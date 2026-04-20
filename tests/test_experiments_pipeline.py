import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("optuna")
pytest.importorskip("imblearn")

import src.experiments_logic as experiments_logic_module
from src.experiments_logic import (
    CONTROLLED_COMPARISON_MODELS,
    ExperimentsRunner,
    build_controlled_comparison_context,
    estimate_controlled_comparison_parallelism,
    preview_controlled_comparison_checkpoint,
)
from src.model_training import temporal_train_test_split
from tests.pipeline_helpers import build_synthetic_base_df


def test_experiments_loop(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)

    runner = ExperimentsRunner(random_state=42)
    base_importance = runner.calculate_feature_importance(
        base_df, base_cols, n_estimators=10
    )
    base_ordered = base_importance["variable"].tolist()
    combined_importance = runner.calculate_feature_importance(
        base_df, feature_cols, n_estimators=10
    )
    combined_ordered = combined_importance["variable"].tolist()

    search_space = {
        "smote": {
            "k_neighbors": {"min": 1, "max": 1},
            "sampling_strategy": {"min": 1.0, "max": 1.0},
        },
        "model": {
            "n_estimators": {"min": 10, "max": 10},
            "max_depth": {"min": 0, "max": 0},
        },
    }
    results = runner.run_iterative_experiment(
        base_df=base_df,
        base_features_ordered=base_ordered,
        cluster_features=combined_ordered,
        model_choice="Random Forest",
        n_trials=1,
        timeout=30,
        far_target=0.2,
        search_space_config=search_space,
        step_size=5,
        test_size=0.2,
        val_size=0.2,
    )
    assert results
    types = {row["type"] for row in results}
    assert "Base" in types
    assert "Base+Cluster" in types


def test_run_optimization_loop_forwards_optuna_n_jobs(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    train_val_df, test_df = temporal_train_test_split(base_df, test_size=0.2)
    train_df, val_df = temporal_train_test_split(train_val_df, test_size=0.25)
    search_space = {
        "smote": {
            "k_neighbors": {"min": 1, "max": 1},
            "sampling_strategy": {"min": 1.0, "max": 1.0},
        },
        "model": {
            "n_estimators": {"min": 10, "max": 10},
            "max_depth": {"min": 0, "max": 0},
        },
    }

    captured = {}
    original_optimize = optuna.study.Study.optimize

    def _wrapped_optimize(self, *args, **kwargs):
        captured["n_jobs"] = kwargs.get("n_jobs")
        return original_optimize(self, *args, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", _wrapped_optimize)

    result = runner.run_optimization_loop(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        model_choice="Random Forest",
        n_trials=1,
        timeout=30,
        far_target=0.2,
        search_space_config=search_space,
        optuna_n_jobs=2,
    )

    assert captured["n_jobs"] == 2
    assert result["optuna_n_jobs"] == 2


def test_run_optimization_loop_applies_requested_calibration(tmp_path, monkeypatch):
    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    train_val_df, test_df = temporal_train_test_split(base_df, test_size=0.2)
    train_df, val_df = temporal_train_test_split(train_val_df, test_size=0.25)
    search_space = {
        "smote": {
            "k_neighbors": {"min": 1, "max": 1},
            "sampling_strategy": {"min": 1.0, "max": 1.0},
        },
        "model": {
            "n_estimators": {"min": 10, "max": 10},
            "max_depth": {"min": 0, "max": 0},
        },
    }

    observed = {"methods": [], "scores": []}

    class _FakeCalibrator:
        def __init__(self, method: str) -> None:
            self.method = method

        def transform(self, scores):
            scores_arr = np.asarray(scores, dtype=float)
            observed["scores"].append(scores_arr + 0.1)
            return scores_arr + 0.1

    def _fake_fit_score_calibrator(y_true, scores, *, method="none"):
        observed["methods"].append(str(method))
        return _FakeCalibrator(str(method))

    def _fake_score_optuna_objective(y_true, scores, **kwargs):
        observed["scores"].append(np.asarray(scores, dtype=float))
        return {
            "score": 1.0,
            "threshold": 0.5,
            "objective_metric": kwargs.get("objective_metric", "f1"),
            "objective_label": "F1",
            "objective_direction": "maximize",
            "metrics": {
                "f1": 1.0,
                "accuracy": 1.0,
                "recall": 1.0,
                "precision": 1.0,
                "roc_auc": 1.0,
                "pr_auc": 1.0,
                "balanced_f1": 1.0,
                "mcc": 1.0,
                "brier_score": 0.0,
                "far": 0.0,
                "sensitivity": 1.0,
                "true_negatives": 1,
                "false_positives": 0,
                "false_negatives": 0,
                "true_positives": 1,
            },
        }

    monkeypatch.setattr(
        experiments_logic_module,
        "fit_score_calibrator",
        _fake_fit_score_calibrator,
    )
    monkeypatch.setattr(
        experiments_logic_module,
        "score_optuna_objective",
        _fake_score_optuna_objective,
    )

    result = runner.run_optimization_loop(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        model_choice="Random Forest",
        n_trials=1,
        timeout=30,
        far_target=0.2,
        search_space_config=search_space,
        threshold_strategy="far",
        calibration_method="sigmoid",
    )

    assert observed["methods"]
    assert set(observed["methods"]) == {"sigmoid"}
    assert observed["scores"]
    assert result["calibration_method"] == "sigmoid"


def _controlled_search_space() -> dict:
    return {
        "smote": {
            "k_neighbors": {"min": 1, "max": 2, "step": 1},
            "sampling_strategy": {"min": 0.5, "max": 1.0, "step": 0.5},
        },
        "rf": {
            "n_estimators": {"min": 10, "max": 10, "step": 1},
            "max_depth": {"min": 3, "max": 3, "step": 1},
            "min_samples_split": {"min": 2, "max": 2, "step": 1},
            "min_samples_leaf": {"min": 1, "max": 1, "step": 1},
            "max_features": ["sqrt"],
        },
        "svm": {
            "C": {"min": 0.5, "max": 0.5, "step": 0.1},
            "kernel": ["linear"],
            "gamma": ["scale"],
            "degree": {"min": 2, "max": 2, "step": 1},
            "coef0": {"min": 0.0, "max": 0.0, "step": 0.1},
        },
        "xgb": {
            "n_estimators": {"min": 10, "max": 10, "step": 1},
            "max_depth": {"min": 3, "max": 3, "step": 1},
            "learning_rate": {"min": 0.1, "max": 0.1, "step": 0.01},
            "subsample": {"min": 1.0, "max": 1.0, "step": 0.1},
            "colsample_bytree": {"min": 1.0, "max": 1.0, "step": 0.1},
            "min_child_weight": {"min": 1.0, "max": 1.0, "step": 1.0},
            "reg_alpha": {"min": 0.0, "max": 0.0, "step": 0.1},
            "reg_lambda": {"min": 1.0, "max": 1.0, "step": 0.1},
            "gamma": {"min": 0.0, "max": 0.0, "step": 0.1},
        },
    }


def _fake_controlled_result(
    *,
    model_name: str,
    feature_set: str,
    balance_mode: str,
    selected_features,
    score: float,
    best_params: dict,
    smote_params: dict,
    optuna_trials_completed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    return {
        "status": "completed",
        "model_name": model_name,
        "feature_set": feature_set,
        "balance_mode": balance_mode,
        "objective_metric": "roc_auc",
        "objective_label": "ROC-AUC",
        "k": int(len(selected_features)),
        "selected_features": list(selected_features),
        "selected_feature_count": int(len(selected_features)),
        "decision_threshold": 0.5,
        "val_objective_score": float(score),
        "test_objective_score": float(score - 0.01),
        "val_roc_auc": float(score),
        "test_roc_auc": float(score - 0.01),
        "val_pr_auc": float(score / 2.0),
        "test_pr_auc": float((score - 0.01) / 2.0),
        "val_f1": 0.5,
        "test_f1": 0.49,
        "val_mcc": 0.3,
        "test_mcc": 0.29,
        "val_recall": 0.4,
        "test_recall": 0.39,
        "val_false_positives": 10,
        "test_false_positives": 12,
        "val_false_alarms_per_day": 1.2,
        "test_false_alarms_per_day": 1.3,
        "val_cost_per_day": 2.4,
        "test_cost_per_day": 2.6,
        "best_params": dict(best_params),
        "effective_model_params": dict(best_params),
        "smote_params": dict(smote_params),
        "optuna_trials_completed": int(optuna_trials_completed),
        "optuna_n_jobs": 1,
        "parallel_jobs": 1,
        "xgb_parallel_jobs": 1,
        "threshold_n_jobs": 1,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "trials_df": pd.DataFrame(
            [{"value": score, "state": "COMPLETE"}]
            if optuna_trials_completed
            else []
        ),
    }


def _expected_k_grid(k_min: int, k_max: int, k_step: int, feature_count: int) -> list[int]:
    if feature_count <= 0:
        return []
    resolved_min = max(1, min(int(k_min), int(feature_count)))
    resolved_max = max(1, min(int(k_max), int(feature_count)))
    resolved_step = max(1, int(k_step))
    if resolved_min > resolved_max:
        resolved_min = resolved_max
    values = list(range(resolved_min, resolved_max + 1, resolved_step))
    if not values:
        values = [resolved_max]
    if values[-1] != resolved_max:
        values.append(resolved_max)
    return sorted({int(value) for value in values if value > 0})


def _synthetic_calibration_base_df(rows: int = 120) -> pd.DataFrame:
    idx = np.arange(rows)
    target = (((idx % 11) == 0) | ((idx % 17) == 0)).astype(int)
    return pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "signal": np.where(target == 1, 0.9, 0.2),
            "aux_signal": np.where(target == 1, 0.7, 0.3) + (idx % 5) * 0.01,
            "target": target,
        }
    )


def test_calibration_sweep_leaderboard_applies_pareto_and_rankable_rules():
    grid_results_df = pd.DataFrame(
        [
            {
                "combo_id": "best",
                "status": "completed",
                "balance_mode": "none",
                "optuna_objective_metric": "mcc",
                "calibration_method": "sigmoid",
                "threshold_objective": "far",
                "val_mcc": 0.70,
                "val_brier_score": 0.10,
                "val_pr_auc": 0.80,
                "val_true_positives": 4,
                "val_false_negatives": 1,
                "val_far": 0.10,
            },
            {
                "combo_id": "dominated",
                "status": "completed",
                "balance_mode": "smote",
                "optuna_objective_metric": "mcc",
                "calibration_method": "isotonic",
                "threshold_objective": "far",
                "val_mcc": 0.50,
                "val_brier_score": 0.20,
                "val_pr_auc": 0.60,
                "val_true_positives": 3,
                "val_false_negatives": 2,
                "val_far": 0.20,
            },
            {
                "combo_id": "not_rankable",
                "status": "completed",
                "balance_mode": "none",
                "optuna_objective_metric": "brier_score",
                "calibration_method": "none",
                "threshold_objective": "f1",
                "val_mcc": 0.40,
                "val_brier_score": 0.15,
                "val_pr_auc": 0.55,
                "val_true_positives": 0,
                "val_false_negatives": 5,
                "val_far": 0.05,
            },
        ]
    )

    leaderboard_df, pareto_front_df = (
        experiments_logic_module._build_calibration_sweep_leaderboard(
            grid_results_df
        )
    )

    best_row = leaderboard_df.loc[leaderboard_df["combo_id"] == "best"].iloc[0]
    dominated_row = leaderboard_df.loc[
        leaderboard_df["combo_id"] == "dominated"
    ].iloc[0]
    not_rankable_row = leaderboard_df.loc[
        leaderboard_df["combo_id"] == "not_rankable"
    ].iloc[0]

    assert bool(best_row["rankable"]) is True
    assert int(best_row["pareto_front"]) == 1
    assert int(best_row["rank"]) == 1
    assert bool(dominated_row["rankable"]) is True
    assert int(dominated_row["pareto_front"]) > int(best_row["pareto_front"])
    assert pd.isna(not_rankable_row["pareto_front"])
    assert bool(not_rankable_row["rankable"]) is False
    assert pd.isna(not_rankable_row["rank"])
    assert pareto_front_df["combo_id"].tolist() == ["best"]


def test_calibration_sweep_leaderboard_breaks_ties_with_stability_score():
    grid_results_df = pd.DataFrame(
        [
            {
                "combo_id": "stable",
                "status": "completed",
                "balance_mode": "none",
                "optuna_objective_metric": "mcc",
                "calibration_method": "sigmoid",
                "threshold_objective": "far",
                "val_mcc": 0.70,
                "val_brier_score": 0.12,
                "val_pr_auc": 0.70,
                "val_true_positives": 4,
                "val_false_negatives": 1,
                "val_far": 0.12,
            },
            {
                "combo_id": "unstable",
                "status": "completed",
                "balance_mode": "smote",
                "optuna_objective_metric": "mcc",
                "calibration_method": "isotonic",
                "threshold_objective": "far",
                "val_mcc": 0.90,
                "val_brier_score": 0.27,
                "val_pr_auc": 0.90,
                "val_true_positives": 4,
                "val_false_negatives": 1,
                "val_far": 0.27,
            },
            {
                "combo_id": "anchor",
                "status": "completed",
                "balance_mode": "none",
                "optuna_objective_metric": "pr_auc",
                "calibration_method": "none",
                "threshold_objective": "mcc",
                "val_mcc": 0.50,
                "val_brier_score": 0.30,
                "val_pr_auc": 0.50,
                "val_true_positives": 2,
                "val_false_negatives": 3,
                "val_far": 0.30,
            },
        ]
    )

    leaderboard_df, _pareto_front_df = (
        experiments_logic_module._build_calibration_sweep_leaderboard(
            grid_results_df
        )
    )
    ordered_ids = leaderboard_df.loc[
        leaderboard_df["rankable"].astype(bool),
        "combo_id",
    ].tolist()
    stable_row = leaderboard_df.loc[leaderboard_df["combo_id"] == "stable"].iloc[0]
    unstable_row = leaderboard_df.loc[
        leaderboard_df["combo_id"] == "unstable"
    ].iloc[0]

    assert int(stable_row["pareto_front"]) == 1
    assert int(unstable_row["pareto_front"]) == 1
    assert float(stable_row["stability_score"]) > float(
        unstable_row["stability_score"]
    )
    assert ordered_ids[:2] == ["stable", "unstable"]


def test_calibration_sweep_search_space_preserves_none_imbalance_knobs():
    runner = ExperimentsRunner(random_state=42)
    y_train = pd.Series(([0] * 24) + ([1] * 6))

    rf_space = runner._controlled_comparison_search_space(
        model_name="Random Forest",
        balance_mode="none",
        search_space_config={},
        y_train=y_train,
    )
    svm_space = runner._controlled_comparison_search_space(
        model_name="SVM",
        balance_mode="none",
        search_space_config={},
        y_train=y_train,
    )
    xgb_space = runner._controlled_comparison_search_space(
        model_name="XGBoost",
        balance_mode="none",
        search_space_config={},
        y_train=y_train,
    )
    nn_space = runner._controlled_comparison_search_space(
        model_name="Neural Network",
        balance_mode="none",
        search_space_config={},
        y_train=y_train,
    )
    balanced_rf_space = runner._controlled_comparison_search_space(
        model_name="Balanced Random Forest",
        balance_mode="none",
        search_space_config={},
        y_train=y_train,
    )

    assert "class_weight" in rf_space["model"]
    assert "class_weight" in svm_space["model"]
    assert "scale_pos_weight" in xgb_space["model"]
    assert "max_delta_step" in xgb_space["model"]
    assert "pos_weight" in nn_space["model"]
    assert "replacement" in balanced_rf_space["model"]


def test_neural_network_search_space_accepts_phase1_opt_in_knobs():
    runner = ExperimentsRunner(random_state=42)
    y_train = pd.Series(([0] * 24) + ([1] * 6))

    nn_space = runner._controlled_comparison_search_space(
        model_name="Neural Network",
        balance_mode="none",
        search_space_config={
            "nn": {
                "use_batch_norm": [False, True],
                "loss_function": ["cross_entropy", "focal"],
                "focal_gamma": {"min": 1.0, "max": 2.0, "step": 1.0},
                "focal_alpha": {"choices": [0.25, 0.75]},
                "max_grad_norm": {"choices": [None, 1.0]},
                "lr_scheduler": ["none", "reduce_on_plateau"],
                "scheduler_factor": {"choices": [0.25]},
                "scheduler_patience": {"choices": [1, 2]},
                "min_lr": {"choices": [1e-6]},
                "temperature_scaling": [False, True],
            }
        },
        y_train=y_train,
    )

    model_space = nn_space["model"]
    assert model_space["use_batch_norm"] == [False, True]
    assert model_space["loss_function"] == ["cross_entropy", "focal"]
    assert model_space["focal_gamma"] == [1.0, 2.0]
    assert model_space["focal_alpha"] == [0.25, 0.75]
    assert model_space["max_grad_norm"] == [None, 1.0]
    assert model_space["lr_scheduler"] == ["none", "reduce_on_plateau"]
    assert model_space["scheduler_factor"] == [0.25]
    assert model_space["scheduler_patience"] == [1, 2]
    assert model_space["min_lr"] == [1e-06]
    assert model_space["temperature_scaling"] == [False, True]


def test_run_calibration_sweep_persists_artifacts_and_ranks_on_validation(
    tmp_path, monkeypatch
):
    base_df = _synthetic_calibration_base_df()
    runner = ExperimentsRunner(random_state=42)
    captured_calls = []

    def _fake_optimize(self, **kwargs):
        balance_mode = str(kwargs["balance_mode"])
        captured_calls.append(dict(kwargs))
        if balance_mode == "none":
            val_metrics = {
                "mcc": 0.82,
                "brier_score": 0.08,
                "pr_auc": 0.84,
                "true_positives": 6,
                "false_negatives": 1,
                "far": 0.08,
            }
            test_metrics = {
                "mcc": 0.40,
                "brier_score": 0.20,
                "pr_auc": 0.55,
                "true_positives": 4,
                "false_negatives": 3,
                "far": 0.20,
            }
        else:
            val_metrics = {
                "mcc": 0.60,
                "brier_score": 0.14,
                "pr_auc": 0.70,
                "true_positives": 5,
                "false_negatives": 2,
                "far": 0.12,
            }
            test_metrics = {
                "mcc": 0.88,
                "brier_score": 0.06,
                "pr_auc": 0.90,
                "true_positives": 7,
                "false_negatives": 0,
                "far": 0.05,
            }
        val_positive_support = (
            val_metrics["true_positives"] + val_metrics["false_negatives"]
        )
        test_positive_support = (
            test_metrics["true_positives"] + test_metrics["false_negatives"]
        )
        return {
            "status": "completed",
            "model_name": str(kwargs["model_name"]),
            "feature_set": str(kwargs["feature_set"]),
            "balance_mode": balance_mode,
            "objective_metric": str(kwargs["objective_metric"]),
            "objective_label": str(kwargs["objective_metric"]).upper(),
            "objective_direction": "maximize",
            "k": int(len(kwargs["selected_features"])),
            "selected_features": list(kwargs["selected_features"]),
            "selected_feature_count": int(len(kwargs["selected_features"])),
            "decision_threshold": 0.42 if balance_mode == "none" else 0.55,
            "val_objective_score": float(val_metrics["mcc"]),
            "test_objective_score": float(test_metrics["mcc"]),
            "val_roc_auc": 0.80 if balance_mode == "none" else 0.72,
            "test_roc_auc": 0.61 if balance_mode == "none" else 0.91,
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "test_pr_auc": float(test_metrics["pr_auc"]),
            "val_brier_score": float(val_metrics["brier_score"]),
            "test_brier_score": float(test_metrics["brier_score"]),
            "val_f1": 0.70,
            "test_f1": 0.55,
            "val_f1_global": 0.70,
            "test_f1_global": 0.55,
            "val_balanced_f1": 0.71,
            "test_balanced_f1": 0.56,
            "val_f1_class_0": 0.90,
            "test_f1_class_0": 0.80,
            "val_f1_class_1": 0.52,
            "test_f1_class_1": 0.40,
            "val_mcc": float(val_metrics["mcc"]),
            "test_mcc": float(test_metrics["mcc"]),
            "val_recall": float(
                val_metrics["true_positives"] / max(1, val_positive_support)
            ),
            "test_recall": float(
                test_metrics["true_positives"] / max(1, test_positive_support)
            ),
            "val_alerts_per_day": 3.0,
            "test_alerts_per_day": 3.5,
            "val_false_alarms_per_day": 0.6,
            "test_false_alarms_per_day": 0.4,
            "val_event_recall_approx": 0.70,
            "test_event_recall_approx": 0.80,
            "val_operational_cost": 8.0,
            "test_operational_cost": 6.0,
            "val_cost_per_day": 1.0,
            "test_cost_per_day": 0.8,
            "alerts_per_day_budget": 5.0,
            "fn_cost": 10.0,
            "fp_cost": 1.0,
            "val_false_negatives": int(val_metrics["false_negatives"]),
            "test_false_negatives": int(test_metrics["false_negatives"]),
            "val_false_positives": 3,
            "test_false_positives": 2,
            "val_true_negatives": 50,
            "test_true_negatives": 48,
            "val_true_positives": int(val_metrics["true_positives"]),
            "test_true_positives": int(test_metrics["true_positives"]),
            "val_positive_support": int(val_positive_support),
            "test_positive_support": int(test_positive_support),
            "val_tp_capture": float(
                val_metrics["true_positives"] / max(1, val_positive_support)
            ),
            "test_tp_capture": float(
                test_metrics["true_positives"] / max(1, test_positive_support)
            ),
            "val_fn_rate": float(
                val_metrics["false_negatives"] / max(1, val_positive_support)
            ),
            "test_fn_rate": float(
                test_metrics["false_negatives"] / max(1, test_positive_support)
            ),
            "val_far": float(val_metrics["far"]),
            "test_far": float(test_metrics["far"]),
            "val_confusion_matrix": [[50, 3], [1, 6]],
            "test_confusion_matrix": [[48, 2], [3, 4]],
            "best_params": {"max_depth": 4},
            "effective_model_params": {"max_depth": 4},
            "smote_params": {} if balance_mode == "none" else {"k_neighbors": 3},
            "optuna_trials_completed": 1,
            "optuna_n_jobs": int(kwargs["optuna_n_jobs"]),
            "parallel_jobs": int(kwargs["parallel_jobs"]),
            "xgb_parallel_jobs": int(kwargs["xgb_parallel_jobs"]),
            "threshold_n_jobs": 1,
            "optuna_jobs_cpu_cap": int(kwargs["optuna_n_jobs"]),
            "cpu_count": 4,
            "train_rows": int(len(kwargs["train_df"])),
            "val_rows": int(len(kwargs["val_df"])),
            "test_rows": int(len(kwargs["test_df"])),
            "trials_df": pd.DataFrame(
                [{"number": 0, "value": float(val_metrics["mcc"]), "state": "COMPLETE"}]
            ),
        }

    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )

    payload = runner.run_calibration_sweep(
        base_df,
        model_name="Random Forest",
        selected_features=["signal", "aux_signal"],
        objective_metrics=["mcc"],
        calibration_methods=["sigmoid"],
        threshold_objectives=["far"],
        n_trials=1,
        timeout=30,
        checkpoint_root=tmp_path / "calibration_runs",
    )

    assert len(captured_calls) == 2
    assert {call["balance_mode"] for call in captured_calls} == {"none", "smote"}
    assert all(call["threshold_protocol"] == "robust" for call in captured_calls)
    assert (tmp_path / "calibration_runs" / payload["run_id"] / "results" / "leaderboard.csv").exists()
    assert (tmp_path / "calibration_runs" / payload["run_id"] / "results" / "pareto_front.csv").exists()
    assert (tmp_path / "calibration_runs" / payload["run_id"] / "manifest.json").exists()
    assert not payload["leaderboard_df"].empty
    assert not payload["pareto_front_df"].empty
    top_balance_mode = payload["leaderboard_df"].iloc[0]["balance_mode"]
    assert top_balance_mode == "none"


def test_estimate_controlled_comparison_parallelism_builds_safe_frontier(tmp_path):
    base_df, _feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)

    estimate = estimate_controlled_comparison_parallelism(
        base_df,
        test_size=0.2,
        val_size=0.25,
        k_min=2,
        k_max=12,
        k_step=5,
        search_space_config=_controlled_search_space(),
        memory_budget_bytes=2 * (1024 ** 3),
        max_cpu_count=8,
    )

    assert estimate["max_parallel_jobs_when_optuna_1"] >= 1
    assert estimate["max_optuna_jobs_when_parallel_1"] >= 1
    assert isinstance(estimate["frontier_df"], pd.DataFrame)
    assert not estimate["frontier_df"].empty
    assert isinstance(estimate["safe_frontier_df"], pd.DataFrame)
    assert not estimate["safe_frontier_df"].empty
    assert isinstance(estimate["recommended_pair"], dict)
    assert int(estimate["recommended_pair"]["parallel_jobs"]) >= 1
    assert int(estimate["recommended_pair"]["optuna_n_jobs"]) >= 1
    assert int(estimate["recommended_pair"]["xgb_parallel_jobs"]) >= 1
    assert estimate["max_xgb_parallel_jobs_when_parallel_1_optuna_1"] >= 1
    assert "xgb_parallel_jobs" in estimate["frontier_df"].columns
    assert "cpu_limited_optuna_jobs" in estimate["frontier_df"].columns


def test_estimate_controlled_comparison_parallelism_handles_tiny_budget(tmp_path):
    base_df, _feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)

    estimate = estimate_controlled_comparison_parallelism(
        base_df,
        test_size=0.2,
        val_size=0.25,
        k_min=2,
        k_max=12,
        k_step=5,
        search_space_config=_controlled_search_space(),
        memory_budget_bytes=64 * (1024 ** 2),
        max_cpu_count=4,
    )

    assert estimate["max_parallel_jobs_when_optuna_1"] == 0
    assert estimate["max_optuna_jobs_when_parallel_1"] == 0
    assert estimate["recommended_pair"] is None
    assert isinstance(estimate["safe_frontier_df"], pd.DataFrame)
    assert estimate["safe_frontier_df"].empty


def test_run_controlled_comparison_normalizes_k_and_builds_summary(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, base_cols, cluster_cols = build_synthetic_base_df(tmp_path)
    event_path = tmp_path / "events.csv"
    features_path = tmp_path / "features.duckdb"
    event_path.write_text("events", encoding="utf-8")
    features_path.write_text("features", encoding="utf-8")

    runner = ExperimentsRunner(random_state=42)

    def _fake_importance(self, df, feature_cols, **kwargs):
        return pd.DataFrame(
            {
                "variable": list(feature_cols),
                "importance": list(range(len(feature_cols), 0, -1)),
            }
        )

    def _fake_optimize(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        **kwargs,
    ):
        base_score = {
            "Base": 0.70,
            "Cluster": 0.65,
            "Base + Cluster": 0.75,
        }[feature_set]
        val_score = base_score + (0.03 if balance_mode == "smote" else 0.0)
        val_score += len(selected_features) * 0.001
        return {
            "status": "completed",
            "model_name": model_name,
            "feature_set": feature_set,
            "balance_mode": balance_mode,
            "objective_metric": "roc_auc",
            "objective_label": "ROC-AUC",
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": 0.5,
            "val_objective_score": float(val_score),
            "test_objective_score": float(val_score - 0.01),
            "val_roc_auc": float(val_score),
            "test_roc_auc": float(val_score - 0.01),
            "val_f1": 0.6,
            "test_f1": 0.59,
            "val_mcc": 0.4,
            "test_mcc": 0.39,
            "best_params": {"model": model_name, "k": len(selected_features)},
            "smote_params": (
                {"k_neighbors": 1, "sampling_strategy": 1.0}
                if balance_mode == "smote"
                else {}
            ),
            "optuna_trials_completed": 1,
            "optuna_n_jobs": int(kwargs["optuna_n_jobs"]),
            "parallel_jobs": int(kwargs["parallel_jobs"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": pd.DataFrame([{"value": val_score, "state": "COMPLETE"}]),
        }

    monkeypatch.setattr(
        ExperimentsRunner,
        "calculate_feature_importance",
        _fake_importance,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )

    k_min = 2
    k_max = max(len(feature_cols), len(cluster_cols), len(base_cols)) + 5
    k_step = 5
    payload = runner.run_controlled_comparison(
        base_df,
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=k_min,
        k_max=k_max,
        k_step=k_step,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=2,
        parallel_jobs=3,
        search_space_config=_controlled_search_space(),
        checkpoint_root=tmp_path / "controlled_ckpt",
        start_fresh=True,
    )

    grid_df = payload["grid_results_df"]
    summary_df = payload["best_summary_df"]

    expected_by_set = {
        "Base": _expected_k_grid(k_min, k_max, k_step, len(base_cols)),
        "Cluster": _expected_k_grid(k_min, k_max, k_step, len(cluster_cols)),
        "Base + Cluster": _expected_k_grid(k_min, k_max, k_step, len(feature_cols)),
    }
    for feature_set, expected_k in expected_by_set.items():
        observed = sorted(
            grid_df.loc[grid_df["feature_set"] == feature_set, "k"]
            .astype(int)
            .unique()
            .tolist()
        )
        assert observed == expected_k

    expected_model_count = len(CONTROLLED_COMPARISON_MODELS)
    expected_protocol_count = 1
    expected_total = (
        expected_model_count
        * 2
        * expected_protocol_count
        * sum(len(values) for values in expected_by_set.values())
    )
    assert len(grid_df) == expected_total
    assert len(summary_df) == expected_model_count * 3 * 2 * expected_protocol_count
    assert {
        "k_optimo",
        "smote_optimo",
        "balance_mode",
        "threshold_protocol",
        "threshold_objective",
        "calibration_method",
        "best_test_accuracy",
        "best_test_recall",
        "best_test_sensitivity",
        "best_test_roc_auc",
        "best_test_pr_auc",
        "best_test_balanced_f1",
        "best_test_alerts_per_day",
        "best_test_false_alarms_per_day",
        "best_test_event_recall_approx",
        "best_test_operational_cost",
        "best_test_f1_global",
        "best_test_f1_class_0",
        "best_test_f1_class_1",
        "best_test_false_negatives",
        "best_test_false_positives",
        "best_test_confusion_matrix",
        "val_roc_auc",
        "val_objective_score",
        "test_objective_score",
    }.issubset(
        summary_df.columns
    )
    assert set(summary_df["balance_mode"].astype(str)) == {"none", "smote"}
    assert set(summary_df["threshold_protocol"].astype(str)) == {"conservative"}


def test_run_controlled_comparison_uses_selected_model_subset(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, base_cols, cluster_cols = build_synthetic_base_df(tmp_path)
    event_path = tmp_path / "events.csv"
    features_path = tmp_path / "features.duckdb"
    event_path.write_text("events", encoding="utf-8")
    features_path.write_text("features", encoding="utf-8")

    runner = ExperimentsRunner(random_state=42)
    selected_models = ["Random Forest", "SVM"]

    def _fake_importance(self, df, feature_cols, **kwargs):
        return pd.DataFrame(
            {
                "variable": list(feature_cols),
                "importance": list(range(len(feature_cols), 0, -1)),
            }
        )

    def _fake_optimize(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        **kwargs,
    ):
        score = 0.7 + (0.01 if balance_mode == "smote" else 0.0)
        return {
            "status": "completed",
            "model_name": model_name,
            "feature_set": feature_set,
            "balance_mode": balance_mode,
            "objective_metric": "roc_auc",
            "objective_label": "ROC-AUC",
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": 0.5,
            "val_objective_score": score,
            "test_objective_score": score - 0.01,
            "val_roc_auc": score,
            "test_roc_auc": score - 0.01,
            "val_f1": 0.6,
            "test_f1": 0.59,
            "val_mcc": 0.4,
            "test_mcc": 0.39,
            "best_params": {"model": model_name},
            "smote_params": {},
            "optuna_trials_completed": 1,
            "optuna_n_jobs": int(kwargs["optuna_n_jobs"]),
            "parallel_jobs": int(kwargs["parallel_jobs"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": pd.DataFrame([{"value": score, "state": "COMPLETE"}]),
        }

    monkeypatch.setattr(
        ExperimentsRunner,
        "calculate_feature_importance",
        _fake_importance,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )

    payload = runner.run_controlled_comparison(
        base_df,
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=1,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=tmp_path / "controlled_ckpt_subset",
        selected_models=selected_models,
        start_fresh=True,
    )

    grid_df = payload["grid_results_df"]
    summary_df = payload["best_summary_df"]

    assert payload["protocol"]["models"] == selected_models
    assert sorted(grid_df["model_name"].unique().tolist()) == sorted(selected_models)
    assert sorted(summary_df["model_name"].unique().tolist()) == sorted(selected_models)
    assert len(grid_df) == len(selected_models) * 3 * 2
    assert len(summary_df) == len(selected_models) * 3 * 2
    assert set(summary_df["threshold_protocol"].astype(str)) == {"conservative"}


def test_run_controlled_comparison_ranking_uses_train_only(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    train_val_df, _test_df = temporal_train_test_split(base_df, test_size=0.2)
    expected_train_df, _expected_val_df = temporal_train_test_split(
        train_val_df, test_size=0.25
    )
    expected_times = expected_train_df["interval_start"].sort_values().tolist()
    ranking_calls = []

    def _record_importance(self, df, feature_cols, **kwargs):
        ranking_calls.append(df["interval_start"].sort_values().tolist())
        return pd.DataFrame(
            {
                "variable": list(feature_cols),
                "importance": list(range(len(feature_cols), 0, -1)),
            }
        )

    def _fake_optimize(self, *, selected_features, train_df, val_df, test_df, **kwargs):
        return {
            "status": "completed",
            "model_name": kwargs["model_name"],
            "feature_set": kwargs["feature_set"],
            "balance_mode": kwargs["balance_mode"],
            "objective_metric": "roc_auc",
            "objective_label": "ROC-AUC",
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": 0.5,
            "val_objective_score": 0.7,
            "test_objective_score": 0.69,
            "val_roc_auc": 0.7,
            "test_roc_auc": 0.69,
            "val_f1": 0.6,
            "test_f1": 0.59,
            "val_mcc": 0.4,
            "test_mcc": 0.39,
            "best_params": {},
            "smote_params": {},
            "optuna_trials_completed": 1,
            "optuna_n_jobs": int(kwargs["optuna_n_jobs"]),
            "parallel_jobs": int(kwargs["parallel_jobs"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": pd.DataFrame([{"value": 0.7, "state": "COMPLETE"}]),
        }

    monkeypatch.setattr(
        ExperimentsRunner,
        "calculate_feature_importance",
        _record_importance,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )

    runner.run_controlled_comparison(
        base_df,
        event_path=tmp_path / "events.csv",
        features_path=tmp_path / "features.duckdb",
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=1,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=tmp_path / "controlled_ckpt_train_only",
        start_fresh=True,
    )

    assert len(ranking_calls) == 3
    for observed_times in ranking_calls:
        assert observed_times == expected_times


def test_run_controlled_comparison_modelos_k_uses_single_global_feature_selection_ranking(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, base_cols, cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    ranking_calls = []
    optimize_calls = []
    global_order = [
        cluster_cols[0],
        base_cols[0],
        cluster_cols[1],
        base_cols[1],
    ]
    global_order.extend(
        col for col in feature_cols if col not in set(global_order)
    )

    def _record_global_importance(self, df, feature_cols, **kwargs):
        ranking_calls.append(
            {
                "rows": len(df),
                "feature_cols": list(feature_cols),
                "kwargs": dict(kwargs),
            }
        )
        return pd.DataFrame(
            {
                "variable": list(global_order),
                "importance": list(range(len(global_order), 0, -1)),
            }
        )

    def _fake_optimize(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        **kwargs,
    ):
        optimize_calls.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "balance_mode": balance_mode,
                "selected_features": list(selected_features),
            }
        )
        score = 0.6 + (len(selected_features) * 0.01)
        return {
            "status": "completed",
            "model_name": model_name,
            "feature_set": feature_set,
            "balance_mode": balance_mode,
            "objective_metric": "roc_auc",
            "objective_label": "ROC-AUC",
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": 0.5,
            "val_objective_score": score,
            "test_objective_score": score - 0.01,
            "val_roc_auc": score,
            "test_roc_auc": score - 0.01,
            "val_f1": 0.5,
            "test_f1": 0.49,
            "val_mcc": 0.3,
            "test_mcc": 0.29,
            "best_params": {"model": model_name},
            "smote_params": {},
            "optuna_trials_completed": 1,
            "optuna_n_jobs": int(kwargs["optuna_n_jobs"]),
            "parallel_jobs": int(kwargs["parallel_jobs"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": pd.DataFrame([{"value": score, "state": "COMPLETE"}]),
        }

    monkeypatch.setattr(
        ExperimentsRunner,
        "calculate_feature_importance",
        _record_global_importance,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )

    payload = runner.run_controlled_comparison(
        base_df,
        event_path=tmp_path / "events.csv",
        features_path=tmp_path / "features.duckdb",
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=2,
        k_max=3,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=tmp_path / "controlled_ckpt_modelos_k",
        start_fresh=True,
        selected_models=["XGBoost"],
        threshold_protocols=["conservative"],
        feature_ranking_mode="feature_selection_global",
        feature_selection_n_estimators=123,
        feature_selection_max_depth=4,
        feature_selection_n_jobs=-1,
    )

    assert len(ranking_calls) == 1
    assert ranking_calls[0]["rows"] == len(base_df)
    assert ranking_calls[0]["feature_cols"] == feature_cols
    assert ranking_calls[0]["kwargs"]["n_estimators"] == 123
    assert ranking_calls[0]["kwargs"]["max_depth"] == 4
    assert ranking_calls[0]["kwargs"]["n_jobs"] == -1

    assert len(optimize_calls) == 2 * 3 * 2
    grid_df = payload["grid_results_df"]
    assert set(grid_df["protocol_family"].unique()) == {"modelos_por_k"}
    assert set(grid_df["feature_ranking_mode"].unique()) == {
        "feature_selection_global"
    }
    assert sorted(grid_df["k"].astype(int).unique().tolist()) == [2, 3]
    assert sorted(grid_df["k_global"].astype(int).unique().tolist()) == [2, 3]

    k2_base = grid_df[
        (grid_df["k"].astype(int) == 2)
        & (grid_df["feature_set"] == "Base")
    ].iloc[0]
    k2_cluster = grid_df[
        (grid_df["k"].astype(int) == 2)
        & (grid_df["feature_set"] == "Cluster")
    ].iloc[0]
    k2_combined = grid_df[
        (grid_df["k"].astype(int) == 2)
        & (grid_df["feature_set"] == "Base + Cluster")
    ].iloc[0]

    assert int(k2_base["effective_k"]) == 1
    assert json.loads(k2_base["selected_features"]) == [base_cols[0]]
    assert int(k2_cluster["effective_k"]) == 1
    assert json.loads(k2_cluster["selected_features"]) == [cluster_cols[0]]
    assert int(k2_combined["effective_k"]) == 2
    assert json.loads(k2_combined["selected_features_global"]) == global_order[:2]


def test_run_controlled_comparison_frozen_tuning_ablation_builds_cross_matrix(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, _feature_cols, base_cols, cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    optimize_calls = []
    frozen_calls = []

    def _fake_importance(self, df, feature_cols, **kwargs):
        ordered = list(feature_cols)
        return pd.DataFrame(
            {
                "variable": ordered,
                "importance": list(range(len(ordered), 0, -1)),
            }
        )

    def _fake_optimize(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        threshold_protocol,
        **kwargs,
    ):
        optimize_calls.append(
            {
                "feature_set": feature_set,
                "balance_mode": balance_mode,
                "threshold_protocol": threshold_protocol,
                "selected_features": list(selected_features),
            }
        )
        best_params = {
            "source": feature_set,
            "feature_count": len(selected_features),
        }
        smote_params = (
            {"k_neighbors": 1, "sampling_strategy": 0.5}
            if balance_mode == "smote"
            else {}
        )
        score = 0.70 + (0.02 if feature_set == "Base + Cluster" else 0.0)
        score += len(selected_features) * 0.001
        return _fake_controlled_result(
            model_name=model_name,
            feature_set=feature_set,
            balance_mode=balance_mode,
            selected_features=selected_features,
            score=score,
            best_params=best_params,
            smote_params=smote_params,
            optuna_trials_completed=3,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    def _fake_frozen_eval(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        frozen_model_params,
        frozen_smote_params,
        threshold_protocol,
        **kwargs,
    ):
        frozen_calls.append(
            {
                "target": feature_set,
                "balance_mode": balance_mode,
                "threshold_protocol": threshold_protocol,
                "frozen_model_params": dict(frozen_model_params),
                "frozen_smote_params": dict(frozen_smote_params),
            }
        )
        score = 0.60
        score += 0.03 if feature_set == "Base + Cluster" else 0.0
        score += 0.02 if frozen_model_params.get("source") == "Base + Cluster" else 0.0
        return _fake_controlled_result(
            model_name=model_name,
            feature_set=feature_set,
            balance_mode=balance_mode,
            selected_features=selected_features,
            score=score,
            best_params=dict(frozen_model_params),
            smote_params=dict(frozen_smote_params),
            optuna_trials_completed=0,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    monkeypatch.setattr(
        ExperimentsRunner,
        "calculate_feature_importance",
        _fake_importance,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_evaluate_controlled_combo_with_frozen_params",
        _fake_frozen_eval,
    )

    payload = runner.run_controlled_comparison(
        base_df,
        event_path=tmp_path / "events.csv",
        features_path=tmp_path / "features.duckdb",
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=2,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=tmp_path / "controlled_ckpt_frozen",
        start_fresh=True,
        selected_models=["XGBoost"],
        threshold_protocols=["conservative"],
        experimental_protocol="frozen_tuning_ablation",
    )

    grid_df = payload["grid_results_df"]
    summary_df = payload["best_summary_df"]
    deltas_df = payload["ablation_deltas_df"]

    assert set(grid_df["protocol_family"].astype(str)) == {"frozen_tuning_ablation"}
    assert set(grid_df["feature_set"].astype(str)) == {"Base", "Base + Cluster"}
    assert "Cluster" not in set(grid_df["target_feature_set"].astype(str))
    assert set(grid_df["params_source_feature_set"].astype(str)) == {
        "Base",
        "Base + Cluster",
    }
    assert set(grid_df["k"].astype(int)) == {1, 2}

    source_rows = grid_df[grid_df["ablation_phase"] == "source_tuning"]
    cross_rows = grid_df[grid_df["ablation_phase"] == "cross_eval"]
    assert len(source_rows) == 2 * 2 * 2
    assert len(cross_rows) == 2 * 2 * 2
    assert set(source_rows["optuna_trials_completed"].astype(int)) == {3}
    assert set(cross_rows["optuna_trials_completed"].astype(int)) == {0}
    assert cross_rows["frozen_tuning"].astype(bool).all()
    assert set(cross_rows["threshold_freeze_policy"].astype(str)) == {
        "recalibrate_per_target"
    }

    assert frozen_calls
    for call in frozen_calls:
        assert call["frozen_model_params"]["source"] != call["target"]
        if call["balance_mode"] == "smote":
            assert call["frozen_smote_params"]["sampling_strategy"] == pytest.approx(0.5)
        else:
            assert call["frozen_smote_params"] == {}

    grouped_pairs = summary_df[
        [
            "params_source_feature_set",
            "target_feature_set",
            "balance_mode",
            "threshold_protocol",
        ]
    ].drop_duplicates()
    assert len(grouped_pairs) == 2 * 2 * 2
    assert len(summary_df) == 2 * 2 * 2

    assert not deltas_df.empty
    assert set(deltas_df["effect_type"].astype(str)) == {
        "feature_effect",
        "tuning_effect",
    }
    assert {
        "delta_val_objective_score",
        "delta_test_roc_auc",
        "delta_test_false_positives",
        "delta_test_false_alarms_per_day",
        "delta_test_cost_per_day",
    }.issubset(deltas_df.columns)
    assert set(payload["protocol"]["ablation_config"]["freeze_scope"]) == {
        "model_params",
        "smote_params",
    }
    assert max(grid_df["effective_k"].astype(int)) <= len(base_cols)
    assert grid_df["selected_cluster_feature_count"].astype(int).max() <= len(cluster_cols)


def test_run_controlled_comparison_frozen_ablation_resume_uses_saved_source_params(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, _feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    event_path = tmp_path / "events.csv"
    features_path = tmp_path / "features.duckdb"
    event_path.write_text("events", encoding="utf-8")
    features_path.write_text("features", encoding="utf-8")
    checkpoint_root = tmp_path / "controlled_ckpt_frozen_resume"
    runner = ExperimentsRunner(random_state=42)
    call_counts = {"optimize": 0, "frozen": 0}

    def _fake_importance(self, df, feature_cols, **kwargs):
        return pd.DataFrame(
            {
                "variable": list(feature_cols),
                "importance": list(range(len(feature_cols), 0, -1)),
            }
        )

    def _fake_optimize(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        **kwargs,
    ):
        call_counts["optimize"] += 1
        return _fake_controlled_result(
            model_name=model_name,
            feature_set=feature_set,
            balance_mode=balance_mode,
            selected_features=selected_features,
            score=0.70,
            best_params={"source": feature_set},
            smote_params=(
                {"k_neighbors": 1, "sampling_strategy": 0.5}
                if balance_mode == "smote"
                else {}
            ),
            optuna_trials_completed=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    def _fake_frozen_eval(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        frozen_model_params,
        frozen_smote_params,
        **kwargs,
    ):
        call_counts["frozen"] += 1
        assert frozen_model_params["source"] in {"Base", "Base + Cluster"}
        return _fake_controlled_result(
            model_name=model_name,
            feature_set=feature_set,
            balance_mode=balance_mode,
            selected_features=selected_features,
            score=0.65,
            best_params=dict(frozen_model_params),
            smote_params=dict(frozen_smote_params),
            optuna_trials_completed=0,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    monkeypatch.setattr(ExperimentsRunner, "calculate_feature_importance", _fake_importance)
    monkeypatch.setattr(ExperimentsRunner, "_optimize_controlled_combo", _fake_optimize)
    monkeypatch.setattr(
        ExperimentsRunner,
        "_evaluate_controlled_combo_with_frozen_params",
        _fake_frozen_eval,
    )

    first_payload = runner.run_controlled_comparison(
        base_df,
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=1,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=checkpoint_root,
        start_fresh=True,
        selected_models=["XGBoost"],
        threshold_protocols=["conservative"],
        experimental_protocol="frozen_tuning_ablation",
    )
    assert call_counts == {"optimize": 4, "frozen": 4}

    run_dir = Path(str(first_payload["checkpoint_run_dir"]))
    grid_path = run_dir / "results" / "grid_results.csv"
    manifest_path = run_dir / "manifest.json"
    grid_df = pd.read_csv(grid_path)
    dropped_idx = grid_df.index[grid_df["ablation_phase"] == "cross_eval"][-1]
    dropped_combo_id = str(grid_df.loc[dropped_idx, "combo_id"])
    grid_df = grid_df.drop(index=dropped_idx).reset_index(drop=True)
    grid_df.to_csv(grid_path, index=False)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["result_status"] = "running"
    steps_index = manifest.get("steps_index") or {}
    if dropped_combo_id in steps_index:
        steps_index[dropped_combo_id]["status"] = "pending"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    second_payload = runner.run_controlled_comparison(
        base_df,
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=1,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=checkpoint_root,
        selected_models=["XGBoost"],
        threshold_protocols=["conservative"],
        experimental_protocol="frozen_tuning_ablation",
    )

    assert call_counts == {"optimize": 4, "frozen": 5}
    assert second_payload["auto_resumed"] is True
    assert len(second_payload["grid_results_df"]) == len(first_payload["grid_results_df"])


def test_run_controlled_comparison_resume_skips_completed_combos(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    event_path = tmp_path / "events.csv"
    features_path = tmp_path / "features.duckdb"
    event_path.write_text("events", encoding="utf-8")
    features_path.write_text("features", encoding="utf-8")
    checkpoint_root = tmp_path / "controlled_ckpt_resume"
    runner = ExperimentsRunner(random_state=42)
    call_counter = {"count": 0}

    def _fake_importance(self, df, feature_cols, **kwargs):
        return pd.DataFrame(
            {
                "variable": list(feature_cols),
                "importance": list(range(len(feature_cols), 0, -1)),
            }
        )

    def _fake_optimize(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        **kwargs,
    ):
        call_counter["count"] += 1
        score = 0.7 + (0.01 if balance_mode == "smote" else 0.0)
        return {
            "status": "completed",
            "model_name": model_name,
            "feature_set": feature_set,
            "balance_mode": balance_mode,
            "objective_metric": "roc_auc",
            "objective_label": "ROC-AUC",
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": 0.5,
            "val_objective_score": score,
            "test_objective_score": score - 0.01,
            "val_roc_auc": score,
            "test_roc_auc": score - 0.01,
            "val_f1": 0.6,
            "test_f1": 0.59,
            "val_mcc": 0.4,
            "test_mcc": 0.39,
            "best_params": {"model": model_name},
            "smote_params": {},
            "optuna_trials_completed": 1,
            "optuna_n_jobs": int(kwargs["optuna_n_jobs"]),
            "parallel_jobs": int(kwargs["parallel_jobs"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": pd.DataFrame([{"value": score, "state": "COMPLETE"}]),
        }

    monkeypatch.setattr(
        ExperimentsRunner,
        "calculate_feature_importance",
        _fake_importance,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )

    first_payload = runner.run_controlled_comparison(
        base_df,
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=1,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=checkpoint_root,
        start_fresh=True,
    )
    initial_calls = call_counter["count"]
    assert initial_calls == len(CONTROLLED_COMPARISON_MODELS) * 3 * 2

    run_dir = Path(str(first_payload["checkpoint_run_dir"]))
    grid_path = run_dir / "results" / "grid_results.csv"
    manifest_path = run_dir / "manifest.json"

    grid_df = pd.read_csv(grid_path)
    dropped_combo_id = str(grid_df.iloc[-1]["combo_id"])
    grid_df = grid_df.iloc[:-1].copy()
    grid_df.to_csv(grid_path, index=False)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["result_status"] = "running"
    manifest["updated_at"] = "2026-01-01T00:00:00"
    steps_index = manifest.get("steps_index") or {}
    if dropped_combo_id in steps_index:
        steps_index[dropped_combo_id]["status"] = "pending"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    second_payload = runner.run_controlled_comparison(
        base_df,
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=1,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=checkpoint_root,
    )

    assert call_counter["count"] == initial_calls + 1
    assert second_payload["auto_resumed"] is True
    assert second_payload["loaded_from_checkpoint"] is False
    assert len(second_payload["grid_results_df"]) == initial_calls


def test_controlled_comparison_checkpoint_compatibility(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, _feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    event_path = tmp_path / "events.csv"
    features_path = tmp_path / "features.duckdb"
    event_path.write_text("events", encoding="utf-8")
    features_path.write_text("features", encoding="utf-8")
    checkpoint_root = tmp_path / "controlled_ckpt_preview"
    runner = ExperimentsRunner(random_state=42)

    def _fake_importance(self, df, feature_cols, **kwargs):
        return pd.DataFrame(
            {
                "variable": list(feature_cols),
                "importance": list(range(len(feature_cols), 0, -1)),
            }
        )

    def _fake_optimize(
        self,
        *,
        model_name,
        feature_set,
        balance_mode,
        selected_features,
        train_df,
        val_df,
        test_df,
        **kwargs,
    ):
        return {
            "status": "completed",
            "model_name": model_name,
            "feature_set": feature_set,
            "balance_mode": balance_mode,
            "objective_metric": "roc_auc",
            "objective_label": "ROC-AUC",
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": 0.5,
            "val_objective_score": 0.7,
            "test_objective_score": 0.69,
            "val_roc_auc": 0.7,
            "test_roc_auc": 0.69,
            "val_f1": 0.6,
            "test_f1": 0.59,
            "val_mcc": 0.4,
            "test_mcc": 0.39,
            "best_params": {},
            "smote_params": {},
            "optuna_trials_completed": 1,
            "optuna_n_jobs": int(kwargs["optuna_n_jobs"]),
            "parallel_jobs": int(kwargs["parallel_jobs"]),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": pd.DataFrame([{"value": 0.7, "state": "COMPLETE"}]),
        }

    monkeypatch.setattr(
        ExperimentsRunner,
        "calculate_feature_importance",
        _fake_importance,
    )
    monkeypatch.setattr(
        ExperimentsRunner,
        "_optimize_controlled_combo",
        _fake_optimize,
    )

    payload = runner.run_controlled_comparison(
        base_df,
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        test_size=0.2,
        val_size=0.25,
        k_min=1,
        k_max=1,
        k_step=1,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        search_space_config=_controlled_search_space(),
        checkpoint_root=checkpoint_root,
        start_fresh=True,
    )

    same_context = build_controlled_comparison_context(
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        protocol=payload["protocol"],
    )
    same_preview = preview_controlled_comparison_checkpoint(
        same_context,
        checkpoint_root=checkpoint_root,
    )
    assert same_preview["checkpoint_available"] is True
    assert same_preview["compatible"] is True

    changed_protocol = dict(payload["protocol"])
    changed_protocol["k_max"] = 2
    changed_context = build_controlled_comparison_context(
        event_path=event_path,
        features_path=features_path,
        segment_info={"segment": "A"},
        protocol=changed_protocol,
    )
    changed_preview = preview_controlled_comparison_checkpoint(
        changed_context,
        checkpoint_root=checkpoint_root,
    )
    assert changed_preview["checkpoint_available"] is False
    assert changed_preview["compatible"] is False


class _DummyModel:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        if len(X) <= 1:
            scores = [0.5] * len(X)
        else:
            scores = [
                0.1 + (0.8 * idx / (len(X) - 1))
                for idx in range(len(X))
            ]
        return pd.DataFrame({"p0": [1 - score for score in scores], "p1": scores}).to_numpy()


class _ReverseDecisionModel:
    def fit(self, X, y):
        return self

    def decision_function(self, X):
        return -np.asarray(X.iloc[:, 0], dtype=float)


def test_controlled_combo_random_forest_forwards_parallel_jobs(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, _feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    train_val_df, test_df = temporal_train_test_split(base_df, test_size=0.2)
    train_df, val_df = temporal_train_test_split(train_val_df, test_size=0.25)
    captured_params = []

    def _fake_build_model(model_name, params, random_state):
        captured_params.append(dict(params))
        return _DummyModel()

    monkeypatch.setattr(
        experiments_logic_module,
        "build_model",
        _fake_build_model,
    )

    result = runner._optimize_controlled_combo(
        model_name="Random Forest",
        feature_set="Base",
        balance_mode="none",
        objective_metric="roc_auc",
        selected_features=base_cols[:2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        search_space_config=_controlled_search_space(),
        parallel_jobs=3,
        xgb_parallel_jobs=1,
    )

    assert captured_params
    assert all(params["n_jobs"] == 3 for params in captured_params)
    assert result["parallel_jobs"] == 3
    assert "n_jobs" not in result["best_params"]
    assert result["effective_model_params"]["n_jobs"] == 3
    assert result["threshold_n_jobs"] == 3


def test_controlled_combo_svm_forwards_optuna_n_jobs(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, _feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    train_val_df, test_df = temporal_train_test_split(base_df, test_size=0.2)
    train_df, val_df = temporal_train_test_split(train_val_df, test_size=0.25)
    captured = {}
    original_optimize = optuna.study.Study.optimize

    def _fake_build_model(model_name, params, random_state):
        return _DummyModel()

    def _wrapped_optimize(self, *args, **kwargs):
        captured["n_jobs"] = kwargs.get("n_jobs")
        return original_optimize(self, *args, **kwargs)

    monkeypatch.setattr(
        experiments_logic_module,
        "build_model",
        _fake_build_model,
    )
    monkeypatch.setattr(optuna.study.Study, "optimize", _wrapped_optimize)

    result = runner._optimize_controlled_combo(
        model_name="SVM",
        feature_set="Base",
        balance_mode="none",
        objective_metric="roc_auc",
        selected_features=base_cols[:2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=2,
        search_space_config=_controlled_search_space(),
        parallel_jobs=4,
        xgb_parallel_jobs=1,
    )

    assert captured["n_jobs"] == 2
    assert result["optuna_n_jobs"] == 2


def test_controlled_trial_params_svm_only_samples_kernel_relevant_values():
    optuna = pytest.importorskip("optuna")

    runner = ExperimentsRunner(random_state=42)
    model_space = {
        "C": [0.5],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"],
        "degree": [2],
        "coef0": [0.0],
    }

    linear_params, linear_smote = runner._controlled_comparison_trial_params(
        optuna.trial.FixedTrial({"kernel": "linear", "C": 0.5}),
        model_name="SVM",
        model_space=model_space,
        smote_space={},
        balance_mode="none",
        parallel_jobs=1,
        xgb_parallel_jobs=1,
    )
    assert linear_smote == {}
    assert linear_params == {
        "kernel": "linear",
        "C": 0.5,
        "probability": False,
    }

    rbf_params, rbf_smote = runner._controlled_comparison_trial_params(
        optuna.trial.FixedTrial({"kernel": "rbf", "C": 0.5, "gamma": "scale"}),
        model_name="SVM",
        model_space=model_space,
        smote_space={},
        balance_mode="none",
        parallel_jobs=1,
        xgb_parallel_jobs=1,
    )
    assert rbf_smote == {}
    assert rbf_params == {
        "kernel": "rbf",
        "C": 0.5,
        "gamma": "scale",
        "probability": False,
    }


def test_controlled_combo_svm_orients_scores_and_disables_probability(monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    runner = ExperimentsRunner(random_state=42)
    train_df = pd.DataFrame(
        {
            "signal": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "target": [0, 0, 0, 1, 1, 1],
            "interval_start": pd.date_range("2024-01-01", periods=6, freq="D"),
        }
    )
    val_df = pd.DataFrame(
        {
            "signal": [0.15, 0.25, 0.75, 0.85],
            "target": [0, 0, 1, 1],
            "interval_start": pd.date_range("2024-02-01", periods=4, freq="D"),
        }
    )
    test_df = pd.DataFrame(
        {
            "signal": [0.12, 0.22, 0.72, 0.82],
            "target": [0, 0, 1, 1],
            "interval_start": pd.date_range("2024-03-01", periods=4, freq="D"),
        }
    )
    captured_params = []

    def _fake_build_model(model_name, params, random_state):
        captured_params.append(dict(params))
        return _ReverseDecisionModel()

    monkeypatch.setattr(
        experiments_logic_module,
        "build_model",
        _fake_build_model,
    )

    result = runner._optimize_controlled_combo(
        model_name="SVM",
        feature_set="Base",
        balance_mode="none",
        objective_metric="roc_auc",
        selected_features=["signal"],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        search_space_config=_controlled_search_space(),
        parallel_jobs=1,
        xgb_parallel_jobs=1,
    )

    assert captured_params
    assert all(params.get("probability") is False for params in captured_params)
    assert result["test_accuracy"] == pytest.approx(1.0)
    assert result["test_recall"] == pytest.approx(1.0)
    assert result["test_sensitivity"] == pytest.approx(1.0)
    assert result["val_roc_auc"] == pytest.approx(1.0)
    assert result["test_roc_auc"] == pytest.approx(1.0)
    assert result["test_pr_auc"] == pytest.approx(1.0)
    assert result["test_f1_global"] == pytest.approx(1.0)
    assert result["test_f1_class_0"] == pytest.approx(1.0)
    assert result["test_f1_class_1"] == pytest.approx(1.0)
    assert result["test_mcc"] == pytest.approx(1.0)
    assert result["test_false_negatives"] == 0
    assert result["test_false_positives"] == 0
    assert result["test_confusion_matrix"] == [[2, 0], [0, 2]]


def test_controlled_combo_xgboost_forwards_xgb_parallel_jobs(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, _feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    train_val_df, test_df = temporal_train_test_split(base_df, test_size=0.2)
    train_df, val_df = temporal_train_test_split(train_val_df, test_size=0.25)
    captured_params = []

    def _fake_build_model(model_name, params, random_state):
        captured_params.append(dict(params))
        return _DummyModel()

    monkeypatch.setattr(
        experiments_logic_module,
        "build_model",
        _fake_build_model,
    )

    result = runner._optimize_controlled_combo(
        model_name="XGBoost",
        feature_set="Base",
        balance_mode="none",
        objective_metric="roc_auc",
        selected_features=base_cols[:2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        search_space_config=_controlled_search_space(),
        parallel_jobs=4,
        xgb_parallel_jobs=6,
    )

    assert captured_params
    assert all(params["n_jobs"] == 6 for params in captured_params)
    assert result["xgb_parallel_jobs"] == 6
    assert "n_jobs" not in result["best_params"]
    assert result["effective_model_params"]["n_jobs"] == 6
    assert result["threshold_n_jobs"] == 6


def test_controlled_combo_xgboost_keeps_ui_optuna_jobs(
    tmp_path, monkeypatch
):
    pytest.importorskip("sklearn")
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, _feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    runner = ExperimentsRunner(random_state=42)
    train_val_df, test_df = temporal_train_test_split(base_df, test_size=0.2)
    train_df, val_df = temporal_train_test_split(train_val_df, test_size=0.25)
    captured = {}
    original_optimize = optuna.study.Study.optimize

    def _fake_build_model(model_name, params, random_state):
        return _DummyModel()

    def _wrapped_optimize(self, *args, **kwargs):
        captured["n_jobs"] = kwargs.get("n_jobs")
        return original_optimize(self, *args, **kwargs)

    monkeypatch.setattr(
        experiments_logic_module,
        "build_model",
        _fake_build_model,
    )
    monkeypatch.setattr(optuna.study.Study, "optimize", _wrapped_optimize)
    monkeypatch.setattr(experiments_logic_module.os, "cpu_count", lambda: 8)

    result = runner._optimize_controlled_combo(
        model_name="XGBoost",
        feature_set="Base",
        balance_mode="none",
        objective_metric="roc_auc",
        selected_features=base_cols[:2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=5,
        search_space_config=_controlled_search_space(),
        parallel_jobs=4,
        xgb_parallel_jobs=3,
    )

    assert captured["n_jobs"] == 5
    assert result["optuna_n_jobs"] == 5
    assert result["requested_optuna_n_jobs"] == 5
    assert result["optuna_jobs_cpu_cap"] == 8
