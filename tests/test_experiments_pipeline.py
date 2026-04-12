import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("optuna")
pytest.importorskip("imblearn")

import src.experiments_logic as experiments_logic_module
from src.experiments_logic import (
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

    expected_total = 3 * 2 * sum(len(values) for values in expected_by_set.values())
    assert len(grid_df) == expected_total
    assert len(summary_df) == 9
    assert {
        "k_optimo",
        "smote_optimo",
        "best_test_roc_auc",
        "val_roc_auc",
        "val_objective_score",
        "test_objective_score",
    }.issubset(
        summary_df.columns
    )
    assert summary_df["smote_optimo"].astype(bool).all()


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
    assert initial_calls == 18

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
    assert len(second_payload["grid_results_df"]) == 18


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
        selected_features=base_cols[:2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        search_space_config=_controlled_search_space(),
        parallel_jobs=3,
    )

    assert captured_params
    assert all(params["n_jobs"] == 3 for params in captured_params)
    assert result["parallel_jobs"] == 3


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
        selected_features=base_cols[:2],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=2,
        search_space_config=_controlled_search_space(),
        parallel_jobs=4,
    )

    assert captured["n_jobs"] == 2
    assert result["optuna_n_jobs"] == 2
