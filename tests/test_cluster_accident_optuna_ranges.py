from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("optuna")
pytest.importorskip("imblearn")

import src.cluster_accident_app as app


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}
        self.number_input_calls: list[dict] = []
        self.slider_calls: list[dict] = []

    def number_input(self, label, **kwargs):
        call = {"label": label, **kwargs}
        self.number_input_calls.append(call)
        key = kwargs.get("key")
        value = kwargs.get("value")
        if key is not None and key not in self.session_state:
            self.session_state[key] = value
        return self.session_state.get(key, value)

    def slider(self, label, **kwargs):
        call = {"label": label, **kwargs}
        self.slider_calls.append(call)
        key = kwargs.get("key")
        value = kwargs.get("value")
        if key is not None and key not in self.session_state:
            self.session_state[key] = value
        return self.session_state.get(key, value)


def _make_optuna_variant(
    *,
    model_choice: str = "XGBoost",
    balance_mode: str = "smote",
    calibration_method: str = "sigmoid",
    objective_metric: str = "balanced_f1",
    objective_label: str = "Balanced F1",
    threshold_objective: str = "far",
    best_score: float = 0.42,
    saved_at: str = "2026-04-21T10:00:00",
    include_trials_df: bool = False,
) -> dict:
    variant = {
        "model_choice": model_choice,
        "balance_mode": balance_mode,
        "calibration_method": calibration_method,
        "best_score": best_score,
        "best_smote_params": (
            {
                "smote_k_neighbors": 5,
                "smote_sampling_strategy": 0.2,
            }
            if balance_mode == "smote"
            else {}
        ),
        "best_model_params": {
            "n_estimators": 120,
            "max_depth": 4,
        },
        "optuna_settings": {
            "objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_SCALAR,
            "objective_metric": objective_metric,
            "objective_label": objective_label,
            "threshold_objective": threshold_objective,
            "calibration_method": calibration_method,
            "n_trials": 20,
            "timeout": 120,
            "n_jobs": 2,
            "random_state": 7,
            "test_size": 0.25,
            "val_size": 0.15,
            "far_target": 0.15,
            "alerts_per_day": 5.0,
            "fn_cost": 11.0,
            "fp_cost": 2.0,
            "tune_topk": True,
            "ranking_method": "rf",
            "ranking_method_label": "Random Forest (importancia)",
            "k_min": 2,
            "k_max": 5,
            "k_step": 1,
            "pruner": {
                "enabled": True,
                "startup_trials": 3,
            },
        },
        "objective_metric": objective_metric,
        "objective_label": objective_label,
        "search_space": {
            "model": {
                "n_estimators": {"min": 100, "max": 200, "step": 10},
            }
        },
        "saved_at": saved_at,
    }
    if include_trials_df:
        variant["trials_df"] = pd.DataFrame(
            [{"number": 0, "value": best_score, "state": "COMPLETE"}]
        )
    return variant


def _make_optuna_entry(
    *,
    optuna_id: str = "optuna_selector_test",
    feature_key: str = "features.duckdb",
    feature_cols: list[str] | None = None,
    dataset_fingerprint: str = "fp-1",
    variants: list[dict] | None = None,
) -> dict:
    feature_cols = feature_cols or ["signal"]
    raw_results: dict[str, dict] = {}
    for variant in variants or []:
        model_choice = str(variant.get("model_choice") or "XGBoost")
        balance_mode = str(variant.get("balance_mode") or "none")
        calibration_method = str(variant.get("calibration_method") or "none")
        model_container = raw_results.setdefault(
            model_choice,
            {
                "model_choice": model_choice,
                "by_balance_mode": {},
            },
        )
        by_balance_mode = model_container.setdefault("by_balance_mode", {})
        mode_container = by_balance_mode.setdefault(
            balance_mode,
            {
                "balance_mode": balance_mode,
                "by_calibration_method": {},
            },
        )
        by_calibration = mode_container.setdefault("by_calibration_method", {})
        by_calibration[calibration_method] = dict(variant)
    return {
        "optuna_id": optuna_id,
        "feature_key": feature_key,
        "feature_id": "feature_id",
        "features_path": feature_key,
        "features_source": "duckdb",
        "feature_cols": list(feature_cols),
        "selected_features": list(feature_cols),
        "dataset_fingerprint": dataset_fingerprint,
        "results": app._normalize_optuna_results_payload(raw_results),
        "saved_at": "2026-04-21T10:00:00",
    }


def _write_optuna_payload(tmp_path: Path, optuna_id: str, payload: dict) -> Path:
    path = tmp_path / f"optuna_{optuna_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

    def slider(self, label, **kwargs):
        call = {"label": label, **kwargs}
        self.slider_calls.append(call)
        key = kwargs.get("key")
        value = kwargs.get("value")
        if key is not None and key not in self.session_state:
            self.session_state[key] = value
        return self.session_state.get(key, value)


def test_suggest_optuna_discrete_int_preserves_non_divisible_upper_bound():
    optuna = pytest.importorskip("optuna")
    trial = optuna.trial.FixedTrial({"top_k": 41})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = app._suggest_optuna_discrete_int(
            trial,
            "top_k",
            10,
            41,
            step=10,
        )

    assert value == 41
    assert not any(
        "not divisible by `step`" in str(warning.message)
        for warning in caught
    )


def test_suggest_optuna_discrete_int_uses_divisible_grid_directly():
    optuna = pytest.importorskip("optuna")
    trial = optuna.trial.FixedTrial({"rf_n_estimators": 40})

    value = app._suggest_optuna_discrete_int(
        trial,
        "rf_n_estimators",
        10,
        40,
        step=10,
    )

    assert value == 40


def test_threshold_field_visibility_for_objective_matches_expected_dependencies():
    assert app._threshold_field_visibility_for_objective("far") == {
        "far_target": True,
        "alerts_per_day": False,
        "fn_cost": False,
        "fp_cost": False,
    }
    assert app._threshold_field_visibility_for_objective(
        "recall_at_alerts_per_day"
    ) == {
        "far_target": False,
        "alerts_per_day": True,
        "fn_cost": False,
        "fp_cost": False,
    }
    assert app._threshold_field_visibility_for_objective("operational_cost") == {
        "far_target": False,
        "alerts_per_day": True,
        "fn_cost": True,
        "fp_cost": True,
    }
    assert app._threshold_field_visibility_for_objective("roc_auc") == {
        "far_target": False,
        "alerts_per_day": False,
        "fn_cost": False,
        "fp_cost": False,
    }


def test_threshold_field_visibility_for_strategy_matches_expected_dependencies():
    assert app._threshold_field_visibility_for_strategy("far") == {
        "far_target": True,
        "alerts_per_day": False,
        "fn_cost": False,
        "fp_cost": False,
    }
    assert app._threshold_field_visibility_for_strategy("Calibrar por FAR") == {
        "far_target": True,
        "alerts_per_day": False,
        "fn_cost": False,
        "fp_cost": False,
    }
    assert app._threshold_field_visibility_for_strategy("optuna") == {
        "far_target": False,
        "alerts_per_day": False,
        "fn_cost": False,
        "fp_cost": False,
    }


def test_calibration_sweep_threshold_objectives_only_expose_operational_metrics():
    values = set(app._calibration_sweep_threshold_objective_options().values())

    assert values == {
        "far",
        "f1",
        "balanced_f1",
        "mcc",
        "recall_at_alerts_per_day",
        "operational_cost",
    }
    assert "pr_auc" not in values
    assert "roc_auc" not in values


def test_calibration_sweep_optuna_objectives_shortlist_and_advanced_catalog():
    default_values = set(
        app._calibration_sweep_optuna_objective_options().values()
    )
    advanced_values = set(
        app._calibration_sweep_optuna_objective_options(
            include_advanced=True
        ).values()
    )

    assert {
        "pr_auc",
        "mcc",
        "brier_score",
        "balanced_f1",
        "recall_at_alerts_per_day",
        "operational_cost",
        "far_sens",
    }.issubset(default_values)
    assert "roc_auc" not in default_values
    assert "roc_auc" in advanced_values
    assert "fnr" in advanced_values


def test_calibration_sweep_protocol_preview_uses_multiobjective_version():
    protocol = app._calibration_sweep_protocol_preview(
        model_name="Random Forest",
        feature_source="feature_selection",
        optuna_objective_mode="multiobjective",
        candidate_feature_cols=["signal", "aux_signal"],
        feature_k_config={"mode": "fixed_top_k", "k": 2, "ranking_method": "rf"},
        objective_metrics=["multiobjective_pareto"],
        calibration_methods=["sigmoid"],
        threshold_objectives=["far"],
        test_size=0.2,
        val_size=0.2,
        n_trials=1,
        timeout=30,
        optuna_n_jobs=1,
        parallel_jobs=1,
        xgb_parallel_jobs=1,
        far_target=0.2,
        alerts_per_day=5.0,
        fn_cost=10.0,
        fp_cost=1.0,
        robust_folds=3,
        search_space={},
        optuna_pruning_config={},
        random_state=42,
        segment_info={},
        event_path="events.csv",
        features_path="features.duckdb",
        dataset_date_start=None,
        dataset_date_end=None,
    )

    assert protocol["protocol_version"] == app.CALIBRATION_SWEEP_MULTIOBJECTIVE_PROTOCOL_VERSION
    assert protocol["optuna_objective_mode"] == app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
    assert protocol["multiobjective_metrics"] == list(app.CALIBRATION_SWEEP_MULTIOBJECTIVE_METRICS)


def test_optuna_objective_mode_options_include_scalar_and_multiobjective():
    options = app._optuna_objective_mode_options()

    assert options["Escalar legacy"] == app.CALIBRATION_SWEEP_OBJECTIVE_MODE_SCALAR
    assert (
        options["Multiobjetivo Pareto"]
        == app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
    )


def test_build_optuna_previous_result_label_uses_expected_format_and_fallbacks():
    assert (
        app._build_optuna_previous_result_label(
            objective_label="Balanced F1",
            objective_metric="balanced_f1",
            calibration_method="Platt scaling (sigmoid)",
            threshold_objective="recall@n",
            optuna_id="optuna_123",
        )
        == "Balanced F1-sigmoid-recall_at_alerts_per_day-optuna_123"
    )
    assert (
        app._build_optuna_previous_result_label(
            objective_label=None,
            objective_metric="f1",
            calibration_method=None,
            threshold_objective=None,
            optuna_id="legacy_id",
        )
        == "f1-none---legacy_id"
    )


def test_list_optuna_previous_result_options_supports_normalized_and_legacy_payloads(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    normalized_entry = _make_optuna_entry(
        optuna_id="normalized_id",
        variants=[
            _make_optuna_variant(balance_mode="none", saved_at="2026-04-21T09:00:00"),
            _make_optuna_variant(balance_mode="smote", saved_at="2026-04-21T10:00:00"),
        ],
    )
    legacy_payload = {
        "optuna_id": "legacy_id",
        "feature_key": "legacy_features.duckdb",
        "feature_cols": ["signal"],
        "dataset_fingerprint": "fp-legacy",
        "model_choice": "Random Forest",
        "best_score": 0.31,
        "best_smote_params": {},
        "best_model_params": {"n_estimators": 100},
        "optuna_settings": {
            "objective_metric": "f1",
            "objective_label": "F1",
        },
        "search_space": {},
        "saved_at": "2026-04-20T08:00:00",
    }

    _write_optuna_payload(tmp_path, "normalized_id", normalized_entry)
    _write_optuna_payload(tmp_path, "legacy_id", legacy_payload)

    options = app._list_optuna_previous_result_options()
    labels = {option["label"]: option for option in options}

    normalized_label = "Balanced F1-sigmoid-far-normalized_id"
    assert normalized_label in labels
    assert labels[normalized_label]["balance_mode"] == "smote"
    assert labels[normalized_label]["model_choice"] == "XGBoost"

    legacy_label = "F1-none---legacy_id"
    assert legacy_label in labels
    assert labels[legacy_label]["model_choice"] == "Random Forest"


def test_load_optuna_previous_result_selection_promotes_compatible_entry(
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    entry = _make_optuna_entry(
        feature_key="features.duckdb",
        feature_cols=["signal"],
        dataset_fingerprint="fp-1",
        variants=[_make_optuna_variant(include_trials_df=True)],
    )
    option = app._optuna_previous_result_options_from_entry(entry)[0]
    current_primary_key = app._optuna_result_key("features.duckdb", ["signal"])

    loaded = app._load_optuna_previous_result_selection(
        option,
        current_feature_key="features.duckdb",
        current_primary_key=current_primary_key,
        current_dataset_fingerprint="fp-1",
    )

    assert loaded["compatible"] is True
    assert fake_st.session_state["optuna_active_key"] == current_primary_key
    assert current_primary_key in fake_st.session_state["optuna_results_store"]
    assert fake_st.session_state["optuna_model_choice"] == "XGBoost"
    assert (
        fake_st.session_state["optuna_objective_mode_label"]
        == "Escalar legacy"
    )
    assert (
        fake_st.session_state["optuna_calibration_method"]
        == "Platt scaling (sigmoid)"
    )
    assert fake_st.session_state["optuna_threshold_objective"] == "FAR"
    assert fake_st.session_state["optuna_best_model_params"] == {
        "n_estimators": 120,
        "max_depth": 4,
    }
    assert isinstance(fake_st.session_state["optuna_trials_df"], pd.DataFrame)
    assert not fake_st.session_state["optuna_trials_df"].empty


def test_load_optuna_previous_result_selection_keeps_incompatible_entry_local(
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    entry = _make_optuna_entry(
        feature_key="features.duckdb",
        feature_cols=["signal"],
        dataset_fingerprint="fp-1",
        variants=[_make_optuna_variant(include_trials_df=True)],
    )
    option = app._optuna_previous_result_options_from_entry(entry)[0]
    incompatible_primary_key = app._optuna_result_key(
        "features.duckdb",
        ["signal", "aux_signal"],
    )

    loaded = app._load_optuna_previous_result_selection(
        option,
        current_feature_key="features.duckdb",
        current_primary_key=incompatible_primary_key,
        current_dataset_fingerprint="fp-2",
    )

    assert loaded["compatible"] is False
    assert fake_st.session_state.get("optuna_active_key") is None
    assert "optuna_results_store" not in fake_st.session_state
    assert loaded["reasons"]
    assert fake_st.session_state["optuna_loaded_result_state"]["compatible"] is False
    assert fake_st.session_state["optuna_best_model_params"] == {
        "n_estimators": 120,
        "max_depth": 4,
    }


def test_persist_optuna_results_keeps_multiobjective_metadata_and_pareto_csv(
    tmp_path,
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    features_df = pd.DataFrame(
        {
            "signal": [0.1, 0.9],
            "interval_start": pd.date_range("2024-01-01", periods=2, freq="D"),
        }
    )
    trials_df = pd.DataFrame(
        [
            {
                "number": 0,
                "pruning_proxy_score": 0.61,
                "pareto_front": True,
                "selected_trial": True,
            }
        ]
    )
    pareto_front_df = pd.DataFrame(
        [
            {
                "number": 0,
                "value_mcc": 0.40,
                "value_pr_auc": 0.52,
                "value_brier_score": 0.18,
                "value_recall_at_alerts_per_day": 0.60,
                "selected_trial": True,
            }
        ]
    )

    app._persist_optuna_results(
        optuna_key="optuna_test_key",
        optuna_id="optuna_test_id",
        feature_key="feature_key",
        feature_id="feature_id",
        features_path="features.duckdb",
        features_source="duckdb",
        features_df=features_df,
        selected_features=["signal"],
        feature_cols=["signal"],
        model_choice="XGBoost",
        balance_mode="none",
        calibration_method="sigmoid",
        best_score=0.61,
        best_smote_params={},
        best_model_params={"n_estimators": 120, "max_depth": 4},
        trials_df=trials_df,
        optuna_settings={
            "objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
            "objective_label": app.CALIBRATION_SWEEP_MULTIOBJECTIVE_LABEL,
        },
        search_space={"model": {"n_estimators": {"min": 100, "max": 200}}},
        extra_result_fields={
            "objective_metric": app.CALIBRATION_SWEEP_MULTIOBJECTIVE_KEY,
            "objective_label": app.CALIBRATION_SWEEP_MULTIOBJECTIVE_LABEL,
            "objective_direction": "multiobjective",
            "objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
            "optuna_objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
            "multiobjective_metrics": list(app.CALIBRATION_SWEEP_MULTIOBJECTIVE_METRICS),
            "multiobjective_directions": list(app.CALIBRATION_SWEEP_MULTIOBJECTIVE_DIRECTIONS),
            "objective_values": {
                "validation": {
                    "mcc": 0.40,
                    "pr_auc": 0.52,
                    "brier_score": 0.18,
                    "recall_at_alerts_per_day": 0.60,
                }
            },
            "pruning_proxy_score": 0.61,
            "far_gate_pass": True,
            "far_gate_fallback": False,
            "decision_threshold": 0.37,
            "best_trial_number": 0,
        },
        pareto_front_df=pareto_front_df,
    )

    store = fake_st.session_state["optuna_results_store"]
    entry = store["optuna_test_key"]
    variant = app._get_optuna_model_result_variant(
        entry["results"],
        model_choice="XGBoost",
        balance_mode="none",
        calibration_method="sigmoid",
    )

    assert variant is not None
    assert (
        variant["optuna_objective_mode"]
        == app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
    )
    assert variant["objective_values"]["validation"]["mcc"] == pytest.approx(0.40)
    assert variant["pruning_proxy_score"] == pytest.approx(0.61)
    assert variant["decision_threshold"] == pytest.approx(0.37)
    assert variant["pareto_front_csv"]
    assert (tmp_path / Path(str(variant["pareto_front_csv"])).name).exists()

    payload, _ = app._load_optuna_result_from_disk("optuna_test_id")
    assert payload is not None
    disk_variant = app._get_optuna_model_result_variant(
        payload["results"],
        model_choice="XGBoost",
        balance_mode="none",
        calibration_method="sigmoid",
    )
    assert disk_variant is not None
    assert disk_variant["objective_values"]["validation"]["pr_auc"] == pytest.approx(0.52)
    assert disk_variant["pareto_front_csv"]
    loaded_trials_df = app._load_optuna_variant_frame(
        disk_variant,
        frame_key="trials_df",
        csv_key="trials_csv",
    )
    loaded_pareto_df = app._load_optuna_variant_frame(
        disk_variant,
        frame_key="pareto_front_df",
        csv_key="pareto_front_csv",
    )
    assert isinstance(loaded_trials_df, pd.DataFrame)
    assert not loaded_trials_df.empty
    assert isinstance(loaded_pareto_df, pd.DataFrame)
    assert not loaded_pareto_df.empty


def test_list_experiment_result_files_includes_calibration_sweep_runs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    legacy_path = tmp_path / "experiments_results_20260101_010203.csv"
    legacy_path.write_text("type,best_f1\nbase,0.1\n", encoding="utf-8")

    completed_run = (
        tmp_path
        / "calibration_experiment_runs"
        / "calibration_sweep_20260418_151657_5f2b8a6f"
    )
    completed_results = completed_run / "results"
    completed_results.mkdir(parents=True)
    completed_best = completed_results / "best_summary.csv"
    completed_best.write_text("rank,model_name\n1,XGBoost\n", encoding="utf-8")
    (completed_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": completed_run.name,
                "status": "completed",
                "result_status": "completed",
                "completed_at": "2026-04-18T23:21:25",
            }
        ),
        encoding="utf-8",
    )

    partial_run = (
        tmp_path
        / "calibration_experiment_runs"
        / "calibration_sweep_20260418_151446_55cc815f"
    )
    partial_results = partial_run / "results"
    partial_results.mkdir(parents=True)
    partial_grid = partial_results / "grid_results.csv"
    partial_grid.write_text("status,model_name\ncompleted,XGBoost\n", encoding="utf-8")

    files = app._list_experiment_result_files()
    relative_files = {str(path.relative_to(tmp_path)) for path in files}

    assert (
        "calibration_experiment_runs/"
        "calibration_sweep_20260418_151657_5f2b8a6f/results/best_summary.csv"
    ) in relative_files
    assert (
        "calibration_experiment_runs/"
        "calibration_sweep_20260418_151446_55cc815f/results/grid_results.csv"
    ) in relative_files
    assert files[0] == completed_best


def test_calibration_sweep_history_helpers_build_labels_and_exports(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    run_dir = (
        tmp_path
        / "calibration_experiment_runs"
        / "calibration_sweep_20260418_151657_5f2b8a6f"
    )
    results_dir = run_dir / "results"
    trials_dir = run_dir / "trials"
    results_dir.mkdir(parents=True)
    trials_dir.mkdir()
    best_path = results_dir / "best_summary.csv"
    trial_path = trials_dir / "combo.csv"
    best_path.write_text("rank,model_name\n1,XGBoost\n", encoding="utf-8")
    trial_path.write_text("number,value\n0,0.5\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "completed",
                "result_status": "completed",
                "completed_at": "2026-04-18T23:21:25",
            }
        ),
        encoding="utf-8",
    )

    label = app._experiment_result_option_label(best_path)
    related_files = app._experiment_result_related_files(
        best_path,
        app._experiment_result_timestamp(best_path),
    )
    state = app._calibration_sweep_result_state_from_path(best_path)

    assert "Calibración score + threshold" in label
    assert "2026-04-18 23:21:25" in label
    assert "completado" in label
    assert app._experiment_result_timestamp(best_path) == "20260418_151657"
    assert best_path in related_files
    assert trial_path in related_files
    assert state["checkpoint_run_dir"] == str(run_dir)
    assert state["run_id"] == run_dir.name


def test_list_calibration_sweep_checkpoints_exposes_sorted_selector_entries(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    newer_run = (
        tmp_path
        / "calibration_experiment_runs"
        / "calibration_sweep_20260420_101500_newer"
    )
    older_run = (
        tmp_path
        / "calibration_experiment_runs"
        / "calibration_sweep_20260418_151657_older"
    )
    (newer_run / "results").mkdir(parents=True)
    (older_run / "results").mkdir(parents=True)
    (newer_run / "results" / "grid_results.csv").write_text(
        "status,model_name\nrunning,XGBoost\n",
        encoding="utf-8",
    )
    (older_run / "results" / "best_summary.csv").write_text(
        "rank,model_name\n1,Random Forest\n",
        encoding="utf-8",
    )
    (newer_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": newer_run.name,
                "status": "running",
                "result_status": "running",
                "updated_at": "2026-04-20T10:15:00",
                "progress": {
                    "completed_steps": 1,
                    "total_steps": 4,
                    "current_step_id": "combo__demo",
                },
            }
        ),
        encoding="utf-8",
    )
    (older_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": older_run.name,
                "status": "completed",
                "result_status": "completed",
                "completed_at": "2026-04-18T23:21:25",
                "progress": {
                    "completed_steps": 4,
                    "total_steps": 4,
                    "current_step_id": None,
                },
            }
        ),
        encoding="utf-8",
    )

    checkpoints = app._list_calibration_sweep_checkpoints()

    assert [item["run_id"] for item in checkpoints[:2]] == [
        newer_run.name,
        older_run.name,
    ]
    assert checkpoints[0]["status_label"] == "en progreso"
    assert checkpoints[0]["completed_steps"] == 1
    assert checkpoints[1]["status_label"] == "completado"
    assert "Calibración score + threshold" in checkpoints[0]["label"]


def test_render_conditional_number_input_preserves_hidden_value(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    first_value = app._render_conditional_number_input(
        "Costo FN",
        visible=True,
        min_value=0.0,
        value=10.0,
        step=1.0,
        key="optuna_fn_cost",
    )
    assert first_value == 10.0

    fake_st.session_state["optuna_fn_cost"] = 17.0
    updated_value = app._render_conditional_number_input(
        "Costo FN",
        visible=True,
        min_value=0.0,
        value=10.0,
        step=1.0,
        key="optuna_fn_cost",
    )
    assert updated_value == 17.0

    fake_st.session_state.pop("optuna_fn_cost", None)
    hidden_value = app._render_conditional_number_input(
        "Costo FN",
        visible=False,
        min_value=0.0,
        value=10.0,
        step=1.0,
        key="optuna_fn_cost",
    )
    assert hidden_value == 17.0

    restored_value = app._render_conditional_number_input(
        "Costo FN",
        visible=True,
        min_value=0.0,
        value=10.0,
        step=1.0,
        key="optuna_fn_cost",
    )
    assert restored_value == 17.0
    assert fake_st.number_input_calls[-1]["value"] == 17.0


def test_render_conditional_slider_preserves_hidden_value(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    first_value = app._render_conditional_slider(
        "FAR target",
        visible=True,
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.01,
        key="exp_far_target",
    )
    assert first_value == 0.2

    fake_st.session_state["exp_far_target"] = 0.33
    updated_value = app._render_conditional_slider(
        "FAR target",
        visible=True,
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.01,
        key="exp_far_target",
    )
    assert updated_value == pytest.approx(0.33)

    fake_st.session_state.pop("exp_far_target", None)
    hidden_value = app._render_conditional_slider(
        "FAR target",
        visible=False,
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.01,
        key="exp_far_target",
    )
    assert hidden_value == pytest.approx(0.33)

    restored_value = app._render_conditional_slider(
        "FAR target",
        visible=True,
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.01,
        key="exp_far_target",
    )
    assert restored_value == pytest.approx(0.33)
    assert fake_st.slider_calls[-1]["value"] == pytest.approx(0.33)
