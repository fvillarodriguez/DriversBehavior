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


class _ProgressBoardFakeElement:
    def __init__(self, root: "_ProgressBoardFakeStreamlit") -> None:
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def empty(self):
        return self

    def container(self):
        return self

    def caption(self, *args, **kwargs):
        self.root.caption_calls.append((args, kwargs))

    def metric(self, *args, **kwargs):
        self.root.metric_calls.append((args, kwargs))

    def progress(self, *args, **kwargs):
        self.root.progress_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.root.info_calls.append((args, kwargs))

    def success(self, *args, **kwargs):
        self.root.success_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.root.error_calls.append((args, kwargs))


class _ProgressBoardFakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}
        self.caption_calls: list[tuple] = []
        self.metric_calls: list[tuple] = []
        self.progress_calls: list[tuple] = []
        self.info_calls: list[tuple] = []
        self.success_calls: list[tuple] = []
        self.error_calls: list[tuple] = []
        self.selectbox_calls: list[dict] = []
        self.chart_calls: list[dict] = []
        self._seen_selectbox_keys: set[str] = set()

    def container(self):
        return _ProgressBoardFakeElement(self)

    def empty(self):
        return _ProgressBoardFakeElement(self)

    def columns(self, spec):
        count = int(spec) if isinstance(spec, int) else len(spec)
        return [_ProgressBoardFakeElement(self) for _ in range(count)]

    def caption(self, *args, **kwargs):
        self.caption_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def selectbox(self, label, options, *, index=0, key=None, **kwargs):
        if key in self._seen_selectbox_keys:
            raise AssertionError(f"duplicate selectbox key: {key}")
        self._seen_selectbox_keys.add(key)
        selected = self.session_state.get(key, options[index])
        self.session_state[key] = selected
        self.selectbox_calls.append(
            {
                "label": label,
                "options": list(options),
                "index": index,
                "key": key,
                **kwargs,
            }
        )
        return selected


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


def test_default_controlled_comparison_search_space_includes_nn_options():
    search_space = app._default_controlled_comparison_search_space()
    nn_space = search_space["nn"]

    assert nn_space["hidden_activation"] == [
        "relu",
        "gelu",
        "leaky_relu",
        "elu",
        "tanh",
    ]
    assert nn_space["output_activation"] == ["softmax", "sigmoid"]
    assert nn_space["loss_function"] == [
        "cross_entropy",
        "binary_cross_entropy",
        "focal",
    ]
    assert nn_space["optimizer_name"] == ["adamw", "adam", "rmsprop"]


def test_history_infer_optuna_feature_set_label_detects_base_cluster_mix():
    assert app._history_infer_optuna_feature_set_label(["flow_light", "speed_light"]) == "Base"
    assert app._history_infer_optuna_feature_set_label(
        ["last_cluster_flow_0", "next_cluster_entropy"]
    ) == "Cluster"
    assert app._history_infer_optuna_feature_set_label(
        ["flow_light", "next_cluster_entropy"]
    ) == "Base + Cluster"


def test_group_optuna_history_records_uses_explicit_batch_key_and_legacy_fallback():
    records = [
        {
            "id": 313,
            "stage": "Optuna",
            "batch_key": "optuna-batch-a",
            "created_at": "2026-04-23T16:01:27",
            "features_path": "features.duckdb",
            "tramo_label": "15 -> 11",
            "model_name": "XGBoost",
            "optuna_objective": "Balanced F1",
            "threshold_objective": "far",
            "calibration_method": "isotonic",
        },
        {
            "id": 312,
            "stage": "Optuna",
            "batch_key": "optuna-batch-a",
            "created_at": "2026-04-23T15:21:41",
            "features_path": "features.duckdb",
            "tramo_label": "15 -> 11",
            "model_name": "XGBoost",
            "optuna_objective": "Balanced F1",
            "threshold_objective": "far",
            "calibration_method": "isotonic",
        },
        {
            "id": 311,
            "stage": "Optuna",
            "batch_key": None,
            "created_at": "2026-04-23T14:49:40",
            "features_path": "features.duckdb",
            "tramo_label": "15 -> 11",
            "model_name": "XGBoost",
            "optuna_objective": "Balanced F1",
            "threshold_objective": "far",
            "calibration_method": "isotonic",
        },
        {
            "id": 310,
            "stage": "Optuna",
            "batch_key": None,
            "created_at": "2026-04-23T14:08:06",
            "features_path": "features.duckdb",
            "tramo_label": "15 -> 11",
            "model_name": "XGBoost",
            "optuna_objective": "Balanced F1",
            "threshold_objective": "far",
            "calibration_method": "isotonic",
        },
        {
            "id": 309,
            "stage": "Optuna",
            "batch_key": None,
            "created_at": "2026-04-23T10:00:00",
            "features_path": "features.duckdb",
            "tramo_label": "15 -> 11",
            "model_name": "XGBoost",
            "optuna_objective": "Balanced F1",
            "threshold_objective": "far",
            "calibration_method": "isotonic",
        },
    ]

    groups = app._group_optuna_history_records(records)

    assert len(groups) == 3
    assert [record["id"] for record in groups[0]["records"]] == [313, 312]
    assert [record["id"] for record in groups[1]["records"]] == [311, 310]
    assert [record["id"] for record in groups[2]["records"]] == [309]


def test_optuna_parameter_profile_defaults_keep_current_wide_profile():
    defaults = app._optuna_parameter_profile_defaults(
        app.OPTUNA_PARAMETER_PROFILE_WIDE,
        model_choice="XGBoost",
    )

    assert defaults["optuna_pruner_startup_trials"] == 5
    assert defaults["optuna_tune_topk"] is False
    assert defaults["optuna_k_min"] == 3
    assert defaults["optuna_k_max"] == 20
    assert defaults["optuna_xgb_n_min"] == 100
    assert defaults["optuna_xgb_n_max"] == 500
    assert defaults["optuna_xgb_depth_min"] == 2
    assert defaults["optuna_xgb_depth_max"] == 10
    assert defaults["optuna_xgb_lr_min"] == pytest.approx(0.01)
    assert defaults["optuna_xgb_lr_max"] == pytest.approx(0.30)
    assert defaults["optuna_xgb_gamma_max"] == pytest.approx(5.0)


def test_optuna_parameter_profile_defaults_refine_xgboost_local_profile():
    defaults = app._optuna_parameter_profile_defaults(
        app.OPTUNA_PARAMETER_PROFILE_LOCAL,
        model_choice="XGBoost",
    )

    assert defaults["optuna_pruner_startup_trials"] == 20
    assert defaults["optuna_tune_topk"] is True
    assert defaults["optuna_k_min"] == 16
    assert defaults["optuna_k_max"] == 20
    assert defaults["optuna_xgb_n_min"] == 80
    assert defaults["optuna_xgb_n_max"] == 400
    assert defaults["optuna_xgb_n_step"] == 20
    assert defaults["optuna_xgb_depth_min"] == 5
    assert defaults["optuna_xgb_depth_max"] == 7
    assert defaults["optuna_xgb_lr_min"] == pytest.approx(0.18)
    assert defaults["optuna_xgb_lr_max"] == pytest.approx(0.23)
    assert defaults["optuna_xgb_sub_min"] == pytest.approx(0.60)
    assert defaults["optuna_xgb_sub_max"] == pytest.approx(0.70)
    assert defaults["optuna_xgb_col_min"] == pytest.approx(0.75)
    assert defaults["optuna_xgb_col_max"] == pytest.approx(0.85)
    assert defaults["optuna_xgb_reg_alpha_min"] == pytest.approx(0.7)
    assert defaults["optuna_xgb_reg_alpha_max"] == pytest.approx(3.0)
    assert defaults["optuna_xgb_reg_lambda_min"] == pytest.approx(4.5)
    assert defaults["optuna_xgb_reg_lambda_max"] == pytest.approx(12.0)
    assert defaults["optuna_xgb_gamma_min"] == pytest.approx(3.8)
    assert defaults["optuna_xgb_gamma_max"] == pytest.approx(4.8)


def test_optuna_parameter_profile_defaults_xgboost_base_cluster_none_profile():
    defaults = app._optuna_parameter_profile_defaults(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_NONE_15141211,
        model_choice="XGBoost",
    )

    assert defaults["optuna_pruner_startup_trials"] == 30
    assert defaults["optuna_tune_topk"] is True
    assert defaults["optuna_k_min"] == 60
    assert defaults["optuna_k_max"] == 100
    assert defaults["optuna_xgb_n_min"] == 400
    assert defaults["optuna_xgb_n_max"] == 800
    assert defaults["optuna_xgb_depth_min"] == 2
    assert defaults["optuna_xgb_depth_max"] == 4
    assert defaults["optuna_xgb_lr_min"] == pytest.approx(0.02)
    assert defaults["optuna_xgb_lr_max"] == pytest.approx(0.07)
    assert defaults["optuna_xgb_reg_lambda_min"] == pytest.approx(6.0)
    assert defaults["optuna_xgb_reg_lambda_max"] == pytest.approx(14.0)
    assert defaults["optuna_xgb_gamma_min"] == pytest.approx(3.5)
    assert defaults["optuna_xgb_gamma_max"] == pytest.approx(6.0)


def test_optuna_parameter_profile_defaults_xgboost_base_cluster_smote_profile():
    defaults = app._optuna_parameter_profile_defaults(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_SMOTE_15141211,
        model_choice="XGBoost",
    )

    assert defaults["optuna_pruner_startup_trials"] == 30
    assert defaults["optuna_tune_topk"] is True
    assert defaults["optuna_k_min"] == 10
    assert defaults["optuna_k_max"] == 30
    assert defaults["optuna_smote_k_min"] == 5
    assert defaults["optuna_smote_k_max"] == 9
    assert defaults["optuna_smote_sampling_min"] == pytest.approx(0.01)
    assert defaults["optuna_smote_sampling_max"] == pytest.approx(0.10)
    assert defaults["optuna_xgb_n_min"] == 50
    assert defaults["optuna_xgb_n_max"] == 200
    assert defaults["optuna_xgb_sub_min"] == pytest.approx(0.50)
    assert defaults["optuna_xgb_sub_max"] == pytest.approx(0.70)
    assert defaults["optuna_xgb_col_min"] == pytest.approx(0.90)
    assert defaults["optuna_xgb_col_max"] == pytest.approx(1.00)
    assert defaults["optuna_xgb_gamma_min"] == pytest.approx(0.0)
    assert defaults["optuna_xgb_gamma_max"] == pytest.approx(2.0)


def test_optuna_base_cluster_profiles_limit_feature_set_options():
    available = ["Base", "Cluster", "Base + Cluster"]

    assert app._optuna_parameter_profile_feature_set_options(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_NONE_15141211,
        available,
    ) == ["Base + Cluster"]
    assert app._optuna_parameter_profile_feature_set_options(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_SMOTE_15141211,
        available,
    ) == ["Base + Cluster"]
    assert app._optuna_parameter_profile_feature_set_options(
        app.OPTUNA_PARAMETER_PROFILE_WIDE,
        available,
    ) == ["Base", "Cluster", "Base + Cluster"]


def test_optuna_base_cluster_profiles_force_balance_options():
    assert app._optuna_parameter_profile_forced_balance_modes(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_NONE_15141211
    ) == ["none"]
    assert app._optuna_parameter_profile_balance_options(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_NONE_15141211
    ) == ["none"]
    assert app._optuna_parameter_profile_forced_balance_modes(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_SMOTE_15141211
    ) == ["smote"]
    assert app._optuna_parameter_profile_balance_options(
        app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_SMOTE_15141211
    ) == ["SMOTE"]


def test_apply_optuna_parameter_profile_scope_clears_stale_widget_values():
    session_state = {
        "optuna_feature_sets_selected": ["Base", "Base + Cluster"],
        "optuna_balance_modes_selected": ["none", "SMOTE"],
    }

    app._apply_optuna_parameter_profile_scope(
        session_state,
        profile_label=app.OPTUNA_PARAMETER_PROFILE_XGB_BASE_CLUSTER_SMOTE_15141211,
        available_feature_sets=["Base", "Cluster", "Base + Cluster"],
    )

    assert session_state["optuna_feature_sets_selected"] == ["Base + Cluster"]
    assert session_state["optuna_balance_modes_selected"] == ["SMOTE"]


def test_optuna_balance_mode_keeps_class_weight_as_distinct_mode():
    assert app._normalize_optuna_balance_mode("class_weight") == "class_weight"
    assert app._normalize_optuna_balance_mode("Class weight") == "class_weight"
    assert app._optuna_normalize_balance_mode_selection(
        ["SMOTE", "none", "class_weight", "SMOTE"]
    ) == ["none", "class_weight", "smote"]


def test_apply_optuna_parameter_profile_updates_session_state():
    session_state = {
        "optuna_tune_topk": False,
        "optuna_xgb_n_min": 100,
        "optuna_pruner_startup_trials": 5,
    }

    app._apply_optuna_parameter_profile(
        session_state,
        profile_label=app.OPTUNA_PARAMETER_PROFILE_LOCAL,
        model_choice="XGBoost",
    )

    assert session_state["optuna_tune_topk"] is True
    assert session_state["optuna_k_min"] == 16
    assert session_state["optuna_xgb_n_min"] == 80
    assert session_state["optuna_xgb_lr_min"] == pytest.approx(0.18)
    assert session_state["optuna_pruner_startup_trials"] == 20


def test_apply_model_params_to_prefix_normalizes_nn_optuna_aliases(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    app._apply_model_params_to_prefix(
        model_choice="Neural Network",
        prefix="cluster_",
        params={
            "hidden_activation": "LeakyReLU",
            "hidden_dim": 16,
            "num_layers": 2,
            "output_activation": "Sigmoid",
            "dropout": 0.1,
            "learning_rate": 0.005,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "optimizer": "Adam",
            "loss_function": "MCE",
            "use_batch_norm": "true",
            "lr_scheduler": "reduce_lr_on_plateau",
            "temperature_scaling": "false",
        },
    )

    assert fake_st.session_state["cluster_model_nn_hidden_activation"] == "leaky_relu"
    assert fake_st.session_state["cluster_model_nn_output_activation"] == "sigmoid"
    assert fake_st.session_state["cluster_model_nn_optimizer_name"] == "adam"
    assert fake_st.session_state["cluster_model_nn_loss_function"] == "cross_entropy"
    assert fake_st.session_state["cluster_model_nn_use_batch_norm"] is True
    assert fake_st.session_state["cluster_model_nn_lr_scheduler"] == (
        "reduce_on_plateau"
    )
    assert fake_st.session_state["cluster_model_nn_temperature_scaling"] is False


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
    assert (
        fake_st.session_state["optuna_last_optimized_feature_key"]
        == "features.duckdb"
    )
    assert (
        fake_st.session_state["optuna_last_optimized_dataset_fingerprint"]
        == "fp-1"
    )


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


def test_history_optuna_previous_options_from_records_groups_batch_key():
    base_none_variant = _make_optuna_variant(
        balance_mode="none",
        saved_at="2026-04-23T12:44:40",
    )
    base_smote_variant = _make_optuna_variant(
        balance_mode="smote",
        saved_at="2026-04-23T13:31:02",
    )
    cluster_none_variant = _make_optuna_variant(
        balance_mode="none",
        saved_at="2026-04-23T14:08:06",
    )
    cluster_smote_variant = _make_optuna_variant(
        balance_mode="smote",
        saved_at="2026-04-23T14:49:40",
    )
    base_cluster_none_variant = _make_optuna_variant(
        balance_mode="none",
        saved_at="2026-04-23T15:21:41",
    )
    base_cluster_smote_variant = _make_optuna_variant(
        balance_mode="smote",
        saved_at="2026-04-23T16:01:27",
    )
    base_entry_older = _make_optuna_entry(
        optuna_id="optuna_base",
        feature_key="features.duckdb",
        feature_cols=["signal"],
        dataset_fingerprint="fp-1",
        variants=[base_none_variant],
    )
    base_entry_latest = _make_optuna_entry(
        optuna_id="optuna_base",
        feature_key="features.duckdb",
        feature_cols=["signal"],
        dataset_fingerprint="fp-1",
        variants=[base_none_variant, base_smote_variant],
    )
    cluster_entry_older = _make_optuna_entry(
        optuna_id="optuna_cluster",
        feature_key="features.duckdb",
        feature_cols=["cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[cluster_none_variant],
    )
    cluster_entry_latest = _make_optuna_entry(
        optuna_id="optuna_cluster",
        feature_key="features.duckdb",
        feature_cols=["cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[cluster_none_variant, cluster_smote_variant],
    )
    base_cluster_entry_older = _make_optuna_entry(
        optuna_id="optuna_base_cluster",
        feature_key="features.duckdb",
        feature_cols=["signal", "cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[base_cluster_none_variant],
    )
    base_cluster_entry_latest = _make_optuna_entry(
        optuna_id="optuna_base_cluster",
        feature_key="features.duckdb",
        feature_cols=["signal", "cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[base_cluster_none_variant, base_cluster_smote_variant],
    )

    batch_key = "optuna-batch-retrofit-1"
    requested_signature = app.history_store.feature_signature(
        ["signal", "cluster_signal"]
    )
    records = [
        {
            "id": 313,
            "batch_key": batch_key,
            "feature_signature": requested_signature,
            "model_name": "XGBoost",
            "balance_strategy": "smote",
            "calibration_method": "sigmoid",
            "threshold_objective": "far",
            "optuna_objective": "Balanced F1",
            "created_at": "2026-04-23T16:01:27",
            "metadata": {
                "store_entry": base_cluster_entry_latest,
                "variant": base_cluster_smote_variant,
                "optuna_batch_key": batch_key,
                "optuna_id": "optuna_base_cluster",
                "feature_cols": ["signal", "cluster_signal"],
            },
        },
        {
            "id": 312,
            "batch_key": batch_key,
            "feature_signature": requested_signature,
            "model_name": "XGBoost",
            "balance_strategy": "none",
            "calibration_method": "sigmoid",
            "threshold_objective": "far",
            "optuna_objective": "Balanced F1",
            "created_at": "2026-04-23T15:21:41",
            "metadata": {
                "store_entry": base_cluster_entry_older,
                "variant": base_cluster_none_variant,
                "optuna_batch_key": batch_key,
                "optuna_id": "optuna_base_cluster",
                "feature_cols": ["signal", "cluster_signal"],
            },
        },
        {
            "id": 311,
            "batch_key": batch_key,
            "feature_signature": app.history_store.feature_signature(
                ["cluster_signal"]
            ),
            "model_name": "XGBoost",
            "balance_strategy": "smote",
            "calibration_method": "sigmoid",
            "threshold_objective": "far",
            "optuna_objective": "Balanced F1",
            "created_at": "2026-04-23T14:49:40",
            "metadata": {
                "store_entry": cluster_entry_latest,
                "variant": cluster_smote_variant,
                "optuna_batch_key": batch_key,
                "optuna_id": "optuna_cluster",
                "feature_cols": ["cluster_signal"],
            },
        },
        {
            "id": 310,
            "batch_key": batch_key,
            "feature_signature": app.history_store.feature_signature(
                ["cluster_signal"]
            ),
            "model_name": "XGBoost",
            "balance_strategy": "none",
            "calibration_method": "sigmoid",
            "threshold_objective": "far",
            "optuna_objective": "Balanced F1",
            "created_at": "2026-04-23T14:08:06",
            "metadata": {
                "store_entry": cluster_entry_older,
                "variant": cluster_none_variant,
                "optuna_batch_key": batch_key,
                "optuna_id": "optuna_cluster",
                "feature_cols": ["cluster_signal"],
            },
        },
        {
            "id": 308,
            "batch_key": batch_key,
            "feature_signature": app.history_store.feature_signature(["signal"]),
            "model_name": "XGBoost",
            "balance_strategy": "none",
            "calibration_method": "sigmoid",
            "threshold_objective": "far",
            "optuna_objective": "Balanced F1",
            "created_at": "2026-04-23T12:44:40",
            "metadata": {
                "store_entry": base_entry_older,
                "variant": base_none_variant,
                "optuna_batch_key": batch_key,
                "optuna_id": "optuna_base",
                "feature_cols": ["signal"],
            },
        },
        {
            "id": 309,
            "batch_key": batch_key,
            "feature_signature": app.history_store.feature_signature(["signal"]),
            "model_name": "XGBoost",
            "balance_strategy": "smote",
            "calibration_method": "sigmoid",
            "threshold_objective": "far",
            "optuna_objective": "Balanced F1",
            "created_at": "2026-04-23T13:31:02",
            "metadata": {
                "store_entry": base_entry_latest,
                "variant": base_smote_variant,
                "optuna_batch_key": batch_key,
                "optuna_id": "optuna_base",
                "feature_cols": ["signal"],
            },
        },
    ]

    options = app._history_optuna_previous_options_from_records(
        records,
        feature_signature_value=requested_signature,
    )

    assert len(options) == 1
    option = options[0]
    assert option["batch_key"] == batch_key
    assert option["subrun_count"] == 6
    assert option["record_ids"] == [308, 309, 310, 311, 312, 313]
    assert len(option["entries"]) == 3
    assert {tuple(entry["feature_cols"]) for entry in option["entries"]} == {
        ("signal",),
        ("cluster_signal",),
        ("signal", "cluster_signal"),
    }
    assert option["entry"]["feature_cols"] == ["signal", "cluster_signal"]
    assert (
        app._get_optuna_model_result_variant(
            option["entry"]["results"],
            model_choice="XGBoost",
            balance_mode="smote",
            calibration_method="sigmoid",
        )
        is not None
    )
    assert "batch 313-308" in option["label"]


def test_load_optuna_previous_result_selection_promotes_compatible_batch(
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    base_none_variant = _make_optuna_variant(balance_mode="none")
    base_smote_variant = _make_optuna_variant(balance_mode="smote")
    cluster_none_variant = _make_optuna_variant(balance_mode="none")
    cluster_smote_variant = _make_optuna_variant(balance_mode="smote")
    base_cluster_none_variant = _make_optuna_variant(balance_mode="none")
    base_cluster_smote_variant = _make_optuna_variant(
        balance_mode="smote",
        include_trials_df=True,
    )
    base_entry_older = _make_optuna_entry(
        optuna_id="optuna_base",
        feature_key="features.duckdb",
        feature_cols=["signal"],
        dataset_fingerprint="fp-1",
        variants=[base_none_variant],
    )
    base_entry_latest = _make_optuna_entry(
        optuna_id="optuna_base",
        feature_key="features.duckdb",
        feature_cols=["signal"],
        dataset_fingerprint="fp-1",
        variants=[base_none_variant, base_smote_variant],
    )
    cluster_entry_older = _make_optuna_entry(
        optuna_id="optuna_cluster",
        feature_key="features.duckdb",
        feature_cols=["cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[cluster_none_variant],
    )
    cluster_entry_latest = _make_optuna_entry(
        optuna_id="optuna_cluster",
        feature_key="features.duckdb",
        feature_cols=["cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[cluster_none_variant, cluster_smote_variant],
    )
    base_cluster_entry_older = _make_optuna_entry(
        optuna_id="optuna_base_cluster",
        feature_key="features.duckdb",
        feature_cols=["signal", "cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[base_cluster_none_variant],
    )
    base_cluster_entry_latest = _make_optuna_entry(
        optuna_id="optuna_base_cluster",
        feature_key="features.duckdb",
        feature_cols=["signal", "cluster_signal"],
        dataset_fingerprint="fp-1",
        variants=[base_cluster_none_variant, base_cluster_smote_variant],
    )

    option = {
        "token": "history_batch_batch_1",
        "label": "Balanced F1-sigmoid-far-optuna_batch | batch 313-308 | 6 subcorridas",
        "batch_key": "optuna-batch-1",
        "subrun_count": 6,
        "entries": [
            base_cluster_entry_latest,
            base_cluster_entry_older,
            cluster_entry_latest,
            cluster_entry_older,
            base_entry_latest,
            base_entry_older,
        ],
        "entry": base_cluster_entry_latest,
        "variant": app._get_optuna_model_result_variant(
            base_cluster_entry_latest["results"],
            model_choice="XGBoost",
            balance_mode="smote",
            calibration_method="sigmoid",
        ),
        "optuna_id": "optuna_batch",
        "model_choice": "XGBoost",
        "balance_mode": "smote",
        "balance_mode_label": "Con SMOTE",
        "calibration_method": "sigmoid",
        "threshold_objective": "far",
        "objective_metric": "Balanced F1",
        "objective_label": "Balanced F1",
        "feature_set_labels": ["Base", "Cluster", "Base + Cluster"],
    }
    current_primary_key = app._optuna_result_key(
        "features.duckdb",
        ["signal", "cluster_signal"],
    )
    base_key = app._optuna_result_key("features.duckdb", ["signal"])
    cluster_key = app._optuna_result_key("features.duckdb", ["cluster_signal"])

    loaded = app._load_optuna_previous_result_selection(
        option,
        current_feature_key="features.duckdb",
        current_primary_key=current_primary_key,
        current_dataset_fingerprint="fp-1",
    )

    assert loaded["compatible"] is True
    assert fake_st.session_state["optuna_active_key"] == current_primary_key
    assert len(fake_st.session_state["optuna_results_store"]) == 3
    assert current_primary_key in fake_st.session_state["optuna_results_store"]
    assert base_key in fake_st.session_state["optuna_results_store"]
    assert cluster_key in fake_st.session_state["optuna_results_store"]
    assert fake_st.session_state["optuna_feature_sets_selected"] == [
        "Base",
        "Cluster",
        "Base + Cluster",
    ]
    assert fake_st.session_state["optuna_loaded_result_state"]["batch_key"] == (
        "optuna-batch-1"
    )
    assert fake_st.session_state["optuna_loaded_result_state"]["subrun_count"] == 6
    assert loaded["feature_set_labels"] == ["Base", "Cluster", "Base + Cluster"]
    assert loaded["balance_modes"] == ["none", "smote"]
    assert fake_st.session_state["optuna_loaded_feature_sets_sync_pending"] is True
    assert fake_st.session_state["optuna_best_model_params"] == {
        "n_estimators": 120,
        "max_depth": 4,
    }
    restored_entry = fake_st.session_state["optuna_results_store"][current_primary_key]
    assert (
        app._get_optuna_model_result_variant(
            restored_entry["results"],
            model_choice="XGBoost",
            balance_mode="smote",
            calibration_method="sigmoid",
        )
        is not None
    )
    assert (
        app._get_optuna_model_result_variant(
            restored_entry["results"],
            model_choice="XGBoost",
            balance_mode="none",
            calibration_method="sigmoid",
        )
        is not None
    )


def test_loaded_optuna_feature_sets_sync_replaces_stale_none(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "optuna_loaded_result_state": {
                "token": "history_batch",
                "batch_key": "optuna-batch-1",
                "feature_set_labels": ["Base", "Cluster", "Base + Cluster"],
            },
            "optuna_feature_sets_selected": ["none"],
            "optuna_loaded_feature_sets_sync_pending": True,
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    selected = app._sync_loaded_optuna_feature_sets_to_form(
        ["Base", "Cluster", "Base + Cluster"]
    )

    assert selected == ["Base", "Cluster", "Base + Cluster"]
    assert fake_st.session_state["optuna_feature_sets_selected"] == selected
    assert fake_st.session_state["optuna_loaded_feature_sets_sync_pending"] is False
    assert app._normalize_model_feature_set_labels("Base") == ["Base"]


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


def test_optuna_trial_metrics_dataframe_includes_val_and_test_metrics():
    class _State:
        name = "COMPLETE"

    class _Trial:
        number = 3
        value = 0.72
        values = None
        params = {"rf_n_estimators": 40}
        state = _State()

        def __init__(self) -> None:
            self.user_attrs = {}

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    trial = _Trial()
    app._record_optuna_trial_metrics(
        trial,
        trial_payload={
            "trial_cols": ["signal"],
            "model_params": {"n_estimators": 40},
        },
        scored_val={
            "score": 0.72,
            "threshold": 0.42,
            "metrics": {
                "accuracy": 0.8,
                "pr_auc": 0.7,
                "mcc": 0.3,
                "confusion_matrix": [[8, 1], [2, 3]],
            },
        },
        scored_test={
            "score": 0.62,
            "threshold": 0.42,
            "metrics": {
                "accuracy": 0.75,
                "pr_auc": 0.65,
                "mcc": 0.2,
                "confusion_matrix": [[7, 2], [2, 3]],
            },
        },
        objective_score=0.72,
        objective_metric="balanced_f1",
        objective_direction="maximize",
    )

    trials_df = app._optuna_trials_dataframe_from_trials(
        [trial],
        objective_direction="maximize",
        pruner_name="MedianPruner",
    )

    expected_cols = {
        "val_accuracy",
        "val_pr_auc",
        "val_mcc",
        "test_accuracy",
        "test_pr_auc",
        "test_mcc",
        "val_confusion_matrix",
        "test_confusion_matrix",
    }
    assert expected_cols.issubset(set(trials_df.columns))
    assert trials_df.loc[0, "val_mcc"] == pytest.approx(0.3)
    assert trials_df.loc[0, "test_pr_auc"] == pytest.approx(0.65)
    assert trials_df.loc[0, "val_confusion_matrix"] == [[8, 1], [2, 3]]
    assert trials_df.loc[0, "test_confusion_matrix"] == [[7, 2], [2, 3]]


def test_svm_gpu_parallel_optuna_scheduler_uses_ask_tell_and_worker_attrs():
    import concurrent.futures
    import optuna

    class _ImmediateExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = int(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def submit(self, fn, payload):
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(fn(payload))
            return future

    def _executor_factory(max_workers: int):
        return _ImmediateExecutor(max_workers)

    def _build_trial_payload(trial):
        c_value = trial.suggest_float("svm_C", 0.1, 0.3, step=0.1)
        return {
            "trial_cols": ["signal"],
            "model_params": {"kernel": "linear", "C": float(c_value)},
        }

    def _worker(payload):
        trial_payload = dict(payload["trial_payload"])
        score = float(trial_payload["model_params"]["C"])
        scored = {
            "score": score,
            "threshold": 0.5,
            "metrics": {
                "accuracy": score,
                "mcc": score,
                "pr_auc": score,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
        }
        return {
            "status": "completed",
            "score": score,
            "scored_val": scored,
            "scored_test": scored,
            "svm_backend": "mlx",
            "svm_fit_warning": "",
            "worker_pid": 1234,
            "execution_backend": "local_process_pool_spawn",
        }

    study = optuna.create_study(direction="maximize")
    app._run_svm_gpu_parallel_optuna(
        study=study,
        n_trials=3,
        timeout=30,
        max_workers=2,
        build_trial_payload=_build_trial_payload,
        worker_payload_base={},
        is_multiobjective=False,
        objective_metric="mcc",
        objective_direction="maximize",
        far_target=0.2,
        executor_factory=_executor_factory,
        worker_fn=_worker,
    )

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    assert len(completed) == 3
    assert all(trial.user_attrs["svm_backend"] == "mlx" for trial in completed)
    assert all(trial.user_attrs["worker_pid"] == 1234 for trial in completed)
    assert all(
        trial.user_attrs["execution_backend"] == "local_process_pool_spawn"
        for trial in completed
    )


def test_persist_optuna_results_writes_trials_json_and_best_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    features_df = pd.DataFrame(
        {
            "signal": [0.1, 0.9, 0.3],
            "interval_start": pd.date_range("2024-01-01", periods=3, freq="D"),
        }
    )
    trials_df = pd.DataFrame(
        [
            {
                "number": 0,
                "value": 0.55,
                "state": "COMPLETE",
                "val_accuracy": 0.70,
                "val_pr_auc": 0.50,
                "val_mcc": 0.10,
                "test_accuracy": 0.65,
                "test_pr_auc": 0.45,
                "test_mcc": 0.08,
                "val_confusion_matrix": [[5, 1], [2, 1]],
                "test_confusion_matrix": [[4, 2], [2, 1]],
            }
        ]
    )

    app._persist_optuna_results(
        optuna_key="optuna_base_key",
        optuna_id="optuna_base_id",
        feature_key="feature_base",
        feature_id="feature_base",
        features_path="features.duckdb",
        features_source="duckdb",
        features_df=features_df,
        selected_features=["signal"],
        feature_cols=["signal"],
        model_choice="Random Forest",
        balance_mode="none",
        calibration_method="none",
        best_score=0.55,
        best_smote_params={},
        best_model_params={"n_estimators": 40},
        trials_df=trials_df,
        optuna_settings={
            "objective_metric": "balanced_f1",
            "objective_label": "Balanced F1",
            "objective_direction": "maximize",
            "feature_set_label": "Base",
        },
        search_space={},
        best_summary={
            "feature_set_label": "Base",
            "model_choice": "Random Forest",
            "balance_mode": "none",
            "balance_mode_label": "Sin SMOTE",
            "calibration_method": "none",
            "calibration_method_label": "Sin calibración",
            "objective_metric": "balanced_f1",
            "objective_label": "Balanced F1",
            "objective_direction": "maximize",
            "best_score": 0.55,
            "best_trial_number": 0,
            "val_mcc": 0.10,
            "test_mcc": 0.08,
        },
    )
    app._persist_optuna_results(
        optuna_key="optuna_cluster_key",
        optuna_id="optuna_cluster_id",
        feature_key="feature_cluster",
        feature_id="feature_cluster",
        features_path="features.duckdb",
        features_source="duckdb",
        features_df=features_df,
        selected_features=["cluster_signal"],
        feature_cols=["cluster_signal"],
        model_choice="Random Forest",
        balance_mode="none",
        calibration_method="none",
        best_score=0.75,
        best_smote_params={},
        best_model_params={"n_estimators": 60},
        trials_df=trials_df.assign(value=0.75, val_mcc=0.2, test_mcc=0.15),
        optuna_settings={
            "objective_metric": "balanced_f1",
            "objective_label": "Balanced F1",
            "objective_direction": "maximize",
            "feature_set_label": "Cluster",
        },
        search_space={},
        best_summary={
            "feature_set_label": "Cluster",
            "model_choice": "Random Forest",
            "balance_mode": "none",
            "balance_mode_label": "Sin SMOTE",
            "calibration_method": "none",
            "calibration_method_label": "Sin calibración",
            "objective_metric": "balanced_f1",
            "objective_label": "Balanced F1",
            "objective_direction": "maximize",
            "best_score": 0.75,
            "best_trial_number": 0,
            "val_mcc": 0.20,
            "test_mcc": 0.15,
        },
    )

    payload = json.loads(
        (tmp_path / "optuna_optuna_base_id.json").read_text(encoding="utf-8")
    )
    payload_variant = app._get_optuna_model_result_variant(
        payload["results"],
        model_choice="Random Forest",
        balance_mode="none",
        calibration_method="none",
    )
    assert payload_variant is not None
    assert payload_variant["trials_json"]
    assert payload_variant["best_summary_json"]
    trials_json_path = Path(str(payload_variant["trials_json"]))
    summary_json_path = Path(str(payload_variant["best_summary_json"]))
    assert trials_json_path.exists()
    assert summary_json_path.exists()
    trials_payload = json.loads(trials_json_path.read_text(encoding="utf-8"))
    assert trials_payload["records"][0]["val_mcc"] == pytest.approx(0.10)
    summary_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary_payload["test_mcc"] == pytest.approx(0.08)

    rows = app._optuna_best_summary_rows_from_store(
        configs=[
            {"key": "optuna_base_key", "label": "Base"},
            {"key": "optuna_cluster_key", "label": "Cluster"},
        ],
        store=fake_st.session_state["optuna_results_store"],
        model_choice="Random Forest",
    )
    by_feature_set = {row["feature_set_label"]: row for row in rows}
    assert by_feature_set["Base"]["is_best_global"] is False
    assert by_feature_set["Cluster"]["is_best_global"] is True


def test_default_model_pareto_feature_sets_prefers_base_and_base_cluster():
    assert app._default_model_pareto_feature_sets(
        ["Base", "Cluster", "Base + Cluster"]
    ) == ["Base", "Base + Cluster"]
    assert app._default_model_pareto_feature_sets(["Cluster"]) == ["Cluster"]


def test_sanitize_optuna_feature_set_selection_prefers_base_and_base_cluster():
    assert app._sanitize_optuna_feature_set_selection(
        ["Base", "Cluster", "Base + Cluster"]
    ) == ["Base", "Base + Cluster"]
    assert app._sanitize_optuna_feature_set_selection(["Base"], ["Cluster"]) == [
        "Base"
    ]


def test_persist_optuna_results_stores_feature_set_context(
    tmp_path,
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    features_df = pd.DataFrame(
        {
            "flow_light": [1.0, 2.0, 3.0],
            "cluster_count_0": [0.1, 0.2, 0.3],
        }
    )
    feature_cols = ["flow_light", "cluster_count_0"]
    feature_key = "/tmp/flow_features.duckdb"
    optuna_key = app._optuna_result_key(feature_key, feature_cols)
    optuna_id = "optuna_feature_sets_ctx"
    optuna_settings = {
        "feature_set_label": "Base + Cluster",
        "selected_feature_sets_in_run": ["Base", "Base + Cluster"],
        "objective_metric": "balanced_f1",
        "objective_label": "Balanced F1",
    }

    app._persist_optuna_results(
        optuna_key=optuna_key,
        optuna_id=optuna_id,
        feature_key=feature_key,
        feature_id="feature_ctx",
        features_path=feature_key,
        features_source="duckdb",
        features_df=features_df,
        selected_features=feature_cols,
        feature_cols=feature_cols,
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        best_score=0.77,
        best_smote_params={"smote_k_neighbors": 5},
        best_model_params={"n_estimators": 120},
        trials_df=None,
        optuna_settings=optuna_settings,
        search_space={},
    )

    entry = fake_st.session_state["optuna_results_store"][optuna_key]
    variant = app._get_optuna_model_result_variant(
        entry["results"],
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
    )
    assert variant is not None
    assert variant["optuna_settings"]["feature_set_label"] == "Base + Cluster"
    assert variant["optuna_settings"]["selected_feature_sets_in_run"] == [
        "Base",
        "Base + Cluster",
    ]

    payload = json.loads(
        (tmp_path / f"optuna_{optuna_id}.json").read_text(encoding="utf-8")
    )
    payload_variant = app._get_optuna_model_result_variant(
        payload["results"],
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
    )
    assert payload_variant is not None
    assert payload_variant["optuna_settings"]["feature_set_label"] == "Base + Cluster"
    assert payload_variant["optuna_settings"]["selected_feature_sets_in_run"] == [
        "Base",
        "Base + Cluster",
    ]


def test_sync_optuna_legacy_top_level_state_hydrates_last_optimized_context(
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    variant = _make_optuna_variant()
    variant["optuna_settings"]["feature_set_label"] = "Base + Cluster"
    variant["optuna_settings"]["selected_feature_sets_in_run"] = [
        "Base",
        "Base + Cluster",
    ]
    entry = _make_optuna_entry(
        feature_key="/tmp/flow_features.duckdb",
        dataset_fingerprint="fp_ctx",
        feature_cols=["flow_light", "cluster_count_0"],
        variants=[variant],
    )

    model_result = app._sync_optuna_legacy_top_level_state(
        entry,
        model_choice="XGBoost",
        calibration_method="sigmoid",
    )

    assert model_result is not None
    assert fake_st.session_state["optuna_last_optimized_feature_sets"] == [
        "Base",
        "Base + Cluster",
    ]
    assert (
        fake_st.session_state["optuna_last_optimized_feature_key"]
        == "/tmp/flow_features.duckdb"
    )
    assert fake_st.session_state["optuna_last_optimized_dataset_fingerprint"] == "fp_ctx"


def test_build_model_tab_training_kwargs_preserves_candidate_smote_params():
    smote_params = {
        "smote_k_neighbors": 7,
        "smote_sampling_strategy": 0.4,
    }

    kwargs = app._build_model_tab_training_kwargs(
        val_size=0.2,
        far_target=0.15,
        random_state=42,
        threshold_protocol="robust",
        threshold_objective="balanced_f1",
        calibration_method="sigmoid",
        alerts_per_day=5.0,
        fn_cost=10.0,
        fp_cost=1.0,
        robust_folds=3,
        balance_strategy="smote",
        use_balanced=False,
        use_split=False,
        smote_params=smote_params,
    )

    assert kwargs["balance_strategy"] == "smote"
    assert kwargs["smote_params"] == smote_params
    assert kwargs["smote_params"] is not smote_params


def test_build_model_tab_training_kwargs_disables_internal_balance_on_prebalanced_split():
    kwargs = app._build_model_tab_training_kwargs(
        val_size=0.2,
        far_target=0.15,
        random_state=42,
        threshold_protocol="robust",
        threshold_objective="balanced_f1",
        calibration_method="sigmoid",
        alerts_per_day=5.0,
        fn_cost=10.0,
        fp_cost=1.0,
        robust_folds=3,
        balance_strategy="smote",
        use_balanced=True,
        use_split=True,
        smote_params={
            "smote_k_neighbors": 9,
            "smote_sampling_strategy": 0.6,
        },
    )

    assert kwargs["balance_strategy"] == "none"
    assert kwargs["smote_params"] == {
        "smote_k_neighbors": 9,
        "smote_sampling_strategy": 0.6,
    }


def test_lookup_optuna_pareto_candidates_reconstructs_trial_params_and_topk():
    feature_key = "features.duckdb"
    feature_cols = ["flow", "speed", "cluster_share_0"]
    optuna_key = app._optuna_result_key(feature_key, feature_cols)
    pareto_front_df = pd.DataFrame(
        [
            {
                "number": 2,
                "selected_trial": False,
                "pruning_proxy_score": 0.61,
                "far_gate_pass": True,
                "decision_threshold": 0.42,
                "params_top_k": 2,
                "params_smote_k_neighbors": 7,
                "params_smote_sampling_strategy": 0.4,
                "params_xgb_n_estimators": 140,
                "params_xgb_max_depth": 6,
                "params_xgb_learning_rate": 0.05,
                "params_xgb_subsample": 0.9,
                "params_xgb_colsample_bytree": 0.8,
                "params_xgb_reg_alpha": 1.2,
                "params_xgb_reg_lambda": 3.4,
                "params_xgb_gamma": 0.1,
                "value_mcc": 0.33,
                "value_pr_auc": 0.44,
                "value_brier_score": 0.19,
                "value_recall_at_alerts_per_day": 0.52,
            },
            {
                "number": 5,
                "selected_trial": True,
                "pruning_proxy_score": 0.65,
                "far_gate_pass": True,
                "decision_threshold": 0.38,
                "params_top_k": 3,
                "params_smote_k_neighbors": 9,
                "params_smote_sampling_strategy": 0.6,
                "params_xgb_n_estimators": 180,
                "params_xgb_max_depth": 4,
                "params_xgb_learning_rate": 0.03,
                "params_xgb_subsample": 1.0,
                "params_xgb_colsample_bytree": 0.7,
                "params_xgb_reg_alpha": 0.0,
                "params_xgb_reg_lambda": 5.0,
                "params_xgb_gamma": 0.0,
                "value_mcc": 0.39,
                "value_pr_auc": 0.51,
                "value_brier_score": 0.17,
                "value_recall_at_alerts_per_day": 0.57,
            },
        ]
    )
    variant = {
        "model_choice": "XGBoost",
        "balance_mode": "smote",
        "calibration_method": "sigmoid",
        "best_score": 0.65,
        "best_smote_params": {
            "smote_k_neighbors": 5,
            "smote_sampling_strategy": 0.2,
        },
        "best_model_params": {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "gamma": 0.0,
            "n_jobs": 4,
        },
        "optuna_settings": {
            "objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
            "objective_mode_label": app.CALIBRATION_SWEEP_MULTIOBJECTIVE_LABEL,
            "balance_mode": "smote",
            "calibration_method": "sigmoid",
            "best_feature_cols": feature_cols,
            "best_top_k": 3,
            "ranked_cols": feature_cols,
        },
        "objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
        "optuna_objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
        "pareto_front_df": pareto_front_df,
        "search_space": {},
    }
    store = {
        optuna_key: _make_optuna_entry(
            feature_key=feature_key,
            feature_cols=feature_cols,
            variants=[variant],
        )
    }

    info = app._lookup_optuna_pareto_candidates(
        store=store,
        feature_key=feature_key,
        cols=feature_cols,
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        allow_calibration_fallback=False,
    )

    assert info is not None
    assert info["objective_mode"] == app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
    assert len(info["candidates"]) == 2

    first_candidate = info["candidates"][0]
    assert first_candidate["trial_number"] == 2
    assert first_candidate["top_k"] == 2
    assert first_candidate["feature_cols"] == ["flow", "speed"]
    assert first_candidate["model_params"]["n_estimators"] == 140
    assert first_candidate["model_params"]["max_depth"] == 6
    assert first_candidate["model_params"]["n_jobs"] == 4
    assert first_candidate["smote_params"]["smote_k_neighbors"] == 7
    assert first_candidate["smote_params"]["smote_sampling_strategy"] == pytest.approx(0.4)
    assert first_candidate["objective_values"]["mcc"] == pytest.approx(0.33)
    assert first_candidate["selected_trial"] is False

    second_candidate = info["candidates"][1]
    assert second_candidate["trial_number"] == 5
    assert second_candidate["top_k"] == 3
    assert second_candidate["feature_cols"] == feature_cols
    assert second_candidate["model_params"]["n_estimators"] == 180
    assert second_candidate["selected_trial"] is True


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


def test_model_optuna_batch_progress_ratio_prefers_live_and_clamps():
    assert app._model_optuna_batch_progress_ratio(
        live_status={"progress_ratio": 1.4},
        manifest_progress={"completed_steps": 1, "total_steps": 4},
    ) == 1.0
    assert app._model_optuna_batch_progress_ratio(
        live_status={},
        manifest_progress={"completed_steps": 2, "total_steps": 4},
    ) == 0.5
    assert app._model_optuna_batch_progress_ratio(
        live_status={"progress_ratio": -0.25},
        manifest_progress={"completed_steps": 2, "total_steps": 4},
    ) == 0.0


def test_model_optuna_batch_best_so_far_excludes_failed_rows():
    rows = [
        {
            "status": "completed",
            "combo_id": "a",
            "candidate_label": "A",
            "val_balanced_f1": 0.31,
        },
        {
            "status": "failed",
            "combo_id": "b",
            "candidate_label": "B",
            "val_balanced_f1": 0.99,
        },
        {
            "status": "completed",
            "combo_id": "c",
            "candidate_label": "C",
            "val_balanced_f1": 0.44,
        },
    ]

    curve_df, metric_col, lower_is_better = app._model_optuna_batch_best_so_far_frame(
        rows,
        objective_metric="balanced_f1",
    )

    assert metric_col == "val_balanced_f1"
    assert lower_is_better is False
    assert curve_df["combo_id"].tolist() == ["a", "c"]
    assert curve_df["combo_index"].tolist() == [1, 2]
    assert curve_df["best_so_far"].round(2).tolist() == [0.31, 0.44]


def test_model_optuna_batch_best_so_far_uses_cumulative_min_for_brier():
    rows = [
        {"status": "completed", "combo_id": "a", "val_brier_score": 0.24},
        {"status": "completed", "combo_id": "b", "val_brier_score": 0.18},
        {"status": "completed", "combo_id": "c", "val_brier_score": 0.21},
    ]

    curve_df, metric_col, lower_is_better = app._model_optuna_batch_best_so_far_frame(
        rows,
        objective_metric="brier_score",
    )

    assert metric_col == "val_brier_score"
    assert lower_is_better is True
    assert curve_df["best_so_far"].round(2).tolist() == [0.24, 0.18, 0.18]
    assert curve_df["new_best"].tolist() == [True, True, False]


def test_model_optuna_batch_best_direction_follows_plotted_fallback_metric():
    rows = [
        {"status": "completed", "combo_id": "a", "val_balanced_f1": 0.47},
        {"status": "completed", "combo_id": "b", "val_balanced_f1": 0.46},
        {"status": "completed", "combo_id": "c", "val_balanced_f1": 0.49},
    ]

    curve_df, metric_col, lower_is_better = app._model_optuna_batch_best_so_far_frame(
        rows,
        objective_metric="far",
    )

    assert metric_col == "val_balanced_f1"
    assert lower_is_better is False
    assert curve_df["best_so_far"].round(2).tolist() == [0.47, 0.47, 0.49]


def test_model_optuna_batch_metric_defaults_follow_protocol_objective():
    columns = ["val_balanced_f1", "val_pr_auc", "val_brier_score"]

    assert app._model_optuna_batch_metric_column("pr_auc", columns) == "val_pr_auc"
    assert (
        app._model_optuna_batch_metric_column("unknown_metric", columns)
        == "val_balanced_f1"
    )


def test_model_optuna_batch_metric_selector_defaults_to_test_roc_auc():
    rows = [
        {
            "status": "failed",
            "combo_id": "a",
            "test_pr_auc": 0.91,
            "test_roc_auc": 0.99,
        },
        {
            "status": "completed",
            "combo_id": "b",
            "test_roc_auc": 0.82,
            "test_brier_score": 0.18,
            "val_balanced_f1": 0.46,
        },
    ]

    options = app._model_optuna_batch_metric_options(rows)

    assert [metric_col for metric_col, _label in options] == [
        "test_roc_auc",
        "test_brier_score",
        "val_balanced_f1",
    ]
    assert app._model_optuna_batch_default_metric_column(options) == "test_roc_auc"


def test_model_optuna_batch_best_so_far_uses_selected_metric():
    rows = [
        {
            "status": "completed",
            "combo_id": "a",
            "test_roc_auc": 0.72,
            "val_balanced_f1": 0.49,
        },
        {
            "status": "completed",
            "combo_id": "b",
            "test_roc_auc": 0.76,
            "val_balanced_f1": 0.47,
        },
    ]

    curve_df, metric_col, lower_is_better = app._model_optuna_batch_best_so_far_frame(
        rows,
        objective_metric="balanced_f1",
        metric_col="test_roc_auc",
    )

    assert metric_col == "test_roc_auc"
    assert lower_is_better is False
    assert curve_df["best_so_far"].round(2).tolist() == [0.72, 0.76]


def test_model_optuna_batch_progress_board_defers_metric_selector_until_terminal(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_st = _ProgressBoardFakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    def _fake_chart(_rows, **kwargs):
        fake_st.chart_calls.append(kwargs)

    monkeypatch.setattr(
        app,
        "_render_model_optuna_batch_best_so_far_chart",
        _fake_chart,
    )
    board = app._ModelOptunaBatchProgressBoard(
        model_name="XGBoost",
        total_steps=2,
        batch_ref={},
        run_dir="Resultados/model_optuna_batch_test",
    )
    rows = [
        {
            "status": "completed",
            "combo_id": "combo-1",
            "test_roc_auc": 0.82,
            "val_balanced_f1": 0.46,
        }
    ]

    def _state(step_id: str, *, status: str = "running") -> dict:
        return {
            "run_id": "model_optuna_batch_test",
            "manifest": {
                "run_id": "model_optuna_batch_test",
                "status": status,
                "result_status": status,
                "progress": {
                    "completed_steps": 1,
                    "total_steps": 2,
                    "current_step_id": step_id,
                    "progress_ratio": 0.5,
                },
            },
            "live_status": {
                "step_id": step_id,
                "message": step_id,
                "progress_ratio": 0.5,
            },
        }

    board.update(_state("combo_done"), rows, objective_metric="balanced_f1")
    board.update(_state("combo_start"), rows, objective_metric="balanced_f1")

    assert fake_st.selectbox_calls == []
    assert [call["metric_col"] for call in fake_st.chart_calls] == [
        "test_roc_auc",
        "test_roc_auc",
    ]

    board.update(
        _state("run_completed", status="completed"),
        rows,
        objective_metric="balanced_f1",
    )
    board.update(
        _state("run_completed", status="completed"),
        rows,
        objective_metric="balanced_f1",
    )

    keys = [call["key"] for call in fake_st.selectbox_calls]
    assert len(keys) == 2
    assert len(set(keys)) == 2
    assert keys[0].startswith(
        "model_optuna_batch_metric_selector_model_optuna_batch_test_"
    )
    assert (
        fake_st.session_state[
            "model_optuna_batch_metric_choice_model_optuna_batch_test"
        ]
        == "test_roc_auc"
    )


def test_model_optuna_batch_result_progress_row_includes_roc_auc_metrics():
    row = app._model_optuna_batch_result_progress_row(
        combo_index=1,
        total_combinations=2,
        subrun={
            "subrun_id": "subrun-1",
            "feature_set_label": "Base",
            "balance_mode": "none",
            "balance_mode_label": "Sin balance",
        },
        candidate={
            "candidate_kind": "trial",
            "trial_number": 3,
            "objective_metric": "roc_auc",
            "decision_threshold": 0.42,
        },
        candidate_label="Trial 3",
        protocol="test",
        protocol_label="Test",
        model_name="Random Forest",
        model_n_jobs=2,
        threshold_objective="balanced_f1",
        threshold_objective_label="Balanced F1",
        calibration_method="sigmoid",
        result={
            "validation_metrics": {"roc_auc": 0.78},
            "metrics": {"roc_auc": 0.83, "threshold": 0.37},
        },
    )

    assert row["val_roc_auc"] == 0.78
    assert row["test_roc_auc"] == 0.83


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
