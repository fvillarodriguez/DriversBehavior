from __future__ import annotations

import warnings

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
