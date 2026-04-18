import numpy as np
import pandas as pd
import pytest

from src import model_training
from src.model_training import (
    compute_extended_metrics,
    select_threshold_for_metric,
    score_optuna_objective,
    train_model,
)


def _toy_eval_frame(rows: int, freq: str = "D") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=rows, freq=freq),
        }
    )


def test_balanced_f1_threshold_selects_macro_f1_optimum():
    y_true = np.asarray([0, 0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.7, 0.9])

    info = select_threshold_for_metric(
        y_true,
        scores,
        objective="balanced_f1",
        eval_df=_toy_eval_frame(len(y_true)),
    )

    assert info["threshold"] == pytest.approx(0.5)
    assert info["objective_score"] == pytest.approx(0.8)


def test_recall_at_alerts_per_day_threshold_respects_alert_budget():
    y_true = np.asarray([0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    scores = np.asarray([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3])
    eval_df = _toy_eval_frame(len(y_true), freq="D")

    info = select_threshold_for_metric(
        y_true,
        scores,
        objective="recall_at_alerts_per_day",
        eval_df=eval_df,
        alerts_per_day=0.3,
    )
    metrics = compute_extended_metrics(
        y_true,
        scores,
        threshold=float(info["threshold"]),
        eval_df=eval_df,
        alerts_per_day=0.3,
    )

    assert metrics["alerts_per_day"] <= 0.3 + 1e-12
    assert metrics["recall"] > 0.0


def test_parallel_threshold_selection_matches_serial():
    y_true = np.asarray(([0, 1, 0, 0, 1] * 20), dtype=int)
    scores = np.linspace(0.01, 0.99, len(y_true))
    eval_df = _toy_eval_frame(len(y_true), freq="5min")

    serial = select_threshold_for_metric(
        y_true,
        scores,
        objective="balanced_f1",
        eval_df=eval_df,
        n_jobs=1,
    )
    parallel = select_threshold_for_metric(
        y_true,
        scores,
        objective="balanced_f1",
        eval_df=eval_df,
        n_jobs=2,
    )

    assert parallel["threshold_n_jobs"] == 2
    assert parallel["threshold"] == pytest.approx(serial["threshold"])
    assert parallel["objective_score"] == pytest.approx(serial["objective_score"])


def test_operational_cost_threshold_prefers_false_positive_when_fn_cost_is_high():
    y_true = np.asarray([0, 1])
    scores = np.asarray([0.6, 0.5])

    high_fn = select_threshold_for_metric(
        y_true,
        scores,
        objective="operational_cost",
        fn_cost=20.0,
        fp_cost=1.0,
    )
    high_fp = select_threshold_for_metric(
        y_true,
        scores,
        objective="operational_cost",
        fn_cost=1.0,
        fp_cost=20.0,
    )

    assert high_fn["threshold"] <= 0.5
    assert high_fp["threshold"] > 0.6


def test_extended_metrics_include_brier_score():
    y_true = np.asarray([0, 1, 1, 0])
    scores = np.asarray([0.1, 0.8, 0.6, 0.3])

    metrics = compute_extended_metrics(y_true, scores, threshold=0.5)

    expected = np.mean((scores - y_true) ** 2)
    assert metrics["brier_score"] == pytest.approx(expected)
    assert metrics["brier"] == pytest.approx(expected)
    assert metrics["positive_support"] == 2
    assert metrics["tp_capture"] == pytest.approx(1.0)
    assert metrics["fn_rate"] == pytest.approx(0.0)


def test_score_optuna_objective_supports_brier_and_mcc():
    y_true = np.asarray([0, 1, 1, 0])
    scores = np.asarray([0.1, 0.8, 0.6, 0.3])

    brier = score_optuna_objective(
        y_true,
        scores,
        objective_metric="brier",
        threshold=0.5,
    )
    mcc = score_optuna_objective(
        y_true,
        scores,
        objective_metric="mcc",
        threshold=0.5,
    )

    assert brier["objective_metric"] == "brier_score"
    assert brier["objective_direction"] == "minimize"
    assert brier["score"] == pytest.approx(np.mean((scores - y_true) ** 2))
    assert mcc["objective_metric"] == "mcc"
    assert mcc["objective_direction"] == "maximize"
    assert mcc["score"] == pytest.approx(1.0)


class _ScoreModel:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        scores = np.asarray(X["signal"], dtype=float)
        return np.column_stack([1.0 - scores, scores])


def _protocol_df(rows: int = 60) -> pd.DataFrame:
    idx = np.arange(rows)
    target = ((idx % 10) >= 7).astype(int)
    signal = np.where(target == 1, 0.8, 0.2)
    return pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "signal": signal,
            "target": target,
        }
    )


def test_conservative_and_robust_protocols_report_distinct_split_metadata(monkeypatch):
    monkeypatch.setattr(
        model_training,
        "build_model",
        lambda model_name, params, random_state: _ScoreModel(),
    )
    df = _protocol_df()

    conservative = train_model(
        df,
        ["signal"],
        "Random Forest",
        {"n_estimators": 10, "max_depth": None},
        test_size=0.2,
        val_size=0.25,
        far_target=0.2,
        random_state=42,
        threshold_protocol="conservative",
        threshold_objective="balanced_f1",
    )
    robust = train_model(
        df,
        ["signal"],
        "Random Forest",
        {"n_estimators": 10, "max_depth": None},
        test_size=0.2,
        val_size=0.25,
        far_target=0.2,
        random_state=42,
        threshold_protocol="robust",
        threshold_objective="balanced_f1",
        robust_folds=3,
    )

    assert conservative["threshold_protocol"] == "conservative"
    assert conservative["split_info"]["robust_folds"] == 0
    assert robust["threshold_protocol"] == "robust"
    assert robust["split_info"]["robust_folds"] > 0
    assert robust["split_info"]["oof_rows"] > 0


def test_internal_smote_is_called_only_on_training_partitions(monkeypatch):
    monkeypatch.setattr(
        model_training,
        "build_model",
        lambda model_name, params, random_state: _ScoreModel(),
    )
    observed_rows = []

    def _fake_smote(X_train, y_train, *, random_state, smote_params=None):
        observed_rows.append(len(X_train))
        return X_train, y_train

    monkeypatch.setattr(model_training, "_apply_internal_smote", _fake_smote)
    df = _protocol_df()

    result = train_model(
        df,
        ["signal"],
        "Random Forest",
        {"n_estimators": 10, "max_depth": None},
        test_size=0.2,
        val_size=0.25,
        far_target=0.2,
        random_state=42,
        threshold_protocol="robust",
        threshold_objective="balanced_f1",
        robust_folds=3,
        balance_strategy="smote",
    )

    assert result["threshold_protocol"] == "robust"
    assert observed_rows
    assert max(observed_rows) < len(df)


@pytest.mark.parametrize(
    ("calibration_method", "shift"),
    [("sigmoid", 0.05), ("isotonic", 0.10)],
)
def test_threshold_selection_uses_calibrated_scores(monkeypatch, calibration_method, shift):
    monkeypatch.setattr(
        model_training,
        "build_model",
        lambda model_name, params, random_state: _ScoreModel(),
    )
    observed = {"scores": None, "methods": []}

    class _FakeCalibrator:
        def __init__(self, method: str, delta: float) -> None:
            self.method = method
            self.delta = float(delta)

        def transform(self, scores):
            return np.asarray(scores, dtype=float) + self.delta

    def _fake_fit_score_calibrator(y_true, scores, *, method="none"):
        observed["methods"].append(str(method))
        return _FakeCalibrator(str(method), shift)

    def _fake_select_threshold_for_metric(y_true, scores, **kwargs):
        observed["scores"] = np.asarray(scores, dtype=float)
        return {
            "threshold": 0.5,
            "objective_score": 1.0,
            "objective": kwargs.get("objective", "balanced_f1"),
            "threshold_n_jobs": 1,
        }

    monkeypatch.setattr(
        model_training,
        "fit_score_calibrator",
        _fake_fit_score_calibrator,
    )
    monkeypatch.setattr(
        model_training,
        "select_threshold_for_metric",
        _fake_select_threshold_for_metric,
    )

    df = _protocol_df()
    result = train_model(
        df,
        ["signal"],
        "Random Forest",
        {"n_estimators": 10, "max_depth": None},
        test_size=0.2,
        val_size=0.25,
        far_target=0.2,
        random_state=42,
        threshold_protocol="conservative",
        threshold_objective="balanced_f1",
        calibration_method=calibration_method,
    )

    assert observed["methods"] == [calibration_method]
    assert observed["scores"] is not None
    assert observed["scores"].max() == pytest.approx(0.8 + shift)
    assert result["calibration_method"] == calibration_method
    assert result["metrics"]["calibration_method"] == calibration_method
