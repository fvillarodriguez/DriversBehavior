import numpy as np
import pandas as pd
import pytest

from src import model_training
from src.model_training import (
    compute_extended_metrics,
    select_threshold_for_metric,
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
