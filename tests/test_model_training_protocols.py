import warnings

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


def test_build_xai_explain_rows_batches_metadata_columns_without_fragmentation_warning():
    rows = 8
    feature_cols = [f"feature_{idx}" for idx in range(130)]
    test_df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "target": np.asarray([0, 1] * 4, dtype=int),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for idx, feature_col in enumerate(feature_cols):
            test_df[feature_col] = np.linspace(0.0, 1.0, rows) + idx

    scores = np.linspace(0.1, 0.8, rows)
    preds = (scores >= 0.5).astype(int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        explain_rows = model_training._build_xai_explain_rows(
            test_df,
            feature_cols=feature_cols,
            scores_test=scores,
            preds=preds,
            threshold=0.5,
        )

    performance_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, pd.errors.PerformanceWarning)
    ]
    assert performance_warnings == []
    assert {"target", "score", "pred", "threshold", "case_hint"}.issubset(
        explain_rows.columns
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


def test_get_model_scores_prefers_decision_function_when_requested():
    class _DualScoreModel:
        _score_preference = "decision_function"

        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.1), np.full(n, 0.9)])

        def decision_function(self, X):
            return np.full(len(X), -3.0)

    frame = pd.DataFrame({"signal": [0.1, 0.2, 0.3]})
    scores = model_training.get_model_scores(_DualScoreModel(), frame)

    assert scores.tolist() == [-3.0, -3.0, -3.0]


def test_svm_build_model_linear_returns_raw_scores_and_probabilities():
    X = pd.DataFrame(
        {
            "speed": [-2.0, -1.0, 1.0, 2.0],
            "flow": [-1.0, -0.5, 0.5, 1.0],
        }
    )
    y = np.asarray([0, 0, 1, 1], dtype=int)

    model = model_training.build_model(
        "SVM",
        {
            "C": 1.0,
            "kernel": "linear",
            "probability": True,
            "epochs": 4,
            "batch_size": 2,
        },
        random_state=17,
    )
    model.fit(X, y)

    scores = model_training.get_model_scores(model, X)
    proba = model.predict_proba(X)

    assert getattr(model, "_score_preference", "") == "decision_function"
    assert getattr(model, "backend_", None) in {"mlx", "sklearn_linear"}
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_svm_build_model_rbf_uses_random_fourier_features():
    X = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0, 0.2, 0.2, 0.8, 0.8],
            "x2": [0.0, 1.0, 0.0, 1.0, 0.2, 0.8, 0.2, 0.8],
        }
    )
    y = np.asarray([0, 1, 1, 0, 0, 1, 1, 0], dtype=int)

    model = model_training.build_model(
        "SVM",
        {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "rff_components": 32,
            "epochs": 4,
            "batch_size": 4,
        },
        random_state=23,
    )
    model.fit(X, y)

    scores = model_training.get_model_scores(model, X)
    proba = model.predict_proba(X)

    assert getattr(model, "kernel_", "") == "rbf"
    assert getattr(model, "feature_map_", None) is not None
    assert getattr(model, "backend_", None) in {"mlx", "sklearn_linear"}
    assert scores.shape == (8,)
    assert np.isfinite(scores).all()
    assert proba.shape == (8, 2)


def test_svm_require_mlx_blocks_linear_cpu_fallback(monkeypatch):
    from src import mlx_svm

    X = pd.DataFrame({"speed": [-1.0, 1.0], "flow": [-0.5, 0.5]})
    y = np.asarray([0, 1], dtype=int)

    def _raise_mlx_unavailable(*_args, **_kwargs):
        raise RuntimeError("forced MLX failure")

    monkeypatch.setattr(
        mlx_svm,
        "_train_linear_svm_with_mlx",
        _raise_mlx_unavailable,
    )
    model = model_training.build_model(
        "SVM",
        {
            "C": 1.0,
            "kernel": "linear",
            "require_mlx": True,
            "epochs": 1,
            "batch_size": 2,
        },
        random_state=17,
    )

    with pytest.raises(RuntimeError, match="requiere backend MLX"):
        model.fit(X, y)

    assert getattr(model, "backend_", None) == "mlx_failed"
    assert not hasattr(model, "linear_model_")


def test_svm_require_mlx_rejects_cpu_only_kernels():
    X = pd.DataFrame({"speed": [-1.0, 1.0], "flow": [-0.5, 0.5]})
    y = np.asarray([0, 1], dtype=int)
    model = model_training.build_model(
        "SVM",
        {
            "C": 1.0,
            "kernel": "poly",
            "require_mlx": True,
        },
        random_state=17,
    )

    with pytest.raises(RuntimeError, match="no tiene backend MLX"):
        model.fit(X, y)

    assert getattr(model, "backend_", None) == "unsupported_mlx_kernel"


def test_neural_network_build_model_passes_phase1_options():
    model = model_training.build_model(
        "Neural Network",
        {
            "hidden_dim": 32,
            "num_layers": 3,
            "dropout": 0.15,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "epochs": 7,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.002,
            "pos_weight": 4.0,
            "val_fraction": 0.25,
            "use_batch_norm": True,
            "hidden_activation": "gelu",
            "output_activation": "sigmoid",
            "loss_function": "focal",
            "optimizer": "Adam",
            "focal_gamma": 1.5,
            "focal_alpha": 0.7,
            "max_grad_norm": 0.75,
            "lr_scheduler": "reduce_on_plateau",
            "scheduler_factor": 0.25,
            "scheduler_patience": 1,
            "min_lr": 1e-5,
            "temperature_scaling": True,
        },
        random_state=123,
    )

    wrapper = model.named_steps["model"]
    assert wrapper.hidden_dim == 32
    assert wrapper.num_layers == 3
    assert wrapper.use_batch_norm is True
    assert wrapper.hidden_activation == "gelu"
    assert wrapper.output_activation == "sigmoid"
    assert wrapper.loss_function == "focal"
    assert wrapper.optimizer_name == "Adam"
    assert wrapper.focal_gamma == pytest.approx(1.5)
    assert wrapper.focal_alpha == pytest.approx(0.7)
    assert wrapper.early_stopping_min_delta == pytest.approx(0.002)
    assert wrapper.max_grad_norm == pytest.approx(0.75)
    assert wrapper.lr_scheduler == "reduce_on_plateau"
    assert wrapper.temperature_scaling is True


def test_mlp_net_inserts_batch_norm_only_when_requested():
    torch = pytest.importorskip("torch")
    from src.mlp_tabular import MLPNet

    with_bn = MLPNet(4, hidden_dim=8, num_layers=2, dropout=0.1, use_batch_norm=True)
    without_bn = MLPNet(4, hidden_dim=8, num_layers=2, dropout=0.1)

    assert sum(isinstance(layer, torch.nn.BatchNorm1d) for layer in with_bn.net) == 2
    assert sum(isinstance(layer, torch.nn.BatchNorm1d) for layer in without_bn.net) == 0


def test_mlp_net_uses_requested_hidden_activation():
    torch = pytest.importorskip("torch")
    from src.mlp_tabular import MLPNet

    gelu_net = MLPNet(
        4,
        hidden_dim=8,
        num_layers=2,
        dropout=0.1,
        hidden_activation="gelu",
    )

    assert sum(isinstance(layer, torch.nn.GELU) for layer in gelu_net.net) == 2
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in gelu_net.net) == 0


def test_mlp_focal_loss_matches_manual_formula():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    wrapper = model_training.MLPClassifierWrapper(
        loss_function="focal",
        focal_gamma=2.0,
        focal_alpha=0.75,
    )
    logits = torch.tensor([[2.0, -0.5], [-0.25, 1.25], [0.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    class_weight = torch.tensor([1.0, 3.0])

    observed = wrapper._compute_loss(logits, targets, class_weight)

    weighted_ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weight,
        reduction="none",
    )
    base_ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-base_ce)
    alpha_t = torch.where(targets == 1, torch.tensor(0.75), torch.tensor(0.25))
    expected = (alpha_t * ((1.0 - pt) ** 2.0) * weighted_ce).mean()
    assert observed.item() == pytest.approx(expected.item())


def test_mlp_classifier_trains_with_bce_sigmoid_and_adam_aliases():
    pytest.importorskip("torch")

    rng = np.random.default_rng(17)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = (X[:, 0] - 0.25 * X[:, 1] > 0.0).astype(int)

    clf = model_training.MLPClassifierWrapper(
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        learning_rate=0.01,
        batch_size=8,
        epochs=2,
        early_stopping_patience=2,
        val_fraction=0.25,
        hidden_activation="LeakyReLU",
        output_activation="Sigmoid",
        loss_function="BCE",
        optimizer_name="Adam",
        random_state=21,
    ).fit(X, y)

    proba = clf.predict_proba(X[:5])

    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(proba).all()
    assert clf.train_loss_history_


def test_mlp_classifier_trains_with_phase1_options(monkeypatch):
    torch = pytest.importorskip("torch")

    scheduler_steps = []
    callback_payloads = []

    class _FakePlateauScheduler:
        def __init__(self, optimizer, **kwargs):
            self.optimizer = optimizer
            self.kwargs = kwargs

        def step(self, metric):
            scheduler_steps.append(float(metric))

    clip_calls = []

    def _fake_clip_grad_norm_(parameters, max_norm):
        clip_calls.append(float(max_norm))
        return torch.tensor(0.0)

    monkeypatch.setattr(
        torch.optim.lr_scheduler,
        "ReduceLROnPlateau",
        _FakePlateauScheduler,
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        _fake_clip_grad_norm_,
    )

    rng = np.random.default_rng(7)
    X = rng.normal(size=(48, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(int)

    clf = model_training.MLPClassifierWrapper(
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        learning_rate=0.01,
        batch_size=8,
        epochs=3,
        early_stopping_patience=3,
        early_stopping_min_delta=0.01,
        val_fraction=0.25,
        use_batch_norm=True,
        loss_function="focal",
        focal_gamma=1.0,
        focal_alpha=0.65,
        max_grad_norm=0.5,
        lr_scheduler="reduce_on_plateau",
        scheduler_patience=0,
        temperature_scaling=True,
        random_state=11,
    ).fit(
        X,
        y,
        epoch_callback=lambda payload: callback_payloads.append(dict(payload)),
    )

    proba = clf.predict_proba(X[:6])
    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert clip_calls
    assert set(clip_calls) == {0.5}
    assert scheduler_steps == pytest.approx(clf.val_loss_history_)
    assert len(clf.train_loss_history_) == len(clf.val_loss_history_)
    assert clf.train_loss_history_
    assert clf.lr_history_
    assert clf.epochs_ran_ == len(clf.train_loss_history_)
    assert len(callback_payloads) >= len(clf.train_loss_history_)
    assert callback_payloads[-1]["train_loss"] == pytest.approx(
        clf.train_loss_history_
    )
    assert clf.temperature_ > 0.0
