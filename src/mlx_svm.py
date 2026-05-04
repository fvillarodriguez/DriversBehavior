"""
SVM wrapper with an MLX training path for Apple Silicon and safe CPU fallbacks.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def _as_float32_matrix(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _normalize_kernel_name(value: object) -> str:
    text = str(value or "linear").strip().lower().replace("-", "_")
    aliases = {
        "linear": "linear",
        "rbf": "rbf",
        "rbf_approx": "rbf",
        "rff": "rbf",
        "poly": "poly",
        "sigmoid": "sigmoid",
    }
    return aliases.get(text, "linear")


def _normalize_class_weight(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        if text == "balanced":
            return "balanced"
        return value
    if isinstance(value, dict):
        return dict(value)
    return value


def _resolve_positive_class_weight(
    y: np.ndarray,
    class_weight: Optional[object],
) -> float:
    y_arr = np.asarray(y).astype(int)
    positives = int((y_arr == 1).sum())
    negatives = int((y_arr == 0).sum())
    normalized = _normalize_class_weight(class_weight)

    if positives <= 0:
        return 1.0
    if normalized == "balanced":
        return max(1.0, float(negatives / positives))
    if isinstance(normalized, dict):
        neg_weight = float(
            normalized.get(0, normalized.get("0", normalized.get(False, 1.0)))
        )
        pos_weight = float(
            normalized.get(1, normalized.get("1", normalized.get(True, 1.0)))
        )
        if neg_weight <= 0:
            neg_weight = 1.0
        if pos_weight <= 0:
            pos_weight = 1.0
        return float(pos_weight / neg_weight)
    return 1.0


def _resolve_rbf_gamma(gamma: object, X_scaled: np.ndarray) -> float:
    feature_count = max(1, int(X_scaled.shape[1]))
    if gamma is None:
        return 1.0 / feature_count
    if isinstance(gamma, str):
        text = gamma.strip().lower()
        if text == "auto":
            return 1.0 / feature_count
        if text == "scale":
            variance = float(np.var(X_scaled))
            if not np.isfinite(variance) or variance <= 0.0:
                variance = 1.0
            return 1.0 / (feature_count * variance)
    value = float(gamma)
    if not np.isfinite(value) or value <= 0.0:
        return 1.0 / feature_count
    return value


def _sigmoid_scores(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _maybe_import_mlx_modules():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        return mx, nn, optim
    except Exception:
        return None


def _train_linear_svm_with_mlx(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    pos_weight: float,
    random_state: int,
) -> tuple[np.ndarray, float]:
    modules = _maybe_import_mlx_modules()
    if modules is None:
        raise ImportError("MLX no esta disponible en el entorno actual.")
    mx, nn, optim = modules

    class _LinearSVMModule(nn.Module):
        def __init__(self, n_features: int, seed: int):
            super().__init__()
            rng = np.random.default_rng(int(seed))
            self.w = mx.array(
                rng.normal(0.0, 0.01, size=(n_features,)).astype(np.float32)
            )
            self.b = mx.array(np.float32(0.0))

        def __call__(self, X):
            return X @ self.w + self.b

    model = _LinearSVMModule(int(X_train.shape[1]), int(random_state))
    optimizer = optim.Adam(learning_rate=float(learning_rate))

    def loss_fn(model, xb, yb, swb):
        scores = model(xb)
        margins = 1.0 - (yb * scores)
        hinge = mx.maximum(0.0, margins)
        data_loss = float(C) * mx.mean(swb * hinge)
        reg_loss = 0.5 * mx.sum(model.w * model.w)
        return data_loss + reg_loss

    eager_loss_and_grad = nn.value_and_grad(model, loss_fn)

    def eager_step(xb, yb, swb):
        loss, grads = eager_loss_and_grad(model, xb, yb, swb)
        optimizer.update(model, grads)
        return loss

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def compiled_step(xb, yb, swb):
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, xb, yb, swb)
        optimizer.update(model, grads)
        return loss

    step_fn = compiled_step
    y_signed = np.where(np.asarray(y_train).astype(int) == 1, 1.0, -1.0).astype(
        np.float32
    )
    sample_weight = np.where(
        np.asarray(y_train).astype(int) == 1,
        float(pos_weight),
        1.0,
    ).astype(np.float32)

    rng = np.random.default_rng(int(random_state))
    batch_size = max(1, int(batch_size))
    epochs = max(1, int(epochs))
    indices = np.arange(len(X_train), dtype=int)

    for _epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = mx.array(X_train[batch_idx])
            yb = mx.array(y_signed[batch_idx])
            swb = mx.array(sample_weight[batch_idx])
            try:
                loss = step_fn(xb, yb, swb)
            except Exception:
                step_fn = eager_step
                loss = step_fn(xb, yb, swb)
            mx.eval(model.state, optimizer.state, loss)

    weights = np.asarray(model.w, dtype=np.float32).reshape(-1)
    bias = float(np.asarray(model.b, dtype=np.float32).reshape(-1)[0])
    return weights, bias


class MLXAcceleratedSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary SVM wrapper.

    `kernel="linear"` trains a linear SVM directly.
    `kernel="rbf"` uses Random Fourier Features + linear SVM.
    Unsupported kernels fall back to sklearn's `SVC` unless `require_mlx=True`.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        kernel: str = "linear",
        gamma: object = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        probability: bool = True,
        cache_size: float = 200.0,
        class_weight: Optional[object] = None,
        random_state: int = 42,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 8192,
        rff_components: int = 2048,
        require_mlx: bool = False,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.rff_components = rff_components
        self.require_mlx = require_mlx
        self._score_preference = "decision_function"

    def _fit_probability_model(self, scores: np.ndarray, y: np.ndarray) -> None:
        self.probability_model_ = None
        if not bool(self.probability):
            return
        y_arr = np.asarray(y).astype(int)
        if np.unique(y_arr).size < 2:
            return
        score_arr = np.asarray(scores, dtype=float).reshape(-1, 1)
        try:
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
            )
            model.fit(score_arr, y_arr)
            self.probability_model_ = model
        except Exception:
            self.probability_model_ = None

    def _fit_feature_map(self, X_scaled: np.ndarray) -> np.ndarray:
        kernel_name = _normalize_kernel_name(self.kernel)
        self.kernel_ = kernel_name
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_scaled).astype(np.float32, copy=False)
        self.feature_map_ = None
        self.gamma_ = None

        if kernel_name == "rbf":
            gamma_value = _resolve_rbf_gamma(self.gamma, X_scaled)
            self.gamma_ = float(gamma_value)
            self.feature_map_ = RBFSampler(
                gamma=float(gamma_value),
                n_components=max(1, int(self.rff_components)),
                random_state=int(self.random_state),
            )
            return self.feature_map_.fit_transform(X_scaled).astype(
                np.float32,
                copy=False,
            )
        return X_scaled

    def _transform_features(self, X: Any) -> np.ndarray:
        if getattr(self, "backend_", None) == "sklearn_svc":
            return _as_float32_matrix(X)
        X_arr = _as_float32_matrix(X)
        X_scaled = self.scaler_.transform(X_arr).astype(np.float32, copy=False)
        if self.feature_map_ is None:
            return X_scaled
        return self.feature_map_.transform(X_scaled).astype(np.float32, copy=False)

    def _build_legacy_svc(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(self.C),
                        kernel=str(self.kernel),
                        gamma=self.gamma,
                        degree=int(self.degree),
                        coef0=float(self.coef0),
                        probability=bool(self.probability),
                        cache_size=float(self.cache_size),
                        class_weight=_normalize_class_weight(self.class_weight),
                        random_state=int(self.random_state),
                    ),
                ),
            ]
        )

    def _fit_linear_backend(self, X_features: np.ndarray, y: np.ndarray) -> None:
        linear_model = LinearSVC(
            C=float(self.C),
            class_weight=_normalize_class_weight(self.class_weight),
            random_state=int(self.random_state),
            max_iter=max(2000, int(self.epochs) * 100),
            dual="auto",
        )
        linear_model.fit(X_features, y)
        self.linear_model_ = linear_model
        self.coef_ = np.asarray(linear_model.coef_, dtype=np.float32)
        self.intercept_ = np.asarray(linear_model.intercept_, dtype=np.float32)

    def fit(self, X: Any, y: Any):
        X_arr = _as_float32_matrix(X)
        y_arr = np.asarray(y).astype(int)
        if np.unique(y_arr).size < 2:
            raise ValueError("La SVM requiere dos clases para entrenar.")

        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = int(X_arr.shape[1])
        self.backend_ = ""
        self.fit_warning_ = ""

        kernel_name = _normalize_kernel_name(self.kernel)
        if kernel_name not in {"linear", "rbf"}:
            if bool(self.require_mlx):
                self.backend_ = "unsupported_mlx_kernel"
                self.fit_warning_ = (
                    f"kernel={kernel_name!r} no tiene backend MLX. "
                    "Use kernel='linear' o kernel='rbf'."
                )
                raise RuntimeError(self.fit_warning_)
            self.backend_ = "sklearn_svc"
            self.legacy_model_ = self._build_legacy_svc()
            self.legacy_model_.fit(X, y_arr)
            return self

        X_features = self._fit_feature_map(X_arr)
        pos_weight = _resolve_positive_class_weight(y_arr, self.class_weight)

        try:
            weights, bias = _train_linear_svm_with_mlx(
                X_features,
                y_arr,
                C=float(self.C),
                learning_rate=float(self.learning_rate),
                epochs=int(self.epochs),
                batch_size=int(self.batch_size),
                pos_weight=float(pos_weight),
                random_state=int(self.random_state),
            )
            self.backend_ = "mlx"
            self.coef_ = np.asarray(weights, dtype=np.float32).reshape(1, -1)
            self.intercept_ = np.asarray([bias], dtype=np.float32)
        except Exception as exc:
            if bool(self.require_mlx):
                self.backend_ = "mlx_failed"
                self.fit_warning_ = str(exc)
                raise RuntimeError(
                    "SVM requiere backend MLX/Metal, pero el entrenamiento MLX "
                    f"fallo: {exc}"
                ) from exc
            self.backend_ = "sklearn_linear"
            self.fit_warning_ = str(exc)
            self._fit_linear_backend(X_features, y_arr)

        train_scores = self.decision_function(X_arr)
        self._fit_probability_model(train_scores, y_arr)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        if getattr(self, "backend_", None) == "sklearn_svc":
            return np.asarray(self.legacy_model_.decision_function(X), dtype=float)

        X_features = self._transform_features(X)
        if getattr(self, "backend_", None) == "sklearn_linear":
            scores = self.linear_model_.decision_function(X_features)
            return np.asarray(scores, dtype=float).reshape(-1)

        scores = X_features @ np.asarray(self.coef_, dtype=np.float32).reshape(-1)
        scores = scores + float(np.asarray(self.intercept_, dtype=np.float32).reshape(-1)[0])
        return np.asarray(scores, dtype=float).reshape(-1)

    def predict_proba(self, X: Any) -> np.ndarray:
        if getattr(self, "backend_", None) == "sklearn_svc":
            if hasattr(self.legacy_model_, "predict_proba"):
                try:
                    return np.asarray(self.legacy_model_.predict_proba(X), dtype=float)
                except Exception:
                    pass
            scores = self.legacy_model_.decision_function(X)
            pos = _sigmoid_scores(scores)
            return np.column_stack([1.0 - pos, pos])

        scores = self.decision_function(X)
        if getattr(self, "probability_model_", None) is not None:
            pos = self.probability_model_.predict_proba(
                np.asarray(scores, dtype=float).reshape(-1, 1)
            )[:, 1]
        else:
            pos = _sigmoid_scores(scores)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: Any) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)
