"""
Shared model training and evaluation logic for the Crash Prediction App.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import importlib
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

XAI_BACKGROUND_MAX_ROWS = 128
XAI_EXPLAIN_MAX_ROWS = 64
THRESHOLD_PROTOCOL_LABELS = {
    "conservative": "Conservador",
    "robust": "Robusto",
}
THRESHOLD_OBJECTIVE_LABELS = {
    "far": "FAR",
    "f1": "F1",
    "balanced_f1": "Balanced F1",
    "mcc": "MCC",
    "recall_at_alerts_per_day": "Recall@N alertas/dia",
    "operational_cost": "Costo operacional",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
}
OPTUNA_OBJECTIVE_LABELS = {
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "accuracy": "Accuracy",
    "recall": "Recall",
    "precision": "Precision",
    "fnr": "FNR",
    "far_sens": "FAR - Sensibilidad",
    "balanced_f1": "Balanced F1",
    "mcc": "MCC",
    "brier_score": "Brier",
    "recall_at_alerts_per_day": "Recall@N alertas/dia",
    "operational_cost": "Costo operacional",
    "net_balanced_rate": "(TP-FP)/P + (TN-FN)/N",
}
OPTUNA_OBJECTIVE_DIRECTIONS = {
    "f1": "maximize",
    "roc_auc": "maximize",
    "pr_auc": "maximize",
    "accuracy": "maximize",
    "recall": "maximize",
    "precision": "maximize",
    "fnr": "minimize",
    "far_sens": "minimize",
    "balanced_f1": "maximize",
    "mcc": "maximize",
    "brier_score": "minimize",
    "recall_at_alerts_per_day": "maximize",
    "operational_cost": "minimize",
    "net_balanced_rate": "maximize",
}
CALIBRATION_METHOD_LABELS = {
    "none": "Sin calibración",
    "sigmoid": "Platt scaling (sigmoid)",
    "isotonic": "Isotonic",
}
BALANCE_STRATEGY_LABELS = {
    "none": "Sin balance interno",
    "class_weight": "Class weight / scale_pos_weight",
    "smote": "SMOTE interno",
}


def _coerce_threshold_n_jobs(n_jobs: Optional[object]) -> int:
    try:
        requested = int(n_jobs if n_jobs is not None else 1)
    except (TypeError, ValueError):
        requested = 1
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(requested, cpu_count))


def normalize_threshold_protocol(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "conservador": "conservative",
        "conservative": "conservative",
        "robusto": "robust",
        "robust": "robust",
    }
    return aliases.get(text, "conservative")


def normalize_threshold_objective(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "far": "far",
        "fpr": "far",
        "f1": "f1",
        "balanced f1": "balanced_f1",
        "balanced_f1": "balanced_f1",
        "macro_f1": "balanced_f1",
        "f1_global": "balanced_f1",
        "mcc": "mcc",
        "recall@n": "recall_at_alerts_per_day",
        "recall_at_n": "recall_at_alerts_per_day",
        "recall_at_alerts_per_day": "recall_at_alerts_per_day",
        "recall_alerts": "recall_at_alerts_per_day",
        "operational_cost": "operational_cost",
        "cost": "operational_cost",
        "costo operacional": "operational_cost",
        "pr_auc": "pr_auc",
        "pr-auc": "pr_auc",
        "average_precision": "pr_auc",
        "roc_auc": "roc_auc",
        "roc-auc": "roc_auc",
        "auc": "roc_auc",
    }
    return aliases.get(text, "far")


def normalize_optuna_objective_metric(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": "f1",
        "best_f1": "f1",
        "f1": "f1",
        "auc": "roc_auc",
        "rocauc": "roc_auc",
        "roc_auc": "roc_auc",
        "roc-auc": "roc_auc",
        "pr_auc": "pr_auc",
        "pr-auc": "pr_auc",
        "average_precision": "pr_auc",
        "accuracy": "accuracy",
        "acc": "accuracy",
        "recall": "recall",
        "sensitivity": "recall",
        "precision": "precision",
        "fnr": "fnr",
        "false_negative_rate": "fnr",
        "far_sens": "far_sens",
        "far_sensitivity": "far_sens",
        "far_minus_sens": "far_sens",
        "balanced_f1": "balanced_f1",
        "balanced f1": "balanced_f1",
        "macro_f1": "balanced_f1",
        "f1_global": "balanced_f1",
        "mcc": "mcc",
        "brier": "brier_score",
        "brier_score": "brier_score",
        "brier score": "brier_score",
        "brier_loss": "brier_score",
        "recall@n": "recall_at_alerts_per_day",
        "recall_at_n": "recall_at_alerts_per_day",
        "recall_at_alerts_per_day": "recall_at_alerts_per_day",
        "recall_alerts": "recall_at_alerts_per_day",
        "operational_cost": "operational_cost",
        "cost": "operational_cost",
        "costo operacional": "operational_cost",
        "net_balanced_rate": "net_balanced_rate",
        "(tp-fp)/p + (tn-fn)/n": "net_balanced_rate",
    }
    return aliases.get(text, "f1")


def optuna_objective_direction(metric: object) -> str:
    metric_key = normalize_optuna_objective_metric(metric)
    return OPTUNA_OBJECTIVE_DIRECTIONS.get(metric_key, "maximize")


def normalize_calibration_method(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "sin calibracion": "none",
        "sin calibración": "none",
        "no": "none",
        "sigmoid": "sigmoid",
        "platt": "sigmoid",
        "platt scaling": "sigmoid",
        "platt scaling (sigmoid)": "sigmoid",
        "isotonic": "isotonic",
        "isotonic_regression": "isotonic",
    }
    return aliases.get(text, "none")


def normalize_balance_strategy(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "sin balance interno": "none",
        "class_weight": "class_weight",
        "class weight": "class_weight",
        "scale_pos_weight": "class_weight",
        "class_weight/scale_pos_weight": "class_weight",
        "smote": "smote",
        "smote interno": "smote",
    }
    return aliases.get(text, "none")


def _import_external_xgboost():
    src_dir = str(Path(__file__).resolve().parent)
    original_sys_path = list(sys.path)
    existing_module = sys.modules.get("xgboost")
    removed_local_module = None
    try:
        if existing_module is not None:
            module_file = Path(str(getattr(existing_module, "__file__", "") or "")).resolve()
            if module_file == (Path(src_dir) / "xgboost.py").resolve():
                removed_local_module = sys.modules.pop("xgboost")
        sys.path = [
            entry
            for entry in original_sys_path
            if str(Path(entry or ".").resolve()) != src_dir
        ]
        xgb = importlib.import_module("xgboost")  # type: ignore
    finally:
        sys.path = original_sys_path
        if removed_local_module is not None:
            sys.modules["xgboost"] = removed_local_module

    module_path = Path(str(getattr(xgb, "__file__", "") or "")).resolve()
    if module_path == (Path(src_dir) / "xgboost.py").resolve():
        raise ImportError(
            "Se importo el modulo local `src/xgboost.py` en lugar del paquete externo `xgboost`. "
            "Revise el entorno o renombre el modulo local para evitar sombreado."
        )
    if not hasattr(xgb, "XGBClassifier"):
        raise ImportError(
            "El paquete `xgboost` importado no expone `XGBClassifier`. "
            f"Modulo cargado: {module_path}"
        )
    return xgb


def _coerce_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "si", "sí", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", ""}:
        return False
    return bool(default)


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "false"}:
        return None
    return float(value)


class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around MLPNet for the crash prediction pipeline."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        epochs: int = 30,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.0,
        pos_weight: float = 1.0,
        val_fraction: float = 0.15,
        random_state: int = 42,
        use_batch_norm: bool = False,
        hidden_activation: str = "relu",
        output_activation: str = "softmax",
        loss_function: str = "cross_entropy",
        optimizer_name: str = "adamw",
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        lr_scheduler: str = "none",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        min_lr: float = 1e-6,
        temperature_scaling: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.pos_weight = pos_weight
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.optimizer_name = optimizer_name
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler = lr_scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr
        self.temperature_scaling = temperature_scaling
        self.model_ = None
        self.in_dim_ = None
        self.device_ = None
        self.classes_ = np.array([0, 1])
        self.temperature_ = 1.0
        self.train_loss_history_: List[float] = []
        self.lr_history_: List[float] = []
        self.val_loss_history_: List[float] = []
        self.epochs_ran_: int = 0
        self.early_stopping_used_: bool = False
        self.early_stopping_triggered_: bool = False
        self.best_epoch_: Optional[int] = None
        self.best_monitor_value_: Optional[float] = None

    @staticmethod
    def _normalize_loss_function(value: object) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "": "cross_entropy",
            "ce": "cross_entropy",
            "mce": "cross_entropy",
            "categorical_cross_entropy": "cross_entropy",
            "crossentropy": "cross_entropy",
            "cross_entropy": "cross_entropy",
            "bce": "binary_cross_entropy",
            "bce_with_logits": "binary_cross_entropy",
            "binary_cross_entropy": "binary_cross_entropy",
            "focal": "focal",
            "focal_loss": "focal",
        }
        return aliases.get(text, "cross_entropy")

    @staticmethod
    def _normalize_lr_scheduler(value: object) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "": "none",
            "none": "none",
            "off": "none",
            "false": "none",
            "reduce_lr_on_plateau": "reduce_on_plateau",
            "reduce_on_plateau": "reduce_on_plateau",
            "plateau": "reduce_on_plateau",
        }
        return aliases.get(text, "none")

    @staticmethod
    def _normalize_hidden_activation(value: object) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "": "relu",
            "relu": "relu",
            "gelu": "gelu",
            "elu": "elu",
            "leaky_relu": "leaky_relu",
            "leakyrelu": "leaky_relu",
            "tanh": "tanh",
        }
        return aliases.get(text, "relu")

    @staticmethod
    def _normalize_output_activation(value: object) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "": "softmax",
            "softmax": "softmax",
            "sigmoid": "sigmoid",
        }
        return aliases.get(text, "softmax")

    @staticmethod
    def _normalize_optimizer_name(value: object) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "": "adamw",
            "adamw": "adamw",
            "adam": "adam",
            "rmsprop": "rmsprop",
        }
        return aliases.get(text, "adamw")

    @staticmethod
    def _positive_logit(logits):
        import torch

        if logits.ndim == 1:
            return logits
        if logits.size(1) <= 1:
            return logits.reshape(-1)
        return logits[:, 1] - logits[:, 0]

    def _positive_probability(self, logits):
        import torch
        import torch.nn.functional as F

        output_key = self._normalize_output_activation(self.output_activation)
        if output_key == "sigmoid":
            return torch.sigmoid(self._positive_logit(logits))
        return F.softmax(logits, dim=1)[:, 1]

    def _compute_loss(self, logits, targets, class_weight):
        import torch
        import torch.nn.functional as F

        loss_key = self._normalize_loss_function(self.loss_function)
        if loss_key == "binary_cross_entropy":
            pos_logits = self._positive_logit(logits)
            pos_weight = torch.tensor(
                float(self.pos_weight),
                dtype=logits.dtype,
                device=logits.device,
            )
            return F.binary_cross_entropy_with_logits(
                pos_logits,
                targets.float(),
                pos_weight=pos_weight,
            )
        if loss_key != "focal":
            return F.cross_entropy(logits, targets, weight=class_weight)

        ce = F.cross_entropy(logits, targets, weight=class_weight, reduction="none")
        base_ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-base_ce)
        loss = ((1.0 - pt) ** float(self.focal_gamma)) * ce
        if self.focal_alpha is not None:
            alpha_pos = float(self.focal_alpha)
            if not 0.0 <= alpha_pos <= 1.0:
                raise ValueError("focal_alpha debe estar entre 0 y 1.")
            alpha = torch.tensor(
                alpha_pos,
                dtype=loss.dtype,
                device=loss.device,
            )
            alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)
            loss = alpha_t * loss
        return loss.mean()

    @staticmethod
    def _fit_temperature_from_logits(logits, labels) -> float:
        import torch
        import torch.nn.functional as F

        logits_cpu = logits.detach().cpu().clone()
        labels_cpu = labels.detach().cpu().long().clone()
        log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)

        def _closure():
            optimizer.zero_grad(set_to_none=True)
            temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
            loss = F.cross_entropy(logits_cpu / temperature, labels_cpu)
            loss.backward()
            return loss

        optimizer.step(_closure)
        with torch.no_grad():
            return float(torch.exp(log_temperature).clamp(0.05, 10.0).item())

    @staticmethod
    def _resolve_device():
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _fits_on_device(n_rows: int, n_cols: int, device) -> bool:
        """Heuristica: decide si el tensor (fp32) cabe comodamente en memoria.

        Para MPS/CUDA preferimos precargar todo cuando el dataset sea <= 2 GB.
        Para CPU siempre devolvemos True (RAM).
        """
        # 4 bytes por float32. Incluimos y (int64, 8B) -> aproximamos con factor 5.
        est_bytes = int(n_rows) * int(n_cols) * 5
        if device.type == "cpu":
            return True
        # 2 GB por defecto como umbral de seguridad.
        return est_bytes <= 2 * 1024 ** 3

    def fit(
        self,
        X,
        y,
        epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ):
        import torch
        import torch.nn.functional as F
        from src.mlp_tabular import MLPNet

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        self.in_dim_ = X_np.shape[1]
        self.device_ = self._resolve_device()
        self.temperature_ = 1.0
        self.train_loss_history_ = []
        self.lr_history_ = []
        self.val_loss_history_ = []
        self.epochs_ran_ = 0
        self.early_stopping_used_ = False
        self.early_stopping_triggered_ = False
        self.best_epoch_ = None
        self.best_monitor_value_ = None

        torch.manual_seed(self.random_state)

        # cudnn.benchmark acelera capas lineales con inputs de tamano fijo (CUDA).
        if self.device_.type == "cuda":
            torch.backends.cudnn.benchmark = True

        use_early_stopping = (
            self.early_stopping_patience > 0
            and int(self.val_fraction * len(y_np)) >= 2
        )
        self.early_stopping_used_ = bool(use_early_stopping)
        if use_early_stopping:
            from sklearn.model_selection import train_test_split as _split

            minority_count = int(np.min(np.bincount(y_np)))
            stratify = y_np if minority_count >= 2 else None
            X_tr, X_es, y_tr, y_es = _split(
                X_np, y_np,
                test_size=self.val_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        else:
            X_tr, y_tr = X_np, y_np
            X_es, y_es = None, None

        class_weight = torch.tensor(
            [1.0, float(self.pos_weight)],
            dtype=torch.float32,
            device=self.device_,
        )

        model = MLPNet(
            self.in_dim_,
            self.hidden_dim,
            self.num_layers,
            self.dropout,
            use_batch_norm=_coerce_bool(self.use_batch_norm),
            hidden_activation=self._normalize_hidden_activation(
                self.hidden_activation
            ),
        ).to(self.device_)

        optimizer_key = self._normalize_optimizer_name(self.optimizer_name)
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        if optimizer_key == "adamw":
            if self.device_.type == "cuda":
                try:
                    optimizer = torch.optim.AdamW(
                        model.parameters(), fused=True, **optimizer_kwargs
                    )
                except (TypeError, RuntimeError):
                    optimizer = torch.optim.AdamW(
                        model.parameters(), **optimizer_kwargs
                    )
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(), **optimizer_kwargs
                )
        elif optimizer_key == "adam":
            if self.device_.type == "cuda":
                try:
                    optimizer = torch.optim.Adam(
                        model.parameters(), fused=True, **optimizer_kwargs
                    )
                except (TypeError, RuntimeError):
                    optimizer = torch.optim.Adam(
                        model.parameters(), **optimizer_kwargs
                    )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), **optimizer_kwargs
                )
        elif optimizer_key == "rmsprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(), **optimizer_kwargs
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

        scheduler_key = self._normalize_lr_scheduler(self.lr_scheduler)
        scheduler = None
        if scheduler_key == "reduce_on_plateau" and use_early_stopping:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.scheduler_factor),
                patience=max(0, int(self.scheduler_patience)),
                min_lr=float(self.min_lr),
            )

        preload_to_device = self._fits_on_device(
            X_tr.shape[0], X_tr.shape[1], self.device_
        )

        # Modo rapido (tabular <= 2GB): cargamos todo a device una sola vez y
        # evitamos H2D por batch. Sliceamos con index_select sobre un tensor
        # de indices generado en device. Esto es clave para saturar la GPU
        # (MPS/CUDA), donde el H2D repetido dominaba el tiempo.
        if preload_to_device:
            X_tr_t = torch.from_numpy(X_tr).to(
                self.device_, non_blocking=True
            )
            y_tr_t = torch.from_numpy(y_tr).to(
                self.device_, non_blocking=True
            )
            if X_es is not None:
                X_es_t = torch.from_numpy(X_es).to(
                    self.device_, non_blocking=True
                )
            else:
                X_es_t = None
        else:
            # Fallback streaming via DataLoader (datasets gigantes).
            from torch.utils.data import TensorDataset, DataLoader

            train_ds = TensorDataset(
                torch.from_numpy(X_tr).float(),
                torch.from_numpy(y_tr).long(),
            )
            pin_mem = self.device_.type == "cuda"
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=pin_mem,
            )

        n_train = int(X_tr.shape[0])
        batch_size = max(1, int(self.batch_size))

        best_val_score = -np.inf
        best_state = None
        no_improve = 0

        cpu_gen = torch.Generator(device="cpu").manual_seed(
            int(self.random_state)
        )

        def _emit_epoch_payload(epoch_number: int, status: str = "running") -> None:
            if epoch_callback is None:
                return
            payload: Dict[str, object] = {
                "epoch": int(epoch_number),
                "epochs": list(range(1, len(self.train_loss_history_) + 1)),
                "train_loss": [float(value) for value in self.train_loss_history_],
                "val_loss": [float(value) for value in self.val_loss_history_],
                "epochs_ran": int(
                    max(
                        len(self.train_loss_history_),
                        len(self.val_loss_history_),
                    )
                ),
                "max_epochs": int(self.epochs),
                "early_stopping_used": bool(self.early_stopping_used_),
                "early_stopping_triggered": bool(
                    self.early_stopping_triggered_
                ),
                "early_stopping_patience": int(
                    self.early_stopping_patience
                ),
                "early_stopping_min_delta": float(
                    self.early_stopping_min_delta
                ),
                "status": str(status),
            }
            if self.best_epoch_ is not None:
                payload["best_epoch"] = int(self.best_epoch_)
            if self.best_monitor_value_ is not None:
                payload["best_monitor_value"] = float(
                    self.best_monitor_value_
                )
            try:
                epoch_callback(payload)
            except Exception:
                pass

        for _epoch in range(self.epochs):
            model.train()
            epoch_loss_total = 0.0
            epoch_seen = 0
            if preload_to_device:
                # Permutacion en CPU (randperm en MPS no es siempre estable)
                # y la subimos al device una sola vez por epoca.
                perm = torch.randperm(n_train, generator=cpu_gen)
                perm_dev = perm.to(self.device_, non_blocking=True)
                for start in range(0, n_train, batch_size):
                    idx = perm_dev[start : start + batch_size]
                    xb = X_tr_t.index_select(0, idx)
                    yb = y_tr_t.index_select(0, idx)
                    if _coerce_bool(self.use_batch_norm) and yb.size(0) < 2:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._compute_loss(model(xb), yb, class_weight)
                    loss.backward()
                    if self.max_grad_norm is not None and float(self.max_grad_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(self.max_grad_norm)
                        )
                    optimizer.step()
                    batch_seen = int(yb.size(0))
                    epoch_loss_total += float(loss.detach().cpu().item()) * batch_seen
                    epoch_seen += batch_seen
            else:
                non_block = self.device_.type == "cuda"
                for xb, yb in train_loader:
                    xb = xb.to(self.device_, non_blocking=non_block)
                    yb = yb.to(self.device_, non_blocking=non_block)
                    if _coerce_bool(self.use_batch_norm) and yb.size(0) < 2:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._compute_loss(model(xb), yb, class_weight)
                    loss.backward()
                    if self.max_grad_norm is not None and float(self.max_grad_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(self.max_grad_norm)
                        )
                    optimizer.step()
                    batch_seen = int(yb.size(0))
                    epoch_loss_total += float(loss.detach().cpu().item()) * batch_seen
                    epoch_seen += batch_seen

            epoch_train_loss = (
                float(epoch_loss_total / epoch_seen) if epoch_seen > 0 else float("nan")
            )
            self.train_loss_history_.append(epoch_train_loss)
            if epoch_seen > 0:
                self.lr_history_.append(float(optimizer.param_groups[0]["lr"]))

            should_stop = False
            if use_early_stopping and X_es is not None:
                model.eval()
                with torch.inference_mode():
                    if preload_to_device and X_es_t is not None:
                        es_tensor = X_es_t
                    else:
                        es_tensor = torch.from_numpy(X_es).to(
                            self.device_, non_blocking=True
                        )
                    es_out = model(es_tensor)
                    es_targets = torch.from_numpy(y_es).to(
                        self.device_, non_blocking=True
                    )
                    val_loss = float(
                        self._compute_loss(es_out, es_targets, class_weight)
                        .detach()
                        .cpu()
                        .item()
                    )
                    es_probs = self._positive_probability(es_out).detach().cpu().numpy()
                from sklearn.metrics import average_precision_score

                self.val_loss_history_.append(val_loss)
                if scheduler is not None:
                    scheduler.step(val_loss)

                if len(np.unique(y_es)) > 1:
                    val_score = average_precision_score(y_es, es_probs)
                else:
                    val_score = 0.0

                min_delta = max(float(self.early_stopping_min_delta), 0.0)
                if val_score > best_val_score + min_delta:
                    best_val_score = val_score
                    self.best_monitor_value_ = float(val_score)
                    self.best_epoch_ = int(_epoch) + 1
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.early_stopping_patience:
                    self.early_stopping_triggered_ = True
                    should_stop = True

            self.epochs_ran_ = max(
                len(self.train_loss_history_),
                len(self.val_loss_history_),
            )
            _emit_epoch_payload(
                int(_epoch) + 1,
                status="completed" if should_stop else "running",
            )
            if should_stop:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(self.device_)
        if (
            _coerce_bool(self.temperature_scaling)
            and X_es is not None
            and len(np.unique(y_es)) > 1
        ):
            model.eval()
            with torch.inference_mode():
                if preload_to_device and X_es_t is not None:
                    es_tensor = X_es_t
                else:
                    es_tensor = torch.from_numpy(X_es).to(
                        self.device_, non_blocking=True
                    )
                logits = model(es_tensor)
                labels = torch.from_numpy(y_es).long()
            self.temperature_ = self._fit_temperature_from_logits(logits, labels)
        else:
            self.temperature_ = 1.0
        self.epochs_ran_ = max(
            len(self.train_loss_history_),
            len(self.val_loss_history_),
        )
        _emit_epoch_payload(int(self.epochs_ran_), status="finished")
        self.model_ = model
        return self

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F

        X_np = np.asarray(X, dtype=np.float32)
        n = int(X_np.shape[0])
        # Chunk para evitar picos de memoria en inferencia sobre datasets grandes.
        # Batch grande pero acotado ~= saturacion de GPU sin OOM.
        chunk = max(1, min(n, 65536))
        out_parts: list = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                tensor = torch.from_numpy(X_np[start:end]).to(
                    self.device_, non_blocking=True
                )
                logits = self.model_(tensor)
                temperature = max(float(getattr(self, "temperature_", 1.0)), 1e-6)
                if (
                    self._normalize_output_activation(self.output_activation)
                    == "sigmoid"
                ):
                    pos_logits = self._positive_logit(logits) / temperature
                    pos_probs = torch.sigmoid(pos_logits)
                    probs_chunk = torch.stack(
                        [1.0 - pos_probs, pos_probs],
                        dim=1,
                    ).detach().cpu().numpy()
                else:
                    logits = logits / temperature
                    probs_chunk = F.softmax(logits, dim=1).detach().cpu().numpy()
                out_parts.append(probs_chunk)
        if not out_parts:
            return np.empty((0, 2), dtype=np.float32)
        return np.concatenate(out_parts, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def _extract_training_curves(model: object) -> Dict[str, object]:
    wrapper = model
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        wrapper = named_steps.get("model")

    train_loss = getattr(wrapper, "train_loss_history_", None)
    val_loss = getattr(wrapper, "val_loss_history_", None)
    if not isinstance(train_loss, list) and not isinstance(val_loss, list):
        return {}

    train_series = (
        [float(value) for value in train_loss]
        if isinstance(train_loss, list)
        else []
    )
    val_series = (
        [float(value) for value in val_loss]
        if isinstance(val_loss, list)
        else []
    )
    max_len = max(len(train_series), len(val_series))
    if max_len <= 0:
        return {}

    best_epoch = getattr(wrapper, "best_epoch_", None)
    epochs_ran = getattr(wrapper, "epochs_ran_", max_len)
    payload: Dict[str, object] = {
        "epochs": list(range(1, max_len + 1)),
        "train_loss": train_series,
        "val_loss": val_series,
        "epochs_ran": int(epochs_ran),
        "max_epochs": int(getattr(wrapper, "epochs", max_len)),
        "early_stopping_used": bool(
            getattr(wrapper, "early_stopping_used_", False)
        ),
        "early_stopping_triggered": bool(
            getattr(wrapper, "early_stopping_triggered_", False)
        ),
        "early_stopping_patience": int(
            getattr(wrapper, "early_stopping_patience", 0)
        ),
        "early_stopping_min_delta": float(
            getattr(wrapper, "early_stopping_min_delta", 0.0)
        ),
        "monitor_metric": "average_precision",
    }
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    best_monitor_value = getattr(wrapper, "best_monitor_value_", None)
    if best_monitor_value is not None:
        payload["best_monitor_value"] = float(best_monitor_value)
    return payload


def _normalize_class_weight(value: object) -> Optional[object]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "false"}:
        return None
    if text == "balanced":
        return "balanced"
    return value


def build_model(model_name: str, params: Dict[str, object], random_state: int):
    if model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=random_state,
            class_weight=_normalize_class_weight(
                params.get("class_weight", "balanced")
            ),
            n_jobs=params.get("n_jobs"),
        )

    if model_name == "Balanced Random Forest":
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier
        except ImportError as exc:
            raise ImportError(
                "Balanced Random Forest requiere `imbalanced-learn` instalado."
            ) from exc

        return BalancedRandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=params.get("n_jobs"),
            replacement=bool(params.get("replacement", False)),
        )

    if model_name == "XGBoost":
        try:
            xgb = _import_external_xgboost()
        except ImportError as exc:
            raise ImportError(
                "No se pudo cargar el paquete externo `xgboost`. "
                "Instale `xgboost` o corrija el sombreado del modulo local."
            ) from exc

        xgb_params = {
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "min_child_weight": float(params.get("min_child_weight", 1.0)),
            "reg_alpha": float(params.get("reg_alpha", 0.0)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "gamma": float(params.get("gamma", 0.0)),
            "n_jobs": int(params.get("n_jobs", 1)),
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        if params.get("scale_pos_weight") is not None:
            xgb_params["scale_pos_weight"] = float(params["scale_pos_weight"])
        if params.get("max_delta_step") is not None:
            xgb_params["max_delta_step"] = float(params["max_delta_step"])

        return xgb.XGBClassifier(**xgb_params)

    if model_name == "SVM":
        from src.mlx_svm import MLXAcceleratedSVMClassifier

        probability = bool(params.get("probability", True))
        cache_size = float(params.get("cache_size", 200.0))

        return MLXAcceleratedSVMClassifier(
            C=float(params["C"]),
            kernel=str(params["kernel"]),
            gamma=params.get("gamma", "scale"),
            degree=int(params.get("degree", 3)),
            coef0=float(params.get("coef0", 0.0)),
            probability=probability,
            cache_size=cache_size,
            class_weight=_normalize_class_weight(params.get("class_weight")),
            random_state=random_state,
            learning_rate=float(params.get("learning_rate", 1e-3)),
            epochs=int(params.get("epochs", 40)),
            batch_size=int(params.get("batch_size", 8192)),
            rff_components=int(params.get("rff_components", 2048)),
        )

    if model_name == "Neural Network":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifierWrapper(
                        hidden_dim=int(params.get("hidden_dim", 256)),
                        num_layers=int(params.get("num_layers", 2)),
                        dropout=float(params.get("dropout", 0.2)),
                        learning_rate=float(params.get("learning_rate", 1e-3)),
                        weight_decay=float(params.get("weight_decay", 1e-5)),
                        batch_size=int(params.get("batch_size", 1024)),
                        epochs=int(params.get("epochs", 30)),
                        early_stopping_patience=int(
                            params.get("early_stopping_patience", 5)
                        ),
                        early_stopping_min_delta=float(
                            params.get("early_stopping_min_delta", 0.0)
                        ),
                        pos_weight=float(params.get("pos_weight", 1.0)),
                        val_fraction=float(params.get("val_fraction", 0.15)),
                        random_state=random_state,
                        use_batch_norm=_coerce_bool(
                            params.get("use_batch_norm", False)
                        ),
                        hidden_activation=str(
                            params.get("hidden_activation", "relu")
                        ),
                        output_activation=str(
                            params.get("output_activation", "softmax")
                        ),
                        loss_function=str(
                            params.get("loss_function", "cross_entropy")
                        ),
                        optimizer_name=str(
                            params.get("optimizer_name", params.get("optimizer", "adamw"))
                        ),
                        focal_gamma=float(params.get("focal_gamma", 2.0)),
                        focal_alpha=_coerce_optional_float(
                            params.get("focal_alpha")
                        ),
                        max_grad_norm=_coerce_optional_float(
                            params.get("max_grad_norm")
                        ),
                        lr_scheduler=str(params.get("lr_scheduler", "none")),
                        scheduler_factor=float(
                            params.get("scheduler_factor", 0.5)
                        ),
                        scheduler_patience=int(
                            params.get("scheduler_patience", 2)
                        ),
                        min_lr=float(params.get("min_lr", 1e-6)),
                        temperature_scaling=_coerce_bool(
                            params.get("temperature_scaling", False)
                        ),
                    ),
                ),
            ]
        )

    raise ValueError(f"Modelo no soportado: {model_name}")


def temporal_train_test_split(
    df: pd.DataFrame,
    *,
    time_col: str = "interval_start",
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if time_col not in df.columns:
        raise ValueError(
            f"No se encontro la columna '{time_col}' para split temporal."
        )
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    work_df = df.copy()
    work_df["_split_time"] = pd.to_datetime(
        work_df[time_col], errors="coerce"
    )
    work_df = work_df.dropna(subset=["_split_time"])
    if work_df.empty:
        raise ValueError("No hay timestamps validos para split temporal.")

    unique_times = np.sort(work_df["_split_time"].unique())
    if len(unique_times) < 2:
        raise ValueError("No hay suficientes timestamps para split temporal.")

    test_count = max(1, int(round(len(unique_times) * float(test_size))))
    if test_count >= len(unique_times):
        test_count = len(unique_times) - 1
    split_idx = len(unique_times) - test_count
    train_times = unique_times[:split_idx]
    test_times = unique_times[split_idx:]

    train_df = work_df[work_df["_split_time"].isin(train_times)].drop(
        columns=["_split_time"]
    )
    test_df = work_df[work_df["_split_time"].isin(test_times)].drop(
        columns=["_split_time"]
    )
    if train_df.empty or test_df.empty:
        raise ValueError("No hay suficientes datos para split temporal.")
    return train_df, test_df


def get_model_scores(model, X: pd.DataFrame) -> np.ndarray:
    if (
        getattr(model, "_score_preference", "") == "decision_function"
        and hasattr(model, "decision_function")
    ):
        try:
            return np.asarray(model.decision_function(X), dtype=float)
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def far_and_sensitivity(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(far), float(sens)


def select_threshold_for_far(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    far_target: float = 0.20,
    *,
    mode: str = "max_sens_under_far",
) -> Dict[str, object]:
    y_val = np.asarray(y_val).astype(int)
    scores_val = np.asarray(scores_val).astype(float)

    if np.unique(y_val).size < 2:
        return {
            "threshold": 0.5,
            "far_val": np.nan,
            "sens_val": np.nan,
            "note": "Validacion con una sola clase.",
        }

    fpr, tpr, thr = roc_curve(y_val, scores_val)
    far_target = float(np.clip(far_target, 0.0, 1.0))

    if mode == "closest_far":
        idx = int(np.argmin(np.abs(fpr - far_target)))
        threshold = float(thr[idx])
    else:
        mask = fpr <= (far_target + 1e-12)
        if np.any(mask):
            idx_local = int(np.argmax(tpr[mask]))
            threshold = float(thr[mask][idx_local])
        else:
            idx = int(np.argmin(np.abs(fpr - far_target)))
            threshold = float(thr[idx])

    yhat_val = (scores_val >= threshold).astype(int)
    far_val, sens_val = far_and_sensitivity(y_val, yhat_val)

    return {
        "threshold": threshold,
        "far_val": far_val,
        "sens_val": sens_val,
        "note": "",
    }


def _maybe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_arr = np.asarray(y_true).astype(int)
    if np.unique(y_arr).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_arr, np.asarray(scores, dtype=float)))
    except Exception:
        return float("nan")


def _maybe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_arr = np.asarray(y_true).astype(int)
    if np.unique(y_arr).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_arr, np.asarray(scores, dtype=float)))
    except Exception:
        return float("nan")


def _maybe_brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    valid_mask = np.isfinite(scores_arr)
    if y_arr.size == 0 or not np.any(valid_mask):
        return float("nan")
    try:
        clipped_scores = np.clip(scores_arr[valid_mask], 0.0, 1.0)
        return float(brier_score_loss(y_arr[valid_mask], clipped_scores))
    except Exception:
        return float("nan")


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.5], dtype=float)
    candidates = np.unique(arr)
    if candidates.size > 256:
        candidates = np.unique(
            np.round(np.quantile(candidates, np.linspace(0.0, 1.0, 257)), 10)
        )
    span = float(np.nanmax(candidates) - np.nanmin(candidates))
    eps = max(1e-12, span * 1e-9)
    candidates = np.append(candidates, [float(np.nanmax(candidates) + eps)])
    candidates = np.append(candidates, [float(np.nanmin(candidates) - eps)])
    if not np.any(np.isclose(candidates, 0.5)):
        candidates = np.append(candidates, 0.5)
    return np.sort(np.unique(candidates.astype(float)))


def estimate_frame_days(df: Optional[pd.DataFrame], row_count: int) -> float:
    if isinstance(df, pd.DataFrame) and "interval_start" in df.columns:
        times = pd.to_datetime(df["interval_start"], errors="coerce").dropna()
        if not times.empty:
            span_days = (
                times.max() - times.min()
            ).total_seconds() / 86400.0
            if span_days > 0:
                return max(span_days, 1.0 / 288.0)
    return max(float(row_count) / 288.0, 1.0 / 288.0)


def _event_groups_from_frame(
    y_true: np.ndarray,
    eval_df: Optional[pd.DataFrame],
) -> List[np.ndarray]:
    y_arr = np.asarray(y_true).astype(int)
    pos_idx = np.flatnonzero(y_arr == 1)
    if pos_idx.size == 0:
        return []
    if not isinstance(eval_df, pd.DataFrame) or "interval_start" not in eval_df.columns:
        groups: List[List[int]] = [[int(pos_idx[0])]]
        for idx in pos_idx[1:]:
            idx_int = int(idx)
            if idx_int == groups[-1][-1] + 1:
                groups[-1].append(idx_int)
            else:
                groups.append([idx_int])
        return [np.asarray(group, dtype=int) for group in groups]

    work = eval_df.reset_index(drop=True).copy()
    work["_event_pos"] = np.arange(len(work))
    work["_event_time"] = pd.to_datetime(work["interval_start"], errors="coerce")
    work["_event_target"] = y_arr
    segment_cols = [
        col
        for col in ["eje", "calzada", "portico", "ultimo_portico", "segment_id"]
        if col in work.columns
    ]
    sort_cols = ["_event_time"] + segment_cols
    work = work.sort_values(sort_cols).reset_index(drop=True)

    groups_out: List[List[int]] = []
    current: List[int] = []
    current_key: Tuple[object, ...] = tuple()
    last_time: Optional[pd.Timestamp] = None
    expected_step = pd.Timedelta(minutes=5)
    for _, row in work.iterrows():
        if int(row["_event_target"]) != 1:
            continue
        time_value = row["_event_time"]
        key = tuple(row[col] for col in segment_cols)
        pos = int(row["_event_pos"])
        contiguous = False
        if current and key == current_key and pd.notna(time_value) and last_time is not None:
            delta = pd.Timestamp(time_value) - pd.Timestamp(last_time)
            contiguous = pd.Timedelta(0) <= delta <= expected_step
        if not current or not contiguous:
            if current:
                groups_out.append(current)
            current = [pos]
            current_key = key
        else:
            current.append(pos)
        last_time = pd.Timestamp(time_value) if pd.notna(time_value) else None
    if current:
        groups_out.append(current)
    return [np.asarray(group, dtype=int) for group in groups_out]


def _event_group_ids_from_frame(
    y_true: np.ndarray,
    eval_df: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, int]:
    y_arr = np.asarray(y_true).astype(int)
    group_ids = np.full(len(y_arr), -1, dtype=np.int32)
    groups = _event_groups_from_frame(y_arr, eval_df)
    for group_id, group in enumerate(groups):
        group_ids[np.asarray(group, dtype=int)] = int(group_id)
    return group_ids, int(len(groups))


def event_recall_approx(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_df: Optional[pd.DataFrame] = None,
    *,
    event_groups: Optional[Sequence[np.ndarray]] = None,
    event_group_ids: Optional[np.ndarray] = None,
    event_group_count: Optional[int] = None,
) -> float:
    if event_group_ids is not None and event_group_count is not None:
        count = int(event_group_count)
        if count <= 0:
            return float("nan")
        preds = np.asarray(y_pred).astype(int)
        ids = np.asarray(event_group_ids, dtype=np.int32)
        hit_ids = ids[(ids >= 0) & (preds == 1)]
        if hit_ids.size == 0:
            return 0.0
        detected = np.bincount(hit_ids, minlength=count)[:count] > 0
        return float(np.count_nonzero(detected) / count)

    groups = (
        list(event_groups)
        if event_groups is not None
        else _event_groups_from_frame(y_true, eval_df)
    )
    if not groups:
        return float("nan")
    preds = np.asarray(y_pred).astype(int)
    detected = sum(1 for group in groups if np.any(preds[group] == 1))
    return float(detected / len(groups))


def compute_extended_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    eval_df: Optional[pd.DataFrame] = None,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    event_groups: Optional[Sequence[np.ndarray]] = None,
    event_group_ids: Optional[np.ndarray] = None,
    event_group_count: Optional[int] = None,
    recall_at_info: Optional[Dict[str, float]] = None,
    frame_days: Optional[float] = None,
) -> Dict[str, object]:
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    preds = (scores_arr >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()
    far_val, sens_val = far_and_sensitivity(y_arr, preds)
    f1_by_class = f1_score(
        y_arr,
        preds,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    days = (
        float(frame_days)
        if frame_days is not None
        else estimate_frame_days(eval_df, len(y_arr))
    )
    alert_count = int(tp + fp)
    false_alarm_count = int(fp)
    operational_cost = float(fn_cost) * float(fn) + float(fp_cost) * float(fp)
    brier = _maybe_brier_score(y_arr, scores_arr)
    positive_support = int(tp + fn)
    tp_capture = (
        float(tp / positive_support) if positive_support > 0 else float("nan")
    )
    fn_rate = (
        float(fn / positive_support) if positive_support > 0 else float("nan")
    )
    if recall_at_info is None:
        recall_at_info = recall_at_alerts_per_day(
            y_arr,
            scores_arr,
            eval_df=eval_df,
            alerts_per_day=float(alerts_per_day),
        )
    return {
        "accuracy": float(accuracy_score(y_arr, preds)),
        "precision": float(precision_score(y_arr, preds, zero_division=0)),
        "recall": float(recall_score(y_arr, preds, zero_division=0)),
        "sensitivity": float(sens_val),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "far": float(far_val),
        "roc_auc": _maybe_roc_auc(y_arr, scores_arr),
        "pr_auc": _maybe_pr_auc(y_arr, scores_arr),
        "brier_score": brier,
        "brier": brier,
        "f1": float(f1_score(y_arr, preds, zero_division=0)),
        "balanced_f1": float(f1_score(y_arr, preds, average="macro", zero_division=0)),
        "f1_global": float(f1_score(y_arr, preds, average="macro", zero_division=0)),
        "f1_class_0": float(f1_by_class[0]),
        "f1_class_1": float(f1_by_class[1]),
        "mcc": float(matthews_corrcoef(y_arr, preds)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "positive_support": int(positive_support),
        "tp_capture": float(tp_capture),
        "fn_rate": float(fn_rate),
        "alerts": int(alert_count),
        "alerts_per_day": float(alert_count / days),
        "false_alarms_per_day": float(false_alarm_count / days),
        "event_recall_approx": event_recall_approx(
            y_arr,
            preds,
            eval_df,
            event_groups=event_groups,
            event_group_ids=event_group_ids,
            event_group_count=event_group_count,
        ),
        "operational_cost": float(operational_cost),
        "cost_per_day": float(operational_cost / days),
        "recall_at_alerts_per_day": float(recall_at_info["recall"]),
        "threshold_at_alerts_per_day": float(recall_at_info["threshold"]),
        "alerts_per_day_budget": float(alerts_per_day),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def recall_at_alerts_per_day(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    eval_df: Optional[pd.DataFrame] = None,
    alerts_per_day: float = 5.0,
) -> Dict[str, float]:
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    days = estimate_frame_days(eval_df, len(y_arr))
    max_alerts = max(1, int(np.floor(float(alerts_per_day) * days)))
    if scores_arr.size == 0:
        return {"recall": float("nan"), "threshold": 0.5, "alerts": 0.0}
    order = np.argsort(scores_arr)[::-1]
    selected = order[: min(max_alerts, len(order))]
    preds = np.zeros_like(y_arr, dtype=int)
    preds[selected] = 1
    recall_val = recall_score(y_arr, preds, zero_division=0)
    threshold = float(scores_arr[selected].min()) if selected.size else float("inf")
    return {
        "recall": float(recall_val),
        "threshold": float(threshold),
        "alerts": float(selected.size),
    }


def _metric_value_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    objective: str,
    eval_df: Optional[pd.DataFrame],
    alerts_per_day: float,
    fn_cost: float,
    fp_cost: float,
    event_groups: Optional[Sequence[np.ndarray]] = None,
    event_group_ids: Optional[np.ndarray] = None,
    event_group_count: Optional[int] = None,
    recall_at_info: Optional[Dict[str, float]] = None,
    frame_days: Optional[float] = None,
) -> float:
    metrics = compute_extended_metrics(
        y_true,
        scores,
        threshold=float(threshold),
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        event_groups=event_groups,
        event_group_ids=event_group_ids,
        event_group_count=event_group_count,
        recall_at_info=recall_at_info,
        frame_days=frame_days,
    )
    return _metric_value_from_metrics(metrics, objective=objective)


def _metric_value_from_metrics(
    metrics: Dict[str, object],
    *,
    objective: str,
) -> float:
    if objective == "operational_cost":
        return -float(metrics.get("operational_cost", float("inf")))
    if objective == "recall_at_alerts_per_day":
        return float(metrics.get("recall", 0.0))
    if objective == "balanced_f1":
        return float(metrics.get("balanced_f1", 0.0))
    if objective in {"f1", "mcc"}:
        return float(metrics.get(objective, 0.0))
    if objective == "far":
        return float(metrics.get("sensitivity", 0.0))
    if objective == "pr_auc":
        return float(metrics.get("pr_auc", float("nan")))
    if objective == "roc_auc":
        return float(metrics.get("roc_auc", float("nan")))
    return float(metrics.get("f1", 0.0))


def _threshold_candidate_metrics(
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str,
    eval_df: Optional[pd.DataFrame],
    alerts_per_day: float,
    fn_cost: float,
    fp_cost: float,
    event_group_ids: np.ndarray,
    event_group_count: int,
    recall_at_info: Dict[str, float],
    frame_days: float,
) -> Tuple[float, Dict[str, object], float]:
    metrics = compute_extended_metrics(
        y_true,
        scores,
        threshold=float(threshold),
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        event_group_ids=event_group_ids,
        event_group_count=int(event_group_count),
        recall_at_info=recall_at_info,
        frame_days=float(frame_days),
    )
    score = _metric_value_from_metrics(metrics, objective=objective)
    return float(threshold), metrics, float(score)


def _orientation_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str,
) -> float:
    objective_key = normalize_threshold_objective(objective)
    if objective_key == "pr_auc":
        return _maybe_pr_auc(y_true, scores)
    if objective_key == "roc_auc":
        return _maybe_roc_auc(y_true, scores)
    info = select_threshold_for_metric(
        y_true,
        scores,
        objective=objective_key,
    )
    return float(info.get("objective_score", float("nan")))


def orient_scores_for_objective(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str = "roc_auc",
) -> Tuple[np.ndarray, float]:
    raw_scores = np.asarray(scores, dtype=float)
    y_arr = np.asarray(y_true).astype(int)
    if raw_scores.size == 0 or np.unique(y_arr).size < 2:
        return raw_scores, 1.0
    pos_score = _orientation_score(y_arr, raw_scores, objective=objective)
    neg_score = _orientation_score(y_arr, -raw_scores, objective=objective)
    if pd.isna(pos_score) and pd.isna(neg_score):
        return raw_scores, 1.0
    if pd.isna(pos_score):
        return -raw_scores, -1.0
    if pd.isna(neg_score):
        return raw_scores, 1.0
    if float(neg_score) > float(pos_score) + 1e-12:
        return -raw_scores, -1.0
    return raw_scores, 1.0


def select_threshold_for_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str = "far",
    eval_df: Optional[pd.DataFrame] = None,
    far_target: float = 0.20,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    n_jobs: int = 1,
) -> Dict[str, object]:
    objective_key = normalize_threshold_objective(objective)
    threshold_n_jobs = _coerce_threshold_n_jobs(n_jobs)
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    if np.unique(y_arr).size < 2:
        return {
            "threshold": 0.5,
            "objective": objective_key,
            "objective_score": float("nan"),
            "threshold_n_jobs": int(threshold_n_jobs),
            "note": "Validacion con una sola clase.",
        }
    if objective_key in {"pr_auc", "roc_auc"}:
        score = (
            _maybe_pr_auc(y_arr, scores_arr)
            if objective_key == "pr_auc"
            else _maybe_roc_auc(y_arr, scores_arr)
        )
        return {
            "threshold": 0.5,
            "objective": objective_key,
            "objective_score": float(score),
            "threshold_n_jobs": int(threshold_n_jobs),
            "note": "Metrica de ranking; threshold operativo 0.5.",
        }
    frame_days = estimate_frame_days(eval_df, len(y_arr))
    event_group_ids, event_group_count = _event_group_ids_from_frame(y_arr, eval_df)
    recall_at_info = recall_at_alerts_per_day(
        y_arr,
        scores_arr,
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
    )
    if objective_key == "far":
        info = select_threshold_for_far(
            y_arr,
            scores_arr,
            far_target=float(far_target),
        )
        threshold = float(info["threshold"])
        score = _metric_value_at_threshold(
            y_arr,
            scores_arr,
            threshold=threshold,
            objective=objective_key,
            eval_df=eval_df,
            alerts_per_day=alerts_per_day,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            event_group_ids=event_group_ids,
            event_group_count=event_group_count,
            recall_at_info=recall_at_info,
            frame_days=frame_days,
        )
        info.update(
            {
                "objective": objective_key,
                "objective_score": float(score),
                "threshold_n_jobs": int(threshold_n_jobs),
            }
        )
        return info

    best_score = float("-inf")
    best_threshold = 0.5
    best_alerts_per_day = float("inf")
    candidates = threshold_candidates(scores_arr)
    if threshold_n_jobs > 1 and len(candidates) >= max(8, threshold_n_jobs * 2):
        def _run_parallel_thresholds(prefer_backend: str) -> List[Tuple[float, Dict[str, object], float]]:
            return Parallel(
                n_jobs=int(threshold_n_jobs),
                prefer=prefer_backend,
                max_nbytes="1M",
                batch_size="auto",
            )(
                delayed(_threshold_candidate_metrics)(
                    float(threshold),
                    y_arr,
                    scores_arr,
                    objective=objective_key,
                    eval_df=None,
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                    event_group_ids=event_group_ids,
                    event_group_count=int(event_group_count),
                    recall_at_info=recall_at_info,
                    frame_days=float(frame_days),
                )
                for threshold in candidates
            )

        try:
            evaluated = _run_parallel_thresholds("processes")
        except (OSError, NotImplementedError, RuntimeError):
            evaluated = _run_parallel_thresholds("threads")
    else:
        evaluated = [
            _threshold_candidate_metrics(
                float(threshold),
                y_arr,
                scores_arr,
                objective=objective_key,
                eval_df=None,
                alerts_per_day=float(alerts_per_day),
                fn_cost=float(fn_cost),
                fp_cost=float(fp_cost),
                event_group_ids=event_group_ids,
                event_group_count=int(event_group_count),
                recall_at_info=recall_at_info,
                frame_days=float(frame_days),
            )
            for threshold in candidates
        ]
    for threshold, metrics, score in evaluated:
        if objective_key == "recall_at_alerts_per_day" and float(
            metrics.get("alerts_per_day", float("inf"))
        ) > float(alerts_per_day) + 1e-12:
            continue
        if pd.isna(score):
            continue
        alerts_value = float(metrics.get("alerts_per_day", float("inf")))
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)
            best_alerts_per_day = alerts_value
            continue
        if abs(score - best_score) <= 1e-12:
            if alerts_value < best_alerts_per_day - 1e-12:
                best_threshold = float(threshold)
                best_alerts_per_day = alerts_value
            elif abs(alerts_value - best_alerts_per_day) <= 1e-12 and abs(
                float(threshold) - 0.5
            ) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)

    if best_score == float("-inf"):
        best_threshold = float(recall_at_info["threshold"])
        best_score = _metric_value_at_threshold(
            y_arr,
            scores_arr,
            threshold=best_threshold,
            objective=objective_key,
            eval_df=eval_df,
            alerts_per_day=alerts_per_day,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            event_group_ids=event_group_ids,
            event_group_count=event_group_count,
            recall_at_info=recall_at_info,
            frame_days=frame_days,
        )
    metrics = compute_extended_metrics(
        y_arr,
        scores_arr,
        threshold=float(best_threshold),
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        event_group_ids=event_group_ids,
        event_group_count=event_group_count,
        recall_at_info=recall_at_info,
        frame_days=frame_days,
    )
    return {
        "threshold": float(best_threshold),
        "objective": objective_key,
        "objective_score": float(best_score),
        "threshold_n_jobs": int(threshold_n_jobs),
        "far_val": float(metrics.get("far", np.nan)),
        "sens_val": float(metrics.get("sensitivity", np.nan)),
        "alerts_per_day": float(metrics.get("alerts_per_day", np.nan)),
        "operational_cost": float(metrics.get("operational_cost", np.nan)),
        "note": "",
    }


def score_optuna_objective(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective_metric: str = "f1",
    threshold: Optional[float] = None,
    threshold_objective: str = "far",
    eval_df: Optional[pd.DataFrame] = None,
    far_target: float = 0.20,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    threshold_n_jobs: Optional[int] = None,
) -> Dict[str, object]:
    """Score a validation fold for Optuna with the Crash Prediction metric catalog."""
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    metric_key = normalize_optuna_objective_metric(objective_metric)
    direction = optuna_objective_direction(metric_key)
    threshold_objective_key = normalize_threshold_objective(threshold_objective)
    effective_threshold_n_jobs = _coerce_threshold_n_jobs(threshold_n_jobs)

    if threshold is None:
        threshold_info = select_threshold_for_metric(
            y_arr,
            scores_arr,
            objective=threshold_objective_key,
            eval_df=eval_df,
            far_target=float(far_target),
            alerts_per_day=float(alerts_per_day),
            fn_cost=float(fn_cost),
            fp_cost=float(fp_cost),
            n_jobs=int(effective_threshold_n_jobs),
        )
        threshold_value = float(threshold_info["threshold"])
    else:
        threshold_value = float(threshold)
        threshold_info = {
            "threshold": threshold_value,
            "objective": threshold_objective_key,
            "objective_score": float("nan"),
            "threshold_n_jobs": int(effective_threshold_n_jobs),
            "note": "Threshold provisto por el trial.",
        }

    metrics = compute_extended_metrics(
        y_arr,
        scores_arr,
        threshold=threshold_value,
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    preds = (scores_arr >= threshold_value).astype(int)

    if metric_key == "fnr":
        fn = float(metrics.get("false_negatives", 0.0))
        tp = float(metrics.get("true_positives", 0.0))
        score = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    elif metric_key == "far_sens":
        score = float(metrics.get("far", 0.0)) - (
            float(metrics.get("sensitivity", 0.0)) * 1e-3
        )
    elif metric_key == "brier_score":
        score = float(metrics.get("brier_score", float("nan")))
    elif metric_key == "net_balanced_rate":
        tp = float(metrics.get("true_positives", 0.0))
        fp = float(metrics.get("false_positives", 0.0))
        tn = float(metrics.get("true_negatives", 0.0))
        fn = float(metrics.get("false_negatives", 0.0))
        total_pos = tp + fn
        total_neg = tn + fp
        pos_term = (tp - fp) / total_pos if total_pos > 0 else 0.0
        neg_term = (tn - fn) / total_neg if total_neg > 0 else 0.0
        score = float(pos_term + neg_term)
    else:
        metric_lookup = {
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "pr_auc",
            "accuracy": "accuracy",
            "recall": "recall",
            "precision": "precision",
            "balanced_f1": "balanced_f1",
            "mcc": "mcc",
            "recall_at_alerts_per_day": "recall_at_alerts_per_day",
            "operational_cost": "operational_cost",
        }
        score = float(metrics.get(metric_lookup.get(metric_key, "f1"), float("nan")))

    return {
        "score": float(score),
        "objective_metric": metric_key,
        "objective_label": OPTUNA_OBJECTIVE_LABELS.get(metric_key, metric_key.upper()),
        "objective_direction": direction,
        "threshold": float(threshold_value),
        "threshold_objective": threshold_objective_key,
        "threshold_info": threshold_info,
        "metrics": metrics,
        "preds": preds,
    }


class ScoreCalibrator:
    def __init__(self, method: str = "none", model: Optional[object] = None):
        self.method = normalize_calibration_method(method)
        self.model = model

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float)
        if self.model is None or self.method == "none":
            return scores_arr
        if self.method == "sigmoid":
            return self.model.predict_proba(scores_arr.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(scores_arr)
        return scores_arr


def fit_score_calibrator(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    method: str = "none",
) -> ScoreCalibrator:
    method_key = normalize_calibration_method(method)
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    if method_key == "none" or np.unique(y_arr).size < 2:
        return ScoreCalibrator("none", None)
    if method_key == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        model.fit(scores_arr.reshape(-1, 1), y_arr)
        return ScoreCalibrator("sigmoid", model)
    if method_key == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores_arr, y_arr)
        return ScoreCalibrator("isotonic", model)
    return ScoreCalibrator("none", None)


def split_train_val_for_threshold(
    train_df: pd.DataFrame,
    *,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    synthetic_mask = (
        train_df["synthetic"].astype(bool)
        if "synthetic" in train_df.columns
        else pd.Series(False, index=train_df.index)
    )
    real_df = train_df.loc[~synthetic_mask].copy()
    synthetic_df = train_df.loc[synthetic_mask].copy()
    if real_df.empty:
        raise ValueError("No hay datos reales para validacion.")

    try:
        train_real, val_df = temporal_train_test_split(
            real_df, time_col="interval_start", test_size=val_size
        )
        # Sinteticos solo en train.
        train_df_final = pd.concat(
            [train_real, synthetic_df], ignore_index=True
        )
        return train_df_final, val_df
    except ValueError:
        from sklearn.model_selection import train_test_split

        stratify = real_df["target"] if real_df["target"].nunique() > 1 else None
        train_real, val_df = train_test_split(
            real_df,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_df_final = pd.concat(
            [train_real, synthetic_df], ignore_index=True
        )
        return train_df_final, val_df


def _sample_frame_rows(df: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if df.empty or len(df) <= max_rows:
        return df.reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def _build_xai_background_rows(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    synthetic_mask = (
        train_df["synthetic"].astype(bool)
        if "synthetic" in train_df.columns
        else pd.Series(False, index=train_df.index)
    )
    real_train_df = train_df.loc[~synthetic_mask].copy()
    if real_train_df.empty:
        real_train_df = train_df.copy()
    feature_df = real_train_df[feature_cols].fillna(0)
    return _sample_frame_rows(feature_df, max_rows=XAI_BACKGROUND_MAX_ROWS)


def _build_xai_explain_rows(
    test_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    scores_test: np.ndarray,
    preds: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    base = test_df.reset_index(drop=False).rename(
        columns={"index": "source_index"}
    )
    feature_col_set = set(feature_cols)
    work_data: Dict[str, object] = {}
    for col in base.columns:
        if col in feature_col_set:
            work_data[col] = base[col].fillna(0)
        elif col == "target":
            work_data[col] = base[col].astype(int)
        else:
            work_data[col] = base[col]
    work_data["score"] = np.asarray(scores_test, dtype=float)
    work_data["pred"] = np.asarray(preds, dtype=int)
    work_data["threshold"] = np.full(len(base), float(threshold), dtype=float)
    work_data["case_hint"] = np.full(len(base), "", dtype=object)
    work = pd.DataFrame(work_data, index=base.index).copy()

    selected_idx: List[int] = []

    def _append_first(mask: pd.Series, label: str) -> None:
        if not mask.any():
            return
        idx = int(
            work.loc[mask]
            .sort_values("score", ascending=False)
            .index[0]
        )
        if idx in selected_idx:
            return
        selected_idx.append(idx)
        work.loc[idx, "case_hint"] = label

    _append_first(pd.Series(True, index=work.index), "highest_score")
    _append_first((work["target"] == 1) & (work["pred"] == 1), "true_positive")
    _append_first((work["target"] == 0) & (work["pred"] == 1), "false_positive")
    _append_first((work["target"] == 1) & (work["pred"] == 0), "false_negative")

    score_ranked = list(
        work.sort_values("score", ascending=False).index.astype(int)
    )
    for idx in score_ranked:
        if idx in selected_idx:
            continue
        selected_idx.append(int(idx))
        if len(selected_idx) >= XAI_EXPLAIN_MAX_ROWS:
            break

    explain_df = work.loc[selected_idx].copy()
    return explain_df.reset_index(drop=True)


def _build_xai_payload(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scores_test: np.ndarray,
    preds: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    return {
        "background_rows": _build_xai_background_rows(train_df, feature_cols),
        "explain_rows": _build_xai_explain_rows(
            test_df,
            feature_cols=feature_cols,
            scores_test=scores_test,
            preds=preds,
            threshold=threshold,
        ),
        "split_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "xai_limits": {
            "background_max_rows": int(XAI_BACKGROUND_MAX_ROWS),
            "explain_max_rows": int(XAI_EXPLAIN_MAX_ROWS),
        },
    }


def _scale_pos_weight_from_y(y_train: pd.Series) -> float:
    y_arr = pd.Series(y_train).astype(int)
    positives = int((y_arr == 1).sum())
    negatives = int((y_arr == 0).sum())
    if positives <= 0:
        return 1.0
    return max(1.0, float(negatives / positives))


def _resolve_training_model_params(
    model_name: str,
    model_params: Dict[str, object],
    y_train: pd.Series,
    *,
    balance_strategy: str,
) -> Dict[str, object]:
    params = dict(model_params or {})
    balance_key = normalize_balance_strategy(balance_strategy)
    if model_name == "XGBoost":
        raw_scale = params.get("scale_pos_weight", None)
        if raw_scale in {"auto", "Auto", "AUTO"} or (
            raw_scale is None and balance_key == "class_weight"
        ):
            params["scale_pos_weight"] = _scale_pos_weight_from_y(y_train)
        elif raw_scale is not None:
            params["scale_pos_weight"] = float(raw_scale)
    elif model_name == "Neural Network":
        raw_pw = params.get("pos_weight")
        if raw_pw in {"auto", "Auto", "AUTO"} or (
            raw_pw is None and balance_key == "class_weight"
        ):
            params["pos_weight"] = _scale_pos_weight_from_y(y_train)
        elif raw_pw is not None:
            params["pos_weight"] = float(raw_pw)
    elif model_name in {"Random Forest", "SVM"}:
        if balance_key == "class_weight" and params.get("class_weight") is None:
            params["class_weight"] = "balanced"
    return params


def _apply_internal_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    smote_params: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("SMOTE requiere `imbalanced-learn` instalado.") from exc

    y_series = pd.Series(y_train).astype(int)
    counts = y_series.value_counts()
    if counts.empty or int(counts.min()) < 2:
        raise ValueError("SMOTE requiere al menos dos muestras en la clase minoritaria.")
    cfg = dict(smote_params or {})
    max_k = max(1, int(counts.min()) - 1)
    k_neighbors = min(max_k, int(cfg.get("k_neighbors", max(1, min(5, max_k)))))
    sampling_strategy = float(cfg.get("sampling_strategy", 1.0))
    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_series)
    return (
        pd.DataFrame(X_res, columns=X_train.columns),
        pd.Series(y_res, name=y_series.name),
    )


def _fit_protocol_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    random_state: int,
    balance_strategy: str,
    smote_params: Optional[Dict[str, object]] = None,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> object:
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].astype(int)
    params = _resolve_training_model_params(
        model_name,
        model_params,
        y_train,
        balance_strategy=balance_strategy,
    )
    X_fit = X_train
    y_fit = y_train
    if normalize_balance_strategy(balance_strategy) == "smote":
        X_fit, y_fit = _apply_internal_smote(
            X_train,
            y_train,
            random_state=random_state,
            smote_params=smote_params,
        )
    model = build_model(model_name, params, random_state)
    fit_kwargs: Dict[str, object] = {}
    if model_name == "Neural Network" and epoch_callback is not None:
        fit_kwargs["model__epoch_callback"] = epoch_callback
    model.fit(X_fit, y_fit, **fit_kwargs)
    return model


def _temporal_oof_splits(
    df: pd.DataFrame,
    *,
    n_splits: int,
    time_col: str = "interval_start",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if time_col not in df.columns:
        return []
    work = df.reset_index(drop=True).copy()
    work["_split_time"] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=["_split_time"])
    if work.empty:
        return []
    unique_times = np.sort(work["_split_time"].unique())
    if len(unique_times) < max(3, int(n_splits) + 1):
        return []
    blocks = [block for block in np.array_split(unique_times, int(n_splits) + 1) if len(block)]
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(1, len(blocks)):
        train_times = np.concatenate(blocks[:fold_idx])
        val_times = blocks[fold_idx]
        train_idx = work.index[work["_split_time"].isin(train_times)].to_numpy()
        val_idx = work.index[work["_split_time"].isin(val_times)].to_numpy()
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        y_train = df.iloc[train_idx]["target"].astype(int)
        y_val = df.iloc[val_idx]["target"].astype(int)
        if y_train.nunique() < 2 or y_val.nunique() < 2:
            continue
        splits.append((train_idx.astype(int), val_idx.astype(int)))
    return splits


def train_model_with_protocol(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    threshold_protocol: str = "conservative",
    threshold_objective: str = "far",
    calibration_method: str = "none",
    far_target: float = 0.20,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    robust_folds: int = 3,
    balance_strategy: str = "none",
    smote_params: Optional[Dict[str, object]] = None,
    threshold_n_jobs: Optional[int] = None,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    random_state: int,
) -> Dict[str, object]:
    protocol_key = normalize_threshold_protocol(threshold_protocol)
    objective_key = normalize_threshold_objective(threshold_objective)
    calibration_key = normalize_calibration_method(calibration_method)
    balance_key = normalize_balance_strategy(balance_strategy)
    effective_threshold_n_jobs = _coerce_threshold_n_jobs(
        threshold_n_jobs
        if threshold_n_jobs is not None
        else dict(model_params or {}).get("n_jobs", 1)
    )

    if train_df["target"].astype(int).nunique() < 2:
        raise ValueError("Solo existe una clase en train.")
    if test_df["target"].astype(int).nunique() < 2:
        raise ValueError("Solo existe una clase en test.")

    note = ""
    if protocol_key == "robust":
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        splits = _temporal_oof_splits(
            train_val_df,
            n_splits=max(2, int(robust_folds)),
        )
        if not splits:
            note = "Robusto sin folds validos; se uso protocolo conservador."
            protocol_key = "conservative"
        else:
            oof_scores = np.full(len(train_val_df), np.nan, dtype=float)
            oof_mask = np.zeros(len(train_val_df), dtype=bool)
            for fold_number, (fold_train_idx, fold_val_idx) in enumerate(splits):
                fold_model = _fit_protocol_model(
                    train_val_df.iloc[fold_train_idx],
                    feature_cols,
                    model_name,
                    model_params,
                    random_state=int(random_state) + int(fold_number),
                    balance_strategy=balance_key,
                    smote_params=smote_params,
                )
                fold_X_val = train_val_df.iloc[fold_val_idx][feature_cols].fillna(0)
                oof_scores[fold_val_idx] = get_model_scores(fold_model, fold_X_val)
                oof_mask[fold_val_idx] = True

            oof_df = train_val_df.loc[oof_mask].reset_index(drop=True)
            oof_y = oof_df["target"].astype(int).to_numpy()
            raw_oof_scores = oof_scores[oof_mask]
            if oof_y.size == 0 or np.unique(oof_y).size < 2:
                note = "Robusto sin OOF de dos clases; se uso protocolo conservador."
                protocol_key = "conservative"
            else:
                score_orientation = 1.0
                if model_name == "SVM":
                    raw_oof_scores, score_orientation = orient_scores_for_objective(
                        oof_y,
                        raw_oof_scores,
                        objective=objective_key,
                    )
                calibrator = fit_score_calibrator(
                    oof_y,
                    raw_oof_scores,
                    method=calibration_key,
                )
                oof_scores_cal = calibrator.transform(raw_oof_scores)
                thr_info = select_threshold_for_metric(
                    oof_y,
                    oof_scores_cal,
                    objective=objective_key,
                    eval_df=oof_df,
                    far_target=float(far_target),
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                    n_jobs=int(effective_threshold_n_jobs),
                )
                threshold = float(thr_info["threshold"])
                final_model = _fit_protocol_model(
                    train_val_df,
                    feature_cols,
                    model_name,
                    model_params,
                    random_state=int(random_state),
                    balance_strategy=balance_key,
                    smote_params=smote_params,
                    epoch_callback=epoch_callback,
                )
                X_test = test_df[feature_cols].fillna(0)
                y_test = test_df["target"].astype(int).to_numpy()
                raw_test_scores = (
                    get_model_scores(final_model, X_test) * float(score_orientation)
                )
                scores_test = calibrator.transform(raw_test_scores)
                preds_test = (scores_test >= threshold).astype(int)
                validation_metrics = compute_extended_metrics(
                    oof_y,
                    oof_scores_cal,
                    threshold=threshold,
                    eval_df=oof_df,
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                )
                test_metrics = compute_extended_metrics(
                    y_test,
                    scores_test,
                    threshold=threshold,
                    eval_df=test_df,
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                )
                metrics = dict(test_metrics)
                metrics.update(
                    {
                        "threshold": threshold,
                        "far_val": float(validation_metrics.get("far", np.nan)),
                        "sens_val": float(validation_metrics.get("sensitivity", np.nan)),
                        "threshold_protocol": protocol_key,
                        "threshold_objective": objective_key,
                        "calibration_method": calibration_key,
                        "balance_strategy": balance_key,
                        "threshold_objective_score": float(
                            thr_info.get("objective_score", np.nan)
                        ),
                        "threshold_n_jobs": int(effective_threshold_n_jobs),
                    }
                )
                split_info = {
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "test_rows": int(len(test_df)),
                    "oof_rows": int(len(oof_df)),
                    "robust_folds": int(len(splits)),
                }
                return {
                    "metrics": metrics,
                    "validation_metrics": validation_metrics,
                    "threshold_info": thr_info,
                    "confusion_matrix": test_metrics["confusion_matrix"],
                    "training_curves": _extract_training_curves(final_model),
                    "model": final_model,
                    "calibrator": calibrator,
                    "split_info": split_info,
                    "threshold_protocol": protocol_key,
                    "threshold_objective": objective_key,
                    "calibration_method": calibration_key,
                    "balance_strategy": balance_key,
                    "note": note,
                    "xai_payload": _build_xai_payload(
                        train_df=train_val_df,
                        val_df=oof_df,
                        test_df=test_df,
                        feature_cols=feature_cols,
                        scores_test=scores_test,
                        preds=preds_test,
                        threshold=threshold,
                    ),
                }

    model = _fit_protocol_model(
        train_df,
        feature_cols,
        model_name,
        model_params,
        random_state=int(random_state),
        balance_strategy=balance_key,
        smote_params=smote_params,
        epoch_callback=epoch_callback,
    )
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target"].astype(int).to_numpy()
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"].astype(int).to_numpy()
    raw_val_scores = get_model_scores(model, X_val)
    score_orientation = 1.0
    if model_name == "SVM":
        raw_val_scores, score_orientation = orient_scores_for_objective(
            y_val,
            raw_val_scores,
            objective=objective_key,
        )
    calibrator = fit_score_calibrator(
        y_val,
        raw_val_scores,
        method=calibration_key,
    )
    scores_val = calibrator.transform(raw_val_scores)
    thr_info = select_threshold_for_metric(
        y_val,
        scores_val,
        objective=objective_key,
        eval_df=val_df,
        far_target=float(far_target),
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        n_jobs=int(effective_threshold_n_jobs),
    )
    threshold = float(thr_info["threshold"])
    scores_test = calibrator.transform(
        get_model_scores(model, X_test) * float(score_orientation)
    )
    preds_test = (scores_test >= threshold).astype(int)
    validation_metrics = compute_extended_metrics(
        y_val,
        scores_val,
        threshold=threshold,
        eval_df=val_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    test_metrics = compute_extended_metrics(
        y_test,
        scores_test,
        threshold=threshold,
        eval_df=test_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    metrics = dict(test_metrics)
    metrics.update(
        {
            "threshold": threshold,
            "far_val": float(validation_metrics.get("far", np.nan)),
            "sens_val": float(validation_metrics.get("sensitivity", np.nan)),
            "threshold_protocol": protocol_key,
            "threshold_objective": objective_key,
            "calibration_method": calibration_key,
            "balance_strategy": balance_key,
            "threshold_objective_score": float(thr_info.get("objective_score", np.nan)),
            "threshold_n_jobs": int(effective_threshold_n_jobs),
        }
    )
    split_info = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "robust_folds": 0,
    }
    return {
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "threshold_info": thr_info,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "training_curves": _extract_training_curves(model),
        "model": model,
        "calibrator": calibrator,
        "split_info": split_info,
        "threshold_protocol": protocol_key,
        "threshold_objective": objective_key,
        "calibration_method": calibration_key,
        "balance_strategy": balance_key,
        "note": note,
        "xai_payload": _build_xai_payload(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            scores_test=scores_test,
            preds=preds_test,
            threshold=threshold,
        ),
    }


def train_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    test_size: float,
    val_size: float,
    far_target: float,
    random_state: int,
    threshold_protocol: str = "conservative",
    threshold_objective: str = "far",
    calibration_method: str = "none",
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    robust_folds: int = 3,
    balance_strategy: str = "none",
    smote_params: Optional[Dict[str, object]] = None,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    y = df["target"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Solo existe una clase en el target.")
    train_val_df, test_df = temporal_train_test_split(
        df, time_col="interval_start", test_size=test_size
    )
    train_df, val_df = temporal_train_test_split(
        train_val_df, time_col="interval_start", test_size=val_size
    )
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].astype(int)
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(
            "El split temporal dejo una sola clase en train. "
            "Ajuste el rango o el test_size."
        )
    if y_test.nunique() < 2:
        raise ValueError(
            "El split temporal dejo una sola clase en test. "
            "Ajuste el rango o el test_size."
        )

    return train_model_with_protocol(
        train_df,
        val_df,
        test_df,
        feature_cols,
        model_name,
        model_params,
        threshold_protocol=threshold_protocol,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        far_target=float(far_target),
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        robust_folds=int(robust_folds),
        balance_strategy=balance_strategy,
        smote_params=smote_params,
        epoch_callback=epoch_callback,
        random_state=int(random_state),
    )


def train_model_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    val_size: float,
    far_target: float,
    random_state: int,
    threshold_protocol: str = "conservative",
    threshold_objective: str = "far",
    calibration_method: str = "none",
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    robust_folds: int = 3,
    balance_strategy: str = "none",
    smote_params: Optional[Dict[str, object]] = None,
    epoch_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    train_df, val_df = split_train_val_for_threshold(
        train_df, val_size=val_size, random_state=random_state
    )
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].astype(int)
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Solo existe una clase en el train.")
    if y_test.nunique() < 2:
        raise ValueError("Solo existe una clase en el test.")

    return train_model_with_protocol(
        train_df,
        val_df,
        test_df,
        feature_cols,
        model_name,
        model_params,
        threshold_protocol=threshold_protocol,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        far_target=float(far_target),
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        robust_folds=int(robust_folds),
        balance_strategy=balance_strategy,
        smote_params=smote_params,
        epoch_callback=epoch_callback,
        random_state=int(random_state),
    )
