from __future__ import annotations

import math
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.config import RESULTADOS_DIR, SEED, get_auto_device
from src.gnn_artifacts import gnn_dir
from src.mlp_tabular import MLPNet


BaselineProgress = Callable[[Dict[str, object]], None]


@dataclass(frozen=True)
class _GraphTensors:
    x: torch.Tensor
    y: torch.Tensor
    masks: Dict[str, torch.Tensor]
    graph_hash: str


@dataclass(frozen=True)
class _BaselineSpec:
    key: str
    model_family: str
    feature_view: str
    model_name: str


_BASELINE_SPECS: Dict[str, _BaselineSpec] = {
    "current": _BaselineSpec(
        key="current",
        model_family="mlp",
        feature_view="current",
        model_name="MLP actual",
    ),
    "temporal": _BaselineSpec(
        key="temporal",
        model_family="mlp",
        feature_view="temporal",
        model_name="MLP temporal",
    ),
    "xgboost_current": _BaselineSpec(
        key="xgboost_current",
        model_family="xgboost",
        feature_view="current",
        model_name="XGBoost actual",
    ),
    "xgboost_temporal": _BaselineSpec(
        key="xgboost_temporal",
        model_family="xgboost",
        feature_view="temporal",
        model_name="XGBoost temporal",
    ),
    "svm_current": _BaselineSpec(
        key="svm_current",
        model_family="svm",
        feature_view="current",
        model_name="SVM actual",
    ),
    "svm_temporal": _BaselineSpec(
        key="svm_temporal",
        model_family="svm",
        feature_view="temporal",
        model_name="SVM temporal",
    ),
}


class _FeatureView:
    baseline: str
    model_name: str
    model_family: str
    feature_view: str
    input_dim: int

    def split_indices(self, split: str, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def features(self, indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def labels(self, indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _CurrentFeatureView(_FeatureView):
    baseline = "current"
    model_name = "MLP actual"
    model_family = "mlp"
    feature_view = "current"

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y
        self.input_dim = int(x.shape[1])

    def split_indices(self, split: str, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = masks.get(split)
        if mask is None:
            return torch.empty(0, dtype=torch.long)
        return torch.nonzero(mask, as_tuple=False).view(-1).to(torch.long)

    def features(self, indices: torch.Tensor) -> torch.Tensor:
        return self.x.index_select(0, indices.to(torch.long))

    def labels(self, indices: torch.Tensor) -> torch.Tensor:
        return self.y.index_select(0, indices.to(torch.long))


class _TemporalFeatureView(_FeatureView):
    baseline = "temporal"
    model_name = "MLP temporal"
    model_family = "mlp"
    feature_view = "temporal"

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sequence_rows: torch.Tensor,
        target_rows: torch.Tensor,
    ) -> None:
        if sequence_rows.dim() != 2:
            raise ValueError("sequence_index.sequence_rows debe ser una matriz [n_seq, seq_len].")
        if target_rows.dim() != 1:
            raise ValueError("sequence_index.target_rows debe ser un vector.")
        n_seq = min(int(sequence_rows.shape[0]), int(target_rows.shape[0]))
        if n_seq <= 0:
            raise ValueError("sequence_index no contiene secuencias.")

        sequence_rows = sequence_rows[:n_seq].to(torch.long).contiguous()
        target_rows = target_rows[:n_seq].to(torch.long).contiguous()
        valid_targets = (target_rows >= 0) & (target_rows < int(x.shape[0]))
        valid_sequences = (sequence_rows >= 0).all(dim=1) & (sequence_rows < int(x.shape[0])).all(dim=1)
        self.valid_sequence_mask = valid_targets & valid_sequences
        self.x = x
        self.y = y
        self.sequence_rows = sequence_rows
        self.target_rows = target_rows
        self.input_dim = int(sequence_rows.shape[1]) * int(x.shape[1])

    def split_indices(self, split: str, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = masks.get(split)
        if mask is None:
            return torch.empty(0, dtype=torch.long)
        target_mask = mask.index_select(0, self.target_rows.clamp(0, int(mask.numel()) - 1))
        selected = self.valid_sequence_mask & target_mask
        return torch.nonzero(selected, as_tuple=False).view(-1).to(torch.long)

    def features(self, indices: torch.Tensor) -> torch.Tensor:
        seq = self.sequence_rows.index_select(0, indices.to(torch.long))
        batch_size = int(seq.shape[0])
        flat_rows = seq.reshape(-1)
        x_seq = self.x.index_select(0, flat_rows)
        return x_seq.reshape(batch_size, self.input_dim)

    def labels(self, indices: torch.Tensor) -> torch.Tensor:
        rows = self.target_rows.index_select(0, indices.to(torch.long))
        return self.y.index_select(0, rows)


class _NamedFeatureView(_FeatureView):
    def __init__(self, base: _FeatureView, spec: _BaselineSpec) -> None:
        self._base = base
        self.baseline = spec.key
        self.model_name = spec.model_name
        self.model_family = spec.model_family
        self.feature_view = spec.feature_view
        self.input_dim = int(base.input_dim)

    def split_indices(self, split: str, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self._base.split_indices(split, masks)

    def features(self, indices: torch.Tensor) -> torch.Tensor:
        return self._base.features(indices)

    def labels(self, indices: torch.Tensor) -> torch.Tensor:
        return self._base.labels(indices)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _emit(progress_callback: Optional[BaselineProgress], **payload: object) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(dict(payload))
    except Exception:
        pass


def _as_cpu_tensor(value: object, *, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype).cpu()


def _mask_to_bool(mask: object, *, n_nodes: int, name: str) -> torch.Tensor:
    mask_t = _as_cpu_tensor(mask, dtype=torch.bool).view(-1)
    if int(mask_t.numel()) != int(n_nodes):
        raise ValueError(f"{name} tiene largo {int(mask_t.numel())}, esperado {int(n_nodes)}.")
    return mask_t


def _graph_hash_from_loaded_obj(loaded_obj: Mapping[str, object]) -> str:
    candidates = [loaded_obj.get("graph_hash"), loaded_obj.get("hash")]
    for meta_key in ("metadata", "meta"):
        meta = loaded_obj.get(meta_key)
        if isinstance(meta, Mapping):
            candidates.extend([meta.get("graph_hash"), meta.get("hash")])
    data = loaded_obj.get("data")
    for attr_name in ("graph_metadata", "metadata"):
        meta = getattr(data, attr_name, None)
        if isinstance(meta, Mapping):
            candidates.extend([meta.get("graph_hash"), meta.get("hash")])
    for attr_name in ("graph_hash", "hash"):
        candidates.append(getattr(data, attr_name, None))

    for raw in candidates:
        if raw is None:
            continue
        text = str(raw).strip().lower()
        if text:
            return re.sub(r"[^0-9a-zA-Z_.-]+", "_", text)
    return "unknown"


def _extract_graph_tensors(loaded_obj: Mapping[str, object], graph_hash: Optional[str]) -> _GraphTensors:
    data = loaded_obj.get("data")
    if data is None:
        raise ValueError("loaded_obj no contiene 'data'.")
    node_types = getattr(data, "node_types", [])
    if "pm" not in node_types:
        raise ValueError("El grafo no contiene nodos 'pm'.")
    pm = data["pm"]
    x_raw = getattr(pm, "x", None)
    y_raw = getattr(pm, "y", None)
    if x_raw is None or y_raw is None:
        raise ValueError("data['pm'] debe contener x e y.")

    x = _as_cpu_tensor(x_raw, dtype=torch.float32)
    y = _as_cpu_tensor(y_raw, dtype=torch.long).view(-1)
    if x.dim() != 2:
        raise ValueError("data['pm'].x debe ser una matriz [n_nodes, n_features].")
    if int(y.numel()) != int(x.shape[0]):
        raise ValueError("data['pm'].y no coincide con data['pm'].x.")

    masks: Dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        mask_raw = getattr(pm, f"{split}_mask", None)
        if mask_raw is None:
            masks[split] = torch.zeros(int(x.shape[0]), dtype=torch.bool)
            continue
        masks[split] = _mask_to_bool(mask_raw, n_nodes=int(x.shape[0]), name=f"{split}_mask")

    if int(masks["train"].sum().item()) <= 0:
        raise ValueError("train_mask no tiene muestras.")
    if int(masks["val"].sum().item()) <= 0:
        raise ValueError("val_mask no tiene muestras.")

    return _GraphTensors(
        x=x,
        y=y,
        masks=masks,
        graph_hash=str(graph_hash or _graph_hash_from_loaded_obj(loaded_obj)),
    )


def _sequence_index_from_loaded_obj(loaded_obj: Mapping[str, object]) -> Optional[object]:
    seq = loaded_obj.get("sequence_index")
    if seq is not None:
        return seq
    data = loaded_obj.get("data")
    for attr in ("sequence_index", "snapshot_sequence_index"):
        seq = getattr(data, attr, None)
        if seq is not None:
            return seq
    return None


def _temporal_view_from_loaded_obj(
    loaded_obj: Mapping[str, object],
    tensors: _GraphTensors,
) -> tuple[Optional[_TemporalFeatureView], Optional[str]]:
    seq = _sequence_index_from_loaded_obj(loaded_obj)
    if seq is None:
        return None, "sequence_index no disponible."
    sequence_rows = getattr(seq, "sequence_rows", None)
    target_rows = getattr(seq, "target_rows", None)
    if sequence_rows is None or target_rows is None:
        return None, "sequence_index no contiene sequence_rows y target_rows."
    try:
        view = _TemporalFeatureView(
            x=tensors.x,
            y=tensors.y,
            sequence_rows=_as_cpu_tensor(sequence_rows, dtype=torch.long),
            target_rows=_as_cpu_tensor(target_rows, dtype=torch.long).view(-1),
        )
    except Exception as exc:
        return None, str(exc)
    if int(view.valid_sequence_mask.sum().item()) <= 0:
        return None, "sequence_index no tiene secuencias validas."
    return view, None


def _resolve_device(device: Optional[object]) -> torch.device:
    if device is None:
        device = get_auto_device()
    if isinstance(device, torch.device):
        resolved = device
    else:
        resolved = torch.device(str(device))
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type == "mps":
        try:
            if not torch.backends.mps.is_available():
                return torch.device("cpu")
        except Exception:
            return torch.device("cpu")
    return resolved


def _iter_batches(
    indices: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[torch.Tensor]:
    n = int(indices.numel())
    if n <= 0:
        return
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        order = torch.randperm(n, generator=generator)
        indices = indices.index_select(0, order)
    for start in range(0, n, int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _class_weights(y_train: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y_train.to(torch.long), minlength=2).to(torch.float32)
    total = counts.sum().clamp_min(1.0)
    weights = total / (2.0 * counts.clamp_min(1.0))
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights


def _train_epoch(
    model: torch.nn.Module,
    view: _FeatureView,
    train_idx: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0
    for batch_idx in _iter_batches(train_idx, batch_size=batch_size, shuffle=True, seed=seed):
        xb = view.features(batch_idx).to(device=device, non_blocking=False)
        yb = view.labels(batch_idx).to(device=device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        n = int(yb.numel())
        total_loss += float(loss.detach().item()) * n
        total_seen += n
    return total_loss / max(1, total_seen)


@torch.no_grad()
def _collect_probabilities(
    model: torch.nn.Module,
    view: _FeatureView,
    split_idx: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_idx in _iter_batches(split_idx, batch_size=batch_size, shuffle=False, seed=0):
        xb = view.features(batch_idx).to(device=device, non_blocking=False)
        logits = model(xb)
        prob = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(prob.astype(np.float64, copy=False))
        labels.append(view.labels(batch_idx).detach().cpu().numpy().astype(np.int64, copy=False))
    if not probs:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)
    return np.concatenate(probs), np.concatenate(labels)


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    if np.unique(y_true).size < 2:
        return float(np.mean(y_true))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _safe_float(average_precision_score(y_true, y_prob))


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _safe_float(roc_auc_score(y_true, y_prob), default=0.5)


def _pick_tau_fbeta(y_true: np.ndarray, y_prob: np.ndarray, *, beta: float = 1.0) -> float:
    if y_true.size == 0 or y_prob.size == 0:
        return 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec, rec, thr = precision_recall_curve(y_true.astype(int), y_prob.astype(float))
    prec_, rec_, thr_ = prec[:-1], rec[:-1], thr
    if len(thr_) == 0 or len(prec_) == 0:
        return 0.5
    beta2 = float(beta) ** 2
    fbeta = (1.0 + beta2) * prec_ * rec_ / np.clip(beta2 * prec_ + rec_, 1e-12, None)
    if not np.isfinite(fbeta).any():
        return 0.5
    idx = int(np.nanargmax(fbeta))
    return _safe_float(thr_[idx], default=0.5)


def _normalize_threshold_strategy(value: object) -> str:
    text = str(value or "far").strip().lower()
    aliases = {
        "auto": "far",
        "auto_far": "far",
        "far_target": "far",
        "far <= target": "far",
        "recall@far": "far",
        "f1": "fbeta",
        "f_beta": "fbeta",
        "f-beta": "fbeta",
    }
    text = aliases.get(text, text)
    return text if text in {"far", "fbeta"} else "far"


def _pick_tau_far_target(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    far_target: float = 0.20,
) -> tuple[float, Dict[str, float]]:
    y_true_i = np.asarray(y_true, dtype=int).ravel()
    y_prob_f = np.asarray(y_prob, dtype=float).ravel()
    if y_true_i.size == 0 or y_true_i.size != y_prob_f.size or np.unique(y_true_i).size < 2:
        return 0.5, {"far": float("nan"), "sens": float("nan")}

    thresholds = np.unique(np.clip(y_prob_f, 0.0, 1.0))
    thresholds = np.r_[thresholds, np.nextafter(1.0, 2.0)]
    target = float(np.clip(far_target, 0.0, 1.0))
    best_tau = 0.5
    best_far = float("inf")
    best_sens = -1.0
    best_precision = -1.0

    for tau in thresholds:
        pred = (y_prob_f >= float(tau)).astype(int)
        tp = float(((pred == 1) & (y_true_i == 1)).sum())
        tn = float(((pred == 0) & (y_true_i == 0)).sum())
        fp = float(((pred == 1) & (y_true_i == 0)).sum())
        fn = float(((pred == 0) & (y_true_i == 1)).sum())
        far = fp / max(fp + tn, 1.0)
        if far > target + 1e-12:
            continue
        sens = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        if (
            sens > best_sens + 1e-12
            or (abs(sens - best_sens) <= 1e-12 and precision > best_precision + 1e-12)
            or (
                abs(sens - best_sens) <= 1e-12
                and abs(precision - best_precision) <= 1e-12
                and far > best_far
            )
        ):
            best_tau = float(tau)
            best_far = float(far)
            best_sens = float(sens)
            best_precision = float(precision)

    if best_sens < 0.0:
        return _pick_tau_fbeta(y_true_i, y_prob_f, beta=1.0), {
            "far": float("nan"),
            "sens": float("nan"),
        }
    return best_tau, {
        "far": _safe_float(best_far),
        "sens": _safe_float(best_sens),
        "precision": _safe_float(best_precision),
    }


def _select_tau_from_validation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold_strategy: str = "far",
    far_target: float = 0.20,
    fbeta_beta: float = 1.0,
) -> tuple[float, str, Dict[str, float]]:
    strategy = _normalize_threshold_strategy(threshold_strategy)
    if strategy == "fbeta":
        tau = _pick_tau_fbeta(y_true, y_prob, beta=float(fbeta_beta))
        return tau, f"val_mask_fbeta_beta_{float(fbeta_beta):g}", {"fbeta_beta": float(fbeta_beta)}
    tau, info = _pick_tau_far_target(y_true, y_prob, far_target=float(far_target))
    return tau, "val_mask_far_target", info


def _threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, tau: float) -> Dict[str, float]:
    if y_true.size == 0:
        return {
            "auprc": 0.0,
            "auc": 0.5,
            "f1_at_tau_val": 0.0,
            "f05_at_tau_val": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "far": 0.0,
            "mcc": 0.0,
            "accuracy": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "brier_score": 0.0,
            "tau": _safe_float(tau, default=0.5),
        }

    y_true_i = y_true.astype(int)
    y_pred = (y_prob >= float(tau)).astype(int)
    tp = float(((y_pred == 1) & (y_true_i == 1)).sum())
    tn = float(((y_pred == 0) & (y_true_i == 0)).sum())
    fp = float(((y_pred == 1) & (y_true_i == 0)).sum())
    fn = float(((y_pred == 0) & (y_true_i == 1)).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    far = fp / max(fp + tn, 1.0)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    beta2 = 0.5**2
    f05 = ((1.0 + beta2) * precision * recall) / max(beta2 * precision + recall, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    brier_score = float(np.mean((np.clip(y_prob.astype(float), 0.0, 1.0) - y_true_i) ** 2))

    return {
        "auprc": _safe_auprc(y_true_i, y_prob),
        "auc": _safe_auc(y_true_i, y_prob),
        "f1_at_tau_val": _safe_float(f1),
        "f05_at_tau_val": _safe_float(f05),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "far": _safe_float(far),
        "mcc": _safe_float(mcc),
        "accuracy": _safe_float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "brier_score": _safe_float(brier_score),
        "tau": _safe_float(tau, default=0.5),
    }


def _skip_rows(
    *,
    baseline: str,
    model_name: str,
    reason: str,
    graph_hash: str,
    splits: Sequence[str] = ("val", "test"),
    sample_counts: Optional[Mapping[str, int]] = None,
    model_family: str = "",
    feature_view: str = "",
    backend: str = "",
    fit_seconds: float = 0.0,
    threshold_strategy: str = "far",
    far_target: float = 0.20,
    tau_source: str = "metadata_or_default",
    calibration_method: str = "not_applied",
    calibration_mask_source: Optional[str] = None,
    threshold_mask_source: Optional[str] = None,
    calibration_count: Optional[int] = None,
    threshold_count: Optional[int] = None,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    counts = dict(sample_counts or {})
    for split in splits:
        row = {
            "model": model_name,
            "baseline": baseline,
            "model_family": model_family,
            "feature_view": feature_view,
            "split": split,
            "status": "skipped",
            "reason": reason,
            "graph_hash": graph_hash,
            "samples": 0,
            "positives": 0,
            "positive_rate": 0.0,
            "train_samples": int(counts.get("train", 0)),
            "val_samples": int(counts.get("val", 0)),
            "test_samples": int(counts.get("test", 0)),
            "epochs_run": 0,
            "best_epoch": 0,
            "best_val_auprc": 0.0,
            "train_loss": 0.0,
            "fit_seconds": _safe_float(fit_seconds),
            "backend": backend,
            "tau_source": tau_source,
            "threshold_strategy": _normalize_threshold_strategy(threshold_strategy),
            "far_target": _safe_float(far_target),
            "calibration_method": calibration_method,
            "calibration_mask_source": calibration_mask_source,
            "threshold_mask_source": threshold_mask_source,
            "calibration_count": calibration_count,
            "threshold_count": threshold_count,
        }
        row.update(_threshold_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), tau=0.5))
        rows.append(row)
    return rows


def _metric_row(
    *,
    view: _FeatureView,
    split: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tau: float,
    graph_hash: str,
    sample_counts: Mapping[str, int],
    epochs_run: int,
    best_epoch: int,
    best_val_auprc: float,
    train_loss: float,
    model_family: Optional[str] = None,
    feature_view: Optional[str] = None,
    backend: str = "",
    fit_seconds: float = 0.0,
    threshold_strategy: str = "far",
    far_target: float = 0.20,
    tau_source: str = "val_mask_far_target",
    calibration_method: str = "not_applied",
    calibration_mask_source: Optional[str] = None,
    threshold_mask_source: Optional[str] = "val_mask",
    calibration_count: Optional[int] = None,
    threshold_count: Optional[int] = None,
) -> Dict[str, object]:
    metrics = _threshold_metrics(y_true, y_prob, tau=tau)
    row: Dict[str, object] = {
        "model": view.model_name,
        "baseline": view.baseline,
        "model_family": str(model_family or getattr(view, "model_family", "")),
        "feature_view": str(feature_view or getattr(view, "feature_view", "")),
        "split": split,
        "status": "completed",
        "reason": "",
        "graph_hash": graph_hash,
        "samples": int(y_true.size),
        "positives": int(y_true.sum()) if y_true.size else 0,
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
        "train_samples": int(sample_counts.get("train", 0)),
        "val_samples": int(sample_counts.get("val", 0)),
        "test_samples": int(sample_counts.get("test", 0)),
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "best_val_auprc": _safe_float(best_val_auprc),
        "train_loss": _safe_float(train_loss),
        "fit_seconds": _safe_float(fit_seconds),
        "backend": backend,
        "tau_source": str(tau_source),
        "threshold_strategy": _normalize_threshold_strategy(threshold_strategy),
        "far_target": _safe_float(far_target),
        "calibration_method": str(calibration_method),
        "calibration_mask_source": calibration_mask_source,
        "threshold_mask_source": threshold_mask_source,
        "calibration_count": calibration_count,
        "threshold_count": threshold_count,
    }
    row.update(metrics)
    return row


def _train_one_baseline(
    view: _FeatureView,
    tensors: _GraphTensors,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    progress_callback: Optional[BaselineProgress],
    threshold_strategy: str = "far",
    far_target: float = 0.20,
    fbeta_beta: float = 1.0,
) -> list[Dict[str, object]]:
    split_idx = {split: view.split_indices(split, tensors.masks) for split in ("train", "val", "test")}
    sample_counts = {split: int(idx.numel()) for split, idx in split_idx.items()}
    if sample_counts["train"] <= 0:
        return _skip_rows(
            baseline=view.baseline,
            model_name=view.model_name,
            reason="train_mask no selecciona muestras para este baseline.",
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            model_family="mlp",
            feature_view=str(getattr(view, "feature_view", "")),
            backend="torch",
            threshold_strategy=threshold_strategy,
            far_target=far_target,
        )
    if sample_counts["val"] <= 0:
        return _skip_rows(
            baseline=view.baseline,
            model_name=view.model_name,
            reason="val_mask no selecciona muestras para este baseline.",
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            model_family="mlp",
            feature_view=str(getattr(view, "feature_view", "")),
            backend="torch",
            threshold_strategy=threshold_strategy,
            far_target=far_target,
        )

    _emit(
        progress_callback,
        event="baseline_start",
        baseline=view.baseline,
        model=view.model_name,
        train_samples=sample_counts["train"],
        val_samples=sample_counts["val"],
        test_samples=sample_counts["test"],
    )
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = MLPNet(
        in_dim=int(view.input_dim),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        num_classes=2,
    ).to(device)
    y_train = view.labels(split_idx["train"])
    weights = _class_weights(y_train).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_auprc = float("-inf")
    best_epoch = 0
    epochs_run = 0
    wait = 0
    last_train_loss = 0.0
    max_epochs = max(1, int(epochs))
    early_patience = max(1, int(patience))
    fit_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        last_train_loss = _train_epoch(
            model,
            view,
            split_idx["train"],
            optimizer=optimizer,
            criterion=criterion,
            batch_size=int(batch_size),
            device=device,
            seed=int(seed) + epoch,
        )
        val_prob, val_y = _collect_probabilities(
            model,
            view,
            split_idx["val"],
            batch_size=int(batch_size),
            device=device,
        )
        val_auprc = _safe_auprc(val_y, val_prob)
        epochs_run = epoch
        improved = val_auprc > best_val_auprc + 1e-12
        if improved:
            best_val_auprc = float(val_auprc)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        _emit(
            progress_callback,
            event="epoch",
            baseline=view.baseline,
            model=view.model_name,
            epoch=epoch,
            train_loss=float(last_train_loss),
            val_auprc=float(val_auprc),
            best_val_auprc=float(best_val_auprc),
        )
        if wait >= early_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if not math.isfinite(best_val_auprc):
        best_val_auprc = 0.0
    fit_seconds = time.perf_counter() - fit_start

    val_prob, val_y = _collect_probabilities(
        model,
        view,
        split_idx["val"],
        batch_size=int(batch_size),
        device=device,
    )
    tau, tau_source, _tau_info = _select_tau_from_validation(
        val_y,
        val_prob,
        threshold_strategy=threshold_strategy,
        far_target=far_target,
        fbeta_beta=fbeta_beta,
    )
    threshold_count = int(sample_counts.get("val", 0))
    rows = [
        _metric_row(
            view=view,
            split="val",
            y_true=val_y,
            y_prob=val_prob,
            tau=tau,
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            best_val_auprc=best_val_auprc,
            train_loss=last_train_loss,
            model_family="mlp",
            feature_view=str(getattr(view, "feature_view", "")),
            backend="torch",
            fit_seconds=fit_seconds,
            threshold_strategy=threshold_strategy,
            far_target=far_target,
            tau_source=tau_source,
            calibration_method="not_applied",
            calibration_mask_source=None,
            threshold_mask_source="val_mask",
            calibration_count=None,
            threshold_count=threshold_count,
        )
    ]
    if sample_counts["test"] > 0:
        test_prob, test_y = _collect_probabilities(
            model,
            view,
            split_idx["test"],
            batch_size=int(batch_size),
            device=device,
        )
        rows.append(
            _metric_row(
                view=view,
                split="test",
                y_true=test_y,
                y_prob=test_prob,
                tau=tau,
                graph_hash=tensors.graph_hash,
                sample_counts=sample_counts,
                epochs_run=epochs_run,
                best_epoch=best_epoch,
                best_val_auprc=best_val_auprc,
                train_loss=last_train_loss,
                model_family="mlp",
                feature_view=str(getattr(view, "feature_view", "")),
                backend="torch",
                fit_seconds=fit_seconds,
                threshold_strategy=threshold_strategy,
                far_target=far_target,
                tau_source=tau_source,
                calibration_method="not_applied",
                calibration_mask_source=None,
                threshold_mask_source="val_mask",
                calibration_count=None,
                threshold_count=threshold_count,
            )
        )
    else:
        rows.extend(
            _skip_rows(
                baseline=view.baseline,
                model_name=view.model_name,
                reason="test_mask no selecciona muestras para este baseline.",
                graph_hash=tensors.graph_hash,
                splits=("test",),
                sample_counts=sample_counts,
                model_family="mlp",
                feature_view=str(getattr(view, "feature_view", "")),
                backend="torch",
                fit_seconds=fit_seconds,
                threshold_strategy=threshold_strategy,
                far_target=far_target,
                tau_source=tau_source,
                calibration_method="not_applied",
                calibration_mask_source=None,
                threshold_mask_source="val_mask",
                calibration_count=None,
                threshold_count=threshold_count,
            )
        )

    _emit(
        progress_callback,
        event="baseline_done",
        baseline=view.baseline,
        model=view.model_name,
        best_epoch=best_epoch,
        best_val_auprc=float(best_val_auprc),
        epochs_run=epochs_run,
    )
    return rows


def _view_arrays(view: _FeatureView, indices: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    x = view.features(indices).detach().cpu().numpy().astype(np.float32, copy=False)
    y = view.labels(indices).detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x, y


def _scale_pos_weight(y_train: np.ndarray) -> float:
    y_arr = np.asarray(y_train, dtype=int)
    positives = int((y_arr == 1).sum())
    negatives = int((y_arr == 0).sum())
    if positives <= 0:
        return 1.0
    return max(1.0, float(negatives / positives))


def _build_tabular_estimator(
    spec: _BaselineSpec,
    *,
    y_train: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[object, str]:
    if spec.model_family == "xgboost":
        from src.model_training import _import_external_xgboost

        xgb = _import_external_xgboost()
        estimator = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=int(seed),
            scale_pos_weight=_scale_pos_weight(y_train),
            verbosity=0,
        )
        return estimator, "xgboost"

    if spec.model_family == "svm":
        from src.mlx_svm import MLXAcceleratedSVMClassifier

        estimator = MLXAcceleratedSVMClassifier(
            C=1.0,
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=int(seed),
            epochs=20,
            batch_size=max(1, int(batch_size)),
            require_mlx=False,
        )
        return estimator, "mlx_svm"

    raise ValueError(f"Familia tabular no soportada: {spec.model_family}")


def _positive_probability(estimator: object, x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 0:
        return np.asarray([], dtype=np.float64)
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(x), dtype=float)
        if proba.ndim == 2 and proba.shape[1] > 0:
            classes = getattr(estimator, "classes_", None)
            pos_idx = 1 if proba.shape[1] > 1 else 0
            if classes is not None:
                try:
                    class_list = list(classes)
                    if 1 in class_list:
                        pos_idx = int(class_list.index(1))
                except Exception:
                    pos_idx = 1 if proba.shape[1] > 1 else 0
            out = proba[:, min(pos_idx, proba.shape[1] - 1)]
        else:
            out = np.asarray(proba, dtype=float).reshape(-1)
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x), dtype=float).reshape(-1)
        out = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
    else:
        pred = np.asarray(estimator.predict(x), dtype=float).reshape(-1)
        out = np.clip(pred, 0.0, 1.0)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _collect_tabular_probabilities(
    estimator: object,
    view: _FeatureView,
    split_idx: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = _view_arrays(view, split_idx)
    return _positive_probability(estimator, x).astype(np.float64, copy=False), y


def _train_one_tabular_baseline(
    view: _FeatureView,
    tensors: _GraphTensors,
    spec: _BaselineSpec,
    *,
    batch_size: int,
    seed: int,
    progress_callback: Optional[BaselineProgress],
    threshold_strategy: str = "far",
    far_target: float = 0.20,
    fbeta_beta: float = 1.0,
) -> list[Dict[str, object]]:
    split_idx = {split: view.split_indices(split, tensors.masks) for split in ("train", "val", "test")}
    sample_counts = {split: int(idx.numel()) for split, idx in split_idx.items()}
    if sample_counts["train"] <= 0:
        return _skip_rows(
            baseline=spec.key,
            model_name=spec.model_name,
            reason="train_mask no selecciona muestras para este baseline.",
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            model_family=spec.model_family,
            feature_view=spec.feature_view,
            threshold_strategy=threshold_strategy,
            far_target=far_target,
        )
    if sample_counts["val"] <= 0:
        return _skip_rows(
            baseline=spec.key,
            model_name=spec.model_name,
            reason="val_mask no selecciona muestras para este baseline.",
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            model_family=spec.model_family,
            feature_view=spec.feature_view,
            threshold_strategy=threshold_strategy,
            far_target=far_target,
        )

    x_train, y_train = _view_arrays(view, split_idx["train"])
    if np.unique(y_train).size < 2:
        reason = f"{spec.model_name} requiere dos clases en train_mask."
        _emit(
            progress_callback,
            event="baseline_skipped",
            baseline=spec.key,
            model=spec.model_name,
            reason=reason,
        )
        return _skip_rows(
            baseline=spec.key,
            model_name=spec.model_name,
            reason=reason,
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            model_family=spec.model_family,
            feature_view=spec.feature_view,
            threshold_strategy=threshold_strategy,
            far_target=far_target,
        )

    _emit(
        progress_callback,
        event="baseline_start",
        baseline=spec.key,
        model=spec.model_name,
        train_samples=sample_counts["train"],
        val_samples=sample_counts["val"],
        test_samples=sample_counts["test"],
    )

    backend = spec.model_family
    fit_start = time.perf_counter()
    try:
        estimator, backend = _build_tabular_estimator(
            spec,
            y_train=y_train,
            batch_size=batch_size,
            seed=seed,
        )
        estimator.fit(x_train, y_train)
        backend = str(getattr(estimator, "backend_", backend))
        fit_seconds = time.perf_counter() - fit_start
        val_prob, val_y = _collect_tabular_probabilities(estimator, view, split_idx["val"])
        best_val_auprc = _safe_auprc(val_y, val_prob)
        tau, tau_source, _tau_info = _select_tau_from_validation(
            val_y,
            val_prob,
            threshold_strategy=threshold_strategy,
            far_target=far_target,
            fbeta_beta=fbeta_beta,
        )
    except Exception as exc:
        fit_seconds = time.perf_counter() - fit_start
        reason = f"Error entrenando {spec.model_name}: {exc}"
        _emit(
            progress_callback,
            event="baseline_skipped",
            baseline=spec.key,
            model=spec.model_name,
            reason=reason,
        )
        return _skip_rows(
            baseline=spec.key,
            model_name=spec.model_name,
            reason=reason,
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            model_family=spec.model_family,
            feature_view=spec.feature_view,
            backend=backend,
            fit_seconds=fit_seconds,
            threshold_strategy=threshold_strategy,
            far_target=far_target,
        )

    rows = [
        _metric_row(
            view=view,
            split="val",
            y_true=val_y,
            y_prob=val_prob,
            tau=tau,
            graph_hash=tensors.graph_hash,
            sample_counts=sample_counts,
            epochs_run=1,
            best_epoch=1,
            best_val_auprc=best_val_auprc,
            train_loss=0.0,
            model_family=spec.model_family,
            feature_view=spec.feature_view,
            backend=backend,
            fit_seconds=fit_seconds,
            threshold_strategy=threshold_strategy,
            far_target=far_target,
            tau_source=tau_source,
            calibration_method="not_applied",
            calibration_mask_source=None,
            threshold_mask_source="val_mask",
            calibration_count=None,
            threshold_count=int(sample_counts.get("val", 0)),
        )
    ]
    if sample_counts["test"] > 0:
        test_prob, test_y = _collect_tabular_probabilities(estimator, view, split_idx["test"])
        rows.append(
            _metric_row(
                view=view,
                split="test",
                y_true=test_y,
                y_prob=test_prob,
                tau=tau,
                graph_hash=tensors.graph_hash,
                sample_counts=sample_counts,
                epochs_run=1,
                best_epoch=1,
                best_val_auprc=best_val_auprc,
                train_loss=0.0,
                model_family=spec.model_family,
                feature_view=spec.feature_view,
                backend=backend,
                fit_seconds=fit_seconds,
                threshold_strategy=threshold_strategy,
                far_target=far_target,
                tau_source=tau_source,
                calibration_method="not_applied",
                calibration_mask_source=None,
                threshold_mask_source="val_mask",
                calibration_count=None,
                threshold_count=int(sample_counts.get("val", 0)),
            )
        )
    else:
        rows.extend(
            _skip_rows(
                baseline=spec.key,
                model_name=spec.model_name,
                reason="test_mask no selecciona muestras para este baseline.",
                graph_hash=tensors.graph_hash,
                splits=("test",),
                sample_counts=sample_counts,
                model_family=spec.model_family,
                feature_view=spec.feature_view,
                backend=backend,
                fit_seconds=fit_seconds,
                threshold_strategy=threshold_strategy,
                far_target=far_target,
                tau_source=tau_source,
                calibration_method="not_applied",
                calibration_mask_source=None,
                threshold_mask_source="val_mask",
                calibration_count=None,
                threshold_count=int(sample_counts.get("val", 0)),
            )
        )

    _emit(
        progress_callback,
        event="baseline_done",
        baseline=spec.key,
        model=spec.model_name,
        best_epoch=1,
        best_val_auprc=float(best_val_auprc),
        epochs_run=1,
    )
    return rows


def run_gnn_mlp_baselines(
    loaded_obj: Mapping[str, object],
    baselines: Sequence[str] = ("current", "temporal"),
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 4096,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: Optional[object] = None,
    save_dir: Optional[Path | str] = None,
    graph_hash: Optional[str] = None,
    progress_callback: Optional[BaselineProgress] = None,
    seed: int = SEED,
    threshold_strategy: str = "far",
    far_target: float = 0.20,
    fbeta_beta: float = 1.0,
) -> pd.DataFrame:
    """Train topology-free tabular baselines on the loaded GNN graph tensors."""
    requested = tuple(dict.fromkeys(str(b).strip().lower() for b in baselines if str(b).strip()))
    if not requested:
        raise ValueError("Debe seleccionar al menos un baseline.")

    tensors = _extract_graph_tensors(loaded_obj, graph_hash=graph_hash)
    device_resolved = _resolve_device(device)
    batch_size = max(1, int(batch_size))
    threshold_strategy = _normalize_threshold_strategy(threshold_strategy)
    far_target = float(np.clip(float(far_target), 0.0, 1.0))
    fbeta_beta = max(float(fbeta_beta), 1e-6)
    rows: list[Dict[str, object]] = []
    current_view: _FeatureView = _CurrentFeatureView(tensors.x, tensors.y)
    temporal_view: Optional[_TemporalFeatureView] = None
    temporal_reason: Optional[str] = None

    for baseline in requested:
        spec = _BASELINE_SPECS.get(baseline)
        if spec is None:
            rows.extend(
                _skip_rows(
                    baseline=baseline,
                    model_name=str(baseline),
                    reason=f"Baseline no reconocido: {baseline}",
                    graph_hash=tensors.graph_hash,
                    threshold_strategy=threshold_strategy,
                    far_target=far_target,
                )
            )
            continue

        if spec.feature_view == "current":
            base_view: Optional[_FeatureView] = current_view
        elif spec.feature_view == "temporal":
            if temporal_view is None and temporal_reason is None:
                temporal_view, temporal_reason = _temporal_view_from_loaded_obj(loaded_obj, tensors)
            base_view = temporal_view
            if base_view is None:
                reason = temporal_reason or "sequence_index no disponible."
                rows.extend(
                    _skip_rows(
                        baseline=spec.key,
                        model_name=spec.model_name,
                        reason=reason,
                        graph_hash=tensors.graph_hash,
                        model_family=spec.model_family,
                        feature_view=spec.feature_view,
                        threshold_strategy=threshold_strategy,
                        far_target=far_target,
                    )
                )
                _emit(
                    progress_callback,
                    event="baseline_skipped",
                    baseline=spec.key,
                    model=spec.model_name,
                    reason=reason,
                )
                continue
        else:
            rows.extend(
                _skip_rows(
                    baseline=spec.key,
                    model_name=spec.model_name,
                    reason=f"Vista de features no reconocida: {spec.feature_view}",
                    graph_hash=tensors.graph_hash,
                    model_family=spec.model_family,
                    feature_view=spec.feature_view,
                    threshold_strategy=threshold_strategy,
                    far_target=far_target,
                )
            )
            continue

        if spec.model_family == "mlp":
            view = base_view
            rows.extend(
                _train_one_baseline(
                    view,
                    tensors,
                    epochs=epochs,
                    patience=patience,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device_resolved,
                    seed=seed,
                    progress_callback=progress_callback,
                    threshold_strategy=threshold_strategy,
                    far_target=far_target,
                    fbeta_beta=fbeta_beta,
                )
            )
            continue

        if spec.model_family in {"xgboost", "svm"}:
            view = _NamedFeatureView(base_view, spec)
            rows.extend(
                _train_one_tabular_baseline(
                    view,
                    tensors,
                    spec,
                    batch_size=batch_size,
                    seed=seed,
                    progress_callback=progress_callback,
                    threshold_strategy=threshold_strategy,
                    far_target=far_target,
                    fbeta_beta=fbeta_beta,
                )
            )
            continue

        rows.extend(
            _skip_rows(
                baseline=spec.key,
                model_name=spec.model_name,
                reason=f"Familia de baseline no reconocida: {spec.model_family}",
                graph_hash=tensors.graph_hash,
                model_family=spec.model_family,
                feature_view=spec.feature_view,
                threshold_strategy=threshold_strategy,
                far_target=far_target,
            )
        )

    df = pd.DataFrame(rows)
    artifact_path: Optional[Path] = None
    out_dir = Path(save_dir) if save_dir is not None else gnn_dir("baselines_mlp", RESULTADOS_DIR, create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_hash = re.sub(r"[^0-9a-zA-Z_.-]+", "_", tensors.graph_hash or "unknown")
    artifact_path = out_dir / f"mlp_baseline_{ts}_{safe_hash[:16]}.csv"
    df.to_csv(artifact_path, index=False)
    df.attrs["artifact_path"] = str(artifact_path)
    return df
