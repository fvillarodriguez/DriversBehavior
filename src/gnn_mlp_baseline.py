from __future__ import annotations

import math
import re
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
from src.mlp_tabular import MLPNet


BaselineProgress = Callable[[Dict[str, object]], None]


@dataclass(frozen=True)
class _GraphTensors:
    x: torch.Tensor
    y: torch.Tensor
    masks: Dict[str, torch.Tensor]
    graph_hash: str


class _FeatureView:
    baseline: str
    model_name: str
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
    prec_, rec_, thr_ = prec[1:], rec[1:], thr
    if len(thr_) == 0 or len(prec_) == 0:
        return 0.5
    beta2 = float(beta) ** 2
    fbeta = (1.0 + beta2) * prec_ * rec_ / np.clip(beta2 * prec_ + rec_, 1e-12, None)
    if not np.isfinite(fbeta).any():
        return 0.5
    idx = int(np.nanargmax(fbeta))
    return _safe_float(thr_[idx], default=0.5)


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
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for split in splits:
        row = {
            "model": model_name,
            "baseline": baseline,
            "split": split,
            "status": "skipped",
            "reason": reason,
            "graph_hash": graph_hash,
            "samples": 0,
            "positives": 0,
            "positive_rate": 0.0,
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "epochs_run": 0,
            "best_epoch": 0,
            "best_val_auprc": 0.0,
            "train_loss": 0.0,
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
) -> Dict[str, object]:
    metrics = _threshold_metrics(y_true, y_prob, tau=tau)
    row: Dict[str, object] = {
        "model": view.model_name,
        "baseline": view.baseline,
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
) -> list[Dict[str, object]]:
    split_idx = {split: view.split_indices(split, tensors.masks) for split in ("train", "val", "test")}
    sample_counts = {split: int(idx.numel()) for split, idx in split_idx.items()}
    if sample_counts["train"] <= 0:
        return _skip_rows(
            baseline=view.baseline,
            model_name=view.model_name,
            reason="train_mask no selecciona muestras para este baseline.",
            graph_hash=tensors.graph_hash,
        )
    if sample_counts["val"] <= 0:
        return _skip_rows(
            baseline=view.baseline,
            model_name=view.model_name,
            reason="val_mask no selecciona muestras para este baseline.",
            graph_hash=tensors.graph_hash,
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

    val_prob, val_y = _collect_probabilities(
        model,
        view,
        split_idx["val"],
        batch_size=int(batch_size),
        device=device,
    )
    tau = _pick_tau_fbeta(val_y, val_prob, beta=1.0)
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
            )
        )
        rows[-1].update(
            {
                "train_samples": sample_counts["train"],
                "val_samples": sample_counts["val"],
                "test_samples": sample_counts["test"],
            }
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
    save_dir: Optional[Path | str] = Path(RESULTADOS_DIR) / "gnn_mlp_baselines",
    graph_hash: Optional[str] = None,
    progress_callback: Optional[BaselineProgress] = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Train topology-free MLP baselines on the loaded GNN graph tensors."""
    requested = tuple(dict.fromkeys(str(b).strip().lower() for b in baselines if str(b).strip()))
    if not requested:
        raise ValueError("Debe seleccionar al menos un baseline.")

    tensors = _extract_graph_tensors(loaded_obj, graph_hash=graph_hash)
    device_resolved = _resolve_device(device)
    batch_size = max(1, int(batch_size))
    rows: list[Dict[str, object]] = []

    for baseline in requested:
        if baseline == "current":
            view: _FeatureView = _CurrentFeatureView(tensors.x, tensors.y)
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
                )
            )
            continue

        if baseline == "temporal":
            view_temporal, reason = _temporal_view_from_loaded_obj(loaded_obj, tensors)
            if view_temporal is None:
                rows.extend(
                    _skip_rows(
                        baseline="temporal",
                        model_name="MLP temporal",
                        reason=reason or "sequence_index no disponible.",
                        graph_hash=tensors.graph_hash,
                    )
                )
                _emit(
                    progress_callback,
                    event="baseline_skipped",
                    baseline="temporal",
                    model="MLP temporal",
                    reason=reason or "sequence_index no disponible.",
                )
                continue
            rows.extend(
                _train_one_baseline(
                    view_temporal,
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
                )
            )
            continue

        rows.extend(
            _skip_rows(
                baseline=baseline,
                model_name=str(baseline),
                reason=f"Baseline no reconocido: {baseline}",
                graph_hash=tensors.graph_hash,
            )
        )

    df = pd.DataFrame(rows)
    artifact_path: Optional[Path] = None
    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_hash = re.sub(r"[^0-9a-zA-Z_.-]+", "_", tensors.graph_hash or "unknown")
        artifact_path = out_dir / f"mlp_baseline_{ts}_{safe_hash[:16]}.csv"
        df.to_csv(artifact_path, index=False)
        df.attrs["artifact_path"] = str(artifact_path)
    return df
