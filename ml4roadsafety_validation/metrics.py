from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _to_numpy(values: object) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def safe_auprc(y_true: object, y_prob: object) -> float | None:
    y = _to_numpy(y_true).astype(int).reshape(-1)
    p = _to_numpy(y_prob).astype(float).reshape(-1)
    if y.size == 0 or y.size != p.size:
        return None
    if np.unique(y).size < 2:
        return float(y.mean())
    return _finite_or_none(average_precision_score(y, p))


def safe_auroc(y_true: object, y_prob: object) -> float | None:
    y = _to_numpy(y_true).astype(int).reshape(-1)
    p = _to_numpy(y_prob).astype(float).reshape(-1)
    if y.size == 0 or y.size != p.size or np.unique(y).size < 2:
        return None
    return _finite_or_none(roc_auc_score(y, p))


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def select_threshold_by_fbeta(
    y_true: object,
    y_prob: object,
    *,
    beta: float = 0.5,
    default: float = 0.5,
) -> float:
    y = _to_numpy(y_true).astype(int).reshape(-1)
    p = _to_numpy(y_prob).astype(float).reshape(-1)
    if y.size == 0 or y.size != p.size or np.unique(y).size < 2:
        return float(default)

    precisions, recalls, thresholds = precision_recall_curve(y, p)
    candidates = list(thresholds.astype(float)) + [float(default)]
    best_threshold = float(default)
    best_score = -1.0
    best_precision = -1.0
    for threshold in candidates:
        pred = p >= threshold
        tp = float(((y == 1) & pred).sum())
        fp = float(((y == 0) & pred).sum())
        fn = float(((y == 1) & ~pred).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        score = _fbeta(precision, recall, beta)
        if (score, precision, threshold) > (best_score, best_precision, best_threshold):
            best_score = score
            best_precision = precision
            best_threshold = float(threshold)
    return best_threshold


def select_threshold_by_top_k(
    y_prob: object,
    *,
    k: int,
    default: float = 0.5,
) -> float:
    p = _to_numpy(y_prob).astype(float).reshape(-1)
    if p.size == 0:
        return float(default)
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return float(default)
    k = max(1, min(int(k), int(finite.size)))
    return float(np.sort(finite)[::-1][k - 1])


def score_diagnostics(y_true: object, y_prob: object) -> dict[str, float | int | None]:
    y = _to_numpy(y_true).astype(int).reshape(-1)
    p = _to_numpy(y_prob).astype(float).reshape(-1)
    if y.size != p.size:
        raise ValueError("y_true e y_prob deben tener el mismo largo.")
    out: dict[str, float | int | None] = {
        "score_n": int(p.size),
        "score_min": None,
        "score_q50": None,
        "score_q90": None,
        "score_q99": None,
        "score_max": None,
        "positive_score_min": None,
        "positive_score_q50": None,
        "positive_score_q90": None,
        "positive_score_max": None,
    }
    if p.size == 0:
        return out
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return out
    q = np.quantile(finite, [0.0, 0.5, 0.9, 0.99, 1.0])
    out.update(
        {
            "score_min": float(q[0]),
            "score_q50": float(q[1]),
            "score_q90": float(q[2]),
            "score_q99": float(q[3]),
            "score_max": float(q[4]),
        }
    )
    positive = p[y == 1]
    positive = positive[np.isfinite(positive)]
    if positive.size:
        pq = np.quantile(positive, [0.0, 0.5, 0.9, 1.0])
        out.update(
            {
                "positive_score_min": float(pq[0]),
                "positive_score_q50": float(pq[1]),
                "positive_score_q90": float(pq[2]),
                "positive_score_max": float(pq[3]),
            }
        )
    return out


def classification_metrics(
    y_true: object,
    y_prob: object,
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    y = _to_numpy(y_true).astype(int).reshape(-1)
    p = _to_numpy(y_prob).astype(float).reshape(-1)
    if y.size != p.size:
        raise ValueError("y_true e y_prob deben tener el mismo largo.")
    pred = p >= float(threshold)

    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = _fbeta(precision, recall, 1.0)
    f05 = _fbeta(precision, recall, 0.5)
    positives = int(y.sum())
    n = int(y.size)
    return {
        "n": n,
        "positives": positives,
        "predicted_positives": int(pred.sum()),
        "prevalence": positives / max(n, 1),
        "auprc": safe_auprc(y, p),
        "auroc": safe_auroc(y, p),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f05": float(f05),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        **score_diagnostics(y, p),
    }


def split_prevalence(data: object, *, node_type: str = "pm") -> dict[str, dict[str, float | int]]:
    node = data[node_type]
    y = node.y.detach().cpu().long()
    out: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "test"):
        mask = getattr(node, f"{split}_mask", None)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool()
        n = int(mask.sum().item())
        positives = int(y[mask].sum().item()) if n else 0
        out[split] = {
            "n": n,
            "positives": positives,
            "prevalence": positives / max(n, 1),
        }
    return out


def flatten_metric_rows(
    model_name: str,
    split_metrics: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, metrics in split_metrics.items():
        row = {"model": model_name, "split": split}
        row.update(dict(metrics))
        rows.append(row)
    return rows
