from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData

from src.imgagn import ImGAGNGenerator
from src.snapshot_sequences import SequenceConfig, SequenceIndex


PM_NODE = "pm"
TEMPORAL_EDGE: Tuple[str, str, str] = ("pm", "temporal", "pm")
SPATIAL_EDGE: Tuple[str, str, str] = ("pm", "spatial", "pm")
RELATIONAL_MODE = "relational_parent_anchored"


@dataclass
class RelationalImGAGNConfig:
    target_pos_ratio: float = 0.003
    dz: int = 64
    hidden_g: int = 256
    n_hidden_g: int = 1
    hidden_d: int = 64
    dropout: float = 0.25
    epochs: int = 30
    d_steps: int = 1
    lr_g: float = 1e-3
    lr_d: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4096
    temperature: float = 0.3
    parent_topk: int = 0
    entropy_weight: float = 0.0
    spatial_copy_k: int = 3
    seed: int = 19092086
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    quality_eval_every: int = 1
    distribution_eval_sample_size: int = 2048
    categorical_unique_threshold: int = 8
    categorical_integer_tolerance: float = 1e-4
    correlation_max_features: int = 32
    classifier_eval_steps: int = 25
    classifier_eval_lr: float = 0.05
    tabsyndex_weight_ks: float = 0.30
    tabsyndex_weight_chi2: float = 0.15
    tabsyndex_weight_corr: float = 0.20
    tabsyndex_weight_distinguishability: float = 0.25
    tabsyndex_weight_utility: float = 0.10


@dataclass
class RelationalImGAGNResult:
    graph_obj: Dict[str, Any]
    build_meta: Dict[str, Any]
    validation: Dict[str, Any]


class NodeImGAGNDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.head_real = nn.Linear(int(hidden_dim), 1)
        self.head_minor = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return h, self.head_real(h).squeeze(-1), self.head_minor(h).squeeze(-1)


def _as_heterodata(graph_obj: Mapping[str, Any]) -> HeteroData:
    data = graph_obj.get("data")
    if not isinstance(data, HeteroData):
        raise ValueError("graph_obj debe contener data=HeteroData.")
    return data


def _require_relational_inputs(graph_obj: Mapping[str, Any]) -> None:
    data = _as_heterodata(graph_obj)
    if PM_NODE not in data.node_types:
        raise ValueError("El grafo debe contener nodos 'pm'.")
    pm = data[PM_NODE]
    for attr in ("x", "y", "train_mask", "val_mask", "test_mask"):
        if not hasattr(pm, attr):
            raise ValueError(f"El grafo pm no contiene {attr}.")
    if graph_obj.get("sequence_index") is None:
        raise ValueError("ImGAGN relacional requiere sequence_index.")
    if TEMPORAL_EDGE not in data.edge_types:
        raise ValueError("ImGAGN relacional requiere aristas ('pm','temporal','pm').")
    if SPATIAL_EDGE not in data.edge_types:
        raise ValueError("ImGAGN relacional requiere aristas ('pm','spatial','pm').")
    for edge_type in (TEMPORAL_EDGE, SPATIAL_EDGE):
        store = data[edge_type]
        if not hasattr(store, "edge_index"):
            raise ValueError(f"La relación {edge_type} no contiene edge_index.")
        edge_attr = getattr(store, "edge_attr", None)
        if edge_attr is None:
            raise ValueError(f"La relación {edge_type} no contiene edge_attr.")
        if edge_attr.dim() != 2:
            raise ValueError(f"La relación {edge_type} debe tener edge_attr 2D.")


def _coerce_sequence_config(
    config: Any,
    *,
    sequence_length: int,
) -> SequenceConfig:
    if isinstance(config, SequenceConfig):
        return config
    values: Dict[str, Any] = {}
    for name in (
        "sequence_length",
        "guard_band_minutes",
        "horizon_minutes",
        "include_downstream",
    ):
        if isinstance(config, Mapping) and name in config:
            values[name] = config.get(name)
        elif hasattr(config, name):
            values[name] = getattr(config, name)
    values.setdefault("sequence_length", int(sequence_length))
    return SequenceConfig(
        sequence_length=int(values.get("sequence_length") or sequence_length),
        guard_band_minutes=int(values.get("guard_band_minutes", 10)),
        horizon_minutes=int(values.get("horizon_minutes", 20)),
        include_downstream=bool(values.get("include_downstream", True)),
    )


def _coerce_sequence_index(sequence_index: Any) -> SequenceIndex:
    if isinstance(sequence_index, SequenceIndex):
        return sequence_index
    if sequence_index is None:
        raise ValueError("sequence_index missing")

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(sequence_index, Mapping):
            return sequence_index.get(name, default)
        return getattr(sequence_index, name, default)

    sequence_rows = np.asarray(_get("sequence_rows"), dtype=np.int64)
    target_rows = np.asarray(_get("target_rows"), dtype=np.int64).reshape(-1)
    if sequence_rows.ndim != 2:
        raise ValueError("sequence_index.sequence_rows debe ser 2D.")
    if target_rows.ndim != 1 or int(target_rows.shape[0]) != int(sequence_rows.shape[0]):
        raise ValueError("sequence_index.target_rows no está alineado con sequence_rows.")

    n_rows = int(sequence_rows.shape[0])

    def _array_or_default(name: str, dtype: Any, default_value: Any) -> np.ndarray:
        raw = _get(name)
        if raw is None:
            return np.full((n_rows,), default_value, dtype=dtype)
        arr = np.asarray(raw, dtype=dtype).reshape(-1)
        if int(arr.shape[0]) != n_rows:
            return np.full((n_rows,), default_value, dtype=dtype)
        return arr

    config = _coerce_sequence_config(
        _get("config"),
        sequence_length=int(sequence_rows.shape[1]),
    )
    return SequenceIndex(
        sequence_rows=sequence_rows,
        target_rows=target_rows,
        labels=_array_or_default("labels", np.int8, 0),
        porticos=_array_or_default("porticos", object, ""),
        target_ts_min=_array_or_default("target_ts_min", np.int64, 0),
        config=config,
    )


def _parent_weights(
    logits: torch.Tensor,
    *,
    temperature: float,
    parent_topk: int,
) -> torch.Tensor:
    scaled = logits / max(float(temperature), 1e-3)
    if int(parent_topk) > 0 and int(parent_topk) < int(scaled.shape[1]):
        top_values, top_idx = torch.topk(scaled, k=int(parent_topk), dim=1)
        top_weights = torch.softmax(top_values, dim=1)
        weights = torch.zeros_like(scaled)
        weights.scatter_(1, top_idx, top_weights)
        return weights
    return torch.softmax(scaled, dim=1)


def _safe_quantile(values: torch.Tensor, q: float) -> Optional[float]:
    if values.numel() == 0:
        return None
    values = values.detach().cpu().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, float(q)).item())


def _latent_synthetic_quality(z_syn: torch.Tensor, z_pos: torch.Tensor) -> Dict[str, Any]:
    if z_syn.numel() == 0 or z_pos.numel() == 0:
        return {}
    z_syn = F.normalize(z_syn.float(), p=2, dim=1, eps=1e-12)
    z_pos = F.normalize(z_pos.float(), p=2, dim=1, eps=1e-12)
    l2 = torch.cdist(z_syn, z_pos, p=2.0).min(dim=1).values
    cosine = (1.0 - (z_syn @ z_pos.T).max(dim=1).values).clamp_min(0.0)
    out: Dict[str, Any] = {
        "latent_min_l2_to_positive_mean": float(l2.mean().item()),
        "latent_min_l2_to_positive_p95": _safe_quantile(l2, 0.95),
        "latent_min_cosine_to_positive_mean": float(cosine.mean().item()),
        "latent_min_cosine_to_positive_p95": _safe_quantile(cosine, 0.95),
    }
    if z_pos.shape[0] >= 2:
        diag = torch.eye(z_pos.shape[0], dtype=torch.bool)
        pos_l2 = torch.cdist(z_pos, z_pos, p=2.0).masked_fill(diag, float("inf")).min(dim=1).values
        pos_cos = (1.0 - (z_pos @ z_pos.T).masked_fill(diag, float("-inf")).max(dim=1).values).clamp_min(0.0)
        syn_l2_p95 = _safe_quantile(l2, 0.95)
        syn_cos_p95 = _safe_quantile(cosine, 0.95)
        pos_l2_p95 = _safe_quantile(pos_l2, 0.95)
        pos_cos_p95 = _safe_quantile(pos_cos, 0.95)
        out.update(
            {
                "real_positive_loo_l2_p95": pos_l2_p95,
                "real_positive_loo_cosine_p95": pos_cos_p95,
                "synthetic_l2_p95_to_real_positive_loo_p95_ratio": (
                    float(syn_l2_p95 / max(pos_l2_p95, 1e-12))
                    if syn_l2_p95 is not None and pos_l2_p95 is not None
                    else None
                ),
                "synthetic_cosine_p95_to_real_positive_loo_p95_ratio": (
                    float(syn_cos_p95 / max(pos_cos_p95, 1e-12))
                    if syn_cos_p95 is not None and pos_cos_p95 is not None
                    else None
                ),
            }
        )
    return out


def _synthetic_feature_quality(
    data: HeteroData,
    source_data: HeteroData,
    synth_mask: torch.Tensor,
) -> Dict[str, Any]:
    quality: Dict[str, Any] = {}
    if not bool(synth_mask.any()):
        return quality

    x_syn = data[PM_NODE].x.detach().cpu().float()[synth_mask]
    source_pm = source_data[PM_NODE]
    y_source = source_pm.y.detach().cpu()
    train_mask = source_pm.train_mask.detach().cpu().bool()
    pos_train_mask = train_mask & (y_source == 1)
    x_train = source_pm.x.detach().cpu().float()[train_mask]
    x_pos = source_pm.x.detach().cpu().float()[pos_train_mask]
    if x_train.numel() == 0 or x_pos.numel() == 0:
        return quality

    eps = 1e-6
    train_mean = x_train.mean(dim=0)
    train_std = x_train.std(dim=0, unbiased=False).clamp_min(eps)
    pos_mean = x_pos.mean(dim=0)
    pos_std = x_pos.std(dim=0, unbiased=False).clamp_min(eps)
    z_syn_train = (x_syn - train_mean) / train_std
    z_pos_train = (x_pos - train_mean) / train_std

    min_train = x_train.min(dim=0).values
    max_train = x_train.max(dim=0).values
    min_pos = x_pos.min(dim=0).values
    max_pos = x_pos.max(dim=0).values
    outside_train = (x_syn < min_train) | (x_syn > max_train)
    outside_pos = (x_syn < min_pos) | (x_syn > max_pos)

    l2_to_pos = torch.cdist(z_syn_train, z_pos_train, p=2.0).min(dim=1).values
    syn_norm = F.normalize(z_syn_train, p=2, dim=1, eps=eps)
    pos_norm = F.normalize(z_pos_train, p=2, dim=1, eps=eps)
    cosine_dist = (1.0 - (syn_norm @ pos_norm.T).max(dim=1).values).clamp_min(0.0)

    loo_l2_p95 = None
    loo_cosine_p95 = None
    l2_p95_ratio = None
    cosine_p95_ratio = None
    if x_pos.shape[0] >= 2:
        pos_pair_l2 = torch.cdist(z_pos_train, z_pos_train, p=2.0)
        diag = torch.eye(pos_pair_l2.shape[0], dtype=torch.bool)
        loo_l2 = pos_pair_l2.masked_fill(diag, float("inf")).min(dim=1).values
        pos_pair_cos = (pos_norm @ pos_norm.T).masked_fill(diag, float("-inf"))
        loo_cosine = (1.0 - pos_pair_cos.max(dim=1).values).clamp_min(0.0)
        loo_l2_p95 = _safe_quantile(loo_l2, 0.95)
        loo_cosine_p95 = _safe_quantile(loo_cosine, 0.95)
        syn_l2_p95 = _safe_quantile(l2_to_pos, 0.95)
        syn_cosine_p95 = _safe_quantile(cosine_dist, 0.95)
        if loo_l2_p95 is not None and syn_l2_p95 is not None:
            l2_p95_ratio = float(syn_l2_p95 / max(loo_l2_p95, eps))
        if loo_cosine_p95 is not None and syn_cosine_p95 is not None:
            cosine_p95_ratio = float(syn_cosine_p95 / max(loo_cosine_p95, eps))

    mean_gap = torch.abs((x_syn.mean(dim=0) - pos_mean) / train_std)
    syn_std = x_syn.std(dim=0, unbiased=False).clamp_min(eps)
    std_ratio = torch.log(syn_std / pos_std).abs()
    nan_count = int(torch.isnan(x_syn).sum().item())
    inf_count = int(torch.isinf(x_syn).sum().item())
    outside_train_frac = float(outside_train.float().mean().item())
    outside_pos_frac = float(outside_pos.float().mean().item())

    quality.update(
        {
            "synthetic_count": int(x_syn.shape[0]),
            "train_positive_count": int(x_pos.shape[0]),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "feature_outside_train_minmax_frac": outside_train_frac,
            "feature_outside_train_positive_minmax_frac": outside_pos_frac,
            "feature_mean_abs_z_gap_to_train_positive": float(mean_gap.mean().item()),
            "feature_p95_abs_z_gap_to_train_positive": _safe_quantile(mean_gap, 0.95),
            "feature_mean_abs_log_std_ratio_to_train_positive": float(std_ratio.mean().item()),
            "min_l2_to_train_positive_mean": float(l2_to_pos.mean().item()),
            "min_l2_to_train_positive_median": _safe_quantile(l2_to_pos, 0.50),
            "min_l2_to_train_positive_p95": _safe_quantile(l2_to_pos, 0.95),
            "min_cosine_distance_to_train_positive_mean": float(cosine_dist.mean().item()),
            "min_cosine_distance_to_train_positive_p95": _safe_quantile(cosine_dist, 0.95),
            "real_positive_loo_l2_p95": loo_l2_p95,
            "real_positive_loo_cosine_p95": loo_cosine_p95,
            "synthetic_l2_p95_to_real_positive_loo_p95_ratio": l2_p95_ratio,
            "synthetic_cosine_p95_to_real_positive_loo_p95_ratio": cosine_p95_ratio,
            "feature_manifold_gate_ok": bool(nan_count == 0 and inf_count == 0),
            "near_duplicate_l2_lt_1e_3_frac": float((l2_to_pos < 1e-3).float().mean().item()),
            "extreme_abs_z_gt_4_frac": float((z_syn_train.abs() > 4.0).float().mean().item()),
        }
    )
    return quality


def _synthetic_edge_quality(
    data: HeteroData,
    source_data: HeteroData,
    synth_mask: torch.Tensor,
) -> Dict[str, Any]:
    quality: Dict[str, Any] = {}
    if not bool(synth_mask.any()):
        return quality
    synth_idx = torch.nonzero(synth_mask, as_tuple=True)[0]
    n_synth = max(1, int(synth_idx.numel()))

    for edge_type in source_data.edge_types:
        if edge_type not in data.edge_types:
            continue
        store = data[edge_type]
        source_store = source_data[edge_type]
        edge_index = store.edge_index.detach().cpu()
        incident = synth_mask[edge_index[0]] | synth_mask[edge_index[1]]
        incident_count = int(incident.sum().item())
        rel_payload: Dict[str, Any] = {
            "synthetic_incident_edges": incident_count,
            "synthetic_incident_edges_per_synthetic": float(incident_count / n_synth),
        }
        edge_attr = getattr(store, "edge_attr", None)
        source_edge_attr = getattr(source_store, "edge_attr", None)
        if edge_attr is not None and source_edge_attr is not None and incident_count > 0:
            ea_syn = edge_attr.detach().cpu().float()[incident]
            ea_src = source_edge_attr.detach().cpu().float()
            if ea_syn.dim() == 1:
                ea_syn = ea_syn.unsqueeze(1)
            if ea_src.dim() == 1:
                ea_src = ea_src.unsqueeze(1)
            eps = 1e-6
            src_std = ea_src.std(dim=0, unbiased=False).clamp_min(eps)
            src_min = ea_src.min(dim=0).values
            src_max = ea_src.max(dim=0).values
            rel_payload.update(
                {
                    "edge_attr_nan_count": int(torch.isnan(ea_syn).sum().item()),
                    "edge_attr_inf_count": int(torch.isinf(ea_syn).sum().item()),
                    "edge_attr_outside_source_minmax_frac": float(((ea_syn < src_min) | (ea_syn > src_max)).float().mean().item()),
                    "edge_attr_mean_abs_z_gap_to_source": float(torch.abs((ea_syn.mean(dim=0) - ea_src.mean(dim=0)) / src_std).mean().item()),
                    "edge_attr_p95_abs_z_gap_to_source": _safe_quantile(torch.abs((ea_syn.mean(dim=0) - ea_src.mean(dim=0)) / src_std), 0.95),
                }
            )
        quality[str(edge_type)] = rel_payload
    return quality


def _to_numpy_2d(values: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = values.detach().cpu().float().numpy() if torch.is_tensor(values) else np.asarray(values)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_rows_np(values: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    if values.shape[0] <= int(max_rows):
        return values
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    idx = rng.choice(values.shape[0], size=int(max_rows), replace=False)
    return values[np.sort(idx)]


def _infer_tabsyndex_schema(
    real_positive_x: torch.Tensor,
    cfg: RelationalImGAGNConfig,
) -> Dict[str, List[int]]:
    arr = _to_numpy_2d(real_positive_x)
    categorical: List[int] = []
    continuous: List[int] = []
    tol = max(float(cfg.categorical_integer_tolerance), 0.0)
    max_categories = max(2, int(cfg.categorical_unique_threshold))
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continuous.append(col_idx)
            continue
        rounded = np.rint(finite)
        is_integer_like = bool(np.max(np.abs(finite - rounded)) <= tol)
        unique_count = int(np.unique(rounded if is_integer_like else finite).size)
        if is_integer_like and unique_count <= max_categories:
            categorical.append(col_idx)
        else:
            continuous.append(col_idx)
    return {"continuous": continuous, "categorical": categorical}


def _ks_statistic_np(real: np.ndarray, synthetic: np.ndarray) -> Optional[float]:
    real = np.sort(real[np.isfinite(real)])
    synthetic = np.sort(synthetic[np.isfinite(synthetic)])
    if real.size == 0 or synthetic.size == 0:
        return None
    values = np.sort(np.concatenate([real, synthetic]))
    cdf_real = np.searchsorted(real, values, side="right") / float(real.size)
    cdf_syn = np.searchsorted(synthetic, values, side="right") / float(synthetic.size)
    return float(np.max(np.abs(cdf_real - cdf_syn)))


def _categorical_chi2_np(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    tolerance: float,
) -> Tuple[Optional[float], Optional[float]]:
    real = real[np.isfinite(real)]
    synthetic = synthetic[np.isfinite(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return None, None
    categories = np.unique(np.rint(real))
    if categories.size == 0:
        return None, None

    def _nearest_counts(values: np.ndarray) -> Tuple[np.ndarray, float]:
        nearest_idx = np.abs(values.reshape(-1, 1) - categories.reshape(1, -1)).argmin(axis=1)
        nearest = categories[nearest_idx]
        frac = float(np.mean(np.abs(values - nearest) > tolerance)) if values.size else 0.0
        counts = np.bincount(nearest_idx, minlength=categories.size).astype(np.float64)
        return counts, frac

    real_counts, _ = _nearest_counts(real)
    syn_counts, fractional_rate = _nearest_counts(synthetic)
    eps = 1e-9
    p = (real_counts + eps) / max(float(real_counts.sum() + eps * categories.size), eps)
    q = (syn_counts + eps) / max(float(syn_counts.sum() + eps * categories.size), eps)
    chi2 = float(np.sum((p - q) ** 2 / np.maximum(p, eps)) / max(categories.size, 1))
    normalized = float(chi2 / (1.0 + chi2))
    return normalized, fractional_rate


def _correlation_diff_np(
    real: np.ndarray,
    synthetic: np.ndarray,
    cols: List[int],
    *,
    max_features: int,
) -> Optional[float]:
    if len(cols) < 2 or real.shape[0] < 3 or synthetic.shape[0] < 3:
        return None
    if len(cols) > int(max_features):
        selected = np.linspace(0, len(cols) - 1, int(max_features), dtype=int)
        cols = [cols[int(i)] for i in selected]
    real_sub = real[:, cols]
    syn_sub = synthetic[:, cols]
    real_corr = np.nan_to_num(np.corrcoef(real_sub, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    syn_corr = np.nan_to_num(np.corrcoef(syn_sub, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    tri = np.triu_indices(len(cols), k=1)
    if tri[0].size == 0:
        return None
    return float(np.mean(np.abs(real_corr[tri] - syn_corr[tri])) / 2.0)


def _binary_auc_np(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    sorted_scores = s[order]
    ranks = np.empty_like(s, dtype=np.float64)
    i = 0
    n = int(s.size)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[y == 1].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(np.clip(auc, 0.0, 1.0))


def _train_linear_auc_np(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> Dict[str, Optional[float]]:
    train_x = _to_numpy_2d(train_x)
    eval_x = _to_numpy_2d(eval_x)
    train_y = np.asarray(train_y, dtype=np.float32).ravel()
    eval_y = np.asarray(eval_y, dtype=np.float32).ravel()
    if train_x.shape[0] < 4 or eval_x.shape[0] < 2:
        return {"auc": None, "accuracy": None}
    if np.unique(train_y).size < 2 or np.unique(eval_y).size < 2:
        return {"auc": None, "accuracy": None}
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_tr = torch.from_numpy(((train_x - mean) / std).astype(np.float32))
    y_tr = torch.from_numpy(train_y.astype(np.float32))
    x_ev = torch.from_numpy(((eval_x - mean) / std).astype(np.float32))
    with torch.enable_grad():
        torch.manual_seed(int(seed))
        model = nn.Linear(int(x_tr.shape[1]), 1)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(max(1, int(steps))):
            logits = model(x_tr).squeeze(-1)
            loss = loss_fn(logits, y_tr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_ev).squeeze(-1)).detach().cpu().numpy()
    auc = _binary_auc_np(eval_y.astype(np.int64), probs)
    acc = float(((probs >= 0.5).astype(np.int64) == eval_y.astype(np.int64)).mean())
    return {"auc": auc, "accuracy": acc}


def _stratified_binary_split_np(
    labels: np.ndarray,
    *,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            continue
        idx = rng.permutation(idx)
        n_test = min(max(1, int(round(float(test_fraction) * idx.size))), max(idx.size - 1, 1))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:] if idx.size > 1 else idx[:0])
    if not train_parts or not test_parts:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty
    train_idx = np.concatenate(train_parts).astype(np.int64)
    test_idx = np.concatenate(test_parts).astype(np.int64)
    return rng.permutation(train_idx), rng.permutation(test_idx)


def _detectability_metrics_np(
    real_positive: np.ndarray,
    synthetic: np.ndarray,
    cfg: RelationalImGAGNConfig,
    *,
    seed: int,
) -> Dict[str, Optional[float]]:
    if int(cfg.classifier_eval_steps) <= 0:
        return {"detect_auc": None, "detect_accuracy": None, "detectability_penalty": None}
    n = min(real_positive.shape[0], synthetic.shape[0], int(cfg.distribution_eval_sample_size))
    if n < 3:
        return {"detect_auc": None, "detect_accuracy": None, "detectability_penalty": None}
    real = _sample_rows_np(real_positive, max_rows=n, seed=seed)
    syn = _sample_rows_np(synthetic, max_rows=n, seed=seed + 17)
    n = min(real.shape[0], syn.shape[0])
    x = np.vstack([real[:n], syn[:n]])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    train_idx, test_idx = _stratified_binary_split_np(y, test_fraction=0.3, seed=seed + 31)
    if train_idx.size < 4 or test_idx.size < 2:
        return {"detect_auc": None, "detect_accuracy": None, "detectability_penalty": None}
    metrics = _train_linear_auc_np(
        x[train_idx],
        y[train_idx],
        x[test_idx],
        y[test_idx],
        steps=int(cfg.classifier_eval_steps),
        lr=float(cfg.classifier_eval_lr),
        seed=seed + 43,
    )
    auc = metrics.get("auc")
    penalty = float(min(1.0, abs(float(auc) - 0.5) * 2.0)) if auc is not None else None
    return {
        "detect_auc": auc,
        "detect_accuracy": metrics.get("accuracy"),
        "detectability_penalty": penalty,
    }


def _utility_metrics_np(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    synthetic: np.ndarray,
    cfg: RelationalImGAGNConfig,
    *,
    seed: int,
) -> Dict[str, Optional[float]]:
    if int(cfg.classifier_eval_steps) <= 0:
        return {"utility_auc_real": None, "utility_auc_aug": None, "utility_auc_gain": None, "utility_penalty": None}
    train_y = np.asarray(train_y, dtype=np.int64).ravel()
    val_y = np.asarray(val_y, dtype=np.int64).ravel()
    if synthetic.shape[0] == 0 or np.unique(train_y).size < 2 or np.unique(val_y).size < 2:
        return {"utility_auc_real": None, "utility_auc_aug": None, "utility_auc_gain": None, "utility_penalty": None}
    max_rows = max(32, int(cfg.distribution_eval_sample_size))
    train_arr = _to_numpy_2d(train_x)
    if train_arr.shape[0] > max_rows:
        rng = np.random.default_rng(int(seed) % (2**32 - 1))
        idx = np.sort(rng.choice(train_arr.shape[0], size=max_rows, replace=False))
        train_x_s = train_arr[idx]
        train_y_s = train_y[idx]
    else:
        train_x_s = train_arr
        train_y_s = train_y
    syn_s = _sample_rows_np(synthetic, max_rows=min(max_rows, synthetic.shape[0]), seed=seed + 59)
    real_metrics = _train_linear_auc_np(
        train_x_s,
        train_y_s,
        val_x,
        val_y,
        steps=int(cfg.classifier_eval_steps),
        lr=float(cfg.classifier_eval_lr),
        seed=seed + 61,
    )
    aug_x = np.vstack([train_x_s, syn_s])
    aug_y = np.concatenate([train_y_s, np.ones(syn_s.shape[0], dtype=np.int64)])
    aug_metrics = _train_linear_auc_np(
        aug_x,
        aug_y,
        val_x,
        val_y,
        steps=int(cfg.classifier_eval_steps),
        lr=float(cfg.classifier_eval_lr),
        seed=seed + 67,
    )
    real_auc = real_metrics.get("auc")
    aug_auc = aug_metrics.get("auc")
    if real_auc is None or aug_auc is None:
        return {"utility_auc_real": real_auc, "utility_auc_aug": aug_auc, "utility_auc_gain": None, "utility_penalty": None}
    gain = float(aug_auc - real_auc)
    return {
        "utility_auc_real": float(real_auc),
        "utility_auc_aug": float(aug_auc),
        "utility_auc_gain": gain,
        "utility_penalty": float(min(1.0, max(0.0, -gain))),
    }


def _tabsyndex_metrics(
    *,
    real_positive_x: torch.Tensor,
    synthetic_x: torch.Tensor,
    schema: Mapping[str, List[int]],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    cfg: RelationalImGAGNConfig,
    seed: int,
) -> Dict[str, Any]:
    real_pos = _sample_rows_np(
        _to_numpy_2d(real_positive_x),
        max_rows=max(32, int(cfg.distribution_eval_sample_size)),
        seed=seed,
    )
    synthetic = _sample_rows_np(
        _to_numpy_2d(synthetic_x),
        max_rows=max(32, int(cfg.distribution_eval_sample_size)),
        seed=seed + 11,
    )
    continuous = list(schema.get("continuous", []))
    categorical = list(schema.get("categorical", []))
    ks_values = [
        value
        for value in (
            _ks_statistic_np(real_pos[:, col], synthetic[:, col])
            for col in continuous
        )
        if value is not None and math.isfinite(float(value))
    ]
    chi2_values: List[float] = []
    fractional_values: List[float] = []
    for col in categorical:
        chi2, frac = _categorical_chi2_np(
            real_pos[:, col],
            synthetic[:, col],
            tolerance=max(float(cfg.categorical_integer_tolerance), 0.0),
        )
        if chi2 is not None and math.isfinite(float(chi2)):
            chi2_values.append(float(chi2))
        if frac is not None and math.isfinite(float(frac)):
            fractional_values.append(float(frac))
    corr_diff = _correlation_diff_np(
        real_pos,
        synthetic,
        continuous,
        max_features=max(2, int(cfg.correlation_max_features)),
    )
    detect = _detectability_metrics_np(real_pos, synthetic, cfg, seed=seed + 101)
    utility = _utility_metrics_np(
        _to_numpy_2d(train_x),
        train_y.detach().cpu().numpy(),
        _to_numpy_2d(val_x),
        val_y.detach().cpu().numpy(),
        synthetic,
        cfg,
        seed=seed + 151,
    )

    components: List[Tuple[float, float]] = []
    ks_mean = float(np.mean(ks_values)) if ks_values else None
    if ks_mean is not None:
        components.append((max(float(cfg.tabsyndex_weight_ks), 0.0), float(np.clip(ks_mean, 0.0, 1.0))))
    chi2_mean = float(np.mean(chi2_values)) if chi2_values else None
    if chi2_mean is not None:
        components.append((max(float(cfg.tabsyndex_weight_chi2), 0.0), float(np.clip(chi2_mean, 0.0, 1.0))))
    if corr_diff is not None:
        components.append((max(float(cfg.tabsyndex_weight_corr), 0.0), float(np.clip(corr_diff, 0.0, 1.0))))
    detect_penalty = detect.get("detectability_penalty")
    if detect_penalty is not None:
        components.append((max(float(cfg.tabsyndex_weight_distinguishability), 0.0), float(np.clip(detect_penalty, 0.0, 1.0))))
    utility_penalty = utility.get("utility_penalty")
    if utility_penalty is not None:
        components.append((max(float(cfg.tabsyndex_weight_utility), 0.0), float(np.clip(utility_penalty, 0.0, 1.0))))
    weight_sum = sum(weight for weight, _ in components if weight > 0.0)
    tabsyndex_loss = (
        float(sum(weight * penalty for weight, penalty in components if weight > 0.0) / weight_sum)
        if weight_sum > 0.0
        else 1.0
    )
    tabsyndex_score = float(np.clip(1.0 - tabsyndex_loss, 0.0, 1.0))
    return {
        "tabsyndex_score": tabsyndex_score,
        "tabsyndex_loss": float(np.clip(tabsyndex_loss, 0.0, 1.0)),
        "tabsyndex_component_weight_sum": float(weight_sum),
        "ks_mean": ks_mean,
        "ks_p95": float(np.quantile(ks_values, 0.95)) if ks_values else None,
        "chi2_mean": chi2_mean,
        "categorical_fractional_rate_mean": float(np.mean(fractional_values)) if fractional_values else None,
        "corr_abs_diff_mean": corr_diff,
        "continuous_feature_count": int(len(continuous)),
        "categorical_feature_count": int(len(categorical)),
        **detect,
        **utility,
    }


def _make_node_imgagn(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_global_idx: torch.Tensor,
    cfg: RelationalImGAGNConfig,
    device: torch.device,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    checkpoint_path: Optional[str | Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    n_train = int(y_train.numel())
    pos_local_idx = torch.nonzero(y_train == 1, as_tuple=True)[0]
    pos = int(pos_local_idx.numel())
    target_pos = int(math.ceil(float(cfg.target_pos_ratio) * float(n_train)))
    n_samples = max(0, target_pos - pos)
    if n_samples <= 0 or pos < 2:
        return (
            torch.empty(0, x_train.shape[1], dtype=x_train.dtype),
            torch.empty(0, dtype=y_train.dtype),
            {
                "synthetic_count": 0,
                "train_count": n_train,
                "train_positive_count": pos,
                "target_pos_ratio": float(cfg.target_pos_ratio),
                "reason": "target_ratio_already_met" if n_samples <= 0 else "not_enough_train_positives",
            },
            {"dominant_parent_global": []},
            [],
        )

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed) % (2**32 - 1))
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(int(cfg.seed))
    x_device = x_train.float().to(device)
    y_device = y_train.long().to(device)
    pos_local_device = pos_local_idx.to(device)
    pos_x = x_device.index_select(0, pos_local_device)
    pos_global = train_global_idx.index_select(0, pos_local_idx).long()
    batch_size = max(32, int(cfg.batch_size))

    neg_local_idx = torch.nonzero(y_train != 1, as_tuple=True)[0]
    pos_perm = pos_local_idx[torch.randperm(pos, generator=gen_cpu)]
    neg_perm = (
        neg_local_idx[torch.randperm(int(neg_local_idx.numel()), generator=gen_cpu)]
        if int(neg_local_idx.numel()) > 0
        else neg_local_idx
    )
    val_pos_count = min(max(1, int(math.ceil(0.2 * pos))), max(pos - 1, 1))
    val_neg_count = min(
        int(neg_local_idx.numel()),
        max(int(val_pos_count * 4), min(64, int(neg_local_idx.numel()))),
    )
    val_real_idx = torch.cat([pos_perm[:val_pos_count], neg_perm[:val_neg_count]]).long()
    if int(val_real_idx.numel()) == 0:
        val_real_idx = torch.arange(min(n_train, batch_size), dtype=torch.long)
    val_fit_mask = torch.ones(n_train, dtype=torch.bool)
    val_fit_mask[val_real_idx] = False
    fit_real_pool = torch.nonzero(val_fit_mask, as_tuple=True)[0].long()
    if int(fit_real_pool.numel()) == 0:
        fit_real_pool = torch.arange(n_train, dtype=torch.long)
    val_real_idx_device = val_real_idx.to(device)
    n_val_syn = max(1, min(n_samples, max(32, int(val_real_idx.numel()))))
    z_val_fixed = torch.randn(n_val_syn, int(cfg.dz), generator=gen_cpu).to(device)
    quality_eval_every = max(1, int(cfg.quality_eval_every))
    tabsyndex_schema = _infer_tabsyndex_schema(pos_x.detach().cpu(), cfg)
    fit_x_cpu = x_train.index_select(0, fit_real_pool).detach().cpu().float()
    fit_y_cpu = y_train.index_select(0, fit_real_pool).detach().cpu().long()
    val_x_cpu = x_train.index_select(0, val_real_idx).detach().cpu().float()
    val_y_cpu = y_train.index_select(0, val_real_idx).detach().cpu().long()

    generator = ImGAGNGenerator(
        dz=int(cfg.dz),
        n_min_train=pos,
        hidden=int(cfg.hidden_g),
        n_hidden=int(cfg.n_hidden_g),
    ).to(device)
    discriminator = NodeImGAGNDiscriminator(
        in_dim=int(x_device.shape[1]),
        hidden_dim=int(cfg.hidden_d),
        dropout=float(cfg.dropout),
    ).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=float(cfg.lr_g), weight_decay=float(cfg.weight_decay))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=float(cfg.lr_d), weight_decay=float(cfg.weight_decay))
    bce = nn.BCEWithLogitsLoss()
    history: List[Dict[str, Any]] = []
    patience = max(0, int(cfg.early_stopping_patience))
    min_delta = max(0.0, float(cfg.early_stopping_min_delta))
    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path is not None else None
    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    best_generator_state: Optional[Dict[str, Any]] = None
    best_discriminator_state: Optional[Dict[str, Any]] = None
    best_history_row: Optional[Dict[str, Any]] = None
    last_quality_metrics: Optional[Dict[str, Any]] = None
    epochs_without_improvement = 0
    early_stopped = False
    stopped_epoch: Optional[int] = None

    def _cpu_state_dict(module: nn.Module) -> Dict[str, Any]:
        return {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in module.state_dict().items()
        }

    def _write_best_checkpoint(row: Mapping[str, Any]) -> None:
        if checkpoint_path_obj is None:
            return
        checkpoint_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": int(best_epoch or row.get("epoch") or 0),
                "best_val_loss": float(best_val_loss),
                "best_tabsyndex_loss": float(best_val_loss),
                "best_tabsyndex_score": row.get("tabsyndex_score"),
                "checkpoint_metric": "tabsyndex_loss",
                "checkpoint_source": "internal_train_holdout_distribution",
                "generator_state_dict": best_generator_state,
                "discriminator_state_dict": best_discriminator_state,
                "config": asdict(cfg),
                "history_row": dict(row),
                "train_count": int(n_train),
                "train_positive_count": int(pos),
                "validation_holdout_count": int(val_real_idx.numel()),
                "validation_holdout_positive_count": int((y_train.index_select(0, val_real_idx) == 1).sum().item()),
            },
            checkpoint_path_obj,
        )

    def _evaluate_internal_val_loss(epoch: int) -> Dict[str, Any]:
        nonlocal last_quality_metrics
        discriminator.eval()
        generator.eval()
        with torch.no_grad():
            val_weights = _parent_weights(
                generator(z_val_fixed),
                temperature=float(cfg.temperature),
                parent_topk=int(cfg.parent_topk),
            )
            val_syn = val_weights @ pos_x
            val_real = x_device.index_select(0, val_real_idx_device)
            val_y_real = y_device.index_select(0, val_real_idx_device).float()
            val_batch = torch.cat([val_real, val_syn], dim=0)
            val_fake_target = torch.cat(
                [
                    torch.zeros(int(val_real.shape[0]), device=device),
                    torch.ones(int(val_syn.shape[0]), device=device),
                ],
                dim=0,
            )
            val_minor_target = torch.cat(
                [
                    val_y_real,
                    torch.ones(int(val_syn.shape[0]), device=device),
                ],
                dim=0,
            )
            val_h, val_logit_fake, val_logit_minor = discriminator(val_batch)
            val_loss_d = bce(val_logit_fake, val_fake_target) + bce(val_logit_minor, val_minor_target)
            val_h_syn = val_h[int(val_real.shape[0]) :]
            h_pos_val, _, _ = discriminator(pos_x)
            val_centroid = val_weights @ h_pos_val
            val_loss_g = (
                bce(val_logit_fake[int(val_real.shape[0]) :], torch.zeros(int(val_syn.shape[0]), device=device))
                + bce(val_logit_minor[int(val_real.shape[0]) :], torch.ones(int(val_syn.shape[0]), device=device))
                + F.mse_loss(val_h_syn, val_centroid)
            )
            if float(cfg.entropy_weight) > 0.0:
                val_entropy = -(val_weights * val_weights.clamp_min(1e-12).log()).sum(dim=1).mean()
                val_loss_g = val_loss_g + float(cfg.entropy_weight) * val_entropy
            adversarial_val_loss = float(val_loss_g.detach().item())
            run_quality_eval = (
                last_quality_metrics is None
                or int(epoch) % quality_eval_every == 0
                or int(epoch) >= int(cfg.epochs)
            )
            if run_quality_eval:
                last_quality_metrics = _tabsyndex_metrics(
                    real_positive_x=pos_x.detach().cpu(),
                    synthetic_x=val_syn.detach().cpu(),
                    schema=tabsyndex_schema,
                    train_x=fit_x_cpu,
                    train_y=fit_y_cpu,
                    val_x=val_x_cpu,
                    val_y=val_y_cpu,
                    cfg=cfg,
                    seed=int(cfg.seed) + int(epoch) * 997,
                )
            quality_metrics = dict(last_quality_metrics or {})
            quality_metrics["quality_eval_ran"] = bool(run_quality_eval)
            tabsyndex_loss = quality_metrics.get("tabsyndex_loss")
            monitor_loss = (
                float(tabsyndex_loss)
                if tabsyndex_loss is not None and math.isfinite(float(tabsyndex_loss))
                else adversarial_val_loss
            )
            return {
                "val_loss": float(monitor_loss),
                "adversarial_val_loss": adversarial_val_loss,
                "val_loss_g": float(val_loss_g.detach().item()),
                "val_loss_d": float(val_loss_d.detach().item()),
                "val_fake_prob_mean": float(torch.sigmoid(val_logit_fake[int(val_real.shape[0]) :]).mean().item()),
                "val_minor_prob_mean": float(torch.sigmoid(val_logit_minor[int(val_real.shape[0]) :]).mean().item()),
                **quality_metrics,
            }

    for epoch in range(1, int(cfg.epochs) + 1):
        generator.train()
        z = torch.randn(n_samples, int(cfg.dz), device=device)
        with torch.no_grad():
            weights = _parent_weights(
                generator(z),
                temperature=float(cfg.temperature),
                parent_topk=int(cfg.parent_topk),
            )
            x_syn = weights @ pos_x

        discriminator.train()
        d_loss_sum = 0.0
        with torch.enable_grad():
            for _ in range(max(1, int(cfg.d_steps))):
                real_take = min(int(fit_real_pool.numel()), max(batch_size - n_samples, batch_size // 2))
                real_perm = torch.randperm(int(fit_real_pool.numel()), generator=gen_cpu)[:real_take]
                real_idx = fit_real_pool.index_select(0, real_perm).to(device)
                if n_samples > batch_size // 2:
                    syn_idx = torch.randperm(n_samples, generator=gen_cpu)[: batch_size // 2].to(device)
                    syn_batch = x_syn.index_select(0, syn_idx)
                else:
                    syn_batch = x_syn
                x_batch = torch.cat([x_device.index_select(0, real_idx), syn_batch.detach()], dim=0)
                y_fake = torch.cat(
                    [
                        torch.zeros(int(real_idx.numel()), device=device),
                        torch.ones(int(syn_batch.shape[0]), device=device),
                    ],
                    dim=0,
                )
                y_minor = torch.cat(
                    [
                        y_device.index_select(0, real_idx).float(),
                        torch.ones(int(syn_batch.shape[0]), device=device),
                    ],
                    dim=0,
                )
                h, logit_fake, logit_minor = discriminator(x_batch)
                loss_fake = bce(logit_fake, y_fake)
                loss_minor = bce(logit_minor, y_minor)
                real_h = h[: int(real_idx.numel())]
                real_y = y_minor[: int(real_idx.numel())]
                if (real_y == 1).any() and (real_y == 0).any():
                    dist = F.pairwise_distance(
                        real_h[real_y == 1].mean(dim=0, keepdim=True),
                        real_h[real_y == 0].mean(dim=0, keepdim=True),
                        p=2,
                    ).mean()
                    loss_sep = F.relu(torch.tensor(1.0, device=device) - dist)
                else:
                    loss_sep = torch.tensor(0.0, device=device)
                loss_d = loss_fake + loss_minor + loss_sep
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                d_loss_sum += float(loss_d.detach().item())

        discriminator.eval()
        generator.train()
        with torch.enable_grad():
            z = torch.randn(n_samples, int(cfg.dz), device=device)
            weights = _parent_weights(
                generator(z),
                temperature=float(cfg.temperature),
                parent_topk=int(cfg.parent_topk),
            )
            x_syn_g = weights @ pos_x
            with torch.no_grad():
                h_pos, _, _ = discriminator(pos_x)
            h_syn, logit_fake_syn, logit_minor_syn = discriminator(x_syn_g)
            centroid = weights @ h_pos
            loss_g = (
                bce(logit_fake_syn, torch.zeros(n_samples, device=device))
                + bce(logit_minor_syn, torch.ones(n_samples, device=device))
                + F.mse_loss(h_syn, centroid)
            )
            if float(cfg.entropy_weight) > 0.0:
                entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean()
                loss_g = loss_g + float(cfg.entropy_weight) * entropy
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
        with torch.no_grad():
            entropy_value = float((-(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean()).item())
            effective_parents = float(torch.exp(torch.tensor(entropy_value)).item())
        row = {
            "epoch": int(epoch),
            "loss_d": float(d_loss_sum / max(1, int(cfg.d_steps))),
            "loss_g": float(loss_g.detach().item()),
            "fake_prob_mean": float(torch.sigmoid(logit_fake_syn.detach()).mean().item()),
            "minor_prob_mean": float(torch.sigmoid(logit_minor_syn.detach()).mean().item()),
            "parent_entropy": entropy_value,
            "effective_parents": effective_parents,
        }
        val_metrics = _evaluate_internal_val_loss(epoch)
        row.update(val_metrics)
        monitor_loss = row.get("val_loss")
        is_best = False
        monitor_active = bool(row.get("quality_eval_ran", True))
        if monitor_active and monitor_loss is not None and math.isfinite(float(monitor_loss)):
            if float(monitor_loss) < (best_val_loss - min_delta):
                best_val_loss = float(monitor_loss)
                best_epoch = int(epoch)
                best_generator_state = _cpu_state_dict(generator)
                best_discriminator_state = _cpu_state_dict(discriminator)
                best_history_row = dict(row)
                epochs_without_improvement = 0
                is_best = True
            else:
                epochs_without_improvement += 1
        row["checkpoint_best"] = bool(is_best)
        row["best_epoch"] = int(best_epoch) if best_epoch is not None else None
        row["best_val_loss"] = float(best_val_loss) if math.isfinite(best_val_loss) else None
        row["early_stop_counter"] = int(epochs_without_improvement)
        row["early_stopping_patience"] = int(patience)
        stop_now = bool(monitor_active and patience > 0 and not is_best and epochs_without_improvement >= patience)
        row["early_stopped"] = stop_now
        if is_best:
            best_history_row = dict(row)
            _write_best_checkpoint(row)
        history.append(row)
        if progress_callback is not None:
            progress_callback(int(epoch), int(cfg.epochs), dict(row))
        if stop_now:
            early_stopped = True
            stopped_epoch = int(epoch)
            break

    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
    if best_discriminator_state is not None:
        discriminator.load_state_dict(best_discriminator_state)

    with torch.no_grad():
        z = torch.randn(n_samples, int(cfg.dz), device=device)
        weights = _parent_weights(
            generator(z),
            temperature=float(cfg.temperature),
            parent_topk=int(cfg.parent_topk),
        )
        syn_x = weights @ pos_x
        dominant_parent_local = weights.argmax(dim=1).detach().cpu().long()
        parent_entropy = (-(weights * weights.clamp_min(1e-12).log()).sum(dim=1)).detach().cpu()
        max_parent_weight = weights.max(dim=1).values.detach().cpu()

    syn_y = torch.ones(n_samples, dtype=y_train.dtype)
    quality = _latent_synthetic_quality(syn_x.detach().cpu(), pos_x.detach().cpu())
    quality.update(
        {
            "synthetic_count": n_samples,
            "train_count": n_train,
            "train_positive_count": pos,
            "target_pos_ratio": float(cfg.target_pos_ratio),
            "actual_augmented_pos_ratio": float((pos + n_samples) / max(n_train + n_samples, 1)),
            "temperature": float(cfg.temperature),
            "parent_topk": int(cfg.parent_topk),
            "entropy_weight": float(cfg.entropy_weight),
            "parent_entropy_mean": float(parent_entropy.mean().item()),
            "effective_parents_mean": float(torch.exp(parent_entropy).mean().item()),
            "max_parent_weight_mean": float(max_parent_weight.mean().item()),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
            "best_tabsyndex_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
            "best_tabsyndex_score": (
                best_history_row.get("tabsyndex_score")
                if isinstance(best_history_row, Mapping)
                else None
            ),
            "checkpoint_metric": "tabsyndex_loss",
            "checkpoint_source": "internal_train_holdout_distribution",
            "checkpoint_path": (
                str(checkpoint_path_obj)
                if checkpoint_path_obj is not None and checkpoint_path_obj.exists()
                else None
            ),
            "early_stopping_patience": int(patience),
            "early_stopping_min_delta": float(min_delta),
            "early_stopped": bool(early_stopped),
            "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else (int(history[-1]["epoch"]) if history else None),
            "validation_holdout_count": int(val_real_idx.numel()),
            "validation_holdout_positive_count": int((y_train.index_select(0, val_real_idx) == 1).sum().item()),
            "tabsyndex_schema": {
                "continuous_feature_count": int(len(tabsyndex_schema.get("continuous", []))),
                "categorical_feature_count": int(len(tabsyndex_schema.get("categorical", []))),
            },
        }
    )
    parent_global = pos_global.index_select(0, dominant_parent_local).long()
    parent_info = {
        "dominant_parent_global": parent_global.tolist(),
        "dominant_parent_positive_local": dominant_parent_local.tolist(),
        "parent_entropy": parent_entropy.tolist(),
        "max_parent_weight": max_parent_weight.tolist(),
        "checkpoint": {
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
            "best_tabsyndex_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
            "best_tabsyndex_score": (
                best_history_row.get("tabsyndex_score")
                if isinstance(best_history_row, Mapping)
                else None
            ),
            "path": str(checkpoint_path_obj) if checkpoint_path_obj is not None else None,
            "metric": "tabsyndex_loss",
            "source": "internal_train_holdout_distribution",
            "history_row": best_history_row,
        },
    }
    return syn_x.detach().cpu().to(dtype=x_train.dtype), syn_y, quality, parent_info, history


def extend_sequence_index_with_synthetics(
    *,
    sequence_index: SequenceIndex,
    parent_nodes: List[int],
    synthetic_nodes: List[int],
) -> Tuple[SequenceIndex, Dict[str, Any], Dict[int, np.ndarray]]:
    target_rows = np.asarray(sequence_index.target_rows, dtype=np.int64)
    sequence_rows = np.asarray(sequence_index.sequence_rows, dtype=np.int64)
    target_to_pos = {int(row): idx for idx, row in enumerate(target_rows.tolist())}
    new_sequences: List[np.ndarray] = []
    new_targets: List[int] = []
    new_labels: List[int] = []
    new_porticos: List[Any] = []
    new_ts: List[int] = []
    synth_to_sequence: Dict[int, np.ndarray] = {}
    skipped = 0

    for parent, synthetic in zip(parent_nodes, synthetic_nodes):
        pos = target_to_pos.get(int(parent))
        if pos is None:
            skipped += 1
            continue
        seq = sequence_rows[pos].copy()
        if seq.size > 0:
            seq[-1] = int(synthetic)
        new_sequences.append(seq.astype(np.int64))
        new_targets.append(int(synthetic))
        new_labels.append(1)
        new_porticos.append(np.asarray(sequence_index.porticos)[pos])
        new_ts.append(int(np.asarray(sequence_index.target_ts_min)[pos]))
        synth_to_sequence[int(synthetic)] = seq.astype(np.int64)

    if new_sequences:
        out_sequence_rows = np.vstack([sequence_rows, np.vstack(new_sequences).astype(np.int64)])
        out_target_rows = np.concatenate([target_rows, np.asarray(new_targets, dtype=np.int64)])
        out_labels = np.concatenate([np.asarray(sequence_index.labels, dtype=np.int8), np.ones(len(new_sequences), dtype=np.int8)])
        out_porticos = np.concatenate([np.asarray(sequence_index.porticos), np.asarray(new_porticos)])
        out_ts = np.concatenate([np.asarray(sequence_index.target_ts_min, dtype=np.int64), np.asarray(new_ts, dtype=np.int64)])
    else:
        out_sequence_rows = sequence_rows
        out_target_rows = target_rows
        out_labels = np.asarray(sequence_index.labels, dtype=np.int8)
        out_porticos = np.asarray(sequence_index.porticos)
        out_ts = np.asarray(sequence_index.target_ts_min, dtype=np.int64)

    return (
        SequenceIndex(
            sequence_rows=out_sequence_rows,
            target_rows=out_target_rows,
            labels=out_labels,
            porticos=out_porticos,
            target_ts_min=out_ts,
            config=sequence_index.config,
        ),
        {
            "added_sequences": int(len(new_sequences)),
            "skipped_without_parent_sequence": int(skipped),
            "sequence_length": int(sequence_rows.shape[1]) if sequence_rows.ndim == 2 else None,
        },
        synth_to_sequence,
    )


def _fit_edge_attr_dim(value: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
    value = value.detach().cpu().flatten().float()
    if value.numel() == int(dim):
        return value.to(dtype=dtype)
    if value.numel() > int(dim):
        return value[: int(dim)].to(dtype=dtype)
    pad = torch.zeros(int(dim) - int(value.numel()), dtype=value.dtype)
    return torch.cat([value, pad], dim=0).to(dtype=dtype)


def _append_relation_edges(
    data: HeteroData,
    *,
    edge_type: Tuple[str, str, str],
    edges: List[Tuple[int, int]],
    attrs: List[torch.Tensor],
) -> Dict[str, Any]:
    if not edges:
        return {"added_edges": 0}
    store = data[edge_type]
    old_edge_index = store.edge_index.detach().cpu().long()
    new_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    store.edge_index = torch.cat([old_edge_index, new_edge_index], dim=1)
    edge_attr = getattr(store, "edge_attr", None)
    if edge_attr is not None:
        old_attr = edge_attr.detach().cpu()
        edge_dim = int(old_attr.shape[1]) if old_attr.dim() > 1 else 1
        new_attr = torch.stack(
            [_fit_edge_attr_dim(attr, edge_dim, old_attr.dtype) for attr in attrs],
            dim=0,
        )
        store.edge_attr = torch.cat([old_attr, new_attr], dim=0)
    return {"added_edges": int(len(edges))}


def build_relational_imgagn_graph(
    graph_obj: Mapping[str, Any],
    cfg: Optional[RelationalImGAGNConfig] = None,
    *,
    device: Optional[torch.device | str] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    validation_artifact_path: Optional[str | Path] = None,
    checkpoint_artifact_path: Optional[str | Path] = None,
) -> RelationalImGAGNResult:
    cfg = cfg or RelationalImGAGNConfig()
    _require_relational_inputs(graph_obj)
    source_data = _as_heterodata(graph_obj)
    source_obj = dict(graph_obj)
    source_obj["data"] = source_data

    obj: Dict[str, Any] = copy.deepcopy(dict(graph_obj))
    obj["data"] = copy.deepcopy(source_data).cpu()
    data = obj["data"]
    pm = data[PM_NODE]
    y = pm.y.detach().cpu().long()
    train_mask = pm.train_mask.detach().cpu().bool()
    train_idx = torch.nonzero(train_mask, as_tuple=True)[0].long()
    if train_idx.numel() == 0:
        raise ValueError("train_mask no contiene nodos.")
    device_obj = torch.device(str(device)) if device is not None else torch.device("cpu")
    checkpoint_path = checkpoint_artifact_path
    if checkpoint_path is None and validation_artifact_path is not None:
        validation_path = Path(validation_artifact_path)
        stem = validation_path.stem
        if stem.endswith("_validation"):
            stem = stem[: -len("_validation")]
        checkpoint_path = validation_path.with_name(f"{stem}_best_checkpoint.pt")

    syn_x, syn_y, synth_quality, parent_info, history = _make_node_imgagn(
        x_train=pm.x.detach().cpu().index_select(0, train_idx),
        y_train=y.index_select(0, train_idx),
        train_global_idx=train_idx,
        cfg=cfg,
        device=device_obj,
        progress_callback=progress_callback,
        checkpoint_path=checkpoint_path,
    )
    n_old = int(pm.x.shape[0])
    n_syn = int(syn_x.shape[0])
    if n_syn <= 0:
        raise ValueError(
            "ImGAGN relacional no produjo nodos sintéticos. "
            f"Detalle: {synth_quality.get('reason', 'sin detalle')}."
        )
    synthetic_nodes = list(range(n_old, n_old + n_syn))
    parents = [int(v) for v in parent_info.get("dominant_parent_global", [])]
    if len(parents) != n_syn:
        raise ValueError("La traza parent-anchored no coincide con los nodos sintéticos.")

    pm.x = torch.cat([pm.x.detach().cpu(), syn_x], dim=0)
    pm.y = torch.cat([pm.y.detach().cpu(), syn_y.to(dtype=pm.y.dtype)], dim=0)
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        old_mask = getattr(pm, mask_name).detach().cpu().bool()
        add = torch.ones(n_syn, dtype=torch.bool) if mask_name == "train_mask" else torch.zeros(n_syn, dtype=torch.bool)
        setattr(pm, mask_name, torch.cat([old_mask, add], dim=0))
    old_synth = (
        pm.is_synthetic.detach().cpu().bool()
        if hasattr(pm, "is_synthetic")
        else torch.zeros(n_old, dtype=torch.bool)
    )
    pm.is_synthetic = torch.cat([old_synth, torch.ones(n_syn, dtype=torch.bool)], dim=0)
    pm.num_nodes = int(pm.x.shape[0])

    try:
        seq_index = _coerce_sequence_index(obj.get("sequence_index"))
    except Exception as exc:
        raise ValueError(
            "sequence_index debe contener sequence_rows y target_rows compatibles."
        ) from exc
    obj["sequence_index"] = seq_index
    new_seq, seq_meta, synth_to_sequence = extend_sequence_index_with_synthetics(
        sequence_index=seq_index,
        parent_nodes=parents,
        synthetic_nodes=synthetic_nodes,
    )
    obj["sequence_index"] = new_seq

    temporal_edges: List[Tuple[int, int]] = []
    temporal_attrs: List[torch.Tensor] = []
    temporal_attr = data[TEMPORAL_EDGE].edge_attr.detach().cpu()
    temporal_dim = int(temporal_attr.shape[1])
    for synth_node, seq in synth_to_sequence.items():
        if len(seq) < 2:
            continue
        pred = int(seq[-2])
        if pred < 0 or pred >= n_old:
            continue
        if not bool(train_mask[pred]):
            continue
        temporal_edges.append((pred, int(synth_node)))
        delta = (pm.x[int(synth_node)].detach().cpu() - pm.x[pred].detach().cpu()).float()
        temporal_attrs.append(_fit_edge_attr_dim(delta, temporal_dim, temporal_attr.dtype))
    temporal_meta = _append_relation_edges(
        data,
        edge_type=TEMPORAL_EDGE,
        edges=temporal_edges,
        attrs=temporal_attrs,
    )

    spatial_meta = {"added_edges": 0}
    store = data[SPATIAL_EDGE]
    ei = store.edge_index.detach().cpu().long()
    ea = store.edge_attr.detach().cpu()
    spatial_edges: List[Tuple[int, int]] = []
    spatial_attrs: List[torch.Tensor] = []
    k = max(0, int(cfg.spatial_copy_k))
    for synth_node, parent in zip(synthetic_nodes, parents):
        outgoing = torch.where((ei[0] == int(parent)) & train_mask.index_select(0, ei[1]).bool())[0]
        incoming = torch.where((ei[1] == int(parent)) & train_mask.index_select(0, ei[0]).bool())[0]
        if k:
            outgoing = outgoing[:k]
            incoming = incoming[:k]
        for col in outgoing.tolist():
            dst = int(ei[1, col].item())
            spatial_edges.append((int(synth_node), dst))
            spatial_attrs.append(ea[col])
        for col in incoming.tolist():
            src = int(ei[0, col].item())
            spatial_edges.append((src, int(synth_node)))
            spatial_attrs.append(ea[col])
    spatial_meta = _append_relation_edges(
        data,
        edge_type=SPATIAL_EDGE,
        edges=spatial_edges,
        attrs=spatial_attrs,
    )

    params = asdict(cfg)
    params.update(
        {
            "mode": RELATIONAL_MODE,
            "synthetic_count": int(n_syn),
            "parents_unique": int(len(set(parents))),
            "checkpoint_path": synth_quality.get("checkpoint_path"),
            "best_epoch": synth_quality.get("best_epoch"),
            "best_val_loss": synth_quality.get("best_val_loss"),
            "best_tabsyndex_loss": synth_quality.get("best_tabsyndex_loss"),
            "best_tabsyndex_score": synth_quality.get("best_tabsyndex_score"),
            "checkpoint_metric": synth_quality.get("checkpoint_metric"),
        }
    )
    obj["imgagn_best_params"] = params
    obj["filename"] = _filename_with_imgagn_tag(str(obj.get("filename") or "graph.pt"))

    build_meta: Dict[str, Any] = {
        "synthetic_count": int(n_syn),
        "parents_unique": int(len(set(parents))),
        "imgagn_feature_quality": synth_quality,
        "imgagn_history": history,
        "imgagn_history_tail": history[-5:],
        "sequence_index": seq_meta,
        "temporal_relation": temporal_meta,
        "spatial_relation": spatial_meta,
        "parent_trace_sample": {
            "synthetic_nodes": synthetic_nodes[:10],
            "parents": parents[:10],
            "max_parent_weight": parent_info.get("max_parent_weight", [])[:10],
        },
    }
    obj["imgagn_relational_build"] = build_meta

    validation = validate_relational_imgagn_graph(
        graph_obj=obj,
        source_obj=source_obj,
        artifact_path=validation_artifact_path,
    )
    obj["relational_validation"] = validation
    if not bool(validation.get("ok")):
        raise ValueError(
            "Validación ImGAGN relacional falló: "
            + "; ".join(str(e) for e in validation.get("errors", []))
        )
    return RelationalImGAGNResult(graph_obj=obj, build_meta=build_meta, validation=validation)


def validate_relational_imgagn_graph(
    *,
    graph_obj: Mapping[str, Any],
    source_obj: Mapping[str, Any],
    artifact_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    data = _as_heterodata(graph_obj)
    source = _as_heterodata(source_obj)
    payload: Dict[str, Any] = {
        "ok": False,
        "errors": [],
        "edges": {},
        "synthetic_masks": {},
        "sequence_index": {},
    }
    errors: List[str] = payload["errors"]

    for required in (TEMPORAL_EDGE, SPATIAL_EDGE):
        if required not in data.edge_types:
            errors.append(f"{required} missing")
    if any(isinstance(et, tuple) and len(et) >= 2 and et[1] == "imgagn" for et in data.edge_types):
        errors.append("Relación homogénea imgagn no permitida")

    for edge_type in source.edge_types:
        if edge_type not in data.edge_types:
            errors.append(f"{edge_type} missing")
            continue
        edge_attr = getattr(data[edge_type], "edge_attr", None)
        source_attr = getattr(source[edge_type], "edge_attr", None)
        edge_count = int(data[edge_type].edge_index.shape[1])
        rows = int(edge_attr.shape[0]) if edge_attr is not None else None
        dim = int(edge_attr.shape[1]) if edge_attr is not None and edge_attr.ndim == 2 else None
        source_dim = int(source_attr.shape[1]) if source_attr is not None and source_attr.ndim == 2 else None
        payload["edges"][str(edge_type)] = {
            "edge_count": edge_count,
            "edge_attr_rows": rows,
            "edge_attr_dim": dim,
            "source_edge_attr_dim": source_dim,
        }
        if source_dim is not None and dim != source_dim:
            errors.append(f"{edge_type} edge_attr dim {dim}; expected {source_dim}")
        if source_dim is not None and rows != edge_count:
            errors.append(f"{edge_type} edge_attr rows {rows}; edge_count {edge_count}")
        if edge_attr is not None and not torch.isfinite(edge_attr.detach().cpu()).all():
            errors.append(f"{edge_type} edge_attr contains non-finite values")

    pm = data[PM_NODE]
    synth = getattr(pm, "is_synthetic", torch.zeros(pm.x.shape[0], dtype=torch.bool)).detach().cpu().bool()
    payload["synthetic_masks"]["synthetic_count"] = int(synth.sum())
    if int(synth.sum()) <= 0:
        errors.append("No synthetic nodes")
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(pm, mask_name).detach().cpu().bool()
        payload["synthetic_masks"][f"{mask_name}_synthetic_count"] = int((synth & mask).sum())
    if bool((synth & ~pm.train_mask.detach().cpu().bool()).any()):
        errors.append("Synthetic nodes outside train_mask")
    if bool((synth & pm.val_mask.detach().cpu().bool()).any()):
        errors.append("Synthetic nodes leaked into val_mask")
    if bool((synth & pm.test_mask.detach().cpu().bool()).any()):
        errors.append("Synthetic nodes leaked into test_mask")
    if not torch.isfinite(pm.x.detach().cpu()).all():
        errors.append("pm.x contains non-finite values")

    seq = graph_obj.get("sequence_index")
    if seq is None:
        errors.append("sequence_index missing")
    else:
        sequence_rows = np.asarray(seq.sequence_rows)
        target_rows = np.asarray(seq.target_rows)
        synth_rows = torch.where(synth)[0].numpy()
        payload["sequence_index"] = {
            "sequence_rows_shape": list(sequence_rows.shape),
            "target_rows_shape": list(target_rows.shape),
            "max_row": int(sequence_rows.max()) if sequence_rows.size else None,
            "target_synthetic_count": int(np.isin(target_rows, synth_rows).sum()),
        }
        if sequence_rows.size and int(sequence_rows.max()) >= int(pm.x.shape[0]):
            errors.append("sequence_index references out-of-range row")
        if payload["sequence_index"]["target_synthetic_count"] != int(synth.sum()):
            errors.append("Not every synthetic node has a SequenceIndex target")

    payload["synthetic_feature_quality"] = _synthetic_feature_quality(data, source, synth)
    payload["synthetic_edge_quality"] = _synthetic_edge_quality(data, source, synth)
    payload["ok"] = not errors
    if artifact_path is not None:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        payload["artifact_path"] = str(path)
    return payload


def _filename_with_imgagn_tag(filename: str) -> str:
    if "_ImGAGN" in filename:
        return filename
    if filename.endswith(".pt"):
        return filename[:-3] + "_ImGAGN.pt"
    return filename + "_ImGAGN"


def _coerce_pm_index_for_pickle(pm_index: Any) -> Any:
    if pm_index is None:
        return None
    p_map = getattr(pm_index, "_map", None)
    r_map = getattr(pm_index, "_rev", None)
    if p_map is None or r_map is None:
        return pm_index
    try:
        from src.graph import PMIndex

        return PMIndex(dict(p_map), dict(r_map))
    except Exception:
        return {
            "_type": "PMIndex",
            "_map": dict(p_map),
            "_rev": dict(r_map),
        }


def to_cpu_graph_object(graph_obj: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(graph_obj))
    data = out.get("data")
    if isinstance(data, HeteroData):
        out["data"] = data.cpu()
    if "pm_index" in out:
        out["pm_index"] = _coerce_pm_index_for_pickle(out.get("pm_index"))
    return out


def create_warm_start_resume_state(
    *,
    source_checkpoint: str | Path,
    output_path: str | Path,
    monitor_metric: str = "val_auprc",
    max_epochs: int = 1,
) -> Path:
    source_path = Path(source_checkpoint)
    if not source_path.exists():
        raise FileNotFoundError(f"No existe checkpoint base: {source_path}")
    ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise ValueError("El checkpoint base no es un diccionario compatible.")
    model_state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if model_state is None and all(hasattr(v, "shape") for v in ckpt.values()):
        model_state = dict(ckpt)
    if model_state is None:
        raise ValueError("El checkpoint base no contiene model_state_dict/model_state.")

    out = {
        "epoch": 0,
        "max_epochs": int(max_epochs),
        "model_state": model_state,
        "model_state_dict": model_state,
        "optimizer_state": None,
        "scheduler_state": None,
        "best_val_loss": float("inf"),
        "best_val_f1": 0.0,
        "monitor_metric": str(monitor_metric),
        "monitor_mode": "max" if str(monitor_metric) != "val_loss" else "min",
        "best_monitor_value": -float("inf") if str(monitor_metric) != "val_loss" else float("inf"),
        "warm_start_source": str(source_path),
        "warm_start_mode": RELATIONAL_MODE,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    return output_path
