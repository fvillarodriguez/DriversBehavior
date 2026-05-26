from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import BATCH_SIZE, NUM_NEIGHBORS, RESULTADOS_DIR, SEED
from src.gnn_artifacts import gnn_dir

try:  # pragma: no cover - exercised in environments with torch_geometric.
    from torch_geometric.loader import NeighborLoader
except Exception:  # pragma: no cover
    NeighborLoader = None


EdgeType = Tuple[str, str, str]


@dataclass
class GNNXAIResult:
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    summary_df: pd.DataFrame
    manifest: Dict[str, Any]
    explanations_df: Optional[pd.DataFrame] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _edge_type_from_key(key: str) -> Tuple[Optional[int], EdgeType]:
    parts = str(key or "").split("_")
    layer = None
    if parts and parts[0].startswith("conv"):
        try:
            layer = int(parts[0].replace("conv", ""))
        except Exception:
            layer = None
        parts = parts[1:]
    if len(parts) >= 3:
        return layer, (parts[0], "_".join(parts[1:-1]), parts[-1])
    return layer, ("pm", str(key), "pm")


def _edge_type_label(edge_type: EdgeType) -> str:
    return f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"


def _edge_type_module_key(edge_type: EdgeType) -> str:
    return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"


def _normalize_edge_type_key(key: object, edge_types: Iterable[EdgeType]) -> Optional[EdgeType]:
    if isinstance(key, tuple) and len(key) == 3:
        return tuple(str(p) for p in key)  # type: ignore[return-value]
    text = str(key or "").strip()
    if not text:
        return None
    for edge_type in edge_types:
        if text in {
            str(edge_type),
            edge_type[1],
            _edge_type_label(edge_type),
            _edge_type_module_key(edge_type),
        }:
            return edge_type
    if "__" in text:
        parts = text.split("__")
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    if "___" in text:
        parts = text.split("___")
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    return None


def _coerce_neighbor_profile(profile: object, num_layers: int) -> List[int]:
    if profile is None:
        return [-1] * max(1, int(num_layers or 1))
    if isinstance(profile, str):
        text = profile.strip()
        if not text:
            return [-1] * max(1, int(num_layers or 1))
        try:
            profile = json.loads(text)
        except Exception:
            sep = "-" if "-" in text and "," not in text and "[" not in text else ","
            profile = [p.strip() for p in text.split(sep) if p.strip()]
    if isinstance(profile, (int, float, np.integer, np.floating)):
        values = [int(profile)]
    elif isinstance(profile, (list, tuple)):
        values = []
        for item in profile:
            try:
                values.append(int(item))
            except Exception:
                continue
    else:
        values = []
    if not values:
        values = [-1]
    return values


def _resolve_num_neighbors(
    num_neighbors: object,
    edge_types: Sequence[EdgeType],
    *,
    num_layers: int,
) -> object:
    spec = NUM_NEIGHBORS if num_neighbors is None else num_neighbors
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            spec = NUM_NEIGHBORS
        else:
            try:
                spec = json.loads(text)
            except Exception:
                spec = text
    if isinstance(spec, Mapping):
        resolved: Dict[EdgeType, List[int]] = {}
        for edge_type in edge_types:
            value = None
            for key in (
                edge_type,
                edge_type[1],
                _edge_type_label(edge_type),
                _edge_type_module_key(edge_type),
            ):
                if key in spec:
                    value = spec[key]
                    break
            if value is None:
                value = spec.get("default", [-1] * max(1, int(num_layers or 1)))
            resolved[edge_type] = _coerce_neighbor_profile(value, num_layers)
        return resolved
    return _coerce_neighbor_profile(spec, num_layers)


def _infer_device(model: torch.nn.Module, device: Optional[object]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _node_count(data: Any, node_type: str) -> int:
    store = data[node_type]
    value = getattr(store, "num_nodes", None)
    if value is not None:
        return int(value)
    return int(store.x.size(0))


def _resolve_node_indices(
    data: Any,
    node_type: str,
    *,
    node_indices: Optional[object],
    mask_name: str,
) -> torch.Tensor:
    if node_indices is not None:
        idx = torch.as_tensor(node_indices, dtype=torch.long).detach().cpu()
        return idx.view(-1)
    mask = getattr(data[node_type], mask_name, None)
    if mask is None and hasattr(data[node_type], "__contains__") and mask_name in data[node_type]:
        mask = data[node_type][mask_name]
    if mask is not None:
        mask_t = torch.as_tensor(mask).detach().cpu()
        if mask_t.dtype == torch.bool:
            return mask_t.nonzero(as_tuple=False).view(-1).long()
        return mask_t.view(-1).long()
    return torch.arange(_node_count(data, node_type), dtype=torch.long)


def _portico_for_node(pm_index: Optional[object], node_idx: int) -> str:
    if pm_index is None:
        return str(node_idx)
    try:
        rev = getattr(pm_index, "_rev", None)
        if rev is not None:
            if isinstance(rev, Mapping):
                value = rev.get(int(node_idx))
                if value is not None:
                    return str(value)
            if isinstance(rev, (list, tuple)) and 0 <= int(node_idx) < len(rev):
                return str(rev[int(node_idx)])
    except Exception:
        pass
    try:
        if isinstance(pm_index, Mapping):
            inverse = {int(v): str(k) for k, v in pm_index.items()}
            if int(node_idx) in inverse:
                return inverse[int(node_idx)]
    except Exception:
        pass
    return str(node_idx)


def _ts_min_for_node(data: Any, node_type: str, node_idx: int) -> object:
    for attr in ("ts_min", "timestamp", "time_idx"):
        try:
            value = getattr(data[node_type], attr)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            item = value[int(node_idx)]
            if hasattr(item, "item"):
                item = item.item()
            return item
        except Exception:
            continue
    return None


def _error_type(y_true: object, y_pred: object) -> str:
    if y_true is None or y_pred is None:
        return "unknown"
    try:
        yt = int(y_true)
        yp = int(y_pred)
    except Exception:
        return "unknown"
    if yt == 1 and yp == 1:
        return "TP"
    if yt == 0 and yp == 0:
        return "TN"
    if yt == 0 and yp == 1:
        return "FP"
    if yt == 1 and yp == 0:
        return "FN"
    return "unknown"


def _mask_label(position: int, batch_size: int, mask_name: str) -> str:
    return str(mask_name) if int(position) < int(batch_size) else "context"


def _as_edge_attr_dict(batch: Any) -> Dict[EdgeType, torch.Tensor]:
    out: Dict[EdgeType, torch.Tensor] = {}
    for edge_type in getattr(batch, "edge_types", []):
        store = batch[edge_type]
        edge_attr = getattr(store, "edge_attr", None)
        if edge_attr is not None:
            out[tuple(edge_type)] = edge_attr
    return out


def _call_model(
    model: torch.nn.Module,
    batch: Any,
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    out = model(batch.x_dict, batch.edge_index_dict, dict(edge_attr_dict))
    return _coerce_model_output(out)


def _call_model_inputs(
    model: torch.nn.Module,
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[EdgeType, torch.Tensor],
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    out = model(dict(x_dict), dict(edge_index_dict), dict(edge_attr_dict))
    return _coerce_model_output(out)


def _coerce_model_output(
    out: object,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if isinstance(out, tuple) and len(out) == 3:
        logits_dict, embeddings_dict, attentions = out
        return dict(logits_dict), dict(embeddings_dict), dict(attentions or {})
    if isinstance(out, tuple) and len(out) == 2:
        logits_dict, embeddings_dict = out
        return dict(logits_dict), dict(embeddings_dict), {}
    if isinstance(out, Mapping):
        return dict(out), {}, {}
    raise RuntimeError("El modelo GNN no devolvió un contrato compatible con XAI.")


def _calibrate_probabilities(probs: np.ndarray, calibration_model: Optional[object]) -> np.ndarray:
    if calibration_model is None:
        return probs
    if probs.size == 0 or probs.shape[1] < 2:
        return probs
    try:
        prob1 = calibration_model.predict_proba(probs[:, [1]])[:, 1]
        prob1 = np.clip(prob1.astype(float), 0.0, 1.0)
        return np.column_stack([1.0 - prob1, prob1])
    except Exception:
        return probs


def _prob1_from_logits(logits: torch.Tensor, calibration_model: Optional[object]) -> np.ndarray:
    logits_cpu = logits.detach().cpu()
    probs = F.softmax(logits_cpu, dim=-1).numpy()
    probs = _calibrate_probabilities(probs, calibration_model)
    return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]


def _target_logits_from_outputs(
    logits_dict: Mapping[str, torch.Tensor],
    embeddings_dict: Mapping[str, torch.Tensor],
    *,
    temporal_module: Optional[torch.nn.Module],
    node_type: str,
    node_n_id: torch.Tensor,
    target_bs: int,
) -> torch.Tensor:
    if temporal_module is not None:
        emb = embeddings_dict.get(node_type)
        if emb is None:
            raise RuntimeError(f"El modelo no devolvio embeddings para node_type={node_type!r}.")
        return temporal_module(emb, node_n_id.to(emb.device), target_bs)
    logits = logits_dict.get(node_type)
    if logits is None:
        raise RuntimeError(f"El modelo no devolvio logits para node_type={node_type!r}.")
    return logits[:target_bs]


def _snapshot_temporal_cache(temporal_module: Optional[torch.nn.Module]) -> Optional[torch.Tensor]:
    if temporal_module is None:
        return None
    cache = getattr(temporal_module, "embedding_cache", None)
    if isinstance(cache, torch.Tensor):
        return cache.detach().clone()
    return None


def _restore_temporal_cache(
    temporal_module: Optional[torch.nn.Module],
    snapshot: Optional[torch.Tensor],
) -> None:
    if temporal_module is None or snapshot is None:
        return
    cache = getattr(temporal_module, "embedding_cache", None)
    if not isinstance(cache, torch.Tensor):
        return
    if tuple(cache.shape) != tuple(snapshot.shape):
        try:
            temporal_module.embedding_cache = snapshot.to(device=cache.device, dtype=cache.dtype).clone()
        except Exception:
            return
    else:
        cache.copy_(snapshot.to(device=cache.device, dtype=cache.dtype))


def _empty_edge_attr_like(edge_attr: Optional[torch.Tensor], edge_index: torch.Tensor) -> Optional[torch.Tensor]:
    if edge_attr is None:
        return None
    if edge_attr.dim() <= 1:
        return edge_attr.new_zeros((0,))
    return edge_attr.new_zeros((0, int(edge_attr.size(1))))


def _ablate_relation_inputs(
    batch: Any,
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    relation_to_remove: EdgeType,
) -> Tuple[Dict[str, torch.Tensor], Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
    x_dict = dict(batch.x_dict)
    edge_index_dict = dict(batch.edge_index_dict)
    edge_attrs = dict(edge_attr_dict)
    edge_index = edge_index_dict.get(relation_to_remove)
    if edge_index is None:
        return x_dict, edge_index_dict, edge_attrs
    edge_index_dict[relation_to_remove] = edge_index.new_zeros((2, 0))
    if relation_to_remove in edge_attrs:
        empty_attr = _empty_edge_attr_like(edge_attrs.get(relation_to_remove), edge_index)
        if empty_attr is not None:
            edge_attrs[relation_to_remove] = empty_attr
    return x_dict, edge_index_dict, edge_attrs


def _keep_only_relation_inputs(
    batch: Any,
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    relation_to_keep: EdgeType,
) -> Tuple[Dict[str, torch.Tensor], Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
    x_dict = dict(batch.x_dict)
    edge_index_dict = dict(batch.edge_index_dict)
    edge_attrs = dict(edge_attr_dict)
    for edge_type, edge_index in list(edge_index_dict.items()):
        edge_type_t = tuple(edge_type)
        if edge_type_t == relation_to_keep:
            continue
        edge_index_dict[edge_type_t] = edge_index.new_zeros((2, 0))
        if edge_type_t in edge_attrs:
            empty_attr = _empty_edge_attr_like(edge_attrs.get(edge_type_t), edge_index)
            if empty_attr is not None:
                edge_attrs[edge_type_t] = empty_attr
    return x_dict, edge_index_dict, edge_attrs


def _ablate_feature_inputs(
    batch: Any,
    *,
    node_type: str,
    feature_idx: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
    x_dict = dict(batch.x_dict)
    edge_index_dict = dict(batch.edge_index_dict)
    edge_attrs = _as_edge_attr_dict(batch)
    x = x_dict.get(node_type)
    if x is None or x.dim() != 2 or feature_idx < 0 or feature_idx >= int(x.size(1)):
        return x_dict, edge_index_dict, edge_attrs
    x_ablated = x.clone()
    x_ablated[:, int(feature_idx)] = 0
    x_dict[node_type] = x_ablated
    return x_dict, edge_index_dict, edge_attrs


def _select_feature_indices(
    x: torch.Tensor,
    *,
    target_bs: int,
    feature_names: Optional[Sequence[str]],
    top_k: int,
) -> List[Tuple[int, str]]:
    if x is None or x.dim() != 2 or top_k <= 0:
        return []
    n_features = int(x.size(1))
    if n_features <= 0:
        return []
    x_target = x[: max(1, int(target_bs))].detach().float().cpu()
    scores = torch.nan_to_num(x_target.abs(), nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0)
    k = min(int(top_k), n_features)
    if k <= 0:
        return []
    _, indices = torch.topk(scores, k=k)
    names = list(feature_names or [])
    out: List[Tuple[int, str]] = []
    for idx in indices.tolist():
        name = names[idx] if 0 <= idx < len(names) else f"x_{idx}"
        out.append((int(idx), str(name)))
    return out


def _predict_ablation_prob1(
    model: torch.nn.Module,
    *,
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[EdgeType, torch.Tensor],
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    temporal_module: Optional[torch.nn.Module],
    node_type: str,
    node_n_id: torch.Tensor,
    target_bs: int,
    calibration_model: Optional[object],
) -> np.ndarray:
    snapshot = _snapshot_temporal_cache(temporal_module)
    try:
        logits_dict, embeddings_dict, _ = _call_model_inputs(model, x_dict, edge_index_dict, edge_attr_dict)
        logits_target = _target_logits_from_outputs(
            logits_dict,
            embeddings_dict,
            temporal_module=temporal_module,
            node_type=node_type,
            node_n_id=node_n_id,
            target_bs=target_bs,
        )
        return _prob1_from_logits(logits_target, calibration_model)
    finally:
        _restore_temporal_cache(temporal_module, snapshot)


def _build_explanation_row(
    *,
    method: str,
    node_row: Mapping[str, Any],
    base_prob1: float,
    perturbed_prob1: float,
    threshold: Optional[float],
    relation: Optional[str] = None,
    feature_idx: Optional[int] = None,
    feature_name: Optional[str] = None,
    sufficient_prob1: Optional[float] = None,
) -> Dict[str, Any]:
    delta = float(perturbed_prob1 - base_prob1)
    tau = 0.5 if threshold is None else float(threshold)
    crossed = (base_prob1 >= tau and perturbed_prob1 < tau) or (base_prob1 < tau and perturbed_prob1 >= tau)
    suff_score = np.nan
    if sufficient_prob1 is not None:
        suff_score = float(max(0.0, 1.0 - abs(float(base_prob1) - float(sufficient_prob1))))
    return {
        "method": str(method),
        "node_idx": int(node_row.get("node_idx")),
        "portico": node_row.get("portico"),
        "ts_min": node_row.get("ts_min"),
        "mask": node_row.get("mask"),
        "y_true": node_row.get("y_true"),
        "y_pred": node_row.get("y_pred"),
        "error_type": node_row.get("error_type"),
        "relation": relation,
        "feature_idx": feature_idx,
        "feature_name": feature_name,
        "base_prob1": float(base_prob1),
        "perturbed_prob1": float(perturbed_prob1),
        "sufficient_prob1": None if sufficient_prob1 is None else float(sufficient_prob1),
        "delta_prob1": delta,
        "score": abs(delta),
        "necessity": float(max(0.0, float(base_prob1) - float(perturbed_prob1))),
        "sufficiency": suff_score,
        "threshold": tau,
        "crossed_threshold": bool(crossed),
    }


def _build_feature_attribution_row(
    *,
    method: str,
    node_row: Mapping[str, Any],
    base_prob1: float,
    threshold: Optional[float],
    feature_idx: int,
    feature_name: str,
    attribution: float,
    target_class: int,
    gradient_steps: Optional[int] = None,
) -> Dict[str, Any]:
    tau = 0.5 if threshold is None else float(threshold)
    return {
        "method": str(method),
        "node_idx": int(node_row.get("node_idx")),
        "portico": node_row.get("portico"),
        "ts_min": node_row.get("ts_min"),
        "mask": node_row.get("mask"),
        "y_true": node_row.get("y_true"),
        "y_pred": node_row.get("y_pred"),
        "error_type": node_row.get("error_type"),
        "relation": None,
        "feature_idx": int(feature_idx),
        "feature_name": str(feature_name),
        "base_prob1": float(base_prob1),
        "perturbed_prob1": np.nan,
        "sufficient_prob1": np.nan,
        "delta_prob1": np.nan,
        "score": abs(float(attribution)),
        "necessity": np.nan,
        "sufficiency": np.nan,
        "threshold": tau,
        "crossed_threshold": False,
        "attribution": float(attribution),
        "target_class": int(target_class),
        "gradient_steps": None if gradient_steps is None else int(gradient_steps),
    }


def _top_feature_attribution_rows(
    *,
    method: str,
    selected_rows: Sequence[Mapping[str, Any]],
    base_prob1: np.ndarray,
    attributions: torch.Tensor,
    threshold: Optional[float],
    feature_names: Optional[Sequence[str]],
    top_features: int,
    target_class: int,
    gradient_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if attributions is None or attributions.dim() != 2 or top_features <= 0:
        return []
    names = list(feature_names or [])
    rows: List[Dict[str, Any]] = []
    selected_count = min(len(selected_rows), int(attributions.size(0)), len(base_prob1))
    if selected_count <= 0:
        return rows
    k = min(int(top_features), int(attributions.size(1)))
    if k <= 0:
        return rows
    attr_cpu = torch.nan_to_num(
        attributions[:selected_count].detach().float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    for pos in range(selected_count):
        _, indices = torch.topk(attr_cpu[pos].abs(), k=k)
        for feature_idx in indices.tolist():
            feature_name = names[feature_idx] if 0 <= feature_idx < len(names) else f"x_{feature_idx}"
            rows.append(
                _build_feature_attribution_row(
                    method=method,
                    node_row=selected_rows[pos],
                    base_prob1=float(base_prob1[pos]),
                    threshold=threshold,
                    feature_idx=int(feature_idx),
                    feature_name=str(feature_name),
                    attribution=float(attr_cpu[pos, feature_idx].item()),
                    target_class=target_class,
                    gradient_steps=gradient_steps,
                )
            )
    return rows


def _gradient_feature_attributions(
    *,
    model: torch.nn.Module,
    batch: Any,
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    temporal_module: Optional[torch.nn.Module],
    node_type: str,
    node_n_id: torch.Tensor,
    target_bs: int,
    selected_count: int,
    target_class: int = 1,
) -> Optional[torch.Tensor]:
    x = batch.x_dict.get(node_type)
    if x is None or x.dim() != 2 or selected_count <= 0:
        return None
    x_grad = x.detach().clone().requires_grad_(True)
    x_dict = dict(batch.x_dict)
    x_dict[node_type] = x_grad
    snapshot = _snapshot_temporal_cache(temporal_module)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            if temporal_module is not None:
                temporal_module.zero_grad(set_to_none=True)
            logits_dict, embeddings_dict, _ = _call_model_inputs(
                model,
                x_dict,
                batch.edge_index_dict,
                edge_attr_dict,
            )
            logits_target = _target_logits_from_outputs(
                logits_dict,
                embeddings_dict,
                temporal_module=temporal_module,
                node_type=node_type,
                node_n_id=node_n_id,
                target_bs=target_bs,
            )
            if logits_target.numel() == 0:
                return None
            cls = min(max(0, int(target_class)), int(logits_target.size(1)) - 1)
            score = logits_target[:selected_count, cls].sum()
            score.backward()
            grad = x_grad.grad
            if grad is None:
                return None
            return (grad[:selected_count].detach() * x_grad[:selected_count].detach()).cpu()
    finally:
        _restore_temporal_cache(temporal_module, snapshot)
        try:
            model.zero_grad(set_to_none=True)
            if temporal_module is not None:
                temporal_module.zero_grad(set_to_none=True)
        except Exception:
            pass


def _integrated_gradient_feature_attributions(
    *,
    model: torch.nn.Module,
    batch: Any,
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    temporal_module: Optional[torch.nn.Module],
    node_type: str,
    node_n_id: torch.Tensor,
    target_bs: int,
    selected_count: int,
    steps: int,
    target_class: int = 1,
) -> Optional[torch.Tensor]:
    x = batch.x_dict.get(node_type)
    if x is None or x.dim() != 2 or selected_count <= 0:
        return None
    effective_steps = max(1, int(steps or 1))
    x_input = x.detach()
    baseline = torch.zeros_like(x_input)
    total_grad = torch.zeros_like(x_input[:selected_count])
    alphas = torch.linspace(
        1.0 / float(effective_steps),
        1.0,
        steps=effective_steps,
        device=x_input.device,
        dtype=x_input.dtype,
    )
    for alpha in alphas:
        x_scaled = (baseline + alpha * (x_input - baseline)).detach().clone().requires_grad_(True)
        x_dict = dict(batch.x_dict)
        x_dict[node_type] = x_scaled
        snapshot = _snapshot_temporal_cache(temporal_module)
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                if temporal_module is not None:
                    temporal_module.zero_grad(set_to_none=True)
                logits_dict, embeddings_dict, _ = _call_model_inputs(
                    model,
                    x_dict,
                    batch.edge_index_dict,
                    edge_attr_dict,
                )
                logits_target = _target_logits_from_outputs(
                    logits_dict,
                    embeddings_dict,
                    temporal_module=temporal_module,
                    node_type=node_type,
                    node_n_id=node_n_id,
                    target_bs=target_bs,
                )
                if logits_target.numel() == 0:
                    continue
                cls = min(max(0, int(target_class)), int(logits_target.size(1)) - 1)
                score = logits_target[:selected_count, cls].sum()
                score.backward()
                if x_scaled.grad is not None:
                    total_grad += x_scaled.grad[:selected_count].detach()
        finally:
            _restore_temporal_cache(temporal_module, snapshot)
            try:
                model.zero_grad(set_to_none=True)
                if temporal_module is not None:
                    temporal_module.zero_grad(set_to_none=True)
            except Exception:
                pass
    avg_grad = total_grad / float(effective_steps)
    return ((x_input[:selected_count] - baseline[:selected_count]) * avg_grad).detach().cpu()


def _batch_advanced_explanations(
    *,
    model: torch.nn.Module,
    batch: Any,
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    logits_dict: Mapping[str, torch.Tensor],
    embeddings_dict: Mapping[str, torch.Tensor],
    temporal_module: Optional[torch.nn.Module],
    node_type: str,
    node_rows: Sequence[Mapping[str, Any]],
    target_bs: int,
    threshold: Optional[float],
    calibration_model: Optional[object],
    explanation_methods: Sequence[str],
    feature_names: Optional[Sequence[str]],
    top_features: int,
    explain_limit: int,
    integrated_gradients_steps: int,
) -> List[Dict[str, Any]]:
    methods = {str(m).strip().lower() for m in explanation_methods if str(m).strip()}
    if not methods or explain_limit <= 0 or target_bs <= 0:
        return []
    node_n_id = getattr(batch[node_type], "n_id").detach().cpu()
    logits_target = _target_logits_from_outputs(
        logits_dict,
        embeddings_dict,
        temporal_module=temporal_module,
        node_type=node_type,
        node_n_id=node_n_id,
        target_bs=target_bs,
    )
    base_prob1 = _prob1_from_logits(logits_target, calibration_model)
    selected_count = min(int(target_bs), int(explain_limit), len(node_rows), len(base_prob1))
    if selected_count <= 0:
        return []
    selected_rows = list(node_rows[:selected_count])
    rows: List[Dict[str, Any]] = []
    target_class = 1

    if "relation_ablation" in methods:
        for edge_type in getattr(batch, "edge_types", []):
            edge_type_t = tuple(edge_type)
            relation = str(edge_type_t[1]) if len(edge_type_t) >= 2 else str(edge_type_t)
            try:
                x_dict, edge_index_dict, edge_attrs = _ablate_relation_inputs(batch, edge_attr_dict, edge_type_t)
                perturbed = _predict_ablation_prob1(
                    model,
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    edge_attr_dict=edge_attrs,
                    temporal_module=temporal_module,
                    node_type=node_type,
                    node_n_id=node_n_id,
                    target_bs=target_bs,
                    calibration_model=calibration_model,
                )
                x_keep, edge_index_keep, edge_attrs_keep = _keep_only_relation_inputs(batch, edge_attr_dict, edge_type_t)
                sufficient = _predict_ablation_prob1(
                    model,
                    x_dict=x_keep,
                    edge_index_dict=edge_index_keep,
                    edge_attr_dict=edge_attrs_keep,
                    temporal_module=temporal_module,
                    node_type=node_type,
                    node_n_id=node_n_id,
                    target_bs=target_bs,
                    calibration_model=calibration_model,
                )
            except Exception:
                continue
            for pos, node_row in enumerate(selected_rows):
                if pos >= len(perturbed) or pos >= len(sufficient):
                    continue
                rows.append(
                    _build_explanation_row(
                        method="relation_ablation",
                        node_row=node_row,
                        base_prob1=float(base_prob1[pos]),
                        perturbed_prob1=float(perturbed[pos]),
                        sufficient_prob1=float(sufficient[pos]),
                        threshold=threshold,
                        relation=relation,
                    )
                )

    if "feature_ablation" in methods:
        feature_candidates = _select_feature_indices(
            batch.x_dict.get(node_type),
            target_bs=target_bs,
            feature_names=feature_names,
            top_k=int(top_features or 0),
        )
        for feature_idx, feature_name in feature_candidates:
            try:
                x_dict, edge_index_dict, edge_attrs = _ablate_feature_inputs(
                    batch,
                    node_type=node_type,
                    feature_idx=int(feature_idx),
                )
                perturbed = _predict_ablation_prob1(
                    model,
                    x_dict=x_dict,
                    edge_index_dict=edge_index_dict,
                    edge_attr_dict=edge_attrs,
                    temporal_module=temporal_module,
                    node_type=node_type,
                    node_n_id=node_n_id,
                    target_bs=target_bs,
                    calibration_model=calibration_model,
                )
            except Exception:
                continue
            for pos, node_row in enumerate(selected_rows):
                if pos >= len(perturbed):
                    continue
                rows.append(
                    _build_explanation_row(
                        method="feature_ablation",
                        node_row=node_row,
                        base_prob1=float(base_prob1[pos]),
                        perturbed_prob1=float(perturbed[pos]),
                        threshold=threshold,
                        feature_idx=int(feature_idx),
                        feature_name=str(feature_name),
                    )
                )

    if "input_gradient" in methods and top_features > 0:
        attr = _gradient_feature_attributions(
            model=model,
            batch=batch,
            edge_attr_dict=edge_attr_dict,
            temporal_module=temporal_module,
            node_type=node_type,
            node_n_id=node_n_id,
            target_bs=target_bs,
            selected_count=selected_count,
            target_class=target_class,
        )
        if attr is not None:
            rows.extend(
                _top_feature_attribution_rows(
                    method="input_gradient",
                    selected_rows=selected_rows,
                    base_prob1=base_prob1,
                    attributions=attr,
                    threshold=threshold,
                    feature_names=feature_names,
                    top_features=int(top_features or 0),
                    target_class=target_class,
                )
            )

    if "integrated_gradients" in methods and top_features > 0:
        attr = _integrated_gradient_feature_attributions(
            model=model,
            batch=batch,
            edge_attr_dict=edge_attr_dict,
            temporal_module=temporal_module,
            node_type=node_type,
            node_n_id=node_n_id,
            target_bs=target_bs,
            selected_count=selected_count,
            steps=int(integrated_gradients_steps or 1),
            target_class=target_class,
        )
        if attr is not None:
            rows.extend(
                _top_feature_attribution_rows(
                    method="integrated_gradients",
                    selected_rows=selected_rows,
                    base_prob1=base_prob1,
                    attributions=attr,
                    threshold=threshold,
                    feature_names=feature_names,
                    top_features=int(top_features or 0),
                    target_class=target_class,
                    gradient_steps=int(integrated_gradients_steps or 1),
                )
            )

    return rows


def _predict_batch_nodes(
    logits: torch.Tensor,
    y: Optional[torch.Tensor],
    global_node_ids: torch.Tensor,
    *,
    pm_index: Optional[object],
    data: Any,
    node_type: str,
    mask_name: str,
    batch_size: int,
    threshold: Optional[float],
    calibration_model: Optional[object],
) -> List[Dict[str, Any]]:
    logits_cpu = logits.detach().cpu()
    probs = F.softmax(logits_cpu, dim=-1).numpy()
    probs = _calibrate_probabilities(probs, calibration_model)
    prob1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    if threshold is None:
        preds = np.argmax(probs, axis=1).astype(int)
    else:
        preds = (prob1 >= float(threshold)).astype(int)
    y_cpu = y.detach().cpu() if y is not None else None
    rows: List[Dict[str, Any]] = []
    for pos, node_id in enumerate(global_node_ids.detach().cpu().view(-1).tolist()):
        yt = None
        if y_cpu is not None and pos < int(y_cpu.numel()):
            yt = int(y_cpu[pos].item())
        yp = int(preds[pos]) if pos < len(preds) else None
        rows.append(
            {
                "node_idx": int(node_id),
                "portico": _portico_for_node(pm_index, int(node_id)),
                "ts_min": _ts_min_for_node(data, node_type, int(node_id)),
                "mask": _mask_label(pos, batch_size, mask_name),
                "y_true": yt,
                "y_pred": yp,
                "prob1": float(prob1[pos]) if pos < len(prob1) else np.nan,
                "error_type": _error_type(yt, yp),
            }
        )
    return rows


def _iter_attention_sources(
    model: torch.nn.Module,
    attentions: Mapping[str, torch.Tensor],
) -> Iterable[Tuple[int, EdgeType, torch.Tensor, Optional[torch.Tensor]]]:
    yielded = set()

    convs = getattr(model, "convs", None)
    if convs is not None:
        for layer_idx, hetero_conv in enumerate(convs, start=1):
            conv_dict = getattr(hetero_conv, "convs", {})
            for edge_type, conv_layer in conv_dict.items():
                edge_type_t = tuple(edge_type)
                key = f"conv{layer_idx}_{_edge_type_label(edge_type_t)}"
                alpha = attentions.get(key)
                if alpha is None:
                    alpha = getattr(conv_layer, "_alpha", None)
                edge_index = getattr(conv_layer, "_alpha_edge_index", None)
                if alpha is not None:
                    yielded.add((layer_idx, edge_type_t))
                    yield layer_idx, edge_type_t, alpha, edge_index

    stages = getattr(model, "stages", None)
    stage_kinds = getattr(model, "stage_kinds", None)
    if stages is not None and stage_kinds is not None:
        conv_idx = 0
        for stage, kind in zip(stages, stage_kinds):
            if kind != "hetero_conv":
                continue
            conv_idx += 1
            conv = getattr(stage, "conv", None)
            conv_dict = getattr(conv, "convs", {}) if conv is not None else {}
            for edge_type, conv_layer in conv_dict.items():
                edge_type_t = tuple(edge_type)
                if (conv_idx, edge_type_t) in yielded:
                    continue
                key = f"conv{conv_idx}_{_edge_type_label(edge_type_t)}"
                alpha = attentions.get(key)
                if alpha is None:
                    alpha = getattr(conv_layer, "_alpha", None)
                edge_index = getattr(conv_layer, "_alpha_edge_index", None)
                if alpha is not None:
                    yield conv_idx, edge_type_t, alpha, edge_index

    for key, alpha in (attentions or {}).items():
        layer_idx, edge_type = _edge_type_from_key(str(key))
        if layer_idx is None:
            layer_idx = 1
        if (layer_idx, edge_type) in yielded:
            continue
        yield layer_idx, edge_type, alpha, None


def _attention_edges_for_batch(
    *,
    model: torch.nn.Module,
    attentions: Mapping[str, torch.Tensor],
    batch: Any,
    layer_filter: Optional[object],
    percentile: float,
    k_heads: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        n_id_by_type = {
            node_type: getattr(batch[node_type], "n_id").detach().cpu()
            for node_type in getattr(batch, "node_types", [])
            if hasattr(batch[node_type], "n_id")
        }
    except Exception:
        n_id_by_type = {}

    selected_layer = None
    if layer_filter not in (None, "auto", "last"):
        if isinstance(layer_filter, str) and layer_filter.startswith("conv"):
            try:
                selected_layer = int(layer_filter.replace("conv", ""))
            except Exception:
                selected_layer = None
        else:
            try:
                selected_layer = int(layer_filter)
            except Exception:
                selected_layer = None

    collected = list(_iter_attention_sources(model, attentions))
    if layer_filter in ("auto", "last", None) and collected:
        selected_layer = max(int(layer_idx) for layer_idx, _, _, _ in collected)

    for layer_idx, edge_type, alpha, eff_edge_index in collected:
        if selected_layer is not None and int(layer_idx) != int(selected_layer):
            continue
        if alpha is None:
            continue
        alpha_cpu = alpha.detach().cpu()
        if alpha_cpu.dim() == 1:
            alpha_heads = alpha_cpu.view(-1, 1)
        else:
            alpha_heads = alpha_cpu.view(alpha_cpu.size(0), -1)
        if alpha_heads.numel() == 0:
            continue

        if eff_edge_index is None:
            try:
                eff_edge_index = batch[edge_type].edge_index
            except Exception:
                eff_edge_index = None
        if eff_edge_index is None:
            continue
        edge_index_cpu = eff_edge_index.detach().cpu()
        edge_count = min(int(edge_index_cpu.size(1)), int(alpha_heads.size(0)))
        if edge_count <= 0:
            continue
        edge_index_cpu = edge_index_cpu[:, :edge_count]
        alpha_heads = alpha_heads[:edge_count]

        flat = alpha_heads.reshape(-1)
        try:
            threshold = float(torch.quantile(flat, float(percentile)).item())
        except Exception:
            threshold = float(flat.max().item())
        heads_above = (alpha_heads >= threshold).sum(dim=1)
        keep = heads_above >= max(1, int(k_heads or 1))
        if not bool(keep.any()):
            continue

        src_type, relation, dst_type = edge_type
        src_n_id = n_id_by_type.get(src_type)
        dst_n_id = n_id_by_type.get(dst_type)
        if src_n_id is None or dst_n_id is None:
            continue
        for edge_pos in torch.nonzero(keep, as_tuple=False).view(-1).tolist():
            local_src = int(edge_index_cpu[0, edge_pos].item())
            local_dst = int(edge_index_cpu[1, edge_pos].item())
            if local_src < 0 or local_dst < 0:
                continue
            if local_src >= int(src_n_id.numel()) or local_dst >= int(dst_n_id.numel()):
                continue
            global_src = int(src_n_id[local_src].item())
            global_dst = int(dst_n_id[local_dst].item())
            if global_src == global_dst:
                continue
            head_values = alpha_heads[edge_pos]
            rows.append(
                {
                    "layer": f"conv{int(layer_idx)}",
                    "relation": str(relation),
                    "source_node_idx": global_src,
                    "dest_node_idx": global_dst,
                    "mean_attention": float(head_values.mean().item()),
                    "max_attention": float(head_values.max().item()),
                    "heads_above_threshold": int(heads_above[edge_pos].item()),
                    "head_count": int(alpha_heads.size(1)),
                    "attention_threshold": float(threshold),
                }
            )
    return rows


def _deduplicate_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        return edges_df
    group_cols = ["layer", "relation", "source_node_idx", "dest_node_idx"]
    agg = {
        "mean_attention": "mean",
        "max_attention": "max",
        "heads_above_threshold": "max",
        "head_count": "max",
        "attention_threshold": "mean",
    }
    dedup = edges_df.groupby(group_cols, as_index=False).agg(agg)
    dedup["batch_occurrences"] = (
        edges_df.groupby(group_cols).size().reset_index(name="batch_occurrences")["batch_occurrences"].values
    )
    return dedup


def _join_predictions(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        for prefix in ("source", "dest"):
            edges_df[f"{prefix}_prob1"] = pd.Series(dtype=float)
            edges_df[f"{prefix}_error_type"] = pd.Series(dtype=str)
            edges_df[f"{prefix}_y_true"] = pd.Series(dtype=float)
            edges_df[f"{prefix}_y_pred"] = pd.Series(dtype=float)
        return edges_df
    node_cols = ["node_idx", "prob1", "error_type", "y_true", "y_pred", "portico", "mask"]
    lookup = nodes_df[node_cols].drop_duplicates("node_idx") if not nodes_df.empty else pd.DataFrame(columns=node_cols)
    merged = edges_df.merge(
        lookup.add_prefix("source_"),
        left_on="source_node_idx",
        right_on="source_node_idx",
        how="left",
    )
    merged = merged.merge(
        lookup.add_prefix("dest_"),
        left_on="dest_node_idx",
        right_on="dest_node_idx",
        how="left",
    )
    return merged


def _summary(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "relation",
                "edge_count",
                "mean_attention",
                "max_attention",
                "mean_heads_above_threshold",
            ]
        )
    return (
        edges_df.groupby(["layer", "relation"], as_index=False)
        .agg(
            edge_count=("source_node_idx", "count"),
            mean_attention=("mean_attention", "mean"),
            max_attention=("max_attention", "max"),
            mean_heads_above_threshold=("heads_above_threshold", "mean"),
        )
        .sort_values(["layer", "relation"])
        .reset_index(drop=True)
    )


def compute_gnn_xai_graph(
    *,
    model: torch.nn.Module,
    graph: Any,
    pm_index: Optional[object] = None,
    mask_name: str = "test_mask",
    node_indices: Optional[object] = None,
    batch_size: Optional[int] = None,
    num_neighbors: Optional[object] = None,
    percentile: float = 0.95,
    k_heads: int = 3,
    layer: Optional[object] = "auto",
    node_type: str = "pm",
    threshold: Optional[float] = None,
    calibration_model: Optional[object] = None,
    temporal_module: Optional[torch.nn.Module] = None,
    device: Optional[object] = None,
    model_path: Optional[object] = None,
    graph_hash: Optional[str] = None,
    explanation_methods: Optional[Sequence[str]] = None,
    explain_max_nodes: int = 0,
    explain_top_features: int = 0,
    integrated_gradients_steps: int = 16,
    feature_names: Optional[Sequence[str]] = None,
) -> GNNXAIResult:
    """Compute graph XAI attention joined with node-level predictions.

    The model forward path is used directly, so variants that encode edge
    attributes (for example HeteroGATWithEdgeEncoder) analyze the encoded
    attributes consumed by attention rather than raw edge attributes.
    """

    if NeighborLoader is None:
        raise ImportError("torch_geometric NeighborLoader no está disponible.")
    if node_type not in graph.node_types:
        raise ValueError(f"El grafo no contiene node_type={node_type!r}.")

    target_indices = _resolve_node_indices(
        graph,
        node_type,
        node_indices=node_indices,
        mask_name=mask_name,
    )
    target_indices = target_indices.detach().cpu().long().view(-1)
    if target_indices.numel() == 0:
        raise ValueError(f"No hay nodos para calcular XAI en mask={mask_name!r}.")

    model_device = _infer_device(model, device)
    model.to(model_device)
    model.eval()
    if temporal_module is not None:
        temporal_module.to(model_device)
        temporal_module.eval()

    graph_for_loader = graph
    try:
        sample_tensor = next(iter(graph.x_dict.values()))
        if getattr(sample_tensor, "device", torch.device("cpu")).type != "cpu":
            graph_for_loader = graph.cpu()
    except Exception:
        try:
            graph_for_loader = graph.cpu()
        except Exception:
            graph_for_loader = graph

    edge_types = [tuple(et) for et in getattr(graph_for_loader, "edge_types", [])]
    num_layers = int(getattr(model, "num_layers", 1) or 1)
    resolved_neighbors = _resolve_num_neighbors(
        num_neighbors,
        edge_types,
        num_layers=num_layers,
    )
    effective_batch_size = int(batch_size or BATCH_SIZE or 256)
    effective_percentile = min(max(float(percentile), 0.0), 1.0)
    effective_k_heads = max(1, int(k_heads or 1))

    loader = NeighborLoader(
        graph_for_loader,
        input_nodes=(node_type, target_indices),
        num_neighbors=resolved_neighbors,
        batch_size=effective_batch_size,
        shuffle=False,
    )

    node_rows_by_id: Dict[int, Dict[str, Any]] = {}
    edge_rows: List[Dict[str, Any]] = []
    explanation_rows: List[Dict[str, Any]] = []
    methods = tuple(str(m).strip().lower() for m in (explanation_methods or []) if str(m).strip())
    remaining_explain_nodes = max(0, int(explain_max_nodes or 0))

    with torch.no_grad():
        for batch in loader:
            target_bs = int(getattr(batch[node_type], "batch_size", 0) or 0)
            if target_bs <= 0:
                continue
            batch = batch.to(model_device)
            edge_attr_dict = _as_edge_attr_dict(batch)
            logits_dict, embeddings_dict, attentions = _call_model(model, batch, edge_attr_dict)
            node_n_id = getattr(batch[node_type], "n_id").detach().cpu()
            logits_all = logits_dict.get(node_type)
            if logits_all is None and embeddings_dict:
                logits_all = embeddings_dict.get(node_type)
            if logits_all is None:
                raise RuntimeError(f"El modelo no devolvió logits para node_type={node_type!r}.")

            if temporal_module is not None:
                logits_target = temporal_module(
                    embeddings_dict[node_type],
                    node_n_id.to(model_device),
                    target_bs,
                )
                global_ids = node_n_id[:target_bs]
                y_target = getattr(batch[node_type], "y", None)
                if y_target is not None:
                    y_target = y_target[:target_bs]
                rows = _predict_batch_nodes(
                    logits_target,
                    y_target,
                    global_ids,
                    pm_index=pm_index,
                    data=graph_for_loader,
                    node_type=node_type,
                    mask_name=mask_name,
                    batch_size=target_bs,
                    threshold=threshold,
                    calibration_model=calibration_model,
                )
            else:
                y_all = getattr(batch[node_type], "y", None)
                rows = _predict_batch_nodes(
                    logits_all,
                    y_all,
                    node_n_id,
                    pm_index=pm_index,
                    data=graph_for_loader,
                    node_type=node_type,
                    mask_name=mask_name,
                    batch_size=target_bs,
                    threshold=threshold,
                    calibration_model=calibration_model,
                )
            for row in rows:
                node_rows_by_id[int(row["node_idx"])] = row

            edge_rows.extend(
                _attention_edges_for_batch(
                    model=model,
                    attentions=attentions,
                    batch=batch,
                    layer_filter=layer,
                    percentile=effective_percentile,
                    k_heads=effective_k_heads,
                )
            )
            if methods and remaining_explain_nodes > 0:
                explain_rows = _batch_advanced_explanations(
                    model=model,
                    batch=batch,
                    edge_attr_dict=edge_attr_dict,
                    logits_dict=logits_dict,
                    embeddings_dict=embeddings_dict,
                    temporal_module=temporal_module,
                    node_type=node_type,
                    node_rows=rows[:target_bs],
                    target_bs=target_bs,
                    threshold=threshold,
                    calibration_model=calibration_model,
                    explanation_methods=methods,
                    feature_names=feature_names,
                    top_features=int(explain_top_features or 0),
                    explain_limit=remaining_explain_nodes,
                    integrated_gradients_steps=int(integrated_gradients_steps or 1),
                )
                explanation_rows.extend(explain_rows)
                explained_ids = {
                    int(row["node_idx"])
                    for row in explain_rows
                    if row.get("node_idx") is not None
                }
                remaining_explain_nodes = max(0, remaining_explain_nodes - len(explained_ids))

    nodes_df = pd.DataFrame(list(node_rows_by_id.values()))
    if nodes_df.empty:
        nodes_df = pd.DataFrame(
            columns=["node_idx", "portico", "ts_min", "mask", "y_true", "y_pred", "prob1", "error_type"]
        )
    else:
        nodes_df = nodes_df.sort_values("node_idx").reset_index(drop=True)

    edges_df = pd.DataFrame(edge_rows)
    if edges_df.empty:
        edges_df = pd.DataFrame(
            columns=[
                "layer",
                "relation",
                "source_node_idx",
                "dest_node_idx",
                "mean_attention",
                "max_attention",
                "heads_above_threshold",
                "head_count",
                "attention_threshold",
            ]
        )
    else:
        edges_df = _deduplicate_edges(edges_df)
    edges_df = _join_predictions(edges_df, nodes_df)
    summary_df = _summary(edges_df)
    explanations_df = pd.DataFrame(explanation_rows)
    if explanations_df.empty:
        explanations_df = pd.DataFrame(
            columns=[
                "method",
                "node_idx",
                "portico",
                "ts_min",
                "mask",
                "y_true",
                "y_pred",
                "error_type",
                "relation",
                "feature_idx",
                "feature_name",
                "base_prob1",
                "perturbed_prob1",
                "sufficient_prob1",
                "delta_prob1",
                "score",
                "necessity",
                "sufficiency",
                "threshold",
                "crossed_threshold",
                "attribution",
                "target_class",
                "gradient_steps",
            ]
        )
    else:
        explanations_df = explanations_df.sort_values("score", ascending=False).reset_index(drop=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    manifest = {
        "schema_version": "2.0",
        "created_at": _utc_now(),
        "run_id": run_id,
        "model_path": str(model_path) if model_path is not None else None,
        "graph_hash": graph_hash,
        "node_type": node_type,
        "mask_name": str(mask_name),
        "node_count": int(len(nodes_df)),
        "edge_count": int(len(edges_df)),
        "explanation_count": int(len(explanations_df)),
        "batch_size": effective_batch_size,
        "num_neighbors": str(resolved_neighbors),
        "percentile": effective_percentile,
        "k_heads": effective_k_heads,
        "layer": str(layer),
        "threshold": None if threshold is None else float(threshold),
        "explanation_methods": list(methods),
        "explain_max_nodes": int(explain_max_nodes or 0),
        "explain_top_features": int(explain_top_features or 0),
        "integrated_gradients_steps": int(integrated_gradients_steps or 1),
        "seed": int(SEED),
        "files": {
            "manifest": "manifest.json",
            "nodes": "nodes.parquet",
            "edges": "edges.parquet",
            "explanations": "explanations.parquet",
            "summary": "summary.csv",
        },
    }
    return GNNXAIResult(
        nodes_df=nodes_df,
        edges_df=edges_df,
        summary_df=summary_df,
        manifest=manifest,
        explanations_df=explanations_df,
    )


def _write_parquet_with_csv_fallback(df: pd.DataFrame, parquet_path: Path) -> str:
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path.name
    except Exception:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path.name


def save_gnn_xai_result(
    result: GNNXAIResult,
    output_dir: Optional[object] = None,
) -> Path:
    base = (
        Path(output_dir)
        if output_dir is not None
        else gnn_dir("xai", RESULTADOS_DIR, create=True) / str(result.manifest.get("run_id"))
    )
    base.mkdir(parents=True, exist_ok=True)
    manifest = dict(result.manifest)
    files = dict(manifest.get("files") or {})
    files["nodes"] = _write_parquet_with_csv_fallback(result.nodes_df, base / "nodes.parquet")
    files["edges"] = _write_parquet_with_csv_fallback(result.edges_df, base / "edges.parquet")
    explanations_df = result.explanations_df if result.explanations_df is not None else pd.DataFrame()
    files["explanations"] = _write_parquet_with_csv_fallback(explanations_df, base / "explanations.parquet")
    result.summary_df.to_csv(base / "summary.csv", index=False)
    files["summary"] = "summary.csv"
    files["manifest"] = "manifest.json"
    manifest["files"] = files
    manifest["output_dir"] = str(base)
    with open(base / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
    result.manifest.clear()
    result.manifest.update(manifest)
    return base


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_gnn_xai_result(result_dir: object) -> GNNXAIResult:
    base = Path(result_dir)
    with open(base / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    files = dict(manifest.get("files") or {})
    nodes_name = files.get("nodes", "nodes.parquet")
    edges_name = files.get("edges", "edges.parquet")
    explanations_name = files.get("explanations")
    summary_name = files.get("summary", "summary.csv")
    nodes_df = _read_table(base / nodes_name)
    edges_df = _read_table(base / edges_name)
    explanations_df = (
        _read_table(base / explanations_name)
        if explanations_name and (base / explanations_name).exists()
        else pd.DataFrame()
    )
    summary_df = _read_table(base / summary_name)
    return GNNXAIResult(
        nodes_df=nodes_df,
        edges_df=edges_df,
        summary_df=summary_df,
        manifest=manifest,
        explanations_df=explanations_df,
    )


def find_latest_gnn_xai_result(
    *,
    base_dir: Optional[object] = None,
    graph_hash: Optional[str] = None,
    model_path: Optional[object] = None,
) -> Optional[Path]:
    root = Path(base_dir) if base_dir is not None else gnn_dir("xai", RESULTADOS_DIR)
    if not root.exists():
        return None
    model_path_str = str(model_path) if model_path is not None else None
    candidates: List[Tuple[float, Path]] = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception:
            continue
        if graph_hash and manifest.get("graph_hash") not in (None, graph_hash):
            continue
        if model_path_str and manifest.get("model_path") not in (None, model_path_str):
            continue
        try:
            mtime = os.path.getmtime(manifest_path)
        except Exception:
            mtime = 0.0
        candidates.append((mtime, manifest_path.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]
