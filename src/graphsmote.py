import os
from typing import Optional, Union, List, Tuple, Dict, Any
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch_geometric.utils import remove_self_loops, coalesce
from torch_geometric.loader import NeighborLoader
from src.config import (
    SEED,
    RESULTADOS_DIR,
    BATCH_SIZE,
    NUM_NEIGHBORS,
    EMB_NUM_NEIGHBORS,
    EMB_BATCH_SIZE,
    GRAPHSMOTE_K,
    DEBUG
)
from src.config import BIDIRECCTION, GRAPHSMOTE_CONECT
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("[GraphSMOTE] %s inválido; usando %.3f.", name, default)
        return float(default)

GRAPHSMOTE_SYNTHETIC_FEATURE_MODE = os.environ.get(
    "GRAPHSMOTE_SYNTHETIC_FEATURE_MODE",
    "z2x",
).strip().lower()
GRAPHSMOTE_Z2X_MINORITY_WEIGHT = _env_float("GRAPHSMOTE_Z2X_MINORITY_WEIGHT", 1.0)
GRAPHSMOTE_Z2X_EARLY_STOP_METRIC = os.environ.get(
    "GRAPHSMOTE_Z2X_EARLY_STOP_METRIC",
    "val_loss",
).strip().lower()

_SYNTHETIC_FEATURE_MODES = {"z2x", "feature_interp", "oracle_copy"}

# ============================================================
# Limpieza de aristas sin romper edge_attr
# ============================================================
def _safe_clean_edges(store, num_src, num_dst):
    """
    Aplica remove_self_loops y coalesce de forma segura.
    Intenta conservar edge_attr durante coalesce. Si coalesce lo elimina
    (porque no se puede agregar/reducir), se elimina del store y se loguea un warning.
    """
    ei = store.edge_index
    ea = getattr(store, 'edge_attr', None)

    # Sanidad básica
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError("edge_index debe tener shape [2, E].")

    E = ei.size(1)
    original_ea_is_not_none = ea is not None
    
    if original_ea_is_not_none:
        # Acepta [E] o [E, d] como válidos
        if (ea.dim() == 1 and E != ea.size(0)) or (ea.dim() >= 2 and E != ea.size(0)):
            logger.warning(f"edge_attr for {store._key} is misaligned (shape {ea.shape} vs {E} edges) and will be ignored during coalesce.")
            ea = None

    ei, ea = remove_self_loops(ei, ea)
    ei, ea_after_coalesce = coalesce(ei, ea, num_nodes=max(num_src, num_dst), reduce='mean')

    store.edge_index = ei
    if ea_after_coalesce is not None:
        store.edge_attr = ea_after_coalesce
    elif original_ea_is_not_none and hasattr(store, 'edge_attr'):
        logger.warning(
            f"edge_attr was removed from store for edge_type {store._key} "
            "because it could not be coalesced. This can happen if the attributes "
            "of duplicate edges are not averageable (e.g., they are not floats)."
        )
        delattr(store, 'edge_attr')

def _resolve_delta_feature_idx(data_or_store, device=None):
    """Resolve delta feature indices for edge_attr deltas."""
    idx = getattr(data_or_store, "delta_feature_idx", None)
    if idx is None:
        return None
    if not torch.is_tensor(idx):
        try:
            idx = torch.tensor(idx, dtype=torch.long)
        except Exception:
            return None
    if idx.numel() == 0:
        return None
    if device is not None:
        try:
            idx = idx.to(device)
        except Exception:
            pass
    return idx

def _delta_from_node_features(src_x: torch.Tensor, dst_x: torch.Tensor, delta_idx: Optional[torch.Tensor]):
    """Compute dst_x - src_x using delta feature indices if provided."""
    if src_x.dim() == 1:
        src_x = src_x.view(-1, 1)
    if dst_x.dim() == 1:
        dst_x = dst_x.view(-1, 1)
    if delta_idx is None:
        return dst_x - src_x
    try:
        if delta_idx.device != src_x.device:
            delta_idx = delta_idx.to(src_x.device)
        if delta_idx.dim() == 0:
            delta_idx = delta_idx.view(1)
        max_dim = src_x.size(1)
        delta_idx = delta_idx[(delta_idx >= 0) & (delta_idx < max_dim)]
        if delta_idx.numel() == 0:
            return dst_x - src_x
        return dst_x.index_select(1, delta_idx) - src_x.index_select(1, delta_idx)
    except Exception:
        return dst_x - src_x

def _resolve_synthetic_feature_mode(mode: Optional[str]) -> str:
    resolved = str(mode or GRAPHSMOTE_SYNTHETIC_FEATURE_MODE or "z2x").strip().lower()
    if resolved not in _SYNTHETIC_FEATURE_MODES:
        logger.warning(
            "[GraphSMOTE] synthetic_feature_mode=%r no soportado; usando 'z2x'.",
            resolved,
        )
        resolved = "z2x"
    return resolved

def _as_feature_matrix(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1, 1) if x.dim() == 1 else x

def _safe_quantile_float(values: torch.Tensor, q: float) -> Optional[float]:
    if values is None or values.numel() == 0:
        return None
    return float(torch.quantile(values.detach().cpu().float(), float(q)).item())

def _feature_manifold_quality(
    x_syn: torch.Tensor,
    x_pos_reference: torch.Tensor,
    x_train_reference: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Quality diagnostics for synthetic node features against real train positives."""
    quality: Dict[str, Any] = {}
    x_syn = _as_feature_matrix(x_syn.detach().cpu().float())
    x_pos = _as_feature_matrix(x_pos_reference.detach().cpu().float())
    x_train = (
        _as_feature_matrix(x_train_reference.detach().cpu().float())
        if x_train_reference is not None and x_train_reference.numel() > 0
        else x_pos
    )
    if x_syn.numel() == 0 or x_pos.numel() == 0:
        return quality

    eps = 1e-6
    train_mean = x_train.mean(dim=0)
    train_std = x_train.std(dim=0).clamp_min(eps)
    z_syn = (x_syn - train_mean) / train_std
    z_pos = (x_pos - train_mean) / train_std

    pos_min = x_pos.min(dim=0).values
    pos_max = x_pos.max(dim=0).values
    train_min = x_train.min(dim=0).values
    train_max = x_train.max(dim=0).values
    outside_pos = (x_syn < pos_min) | (x_syn > pos_max)
    outside_train = (x_syn < train_min) | (x_syn > train_max)

    l2_to_pos = torch.cdist(z_syn, z_pos, p=2.0).min(dim=1).values
    syn_norm = F.normalize(z_syn, p=2, dim=1, eps=eps)
    pos_norm = F.normalize(z_pos, p=2, dim=1, eps=eps)
    cosine_dist = (1.0 - (syn_norm @ pos_norm.T).max(dim=1).values).clamp_min(0.0)

    quality.update(
        {
            "synthetic_count": int(x_syn.shape[0]),
            "train_positive_count": int(x_pos.shape[0]),
            "nan_count": int(torch.isnan(x_syn).sum().item()),
            "inf_count": int(torch.isinf(x_syn).sum().item()),
            "feature_outside_train_minmax_frac": float(outside_train.float().mean().item()),
            "feature_outside_train_positive_minmax_frac": float(outside_pos.float().mean().item()),
            "min_l2_to_train_positive_mean": float(l2_to_pos.mean().item()),
            "min_l2_to_train_positive_median": _safe_quantile_float(l2_to_pos, 0.50),
            "min_l2_to_train_positive_p95": _safe_quantile_float(l2_to_pos, 0.95),
            "min_cosine_distance_to_train_positive_mean": float(cosine_dist.mean().item()),
            "min_cosine_distance_to_train_positive_p95": _safe_quantile_float(cosine_dist, 0.95),
        }
    )

    if x_pos.shape[0] >= 2:
        pos_l2 = torch.cdist(z_pos, z_pos, p=2.0)
        diag = torch.eye(pos_l2.shape[0], dtype=torch.bool)
        pos_l2 = pos_l2.masked_fill(diag, float("inf"))
        loo_l2 = pos_l2.min(dim=1).values

        pos_cos = pos_norm @ pos_norm.T
        pos_cos = pos_cos.masked_fill(diag, float("-inf"))
        loo_cos = (1.0 - pos_cos.max(dim=1).values).clamp_min(0.0)

        ref_l2_p95 = _safe_quantile_float(loo_l2, 0.95)
        ref_cos_p95 = _safe_quantile_float(loo_cos, 0.95)
        syn_l2_p95 = quality["min_l2_to_train_positive_p95"]
        syn_cos_p95 = quality["min_cosine_distance_to_train_positive_p95"]
        quality.update(
            {
                "real_positive_loo_l2_p95": ref_l2_p95,
                "real_positive_loo_cosine_p95": ref_cos_p95,
                "synthetic_l2_p95_to_real_positive_loo_p95_ratio": (
                    float(syn_l2_p95 / max(ref_l2_p95, eps))
                    if syn_l2_p95 is not None and ref_l2_p95 is not None
                    else None
                ),
                "synthetic_cosine_p95_to_real_positive_loo_p95_ratio": (
                    float(syn_cos_p95 / max(ref_cos_p95, eps))
                    if syn_cos_p95 is not None and ref_cos_p95 is not None
                    else None
                ),
            }
        )
    l2_ratio = quality.get("synthetic_l2_p95_to_real_positive_loo_p95_ratio")
    cosine_ratio = quality.get("synthetic_cosine_p95_to_real_positive_loo_p95_ratio")
    quality["feature_manifold_gate_ok"] = bool(
        quality["nan_count"] == 0
        and quality["inf_count"] == 0
        and quality["feature_outside_train_positive_minmax_frac"] == 0.0
        and (l2_ratio is None or l2_ratio <= 1.25)
        and (cosine_ratio is None or cosine_ratio <= 1.25)
    )
    quality["feature_manifold_gate_policy"] = (
        "finite and inside train-positive minmax; synthetic p95 nearest-positive "
        "<= 1.25x real-positive leave-one-out p95 when available"
    )
    return quality

def _sanitize_synthetic_features(
    x_syn: torch.Tensor,
    x_reference: torch.Tensor,
) -> torch.Tensor:
    """Replace non-finite synthetic values and clamp to the real positive train range."""
    x_syn = _as_feature_matrix(x_syn)
    x_ref = _as_feature_matrix(x_reference.to(device=x_syn.device, dtype=x_syn.dtype))
    if x_ref.numel() == 0:
        return torch.nan_to_num(x_syn)

    ref_mean = x_ref.mean(dim=0)
    finite = torch.isfinite(x_syn)
    x_syn = torch.where(finite, x_syn, ref_mean.unsqueeze(0).expand_as(x_syn))

    ref_min = x_ref.min(dim=0).values
    ref_max = x_ref.max(dim=0).values
    return torch.maximum(torch.minimum(x_syn, ref_max), ref_min)

def _project_synthetic_features_from_smote(
    *,
    node_type: str,
    syn_z: torch.Tensor,
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    parent_info: Optional[Dict[str, torch.Tensor]],
    minority_class: int,
    z2x_decoders,
    feature_mode: Optional[str],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create synthetic node features using the requested source:
    z2x decoder, feature-space interpolation from SMOTE parents, or oracle copy.
    """
    mode = _resolve_synthetic_feature_mode(feature_mode)
    x_source = _as_feature_matrix(x_source.to(device=syn_z.device, dtype=torch.float32))
    y_source = y_source.to(device=syn_z.device)
    pos_source = x_source[y_source == int(minority_class)]
    if pos_source.numel() == 0:
        pos_source = x_source

    if mode == "z2x":
        if z2x_decoders is None:
            raise ValueError("synthetic_feature_mode='z2x' requiere z2x_decoders.")
        with torch.no_grad():
            x_syn = z2x_decoders.project_one(node_type, syn_z).to(dtype=x_source.dtype)
    else:
        if parent_info is None:
            raise ValueError(f"synthetic_feature_mode='{mode}' requiere parent_info de SMOTE.")
        base_idx = parent_info["base_idx"].to(device=syn_z.device).long()
        neighbor_idx = parent_info["neighbor_idx"].to(device=syn_z.device).long()
        alpha = parent_info["alpha"].to(device=syn_z.device, dtype=x_source.dtype).view(-1, 1)
        x_a = x_source.index_select(0, base_idx)
        if mode == "oracle_copy":
            x_syn = x_a.clone()
        else:
            x_b = x_source.index_select(0, neighbor_idx)
            x_syn = (1.0 - alpha) * x_a + alpha * x_b

    x_syn = _sanitize_synthetic_features(x_syn, pos_source)
    quality = _feature_manifold_quality(x_syn, pos_source, x_source)
    quality["synthetic_feature_mode"] = mode
    return x_syn, quality

# ============================================================
# 1) Edge generator bilineal por relación
# ============================================================
class RelEdgeGen(torch.nn.Module):
    def __init__(self, z_dim_dict, rels):
        super().__init__()
        # un S_r por relación (src_type, rel, dst_type)
        self.S = torch.nn.ParameterDict({
            f"{src}:{rel}:{dst}": torch.nn.Parameter(
                torch.empty(z_dim_dict[src], z_dim_dict[dst]).normal_(0, 0.02)
            )
            for (src, rel, dst) in rels
        })

    def score(self, z_src, z_dst, key):  # z_*: [N, d]
        S = self.S[key]
        return (z_src @ S @ z_dst.t())   # [N_src, N_dst]

    def predict(
        self,
        z_src,
        z_dst,
        key,
        topk=None,
        tau=None,
        batch_size=BATCH_SIZE,
        force_cpu: Optional[bool] = None,
    ):
        if topk is None:
            # Original behavior for non-topk cases (might still cause OOM if z_src is large)
            logits = self.score(z_src, z_dst, key)
            if tau is not None:
                return torch.sigmoid(logits / tau)
            return torch.sigmoid(logits)

        # Batched top-k prediction to prevent OOM
        num_src = z_src.size(0)
        num_dst = z_dst.size(0)
        
        device = z_src.device
        is_cuda = device.type == 'cuda'
        
        if force_cpu is None:
            compute_device = 'cpu' if 'mps' in str(device) else device
        else:
            compute_device = torch.device('cpu') if force_cpu else device
        
        z_src_compute = z_src.to(compute_device)
        z_dst_compute = z_dst.to(compute_device)
        s_compute = self.S[key].to(compute_device)

        final_top_indices = torch.full((num_src, topk), -1, dtype=torch.long, device=device)

        for i in tqdm(range(0, num_src, batch_size), desc="Predicting edges in batches (src)"):
            batch_z_src = z_src_compute[i:i+batch_size]
            
            batch_top_scores = torch.full((batch_z_src.size(0), topk), -float('inf'), device=compute_device)
            batch_top_indices = torch.full((batch_z_src.size(0), topk), -1, dtype=torch.long, device=compute_device)

            for j in range(0, num_dst, batch_size):
                batch_z_dst = z_dst_compute[j:j+batch_size]
                
                logits = (batch_z_src @ s_compute @ batch_z_dst.t())
                
                k_candidates = min(topk, logits.size(1))
                if k_candidates == 0:
                    del logits
                    continue

                top_scores_new, top_indices_new_relative = logits.topk(k=k_candidates, dim=1)

                combined_scores = torch.cat([batch_top_scores, top_scores_new], dim=1)
                
                top_indices_new_absolute = top_indices_new_relative + j
                combined_indices = torch.cat([batch_top_indices, top_indices_new_absolute], dim=1)

                _, top_indices_in_combined = torch.topk(combined_scores, k=topk, dim=1)
                
                batch_top_scores = torch.gather(combined_scores, 1, top_indices_in_combined)
                batch_top_indices = torch.gather(combined_indices, 1, top_indices_in_combined)
                
                del top_scores_new, top_indices_new_relative, combined_scores, top_indices_new_absolute, combined_indices, top_indices_in_combined, logits

            final_top_indices[i:i+batch_z_src.size(0)] = batch_top_indices.to(device)
            del batch_z_src, batch_top_scores, batch_top_indices
            if is_cuda:
                torch.cuda.empty_cache()
        
        return final_top_indices

# ============================================================
# 2) Extracción de embeddings
# ============================================================
def _normalize_num_neighbors(num_neighbors, data):
    edge_types = getattr(data, "edge_types", None)

    def _profile(val):
        if isinstance(val, (int, float)):
            return [int(val)]
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        return None

    if edge_types is None:
        return _profile(num_neighbors) or [1]

    if isinstance(num_neighbors, dict):
        default_profile = None
        for v in num_neighbors.values():
            prof = _profile(v)
            if prof:
                default_profile = prof
                break
        if default_profile is None:
            default_profile = [1]
        return {
            edge_type: _profile(num_neighbors.get(edge_type)) or default_profile
            for edge_type in edge_types
        }

    profile = _profile(num_neighbors) or [1]
    return {edge_type: profile for edge_type in edge_types}


def compute_epoch_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        edge_attr_dict = getattr(data, "edge_attr_dict", {})
        _, z_dict, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    return z_dict

@torch.no_grad()
def get_embeddings_minibatch(
    model,
    data,
    *,
    num_neighbors=NUM_NEIGHBORS,
    store_on_cpu: bool = True,
):
    model.eval()
    device = next(model.parameters()).device
    data_cpu = data.cpu()
    num_neighbors = _normalize_num_neighbors(num_neighbors, data_cpu)
    z_dict = {node_type: [] for node_type in data.node_types}

    for node_type in data.node_types:
        # Skip node types with no nodes
        if data_cpu[node_type].num_nodes == 0:
            continue

        loader = NeighborLoader(
            data_cpu,
            input_nodes=(node_type, torch.arange(data_cpu[node_type].num_nodes)),
            num_neighbors=num_neighbors,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        
        for batch in tqdm(loader, desc=f"Generating embeddings for node type {node_type}"):
            batch = batch.to(device)
            edge_attr_dict = {
                et: batch[et].edge_attr
                for et in batch.edge_types
                if 'edge_attr' in batch[et]
            }
            
            _, batch_z_dict, _ = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
            
            if node_type in batch_z_dict:
                # We are only interested in the embeddings of the input nodes of the batch, not the neighbors
                z_batch = batch_z_dict[node_type][:batch[node_type].batch_size]
                if store_on_cpu:
                    z_batch = z_batch.cpu()
                z_dict[node_type].append(z_batch)
                if DEBUG:
                    logger.info(f"DEBUG: get_embeddings_minibatch - batch_z_dict[{node_type}].shape={batch_z_dict[node_type].shape}")
                

    for node_type in data.node_types:
        if z_dict[node_type]:
            z_dict[node_type] = torch.cat(z_dict[node_type], dim=0)
            z_dict[node_type] = F.normalize(z_dict[node_type], p=2, dim=1, eps=1e-12)
        else:
            # If no embeddings were generated, create an empty tensor with the correct shape
            out_channels = model.out_channels if hasattr(model, 'out_channels') else (model.hidden_channels * model.num_heads)
            empty_device = torch.device("cpu") if store_on_cpu else device
            z_dict[node_type] = torch.empty(0, out_channels, device=empty_device)
            
    return z_dict

@torch.no_grad()
def get_embeddings(model, data):
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    edge_attr_dict = getattr(data, "edge_attr_dict", {})
    _, z_dict, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)

    for ntype in z_dict:
        z_dict[ntype] = F.normalize(z_dict[ntype], p=2, dim=1, eps=1e-12)

    return z_dict


# ============================================================
# Probes de validación de los modelos internos de GraphSMOTE
# ============================================================
@torch.no_grad()
def evaluate_edge_generator_auc(
    edge_gen,
    z_dict: Dict[str, torch.Tensor],
    edge_index_dict,
    *,
    holdout_frac: float = 0.1,
    num_neg: int = 1,
    seed: int = SEED,
    min_edges: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    AUC-ROC y AP de link prediction por relación para el `RelEdgeGen`.

    Para cada relación con >= ``min_edges`` aristas: toma una fracción holdout
    de positivos, samplea destinos aleatorios como negativos (colisión con
    aristas reales tiene probabilidad despreciable a las densidades típicas),
    aplica el score bilineal `z_src^T S_r z_dst` y calcula AUC/AP.

    Devuelve `{'src:rel:dst': {'auc','ap','n_pos','n_neg'}}` y una entrada
    `'_macro'` con el promedio. Si no hay relaciones evaluables, dict vacío.
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except Exception as exc:
        logger.warning(f"[EdgeGen eval] sklearn no disponible: {exc}")
        return {}

    rng = np.random.RandomState(int(seed))
    out: Dict[str, Dict[str, float]] = {}
    aucs, aps = [], []

    for et, ei in edge_index_dict.items():
        if not isinstance(et, tuple) or len(et) != 3:
            continue
        src_t, rel, dst_t = et
        if src_t not in z_dict or dst_t not in z_dict:
            continue
        zs = z_dict[src_t]
        zd = z_dict[dst_t]
        if zs.numel() == 0 or zd.numel() == 0:
            continue
        E = ei.size(1)
        if E < int(min_edges):
            continue
        key = f"{src_t}:{rel}:{dst_t}"
        if not hasattr(edge_gen, 'S') or key not in edge_gen.S:
            continue

        n_holdout = max(1, int(round(E * float(holdout_frac))))
        perm = rng.permutation(E)[:n_holdout]
        device = zs.device
        ei_dev = ei.to(device)

        pos_src = ei_dev[0, perm]
        pos_dst = ei_dev[1, perm]

        n_neg_total = n_holdout * int(num_neg)
        neg_src = torch.from_numpy(
            rng.randint(0, zs.size(0), size=n_neg_total)
        ).long().to(device)
        neg_dst = torch.from_numpy(
            rng.randint(0, zd.size(0), size=n_neg_total)
        ).long().to(device)

        S = edge_gen.S[key].to(device)
        pos_logits = torch.sum((zs.index_select(0, pos_src) @ S) * zd.index_select(0, pos_dst), dim=1)
        neg_logits = torch.sum((zs.index_select(0, neg_src) @ S) * zd.index_select(0, neg_dst), dim=1)

        y_true = np.concatenate(
            [np.ones(pos_logits.numel(), dtype=np.int64),
             np.zeros(neg_logits.numel(), dtype=np.int64)]
        )
        y_score = np.concatenate(
            [pos_logits.detach().cpu().numpy(), neg_logits.detach().cpu().numpy()]
        )

        try:
            auc = float(roc_auc_score(y_true, y_score))
            ap = float(average_precision_score(y_true, y_score))
        except Exception as exc:
            logger.warning(f"[EdgeGen eval] {key}: no se pudo calcular AUC ({exc})")
            continue

        out[key] = {
            'auc': auc,
            'ap': ap,
            'n_pos': int(pos_logits.numel()),
            'n_neg': int(neg_logits.numel()),
        }
        aucs.append(auc)
        aps.append(ap)

    if aucs:
        out['_macro'] = {
            'auc': float(sum(aucs) / len(aucs)),
            'ap': float(sum(aps) / len(aps)),
            'n_relations': len(aucs),
        }
    return out


def evaluate_embedding_separability(
    z_dict: Dict[str, torch.Tensor],
    data,
    node_type: str = 'pm',
    *,
    seed: int = SEED,
    max_iter: int = 1000,
) -> Optional[Dict[str, float]]:
    """
    Sondea la separabilidad lineal del encoder. Ajusta una LogisticRegression
    sobre embeddings de TRAIN y reporta F1 (macro y minoritaria), AUPRC y
    ROC-AUC sobre VAL. Si los embeddings codifican señal de la clase, este
    probe es alto incluso sin la cabeza GAT — útil para aislar si el cuello de
    botella está en el encoder o en el clasificador.

    Devuelve None si faltan máscaras, si no hay dos clases en train+val o si
    el `z_dict[node_type]` no está alineado con todos los nodos del tipo.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
    except Exception as exc:
        logger.warning(f"[Embed probe] sklearn no disponible: {exc}")
        return None

    if node_type not in z_dict:
        return None
    z_tensor = z_dict[node_type]
    if z_tensor is None or z_tensor.numel() == 0:
        return None
    z = z_tensor.detach().cpu().to(torch.float32).numpy()

    store = data[node_type]
    if 'y' not in store or 'train_mask' not in store or 'val_mask' not in store:
        return None
    y = store.y.cpu().numpy().astype(int)
    train_mask = store.train_mask.cpu().numpy().astype(bool)
    val_mask = store.val_mask.cpu().numpy().astype(bool)

    if z.shape[0] != y.shape[0]:
        # z viene filtrado a TRAIN (p.ej. compute_train_embeddings); este probe
        # requiere alineación completa para usar val_mask. Avisamos y salimos.
        logger.info(
            f"[Embed probe] z[{node_type}]={z.shape[0]} no coincide con y={y.shape[0]}; "
            "el probe necesita embeddings de todos los nodos."
        )
        return None

    z_tr, y_tr = z[train_mask], y[train_mask]
    z_va, y_va = z[val_mask], y[val_mask]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return None
    if z_tr.shape[0] < 5 or z_va.shape[0] < 5:
        return None

    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    clf.fit(z_tr, y_tr)
    proba_va = clf.predict_proba(z_va)[:, 1]
    pred = (proba_va >= 0.5).astype(int)

    return {
        'val_f1_macro': float(f1_score(y_va, pred, average='macro', zero_division=0)),
        'val_f1_minority': float(f1_score(y_va, pred, pos_label=1, zero_division=0)),
        'val_auprc': float(average_precision_score(y_va, proba_va)),
        'val_roc_auc': float(roc_auc_score(y_va, proba_va)),
        'n_train': int(z_tr.shape[0]),
        'n_val': int(z_va.shape[0]),
    }


# ============================================================
# 3) Decodificadores z->x por tipo (entrenados una sola vez)
# ============================================================
class Z2XDecoders(torch.nn.Module):
    """
    Un MLP por tipo de nodo: toma z (embedding) y predice x (feature original).
    """
    def __init__(self, dims_per_type):  # dict: ntype -> (z_dim, x_dim)
        super().__init__()
        mods = {}
        for ntype, (z_dim, x_dim) in dims_per_type.items():
            hidden = max(64, 2 * min(z_dim, x_dim))
            mods[ntype] = torch.nn.Sequential(
                torch.nn.Linear(z_dim, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, x_dim)
            )
        self.heads = torch.nn.ModuleDict(mods)
        self._loaded_types = set()  # se completa en load/train

    def forward(self, z_dict):
        out = {}
        for ntype, z in z_dict.items():
            if ntype in self.heads:
                out[ntype] = self.heads[ntype](z)
        return out

    def project_one(self, ntype, z_batch):
        return self.heads[ntype](z_batch)

def train_z2x_decoders(
    model,
    data,
    node_types=None,
    lr: float = 1e-3,
    epochs: int = 50,
    loss_type: str = "huber",
    weight_decay: float = 1e-4,
    device=None,
    use_masks: bool = True,
    save_dir: str = os.path.join(RESULTADOS_DIR, "z2x_decoders"),
    early_stop: bool = True,
    patience: int = 3,
    min_delta: float = 1e-4,
    val_split: float = 0.1,
    log_every: int = 1,
    show_progress: bool = True, #False para mostrar resultados de cada epoch de entrenamiento en la consola
    num_neighbors=NUM_NEIGHBORS,
    progress_callback: Optional[callable] = None,
    *,
    minority_classes: Optional[Dict[str, int]] = None,
    use_mixup: bool = True,
    mixup_k: int = 5,
    mixup_ratio: float = 1.0,
    mixup_weight: float = 1.0,
    minority_recon_weight: Optional[float] = None,
    early_stop_metric: Optional[str] = None,
):
    """
    Entrena un decodificador z->x por tipo usando z extraído en modo eval.

    Correcciones respecto al diseño anterior:
    - **Excluye nodos sintéticos previos** (`is_synthetic`) del set de entrenamiento
      y validación para evitar que el decoder aprenda de sus propias predicciones.
    - **Augmentación tipo mixup** sobre vecinos minoritarios: en cada época se
      construyen pares interpolados `(z_mix, x_mix)` con `α~U[0,1]` y se entrena
      al decoder explícitamente sobre puntos del tipo que generará SMOTE. El
      `z_mix` se L2-normaliza para mantener consistencia con la entrada que el
      decoder verá en inferencia.
    - **Validación desagregada por clase** (mayoritaria vs. minoritaria) para
      diagnosticar la calidad de la reconstrucción donde más importa.

    `minority_classes` es un dict `{ntype: clase_minoritaria}`; por defecto se
    asume `{'pm': 1}` (convención del pipeline). Pasar `use_mixup=False`
    desactiva la augmentación.

    Guarda pesos por tipo en save_dir y retorna el Z2XDecoders entrenado.
    """
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    z_dict = get_embeddings_minibatch(
        model, data, num_neighbors=num_neighbors
    )

    # Probe de separabilidad lineal del encoder sobre val (sanity del feature
    # extractor antes de gastar epochs entrenando el decoder z->x).
    for ntype in z_dict.keys():
        if (node_types is not None) and (ntype not in node_types):
            continue
        try:
            probe = evaluate_embedding_separability(z_dict, data, node_type=ntype)
        except Exception as exc:
            logger.warning(f"[Z2X] probe de separabilidad falló para '{ntype}': {exc}")
            probe = None
        if probe is not None:
            logger.info(
                f"[Z2X] separabilidad encoder '{ntype}' (LogReg sobre z): "
                f"F1_macro={probe['val_f1_macro']:.3f} "
                f"F1_min={probe['val_f1_minority']:.3f} "
                f"AUPRC={probe['val_auprc']:.3f} "
                f"ROC_AUC={probe['val_roc_auc']:.3f} "
                f"(n_train={probe['n_train']}, n_val={probe['n_val']})"
            )
            if probe['val_auprc'] < 0.1 and probe['val_roc_auc'] < 0.6:
                logger.warning(
                    f"[Z2X] embeddings de '{ntype}' tienen muy poca señal de clase "
                    "(AUPRC<0.1 y ROC_AUC<0.6); SMOTE en este espacio probablemente "
                    "no ayudará. Re-entrena el encoder antes de aumentar."
                )

    # dims por tipo
    dims_per_type = {}
    for ntype, z in z_dict.items():
        if (node_types is None) or (ntype in node_types):
            if z.nelement() > 0:
                z_dim = z.size(1)
                x_dim = data[ntype].x.size(1)
                dims_per_type[ntype] = (z_dim, x_dim)

    if minority_classes is None:
        minority_classes = {ntype: 1 for ntype in dims_per_type.keys()}

    dec = Z2XDecoders(dims_per_type).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=weight_decay)
    crit = (
        torch.nn.SmoothL1Loss(reduction="none")
        if loss_type == "huber"
        else torch.nn.MSELoss(reduction="none")
    )
    if minority_recon_weight is None:
        minority_recon_weight = float(GRAPHSMOTE_Z2X_MINORITY_WEIGHT)
    minority_recon_weight = float(max(1.0, minority_recon_weight))
    early_stop_metric = str(early_stop_metric or GRAPHSMOTE_Z2X_EARLY_STOP_METRIC or "val_loss").strip().lower()

    def _reconstruction_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_elem = crit(pred, target)
        if per_elem.dim() > 1:
            per_sample = per_elem.view(per_elem.size(0), -1).mean(dim=1)
        else:
            per_sample = per_elem.view(-1)
        if sample_weights is not None:
            weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype).view(-1)
            return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)
        return per_sample.mean()

    # Precompute device tensors and masks per type
    z_all = {}
    x_all = {}
    y_all = {}
    train_idx = {}
    val_idx = {}
    minority_train_idx = {}        # subset of train_idx donde y == clase minoritaria
    mixup_neighbors = {}           # ntype -> (k_eff, nbr_idx tensor [N_min, k_eff])
    has_any_val = False

    rng = np.random.RandomState(SEED)

    for ntype in dims_per_type.keys():
        z_all[ntype] = z_dict[ntype].to(device)
        x_all[ntype] = data[ntype].x.to(device)
        if 'y' in data[ntype]:
            y_all[ntype] = data[ntype].y.to(device)
        else:
            y_all[ntype] = None

        N = x_all[ntype].size(0)
        m_train_base = torch.ones(N, dtype=torch.bool, device=device)
        m_val = torch.zeros(N, dtype=torch.bool, device=device)

        if use_masks and ('train_mask' in data[ntype]):
            m_train_base = data[ntype]['train_mask'].to(device)
            if m_train_base.dtype != torch.bool:
                m_train_base = m_train_base.bool()
        if use_masks and ('val_mask' in data[ntype]):
            m_val = data[ntype]['val_mask'].to(device)
            if m_val.dtype != torch.bool:
                m_val = m_val.bool()
        else:
            # Create a small validation split from train if none provided
            base_idx = torch.nonzero(m_train_base, as_tuple=True)[0]
            if base_idx.numel() > 0 and val_split > 0.0:
                n_val = max(1, int(round(base_idx.numel() * val_split)))
                chosen = torch.from_numpy(rng.choice(base_idx.cpu().numpy(), size=n_val, replace=False)).to(device)
                m_val[chosen] = True

        # Excluir nodos sintéticos previos para evitar feedback loop:
        # el decoder no debe entrenarse sobre x's que él mismo (o una versión
        # anterior) produjo en una iteración previa de aumentación.
        if 'is_synthetic' in data[ntype]:
            is_synth = data[ntype]['is_synthetic'].to(device)
            if is_synth.dtype != torch.bool:
                is_synth = is_synth.bool()
            n_synth = int(is_synth.sum().item())
            if n_synth > 0:
                m_train_base = m_train_base & (~is_synth)
                m_val = m_val & (~is_synth)
                logger.info(
                    f"[Z2X] '{ntype}': excluyendo {n_synth} nodo(s) sintético(s) previo(s)."
                )

        # Ensure disjoint
        m_train = m_train_base & (~m_val)

        train_idx[ntype] = torch.nonzero(m_train, as_tuple=True)[0]
        val_idx[ntype] = torch.nonzero(m_val, as_tuple=True)[0]
        has_any_val = has_any_val or (val_idx[ntype].numel() > 0)

        # Subset minoritario y kNN para mixup
        min_label = minority_classes.get(ntype, None)
        if (
            use_mixup
            and min_label is not None
            and y_all[ntype] is not None
            and train_idx[ntype].numel() > 0
        ):
            y_train_t = y_all[ntype].index_select(0, train_idx[ntype])
            min_local = (y_train_t == int(min_label))
            min_train_idx = train_idx[ntype][min_local]
            minority_train_idx[ntype] = min_train_idx
            n_min = int(min_train_idx.numel())
            if n_min >= 2:
                k_eff = min(int(mixup_k), n_min - 1)
                z_min_np = (
                    z_all[ntype]
                    .index_select(0, min_train_idx)
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric='cosine').fit(z_min_np)
                nbr_np = nbrs.kneighbors(z_min_np, return_distance=False)[:, 1:]
                mixup_neighbors[ntype] = (
                    int(k_eff),
                    torch.from_numpy(nbr_np).long().to(device),
                )
                logger.info(
                    f"[Z2X] '{ntype}': mixup activo sobre {n_min} minoritarios (k_eff={k_eff})."
                )
            else:
                logger.info(
                    f"[Z2X] '{ntype}': mixup desactivado (solo {n_min} minoritario(s) en train)."
                )

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    epoch_iter = range(1, epochs + 1)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Z2X training (epochs)", leave=False)

    history = []

    for epoch in epoch_iter:
        dec.train()
        opt.zero_grad()
        total_train_loss = 0.0
        total_mixup_loss = 0.0

        for ntype in dims_per_type.keys():
            if train_idx[ntype].numel() == 0:
                continue
            z = z_all[ntype].index_select(0, train_idx[ntype])
            x = x_all[ntype].index_select(0, train_idx[ntype])
            xhat = dec.project_one(ntype, z)
            weights = None
            min_label = minority_classes.get(ntype, None)
            if (
                minority_recon_weight > 1.0
                and min_label is not None
                and y_all[ntype] is not None
            ):
                y_train_batch = y_all[ntype].index_select(0, train_idx[ntype])
                weights = torch.ones(y_train_batch.numel(), device=device, dtype=xhat.dtype)
                weights[y_train_batch == int(min_label)] = float(minority_recon_weight)
            loss = _reconstruction_loss(xhat, x, weights)
            loss.backward()
            total_train_loss += float(loss.item())

            # --- Mixup augmentation ---
            if ntype in mixup_neighbors:
                k_eff, nbr_idx = mixup_neighbors[ntype]
                m_idx = minority_train_idx[ntype]
                n_min = int(m_idx.numel())
                n_mix = max(1, int(round(mixup_ratio * n_min)))
                base_np = rng.randint(0, n_min, size=n_mix)
                nbr_np = rng.randint(0, k_eff, size=n_mix)
                alpha_np = rng.random_sample(size=n_mix).astype(np.float32)

                base_t = torch.from_numpy(base_np).long().to(device)
                nbr_t = torch.from_numpy(nbr_np).long().to(device)
                j_t = nbr_idx[base_t, nbr_t]
                alpha_t = torch.from_numpy(alpha_np).to(device).unsqueeze(1)

                z_min_full = z_all[ntype].index_select(0, m_idx)
                x_min_full = x_all[ntype].index_select(0, m_idx)
                z_a = z_min_full.index_select(0, base_t)
                z_b = z_min_full.index_select(0, j_t)
                x_a = x_min_full.index_select(0, base_t)
                x_b = x_min_full.index_select(0, j_t)

                z_mix = (1.0 - alpha_t) * z_a + alpha_t * z_b
                z_mix = F.normalize(z_mix, p=2, dim=1, eps=1e-12)
                x_mix = (1.0 - alpha_t) * x_a + alpha_t * x_b

                xhat_mix = dec.project_one(ntype, z_mix)
                mloss = float(mixup_weight) * _reconstruction_loss(xhat_mix, x_mix)
                mloss.backward()
                total_mixup_loss += float(mloss.item())

        opt.step()

        # Validation (total + per-class si aplica)
        total_val_loss = None
        val_loss_minority = None
        val_loss_majority = None
        if early_stop and has_any_val:
            dec.eval()
            with torch.no_grad():
                vloss = 0.0
                vcount = 0
                v_min_losses = []
                v_maj_losses = []
                for ntype in dims_per_type.keys():
                    if val_idx[ntype].numel() == 0:
                        continue
                    z = z_all[ntype].index_select(0, val_idx[ntype])
                    x = x_all[ntype].index_select(0, val_idx[ntype])
                    xhat = dec.project_one(ntype, z)
                    loss = _reconstruction_loss(xhat, x)
                    vloss += float(loss.item())
                    vcount += 1

                    min_label = minority_classes.get(ntype, None)
                    if min_label is not None and y_all[ntype] is not None:
                        y_val = y_all[ntype].index_select(0, val_idx[ntype])
                        min_mask = (y_val == int(min_label))
                        if min_mask.any():
                            v_min_losses.append(float(_reconstruction_loss(xhat[min_mask], x[min_mask]).item()))
                        if (~min_mask).any():
                            v_maj_losses.append(float(_reconstruction_loss(xhat[~min_mask], x[~min_mask]).item()))
                if vcount > 0:
                    total_val_loss = vloss / vcount
                if v_min_losses:
                    val_loss_minority = sum(v_min_losses) / len(v_min_losses)
                if v_maj_losses:
                    val_loss_majority = sum(v_maj_losses) / len(v_maj_losses)

            # Early stopping check
            monitor_val = total_val_loss
            if early_stop_metric in {"val_loss_minority", "minority", "minority_val_loss"}:
                monitor_val = val_loss_minority if val_loss_minority is not None else total_val_loss
            elif early_stop_metric in {"weighted_val_loss", "val_loss_weighted"}:
                if total_val_loss is not None and val_loss_minority is not None:
                    monitor_val = 0.5 * total_val_loss + 0.5 * val_loss_minority
            if monitor_val is not None:
                improved = (best_val - monitor_val) > min_delta
                if improved:
                    best_val = monitor_val
                    best_state = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Append to history
        history.append({
            "epoch": epoch,
            "train_loss": total_train_loss,
            "mixup_loss": total_mixup_loss if total_mixup_loss > 0.0 else None,
            "val_loss": total_val_loss if total_val_loss is not None else None,
            "val_loss_minority": val_loss_minority,
            "val_loss_majority": val_loss_majority,
            "minority_recon_weight": float(minority_recon_weight),
            "early_stop_metric": early_stop_metric,
        })

        if (log_every is not None) and (log_every > 0) and (epoch % log_every == 0):
            if show_progress and hasattr(epoch_iter, 'set_postfix'):
                postfix = {
                    'train': f"{total_train_loss:.4f}",
                }
                if total_mixup_loss > 0.0:
                    postfix['mix'] = f"{total_mixup_loss:.4f}"
                if total_val_loss is not None:
                    postfix['val'] = f"{total_val_loss:.4f}"
                    postfix['best'] = (
                        f"{best_val:.4f}" if best_val != float('inf') else "nan"
                    )
                if val_loss_minority is not None:
                    postfix['val_min'] = f"{val_loss_minority:.4f}"
                epoch_iter.set_postfix(postfix)
            else:
                msg = f"[Z2X] Epoch {epoch:03d}/{epochs} | train={total_train_loss:.4f}"
                if total_mixup_loss > 0.0:
                    msg += f" | mixup={total_mixup_loss:.4f}"
                if total_val_loss is not None:
                    msg += f" | val={total_val_loss:.4f}"
                if val_loss_minority is not None:
                    msg += f" | val_min={val_loss_minority:.4f}"
                logger.info(msg)

        if progress_callback is not None:
            progress_callback(
                epoch=epoch,
                total=epochs,
                train_loss=total_train_loss,
                val_loss=total_val_loss
            )

        if early_stop and has_any_val and patience > 0 and epochs_no_improve >= patience:
            if show_progress and hasattr(epoch_iter, 'set_description'):
                epoch_iter.set_description("Z2X training (early stop)")
            logger.info(f"[Z2X] Early stopping at epoch {epoch:03d} (no improvement for {patience} epochs)")
            break

    # Load best state (if any)
    if show_progress and hasattr(epoch_iter, 'close'):
        epoch_iter.close()

    if best_state is not None:
        dec.load_state_dict(best_state)

    # Guardar por tipo
    for ntype in dims_per_type.keys():
        torch.save(dec.heads[ntype].state_dict(), os.path.join(save_dir, f"{ntype}.pt"))

    # Guardar historial de entrenamiento
    history_path = os.path.join(save_dir, "history.json")
    try:
        import json
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save z2x history: {e}")

    dec.eval()
    dec._loaded_types = set(dims_per_type.keys())
    return dec

def load_z2x_decoders(model, data, node_types=None, device=None,
                      save_dir=os.path.join(RESULTADOS_DIR, "z2x_decoders")):
    """
    Construye Z2XDecoders con dims correctas y carga pesos si existen.
    Deja dec._loaded_types con los tipos efectivamente cargados desde disco.
    """
    if device is None:
        device = next(model.parameters()).device

    z_dict = get_embeddings_minibatch(model, data)

    dims_per_type = {}
    for ntype, z in z_dict.items():
        if (node_types is None) or (ntype in node_types):
            if z.nelement() > 0:
                z_dim = z.size(1)
                x_dim = data[ntype].x.size(1)
                dims_per_type[ntype] = (z_dim, x_dim)

    dec = Z2XDecoders(dims_per_type).to(device)
    loaded = set()
    for ntype in dims_per_type.keys():
        p = os.path.join(save_dir, f"{ntype}.pt")
        if os.path.exists(p):
            try:
                dec.heads[ntype].load_state_dict(
                    torch.load(p, map_location=device, weights_only=True)
                )
                loaded.add(ntype)
            except RuntimeError as e:
                print(f"Warning: could not load state_dict for ntype '{ntype}'. It might be retrained if necessary. Error: {e}")
    dec.eval()
    dec._loaded_types = loaded
    return dec

# ============================================================
# 3.5) Decoder de atributos de arista para SMOTE
# ============================================================
class RelEdgeAttrDecoder(torch.nn.Module):
    """
    Un MLP por tipo de relación que mapea [z_src ⊕ z_dst] → edge_attr.

    Entrenado con aristas reales como supervisión, permite reemplazar el
    zero-fill (o delta truncado) que aplicaba GraphSMOTE a las aristas
    sintéticas. Cada relación conserva su propia dimensión de `edge_attr`;
    esto evita mezclar, por ejemplo, atributos temporales y espaciales con
    anchos distintos.
    """

    def __init__(
        self,
        rels,
        z_dim_dict: Dict[str, int],
        edge_attr_dims_by_rel: Dict[str, int],
        hidden_dim: int = 128,
        clip_predictions: bool = True,
    ):
        super().__init__()
        self.edge_attr_dims_by_rel = {str(k): int(v) for k, v in edge_attr_dims_by_rel.items()}
        self.edge_attr_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.clip_predictions = bool(clip_predictions)
        self.heads = torch.nn.ModuleDict()
        for (src, rel, dst) in rels:
            key = f"{src}:{rel}:{dst}"
            if key not in self.edge_attr_dims_by_rel:
                continue
            in_dim = int(z_dim_dict[src]) + int(z_dim_dict[dst])
            out_dim = self.edge_attr_dims_by_rel[key]
            self.heads[key] = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, key: str) -> torch.Tensor:
        x = torch.cat([z_src, z_dst], dim=-1)
        return self.heads[key](x)

    def set_edge_attr_stats(
        self,
        key: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        min_value: torch.Tensor,
        max_value: torch.Tensor,
    ) -> None:
        self.edge_attr_stats[key] = {
            "mean": mean.detach().clone(),
            "std": std.detach().clone().clamp_min(1e-6),
            "min": min_value.detach().clone(),
            "max": max_value.detach().clone(),
        }

    def _stats_for(
        self,
        key: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        stats = self.edge_attr_stats.get(key)
        if stats is None:
            return None
        out: Dict[str, torch.Tensor] = {}
        for stat_name, tensor in stats.items():
            t = tensor
            if device is not None:
                t = t.to(device)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype=dtype)
            out[stat_name] = t
        return out

    def normalize_target(self, key: str, target: torch.Tensor) -> torch.Tensor:
        stats = self._stats_for(key, device=target.device, dtype=target.dtype)
        if stats is None:
            return target
        return (target - stats["mean"]) / stats["std"]

    def denormalize_prediction(self, key: str, pred: torch.Tensor) -> torch.Tensor:
        stats = self._stats_for(key, device=pred.device, dtype=pred.dtype)
        if stats is None:
            return pred
        out = pred * stats["std"] + stats["mean"]
        if self.clip_predictions:
            out = torch.minimum(torch.maximum(out, stats["min"]), stats["max"])
        return out

    @torch.no_grad()
    def predict(self, z_src: torch.Tensor, z_dst: torch.Tensor, key: str) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.denormalize_prediction(key, self.forward(z_src, z_dst, key))
        if was_training:
            self.train()
        return out


def train_edge_attr_decoders(
    model,
    data: HeteroData,
    *,
    device=None,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    batch_size: int = 4096,
    val_split: float = 0.1,
    save_dir: str = os.path.join(RESULTADOS_DIR, "edge_attr_decoders"),
    seed: int = 0,
    num_neighbors=NUM_NEIGHBORS,
    show_progress: bool = True,
) -> Optional[RelEdgeAttrDecoder]:
    """
    Entrena `RelEdgeAttrDecoder` por tipo de relación usando aristas reales.

    Solo se usan aristas cuyo nodo origen pertenece a `train_mask` (si existe)
    y que NO son sintéticas (en caso de un grafo ya aumentado se ignoran las
    `is_synthetic` para evitar que el decoder aprenda de sus propias
    predicciones).

    Devuelve el decoder en modo eval o `None` si no hay aristas con
    `edge_attr` o no se cumple ningún tipo entrenable.
    """
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = next(model.parameters()).device

    # 1) Embeddings z para todos los tipos de nodo
    z_dict = get_embeddings_minibatch(model, data, num_neighbors=num_neighbors)
    z_dim_dict = {ntype: int(z.shape[1]) for ntype, z in z_dict.items()}

    # 2) Detectar edge_attr_dim por relación. No asumir dimensión global:
    # temporal y spatial pueden tener anchos distintos en el grafo real.
    edge_attr_dims_by_rel: Dict[str, int] = {}
    for et in data.edge_types:
        src, rel, dst = et
        e_store = data[et]
        if hasattr(e_store, "edge_attr") and e_store.edge_attr is not None and e_store.edge_attr.numel() > 0:
            key = f"{src}:{rel}:{dst}"
            edge_attr_dims_by_rel[key] = int(e_store.edge_attr.shape[1]) if e_store.edge_attr.dim() > 1 else 1
    if not edge_attr_dims_by_rel:
        logger.info("[EdgeAttrDec] grafo sin edge_attr; saltando entrenamiento.")
        return None

    decoder = RelEdgeAttrDecoder(
        data.edge_types, z_dim_dict, edge_attr_dims_by_rel, hidden_dim=hidden_dim
    ).to(device)
    if not decoder.heads:
        logger.info("[EdgeAttrDec] no hay relaciones con dimensiones entrenables; saltando entrenamiento.")
        return None
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.HuberLoss()

    # 3) Construir splits train/val por tipo de relación
    rng = np.random.RandomState(seed)
    per_rel: Dict[str, Dict[str, torch.Tensor]] = {}
    rel_diagnostics: Dict[str, Dict[str, int]] = {}
    for (src, rel, dst) in data.edge_types:
        key = f"{src}:{rel}:{dst}"
        if key not in decoder.heads:
            continue
        e_store = data[(src, rel, dst)]
        if not hasattr(e_store, "edge_attr") or e_store.edge_attr is None or e_store.edge_attr.numel() == 0:
            continue
        if src not in z_dict or dst not in z_dict:
            logger.warning(f"[EdgeAttrDec] faltan embeddings para '{key}'; saltando relación.")
            continue
        ei = e_store.edge_index.cpu()
        ea = e_store.edge_attr.cpu().float()
        if ea.dim() == 1:
            ea = ea.unsqueeze(1)
        expected_dim = int(edge_attr_dims_by_rel[key])
        if ea.size(1) != expected_dim:
            logger.warning(
                f"[EdgeAttrDec] edge_attr dim inesperada en '{key}': "
                f"{ea.size(1)} != {expected_dim}; saltando relación."
            )
            continue

        # Filtros: train_mask del origen + excluir sintéticos si el grafo está aumentado
        mask = torch.ones(ei.shape[1], dtype=torch.bool)
        if hasattr(data[src], "train_mask"):
            mask = mask & data[src].train_mask.cpu()[ei[0]]
        if hasattr(data[src], "is_synthetic"):
            mask = mask & (~data[src].is_synthetic.cpu()[ei[0]])
        if hasattr(data[dst], "is_synthetic"):
            mask = mask & (~data[dst].is_synthetic.cpu()[ei[1]])

        ei_filt = ei[:, mask]
        ea_filt = ea[mask]
        n = ei_filt.shape[1]
        if n == 0:
            continue

        idx = rng.permutation(n)
        n_val = max(1, int(round(n * val_split)))
        if n_val >= n:
            n_val = max(0, n - 1)
        val_sel = torch.from_numpy(idx[:n_val])
        tr_sel = torch.from_numpy(idx[n_val:])
        if tr_sel.numel() == 0:
            logger.info(f"[EdgeAttrDec] '{key}' sin aristas train tras split; saltando relación.")
            continue

        ea_train_cpu = ea_filt[tr_sel].float()
        scale_mean = ea_train_cpu.mean(dim=0)
        scale_std = ea_train_cpu.std(dim=0, unbiased=False).clamp_min(1e-6)
        scale_min = ea_train_cpu.min(dim=0).values
        scale_max = ea_train_cpu.max(dim=0).values
        decoder.set_edge_attr_stats(
            key,
            scale_mean.to(device),
            scale_std.to(device),
            scale_min.to(device),
            scale_max.to(device),
        )

        per_rel[key] = {
            "ei_train": ei_filt[:, tr_sel].to(device),
            "ea_train": ea_filt[tr_sel].to(device),
            "ei_val": ei_filt[:, val_sel].to(device),
            "ea_val": ea_filt[val_sel].to(device),
            "z_src_full": z_dict[src].to(device),
            "z_dst_full": z_dict[dst].to(device),
            "n_train": int(tr_sel.numel()),
            "n_val": int(val_sel.numel()),
        }
        rel_diagnostics[key] = {
            "edge_attr_dim": int(expected_dim),
            "n_edges_filtered": int(n),
            "n_train": int(tr_sel.numel()),
            "n_val": int(val_sel.numel()),
            "normalized_target": True,
            "clip_predictions_to_train_minmax": bool(decoder.clip_predictions),
            "target_std_min": float(scale_std.min().item()),
            "target_std_median": float(scale_std.median().item()),
            "target_std_max": float(scale_std.max().item()),
        }

    if not per_rel:
        logger.info("[EdgeAttrDec] no hay aristas reales filtrables; saltando entrenamiento.")
        return None

    history: List[Dict[str, float]] = []
    iter_range = range(epochs)
    if show_progress:
        iter_range = tqdm(iter_range, desc="EdgeAttrDec")

    last_val_per_rel: Dict[str, float] = {}

    for ep in iter_range:
        decoder.train()
        train_losses_ep: List[float] = []
        for key, store in per_rel.items():
            ei = store["ei_train"]
            ea = store["ea_train"]
            n = ei.shape[1]
            if n == 0:
                continue
            order = torch.randperm(n, device=device)
            for s in range(0, n, batch_size):
                b = order[s:s + batch_size]
                z_src_b = store["z_src_full"][ei[0, b]]
                z_dst_b = store["z_dst_full"][ei[1, b]]
                pred = decoder(z_src_b, z_dst_b, key)
                target = ea[b]
                if target.dtype != pred.dtype:
                    target = target.to(pred.dtype)
                target = decoder.normalize_target(key, target)
                loss = loss_fn(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses_ep.append(float(loss.item()))

        # Validación
        decoder.eval()
        val_losses_ep: List[float] = []
        with torch.no_grad():
            for key, store in per_rel.items():
                ei_v = store["ei_val"]
                ea_v = store["ea_val"]
                if ei_v.shape[1] == 0:
                    continue
                z_src_v = store["z_src_full"][ei_v[0]]
                z_dst_v = store["z_dst_full"][ei_v[1]]
                pred_v = decoder(z_src_v, z_dst_v, key)
                target_v = ea_v.to(pred_v.dtype)
                target_v = decoder.normalize_target(key, target_v)
                vl = float(loss_fn(pred_v, target_v).item())
                val_losses_ep.append(vl)
                last_val_per_rel[key] = vl

        if train_losses_ep or val_losses_ep:
            entry = {
                "epoch": int(ep),
                "train_loss": float(np.mean(train_losses_ep)) if train_losses_ep else 0.0,
                "val_loss": float(np.mean(val_losses_ep)) if val_losses_ep else 0.0,
            }
            history.append(entry)
            if show_progress:
                iter_range.set_postfix({
                    "train": f"{entry['train_loss']:.4f}",
                    "val": f"{entry['val_loss']:.4f}",
                })

    # 4) Persistir pesos y diagnóstico
    try:
        import json as _json
        torch.save(decoder.state_dict(), os.path.join(save_dir, "edge_attr_decoder.pt"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            _json.dump(history, f, indent=2)
        with open(os.path.join(save_dir, "val_loss_by_rel.json"), "w") as f:
            _json.dump({k: float(v) for k, v in last_val_per_rel.items()}, f, indent=2)
        with open(os.path.join(save_dir, "relation_diagnostics.json"), "w") as f:
            _json.dump(rel_diagnostics, f, indent=2)
    except Exception as exc:
        logger.warning(f"[EdgeAttrDec] persistencia falló: {exc}")

    decoder.eval()
    return decoder

# ============================================================
# 4) SMOTE en espacio z — helper canónico (Eq. 4 del paper)
# ============================================================
def _smote_in_z_space(
    z: torch.Tensor,
    y: torch.Tensor,
    minority_class: int,
    k: int,
    n_samples: int,
    rng: np.random.RandomState,
    *,
    return_renormalized: bool = True,
    return_parent_info: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Interpolación SMOTE en el espacio de embeddings (Zhao et al. 2021, Eq. 4):

        z_syn = (1 - α)·z_a + α·z_b,   α ~ U[0, 1]

    donde z_a es un nodo minoritario aleatorio y z_b uno de sus k vecinos
    minoritarios más cercanos (distancia coseno).

    Si `return_renormalized=True`, los z sintéticos se proyectan de vuelta a la
    hiperesfera unitaria via L2-normalización para coincidir con la distribución
    en la que el encoder fue entrenado (`get_embeddings_minibatch` siempre
    normaliza, así que el decoder z->x espera embeddings unitarios).

    Devuelve `(syn_z [n_samples, d], syn_labels [n_samples])`. Si
    `return_parent_info=True`, añade un dict con `base_idx`, `neighbor_idx` y
    `alpha`, todos alineados con el tensor `z` recibido. Si no hay suficientes
    minoritarios para hacer kNN, devuelve tensores vacíos.
    """
    def _empty_result():
        empty_z = torch.empty(0, z.size(1), device=z.device, dtype=z.dtype)
        empty_y = torch.empty(0, dtype=y.dtype, device=y.device)
        if not return_parent_info:
            return empty_z, empty_y
        empty_long = torch.empty(0, dtype=torch.long, device=z.device)
        empty_alpha = torch.empty(0, dtype=z.dtype, device=z.device)
        return empty_z, empty_y, {
            "base_idx": empty_long,
            "neighbor_idx": empty_long.clone(),
            "alpha": empty_alpha,
        }

    if z.dim() != 2:
        raise ValueError(f"_smote_in_z_space espera z de shape [N,d], recibió {tuple(z.shape)}")
    if n_samples <= 0:
        return _empty_result()

    minority_mask = (y == int(minority_class))
    minority_input_idx = torch.nonzero(minority_mask, as_tuple=True)[0].to(z.device)
    pos_emb = z[minority_mask]
    n_pos = pos_emb.size(0)

    if n_pos < 2:
        logger.warning(
            f"[GraphSMOTE] Solo {n_pos} muestra(s) de la clase minoritaria {minority_class}; "
            "no se puede interpolar."
        )
        return _empty_result()

    k_eff = min(int(k), n_pos - 1)
    if k_eff < int(k):
        logger.warning(
            f"[GraphSMOTE] k_eff={k_eff} < k={k} (solo {n_pos} muestras minoritarias)."
        )

    pos_cpu = pos_emb.detach().cpu().to(torch.float32).numpy()
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(pos_cpu)
    nbr_idx = nbrs.kneighbors(pos_cpu, return_distance=False)[:, 1:]  # quitar self

    base = rng.randint(0, n_pos, size=int(n_samples))
    nbr_choice = rng.randint(0, k_eff, size=int(n_samples))
    j = nbr_idx[base, nbr_choice]
    alpha = rng.random_sample(size=int(n_samples)).astype(np.float32)

    base_t = torch.from_numpy(base).long().to(z.device)
    j_t = torch.from_numpy(j).long().to(z.device)
    alpha_t = torch.from_numpy(alpha).to(z.device).unsqueeze(1)

    z_a = pos_emb.index_select(0, base_t)
    z_b = pos_emb.index_select(0, j_t)
    syn_z = (1.0 - alpha_t) * z_a + alpha_t * z_b

    if return_renormalized:
        syn_z = F.normalize(syn_z, p=2, dim=1, eps=1e-12)

    syn_labels = torch.full(
        (int(n_samples),), int(minority_class), dtype=y.dtype, device=y.device
    )
    if return_parent_info:
        parent_info = {
            "base_idx": minority_input_idx.index_select(0, base_t).detach(),
            "neighbor_idx": minority_input_idx.index_select(0, j_t).detach(),
            "alpha": torch.from_numpy(alpha).to(z.device, dtype=z.dtype).detach(),
            "base_pos_idx": base_t.detach(),
            "neighbor_pos_idx": j_t.detach(),
        }
        return syn_z, syn_labels, parent_info
    return syn_z, syn_labels

# ============================================================
# 5) GraphSMOTE (integrado con decodificadores z->x y edge generator)
# ============================================================
def save_augmented_graph(data, save_path):
    """
    Guarda un grafo de datos aumentado en el disco.
    """
    if save_path:
        try:
            # Mover datos a la CPU antes de guardar para evitar problemas de GPU
            data_cpu = data.cpu()
            torch.save(data_cpu, save_path)
            logger.info(f"Grafo aumentado guardado en: {save_path}")
        except Exception as e:
            logger.error(f"Error al guardar el grafo aumentado en {save_path}: {e}")

def run_graphsmote(
    model,
    data,
    nodes_to_smote,
    k_smote,
    k_neighbors_edges,
    random_state,
    z2x_decoders=None,
    save_dir=os.path.join(RESULTADOS_DIR, "z2x_decoders"),
    device=None,
    add_to_train_mask=True,
    edge_gen=None,
    save_path=None,
    progress_callback: Optional[callable] = None,
    force_cpu: bool = True,
    synthetic_feature_mode: Optional[str] = None,
):
    device = next(model.parameters()).device
    data = data.to(device)

    # Garantizar decodificadores z->x entrenados/cargados una vez
    if z2x_decoders is None:
        z2x_decoders = load_z2x_decoders(model, data, device=device, save_dir=save_dir)
        expected_types = set(data.node_types)
        if z2x_decoders._loaded_types != expected_types:
            print("Entrenando z->x decoders ...")
            z2x_decoders = train_z2x_decoders(
                model, data, device=device, save_dir=save_dir,
                progress_callback=progress_callback
            )
    z2x_decoders = z2x_decoders.to(device)
    z2x_decoders.eval()

    # Embeddings actuales normalizados
    z_dict = get_embeddings_minibatch(model, data, store_on_cpu=bool(force_cpu))

    # Edge generator (si no se pasa, se crea uno)
    if edge_gen is None:
        print("Entrenando RelEdgeGen ...")
        z_dim_dict = {ntype: z.shape[1] for ntype, z in z_dict.items() if z.nelement() > 0}
        # Asegurarse de que todos los tipos de nodos en las relaciones tengan una dimensión,
        # incluso si no hay nodos de ese tipo en el grafo actual.
        for edge_type in data.edge_types:
            for node_type_in_edge in [edge_type[0], edge_type[2]]:
                if node_type_in_edge not in z_dim_dict:
                    # Usar la dimensión de salida del modelo como fallback.
                    z_dim_dict[node_type_in_edge] = model.hidden_channels * model.num_heads
        edge_gen = RelEdgeGen(z_dim_dict, data.edge_types).to(device)
        # Opcional: pre-entrenar aquí si es necesario
    edge_gen.eval()

    augmented_data = data.clone()
    if hasattr(data, "delta_feature_idx"):
        try:
            augmented_data.delta_feature_idx = data.delta_feature_idx
        except Exception:
            pass

    for smote_params in nodes_to_smote:
        node_type = smote_params['node_type']
        minority_class = smote_params['minority_class']
        n_samples = smote_params['n_samples']

        z_node = z_dict[node_type].to(device)
        y_node = data[node_type].y.to(z_node.device)

        # Restringir SMOTE al subset de entrenamiento si existe train_mask
        z_smote = z_node
        y_smote = y_node
        x_smote = data[node_type].x.to(z_node.device)
        try:
            if hasattr(data[node_type], "train_mask"):
                train_mask = data[node_type].train_mask.to(z_node.device)
                if train_mask.dtype != torch.bool:
                    train_mask = train_mask.bool()
                train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1)
                if train_idx.numel() > 0:
                    z_smote = z_node.index_select(0, train_idx)
                    y_smote = y_node.index_select(0, train_idx)
                    x_smote = data[node_type].x.to(z_node.device).index_select(0, train_idx)
        except Exception:
            # Si falla, se usa el conjunto completo por compatibilidad
            z_smote = z_node
            y_smote = y_node
            x_smote = data[node_type].x.to(z_node.device)

        rng = np.random.RandomState(int(random_state) if random_state is not None else SEED)
        syn_z, syn_labels, parent_info = _smote_in_z_space(
            z_smote, y_smote,
            minority_class=int(minority_class),
            k=int(k_smote),
            n_samples=int(n_samples),
            rng=rng,
            return_renormalized=True,  # match decoder training distribution
            return_parent_info=True,
        )

        if syn_z.numel() == 0:
            print(f"GraphSMOTE did not generate any new {node_type} samples.")
            continue

        syn_x, feature_quality = _project_synthetic_features_from_smote(
            node_type=node_type,
            syn_z=syn_z,
            x_source=x_smote,
            y_source=y_smote,
            parent_info=parent_info,
            minority_class=int(minority_class),
            z2x_decoders=z2x_decoders,
            feature_mode=synthetic_feature_mode,
        )
        logger.info("[GraphSMOTE] synthetic feature quality (%s): %s", node_type, feature_quality)

        if syn_x.numel() == 0:
            print(f"GraphSMOTE did not generate any new {node_type} samples after decoding.")
            continue

        N_old = augmented_data[node_type].num_nodes
        N_new = syn_x.size(0)

        # features y labels (alinear dispositivo con los tensores base)
        base_x = augmented_data[node_type].x
        base_y = augmented_data[node_type].y
        syn_x = syn_x.to(device=base_x.device, dtype=base_x.dtype)
        syn_labels = syn_labels.to(base_y.device)
        augmented_data[node_type].x = torch.cat([base_x, syn_x], dim=0)
        augmented_data[node_type].y = torch.cat([base_y, syn_labels], dim=0)

        # masks
        for m in ['train_mask', 'val_mask', 'test_mask', 'is_accident_pm']:
            if hasattr(augmented_data[node_type], m):
                old = getattr(augmented_data[node_type], m)
                # is_accident_pm es booleano, los nuevos nodos no son accidentes
                pad = torch.zeros(N_new, dtype=torch.bool, device=old.device)
                setattr(augmented_data[node_type], m, torch.cat([old, pad], dim=0))
            elif m == 'is_accident_pm':
                # Si no existe, se crea para todos los nodos (antiguos y nuevos) como False.
                total_nodes = N_old + N_new
                new_mask = torch.zeros(total_nodes, dtype=torch.bool, device=base_x.device)
                setattr(augmented_data[node_type], m, new_mask)
        
        # sintéticos solo entrenan:
        if add_to_train_mask and hasattr(augmented_data[node_type], 'train_mask'):
            augmented_data[node_type].train_mask[-N_new:] = True

        # flag diagnóstico
        if not hasattr(augmented_data[node_type], 'is_synthetic'):
            augmented_data[node_type].is_synthetic = torch.zeros(N_old, dtype=torch.bool, device=base_x.device)

        old_syn = augmented_data[node_type].is_synthetic
        if old_syn.device != base_x.device:
            old_syn = old_syn.to(base_x.device)
        new_synthetic_flag = torch.ones(N_new, dtype=torch.bool, device=base_x.device)
        augmented_data[node_type].is_synthetic = torch.cat([old_syn, new_synthetic_flag], dim=0)


        # --- CONEXIÓN DE NODOS SINTÉTICOS ---
        if node_type == 'pm':
            # 1. Elegir candidatos reales según configuración
            if int(GRAPHSMOTE_CONECT) == 1:
                # Método actual: todos los reales en TRAIN
                if hasattr(data[node_type], 'train_mask'):
                    train_mask_real = data[node_type].train_mask.to(device)
                    candidate_real_indices = torch.where(train_mask_real)[0]
                else:
                    candidate_real_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                # Alternativa: vecinos 1-hop de accidentes reales (clase=1)
                candidate_real_indices = _first_hop_neighbors_of_accidents(
                    data, node_type=node_type, use_train_only=True, device=device, log_context="run_graphsmote"
                )

            if candidate_real_indices.numel() == 0:
                logger.warning(f"No hay candidatos reales para conectar (modo GRAPHSMOTE_CONECT={GRAPHSMOTE_CONECT}).")
                continue

            candidate_real_indices = candidate_real_indices.to(z_node.device)
            z_real_train_pm = z_node.index_select(0, candidate_real_indices)
            z_syn_pm = syn_z
            
            num_syn_nodes = syn_x.size(0)

            # Conectar pm_syn -> pm_real_train
            for rel_type in [('pm', 'spatial', 'pm'), ('pm', 'temporal', 'pm'), ('pm', 'st_fwd', 'pm')]:
                if rel_type not in data.edge_types: continue
                
                key = f"{rel_type[0]}:{rel_type[1]}:{rel_type[2]}"
                
                # Predecir vecinos dentro del conjunto candidato
                k_eff = min(k_neighbors_edges, z_real_train_pm.size(0))
                if k_eff == 0:
                    continue
                neighbor_indices_in_train = edge_gen.predict(
                    z_syn_pm, z_real_train_pm, key, topk=k_eff, force_cpu=bool(force_cpu)
                )
                
                # Mapear de vuelta a índices globales
                dst_nodes = candidate_real_indices[neighbor_indices_in_train.flatten()]
                src_nodes = torch.arange(N_old, N_old + N_new, device=device).repeat_interleave(k_eff)
                
                # Construir aristas según configuración:
                # BIDIRECCTION=1 -> bidireccionales (sintético<->real)
                # BIDIRECCTION=0 -> solo direccionales real->sintético
                if int(BIDIRECCTION) == 1:
                    # sintético -> real y real -> sintético
                    base_edges = torch.stack([src_nodes, dst_nodes])
                    new_edges = torch.cat([base_edges, base_edges.flip(0)], dim=1)
                else:
                    # solo real -> sintético
                    new_edges = torch.stack([dst_nodes, src_nodes])

                # Build edge attributes from node features (dst_x - src_x)
                try:
                    src_x_store = augmented_data[rel_type[0]].x
                    dst_x_store = augmented_data[rel_type[2]].x
                    base_device = src_x_store.device
                    if dst_x_store.device != base_device:
                        dst_x_store = dst_x_store.to(base_device)
                    new_edges = new_edges.to(base_device)
                    src_x = src_x_store.index_select(0, new_edges[0].long())
                    dst_x = dst_x_store.index_select(0, new_edges[1].long())
                    delta_idx = _resolve_delta_feature_idx(augmented_data, base_device)
                    new_ea = _delta_from_node_features(src_x, dst_x, delta_idx)
                    if hasattr(augmented_data[rel_type], 'edge_attr') and augmented_data[rel_type].edge_attr is not None:
                        edge_attr_dim = augmented_data[rel_type].edge_attr.shape[1] if augmented_data[rel_type].edge_attr.dim() > 1 else 1
                        if new_ea.dim() == 1:
                            new_ea = new_ea.unsqueeze(1)
                        if new_ea.size(1) > edge_attr_dim:
                            new_ea = new_ea[:, :edge_attr_dim]
                        elif new_ea.size(1) < edge_attr_dim:
                            pad = torch.zeros(new_ea.size(0), edge_attr_dim - new_ea.size(1), device=new_ea.device, dtype=new_ea.dtype)
                            new_ea = torch.cat([new_ea, pad], dim=1)
                        new_ea = new_ea.to(dtype=augmented_data[rel_type].edge_attr.dtype, device=base_device)
                    # Append edge_index and edge_attr
                    edge_index_base = augmented_data[rel_type].edge_index.to(base_device)
                    augmented_data[rel_type].edge_index = torch.cat([edge_index_base, new_edges], dim=1)
                    if hasattr(augmented_data[rel_type], 'edge_attr') and augmented_data[rel_type].edge_attr is not None:
                        old_ea = augmented_data[rel_type].edge_attr
                        if old_ea.device != new_ea.device:
                            old_ea = old_ea.to(new_ea.device)
                        augmented_data[rel_type].edge_attr = torch.cat([old_ea, new_ea], dim=0)
                    else:
                        augmented_data[rel_type].edge_attr = new_ea
                except Exception:
                    # Fallback: append edges and rely on cleaning; edge_attr may be dropped if misaligned
                    edge_index_base = augmented_data[rel_type].edge_index.to(new_edges.device)
                    augmented_data[rel_type].edge_index = torch.cat([edge_index_base, new_edges], dim=1)
                _safe_clean_edges(augmented_data[rel_type], augmented_data[node_type].num_nodes, augmented_data[node_type].num_nodes)
        
        # Cleanup intermediate tensors from the loop
        del syn_z, syn_x, syn_labels, z_node, y_node
        if 'z_syn_pm' in locals() and 'z_syn_pm' in vars(): del z_syn_pm


    # Save augmented graph if a path is provided
    if save_path:
        save_augmented_graph(augmented_data, save_path)

    # Cleanup to reduce memory
    del z_dict
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return augmented_data.cpu()

# =========================
# Helpers de sanidad / masks
# =========================

def _assert_sanity_after_augment(data: HeteroData):
    assert 'pm' in data.node_types, "Se espera tipo de nodo 'pm'."
    N = data['pm'].num_nodes
    assert data['pm'].train_mask.shape[0] == N
    assert data['pm'].val_mask.shape[0] == N
    assert data['pm'].test_mask.shape[0] == N
    # EdgeAttr alineado (si existe)
    for et in data.edge_types:
        store = data[et]
        if hasattr(store, 'edge_attr') and store.edge_attr is not None:
            assert store.edge_attr.shape[0] == store.edge_index.shape[1], f"edge_attr desalineado en {et}"

def _drop_edge_attr_if_misaligned(store):
    ei = store.edge_index
    ea = getattr(store, 'edge_attr', None)
    if ea is None:
        return
    if ea.shape[0] != ei.shape[1]:
        # Fallback conservador: dropea edge_attr
        delattr(store, 'edge_attr')

def _coalesce_and_align_all_edges(data: HeteroData):
    for et in data.edge_types:
        store = data[et]
        ei = store.edge_index
        ea = getattr(store, 'edge_attr', None)
        
        # Pass edge_attr to remove_self_loops to keep it aligned
        ei, ea = remove_self_loops(ei, ea)
        
        m = data[et[0]].num_nodes
        n = data[et[2]].num_nodes
        
        # The `coalesce` function expects `num_nodes` as the size of the square adjacency matrix.
        # For heterogeneous graphs, we should use the max of source and destination nodes.
        num_nodes = max(m, n)

        # Pass `ea` to coalesce to correctly handle edge attributes
        ei, ea = coalesce(ei, ea, num_nodes=num_nodes, reduce='min')
        
        store.edge_index = ei
        if ea is not None:
            store.edge_attr = ea
        elif hasattr(store, 'edge_attr'):
            delattr(store, 'edge_attr') # remove dangling edge_attr
        
        _drop_edge_attr_if_misaligned(store)

# 2.2. Embeddings solo de TRAIN (evitar fuga)

@torch.no_grad()
def compute_train_embeddings(model, data: HeteroData, device, num_neighbors, batch_size, node_type='pm'):
    model.eval()
    num_neighbors = _normalize_num_neighbors(num_neighbors, data)
    # Get embeddings only for the training nodes
    train_nodes = data[node_type].train_mask.nonzero(as_tuple=True)[0]
    loader = NeighborLoader(
        data.cpu(), # Loader works on CPU data
        num_neighbors=num_neighbors,
        input_nodes=(node_type, train_nodes),
        batch_size=batch_size,
        shuffle=False
    )
    
    z_list = []
    for batch in tqdm(loader, desc=f"Generando embeddings para SMOTE ({node_type})"):
        batch = batch.to(device)
        
        # The model forward pass returns embeddings for all nodes in the batch (inputs + neighbors)
        edge_attr_dict = {et: batch[et].edge_attr for et in batch.edge_types if 'edge_attr' in batch[et]}
        _, z_b, _ = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        
        # We only want the embeddings for the input nodes of the batch
        if node_type in z_b:
            z_list.append(z_b[node_type][:batch[node_type].batch_size].detach().cpu())

    # This will fail if z_list is empty
    if not z_list:
        # This case should ideally not be hit if train_nodes is not empty
        z_dict = {node_type: torch.empty(0, model.hidden_channels * model.num_heads, device='cpu')}
    else:
        z_cat = torch.cat(z_list, dim=0)
        # L2-normalizar para coincidir con get_embeddings_minibatch (el decoder z->x
        # se entrena sobre embeddings unitarios; mezclar normalizados y no normalizados
        # en ramas distintas del pipeline produce desajuste de distribución).
        z_cat = F.normalize(z_cat, p=2, dim=1, eps=1e-12)
        z_dict = {node_type: z_cat}

    # We only care about the target node type's embeddings for this function's purpose
    final_z_dict = {node_type: z_dict.get(node_type)}

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return final_z_dict

# 2.3. Síntesis de nodos y actualización de masks (modo agnóstico)

def _synthesize_minority_nodes_from_embeddings(
    data: HeteroData,
    z_pm: torch.Tensor,
    target_pos_ratio: float,
    k: int,
    rng: np.random.RandomState,
    z2x_decoders,
    device,
    *,
    synthetic_feature_mode: Optional[str] = None,
):
    """
    Devuelve: dict con tensores nuevos para 'pm'.x, .y y un índice booleano de nuevos nodos.
    Asume binario y que la clase positiva es 1.

    Implementa la Eq. 4 del paper vía `_smote_in_z_space` con α~U[0,1] y
    re-normalización L2 de los embeddings sintéticos para que la entrada al
    decoder z->x viva en la misma hiperesfera unitaria que vio en el entrenamiento.
    """
    y_full = data['pm'].y
    train_mask = data['pm'].train_mask
    if y_full.dim() == 0 or train_mask.numel() == 0:
        logger.warning("El conjunto de entrenamiento está vacío. No se puede aplicar SMOTE.")
        return None

    y_train = y_full[train_mask].to(z_pm.device)
    n_train = int(y_train.numel())

    if n_train == 0:
        logger.warning("El conjunto de entrenamiento está vacío. No se puede aplicar SMOTE.")
        return None

    n_pos = int((y_train == 1).sum().item())
    n_target_pos = int(np.ceil(target_pos_ratio * n_train))
    n_to_add = max(0, n_target_pos - n_pos)

    if n_to_add == 0:
        logger.info("El ratio de positivos ya es igual o mayor al objetivo. No se añaden nodos.")
        return None

    # z_pm viene de compute_train_embeddings (alineado con train_mask).
    # Renormalizar para coherencia con get_embeddings_minibatch (decoder fue entrenado
    # sobre embeddings unitarios).
    z_train = F.normalize(z_pm, p=2, dim=1, eps=1e-12)

    syn_z, syn_labels, parent_info = _smote_in_z_space(
        z_train.to(z_pm.device),
        y_train,
        minority_class=1,
        k=int(k),
        n_samples=int(n_to_add),
        rng=rng,
        return_renormalized=True,
        return_parent_info=True,
    )

    if syn_z.numel() == 0:
        return None

    syn_z = syn_z.to(device)

    x_train = data['pm'].x[train_mask].to(device)
    x_syn, feature_quality = _project_synthetic_features_from_smote(
        node_type='pm',
        syn_z=syn_z,
        x_source=x_train,
        y_source=y_train.to(device),
        parent_info=parent_info,
        minority_class=1,
        z2x_decoders=z2x_decoders,
        feature_mode=synthetic_feature_mode,
    )
    logger.info("[GraphSMOTE] synthetic feature quality (pm): %s", feature_quality)
    x_syn = x_syn.cpu()

    y_syn = syn_labels.to(dtype=data['pm'].y.dtype, device='cpu')

    return {
        'x_syn': x_syn,
        'y_syn': y_syn,
        'n_new': int(syn_z.size(0)),
        'z_syn': syn_z.detach().cpu(),
        'smote_parent_base_idx': parent_info['base_idx'].detach().cpu(),
        'smote_parent_neighbor_idx': parent_info['neighbor_idx'].detach().cpu(),
        'smote_alpha': parent_info['alpha'].detach().cpu(),
        'synthetic_feature_mode': _resolve_synthetic_feature_mode(synthetic_feature_mode),
        'synthetic_feature_quality': feature_quality,
    }

# 2.4. Generación de aristas para sintéticos (usando tu edge generator)

def _generate_edges_for_synthetics(
    model,
    aug_data: HeteroData,
    edge_gen,
    device,
    node_type='pm',
    topK=10,
    force_cpu: bool = True,
    edge_attr_decoder: Optional["RelEdgeAttrDecoder"] = None,
):
    """
    Para cada sintético, conecta con topK reales/vecinos probables por cada relación que involucre 'pm'.

    Si `edge_attr_decoder` se pasa, los atributos de las nuevas aristas se generan
    desde [z_src, z_dst] vía el decoder entrenado con aristas reales (recomendado).
    Sin decoder, cae al cómputo delta entre features de nodos (legacy) y, ante
    fallo, a ceros — lo que sesgaba al modelo cuando edge_attr_dim > delta_dim.
    """
    if edge_gen is None:
        return  # si no tienes generador, puedes conectar heurísticamente por k-NN en z

    aug_data = aug_data.to(device)
    with torch.no_grad():
        # Nodos de entrenamiento (reales) candidatos para conexión
        if int(GRAPHSMOTE_CONECT) == 1:
            train_idx = aug_data[node_type].train_mask.nonzero(as_tuple=True)[0]
        else:
            train_idx = _first_hop_neighbors_of_accidents(
                aug_data,
                node_type=node_type,
                use_train_only=True,
                device=aug_data[node_type].x.device,
                log_context="_generate_edges_for_synthetics",
            )
        # Nodos sintéticos que necesitan aristas
        syn_idx   = aug_data[node_type].is_synthetic.nonzero(as_tuple=True)[0]
        
        if syn_idx.numel() == 0 or train_idx.numel() == 0:
            return

        # embeddings para edge_gen
        # Se necesita recalcular embeddings sobre el grafo aumentado
        model.eval()
        _, z_dict, _ = model(aug_data.x_dict, aug_data.edge_index_dict, getattr(aug_data, "edge_attr_dict", {}))
        z = z_dict[node_type]

        for (src, rel, dst) in aug_data.edge_types:
            if src != node_type or dst != node_type:
                continue
            
            key = f"{src}:{rel}:{dst}"
            
            # topK por fila
            k_actual = min(topK, train_idx.numel())
            if k_actual == 0:
                continue
            
            topi = edge_gen.predict(
                z[syn_idx], z[train_idx], key, topk=k_actual, force_cpu=bool(force_cpu)
            )
            dst_sel = train_idx[topi]                                # indices reales de training
            src_sel = syn_idx.unsqueeze(1).expand_as(dst_sel)        # broadcast sintéticos

            # evitar auto-loops por seguridad (un sintético no debe conectarse a sí mismo)
            mask = (src_sel != dst_sel)
            src_sel = src_sel[mask]
            dst_sel = dst_sel[mask]

            if src_sel.numel() == 0:
                continue

            e_store = aug_data[(src, rel, dst)]
            # Construir aristas segun BIDIRECCTION
            if int(BIDIRECCTION) == 1:
                # sintético -> real y (si simétrica) real -> sintético
                new_ei = torch.stack([src_sel.flatten(), dst_sel.flatten()], dim=0)
                if src == dst:
                    new_ei = torch.cat([new_ei, new_ei.flip(0)], dim=1)
            else:
                # solo real -> sintético
                new_ei = torch.stack([dst_sel.flatten(), src_sel.flatten()], dim=0)

            # Construye atributos de las nuevas aristas. Prioridad:
            #   1. RelEdgeAttrDecoder (si fue pasado): predice [E, edge_attr_dim]
            #      desde [z_src ⊕ z_dst]. Cubre el ancho completo de edge_attr.
            #   2. Delta de features (legacy): dst_x - src_x truncado/padeado a
            #      edge_attr_dim. Si delta_dim < edge_attr_dim, los faltantes
            #      quedaban en cero — gap conocido que el decoder cierra.
            #   3. Zeros, si todo falla.
            new_ea = None
            base_device = aug_data[src].x.device
            new_ei = new_ei.to(base_device)

            # Ruta 1: decoder de edge_attr
            if edge_attr_decoder is not None and key in edge_attr_decoder.heads:
                try:
                    z_src_e = z[new_ei[0].long()]
                    z_dst_e = z[new_ei[1].long()]
                    ea = edge_attr_decoder.predict(z_src_e, z_dst_e, key)
                    if hasattr(e_store, 'edge_attr') and e_store.edge_attr is not None:
                        expected_dim = e_store.edge_attr.shape[1] if e_store.edge_attr.dim() > 1 else 1
                        got_dim = ea.shape[1] if ea.dim() > 1 else 1
                        if got_dim != expected_dim:
                            raise ValueError(
                                f"edge_attr_decoder produjo dim {got_dim}; se esperaba {expected_dim}"
                            )
                    if hasattr(e_store, 'edge_attr') and e_store.edge_attr is not None:
                        ea = ea.to(dtype=e_store.edge_attr.dtype, device=base_device)
                    new_ea = ea
                except Exception as exc:
                    logger.warning(
                        f"[SMOTE] edge_attr_decoder falló para '{key}' "
                        f"({type(exc).__name__}: {exc}); usando ruta delta."
                    )
                    new_ea = None

            # Ruta 2: delta legacy (dst_x - src_x)
            if new_ea is None:
                try:
                    src_x_store = aug_data[src].x
                    dst_x_store = aug_data[dst].x
                    if dst_x_store.device != base_device:
                        dst_x_store = dst_x_store.to(base_device)
                    src_x = src_x_store.index_select(0, new_ei[0].long())
                    dst_x = dst_x_store.index_select(0, new_ei[1].long())
                    delta_idx = _resolve_delta_feature_idx(aug_data, base_device)
                    ea = _delta_from_node_features(src_x, dst_x, delta_idx)

                    if hasattr(e_store, 'edge_attr') and e_store.edge_attr is not None:
                        edge_attr_dim = e_store.edge_attr.shape[1] if e_store.edge_attr.dim() > 1 else 1
                        if ea.dim() == 1:
                            ea = ea.unsqueeze(1)
                        if ea.size(1) > edge_attr_dim:
                            ea = ea[:, :edge_attr_dim]
                        elif ea.size(1) < edge_attr_dim:
                            pad = torch.zeros(ea.size(0), edge_attr_dim - ea.size(1), device=ea.device, dtype=ea.dtype)
                            ea = torch.cat([ea, pad], dim=1)
                        ea = ea.to(dtype=e_store.edge_attr.dtype, device=base_device)
                    new_ea = ea
                except Exception:
                    # Ruta 3: ceros explícitos (último recurso).
                    if hasattr(e_store, 'edge_attr') and e_store.edge_attr is not None:
                        edge_attr_dim = e_store.edge_attr.shape[1] if e_store.edge_attr.dim() > 1 else 1
                        new_ea = torch.zeros((new_ei.shape[1], edge_attr_dim), dtype=e_store.edge_attr.dtype, device=new_ei.device)
            
            if hasattr(e_store, 'edge_index') and e_store.edge_index.numel() > 0:
                edge_index_base = e_store.edge_index
                if edge_index_base.device != new_ei.device:
                    edge_index_base = edge_index_base.to(new_ei.device)
                e_store.edge_index = torch.cat([edge_index_base, new_ei], dim=1)
                if new_ea is not None:
                    if hasattr(e_store, 'edge_attr') and e_store.edge_attr is not None:
                        edge_attr_base = e_store.edge_attr
                        if edge_attr_base.device != new_ea.device:
                            edge_attr_base = edge_attr_base.to(new_ea.device)
                        e_store.edge_attr = torch.cat([edge_attr_base, new_ea], dim=0)
                    else:
                        e_store.edge_attr = new_ea
            else:
                e_store.edge_index = new_ei
                if new_ea is not None:
                    e_store.edge_attr = new_ea
            
            _safe_clean_edges(e_store, aug_data[src].num_nodes, aug_data[dst].num_nodes)

# 2.5. Aumento offline-once (todo junto)

def augment_graph_offline_once(
    model,
    data: HeteroData,
    device,
    target_pos_ratio: float,
    z2x_decoders,
    k: int,
    edge_gen=None,
    save_path: str = None,
    seed: int = 0,
    num_neighbors=EMB_NUM_NEIGHBORS,
    edge_attr_decoder: Optional["RelEdgeAttrDecoder"] = None,
    synthetic_feature_mode: Optional[str] = None,
):
    rng = np.random.RandomState(seed)
    try:
        mode = "bidirectional (real <-> synthetic)" if int(BIDIRECCTION) == 1 else "directed (real -> synthetic)"
        logger.info(f"[GraphSMOTE] Edge direction mode: {mode} (BIDIRECCTION={BIDIRECCTION})")
    except Exception:
        pass
    # 1) Embeddings SOLO train
    z_dict = compute_train_embeddings(
        model, data, device,
        num_neighbors=num_neighbors,
        batch_size=EMB_BATCH_SIZE,
        node_type='pm'
    )
    z_pm = z_dict['pm'] 

    # 2) Síntesis
    syn = _synthesize_minority_nodes_from_embeddings(
        data,
        z_pm,
        target_pos_ratio,
        k,
        rng,
        z2x_decoders,
        device,
        synthetic_feature_mode=synthetic_feature_mode,
    )
    if syn is None:
        if save_path:
            save_augmented_graph(data, save_path)
        return data, {'pm': {'n_new': 0}}

    # 3) Construye grafo aumentado
    aug = data.clone()
    if hasattr(data, "delta_feature_idx"):
        try:
            aug.delta_feature_idx = data.delta_feature_idx
        except Exception:
            pass

    N = aug['pm'].num_nodes
    x_new = torch.cat([aug['pm'].x.cpu(), syn['x_syn'].to(dtype=aug['pm'].x.dtype)], dim=0)
    y_new = torch.cat([aug['pm'].y.cpu(), syn['y_syn']], dim=0)
    aug['pm'].x = x_new
    aug['pm'].y = y_new

    # Máscaras: solo TRAIN crece
    train_mask = aug['pm'].train_mask.cpu()
    val_mask   = aug['pm'].val_mask.cpu()
    test_mask  = aug['pm'].test_mask.cpu()
    
    # Manejo seguro de 'is_accident_pm'
    if hasattr(aug['pm'], 'is_accident_pm'):
        is_accident_pm_mask = aug['pm'].is_accident_pm.cpu()
    else:
        # Si no existe, se crea un tensor de Falsos para los nodos originales
        is_accident_pm_mask = torch.zeros(N, dtype=torch.bool)

    add_mask = torch.ones((syn['n_new'],), dtype=torch.bool)
    aug['pm'].train_mask = torch.cat([train_mask, add_mask], dim=0)
    aug['pm'].val_mask   = torch.cat([val_mask,   torch.zeros_like(add_mask)], dim=0)
    aug['pm'].test_mask  = torch.cat([test_mask,  torch.zeros_like(add_mask)], dim=0)
    aug['pm'].is_accident_pm = torch.cat([is_accident_pm_mask, torch.zeros_like(add_mask)], dim=0)

    # flag diagnóstico para identificar nodos sintéticos
    aug['pm'].is_synthetic = torch.zeros(N, dtype=torch.bool)
    new_synthetic_flag = torch.ones(syn['n_new'], dtype=torch.bool)
    aug['pm'].is_synthetic = torch.cat([aug['pm'].is_synthetic, new_synthetic_flag])

    # 4) Conecta aristas para sintéticos
    _generate_edges_for_synthetics(
        model=model,
        aug_data=aug,
        edge_gen=edge_gen,
        device=device,
        topK=GRAPHSMOTE_K,
        edge_attr_decoder=edge_attr_decoder,
    )

    # 5) Limpieza y coalesce
    _coalesce_and_align_all_edges(aug)
    _assert_sanity_after_augment(aug)

    if save_path:
        save_augmented_graph(aug, save_path)

    new_idx = torch.zeros(aug['pm'].num_nodes, dtype=torch.bool)
    new_idx[N:] = True
    registry = {
        'pm': {
            'n_new': syn['n_new'],
            'new_idx': new_idx,
            'synthetic_feature_mode': syn.get('synthetic_feature_mode'),
            'synthetic_feature_quality': syn.get('synthetic_feature_quality', {}),
            'smote_parent_base_idx': syn.get('smote_parent_base_idx'),
            'smote_parent_neighbor_idx': syn.get('smote_parent_neighbor_idx'),
            'smote_alpha': syn.get('smote_alpha'),
        }
    }
    return aug.cpu(), registry

# 2.6. Refresco online-periódico (reemplaza sintéticos)

def refresh_synthetics_online(
    model,
    base_data: HeteroData,
    device,
    target_pos_ratio: float,
    k: int,
    z2x_decoders,
    edge_gen=None,
    seed: int = 0,
    num_neighbors=EMB_NUM_NEIGHBORS,
    edge_attr_decoder: Optional["RelEdgeAttrDecoder"] = None,
    synthetic_feature_mode: Optional[str] = None,
):
    """
    Parte siempre del grafo BASE (solo reales) y vuelve a sintetizar,
    devolviendo un grafo AUGMENTED nuevo. No acumula.
    """
    augmented, registry = augment_graph_offline_once(
        model=model,
        data=base_data,
        device=device,
        target_pos_ratio=target_pos_ratio,
        k=k,
        z2x_decoders=z2x_decoders,
        edge_gen=edge_gen,
        save_path=None,
        seed=seed,
        num_neighbors=num_neighbors,
        edge_attr_decoder=edge_attr_decoder,
        synthetic_feature_mode=synthetic_feature_mode,
    )
    return augmented, registry
def _first_hop_neighbors_of_accidents(
    data: HeteroData,
    node_type: str = 'pm',
    use_train_only: bool = True,
    device=None,
    log_context: Optional[str] = None,
):
    """Return indices of real (non-synthetic) nodes in the first-hop neighborhood of
    real accident nodes (class=1) for the given node_type.
    If use_train_only=True, restricts both seed accidents and returned neighbors to TRAIN nodes.
    """
    if device is None:
        device = next(iter(data.x_dict.values())).device

    y = data[node_type].y.to(device)
    is_synth = getattr(data[node_type], 'is_synthetic', torch.zeros(y.size(0), dtype=torch.bool, device=device)).to(device)
    train_mask = getattr(data[node_type], 'train_mask', torch.ones(y.size(0), dtype=torch.bool, device=device)).to(device)

    # Seeds: real accident nodes
    seeds_mask = (y == 1) & (~is_synth)
    if use_train_only:
        seeds_mask = seeds_mask & train_mask
    seed_idx = seeds_mask.nonzero(as_tuple=True)[0]
    if seed_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    neighbors = set()
    # Iterate over pm->pm relations
    for (src, rel, dst) in data.edge_types:
        if src != node_type or dst != node_type:
            continue
        eidx = data[(src, rel, dst)].edge_index.to(device)
        if eidx.numel() == 0:
            continue
        src_nodes = eidx[0]
        dst_nodes = eidx[1]

        # Out-neighbors of seeds
        mask_out = torch.isin(src_nodes, seed_idx)
        if mask_out.any():
            neighbors.update(dst_nodes[mask_out].tolist())

        # In-neighbors of seeds
        mask_in = torch.isin(dst_nodes, seed_idx)
        if mask_in.any():
            neighbors.update(src_nodes[mask_in].tolist())

    if not neighbors:
        # Logging (optional)
        if log_context is not None:
            logger.info(f"[GraphSMOTE] {log_context}: accidents={seed_idx.numel()}, first_hop_neighbors=0, candidates=0")
        return torch.empty(0, dtype=torch.long, device=device)
    neigh_idx = torch.tensor(sorted(neighbors), dtype=torch.long, device=device)
    # Remove seeds themselves and restrict to real nodes (and train if requested)
    mask_real = (~is_synth)
    if use_train_only:
        mask_real = mask_real & train_mask
    valid_mask = mask_real
    valid_mask[seed_idx] = False
    if valid_mask.numel() == 0:
        if log_context is not None:
            logger.info(f"[GraphSMOTE] {log_context}: accidents={seed_idx.numel()}, first_hop_neighbors={len(neighbors)}, candidates=0")
        return torch.empty(0, dtype=torch.long, device=device)
    filtered = neigh_idx[valid_mask[neigh_idx]]
    if log_context is not None:
        logger.info(f"[GraphSMOTE] {log_context}: accidents={seed_idx.numel()}, first_hop_neighbors={len(neighbors)}, candidates={filtered.numel()}")
    return filtered
