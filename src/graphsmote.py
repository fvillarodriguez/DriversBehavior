import os
from typing import Optional, Union, List, Tuple
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
from src.smote_cpu_cache import get_knn_cached, simple_smote_from_knn
logger = logging.getLogger(__name__)

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
):
    """
    Entrena un decodificador z->x por tipo usando z extraído en modo eval.
    Guarda pesos por tipo en save_dir y retorna el Z2XDecoders entrenado.
    """
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device
    
    z_dict = get_embeddings_minibatch(
        model, data, num_neighbors=num_neighbors
    )

    # dims por tipo
    dims_per_type = {}
    for ntype, z in z_dict.items():
        if (node_types is None) or (ntype in node_types):
            if z.nelement() > 0:
                z_dim = z.size(1)
                x_dim = data[ntype].x.size(1)
                dims_per_type[ntype] = (z_dim, x_dim)

    dec = Z2XDecoders(dims_per_type).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.SmoothL1Loss() if loss_type == "huber" else torch.nn.MSELoss()

    # Precompute device tensors and masks per type
    z_all = {}
    x_all = {}
    train_idx = {}
    val_idx = {}
    has_any_val = False

    rng = np.random.RandomState(SEED)

    for ntype in dims_per_type.keys():
        z_all[ntype] = z_dict[ntype].to(device)
        x_all[ntype] = data[ntype].x.to(device)

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

        # Ensure disjoint
        m_train = m_train_base & (~m_val)

        train_idx[ntype] = torch.nonzero(m_train, as_tuple=True)[0]
        val_idx[ntype] = torch.nonzero(m_val, as_tuple=True)[0]
        has_any_val = has_any_val or (val_idx[ntype].numel() > 0)

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

        for ntype in dims_per_type.keys():
            if train_idx[ntype].numel() == 0:
                continue
            z = z_all[ntype].index_select(0, train_idx[ntype])
            x = x_all[ntype].index_select(0, train_idx[ntype])
            xhat = dec.project_one(ntype, z)
            loss = crit(xhat, x)
            loss.backward()
            total_train_loss += float(loss.item())

        opt.step()

        # Validation
        total_val_loss = None
        if early_stop and has_any_val:
            dec.eval()
            with torch.no_grad():
                vloss = 0.0
                vcount = 0
                for ntype in dims_per_type.keys():
                    if val_idx[ntype].numel() == 0:
                        continue
                    z = z_all[ntype].index_select(0, val_idx[ntype])
                    x = x_all[ntype].index_select(0, val_idx[ntype])
                    xhat = dec.project_one(ntype, z)
                    loss = crit(xhat, x)
                    vloss += float(loss.item())
                    vcount += 1
                if vcount > 0:
                    total_val_loss = vloss / vcount
                else:
                    total_val_loss = None

            # Early stopping check
            if total_val_loss is not None:
                improved = (best_val - total_val_loss) > min_delta
                if improved:
                    best_val = total_val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Append to history
        history.append({
            "epoch": epoch,
            "train_loss": total_train_loss,
            "val_loss": total_val_loss if total_val_loss is not None else None
        })

        if (log_every is not None) and (log_every > 0) and (epoch % log_every == 0):
            if show_progress and hasattr(epoch_iter, 'set_postfix'):
                if total_val_loss is not None:
                    epoch_iter.set_postfix({
                        'train': f"{total_train_loss:.4f}",
                        'val': f"{total_val_loss:.4f}",
                        'best': f"{best_val if best_val!=float('inf') else float('nan'):.4f}"
                    })
                else:
                    logger.info(f"[Z2X] Epoch {epoch:03d}/{epochs} | train_loss={total_train_loss:.4f}")

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
                dec.heads[ntype].load_state_dict(torch.load(p, map_location=device))
                loaded.add(ntype)
            except RuntimeError as e:
                print(f"Warning: could not load state_dict for ntype '{ntype}'. It might be retrained if necessary. Error: {e}")
    dec.eval()
    dec._loaded_types = loaded
    return dec

# ============================================================
# 4) SMOTE en espacio z (corregido: semillas NumPy + random)
# ============================================================
def _knn_torch_chunked_device(
    X: torch.Tensor,
    k: int,
    *,
    metric: str = "cosine",
    q_block: int = 2048,
    r_block: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked kNN on the same device as X (GPU/MPS/CPU)."""
    assert metric in {"cosine", "euclidean"}
    Xd = X.detach().contiguous()
    N, _ = Xd.shape
    k1 = k + 1

    top_val = torch.full((N, k1), float("-inf"), device=Xd.device)
    top_idx = torch.full((N, k1), -1, dtype=torch.long, device=Xd.device)

    if metric == "cosine":
        Xn = torch.nn.functional.normalize(Xd, p=2.0, dim=1)
        for q0 in range(0, N, q_block):
            q1 = min(N, q0 + q_block)
            Q = Xn[q0:q1]
            cur_val = top_val[q0:q1]
            cur_idx = top_idx[q0:q1]
            for r0 in range(0, N, r_block):
                r1 = min(N, r0 + r_block)
                R = Xn[r0:r1]
                sim = Q @ R.T
                cand_val = torch.cat([cur_val, sim], dim=1)
                cand_idx = torch.cat(
                    [cur_idx, torch.arange(r0, r1, device=Xd.device).expand(q1 - q0, -1)],
                    dim=1,
                )
                new_val, ord_idx = torch.topk(cand_val, k1, largest=True, dim=1)
                cur_val = new_val
                cur_idx = torch.gather(cand_idx, 1, ord_idx)
                del sim, cand_val, cand_idx, new_val, ord_idx
            top_val[q0:q1] = cur_val
            top_idx[q0:q1] = cur_idx
        row_ids = torch.arange(N, device=Xd.device).view(-1, 1).expand(-1, k1)
        cur_val = top_val.clone()
        cur_val[top_idx == row_ids] = float("-inf")
        final_val, ord_idx = torch.topk(cur_val, k, largest=True, dim=1)
        final_idx = torch.gather(top_idx, 1, ord_idx)
        dist = 1.0 - final_val.clamp(-1, 1)
        return final_idx.long(), dist.float()

    X2 = (Xd * Xd).sum(dim=1, keepdim=True)
    for q0 in range(0, N, q_block):
        q1 = min(N, q0 + q_block)
        Q = Xd[q0:q1]
        Q2 = X2[q0:q1]
        cur_val = top_val[q0:q1]
        cur_idx = top_idx[q0:q1]
        for r0 in range(0, N, r_block):
            r1 = min(N, r0 + r_block)
            R = Xd[r0:r1]
            R2 = X2[r0:r1].T
            d2 = Q2 + R2 - 2.0 * (Q @ R.T)
            score = -d2
            cand_val = torch.cat([cur_val, score], dim=1)
            cand_idx = torch.cat(
                [cur_idx, torch.arange(r0, r1, device=Xd.device).expand(q1 - q0, -1)],
                dim=1,
            )
            new_val, ord_idx = torch.topk(cand_val, k1, largest=True, dim=1)
            cur_val = new_val
            cur_idx = torch.gather(cand_idx, 1, ord_idx)
            del d2, score, cand_val, cand_idx, new_val, ord_idx
        top_val[q0:q1] = cur_val
        top_idx[q0:q1] = cur_idx
    row_ids = torch.arange(N, device=Xd.device).view(-1, 1).expand(-1, k1)
    cur_val = top_val.clone()
    cur_val[top_idx == row_ids] = float("-inf")
    final_val, ord_idx = torch.topk(cur_val, k, largest=True, dim=1)
    final_idx = torch.gather(top_idx, 1, ord_idx)
    dist2 = -final_val
    dist = torch.sqrt(torch.clamp_min(dist2, 0.0))
    return final_idx.long(), dist.float()


def _simple_smote_from_knn_device(
    pos_emb: torch.Tensor,
    idx: torch.Tensor,
    num_new: int,
    *,
    alpha: float = 0.5,
) -> torch.Tensor:
    N, d = pos_emb.shape
    if num_new <= 0:
        return torch.empty((0, d), dtype=pos_emb.dtype, device=pos_emb.device)
    base = torch.randint(0, N, (num_new,), device=pos_emb.device)
    nbrs = torch.randint(0, idx.size(1), (num_new,), device=pos_emb.device)
    j = idx[base, nbrs]
    X = pos_emb.detach()
    synth = (1 - alpha) * X[base] + alpha * X[j]
    return synth


def smote_nodes(
    z,
    y,
    minority_class=1,
    k=5,
    n_samples=100,
    random_state=SEED,
    *,
    force_cpu: bool = True,
    gpu_knn_max_n: int = 5000,
    gpu_knn_block: int = 2048,
):
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

    minority_mask = (y == minority_class)
    pos_emb = z[minority_mask]

    if pos_emb.shape[0] < k or pos_emb.shape[0] == 0:
        logger.warning(f"Warning: # of minority samples ({pos_emb.shape[0]}) < k ({k}). Cannot apply SMOTE.")
        return torch.empty(0, pos_emb.shape[1], device=z.device), torch.empty(0, dtype=torch.long, device=y.device), None

    # --- k-NN (CPU cache or device, depending on toggle) ---
    extra = {
        "dataset": "unknown",  # placeholder for caching signature
        "seed": int(random_state if random_state is not None else 0),
        "version": "v1",
    }
    cache_dir = os.path.join(".", "cache", "graphsmote")

    use_device_knn = (
        (not force_cpu)
        and pos_emb.device.type != "cpu"
        and pos_emb.shape[0] <= int(gpu_knn_max_n)
    )
    idx = None
    if use_device_knn:
        try:
            idx, _ = _knn_torch_chunked_device(
                pos_emb, k, metric="cosine", q_block=int(gpu_knn_block), r_block=int(gpu_knn_block)
            )
            synthetic_features = _simple_smote_from_knn_device(
                pos_emb, idx, num_new=n_samples, alpha=0.5
            )
            synthetic_features = synthetic_features.to(device=z.device, dtype=z.dtype)
        except Exception as e:
            logger.warning(f"[GraphSMOTE] GPU kNN failed ({e}); falling back to CPU.")
            idx = None

    if idx is None:
        idx, _ = get_knn_cached(
            pos_emb=pos_emb.to("cpu"),
            k=k,
            cache_dir=cache_dir,
            extra_params=extra,
            metric="cosine",
        )
        # Generar sintéticos en CPU y mover al device del modelo
        synthetic_features = simple_smote_from_knn(
            pos_emb.to("cpu"), idx, num_new=n_samples, alpha=0.5
        )
        synthetic_features = synthetic_features.to(device=z.device, dtype=z.dtype)
    # --- Fin k-NN ---
    
    synthetic_labels = torch.full((n_samples,), minority_class, dtype=torch.long, device=y.device)
    
    return synthetic_features, synthetic_labels, None

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
        try:
            if hasattr(data[node_type], "train_mask"):
                train_mask = data[node_type].train_mask.to(z_node.device)
                if train_mask.dtype != torch.bool:
                    train_mask = train_mask.bool()
                train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1)
                if train_idx.numel() > 0:
                    z_smote = z_node.index_select(0, train_idx)
                    y_smote = y_node.index_select(0, train_idx)
        except Exception:
            # Si falla, se usa el conjunto completo por compatibilidad
            z_smote = z_node
            y_smote = y_node

        syn_z, syn_labels, minority_indices = smote_nodes(
            z_smote, y_smote,
            minority_class=minority_class,
            k=k_smote,
            n_samples=n_samples,
            random_state=random_state,
            force_cpu=bool(force_cpu),
        )

        if syn_z.numel() == 0:
            print(f"GraphSMOTE did not generate any new {node_type} samples.")
            continue

        with torch.no_grad():
            syn_x = z2x_decoders.project_one(node_type, syn_z)

        if syn_x.numel() == 0:
            print(f"GraphSMOTE did not generate any new {node_type} samples after decoding.")
            continue

        N_old = augmented_data[node_type].num_nodes
        N_new = syn_x.size(0)

        # features y labels (alinear dispositivo con los tensores base)
        base_x = augmented_data[node_type].x
        base_y = augmented_data[node_type].y
        syn_x = syn_x.to(base_x.device)
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
        del syn_z, syn_x, syn_labels, minority_indices, z_node, y_node
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
        z_dict = {node_type: torch.cat(z_list, dim=0)}

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
    device
):
    """
    Devuelve: dict con tensores nuevos para 'pm'.x, .y y un índice booleano de nuevos nodos.
    Asume binario y que la clase positiva es 1.
    """
    y = data['pm'].y.cpu().numpy().astype(int)
    train_mask = data['pm'].train_mask.cpu().numpy().astype(bool)
    y_train = y[train_mask]
    n_train = y_train.shape[0]
    
    if n_train == 0:
        logger.warning("El conjunto de entrenamiento está vacío. No se puede aplicar SMOTE.")
        return None

    n_pos = y_train.sum()
    n_target_pos = int(np.ceil(target_pos_ratio * n_train))
    n_to_add = max(0, n_target_pos - n_pos)
    
    if n_to_add == 0:
        logger.info("El ratio de positivos ya es igual o mayor al objetivo. No se añaden nodos.")
        return None

    # Get embeddings of training nodes
    z_train = z_pm
    
    # Get indices of minority class samples IN THE TRAINING SET
    minority_idx_in_train = np.where(y_train == 1)[0]
    
    if len(minority_idx_in_train) == 0:
        logger.warning("No se encontraron muestras de clase minoritaria en el conjunto de entrenamiento. No se puede aplicar SMOTE.")
        return None
    
    if len(minority_idx_in_train) < k:
        logger.warning(f"Número de muestras minoritarias ({len(minority_idx_in_train)}) es menor que k ({k}). Se usará k={len(minority_idx_in_train)}.")
        k = len(minority_idx_in_train)

    # Get embeddings of only the minority samples
    z_train_minority = z_train[minority_idx_in_train]
    
    # Fit NearestNeighbors on minority samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(z_train_minority.numpy())
    indices = nbrs.kneighbors(z_train_minority.numpy(), return_distance=False)

    syn_z_list = []
    for _ in range(n_to_add):
        # 1. Choose a random minority sample (index into the minority array)
        base_idx = rng.choice(len(minority_idx_in_train))
        
        # 2. Choose one of its k-nearest neighbors (index into the minority array)
        neighbor_relative_idx = rng.choice(indices[base_idx])

        # 3. Get the embeddings of the sample and its neighbor
        base_z = z_train_minority[base_idx]
        neighbor_z = z_train_minority[neighbor_relative_idx]

        # 4. Create synthetic sample
        alpha = rng.rand()
        z_syn = alpha * base_z + (1 - alpha) * neighbor_z
        syn_z_list.append(z_syn.unsqueeze(0))

    if not syn_z_list:
        return None

    z_syn = torch.cat(syn_z_list, dim=0).to(device)

    # Decode back to feature space
    with torch.no_grad():
        x_syn = z2x_decoders.project_one('pm', z_syn).cpu()

    y_syn = torch.ones((n_to_add,), dtype=data['pm'].y.dtype)
    
    return {
        'x_syn': x_syn,
        'y_syn': y_syn,
        'n_new': n_to_add
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
):
    """
    Para cada sintético, conecta con topK reales/vecinos probables por cada relación que involucre 'pm'.
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

            # Build edge attributes from node features (dst_x - src_x),
            # mirroring how base edges were created. This leverages that
            # synthetic node features were decoded from embeddings.
            new_ea = None
            try:
                src_x_store = aug_data[src].x
                dst_x_store = aug_data[dst].x
                base_device = src_x_store.device
                if dst_x_store.device != base_device:
                    dst_x_store = dst_x_store.to(base_device)
                new_ei = new_ei.to(base_device)
                src_x = src_x_store.index_select(0, new_ei[0].long())
                dst_x = dst_x_store.index_select(0, new_ei[1].long())
                delta_idx = _resolve_delta_feature_idx(aug_data, base_device)
                ea = _delta_from_node_features(src_x, dst_x, delta_idx)

                # Align dtype and dimensionality with existing edge_attr (if any)
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
                # Fallback: if something goes wrong, preserve previous behavior (zeros)
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
        data, z_pm, target_pos_ratio, k, rng, z2x_decoders, device
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
    x_new = torch.cat([aug['pm'].x.cpu(), syn['x_syn']], dim=0)
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
    _generate_edges_for_synthetics(model=model, aug_data=aug, edge_gen=edge_gen, device=device, topK=GRAPHSMOTE_K)

    # 5) Limpieza y coalesce
    _coalesce_and_align_all_edges(aug)
    _assert_sanity_after_augment(aug)

    if save_path:
        save_augmented_graph(aug, save_path)

    new_idx = torch.zeros(aug['pm'].num_nodes, dtype=torch.bool)
    new_idx[N:] = True
    registry = {
        'pm': {'n_new': syn['n_new'], 'new_idx': new_idx}
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
