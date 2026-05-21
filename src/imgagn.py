
import math
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import SEED, EMB_BATCH_SIZE, EMB_NUM_NEIGHBORS, BIDIRECCTION
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm


# --------- Utils ---------

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _neighbor_loader_available() -> bool:
    try:
        from torch_geometric.loader import NeighborLoader  # noqa: F401
    except Exception:
        return False
    try:
        import pyg_lib  # type: ignore  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import torch_sparse  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False

def to_homogeneous_if_needed(data, target_ntype: Optional[str] = None):
    """
    Accepts either torch_geometric.data.Data or HeteroData.
    If HeteroData, it extracts a homogeneous view centered on target_ntype nodes:
      - Keeps only edges (target_ntype, *, target_ntype) if present; otherwise, builds a kNN over X for structure.
      - Returns (hom_data, mapping) where mapping carries original indices for target_ntype.
    """
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.utils import to_undirected
    if isinstance(data, Data):
        return data, None

    assert isinstance(data, HeteroData), "data must be Data or HeteroData"
    assert target_ntype is not None, "For HeteroData, provide target_ntype"

    x = data[target_ntype].x
    y = data[target_ntype].y if 'y' in data[target_ntype] else None
    # Prefer an existing homogeneous edge set between target nodes
    # Search any edge type where src==dst==target_ntype
    eidx = None
    eattr = None
    for (src, _, dst), store in data.edge_items():
        if src == target_ntype and dst == target_ntype and 'edge_index' in store:
            eidx = store['edge_index']
            try:
                eattr = store['edge_attr'] if 'edge_attr' in store else None
            except Exception:
                eattr = None
            break

    if eidx is None:
        # Build a simple kNN structure over features if needed
        # (lightweight; you can replace with your graph's actual topology if desired)
        k = min(10, x.size(0) - 1) if x.size(0) > 1 else 0
        if k > 0:
            # cosine similarity kNN (no weights)
            x_norm = F.normalize(x, p=2, dim=-1)
            sims = x_norm @ x_norm.t()
            sims.fill_diagonal_(-1.0)
            topk = sims.topk(k=k, dim=1).indices
            rows = torch.arange(x.size(0), device=x.device).view(-1, 1).repeat(1, k)
            eidx = torch.stack([rows.reshape(-1), topk.reshape(-1)], dim=0)
        else:
            eidx = torch.empty(2, 0, dtype=torch.long, device=x.device)
        eidx = to_undirected(eidx, num_nodes=x.size(0))

    hom = type('Dummy', (), {})()  # simple struct-like
    from torch_geometric.data import Data
    hom = Data(x=x, edge_index=eidx, y=y)
    if eattr is not None:
        try:
            if eattr.size(0) == eidx.size(1):
                hom.edge_attr = eattr
        except Exception:
            pass
    # Preserve optional delta feature indices if provided in the original data
    try:
        if hasattr(data, "delta_feature_idx"):
            hom.delta_feature_idx = data.delta_feature_idx
    except Exception:
        pass
    return hom, {'target_ntype': target_ntype}

# --------- Generator (GraphGenerator) ---------

class ImGAGNGenerator(nn.Module):
    """
    GraphGenerator G: maps noise Z -> row logits (for softmax) over TRAIN minority nodes.
    Each generated node i gets a distribution over real minority nodes (weights T_i*).
    Features for node i: Xg[i] = sum_j softmax(logits[i,j]) * X_min[j].
    Edges: connect i to its top-k minority neighbors by weight.
    """
    def __init__(self, dz: int, n_min_train: int, hidden: int = 200, n_hidden: int = 1):
        super().__init__()
        layers = []
        in_dim = dz
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, n_min_train)]  # output row logits
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        # Returns raw logits O (before row-wise softmax)
        return self.mlp(z)  # [ng, n_min_train]

    @torch.no_grad()
    def build(self, z: Tensor, x_min: Tensor, topk: int = 5) -> Tuple[Tensor, Tensor]:
        """Produce (Xg, links) from z and real minority features x_min.
        links: LongTensor [2, E_add] with edges from generated node indices (offset added outside) to chosen minority nodes.
        """
        logits = self.forward(z)                              # [ng, n_min_train]
        T = torch.softmax(logits, dim=-1)                     # [ng, n_min_train]
        Xg = T @ x_min                                        # [ng, F]

        # Top-k linking to avoid dense edges
        k = min(topk, T.size(1)) if T.size(1) > 0 else 0
        if k == 0:
            links = torch.empty(2, 0, dtype=torch.long, device=T.device)
            return Xg, links

        idx = T.topk(k=k, dim=1).indices                      # [ng, k]
        rows = torch.arange(T.size(0), device=T.device).view(-1, 1).repeat(1, k)
        links = torch.stack([rows.reshape(-1), idx.reshape(-1)], dim=0)  # indices in (generated, real_min_train)
        return Xg, links

# --------- Discriminator (GCN + two heads) ---------

class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_dim, hid, cached=False, normalize=True)
        self.conv2 = GCNConv(hid, out_dim, cached=False, normalize=True)
        self.norm1 = nn.LayerNorm(hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.conv1(x, edge_index); h = self.norm1(h); h = F.relu(h); h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h  # embedding

class ImGAGNDiscriminator(nn.Module):
    """Two-head discriminator D:
      - head_real: real(0)/fake(1)
      - head_minor: majority(0)/minority(1)
    Output embeddings h are used for distance regularizers.
    """
    def __init__(self, in_dim: int, hid: int = 128, emb_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        # TODO: swap encoder to your HeteroGAT if you prefer (keep emb dims consistent)
        self.encoder = GCNEncoder(in_dim, hid, emb_dim, dropout)
        self.head_real = nn.Linear(emb_dim, 1)
        self.head_minor = nn.Linear(emb_dim, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(x, edge_index)
        logit_real = self.head_real(h).squeeze(-1)    # higher -> fake (we'll use BCEWithLogitsLoss)
        logit_minor = self.head_minor(h).squeeze(-1)  # higher -> minority
        return h, logit_real, logit_minor

# --------- Core training ---------

@dataclass
class ImGAGNConfig:
    dz: int = 100
    hidden_g: int = 200
    n_hidden_g: int = 1
    topk_links: int = 5
    emb_dim: int = 128
    hid_d: int = 128
    dropout: float = 0.2
    lambda1_ratio: float = 1.0     # ratio of (train minority + generated) / train majority
    d_steps: int = 50              # discriminator steps per generator step (paper suggests ~100)
    epochs: int = 100
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    wd: float = 5e-4
    alpha_reg: float = 1e-4        # L2 on G
    beta_reg: float = 1e-4         # L2 on D
    margin_mm: float = 1.0         # for minority/majority separation term
    grad_clip: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Mini-batch settings (NeighborLoader)
    use_neighbor_loader: bool = True
    batch_size: int = EMB_BATCH_SIZE
    num_neighbors: List[int] = None  # if None, resolves from EMB_NUM_NEIGHBORS
    # Safety: cap number of generated nodes to avoid OOM
    max_new_nodes: int = 100000
    # UI
    show_progress: bool = True

def _build_augmented_graph(
    x: Tensor,
    edge_index: Tensor,
    xg: Tensor,
    links: Tensor,
    minority_index_map: Tensor,
    edge_attr: Optional[Tensor] = None,
    delta_feature_idx: Optional[Tensor] = None,
):
    """Append generated nodes and connect them to real TRAIN minority nodes.
    minority_index_map maps link column indices -> original node indices in x.
    Returns augmented (x_aug, edge_index_aug) and index slices.
    """
    N = x.size(0)
    ng = xg.size(0)
    x_aug = torch.cat([x, xg], dim=0)  # [N+ng, F]

    # links has shape [2, E], with rows in [0..ng-1], cols in [0..n_min_train-1]
    if links.numel() == 0:
        e_add = torch.empty(2, 0, dtype=torch.long, device=x.device)
    else:
        gen_rows = links[0] + N                              # shift generated nodes indices by N
        real_cols = minority_index_map[links[1]]            # map to original node indices
        # Build edges according to config: bidirectional or directed real->synthetic
        if int(BIDIRECCTION) == 1:
            # gen -> real and real -> gen
            e_base = torch.stack([gen_rows, real_cols], dim=0)
            e_add = torch.cat([e_base, e_base.flip(0)], dim=1)
        else:
            # only real -> synthetic (dst are generated)
            e_add = torch.stack([real_cols, gen_rows], dim=0)

    from torch_geometric.utils import coalesce
    e_aug = torch.cat([edge_index, e_add], dim=1)
    edge_attr_aug = None
    if edge_attr is not None:
        try:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            if edge_attr.size(0) != edge_index.size(1):
                edge_attr = None
        except Exception:
            edge_attr = None
    if edge_attr is not None:
        edge_dim = int(edge_attr.size(1))
        delta_idx = None
        try:
            if delta_feature_idx is not None:
                if torch.is_tensor(delta_feature_idx):
                    delta_idx = delta_feature_idx.to(device=x_aug.device)
                else:
                    delta_idx = torch.tensor(
                        delta_feature_idx, dtype=torch.long, device=x_aug.device
                    )
                if delta_idx.numel() > 0:
                    delta_idx = delta_idx[
                        (delta_idx >= 0) & (delta_idx < x_aug.size(1))
                    ]
                if delta_idx.numel() == 0:
                    delta_idx = None
        except Exception:
            delta_idx = None
        # Build edge_attr for new edges as delta of node features + zeros for physical extras
        if e_add.size(1) > 0:
            src_new = e_add[0]
            dst_new = e_add[1]
            if delta_idx is not None:
                delta_new = (
                    x_aug[dst_new][:, delta_idx]
                    - x_aug[src_new][:, delta_idx]
                )
            else:
                delta_new = x_aug[dst_new] - x_aug[src_new]
            if edge_dim > delta_new.size(1):
                extras = torch.zeros(
                    (delta_new.size(0), edge_dim - delta_new.size(1)),
                    dtype=delta_new.dtype,
                    device=delta_new.device,
                )
                edge_attr_new = torch.cat([delta_new, extras], dim=1)
            elif edge_dim < delta_new.size(1):
                edge_attr_new = delta_new[:, :edge_dim]
            else:
                edge_attr_new = delta_new
        else:
            edge_attr_new = edge_attr.new_zeros((0, edge_dim))
        edge_attr_aug = torch.cat([edge_attr, edge_attr_new], dim=0)
        e_aug, edge_attr_aug = coalesce(
            e_aug,
            edge_attr_aug,
            num_nodes=N + ng,
            reduce="mean",
        )
    else:
        e_aug = coalesce(e_aug, num_nodes=N + ng)
    return x_aug, e_aug, edge_attr_aug, (slice(N, N+ng),)

def _mm_separation(h: Tensor, y_minor: Tensor, margin: float = 1.0) -> Tensor:
    """Push minority and majority embeddings apart: simple pairwise hinge.
    Uses class means to avoid O(N^2).
    """
    if y_minor.float().sum() == 0 or y_minor.float().sum() == y_minor.numel():
        return torch.tensor(0.0, device=h.device)
    h_min = h[y_minor == 1].mean(dim=0, keepdim=True)
    h_maj = h[y_minor == 0].mean(dim=0, keepdim=True)
    dist = F.pairwise_distance(h_min, h_maj, p=2).mean()
    # Maximize distance -> minimize negative distance with margin
    return F.relu(margin - dist)

def train_imgagn(
    data,
    train_mask: Tensor,
    y_binary: Tensor,
    cfg: ImGAGNConfig = ImGAGNConfig(),
    target_ntype: Optional[str] = None,
    delta_feature_idx: Optional[Tensor] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Tensor]:
    """Train ImGAGN on a (homogeneous or heterogeneous) graph for binary minority detection.
    y_binary: 0=majority, 1=minority, defined on the target node set.
    Returns: dict with model states and the augmented graph tensors.
    """
    raise RuntimeError(
        "train_imgagn homogéneo fue retirado del flujo productivo. "
        "Use src.imgagn_relational.build_relational_imgagn_graph."
    )
    set_seed(SEED)
    device = torch.device(cfg.device)

    if cfg.use_neighbor_loader and not _neighbor_loader_available():
        logger.warning(
            "[ImGAGN] NeighborLoader no disponible (requiere pyg-lib o torch-sparse). "
            "Se usara modo full-batch."
        )
        cfg.use_neighbor_loader = False

    # Log edge direction mode
    try:
        mode = "bidirectional (real <-> synthetic)" if int(BIDIRECCTION) == 1 else "directed (real -> synthetic)"
        logger.info(f"[ImGAGN] Edge direction mode: {mode} (BIDIRECCTION={BIDIRECCTION})")
    except Exception:
        pass

    # Homogeneous view
    hom, meta = to_homogeneous_if_needed(data, target_ntype)
    x: Tensor = hom.x.to(device)
    edge_index: Tensor = hom.edge_index.to(device)
    edge_attr: Optional[Tensor] = getattr(hom, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    if delta_feature_idx is None:
        delta_feature_idx = getattr(hom, "delta_feature_idx", None)
        if delta_feature_idx is None:
            delta_feature_idx = getattr(data, "delta_feature_idx", None)
    y: Tensor = y_binary.to(device)
    train_mask = train_mask.to(device)

    # TRAIN minority pool (no leakage)
    idx_train_min = (train_mask & (y == 1)).nonzero(as_tuple=False).view(-1)
    idx_train_maj = (train_mask & (y == 0)).nonzero(as_tuple=False).view(-1)
    n_min, n_maj = idx_train_min.numel(), idx_train_maj.numel()

    if n_min == 0:
        raise ValueError("No TRAIN minority nodes found. Cannot run ImGAGN.")

    # Number of generated nodes to reach desired lambda1 ratio
    # lambda1 = (n_min + ng) / n_maj  -> ng = lambda1*n_maj - n_min
    ng = max(0, int(round(cfg.lambda1_ratio * n_maj - n_min)))
    # Cap to avoid exploding augmented graph size
    if ng > cfg.max_new_nodes:
        ng = cfg.max_new_nodes
    if ng == 0:
        # Nothing to generate -> trivial no-op
        return {
            'message': 'lambda1_ratio small; no nodes generated',
            'x_aug': x,
            'edge_index_aug': edge_index,
            'idx_gen_slice': slice(x.size(0), x.size(0)),
        }

    # Models
    G = ImGAGNGenerator(cfg.dz, n_min, hidden=cfg.hidden_g, n_hidden=cfg.n_hidden_g).to(device)
    D = ImGAGNDiscriminator(in_dim=x.size(1), hid=cfg.hid_d, emb_dim=cfg.emb_dim, dropout=cfg.dropout).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, weight_decay=cfg.alpha_reg)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, weight_decay=cfg.beta_reg)

    bce = nn.BCEWithLogitsLoss()

    # Fixed TRAIN minority features for generator
    x_min = x[idx_train_min]  # [n_min, F]

    best = {'recall': -1.0}

    # Helper to build synthetic nodes and links in chunks (memory safe, no grad)
    @torch.no_grad()
    def _build_synth_nograd(z_all: Tensor, topk: int) -> Tuple[Tensor, Tensor]:
        ng_total = z_all.size(0)
        chunk = max(512, int(cfg.batch_size))
        xg_parts = []
        rows_list = []
        cols_list = []
        for start in range(0, ng_total, chunk):
            end = min(ng_total, start + chunk)
            z_i = z_all[start:end]
            logits_i = G(z_i)
            T_i = torch.softmax(logits_i, dim=-1)
            xg_i = T_i @ x_min
            xg_parts.append(xg_i)
            k_eff = min(topk, T_i.size(1)) if T_i.size(1) > 0 else 0
            if k_eff > 0:
                idx_i = T_i.topk(k=k_eff, dim=1).indices
                rows = torch.arange(end - start, device=device).view(-1, 1).repeat(1, k_eff) + start
                rows_list.append(rows.reshape(-1))
                cols_list.append(idx_i.reshape(-1))
        xg_full = torch.cat(xg_parts, dim=0) if xg_parts else torch.empty(0, x_min.size(1), device=device)
        if rows_list:
            rows = torch.cat(rows_list, dim=0)
            cols = torch.cat(cols_list, dim=0)
            links = torch.stack([rows, cols], dim=0)
        else:
            links = torch.empty(2, 0, dtype=torch.long, device=device)
        return xg_full, links

    # Helper to build xg (requires grad), centroid (requires grad) and links (no grad) in chunks
    def _build_xg_centroid_with_links(z_all: Tensor, h_min: Tensor, topk: int) -> Tuple[Tensor, Tensor, Tensor]:
        ng_total = z_all.size(0)
        chunk = max(512, int(cfg.batch_size))
        xg_parts = []
        centroid_parts = []
        rows_list = []
        cols_list = []
        for start in range(0, ng_total, chunk):
            end = min(ng_total, start + chunk)
            z_i = z_all[start:end]
            logits_i = G(z_i)
            T_i = torch.softmax(logits_i, dim=-1)
            xg_i = T_i @ x_min
            cent_i = T_i @ h_min
            xg_parts.append(xg_i)
            centroid_parts.append(cent_i)
            # links top-k (no grad needed for links)
            with torch.no_grad():
                k_eff = min(topk, T_i.size(1)) if T_i.size(1) > 0 else 0
                if k_eff > 0:
                    idx_i = T_i.topk(k=k_eff, dim=1).indices
                    rows = torch.arange(end - start, device=device).view(-1, 1).repeat(1, k_eff) + start
                    rows_list.append(rows.reshape(-1))
                    cols_list.append(idx_i.reshape(-1))
        xg_full = torch.cat(xg_parts, dim=0) if xg_parts else torch.empty(0, x_min.size(1), device=device)
        centroid_full = torch.cat(centroid_parts, dim=0) if centroid_parts else torch.empty(0, h_min.size(1), device=device)
        if rows_list:
            rows = torch.cat(rows_list, dim=0)
            cols = torch.cat(cols_list, dim=0)
            links = torch.stack([rows, cols], dim=0)
        else:
            links = torch.empty(2, 0, dtype=torch.long, device=device)
        return xg_full, centroid_full, links

    # Resolve homogeneous num_neighbors from cfg or global config
    def _resolve_num_neighbors(nn_cfg):
        nn = nn_cfg if nn_cfg is not None else EMB_NUM_NEIGHBORS
        # If dict (hetero), pick a representative per-layer list
        if isinstance(nn, dict):
            # Prefer pm relations if present
            for k in [( 'pm','spatial','pm'), ('pm','temporal','pm')]:
                if k in nn and isinstance(nn[k], (list, tuple)):
                    return list(nn[k])
            for v in nn.values():
                if isinstance(v, (list, tuple)):
                    return list(v)
            return [-1, -1]
        if isinstance(nn, (list, tuple)):
            return list(nn)
        if isinstance(nn, int):
            return [nn]
        return [-1, -1]
    resolved_neighbors = _resolve_num_neighbors(cfg.num_neighbors)

    # Helper: full-graph inference via NeighborLoader (homogeneous) or full-batch fallback
    def _infer_full(
        D_model: nn.Module, data_x: Tensor, data_edge_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        D_model.eval()
        if not cfg.use_neighbor_loader:
            with torch.no_grad():
                h_out, logit_real_out, logit_minor_out = D_model(
                    data_x, data_edge_index
                )
            return h_out, logit_real_out, logit_minor_out

        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        N = data_x.size(0)
        graph = Data(
            x=data_x.detach().cpu(), edge_index=data_edge_index.detach().cpu()
        )
        loader = NeighborLoader(
            graph,
            num_neighbors=resolved_neighbors,
            batch_size=cfg.batch_size,
            input_nodes=torch.arange(N),
        )
        h_out = torch.zeros(N, D_model.encoder.conv2.out_channels, device=device)
        logit_real_out = torch.zeros(N, device=device)
        logit_minor_out = torch.zeros(N, device=device)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                hb, lr_b, lm_b = D_model(batch.x, batch.edge_index)
                bs = int(batch.batch_size)
                seeds = batch.n_id[:bs].to(device)
                h_out[seeds] = hb[:bs]
                logit_real_out[seeds] = lr_b[:bs]
                logit_minor_out[seeds] = lm_b[:bs]
        return h_out, logit_real_out, logit_minor_out

    def _infer_nodes(
        D_model: nn.Module,
        data_x: Tensor,
        data_edge_index: Tensor,
        seed_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        D_model.eval()
        if not cfg.use_neighbor_loader:
            with torch.no_grad():
                h_all, lr_all, lm_all = D_model(data_x, data_edge_index)
            return (
                h_all[seed_indices],
                lr_all[seed_indices],
                lm_all[seed_indices],
            )

        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        seeds_cpu = seed_indices.detach().cpu()
        graph = Data(
            x=data_x.detach().cpu(), edge_index=data_edge_index.detach().cpu()
        )
        loader = NeighborLoader(
            graph,
            num_neighbors=resolved_neighbors,
            batch_size=cfg.batch_size,
            input_nodes=seeds_cpu,
        )
        out_N = seeds_cpu.numel()
        h_out = torch.zeros(out_N, D_model.encoder.conv2.out_channels, device=device)
        logit_real_out = torch.zeros(out_N, device=device)
        logit_minor_out = torch.zeros(out_N, device=device)
        pos = {int(seeds_cpu[i]): i for i in range(out_N)}
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                hb, lr_b, lm_b = D_model(batch.x, batch.edge_index)
                bs = int(batch.batch_size)
                g_ids = batch.n_id[:bs].cpu().tolist()
                for j, gid in enumerate(g_ids):
                    idx = pos.get(gid, None)
                    if idx is not None:
                        h_out[idx] = hb[j]
                        logit_real_out[idx] = lr_b[j]
                        logit_minor_out[idx] = lm_b[j]
        return h_out, logit_real_out, logit_minor_out

    epoch_iter = range(1, cfg.epochs + 1)
    if cfg.show_progress:
        epoch_iter = tqdm(epoch_iter, desc="ImGAGN (epochs)", leave=False)

    last_recall = None
    for epoch in epoch_iter:
        # Sample noise
        z = torch.randn(ng, cfg.dz, device=device)

        # Build synthetic nodes (no grad) for topology in chunks
        with torch.no_grad():
            xg, links = _build_synth_nograd(z, topk=cfg.topk_links)

        # Augmented graph
        x_aug, e_aug, edge_attr_aug, (gen_slice,) = _build_augmented_graph(
            x,
            edge_index,
            xg,
            links,
            idx_train_min,
            edge_attr=edge_attr,
            delta_feature_idx=delta_feature_idx,
        )
        N_tot = x_aug.size(0)
        y_real_all = torch.cat([torch.zeros(x.size(0), device=device), torch.ones(ng, device=device)], dim=0)  # 0=real,1=fake
        y_minor_all = torch.cat([y, torch.ones(ng, device=device)], dim=0)  # generated are minority

        # --------- Train D (multiple steps) ---------
        D.train()
        avg_loss_D = 0.0
        cnt_loss_D = 0
        # Progress bar for Discriminator steps
        d_pbar = tqdm(total=cfg.d_steps, desc="ImGAGN D steps", leave=False) if cfg.show_progress else None
        if cfg.use_neighbor_loader:
            # Move augmented tensors to CPU and free VRAM to avoid OOM
            x_aug_cpu = x_aug.detach().cpu(); e_aug_cpu = e_aug.detach().cpu()
            del x_aug, e_aug
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Build NeighborLoader over augmented graph for D updates (restrict seeds to train real + generated)
            from torch_geometric.data import Data
            from torch_geometric.loader import NeighborLoader
            data_aug = Data(x=x_aug_cpu, edge_index=e_aug_cpu)
            train_seeds = torch.arange(x.size(0))[train_mask.cpu()]
            gen_seeds = torch.arange(x.size(0), x.size(0) + ng)
            seeds_all = torch.cat([train_seeds, gen_seeds], dim=0)
            # iterate d_steps mini-batches per epoch
            loader = NeighborLoader(data_aug, num_neighbors=resolved_neighbors, batch_size=cfg.batch_size,
                                    input_nodes=seeds_all)
            it_loader = iter(loader)
            for _ in range(cfg.d_steps):
                try:
                    batch = next(it_loader)
                except StopIteration:
                    it_loader = iter(loader)
                    batch = next(it_loader)
                batch = batch.to(device)
                hb, lr_b, lm_b = D(batch.x, batch.edge_index)
                bs = int(batch.batch_size)
                seeds = batch.n_id[:bs].to(device)
                y_real_b = y_real_all[seeds]
                y_minor_b = y_minor_all[seeds]

                loss_fa = bce(lr_b[:bs], y_real_b)
                loss_cl = bce(lm_b[:bs], y_minor_b)
                # mm separation on REAL seeds only
                mask_real = ~y_real_b.bool()
                loss_mm = _mm_separation(hb[:bs][mask_real], y_minor_b[mask_real], margin=cfg.margin_mm) if mask_real.any() else torch.tensor(0.0, device=device)
                loss_D = loss_fa + loss_cl + loss_mm

                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
                optD.step()
                avg_loss_D += float(loss_D.detach().item())
                cnt_loss_D += 1
                if d_pbar is not None:
                    d_pbar.update(1)
                    d_pbar.set_postfix({'loss_D': f"{(avg_loss_D / max(1, cnt_loss_D)):.4f}"})
        else:
            # Full-batch fallback
            for _ in range(cfg.d_steps):
                h, logit_real, logit_minor = D(x_aug, e_aug)
                loss_fa = bce(logit_real, y_real_all)
                loss_cl = bce(logit_minor, y_minor_all)
                loss_mm = _mm_separation(h[~y_real_all.bool()], y_minor_all[~y_real_all.bool()], margin=cfg.margin_mm)
                loss_D = loss_fa + loss_cl + loss_mm
                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
                optD.step()
                avg_loss_D += float(loss_D.detach().item())
                cnt_loss_D += 1
                if d_pbar is not None:
                    d_pbar.update(1)
                    d_pbar.set_postfix({'loss_D': f"{(avg_loss_D / max(1, cnt_loss_D)):.4f}"})

        if d_pbar is not None:
            d_pbar.close()

        # --------- Train G (one step) ---------
        D.eval()
        # Recompute with grad for G path (chunked + NeighborLoader on generated seeds)
        # 1) Compute h_min (embeddings of TRAIN minority) once via NeighborLoader
        h_min_only, _, _ = _infer_nodes(D, x, edge_index, idx_train_min)
        h_min = h_min_only.to(device)  # [n_min, d]

        # 2) Build xg (requires grad), centroid (requires grad) and links (no grad) in chunks
        xg_g, centroid_full, links_g = _build_xg_centroid_with_links(z, h_min, topk=cfg.topk_links)
        x_aug_g, e_aug_g, edge_attr_aug_g, (gen_slice_g,) = _build_augmented_graph(
            x,
            edge_index,
            xg_g,
            links_g,
            idx_train_min,
            edge_attr=edge_attr,
            delta_feature_idx=delta_feature_idx,
        )

        sum_rf = torch.tensor(0.0, device=device)
        sum_mi = torch.tensor(0.0, device=device)
        sum_di = torch.tensor(0.0, device=device)
        tot = 0

        if cfg.use_neighbor_loader:
            from torch_geometric.data import Data
            from torch_geometric.loader import NeighborLoader

            data_aug_g = Data(
                x=x_aug_g.detach().cpu(), edge_index=e_aug_g.detach().cpu()
            )
            gen_indices = torch.arange(x.size(0), x.size(0) + ng)
            loader_g = NeighborLoader(
                data_aug_g,
                num_neighbors=resolved_neighbors,
                batch_size=cfg.batch_size,
                input_nodes=gen_indices,
            )

            # Progress bar for Generator minibatches
            g_pbar = None
            if cfg.show_progress:
                try:
                    total_mb = len(loader_g)
                except Exception:
                    total_mb = None
                g_pbar = tqdm(
                    total=total_mb, desc="ImGAGN G minibatches", leave=False
                )
            for batch in loader_g:
                batch = batch.to(device)
                h_b, lr_b, lm_b = D(batch.x, batch.edge_index)
                bs = int(batch.batch_size)
                seeds = batch.n_id[:bs].to(device)
                idx_local = seeds - x.size(0)
                target_real_b = torch.zeros(bs, device=device)
                target_minor_b = torch.ones(bs, device=device)
                sum_rf = sum_rf + F.binary_cross_entropy_with_logits(
                    lr_b[:bs], target_real_b, reduction="sum"
                )
                sum_mi = sum_mi + F.binary_cross_entropy_with_logits(
                    lm_b[:bs], target_minor_b, reduction="sum"
                )
                sum_di = sum_di + F.mse_loss(
                    h_b[:bs], centroid_full[idx_local], reduction="sum"
                )
                tot += bs
                if g_pbar is not None:
                    if g_pbar.total is not None:
                        g_pbar.update(1)
                    # Show running averages (normalized by processed seeds)
                    g_pbar.set_postfix(
                        {
                            "rf": f"{(sum_rf / max(1, tot)):.4f}",
                            "mi": f"{(sum_mi / max(1, tot)):.4f}",
                            "di": f"{(sum_di / max(1, tot)):.4f}",
                        }
                    )

            if g_pbar is not None:
                g_pbar.close()
        else:
            h_all, lr_all, lm_all = D(x_aug_g, e_aug_g)
            gen_start = int(x.size(0))
            gen_end = gen_start + int(ng)
            target_real = torch.zeros(gen_end - gen_start, device=device)
            target_minor = torch.ones(gen_end - gen_start, device=device)
            sum_rf = F.binary_cross_entropy_with_logits(
                lr_all[gen_start:gen_end], target_real, reduction="sum"
            )
            sum_mi = F.binary_cross_entropy_with_logits(
                lm_all[gen_start:gen_end], target_minor, reduction="sum"
            )
            sum_di = F.mse_loss(
                h_all[gen_start:gen_end], centroid_full, reduction="sum"
            )
            tot = gen_end - gen_start

        loss_rf = sum_rf / max(1, tot)
        loss_mi = sum_mi / max(1, tot)
        loss_di = sum_di / max(1, tot)

        loss_G = loss_rf + loss_mi + loss_di
        optG.zero_grad(set_to_none=True)
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
        optG.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                # Proxy: recall en TRAIN real usando NeighborLoader (solo seeds de train)
                tr_ids = torch.nonzero(train_mask, as_tuple=False).view(-1)
                _, _, lm_tr = _infer_nodes(D, x, edge_index, tr_ids)
                pred_minor = (torch.sigmoid(lm_tr) > 0.5).long()
                y_tr = y[train_mask]
                p_tr = pred_minor
                tp = ((p_tr == 1) & (y_tr == 1)).sum().item()
                fn = ((p_tr == 0) & (y_tr == 1)).sum().item()
                recall = tp / (tp + fn + 1e-9)
                last_recall = recall

                best = max(best, {'recall': recall, 'epoch': epoch}, key=lambda d: d['recall'])

        # Update progress bar postfix
        if cfg.show_progress and hasattr(epoch_iter, 'set_postfix'):
            postfix = {
                'D': f"{(avg_loss_D / max(1, cnt_loss_D)):.4f}",
                'G': f"{float(loss_G.detach().item()):.4f}"
            }
            if last_recall is not None:
                postfix['recall'] = f"{last_recall:.3f}"
            epoch_iter.set_postfix(postfix)

        if progress_callback is not None:
            progress_callback(
                epoch=epoch,
                total=cfg.epochs,
                loss_d=(avg_loss_D / max(1, cnt_loss_D)),
                loss_g=float(loss_G.detach().item()),
                recall=last_recall
            )

    if cfg.show_progress and hasattr(epoch_iter, 'close'):
        epoch_iter.close()

    # Final augmented graph (using last G)
    with torch.no_grad():
        z = torch.randn(ng, cfg.dz, device=device)
        xg, links = _build_synth_nograd(z, topk=cfg.topk_links)
        x_aug, e_aug, edge_attr_aug, (gen_slice,) = _build_augmented_graph(
            x,
            edge_index,
            xg,
            links,
            idx_train_min,
            edge_attr=edge_attr,
            delta_feature_idx=delta_feature_idx,
        )

    return {
        'G_state': G.state_dict(),
        'D_state': D.state_dict(),
        'x_aug': x_aug.detach().cpu(),
        'edge_index_aug': e_aug.detach().cpu(),
        'edge_attr_aug': edge_attr_aug.detach().cpu() if edge_attr_aug is not None else None,
        'gen_slice': gen_slice,
        'best_train_recall': torch.tensor(best.get('recall', -1.0)),
        'epochs': torch.tensor(cfg.epochs),
        'seed': int(SEED),
        'config': asdict(cfg),
    }
