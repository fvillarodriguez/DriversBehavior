# smote_cpu_cache.py
# CPU k-NN (chunked) + persistent cache for GraphSMOTE
# Drop-in helper to avoid CUDA OOM from full cdist/topk on GPU.

from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch

def _hash_tensor(t: torch.Tensor, extra: Dict) -> str:
    """Compact signature for caching (shape+dtype+tiny sample+extra)."""
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    if t.numel() > 0:
        sample = t.detach().flatten()[: min(2048, t.numel())].to("cpu", torch.float32)
        h.update(sample.numpy().tobytes())
    if extra:
        h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]

def _knn_sklearn(X_cpu: torch.Tensor, k: int, metric: str = "cosine") -> Tuple[torch.Tensor, torch.Tensor]:
    """Use sklearn NearestNeighbors if available (fast, multithreaded)."""
    try:
        import numpy as _np  # noqa: F401
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise RuntimeError("scikit-learn no está disponible para k-NN") from e

    Xn = X_cpu.detach().to("cpu", torch.float32).contiguous().numpy()
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nn.fit(Xn)
    dist, idx = nn.kneighbors(Xn, return_distance=True)
    idx = torch.from_numpy(idx[:, 1:]).long()
    dist = torch.from_numpy(dist[:, 1:]).float()
    return idx, dist

def _knn_torch_chunked(X_cpu: torch.Tensor, k: int, metric: str = "cosine",
                       q_block: int = 20000, r_block: int = 50000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked kNN in pure torch (CPU). Keeps only top-k per row to bound memory.
    metric: 'cosine' or 'euclidean' (cosine recommended).
    """
    assert metric in {"cosine", "euclidean"}
    X = X_cpu.detach().to("cpu", torch.float32).contiguous()
    N, d = X.shape
    k1 = k + 1  # keep self; we'll drop later

    top_val = torch.full((N, k1), float("-inf"))
    top_idx = torch.full((N, k1), -1, dtype=torch.long)

    if metric == "cosine":
        Xn = torch.nn.functional.normalize(X, p=2.0, dim=1)
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
                cand_idx = torch.cat([cur_idx, torch.arange(r0, r1).expand(q1 - q0, -1)], dim=1)
                new_val, ord_idx = torch.topk(cand_val, k1, largest=True, dim=1)
                cur_val = new_val
                cur_idx = torch.gather(cand_idx, 1, ord_idx)
                del sim, cand_val, cand_idx, new_val, ord_idx
            top_val[q0:q1] = cur_val
            top_idx[q0:q1] = cur_idx
        row_ids = torch.arange(N).view(-1, 1).expand(-1, k1)
        cur_val = top_val.clone()
        cur_val[top_idx == row_ids] = float("-inf")
        final_val, ord_idx = torch.topk(cur_val, k, largest=True, dim=1)
        final_idx = torch.gather(top_idx, 1, ord_idx)
        dist = 1.0 - final_val.clamp(-1, 1)
        return final_idx.long(), dist.float()

    else:
        X2 = (X * X).sum(dim=1, keepdim=True)
        for q0 in range(0, N, q_block):
            q1 = min(N, q0 + q_block)
            Q = X[q0:q1]
            Q2 = X2[q0:q1]
            cur_val = top_val[q0:q1]
            cur_idx = top_idx[q0:q1]
            for r0 in range(0, N, r_block):
                r1 = min(N, r0 + r_block)
                R = X[r0:r1]
                R2 = X2[r0:r1].T
                d2 = Q2 + R2 - 2.0 * (Q @ R.T)
                score = -d2
                cand_val = torch.cat([cur_val, score], dim=1)
                cand_idx = torch.cat([cur_idx, torch.arange(r0, r1).expand(q1 - q0, -1)], dim=1)
                new_val, ord_idx = torch.topk(cand_val, k1, largest=True, dim=1)
                cur_val = new_val
                cur_idx = torch.gather(cand_idx, 1, ord_idx)
                del d2, score, cand_val, cand_idx, new_val, ord_idx
            top_val[q0:q1] = cur_val
            top_idx[q0:q1] = cur_idx
        row_ids = torch.arange(N).view(-1, 1).expand(-1, k1)
        cur_val = top_val.clone()
        cur_val[top_idx == row_ids] = float("-inf")
        final_val, ord_idx = torch.topk(cur_val, k, largest=True, dim=1)
        final_idx = torch.gather(top_idx, 1, ord_idx)
        dist2 = -final_val
        dist = torch.sqrt(torch.clamp_min(dist2, 0.0))
        return final_idx.long(), dist.float()

def get_knn_cached(pos_emb: torch.Tensor,
                   k: int,
                   cache_dir: str,
                   extra_params: Optional[Dict] = None,
                   metric: str = "cosine",
                   prefer_sklearn: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute k-NN for positive-class embeddings on CPU, chunked, with persistent cache.
    Returns (indices [N,k], distances [N,k]) on CPU tensors.
    """
    extra_params = extra_params or {}
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    pos_cpu = pos_emb.detach().to("cpu", torch.float32).contiguous()

    k_eff = int(max(1, min(k, 32)))

    sig = _hash_tensor(pos_cpu, {"k": k_eff, "metric": metric, **extra_params})
    cache_file = Path(cache_dir) / f"knn_{sig}.pt"
    if cache_file.exists():
        # Prefer safe load (weights_only=True) when available to avoid pickle warnings.
        try:
            out = torch.load(cache_file, map_location="cpu", weights_only=True)
        except TypeError:
            out = torch.load(cache_file, map_location="cpu")
        return out["idx"], out["dist"]

    if prefer_sklearn:
        try:
            idx, dist = _knn_sklearn(pos_cpu, k_eff, metric=metric)
        except Exception:
            idx, dist = _knn_torch_chunked(pos_cpu, k_eff, metric=metric)
    else:
        idx, dist = _knn_torch_chunked(pos_cpu, k_eff, metric=metric)

    torch.save({"idx": idx, "dist": dist, "k": k_eff, "metric": metric}, cache_file)
    return idx, dist

def simple_smote_from_knn(pos_emb: torch.Tensor,
                          idx: torch.Tensor,
                          num_new: int,
                          alpha: float = 0.5) -> torch.Tensor:
    """
    Minimal SMOTE mixing using precomputed neighbors idx.
    Returns synthetic embeddings on CPU.
    """
    N, d = pos_emb.shape
    if num_new <= 0:
        return torch.empty((0, d), dtype=pos_emb.dtype)

    base = torch.randint(0, N, (num_new,))
    nbrs = torch.randint(0, idx.size(1), (num_new,))
    j = idx[base, nbrs]
    X = pos_emb.detach().to("cpu", torch.float32)
    synth = (1 - alpha) * X[base] + alpha * X[j]
    return synth
