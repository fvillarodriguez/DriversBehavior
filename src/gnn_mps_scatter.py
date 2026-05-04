"""GNN-scoped PyG scatter policy for Apple MPS.

PyG attention layers use sparse reductions for message aggregation and sparse
softmax. On Apple MPS, ``scatter_reduce_`` is still routed through PyTorch's CPU
fallback for min/max reductions, which produces noisy warnings and unnecessary
fallbacks from the generic PyG path. This module installs a narrow replacement
used by the GNN stack only:

- sum/add and mean are kept on MPS with ``index_add_``.
- min/max are executed on CPU and copied back to the original device.
- all other devices/reductions keep PyG's original implementation.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_ORIGINAL_SCATTER: Optional[Callable[..., Tensor]] = None
_PATCHED_SCATTER: Optional[Callable[..., Tensor]] = None
_INSTALLED = False
_LOGGED_ACTIVE = False


@dataclass(frozen=True)
class GNNMPSScatterPolicyStatus:
    installed: bool
    patched_modules: tuple[str, ...]
    sum_mean_backend: str = "index_add_"
    min_max_backend: str = "cpu"


def _normalize_reduce(reduce: str) -> str:
    value = str(reduce or "sum").lower()
    aliases = {
        "add": "sum",
        "amax": "max",
        "amin": "min",
    }
    return aliases.get(value, value)


def _normalize_dim(src: Tensor, dim: int) -> int:
    dim = src.dim() + dim if dim < 0 else dim
    if dim < 0 or dim >= src.dim():
        raise ValueError(
            f"The `dim` argument must lay between 0 and {src.dim() - 1} "
            f"(got {dim})"
        )
    return dim


def _resolve_dim_size(index: Tensor, dim_size: Optional[int]) -> int:
    if dim_size is not None:
        return int(dim_size)
    return int(index.max().item()) + 1 if index.numel() > 0 else 0


def _move_dim_to_front(src: Tensor, dim: int) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    if dim == 0:
        return src, lambda out: out
    return src.movedim(dim, 0), lambda out: out.movedim(0, dim)


def _index_add_scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    """Scatter sum/mean implemented via ``index_add_``.

    This intentionally supports the 1-D index case used by PyG message passing.
    Other index layouts fall back to the original PyG implementation.
    """
    if index.dim() != 1:
        raise ValueError("GNN MPS scatter policy expects a one-dimensional index")

    reduce_norm = _normalize_reduce(reduce)
    if reduce_norm not in {"sum", "mean"}:
        raise ValueError(f"Unsupported index_add_ reduce: {reduce}")

    dim = _normalize_dim(src, dim)
    index = index.to(device=src.device, dtype=torch.long).view(-1)
    src_front, restore_dim = _move_dim_to_front(src, dim)

    if src_front.size(0) != index.numel():
        raise ValueError(
            "The index length must match the source size along the scatter "
            f"dimension (got {index.numel()} and {src_front.size(0)})"
        )

    dim_size_int = _resolve_dim_size(index, dim_size)
    out_shape = list(src_front.shape)
    out_shape[0] = dim_size_int
    out_front = src_front.new_zeros(out_shape)
    if index.numel() > 0:
        out_front.index_add_(0, index, src_front)

    if reduce_norm == "mean":
        count = src_front.new_zeros((dim_size_int,))
        if index.numel() > 0:
            ones = torch.ones(index.numel(), dtype=src_front.dtype, device=src_front.device)
            count.index_add_(0, index, ones)
        count = count.clamp_min(1)
        count_shape = [dim_size_int] + [1] * (src_front.dim() - 1)
        out_front = out_front / count.view(count_shape)

    return restore_dim(out_front)


def _cpu_minmax_scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = "max",
    original_scatter: Optional[Callable[..., Tensor]] = None,
) -> Tensor:
    """Run min/max scatter on CPU and return the result to ``src.device``."""
    reduce_norm = _normalize_reduce(reduce)
    if reduce_norm not in {"min", "max"}:
        raise ValueError(f"Unsupported CPU reduce: {reduce}")

    dim = _normalize_dim(src, dim)
    scatter_impl = original_scatter or _ORIGINAL_SCATTER
    if getattr(scatter_impl, "_gnn_mps_scatter_policy", False):
        scatter_impl = _ORIGINAL_SCATTER

    src_cpu = src.cpu()
    index_cpu = index.to(device="cpu", dtype=torch.long)
    if scatter_impl is None:
        return _native_cpu_minmax_scatter(
            src_cpu,
            index_cpu,
            dim=dim,
            dim_size=dim_size,
            reduce=reduce_norm,
        ).to(device=src.device)

    out_cpu = scatter_impl(
        src_cpu,
        index_cpu,
        dim=dim,
        dim_size=dim_size,
        reduce=reduce_norm,
    )
    return out_cpu.to(device=src.device)


def _native_cpu_minmax_scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = "max",
) -> Tensor:
    reduce_norm = _normalize_reduce(reduce)
    if reduce_norm not in {"min", "max"}:
        raise ValueError(f"Unsupported native CPU reduce: {reduce}")
    if index.dim() != 1:
        raise ValueError("GNN MPS scatter policy expects a one-dimensional index")

    dim = _normalize_dim(src, dim)
    index = index.to(device=src.device, dtype=torch.long).view(-1)
    src_front, restore_dim = _move_dim_to_front(src, dim)
    dim_size_int = _resolve_dim_size(index, dim_size)

    out_shape = list(src_front.shape)
    out_shape[0] = dim_size_int
    out_front = src_front.new_zeros(out_shape)
    if index.numel() > 0:
        index_view = index.view([index.numel()] + [1] * (src_front.dim() - 1))
        index_view = index_view.expand_as(src_front)
        out_front.scatter_reduce_(
            0,
            index_view,
            src_front,
            reduce="amax" if reduce_norm == "max" else "amin",
            include_self=False,
        )

    return restore_dim(out_front)


def _should_use_mps_policy(src: Tensor, index: Tensor) -> bool:
    return (
        isinstance(src, Tensor)
        and isinstance(index, Tensor)
        and src.device.type == "mps"
        and index.dim() == 1
    )


def _make_patched_scatter(original_scatter: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def _patched_scatter(
        src: Tensor,
        index: Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
        reduce: str = "sum",
    ) -> Tensor:
        global _LOGGED_ACTIVE

        reduce_norm = _normalize_reduce(reduce)
        if not _should_use_mps_policy(src, index):
            return original_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)

        if not _LOGGED_ACTIVE:
            logger.info(
                "GNN MPS scatter policy active: sum/mean -> index_add_, "
                "min/max -> CPU fallback."
            )
            _LOGGED_ACTIVE = True

        try:
            if reduce_norm in {"sum", "mean"}:
                return _index_add_scatter(
                    src,
                    index,
                    dim=dim,
                    dim_size=dim_size,
                    reduce=reduce_norm,
                )
            if reduce_norm in {"min", "max"}:
                return _cpu_minmax_scatter(
                    src,
                    index,
                    dim=dim,
                    dim_size=dim_size,
                    reduce=reduce_norm,
                    original_scatter=original_scatter,
                )
        except Exception as exc:
            logger.debug(
                "Falling back to PyG scatter for reduce=%s on MPS: %s",
                reduce,
                exc,
            )

        return original_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)

    setattr(_patched_scatter, "_gnn_mps_scatter_policy", True)
    return _patched_scatter


def _patch_module_attr(module_name: str, attr: str, value: Callable[..., Tensor]) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    if not hasattr(module, attr):
        return False
    setattr(module, attr, value)
    return True


def install_gnn_mps_scatter_policy() -> GNNMPSScatterPolicyStatus:
    """Install the PyG scatter policy used by the Graph Neural Network stack."""
    global _ORIGINAL_SCATTER, _PATCHED_SCATTER, _INSTALLED

    patched_modules: list[str] = []

    try:
        scatter_module = importlib.import_module("torch_geometric.utils._scatter")
    except Exception as exc:
        logger.debug("Could not import PyG scatter module for GNN MPS policy: %s", exc)
        return GNNMPSScatterPolicyStatus(False, tuple())

    current_scatter = getattr(scatter_module, "scatter", None)
    if current_scatter is None:
        return GNNMPSScatterPolicyStatus(False, tuple())

    if _INSTALLED and _PATCHED_SCATTER is not None:
        patched_scatter = _PATCHED_SCATTER
    else:
        if getattr(current_scatter, "_gnn_mps_scatter_policy", False):
            patched_scatter = current_scatter
        else:
            _ORIGINAL_SCATTER = current_scatter
            patched_scatter = _make_patched_scatter(current_scatter)
        _PATCHED_SCATTER = patched_scatter
        _INSTALLED = True

    patch_targets = (
        ("torch_geometric.utils._scatter", "scatter"),
        ("torch_geometric.utils", "scatter"),
        ("torch_geometric.utils._softmax", "scatter"),
        ("torch_geometric.utils._coalesce", "scatter"),
        ("torch_geometric.nn.aggr.base", "scatter"),
        ("torch_geometric.nn.aggr.fused", "scatter"),
        ("torch_geometric.nn.pool.glob", "scatter"),
    )
    for module_name, attr in patch_targets:
        if _patch_module_attr(module_name, attr, patched_scatter):
            patched_modules.append(module_name)

    return GNNMPSScatterPolicyStatus(True, tuple(sorted(set(patched_modules))))


def is_gnn_mps_scatter_policy_installed() -> bool:
    return bool(_INSTALLED)
