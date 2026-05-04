import torch
from torch_geometric.data import HeteroData

from src.graph_builder_app import (
    _advance_batch_index_by_jump,
    _adaptive_probe_jump,
    _build_sampler_memory_loader,
    _infer_batch_step_size,
    _memory_budget_bytes,
    _resolve_probe_loader_limit,
    _select_best_sampler_memory_row,
)


def test_memory_budget_bytes_uses_fraction():
    expected = int(24.0 * (1024 ** 3) * 0.95)
    assert _memory_budget_bytes(24.0, 0.95) == expected


def test_select_best_sampler_memory_row_prefers_highest_under_budget():
    rows = [
        {"config_name": "cfg_a", "status": "ok", "memory_peak_bytes": 10, "batch_size": 128},
        {"config_name": "cfg_b", "status": "ok", "memory_peak_bytes": 18, "batch_size": 256},
        {"config_name": "cfg_c", "status": "ok", "memory_peak_bytes": 25, "batch_size": 512},
    ]
    best_under, closest_over = _select_best_sampler_memory_row(rows, budget_bytes=20)
    assert best_under is not None
    assert best_under["config_name"] == "cfg_b"
    assert closest_over is not None
    assert closest_over["config_name"] == "cfg_c"


def test_select_best_sampler_memory_row_handles_no_under_budget():
    rows = [
        {"config_name": "cfg_x", "status": "ok", "memory_peak_bytes": 30, "batch_size": 128},
        {"config_name": "cfg_y", "status": "ok", "memory_peak_bytes": 40, "batch_size": 256},
    ]
    best_under, closest_over = _select_best_sampler_memory_row(rows, budget_bytes=20)
    assert best_under is None
    assert closest_over is not None
    assert closest_over["config_name"] == "cfg_x"


def test_adaptive_probe_jump_reduces_step_when_usage_is_high():
    assert _adaptive_probe_jump(0.20) == 10
    assert _adaptive_probe_jump(0.50) >= 8
    assert _adaptive_probe_jump(0.80) <= 5
    assert _adaptive_probe_jump(0.97) <= 2
    assert _adaptive_probe_jump(1.10) == 1


def test_adaptive_probe_jump_uses_custom_jump_sizes():
    jump_sizes = [1, 3, 6, 10]
    assert _adaptive_probe_jump(None, jump_sizes) == 10
    assert _adaptive_probe_jump(0.20, jump_sizes) == 10
    assert _adaptive_probe_jump(0.80, jump_sizes) == 3
    assert _adaptive_probe_jump(0.97, jump_sizes) == 1


def test_infer_batch_step_size_uses_smallest_positive_delta():
    values = [1024, 1152, 1280, 1664]
    assert _infer_batch_step_size(values) == 128


def test_advance_batch_index_by_jump_applies_step_over_batch_range():
    values = [1024, 1152, 1280, 1408, 1536]
    idx = _advance_batch_index_by_jump(
        values,
        current_idx=0,
        jump_units=3,
        batch_step_size=128,
    )
    assert idx == 3


def test_advance_batch_index_by_jump_respects_upper_bound():
    values = [1024, 1152, 1280, 1408, 1536]
    idx = _advance_batch_index_by_jump(
        values,
        current_idx=1,
        jump_units=4,
        batch_step_size=128,
        upper_bound_idx=3,
    )
    assert idx == 3


def test_resolve_probe_loader_limit_fast_mode_uses_probe_steps():
    assert _resolve_probe_loader_limit(
        base_len=100,
        probe_steps=3,
        fidelity_full_epoch=False,
    ) == 3


def test_resolve_probe_loader_limit_fidelity_mode_uses_full_epoch():
    assert _resolve_probe_loader_limit(
        base_len=100,
        probe_steps=3,
        fidelity_full_epoch=True,
    ) == 100


def test_resolve_probe_loader_limit_fidelity_fallbacks_to_probe_steps():
    assert _resolve_probe_loader_limit(
        base_len=0,
        probe_steps=4,
        fidelity_full_epoch=True,
    ) == 4


def _make_probe_graph() -> HeteroData:
    graph = HeteroData()
    n_nodes = 120
    graph["pm"].x = torch.randn(n_nodes, 8)
    graph["pm"].y = torch.randint(0, 2, (n_nodes,))
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:90] = True
    graph["pm"].train_mask = train_mask
    graph["pm"].val_mask = ~train_mask

    src = torch.randint(0, n_nodes, (400,))
    dst = torch.randint(0, n_nodes, (400,))
    graph[("pm", "spatial", "pm")].edge_index = torch.stack([src, dst], dim=0)
    graph[("pm", "temporal", "pm")].edge_index = torch.stack([src, dst], dim=0)
    graph[("pm", "spatial_back", "pm")].edge_index = torch.stack([dst, src], dim=0)
    graph[("pm", "st_fwd", "pm")].edge_index = torch.stack([src, dst], dim=0)
    return graph


def test_build_sampler_memory_loader_cluster_gcn_is_native():
    graph = _make_probe_graph()
    loader, err = _build_sampler_memory_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "cluster_gcn",
            "cluster_gcn_num_parts": 16,
            "cluster_gcn_parts_per_epoch": 8,
        },
        batch_size=128,
        sampling_seed=42,
    )
    assert err is None
    assert loader is not None
    assert getattr(loader, "sampler_impl", "") == "cluster_gcn_native"
    assert loader.__class__.__name__ != "NeighborLoader"


def test_build_sampler_memory_loader_graphsaint_is_native():
    graph = _make_probe_graph()
    loader, err = _build_sampler_memory_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "graphsaint",
            "graphsaint_mode": "node",
            "graphsaint_batch_size": 64,
            "graphsaint_num_steps": 4,
            "graphsaint_walk_length": 2,
        },
        batch_size=128,
        sampling_seed=42,
    )
    assert err is None
    assert loader is not None
    assert str(getattr(loader, "sampler_impl", "")).startswith("graphsaint_native_")
    assert loader.__class__.__name__ != "NeighborLoader"


def test_native_sampler_batch_supervises_train_nodes_only():
    graph = _make_probe_graph()
    loader, err = _build_sampler_memory_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "graphsaint",
            "graphsaint_mode": "node",
            "graphsaint_batch_size": 128,
            "graphsaint_num_steps": 1,
        },
        batch_size=128,
        sampling_seed=42,
    )

    assert err is None
    batch = next(iter(loader))
    supervised = batch["pm"].n_id[: batch["pm"].batch_size]
    assert graph["pm"].train_mask[supervised].all()
    assert batch["pm"].train_mask[: batch["pm"].batch_size].all()
    assert not batch["pm"].val_mask[: batch["pm"].batch_size].any()
