import torch
from torch_geometric.data import HeteroData

from src import gnn_main
from src.graph_builder_app import (
    SAMPLER_CLUSTER_GCN_PROFILE_PRESETS,
    SAMPLER_GRAPHSAINT_PROFILE_PRESETS,
    _advance_batch_index_by_jump,
    _adaptive_probe_jump,
    _build_sampler_memory_loader,
    _infer_batch_step_size,
    _memory_budget_bytes,
    _resolve_probe_loader_limit,
    _select_best_sampler_memory_row,
)
from src.train_pretrain import train_minibatch


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
    graph[("pm", "spatial", "pm")].edge_attr = torch.randn(src.numel(), 3)
    graph[("pm", "temporal", "pm")].edge_index = torch.stack([src, dst], dim=0)
    graph[("pm", "temporal", "pm")].edge_attr = torch.randn(src.numel(), 3)
    graph[("pm", "spatial_back", "pm")].edge_index = torch.stack([dst, src], dim=0)
    graph[("pm", "spatial_back", "pm")].edge_attr = torch.randn(src.numel(), 3)
    graph[("pm", "st_fwd", "pm")].edge_index = torch.stack([src, dst], dim=0)
    graph[("pm", "st_fwd", "pm")].edge_attr = torch.randn(src.numel(), 3)
    return graph


def test_highway_sampler_profile_presets_are_graph_specific():
    assert dict(SAMPLER_CLUSTER_GCN_PROFILE_PRESETS) == {
        "highway_stable": (64, 0),
        "highway_broad": (32, 0),
        "highway_local": (128, 0),
        "highway_probe": (64, 16),
    }
    assert dict(SAMPLER_GRAPHSAINT_PROFILE_PRESETS) == {
        "highway_rw_stable": ("random_walk", 4096, 16, 3),
        "highway_node_stable": ("node", 4096, 16, 1),
        "highway_edge_stable": ("edge", 4096, 16, 1),
        "highway_rw_broad": ("random_walk", 8192, 12, 3),
    }


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


def test_native_sampler_batch_preserves_edge_attr_alignment():
    graph = _make_probe_graph()
    loader, err = _build_sampler_memory_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "cluster_gcn",
            "cluster_gcn_num_parts": 4,
            "cluster_gcn_parts_per_epoch": 0,
        },
        batch_size=128,
        sampling_seed=42,
    )

    assert err is None
    batch = next(iter(loader))
    aligned_edge_types = 0
    for edge_type in batch.edge_types:
        edge_attr = getattr(batch[edge_type], "edge_attr", None)
        if edge_attr is None:
            continue
        aligned_edge_types += 1
        assert edge_attr.size(0) == batch[edge_type].edge_index.size(1)
        assert edge_attr.size(1) == 3
    assert aligned_edge_types >= 1


class _ToyHeteroClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        logits = {"pm": self.lin(x_dict["pm"])}
        embeddings = {"pm": x_dict["pm"]}
        return logits, embeddings, {}


def test_train_minibatch_accepts_native_cluster_and_graphsaint_loaders():
    for sampler_config in (
        {
            "train_sampler_mode": "cluster_gcn",
            "cluster_gcn_num_parts": 4,
            "cluster_gcn_parts_per_epoch": 0,
        },
        {
            "train_sampler_mode": "graphsaint",
            "graphsaint_mode": "node",
            "graphsaint_batch_size": 64,
            "graphsaint_num_steps": 2,
            "graphsaint_walk_length": 1,
        },
    ):
        graph = _make_probe_graph()
        base_seeds = graph["pm"].train_mask.nonzero(as_tuple=False).view(-1)
        loader, err = gnn_main._build_native_sampler_loader(
            graph_cpu=graph,
            sampler_config=sampler_config,
            batch_size=64,
            sampling_seed=42,
            base_seeds=base_seeds,
            deterministic=True,
        )
        assert err is None
        assert loader is not None
        assert loader.__class__.__name__ != "NeighborLoader"

        model = _ToyHeteroClassifier(in_channels=8, out_channels=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        avg_loss, avg_cls_loss, avg_edge_loss, avg_l2_att_loss = train_minibatch(
            model,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            accumulation_steps=1,
        )

        assert avg_loss >= 0.0
        assert avg_cls_loss >= 0.0
        assert avg_edge_loss == 0.0
        assert avg_l2_att_loss == 0.0
