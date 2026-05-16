import torch
from torch_geometric.data import HeteroData

from src import gnn_main
from src.train_pretrain import train_minibatch


class _DummyPMIndex:
    def __init__(self, reverse):
        self._rev = dict(reverse)


def _make_positive_sampler_graph() -> tuple[HeteroData, _DummyPMIndex]:
    graph = HeteroData()
    n_nodes = 16
    graph["pm"].x = torch.randn(n_nodes, 5)
    y = torch.zeros(n_nodes, dtype=torch.long)
    y[[1, 6, 13]] = 1
    graph["pm"].y = y

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:12] = True
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask[12:14] = True
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask[14:] = True
    graph["pm"].train_mask = train_mask
    graph["pm"].val_mask = val_mask
    graph["pm"].test_mask = test_mask

    reverse = {
        0: ("A", 0),
        1: ("A", 60),
        2: ("A", 55),
        3: ("A", 120),
        4: ("B", 60),
        5: ("B", 65),
        6: ("B", 180),
        7: ("B", 170),
        8: ("A", 180),
        9: ("A", 240),
        10: ("C", 60),
        11: ("C", 180),
        12: ("A", 60),
        13: ("B", 180),
        14: ("A", 55),
        15: ("B", 170),
    }
    pm_index = _DummyPMIndex(reverse)

    spatial_edges = torch.tensor(
        [
            [1, 1, 6, 6, 13, 14],
            [4, 12, 7, 15, 7, 2],
        ],
        dtype=torch.long,
    )
    graph[("pm", "spatial", "pm")].edge_index = spatial_edges
    graph[("pm", "spatial", "pm")].edge_attr = torch.ones(spatial_edges.size(1), 2)

    temporal_edges = torch.tensor(
        [
            [0, 1, 2, 6, 7, 8],
            [1, 2, 3, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    graph[("pm", "temporal", "pm")].edge_index = temporal_edges
    graph[("pm", "temporal", "pm")].edge_attr = torch.ones(temporal_edges.size(1), 2)
    return graph, pm_index


def test_positive_aware_seed_order_balances_batches_and_keeps_train_only():
    graph, pm_index = _make_positive_sampler_graph()

    seed_order, stats = gnn_main.build_positive_aware_seed_order(
        graph,
        pm_index=pm_index,
        batch_size=8,
        sampling_seed=123,
        epoch=1,
        target_positive_fraction=0.25,
        hard_negative_window_minutes=60,
        hard_negatives_per_positive=2,
    )

    assert stats["sampler_impl"] == "positive_aware_neighbor"
    assert stats["pos_per_batch"] == 2
    assert stats["temporal_hard_negative_candidates"] > 0
    assert stats["spatial_hard_negative_candidates"] > 0
    assert graph["pm"].train_mask[seed_order].all()
    assert not graph["pm"].val_mask[seed_order].any()
    assert not graph["pm"].test_mask[seed_order].any()

    y = graph["pm"].y
    for start in range(0, seed_order.numel(), 8):
        batch = seed_order[start: start + 8]
        if batch.numel() == 8:
            assert int((y[batch] == 1).sum().item()) == 2


def test_positive_aware_seed_order_is_deterministic_by_seed_and_epoch():
    graph, pm_index = _make_positive_sampler_graph()

    first, _ = gnn_main.build_positive_aware_seed_order(
        graph,
        pm_index=pm_index,
        batch_size=8,
        sampling_seed=321,
        epoch=1,
        target_positive_fraction=0.25,
    )
    second, _ = gnn_main.build_positive_aware_seed_order(
        graph,
        pm_index=pm_index,
        batch_size=8,
        sampling_seed=321,
        epoch=1,
        target_positive_fraction=0.25,
    )
    third, _ = gnn_main.build_positive_aware_seed_order(
        graph,
        pm_index=pm_index,
        batch_size=8,
        sampling_seed=321,
        epoch=2,
        target_positive_fraction=0.25,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_positive_aware_seed_order_fallback_without_positives():
    graph, pm_index = _make_positive_sampler_graph()
    graph["pm"].y.zero_()

    seed_order, stats = gnn_main.build_positive_aware_seed_order(
        graph,
        pm_index=pm_index,
        batch_size=8,
        sampling_seed=11,
        epoch=1,
    )

    assert stats["fallback_reason"] == "no_train_positives"
    assert seed_order.numel() == int(graph["pm"].train_mask.sum().item())
    assert graph["pm"].train_mask[seed_order].all()


def test_positive_aware_seed_order_without_pm_index_still_balances_positives():
    graph, _ = _make_positive_sampler_graph()

    seed_order, stats = gnn_main.build_positive_aware_seed_order(
        graph,
        pm_index=None,
        batch_size=8,
        sampling_seed=123,
        epoch=1,
        target_positive_fraction=0.25,
    )

    assert stats["pm_index_available"] is False
    assert stats["fallback_reason"] == "missing_pm_index_hard_negatives_random"
    assert graph["pm"].train_mask[seed_order].all()
    assert int((graph["pm"].y[seed_order[:8]] == 1).sum().item()) == 2


def test_build_native_positive_aware_loader_supervises_positive_train_nodes():
    graph, pm_index = _make_positive_sampler_graph()
    loader, err = gnn_main._build_native_sampler_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "positive_aware",
            "pm_index": pm_index,
            "positive_sampler_target_fraction": 0.25,
            "positive_sampler_hard_window_minutes": 60,
            "positive_sampler_hard_negatives_per_positive": 2,
            "positive_sampler_epoch": 1,
        },
        batch_size=8,
        sampling_seed=123,
        deterministic=True,
    )

    assert err is None
    assert loader is not None
    assert getattr(loader, "sampler_impl", "") == "positive_aware_neighbor"
    stats = getattr(loader, "positive_sampler_stats", {})
    assert stats["hard_negative_candidates"] > 0

    batch = next(iter(loader))
    supervised = batch["pm"].n_id[: batch["pm"].batch_size]
    assert graph["pm"].train_mask[supervised].all()
    assert int((graph["pm"].y[supervised] == 1).sum().item()) >= 1


class _ToyHeteroClassifier(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return {"pm": self.lin(x_dict["pm"])}, {"pm": x_dict["pm"]}, {}


def test_train_minibatch_accepts_positive_aware_neighbor_loader():
    graph, pm_index = _make_positive_sampler_graph()
    loader, err = gnn_main._build_native_sampler_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "positive_aware",
            "pm_index": pm_index,
            "positive_sampler_target_fraction": 0.25,
            "positive_sampler_epoch": 1,
        },
        batch_size=8,
        sampling_seed=123,
        deterministic=True,
    )

    assert err is None
    model = _ToyHeteroClassifier(in_channels=5, out_channels=2)
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
