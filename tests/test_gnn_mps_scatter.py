import torch
import pytest

from src.gnn_mps_scatter import (
    _cpu_minmax_scatter,
    _index_add_scatter,
    install_gnn_mps_scatter_policy,
    is_gnn_mps_scatter_policy_installed,
)


def test_index_add_scatter_sum_matches_expected():
    src = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    index = torch.tensor([0, 1, 0, 2])

    out = _index_add_scatter(src, index, dim=0, dim_size=4, reduce="sum")

    expected = torch.tensor(
        [
            [6.0, 8.0],
            [3.0, 4.0],
            [7.0, 8.0],
            [0.0, 0.0],
        ]
    )
    assert torch.allclose(out, expected)


def test_index_add_scatter_mean_handles_empty_groups():
    src = torch.tensor(
        [
            [2.0, 4.0],
            [6.0, 8.0],
            [10.0, 12.0],
        ]
    )
    index = torch.tensor([0, 0, 2])

    out = _index_add_scatter(src, index, dim=0, dim_size=4, reduce="mean")

    expected = torch.tensor(
        [
            [4.0, 6.0],
            [0.0, 0.0],
            [10.0, 12.0],
            [0.0, 0.0],
        ]
    )
    assert torch.allclose(out, expected)


def test_index_add_scatter_supports_nonzero_dim():
    src = torch.tensor(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )
    index = torch.tensor([0, 1, 0, 2])

    out = _index_add_scatter(src, index, dim=1, dim_size=3, reduce="sum")

    expected = torch.tensor(
        [
            [6.0, 3.0, 7.0],
            [8.0, 4.0, 8.0],
        ]
    )
    assert torch.allclose(out, expected)


def test_cpu_minmax_scatter_matches_pyg_scatter():
    src = torch.tensor(
        [
            [1.0, 7.0],
            [3.0, 2.0],
            [5.0, 4.0],
            [0.5, 9.0],
        ]
    )
    index = torch.tensor([0, 1, 0, 1])

    out_max = _cpu_minmax_scatter(
        src,
        index,
        dim=0,
        dim_size=2,
        reduce="max",
    )
    out_min = _cpu_minmax_scatter(
        src,
        index,
        dim=0,
        dim_size=2,
        reduce="min",
    )

    assert torch.allclose(out_max, torch.tensor([[5.0, 7.0], [3.0, 9.0]]))
    assert torch.allclose(out_min, torch.tensor([[1.0, 4.0], [0.5, 2.0]]))


def test_install_gnn_mps_scatter_policy_is_idempotent_and_patches_softmax():
    first = install_gnn_mps_scatter_policy()
    second = install_gnn_mps_scatter_policy()

    from torch_geometric.utils import _softmax

    assert first.installed is True
    assert second.installed is True
    assert is_gnn_mps_scatter_policy_installed() is True
    assert getattr(_softmax.scatter, "_gnn_mps_scatter_policy", False) is True


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple MPS",
)
def test_prime_temporal_cache_clones_cpu_graph_to_model_device():
    from torch_geometric.data import HeteroData

    from src.gnn_main import _prime_temporal_cache_if_needed
    from src.temporal_head import TemporalAggregator

    device = torch.device("mps")
    graph = HeteroData()
    graph["pm"].x = torch.randn(3, 4)
    graph["pm"].y = torch.tensor([0, 1, 0])
    graph[("pm", "spatial", "pm")].edge_index = torch.tensor([[0, 1], [1, 2]])

    class DummyTemporalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1, device=device))
            self.temporal_head = TemporalAggregator(
                sequence_rows=torch.tensor([[0, 1], [1, 2], [0, 2]]),
                target_rows=torch.tensor([0, 1, 2]),
                num_nodes=3,
                embedding_dim=4,
                sequence_length=2,
                num_classes=2,
                cache_strategy="global_epoch",
            ).to(device)

        def forward(self, x_dict, edge_index_dict, edge_attr_dict):
            assert x_dict["pm"].device == self.scale.device
            z_pm = x_dict["pm"] * self.scale
            logits = torch.zeros(z_pm.size(0), 2, device=z_pm.device)
            return {"pm": logits}, {"pm": z_pm}, {}

    model = DummyTemporalModel()

    assert _prime_temporal_cache_if_needed(model, graph) is True
    assert graph["pm"].x.device.type == "cpu"
    assert model.temporal_head.embedding_cache.device.type == "mps"
