import math

import torch
from torch_geometric.data import HeteroData

from src import gnn_main
from src.gnn_rl_sampler import (
    RioGNNThresholdController,
    build_rl_top_p_batch,
)
from src.train_pretrain import train_minibatch


def _make_rl_graph() -> HeteroData:
    graph = HeteroData()
    graph["pm"].x = torch.arange(5, dtype=torch.float32).view(-1, 1)
    graph["pm"].y = torch.tensor([1, 0, 0, 1, 0], dtype=torch.long)
    train_mask = torch.tensor([True, True, True, False, False])
    graph["pm"].train_mask = train_mask
    graph["pm"].val_mask = torch.tensor([False, False, False, True, False])
    graph["pm"].test_mask = torch.tensor([False, False, False, False, True])
    edge_index = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    graph[("pm", "spatial", "pm")].edge_index = edge_index
    graph[("pm", "spatial", "pm")].edge_attr = torch.tensor(
        [[10.0], [20.0], [30.0], [40.0]],
        dtype=torch.float32,
    )
    return graph


def _controller(action_space: str = "discrete") -> RioGNNThresholdController:
    ctrl = RioGNNThresholdController(
        edge_types=[("pm", "spatial", "pm")],
        num_layers=1,
        in_channels=1,
        max_degree_by_edge_type={("pm", "spatial", "pm"): 4},
        action_space=action_space,
        initial_p=0.5,
        min_p=0.05,
        max_p=1.0,
        min_keep=1,
        seed=123,
    )

    def fixed_probs(x):
        return torch.tensor([0.9, 0.85, 0.1, 0.8, 0.0], dtype=torch.float32, device=x.device)

    ctrl.scorer.positive_probability = fixed_probs
    return ctrl


def test_top_p_selects_expected_edges_and_preserves_edge_attr():
    graph = _make_rl_graph()
    ctrl = _controller()

    batch, err = build_rl_top_p_batch(
        graph,
        torch.tensor([0]),
        controller=ctrl,
        num_neighbors_cfg={("pm", "spatial", "pm"): [4]},
        num_layers=1,
    )

    assert err is None
    assert batch is not None
    assert batch["pm"].n_id[: batch["pm"].batch_size].tolist() == [0]
    assert batch[("pm", "spatial", "pm")].edge_attr.view(-1).tolist() == [10.0, 30.0]
    assert batch[("pm", "spatial", "pm")].edge_index.size(1) == 2


def test_top_p_respects_min_keep_and_num_neighbors_cap():
    graph = _make_rl_graph()
    ctrl = _controller()
    ctrl.set_threshold(0, ("pm", "spatial", "pm"), 0.05)

    batch, err = build_rl_top_p_batch(
        graph,
        torch.tensor([0]),
        controller=ctrl,
        num_neighbors_cfg={("pm", "spatial", "pm"): [1]},
        num_layers=1,
    )

    assert err is None
    assert batch is not None
    assert batch[("pm", "spatial", "pm")].edge_attr.view(-1).tolist() == [10.0]


def test_rl_loader_keeps_supervised_train_seeds_first():
    graph = _make_rl_graph()
    loader, err = gnn_main._build_native_sampler_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "rl_top_p",
            "rl_action_space": "discrete",
            "rl_initial_p": 0.5,
            "num_layers": 1,
        },
        batch_size=5,
        sampling_seed=7,
        base_seeds=torch.arange(5),
        num_neighbors_cfg={("pm", "spatial", "pm"): [2]},
        deterministic=True,
    )

    assert err is None
    batch = next(iter(loader))
    supervised = batch["pm"].n_id[: batch["pm"].batch_size]
    assert graph["pm"].train_mask[supervised].all()
    assert not graph["pm"].val_mask[supervised].any()
    assert not graph["pm"].test_mask[supervised].any()


def test_discrete_rsrl_reduces_range_and_converges():
    ctrl = _controller()
    ctrl.update_after_validation(0.10, epoch=0)
    initial_width = next(iter(ctrl.records.values())).high - next(iter(ctrl.records.values())).low

    for epoch in range(1, 12):
        ctrl.update_after_validation(0.10 + epoch * 0.02, epoch=epoch)

    record = next(iter(ctrl.records.values()))
    assert record.high - record.low < initial_width
    assert record.converged
    assert ctrl.reward_history[-1]["valid"] is True


def test_continuous_actor_thresholds_are_bounded_and_reproducible():
    ctrl_a = _controller(action_space="continuous_actor")
    ctrl_b = _controller(action_space="continuous_actor")
    for ctrl in (ctrl_a, ctrl_b):
        ctrl.update_after_validation(0.1, epoch=0)
        ctrl.update_after_validation(0.2, epoch=1)

    threshold_a = next(iter(ctrl_a.thresholds_serializable().values()))
    threshold_b = next(iter(ctrl_b.thresholds_serializable().values()))
    assert 0.05 <= threshold_a <= 1.0
    assert math.isclose(threshold_a, threshold_b, rel_tol=0.0, abs_tol=1e-6)


def test_invalid_auprc_does_not_update_thresholds():
    ctrl = _controller()
    before = ctrl.thresholds_serializable()
    payload = ctrl.update_after_validation(float("nan"), epoch=1)

    assert payload["valid"] is False
    assert ctrl.thresholds_serializable() == before
    assert ctrl.invalid_update_count == 1


class _ToyHeteroClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 2)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        logits = {"pm": self.lin(x_dict["pm"])}
        embeddings = {"pm": x_dict["pm"]}
        return logits, embeddings, {}


def test_train_minibatch_accepts_rl_top_p_loader_with_similarity_loss():
    graph = _make_rl_graph()
    loader, err = gnn_main._build_native_sampler_loader(
        graph_cpu=graph,
        sampler_config={
            "train_sampler_mode": "rl_top_p",
            "rl_action_space": "discrete",
            "rl_initial_p": 0.5,
            "num_layers": 1,
        },
        batch_size=3,
        sampling_seed=11,
        base_seeds=graph["pm"].train_mask.nonzero(as_tuple=False).view(-1),
        num_neighbors_cfg={("pm", "spatial", "pm"): [2]},
        deterministic=True,
    )
    assert err is None
    assert loader is not None

    model = _ToyHeteroClassifier()
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(loader.rl_sampler_controller.parameters()),
        lr=0.01,
    )
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss, avg_cls_loss, avg_edge_loss, avg_l2_att_loss = train_minibatch(
        model,
        loader,
        optimizer,
        criterion,
        device=torch.device("cpu"),
        accumulation_steps=1,
        rl_sampler_controller=loader.rl_sampler_controller,
        lambda_simi=0.5,
    )

    assert avg_loss >= 0.0
    assert avg_cls_loss >= 0.0
    assert avg_edge_loss == 0.0
    assert avg_l2_att_loss == 0.0

