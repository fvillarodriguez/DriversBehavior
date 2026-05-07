from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import graph_builder_app as app
from src import gnn_main


def test_checkpoint_edge_type_parser_handles_pyg_and_legacy_keys():
    state_dict = {
        "convs.0.convs.<pm___spatial_back___pm>.att_src": torch.randn(1),
        "convs.0.convs.pm__temporal__pm.att_dst": torch.randn(1),
        "convs.1.convs.<pm___spatial_back___pm>.bias": torch.randn(1),
    }

    assert app._infer_edge_types_from_state_dict(state_dict) == (
        ("pm", "spatial_back", "pm"),
        ("pm", "temporal", "pm"),
    )


def test_checkpoint_edge_types_rebuild_without_legacy_st_fwd():
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    edge_types = (
        ("pm", "temporal", "pm"),
        ("pm", "spatial", "pm"),
        ("pm", "spatial_back", "pm"),
    )
    graph = HeteroData()
    for edge_type in edge_types:
        graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
    model = gnn_main._build_gnn_model(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=1,
        num_layers=1,
        gnn_variant="gat_snapshot",
        num_nodes=4,
        edge_types=edge_types,
    )
    state_dict = model.state_dict()

    resolved_edge_types = app._resolve_checkpoint_edge_types_for_model(
        state_dict,
        graph,
    )
    rebuilt = gnn_main._build_gnn_model(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=1,
        num_layers=1,
        gnn_variant="gat_snapshot",
        num_nodes=4,
        edge_types=resolved_edge_types,
    )

    assert ("pm", "st_fwd", "pm") not in resolved_edge_types
    missing, unexpected = rebuilt.load_state_dict(state_dict, strict=True)
    assert missing == []
    assert unexpected == []


def test_checkpoint_graph_compat_blocks_missing_checkpoint_edge_type():
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    graph = HeteroData()
    graph["pm"].x = torch.zeros((2, 3), dtype=torch.float32)
    graph[("pm", "spatial", "pm")].edge_index = torch.zeros((2, 0), dtype=torch.long)
    state_dict = {
        "convs.0.convs.<pm___temporal___pm>.att_src": torch.randn(1),
    }

    is_ok, msg = app._check_model_graph_compat(
        "/tmp/checkpoint.pt",
        graph,
        meta={},
        state_dict=state_dict,
    )

    assert not is_ok
    assert "temporal" in str(msg)


def test_trained_model_eval_final_masks_are_test_only():
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    graph = HeteroData()
    graph["pm"].test_mask = torch.tensor([False, True, False])

    assert app._test_only_eval_masks(graph) == ["test_mask"]

    graph["pm"].test_mask = torch.zeros(3, dtype=torch.bool)
    assert app._test_only_eval_masks(graph) == []


def test_checkpoint_temporal_state_overrides_snapshot_metadata():
    state_dict = {
        "temporal_head.sequence_rows": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "temporal_head.target_rows": torch.tensor([2], dtype=torch.long),
        "temporal_head.temporal.weight_ih_l0": torch.randn(12, 4),
    }

    variant = app._resolve_gnn_variant_for_checkpoint(
        "/tmp/gat_model_BEST_GNN_gat_snapshot.pt",
        {"gnn_variant": "gat_snapshot"},
        state_dict,
    )

    assert variant == "gat_gru"


def test_checkpoint_without_temporal_state_strips_temporal_metadata():
    state_dict = {"convs.0.convs.pm__spatial__pm.att_src": torch.randn(1, 2, 8)}

    variant = app._resolve_gnn_variant_for_checkpoint(
        "/tmp/gat_model_BEST_GNN_gat_gru.pt",
        {"gnn_variant": "gat_gru"},
        state_dict,
    )

    assert variant == "gat_snapshot"


def test_sequence_index_can_be_reconstructed_from_checkpoint_buffers():
    state_dict = {
        "temporal_head.sequence_rows": torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long),
        "temporal_head.target_rows": torch.tensor([2, 4], dtype=torch.long),
    }

    sequence_index = app._resolve_sequence_index_for_checkpoint(None, state_dict)

    assert sequence_index is not None
    assert np.array_equal(sequence_index.sequence_rows, np.array([[0, 1, 2], [2, 3, 4]]))
    assert np.array_equal(sequence_index.target_rows, np.array([2, 4]))


def test_temporal_checkpoint_rebuilds_model_for_strict_load():
    sequence_index = SimpleNamespace(
        sequence_rows=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        target_rows=np.array([2, 3], dtype=np.int64),
    )
    model = gnn_main._build_gnn_model(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=1,
        num_layers=1,
        gnn_variant="gat_gru",
        sequence_index=sequence_index,
        num_nodes=4,
    )
    state_dict = model.state_dict()

    variant = app._resolve_gnn_variant_for_checkpoint(
        "/tmp/gat_model_BEST_GNN_gat_gru.pt",
        {},
        state_dict,
    )
    rebuilt_sequence_index = app._resolve_sequence_index_for_checkpoint(None, state_dict)
    rebuilt = gnn_main._build_gnn_model(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=1,
        num_layers=1,
        gnn_variant=variant,
        sequence_index=rebuilt_sequence_index,
        num_nodes=app._temporal_num_nodes_from_state_dict(state_dict, 4),
    )

    missing, unexpected = rebuilt.load_state_dict(state_dict, strict=True)
    assert missing == []
    assert unexpected == []
    assert hasattr(rebuilt, "temporal_head")


def test_gat_gru_requires_sequence_index_in_strict_mode():
    with pytest.raises(ValueError, match="SequenceIndex"):
        gnn_main._build_gnn_model(
            in_channels=3,
            hidden_channels=4,
            out_channels=2,
            num_heads=2,
            dropout=0.0,
            edge_feature_dim=1,
            num_layers=1,
            gnn_variant="gat_gru",
            sequence_index=None,
            num_nodes=4,
            require_temporal_head=True,
        )


def test_snapshot_variant_does_not_require_sequence_index():
    model = gnn_main._build_gnn_model(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=1,
        num_layers=1,
        gnn_variant="gat_snapshot",
        sequence_index=None,
        num_nodes=4,
        require_temporal_head=True,
    )

    assert not hasattr(model, "temporal_head")
