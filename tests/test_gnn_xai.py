from __future__ import annotations

import pytest
import numpy as np

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData

from src.gat_model import HeteroGAT, HeteroGATWithEdgeEncoder
from src.gnn_xai import compute_gnn_xai_graph, load_gnn_xai_result, save_gnn_xai_result


EDGE_TYPES = (("pm", "spatial", "pm"), ("pm", "temporal", "pm"))


def _dummy_xai_graph(edge_dim: int = 4) -> HeteroData:
    torch.manual_seed(7)
    data = HeteroData()
    data["pm"].x = torch.randn(6, 3)
    data["pm"].y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    data["pm"].test_mask = torch.ones(6, dtype=torch.bool)
    data["pm"].val_mask = torch.ones(6, dtype=torch.bool)
    spatial = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 3],
            [0, 1, 2, 2, 3, 4, 5, 0, 0, 5, 1, 2],
        ],
        dtype=torch.long,
    )
    temporal = torch.tensor(
        [
            [0, 1, 2, 3, 4, 4, 5],
            [1, 2, 3, 4, 5, 5, 0],
        ],
        dtype=torch.long,
    )
    data["pm", "spatial", "pm"].edge_index = spatial
    data["pm", "temporal", "pm"].edge_index = temporal
    data["pm", "spatial", "pm"].edge_attr = torch.randn(spatial.size(1), edge_dim)
    data["pm", "temporal", "pm"].edge_attr = torch.randn(temporal.size(1), edge_dim)
    return data


def _base_model(edge_dim: int = 4) -> HeteroGAT:
    model = HeteroGAT(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=edge_dim,
        num_layers=2,
        use_checkpointing=False,
        edge_types=EDGE_TYPES,
        use_relation_self_loops=True,
    )
    model.eval()
    return model


class _IdentityCalibrator:
    def predict_proba(self, values):
        prob1 = np.asarray(values, dtype=float).reshape(-1)
        prob1 = np.clip(prob1, 0.0, 1.0)
        return np.column_stack([1.0 - prob1, prob1])


def test_compute_gnn_xai_graph_returns_minimum_columns_and_filters_edges():
    data = _dummy_xai_graph(edge_dim=4)
    model = _base_model(edge_dim=4)

    result = compute_gnn_xai_graph(
        model=model,
        graph=data,
        node_indices=torch.arange(6),
        batch_size=6,
        num_neighbors=[-1, -1],
        percentile=0.0,
        k_heads=1,
        layer="conv1",
        threshold=0.5,
    )

    expected_node_cols = {
        "node_idx",
        "portico",
        "ts_min",
        "mask",
        "y_true",
        "y_pred",
        "prob1",
        "error_type",
    }
    expected_edge_cols = {
        "layer",
        "relation",
        "source_node_idx",
        "dest_node_idx",
        "mean_attention",
        "max_attention",
        "heads_above_threshold",
        "head_count",
        "source_prob1",
        "dest_prob1",
        "source_error_type",
        "dest_error_type",
    }
    assert expected_node_cols.issubset(result.nodes_df.columns)
    assert expected_edge_cols.issubset(result.edges_df.columns)
    assert not result.nodes_df.empty
    assert not result.edges_df.empty
    assert not (
        result.edges_df["source_node_idx"] == result.edges_df["dest_node_idx"]
    ).any()

    dedup_cols = ["layer", "relation", "source_node_idx", "dest_node_idx"]
    assert len(result.edges_df) == len(result.edges_df.drop_duplicates(dedup_cols))
    assert result.edges_df["source_prob1"].notna().any()
    assert result.edges_df["dest_prob1"].notna().any()


def test_compute_gnn_xai_graph_joins_attention_with_prediction_by_node():
    data = _dummy_xai_graph(edge_dim=4)
    model = _base_model(edge_dim=4)

    result = compute_gnn_xai_graph(
        model=model,
        graph=data,
        node_indices=torch.arange(6),
        batch_size=6,
        num_neighbors=[-1, -1],
        percentile=0.0,
        k_heads=1,
        layer="conv2",
        threshold=0.5,
    )

    node_lookup = result.nodes_df.set_index("node_idx")
    first_edge = result.edges_df.iloc[0]
    src = int(first_edge["source_node_idx"])
    dst = int(first_edge["dest_node_idx"])
    assert first_edge["source_error_type"] == node_lookup.loc[src, "error_type"]
    assert first_edge["dest_error_type"] == node_lookup.loc[dst, "error_type"]


def test_compute_gnn_xai_graph_uses_encoded_edge_attrs_for_edge_encoder_variant():
    raw_edge_dim = 3
    encoded_edge_dim = 2
    data = _dummy_xai_graph(edge_dim=raw_edge_dim)
    model = HeteroGATWithEdgeEncoder(
        in_channels=3,
        hidden_channels=4,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=raw_edge_dim,
        num_layers=2,
        use_checkpointing=False,
        edge_types=EDGE_TYPES,
        edge_feature_dims={edge_type: raw_edge_dim for edge_type in EDGE_TYPES},
        edge_encoded_dims={edge_type: encoded_edge_dim for edge_type in EDGE_TYPES},
        edge_encoder_hidden_dims={edge_type: 5 for edge_type in EDGE_TYPES},
        use_relation_self_loops=True,
    )
    model.eval()

    result = compute_gnn_xai_graph(
        model=model,
        graph=data,
        node_indices=torch.arange(6),
        batch_size=6,
        num_neighbors=[-1, -1],
        percentile=0.0,
        k_heads=1,
        layer="conv1",
        threshold=0.5,
    )

    assert not result.nodes_df.empty
    assert not result.edges_df.empty
    assert {"mean_attention", "source_prob1", "dest_prob1"}.issubset(result.edges_df.columns)


def test_compute_gnn_xai_graph_returns_advanced_explanations():
    data = _dummy_xai_graph(edge_dim=4)
    model = _base_model(edge_dim=4)

    result = compute_gnn_xai_graph(
        model=model,
        graph=data,
        node_indices=torch.arange(6),
        batch_size=6,
        num_neighbors=[-1, -1],
        percentile=0.0,
        k_heads=1,
        layer="conv1",
        threshold=0.5,
        explanation_methods=["relation_ablation", "feature_ablation"],
        explain_max_nodes=3,
        explain_top_features=2,
        feature_names=["flow", "speed", "density"],
    )

    assert result.explanations_df is not None
    assert not result.explanations_df.empty
    assert {"relation_ablation", "feature_ablation"}.issubset(
        set(result.explanations_df["method"].astype(str))
    )
    assert result.explanations_df["score"].notna().all()
    assert result.explanations_df["delta_prob1"].notna().all()
    feature_rows = result.explanations_df[result.explanations_df["method"] == "feature_ablation"]
    assert feature_rows["feature_name"].notna().any()
    assert int(result.manifest["explanation_count"]) == len(result.explanations_df)
    assert result.manifest["schema_version"] == "2.0"


def test_save_load_gnn_xai_result_round_trips_explanations(tmp_path):
    data = _dummy_xai_graph(edge_dim=4)
    model = _base_model(edge_dim=4)

    result = compute_gnn_xai_graph(
        model=model,
        graph=data,
        node_indices=torch.arange(6),
        batch_size=6,
        num_neighbors=[-1, -1],
        percentile=0.0,
        k_heads=1,
        layer="conv1",
        threshold=0.5,
        explanation_methods=["relation_ablation"],
        explain_max_nodes=2,
    )

    out_dir = save_gnn_xai_result(result, tmp_path / "xai_result")
    loaded = load_gnn_xai_result(out_dir)

    assert loaded.explanations_df is not None
    assert len(loaded.explanations_df) == len(result.explanations_df)
    assert loaded.manifest["files"].get("explanations")


def test_compute_gnn_xai_graph_keeps_calibration_for_advanced_explanations():
    data = _dummy_xai_graph(edge_dim=4)
    model = _base_model(edge_dim=4)

    result = compute_gnn_xai_graph(
        model=model,
        graph=data,
        node_indices=torch.arange(6),
        batch_size=6,
        num_neighbors=[-1, -1],
        percentile=0.0,
        k_heads=1,
        layer="conv1",
        threshold=0.5,
        calibration_model=_IdentityCalibrator(),
        explanation_methods=["relation_ablation"],
        explain_max_nodes=2,
    )

    assert result.explanations_df is not None
    assert not result.explanations_df.empty
    assert result.explanations_df["base_prob1"].between(0.0, 1.0).all()
    assert result.explanations_df["perturbed_prob1"].between(0.0, 1.0).all()


def test_compute_gnn_xai_graph_returns_gradient_feature_attributions_without_mutating_graph():
    data = _dummy_xai_graph(edge_dim=4)
    original_x = data["pm"].x.clone()
    model = _base_model(edge_dim=4)

    result = compute_gnn_xai_graph(
        model=model,
        graph=data,
        node_indices=torch.arange(6),
        batch_size=6,
        num_neighbors=[-1, -1],
        percentile=0.0,
        k_heads=1,
        layer="conv1",
        threshold=0.5,
        explanation_methods=["input_gradient", "integrated_gradients"],
        explain_max_nodes=2,
        explain_top_features=2,
        integrated_gradients_steps=3,
        feature_names=["flow", "speed", "density"],
    )

    assert torch.equal(data["pm"].x, original_x)
    assert not data["pm"].x.requires_grad
    assert result.explanations_df is not None
    gradient_rows = result.explanations_df[
        result.explanations_df["method"].isin(["input_gradient", "integrated_gradients"])
    ]
    assert not gradient_rows.empty
    assert {"input_gradient", "integrated_gradients"}.issubset(
        set(gradient_rows["method"].astype(str))
    )
    assert gradient_rows["attribution"].notna().all()
    assert gradient_rows["score"].notna().all()
    ig_rows = gradient_rows[gradient_rows["method"] == "integrated_gradients"]
    assert set(ig_rows["gradient_steps"].dropna().astype(int)) == {3}
    assert result.manifest["integrated_gradients_steps"] == 3
