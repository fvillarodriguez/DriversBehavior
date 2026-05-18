import torch
import pytest


def test_edge_attr_decoder_keeps_relation_specific_output_dims(monkeypatch, tmp_path):
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData

    from src import graphsmote

    data = HeteroData()
    data["pm"].x = torch.randn(6, 4)
    data["pm"].y = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.long)
    data["pm"].train_mask = torch.ones(6, dtype=torch.bool)

    spatial_edges = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
        dtype=torch.long,
    )
    temporal_edges = torch.tensor(
        [[0, 1, 2, 3, 4], [2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    data["pm", "spatial", "pm"].edge_index = spatial_edges
    data["pm", "spatial", "pm"].edge_attr = torch.randn(spatial_edges.size(1), 3)
    data["pm", "temporal", "pm"].edge_index = temporal_edges
    data["pm", "temporal", "pm"].edge_attr = torch.randn(temporal_edges.size(1), 5)

    monkeypatch.setattr(
        graphsmote,
        "get_embeddings_minibatch",
        lambda model, graph, num_neighbors=None: {"pm": torch.randn(graph["pm"].num_nodes, 7)},
    )

    model = torch.nn.Linear(4, 2)
    decoder = graphsmote.train_edge_attr_decoders(
        model,
        data,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        save_dir=str(tmp_path),
        show_progress=False,
    )

    assert decoder is not None
    assert decoder.edge_attr_dims_by_rel["pm:spatial:pm"] == 3
    assert decoder.edge_attr_dims_by_rel["pm:temporal:pm"] == 5

    z_src = torch.randn(2, 7)
    z_dst = torch.randn(2, 7)
    assert decoder.predict(z_src, z_dst, "pm:spatial:pm").shape == (2, 3)
    assert decoder.predict(z_src, z_dst, "pm:temporal:pm").shape == (2, 5)
    assert (tmp_path / "relation_diagnostics.json").exists()
