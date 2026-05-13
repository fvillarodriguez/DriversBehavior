from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _sparse_tensor(torch, edges, values, size):
    index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.sparse_coo_tensor(index, torch.tensor(values, dtype=torch.float32), size).coalesce()


def _write_synthetic_state(tmp_path: Path):
    torch = pytest.importorskip("torch")
    state_dir = tmp_path / "MA"
    (state_dir / "Edges").mkdir(parents=True)
    (state_dir / "Nodes").mkdir(parents=True)

    road_edges = [(0, 1), (1, 2), (2, 3), (1, 3)]
    adj = _sparse_tensor(torch, road_edges, [1, 1, 1, 1], (4, 4))
    torch.save(adj, state_dir / "adj_matrix.pt")

    length = _sparse_tensor(torch, road_edges, [10, 20, 30, 40], (4, 4))
    oneway = _sparse_tensor(torch, road_edges, [1, 0, 1, 0], (4, 4))
    torch.save({"length": length, "oneway": oneway}, state_dir / "Edges" / "edge_features.pt")
    traffic = _sparse_tensor(torch, road_edges, [1000, 2000, 3000, 4000], (4, 4))
    torch.save({"AADT": traffic}, state_dir / "Edges" / "edge_features_traffic_2022.pt")

    accidents = pd.DataFrame(
        [
            {"year": 2022, "month": 1, "node_1_idx": 0, "node_2_idx": 1, "acc_count": 2},
            {"year": 2022, "month": 2, "node_1_idx": 1, "node_2_idx": 2, "acc_count": 1},
            {"year": 2022, "month": 3, "node_1_idx": 2, "node_2_idx": 3, "acc_count": 1},
        ]
    )
    accidents.to_csv(state_dir / "accidents_monthly.csv", index=False)

    for month in (1, 2, 3):
        df = pd.DataFrame(
            {
                "tavg": [10 + month, 11 + month, 12 + month, 13 + month],
                "tmin": [1, 2, 3, 4],
                "tmax": [20, 21, 22, 23],
                "prcp": [0.0, 0.1, 0.2, 0.3],
                "wspd": [5, 6, 7, 8],
                "pres": [1000, 1001, 1002, 1003],
            }
        )
        df.to_csv(state_dir / "Nodes" / f"node_features_2022_{month}.csv", index=False)
    return state_dir


def test_build_graph_adapter_contract(tmp_path):
    pytest.importorskip("torch_geometric")
    from ml4roadsafety_validation.build_graph import build_ml4roadsafety_graph

    _write_synthetic_state(tmp_path)
    data = build_ml4roadsafety_graph(
        data_dir=tmp_path,
        state="MA",
        months=["2022-01", "2022-02", "2022-03"],
        max_segments=4,
        seed=7,
    )

    assert data["pm"].x.shape[0] == 12
    assert data["pm"].y.shape == (12,)
    assert set(data["pm"].y.tolist()) <= {0, 1}
    assert int(data["pm"].train_mask.sum()) == 4
    assert int(data["pm"].val_mask.sum()) == 4
    assert int(data["pm"].test_mask.sum()) == 4
    overlap = (
        data["pm"].train_mask.long()
        + data["pm"].val_mask.long()
        + data["pm"].test_mask.long()
    )
    assert int(overlap.max().item()) == 1
    for edge_type in data.edge_types:
        assert data[edge_type].edge_index.shape[0] == 2
        assert data[edge_type].edge_attr.shape[0] == data[edge_type].edge_index.shape[1]
    assert data["pm", "temporal", "pm"].edge_index.shape[1] == 8
    assert data.ml4rs_metadata["normalization"]["fit_split"] == "train"


def test_heterogat_forward_on_synthetic_adapter(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from ml4roadsafety_validation.build_graph import build_ml4roadsafety_graph
    from ml4roadsafety_validation.run_pilot import _edge_feature_dims
    from src.gat_model import HeteroGAT

    _write_synthetic_state(tmp_path)
    data = build_ml4roadsafety_graph(
        data_dir=tmp_path,
        state="MA",
        months=["2022-01", "2022-02", "2022-03"],
        max_segments=4,
        seed=7,
    )
    edge_dims = _edge_feature_dims(data)
    model = HeteroGAT(
        in_channels=int(data["pm"].x.shape[1]),
        hidden_channels=8,
        out_channels=2,
        num_heads=1,
        dropout=0.0,
        edge_feature_dim=max(edge_dims.values()),
        edge_feature_dims=edge_dims,
        edge_types=tuple(data.edge_types),
        num_layers=1,
        use_residual=True,
    )
    edge_attr_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}
    with torch.no_grad():
        logits, _, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    assert logits["pm"].shape == (data["pm"].num_nodes, 2)
    assert torch.isfinite(logits["pm"]).all()

