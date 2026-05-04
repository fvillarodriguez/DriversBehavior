
import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gat_model import HeteroGAT
from src.graphsmote import smote_nodes, Z2XDecoders
from src.graph_builder_app import (
    _network_config_to_hparams,
    _compute_n_syn_from_ratio,
    _infer_edge_feature_dim,
    _apply_temporal_split_to_graph,
    _attach_graph_metadata_hash,
    _build_edge_feature_extras,
    _build_raw_global_edge_features,
    _compute_normalization_stats_from_train,
    _compute_temporal_split_masks,
    _compute_graph_semantic_hash,
    _resolve_graph_hash_for_loaded_graph,
    PMIndex,
)



@pytest.fixture
def model_config():
    return {
        "hidden_channels": 32,
        "num_heads": 2,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 0.01,
        "weight_decay": 1e-4
    }

def test_gat_model_forward(dummy_graph_data):
    """
    Test Step 1: Verify GAT Model Initialization and Forward Pass
    Goal: Ensure the GNN model architecture works with provided data.
    """
    data = dummy_graph_data
    in_channels = data['pm'].x.shape[1]
    out_channels = 2
    edge_dim = _infer_edge_feature_dim(data)
    
    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=out_channels,
        num_heads=2,
        dropout=0.1,
        edge_feature_dim=edge_dim,
        num_layers=2
    )
    
    # Run forward pass
    edge_attr_dict = {
        ('pm', 'spatial', 'pm'): data['pm', 'spatial', 'pm'].edge_attr,
        ('pm', 'temporal', 'pm'): data['pm', 'temporal', 'pm'].edge_attr
    }
    
    x_dict, _, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    
    # Assertions
    assert 'pm' in x_dict
    assert x_dict['pm'].shape == (data['pm'].num_nodes, out_channels)
    assert torch.isfinite(x_dict['pm']).all()

def test_smote_nodes_logic():
    """
    Test Step 2: Verify GraphSMOTE Node Generation
    Goal: Verify synthetic node generation logic works for imbalanced classes.
    """
    # Create synthetic imbalanced data
    # Class 0: 100 samples
    # Class 1: 5 samples (Minority)
    z_maj = torch.randn(100, 10)
    z_min = torch.randn(5, 10) + 5 # Shift mean to make them distinct
    
    z = torch.cat([z_maj, z_min], dim=0)
    y = torch.cat([torch.zeros(100), torch.ones(5)], dim=0).long()
    
    n_new = 10
    k_neighbors = 2
    
    syn_features, syn_labels, _ = smote_nodes(
        z, y, 
        minority_class=1, 
        k=k_neighbors, 
        n_samples=n_new, 
        random_state=42
    )
    
    # Assertions
    assert syn_features.shape == (n_new, 10)
    assert syn_labels.shape == (n_new,)
    assert (syn_labels == 1).all() # All should be minority class
    assert torch.isfinite(syn_features).all()

def test_compute_n_syn_from_ratio():
    """
    Test Step 3: Verify Synthetic Count Calculation
    Goal: Check if correct number of synthetic nodes is calculated based on ratio.
    """
    y = torch.cat([torch.zeros(90), torch.ones(10)]) # 10% Positives
    train_mask = torch.ones(100, dtype=torch.bool)
    
    # Target: 50% Positives
    # Currently: 10 Pos, 90 Neg (Total 100)
    # Target Ratio = 0.5 => Pos / (Pos + Neg) = 0.5
    # (10 + New) / (100 + New) = 0.5 
    # 10 + New = 50 + 0.5*New => 0.5*New = 40 => New = 80
    
    n_syn = _compute_n_syn_from_ratio(
        y, train_mask, 
        minority_class=1, 
        target_pos_ratio=0.5
    )
    
    # Allow for rounding differences +/- 1
    assert 79 <= n_syn <= 81 


def test_temporal_split_uses_full_timeline_and_purges_label_window():
    ts_values = np.arange(10) * 5

    split = _compute_temporal_split_masks(
        ts_values,
        train_ratio=60,
        val_ratio=20,
        label_lookahead_minutes=5,
    )

    meta = split["metadata"]
    assert meta["timeline_count"] == 10
    assert meta["train_ts_cutoff"] == 30.0
    assert meta["val_ts_cutoff"] == 40.0
    assert np.where(split["train_mask"])[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.where(split["val_mask"])[0].tolist() == [7]
    assert np.where(split["test_mask"])[0].tolist() == [9]
    assert not bool(split["train_mask"][6])
    assert not bool(split["val_mask"][8])


def test_apply_temporal_split_respects_metadata_lookahead():
    data = HeteroData()
    data["pm"].x = torch.randn(10, 2)
    data["pm"].y = torch.zeros(10, dtype=torch.long)
    pm_map = {("P1", int(idx * 5)): idx for idx in range(10)}
    pm_rev = {idx: ("P1", int(idx * 5)) for idx in range(10)}
    pm_index = PMIndex(pm_map, pm_rev)

    split_info = _apply_temporal_split_to_graph(
        data,
        pm_index,
        train_ratio=60,
        val_ratio=20,
        graph_metadata={"labeling": {"label_lookahead_minutes": 5}},
    )

    assert split_info["label_lookahead_minutes"] == 5
    assert split_info["label_lookahead_source"] == "metadata"
    assert data["pm"].train_mask.nonzero(as_tuple=False).view(-1).tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert data["pm"].val_mask.nonzero(as_tuple=False).view(-1).tolist() == [7]
    assert data["pm"].test_mask.nonzero(as_tuple=False).view(-1).tolist() == [9]


def test_normalization_stats_use_purged_train_mask_only():
    df = pd.DataFrame(
        {
            "ts_min": np.arange(10) * 5,
            "flow": np.arange(10, dtype=float),
        }
    )
    split = _compute_temporal_split_masks(
        df["ts_min"].to_numpy(),
        train_ratio=60,
        val_ratio=20,
        label_lookahead_minutes=5,
    )

    mu, sigma = _compute_normalization_stats_from_train(
        df,
        ["flow"],
        split["train_mask"],
    )

    assert mu[0] == pytest.approx(np.mean(np.arange(6, dtype=float)))
    assert sigma[0] == pytest.approx(np.std(np.arange(6, dtype=float)) + 1e-9)


def test_raw_global_edge_features_are_computed_before_normalization():
    df_raw = pd.DataFrame(
        {
            "portico": ["A", "B"],
            "ts_min": [0, 0],
            "flujo_total": [100.0, 140.0],
            "densidad_total": [20.0, 35.0],
            "v_armonica": [60.0, 55.0],
            "prop_pesados": [0.10, 0.25],
        }
    )
    df_normalized = df_raw.copy()
    for col in ["flujo_total", "densidad_total", "v_armonica", "prop_pesados"]:
        df_normalized[col] = (
            df_normalized[col] - df_normalized[col].mean()
        ) / df_normalized[col].std(ddof=0)

    raw_globals = _build_raw_global_edge_features(df_raw, "portico")

    assert raw_globals["q_total_raw"].tolist() == [100.0, 140.0]
    assert raw_globals["k_total_raw"].tolist() == [20.0, 35.0]
    assert raw_globals["v_armonica_raw"].tolist() == [60.0, 55.0]
    assert raw_globals["prop_pesados_raw"].tolist() == [0.10, 0.25]
    assert raw_globals["q_total_raw"].tolist() != df_normalized["flujo_total"].tolist()


def test_raw_global_edge_features_support_snapshot_schema():
    df_raw = pd.DataFrame(
        {
            "portico": ["A", "B"],
            "ts_min": [0, 0],
            "flow_total": [100.0, 140.0],
            "speed_harmonic": [50.0, 35.0],
            "slow_pct": [0.10, 0.35],
            "mix_cat_2": [0.20, 0.40],
        }
    )

    raw_globals = _build_raw_global_edge_features(df_raw, "portico")

    assert raw_globals.attrs["raw_feature_family"] == "snapshot"
    assert raw_globals.attrs["edge_feature_sources"]["q"] == "flow_total"
    assert raw_globals["q_total_raw"].tolist() == [100.0, 140.0]
    assert raw_globals["v_armonica_raw"].tolist() == [50.0, 35.0]
    assert raw_globals["k_total_raw"].tolist() == [2.0, 4.0]
    assert raw_globals["prop_pesados_raw"].tolist() == [0.20, 0.40]
    assert raw_globals["slow_pct_raw"].tolist() == [0.10, 0.35]


def test_raw_global_edge_features_snapshot_fallbacks():
    df_raw = pd.DataFrame(
        {
            "portico": ["A", "B"],
            "ts_min": [0, 0],
            "flow_total": [120.0, 80.0],
            "speed_mean": [60.0, 40.0],
            "flow_cat_2": [30.0, 20.0],
        }
    )

    raw_globals = _build_raw_global_edge_features(df_raw, "portico")

    assert raw_globals.attrs["raw_feature_family"] == "snapshot"
    assert raw_globals.attrs["edge_feature_sources"]["v"] == "speed_mean"
    assert raw_globals.attrs["edge_feature_sources"]["pi"] == "flow_cat_2/flow_total"
    assert raw_globals["v_armonica_raw"].tolist() == [60.0, 40.0]
    assert raw_globals["k_total_raw"].tolist() == [2.0, 2.0]
    assert raw_globals["prop_pesados_raw"].tolist() == pytest.approx([0.25, 0.25])
    assert raw_globals["slow_pct_raw"].tolist() == [0.0, 0.0]


def test_edge_feature_extras_recompute_directional_and_st_fwd_gradients():
    df_raw = pd.DataFrame(
        {
            "portico": ["A", "B", "A", "B"],
            "ts_min": [0, 0, 5, 5],
            "flow_total": [100.0, 150.0, 110.0, 210.0],
            "speed_mean": [50.0, 40.0, 55.0, 35.0],
            "slow_pct": [0.10, 0.30, 0.20, 0.60],
            "mix_cat_2": [0.10, 0.20, 0.10, 0.30],
        }
    )
    raw_globals = _build_raw_global_edge_features(df_raw, "portico")
    features = ["dist_km", "grad_q", "grad_v", "grad_slow_pct"]

    spatial_edges = pd.DataFrame(
        {
            "portico_src": ["A"],
            "portico_dst": ["B"],
            "ts_min": [0],
            "dist_km": [2.0],
        }
    )
    spatial = _build_edge_feature_extras(
        spatial_edges,
        raw_globals,
        "portico",
        features,
    )

    back_edges = pd.DataFrame(
        {
            "portico_src": ["B"],
            "portico_dst": ["A"],
            "ts_min": [0],
            "dist_km": [2.0],
        }
    )
    back = _build_edge_feature_extras(
        back_edges,
        raw_globals,
        "portico",
        features,
    )

    st_edges = pd.DataFrame(
        {
            "portico_src": ["A"],
            "portico_dst": ["B"],
            "ts_min_src": [0],
            "ts_min_dst": [5],
            "dist_km": [2.0],
        }
    )
    st_fwd = _build_edge_feature_extras(
        st_edges,
        raw_globals,
        "portico",
        features,
        src_ts_col="ts_min_src",
        dst_ts_col="ts_min_dst",
    )

    assert spatial[0].tolist() == pytest.approx([2.0, 25.0, -5.0, 0.10])
    assert back[0].tolist() == pytest.approx([2.0, -25.0, 5.0, -0.10])
    assert st_fwd[0].tolist() == pytest.approx([2.0, 55.0, -7.5, 0.25])


def test_attach_graph_metadata_hash_records_reproducibility_payload():
    data = HeteroData()
    data["pm"].x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    data["pm"].y = torch.tensor([0, 1], dtype=torch.long)
    data["pm"].train_mask = torch.tensor([True, False])
    data["pm"].val_mask = torch.tensor([False, True])
    data["pm"].test_mask = torch.tensor([False, False])
    data[("pm", "temporal", "pm")].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    data[("pm", "temporal", "pm")].edge_attr = torch.tensor([[0.5]], dtype=torch.float32)
    pm_index = PMIndex({("A", 0): 0, ("A", 5): 1}, {0: ("A", 0), 1: ("A", 5)})
    split_metadata = {
        "strategy": "temporal_purged",
        "train_ts_cutoff": 0.0,
        "val_ts_cutoff": 5.0,
        "train_count": 1,
        "val_count": 1,
        "test_count": 0,
    }
    edge_config = {"physical_features": ["dist_km"], "delta_features": ["flow"]}
    metadata = {
        "schema_version": 2,
        "labeling": {"label_lookahead_minutes": 5},
        "split": split_metadata,
        "edge_config": edge_config,
    }

    metadata_out, graph_hash = _attach_graph_metadata_hash(
        data=data,
        pm_index=pm_index,
        feature_cols=["flow", "speed"],
        split_metadata=split_metadata,
        edge_config=edge_config,
        metadata=metadata,
    )

    assert len(graph_hash) == 64
    assert metadata_out["graph_hash"] == graph_hash
    assert metadata_out["split"]["train_count"] == 1
    assert metadata_out["labeling"]["label_lookahead_minutes"] == 5
    assert metadata_out["edge_config"]["physical_features"] == ["dist_km"]
    assert data.graph_metadata["graph_hash"] == graph_hash


def test_graph_semantic_hash_ignores_non_semantic_metadata_fields():
    data = HeteroData()
    data["pm"].x = torch.tensor([[1.0], [2.0]])
    data["pm"].y = torch.tensor([0, 1], dtype=torch.long)
    data["pm"].train_mask = torch.tensor([True, False])
    data["pm"].val_mask = torch.tensor([False, True])
    data["pm"].test_mask = torch.tensor([False, False])
    data[("pm", "temporal", "pm")].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    pm_index = PMIndex({("A", 0): 0, ("A", 5): 1}, {0: ("A", 0), 1: ("A", 5)})
    split_metadata = {"strategy": "temporal_purged", "train_count": 1}
    edge_config = {"edge_types": ["temporal"]}
    metadata_a = {
        "schema_version": 2,
        "created_at": "2026-05-04T10:00:00",
        "normalization": {"stats_path": "/tmp/a/stats.json", "feature_count": 1},
        "event_source": {"loaded_acc_path": "/tmp/a/events.csv", "event_files": ["/tmp/a/events.csv"]},
    }
    metadata_b = {
        "schema_version": 2,
        "created_at": "2026-05-04T11:00:00",
        "normalization": {"stats_path": "/tmp/b/stats.json", "feature_count": 1},
        "event_source": {"loaded_acc_path": "/tmp/b/events.csv", "event_files": ["/tmp/b/events.csv"]},
    }

    hash_a = _compute_graph_semantic_hash(
        data=data,
        pm_index=pm_index,
        feature_cols=["flow"],
        split_metadata=split_metadata,
        edge_config=edge_config,
        metadata=metadata_a,
    )
    hash_b = _compute_graph_semantic_hash(
        data=data,
        pm_index=pm_index,
        feature_cols=["flow"],
        split_metadata=split_metadata,
        edge_config=edge_config,
        metadata=metadata_b,
    )

    assert hash_a == hash_b


def test_resolve_graph_hash_prefers_semantic_metadata_over_file_hash():
    data = HeteroData()
    data.graph_metadata = {"graph_hash": "a" * 64}

    graph_hash = _resolve_graph_hash_for_loaded_graph(
        {
            "data": data,
            "metadata": {"graph_hash": "b" * 64},
            "graph_hash": "c" * 64,
            "filename": "missing_graph.pt",
        }
    )

    assert graph_hash == "c" * 64

def test_network_config_translation():
    """
    Test Step 4: Verify Config Translation
    Goal: Ensure parameter mapping function handles graphsmote flag correctly.
    """
    cfg = {
        "hidden_channels": 64,
        "smote_k": 5,
        "target_pos_ratio": 0.4
    }
    
    # Case 1: Use GraphSMOTE = True
    hparams_smote = _network_config_to_hparams(cfg, use_graphsmote=True)
    assert hparams_smote["use_graphsmote"] is True
    assert hparams_smote["smote_k"] == 5
    assert hparams_smote["target_pos_ratio"] == 0.4
    
    # Case 2: Use GraphSMOTE = False
    hparams_no_smote = _network_config_to_hparams(cfg, use_graphsmote=False)
    assert hparams_no_smote["use_graphsmote"] is False
    assert "smote_k" not in hparams_no_smote # Should not be present or not used
    
    
