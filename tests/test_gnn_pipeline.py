
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
from src.graphsmote import (
    _smote_in_z_space,
    Z2XDecoders,
    RelEdgeGen,
    evaluate_edge_generator_auc,
    evaluate_embedding_separability,
)
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
    _resolve_graph_identity_for_loaded_graph,
    _resolve_graph_hash_for_loaded_graph,
    _default_gnn_objective_metrics,
    _split_val_mask_for_calibration_threshold,
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
    Goal: Verify synthetic node generation logic works for imbalanced classes,
    using the canonical helper (_smote_in_z_space) with α~U[0,1] (Eq. 4 del paper).
    """
    # Class 0: 100 samples
    # Class 1: 5 samples (Minority)
    z_maj = torch.randn(100, 10)
    z_min = torch.randn(5, 10) + 5  # shift mean to make them distinct

    z = torch.cat([z_maj, z_min], dim=0)
    y = torch.cat([torch.zeros(100), torch.ones(5)], dim=0).long()

    n_new = 10
    k_neighbors = 2

    rng = np.random.RandomState(42)
    syn_features, syn_labels = _smote_in_z_space(
        z, y,
        minority_class=1,
        k=k_neighbors,
        n_samples=n_new,
        rng=rng,
        return_renormalized=True,
    )

    assert syn_features.shape == (n_new, 10)
    assert syn_labels.shape == (n_new,)
    assert (syn_labels == 1).all()
    assert torch.isfinite(syn_features).all()
    # Re-normalized: should be on the unit hypersphere (within fp tolerance)
    norms = syn_features.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_smote_in_z_space_handles_too_few_minority():
    """If minority has fewer than 2 samples, helper returns empty tensors gracefully."""
    z = torch.randn(50, 8)
    y = torch.zeros(50, dtype=torch.long)
    y[0] = 1  # solo 1 minoritario

    rng = np.random.RandomState(0)
    syn_z, syn_y = _smote_in_z_space(
        z, y, minority_class=1, k=3, n_samples=10, rng=rng
    )
    assert syn_z.numel() == 0
    assert syn_y.numel() == 0


def test_edge_generator_auc_separates_real_from_random():
    """RelEdgeGen entrenado a mano sobre embeddings clusterizados debe dar AUC > 0.8."""
    torch.manual_seed(0)
    np.random.seed(0)

    n = 80
    d = 16
    # Dos clusters bien separados; aristas reales solo dentro de un cluster.
    z_cluster_a = torch.nn.functional.normalize(torch.randn(n // 2, d), p=2, dim=1)
    z_cluster_b = torch.nn.functional.normalize(torch.randn(n // 2, d) + 5.0, p=2, dim=1)
    z = torch.cat([z_cluster_a, z_cluster_b], dim=0)
    z_dict = {'pm': z}

    # Aristas reales: dentro de cluster A (todos los pares (i, i+1))
    edges = []
    for i in range(n // 2 - 1):
        edges.append([i, i + 1])
    ei = torch.tensor(edges, dtype=torch.long).t()
    edge_index_dict = {('pm', 'spatial', 'pm'): ei}

    # Inicializar S como I/sqrt(d) y entrenar 100 pasos contra negativos
    rels = list(edge_index_dict.keys())
    edge_gen = RelEdgeGen({'pm': d}, rels)
    opt = torch.optim.Adam(edge_gen.parameters(), lr=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    key = 'pm:spatial:pm'
    for _ in range(100):
        S = edge_gen.S[key]
        pos = (z[ei[0]] @ S * z[ei[1]]).sum(dim=1)
        neg_dst = torch.randint(0, n, (ei.size(1),))
        neg = (z[ei[0]] @ S * z[neg_dst]).sum(dim=1)
        loss = bce(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad(); loss.backward(); opt.step()

    report = evaluate_edge_generator_auc(
        edge_gen, z_dict, edge_index_dict, holdout_frac=0.5, num_neg=2, min_edges=1
    )
    macro = report.get('_macro')
    assert macro is not None, f"esperaba métricas macro, recibí {report}"
    assert macro['auc'] > 0.8, f"AUC={macro['auc']:.3f} es muy bajo para datos separables"


def test_embedding_separability_probe_returns_sane_metrics():
    """Probe sobre embeddings sintéticos linealmente separables debe dar AUPRC alto."""
    torch.manual_seed(0)
    np.random.seed(0)
    n = 100
    d = 8

    z_neg = torch.randn(n, d)
    z_pos = torch.randn(n, d) + 4.0
    z = torch.cat([z_neg, z_pos], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)], dim=0).long()

    train_mask = torch.zeros(2 * n, dtype=torch.bool)
    val_mask = torch.zeros(2 * n, dtype=torch.bool)
    train_mask[: int(0.8 * n)] = True            # 80 negativos
    train_mask[n : n + int(0.8 * n)] = True       # 80 positivos
    val_mask[int(0.8 * n) : n] = True             # 20 negativos
    val_mask[n + int(0.8 * n) :] = True           # 20 positivos

    data = HeteroData()
    data['pm'].x = z
    data['pm'].y = y
    data['pm'].train_mask = train_mask
    data['pm'].val_mask = val_mask

    out = evaluate_embedding_separability({'pm': z}, data, node_type='pm')
    assert out is not None
    assert out['val_auprc'] > 0.8
    assert out['val_roc_auc'] > 0.8
    assert out['n_train'] == int(train_mask.sum().item())
    assert out['n_val'] == int(val_mask.sum().item())


def test_embedding_separability_probe_returns_none_when_unaligned():
    """Si z no está alineado con todos los nodos, el probe debe salir limpio."""
    data = HeteroData()
    n = 50
    data['pm'].x = torch.randn(n, 4)
    data['pm'].y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    data['pm'].train_mask = torch.tensor([True] * (n // 2) + [False] * (n // 2))
    data['pm'].val_mask = ~data['pm'].train_mask

    # z solo trae train (ej. compute_train_embeddings)
    z_train_only = torch.randn(n // 2, 4)
    out = evaluate_embedding_separability({'pm': z_train_only}, data, node_type='pm')
    assert out is None

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


def test_apply_temporal_split_accepts_top_level_lookahead_metadata():
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
        graph_metadata={"label_lookahead_minutes": 5},
    )

    assert split_info["label_lookahead_minutes"] == 5
    assert split_info["label_lookahead_source"] == "metadata"


def test_apply_temporal_split_strict_rejects_missing_lookahead_metadata():
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
        graph_metadata={},
        strict_lookahead=True,
    )

    assert split_info is None


def test_val_mask_split_uses_time_order_for_calibration_and_threshold():
    data = HeteroData()
    n_nodes = 12
    data["pm"].x = torch.randn(n_nodes, 2)
    data["pm"].y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    data["pm"].val_mask = torch.tensor(
        [False, False, True, True, True, True, True, True, True, True, False, False]
    )
    pm_map = {("P1", int(idx * 5)): idx for idx in range(n_nodes)}
    pm_rev = {idx: ("P1", int(idx * 5)) for idx in range(n_nodes)}
    pm_index = PMIndex(pm_map, pm_rev)

    split = _split_val_mask_for_calibration_threshold(data, pm_index)

    assert split["calibration_mask_source"] == "val_calib"
    assert split["threshold_mask_source"] == "val_threshold"
    assert split["calib_idx"].max().item() < split["threshold_idx"].min().item()
    for idx_key in ("calib_idx", "threshold_idx"):
        labels = set(data["pm"].y[split[idx_key]].tolist())
        assert labels == {0, 1}


def test_val_mask_split_rejects_single_class_side():
    data = HeteroData()
    n_nodes = 8
    data["pm"].x = torch.randn(n_nodes, 2)
    data["pm"].y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    data["pm"].val_mask = torch.ones(n_nodes, dtype=torch.bool)
    pm_map = {("P1", int(idx * 5)): idx for idx in range(n_nodes)}
    pm_rev = {idx: ("P1", int(idx * 5)) for idx in range(n_nodes)}
    pm_index = PMIndex(pm_map, pm_rev)

    with pytest.raises(ValueError, match="ambas clases"):
        _split_val_mask_for_calibration_threshold(data, pm_index)


def test_gnn_objective_defaults_focus_on_extreme_imbalance_metrics():
    assert _default_gnn_objective_metrics() == ["AUPRC", "MCC", "Recall-FAR"]


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


def test_resolve_graph_hash_prefers_semantic_metadata_over_file_hash(tmp_path):
    data = HeteroData()
    data.graph_metadata = {"graph_hash": "a" * 64}
    graph_path = tmp_path / "graph.pt"
    graph_path.write_bytes(b"graph bytes")
    expected_file_hash = __import__("hashlib").sha256(b"graph bytes").hexdigest()

    loaded = {
        "data": data,
        "metadata": {"graph_hash": "b" * 64},
        "graph_hash": "c" * 64,
        "graph_path": str(graph_path),
    }
    graph_hash = _resolve_graph_hash_for_loaded_graph(loaded)
    identity = _resolve_graph_identity_for_loaded_graph(loaded)

    assert graph_hash == "c" * 64
    assert identity["graph_hash"] == "c" * 64
    assert identity["graph_file_hash"] == expected_file_hash
    assert identity["graph_hash_source"] == "semantic_metadata"

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
    
    
