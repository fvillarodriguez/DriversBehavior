
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
    _infer_edge_feature_dim
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
    
    
