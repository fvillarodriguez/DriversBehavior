
import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gat_model import HeteroGAT
from src.graph_builder_app import _infer_edge_feature_dim

def test_gat_model_training_step(dummy_graph_data):
    """
    Test Goal: Verify that the GNN model can perform a complete training step:
    - Forward pass
    - Loss calculation
    - Backward pass (Gradient computation)
    - Optimizer step
    """
    data = dummy_graph_data
    in_channels = data['pm'].x.shape[1]
    out_channels = 2
    edge_dim = _infer_edge_feature_dim(data)
    
    # 1. Initialize Model
    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=out_channels,
        num_heads=2,
        dropout=0.1,
        edge_feature_dim=edge_dim,
        num_layers=2
    )
    
    # 2. Setup Optimizer and Criteria
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    optimizer.zero_grad()
    
    # 3. Forward Pass
    edge_attr_dict = {
        ('pm', 'spatial', 'pm'): data['pm', 'spatial', 'pm'].edge_attr,
        ('pm', 'temporal', 'pm'): data['pm', 'temporal', 'pm'].edge_attr
    }
    
    # Simulate a batch (using full graph here for simplicity)
    x_dict, _, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    
    # 4. Compute Loss
    logits = x_dict['pm']
    target = data['pm'].y
    print(f"Logits shape: {logits.shape}, Target shape: {target.shape}")
    loss = criterion(logits, target)
    
    # Check loss is valid
    assert torch.isfinite(loss)
    assert loss.item() > 0
    
    # 5. Backward Pass
    loss.backward()
    
    # 6. Verify Gradients
    # Check that at least one parameter has gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            assert torch.isfinite(param.grad).all()
            # break # check all?
    
    assert has_grads, "Model parameters should have gradients after backward pass"
    
    # 7. Optimizer Step
    optimizer.step()
    
    # 8. Check that weights changed (optional but good)
    # Ideally checking before/after, but basic gradient check is sufficient for 'can train' verification.
