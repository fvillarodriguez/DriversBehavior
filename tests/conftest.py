import os
import sys

import pytest
import pandas as pd
import numpy as np

# Ensure repository root is importable so `from src...` works from any pytest cwd.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

@pytest.fixture
def sample_flow_data():
    """Creates a sample flow dataframe for testing."""
    # Create valid synthetic data
    # 3 categories, 2 porticos, 1 hour of data at 1 min resolution
    data = []
    start_time = pd.Timestamp("2024-01-01 08:00:00")
    
    for portico in ["P1", "P2"]:
        for i in range(20): # 20 data points
            ts = start_time + pd.Timedelta(minutes=i)
            # Add entries for multiple categories
            for cat in [1, 2, 3]:
                # Generate random speed and flows
                for _ in range(5): 
                    data.append({
                        "fecha": ts,
                        "portico": portico,
                        "velocidad": np.random.normal(80, 10), # Mean 80 km/h
                        "categoria": cat,
                        "carril": 1
                    })
    
    return pd.DataFrame(data)

@pytest.fixture
def dummy_graph_data():
    """Creates a basic HeteroData object for testing."""
    torch = pytest.importorskip("torch")
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    data = HeteroData()
    
    # Nodes: 'pm' (Portico-Maestro)
    num_nodes = 50
    in_channels = 16
    data['pm'].x = torch.randn(num_nodes, in_channels)
    data['pm'].y = torch.randint(0, 2, (num_nodes,))
    
    # Edges: 'spatial'
    # Create random edges
    edge_index = torch.randint(0, num_nodes, (2, 200))
    edge_attr = torch.randn(200, 4) # Edge features dim=4
    
    data['pm', 'spatial', 'pm'].edge_index = edge_index
    data['pm', 'spatial', 'pm'].edge_attr = edge_attr
    
    data['pm', 'temporal', 'pm'].edge_index = edge_index
    data['pm', 'temporal', 'pm'].edge_attr = edge_attr

    return data
