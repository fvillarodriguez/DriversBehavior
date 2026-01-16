import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.utils import compute_flow_features, compute_cluster_features

def test_compute_flow_features_consistency():
    # Setup dummy flow data
    data = {
        "FECHA": pd.to_datetime([
            "2023-01-01 10:00:00", "2023-01-01 10:01:00", 
            "2023-01-01 10:02:00", "2023-01-01 10:06:00"
        ]),
        "VELOCIDAD": [80, 85, 90, 88],
        "CATEGORIA": [1, 1, 2, 1],
        "MATRICULA": ["A1", "A2", "B1", "A3"],
        "PORTICO": ["P1", "P1", "P1", "P1"]
    }
    df = pd.DataFrame(data)
    
    # Run function
    result = compute_flow_features(
        df,
        interval_minutes=5,
        lanes=2,
        metrics=["flow", "speed"]
    )
    
    # Assertions
    assert not result.empty
    assert "portico" in result.columns
    assert "interval_start" in result.columns
    
    # Check if we have expected columns based on categories (Light, Heavy)
    # Categories: 1->Light, 2->Heavy.
    expected_cols = [
        "flow_light", "speed_light", 
        "flow_heavy", "speed_heavy"
    ]
    for col in expected_cols:
        assert col in result.columns
        
    # Check consistency of values (basic check)
    # First interval (10:00) has 3 cars: 2 Light (80, 85), 1 Heavy (90)
    row0 = result.iloc[0]
    assert row0["flow_light"] > 0
    assert row0["speed_light"] == 82.5  # (80+85)/2
    assert row0["speed_heavy"] == 90.0

def test_compute_cluster_features_consistency():
    # Setup dummy flow data
    flow_data = {
        "FECHA": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:01:00"]),
        "VELOCIDAD": [80, 85],
        "MATRICULA": ["A1", "A2"],
        "PORTICO": ["P1", "P1"]
    }
    flows_df = pd.DataFrame(flow_data)
    
    # Setup dummy cluster labels
    cluster_data = {
        "plate": ["A1", "A2"],
        "cluster_label": [0, 1]
    }
    cluster_df = pd.DataFrame(cluster_data)
    
    # Run function
    result = compute_cluster_features(
        flows_df,
        cluster_df,
        interval_minutes=5,
        include_counts=True,
        include_speed=True
    )
    
    # Assertions
    assert not result.empty
    assert "portico" in result.columns
    assert "interval_start" in result.columns
    
    # Check cluster columns
    expected_cols = [
        "cluster_share_0", "cluster_share_1",
        "cluster_flow_0", "cluster_flow_1",
        "cluster_speed_0", "cluster_speed_1"
    ]
    for col in expected_cols:
        assert col in result.columns

