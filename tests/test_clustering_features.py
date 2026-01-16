import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.clustering import Clusterization, FlowColumns

def test_clustering_features_generation():
    # Setup dummy flow data
    # Needs: timestamp, speed, portico, lane, plate
    data = {
        "FECHA": pd.to_datetime([
            "2023-01-01 10:00:00", "2023-01-01 10:00:10", 
            "2023-01-01 10:00:20", "2023-01-01 10:01:00"
        ]),
        "VELOCIDAD": [100, 90, 80, 95],
        "PORTICO": ["P1", "P1", "P1", "P1"],
        "CARRIL": [1, 1, 1, 2],
        "MATRICULA": ["AAAAA1", "AAAAA1", "AAAAA1", "AAAAA2"]
    }
    df = pd.DataFrame(data)
    
    flow_cols = FlowColumns(
        timestamp="FECHA",
        speed_kmh="VELOCIDAD",
        portico="PORTICO",
        lane="CARRIL",
        plate_id="MATRICULA"
    )
    
    # Run Clusterization
    result = Clusterization(
        df,
        flow_cols,
        ttc_max_map=None,
        include_counts=True
    )
    
    # Assertions
    assert not result.empty
    assert "plate" in result.columns
    assert "total_passes" in result.columns
    assert "avg_speed_kmh" in result.columns
    
    # Validation
    # AAAAA1 passed 3 times, AAAAA2 passed 1 time
    row_a1 = result[result["plate"] == "AAAAA1"].iloc[0]
    assert row_a1["total_passes"] == 3
    assert row_a1["avg_speed_kmh"] == 90.0 # (100+90+80)/3
    
    row_a2 = result[result["plate"] == "AAAAA2"].iloc[0]
    assert row_a2["total_passes"] == 1
    assert row_a2["avg_speed_kmh"] == 95.0

    # Verify other columns exist
    feature_cols = [
        "avg_relative_speed",
        "avg_headway_s",
        "conflict_rate",
        "lane_prop_1",
        "lane_prop_2",
        "lane_change_rate"
    ]
    for col in feature_cols:
        assert col in result.columns
