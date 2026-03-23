import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustering import Clusterization, FlowColumns, _aggregate_batch_features  # noqa: E402

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


def test_clustering_features_generation_with_monthly_weighting():
    data = {
        "FECHA": pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-01 10:00:10",
                "2023-02-01 10:00:00",
                "2023-02-01 10:00:10",
                "2023-02-02 11:00:00",
                "2023-02-02 11:00:10",
            ]
        ),
        "VELOCIDAD": [100, 90, 80, 70, 60, 50],
        "PORTICO": ["P1", "P1", "P1", "P1", "P1", "P1"],
        "CARRIL": [1, 1, 1, 1, 2, 2],
        "MATRICULA": ["AAAAA1", "AAAAA1", "AAAAA1", "AAAAA1", "BBBBB2", "BBBBB2"],
    }
    df = pd.DataFrame(data)

    flow_cols = FlowColumns(
        timestamp="FECHA",
        speed_kmh="VELOCIDAD",
        portico="PORTICO",
        lane="CARRIL",
        plate_id="MATRICULA",
    )

    result = Clusterization(
        df,
        flow_cols,
        ttc_max_map=None,
        monthly_weighting=True,
        include_counts=False,
    )

    assert not result.empty
    row_a1 = result[result["plate"] == "AAAAA1"].iloc[0]
    row_b2 = result[result["plate"] == "BBBBB2"].iloc[0]

    assert row_a1["total_passes"] == 4
    assert row_a1["n_months_active"] == 2
    assert row_a1["avg_speed_kmh"] == pytest.approx(85.0)
    assert row_b2["total_passes"] == 2
    assert row_b2["n_months_active"] == 1


def test_aggregate_batch_features_monthly_weighting():
    batches_df = pd.DataFrame(
        {
            "plate": ["A", "A", "B"],
            "total_passes": [2, 4, 3],
            "avg_speed_kmh": [10.0, 20.0, 30.0],
            "avg_relative_speed": [1.0, 2.0, 3.0],
            "avg_headway_s": [5.0, 10.0, 15.0],
            "conflict_rate": [0.1, 0.2, 0.3],
            "lane_prop_1": [1.0, 0.5, 0.0],
            "lane_prop_2": [0.0, 0.5, 1.0],
            "lane_prop_3": [0.0, 0.0, 0.0],
            "lane_changes": [1, 2, 1],
            "n_days_active": [1, 2, 3],
            "n_months_active": [1, 1, 1],
        }
    )

    result = _aggregate_batch_features(
        batches_df,
        batch_mode="month",
        monthly_weighting=True,
    )
    result = result.set_index("plate")

    assert result.loc["A", "total_passes"] == 6
    assert result.loc["A", "n_months_active"] == 2
    assert result.loc["A", "avg_speed_kmh"] == pytest.approx((10.0 * 2 + 20.0 * 4) / 6)
    assert result.loc["A", "lane_change_rate"] == pytest.approx(3 / 4)
