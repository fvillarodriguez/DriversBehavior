
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import compute_flow_features

def generate_synthetic_flows(
    start_time="2024-01-01 08:00",
    end_time="2024-01-01 12:00",
    interval_seconds=60,
    porticos=["P1"],
    categories=[1],
):
    """Generate synthetic flow data."""
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{interval_seconds}s")
    data = []
    
    for ts in timestamps:
        for portico in porticos:
            for cat in categories:
                # Random speed and flow
                # We add some entries per timestamp/portico/cat
                n_vehicles = np.random.randint(1, 5)
                for _ in range(n_vehicles):
                    data.append({
                        "FECHA": ts,
                        "PORTICO": portico,
                        "CATEGORIA": cat,
                        "VELOCIDAD": np.random.normal(80, 10),
                    })
    
    return pd.DataFrame(data)

def test_compute_flow_features_batch_consistency():
    """
    Verify that computing features in batches yields consistent results with full processing,
    except for known boundary effects (delta variables).
    """
    # 1. Setup Data
    # 4 hours of data, enough for multiple batches
    flows_df = generate_synthetic_flows(
        start_time="2024-01-01 08:00:00",
        end_time="2024-01-01 12:00:00",
        interval_seconds=30
    )
    
    # Define parameters
    interval_minutes = 15
    # Use simple category map for testing
    cat_remap = {1: 1}
    cat_labels = {1: "Light"}
    
    # 2. Compute Full Features
    full_features = compute_flow_features(
        flows_df,
        interval_minutes=interval_minutes,
        category_remap=cat_remap,
        category_labels=cat_labels,
        metrics=["flow", "speed", "density", "delta_speed", "delta_density"],
        progress=None
    )
    
    # Sort for consistent comparison
    full_features = full_features.sort_values(["portico", "interval_start"]).reset_index(drop=True)
    
    # 3. Compute Batched Features
    # Split input data into 2-hour chunks
    mid_point = pd.Timestamp("2024-01-01 10:00:00")
    batch1_df = flows_df[flows_df["FECHA"] < mid_point].copy()
    batch2_df = flows_df[flows_df["FECHA"] >= mid_point].copy()
    
    # Compute for each batch
    b1_features = compute_flow_features(
        batch1_df,
        interval_minutes=interval_minutes,
        category_remap=cat_remap,
        category_labels=cat_labels,
        metrics=["flow", "speed", "density", "delta_speed", "delta_density"],
        progress=None
    )
    
    b2_features = compute_flow_features(
        batch2_df,
        interval_minutes=interval_minutes,
        category_remap=cat_remap,
        category_labels=cat_labels,
        metrics=["flow", "speed", "density", "delta_speed", "delta_density"],
        progress=None
    )
    
    # Combine batches
    batch_combined = pd.concat([b1_features, b2_features], ignore_index=True)
    batch_combined = batch_combined.sort_values(["portico", "interval_start"]).reset_index(drop=True)
    
    # 4. Compare Results
    # We expect full consistency for static metrics: flow, speed, density
    # We expect discrepancies for delta metrics at the specific batch boundary
    
    # Identify the boundary interval where batch 2 starts
    # Since batch 2 starts at 10:00:00, the first interval is 10:00-10:15
    boundary_start = pd.Timestamp("2024-01-01 10:00:00")
    
    # Static columns check
    static_cols = [c for c in full_features.columns if "delta" not in c]
    pd.testing.assert_frame_equal(
        full_features[static_cols], 
        batch_combined[static_cols], 
        obj="Static Features (Flow, Speed, Density)"
    )
    
    # Delta columns check
    # We exclude the boundary interval from comparison because .diff() will be NaN/0 in batch mode
    # for the first element, whereas full mode has the previous value.
    mask_not_boundary = full_features["interval_start"] != boundary_start
    
    full_deltas_safe = full_features.loc[mask_not_boundary].copy()
    batch_deltas_safe = batch_combined.loc[mask_not_boundary].copy()
    
    # We rely on previous frame_equal for static cols, so the indexing is aligned. 
    # Just need to check values.
    # Note: we also might need to check if there are NaNs that need filling, 
    # but compute_flow_features fills NaNs with 0 in delta calculation.
    
    pd.testing.assert_frame_equal(
        full_deltas_safe.reset_index(drop=True),
        batch_deltas_safe.reset_index(drop=True),
        obj="Delta Features (excluding batch boundary)"
    )
    
    # Verify the boundary discrepancy exists (optional, but good for validity)
    # The batch result for delta at boundary should be 0 (no history), 
    # while full result should likely be non-zero (if there was change).
    
    full_boundary = full_features.loc[~mask_not_boundary]
    batch_boundary = batch_combined.loc[~mask_not_boundary]
    
    if not full_boundary.empty:
        # Check one delta column, e.g., delta_speed_light
        # Assuming density/speed changed, full should be != 0, batch should be 0
        delta_cols = [c for c in full_features.columns if "delta" in c]
        for col in delta_cols:
            # Batch value at start of batch is 0 (filledna)
            assert (batch_boundary[col] == 0).all(), \
                f"Batch {col} at boundary should be 0 (no history)"
