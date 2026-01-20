
import pytest
import pandas as pd
import polars as pl
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import compute_pm_features

def test_gnn_batch_consistency(sample_flow_data):
    """
    Goal: Verify that computing features on the full dataset yields the same result
          as splitting the dataset, computing separately, and concatenating.
    """
    # 1. Compute on full dataset
    df_full = compute_pm_features(
        flujos_df=sample_flow_data,
        interactive=False,
        dt_minutes=5
    )
    
    # 2. Split dataset into 2 chunks (halves)
    mid_idx = len(sample_flow_data) // 2
    part1 = sample_flow_data.iloc[:mid_idx]
    part2 = sample_flow_data.iloc[mid_idx:]
    
    # 3. Compute on chunks
    df_part1 = compute_pm_features(part1, interactive=False, dt_minutes=5)
    df_part2 = compute_pm_features(part2, interactive=False, dt_minutes=5)
    
    # 4. Concatenate results
    df_batched = pd.concat([df_part1, df_part2], ignore_index=True)
    
    # Sort both to ensure order matches for comparison
    df_full = df_full.sort_values(["portico", "ts_min"]).reset_index(drop=True)
    df_batched = df_batched.sort_values(["portico", "ts_min"]).reset_index(drop=True)
    
    # 5. Compare
    # Check total flow
    total_flow_full = df_full["flujo_total"].sum()
    total_flow_batched = df_batched["flujo_total"].sum()
    
    assert abs(total_flow_full - total_flow_batched) < 1e-5, f"Total flow mismatch: {total_flow_full} vs {total_flow_batched}"
    
    # Check disjoint split match
    unique_porticos = sample_flow_data["portico"].unique()
    if len(unique_porticos) > 1:
        p1_data = sample_flow_data[sample_flow_data["portico"] == unique_porticos[0]]
        p2_data = sample_flow_data[sample_flow_data["portico"] == unique_porticos[1]]
        
        df_p1 = compute_pm_features(p1_data, interactive=False, dt_minutes=5)
        df_p2 = compute_pm_features(p2_data, interactive=False, dt_minutes=5)
        df_batched_clean = pd.concat([df_p1, df_p2], ignore_index=True).sort_values(["portico", "ts_min"]).reset_index(drop=True)
        
        # Now this should match df_full exactly because keys are disjoint.
        pd.testing.assert_frame_equal(df_full, df_batched_clean, check_like=True)
