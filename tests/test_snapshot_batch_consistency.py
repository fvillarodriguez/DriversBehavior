import pytest
import pandas as pd
import polars as pl
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.snapshot_pipeline import SnapshotConfig, SnapshotFeatureBuilder, flatten_snapshot_features
from src.graph_builder_app import _filter_snapshot_chunk

def create_synthetic_data(start_time, end_time, porticos):
    """Creates a deterministic synthetic flow dataframe."""
    dates = pd.date_range(start_time, end_time, freq="1min")
    data = []
    
    for p in porticos:
        for d in dates:
            # Deterministic speed pattern: 60 + (minute % 30)
            # This ensures we can verify smooth windows
            speed = 60.0 + (d.minute % 30)
            
            # Simple flow pattern
            flow = 10 if d.hour < 10 else 20
            
            data.append({
                "FECHA": d,
                "PORTICO": p,
                "VELOCIDAD": speed,
                "SENTIDO": "N",
                "CATEGORIA": "Light",
                "CARRIL": 1,
            })
            
    return pd.DataFrame(data)

def test_snapshot_batch_consistency():
    """
    Verifies that processing data in batches with overlap yields 
    identical results to processing the entire dataset at once.
    """
    
    # 1. Setup
    start_time = pd.Timestamp("2023-01-01 08:00:00")
    end_time = pd.Timestamp("2023-01-01 12:00:00")
    porticos = [1, 2]
    
    full_df = create_synthetic_data(start_time, end_time, porticos)
    
    # Porticos DF (Mock)
    df_porticos = pd.DataFrame({
        "PORTICO": porticos,
        "km_ruta": [10.0, 15.0],
        "latitud": [0.0, 0.1],
        "longitud": [0.0, 0.1]
    })
    
    # Config
    config = SnapshotConfig(
        dt_minutes=5,
        window_minutes=30, # 6 steps lookback
        slow_speed_kmh=30.0
    )
    
    builder = SnapshotFeatureBuilder(config)
    
    # 2. Run Global (Memory Mode)
    print("\nRunning Global Mode...")
    snap_global, _ = builder.build(pl.from_pandas(full_df), df_porticos)
    df_global = flatten_snapshot_features(snap_global, include_timestamp=True)
    # Apply same filter as batch (exclusive end) because full_df includes end_time inclusive
    df_global = _filter_snapshot_chunk(df_global, start_time, end_time)
    df_global = df_global.sort_values(["portico", "ts_min"]).reset_index(drop=True)
    
    # 3. Run Batched (Batch Mode)
    # Split: Batch 1 [08:00, 10:00), Batch 2 [10:00, 12:00)
    # Batch 2 needs overlap from Batch 1.
    # Overlap needed = window_minutes (30) + a bit of margin. Say 40 min.
    
    mid_time = pd.Timestamp("2023-01-01 10:00:00")
    
    # --- Batch 1 ---
    # Range: 08:00 to 10:00
    # Input: 08:00 to 10:00 (Start is exact for first batch)
    print("Running Batch 1...")
    df_b1_input = full_df[
        (full_df["FECHA"] >= start_time) & 
        (full_df["FECHA"] < mid_time)
    ].copy()
    
    snap_b1, _ = builder.build(pl.from_pandas(df_b1_input), df_porticos)
    df_b1 = flatten_snapshot_features(snap_b1, include_timestamp=True)
    
    # Filter Batch 1: Keep strict range [08:00, 10:00)
    # Using _filter_snapshot_chunk logic (usually removes edges if not strict)
    # Here we simulate the logic: we want features that Fall in valid range.
    # The builder generates timestamps for the END of the dt bin.
    # e.g. 08:00 to 08:05 -> ts_min represents 08:00 (start) or 08:05 (end)? 
    # Usually `ts_min` in flattened is integer minutes from epoch.
    
    # Let's inspect ONE timestamp to be sure.
    # If dt=5, features are emitted at t_end.
    # Flatten snapshot features usually produces 'ts_min' as integer minutes.
    
    df_b1 = _filter_snapshot_chunk(df_b1, start_time, mid_time)
    
    # --- Batch 2 ---
    # Range: 10:00 to 12:00
    # Overlap: 40 min -> Start input at 09:20
    print("Running Batch 2...")
    overlap_start = mid_time - pd.Timedelta(minutes=40)
    
    df_b2_input = full_df[
        (full_df["FECHA"] >= overlap_start) & 
        (full_df["FECHA"] < end_time)
    ].copy()
    
    snap_b2, _ = builder.build(pl.from_pandas(df_b2_input), df_porticos)
    df_b2 = flatten_snapshot_features(snap_b2, include_timestamp=True)
    
    # Filter Batch 2: Keep strict range [10:00, 12:00)
    df_b2 = _filter_snapshot_chunk(df_b2, mid_time, end_time)
    
    # 4. Concatenate
    df_batched = pd.concat([df_b1, df_b2], ignore_index=True)
    df_batched = df_batched.sort_values(["portico", "ts_min"]).reset_index(drop=True)
    
    # 5. Assertions
    # Check shapes
    print(f"Global Shape: {df_global.shape}")
    print(f"Batched Shape: {df_batched.shape}")
    
    # We might lose the very first few rows in Global if window is not full?
    # Or builder fills with NaN? (SnapshotBuilder usually handles start)
    # Actually, SnapshotFeatureBuilder usually drops invalid windows if not handled?
    # Let's align by index.
    
    common_idx = df_global.index.intersection(df_batched.index)
    assert len(common_idx) > 0, "No intersection in rows"
    
    # Outer join to see diffs
    merged = pd.merge(
        df_global, 
        df_batched, 
        on=["portico", "ts_min"], 
        suffixes=("_g", "_b"), 
        how="outer",
        indicator=True
    )
    
    # Check for missing rows
    missing_in_batch = merged[merged["_merge"] == "left_only"]
    missing_in_global = merged[merged["_merge"] == "right_only"]
    
    if not missing_in_batch.empty:
        print("Rows missing in Batch:")
        print(missing_in_batch[["portico", "ts_min"]].head(20))
        # Print min/max ts of missing
        print(f"Missing TS Range: {missing_in_batch['ts_min'].min()} - {missing_in_batch['ts_min'].max()}")
        
    if not missing_in_global.empty:
        print("Rows missing in Global:")
        print(missing_in_global[["portico", "ts_min"]].head(20))
        
    assert missing_in_batch.empty, "Batch mode missed rows found in global"
    assert missing_in_global.empty, "Batch mode generated extra rows not in global"
    
    # Check values
    # We check a specific numeric column that depends on history
    col_check = "speed_mean_w_mean" # Moving average of speed
    
    # If the column exists
    if f"{col_check}_g" in merged.columns:
        diff = np.abs(merged[f"{col_check}_g"] - merged[f"{col_check}_b"])
        max_diff = diff.max()
        print(f"Max difference in {col_check}: {max_diff}")
        
        # We expect exact matches or very close float
        assert max_diff < 1e-6, f"Significant difference in {col_check}"
    
    print("Test Passed: Batch consistency verified.")

if __name__ == "__main__":
    test_snapshot_batch_consistency()
