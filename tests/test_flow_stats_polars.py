import duckdb
import pandas as pd
import polars as pl
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from src.flow_stats_legacy import (
    process_monthly_and_store,
    generate_table_a_polars,
    build_hourly_profile_tables_polars
)

@pytest.fixture
def mock_flow_db(tmp_path):
    """
    Creates a temporary DuckDB with 2 months of synthetic flow data.
    """
    db_path = tmp_path / "test_flujos.duckdb"
    conn = duckdb.connect(str(db_path))
    
    conn.execute("""
        CREATE TABLE flujos_duckdb (
            FECHA TIMESTAMP,
            VELOCIDAD DOUBLE,
            CATEGORIA INTEGER,
            PORTICO VARCHAR
        )
    """)
    
    # Generate data
    # Month 1: 2024-01-01 ...
    # Month 2: 2024-02-01 ...
    
    dates = []
    base_date = datetime(2024, 1, 1, 10, 0)
    for i in range(100):
        dates.append(base_date + timedelta(minutes=i))
        
    base_date_2 = datetime(2024, 2, 1, 10, 0)
    for i in range(100):
        dates.append(base_date_2 + timedelta(minutes=i))
        
    # Insert data
    # Portico A, Cat 1, Speed ~80
    for d in dates:
        conn.execute("INSERT INTO flujos_duckdb VALUES (?, ?, ?, ?)", [d, 80.0, 1, "P1"])
        
    # Add some outliers
    conn.execute("INSERT INTO flujos_duckdb VALUES (?, ?, ?, ?)", [datetime(2024, 1, 15, 12, 0), 200.0, 1, "P1"])
    
    conn.close()
    return db_path

def test_process_monthly_and_store(mock_flow_db, tmp_path):
    temp_res = tmp_path / "temp_agg.duckdb"
    
    # Run processing
    # Min speed 10, remove outliers False (to keep simple)
    df_res = process_monthly_and_store(
        db_path=mock_flow_db,
        min_speed=10.0,
        remove_outliers=False,
        temp_db_path=temp_res
    )
    
    assert df_res is not None
    assert not df_res.is_empty()
    assert "flow" in df_res.columns
    assert "mean_speed" in df_res.columns
    
    # Check if we have data from both months
    dates = df_res["FECHA"].dt.month().unique().to_list()
    assert 1 in dates
    assert 2 in dates
    
    # Check aggregation (we inserted 100 rows per month per portico)
    # The result is hourly aggregated.
    # 100 minutes spans roughly 2 hours (10:00 to 11:39)
    # So we expect 2 or 3 rows per month per portico-category
    assert len(df_res) > 0
    
    # Check temp file exists
    assert temp_res.exists()

def test_generate_table_a_polars():
    # Create synthetic aggregated data
    df = pl.DataFrame({
        "PORTICO": ["P1"] * 24,
        "CATEGORIA": ["Light"] * 24,
        "FECHA": [datetime(2024, 1, 1, h, 0) for h in range(24)],
        "flow": [100] * 24,
        "mean_speed": [80.0] * 24,
        "speed_std": [5.0] * 24
    })
    
    tables = generate_table_a_polars(df)
    
    assert len(tables) > 0
    t1 = tables[0]
    assert t1["title"] == "Tabla A - Estadisticas Globales por Categoria"
    assert not t1["stats"].empty
    
    # Check stats content
    stats = t1["stats"]
    # Should have Average Flow = 100 (if sum logic is used per hour)
    # Our logic: sum of flow in hour. Input is already hourly.
    row = stats[(stats["CATEGORIA"] == "Light") & (stats["Variable"] == "Flow [veh/h]")]
    assert not row.empty
    # We put 100 flow per hour. The avg should be 100.
    assert row["Average"].iloc[0] == 100.0

def test_build_hourly_profile_tables_polars():
    df = pl.DataFrame({
        "PORTICO": ["P1"] * 2,
        "CATEGORIA": ["Light"] * 2,
        "FECHA": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0)], # Monday
        "flow": [100, 200],
        "mean_speed": [80.0, 90.0]
    })
    
    res = build_hourly_profile_tables_polars(df)
    assert res is not None
    # Index should be day_type, hour
    # Check values
    assert res.loc[("Weekday", 10), ("avg_flow", "Light")] == 100
    assert res.loc[("Weekday", 11), ("avg_flow", "Light")] == 200
