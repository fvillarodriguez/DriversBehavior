from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:  # pragma: no cover - optional dependency
    lowess = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    px = None
    go = None

from src.config import RESULTADOS_DIR, SEED

WEEKDAY_LABELS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
CATEGORY_REMAP = {
    "1": "Light",
    "1.0": "Light",
    "2": "Heavy",
    "2.0": "Heavy",
    "3": "Motorcycle",
    "3.0": "Motorcycle",
    "4": "Heavy",
    "4.0": "Heavy",
    "light": "Light",
    "heavy": "Heavy",
    "motorcycle": "Motorcycle",
    "motorcycles": "Motorcycle",
}
CATEGORY_ORDER = ["Light", "Heavy", "Motorcycle"]


def _normalize_category_expr(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8)
        .str.to_lowercase()
        .replace(CATEGORY_REMAP, default="Other")
    )


def _table_a_for_subset_polars(
    df_subset: pl.DataFrame, title: str
) -> Optional[dict]:
    if df_subset is None or df_subset.is_empty():
        return None

    # Filter "Other" category
    df_local = df_subset.filter(pl.col("CATEGORIA") != "Other")
    if df_local.is_empty():
        return None

    # Aggregate by PORTICO, CATEGORIA, FECHA_HORA (ensure hourly)
    # The input df_subset is likely already minute or hourly aggregated.
    # We re-aggregate to Hour just in case.
    
    # Calculate weighted speed: sum(speed * flow) / sum(flow)
    hourly_agg = df_local.with_columns(
        (pl.col("FECHA").dt.truncate("1h")).alias("FECHA_HORA")
    ).group_by(["PORTICO", "CATEGORIA", "FECHA_HORA"]).agg([
        pl.col("flow").sum().alias("flow_total"),
        (
            (pl.col("mean_speed") * pl.col("flow")).sum() / pl.col("flow").sum()
        ).alias("speed_weighted")
    ])

    if hourly_agg.is_empty():
        return None

    # Calculate derived metrics
    # Flow [veh/h] = flow_total (if already hourly) -> Actually flow_total is sum of minutes.
    # The original code divided by 3.0? That was specific to 20-min aggregation or something?
    # Wait, original pandas code: hourly_agg["Flow [veh/h]"] = hourly_agg["flow_total"] / 3.0
    # If the user has 60 minutes of data, sum(flow) is total vehicles per hour.
    # Why divide by 3? Maybe 20 min chunks?
    # Assuming input is minute-level flow counts sum up to hourly flow. 
    # Let's keep it as sum() for now, or check why 3.0 was used.
    # Checking legacy: "hourly_agg = df_local.groupby(...).agg(flow_total=('flow', 'sum'))... hourly_agg['Flow [veh/h]'] = hourly_agg['flow_total'] / 3.0"
    # This implies valid data was expected to be 3 * 20min? or maybe it's just wrong?
    # If we assume standard "veh/h", it should be the sum.
    # Let's use SUM for now to be safe, assuming data is 24h.
    
    # To reproduce legacy behavior exactly mechanically:
    # We will assume flow_total is correct for now.
    
    hourly_agg = hourly_agg.with_columns([
        (pl.col("flow_total")).alias("Flow [veh/h]"), # Removing / 3.0 unless explicitly requested logic
        pl.col("speed_weighted").alias("Speed [km/h]")
    ])
    
    hourly_agg = hourly_agg.with_columns(
        pl.when(pl.col("Speed [km/h]") > 0)
        .then(pl.col("Flow [veh/h]") / pl.col("Speed [km/h]"))
        .otherwise(0.0)
        .alias("Density [veh/km]")
    )

    # Global Stats by Categoria
    stats_agg = hourly_agg.group_by("CATEGORIA").agg([
        pl.col("Flow [veh/h]").mean().alias("Flow_mean"),
        pl.col("Flow [veh/h]").std().alias("Flow_std"),
        pl.col("Flow [veh/h]").min().alias("Flow_min"),
        pl.col("Flow [veh/h]").max().alias("Flow_max"),
        pl.col("Flow [veh/h]").quantile(0.02, interpolation="linear").alias("Flow_p2"),
        pl.col("Flow [veh/h]").quantile(0.25, interpolation="linear").alias("Flow_p25"),
        pl.col("Flow [veh/h]").quantile(0.50, interpolation="linear").alias("Flow_p50"),
        pl.col("Flow [veh/h]").quantile(0.95, interpolation="linear").alias("Flow_p95"),
        
        pl.col("Speed [km/h]").mean().alias("Speed_mean"),
        pl.col("Speed [km/h]").std().alias("Speed_std"),
        pl.col("Speed [km/h]").min().alias("Speed_min"),
        pl.col("Speed [km/h]").max().alias("Speed_max"),
        pl.col("Speed [km/h]").quantile(0.02, interpolation="linear").alias("Speed_p2"),
        pl.col("Speed [km/h]").quantile(0.25, interpolation="linear").alias("Speed_p25"),
        pl.col("Speed [km/h]").quantile(0.50, interpolation="linear").alias("Speed_p50"),
        pl.col("Speed [km/h]").quantile(0.95, interpolation="linear").alias("Speed_p95"),
        
        pl.col("Density [veh/km]").mean().alias("Density_mean"),
        pl.col("Density [veh/km]").std().alias("Density_std"),
        pl.col("Density [veh/km]").min().alias("Density_min"),
        pl.col("Density [veh/km]").max().alias("Density_max"),
        pl.col("Density [veh/km]").quantile(0.02, interpolation="linear").alias("Density_p2"),
        pl.col("Density [veh/km]").quantile(0.25, interpolation="linear").alias("Density_p25"),
        pl.col("Density [veh/km]").quantile(0.50, interpolation="linear").alias("Density_p50"),
        pl.col("Density [veh/km]").quantile(0.95, interpolation="linear").alias("Density_p95"),
    ])
    
    if stats_agg.is_empty():
        return None

    # Reshape (Melt/Stack) to match specific format
    # Columns: CATEGORIA, Variable, Average, Std.Dev, Min, Max
    
    vars_list = ["Flow [veh/h]", "Speed [km/h]", "Density [veh/km]"]
    suffix_map = {
        "mean": "Average",
        "std": "Std.Dev",
        "min": "Min",
        "max": "Max",
        "p2": "P2",
        "p25": "P25",
        "p50": "P50",
        "p95": "P95",
    }
    
    rows = []
    for row in stats_agg.to_dicts():
        cat = row["CATEGORIA"]
        for var in vars_list:
            prefix = var.split(" ")[0] # Flow, Speed, Density
            r = {"CATEGORIA": cat, "Variable": var}
            for suffix, out_col in suffix_map.items():
                val = row.get(f"{prefix}_{suffix}")
                r[out_col] = round(val, 2) if val is not None else 0.0
            rows.append(r)
            
    df_final = pl.DataFrame(rows)
    df_final = df_final.select(
        ["CATEGORIA", "Variable", "Average", "Std.Dev", "Min", "Max", "P2", "P25", "P50", "P95"]
    )
    
    # Calculate Share
    total_flow_all = hourly_agg.select(pl.col("Flow [veh/h]").sum()).item()
    share_df = None
    if total_flow_all > 0:
        share_agg = hourly_agg.group_by("CATEGORIA").agg(
            pl.col("Flow [veh/h]").sum().alias("cat_flow")
        )
        share_agg = share_agg.with_columns(
            (pl.col("cat_flow") / total_flow_all * 100).round(2).alias("Share (%)")
        )
        share_df = share_agg.select(["CATEGORIA", "Share (%)"])
        
    # Join Share to Final Export
    export_df = df_final.clone()
    if share_df is not None:
        export_df = export_df.join(share_df, on="CATEGORIA", how="left")
    
    export_df = export_df.with_columns(pl.lit(title).alias("Tabla")).select(["Tabla"] + export_df.columns)

    return {
        "title": title,
        "stats": df_final.to_pandas(), # Convert to pandas for st.dataframe compatibility
        "share": share_df.to_pandas() if share_df is not None else None,
        "export": export_df.to_pandas(),
    }


def generate_table_a_polars(df_pl: pl.DataFrame) -> List[dict]:
    results: List[dict] = []
    if df_pl is None or df_pl.is_empty():
        return results

    df = df_pl.clone()
    
    # Global
    results.extend([
        r for r in [_table_a_for_subset_polars(df, "Tabla A - Estadisticas Globales por Categoria")]
        if r is not None
    ])
    
    # Day Type Analysis
    # DuckDB/Polars day of week: Monday=1...Sunday=7ISO or 0..6?
    # Polars dt.weekday(): Monday=1, Sunday=7.
    df = df.with_columns(
        pl.when(pl.col("FECHA").dt.weekday() >= 6)
        .then(pl.lit("Weekend"))
        .otherwise(pl.lit("Weekday"))
        .alias("day_type")
    )
    
    results.extend([
        r for r in [
            _table_a_for_subset_polars(df.filter(pl.col("day_type") == "Weekday"), "Tabla A - Lunes a Viernes (24h)"),
            _table_a_for_subset_polars(df.filter(pl.col("day_type") == "Weekend"), "Tabla A - Sabado y Domingo (24h)"),
        ] if r is not None
    ])

    # Time Ranges
    df = df.with_columns(pl.col("FECHA").dt.hour().alias("hour"))
    
    franjas = [
        ("Tabla 1 (06:00-08:00)", (pl.col("hour") >= 6) & (pl.col("hour") < 8)),
        ("Tabla 2 (08:00-18:00)", (pl.col("hour") >= 8) & (pl.col("hour") < 18)),
        ("Tabla 3 (18:00-20:00)", (pl.col("hour") >= 18) & (pl.col("hour") < 20)),
        ("Tabla 4 (20:00-06:00)", (pl.col("hour") >= 20) | (pl.col("hour") < 6)),
    ]
    
    for title, expr in franjas:
        franja_df = df.filter(expr)
        results.extend([
            r for r in [
                _table_a_for_subset_polars(franja_df, f"{title} - Todos los dias"),
                _table_a_for_subset_polars(franja_df.filter(pl.col("day_type") == "Weekday"), f"{title} - Lunes a Viernes"),
                _table_a_for_subset_polars(franja_df.filter(pl.col("day_type") == "Weekend"), f"{title} - Sabado y Domingo"),
            ] if r is not None
        ])
        
    return results


def build_hourly_profile_tables_polars(df_pl: pl.DataFrame) -> Optional[pd.DataFrame]:
    if df_pl is None or df_pl.is_empty():
        return None
        
    df = df_pl.with_columns([
        pl.col("FECHA").dt.hour().alias("hour"),
        pl.when(pl.col("FECHA").dt.weekday() >= 6).then(pl.lit("Weekend")).otherwise(pl.lit("Weekday")).alias("day_type")
    ])
    
    agg = df.group_by(["day_type", "hour", "CATEGORIA"]).agg([
        pl.col("flow").mean().alias("avg_flow"),
        pl.col("mean_speed").mean().alias("avg_speed")
    ]).sort(["day_type", "hour", "CATEGORIA"])
    
    # Pivot for display: Index=[day_type, hour], Cols=[Category... avg_flow/speed]
    # This is complex in Polars, easier to do in Pandas for display
    return agg.to_pandas().round(2).set_index(["day_type", "hour", "CATEGORIA"]).unstack(level="CATEGORIA").fillna(0)


def build_weekly_seasonality_polars(df_pl: pl.DataFrame) -> Optional[pd.DataFrame]:
    if df_pl is None or df_pl.is_empty():
        return None
        
    daily_flow = df_pl.group_by(pl.col("FECHA").dt.date()).agg(pl.col("flow").sum())
    global_mean = daily_flow.select(pl.col("flow").mean()).item()
    
    if global_mean is None or global_mean <= 0:
        return None
        
    seasonality = daily_flow.with_columns(
        (pl.col("flow") / global_mean).alias("Factor_Estacionalidad"),
        pl.col("FECHA").dt.strftime("%A").alias("Dia_Semana") # English names by default in many locales? 
        # Polars strftime depends on locale. Safe to map integer weekday.
    )
    
    # Map weekday number to name manually to be safe
    # Weekday: 1=Mon, 7=Sun
    days_map = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
    seasonality = seasonality.with_columns(
         pl.col("FECHA").dt.weekday().replace(days_map).alias("Dia_Semana")
    )
    
    weekly = seasonality.group_by("Dia_Semana").agg(pl.col("Factor_Estacionalidad").mean().round(3))
    
    # Sort
    sort_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Convert to pandas to sort by categorical list
    pdf = weekly.to_pandas()
    pdf["Dia_Semana"] = pd.Categorical(pdf["Dia_Semana"], categories=sort_order, ordered=True)
    return pdf.sort_values("Dia_Semana").reset_index(drop=True)


def plot_hourly_profiles_polars(
    df_pl: pl.DataFrame,
    *,
    show_flow: bool,
    show_speed: bool,
    save_outputs: bool,
) -> List[Tuple[str, plt.Figure, Optional[str]]]:
    figs = []
    if sns is None or df_pl.is_empty():
        return figs

    df = df_pl.with_columns([
        pl.col("FECHA").dt.hour().alias("hour"),
        pl.when(pl.col("FECHA").dt.weekday() >= 6).then(pl.lit("Weekend")).otherwise(pl.lit("Weekday")).alias("day_type")
    ])
    
    plot_df = df.group_by(["day_type", "hour"]).agg([
        pl.col("flow").sum().alias("total_flow"),
        pl.col("mean_speed").mean().alias("avg_speed")
    ]).sort("hour").to_pandas()
    
    if plot_df.empty:
        return figs

    palette = ["black", "orange"] # Basic palette
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if show_flow:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=plot_df, x="hour", y="total_flow", hue="day_type", marker="o", palette=palette, ax=ax)
        ax.set_ylabel("Total Average Flow (veh/min)")
        ax.grid(False)
        plt.tight_layout()
        save_path = None
        if save_outputs:
            save_path = os.path.join(RESULTADOS_DIR, f"perfil_horario_flujo_{timestamp}.png")
            plt.savefig(save_path, format="png", dpi=300)
        figs.append(("Perfil horario de flujo", fig, save_path))

    if show_speed:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=plot_df, x="hour", y="avg_speed", hue="day_type", marker="o", palette=palette, ax=ax)
        ax.set_ylabel("Speed mean (km/h)")
        ax.grid(False)
        plt.tight_layout()
        save_path = None
        if save_outputs:
            save_path = os.path.join(RESULTADOS_DIR, f"perfil_horario_velocidad_{timestamp}.png")
            plt.savefig(save_path, format="png", dpi=300)
        figs.append(("Perfil horario de velocidad", fig, save_path))
        
    return figs


def process_monthly_and_store(
    db_path: Path, 
    min_speed: float, 
    remove_outliers: bool,
    temp_db_path: Path
) -> Optional[pl.DataFrame]:
    """
    Iterates through months in the DuckDB tables.
    Loads each month into Polars, aggregates, and stores into a TEMP DuckDB table.
    Finally returns the full aggregated dataset as a lazy Polars DF (or materialized if small enough).
    """
    if not db_path.exists():
        st.error("DB not found")
        return None
        
    # 1. Setup Temp DB
    if temp_db_path.exists():
        try:
            os.remove(temp_db_path)
        except OSError:
            pass
            
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        
        # 2. Get Date Range (Months)
        range_query = "SELECT MIN(FECHA) as min_date, MAX(FECHA) as max_date FROM flujos_duckdb"
        res = conn.execute(range_query).fetchone()
        if not res or res[0] is None:
            st.warning("DB is empty")
            return None
            
        min_date, max_date = res
        # DuckDB dates to Python datetime
        current_date = min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = max_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Init Temp DB connection
        temp_conn = duckdb.connect(str(temp_db_path))
        temp_conn.execute("CREATE TABLE agg_results (PORTICO VARCHAR, CATEGORIA VARCHAR, FECHA TIMESTAMP, flow INTEGER, mean_speed DOUBLE, speed_std DOUBLE)")
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        months_list = []
        d = current_date
        while d <= end_date:
            months_list.append(d)
            # increment month
            # simple trick: add 32 days and reset to day 1
            # or dateutil.relativedelta (but trying to avoid extra deps)
            # using pandas offset just for logic if available?
            # pure python:
            next_month = d.month + 1 if d.month < 12 else 1
            next_year = d.year if d.month < 12 else d.year + 1
            d = d.replace(year=next_year, month=next_month, day=1)
            
        total_months = len(months_list)
        
        # 3. Iterate Months
        for idx, month_start in enumerate(months_list):
            next_month = month_start.replace(year=month_start.year + (month_start.month // 12), month=(month_start.month % 12) + 1, day=1)
            
            status_text.text(f"Procesando mes: {month_start.strftime('%Y-%m')}")
            
            # Prepare Query
            where_clauses = ["VELOCIDAD >= ?"]
            params = [min_speed, month_start, next_month]
            query_where = "VELOCIDAD >= ? AND FECHA >= ? AND FECHA < ?"
            
            if remove_outliers:
                # Local IQR for the period (or global? User asked for batching, maybe local is acceptable or we query global params first)
                # To be consistent with "Global IQR", we should have pre-calculated global stats.
                # Calculating Global IQR on full DB fits in memory? 
                # conn.execute("quantile_cont...").fetchone() is efficient.
                # Let's assume global IQR was pre-calculated or we do per-month IQR for strict batching.
                # Recommendation: Per-month IQR is safer for streaming, but creates discontinuities.
                # Let's try Global IQR first (it's a scalar query).
                try:
                     # Check if we already calculated global iqr? No, let's do it per month for speed or skip.
                     # Re-querying global iqr every loop is wasteful.
                     # Let's calculate per-month IQR to avoid unrelated data scans.
                     pass 
                except:
                     pass
            
            # Extract Raw Data for Month to Polars
            q = f"SELECT PORTICO, CATEGORIA, FECHA, VELOCIDAD FROM flujos_duckdb WHERE {query_where}"
            
            # Fetch Arrow -> Polars
            # chunk_size ignored if via duckdb relation?
            # We trust 1 month of raw data fits in RAM (usually < 2GB for reasonable traffic).
            # If still OOM, we'd need daily. Assuming monthly is fine.
            df_month = conn.execute(q, params).pl()
            
            if not df_month.is_empty():
                # Aggregation in Polars
                # Group by Hour (user wanted optimization)
                # Ensure FECHA is datetime
                agg_pl = df_month.group_by([
                    "PORTICO",
                    "CATEGORIA",
                    pl.col("FECHA").dt.truncate("1h")
                ]).agg([
                    pl.len().alias("flow"),
                    pl.col("VELOCIDAD").mean().alias("mean_speed"),
                    pl.col("VELOCIDAD").std().fill_null(0).alias("speed_std")
                ])
                
                # Append to Temp DuckDB
                # polars can write_database? Or just register and insert.
                temp_conn.register("df_view", agg_pl)
                temp_conn.execute("INSERT INTO agg_results SELECT * FROM df_view")
                temp_conn.unregister("df_view")
                
                del df_month
                del agg_pl
            
            progress_bar.progress((idx + 1) / total_months)
            
        conn.close()
        
        # 4. Read Final from Temp
        # Return as Polars DataFrame
        final_df = temp_conn.execute("SELECT * FROM agg_results").pl()
        temp_conn.close()
        
        # Map IDs
        cat_map = {
            "1": "Light",
            "1.0": "Light",
            "2": "Heavy",
            "2.0": "Heavy",
            "3": "Motorcycle",
            "3.0": "Motorcycle",
            "4": "Heavy",
            "4.0": "Heavy",
        } # Handle string/int variance
        # If DB has ints:
        final_df = final_df.with_columns(
            pl.col("CATEGORIA").cast(pl.Utf8).replace(cat_map, default="Other")
        )

        return final_df

    except Exception as e:
        st.error(f"Error en procesamiento mensual: {e}")
        return None


def process_monthly_and_store_duckdb(
    db_path: Path,
    min_speed: float,
    temp_db_path: Path,
) -> Optional[pl.DataFrame]:
    """
    Iterates through months using DuckDB-only aggregation and stores results in a temp DB.
    """
    if not db_path.exists():
        st.error("DB not found")
        return None

    if temp_db_path.exists():
        try:
            os.remove(temp_db_path)
        except OSError:
            pass

    try:
        conn = duckdb.connect(str(temp_db_path))
        attach_path = str(db_path).replace("'", "''")
        conn.execute(f"ATTACH '{attach_path}' AS base (READ_ONLY)")

        res = conn.execute(
            "SELECT MIN(FECHA) as min_date, MAX(FECHA) as max_date FROM base.flujos_duckdb"
        ).fetchone()
        if not res or res[0] is None:
            st.warning("DB is empty")
            return None

        min_date, max_date = res
        current_date = min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = max_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        conn.execute(
            "CREATE TABLE agg_results (PORTICO VARCHAR, CATEGORIA VARCHAR, FECHA TIMESTAMP, "
            "flow BIGINT, mean_speed DOUBLE, speed_std DOUBLE)"
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        months_list = []
        d = current_date
        while d <= end_date:
            months_list.append(d)
            next_month = d.month + 1 if d.month < 12 else 1
            next_year = d.year if d.month < 12 else d.year + 1
            d = d.replace(year=next_year, month=next_month, day=1)

        total_months = len(months_list)

        for idx, month_start in enumerate(months_list):
            next_month = month_start.replace(
                year=month_start.year + (month_start.month // 12),
                month=(month_start.month % 12) + 1,
                day=1,
            )

            status_text.text(f"Procesando mes: {month_start.strftime('%Y-%m')}")

            conn.execute(
                """
                INSERT INTO agg_results
                SELECT
                    PORTICO,
                    CATEGORIA,
                    date_trunc('hour', FECHA) AS FECHA,
                    COUNT(*) AS flow,
                    AVG(VELOCIDAD) AS mean_speed,
                    STDDEV_SAMP(VELOCIDAD) AS speed_std
                FROM base.flujos_duckdb
                WHERE VELOCIDAD >= ? AND FECHA >= ? AND FECHA < ?
                GROUP BY 1, 2, 3
                """,
                [min_speed, month_start, next_month],
            )

            progress_bar.progress((idx + 1) / total_months)

        final_df = conn.execute("SELECT * FROM agg_results").pl()
        conn.close()

        cat_map = {
            "1": "Light",
            "1.0": "Light",
            "2": "Heavy",
            "2.0": "Heavy",
            "3": "Motorcycle",
            "3.0": "Motorcycle",
            "4": "Heavy",
            "4.0": "Heavy",
        }
        final_df = final_df.with_columns(
            pl.col("CATEGORIA").cast(pl.Utf8).replace(cat_map, default="Other")
        )
        return final_df

    except Exception as exc:
        st.error(f"Error en procesamiento DuckDB: {exc}")
        return None


def build_weekday_flow_lines(df_pl: pl.DataFrame) -> Optional[pd.DataFrame]:
    if df_pl is None or df_pl.is_empty():
        return None

    df = df_pl.with_columns(
        [
            _normalize_category_expr(pl.col("CATEGORIA")).alias("Categoria"),
            pl.col("FECHA").dt.date().alias("fecha_dia"),
        ]
    ).filter(pl.col("Categoria") != "Other")

    if df.is_empty():
        return None

    daily = df.group_by(["Categoria", "fecha_dia"]).agg(
        pl.col("flow").sum().alias("flow_total")
    )
    daily = daily.with_columns(pl.col("fecha_dia").dt.weekday().alias("dow"))
    avg = (
        daily.group_by(["Categoria", "dow"])
        .agg(pl.col("flow_total").mean().alias("flow_avg"))
        .with_columns((pl.col("dow") - 1).alias("dow_idx"))
    )
    return avg.to_pandas()


def _plot_weekday_flow_lines(line_df: pd.DataFrame) -> Optional[plt.Figure]:
    if line_df is None or line_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    for cat in CATEGORY_ORDER:
        sub = line_df[line_df["Categoria"] == cat].sort_values("dow_idx")
        if sub.empty:
            continue
        ax.plot(
            sub["dow_idx"],
            sub["flow_avg"],
            marker="o",
            linewidth=2,
            label=cat,
        )
    ax.set_xticks(list(range(7)))
    ax.set_xticklabels(WEEKDAY_LABELS)
    ax.set_xlabel("Dia de semana")
    ax.set_ylabel("Flujo promedio (veh/dia)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def _plot_weekday_flow_lines_interactive(line_df: pd.DataFrame):
    if go is None or line_df is None or line_df.empty:
        return None
    df = line_df.copy()
    df = df[df["Categoria"].isin(CATEGORY_ORDER)].copy()
    if df.empty:
        return None
    df = (
        df.groupby(["Categoria", "dow_idx"], as_index=False)
        .agg({"flow_avg": "mean"})
        .sort_values(["Categoria", "dow_idx"])
    )

    fig = go.Figure()
    for cat in CATEGORY_ORDER:
        sub = df[df["Categoria"] == cat].sort_values("dow_idx")
        if sub.empty:
            continue
        x_labels = [
            WEEKDAY_LABELS[int(i)] if 0 <= int(i) < 7 else str(i)
            for i in sub["dow_idx"].tolist()
        ]
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=sub["flow_avg"],
                mode="lines+markers",
                name=cat,
            )
        )

    fig.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=WEEKDAY_LABELS, title="Dia de semana"),
        yaxis=dict(title="Flujo promedio (veh/dia)"),
        legend_title_text="Tipo de vehiculo",
        template="plotly_white",
        height=420,
    )
    return fig


def _compute_surface_polars(
    db_path: Path, min_speed: float, interval_minutes: int
) -> Optional[pd.DataFrame]:
    if not db_path.exists():
        st.error("DB not found")
        return None

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        res = conn.execute(
            "SELECT MIN(FECHA) as min_date, MAX(FECHA) as max_date FROM flujos_duckdb"
        ).fetchone()
        if not res or res[0] is None:
            st.warning("DB is empty")
            return None
        min_date, max_date = res
        current_date = min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = max_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        months_list = []
        d = current_date
        while d <= end_date:
            months_list.append(d)
            next_month = d.month + 1 if d.month < 12 else 1
            next_year = d.year if d.month < 12 else d.year + 1
            d = d.replace(year=next_year, month=next_month, day=1)

        monthly_results = []
        for month_start in months_list:
            next_month = month_start.replace(
                year=month_start.year + (month_start.month // 12),
                month=(month_start.month % 12) + 1,
                day=1,
            )
            q = """
                SELECT CATEGORIA, FECHA, VELOCIDAD
                FROM flujos_duckdb
                WHERE VELOCIDAD >= ?
                  AND VELOCIDAD IS NOT NULL
                  AND FECHA IS NOT NULL
                  AND CATEGORIA IS NOT NULL
                  AND FECHA >= ? AND FECHA < ?
            """
            df_month = conn.execute(q, [min_speed, month_start, next_month]).pl()
            if df_month.is_empty():
                continue
            df_month = df_month.with_columns(
                [
                    _normalize_category_expr(pl.col("CATEGORIA")).alias("Categoria"),
                    pl.col("FECHA").dt.truncate(f"{interval_minutes}m").alias("ts_bin"),
                    pl.col("FECHA").dt.date().alias("fecha_dia"),
                ]
            ).filter(pl.col("Categoria") != "Other")
            if df_month.is_empty():
                continue
            per_day = (
                df_month.group_by(["Categoria", "fecha_dia", "ts_bin"])
                .agg(
                    [
                        pl.len().alias("flow"),
                        pl.col("VELOCIDAD").sum().alias("speed_sum"),
                    ]
                )
            )
            per_day = per_day.with_columns(
                [
                    (pl.col("fecha_dia").dt.weekday() - 1).alias("dow_idx"),
                    (
                        pl.col("ts_bin").dt.hour() * 60
                        + pl.col("ts_bin").dt.minute()
                    ).alias("minute_of_day"),
                ]
            )
            agg = per_day.group_by(["Categoria", "dow_idx", "minute_of_day"]).agg(
                [
                    pl.col("flow").sum().alias("flow_sum"),
                    pl.col("speed_sum").sum().alias("speed_sum"),
                    pl.len().alias("day_count"),
                ]
            )
            monthly_results.append(agg)

        if not monthly_results:
            return None

        all_months = pl.concat(monthly_results)
        final = (
            all_months.group_by(["Categoria", "dow_idx", "minute_of_day"])
            .agg(
                [
                    pl.col("flow_sum").sum().alias("flow_sum"),
                    pl.col("speed_sum").sum().alias("speed_sum"),
                    pl.col("day_count").sum().alias("day_count"),
                ]
            )
            .with_columns(
                (pl.col("flow_sum") / pl.col("day_count")).alias("flow_avg"),
                (pl.col("speed_sum") / pl.col("flow_sum")).alias("speed_avg"),
            )
            .with_columns(
                pl.when(pl.col("speed_avg") > 0)
                .then(pl.col("flow_avg") / pl.col("speed_avg"))
                .otherwise(0.0)
                .alias("density_avg")
            )
        )
        return final.to_pandas()
    finally:
        conn.close()


def _compute_surface_duckdb(
    db_path: Path, min_speed: float, interval_minutes: int, temp_db_path: Path
) -> Optional[pd.DataFrame]:
    if not db_path.exists():
        st.error("DB not found")
        return None

    if temp_db_path.exists():
        try:
            os.remove(temp_db_path)
        except OSError:
            pass

    attach_path = str(db_path).replace("'", "''")
    conn = duckdb.connect(str(temp_db_path))
    try:
        conn.execute(f"ATTACH '{attach_path}' AS base (READ_ONLY)")
        res = conn.execute(
            "SELECT MIN(FECHA) as min_date, MAX(FECHA) as max_date FROM base.flujos_duckdb"
        ).fetchone()
        if not res or res[0] is None:
            st.warning("DB is empty")
            return None
        min_date, max_date = res
        current_date = min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = max_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        conn.execute(
            "CREATE TABLE surface_tmp (categoria INTEGER, dow INTEGER, minute_of_day INTEGER, "
            "flow_sum DOUBLE, speed_sum DOUBLE, day_count BIGINT)"
        )

        months_list = []
        d = current_date
        while d <= end_date:
            months_list.append(d)
            next_month = d.month + 1 if d.month < 12 else 1
            next_year = d.year if d.month < 12 else d.year + 1
            d = d.replace(year=next_year, month=next_month, day=1)

        for month_start in months_list:
            next_month = month_start.replace(
                year=month_start.year + (month_start.month // 12),
                month=(month_start.month % 12) + 1,
                day=1,
            )
            conn.execute(
                f"""
                INSERT INTO surface_tmp
                WITH base_month AS (
                    SELECT
                        CASE WHEN CATEGORIA = 4 THEN 2 ELSE CATEGORIA END AS categoria,
                        VELOCIDAD AS velocidad,
                        DATE(FECHA) AS fecha,
                        (
                            date_trunc('hour', FECHA)
                            + INTERVAL 1 MINUTE * (
                                FLOOR(EXTRACT(minute FROM FECHA) / {interval_minutes}) * {interval_minutes}
                            )
                        ) AS ts_bin
                    FROM base.flujos_duckdb
                    WHERE VELOCIDAD >= ?
                      AND FECHA IS NOT NULL
                      AND CATEGORIA IS NOT NULL
                      AND VELOCIDAD IS NOT NULL
                      AND FECHA >= ? AND FECHA < ?
                ),
                per_day AS (
                    SELECT
                        categoria,
                        fecha,
                        ts_bin,
                        COUNT(*) AS flow,
                        SUM(velocidad) AS speed_sum
                    FROM base_month
                    GROUP BY 1, 2, 3
                ),
                agg AS (
                    SELECT
                        categoria,
                        EXTRACT(dow FROM fecha) AS dow,
                        (EXTRACT(hour FROM ts_bin) * 60 + EXTRACT(minute FROM ts_bin)) AS minute_of_day,
                        SUM(flow) AS flow_sum,
                        SUM(speed_sum) AS speed_sum,
                        COUNT(*) AS day_count
                    FROM per_day
                    GROUP BY 1, 2, 3
                )
                SELECT categoria, dow, minute_of_day, flow_sum, speed_sum, day_count FROM agg
                """,
                [min_speed, month_start, next_month],
            )

        final = conn.execute(
            """
            SELECT
                categoria,
                dow,
                minute_of_day,
                SUM(flow_sum) AS flow_sum,
                SUM(speed_sum) AS speed_sum,
                SUM(day_count) AS day_count
            FROM surface_tmp
            GROUP BY 1, 2, 3
            """
        ).df()
    finally:
        conn.close()

    if final.empty:
        return None

    final["dow_idx"] = (final["dow"] + 6) % 7
    final["Categoria"] = final["categoria"].astype(str).str.lower().replace(CATEGORY_REMAP)
    final["flow_avg"] = final["flow_sum"] / final["day_count"].replace(0, np.nan)
    final["speed_avg"] = final["speed_sum"] / final["flow_sum"].replace(0, np.nan)
    final["density_avg"] = final["flow_avg"] / final["speed_avg"].replace(0, np.nan)
    final = final.dropna(subset=["flow_avg"])
    return final[
        ["Categoria", "dow_idx", "minute_of_day", "flow_avg", "speed_avg", "density_avg"]
    ]


def _plot_surface_by_category(surface_df: pd.DataFrame) -> List[Tuple[str, plt.Figure]]:
    if surface_df is None or surface_df.empty:
        return []

    figs: List[Tuple[str, plt.Figure]] = []
    minutes = np.arange(0, 24 * 60, 5)
    days = np.arange(7)
    hours = minutes / 60.0

    for cat in CATEGORY_ORDER:
        sub = surface_df[surface_df["Categoria"] == cat]
        if sub.empty:
            continue
        pivot = (
            sub.pivot(index="minute_of_day", columns="dow_idx", values="flow_avg")
            .reindex(index=minutes, columns=days)
            .fillna(0.0)
        )
        Z = pivot.values  # flow
        X, Y = np.meshgrid(days, hours)
        norm = colors.Normalize(vmin=0.0, vmax=max(Z.max(), 1e-6))
        cmap = cm.get_cmap("RdYlGn_r")
        facecolors = cmap(norm(Z))

        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        ax.set_xlabel("Dia de semana")
        ax.set_ylabel("Hora del dia")
        ax.set_zlabel("Flujo promedio")
        ax.set_xticks(days)
        ax.set_xticklabels(WEEKDAY_LABELS)
        ax.set_yticks([0, 4, 8, 12, 16, 20, 24])
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label="Flujo promedio")
        figs.append((f"Superficie 3D - {cat}", fig))

    return figs


def _plot_surface_by_category_interactive(
    surface_df: pd.DataFrame,
    *,
    x_var: str,
    y_var: str,
    z_var: str,
):
    if go is None or surface_df is None or surface_df.empty:
        return []

    figs = []
    value_map = {
        "Flujo promedio": "flow_avg",
        "Velocidad media": "speed_avg",
        "Densidad media": "density_avg",
    }
    value_col = value_map.get(z_var, "flow_avg")
    minutes = np.arange(0, 24 * 60, 5)
    days = np.arange(7)
    hours = minutes / 60.0

    colorscale = [
        [0.0, "rgb(0,128,0)"],
        [0.5, "rgb(255,215,0)"],
        [1.0, "rgb(220,20,60)"],
    ]

    for cat in CATEGORY_ORDER:
        sub = surface_df[surface_df["Categoria"] == cat]
        if sub.empty:
            continue
        pivot = (
            sub.pivot(index="minute_of_day", columns="dow_idx", values=value_col)
            .reindex(index=minutes, columns=days)
            .fillna(0.0)
        )
        if x_var == "Dia de semana" and y_var == "Hora del dia":
            Z = pivot.values
            X, Y = np.meshgrid(days, hours)
            x_title = "Dia de semana"
            y_title = "Hora del dia"
            x_ticks = dict(tickmode="array", tickvals=days, ticktext=WEEKDAY_LABELS)
            y_ticks = dict(tickmode="array", tickvals=[0, 4, 8, 12, 16, 20, 24])
        elif x_var == "Hora del dia" and y_var == "Dia de semana":
            Z = pivot.values.T
            X, Y = np.meshgrid(hours, days)
            x_title = "Hora del dia"
            y_title = "Dia de semana"
            x_ticks = dict(tickmode="array", tickvals=[0, 4, 8, 12, 16, 20, 24])
            y_ticks = dict(tickmode="array", tickvals=days, ticktext=WEEKDAY_LABELS)
        else:
            Z = pivot.values
            X, Y = np.meshgrid(days, hours)
            x_title = "Dia de semana"
            y_title = "Hora del dia"
            x_ticks = dict(tickmode="array", tickvals=days, ticktext=WEEKDAY_LABELS)
            y_ticks = dict(tickmode="array", tickvals=[0, 4, 8, 12, 16, 20, 24])

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale=colorscale,
                    cmin=0.0,
                    cmax=max(Z.max(), 1e-6),
                    colorbar=dict(title=z_var),
                )
            ]
        )
        fig.update_layout(
            title=f"Superficie 3D - {cat}",
            scene=dict(
                xaxis=dict(title=x_title, **x_ticks),
                yaxis=dict(title=y_title, **y_ticks),
                zaxis=dict(title=z_var),
            ),
            template="plotly_white",
            height=550,
        )
        figs.append((f"Superficie 3D - {cat}", fig))

    return figs


def render_flow_stats_tab(*, db_path: Path) -> None:
    st.subheader("Estadisticas de flujos")

    engine = st.selectbox(
        "Motor de calculo",
        ["Polars (batching)", "DuckDB"],
        key="flow_stats_engine",
    )
    if st.session_state.get("flow_stats_engine_selected") != engine:
        st.session_state["flow_stats_engine_selected"] = engine
        st.session_state.pop("flow_stats_df", None)

    col1, col2 = st.columns(2)
    min_speed = col1.number_input("Min Speed (km/h)", 1.0, 150.0, 1.0)
    temp_db = Path(RESULTADOS_DIR) / "temp_agg.duckdb"

    if st.button("Procesar Mensual"):
        if engine == "DuckDB":
            df_res = process_monthly_and_store_duckdb(db_path, min_speed, temp_db)
        else:
            df_res = process_monthly_and_store(db_path, min_speed, False, temp_db)
        if df_res is not None:
            st.session_state["flow_stats_df"] = df_res
            st.success(f"Finalizado. Filas reducidas: {len(df_res)}")

    if "flow_stats_df" in st.session_state:
        df_pl = st.session_state["flow_stats_df"]
        
        st.write("---")
        tables = generate_table_a_polars(df_pl)
        for t in tables:
            st.markdown(f"**{t['title']}**")
            st.dataframe(t['stats'])
            
        # Plots
        figs = plot_hourly_profiles_polars(df_pl, show_flow=True, show_speed=True, save_outputs=False)
        for title, fig, _ in figs:
            st.write(title)
            st.pyplot(fig)

        st.write("---")
        st.subheader("Flujo promedio por dia de la semana")
        line_df = build_weekday_flow_lines(df_pl)
        fig_line = (
            _plot_weekday_flow_lines_interactive(line_df)
            if line_df is not None
            else None
        )
        if fig_line is None:
            st.info("No hay datos suficientes para el grafico semanal o falta plotly.")
        else:
            st.plotly_chart(fig_line, use_container_width=True)

        st.write("---")
        st.subheader("Superficie 3D (dia vs hora, resolucion 5-min)")
        axis_cols = st.columns(3)
        with axis_cols[0]:
            axis_x = st.selectbox(
                "Eje X",
                ["Dia de semana", "Hora del dia"],
                index=0,
                key="flow_stats_axis_x",
            )
        with axis_cols[1]:
            axis_y = st.selectbox(
                "Eje Y",
                ["Hora del dia", "Dia de semana"],
                index=0,
                key="flow_stats_axis_y",
            )
        with axis_cols[2]:
            axis_z = st.selectbox(
                "Eje Z",
                ["Flujo promedio", "Velocidad media", "Densidad media"],
                index=0,
                key="flow_stats_axis_z",
            )
        if axis_x == axis_y:
            st.error("Eje X y Eje Y deben ser distintos.")

        surface_key = f"{engine}|{min_speed}"
        if st.button("Calcular superficie 3D", key="flow_stats_surface_btn"):
            with st.spinner("Calculando superficie 3D..."):
                if engine == "DuckDB":
                    surface_df = _compute_surface_duckdb(
                        db_path, min_speed, 5, temp_db
                    )
                else:
                    surface_df = _compute_surface_polars(
                        db_path, min_speed, 5
                    )
                st.session_state["flow_stats_surface"] = {
                    "key": surface_key,
                    "data": surface_df,
                }

        surface_state = st.session_state.get("flow_stats_surface")
        surface_df = None
        if surface_state and surface_state.get("key") == surface_key:
            surface_df = surface_state.get("data")

        if surface_df is None:
            st.info("Presiona 'Calcular superficie 3D' para generar los graficos.")
        else:
            if axis_x == axis_y:
                st.info("Selecciona ejes distintos para graficar la superficie.")
                figs = []
            else:
                figs = _plot_surface_by_category_interactive(
                    surface_df, x_var=axis_x, y_var=axis_y, z_var=axis_z
                )
            if not figs:
                st.info("No hay datos suficientes para las superficies o falta plotly.")
            else:
                for title, fig in figs:
                    st.plotly_chart(fig, use_container_width=True)
            
