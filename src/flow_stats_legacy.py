from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
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

import matplotlib.pyplot as plt

from src.config import RESULTADOS_DIR, SEED


def _count_lines(path: Path) -> int:
    with open(path, "rb") as handle:
        buf_gen = iter(lambda: handle.read(1024 * 1024), b"")
        return sum(buf.count(b"\n") for buf in buf_gen)


def _normalize_column_key(value: object) -> str:
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    normalized = {_normalize_column_key(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_column_key(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = _normalize_column_key(candidate)
        for norm, original in normalized.items():
            if key and key in norm:
                return original
    return None


def compute_minute_aggregates(df: pd.DataFrame, freq: str = "min") -> pd.DataFrame:
    df = df.infer_objects(copy=False)
    df_indexed = df.set_index("FECHA")
    agg = (
        df_indexed.groupby(
            ["PORTICO", pd.Grouper(freq=freq), "CATEGORIA"]
        )["VELOCIDAD"]
        .agg(["count", "mean", "std"])
        .rename(
            columns={
                "count": "flow",
                "mean": "mean_speed",
                "std": "speed_std",
            }
        )
    )
    agg["speed_std"] = agg["speed_std"].fillna(0)
    return agg.reset_index()


def _process_flow_chunk(
    chunk: pd.DataFrame,
    *,
    remove_outliers: bool,
    min_speed: float,
) -> Optional[pd.DataFrame]:
    if chunk is None or chunk.empty:
        return None
    chunk = chunk.copy()
    chunk.columns = [str(c).strip().upper() for c in chunk.columns]

    fecha_col = _find_column(chunk, ["FECHA", "FECHAHORA", "DATE"])
    velo_col = _find_column(chunk, ["VELOCIDAD", "SPEED"])
    cat_col = _find_column(chunk, ["CATEGORIA", "CATEGORY"])
    portico_col = _find_column(chunk, ["PORTICO", "GANTRY", "ID_PORTICO"])

    if not all([fecha_col, velo_col, cat_col, portico_col]):
        return None

    chunk.loc[:, velo_col] = pd.to_numeric(chunk[velo_col], errors="coerce")
    if remove_outliers and pd.api.types.is_numeric_dtype(chunk[velo_col]):
        q1 = chunk[velo_col].quantile(0.25)
        q3 = chunk[velo_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = float(min_speed)
        upper_bound = q3 + 1.5 * iqr if pd.notnull(iqr) else None
        if upper_bound is not None and pd.notnull(upper_bound):
            chunk = chunk[
                (chunk[velo_col] >= lower_bound)
                & (chunk[velo_col] <= upper_bound)
            ].copy()

    chunk.loc[:, fecha_col] = pd.to_datetime(
        chunk[fecha_col], errors="coerce", dayfirst=True
    )
    chunk = chunk.dropna(subset=[fecha_col, velo_col])

    cat_map = {"1": "Light", "2": "Heavy", "3": "Motorcycle"}
    chunk[cat_col] = chunk[cat_col].astype(str).map(cat_map).fillna("Other")

    df_proc = chunk[[fecha_col, portico_col, cat_col, velo_col]].rename(
        columns={
            fecha_col: "FECHA",
            portico_col: "PORTICO",
            cat_col: "CATEGORIA",
            velo_col: "VELOCIDAD",
        }
    )
    return df_proc


def build_minute_agg_from_csvs(
    files: List[Path],
    *,
    chunksize: int,
    remove_outliers: bool,
    min_speed: float,
) -> pd.DataFrame:
    all_minute_agg: List[pd.DataFrame] = []
    progress = st.progress(0.0)
    status = st.empty()

    total_files = len(files)
    for idx, file_path in enumerate(files):
        status.text(f"Procesando {file_path.name}...")
        num_lines = _count_lines(file_path)
        total_chunks = max(1, math.ceil(num_lines / chunksize)) if num_lines else 1
        chunk_iter = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
        for chunk_idx, chunk in enumerate(chunk_iter, 1):
            df_proc = _process_flow_chunk(
                chunk, remove_outliers=remove_outliers, min_speed=min_speed
            )
            if df_proc is None or df_proc.empty:
                continue
            minute_agg_chunk = compute_minute_aggregates(df_proc, freq="min")
            minute_agg_chunk["flow"] = minute_agg_chunk["flow"].astype(
                "int32", copy=False
            )
            minute_agg_chunk["mean_speed"] = minute_agg_chunk[
                "mean_speed"
            ].astype("float32", copy=False)
            minute_agg_chunk["speed_std"] = minute_agg_chunk[
                "speed_std"
            ].astype("float32", copy=False)
            all_minute_agg.append(minute_agg_chunk)
            step_ratio = (idx + (chunk_idx / total_chunks)) / total_files
            progress.progress(min(step_ratio, 1.0))

    progress.progress(1.0)
    status.empty()
    if not all_minute_agg:
        return pd.DataFrame()
    return pd.concat(all_minute_agg, ignore_index=True)


def _table_a_for_subset(
    df_subset: pd.DataFrame, title: str
) -> Optional[dict]:
    if df_subset is None or df_subset.empty:
        return None

    df_local = df_subset[df_subset["CATEGORIA"] != "Other"].copy()
    if df_local.empty:
        return None

    df_local["FECHA_HORA"] = df_local["FECHA"].dt.floor("h")

    def _weighted_speed(series: pd.Series) -> float:
        weights = df_local.loc[series.index, "flow"]
        total_weight = weights.sum()
        return np.average(series, weights=weights) if total_weight > 0 else np.nan

    hourly_agg = (
        df_local.groupby(["PORTICO", "CATEGORIA", "FECHA_HORA"])
        .agg(
            flow_total=("flow", "sum"),
            speed_weighted=("mean_speed", _weighted_speed),
        )
        .reset_index()
    )
    if hourly_agg.empty:
        return None

    hourly_agg["Flow [veh/h]"] = hourly_agg["flow_total"] / 3.0
    hourly_agg["Speed [km/h]"] = hourly_agg["speed_weighted"]
    hourly_agg["Density [veh/km]"] = hourly_agg.apply(
        lambda row: row["Flow [veh/h]"] / row["Speed [km/h]"]
        if row["Speed [km/h]"] > 0
        else 0,
        axis=1,
    )

    stats_agg = hourly_agg.groupby("CATEGORIA").agg(
        {
            "Flow [veh/h]": ["mean", "std", "min", "max"],
            "Speed [km/h]": ["mean", "std", "min", "max"],
            "Density [veh/km]": ["mean", "std", "min", "max"],
        }
    )
    if stats_agg.empty:
        return None

    stats_agg.columns.names = ["Variable", "Estadistica"]
    df_final = stats_agg.stack(level="Variable", future_stack=True)
    df_final = df_final.rename(
        columns={"mean": "Average", "std": "Std.Dev", "min": "Min", "max": "Max"}
    ).round(2)
    df_final = df_final[["Average", "Std.Dev", "Min", "Max"]]

    share_df = None
    category_total_flow = hourly_agg.groupby("CATEGORIA")["Flow [veh/h]"].sum()
    total_flow = category_total_flow.sum()
    if total_flow > 0:
        share = (category_total_flow / total_flow * 100).round(2)
        share_df = pd.DataFrame(share).rename(columns={"Flow [veh/h]": "Share (%)"})

    export_df = df_final.reset_index().rename(
        columns={"level_0": "CATEGORIA", "level_1": "Variable"}
    )
    if share_df is not None:
        export_df["Share (%)"] = export_df["CATEGORIA"].map(share_df["Share (%)"])
    export_df.insert(0, "Tabla", title)

    return {
        "title": title,
        "stats": df_final,
        "share": share_df,
        "export": export_df,
    }


def generate_table_a(minute_agg: pd.DataFrame) -> List[dict]:
    results: List[dict] = []
    if minute_agg is None or minute_agg.empty:
        return results

    df = minute_agg.copy()
    results.extend(
        [
            r
            for r in [
                _table_a_for_subset(
                    df, "Tabla A - Estadisticas Globales por Categoria"
                )
            ]
            if r is not None
        ]
    )

    df["hour"] = df["FECHA"].dt.hour
    df["day_type"] = df["FECHA"].dt.dayofweek.apply(
        lambda x: "Weekday" if x < 5 else "Weekend"
    )
    results.extend(
        [
            r
            for r in [
                _table_a_for_subset(
                    df[df["day_type"] == "Weekday"],
                    "Tabla A - Lunes a Viernes (24h)",
                ),
                _table_a_for_subset(
                    df[df["day_type"] == "Weekend"],
                    "Tabla A - Sabado y Domingo (24h)",
                ),
            ]
            if r is not None
        ]
    )

    franjas = [
        ("Tabla 1 (06:00-08:00)", (df["hour"] >= 6) & (df["hour"] < 8)),
        ("Tabla 2 (08:00-18:00)", (df["hour"] >= 8) & (df["hour"] < 18)),
        ("Tabla 3 (18:00-20:00)", (df["hour"] >= 18) & (df["hour"] < 20)),
        ("Tabla 4 (20:00-06:00)", (df["hour"] >= 20) | (df["hour"] < 6)),
    ]
    for title, mask in franjas:
        franja_df = df[mask]
        results.extend(
            [
                r
                for r in [
                    _table_a_for_subset(
                        franja_df, f"{title} - Todos los dias"
                    ),
                    _table_a_for_subset(
                        franja_df[franja_df["day_type"] == "Weekday"],
                        f"{title} - Lunes a Viernes",
                    ),
                    _table_a_for_subset(
                        franja_df[franja_df["day_type"] == "Weekend"],
                        f"{title} - Sabado y Domingo",
                    ),
                ]
                if r is not None
            ]
        )

    return results


def build_hourly_profile_tables(
    minute_agg: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    minute_agg = minute_agg.copy()
    minute_agg["hour"] = minute_agg["FECHA"].dt.hour
    minute_agg["day_type"] = minute_agg["FECHA"].dt.dayofweek.apply(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )
    hourly_profile_table = (
        minute_agg.groupby(["day_type", "hour", "CATEGORIA"])
        .agg(avg_flow=("flow", "mean"), avg_speed=("mean_speed", "mean"))
        .round(2)
        .unstack(level="CATEGORIA")
        .fillna(0)
    )
    return hourly_profile_table


def build_weekly_seasonality(
    minute_agg: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    daily_flow = minute_agg.groupby(minute_agg["FECHA"].dt.date)["flow"].sum()
    global_mean_flow = daily_flow.mean()
    if not pd.notnull(global_mean_flow) or global_mean_flow <= 0:
        return None
    seasonality_factor = daily_flow / global_mean_flow
    seasonality_df = seasonality_factor.reset_index()
    seasonality_df.columns = ["Fecha", "Factor_Estacionalidad"]
    seasonality_df["Dia_Semana"] = pd.to_datetime(
        seasonality_df["Fecha"]
    ).dt.day_name()
    weekly = (
        seasonality_df.groupby("Dia_Semana")["Factor_Estacionalidad"]
        .mean()
        .round(3)
    )
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly = weekly.reindex(days_order).reset_index()
    return weekly


def plot_hourly_profiles(
    minute_agg: pd.DataFrame,
    *,
    show_flow: bool,
    show_speed: bool,
    save_outputs: bool,
) -> List[Tuple[str, plt.Figure, Optional[str]]]:
    figs: List[Tuple[str, plt.Figure, Optional[str]]] = []
    if sns is None:
        st.error("Se requiere seaborn para graficos de perfil horario.")
        return figs

    minute_agg = minute_agg.copy()
    minute_agg["hour"] = minute_agg["FECHA"].dt.hour
    minute_agg["day_type"] = minute_agg["FECHA"].dt.dayofweek.apply(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )
    hourly_profile_plot = (
        minute_agg.groupby(["day_type", "hour"])
        .agg(total_flow=("flow", "sum"), avg_speed=("mean_speed", "mean"))
        .reset_index()
    )
    if hourly_profile_plot.empty:
        return figs

    palette = ["black", "orange"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if show_flow:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=hourly_profile_plot,
            x="hour",
            y="total_flow",
            hue="day_type",
            marker="o",
            palette=palette,
            ax=ax,
        )
        ax.set_ylabel("Total Average Flow (veh/min)")
        ax.set_xlabel("Time of Day")
        ax.set_xticks(range(0, 24))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        plt.tight_layout()
        save_path = None
        if save_outputs:
            save_path = os.path.join(
                RESULTADOS_DIR, f"perfil_horario_flujo_{timestamp}.png"
            )
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        figs.append(("Perfil horario de flujo", fig, save_path))

    if show_speed:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=hourly_profile_plot,
            x="hour",
            y="avg_speed",
            hue="day_type",
            marker="o",
            palette=palette,
            ax=ax,
        )
        ax.set_ylabel("Speed mean (km/h)")
        ax.set_xlabel("Time of Day")
        ax.set_xticks(range(0, 24))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        plt.tight_layout()
        save_path = None
        if save_outputs:
            save_path = os.path.join(
                RESULTADOS_DIR, f"perfil_horario_velocidad_{timestamp}.png"
            )
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        figs.append(("Perfil horario de velocidad", fig, save_path))

    return figs


def plot_vehicle_composition(
    minute_agg: pd.DataFrame,
    *,
    save_outputs: bool,
) -> Tuple[Optional[plt.Figure], Optional[str], Optional[pd.DataFrame]]:
    if sns is None:
        st.error("Se requiere seaborn para composicion vehicular.")
        return None, None, None

    minute_agg = minute_agg.copy()
    minute_agg["FECHA_DIA"] = minute_agg["FECHA"].dt.floor("D")
    daily_composition = (
        minute_agg.groupby(["FECHA_DIA", "CATEGORIA"])["flow"]
        .sum()
        .unstack(fill_value=0)
    )
    if daily_composition.empty:
        return None, None, None

    daily_pct = daily_composition.apply(
        lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1
    )
    if len(daily_pct) <= 7:
        return None, None, daily_pct

    daily_smooth = daily_pct.rolling(window=7, min_periods=1).mean()
    palette = ["black", "orange", "#666666"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.subplots_adjust(hspace=0.05)
    for idx, cat in enumerate(daily_smooth.columns):
        color = palette[idx % len(palette)]
        ax1.plot(daily_smooth.index, daily_smooth[cat], label=cat, color=color)
        ax2.plot(daily_smooth.index, daily_smooth[cat], label=cat, color=color)

    ax1.set_ylim(85, 100)
    ax2.set_ylim(0, 10)
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.suptitle("Composicion Vehicular (Media Movil de 7 Dias)", fontsize=14)
    fig.text(
        0.04,
        0.5,
        "Porcentaje del Total (%)",
        va="center",
        rotation="vertical",
    )
    ax2.set_xlabel("Fecha")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Categoria",
        loc="upper right",
    )
    ax1.grid(False)
    ax2.grid(False)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])

    save_path = None
    if save_outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            RESULTADOS_DIR, f"composicion_vehicular_{timestamp}.png"
        )
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    percentiles = daily_pct.quantile([0.1, 0.5, 0.9]).round(2)
    percentiles.index = ["p10", "p50", "p90"]
    return fig, save_path, percentiles


def plot_speed_flow_relation(
    minute_agg: pd.DataFrame,
    *,
    agg_freq_minutes: int,
    save_outputs: bool,
) -> Tuple[Optional[plt.Figure], Optional[str]]:
    if lowess is None:
        st.error("Se requiere statsmodels para el ajuste LOESS.")
        return None, None

    highway_agg = (
        minute_agg.groupby(pd.Grouper(key="FECHA", freq=f"{agg_freq_minutes}min"))
        .agg(total_flow=("flow", "sum"), avg_speed=("mean_speed", "mean"))
        .dropna()
    )
    if len(highway_agg) < 20:
        return None, None

    sample_df = highway_agg.sample(
        n=min(5000, len(highway_agg)), random_state=SEED
    )
    frac = 0.3
    lowess_smooth = lowess(
        sample_df["avg_speed"], sample_df["total_flow"], frac=frac
    )
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        sample_df["total_flow"],
        sample_df["avg_speed"],
        alpha=0.5,
        label="Subsample",
        color="orange",
        edgecolors="none",
        s=30,
    )
    ax.plot(
        lowess_smooth[:, 0],
        lowess_smooth[:, 1],
        color="black",
        lw=2.5,
        label="LOESS curve",
    )
    ax.set_xlabel("Total flow (veh/h)")
    ax.set_ylabel("Speed mean (km/h)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.legend(loc="lower left")
    plt.tight_layout()

    save_path = None
    if save_outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            RESULTADOS_DIR, f"velocidad_flujo_{timestamp}.png"
        )
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    return fig, save_path


def render_flow_stats_tab(*, data_dir: Path, db_path: Path) -> None:
    st.subheader("Estadisticas de flujos (legacy)")

    source = st.radio(
        "Fuente de datos",
        ["CSV en Datos/", "DuckDB (tabla flujos)"],
        horizontal=True,
        key="flow_stats_source",
    )
    row_limit = st.number_input(
        "Limite de filas (0 = sin limite)",
        min_value=0,
        value=0,
        step=10000,
        key="flow_stats_row_limit",
    )
    chunksize = st.number_input(
        "Chunk size",
        min_value=10000,
        value=1000000,
        step=50000,
        key="flow_stats_chunksize",
    )
    remove_outliers = st.checkbox(
        "Eliminar outliers (IQR) en velocidad",
        value=True,
        key="flow_stats_outliers",
    )
    min_speed = st.number_input(
        "Velocidad minima (km/h)",
        min_value=0.0,
        value=1.0,
        step=1.0,
        key="flow_stats_min_speed",
    )

    minute_agg = None
    if source == "CSV en Datos/":
        csv_files = sorted([p for p in data_dir.glob("*.csv")])
        if not csv_files:
            st.warning("No se encontraron CSV en Datos/.")
            return
        selected = st.multiselect(
            "Archivos CSV",
            [p.name for p in csv_files],
            key="flow_stats_csv_selection",
        )
        if selected and st.button("Procesar flujos", key="flow_stats_run_csv"):
            selected_paths = [data_dir / name for name in selected]
            minute_agg = build_minute_agg_from_csvs(
                selected_paths,
                chunksize=int(chunksize),
                remove_outliers=bool(remove_outliers),
                min_speed=float(min_speed),
            )
    else:
        if row_limit == 0:
            st.warning(
                "Defina un limite de filas para leer desde DuckDB."
            )
        if st.button("Procesar flujos", key="flow_stats_run_db"):
            try:
                import duckdb

                conn = duckdb.connect(str(db_path), read_only=True)
                limit_clause = ""
                params = []
                if row_limit and int(row_limit) > 0:
                    limit_clause = " LIMIT ?"
                    params.append(int(row_limit))
                query = (
                    "SELECT FECHA, VELOCIDAD, CATEGORIA, PORTICO "
                    "FROM flujos_duckdb" + limit_clause
                )
                df = conn.execute(query, params).df()
                conn.close()
                df_proc = _process_flow_chunk(
                    df,
                    remove_outliers=bool(remove_outliers),
                    min_speed=float(min_speed),
                )
                if df_proc is None or df_proc.empty:
                    st.warning("No se pudieron preparar datos desde DuckDB.")
                else:
                    minute_agg = compute_minute_aggregates(df_proc, freq="min")
            except Exception as exc:
                st.error(f"No se pudo leer DuckDB: {exc}")

    if minute_agg is None:
        minute_agg = st.session_state.get("flow_stats_minute_agg")
    elif minute_agg is not None and not minute_agg.empty:
        st.session_state["flow_stats_minute_agg"] = minute_agg

    if minute_agg is None or minute_agg.empty:
        st.info("No hay datos agregados para analizar.")
        return

    st.caption(f"Filas agregadas: {len(minute_agg):,}")
    save_outputs = st.checkbox(
        "Guardar graficos/tablas en Resultados/",
        value=False,
        key="flow_stats_save_outputs",
    )

    st.markdown("### Analisis")
    show_table_a = st.checkbox(
        "Tabla A - Estadisticas por categoria",
        value=True,
        key="flow_stats_table_a",
    )
    show_temporal = st.checkbox(
        "Dinamica temporal",
        value=False,
        key="flow_stats_temporal",
    )
    show_composition = st.checkbox(
        "Composicion vehicular",
        value=False,
        key="flow_stats_composition",
    )
    show_speed_flow = st.checkbox(
        "Relacion velocidad-flujo",
        value=False,
        key="flow_stats_speed_flow",
    )

    if show_table_a:
        st.subheader("Tabla A")
        tables = generate_table_a(minute_agg)
        exports = []
        for table in tables:
            st.markdown(table["title"])
            if table.get("share") is not None:
                st.caption("Share por categoria")
                st.dataframe(table["share"], width="stretch")
            st.dataframe(table["stats"], width="stretch")
            exports.append(table["export"])
        if save_outputs and exports:
            export_path = os.path.join(
                RESULTADOS_DIR,
                f"tabla_A_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )
            pd.concat(exports, ignore_index=True).to_csv(
                export_path, index=False
            )
            st.caption(f"Tablas exportadas: {export_path}")

    if show_temporal:
        st.subheader("Dinamica temporal")
        hourly_table = build_hourly_profile_tables(minute_agg)
        if hourly_table is not None and not hourly_table.empty:
            st.dataframe(hourly_table, width="stretch")
        weekly = build_weekly_seasonality(minute_agg)
        if weekly is not None and not weekly.empty:
            st.markdown("Factor de estacionalidad semanal")
            st.dataframe(weekly, width="stretch")
        fig_outputs = plot_hourly_profiles(
            minute_agg,
            show_flow=True,
            show_speed=True,
            save_outputs=save_outputs,
        )
        for title, fig, save_path in fig_outputs:
            st.markdown(title)
            st.pyplot(fig, clear_figure=True)
            if save_path:
                st.caption(f"Guardado en: {save_path}")

    if show_composition:
        st.subheader("Composicion vehicular")
        fig, save_path, percentiles = plot_vehicle_composition(
            minute_agg, save_outputs=save_outputs
        )
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
            if save_path:
                st.caption(f"Guardado en: {save_path}")
        if percentiles is not None:
            st.markdown("Percentiles de composicion diaria")
            st.dataframe(percentiles, width="stretch")
        else:
            st.info("No hay suficientes dias para composicion vehicular.")

    if show_speed_flow:
        st.subheader("Relacion velocidad-flujo")
        agg_freq = st.number_input(
            "Frecuencia de agregacion (min)",
            min_value=5,
            value=60,
            step=5,
            key="flow_stats_speed_flow_freq",
        )
        fig, save_path = plot_speed_flow_relation(
            minute_agg,
            agg_freq_minutes=int(agg_freq),
            save_outputs=save_outputs,
        )
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
            if save_path:
                st.caption(f"Guardado en: {save_path}")
        else:
            st.info("No hay suficientes datos para velocidad-flujo.")
