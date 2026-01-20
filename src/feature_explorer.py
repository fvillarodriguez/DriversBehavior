from __future__ import annotations

import glob
import os
from pathlib import Path
import math
from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import visualization as viz


try:
    import duckdb
except ImportError:
    duckdb = None

from src.config import RESULTADOS_DIR


def _feature_db_path() -> Path:
    return Path(RESULTADOS_DIR) / "features.duckdb"


def _list_duckdb_tables(db_path: Path) -> List[str]:
    if not duckdb or not db_path.exists():
        return []
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        # duckdb < 0.9 might need different queries, but information_schema is standard-ish
        # listing all tables in main schema
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        tables = [row[0] for row in con.execute(query).fetchall()]
        con.close()
        
        # Filter for feature tables
        filtered = [
            t for t in tables 
            if t.startswith("features_") and not t.endswith("_latest")
        ]
        return sorted(filtered, reverse=True)
    except Exception as e:
        st.warning(f"Error reading DuckDB tables: {e}")
        return []


def _load_data_from_duckdb(db_path: Path, table_name: str) -> Optional[pd.DataFrame]:
    if not duckdb:
        return None
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        df = con.execute(f"SELECT * FROM {table_name}").df()
        con.close()
        
        # Ensure timestamp column exists (from ts_min)
        if 'ts_min' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_min'], unit='m')
        elif 'timestamp' in df.columns:
             # Ensure it is datetime
             df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        st.error(f"Error loading table {table_name} from DuckDB: {e}")
        return None


def _list_feature_files(results_dir: Path) -> List[str]:
    all_feature_files = glob.glob(os.path.join(results_dir, "features_*.csv"))
    filtered = []
    for path in all_feature_files:
        name = os.path.basename(path)
        if "selected_names" in name:
            continue
        if "subset" in name:
            continue
        if not name.startswith("features_"):
            continue
        filtered.append(path)
    return sorted(filtered, key=os.path.basename, reverse=True)



def _render_controls(
    *,
    df: pd.DataFrame,
    all_vars: List[str],
    portico_col_name: str,
    accident_df: pd.DataFrame,
) -> Tuple[
    List[str],
    List[str],
    Tuple[pd.Timestamp, pd.Timestamp],
    str,
    bool,
    Optional[str],
    Optional[str],
    Optional[str],
]:
    selected_variables = st.multiselect(
        "Variables a visualizar",
        all_vars,
        default=all_vars if all_vars else None,
        key="feature_explorer_vars",
    )
    selected_porticos = st.multiselect(
        "Porticos",
        sorted(df[portico_col_name].unique()),
        default=sorted(df[portico_col_name].unique()),
        key="feature_explorer_porticos",
    )
    min_time, max_time = df["timestamp"].min(), df["timestamp"].max()
    py_min_time = min_time.to_pydatetime()
    py_max_time = max_time.to_pydatetime()
    start_time, end_time = st.slider(
        "Ventana temporal",
        min_value=py_min_time,
        max_value=py_max_time,
        value=(py_min_time, py_max_time),
        key="feature_explorer_time",
    )
    chart_type = st.selectbox(
        "Tipo de grafico",
        ["Lineas", "Dispersion", "Dispersion 3D", "Histograma", "Box Plot"],
        key="feature_explorer_chart",
    )

    x_var = y_var = z_var = None
    if chart_type == "Dispersion 3D":
        st.markdown("Ejes 3D")
        x_var = st.selectbox(
            "Eje X",
            all_vars,
            key="feature_explorer_x_var",
        )
        y_options = [v for v in all_vars if v != x_var] or all_vars
        y_var = st.selectbox(
            "Eje Y",
            y_options,
            key="feature_explorer_y_var",
        )
        z_options = [v for v in all_vars if v not in {x_var, y_var}] or all_vars
        z_var = st.selectbox(
            "Eje Z",
            z_options,
            key="feature_explorer_z_var",
        )

    normalize = st.checkbox(
        "Normalizar variables (Min-Max)",
        value=False,
        key="feature_explorer_normalize",
    )

    return (
        selected_variables,
        selected_porticos,
        (start_time, end_time),
        chart_type,
        normalize,
        x_var,
        y_var,
        z_var,
    )


def render_feature_explorer() -> None:
    st.subheader("Feature explorer")

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "Resultados"
    data_dir = project_root / "Datos"

    feature_files = _list_feature_files(results_dir)
    
    # DuckDB check
    db_path = _feature_db_path()
    duck_tables = _list_duckdb_tables(db_path)

    if not feature_files and not duck_tables:
        st.error(
            "No se encontraron archivos de features (`features_*.csv`) en Resultados/ ni tablas en features.duckdb."
        )
        return

    # User choice: Source type or unified list?
    # Let's unify them but label them
    # options: "DK: table_name", "CSV: file_name"
    
    # Actually, simpler: if duck_tables exist, prefer them or show them mixed?
    # The user said "features de GNN en duckdb...no csv!", so let's show tables if available.
    
    options = [{"type": "none", "name": None, "label": "Seleccione una fuente"}]
    for t in duck_tables:
        options.append({"type": "duckdb", "name": t, "label": f"[DB]  {t}"})
    for f in feature_files:
        options.append({"type": "csv", "name": f, "label": f"[CSV] {os.path.basename(f)}"})
    
    # If user strictly wants NO CSV, we could filter, but having them as fallback is nice. 
    # But since the error message was specific about not finding CSV, maybe they only have DB now.
    
    selected_option = st.selectbox(
        "Fuente de features",
        options,
        format_func=lambda x: x["label"],
        key="feature_explorer_source",
        index=0,
    )

    if selected_option["type"] == "none":
        st.info("Seleccione una fuente para cargar features.")
        return


    portico_files = glob.glob(os.path.join(data_dir, "*orticos.csv"))
    if not portico_files:
        st.warning("No se encontraron archivos de porticos en Datos/.")
        return
    selected_portico_file = st.selectbox(
        "Archivo de porticos",
        sorted(portico_files, key=os.path.basename, reverse=True),
        format_func=os.path.basename,
        key="feature_explorer_portico_file",
    )

    event_files = glob.glob(os.path.join(data_dir, "*ventos*.csv"))
    event_files.insert(0, "No mostrar accidentes")
    selected_event_file = st.selectbox(
        "Archivo de accidentes (opcional)",
        sorted(event_files, key=os.path.basename, reverse=True),
        format_func=lambda x: os.path.basename(x)
        if x != "No mostrar accidentes"
        else "No mostrar accidentes",
        key="feature_explorer_event_file",
    )

    if selected_option["type"] == "duckdb":
        df = _load_data_from_duckdb(db_path, selected_option["name"])
    else:
        df = viz.load_data(selected_option["name"])
    gantry_df = viz.load_gantry_data(selected_portico_file)
    if selected_event_file == "No mostrar accidentes":
        accident_df = pd.DataFrame()
    else:
        accident_df = viz.load_accident_data(selected_event_file, gantry_df)

    if df is None or df.empty:
        return

    portico_col_name = next(
        (col for col in ["portico", "id_portico"] if col in df.columns), None
    )
    if not portico_col_name:
        st.error("No se encontro columna de portico en features.")
        return

    all_vars = sorted(
        [
            col
            for col in df.select_dtypes(include=["number"]).columns
            if col not in [portico_col_name, "ts_min"]
        ]
    )


    with st.expander("Filtros y visualizacion", expanded=True):
        (
            selected_variables,
            selected_porticos,
            time_window,
            chart_type,
            normalize,
            x_var,
            y_var,
            z_var,
        ) = _render_controls(
            df=df,
            all_vars=all_vars,
            portico_col_name=portico_col_name,
            accident_df=accident_df,
        )
        default_max_points = 20000 if chart_type == "Dispersion 3D" else 50000
        max_points = st.number_input(
            "Max puntos a graficar",
            min_value=1000,
            max_value=200000,
            value=default_max_points,
            step=1000,
            key="feature_explorer_max_points",
        )

    required_vars = (
        selected_variables
        if chart_type != "Dispersion 3D"
        else [x_var, y_var, z_var]
    )
    if not required_vars or not selected_porticos:
        st.info("Seleccione variables y porticos para generar la grafica.")
        return

    start_time, end_time = time_window
    filtered_df = df[
        (df[portico_col_name].isin(selected_porticos))
        & (df["timestamp"] >= start_time)
        & (df["timestamp"] <= end_time)
    ]

    if filtered_df.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    df_to_plot = filtered_df
    if len(filtered_df) > max_points:
        stride = max(1, int(math.ceil(len(filtered_df) / max_points)))
        df_to_plot = filtered_df.iloc[::stride].copy()
        st.caption(
            f"Se muestrean {len(df_to_plot):,} de {len(filtered_df):,} filas "
            f"(stride={stride})."
        )
    else:
        df_to_plot = filtered_df.copy()
    vars_to_normalize = (
        selected_variables
        if chart_type != "Dispersion 3D"
        else [x_var, y_var, z_var]
    )
    if normalize and vars_to_normalize:
        for col in vars_to_normalize:
            min_val = df_to_plot[col].min()
            max_val = df_to_plot[col].max()
            if max_val > min_val:
                df_to_plot[col] = (df_to_plot[col] - min_val) / (
                    max_val - min_val
                )

    viz_map = {
        "Lineas": "Líneas",
        "Dispersion": "Dispersión (Scatter)",
        "Dispersion 3D": "Dispersión 3D",
    }
    viz_chart_type = viz_map.get(chart_type, chart_type)

    fig = viz.create_figure(
        df_to_plot,
        viz_chart_type,
        selected_variables
        if chart_type != "Dispersion 3D"
        else [x_var, y_var, z_var],
        portico_col_name,
    )

    if fig is None:
        st.warning("No se pudo generar la grafica.")
        return

    if not accident_df.empty and chart_type in ["Lineas", "Dispersion"]:
        accidents_in_view = accident_df[
            (accident_df["portico"].isin(selected_porticos))
            & (accident_df["timestamp"] >= start_time)
            & (accident_df["timestamp"] <= end_time)
        ]
        annotations = (
            list(fig.layout.annotations) if fig.layout.annotations else []
        )
        if not accidents_in_view.empty and selected_variables:
            y_col = selected_variables[0]
            for _, acc in accidents_in_view.iterrows():
                y_val = None
                rows = df_to_plot[df_to_plot[portico_col_name] == acc["portico"]]
                if not rows.empty:
                    y_val = pd.to_numeric(rows[y_col], errors="coerce").mean()
                if not pd.notna(y_val):
                    y_val = df_to_plot[y_col].mean()
                annotations.append(
                    dict(
                        x=acc["timestamp"],
                        y=y_val,
                        xref="x",
                        yref="y",
                        text="X",
                        showarrow=False,
                        font=dict(
                            family="sans serif",
                            size=16,
                            color="red",
                        ),
                        align="center",
                        bordercolor="white",
                        borderwidth=1,
                        bgcolor="rgba(255,255,255,0.5)",
                    )
                )
        if annotations:
            fig.update_layout(annotations=annotations)
    elif not accident_df.empty and chart_type == "Dispersion 3D":
        for tr in fig.data:
            try:
                tr.update(marker=dict(size=1, symbol="x"))
            except Exception:
                tr.update(marker=dict(size=3))

        accidents_in_view = accident_df[
            (accident_df["portico"].isin(selected_porticos))
            & (accident_df["timestamp"] >= start_time)
            & (accident_df["timestamp"] <= end_time)
        ]
        if not accidents_in_view.empty and x_var and y_var and z_var:
            xs, ys, zs, texts = [], [], [], []
            df_by_port = {
                p: g.sort_values("timestamp")
                for p, g in df_to_plot.groupby(portico_col_name)
            }
            for _, acc in accidents_in_view.iterrows():
                p = acc["portico"]
                sub = df_by_port.get(p)
                if sub is None or sub.empty:
                    continue
                try:
                    nearest_idx = (
                        sub["timestamp"] - acc["timestamp"]
                    ).abs().idxmin()
                    row = sub.loc[nearest_idx]
                except Exception:
                    sub_prior = sub[sub["timestamp"] <= acc["timestamp"]]
                    row = sub.iloc[-1] if sub_prior.empty else sub_prior.iloc[-1]
                xv = pd.to_numeric(row.get(x_var, None), errors="coerce")
                yv = pd.to_numeric(row.get(y_var, None), errors="coerce")
                zv = pd.to_numeric(row.get(z_var, None), errors="coerce")
                if pd.notna(xv) and pd.notna(yv) and pd.notna(zv):
                    xs.append(float(xv))
                    ys.append(float(yv))
                    zs.append(float(zv))
                    texts.append(f"{p}<br>{acc['timestamp']}")
            if xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="markers",
                        marker=dict(size=6, color="red"),
                        name="Accidentes",
                        text=texts,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

    st.plotly_chart(fig, width="stretch")
