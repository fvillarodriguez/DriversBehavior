#!/usr/bin/env python3
"""
Streamlit app to explore cluster features stored in DuckDB.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
TABLE_NAME = "cluster_features"


@st.cache_data(show_spinner=False)
def load_features(path: Path) -> pd.DataFrame:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "duckdb no esta instalado. Ejecute `pip install duckdb`."
        ) from exc

    conn = duckdb.connect(str(path), read_only=True)
    try:
        info = conn.execute(
            f"PRAGMA table_info('{TABLE_NAME}')"
        ).fetchall()
        if not info:
            return pd.DataFrame()
        return conn.execute(f"SELECT * FROM {TABLE_NAME}").df()
    finally:
        conn.close()


def list_feature_dbs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("cluster_features*.duckdb"))


def main() -> None:
    st.set_page_config(page_title="Cluster features", layout="wide")
    st.title("Cluster features explorer")

    if st.sidebar.button("Cerrar app"):
        st.sidebar.write("Cerrando...")
        os._exit(0)

    db_paths = list_feature_dbs()
    if not db_paths:
        st.error("No se encontraron archivos cluster_features*.duckdb en Resultados.")
        return

    file_names = [path.name for path in db_paths]
    selected_name = st.sidebar.selectbox("Archivo de variables", file_names)
    selected_path = RESULTS_DIR / selected_name

    try:
        df = load_features(selected_path)
    except ImportError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"No se pudo cargar el archivo: {exc}")
        return

    if df.empty:
        st.warning("El archivo existe pero no contiene datos.")
        return

    total_rows = len(df)
    st.sidebar.caption(f"Archivo: {selected_path}")
    st.sidebar.caption(f"Filas: {total_rows:,}")
    st.sidebar.caption(f"Columnas: {len(df.columns)}")

    use_sample = st.sidebar.checkbox(
        "Usar muestra para graficos", value=total_rows > 200_000
    )
    if use_sample:
        default_sample = min(100_000, total_rows)
        max_sample = min(500_000, total_rows)
        sample_size = st.sidebar.slider(
            "Tamano de muestra",
            min_value=1_000,
            max_value=max_sample,
            value=default_sample,
            step=1_000,
        )
        df_plot = df.sample(sample_size, random_state=42)
    else:
        df_plot = df

    preview_rows = st.sidebar.slider(
        "Filas para vista previa",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    st.subheader("Vista previa de datos")
    st.dataframe(df.head(preview_rows), use_container_width=True)

    st.subheader("Estadisticas descriptivas (todas las variables)")
    try:
        desc = df.describe(include="all").transpose()
    except Exception as exc:  # pragma: no cover - defensive path
        st.error(f"No se pudieron calcular las estadisticas: {exc}")
    else:
        st.dataframe(desc, use_container_width=True)

    st.subheader("Graficos")
    numeric_cols = df_plot.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No hay columnas numericas para graficos.")
        return

    try:
        import plotly.express as px
    except ImportError:
        st.error("plotly no esta instalado. Ejecute `pip install plotly`.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        hist_col = st.selectbox("Histograma", numeric_cols, index=0)
        bins = st.slider("Bins", min_value=10, max_value=100, value=50, step=5)
    with col2:
        box_index = 1 if len(numeric_cols) > 1 else 0
        box_col = st.selectbox("Boxplot", numeric_cols, index=box_index)
    with col3:
        x_col = st.selectbox("Scatter X", numeric_cols, index=0)
        y_index = 1 if len(numeric_cols) > 1 else 0
        y_col = st.selectbox("Scatter Y", numeric_cols, index=y_index)

    hist_fig = px.histogram(df_plot, x=hist_col, nbins=bins)
    st.plotly_chart(hist_fig, use_container_width=True)

    box_fig = px.box(df_plot, y=box_col, points="outliers")
    st.plotly_chart(box_fig, use_container_width=True)

    scatter_fig = px.scatter(df_plot, x=x_col, y=y_col, opacity=0.6)
    st.plotly_chart(scatter_fig, use_container_width=True)


if __name__ == "__main__":
    main()
