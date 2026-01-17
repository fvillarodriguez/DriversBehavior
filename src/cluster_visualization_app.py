#!/usr/bin/env python3
"""
Streamlit app to explore cluster results interactively.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
CLUSTER_LABEL_PATTERN = re.compile(
    r"^cluster_(?P<method>kmeans|gmm|hdbscan)(?:_k(?P<k>\d+))?(?:.*)?\.csv$"
)


@st.cache_data(show_spinner=False)
def load_cluster_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def list_cluster_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    candidates = sorted(RESULTS_DIR.glob("cluster_*.csv"))
    return [path for path in candidates if CLUSTER_LABEL_PATTERN.match(path.name)]


def normalize_profile(profile: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "Z-score":
        means = profile.mean()
        stds = profile.std(ddof=0).replace(0, 1)
        return (profile - means) / stds
    if method == "Min-max":
        mins = profile.min()
        ranges = (profile.max() - mins).replace(0, 1)
        return (profile - mins) / ranges
    return profile


def main() -> None:
    st.set_page_config(page_title="Cluster visualization", layout="wide")
    st.title("Cluster visualization")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.error("plotly no esta instalado. Ejecute `pip install plotly`.")
        return

    files = list_cluster_files()
    if not files:
        st.warning("No se encontraron archivos cluster_*.csv de clustering en Resultados.")
        return

    file_names = [path.name for path in files]
    selected_name = st.sidebar.selectbox("Archivo de clusters", file_names)
    selected_path = RESULTS_DIR / selected_name

    if not CLUSTER_LABEL_PATTERN.match(selected_path.name):
        st.error("El nombre del archivo no coincide con el formato esperado.")
        return

    df = load_cluster_file(selected_path)
    if "cluster_label" not in df.columns:
        st.error("El archivo no contiene la columna cluster_label.")
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "cluster_label"]
    if len(numeric_cols) < 3:
        st.error("No hay suficientes variables numericas para la visualizacion 3D.")
        return

    cluster_values = sorted(df["cluster_label"].dropna().unique().tolist())
    selected_clusters = st.sidebar.multiselect(
        "Clusters", cluster_values, default=cluster_values
    )
    if not selected_clusters:
        st.warning("Seleccione al menos un cluster.")
        return

    df = df[df["cluster_label"].isin(selected_clusters)]
    total_rows = len(df)
    st.sidebar.caption(f"Filas disponibles: {total_rows:,}")

    if total_rows == 0:
        st.warning("No hay datos para los clusters seleccionados.")
        return

    use_sample = st.sidebar.checkbox(
        "Usar muestra para graficos", value=total_rows > 200_000
    )
    if use_sample:
        default_sample = min(100_000, total_rows)
        max_sample = min(200_000, total_rows)
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

    st.subheader("3D scatter por cluster")
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x_col = st.selectbox("Eje X", numeric_cols, index=0)
    with col_y:
        y_col = st.selectbox(
            "Eje Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0
        )
    with col_z:
        z_col = st.selectbox(
            "Eje Z", numeric_cols, index=2 if len(numeric_cols) > 2 else 0
        )

    scatter_fig = px.scatter_3d(
        df_plot,
        x=x_col,
        y=y_col,
        z=z_col,
        color="cluster_label",
        opacity=0.7,
    )
    scatter_fig.update_traces(marker={"size": 2})
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Distribucion de una variable")
    hist_col = st.selectbox("Variable", numeric_cols, index=0, key="hist_var")
    bins = st.slider("Bins", min_value=10, max_value=100, value=50, step=5)
    hist_fig = px.histogram(
        df_plot, x=hist_col, nbins=bins, color="cluster_label"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader("Caracterizacion de clusters")
    default_profile_vars = numeric_cols[: min(8, len(numeric_cols))]
    default_matrix_vars = numeric_cols[: min(4, len(numeric_cols))]

    tabs = st.tabs(["Perfil medio", "Violin/box", "Matriz de dispersion", "Radar"])
    with tabs[0]:
        st.caption(
            "Promedios por cluster con opcion de normalizacion para resaltar contrastes."
        )
        profile_vars = st.multiselect(
            "Variables para el perfil",
            numeric_cols,
            default=default_profile_vars,
            key="profile_vars",
        )
        norm_method = st.selectbox(
            "Normalizacion", ["Z-score", "Min-max", "Sin normalizar"], index=0
        )
        if not profile_vars:
            st.info("Seleccione al menos una variable.")
        else:
            profile = (
                df.groupby("cluster_label")[profile_vars]
                .mean()
                .sort_index()
            )
            profile_norm = normalize_profile(profile, norm_method)
            color_scale = "RdBu" if norm_method == "Z-score" else "Viridis"
            heat_fig = px.imshow(
                profile_norm,
                aspect="auto",
                color_continuous_scale=color_scale,
            )
            heat_fig.update_layout(
                xaxis_title="Variables",
                yaxis_title="Cluster",
                coloraxis_colorbar_title=norm_method,
            )
            st.plotly_chart(heat_fig, use_container_width=True)
            show_table = st.checkbox(
                "Mostrar tabla de promedios", value=False, key="profile_table"
            )
            if show_table:
                st.dataframe(profile.round(3), use_container_width=True)

    with tabs[1]:
        st.caption("Distribucion por cluster para una variable con violin o boxplot.")
        dist_col = st.selectbox(
            "Variable", numeric_cols, index=0, key="dist_var"
        )
        dist_type = st.radio(
            "Tipo", ["Violin", "Box"], horizontal=True, key="dist_type"
        )
        if dist_type == "Violin":
            dist_fig = px.violin(
                df_plot,
                x="cluster_label",
                y=dist_col,
                color="cluster_label",
                box=True,
                points="outliers",
            )
        else:
            dist_fig = px.box(
                df_plot,
                x="cluster_label",
                y=dist_col,
                color="cluster_label",
                points="outliers",
            )
        dist_fig.update_layout(xaxis_title="Cluster")
        st.plotly_chart(dist_fig, use_container_width=True)

    with tabs[2]:
        st.caption(
            "Matriz de dispersion con muestra para comparar relaciones entre variables."
        )
        matrix_vars = st.multiselect(
            "Variables para matriz",
            numeric_cols,
            default=default_matrix_vars,
            key="matrix_vars",
        )
        if len(matrix_vars) < 2:
            st.info("Seleccione al menos dos variables.")
        else:
            matrix_df = df_plot
            if len(df_plot) > 5_000:
                max_sample = min(20_000, len(df_plot))
                default_sample = min(5_000, max_sample)
                sample_size = st.slider(
                    "Tamano de muestra",
                    min_value=1_000,
                    max_value=max_sample,
                    value=default_sample,
                    step=1_000,
                    key="matrix_sample",
                )
                matrix_df = df_plot.sample(sample_size, random_state=42)
            matrix_fig = px.scatter_matrix(
                matrix_df,
                dimensions=matrix_vars,
                color="cluster_label",
            )
            matrix_fig.update_traces(diagonal_visible=False)
            st.plotly_chart(matrix_fig, use_container_width=True)

    with tabs[3]:
        st.caption("Radar con perfiles normalizados para comparar clusters.")
        radar_vars = st.multiselect(
            "Variables para radar",
            numeric_cols,
            default=default_profile_vars[: min(6, len(default_profile_vars))],
            key="radar_vars",
        )
        radar_clusters = st.multiselect(
            "Clusters en radar",
            selected_clusters,
            default=selected_clusters[: min(6, len(selected_clusters))],
            key="radar_clusters",
        )
        if len(radar_vars) < 3:
            st.info("Seleccione al menos tres variables.")
        elif not radar_clusters:
            st.info("Seleccione al menos un cluster.")
        else:
            radar_profile = (
                df[df["cluster_label"].isin(radar_clusters)]
                .groupby("cluster_label")[radar_vars]
                .mean()
                .sort_index()
            )
            radar_profile = normalize_profile(radar_profile, "Min-max")
            theta = radar_vars + [radar_vars[0]]
            fill_mode = "toself" if len(radar_clusters) <= 6 else None
            radar_fig = go.Figure()
            for cluster_label, row in radar_profile.iterrows():
                values = row.tolist()
                values += values[:1]
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=theta,
                        name=f"Cluster {cluster_label}",
                        fill=fill_mode,
                        opacity=0.6 if fill_mode else 0.8,
                    )
                )
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                ),
                showlegend=True,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("Tamanos de cluster")
    counts = (
        df["cluster_label"]
        .value_counts()
        .sort_index()
        .reset_index(name="count")
        .rename(columns={"index": "cluster_label"})
    )
    bar_fig = px.bar(
        counts, x="cluster_label", y="count", text="count", color="cluster_label"
    )
    bar_fig.update_traces(textposition="outside")
    bar_fig.update_layout(xaxis_title="Cluster", yaxis_title="Filas")
    st.plotly_chart(bar_fig, use_container_width=True)
    st.dataframe(counts, use_container_width=True)


if __name__ == "__main__":
    main()
