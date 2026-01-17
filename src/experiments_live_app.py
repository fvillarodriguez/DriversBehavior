#!/usr/bin/env python3
"""
Streamlit app to monitor experiment results in real time.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"


def _list_live_db_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        RESULTS_DIR.glob("experiment_live_*.sqlite"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def _read_live_db(
    path: Path,
) -> Tuple[Dict[str, object], pd.DataFrame, Optional[Dict[str, object]]]:
    meta: Dict[str, object] = {}
    rows_df = pd.DataFrame()
    best_row: Optional[Dict[str, object]] = None

    con = sqlite3.connect(path, timeout=1)
    try:
        if _table_exists(con, "meta"):
            rows = con.execute("SELECT key, value FROM meta").fetchall()
            for key, value in rows:
                try:
                    meta[key] = json.loads(value)
                except Exception:
                    meta[key] = value

        if _table_exists(con, "results"):
            rows = con.execute(
                "SELECT id, created_at, payload_json FROM results ORDER BY id"
            ).fetchall()
            payloads = []
            for row_id, created_at, payload_json in rows:
                try:
                    payload = json.loads(payload_json)
                except Exception:
                    payload = {"raw": payload_json}
                payload["_row_id"] = row_id
                payload["_created_at"] = created_at
                payloads.append(payload)
            if payloads:
                rows_df = pd.DataFrame(payloads)

        if _table_exists(con, "best"):
            row = con.execute(
                "SELECT payload_json FROM best ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                try:
                    best_row = json.loads(row[0])
                except Exception:
                    best_row = {"raw": row[0]}
    finally:
        con.close()
    return meta, rows_df, best_row


def _render_find_samples_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: Find samples sizes")
    metric_options = {
        "best_f1": "F1",
        "accuracy": "Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "roc_auc": "ROC-AUC",
        "fnr": "FNR (menor es mejor)",
    }
    available_metrics = {k: v for k, v in metric_options.items() if k in df.columns}
    if not available_metrics:
        st.info("No hay metricas disponibles para graficar.")
        st.dataframe(df, width="stretch")
        return

    metric_labels = list(available_metrics.values())
    selected_metric_label = st.selectbox(
        "Metrica a graficar",
        metric_labels,
        key="live_find_samples_metric",
    )
    metric_key = next(
        k for k, v in available_metrics.items() if v == selected_metric_label
    )

    plot_df = df.copy()
    if "error" in plot_df.columns:
        plot_df = plot_df[
            plot_df["error"].isna() | (plot_df["error"] == "")
        ]

    if best_row is None and not plot_df.empty and metric_key in plot_df.columns:
        if metric_key == "fnr":
            best_row = plot_df.loc[plot_df[metric_key].idxmin()].to_dict()
        else:
            best_row = plot_df.loc[plot_df[metric_key].idxmax()].to_dict()

    if best_row:
        st.markdown("**Resultado optimo**")
        objective_label = best_row.get("objective_label")
        if objective_label:
            st.caption(f"Objetivo: {objective_label}")
        st.caption(
            f"{best_row.get('segment_portico_last', '?')} -> {best_row.get('segment_portico_next', '?')} "
            f"| {best_row.get('window_start', '?')} a {best_row.get('window_end', '?')}"
        )
        metrics_cols = [
            "best_f1",
            "accuracy",
            "recall",
            "precision",
            "roc_auc",
            "fnr",
        ]
        metrics_payload = {
            key: best_row.get(key) for key in metrics_cols if key in best_row
        }
        if metrics_payload:
            st.json(metrics_payload)
        model_path = best_row.get("model_path")
        if model_path and isinstance(model_path, str):
            st.caption(f"Modelo: {model_path}")

    tab_viz, tab_data = st.tabs(["Grafico", "Datos"])
    with tab_viz:
        if "candidate_rank" in plot_df.columns and metric_key in plot_df.columns:
            try:
                import altair as alt
                chart = (
                    alt.Chart(plot_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "candidate_rank:O",
                            axis=alt.Axis(title="Candidato"),
                        ),
                        y=alt.Y(
                            metric_key,
                            axis=alt.Axis(title=available_metrics[metric_key]),
                        ),
                        color=alt.Color(
                            "window_days:O",
                            title="Ventana (dias)",
                        ),
                        tooltip=[
                            "candidate_rank",
                            "window_days",
                            "accidents_per_day",
                            metric_key,
                            "segment_portico_last",
                            "segment_portico_next",
                        ],
                    )
                    .interactive()
                )
                st.altair_chart(chart, width="stretch")
            except ImportError:
                st.warning("Altair no instalado.")
        else:
            st.info("No hay columnas suficientes para graficar.")

        if {"window_days", "accidents_per_day"}.issubset(plot_df.columns):
            try:
                import altair as alt
                scatter = (
                    alt.Chart(plot_df)
                    .mark_circle(size=70, opacity=0.7)
                    .encode(
                        x=alt.X(
                            "window_days:Q",
                            axis=alt.Axis(title="Ventana (dias)"),
                        ),
                        y=alt.Y(
                            "accidents_per_day:Q",
                            axis=alt.Axis(title="Accidentes por dia"),
                        ),
                        color=alt.Color(
                            "segment_portico_last:N",
                            title="Portico inicio",
                        ),
                        tooltip=[
                            "window_days",
                            "accidents_per_day",
                            "segment_portico_last",
                            "segment_portico_next",
                        ],
                    )
                    .interactive()
                )
                st.altair_chart(scatter, width="stretch")
            except ImportError:
                pass

    with tab_data:
        st.dataframe(df, width="stretch")


def _render_features_sampler_view(df: pd.DataFrame) -> None:
    if "best_f1" in df.columns and "type" in df.columns:
        st.caption("Mejor F1 por estrategia:")
        best_by_type = df.loc[df.groupby("type")["best_f1"].idxmax()]
        if not best_by_type.empty:
            cols = st.columns(len(best_by_type))
            for idx, row in enumerate(best_by_type.itertuples(), start=0):
                with cols[idx]:
                    delta_label = ""
                    if "k" in best_by_type.columns:
                        delta_label = f"k={row.k}"
                    st.metric(
                        label=row.type,
                        value=f"{row.best_f1:.4f}",
                        delta=delta_label,
                    )

    tab_viz, tab_data = st.tabs(["Grafico", "Datos"])
    with tab_viz:
        if "k" in df.columns:
            metric_options = {
                "best_f1": "Best F1 Score",
                "accuracy": "Accuracy",
                "recall": "Recall (Sens)",
                "precision": "Precision",
                "roc_auc": "ROC-AUC",
                "fnr": "FNR",
            }
            available_metrics = {
                k: v for k, v in metric_options.items() if k in df.columns
            }
            if not available_metrics:
                available_metrics = (
                    {"best_f1": "Best F1 Score"} if "best_f1" in df.columns else {}
                )
            selected_metric_key = "best_f1"
            if available_metrics:
                col_sel, _ = st.columns([0.3, 0.7])
                with col_sel:
                    selected_metric_label = st.selectbox(
                        "Metrica a graficar",
                        options=list(available_metrics.values()),
                        index=0,
                        key="live_features_metric",
                    )
                    selected_metric_key = next(
                        k
                        for k, v in available_metrics.items()
                        if v == selected_metric_label
                    )
            if selected_metric_key in df.columns and "type" in df.columns:
                try:
                    import altair as alt
                    y_min = df[selected_metric_key].min()
                    y_max = df[selected_metric_key].max()
                    padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
                    chart = (
                        alt.Chart(df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "k", axis=alt.Axis(title="Top K Features")
                            ),
                            y=alt.Y(
                                selected_metric_key,
                                scale=alt.Scale(
                                    domain=[
                                        max(0, y_min - padding),
                                        min(1, y_max + padding),
                                    ]
                                ),
                                axis=alt.Axis(
                                    title=available_metrics[selected_metric_key]
                                ),
                            ),
                            color="type",
                            tooltip=["k", selected_metric_key, "type", "n_features"],
                        )
                        .interactive()
                    )
                    st.altair_chart(chart, width="stretch")
                except ImportError:
                    st.warning("Altair no instalado.")
            else:
                st.info("Columnas insuficientes para graficar.")
        else:
            st.info("No hay columna 'k' para graficar.")

    with tab_data:
        st.dataframe(df, width="stretch")


def _render_best_highway_section_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: Best highway section")

    plot_df = df.copy()
    if "error" in plot_df.columns:
        plot_df = plot_df[
            plot_df["error"].isna() | (plot_df["error"] == "")
        ]
    if plot_df.empty:
        st.info("No hay resultados validos para graficar.")
        st.dataframe(df, width="stretch")
        return

    dataset_types = []
    if "type" in plot_df.columns:
        dataset_types = sorted(
            [t for t in plot_df["type"].dropna().unique().tolist() if t]
        )
    selected_type = "Todos"
    if dataset_types:
        selected_type = st.selectbox(
            "Dataset",
            ["Todos"] + dataset_types,
            key="live_best_section_type",
        )
    if selected_type != "Todos" and "type" in plot_df.columns:
        plot_df = plot_df[plot_df["type"] == selected_type]
        if plot_df.empty:
            st.info("No hay resultados para el dataset seleccionado.")
            return

    metric_cols = [
        col
        for col in ("accuracy", "recall", "roc_auc")
        if col in plot_df.columns
    ]
    if not metric_cols:
        st.info("No hay metricas disponibles para graficar.")
        st.dataframe(df, width="stretch")
        return

    def _segment_label(row: pd.Series) -> str:
        last = row.get("segment_portico_last")
        nxt = row.get("segment_portico_next")
        eje = row.get("segment_eje")
        calzada = row.get("segment_calzada")
        last = "?" if pd.isna(last) else str(last)
        nxt = "?" if pd.isna(nxt) else str(nxt)
        label = f"{last}->{nxt}"
        if pd.notna(eje) or pd.notna(calzada):
            eje_val = "-" if pd.isna(eje) else str(eje)
            calzada_val = "-" if pd.isna(calzada) else str(calzada)
            label = f"{eje_val}/{calzada_val} {label}"
        return label

    plot_df = plot_df.copy()
    plot_df["segment_label"] = plot_df.apply(_segment_label, axis=1)
    if "segment_index" in plot_df.columns:
        plot_df["segment_index"] = pd.to_numeric(
            plot_df["segment_index"], errors="coerce"
        )
    else:
        plot_df["segment_index"] = range(1, len(plot_df) + 1)

    long_df = plot_df.melt(
        id_vars=["segment_label", "segment_index"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    if long_df.empty:
        st.info("No hay datos validos para graficar.")
        st.dataframe(df, width="stretch")
        return

    metric_labels = {
        "accuracy": "Accuracy",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
    }
    long_df["metric_label"] = long_df["metric"].map(metric_labels)
    order = (
        plot_df.sort_values("segment_index")["segment_label"]
        .drop_duplicates()
        .tolist()
    )

    try:
        import plotly.express as px
    except ImportError:
        pivot = (
            long_df.pivot_table(
                index="segment_label", columns="metric_label", values="value"
            )
            .reindex(order)
        )
        st.line_chart(pivot)
    else:
        fig = px.line(
            long_df,
            x="segment_label",
            y="value",
            color="metric_label",
            markers=True,
            category_orders={"segment_label": order},
        )
        fig.update_layout(
            xaxis_title="Tramo",
            yaxis_title="Metrica",
            legend_title_text="Metrica",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Datos")
    st.dataframe(plot_df, width="stretch")


def main(*, set_page_config: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Experiments Live", layout="wide")
    st.title("Experimentos en vivo")

    db_files = _list_live_db_files()
    db_names = [p.name for p in db_files]
    if not db_names:
        st.info("No hay bases de datos de experimentos.")
        return
    selected = st.selectbox(
        "Base de datos",
        options=db_names,
        index=0,
    )

    auto_refresh = st.sidebar.checkbox(
        "Actualizar automaticamente", value=True
    )
    refresh_seconds = st.sidebar.number_input(
        "Intervalo (segundos)",
        min_value=1,
        value=10,
        step=1,
    )
    if st.sidebar.button("Actualizar ahora"):
        st.rerun()

    path = next((p for p in db_files if p.name == selected), None)
    if path is None:
        st.warning("Seleccione una base valida.")
        return

    meta, df, best_row = _read_live_db(path)
    st.caption(f"Archivo: {path}")
    if meta:
        with st.expander("Meta", expanded=False):
            st.json(meta)

    if df.empty:
        st.warning("No hay resultados en la base de datos.")
    else:
        experiment_name = str(meta.get("experiment", "")).lower()
        is_find_samples = (
            "find samples" in experiment_name
            or df.get("experiment", pd.Series())
            .astype(str)
            .str.contains("find samples", case=False, na=False)
            .any()
            or "candidate_rank" in df.columns
        )
        is_best_section = (
            "best highway section" in experiment_name
            or df.get("experiment", pd.Series())
            .astype(str)
            .str.contains("best highway section", case=False, na=False)
            .any()
        )

        if is_best_section:
            _render_best_highway_section_view(df, best_row)
        elif is_find_samples:
            _render_find_samples_view(df, best_row)
        else:
            _render_features_sampler_view(df)

    if auto_refresh:
        time.sleep(float(refresh_seconds))
        st.rerun()


if __name__ == "__main__":
    main()
