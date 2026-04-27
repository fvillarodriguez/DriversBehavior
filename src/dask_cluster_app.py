#!/usr/bin/env python3
"""
Streamlit page for managing the local Dask cluster.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
DASK_APP_DIR = ROOT_DIR / "dask"


def _ensure_cluster_app_importable() -> None:
    try:
        import cluster_app  # noqa: F401

        return
    except ModuleNotFoundError:
        if str(DASK_APP_DIR) not in sys.path:
            sys.path.insert(0, str(DASK_APP_DIR))
    try:
        import cluster_app  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No se puede importar Dask Cluster App. Instale el complemento con "
            "`pip install -e ./dask` en el entorno de Streamlit."
        ) from exc


def main(set_page_config: bool = False, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="Dask Cluster", layout="wide")

    st.title("Dask Cluster")
    st.markdown(
        "Administra el scheduler y worker Dask usados por los experimentos distribuidos."
    )

    try:
        _ensure_cluster_app_importable()
        from cluster_app.integrations.streamlit import render_cluster_panel

        render_cluster_panel(key="tesis_dask_cluster")
    except Exception as exc:
        st.error(f"Dask Cluster no disponible: {exc}")
        st.code("pip install -e ./dask", language="bash")

    if show_exit_button and st.button("Cerrar"):
        raise SystemExit(0)


if __name__ == "__main__":
    main(set_page_config=True)
