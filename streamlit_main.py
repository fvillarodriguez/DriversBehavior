#!/usr/bin/env python3
"""
Streamlit main menu to access SUMO tools.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.flow_database_app as flow_database_app
import src.clustering_tabs_app as clustering_tabs_app
import src.cluster_accident_app as cluster_accident_app
import src.events_map_app as events_map_app
import src.experiments_live_app as experiments_live_app
import src.files_app as files_app
import src.test_page as test_page


def _render_home() -> None:
    st.title("Drivers Behavior")
    st.markdown(
        "Modeling and simulation"
    )
    
    st.markdown(
        """
        This application is a comprehensive pipeline for analyzing and predicting traffic accidents using flow and clustering data.
        
        **Modules Overview:**
        - **DuckDB:** Admin flows database.
        - **Clustering:** Features, clustering and analysis.
        - **Crash prediction:** Train and evaluate accident prediction models.
        - **Eventos:** Load and visualize accident events on an interactive map.
        - **Files:** Manage and browse data files.
        - **Test:** Testing playground.
        """
    )


def main() -> None:
    st.set_page_config(page_title="SUMO Streamlit", layout="wide")

    st.sidebar.title("Menu")
    

    pages: Dict[str, Callable[[], None]] = {
        "Inicio": _render_home,
        "Flow database": lambda: flow_database_app.main(
            set_page_config=False, show_exit_button=False
        ),
        "Clustering": lambda: clustering_tabs_app.main(
            set_page_config=False, show_exit_button=False
        ),
        "Crash prediction": lambda: cluster_accident_app.main(
            set_page_config=False, show_exit_button=False
        ),
        "Experiments Live": lambda: experiments_live_app.main(
            set_page_config=False
        ),
        "Events": lambda: events_map_app.main(
            set_page_config=False, show_exit_button=False
        ),
        "Files": lambda: files_app.main(
            set_page_config=False, show_exit_button=False
        ),
        "Test": lambda: test_page.main(
             set_page_config=False, show_exit_button=False
        ),
    }

    selection = st.sidebar.radio("Secciones", list(pages.keys()))
    pages[selection]()

    if st.sidebar.button("Cerrar"):
        os._exit(0)
        
if __name__ == "__main__":
    main()
