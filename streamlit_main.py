#!/usr/bin/env python3
"""
Streamlit main menu to access SUMO tools.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict
import psutil

# Configurar fallback para MPS (Apple Silicon) antes de importar librerías de ML
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)


def _patch_sklearn_parallel_warning_noise() -> None:
    try:
        from sklearn.utils import parallel as sklearn_parallel  # type: ignore
    except Exception:
        return

    func_wrapper = getattr(sklearn_parallel, "_FuncWrapper", None)
    config_context = getattr(sklearn_parallel, "config_context", None)
    if func_wrapper is None or config_context is None:
        return

    current_call = getattr(func_wrapper, "__call__", None)
    if current_call is None or getattr(current_call, "__name__", "") == "_sumo_quiet_sklearn_funcwrapper_call":
        return

    warning_filter_keys = ["action", "message", "category", "module", "lineno"]

    def _sumo_quiet_sklearn_funcwrapper_call(self, *args, **kwargs):
        config = getattr(self, "config", {})
        warning_filters = getattr(self, "warning_filters", [])

        with config_context(**config), warnings.catch_warnings():
            if warning_filters:
                warnings.resetwarnings()
                for filter_args in warning_filters:
                    this_warning_filter_dict = {
                        key: value
                        for key, value in zip(warning_filter_keys, filter_args)
                        if value is not None
                    }
                    if (
                        "message" not in this_warning_filter_dict
                        and "module" not in this_warning_filter_dict
                    ):
                        warnings.simplefilter(**this_warning_filter_dict, append=True)
                    else:
                        for special_key in ["message", "module"]:
                            this_value = this_warning_filter_dict.get(special_key)
                            if this_value is not None and not isinstance(this_value, str):
                                this_warning_filter_dict[special_key] = this_value.pattern
                        warnings.filterwarnings(**this_warning_filter_dict, append=True)
            return self.function(*args, **kwargs)

    _sumo_quiet_sklearn_funcwrapper_call.__name__ = "_sumo_quiet_sklearn_funcwrapper_call"
    func_wrapper.__call__ = _sumo_quiet_sklearn_funcwrapper_call


_patch_sklearn_parallel_warning_noise()

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
import src.sumo_simulation_app as sumo_simulation_app
import src.test_page as test_page
import src.drift_detection_app as drift_detection_app
import src.github_sync_app as github_sync_app
import src.notification_system as notification_system
import src.multi_agent_rl_app as multi_agent_rl_app
import src.latex_viewer_app as latex_viewer_app
import src.nlp_severity_app as nlp_severity_app


def _render_home() -> None:
    st.title("Drivers Behavior Modeling and Simulation")
    st.markdown(
        """
        This application is a comprehensive pipeline for analyzing and predicting traffic accidents using flow and clustering data.
        
        **Modules Overview:**
        - **Flow database:** Admin flows database DuckDB.
        - **Files:** Manage and browse data files.
        - **GitHub Sync:** Sync local files with the remote repository.
        - **Clustering:** Features, clustering and analysis.
        - **Crash prediction:** Train and evaluate accident prediction models.
        - **NLP in Severity:** Build granular flow + text datasets to model accident severity.
        - **Drift detection:** Recalibration and drift detection replication module.
        - **Graph Neural Network:** Construct graphs for GNN models using crash prediction features.
        - **Multi Agent RL:** Multi-Agent Reinforcement Learning modules.
        - **Events:** Load and visualize accident events on an interactive map.
        - **Experiments Live:** Real-time experiment monitoring.
        - **Simulacion SUMO:** SUMO traffic simulation integration.
        - **Notification system:** Configure system notifications.
        - **Test:** Testing playground.
        """
    )


def _render_graph_builder() -> None:
    import src.graph_builder_app as graph_builder_app

    graph_builder_app.main(set_page_config=False, show_exit_button=False)


def main() -> None:
    st.set_page_config(page_title="Drivers Behavior", layout="wide", page_icon="🚗")

    # --- PAGE DEFINITIONS ---
    # Wrappers to provide unique function names for st.Page URL inference
    
    def p_flow_database():
        flow_database_app.main(set_page_config=False, show_exit_button=False)
        
    def p_files():
        files_app.main(set_page_config=False, show_exit_button=False)
        
    def p_github():
        github_sync_app.main(set_page_config=False, show_exit_button=False)
        
    def p_clustering():
        clustering_tabs_app.main(set_page_config=False, show_exit_button=False)
        
    def p_crash():
        cluster_accident_app.main(set_page_config=False, show_exit_button=False)

    def p_nlp_severity():
        nlp_severity_app.main(set_page_config=False, show_exit_button=False)

    def p_drift():
        drift_detection_app.main(set_page_config=False, show_exit_button=False)
        
    def p_events():
        events_map_app.main(set_page_config=False, show_exit_button=False)
        
    def p_experiments():
        experiments_live_app.main(set_page_config=False)
        
    def p_sumo():
        sumo_simulation_app.main(set_page_config=False, show_exit_button=False)
        
    def p_marl():
        multi_agent_rl_app.main(set_page_config=False, show_exit_button=False)

    def p_test():
        test_page.main(set_page_config=False, show_exit_button=False)
        
    def p_latex():
        latex_viewer_app.Latex(set_page_config=False, show_exit_button=False)

    ps_databases = st.Page(p_flow_database, title="Flow database", icon=":material/database:")
    ps_files = st.Page(p_files, title="Files", icon=":material/folder:")
    ps_github = st.Page(p_github, title="GitHub Sync", icon=":material/sync:")
    
    ps_clustering = st.Page(p_clustering, title="Clustering", icon=":material/scatter_plot:")
    ps_crash = st.Page(p_crash, title="Crash prediction", icon=":material/warning:")
    ps_nlp_sev = st.Page(p_nlp_severity, title="NLP in Severity", icon=":material/text_fields:")
    ps_drift = st.Page(p_drift, title="Drift detection", icon=":material/timeline:")
    ps_gnn = st.Page(_render_graph_builder, title="Graph Neural Network", icon=":material/hub:")
    ps_marl = st.Page(p_marl, title="Multi Agent RL", icon=":material/groups:")
    
    ps_events = st.Page(p_events, title="Events", icon=":material/map:")
    ps_exp = st.Page(p_experiments, title="Experiments Live", icon=":material/science:")
    ps_sumo = st.Page(p_sumo, title="Simulacion SUMO", icon=":material/directions_car:")
    
    ps_notify = st.Page(notification_system.render_notification_config, title="Notification system", icon=":material/notifications:")
    ps_test = st.Page(p_test, title="Test", icon=":material/bug_report:")
    ps_latex = st.Page(p_latex, title="LaTeX", icon=":material/picture_as_pdf:")
    ps_home = st.Page(_render_home, title="Inicio", icon=":material/home:", default=True)

    # --- NAVIGATION SETUP ---
    pg = st.navigation(
        {
            "Navegación": [ps_home],
            "Data & Gestión": [ps_databases, ps_files, ps_github],
            "Análisis & Modelos": [ps_clustering, ps_crash, ps_nlp_sev, ps_drift, ps_gnn, ps_marl],
            "Simulación & Vizualización": [ps_events, ps_exp, ps_sumo],
            "Configuración": [ps_notify, ps_test, ps_latex],
        }
    )
    
    pg.run()

    #st.sidebar.markdown("---")

    @st.fragment(run_every=2)
    def _render_system_stats():
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / 1e9  # GB
        cpu_usage = process.cpu_percent(interval=None) # Non-blocking immediate check
        st.caption(f"**Proceso** | CPU: {cpu_usage:.1f}% | RAM: {mem_usage:.2f} GB")

    with st.sidebar:
        _render_system_stats()

    def _clear_sidebar_memory():
        keys_to_clear = [
            "loaded_graph",
            "graph_path",
            "df_pm_cache",
            "df_port",
            "df_acc",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = None
        try:
            import pandas as pd
            import polars as pl
            import numpy as np
            import torch
            for key, value in list(st.session_state.items()):
                if isinstance(value, (pd.DataFrame, pl.DataFrame, pl.LazyFrame, np.ndarray, torch.Tensor)):
                    st.session_state[key] = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    col_clear, col_exit = st.sidebar.columns(2)
    if col_clear.button("",type='tertiary', icon="🧹"):
        _clear_sidebar_memory()
        col_clear.success("Memoria limpiada")
    if col_exit.button("Cerrar"):
        os._exit(0)
        
if __name__ == "__main__":
    main()
