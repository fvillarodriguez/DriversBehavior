#!/usr/bin/env python3
"""
Streamlit main menu to access SUMO tools.
"""
from __future__ import annotations

import os
import sys
import warnings
import importlib
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


import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_page_module(module_name: str):
    return importlib.import_module(module_name)


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
        - **Reward Learning:** Reconstruct AVI transitions and learn driver-behavior rewards.
        - **Events:** Load and visualize accident events on an interactive map.
        - **Experiments Live:** Real-time experiment monitoring.
        - **Simulacion SUMO:** SUMO traffic simulation integration.
        - **Notification system:** Configure system notifications.
        - **Rockfish Cluster:** Conectar, sincronizar y lanzar jobs GPU en el HPC de JHU.
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
        flow_database_app = _load_page_module("src.flow_database_app")
        flow_database_app.main(set_page_config=False, show_exit_button=False)
        
    def p_files():
        files_app = _load_page_module("src.files_app")
        files_app.main(set_page_config=False, show_exit_button=False)
        
    def p_github():
        github_sync_app = _load_page_module("src.github_sync_app")
        github_sync_app.main(set_page_config=False, show_exit_button=False)
        
    def p_clustering():
        _patch_sklearn_parallel_warning_noise()
        clustering_tabs_app = _load_page_module("src.clustering_tabs_app")
        clustering_tabs_app.main(set_page_config=False, show_exit_button=False)
        
    def p_crash():
        _patch_sklearn_parallel_warning_noise()
        cluster_accident_app = _load_page_module("src.cluster_accident_app")
        cluster_accident_app.main(set_page_config=False, show_exit_button=False)

    def p_nlp_severity():
        nlp_severity_app = _load_page_module("src.nlp_severity_app")
        nlp_severity_app.main(set_page_config=False, show_exit_button=False)

    def p_drift():
        _patch_sklearn_parallel_warning_noise()
        drift_detection_app = _load_page_module("src.drift_detection_app")
        drift_detection_app.main(set_page_config=False, show_exit_button=False)
        
    def p_events():
        events_map_app = _load_page_module("src.events_map_app")
        events_map_app.main(set_page_config=False, show_exit_button=False)
        
    def p_experiments():
        experiments_live_app = _load_page_module("src.experiments_live_app")
        experiments_live_app.main(set_page_config=False)
        
    def p_sumo():
        sumo_simulation_app = _load_page_module("src.sumo_simulation_app")
        sumo_simulation_app.main(set_page_config=False, show_exit_button=False)
        
    def p_marl():
        multi_agent_rl_app = _load_page_module("src.multi_agent_rl_app")
        multi_agent_rl_app.main(set_page_config=False, show_exit_button=False)

    def p_test():
        test_page = _load_page_module("src.test_page")
        test_page.main(set_page_config=False, show_exit_button=False)
        
    def p_latex():
        latex_viewer_app = _load_page_module("src.latex_viewer_app")
        latex_viewer_app.Latex(set_page_config=False, show_exit_button=False)

    def p_documentacion():
        documentation_app = _load_page_module("src.documentation_app")
        documentation_app.main(set_page_config=False, show_exit_button=False)

    def p_dask_cluster():
        dask_cluster_app = _load_page_module("src.dask_cluster_app")
        dask_cluster_app.main(set_page_config=False, show_exit_button=False)

    def p_rockfish_cluster():
        rockfish_cluster_app = _load_page_module("src.rockfish_cluster_app")
        rockfish_cluster_app.main(set_page_config=False, show_exit_button=False)

    def p_notify():
        notification_system = _load_page_module("src.notification_system")
        notification_system.render_notification_config()

    def p_trc_paper():
        trc_paper_app = _load_page_module("src.trc_paper_app")
        trc_paper_app.main(set_page_config=False, show_exit_button=False)

    ps_databases = st.Page(p_flow_database, title="Flow database", icon=":material/database:")
    ps_files = st.Page(p_files, title="Files", icon=":material/folder:")
    ps_github = st.Page(p_github, title="GitHub Sync", icon=":material/sync:")
    
    ps_clustering = st.Page(p_clustering, title="Clustering", icon=":material/scatter_plot:")
    ps_crash = st.Page(p_crash, title="Crash prediction", icon=":material/warning:")
    ps_nlp_sev = st.Page(p_nlp_severity, title="NLP in Severity", icon=":material/text_fields:")
    ps_drift = st.Page(p_drift, title="Drift detection", icon=":material/timeline:")
    ps_gnn = st.Page(_render_graph_builder, title="Graph Neural Network", icon=":material/hub:")
    ps_marl = st.Page(p_marl, title="Reward Learning", icon=":material/groups:")
    
    ps_events = st.Page(p_events, title="Events", icon=":material/map:")
    ps_exp = st.Page(p_experiments, title="Experiments Live", icon=":material/science:")
    ps_sumo = st.Page(p_sumo, title="Simulacion SUMO", icon=":material/directions_car:")
    
    ps_notify = st.Page(p_notify, title="Notification system", icon=":material/notifications:")
    ps_dask = st.Page(p_dask_cluster, title="Dask Cluster", icon=":material/account_tree:")
    ps_rockfish = st.Page(p_rockfish_cluster, title="Rockfish Cluster", icon=":material/dns:")
    ps_test = st.Page(p_test, title="Test", icon=":material/bug_report:")
    ps_latex = st.Page(p_latex, title="LaTeX", icon=":material/picture_as_pdf:")
    ps_docs = st.Page(p_documentacion, title="Documentación", icon=":material/description:")
    ps_home = st.Page(_render_home, title="Inicio", icon=":material/home:", default=True)
    ps_trc_paper = st.Page(p_trc_paper, title="TRC Paper Pipeline", icon=":material/science:")

    # --- NAVIGATION SETUP ---
    pg = st.navigation(
        {
            "Navegación": [ps_home, ps_docs],
            "Data & Gestión": [ps_databases, ps_files, ps_github],
            "Análisis & Modelos": [ps_clustering, ps_crash, ps_nlp_sev, ps_drift, ps_gnn, ps_marl],
            "Simulación & Vizualización": [ps_events, ps_exp, ps_sumo],
            "Publicaciones": [ps_trc_paper],
            "Configuración": [ps_notify, ps_dask, ps_rockfish, ps_test, ps_latex],
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
