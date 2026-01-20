from __future__ import annotations

import builtins
import glob
import hashlib
import json
import os
import re
import sys
import traceback
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import streamlit as st
import torch
from torch_geometric.data import HeteroData

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Auto-batch thresholds
AUTO_BATCH_RANGE_DAYS = 14
AUTO_BATCH_ROW_THRESHOLD = 2_000_000
DATA_DIR = ROOT_DIR / "Datos"
FORCE_SNAPSHOT_FEATURES = True

# --- Imports from SRC ---
# We assume app.py has added the parent dir to sys.path
from src.graphsmote import _safe_clean_edges
from src.graph import PMIndex, _compute_time_cutoffs, _save_normalization_stats
from src.utils import (
    DEFAULT_DUCKDB_FILE,
    FLOW_TABLE_NAME,
    _slugify,
    buscar_columna,
    find_candidate_porticos,
    load_accidentes,
    load_porticos,
    process_accidentes_df,
    reconstruct_portico_sequence,
)
from src.snapshot_pipeline import (
    SnapshotConfig,
    SnapshotFeatureBuilder,
    flatten_snapshot_features,
    build_static_topology,
)
from src.snapshot_sequences import (
    SequenceConfig,
    build_sequence_index,
)
from src.config import (
    BATCH_SIZE,
    DT_MINUTES,
    FLOAT,
    GRAPHSMOTE_K,
    DECODER_EPOCHS,
    GUARD_BAND_MINUTES,
    HORIZON_MINUTES,
    INCLUDE_DOWNSTREAM_IN_LABEL,
    N_TRIALS,
    NUM_EPOCHS_OPTUNA,
    RESULTADOS_DIR,
    SEED,
    SEQ_LENGTH,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.features import compute_pm_features
from src.graph_visualization import render_visual_graph_tab
from src.feature_explorer import render_feature_explorer

FEATURE_SELECTION_DIR = Path(RESULTADOS_DIR)

try:
    import duckdb
except ImportError:
    duckdb = None

# --- HELPERS ---
import logging
import time

class StreamlitLogHandler(logging.Handler):
    """
    Custom Logging Handler to stream logs to a Streamlit container.
    Maintains a buffer of the last N lines.
    """
    def __init__(self, container, max_lines=100):
        super().__init__()
        self.container = container
        self.max_lines = max_lines
        self.lines = []
        self.last_update = 0
        self.update_interval = 0.5 # update at most every 0.5s to reduce lag

    def emit(self, record):
        try:
            msg = self.format(record)
            self.lines.append(msg)
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]
            
            current_time = time.time()
            if current_time - self.last_update > self.update_interval:
                log_text = "\n".join(self.lines)
                # We use code block for monospaced font
                self.container.code(log_text, language="text")
                self.last_update = current_time
        except Exception:
            self.handleError(record)

# --- HELPERS ---
def _normalize_portico_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    lowered = cleaned.str.lower()
    invalid = lowered.isin({"", "nan", "none", "null"})
    cleaned = cleaned.mask(invalid, None)
    return cleaned.str.replace(r"\.0$", "", regex=True)


def _ensure_porticos_loaded() -> Optional[pd.DataFrame]:
    if st.session_state.get("df_port") is None:
        with st.spinner("Cargando pórticos automáticamente..."):
            df_port = load_porticos()
            if df_port is not None:
                st.session_state.df_port = df_port
            else:
                st.error("No se pudo cargar 'Datos/Porticos.csv'. Verifique el archivo.")
                return None
    return st.session_state.get("df_port")


def _render_tramo_selector(
    df_port: pd.DataFrame,
    key_prefix: str,
) -> Tuple[Optional[List[str]], Optional[str]]:
    if df_port is None or df_port.empty:
        st.warning("No hay pórticos disponibles para definir el tramo.")
        return None, None

    tramo_mode = st.radio(
        "Definición del Grafo (Tramo)",
        ["Manual", "Puntos Negros (Reporte)"],
        index=0,
        key=f"{key_prefix}_tramo_mode",
    )

    porticos_a_incluir = None
    preselected_time_choice = None

    if tramo_mode == "Puntos Negros (Reporte)":
        pn_reports = sorted(
            glob.glob(os.path.join(RESULTADOS_DIR, "analisis_puntos_negros_*.csv"))
        )
        pn_mode_options = ["Usar reporte existente", "Calcular nuevo"]
        if pn_reports:
            pn_mode = st.radio(
                "Puntos Negros",
                pn_mode_options,
                horizontal=True,
                key=f"{key_prefix}_pn_mode_select",
            )
        else:
            pn_mode = "Calcular nuevo"

        if pn_mode == "Calcular nuevo":
            if st.button("Calcular nuevo reporte", key=f"{key_prefix}_pn_calc_report"):
                acc_df = st.session_state.get("df_acc")
                if acc_df is None or acc_df.empty:
                    st.warning("Cargue accidentes en la pestaña Eventos.")
                else:
                    stats = _compute_puntos_negros_report(acc_df, df_port)
                    if stats is None or stats.empty:
                        st.warning("No se pudo generar el reporte de puntos negros.")
                    else:
                        timestamp = datetime.now().strftime("%d%m%Y_%H%M")
                        output_path = os.path.join(
                            RESULTADOS_DIR,
                            f"analisis_puntos_negros_{timestamp}.csv",
                        )
                        stats.to_csv(
                            output_path,
                            index=False,
                            sep=";",
                            decimal=",",
                        )
                        st.session_state[f"{key_prefix}_pn_report_select"] = (
                            output_path
                        )
                        pn_reports = sorted(
                            glob.glob(
                                os.path.join(
                                    RESULTADOS_DIR, "analisis_puntos_negros_*.csv"
                                )
                            )
                        )

        if not pn_reports:
            st.warning("No se encontraron reportes de puntos negros.")
        else:
            selected_report = st.selectbox(
                "Reporte de puntos negros",
                pn_reports,
                format_func=lambda x: os.path.basename(x),
                key=f"{key_prefix}_pn_report_select",
            )
            if selected_report:
                try:
                    df_pn = pd.read_csv(selected_report, sep=";", decimal=",")
                    if not df_pn.empty:
                        top = df_pn.iloc[0]
                        st.write("Top Punto Negro:", top)
                        tramo_str = str(top["tramo"])
                        franja = str(top.get("franja_horaria", ""))

                        if "06-10" in franja:
                            preselected_time_choice = "2"
                        elif "18-22" in franja:
                            preselected_time_choice = "3"
                        else:
                            preselected_time_choice = "1"

                        p1_raw, p2_raw = [s.strip() for s in tramo_str.split("->")]
                        porticos_a_incluir = reconstruct_portico_sequence(
                            df_porticos=df_port,
                            p1=p1_raw,
                            p2=p2_raw,
                            max_anterior=4,
                            max_posterior=0,
                            prefer_family=True,
                            deduplicate_slice=True,
                        )
                        st.info(
                            "Pórticos seleccionados "
                            f"({len(porticos_a_incluir)}): {porticos_a_incluir}"
                        )
                except Exception as e:
                    st.error(f"Error leyendo reporte: {e}")

    else:
        autopista = None
        if "autopista" in df_port.columns:
            autopista_options = list(
                pd.Index(df_port["autopista"].dropna().unique())
            )
            autopista_map = {
                "C": "Ruta 5 (Central)",
                "NO": "General Velásquez",
            }
            if autopista_options:
                auto_key = f"{key_prefix}_autopista"
                if (
                    auto_key not in st.session_state
                    or st.session_state[auto_key] not in autopista_options
                ):
                    st.session_state[auto_key] = autopista_options[0]
                autopista = st.selectbox(
                    "Autopista",
                    autopista_options,
                    format_func=lambda x: autopista_map.get(x, x),
                    key=auto_key,
                )
                mask_aut = df_port["autopista"] == autopista
            else:
                mask_aut = pd.Series(True, index=df_port.index)
        else:
            mask_aut = pd.Series(True, index=df_port.index)

        calzada_options = list(
            pd.Index(df_port[mask_aut]["calzada"].dropna().unique())
        )
        if not calzada_options:
            st.warning("No hay calzadas disponibles para esta autopista.")
            calzada = None
            mask_calz = pd.Series(False, index=df_port.index)
            eje = None
            mask_eje = mask_calz
        else:
            calzada_key = f"{key_prefix}_calzada"
            if (
                calzada_key not in st.session_state
                or st.session_state[calzada_key] not in calzada_options
            ):
                st.session_state[calzada_key] = calzada_options[0]
            calzada = st.selectbox("Calzada", calzada_options, key=calzada_key)
            mask_calz = mask_aut & (df_port["calzada"] == calzada)
            eje_options = list(
                pd.Index(df_port[mask_calz]["eje"].dropna().unique())
            )
            if not eje_options:
                st.warning("No hay ejes disponibles para esta calzada.")
                eje = None
                mask_eje = mask_calz
            elif len(eje_options) == 1:
                eje = eje_options[0]
                st.session_state[f"{key_prefix}_eje"] = eje
                mask_eje = mask_calz & (df_port["eje"] == eje)
            else:
                eje_key = f"{key_prefix}_eje"
                if (
                    eje_key not in st.session_state
                    or st.session_state[eje_key] not in eje_options
                ):
                    st.session_state[eje_key] = eje_options[0]
                eje = st.selectbox("Eje", eje_options, key=eje_key)
                mask_eje = mask_calz & (df_port["eje"] == eje)

        df_sel = df_port[mask_eje].copy()
        if "orden" in df_sel.columns:
            df_sel["orden_num"] = pd.to_numeric(df_sel["orden"], errors="coerce")
        df_sel["km_num"] = pd.to_numeric(df_sel.get("km"), errors="coerce")
        sort_cols = []
        if "orden_num" in df_sel.columns and df_sel["orden_num"].notna().any():
            sort_cols.append("orden_num")
        if "km_num" in df_sel.columns and df_sel["km_num"].notna().any():
            sort_cols.append("km_num")
        if sort_cols:
            df_sel = df_sel.sort_values(sort_cols)
        else:
            df_sel = df_sel.sort_values("portico")
        portico_series = df_sel["portico"].dropna().astype(str).str.strip()
        porticos_disp = list(pd.Index(portico_series).drop_duplicates())

        if not porticos_disp:
            st.warning("No hay pórticos disponibles para esta calzada.")
            p_start = None
            p_end = None
        else:
            tramo_key = (autopista, calzada, eje)
            tramo_key_state = f"{key_prefix}_tramo_key"
            if st.session_state.get(tramo_key_state) != tramo_key:
                st.session_state[tramo_key_state] = tramo_key
                st.session_state[f"{key_prefix}_portico_start"] = porticos_disp[0]
                st.session_state[f"{key_prefix}_portico_end"] = porticos_disp[-1]
                st.session_state.pop(f"{key_prefix}_manual_porticos", None)
            c1, c2 = st.columns(2)
            with c1:
                start_key = f"{key_prefix}_portico_start"
                if st.session_state.get(start_key) not in porticos_disp:
                    st.session_state[start_key] = porticos_disp[0]
                p_start = st.selectbox(
                    "Pórtico Inicio", porticos_disp, key=start_key
                )
            with c2:
                try:
                    start_idx = porticos_disp.index(p_start)
                except ValueError:
                    start_idx = 0
                end_options = porticos_disp[start_idx + 1 :]
                if not end_options:
                    end_options = [p_start]
                end_key = f"{key_prefix}_portico_end"
                if st.session_state.get(end_key) not in end_options:
                    st.session_state[end_key] = end_options[-1]
                p_end = st.selectbox("Pórtico Fin", end_options, key=end_key)

        if p_start and p_end and st.button(
            "Confirmar Tramo", key=f"{key_prefix}_confirm_tramo"
        ):
            try:
                p_list = porticos_disp
                idx_s = p_list.index(p_start)
                idx_e = p_list.index(p_end)
                if idx_s > idx_e:
                    idx_s, idx_e = idx_e, idx_s
                porticos_a_incluir = p_list[idx_s : idx_e + 1]
                st.session_state[f"{key_prefix}_manual_porticos"] = (
                    porticos_a_incluir
                )
                st.success(
                    "Tramo manual seleccionado: "
                    f"{len(porticos_a_incluir)} pórticos"
                )
            except ValueError:
                st.error("Error calculando rango de pórticos")

        manual_key = f"{key_prefix}_manual_porticos"
        if manual_key in st.session_state:
            porticos_a_incluir = st.session_state[manual_key]
            st.write(f"Selección actual: {porticos_a_incluir}")

    return porticos_a_incluir, preselected_time_choice


def _apply_tramo_filter(
    df_pm: Optional[pd.DataFrame],
    porticos_filter: Optional[Sequence[str]],
    *,
    status: Optional[object] = None,
) -> Optional[pd.DataFrame]:
    if df_pm is None or df_pm.empty or not porticos_filter:
        return df_pm

    porticos_norm = _normalize_portico_series(pd.Series(porticos_filter))
    porticos_norm = list(pd.Index(porticos_norm).dropna().drop_duplicates())
    if not porticos_norm:
        msg = "No hay pórticos válidos en el tramo seleccionado."
        if status is not None:
            status.warning(msg)
        else:
            st.warning(msg)
        return df_pm

    portico_col = _find_column_by_keywords(
        df_pm, ["portico", "cod_portico", "id_portico"]
    )
    if not portico_col:
        msg = "Columna de pórtico no encontrada en features."
        if status is not None:
            status.warning(msg)
        else:
            st.warning(msg)
        return df_pm

    df_pm[portico_col] = _normalize_portico_series(df_pm[portico_col])
    initial_len = len(df_pm)
    df_pm = df_pm[df_pm[portico_col].isin(porticos_norm)].copy()
    msg = f"Filtrado por tramo: {initial_len} -> {len(df_pm)} registros."
    if status is not None:
        status.text(msg)
    else:
        st.info(msg)
    return df_pm


def _rename_vehicle_type_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    suffix_map = {
        "_1": "_light",
        "_2": "_heavy",
        "_3": "_motorcycles",
    }
    rename_map = {}
    for col in df.columns:
        for suffix, label in suffix_map.items():
            if col.endswith(suffix):
                rename_map[col] = col[: -len(suffix)] + label
                break
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


SNAPSHOT_GROUP_OPTIONS = {
    "Velocidad base": "speed_stats",
    "Flujo total + slow pct": "flow_base",
    "Flow por categoria": "flow_cat",
    "Mix por categoria": "mix_cat",
    "Carriles": "lane_stats",
    "Ventanas / Lags / Pendientes": "window_features",
    "Gradientes espaciales": "spatial_gradients",
    "Codificacion temporal": "temporal_encodings",
}
SNAPSHOT_GROUP_KEYS = list(SNAPSHOT_GROUP_OPTIONS.values())
SNAPSHOT_SPEED_COLUMNS = [
    "speed_mean",
    "speed_std",
    "speed_var",
    "speed_median",
    "speed_min",
    "speed_max",
    "speed_q25",
    "speed_q75",
    "speed_q10",
    "speed_q90",
    "speed_iqr",
    "speed_harmonic",
    "speed_entropy",
]
SNAPSHOT_FLOW_COLUMNS = ["flow_total", "slow_pct"]
SNAPSHOT_LANE_COLUMNS = [
    "lane_speed_mean_std",
    "lane_flow_std",
    "lane_count",
]
SNAPSHOT_TEMPORAL_COLUMNS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]
SNAPSHOT_WINDOW_SUFFIXES = ("_w_mean", "_w_std", "_lag1", "_lag2", "_slope")
SNAPSHOT_GRADIENT_SUFFIXES = ("_grad_upstream", "_grad_downstream")
SNAPSHOT_WINDOW_DEFAULT_COLS = [
    "speed_mean",
    "speed_std",
    "speed_var",
    "speed_median",
    "speed_harmonic",
    "flow_total",
    "slow_pct",
]
SNAPSHOT_WINDOW_COL_OPTIONS = [
    "speed_mean",
    "speed_std",
    "speed_var",
    "speed_median",
    "speed_min",
    "speed_max",
    "speed_q25",
    "speed_q75",
    "speed_q10",
    "speed_q90",
    "speed_iqr",
    "speed_harmonic",
    "speed_entropy",
    "flow_total",
    "slow_pct",
    "lane_speed_mean_std",
    "lane_flow_std",
    "lane_count",
]
SNAPSHOT_GRADIENT_DEFAULT_COLS = ["speed_mean", "speed_mean_w_mean"]


def _filter_flow_features(
    df_pm: Optional[pd.DataFrame],
    gen_params: Dict[str, object],
) -> Optional[pd.DataFrame]:
    if df_pm is None or df_pm.empty:
        return df_pm
    metrics = gen_params.get("metrics") or []
    categories = gen_params.get("categories") or []
    global_metrics = gen_params.get("global_metrics") or []
    if not metrics and not global_metrics:
        return df_pm

    metric_map = {
        "Flow": "flujo",
        "Flow (veh/h)": "flujo_hph",
        "Speed (Mean)": "velocidad_media",
        "Speed (Std)": "velocidad_std",
        "Speed (Harmonic)": "velocidad_harmonica",
        "Speed (Median)": "velocidad_mediana",
        "Speed (Q25)": "velocidad_q25",
        "Speed (Q75)": "velocidad_q75",
        "Speed (IQR)": "velocidad_iqr",
        "Speed (MAD)": "velocidad_mad",
        "Speed (CV)": "velocidad_cv",
        "Speed (Skew)": "velocidad_skew",
        "Speed (Kurtosis)": "velocidad_kurt_exceso",
        "Speed (Entropy)": "velocidad_entropy",
        "Robust Flow (Median)": "flujo_mediana_subbins",
        "Robust Flow (Trimmed)": "flujo_media_recortada",
        "Robust Flow (Winsor)": "flujo_media_winsor",
        "Density": "densidad",
        "Gap Time-Space": "brecha_ts",
        "Flow Diff (dot_q)": "dot_q",
        "Smooth Speed": "smooth_v",
    }
    cat_number_map = {"Light": 1, "Heavy": 2, "Motorcycles": 3}
    keep_cols = ["portico", "ts_min"]
    lanes = gen_params.get("lanes", 3)

    if categories and metrics:
        selected_prefixes = []
        for cat_ui in categories:
            cat_num = cat_number_map.get(cat_ui)
            if cat_num is None:
                continue
            for met_ui, met_db in metric_map.items():
                if met_ui in metrics:
                    selected_prefixes.append(met_db)
                    col_name = f"{met_db}_{cat_num}"
                    if col_name in df_pm.columns:
                        if met_ui == "Flow" and lanes > 0:
                            df_pm[col_name] = df_pm[col_name] / lanes
                        keep_cols.append(col_name)
        for met_db in set(selected_prefixes):
            if not any(col.startswith(f"{met_db}_") for col in keep_cols):
                keep_cols.extend(
                    [c for c in df_pm.columns if c.startswith(f"{met_db}_")]
                )

    global_map = {
        "Total Flow": "flujo_total",
        "Total Density": "densidad_total",
        "Global Harmonic Speed": "v_armonica",
        "Heavy Prop.": "prop_pesados",
        "Congestion Index": "indice_congestion",
        "Mix Entropy": "entropia_mezcla",
        "Capacity (est)": "q_max_p95",
        "V/C Ratio": "vc_ratio",
        "FreeFlow Speed (Night)": "v_freeflow_p95_noche",
        "Relative FF Speed": "v_rel_ff",
        "Speed Drop": "v_drop",
        "Slow Share (30%)": "slowshare_0p30",
        "Slow Share (50%)": "slowshare_0p50",
        "Slow Share (70%)": "slowshare_0p70",
        "FD Envelope": "envolvente_FD",
        "Critical Density (k_crit)": "k_crit",
        "Density Ratio (k/k_crit)": "dens_ratio",
        "FD Residual": "resid_FD",
        "Capacity Slack": "slack_cap",
        "Outlier Rate": "outlier_rate",
        "Missing Rate": "missing_rate",
    }
    if global_metrics:
        for glob_ui, glob_db in global_map.items():
            if glob_ui in global_metrics and glob_db in df_pm.columns:
                keep_cols.append(glob_db)

    available_cols = [c for c in keep_cols if c in df_pm.columns]
    df_pm = df_pm[available_cols].copy()
    df_pm = _rename_vehicle_type_suffixes(df_pm)
    return df_pm


def _filter_snapshot_features(
    df_pm: Optional[pd.DataFrame],
    gen_params: Dict[str, object],
) -> Optional[pd.DataFrame]:
    if df_pm is None or df_pm.empty:
        return df_pm

    selected_groups = gen_params.get("snapshot_groups") or []
    if not selected_groups:
        return df_pm

    group_set = set(selected_groups)
    keep_cols: List[str] = []
    for base_col in ("portico", "ts_min", "timestamp"):
        if base_col in df_pm.columns:
            keep_cols.append(base_col)

    if "speed_stats" in group_set:
        keep_cols.extend([c for c in SNAPSHOT_SPEED_COLUMNS if c in df_pm.columns])
    if "flow_base" in group_set:
        keep_cols.extend([c for c in SNAPSHOT_FLOW_COLUMNS if c in df_pm.columns])
    if "flow_cat" in group_set:
        keep_cols.extend([c for c in df_pm.columns if c.startswith("flow_cat_")])
    if "mix_cat" in group_set:
        keep_cols.extend([c for c in df_pm.columns if c.startswith("mix_cat_")])
    if "lane_stats" in group_set:
        keep_cols.extend([c for c in SNAPSHOT_LANE_COLUMNS if c in df_pm.columns])
    if "window_features" in group_set:
        keep_cols.extend(
            [c for c in df_pm.columns if c.endswith(SNAPSHOT_WINDOW_SUFFIXES)]
        )
    if "spatial_gradients" in group_set:
        keep_cols.extend(
            [c for c in df_pm.columns if c.endswith(SNAPSHOT_GRADIENT_SUFFIXES)]
        )
    if "temporal_encodings" in group_set:
        keep_cols.extend([c for c in SNAPSHOT_TEMPORAL_COLUMNS if c in df_pm.columns])

    keep_cols = list(dict.fromkeys(keep_cols))
    if not keep_cols:
        return df_pm
    return df_pm.loc[:, keep_cols].copy()


def _normalize_column_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _normalize_match_key(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def _find_match_column(
    df: pd.DataFrame, candidates: Sequence[str]
) -> Optional[str]:
    normalized = {_normalize_match_key(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_match_key(candidate)
        col = normalized.get(key)
        if col:
            return col
    return None


def _find_column_by_keywords(
    df: object, keywords: Sequence[str]
) -> Optional[str]:
    if df is None or _is_empty_frame(df):
        return None
    normalized = {_normalize_column_key(col): col for col in df.columns}
    for key in keywords:
        key_norm = _normalize_column_key(key)
        if key_norm in normalized:
            return normalized[key_norm]
    for key in keywords:
        key_norm = _normalize_column_key(key)
        for norm, original in normalized.items():
            if key_norm and key_norm in norm:
                return original
    return None


def _list_event_files() -> list:
    if not DATA_DIR.exists():
        return []
    candidates = []
    for path in DATA_DIR.glob("*.csv"):
        if path.name.lower().startswith("eventos"):
            candidates.append(path)
    return sorted(candidates)


def _infer_edge_feature_dim(graph_data: HeteroData) -> int:
    for edge_type in graph_data.edge_types:
        store = graph_data[edge_type]
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            edge_attr = store.edge_attr
            if edge_attr.dim() > 1:
                return int(edge_attr.shape[1])
            return 1
    return 0


def _infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    num_layers = 0
    num_heads = None
    hidden = None
    for key in state_dict.keys():
        if key.startswith("convs."):
            try:
                idx = int(key.split(".")[1])
                num_layers = max(num_layers, idx + 1)
            except Exception:
                pass
        if key.endswith(".att_src") and num_heads is None:
            t = state_dict[key]
            if t.dim() >= 3:
                num_heads = int(t.shape[1])
                hidden = int(t.shape[2])
    if (hidden is None or num_heads is None) and "norms.0.pm.weight" in state_dict:
        d = int(state_dict["norms.0.pm.weight"].shape[0])
        for h in (8, 6, 4, 2, 1):
            if d % h == 0:
                num_heads = h
                hidden = d // h
                break
    return {
        "num_layers": int(num_layers if num_layers else 2),
        "num_heads": int(num_heads if num_heads else 4),
        "hidden_channels": int(hidden if hidden else 32),
    }


def _load_hparams_for_model(model_path: str) -> Dict[str, object]:
    hparams_path = Path(model_path).with_name(
        f"{Path(model_path).stem}_hparams.json"
    )
    if not hparams_path.exists():
        return {}
    try:
        with open(hparams_path, "r") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _select_latest_gat_model(
    *,
    use_graphsmote: bool,
    graph_obj: Dict[str, object],
) -> Optional[str]:
    files = sorted(
        glob.glob(os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt")),
        key=os.path.getmtime,
    )
    if not files:
        return None
    want_imgagn = bool(graph_obj.get("imgagn_best_params")) or (
        "ImGAGN" in str(graph_obj.get("filename", ""))
    )
    filtered = []
    for path in files:
        name = os.path.basename(path)
        if use_graphsmote and "_GraphSMOTE" not in name:
            continue
        if (not use_graphsmote) and "_GraphSMOTE" in name:
            continue
        if want_imgagn and "_ImGAGN" not in name:
            continue
        filtered.append(path)
    candidates = filtered if filtered else files
    return candidates[-1] if candidates else None


def _list_hpo_files_for_training(
    *,
    use_graphsmote: bool,
    graph_obj: Dict[str, object],
) -> List[str]:
    all_hp_files = sorted(
        glob.glob(os.path.join(RESULTADOS_DIR, "optuna_hyperparams_*.csv")),
        key=os.path.getmtime,
    )
    if not all_hp_files:
        return []
    want_imgagn = bool(graph_obj.get("imgagn_best_params")) or (
        "ImGAGN" in str(graph_obj.get("filename", ""))
    )

    def _score_hp(path: str) -> Tuple[int, int, int, float]:
        name = os.path.basename(path)
        has_smote = "_GraphSMOTE" in name
        has_imgagn = "_ImGAGN" in name
        has_base = "_Base" in name
        ok_smote = (has_smote == use_graphsmote)
        ok_imgagn = (has_imgagn == want_imgagn)
        base_pref = has_base and not use_graphsmote
        return (int(ok_smote), int(ok_imgagn), int(base_pref), os.path.getmtime(path))

    hp_files_sorted = sorted(all_hp_files, key=_score_hp)
    return [
        f
        for f in hp_files_sorted
        if ("_GraphSMOTE" in os.path.basename(f)) == use_graphsmote
    ]


def _compute_binary_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    if cm.shape != (2, 2):
        return {}
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    far = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "specificity": float(specificity),
    }


def _compute_n_syn_from_ratio(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    minority_class: int,
    target_pos_ratio: float,
    max_new: Optional[int] = None,
) -> int:
    y_train = y[train_mask]
    pos = int((y_train == minority_class).sum().item())
    neg = int((y_train != minority_class).sum().item())
    total = pos + neg
    if total == 0:
        return 0
    ratio = float(target_pos_ratio)
    if ratio <= 0.0:
        return 0
    if ratio >= 0.999999:
        ratio = 0.999999
    n_syn = int(max(0, round((ratio * total - pos) / (1.0 - ratio))))
    if max_new is not None:
        n_syn = min(n_syn, int(max_new))
    return n_syn


def _load_optuna_params_from_csv(path: str) -> Dict[str, object]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _network_config_to_hparams(
    cfg: Dict[str, object], *, use_graphsmote: bool
) -> Dict[str, object]:
    params: Dict[str, object] = {
        "hidden_channels": int(cfg.get("hidden_channels", 64)),
        "num_heads": int(cfg.get("num_heads", 4)),
        "num_layers": int(cfg.get("num_layers", 2)),
        "dropout": float(cfg.get("dropout", 0.2)),
        "aggr1": cfg.get("aggr1", "sum"),
        "aggr2": cfg.get("aggr2", "sum"),
        "use_checkpointing": bool(cfg.get("use_checkpointing", False)),
        "optimizer": cfg.get("optimizer", "Adam"),
        "lr": float(cfg.get("lr", 1e-3)),
        "weight_decay": float(cfg.get("weight_decay", 1e-6)),
        "lambda_l2_att": float(cfg.get("lambda_l2_att", 1e-4)),
        "lambda_H_mode": cfg.get("lambda_H_mode", "fixed"),
        "lambda_H_fixed": float(cfg.get("lambda_H_fixed", 1e-4)),
        "initial_lambda_H": float(cfg.get("initial_lambda_H", 1e-4)),
        "final_lambda_H": float(cfg.get("final_lambda_H", 1e-2)),
        "loss_type": cfg.get("loss_type", "CrossEntropy"),
        "focal_gamma": float(cfg.get("focal_gamma", 2.0)),
        "focal_alpha": float(cfg.get("focal_alpha", 0.75)),
        "batch_size": int(cfg.get("batch_size", 512)),
        "num_neighbors": json.dumps(cfg.get("num_neighbors", [15, 10])),
        "use_graphsmote": bool(use_graphsmote),
        "value": 0.0,
    }
    if use_graphsmote:
        params.update(
            {
                "smote_k": int(cfg.get("smote_k", 5)),
                "target_pos_ratio": float(cfg.get("target_pos_ratio", 0.35)),
                "smote_every_n_epochs": int(
                    cfg.get("smote_every_n_epochs", 2)
                ),
                "lambda_edge": float(cfg.get("lambda_edge", 1e-6)),
            }
        )
    return params


def _save_network_hparams(
    cfg: Dict[str, object], *, use_graphsmote: bool
) -> Optional[str]:
    if not cfg:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_tag = "_GraphSMOTE" if use_graphsmote else "_Base"
    path = os.path.join(
        RESULTADOS_DIR,
        f"optuna_hyperparams_network_{timestamp}{variant_tag}.csv",
    )
    try:
        params = _network_config_to_hparams(cfg, use_graphsmote=use_graphsmote)
        pd.DataFrame([params]).to_csv(path, index=False)
        return path
    except Exception:
        return None


def _extract_ts_min_from_pm_index(
    pm_index: object, num_nodes: int
) -> Optional[np.ndarray]:
    if pm_index is None:
        return None
    rev = getattr(pm_index, "_rev", None)
    if rev is None and isinstance(pm_index, dict):
        rev = pm_index
    if not isinstance(rev, dict):
        return None
    ts_arr = np.full(num_nodes, np.nan, dtype=float)
    for node_idx, value in rev.items():
        try:
            idx = int(node_idx)
        except Exception:
            continue
        if idx < 0 or idx >= num_nodes:
            continue
        ts_val = None
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            ts_val = value[1]
        elif isinstance(value, dict):
            ts_val = value.get("ts_min")
        if ts_val is None:
            continue
        try:
            ts_arr[idx] = float(ts_val)
        except Exception:
            continue
    return ts_arr


def _compute_temporal_cutoffs_from_ts(
    ts_values: np.ndarray, train_ratio: int, val_ratio: int
) -> Optional[Tuple[float, float]]:
    ts_values = np.asarray(ts_values, dtype=float)
    ts_values = ts_values[np.isfinite(ts_values)]
    if ts_values.size < 3:
        return None
    unique_ts = np.unique(ts_values)
    if unique_ts.size < 3:
        return None
    n_ts = len(unique_ts)
    raw_train_end = int((train_ratio / 100.0) * n_ts)
    raw_val_end = int(((train_ratio + val_ratio) / 100.0) * n_ts)
    train_end_idx = max(0, min(raw_train_end, n_ts - 3))
    val_end_idx = max(train_end_idx + 1, min(raw_val_end, n_ts - 2))
    return float(unique_ts[train_end_idx]), float(unique_ts[val_end_idx])


def _apply_temporal_split_to_graph(
    graph_data: HeteroData,
    pm_index: object,
    train_ratio: int,
    val_ratio: int,
) -> Optional[Dict[str, object]]:
    num_nodes = int(graph_data["pm"].num_nodes)
    ts_arr = _extract_ts_min_from_pm_index(pm_index, num_nodes)
    if ts_arr is None:
        return None
    cutoffs = _compute_temporal_cutoffs_from_ts(
        ts_arr, int(train_ratio), int(val_ratio)
    )
    if cutoffs is None:
        return None
    train_cutoff, val_cutoff = cutoffs

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    valid_mask = np.isfinite(ts_arr)
    train_mask[valid_mask] = ts_arr[valid_mask] <= train_cutoff
    val_mask[valid_mask] = (ts_arr[valid_mask] > train_cutoff) & (
        ts_arr[valid_mask] <= val_cutoff
    )
    test_mask[valid_mask] = ts_arr[valid_mask] > val_cutoff

    sequence_mask = getattr(graph_data["pm"], "sequence_mask", None)
    if sequence_mask is not None and len(sequence_mask) == num_nodes:
        seq_mask_np = sequence_mask.cpu().numpy().astype(bool)
        train_mask = train_mask & seq_mask_np
        val_mask = val_mask & seq_mask_np
        test_mask = test_mask & seq_mask_np

    is_synthetic = getattr(graph_data["pm"], "is_synthetic", None)
    if is_synthetic is not None and len(is_synthetic) == num_nodes:
        synth_mask = is_synthetic.cpu().numpy().astype(bool)
        train_mask[synth_mask] = True
        val_mask[synth_mask] = False
        test_mask[synth_mask] = False

    graph_data["pm"].train_mask = torch.from_numpy(train_mask)
    graph_data["pm"].val_mask = torch.from_numpy(val_mask)
    graph_data["pm"].test_mask = torch.from_numpy(test_mask)

    return {
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "test_count": int(test_mask.sum()),
    }


def _run_optuna_search(
    *,
    graph_obj: Dict[str, object],
    balancing_strategy: str,
    search_space: Dict[str, object],
    optuna_settings: Dict[str, object],
    objective_settings: Dict[str, object],
    log_container: Optional[object] = None, # New argument
) -> Dict[str, object]:
    
    # --- LOGGING SETUP ---
    handler = None
    loggers_to_attach = ["optuna", "src.gnn_main", "src.train_pretrain"]
    if log_container is not None:
        handler = StreamlitLogHandler(log_container)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        
        for name in loggers_to_attach:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            
    try:
        import optuna

    except Exception as exc:
        raise RuntimeError(f"Optuna no esta disponible: {exc}") from exc

    from src.gat_model import HeteroGAT
    from src.temporal_head import TemporalAggregator
    from src.train_pretrain import train_minibatch
    from src.graphsmote import (
        RelEdgeGen,
        refresh_synthetics_online,
        train_z2x_decoders,
    )
    from src import gnn_main as graph_main
    from src.imgagn import train_imgagn, ImGAGNConfig
    from torch_geometric.loader import NeighborLoader

    graph_data = graph_obj["data"]
    device = torch.device(str(objective_settings["device"]))
    # Use copy/clone if needed to avoid mutating original data across trials? 
    # Actually graph_data is reused but ImGAGN might mutate it. 
    # Safest to clone or handle carefully in trial.
    data_orig = graph_data.to(device)
    sequence_index = graph_obj.get("sequence_index")

    sampler = optuna.samplers.TPESampler(
        seed=int(optuna_settings["seed"]),
        multivariate=bool(optuna_settings["multivariate"]),
        group=bool(optuna_settings["group"]),
    )
    pruner_type = optuna_settings["pruner"]
    if pruner_type == "Hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=int(optuna_settings["min_resource"]),
            max_resource=int(optuna_settings["max_resource"]),
            reduction_factor=int(optuna_settings["reduction_factor"]),
        )
    elif pruner_type == "Median":
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=int(optuna_settings["n_warmup_steps"])
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner
    )

    num_epochs = int(optuna_settings["epochs"])
    eval_every = int(optuna_settings["eval_every"])
    objective_metric = str(objective_settings["metric"])
    threshold_beta = float(objective_settings["threshold_beta"])

    use_graphsmote = (balancing_strategy == "Opt. balance con GraphSMOTE")
    use_imgagn = (balancing_strategy == "Opt. balance con ImGAGN")

    def _score_from_metrics(
        metrics: Dict[str, float], *, fbeta: float, auprc: float, mcc: float
    ) -> float:
        if objective_metric == "F1":
            return metrics.get("f1", 0.0)
        if objective_metric == "Recall":
            return metrics.get("recall", 0.0)
        if objective_metric == "FAR":
            return 1.0 - metrics.get("far", 0.0)
        if objective_metric == "Recall-FAR":
            return metrics.get("recall", 0.0) - metrics.get("far", 0.0)
        if objective_metric == "F0.5":
            return fbeta
        if objective_metric == "AUPRC":
            return auprc
        if objective_metric == "MCC":
            return mcc
        if objective_metric == "Accuracy":
            return metrics.get("accuracy", 0.0)
        return metrics.get("f1", 0.0)

    def objective(trial: optuna.Trial) -> float:
        trial_seed = int(SEED) + trial.number
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        
        # Clone data for this trial to avoid contamination
        data = data_orig.clone()

        hidden_cfg = search_space["hidden_channels"]
        num_heads_cfg = search_space["num_heads"]
        num_layers_cfg = search_space["num_layers"]
        dropout_cfg = search_space["dropout"]
        batch_cfg = search_space["batch_size"]

        hidden_channels = trial.suggest_int(
            "hidden_channels",
            int(hidden_cfg["min"]),
            int(hidden_cfg["max"]),
            step=int(hidden_cfg["step"]),
        )
        num_heads = trial.suggest_int(
            "num_heads",
            int(num_heads_cfg["min"]),
            int(num_heads_cfg["max"]),
            step=int(num_heads_cfg["step"]),
        )
        num_layers = trial.suggest_int(
            "num_layers",
            int(num_layers_cfg["min"]),
            int(num_layers_cfg["max"]),
            step=int(num_layers_cfg["step"]),
        )
        dropout = trial.suggest_float(
            "dropout",
            float(dropout_cfg["min"]),
            float(dropout_cfg["max"]),
            step=float(dropout_cfg["step"]),
        )

        aggr1 = trial.suggest_categorical(
            "aggr1", list(search_space["aggr1"])
        )
        aggr2 = trial.suggest_categorical(
            "aggr2", list(search_space["aggr2"])
        )
        use_checkpointing_flag = trial.suggest_categorical(
            "use_checkpointing", list(search_space["use_checkpointing"])
        )

        neighbor_choice = trial.suggest_categorical(
            "num_neighbors_choice", list(search_space["neighbor_choices"])
        )
        neighbor_profile = search_space["neighbor_profiles"][neighbor_choice]

        lr_cfg = search_space["lr"]
        wd_cfg = search_space["weight_decay"]
        lambda_l2_cfg = search_space["lambda_l2_att"]

        lr = trial.suggest_float(
            "lr", float(lr_cfg["min"]), float(lr_cfg["max"]), log=True
        )
        optimizer_name = trial.suggest_categorical(
            "optimizer", list(search_space["optimizers"])
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            float(wd_cfg["min"]),
            float(wd_cfg["max"]),
            log=True,
        )
        lambda_l2_att = trial.suggest_float(
            "lambda_l2_att",
            float(lambda_l2_cfg["min"]),
            float(lambda_l2_cfg["max"]),
            log=True,
        )

        lambda_H_mode = trial.suggest_categorical(
            "lambda_H_mode", list(search_space["lambda_H_mode"])
        )
        if lambda_H_mode == "fixed":
            lambda_H_cfg = search_space["lambda_H_fixed"]
            initial_lambda_H = final_lambda_H = trial.suggest_float(
                "lambda_H_fixed",
                float(lambda_H_cfg["min"]),
                float(lambda_H_cfg["max"]),
                log=True,
            )
        else:
            lambda_init_cfg = search_space["initial_lambda_H"]
            lambda_final_cfg = search_space["final_lambda_H"]
            initial_lambda_H = trial.suggest_float(
                "initial_lambda_H",
                float(lambda_init_cfg["min"]),
                float(lambda_init_cfg["max"]),
                log=True,
            )
            final_lambda_H = trial.suggest_float(
                "final_lambda_H",
                float(lambda_final_cfg["min"]),
                float(lambda_final_cfg["max"]),
                log=True,
            )

        loss_type = trial.suggest_categorical(
            "loss_type", list(search_space["loss_type"])
        )
        focal_gamma = None
        focal_alpha = None
        if loss_type == "FocalLoss":
            focal_gamma_cfg = search_space["focal_gamma"]
            focal_gamma = trial.suggest_float(
                "focal_gamma",
                float(focal_gamma_cfg["min"]),
                float(focal_gamma_cfg["max"]),
                step=float(focal_gamma_cfg["step"]),
            )
            if not use_graphsmote:
                focal_alpha_cfg = search_space["focal_alpha"]
                focal_alpha = trial.suggest_float(
                    "focal_alpha",
                    float(focal_alpha_cfg["min"]),
                    float(focal_alpha_cfg["max"]),
                    step=float(focal_alpha_cfg["step"]),
                )

        smote_k = None
        target_pos_ratio = None
        smote_every_n_epochs = None
        lambda_edge = 0.0
        
        # --- GraphSMOTE ---
        if use_graphsmote:
            smote_k_cfg = search_space["smote_k"]
            smote_k = trial.suggest_int(
                "smote_k",
                int(smote_k_cfg["min"]),
                int(smote_k_cfg["max"]),
                step=int(smote_k_cfg["step"]),
            )
            target_pos_ratio = trial.suggest_categorical(
                "target_pos_ratio",
                list(search_space["target_pos_ratio"]),
            )
            smote_every_n_epochs = trial.suggest_categorical(
                "smote_every_n_epochs",
                list(search_space["smote_every_n_epochs"]),
            )
            lambda_edge_cfg = search_space["lambda_edge"]
            lambda_edge = trial.suggest_float(
                "lambda_edge",
                float(lambda_edge_cfg["min"]),
                float(lambda_edge_cfg["max"]),
                log=True,
            )

        # --- ImGAGN ---
        if use_imgagn:
            ig_dz = trial.suggest_int(
                "imgagn_dz", 
                int(search_space["imgagn_dz"]["min"]), 
                int(search_space["imgagn_dz"]["max"]), 
                step=int(search_space["imgagn_dz"]["step"])
            )
            ig_hidden_g = trial.suggest_int(
                "imgagn_hidden_g", 
                int(search_space["imgagn_hidden_g"]["min"]), 
                int(search_space["imgagn_hidden_g"]["max"]), 
                step=int(search_space["imgagn_hidden_g"]["step"])
            )
            ig_topk = trial.suggest_int(
                "imgagn_topk",
                int(search_space["imgagn_topk"]["min"]),
                int(search_space["imgagn_topk"]["max"]),
                step=int(search_space["imgagn_topk"]["step"])
            )
            ig_lambda1 = trial.suggest_float(
                "imgagn_lambda1",
                float(search_space["imgagn_lambda1"]["min"]),
                float(search_space["imgagn_lambda1"]["max"]),
                step=float(search_space["imgagn_lambda1"]["step"])
            )
            ig_epochs = trial.suggest_int(
                "imgagn_epochs",
                int(search_space["imgagn_epochs"]["min"]),
                int(search_space["imgagn_epochs"]["max"]),
                step=int(search_space["imgagn_epochs"]["step"])
            )
            ig_lr_g = trial.suggest_float(
                "imgagn_lr_g", 
                float(search_space["imgagn_lr_g"]["min"]), 
                float(search_space["imgagn_lr_g"]["max"]), 
                log=True
            )
            ig_lr_d = trial.suggest_float(
                "imgagn_lr_d", 
                float(search_space["imgagn_lr_d"]["min"]), 
                float(search_space["imgagn_lr_d"]["max"]), 
                log=True
            )
            ig_alpha = trial.suggest_float(
                "imgagn_alpha_reg", float(search_space["imgagn_alpha_reg"]["min"]), float(search_space["imgagn_alpha_reg"]["max"]), log=True
            )
            ig_beta = trial.suggest_float(
                "imgagn_beta_reg", float(search_space["imgagn_beta_reg"]["min"]), float(search_space["imgagn_beta_reg"]["max"]), log=True
            )
            
            # Ejecutar ImGAGN en data["pm"]
            imgagn_cfg = ImGAGNConfig(
                dz=ig_dz,
                hidden_g=ig_hidden_g,
                topk_links=ig_topk,
                lambda1_ratio=ig_lambda1,
                epochs=ig_epochs,
                lr_g=ig_lr_g,
                lr_d=ig_lr_d,
                alpha_reg=ig_alpha,
                beta_reg=ig_beta,
                device=str(device),
                show_progress=False # No llenar consola
            )
            # Asumimos que data tiene estructura para ImGAGN. 
            # ImGAGN toma y_binary.
            # Convertimos 'pm' a un grafo aumentado.
            try:
                res = train_imgagn(
                    data, 
                    train_mask=data["pm"].train_mask, 
                    y_binary=data["pm"].y, 
                    cfg=imgagn_cfg, 
                    target_ntype="pm"
                )
                if 'x_aug' in res:
                    # Reemplazar features y aristas con las aumentadas para el GNN
                    data["pm"].x = res["x_aug"].to(device)
                    # Ojo: ImGAGN devuelve aristas homogeneas. 
                    # El HeteroGAT espera edge_index_dict.
                    # Debemos asignar las aristas aumentadas a 'spatial' o similar.
                    # OJO: HeteroGAT usa edge_types. Si ImGAGN devuelve un solo edge_index,
                    # asumimos que corresponde a ('pm', 'spatial', 'pm').
                    data["pm", "spatial", "pm"].edge_index = res["edge_index_aug"].to(device)
                    
                    # Actualizar train_mask para incluir los nuevos nodos sinteticos
                    gen_slice = res["gen_slice"]
                    old_mask = data["pm"].train_mask
                    # Nueva mascara es old_mask + True para los nuevos
                    n_new = gen_slice.stop - gen_slice.start
                    new_ones = torch.ones(n_new, dtype=torch.bool, device=device)
                    data["pm"].train_mask = torch.cat([old_mask, new_ones], dim=0)
                    
                    # Ajustar otras mascaras y etiquetas
                    data["pm"].val_mask = torch.cat([data["pm"].val_mask, torch.zeros(n_new, dtype=torch.bool, device=device)], dim=0)
                    data["pm"].test_mask = torch.cat([data["pm"].test_mask, torch.zeros(n_new, dtype=torch.bool, device=device)], dim=0)
                    
                    # Las etiquetas de los nuevos nodos son 1 (minoria)
                    data["pm"].y = torch.cat([data["pm"].y, torch.ones(n_new, dtype=torch.long, device=device)], dim=0)
                    
                    # Ajustar num_nodes
                    data["pm"].num_nodes = data["pm"].x.size(0)
            except Exception as e:
                # Si falla ImGAGN, abortamos trial
                raise optuna.exceptions.TrialPruned(f"ImGAGN failed: {e}")

        batch_size_candidate = trial.suggest_int(
            "batch_size",
            int(batch_cfg["min"]),
            int(batch_cfg["max"]),
            step=int(batch_cfg["step"]),
        )
        trial.set_user_attr("num_neighbors", neighbor_profile)
        trial.set_user_attr("use_checkpointing", use_checkpointing_flag)

        num_classes = 2
        edge_feature_dim = _infer_edge_feature_dim(data)
        in_channels = data["pm"].x.shape[1]

        model = HeteroGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_heads=num_heads,
            dropout=dropout,
            edge_feature_dim=edge_feature_dim,
            num_layers=num_layers,
            aggr1=aggr1,
            aggr2=aggr2,
            use_checkpointing=use_checkpointing_flag,
        ).to(device)

        temporal_module = None
        if sequence_index and getattr(sequence_index, "sequence_rows", None) is not None:
            if sequence_index.sequence_rows.size:
                seq_rows_tensor = torch.as_tensor(
                    sequence_index.sequence_rows, dtype=torch.long
                )
                target_rows_tensor = torch.as_tensor(
                    sequence_index.target_rows, dtype=torch.long
                )
                temporal_module = TemporalAggregator(
                    sequence_rows=seq_rows_tensor,
                    target_rows=target_rows_tensor,
                    num_nodes=data["pm"].num_nodes,
                    embedding_dim=hidden_channels * num_heads,
                    sequence_length=sequence_index.sequence_rows.shape[1],
                    hidden_dim=hidden_channels * num_heads,
                    num_classes=num_classes,
                ).to(device)
                temporal_module.reset_cache()
                temporal_module.train()
                model.temporal_head = temporal_module

        edge_gen = (
            RelEdgeGen(
                {ntype: hidden_channels * num_heads for ntype in data.node_types},
                data.edge_types,
            ).to(device)
            if use_graphsmote
            else None
        )

        optimizer = getattr(torch.optim, optimizer_name)(
            list(model.parameters())
            + (list(edge_gen.parameters()) if edge_gen else []),
            lr=lr,
            weight_decay=weight_decay,
        )

        if loss_type == "FocalLoss":
            alpha_val = None
            if not use_graphsmote and focal_alpha is not None:
                alpha_val = [1 - focal_alpha, focal_alpha]
            criterion = graph_main.FocalLoss(
                gamma=float(focal_gamma or 2.0), alpha=alpha_val
            )
        else:
            weight = None
            if not use_graphsmote:
                y_train = data["pm"].y[data["pm"].train_mask]
                counts = torch.bincount(y_train)
                if counts.numel() > 1:
                    weight = (1.0 / counts.float()).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

        base_graph = data
        train_graph = base_graph
        z2x_decoders = (
            train_z2x_decoders(
                model, base_graph, device=device, epochs=DECODER_EPOCHS
            )
            if use_graphsmote
            else None
        )

        num_neighbors_dict = {
            edge_type: neighbor_profile for edge_type in train_graph.edge_types
        }
        train_loader = NeighborLoader(
            train_graph.cpu(),
            input_nodes=("pm", train_graph["pm"].train_mask),
            num_neighbors=num_neighbors_dict,
            batch_size=batch_size_candidate,
            shuffle=True,
        )

        best_score = -1e9
        best_tau = 0.5
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            if use_graphsmote and smote_every_n_epochs:
                if epoch % int(smote_every_n_epochs) == 0:
                    aug_data, _ = refresh_synthetics_online(
                        model,
                        base_graph,
                        device,
                        target_pos_ratio=float(target_pos_ratio),
                        k=int(smote_k),
                        z2x_decoders=z2x_decoders,
                        edge_gen=edge_gen,
                        seed=trial_seed + epoch,
                    )
                    train_graph = aug_data.to(device)
                    train_loader = NeighborLoader(
                        train_graph.cpu(),
                        input_nodes=("pm", train_graph["pm"].train_mask),
                        num_neighbors={
                            edge_type: neighbor_profile
                            for edge_type in train_graph.edge_types
                        },
                        batch_size=batch_size_candidate,
                        shuffle=True,
                    )

            if lambda_H_mode == "fixed":
                current_lambda_H = float(initial_lambda_H)
            else:
                current_lambda_H = float(
                    initial_lambda_H
                    + (final_lambda_H - initial_lambda_H)
                    * (epoch - 1)
                    / max(num_epochs - 1, 1)
                )

            train_minibatch(
                model,
                train_loader,
                optimizer,
                criterion,
                grad_clip_value=1.0,
                device=device,
                use_amp=False,
                scaler=None,
                scheduler=None,
                writer=None,
                epoch=epoch,
                lambda_H=current_lambda_H,
                node_type="pm",
                edge_gen=edge_gen,
                lambda_edge=float(lambda_edge),
                lambda_l2_att=float(lambda_l2_att),
            )

            if epoch % eval_every != 0:
                continue

            val_results = graph_main.test(
                model,
                base_graph,
                node_type="pm",
                batch_size=batch_size_candidate,
            )
            if temporal_module is not None:
                temporal_module.train()
            if not val_results or "val_mask" not in val_results:
                continue

            y_true_val = val_results["val_mask"]["true"].numpy().ravel()
            y_prob1_val = (
                val_results["val_mask"]["probs"][:, 1].numpy().ravel()
            )
            if np.unique(y_true_val).size < 2:
                continue

            tau, _, _, fbeta = graph_main.pick_tau_fbeta(
                y_true_val, y_prob1_val, beta=threshold_beta
            )
            preds = (y_prob1_val >= tau).astype(int)
            cm = np.array(
                graph_main.confusion_matrix(y_true_val, preds, labels=[0, 1])
            )
            metrics = _compute_binary_metrics_from_cm(cm)
            auprc = float(val_results["val_mask"].get("auprc") or 0.0)
            mcc = float(val_results["val_mask"].get("mcc") or 0.0)

            score = _score_from_metrics(
                metrics, fbeta=float(fbeta), auprc=auprc, mcc=mcc
            )
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if score > best_score:
                best_score = score
                best_tau = tau
                best_epoch = epoch

        trial.set_user_attr("best_metric", best_score)
        trial.set_user_attr("best_tau", float(best_tau))
        trial.set_user_attr("best_epoch", int(best_epoch))
        return float(best_score)

    try:
        study.optimize(
            objective,
            n_trials=int(optuna_settings["n_trials"]),
            show_progress_bar=True,
        )
    finally:
        # CLEANUP HANDLER
        if handler:
            for name in loggers_to_attach:
                logger = logging.getLogger(name)
                logger.removeHandler(handler)
        
        import gc

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = study.best_trial
    best_params = best.params.copy()
    best_params["value"] = best.value
    best_params["objective_metric"] = objective_metric
    for key, value in best.user_attrs.items():
        best_params[key] = value
    best_params["use_graphsmote"] = use_graphsmote

    try:
        best_params["use_imgagn_aug"] = bool(graph_main._has_imgagn(graph_obj))
    except Exception:
        best_params["use_imgagn_aug"] = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_hash = graph_main._calculate_graph_hash(graph_obj.get("filename"))
    variant_tag = graph_main._variant_tags(use_graphsmote, graph_obj)
    hash_tag = f"_{graph_hash[:16]}" if graph_hash else ""
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    best_params_path = os.path.join(
        RESULTADOS_DIR,
        f"optuna_hyperparams_{timestamp}{hash_tag}{variant_tag}.csv",
    )
    pd.DataFrame([best_params]).to_csv(best_params_path, index=False)

    trials_df = study.trials_dataframe()
    trials_df["use_graphsmote"] = use_graphsmote
    trials_df["graph_hash"] = graph_hash
    full_study_path = os.path.join(
        RESULTADOS_DIR,
        f"optuna_full_study_{timestamp}{hash_tag}{variant_tag}.csv",
    )
    trials_df.to_csv(full_study_path, index=False)

    return {
        "best_params": best_params,
        "best_path": best_params_path,
        "full_path": full_study_path,
        "study": study,
    }


def _feature_selection_key(
    features_path: Optional[str],
    features_source: Optional[str],
    features_df: pd.DataFrame,
) -> str:
    if features_path:
        try:
            return str(Path(features_path).resolve())
        except Exception:
            return str(features_path)
    source = features_source or "memory"
    return f"{source}:{len(features_df)}:{len(features_df.columns)}"


def _feature_selection_id(
    features_path: Optional[str],
    features_source: Optional[str],
    features_df: pd.DataFrame,
) -> str:
    if features_path:
        base = Path(features_path).stem
    else:
        source = features_source or "memory"
        base = f"features_{source}_{len(features_df)}_{len(features_df.columns)}"
    return _slugify(base)


def _feature_selection_paths(feature_id: str) -> Tuple[Path, Path]:
    json_path = FEATURE_SELECTION_DIR / f"feature_selection_{feature_id}.json"
    csv_path = (
        FEATURE_SELECTION_DIR / f"feature_selection_{feature_id}_importance.csv"
    )
    return json_path, csv_path


def _load_feature_selection_from_disk(
    feature_id: str,
) -> Tuple[Optional[Dict[str, object]], Optional[pd.DataFrame]]:
    json_path, csv_path = _feature_selection_paths(feature_id)
    payload: Optional[Dict[str, object]] = None
    importance_df: Optional[pd.DataFrame] = None
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = None
    if csv_path.exists():
        try:
            importance_df = pd.read_csv(csv_path)
        except Exception:
            importance_df = None
    return payload, importance_df


def _persist_feature_selection(
    *,
    feature_key: str,
    feature_id: str,
    features_path: Optional[str],
    features_source: Optional[str],
    features_df: pd.DataFrame,
    selected_features: List[str],
    importance_df: Optional[pd.DataFrame],
    params: Dict[str, object],
) -> None:
    store = st.session_state.setdefault("feature_selection_store", {})
    prev = store.get(feature_key, {})
    prev_selected = prev.get("selected_features")
    prev_hash = prev.get("importance_hash")
    prev_importance = prev.get("importance_df")

    importance_hash = prev_hash
    if importance_df is None:
        importance_df = prev_importance
    elif importance_df is not None and not importance_df.empty:
        try:
            importance_hash = int(
                pd.util.hash_pandas_object(importance_df, index=True).sum()
            )
        except Exception:
            importance_hash = None

    entry = {
        "feature_id": feature_id,
        "features_path": features_path,
        "features_source": features_source,
        "features_rows": int(len(features_df)),
        "features_cols": int(len(features_df.columns)),
        "selected_features": list(selected_features),
        "importance_df": importance_df,
        "importance_hash": importance_hash,
        "params": dict(params),
        "saved_at": datetime.now().isoformat(),
    }
    store[feature_key] = entry

    selected_changed = prev_selected != selected_features
    importance_changed = (
        importance_df is not None and importance_hash != prev_hash
    )
    if not (selected_changed or importance_changed or prev == {}):
        return

    FEATURE_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = _feature_selection_paths(feature_id)
    payload = {
        "feature_key": feature_key,
        "feature_id": feature_id,
        "features_path": features_path,
        "features_source": features_source,
        "features_rows": int(len(features_df)),
        "features_cols": int(len(features_df.columns)),
        "selected_features": list(selected_features),
        "params": dict(params),
        "saved_at": datetime.now().isoformat(),
        "importance_csv": None,
    }
    if importance_df is not None and not importance_df.empty:
        try:
            importance_df.to_csv(csv_path, index=False)
            payload["importance_csv"] = str(csv_path)
        except Exception:
            payload["importance_csv"] = None
    else:
        if csv_path.exists():
            payload["importance_csv"] = str(csv_path)
    try:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
    except Exception:
        return


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {
        "target",
        "synthetic",
        "portico",
        "ts_min",
        "timestamp",
        "interval_start",
        "node_idx",
        "next_ts_min",
        "future_label",
    }
    return [
        col
        for col in df.columns
        if col not in exclude
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def _parse_neighbor_profile(
    text_value: str,
    *,
    default: Optional[List[int]] = None,
    min_value: int = 1,
) -> List[int]:
    if default is None:
        default = [15, 10]
    if not text_value:
        return default
    parts = re.split(r"[,\s]+", text_value.strip())
    values = []
    for part in parts:
        if not part:
            continue
        try:
            val = int(part)
        except ValueError:
            continue
        if val < min_value:
            continue
        values.append(val)
    return values or default


def _edge_attr_dict_from_data(
    data: HeteroData,
) -> Dict[Tuple[str, str, str], Optional[torch.Tensor]]:
    edge_attr_dict: Dict[Tuple[str, str, str], Optional[torch.Tensor]] = {}
    for edge_type in data.edge_types:
        store = data[edge_type]
        edge_attr_dict[edge_type] = (
            store.edge_attr if hasattr(store, "edge_attr") else None
        )
    return edge_attr_dict


def _run_gnn_gradient_importance(
    *,
    graph_data: HeteroData,
    feature_names: Sequence[str],
    device: torch.device,
    hidden_channels: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    neighbor_profile: Sequence[int],
    target_class: int,
    target_mode: str,
    attribution_method: str,
    ig_steps: int,
    max_train_batches: int,
    max_attr_batches: int,
    seed: int,
) -> pd.DataFrame:
    from src.gat_model import HeteroGAT
    from torch_geometric.loader import NeighborLoader

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    data_cpu = graph_data.cpu()
    if "pm" not in data_cpu.node_types:
        raise ValueError("El grafo no contiene nodos 'pm'.")
    if not hasattr(data_cpu["pm"], "y"):
        raise ValueError("El grafo no contiene etiquetas 'y'.")
    num_classes = int(torch.unique(data_cpu["pm"].y).numel())
    if num_classes < 2:
        raise ValueError(
            "No hay dos clases en el target para calcular importancia GNN."
        )

    num_features = int(data_cpu["pm"].x.shape[1])
    if len(feature_names) != num_features:
        feature_names = [f"feat_{idx}" for idx in range(num_features)]

    train_mask = (
        data_cpu["pm"].train_mask
        if hasattr(data_cpu["pm"], "train_mask")
        else torch.ones(data_cpu["pm"].num_nodes, dtype=torch.bool)
    )
    if train_mask.sum() == 0:
        raise ValueError("No hay nodos de entrenamiento disponibles.")

    neighbor_profile = list(neighbor_profile)
    if len(neighbor_profile) < num_layers:
        neighbor_profile = neighbor_profile + [neighbor_profile[-1]] * (
            num_layers - len(neighbor_profile)
        )
    num_neighbors_dict = {
        edge_type: neighbor_profile for edge_type in data_cpu.edge_types
    }

    edge_feature_dim = _infer_edge_feature_dim(graph_data)
    model = HeteroGAT(
        in_channels=num_features,
        hidden_channels=int(hidden_channels),
        out_channels=num_classes,
        num_heads=int(num_heads),
        dropout=float(dropout),
        edge_feature_dim=int(edge_feature_dim),
        num_layers=int(num_layers),
        aggr1="sum",
        aggr2="sum",
        use_checkpointing=False,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = NeighborLoader(
        data_cpu,
        input_nodes=("pm", train_mask),
        num_neighbors=num_neighbors_dict,
        batch_size=int(batch_size),
        shuffle=True,
    )

    model.train()
    for epoch in range(int(epochs)):
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            if max_train_batches and batch_count > max_train_batches:
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            edge_attr_dict = _edge_attr_dict_from_data(batch)
            out_dict, _, _ = model(
                batch.x_dict, batch.edge_index_dict, edge_attr_dict
            )
            logits = out_dict["pm"]
            batch_size_eff = int(
                getattr(batch["pm"], "batch_size", logits.size(0))
            )
            logits = logits[:batch_size_eff]
            y_true = batch["pm"].y[:batch_size_eff]
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()

    attr_loader = NeighborLoader(
        data_cpu,
        input_nodes=("pm", train_mask),
        num_neighbors=num_neighbors_dict,
        batch_size=int(batch_size),
        shuffle=False,
    )

    model.eval()
    importance_sum = torch.zeros(num_features, device=device)
    total_nodes = 0

    for batch_idx, batch in enumerate(attr_loader, start=1):
        if max_attr_batches and batch_idx > max_attr_batches:
            break
        batch = batch.to(device)
        edge_attr_dict = _edge_attr_dict_from_data(batch)
        base_x = batch["pm"].x
        batch_size_eff = int(
            getattr(batch["pm"], "batch_size", base_x.size(0))
        )
        y_true = batch["pm"].y[:batch_size_eff]
        if target_mode == "true":
            target = y_true
        else:
            target = torch.full(
                (batch_size_eff,),
                int(target_class),
                dtype=torch.long,
                device=device,
            )

        if attribution_method == "ig":
            baseline = torch.zeros_like(base_x)
            total_grad = torch.zeros_like(base_x)
            for step in range(1, int(ig_steps) + 1):
                alpha = float(step) / float(ig_steps)
                x_interp = baseline + alpha * (base_x - baseline)
                x_interp = x_interp.detach().requires_grad_(True)
                batch["pm"].x = x_interp
                out_dict, _, _ = model(
                    batch.x_dict, batch.edge_index_dict, edge_attr_dict
                )
                logits = out_dict["pm"][:batch_size_eff]
                score = logits.gather(1, target.unsqueeze(1)).sum()
                model.zero_grad(set_to_none=True)
                if x_interp.grad is not None:
                    x_interp.grad.zero_()
                score.backward()
                total_grad += x_interp.grad.detach()
            avg_grad = total_grad / float(ig_steps)
            attr = (base_x - baseline) * avg_grad
        else:
            x_req = base_x.detach().requires_grad_(True)
            batch["pm"].x = x_req
            out_dict, _, _ = model(
                batch.x_dict, batch.edge_index_dict, edge_attr_dict
            )
            logits = out_dict["pm"][:batch_size_eff]
            score = logits.gather(1, target.unsqueeze(1)).sum()
            model.zero_grad(set_to_none=True)
            if x_req.grad is not None:
                x_req.grad.zero_()
            score.backward()
            attr = x_req.grad * x_req

        attr = attr[:batch_size_eff]
        importance_sum += attr.abs().sum(dim=0)
        total_nodes += batch_size_eff

    if total_nodes == 0:
        raise ValueError("No se pudieron calcular atribuciones.")

    importance = (importance_sum / float(total_nodes)).detach().cpu().numpy()
    importance_df = pd.DataFrame(
        {"variable": list(feature_names), "importance": importance}
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


def _get_cluster_cols(df: pd.DataFrame) -> List[str]:
    cluster_prefixes = (
        "cluster_share_",
        "cluster_flow_",
        "cluster_count_",
        "cluster_speed_",
        "cluster_density_",
        "cluster_delta_speed_",
        "cluster_delta_density_",
        "cluster_entropy",
    )
    valid_cols = []
    for col in df.columns:
        if col.startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
        if col.startswith("last_") and col[5:].startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
        if col.startswith("next_") and col[5:].startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
    return valid_cols


def _is_snapshot_features(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    if "portico" not in df.columns or "ts_min" not in df.columns:
        return False
    columns = set(df.columns)
    if any(col.startswith("speed_") for col in columns):
        return True
    if any(col.endswith(SNAPSHOT_WINDOW_SUFFIXES) for col in columns):
        return True
    if any(col.endswith(SNAPSHOT_GRADIENT_SUFFIXES) for col in columns):
        return True
    if any(col in columns for col in SNAPSHOT_FLOW_COLUMNS):
        return True
    if any(col in columns for col in SNAPSHOT_TEMPORAL_COLUMNS):
        return True
    return False


def _is_empty_frame(obj: object) -> bool:
    if obj is None:
        return True
    if isinstance(obj, pl.LazyFrame):
        return obj.collect().is_empty()
    if isinstance(obj, pl.DataFrame):
        return obj.is_empty()
    if isinstance(obj, pl.Series):
        return obj.is_empty()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.empty
    if hasattr(obj, "empty"):
        return bool(getattr(obj, "empty"))
    return False


def _debug_cluster(enabled: bool, message: str) -> None:
    if enabled:
        st.caption(f"[cluster-debug] {message}")


def _debug_labels(enabled: bool, message: str) -> None:
    if enabled:
        st.caption(f"[label-debug] {message}")


def _frame_debug_info(obj: object, name: str) -> str:
    if obj is None:
        return f"{name}: None"
    obj_type = type(obj).__name__
    shape = None
    cols = None
    try:
        if isinstance(obj, pl.LazyFrame):
            shape = obj.collect().shape
            cols = obj.columns
        elif isinstance(obj, pl.DataFrame):
            shape = obj.shape
            cols = obj.columns
        elif isinstance(obj, pl.Series):
            shape = (len(obj),)
        elif isinstance(obj, pd.DataFrame):
            shape = obj.shape
            cols = list(obj.columns)
        elif isinstance(obj, pd.Series):
            shape = obj.shape
        else:
            shape = getattr(obj, "shape", None)
            cols = getattr(obj, "columns", None)
    except Exception as exc:
        return f"{name}: type={obj_type} (shape/cols error: {exc})"
    cols_text = ""
    if cols is not None:
        cols_list = list(cols)
        preview = ", ".join(str(c) for c in cols_list[:6])
        suffix = f", ... (+{len(cols_list) - 6})" if len(cols_list) > 6 else ""
        cols_text = f", cols=[{preview}{suffix}]"
    return f"{name}: type={obj_type}, shape={shape}{cols_text}"


def _merge_on_portico_ts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    debug: bool = False,
    label: str = "merge",
    dt_minutes: Optional[int] = None,
) -> pd.DataFrame:
    work_left = left.copy()
    work_right = right.copy()
    if dt_minutes is not None:
        dt_minutes = int(dt_minutes)
    if "ts_min" in work_left.columns:
        work_left["ts_min"] = pd.to_numeric(
            work_left["ts_min"], errors="coerce"
        )
        if dt_minutes:
            work_left["ts_min"] = (
                (work_left["ts_min"] // dt_minutes) * dt_minutes
            )
        work_left["ts_min"] = work_left["ts_min"].astype("Int64")
    if "ts_min" in work_right.columns:
        work_right["ts_min"] = pd.to_numeric(
            work_right["ts_min"], errors="coerce"
        )
        if dt_minutes:
            work_right["ts_min"] = (
                (work_right["ts_min"] // dt_minutes) * dt_minutes
            )
        work_right["ts_min"] = work_right["ts_min"].astype("Int64")
    if "portico" in work_left.columns:
        work_left["portico"] = _normalize_portico_series(
            work_left["portico"]
        )
    if "portico" in work_right.columns:
        work_right["portico"] = _normalize_portico_series(
            work_right["portico"]
        )
    work_left["portico_key"] = pd.to_numeric(
        work_left["portico"], errors="coerce"
    )
    work_right["portico_key"] = pd.to_numeric(
        work_right["portico"], errors="coerce"
    )
    use_numeric = (
        work_left["portico_key"].notna().any()
        and work_right["portico_key"].notna().any()
    )
    if debug:
        left_range = (
            (work_left["ts_min"].min(), work_left["ts_min"].max())
            if "ts_min" in work_left.columns
            else (None, None)
        )
        right_range = (
            (work_right["ts_min"].min(), work_right["ts_min"].max())
            if "ts_min" in work_right.columns
            else (None, None)
        )
        _debug_cluster(
            debug,
            f"{label}: ts_min left={left_range} right={right_range}",
        )
        if use_numeric:
            left_keys = (
                work_left[["portico_key", "ts_min"]]
                .dropna()
                .drop_duplicates()
            )
            right_keys = (
                work_right[["portico_key", "ts_min"]]
                .dropna()
                .drop_duplicates()
            )
            match = left_keys.merge(
                right_keys, on=["portico_key", "ts_min"], how="inner"
            )
        else:
            left_keys = (
                work_left[["portico", "ts_min"]]
                .dropna()
                .drop_duplicates()
            )
            right_keys = (
                work_right[["portico", "ts_min"]]
                .dropna()
                .drop_duplicates()
            )
            match = left_keys.merge(
                right_keys, on=["portico", "ts_min"], how="inner"
            )
        _debug_cluster(
            debug,
            f"{label}: left={len(left_keys)} right={len(right_keys)} match={len(match)}",
        )
    right_value_cols = [
        col
        for col in work_right.columns
        if col not in {"portico", "ts_min", "portico_key"}
    ]
    if use_numeric:
        merged = work_left.merge(
            work_right.drop(columns=["portico"], errors="ignore"),
            left_on=["portico_key", "ts_min"],
            right_on=["portico_key", "ts_min"],
            how="left",
        )
    else:
        merged = work_left.merge(
            work_right,
            on=["portico", "ts_min"],
            how="left",
        )
    if right_value_cols and merged[right_value_cols].notna().any().any():
        return merged.drop(columns=["portico_key"], errors="ignore")

    if use_numeric:
        merged = work_left.merge(
            work_right,
            on=["portico", "ts_min"],
            how="left",
        )
    return merged.drop(columns=["portico_key"], errors="ignore")


def _compute_puntos_negros_report(
    accidents_df: pd.DataFrame, porticos_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    if accidents_df is None or accidents_df.empty:
        return None
    if porticos_df is None or porticos_df.empty:
        return None

    df_port = porticos_df.copy()
    df_port["portico_clean"] = _normalize_portico_series(df_port["portico"])
    df_port["orden_num"] = pd.to_numeric(
        df_port.get("orden"), errors="coerce"
    )
    df_port["calzada_norm"] = (
        df_port["calzada"].astype(str).str.strip().str.upper()
    )
    df_port["eje_norm"] = (
        df_port["eje"].astype(str).str.strip().str.upper()
    )

    acc_eje_col = _find_match_column(accidents_df, ["Eje"])
    acc_calzada_col = _find_match_column(accidents_df, ["Calzada"])
    acc_autopista_col = _find_match_column(accidents_df, ["Autopista"])
    use_context = bool(acc_eje_col and acc_calzada_col)

    group_cols = ["eje_norm", "calzada_norm"]
    if (
        acc_autopista_col
        and "autopista" in df_port.columns
    ):
        df_port["autopista_norm"] = (
            df_port["autopista"].astype(str).str.strip().str.upper()
        )
        group_cols = ["autopista_norm"] + group_cols

    df_port = df_port.dropna(
        subset=group_cols + ["orden_num", "portico_clean"]
    )
    df_port = df_port.sort_values(group_cols + ["orden_num"])

    if use_context:
        next_map: Dict[tuple, str] = {}
        for _, group in df_port.groupby(group_cols):
            porticos = group["portico_clean"].tolist()
            for idx in range(len(porticos) - 1):
                if "autopista_norm" in group.columns:
                    map_key = (
                        group["autopista_norm"].iloc[idx],
                        group["eje_norm"].iloc[idx],
                        group["calzada_norm"].iloc[idx],
                        porticos[idx],
                    )
                else:
                    map_key = (
                        group["eje_norm"].iloc[idx],
                        group["calzada_norm"].iloc[idx],
                        porticos[idx],
                    )
                next_map[map_key] = porticos[idx + 1]
        if not next_map:
            return None
    else:
        next_map_simple: Dict[str, str] = {}
        for _, group in df_port.groupby(group_cols):
            porticos = group["portico_clean"].tolist()
            for idx in range(len(porticos) - 1):
                next_map_simple[porticos[idx]] = porticos[idx + 1]
        if not next_map_simple:
            return None

    df_acc = accidents_df.copy()
    if "accidente_time" not in df_acc.columns:
        return None
    df_acc["ultimo_portico"] = _normalize_portico_series(
        df_acc["ultimo_portico"]
    )
    df_acc["accidente_time"] = pd.to_datetime(
        df_acc["accidente_time"], errors="coerce"
    )
    df_acc = df_acc.dropna(subset=["ultimo_portico", "accidente_time"])
    if use_context:
        df_acc["eje_norm"] = (
            df_acc[acc_eje_col].astype(str).str.strip().str.upper()
        )
        df_acc["calzada_norm"] = (
            df_acc[acc_calzada_col].astype(str).str.strip().str.upper()
        )
        if acc_autopista_col and "autopista_norm" in df_port.columns:
            df_acc["autopista_norm"] = (
                df_acc[acc_autopista_col]
                .astype(str)
                .str.strip()
                .str.upper()
            )
            df_acc["map_key"] = list(
                zip(
                    df_acc["autopista_norm"],
                    df_acc["eje_norm"],
                    df_acc["calzada_norm"],
                    df_acc["ultimo_portico"],
                )
            )
        else:
            df_acc["map_key"] = list(
                zip(
                    df_acc["eje_norm"],
                    df_acc["calzada_norm"],
                    df_acc["ultimo_portico"],
                )
            )
        df_acc["siguiente_portico"] = df_acc["map_key"].map(next_map)
    else:
        df_acc["siguiente_portico"] = df_acc["ultimo_portico"].map(
            next_map_simple
        )
    df_acc = df_acc.dropna(subset=["siguiente_portico"])
    if df_acc.empty:
        return None

    df_acc["tramo"] = (
        df_acc["ultimo_portico"].astype(str)
        + " -> "
        + df_acc["siguiente_portico"].astype(str)
    )
    hours = df_acc["accidente_time"].dt.hour
    conditions = [
        (hours >= 6) & (hours < 10),
        (hours >= 18) & (hours < 22),
    ]
    choices = ["Mañana (06-10h)", "Tarde (18-22h)"]
    df_acc["franja_horaria"] = np.select(
        conditions, choices, default="Resto del día"
    )

    stats = (
        df_acc.groupby(["tramo", "franja_horaria"])
        .size()
        .reset_index(name="n_accidentes")
    )
    if stats.empty:
        return None

    horas_map = {
        "Mañana (06-10h)": 4.0,
        "Tarde (18-22h)": 4.0,
        "Resto del día": 16.0,
    }
    stats["horas_franja"] = stats["franja_horaria"].map(horas_map)
    stats["accidentes_por_hora"] = (
        stats["n_accidentes"]
        / stats["horas_franja"].replace({0.0: np.nan})
    )
    stats = stats.sort_values(
        by=["accidentes_por_hora", "n_accidentes", "tramo"],
        ascending=[False, False, True],
    )
    return stats.reset_index(drop=True)


def _infer_flow_columns(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    timestamp_col = _find_column_by_keywords(
        df, ["fecha", "timestamp", "datetime", "time"]
    )
    portico_col = _find_column_by_keywords(df, ["portico", "cod_portico"])
    plate_col = _find_column_by_keywords(
        df, ["matricula", "patente", "placa", "plate"]
    )
    speed_col = _find_column_by_keywords(df, ["velocidad", "speed"])
    if not timestamp_col or not portico_col or not plate_col:
        return None
    return {
        "timestamp": timestamp_col,
        "portico": portico_col,
        "plate": plate_col,
        "speed": speed_col or "VELOCIDAD",
    }


def _prepare_cluster_features_df(
    df: pd.DataFrame,
    portico_col: str,
    time_col: str,
    dt_minutes: int,
) -> pd.DataFrame:
    cluster_df = df.copy()
    if portico_col != "portico":
        cluster_df = cluster_df.rename(columns={portico_col: "portico"})
    cluster_df["portico"] = _normalize_portico_series(cluster_df["portico"])
    time_key = _normalize_column_key(time_col)
    if time_key == "tsmin":
        cluster_df["ts_min"] = pd.to_numeric(
            cluster_df[time_col], errors="coerce"
        )
    else:
        ts = pd.to_datetime(cluster_df[time_col], errors="coerce")
        ts = ts.astype("datetime64[ns]")
        cluster_df["ts_min"] = (ts.astype("int64") // 60_000_000_000)
        cluster_df["ts_min"] = (
            (cluster_df["ts_min"] // int(dt_minutes)) * int(dt_minutes)
        )
    if time_col != "ts_min":
        cluster_df = cluster_df.drop(columns=[time_col], errors="ignore")
    cluster_df = cluster_df.dropna(subset=["portico", "ts_min"])
    cluster_df["ts_min"] = cluster_df["ts_min"].astype(int)
    return cluster_df


def _filter_snapshot_chunk(
    df: Optional[pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    if df is None or _is_empty_frame(df):
        return df
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        mask = (ts >= start_ts) & (ts < end_ts)
        filtered = df.loc[mask].copy()
        filtered = filtered.drop(columns=["timestamp"], errors="ignore")
        _debug_cluster(
            debug,
            f"snapshot filter by timestamp: before={len(df)} after={len(filtered)} "
            f"start={start_ts} end={end_ts}",
        )
        return filtered
    start_min = int(pd.Timestamp(start_ts).value // 60_000_000_000)
    end_min = int(pd.Timestamp(end_ts).value // 60_000_000_000)
    if "ts_min" not in df.columns:
        return df
    filtered = df.loc[
        (df["ts_min"] >= start_min) & (df["ts_min"] < end_min)
    ].copy()
    _debug_cluster(
        debug,
        f"snapshot filter by ts_min: before={len(df)} after={len(filtered)} "
        f"start_min={start_min} end_min={end_min}",
    )
    return filtered


def _filter_cluster_chunk(
    df: Optional[pd.DataFrame],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    if df is None or _is_empty_frame(df):
        return df
    if "interval_start" in df.columns:
        interval = pd.to_datetime(df["interval_start"], errors="coerce")
        return df.loc[
            (interval >= start_ts) & (interval < end_ts)
        ].copy()
    if "ts_min" in df.columns:
        start_min = int(pd.Timestamp(start_ts).value // 60_000_000_000)
        end_min = int(pd.Timestamp(end_ts).value // 60_000_000_000)
        return df.loc[
            (df["ts_min"] >= start_min) & (df["ts_min"] < end_min)
        ].copy()
    return df


def _compute_cluster_features_from_flows(
    flows_df: pd.DataFrame,
    cluster_labels_df: pd.DataFrame,
    *,
    dt_minutes: int,
    lanes: int,
    include_counts: bool = False,
    include_entropy: bool = False,
    include_speed: bool = False,
    include_density: bool = False,
    include_delta_speed: bool = False,
    include_delta_density: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    from src.utils import compute_cluster_features, normalize_plate_series

    col_map = _infer_flow_columns(flows_df)
    _debug_cluster(debug, _frame_debug_info(flows_df, "flows_df"))
    _debug_cluster(debug, _frame_debug_info(cluster_labels_df, "cluster_labels_df"))
    if col_map is None:
        raise ValueError(
            "No se encontraron columnas de flujo requeridas (fecha/portico/matricula)."
        )
    _debug_cluster(debug, f"col_map={col_map}")
    labels_df = cluster_labels_df
    if isinstance(flows_df, pl.LazyFrame):
        flows_df = flows_df.collect()

    if isinstance(flows_df, pl.DataFrame):
        plate_col = col_map["plate"]
        if plate_col in flows_df.columns:
            plate_series = (
                flows_df.select(
                    pl.col(plate_col)
                    .cast(pl.Utf8, strict=False)
                    .str.strip_chars()
                    .str.to_uppercase()
                    .alias("plate_clean")
                )
                .get_column("plate_clean")
                .drop_nulls()
                .unique()
            )
            _debug_cluster(
                debug,
                f"plate_series (polars): count={len(plate_series)}",
            )
            if plate_series.is_empty():
                return pd.DataFrame()
            if "plate" in labels_df.columns:
                labels_df = labels_df[
                    labels_df["plate"].isin(plate_series.to_list())
                ]
                if _is_empty_frame(labels_df):
                    return pd.DataFrame()
    else:
        plate_series = normalize_plate_series(flows_df[col_map["plate"]])
        plate_series = plate_series.dropna()
        _debug_cluster(
            debug,
            f"plate_series (pandas): count={len(plate_series)}",
        )
        if _is_empty_frame(plate_series):
            return pd.DataFrame()
        plate_set = set(plate_series.unique())
        if "plate" in labels_df.columns:
            labels_df = labels_df[labels_df["plate"].isin(plate_set)]
            if _is_empty_frame(labels_df):
                return pd.DataFrame()
    _debug_cluster(debug, _frame_debug_info(labels_df, "labels_df_filtered"))
    return compute_cluster_features(
        flows_df,
        labels_df,
        interval_minutes=int(dt_minutes),
        lanes=int(lanes),
        timestamp_col=col_map["timestamp"],
        portico_col=col_map["portico"],
        plate_col_flow=col_map["plate"],
        speed_col=col_map["speed"],
        include_counts=include_counts,
        include_entropy=include_entropy,
        include_speed=include_speed,
        include_density=include_density,
        include_delta_speed=include_delta_speed,
        include_delta_density=include_delta_density,
    )

def _build_batch_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    mode: str,
) -> list:
    if mode not in {"month", "week"}:
        raise ValueError("mode must be 'month' or 'week'")

    if mode == "month":
        range_start = start_ts.to_period("M").start_time
        range_end = (end_ts + pd.offsets.MonthBegin(1))
        boundaries = pd.date_range(start=range_start, end=range_end, freq="MS")
        ranges = []
        for i in range(len(boundaries) - 1):
            chunk_start = max(start_ts, boundaries[i])
            chunk_end = min(end_ts, boundaries[i + 1])
            if chunk_start >= chunk_end:
                continue
            ranges.append(
                (chunk_start, chunk_end, boundaries[i].strftime("%Y-%m"))
            )
        return ranges

    # Week
    range_start = start_ts.normalize() - pd.Timedelta(days=start_ts.weekday())
    range_end = end_ts.normalize() + pd.Timedelta(days=7)
    boundaries = pd.date_range(start=range_start, end=range_end, freq="7D")
    ranges = []
    for i in range(len(boundaries) - 1):
        chunk_start = max(start_ts, boundaries[i])
        chunk_end = min(end_ts, boundaries[i + 1])
        if chunk_start >= chunk_end:
            continue
        label = (
            f"{chunk_start:%Y-%m-%d}_to_"
            f"{(chunk_end - pd.Timedelta(days=1)):%Y-%m-%d}"
        )
        ranges.append((chunk_start, chunk_end, label))
    return ranges


def _pick_duckdb_table(
    tables: Sequence[str],
    preferred: Sequence[str],
) -> Optional[str]:
    for name in preferred:
        if name in tables:
            return name
    return tables[0] if tables else None


def _load_duckdb_df(path: str, preferred_tables: Sequence[str]) -> Optional[pd.DataFrame]:
    if duckdb is None:
        return None
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        table_name = _pick_duckdb_table(tables, preferred_tables)
        if not table_name:
            return None
        return con.execute(f"SELECT * FROM {table_name}").df()
    finally:
        con.close()


def _estimate_flow_rows(
    start_date: Optional[object],
    end_date: Optional[object],
) -> Optional[int]:
    if duckdb is None or not DEFAULT_DUCKDB_FILE.exists():
        return None
    con = duckdb.connect(str(DEFAULT_DUCKDB_FILE), read_only=True)
    try:
        query = f"SELECT COUNT(*) FROM {FLOW_TABLE_NAME}"
        conditions = []
        if start_date is not None:
            start_ts = pd.to_datetime(start_date)
            conditions.append(f"FECHA >= '{start_ts}'")
        if end_date is not None:
            end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            conditions.append(f"FECHA < '{end_ts}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        result = con.execute(query).fetchone()
        if not result:
            return None
        return int(result[0])
    finally:
        con.close()




def _infer_dt_minutes(df_pm: pd.DataFrame, portico_col: str, ts_col: str = "ts_min") -> int:
    if df_pm is None or df_pm.empty or ts_col not in df_pm.columns:
        return int(DT_MINUTES)

    for _, group in df_pm[[portico_col, ts_col]].dropna().groupby(portico_col):
        series = pd.to_numeric(group[ts_col], errors="coerce").dropna().sort_values()
        if len(series) < 2:
            continue
        diffs = series.diff().dropna()
        diffs = diffs[diffs > 0]
        if not diffs.empty:
            return max(1, int(diffs.min()))
    return int(DT_MINUTES)


def run_feature_generation_workflow(params):
    """
    Executes the feature generation workflow (Batch or Memory) based on params.
    Updates st.session_state.df_pm_cache with the result.
    Returns: df_pm (DataFrame) or None if failed.
    """
    source_choice = params.get("source", "Generar nuevas")
    # If source is "Cargar existentes", logic is simpler but let's handle generation first
    # Actually, the user might want to generate from scratch even if source is "En memoria" (Regenarate)

    gen_params = params.get("params", {})
    flow_source = gen_params.get("flow_source", "Archivo CSV")
    selected_raw_path = gen_params.get("flow_file_path")
    use_batches = gen_params.get("use_batches", False)
    force_snapshot = bool(gen_params.get("force_snapshot"))
    debug_cluster = bool(gen_params.get("debug_cluster"))

    cluster_features_df = None
    cluster_labels_df = None
    cluster_portico_col = None
    cluster_time_col = None
    cluster_label_mode = False
    cluster_compute_failed = False
    cluster_file = gen_params.get("cluster_file")
    cluster_vars = gen_params.get("cluster_vars") or []
    include_cluster = bool(gen_params.get("include_cluster"))
    if include_cluster and not cluster_vars:
        cluster_vars = [
            "Proporciones por cluster",
            "Flow por tipo de cluster",
            "Entropia de cluster",
            "Speed por tipo de cluster",
            "Density por tipo de cluster",
            "Delta.Speed por tipo de cluster",
            "Delta.Density por tipo de cluster",
        ]
    include_shares = "Proporciones por cluster" in cluster_vars
    include_flow = "Flow por tipo de cluster" in cluster_vars
    include_entropy = "Entropia de cluster" in cluster_vars
    include_speed = "Speed por tipo de cluster" in cluster_vars
    include_density = "Density por tipo de cluster" in cluster_vars
    include_delta_speed = "Delta.Speed por tipo de cluster" in cluster_vars
    include_delta_density = "Delta.Density por tipo de cluster" in cluster_vars
    include_delta_density = "Delta.Density por tipo de cluster" in cluster_vars

    feature_mode = gen_params.get("feature_mode")
    if not feature_mode:
        feature_groups = gen_params.get("feature_groups") or []
        if (
            "Flow 5 min (Basic & Global)" in feature_groups
            and "Snapshot Features (Window/Slopes)" not in feature_groups
        ):
            feature_mode = "Flow 5 min"
        else:
            feature_mode = "Snapshot"
    use_snapshot = feature_mode == "Snapshot"
    use_flow = feature_mode == "Flow 5 min"
    force_snapshot = use_snapshot
    include_cluster = include_cluster and use_flow

    snapshot_groups = gen_params.get("snapshot_groups") or []
    if use_snapshot and not snapshot_groups:
        snapshot_groups = list(SNAPSHOT_GROUP_KEYS)
        gen_params["snapshot_groups"] = snapshot_groups
    snapshot_group_set = set(snapshot_groups) if use_snapshot else set()
    include_window_features = "window_features" in snapshot_group_set
    include_spatial_gradients = "spatial_gradients" in snapshot_group_set
    include_temporal_encodings = "temporal_encodings" in snapshot_group_set
    snapshot_window_cols = gen_params.get("snapshot_window_cols") or []
    if include_window_features and not snapshot_window_cols:
        snapshot_window_cols = list(SNAPSHOT_WINDOW_DEFAULT_COLS)
        gen_params["snapshot_window_cols"] = snapshot_window_cols
    snapshot_grad_cols = gen_params.get("snapshot_grad_cols") or []
    if include_spatial_gradients and not snapshot_grad_cols:
        snapshot_grad_cols = list(SNAPSHOT_GRADIENT_DEFAULT_COLS)
        gen_params["snapshot_grad_cols"] = snapshot_grad_cols

    if include_cluster and cluster_file:
        try:
            cluster_sample = pd.read_csv(cluster_file, nrows=50)
        except Exception as exc:
            st.warning(f"No se pudo leer {os.path.basename(cluster_file)}: {exc}")
            cluster_sample = None
        if cluster_sample is not None and not cluster_sample.empty:
            cluster_portico_col = _find_column_by_keywords(
                cluster_sample, ["portico", "cod_portico"]
            )
            cluster_time_col = _find_column_by_keywords(
                cluster_sample, ["ts_min", "interval_start"]
            )
            if cluster_portico_col and cluster_time_col:
                available_cols = list(cluster_sample.columns)
                usecols = {cluster_portico_col, cluster_time_col}
                if include_shares:
                    usecols.update(
                        [c for c in available_cols if c.startswith("cluster_share_")]
                    )
                if include_flow:
                    usecols.update(
                        [c for c in available_cols if c.startswith("cluster_flow_")]
                    )
                if include_entropy and "cluster_entropy" in available_cols:
                    usecols.add("cluster_entropy")
                if include_speed:
                    usecols.update(
                        [c for c in available_cols if c.startswith("cluster_speed_")]
                    )
                if include_density:
                    usecols.update(
                        [c for c in available_cols if c.startswith("cluster_density_")]
                    )
                if include_delta_speed:
                    usecols.update(
                        [
                            c
                            for c in available_cols
                            if c.startswith("cluster_delta_speed_")
                        ]
                    )
                if include_delta_density:
                    usecols.update(
                        [
                            c
                            for c in available_cols
                            if c.startswith("cluster_delta_density_")
                        ]
                    )
                try:
                    cluster_features_df = pd.read_csv(
                        cluster_file, usecols=sorted(usecols)
                    )
                except Exception as exc:
                    st.warning(
                        f"No se pudo leer {os.path.basename(cluster_file)}: {exc}"
                    )
                    cluster_features_df = None
            else:
                plate_col = _find_column_by_keywords(
                    cluster_sample, ["plate", "matricula", "patente", "placa"]
                )
                label_col = _find_column_by_keywords(
                    cluster_sample, ["cluster_label", "cluster"]
                )
                if plate_col and label_col:
                    from src.utils import normalize_plate_series

                    cluster_label_mode = True
                    try:
                        cluster_labels_df = pd.read_csv(
                            cluster_file, usecols=[plate_col, label_col]
                        )
                    except Exception as exc:
                        st.warning(
                            f"No se pudo leer {os.path.basename(cluster_file)}: {exc}"
                        )
                        cluster_labels_df = None
                    if cluster_labels_df is not None:
                        cluster_labels_df = cluster_labels_df.rename(
                            columns={
                                plate_col: "plate",
                                label_col: "cluster_label",
                            }
                        )
                        cluster_labels_df["plate"] = normalize_plate_series(
                            cluster_labels_df["plate"]
                        )
                        cluster_labels_df = cluster_labels_df.dropna(
                            subset=["plate", "cluster_label"]
                        )
                        cluster_labels_df = cluster_labels_df.drop_duplicates(
                            subset=["plate"]
                        )
                        _debug_cluster(
                            bool(gen_params.get("debug_cluster")),
                            _frame_debug_info(
                                cluster_labels_df, "cluster_labels_df_loaded"
                            ),
                        )
                else:
                    st.warning(
                        "⚠️ No se encontró columna de pórtico/tiempo ni etiquetas de cluster en "
                        f"'{os.path.basename(cluster_file)}'."
                    )
    
    # Status Containers
    status = st.empty()
    progress = st.progress(0)
    status.info("Iniciando generación de features...")

    try:
        df_pm = None


        # --- VALIDATION ---
        if flow_source == "Archivo CSV" and not selected_raw_path and source_choice == "Generar nuevas":
             st.error("No se ha seleccionado un archivo CSV de flujos.")
             return None
        
        if (use_batches or flow_source == "Base de Datos (DuckDB)") and duckdb is None:
             st.error("Librería duckdb no encontrada. Instale `duckdb`.")
             return None

        # --- BATCH MODE ---
        if use_batches and source_choice == "Generar nuevas":
            con = None
            try:
                table_name = "work_flujos" 
                is_persistent_db = False

                if flow_source == "Base de Datos (DuckDB)":
                     if not os.path.exists(DEFAULT_DUCKDB_FILE):
                         st.error(f"No existe la DB: {DEFAULT_DUCKDB_FILE}")
                         return None
                     status.text("Conectando a DuckDB...")
                     con = duckdb.connect(str(DEFAULT_DUCKDB_FILE), read_only=True)
                     table_name = FLOW_TABLE_NAME
                     is_persistent_db = True
                     cols = [c[0] for c in con.execute(f"DESCRIBE {table_name}").fetchall()]
                else:
                     status.text(f"Inicializando lotes desde CSV {os.path.basename(selected_raw_path)}...")
                     con = duckdb.connect(":memory:")
                     con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{selected_raw_path}', sample_size=20000)")
                     
                     # Fix columns
                     cols = [c[0] for c in con.execute(f"DESCRIBE {table_name}").fetchall()]
                     fecha_col = next((c for c in cols if "fecha" in c.lower() or "time" in c.lower()), None)
                     if not fecha_col:
                         st.error(f"No coinciden columnas de fecha en {cols}")
                         con.close(); return None
                     if fecha_col != "FECHA":
                         con.execute(f"ALTER TABLE {table_name} RENAME {fecha_col} TO FECHA")
                
                # Date Range
                q_minmax = ""
                if is_persistent_db:
                     q_minmax = f"SELECT MIN(FECHA), MAX(FECHA) FROM {table_name}"
                else:
                     q_minmax = f"SELECT MIN(try_cast(FECHA as TIMESTAMP)), MAX(try_cast(FECHA as TIMESTAMP)) FROM {table_name}"
                
                min_max = con.execute(q_minmax).fetchone()
                if min_max[0] is None and not is_persistent_db: # Retry fallback
                     min_max = con.execute(f"SELECT MIN(FECHA), MAX(FECHA) FROM {table_name}").fetchone()

                if min_max[0] is None:
                     st.error("Sin fechas válidas en los datos.")
                     con.close(); return None
                
                start_t = pd.to_datetime(min_max[0])
                end_t = pd.to_datetime(min_max[1])
                
                # Apply User Date Filter
                if gen_params.get("start_date"):
                    user_start = pd.to_datetime(gen_params["start_date"])
                    if user_start > start_t: start_t = user_start
                
                if gen_params.get("end_date"):
                    user_end = pd.to_datetime(gen_params["end_date"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    if user_end < end_t: end_t = user_end
                
                ranges = _build_batch_ranges(start_t, end_t, gen_params.get("batch_mode", "month"))
                acc_dfs = []
                dt_m = gen_params.get("dt_minutes", int(DT_MINUTES))
                snap_config = None
                builder = None
                df_p = None
                overlap_minutes = 0
                if force_snapshot:
                    snap_config = SnapshotConfig()
                    if gen_params:
                        snap_config.dt_minutes = gen_params.get(
                            "dt_minutes", snap_config.dt_minutes
                        )
                        snap_config.window_minutes = gen_params.get(
                            "window_minutes", snap_config.window_minutes
                        )
                        snap_config.slow_speed_kmh = gen_params.get(
                            "slow_speed_kmh", snap_config.slow_speed_kmh
                        )
                    builder = SnapshotFeatureBuilder(snap_config)
                    overlap_steps = max(snap_config.window_steps, 2)
                    overlap_minutes = int(overlap_steps * snap_config.dt_minutes)
                    df_p = st.session_state.get("df_port")
                    if df_p is None:
                        df_p = load_porticos()
                cluster_frames = []
                cluster_failed = False
                desired_cols = ["FECHA", "VELOCIDAD", "CATEGORIA", "CARRIL", "PORTICO"]
                if cluster_label_mode:
                    desired_cols.append("MATRICULA")
                select_clause = "*"
                required_cols = {"FECHA", "VELOCIDAD", "PORTICO"}
                if required_cols.issubset(set(cols)):
                    select_cols = [c for c in desired_cols if c in cols]
                    if select_cols:
                        select_clause = ", ".join(select_cols)
                for idx, (r_start, r_end, label) in enumerate(ranges):
                    status.text(f"Procesando lote {idx+1}/{len(ranges)}: {label}")
                    progress.progress(idx / len(ranges))

                    query_start = r_start
                    if force_snapshot and overlap_minutes > 0:
                        query_start = r_start - pd.Timedelta(
                            minutes=overlap_minutes
                        )
                        if query_start < start_t:
                            query_start = start_t
                    q = (
                        f"SELECT {select_clause} FROM {table_name} "
                        f"WHERE try_cast(FECHA as TIMESTAMP) >= '{query_start}' "
                        f"AND try_cast(FECHA as TIMESTAMP) < '{r_end}'"
                    )
                    batch_df = con.execute(q).pl() # Polars Fetch
                    
                    if not batch_df.is_empty():
                        flow_pd = None
                        if force_snapshot:
                            flow_pd = (
                                batch_df.to_pandas()
                                if hasattr(batch_df, "to_pandas")
                                else batch_df
                            )
                            if not _is_empty_frame(flow_pd):
                                snapshot_features, _ = builder.build(
                                    flow_pd,
                                    df_p,
                                    tracked_columns=snapshot_window_cols,
                                    gradient_columns=snapshot_grad_cols,
                                    include_window_features=include_window_features,
                                    include_spatial_gradients=include_spatial_gradients,
                                    include_temporal_encodings=include_temporal_encodings,
                                )
                                feat_batch = flatten_snapshot_features(
                                    snapshot_features,
                                    include_timestamp=True,
                                )
                                feat_batch = _filter_snapshot_chunk(
                                    feat_batch,
                                    r_start,
                                    r_end,
                                    debug=debug_cluster,
                                )
                                _debug_cluster(
                                    debug_cluster,
                                    _frame_debug_info(
                                        feat_batch, "feat_batch_filtered"
                                    ),
                                )
                            else:
                                feat_batch = None
                        else:
                            feat_batch = compute_pm_features(
                                batch_df,
                                interactive=False,
                                dt_minutes=dt_m,
                            )
                        if (
                            feat_batch is not None
                            and not _is_empty_frame(feat_batch)
                        ):
                            feat_batch["portico"] = _normalize_portico_series(
                                feat_batch["portico"]
                            )
                            acc_dfs.append(feat_batch)
                        if (
                            cluster_label_mode
                            and cluster_labels_df is not None
                            and not cluster_failed
                        ):
                            try:
                                cluster_source = (
                                    flow_pd if force_snapshot else batch_df
                                )
                                if _is_empty_frame(cluster_source):
                                    cluster_batch = None
                                else:
                                    cluster_batch = _compute_cluster_features_from_flows(
                                        cluster_source,
                                        cluster_labels_df,
                                        dt_minutes=dt_m,
                                        lanes=gen_params.get("lanes", 3),
                                        include_counts=include_flow,
                                        include_entropy=include_entropy,
                                        include_speed=include_speed,
                                        include_density=include_density,
                                        include_delta_speed=include_delta_speed,
                                        include_delta_density=include_delta_density,
                                        debug=debug_cluster,
                                    )
                                    if force_snapshot and cluster_batch is not None:
                                        cluster_batch = _filter_cluster_chunk(
                                            cluster_batch, r_start, r_end
                                        )
                                if (
                                    cluster_batch is not None
                                    and not _is_empty_frame(cluster_batch)
                                ):
                                    cluster_frames.append(cluster_batch)
                            except Exception as exc:
                                st.warning(
                                    f"No se pudieron calcular variables de cluster: {exc}"
                                )
                                if debug_cluster:
                                    st.code(traceback.format_exc(), language="text")
                                cluster_failed = True
                                cluster_compute_failed = True

                con.close()
                
                if acc_dfs:
                    df_pm = pd.concat(acc_dfs, ignore_index=True)
                    if (
                        force_snapshot
                        and "portico" in df_pm.columns
                        and "ts_min" in df_pm.columns
                    ):
                        df_pm = df_pm.drop_duplicates(
                            subset=["portico", "ts_min"],
                            keep="last",
                        )
                else:
                    st.warning("No se generaron features en ningún lote.")
                    return None

                if cluster_label_mode and cluster_labels_df is not None and cluster_frames:
                    cluster_features_df = pd.concat(
                        cluster_frames, ignore_index=True
                    )
                    cluster_portico_col = "portico"
                    cluster_time_col = "interval_start"
                elif cluster_label_mode and cluster_labels_df is not None:
                    cluster_compute_failed = True

            except Exception as e:
                st.error(f"Error en Batch Mode: {e}")
                try: con.close() 
                except: pass
                return None
        
        # --- MEMORY MODE (or source=existing logic handled separately?) ---
        elif source_choice == "Generar nuevas": # Memory Mode (No Batch)
             status.text("Cargando flujos completos (Sin lotes)...")
             df_flows = None
             try:
                 if flow_source == "Base de Datos (DuckDB)":
                     if not os.path.exists(DEFAULT_DUCKDB_FILE):
                         st.error(f"No existe la DB: {DEFAULT_DUCKDB_FILE}")
                         return None
                     con = duckdb.connect(str(DEFAULT_DUCKDB_FILE), read_only=True)
                     
                     # Construct Query with Filters
                     cols = [c[0] for c in con.execute(f"DESCRIBE {FLOW_TABLE_NAME}").fetchall()]
                     desired_cols = ["FECHA", "VELOCIDAD", "PORTICO", "CATEGORIA", "CARRIL"]
                     if cluster_label_mode:
                         desired_cols.append("MATRICULA")
                     select_clause = "*"
                     required_cols = {"FECHA", "VELOCIDAD", "PORTICO"}
                     if required_cols.issubset(set(cols)):
                         select_cols = [c for c in desired_cols if c in cols]
                         if select_cols:
                             select_clause = ", ".join(select_cols)
                     query = f"SELECT {select_clause} FROM {FLOW_TABLE_NAME}"
                     conditions = []
                     
                     if gen_params.get("start_date"):
                         conditions.append(f"try_cast(FECHA as TIMESTAMP) >= '{gen_params['start_date']}'")
                     if gen_params.get("end_date"):
                         # Add 1 day to include end_date fully (as typical with date pickers)
                         end_dt = pd.to_datetime(gen_params['end_date']) + pd.Timedelta(days=1)
                         conditions.append(f"try_cast(FECHA as TIMESTAMP) < '{end_dt.strftime('%Y-%m-%d')}'")
                         
                     if conditions:
                         query += " WHERE " + " AND ".join(conditions)
                         status.text(f"Cargando DuckDB filtrado: {query}")
                     
                     df_flows = con.execute(query).pl()
                     con.close()
                 else:
                     df_flows = pl.read_csv(selected_raw_path)
             except Exception as e:
                 st.error(f"Error cargando flujos completo: {e}")
                 return None

             if df_flows is None or df_flows.is_empty():
                 st.error("Flujos vacíos.")
                 return None

             flow_pd = df_flows.to_pandas() if hasattr(df_flows, "to_pandas") else df_flows
             dt_m = gen_params.get("dt_minutes", int(DT_MINUTES))
             cluster_source = (
                 df_flows
                 if isinstance(df_flows, (pl.DataFrame, pl.LazyFrame))
                 else flow_pd
             )
             if cluster_label_mode and cluster_labels_df is not None:
                 try:
                     cluster_features_df = _compute_cluster_features_from_flows(
                         cluster_source,
                         cluster_labels_df,
                         dt_minutes=dt_m,
                         lanes=gen_params.get("lanes", 3),
                         include_counts=include_flow,
                         include_entropy=include_entropy,
                         include_speed=include_speed,
                         include_density=include_density,
                         include_delta_speed=include_delta_speed,
                         include_delta_density=include_delta_density,
                         debug=debug_cluster,
                     )
                     cluster_portico_col = "portico"
                     cluster_time_col = "interval_start"
                 except Exception as exc:
                     st.warning(
                         f"No se pudieron calcular variables de cluster: {exc}"
                     )
                     if debug_cluster:
                         st.code(traceback.format_exc(), language="text")
                     cluster_compute_failed = True
                 if cluster_features_df is not None and _is_empty_frame(cluster_features_df):
                     st.warning(
                         "No se encontraron coincidencias entre clusters y patentes en los flujos."
                     )
                     cluster_features_df = None
                     cluster_compute_failed = True

             if force_snapshot:
                 snap_config = SnapshotConfig()
                 if gen_params:
                    snap_config.dt_minutes = gen_params.get("dt_minutes", snap_config.dt_minutes)
                    snap_config.window_minutes = gen_params.get("window_minutes", snap_config.window_minutes)
                    snap_config.slow_speed_kmh = gen_params.get("slow_speed_kmh", snap_config.slow_speed_kmh)

                 builder = SnapshotFeatureBuilder(snap_config)
                 df_p = st.session_state.get('df_port')
                 if df_p is None:
                     df_p = load_porticos()

                 snapshot_features, _ = builder.build(
                     flow_pd,
                     df_p,
                     tracked_columns=snapshot_window_cols,
                     gradient_columns=snapshot_grad_cols,
                     include_window_features=include_window_features,
                     include_spatial_gradients=include_spatial_gradients,
                     include_temporal_encodings=include_temporal_encodings,
                 )
                 df_pm = flatten_snapshot_features(snapshot_features)
             else:
                 df_pm = compute_pm_features(
                     df_flows,
                     interactive=False,
                     dt_minutes=dt_m,
                 )

        elif source_choice == "Cargar existentes":
            csv_p = params.get("csv_path")
            if csv_p:
                 status.text(f"Cargando {os.path.basename(csv_p)}...")
                 df_pm = pd.read_csv(csv_p)
                 if force_snapshot and not _is_snapshot_features(df_pm):
                     st.error(
                         "Las features cargadas no corresponden a snapshots. "
                         "Genere nuevamente en modo snapshot."
                     )
                     return None

        # --- POST PROCESSING (Sorting & Filtering) ---
        if df_pm is not None:
            if "portico" in df_pm.columns:
                df_pm["portico"] = _normalize_portico_series(df_pm["portico"])
            df_pm = df_pm.sort_values(["portico", "ts_min"]).reset_index(drop=True)

            porticos_filter = gen_params.get("porticos_filter")
            if porticos_filter:
                df_pm = _apply_tramo_filter(
                    df_pm, porticos_filter, status=status
                )
                if df_pm is None or df_pm.empty:
                    status.error(
                        "Features vacías tras filtrar por tramo. "
                        "Verifique la selección."
                    )
                    return None

            # Filter Columns
            if force_snapshot and gen_params.get("snapshot_groups"):
                status.text("Filtrando columnas snapshot seleccionadas...")
                df_pm = _filter_snapshot_features(df_pm, gen_params)
            if not force_snapshot and (
                gen_params.get("metrics") or gen_params.get("global_metrics")
            ):
                status.text("Filtrando columnas seleccionadas...")
                df_pm = _filter_flow_features(df_pm, gen_params)

            # Cluster Merging
            if include_cluster and gen_params.get("cluster_file"):
                status.text("Fusionando clusters...")
                try:
                    dt_m = gen_params.get("dt_minutes", int(DT_MINUTES))
                    if cluster_features_df is None:
                        if cluster_compute_failed:
                            st.warning(
                                "⚠️ No se pudieron calcular variables de cluster con las etiquetas."
                            )
                        elif cluster_label_mode:
                            st.warning(
                                "⚠️ Se detectaron etiquetas de cluster por patente, "
                                "pero no se pudieron calcular variables sin flujos."
                            )
                        else:
                            st.warning(
                                "⚠️ No se encontraron variables de cluster válidas para unir."
                            )
                    else:
                        p_col = cluster_portico_col or _find_column_by_keywords(
                            cluster_features_df, ["portico", "cod_portico"]
                        )
                        t_col = cluster_time_col or _find_column_by_keywords(
                            cluster_features_df, ["ts_min", "interval_start"]
                        )
                        if not p_col or not t_col:
                            st.warning(
                                "⚠️ Las variables de cluster no contienen portico e intervalo."
                            )
                        else:
                            cluster_ready = _prepare_cluster_features_df(
                                cluster_features_df,
                                p_col,
                                t_col,
                                dt_m,
                            )
                            if cluster_ready.empty:
                                st.warning(
                                    "⚠️ No se encontraron coincidencias entre clusters y pórticos/tiempo."
                                )
                            else:
                                drop_cols = []
                                if not include_shares:
                                    drop_cols.extend(
                                        [
                                            c
                                            for c in cluster_ready.columns
                                            if c.startswith("cluster_share_")
                                        ]
                                    )
                                if not include_flow:
                                    drop_cols.extend(
                                        [
                                            c
                                            for c in cluster_ready.columns
                                            if c.startswith("cluster_flow_")
                                        ]
                                    )
                                if not include_entropy:
                                    drop_cols.append("cluster_entropy")
                                if not include_speed:
                                    drop_cols.extend(
                                        [
                                            c
                                            for c in cluster_ready.columns
                                            if c.startswith("cluster_speed_")
                                        ]
                                    )
                                if not include_density:
                                    drop_cols.extend(
                                        [
                                            c
                                            for c in cluster_ready.columns
                                            if c.startswith("cluster_density_")
                                        ]
                                    )
                                if not include_delta_speed:
                                    drop_cols.extend(
                                        [
                                            c
                                            for c in cluster_ready.columns
                                            if c.startswith("cluster_delta_speed_")
                                        ]
                                    )
                                if not include_delta_density:
                                    drop_cols.extend(
                                        [
                                            c
                                            for c in cluster_ready.columns
                                            if c.startswith("cluster_delta_density_")
                                        ]
                                    )
                                if drop_cols:
                                    cluster_ready = cluster_ready.drop(
                                        columns=drop_cols, errors="ignore"
                                    )
                                cluster_vars = [
                                    c
                                    for c in cluster_ready.columns
                                    if c not in ["portico", "ts_min"]
                                ]
                                if not cluster_vars:
                                    st.warning(
                                        "⚠️ No se encontraron variables de cluster para unir."
                                    )
                                else:
                                    df_pm = _merge_on_portico_ts(
                                        df_pm,
                                        cluster_ready,
                                        debug=debug_cluster,
                                        label="cluster_merge",
                                        dt_minutes=dt_m,
                                    )
                                    df_pm[cluster_vars] = df_pm[cluster_vars].fillna(0)
                except Exception as e:
                    st.warning(f"Error fusionando clusters: {e}")
        
            # Cache Result
            st.session_state.df_pm_cache = df_pm.copy()

            # Auto-save to DuckDB (with versioning)
            try:
                features_db_path = os.path.join(RESULTADOS_DIR, "features.duckdb")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                table_name = f"features_GNN_{timestamp}"
                con_feat = duckdb.connect(str(features_db_path))

                # Create versioned table
                con_feat.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_pm")

                # Also update 'latest' view for convenience
                con_feat.execute(
                    f"CREATE OR REPLACE VIEW features_latest AS SELECT * FROM {table_name}"
                )
                con_feat.close()

            except Exception as e:
                st.warning(
                    f"Features generadas pero no se pudieron guardar en DuckDB: {e}"
                )
                status.success(
                    f"Features generadas exitosamente: {len(df_pm)} registros."
                )
                st.session_state["feature_notice"] = {
                    "kind": "warning",
                    "text": (
                        "Features generadas pero no se pudieron guardar en DuckDB: "
                        f"{e}"
                    ),
                }

            progress.empty()
            return df_pm

    except Exception as e:
        status.error(f"Error en workflow de generación: {e}")
        st.session_state["feature_notice"] = {
            "kind": "error",
            "text": f"Error en workflow de generación: {e}",
        }
        return None

    return None

def render_graph_builder():
    # --- Tabs for Eventos, Feature Engineering, Feature Selection, Graph, Network, Optuna, Balance, Training ---
    (
        tab_events,
        tab_features,
        tab_feature_explorer,
        tab_selection,
        tab_graph,
        tab_visual,
        tab_network,
        tab_optuna,
        tab_balance,
        tab_training,
    ) = st.tabs(
        [
            "Eventos",
            "Feature engineering",
            "Feature explorer",
            "Feature selection",
            "Graph",
            "Visual Graph",
            "Network",
            "Optuna",
            "Balance",
            "Training",
        ]
    )

    with tab_events:
        _render_events_tab()

    with tab_features:
        _render_feature_engineering()

    with tab_feature_explorer:
        render_feature_explorer()

    with tab_selection:
        _render_feature_selection_tab()

    with tab_graph:
        mode = st.radio(
            "Modo de Grafo",
            ["Crear Nuevo", "Cargar", "En memoria"],
            horizontal=True,
            key="graph_mode_radio",
        )

        if mode == "Crear Nuevo":
            _render_create_graph()
        elif mode == "Cargar":
            _render_load_graph()
        else:
            _render_in_memory_graph()

    with tab_visual:
        render_visual_graph_tab()

    with tab_network:
        _render_network_tab()

    with tab_optuna:
        _render_optuna_tab()

    with tab_balance:
        _render_balance_tab()

    with tab_training:
        _render_training_tab()

def _render_in_memory_graph():
    st.markdown("### Grafo en Memoria")
    loaded_graph = st.session_state.get("loaded_graph")
    
    if loaded_graph is None:
        st.warning("⚠️ No hay ningún grafo cargado en memoria.")
        st.info("Puedes crear un nuevo grafo o cargar uno existente desde las opciones anteriores.")
    else:
        st.success(f"✅ Grafo cargado: **{loaded_graph.get('filename', 'Sin nombre')}**")
        
        # Display metadata
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Metadatos:**")
            if 'data' in loaded_graph:
                st.code(str(loaded_graph['data']), language="text")
        
        with col2:
            if 'feature_cols' in loaded_graph:
                st.markdown(f"**Features ({len(loaded_graph['feature_cols'])}):**")
                st.write(", ".join(loaded_graph['feature_cols'][:10]) + ("..." if len(loaded_graph['feature_cols']) > 10 else ""))
            
            if 'pm_index' in loaded_graph:
                pm_index = loaded_graph["pm_index"]
                try:
                    pm_count = len(pm_index)
                except TypeError:
                    pm_count = len(getattr(pm_index, "_rev", []))
                st.markdown(f"**Índice PM:** {pm_count} nodos.")

        graph_data = loaded_graph.get("data")
        if isinstance(graph_data, HeteroData) and graph_data.edge_types:
            st.markdown("**Muestra de grafo (nodos y aristas)**")
            edge_labels = []
            edge_types = list(graph_data.edge_types)
            for edge_type in edge_types:
                src, rel, dst = edge_type
                edge_labels.append(f"{src}-{rel}-{dst}")
            edge_choice = st.selectbox(
                "Tipo de arista",
                edge_labels,
                index=0,
                key="gnn_mem_edge_type",
            )
            edge_type = edge_types[edge_labels.index(edge_choice)]
            edge_store = graph_data[edge_type]
            edge_index = edge_store.edge_index
            if edge_index is None or edge_index.numel() == 0:
                st.warning("No hay aristas para este tipo.")
            else:
                edge_idx = edge_index.detach().cpu().numpy()
                edge_count = edge_idx.shape[1]
                max_edges = min(50, edge_count)
                edge_limit = st.slider(
                    "Aristas a mostrar",
                    min_value=1,
                    max_value=max_edges,
                    value=min(15, max_edges),
                    step=1,
                    key="gnn_mem_edge_limit",
                )
                take = min(edge_limit, edge_count)
                sample_src = edge_idx[0][:take]
                sample_dst = edge_idx[1][:take]

                pm_rev = None
                if "pm_index" in loaded_graph:
                    pm_rev = getattr(loaded_graph["pm_index"], "_rev", None)

                def _label_for_node(idx: int) -> str:
                    if pm_rev and idx in pm_rev:
                        portico, ts_min = pm_rev[idx]
                        return f"p{portico}@{ts_min}"
                    return f"n{idx}"

                edge_rows = []
                node_set = set()
                for src_idx, dst_idx in zip(sample_src, sample_dst):
                    node_set.add(int(src_idx))
                    node_set.add(int(dst_idx))
                    edge_rows.append(
                        {
                            "src": int(src_idx),
                            "dst": int(dst_idx),
                            "src_label": _label_for_node(int(src_idx)),
                            "dst_label": _label_for_node(int(dst_idx)),
                        }
                    )
                st.dataframe(pd.DataFrame(edge_rows), width="stretch")

                max_nodes = min(len(node_set), 20)
                nodes_ordered = sorted(node_set)[:max_nodes]
                node_labels = {idx: _label_for_node(idx) for idx in nodes_ordered}
                dot_lines = ["digraph G {", "  rankdir=LR;"]
                for idx in nodes_ordered:
                    label = node_labels[idx].replace('"', "'")
                    dot_lines.append(f'  n{idx} [label="{label}"];')
                for src_idx, dst_idx in zip(sample_src, sample_dst):
                    if int(src_idx) in node_labels and int(dst_idx) in node_labels:
                        dot_lines.append(f"  n{int(src_idx)} -> n{int(dst_idx)};")
                dot_lines.append("}")
                st.graphviz_chart("\n".join(dot_lines))

        if st.button("Limpiar Memoria"):
            st.session_state.loaded_graph = None
            st.session_state.graph_path = None
            st.rerun()

def _render_feature_engineering():
    st.markdown("### Feature Engineering")
    #st.info("Configure la generación de variables (features) para los nodos del grafo.")

    # Source Selection
    source = st.radio(
        "Fuente de Features",
        ["Generar nuevas", "Cargar existentes", "En memoria"],
        key="feature_source_radio",
        horizontal=True,
    )

    selected_csv_features = None
    gen_params = {}
    porticos_filter = None
    if "feat_mode" not in st.session_state:
        st.session_state["feat_mode"] = "Snapshot"
    feature_mode_state = st.session_state.get("feat_mode", "Snapshot")

    if source in ("Generar nuevas", "Cargar existentes"):
        st.markdown("#### Filtro por tramo")
        use_tramo_filter = st.checkbox(
            "Aplicar filtro por tramo",
            value=False,
            key="feat_use_tramo_filter",
        )
        if use_tramo_filter:
            df_port = _ensure_porticos_loaded()
            if df_port is not None:
                porticos_filter, _ = _render_tramo_selector(
                    df_port, key_prefix="feat_tramo"
                )
                if porticos_filter:
                    st.caption(
                        f"Tramo seleccionado: {len(porticos_filter)} pórticos."
                    )
                    st.session_state["feature_tramo_porticos"] = porticos_filter

    # --- EN MEMORIA ---
    if source == "En memoria":
        df_pm_cache = st.session_state.get("df_pm_cache")
        if isinstance(df_pm_cache, pd.DataFrame) and not df_pm_cache.empty:
            st.success(
                f"✅ Features en memoria disponibles: {len(df_pm_cache)} registros."
            )

            show_n = st.number_input(
                "Filas a mostrar",
                min_value=1,
                max_value=len(df_pm_cache),
                value=50,
                step=50,
                key="mem_preview_n",
            )
            st.dataframe(
                df_pm_cache.head(show_n),
                width='stretch',
            )
        else:
            st.warning(
                "⚠️ No hay features en memoria. Se generaran o cargaran en la proxima construccion."
            )

    # --- CARGAR EXISTENTE ---
    elif source == "Cargar existentes":
        features_db_path = os.path.join(RESULTADOS_DIR, "features.duckdb")
        
        # List tables from DuckDB
        duckdb_tables = []
        if os.path.exists(features_db_path) and duckdb is not None:
            try:
                con_list = duckdb.connect(str(features_db_path), read_only=True)
                tables_result = con_list.execute("SHOW TABLES").fetchall()
                duckdb_tables = [t[0] for t in tables_result if t[0].startswith("features_") and not t[0].endswith("_latest")]
                duckdb_tables = sorted(duckdb_tables, reverse=True)  # Newest first
                con_list.close()
            except Exception as e:
                st.warning(f"Error leyendo tablas DuckDB: {e}")
        
        # Also list CSV files as fallback
        csv_files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "features_*.csv")), reverse=True)
        csv_files = [f for f in csv_files if not os.path.basename(f).startswith("features_selected_names_")]
        
        # Build combined options
        options = []
        for t in duckdb_tables:
            options.append(("duckdb", t, f"📊 {t} (DuckDB)"))
        for f in csv_files:
            options.append(("csv", f, f"📄 {os.path.basename(f)} (CSV)"))
        
        if options:
            selected_opt = st.selectbox(
                "Selecciona versión de Features",
                options,
                format_func=lambda x: x[2],
                key="feature_source_select"
            )
            
            if st.button("Cargar en Memoria", type="primary"):
                source_type, source_path, label = selected_opt
                with st.spinner(f"Cargando {label}..."):
                    try:
                        if source_type == "duckdb":
                            con_load = duckdb.connect(str(features_db_path), read_only=True)
                            df_tmp = con_load.execute(f"SELECT * FROM {source_path}").df()
                            con_load.close()
                        else:
                            df_tmp = pd.read_csv(source_path)

                        if porticos_filter:
                            df_tmp = _apply_tramo_filter(
                                df_tmp, porticos_filter
                            )

                        if df_tmp is None or df_tmp.empty:
                            st.error(
                                "Features vacías tras filtrar por tramo. "
                                "Verifique la selección."
                            )
                        else:
                            st.session_state.df_pm_cache = df_tmp
                            st.success(
                                "Features cargadas en memoria: "
                                f"{len(df_tmp)} registros, {len(df_tmp.columns)} columnas."
                            )
                    except Exception as e:
                        st.error(f"Error cargando: {e}")
        else:
            st.warning("No hay features guardadas. Genera nuevas features primero.")

    # --- GENERAR DESDE CERO (SRC_SUMO STYLE) ---
    elif source == "Generar nuevas":
        # Force DuckDB Source as per requirement
        if not DEFAULT_DUCKDB_FILE.exists():
             st.warning(f"⚠️ La base de datos de flujos no existe en {DEFAULT_DUCKDB_FILE}. Por favor vaya a 'Base de Datos Flujos' en el menú y cargue datos.")
        else:
             st.markdown(f"Usando Base de Datos Central: {DEFAULT_DUCKDB_FILE}")
        
        # Date Filter
        use_date_filter = st.checkbox("Filtrar por Rango de Fechas", value=False, key="feat_use_date_filter")
        start_d, end_d = None, None
        if use_date_filter:
            cd1, cd2 = st.columns(2)
            start_d = cd1.date_input("Fecha Inicio", value=datetime(2023,1,1), key="feat_start_date")
            end_d = cd2.date_input("Fecha Fin", value=datetime(2023,12,31), key="feat_end_date")

        st.markdown("#### Configuración de Variables")
        
        # 1. Toggles
        c_tog1, c_tog2 = st.columns(2)
        with c_tog1:
            use_batches = st.checkbox(
                "Procesar por lotes (Optimizado)",
                value=True,
                key="feat_use_batches",
                help="Recomendado para archivos grandes.",
            )
            auto_batch = st.checkbox(
                "Auto-batch (segun rango/volumen)",
                value=True,
                key="feat_auto_batch",
            )
            auto_batch_trigger = False
            auto_batch_reason = []
            if auto_batch:
                if not use_date_filter:
                    auto_batch_trigger = True
                    auto_batch_reason.append("sin filtro de fechas")
                else:
                    if start_d and end_d:
                        range_days = (end_d - start_d).days + 1
                        if range_days >= AUTO_BATCH_RANGE_DAYS:
                            auto_batch_trigger = True
                            auto_batch_reason.append(
                                f"rango {range_days} dias"
                            )
                    if (
                        start_d
                        and end_d
                        and duckdb is not None
                        and DEFAULT_DUCKDB_FILE.exists()
                    ):
                        cache_key = f"{start_d}_{end_d}"
                        cached = st.session_state.get("auto_batch_estimate")
                        if (
                            isinstance(cached, dict)
                            and cached.get("key") == cache_key
                        ):
                            est_rows = cached.get("rows")
                        else:
                            est_rows = _estimate_flow_rows(start_d, end_d)
                            st.session_state["auto_batch_estimate"] = {
                                "key": cache_key,
                                "rows": est_rows,
                            }
                        if (
                            est_rows is not None
                            and est_rows >= AUTO_BATCH_ROW_THRESHOLD
                        ):
                            auto_batch_trigger = True
                            auto_batch_reason.append(
                                f"volumen {est_rows:,} filas"
                            )
            use_batches_effective = use_batches or (
                auto_batch and auto_batch_trigger
            )
            batch_mode_sel = "month"
            if use_batches_effective:
                batch_mode_sel = st.radio(
                    "Modo de lotes",
                    ["month", "week"],
                    horizontal=True,
                    key="feat_batch_mode",
                )
            if feature_mode_state == "Snapshot" and use_batches_effective:
                st.caption(
                    "Modo snapshot activo: el batch usa solapamiento para mantener la ventana."
                )
            if auto_batch and auto_batch_trigger and not use_batches:
                reason_text = (
                    ", ".join(auto_batch_reason)
                    if auto_batch_reason
                    else "umbral superado"
                )
                st.caption(f"Auto-batch activado: {reason_text}.")
            
            # Common: Granularity
            dt_min = st.number_input("Granularidad (min)", min_value=1, value=1, key="fp_dt", help="Intervalo de tiempo en minutos para agrupar los datos (e.g., cada 5 min). Aplica a ambos modos.")

        # --- TABS: SNAPSHOT vs FLOW ---
        tab_snap, tab_flow = st.tabs(["Snapshot Features", "Flow 5 min"])

        # --- TAB 1: SNAPSHOT ---
        with tab_snap:
            st.caption("Genera variables temporales (ventanas, pendientes, retardos) para GNN/LSTM.")
            
            # Snapshot Params
            c3, c4, c5 = st.columns(3)
            with c3:
                seq_len = st.number_input("Largo Secuencia (steps)", min_value=1, value=10, key="fp_seq", help="Cantidad de pasos de tiempo (granularidad) hacia atrás que el modelo usará como input.")
            with c4:
                win_min = st.number_input("Ventana Agregación (min)", min_value=1, value=30, key="fp_win", help="Ventana de tiempo para agregar estadísticas (e.g. media móvil).")
                guard_min = st.number_input("Guard Band (min)", min_value=0, value=0, key="fp_guard", help="Tiempo de separación entre el fin de los datos de input y el inicio del evento objetivo.")
            with c5:
                # dt_min is from common
                slow_v = st.number_input("Velocidad Lenta (km/h)", min_value=0.0, value=30.0, key="fp_slow", help="Umbral de velocidad bajo el cual se considera congestión.")
                hor_min = st.number_input("Horizonte (min)", min_value=1, value=5, key="fp_hor", help="Tiempo a futuro desde el input para predecir el evento (Target).")

            st.markdown("#### Variables Snapshot")
            snapshot_group_labels = list(SNAPSHOT_GROUP_OPTIONS.keys())
            snapshot_default = snapshot_group_labels
            snapshot_labels = st.multiselect(
                "Selecciona grupos de features",
                options=snapshot_group_labels,
                default=snapshot_default,
                key="snapshot_groups_select",
            )
            snapshot_groups = [
                SNAPSHOT_GROUP_OPTIONS[label] for label in snapshot_labels
            ]
            if not snapshot_groups:
                st.warning(
                    "Selecciona al menos un grupo de features. "
                    "Se usaran todos por defecto."
                )
                snapshot_groups = list(SNAPSHOT_GROUP_KEYS)

            snapshot_window_cols: List[str] = []
            if "window_features" in snapshot_groups:
                snapshot_window_cols = st.multiselect(
                    "Variables para ventanas/pendientes",
                    options=SNAPSHOT_WINDOW_COL_OPTIONS,
                    default=SNAPSHOT_WINDOW_DEFAULT_COLS,
                    key="snapshot_window_cols",
                )
                if not snapshot_window_cols:
                    snapshot_window_cols = list(SNAPSHOT_WINDOW_DEFAULT_COLS)

            snapshot_grad_cols: List[str] = []
            if "spatial_gradients" in snapshot_groups:
                grad_options = ["speed_mean", "flow_total", "slow_pct"]
                if "window_features" in snapshot_groups and snapshot_window_cols:
                    grad_options.extend(
                        [f"{col}_w_mean" for col in snapshot_window_cols]
                    )
                grad_options = list(dict.fromkeys(grad_options))
                default_grad = [
                    col for col in SNAPSHOT_GRADIENT_DEFAULT_COLS if col in grad_options
                ] or grad_options
                snapshot_grad_cols = st.multiselect(
                    "Variables para gradientes espaciales",
                    options=grad_options,
                    default=default_grad,
                    key="snapshot_grad_cols",
                )

            # Guide & Info
            seq_minutes = int(dt_min) * int(seq_len)
            label_start = int(guard_min) + int(dt_min)
            label_end = int(guard_min) + int(dt_min) + int(hor_min)
            example_input_end_minutes = 10 * 60
            example_input_start_minutes = example_input_end_minutes - int(dt_min)
            example_label_start_minutes = example_input_end_minutes + int(guard_min)
            example_label_end_minutes = example_label_start_minutes + int(hor_min)
            example_acc_minutes = example_label_start_minutes + min(2, int(hor_min))

            def _fmt_clock(total_minutes: int) -> str:
                hour = (total_minutes // 60) % 24
                minute = total_minutes % 60
                return f"{hour:02d}:{minute:02d}"

            example_input_start_str = _fmt_clock(example_input_start_minutes)
            example_input_end_str = _fmt_clock(example_input_end_minutes)
            example_label_start_str = _fmt_clock(example_label_start_minutes)
            example_label_end_str = _fmt_clock(example_label_end_minutes)
            example_acc_str = _fmt_clock(example_acc_minutes)
            
            st.markdown(
                f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; font-size: 0.9em;">
            <strong>Guía de Ajuste de Parámetros Temporales:</strong>
            <ul style="margin-top: 5px; padding-left: 20px;">
                <li><b>Granularidad (dt):</b> Define la resolución temporal. <i>Menor valor</i> (e.g. 1 min) captura cambios rápidos pero aumenta el ruido y el cómputo. <i>Mayor valor</i> (e.g. 15 min) suaviza la señal.</li>
                <li><b>Largo Secuencia:</b> Cuánto "pasado" observa el modelo. Con <i>dt={int(dt_min)}</i> y <i>Largo={int(seq_len)}</i>, el modelo mira {seq_minutes} minutos hacia atrás.</li>
                <li><b>Ventana Agregación:</b> Suaviza métricas antes de generar features. Con <i>{int(win_min)} min</i> se agrega sobre ese rango temporal.</li>
                <li><b>Guard Band:</b> Tiempo "ciego" entre el fin de la ventana de input y el inicio de la etiqueta. Con <i>{int(guard_min)} min</i> se excluye ese margen.</li>
                <li><b>Horizonte:</b> Rango temporal posterior al fin del input donde se busca el accidente. Con <i>{int(hor_min)} min</i> se etiqueta si ocurre un accidente en ese rango.</li>
                <li><b>Velocidad Lenta:</b> Umbral para definir congestión. Con <i>{float(slow_v):g} km/h</i> se activan features de congestión.</li>
            </ul>
            <div style="margin-top: 8px;">
                <strong>Etiquetado de accidentes:</strong> se etiqueta la ventana previa al accidente. Para cada fila en <i>ts_min</i> (inicio del intervalo),
                se marca positivo si existe un evento entre <i>ts_min + {label_start} min</i> y <i>ts_min + {label_end} min</i>, donde
                <i>{int(dt_min)} min</i> corresponde al fin del intervalo y se suma el guard band + horizonte.
            </div>
            <div style="margin-top: 6px;">
                <strong>Ejemplo:</strong> si la fila es <i>{example_input_start_str} → {example_input_end_str}</i>, la ventana positiva es
                <i>{example_label_start_str}</i> a <i>{example_label_end_str}</i>. Si ocurre un accidente a las <i>{example_acc_str}</i>,
                entonces esa fila se etiqueta como positiva.
            </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("---")
            if st.button("🚀 Generar Snapshot Features", type="primary", key="btn_run_snapshot"):
                # Prepare Params for Snapshot
                snap_params = {
                    "use_batches": use_batches_effective,
                    "batch_mode": batch_mode_sel,
                    "dt_minutes": dt_min,
                    "window_minutes": win_min,
                    "slow_speed_kmh": slow_v,
                    "seq_length": seq_len,
                    "guard_band_minutes": guard_min,
                    "horizon_minutes": hor_min,
                    "flow_source": "Base de Datos (DuckDB)",
                    "flow_file_path": None,
                    "start_date": start_d,
                    "end_date": end_d,
                    "snapshot_groups": snapshot_groups,
                    "snapshot_window_cols": snapshot_window_cols,
                    "snapshot_grad_cols": snapshot_grad_cols,
                    "feature_mode": "Snapshot",
                    "force_snapshot": True,
                    "debug_cluster": False,
                }
                if porticos_filter:
                    snap_params["porticos_filter"] = porticos_filter
                
                st.session_state.feature_config = {
                    "source": source,
                    "csv_path": None,
                    "params": snap_params,
                }
                result = run_feature_generation_workflow(st.session_state.feature_config)
                if result is not None:
                     st.success("Snapshot Features generadas correctamente.")

        # --- TAB 2: FLOW 5 MIN ---
        with tab_flow:
            st.caption("Genera métricas descriptivas, de congestión y clusters.")
            
            # Flow Params
            col1, col2 = st.columns(2)
            with col1:
                # Metric Options
                metric_options = {
                    "Flow": "flujo",
                    "Flow (veh/h)": "flujo_hph",
                    "Speed (Mean)": "velocidad_media",
                    "Speed (Std)": "velocidad_std",
                    "Speed (Harmonic)": "velocidad_harmonica",
                    "Speed (Median)": "velocidad_mediana",
                    "Speed (Q25)": "velocidad_q25",
                    "Speed (Q75)": "velocidad_q75",
                    "Speed (IQR)": "velocidad_iqr",
                    "Speed (MAD)": "velocidad_mad",
                    "Speed (CV)": "velocidad_cv",
                    "Speed (Skew)": "velocidad_skew",
                    "Speed (Kurtosis)": "velocidad_kurt_exceso",
                    "Speed (Entropy)": "velocidad_entropy",
                    "Robust Flow (Median)": "flujo_mediana_subbins",
                    "Robust Flow (Trimmed)": "flujo_media_recortada",
                    "Robust Flow (Winsor)": "flujo_media_winsor",
                    "Density": "densidad",
                    "Gap Time-Space": "brecha_ts",
                    "Flow Diff (dot_q)": "dot_q",
                    "Smooth Speed": "smooth_v"
                }
                metrics_selected = st.multiselect(
                    "Variables por Categoría",
                    list(metric_options.keys()),
                    default=["Flow", "Speed (Mean)", "Density"],
                    key="feat_metrics_v2",
                )
                
                # Global / Congestion Options
                global_options = {
                    "Total Flow": "flujo_total",
                    "Total Density": "densidad_total",
                    "Global Harmonic Speed": "v_armonica",
                    "Heavy Prop.": "prop_pesados",
                    "Congestion Index": "indice_congestion",
                    "Mix Entropy": "entropia_mezcla",
                    "Capacity (est)": "q_max_p95",
                    "V/C Ratio": "vc_ratio",
                    "FreeFlow Speed (Night)": "v_freeflow_p95_noche",
                    "Relative FF Speed": "v_rel_ff",
                    "Speed Drop": "v_drop",
                    "Slow Share (30%)": "slowshare_0p30",
                    "Slow Share (50%)": "slowshare_0p50",
                    "Slow Share (70%)": "slowshare_0p70",
                    "FD Envelope": "envolvente_FD",
                    "Critical Density (k_crit)": "k_crit",
                    "Density Ratio (k/k_crit)": "dens_ratio",
                    "FD Residual": "resid_FD",
                    "Capacity Slack": "slack_cap",
                    "Outlier Rate": "outlier_rate",
                    "Missing Rate": "missing_rate"
                }
                global_selected = st.multiselect(
                    "Indicadores Globales (Congestión)",
                    list(global_options.keys()),
                    default=["Congestion Index", "V/C Ratio", "Slow Share (50%)"],
                    key="feat_global_v2",
                )
                
                # Categories
                cat_options = ["Light", "Heavy", "Motorcycles"]
                cats_selected = st.multiselect(
                    "Tipos de vehículo",
                    cat_options,
                    default=["Light", "Heavy"],
                    key="feat_cats",
                )

            with col2:
                # Lanes
                lanes = st.number_input("Carriles (para normalizar Flow)", min_value=1, value=3, step=1, key="feat_lanes")
                
                # Cluster Checkbox
                include_cluster = st.checkbox(
                     "Incluir variables de cluster",
                     value=False,
                     key="feat_include_cluster",
                )
                
                # Cluster File
                cluster_file = None
                cluster_vars = []
                if include_cluster:
                    # Broader search for cluster files
                    all_cluster_csv = glob.glob(os.path.join(RESULTADOS_DIR, "cluster_*.csv"))
                    exclude_keywords = ["summary", "descriptive", "metrics", "quality", "k_optimo", "color_map"]
                    c_files = [
                        f for f in all_cluster_csv 
                        if not any(k in os.path.basename(f).lower() for k in exclude_keywords)
                    ]
                    c_files = sorted(c_files, key=os.path.getmtime, reverse=True)
                    
                    if c_files:
                        cluster_file = st.selectbox("Archivo de etiquetas de cluster", c_files, format_func=os.path.basename, key="feat_cluster_file")
                    else:
                        st.warning("No se encontraron archivos de cluster (cluster_*.csv)")

                    cluster_var_options = [
                        "Proporciones por cluster",
                        "Flow por tipo de cluster",
                        "Entropia de cluster",
                        "Speed por tipo de cluster",
                        "Density por tipo de cluster",
                        "Delta.Speed por tipo de cluster",
                        "Delta.Density por tipo de cluster",
                    ]
                    default_vars = cluster_var_options
                    existing_vars = st.session_state.get("gnn_cluster_vars")
                    if isinstance(existing_vars, list):
                        default_vars = [
                            item for item in existing_vars if item in cluster_var_options
                        ] or cluster_var_options
                    cluster_vars = st.multiselect(
                        "Variables de cluster",
                        options=cluster_var_options,
                        default=default_vars,
                        key="gnn_cluster_vars",
                    )
                    st.checkbox(
                        "Debug cluster (mostrar trazas)",
                        key="gnn_debug_cluster",
                        help="Muestra información de debug cuando falla el cálculo de clusters.",
                    )

            st.markdown("---")
            if st.button("🚀 Generar Flow Features", type="primary", key="btn_run_flow"):
                 # Prepare Params for Flow
                 flow_params = {
                    "use_batches": use_batches_effective,
                    "batch_mode": batch_mode_sel,
                    "metrics": metrics_selected,
                    "global_metrics": global_selected,
                    "categories": cats_selected,
                    "lanes": lanes,
                    "include_cluster": include_cluster,
                    "cluster_file": cluster_file,
                    "cluster_vars": cluster_vars,
                    "dt_minutes": dt_min,
                    "flow_source": "Base de Datos (DuckDB)",
                    "flow_file_path": None,
                    "start_date": start_d,
                    "end_date": end_d,
                    "feature_mode": "Flow 5 min",
                    "force_snapshot": False,
                    "debug_cluster": st.session_state.get("gnn_debug_cluster", False),
                 }
                 if porticos_filter:
                    flow_params["porticos_filter"] = porticos_filter
                 
                 st.session_state.feature_config = {
                    "source": source,
                    "csv_path": None,
                    "params": flow_params,
                 }
                 result = run_feature_generation_workflow(st.session_state.feature_config)
                 if result is not None:
                      st.success("Flow Features generadas correctamente.")

    notice = st.session_state.get("feature_notice")
    if notice:
        kind = notice.get("kind", "info")
        text = notice.get("text", "")
        if kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        elif kind == "error":
            st.error(text)
        else:
            st.info(text)


def _render_feature_selection_tab() -> None:
    st.subheader("Feature selection")

    accidents_df = st.session_state.get("df_acc")
    if accidents_df is None or accidents_df.empty:
        accidents_df = st.session_state.get("accidents_df")

    features_df = st.session_state.get("df_pm_cache")
    features_source = "GNN"
    features_path = None
    if features_df is None or getattr(features_df, "empty", True):
        features_df = st.session_state.get("flow_features_df")
        features_path = st.session_state.get("flow_features_path")
        features_source = st.session_state.get("flow_features_source")
    else:
        feat_cfg = st.session_state.get("feature_config", {})
        if isinstance(feat_cfg, dict):
            features_path = feat_cfg.get("csv_path")
            features_source = feat_cfg.get("source") or "GNN"

    if accidents_df is None or accidents_df.empty:
        st.info("Cargue accidentes en la pestana Eventos.")
        return
    if features_df is None or features_df.empty:
        st.info("Calcule variables de flujo en la pestana Feature engineering.")
        return

    dt_min = int(DT_MINUTES)
    if isinstance(st.session_state.get("feature_config"), dict):
        params = st.session_state["feature_config"].get("params", {})
        dt_min = int(params.get("dt_minutes", dt_min))

    features_work = features_df.copy()
    if "interval_start" not in features_work.columns and "ts_min" in features_work.columns:
        features_work["interval_start"] = (
            pd.to_datetime(features_work["ts_min"], unit="m", origin="unix")
            - pd.Timedelta(minutes=dt_min)
        )
    if "interval_start" not in features_work.columns:
        st.warning("No se encontro columna temporal para Feature selection.")
        return
    accidents_work = accidents_df.copy()
    if "ultimo_portico" in accidents_work.columns:
        accidents_work["ultimo_portico"] = _normalize_portico_series(
            accidents_work["ultimo_portico"]
        )

    params: Dict[str, object] = {}
    feature_mode = None
    if isinstance(st.session_state.get("feature_config"), dict):
        params = st.session_state["feature_config"].get("params", {}) or {}
        feature_mode = params.get("feature_mode")
    if not feature_mode:
        feature_mode = (
            "Snapshot" if _is_snapshot_features(features_work) else "Flow 5 min"
        )

    label_debug_lines: List[str] = []
    label_debug_lines.append(f"feature_mode={feature_mode} dt_min={dt_min}")

    if feature_mode == "Snapshot":
        accidents_work = accidents_work.copy()
        if "accidente_time" not in accidents_work.columns:
            st.warning("No se encontro columna de accidentes para etiquetar.")
            return
        accidents_work["ts_min"] = (
            accidents_work["accidente_time"]
            .dt.floor(f"{int(dt_min)}min")
            .astype("int64")
            // 60_000_000_000
        )
        df_acc_sel = accidents_work[["ultimo_portico", "ts_min"]].dropna()
        df_acc_sel["ultimo_portico"] = _normalize_portico_series(
            df_acc_sel["ultimo_portico"]
        )
        label_debug_lines.append(
            f"accidents_rows={len(df_acc_sel)} acc_ts_range=({df_acc_sel['ts_min'].min()}, {df_acc_sel['ts_min'].max()})"
        )

        df_port = st.session_state.get("df_port")
        if df_port is None:
            df_port = load_porticos()
        snapshot_topology = None
        if isinstance(df_port, pd.DataFrame) and not df_port.empty:
            try:
                snapshot_topology = build_static_topology(
                    df_port,
                    SnapshotConfig(dt_minutes=int(dt_min)),
                )
            except Exception:
                snapshot_topology = None

        seq_l = params.get("seq_length", SEQ_LENGTH)
        gb_min = params.get("guard_band_minutes", GUARD_BAND_MINUTES)
        hor_min = params.get("horizon_minutes", HORIZON_MINUTES)
        gb_effective = int(gb_min) + int(dt_min)
        sequence_config = SequenceConfig(
            sequence_length=int(seq_l),
            guard_band_minutes=gb_effective,
            horizon_minutes=int(hor_min),
            include_downstream=bool(INCLUDE_DOWNSTREAM_IN_LABEL),
        )
        _, labels_df = build_sequence_index(
            features_work,
            df_acc_sel,
            snapshot_topology,
            sequence_config,
            step_minutes=int(dt_min),
        )
        base_df = features_work.copy()
        base_df["target"] = labels_df["label"].to_numpy(dtype=int)
        label_debug_lines.append(
            f"seq_len={sequence_config.sequence_length} guard={sequence_config.guard_band_minutes} horizon={sequence_config.horizon_minutes} downstream={sequence_config.include_downstream}"
        )
        label_debug_lines.append(
            f"labels_pos={int(base_df['target'].sum())} total={len(base_df)}"
        )
    else:
        base_df = features_work.copy()
        base_df = base_df.drop(columns=["target"], errors="ignore")
        if (
            "accidente_time" in accidents_work.columns
            and not accidents_work.empty
            and "ts_min" in base_df.columns
        ):
            portico_col = buscar_columna(base_df, "portico") or "portico"
            acc_labels = accidents_work[["ultimo_portico", "accidente_time"]].copy()
            acc_labels["ts_min"] = (
                acc_labels["accidente_time"]
                .dt.floor(f"{int(dt_min)}min")
                .astype("int64")
                // 60_000_000_000
            ) - int(dt_min)
            acc_labels["ultimo_portico"] = _normalize_portico_series(
                acc_labels["ultimo_portico"]
            )
            acc_pairs = acc_labels[["ultimo_portico", "ts_min"]].dropna()
            acc_pairs = acc_pairs.drop_duplicates()
            acc_pairs = acc_pairs.rename(columns={"ultimo_portico": portico_col})
            base_df[portico_col] = _normalize_portico_series(base_df[portico_col])
            base_df = base_df.merge(
                acc_pairs.assign(target=1),
                on=[portico_col, "ts_min"],
                how="left",
            )
            label_debug_lines.append(
                f"flow_pairs={len(acc_pairs)} acc_ts_shifted_range=({acc_pairs['ts_min'].min()}, {acc_pairs['ts_min'].max()})"
            )
        if "target" in base_df.columns:
            base_df["target"] = base_df["target"].fillna(0).astype(int)
        else:
            base_df["target"] = 0
        label_debug_lines.append(
            f"labels_pos={int(base_df['target'].sum())} total={len(base_df)}"
        )
    if base_df.empty:
        st.warning("No se pudo preparar el dataset base.")
        return

    feature_key = _feature_selection_key(
        features_path, features_source, features_work
    )
    feature_id = _feature_selection_id(
        features_path, features_source, features_work
    )
    active_key = st.session_state.get("feature_selection_active_key")
    if active_key != feature_key:
        st.session_state["feature_selection_active_key"] = feature_key
        store = st.session_state.get("feature_selection_store", {})
        entry = store.get(feature_key)
        if entry is None:
            payload, importance_df = _load_feature_selection_from_disk(
                feature_id
            )
            if payload or importance_df is not None:
                entry = {
                    "feature_id": feature_id,
                    "selected_features": payload.get("selected_features")
                    if payload
                    else None,
                    "importance_df": importance_df,
                    "importance_hash": None,
                    "params": payload.get("params") if payload else {},
                }
                store[feature_key] = entry
                st.session_state["feature_selection_store"] = store
        if entry:
            if entry.get("importance_df") is not None:
                method = (entry.get("params") or {}).get(
                    "importance_method", "Importancia de Gini (RF)"
                )
                if method == "Gradientes (GNN)":
                    st.session_state["feature_importances_gnn_df"] = entry.get(
                        "importance_df"
                    )
                else:
                    st.session_state["feature_importances_rf_df"] = entry.get(
                        "importance_df"
                    )
            if entry.get("selected_features") is not None:
                st.session_state["selected_features"] = entry.get(
                    "selected_features"
                )
        else:
            st.session_state["selected_features"] = None
            st.session_state["feature_importances_rf_df"] = None
            st.session_state["feature_importances_gnn_df"] = None

    if features_path:
        st.caption(f"Archivo de features: {features_path}")
    else:
        st.caption("Archivo de features: (sin archivo)")

    feature_cols = _get_feature_cols(base_df)
    if not feature_cols:
        st.warning("No hay variables numericas disponibles.")
        return

    st.caption(
        f"Filas: {len(base_df):,} | Variables numericas: {len(feature_cols)}"
    )
    debug_labels = st.checkbox(
        "Debug etiquetas (mostrar resumen)",
        value=False,
        key="fs_debug_labels",
    )
    if debug_labels:
        st.code("\n".join(label_debug_lines), language="text")

    importance_method = st.radio(
        "Metodo de importancia",
        ["Importancia de Gini (RF)", "Gradientes (GNN)"],
        horizontal=True,
        key="fs_importance_method",
    )
    importance_key = (
        "feature_importances_rf_df"
        if importance_method == "Importancia de Gini (RF)"
        else "feature_importances_gnn_df"
    )
    importance_df = st.session_state.get(importance_key)

    fs_params: Dict[str, object] = {"importance_method": importance_method}

    if importance_method == "Importancia de Gini (RF)":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input(
                "n_estimators",
                min_value=50,
                value=200,
                step=50,
                key="fs_n_estimators",
            )
        with col2:
            max_depth = st.number_input(
                "max_depth (0 = sin limite)",
                min_value=0,
                value=0,
                step=1,
                key="fs_max_depth",
            )
        random_state = st.number_input(
            "random_state",
            min_value=0,
            value=42,
            step=1,
            key="fs_random_state",
        )

        if st.button("Calcular importancia (RF)"):
            try:
                from sklearn.ensemble import RandomForestClassifier
            except ImportError:
                st.error(
                    "scikit-learn no esta instalado. Ejecute `pip install scikit-learn`."
                )
                return

            progress = st.progress(0)
            try:
                X = base_df[feature_cols].fillna(0)
                progress.progress(10)
                y = base_df["target"].astype(int)
                progress.progress(20)
                if y.nunique() < 2:
                    st.warning(
                        "No hay dos clases en el target para calcular importancia."
                    )
                    return
                with st.spinner("Calculando importancia..."):
                    progress.progress(30)
                    model = RandomForestClassifier(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth) if max_depth else None,
                        criterion="gini",
                        random_state=int(random_state),
                        class_weight="balanced",
                        n_jobs=-1,
                    )
                    model.fit(X, y)
                    progress.progress(80)
                importance_df = pd.DataFrame(
                    {
                        "variable": feature_cols,
                        "importance": model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                importance_df = importance_df.reset_index(drop=True)
                progress.progress(95)
                st.session_state[importance_key] = importance_df
                progress.progress(100)
                st.success("Importancias calculadas.")
            finally:
                progress.empty()

        fs_params.update(
            {
                "n_estimators": int(n_estimators),
                "max_depth": int(max_depth) if max_depth else None,
                "random_state": int(random_state),
            }
        )
    else:
        graph_obj = st.session_state.get("loaded_graph")
        graph_data = graph_obj.get("data") if isinstance(graph_obj, dict) else None
        if not isinstance(graph_data, HeteroData):
            st.warning(
                "Cargue un grafo en la pestaña Graph para calcular Gradientes (GNN)."
            )
        else:
            num_graph_features = int(graph_data["pm"].x.shape[1])
            if len(feature_cols) != num_graph_features:
                st.warning(
                    "Las features del grafo no coinciden con las actuales. "
                    "Se usaran nombres genericos para el ranking."
                )
            gnn_has_classes = True
            y_all = graph_data["pm"].y
            y_check = y_all
            if hasattr(graph_data["pm"], "train_mask"):
                mask = graph_data["pm"].train_mask
                if mask is not None and mask.numel() == y_all.numel():
                    y_check = y_all[mask]
            unique_vals, counts = torch.unique(
                y_check.detach().cpu(), return_counts=True
            )
            if unique_vals.numel() < 2:
                gnn_has_classes = False
                st.warning(
                    "No hay dos clases en el target del grafo. "
                    f"Distribucion: {dict(zip(unique_vals.tolist(), counts.tolist()))}"
                )
            col1, col2 = st.columns(2)
            with col1:
                gnn_epochs = st.number_input(
                    "epochs",
                    min_value=1,
                    value=5,
                    step=1,
                    key="fs_gnn_epochs",
                )
                gnn_hidden = st.number_input(
                    "hidden_channels",
                    min_value=4,
                    value=16,
                    step=4,
                    key="fs_gnn_hidden",
                )
                gnn_heads = st.number_input(
                    "num_heads",
                    min_value=1,
                    value=2,
                    step=1,
                    key="fs_gnn_heads",
                )
                gnn_layers = st.number_input(
                    "num_layers",
                    min_value=1,
                    value=2,
                    step=1,
                    key="fs_gnn_layers",
                )
            with col2:
                gnn_lr = st.number_input(
                    "lr",
                    min_value=1e-5,
                    value=1e-3,
                    step=1e-4,
                    format="%.5f",
                    key="fs_gnn_lr",
                )
                gnn_weight_decay = st.number_input(
                    "weight_decay",
                    min_value=0.0,
                    value=1e-4,
                    step=1e-5,
                    format="%.5f",
                    key="fs_gnn_wd",
                )
                gnn_dropout = st.slider(
                    "dropout",
                    min_value=0.0,
                    max_value=0.8,
                    value=0.2,
                    step=0.05,
                    key="fs_gnn_dropout",
                )
                gnn_batch_size = st.number_input(
                    "batch_size",
                    min_value=64,
                    value=256,
                    step=64,
                    key="fs_gnn_batch",
                )

            neigh_text = st.text_input(
                "Vecinos por capa (ej: 15,10)",
                value="15,10",
                key="fs_gnn_neighbors",
            )
            neighbor_profile = _parse_neighbor_profile(neigh_text)

            attr_method = st.selectbox(
                "Metodo de atribucion",
                ["Gradient x Input", "Integrated Gradients"],
                index=0,
                key="fs_gnn_attr_method",
            )
            ig_steps = 20
            if attr_method == "Integrated Gradients":
                ig_steps = st.number_input(
                    "IG steps",
                    min_value=5,
                    value=20,
                    step=5,
                    key="fs_gnn_ig_steps",
                )
            target_mode = st.selectbox(
                "Objetivo para gradientes",
                ["Clase positiva (1)", "Etiqueta real"],
                index=0,
                key="fs_gnn_target_mode",
            )
            max_train_batches = st.number_input(
                "Max batches entrenamiento",
                min_value=1,
                value=50,
                step=5,
                key="fs_gnn_train_batches",
            )
            max_attr_batches = st.number_input(
                "Max batches atribuciones",
                min_value=1,
                value=50,
                step=5,
                key="fs_gnn_attr_batches",
            )
            seed_val = st.number_input(
                "seed",
                min_value=0,
                value=int(SEED),
                step=1,
                key="fs_gnn_seed",
            )

            if st.button("Calcular importancia (GNN)"):
                if not gnn_has_classes:
                    st.warning(
                        "No hay dos clases en el target para calcular importancia GNN."
                    )
                    return
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                method_code = (
                    "ig" if attr_method == "Integrated Gradients" else "grad"
                )
                target_mode_code = (
                    "true" if target_mode == "Etiqueta real" else "pos"
                )
                try:
                    with st.spinner(
                        "Entrenando GNN y calculando gradientes..."
                    ):
                        importance_df = _run_gnn_gradient_importance(
                            graph_data=graph_data,
                            feature_names=feature_cols,
                            device=device,
                            hidden_channels=int(gnn_hidden),
                            num_heads=int(gnn_heads),
                            num_layers=int(gnn_layers),
                            dropout=float(gnn_dropout),
                            lr=float(gnn_lr),
                            weight_decay=float(gnn_weight_decay),
                            epochs=int(gnn_epochs),
                            batch_size=int(gnn_batch_size),
                            neighbor_profile=neighbor_profile,
                            target_class=1,
                            target_mode=target_mode_code,
                            attribution_method=method_code,
                            ig_steps=int(ig_steps),
                            max_train_batches=int(max_train_batches),
                            max_attr_batches=int(max_attr_batches),
                            seed=int(seed_val),
                        )
                    st.session_state[importance_key] = importance_df
                    st.success("Importancias GNN calculadas.")
                except ValueError as exc:
                    st.warning(str(exc))

            fs_params.update(
                {
                    "epochs": int(gnn_epochs),
                    "hidden_channels": int(gnn_hidden),
                    "num_heads": int(gnn_heads),
                    "num_layers": int(gnn_layers),
                    "lr": float(gnn_lr),
                    "weight_decay": float(gnn_weight_decay),
                    "dropout": float(gnn_dropout),
                    "batch_size": int(gnn_batch_size),
                    "neighbor_profile": list(neighbor_profile),
                    "attribution_method": attr_method,
                    "ig_steps": int(ig_steps),
                    "target_mode": target_mode,
                    "max_train_batches": int(max_train_batches),
                    "max_attr_batches": int(max_attr_batches),
                    "seed": int(seed_val),
                }
            )
    ordered_vars = feature_cols
    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        importance_df = importance_df[
            importance_df["variable"].isin(feature_cols)
        ].copy()
        if importance_df.empty:
            st.session_state[importance_key] = None
            st.info("Calcule la importancia para ordenar las variables.")
        else:
            st.session_state[importance_key] = importance_df
            st.dataframe(importance_df, width="stretch")
            ordered_vars = importance_df["variable"].tolist()
    else:
        st.info("Calcule la importancia para ordenar las variables.")

    selected_features = st.session_state.get("selected_features")
    if selected_features is None:
        selected_features = list(ordered_vars)
    else:
        selected_features = [
            feature for feature in selected_features if feature in ordered_vars
        ]

    col_imp, col_btn = st.columns([2, 1])
    with col_imp:
        if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
            min_imp = float(importance_df["importance"].min())
            max_imp = float(importance_df["importance"].max())
            range_span = max_imp - min_imp
            slider_kwargs = {
                "min_value": min_imp,
                "max_value": max_imp,
                "value": min_imp,
                "format": "%.6f",
            }
            if range_span > 0:
                step = max(range_span / 1000.0, 1e-6)
                if step > range_span:
                    step = range_span
                slider_kwargs["step"] = step
            threshold = st.slider(
                "Seleccionar por umbral de importancia (> value)",
                **slider_kwargs,
            )
            if st.button("Seleccionar > Umbral"):
                subset = importance_df[importance_df["importance"] > threshold]
                new_selection = subset["variable"].tolist()
                st.session_state["selected_features"] = new_selection
                for idx, feature in enumerate(ordered_vars):
                    safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
                    k = f"feature_sel_{idx}_{safe_key}"
                    st.session_state[k] = feature in new_selection
                st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Seleccionar todas"):
            st.session_state["selected_features"] = list(ordered_vars)
            for idx, feature in enumerate(ordered_vars):
                safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
                k = f"feature_sel_{idx}_{safe_key}"
                st.session_state[k] = True
            st.rerun()
    with col_b:
        if st.button("Limpiar seleccion"):
            st.session_state["selected_features"] = []
            for idx, feature in enumerate(ordered_vars):
                safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
                k = f"feature_sel_{idx}_{safe_key}"
                st.session_state[k] = False
            st.rerun()

    st.caption("Selecciona variables para usar en Balance y Modelos.")
    selected = []
    for idx, feature in enumerate(ordered_vars):
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
        checkbox_key = f"feature_sel_{idx}_{key}"
        if checkbox_key in st.session_state:
            checked = st.checkbox(feature, key=checkbox_key)
        else:
            checked = st.checkbox(
                feature,
                value=feature in selected_features,
                key=checkbox_key,
            )
        if checked:
            selected.append(feature)
    st.session_state["selected_features"] = selected
    st.caption(f"Variables seleccionadas: {len(selected)}")
    st.markdown("**Resumen de variables seleccionadas**")
    if selected:
        st.dataframe(pd.DataFrame({"Variable": selected}), width="stretch")
    else:
        st.info("No hay variables seleccionadas.")

    _persist_feature_selection(
        feature_key=feature_key,
        feature_id=feature_id,
        features_path=features_path,
        features_source=features_source,
        features_df=features_work,
        selected_features=selected,
        importance_df=importance_df if isinstance(importance_df, pd.DataFrame) else None,
        params=fs_params,
    )


def _render_balance_tab() -> None:
    st.subheader("Balance (GNN)")

    balance_graph_obj = st.session_state.get("loaded_graph")
    if balance_graph_obj is None:
        st.warning(
            "No hay grafo cargado en memoria. Cargue o construya un grafo en la pestaña Graph."
        )
        return

    if not isinstance(balance_graph_obj, dict):
        st.info("Cargue un grafo para continuar.")
        return

    graph_data = balance_graph_obj.get("data")
    if not isinstance(graph_data, HeteroData):
        st.warning("El grafo seleccionado no es HeteroData.")
        return

    if "pm" not in graph_data.node_types:
        st.warning("El grafo no contiene nodos 'pm'.")
        return
    if not hasattr(graph_data["pm"], "y"):
        st.warning("El grafo no contiene etiquetas 'y'.")
        return
    if not hasattr(graph_data["pm"], "train_mask"):
        st.warning("El grafo no contiene train_mask.")
        return

    method = st.radio(
        "Metodo de balance",
        ["GraphSMOTE", "ImGAGN"],
        horizontal=True,
        key="gnn_balance_method",
    )

    if method == "GraphSMOTE":
        node_types = list(graph_data.node_types)
        node_type = st.selectbox(
            "Tipo de nodo",
            node_types,
            index=node_types.index("pm")
            if "pm" in node_types
            else 0,
            key="gnn_smote_node_type",
        )
        if node_type != "pm":
            st.warning(
                "GraphSMOTE actualmente solo esta soportado para nodos 'pm'."
            )
            return
        minority_class = st.number_input(
            "Clase minoritaria",
            min_value=0,
            value=1,
            step=1,
            key="gnn_smote_minority",
        )
        optuna_files = sorted(
            glob.glob(os.path.join(RESULTADOS_DIR, "optuna_hyperparams_*.csv")),
            key=os.path.getmtime,
            reverse=True,
        )
        optuna_choice = st.selectbox(
            "Parametros Optuna (opcional)",
            options=["(ninguno)"] + optuna_files,
            format_func=lambda x: "(ninguno)"
            if x == "(ninguno)"
            else os.path.basename(x),
            key="gnn_smote_optuna_file",
        )
        if st.button("Aplicar Optuna a Balance", key="gnn_smote_apply_optuna"):
            if optuna_choice == "(ninguno)":
                st.warning("Seleccione un archivo de Optuna.")
            else:
                params = _load_optuna_params_from_csv(optuna_choice)
                if "smote_k" in params:
                    try:
                        st.session_state["gnn_smote_k"] = int(params["smote_k"])
                    except Exception:
                        pass
                if (
                    "target_pos_ratio" in params
                    and hasattr(graph_data[node_type], "train_mask")
                    and hasattr(graph_data[node_type], "y")
                ):
                    try:
                        n_syn = _compute_n_syn_from_ratio(
                            graph_data[node_type].y,
                            graph_data[node_type].train_mask,
                            int(minority_class),
                            float(params["target_pos_ratio"]),
                        )
                        if n_syn > 0:
                            st.session_state["gnn_smote_n_samples"] = int(n_syn)
                    except Exception:
                        pass
                st.success("Parametros Optuna aplicados.")
                st.rerun()
        n_samples = st.number_input(
            "Nuevos nodos sinteticos",
            min_value=1,
            value=500,
            step=50,
            key="gnn_smote_n_samples",
        )
        k_smote = st.number_input(
            "k_smote",
            min_value=1,
            value=int(GRAPHSMOTE_K),
            step=1,
            key="gnn_smote_k",
        )
        k_neighbors_edges = st.number_input(
            "k_neighbors_edges",
            min_value=1,
            value=5,
            step=1,
            key="gnn_smote_k_edges",
        )
        random_state = st.number_input(
            "random_state",
            min_value=0,
            value=int(SEED),
            step=1,
            key="gnn_smote_seed",
        )
        add_to_train_mask = st.checkbox(
            "Agregar sinteticos a train_mask",
            value=True,
            key="gnn_smote_add_train",
        )
        disable_acc_smote_pm_generation = st.checkbox(
            "Desactivar generacion de accidentes sinteticos",
            value=True,
            key="gnn_smote_disable_acc",
        )
        conect_mode = st.radio(
            "Conexion de sinteticos",
            ["Top-K TRAIN real", "Vecinos de accidentes (1-hop)"],
            horizontal=True,
            key="gnn_smote_conect_mode",
        )
        bidirectional = st.checkbox(
            "Aristas bidireccionales (real <-> sintetico)",
            value=False,
            key="gnn_smote_bidir",
        )

        st.markdown("#### Modelo GNN entrenado")
        st.write(
            "GraphSMOTE necesita un GNN entrenado para generar embeddings y "
            "decodificadores z->x. El entrenamiento guarda modelos "
            "`gat_model_BEST*.pt` en Resultados y un _hparams.json asociado."
        )
        st.write(
            "Para usarlo aqui: entrena el modelo con el grafo en memoria, "
            "selecciona el archivo .pt y luego ejecuta GraphSMOTE."
        )

        model_files = sorted(
            glob.glob(os.path.join(RESULTADOS_DIR, "*.pt"))
        )
        model_path = st.selectbox(
            "Modelo GNN (.pt)",
            options=["(ninguno)"] + model_files,
            format_func=lambda x: "(ninguno)"
            if x == "(ninguno)"
            else os.path.basename(x),
            key="gnn_smote_model_file",
        )
        model_path_override = st.text_input(
            "Ruta manual del modelo (opcional)",
            value="",
            key="gnn_smote_model_path",
        )
        if model_path_override.strip():
            model_path = model_path_override.strip()

        train_use_smote = st.checkbox(
            "Entrenar con GraphSMOTE (mismo grafo)",
            value=False,
            key="gnn_smote_train_with_smote",
        )
        train_reuse_hparams = st.checkbox(
            "Reutilizar hiperparametros guardados si existen",
            value=True,
            key="gnn_smote_train_reuse_hparams",
        )
        col_train, col_delete = st.columns(2)
        with col_train:
            if st.button("Entrenar modelo GNN", key="gnn_smote_train_model"):
                graph_obj = dict(balance_graph_obj)
                graph_obj["data"] = graph_data
                if not graph_obj.get("filename"):
                    graph_path = st.session_state.get("graph_path")
                    if graph_path:
                        graph_obj["filename"] = os.path.basename(graph_path)

                def _has_imgagn(obj: dict) -> bool:
                    try:
                        return bool(obj.get("imgagn_best_params")) or (
                            "ImGAGN" in str(obj.get("filename", ""))
                        )
                    except Exception:
                        return False

                def _list_hpo_files(
                    use_graphsmote: bool, obj: dict
                ) -> list[str]:
                    all_hp_files = sorted(
                        glob.glob(
                            os.path.join(
                                RESULTADOS_DIR, "optuna_hyperparams_*.csv"
                            )
                        ),
                        key=os.path.getmtime,
                    )
                    want_imgagn = _has_imgagn(obj)

                    def _score_hp(path: str) -> tuple:
                        name = os.path.basename(path)
                        has_smote = "_GraphSMOTE" in name
                        has_imgagn = "_ImGAGN" in name
                        has_base = "_Base" in name
                        ok_smote = (has_smote == use_graphsmote)
                        ok_imgagn = (has_imgagn == want_imgagn)
                        base_pref = (has_base and not use_graphsmote)
                        return (
                            int(ok_smote),
                            int(ok_imgagn),
                            int(base_pref),
                            os.path.getmtime(path),
                        )

                    hp_files_sorted = sorted(all_hp_files, key=_score_hp)
                    return [
                        f
                        for f in hp_files_sorted
                        if ("_GraphSMOTE" in os.path.basename(f))
                        == use_graphsmote
                    ]

                hp_files = _list_hpo_files(
                    bool(train_use_smote), graph_obj
                )
                hp_choice = (
                    len(hp_files)
                    if train_reuse_hparams and hp_files
                    else 0
                )

                try:
                    from src import gnn_main as graph_main
                except Exception as exc:
                    st.error(f"No se pudo cargar el entrenador GNN: {exc}")
                    return

                original_input = builtins.input

                def _auto_input(prompt: str = "") -> str:
                    norm = (
                        unicodedata.normalize("NFKD", prompt)
                        .encode("ascii", "ignore")
                        .decode("ascii")
                        .lower()
                    )
                    if "seleccione el numero del archivo" in norm:
                        return str(hp_choice)
                    if "reutilizar estos" in norm:
                        return "s" if train_reuse_hparams else "n"
                    return "0"

                builtins.input = _auto_input
                with st.spinner("Entrenando modelo GNN..."):
                    try:
                        graph_main.run_gat_training(
                            graph_obj,
                            force_use_graphsmote=bool(train_use_smote),
                        )
                    except Exception as exc:
                        st.error(f"Error al entrenar el modelo: {exc}")
                        return
                    finally:
                        builtins.input = original_input

                latest_model = None
                latest_candidates = glob.glob(
                    os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt")
                )
                if latest_candidates:
                    latest_model = max(
                        latest_candidates, key=os.path.getmtime
                    )
                if latest_model:
                    st.session_state["gnn_smote_model_file"] = latest_model
                st.success(
                    "Entrenamiento finalizado. Seleccione el modelo en la lista."
                )
                st.rerun()

        with col_delete:
            confirm_delete = st.checkbox(
                "Confirmar borrado del modelo seleccionado",
                value=False,
                key="gnn_smote_confirm_delete",
            )
            if st.button(
                "Borrar modelo seleccionado", key="gnn_smote_delete_model"
            ):
                if not model_path or model_path == "(ninguno)":
                    st.error("Seleccione un modelo para borrar.")
                elif not confirm_delete:
                    st.warning("Confirme el borrado antes de continuar.")
                else:
                    try:
                        model_path_obj = Path(model_path).resolve()
                        resultados_dir = Path(RESULTADOS_DIR).resolve()
                        if resultados_dir not in model_path_obj.parents:
                            st.error(
                                "Solo se pueden borrar modelos dentro de Resultados."
                            )
                        elif not model_path_obj.exists():
                            st.error("El archivo seleccionado no existe.")
                        else:
                            hparams_path = model_path_obj.with_name(
                                model_path_obj.stem + "_hparams.json"
                            )
                            model_path_obj.unlink()
                            if hparams_path.exists():
                                hparams_path.unlink()
                            st.session_state["gnn_smote_model_file"] = (
                                "(ninguno)"
                            )
                            st.success("Modelo borrado.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"No se pudo borrar el modelo: {exc}")

        edge_feature_dim = 0
        for edge_type in graph_data.edge_types:
            store = graph_data[edge_type]
            if hasattr(store, "edge_attr") and store.edge_attr is not None:
                edge_attr = store.edge_attr
                edge_feature_dim = (
                    int(edge_attr.shape[1])
                    if edge_attr.dim() > 1
                    else 1
                )
                break
        in_channels = int(graph_data[node_type].x.shape[1])
        st.caption(
            f"Dimensiones detectadas: in_channels={in_channels}, edge_feature_dim={edge_feature_dim}"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            hidden_channels = st.number_input(
                "hidden_channels",
                min_value=1,
                value=128,
                step=16,
                key="gnn_smote_hidden",
            )
            num_heads = st.number_input(
                "num_heads",
                min_value=1,
                value=4,
                step=1,
                key="gnn_smote_heads",
            )
            num_layers = st.number_input(
                "num_layers",
                min_value=1,
                value=2,
                step=1,
                key="gnn_smote_layers",
            )
        with col_b:
            dropout = st.number_input(
                "dropout",
                min_value=0.0,
                max_value=0.9,
                value=0.2,
                step=0.05,
                key="gnn_smote_dropout",
            )
            out_channels = st.number_input(
                "out_channels",
                min_value=1,
                value=1,
                step=1,
                key="gnn_smote_out",
            )
            device = st.selectbox(
                "Dispositivo",
                ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
                key="gnn_smote_device",
            )

        save_dir = st.text_input(
            "Directorio z2x decoders",
            value=os.path.join(RESULTADOS_DIR, "z2x_decoders"),
            key="gnn_smote_save_dir",
        )
        save_aug = st.checkbox(
            "Guardar grafo balanceado",
            value=False,
            key="gnn_smote_save_graph",
        )
        save_path = None
        if save_aug:
            default_name = f"graph_smote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            save_path = st.text_input(
                "Ruta de guardado",
                value=os.path.join(RESULTADOS_DIR, default_name),
                key="gnn_smote_save_path",
            )

        if st.button("Balancear con GraphSMOTE", key="gnn_smote_run"):
            if not model_path or model_path == "(ninguno)":
                st.error("Seleccione un modelo GNN entrenado.")
                return
            if not hasattr(graph_data[node_type], "train_mask"):
                st.error("El grafo no tiene train_mask para balancear.")
                return
            if not hasattr(graph_data[node_type], "y"):
                st.error("El grafo no tiene etiquetas 'y' para balancear.")
                return

            try:
                import src.graphsmote as graphsmote_mod
            except Exception as exc:
                st.error(f"No se pudo cargar GraphSMOTE: {exc}")
                return

            graphsmote_mod.BIDIRECCTION = 1 if bidirectional else 0
            graphsmote_mod.GRAPHSMOTE_CONECT = (
                1 if conect_mode == "Top-K TRAIN real" else 0
            )

            try:
                from src.gat_model import HeteroGAT
            except Exception as exc:
                st.error(f"No se pudo cargar el modelo GAT: {exc}")
                return

            try:
                model_obj = torch.load(
                    model_path,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
            except Exception as exc:
                st.error(f"No se pudo cargar el modelo: {exc}")
                return

            model = None
            state_dict = None
            if isinstance(model_obj, torch.nn.Module):
                model = model_obj
            elif isinstance(model_obj, dict):
                if isinstance(model_obj.get("model"), torch.nn.Module):
                    model = model_obj.get("model")
                else:
                    state_dict = (
                        model_obj.get("model_state")
                        or model_obj.get("state_dict")
                    )
            if model is None:
                if state_dict is None:
                    st.error("El modelo cargado no es compatible.")
                    return
                model = HeteroGAT(
                    in_channels=in_channels,
                    hidden_channels=int(hidden_channels),
                    out_channels=int(out_channels),
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    edge_feature_dim=int(edge_feature_dim),
                    num_layers=int(num_layers),
                    use_checkpointing=False,
                )
                missing, unexpected = model.load_state_dict(
                    state_dict, strict=False
                )
                if missing or unexpected:
                    st.warning(
                        f"Estado cargado con diferencias. missing={len(missing)} unexpected={len(unexpected)}"
                    )

            model = model.to(device)
            model.eval()

            nodes_to_smote = [
                {
                    "node_type": node_type,
                    "minority_class": int(minority_class),
                    "n_samples": int(n_samples),
                }
            ]
            with st.spinner("Ejecutando GraphSMOTE..."):
                try:
                    augmented = graphsmote_mod.run_graphsmote(
                        model,
                        graph_data,
                        nodes_to_smote,
                        int(k_smote),
                        int(k_neighbors_edges),
                        int(random_state),
                        save_dir=save_dir,
                        add_to_train_mask=bool(add_to_train_mask),
                        disable_acc_smote_pm_generation=bool(
                            disable_acc_smote_pm_generation
                        ),
                        save_path=save_path,
                    )
                except Exception as exc:
                    st.error(f"Error en GraphSMOTE: {exc}")
                    return

            st.session_state["balanced_graph"] = {
                "data": augmented,
                "source": "GraphSMOTE",
                "params": {
                    "node_type": node_type,
                    "minority_class": int(minority_class),
                    "n_samples": int(n_samples),
                    "k_smote": int(k_smote),
                    "k_neighbors_edges": int(k_neighbors_edges),
                },
            }
            st.success(
                f"Balance GraphSMOTE completado. Nodos: {augmented[node_type].num_nodes}"
            )

    else:
        node_types = list(graph_data.node_types)
        target_ntype = st.selectbox(
            "Tipo de nodo",
            node_types,
            index=node_types.index("pm")
            if "pm" in node_types
            else 0,
            key="gnn_imgagn_target",
        )
        minority_class = st.number_input(
            "Clase minoritaria",
            min_value=0,
            value=1,
            step=1,
            key="gnn_imgagn_minority",
        )
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.append("cuda")
        device = st.selectbox(
            "Dispositivo",
            device_options,
            key="gnn_imgagn_device",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            dz = st.number_input(
                "dz",
                min_value=10,
                value=100,
                step=10,
                key="gnn_imgagn_dz",
            )
            hidden_g = st.number_input(
                "hidden_g",
                min_value=16,
                value=200,
                step=10,
                key="gnn_imgagn_hidden_g",
            )
            n_hidden_g = st.number_input(
                "n_hidden_g",
                min_value=1,
                value=1,
                step=1,
                key="gnn_imgagn_n_hidden_g",
            )
            topk_links = st.number_input(
                "topk_links",
                min_value=1,
                value=5,
                step=1,
                key="gnn_imgagn_topk",
            )
        with col2:
            emb_dim = st.number_input(
                "emb_dim",
                min_value=16,
                value=128,
                step=16,
                key="gnn_imgagn_emb_dim",
            )
            hid_d = st.number_input(
                "hid_d",
                min_value=16,
                value=128,
                step=16,
                key="gnn_imgagn_hid_d",
            )
            dropout = st.number_input(
                "dropout",
                min_value=0.0,
                max_value=0.9,
                value=0.2,
                step=0.05,
                key="gnn_imgagn_dropout",
            )
            lambda1_ratio = st.number_input(
                "lambda1_ratio",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="gnn_imgagn_lambda1",
            )
        with col3:
            d_steps = st.number_input(
                "d_steps",
                min_value=1,
                value=50,
                step=1,
                key="gnn_imgagn_d_steps",
            )
            epochs = st.number_input(
                "epochs",
                min_value=1,
                value=100,
                step=5,
                key="gnn_imgagn_epochs",
            )
            lr_g = st.number_input(
                "lr_g",
                min_value=1e-5,
                value=1e-3,
                step=1e-5,
                format="%.5f",
                key="gnn_imgagn_lr_g",
            )
            lr_d = st.number_input(
                "lr_d",
                min_value=1e-5,
                value=1e-3,
                step=1e-5,
                format="%.5f",
                key="gnn_imgagn_lr_d",
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            wd = st.number_input(
                "wd",
                min_value=0.0,
                value=5e-4,
                step=1e-4,
                format="%.5f",
                key="gnn_imgagn_wd",
            )
            alpha_reg = st.number_input(
                "alpha_reg",
                min_value=0.0,
                value=1e-4,
                step=1e-5,
                format="%.5f",
                key="gnn_imgagn_alpha",
            )
            beta_reg = st.number_input(
                "beta_reg",
                min_value=0.0,
                value=1e-4,
                step=1e-5,
                format="%.5f",
                key="gnn_imgagn_beta",
            )
        with col5:
            margin_mm = st.number_input(
                "margin_mm",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="gnn_imgagn_margin",
            )
            grad_clip = st.number_input(
                "grad_clip",
                min_value=0.5,
                value=2.0,
                step=0.5,
                key="gnn_imgagn_grad",
            )
            max_new_nodes = st.number_input(
                "max_new_nodes",
                min_value=1,
                value=100000,
                step=1000,
                key="gnn_imgagn_max_new",
            )
        with col6:
            use_neighbor_loader = st.checkbox(
                "use_neighbor_loader",
                value=True,
                key="gnn_imgagn_use_loader",
            )
            batch_size = st.number_input(
                "batch_size",
                min_value=32,
                value=512,
                step=32,
                key="gnn_imgagn_batch",
            )
            num_neighbors_raw = st.text_input(
                "num_neighbors (por capa, ej: 10,5,5)",
                value="10,5,5",
                key="gnn_imgagn_neighbors",
            )

        show_progress = st.checkbox(
            "Mostrar progreso",
            value=True,
            key="gnn_imgagn_progress",
        )

        if st.button("Balancear con ImGAGN", key="gnn_imgagn_run"):
            if not hasattr(graph_data[target_ntype], "train_mask"):
                st.error("El grafo no tiene train_mask para balancear.")
                return
            if not hasattr(graph_data[target_ntype], "y"):
                st.error("El grafo no tiene etiquetas 'y' para balancear.")
                return
            y = graph_data[target_ntype].y
            y_binary = (y == int(minority_class)).long()

            num_neighbors = None
            if num_neighbors_raw.strip():
                try:
                    num_neighbors = [
                        int(n) for n in num_neighbors_raw.split(",") if n.strip()
                    ]
                except ValueError:
                    st.error("num_neighbors debe ser una lista de enteros.")
                    return

            try:
                import src.imgagn as imgagn_mod
            except Exception as exc:
                st.error(f"No se pudo cargar ImGAGN: {exc}")
                return

            cfg = imgagn_mod.ImGAGNConfig(
                dz=int(dz),
                hidden_g=int(hidden_g),
                n_hidden_g=int(n_hidden_g),
                topk_links=int(topk_links),
                emb_dim=int(emb_dim),
                hid_d=int(hid_d),
                dropout=float(dropout),
                lambda1_ratio=float(lambda1_ratio),
                d_steps=int(d_steps),
                epochs=int(epochs),
                lr_g=float(lr_g),
                lr_d=float(lr_d),
                wd=float(wd),
                alpha_reg=float(alpha_reg),
                beta_reg=float(beta_reg),
                margin_mm=float(margin_mm),
                grad_clip=float(grad_clip),
                device=str(device),
                use_neighbor_loader=bool(use_neighbor_loader),
                batch_size=int(batch_size),
                num_neighbors=num_neighbors,
                max_new_nodes=int(max_new_nodes),
                show_progress=bool(show_progress),
            )

            with st.spinner("Ejecutando ImGAGN..."):
                try:
                    result = imgagn_mod.train_imgagn(
                        graph_data,
                        graph_data[target_ntype].train_mask,
                        y_binary,
                        cfg=cfg,
                        target_ntype=target_ntype,
                    )
                except Exception as exc:
                    st.error(f"Error en ImGAGN: {exc}")
                    return

            x_aug = result.get("x_aug")
            edge_index_aug = result.get("edge_index_aug")
            if x_aug is None or edge_index_aug is None:
                st.error("ImGAGN no devolvio grafo aumentado.")
                return

            n_original = graph_data[target_ntype].num_nodes
            n_aug = x_aug.size(0)
            n_new = max(0, n_aug - n_original)
            y_aug = torch.cat(
                [y_binary, torch.ones(n_new, dtype=y_binary.dtype)], dim=0
            )
            train_mask_base = graph_data[target_ntype].train_mask
            val_mask_base = (
                graph_data[target_ntype].val_mask
                if hasattr(graph_data[target_ntype], "val_mask")
                else torch.zeros(n_original, dtype=torch.bool)
            )
            test_mask_base = (
                graph_data[target_ntype].test_mask
                if hasattr(graph_data[target_ntype], "test_mask")
                else torch.zeros(n_original, dtype=torch.bool)
            )
            train_mask_aug = torch.cat(
                [train_mask_base, torch.ones(n_new, dtype=torch.bool)],
                dim=0,
            )
            val_mask_aug = torch.cat(
                [val_mask_base, torch.zeros(n_new, dtype=torch.bool)],
                dim=0,
            )
            test_mask_aug = torch.cat(
                [test_mask_base, torch.zeros(n_new, dtype=torch.bool)],
                dim=0,
            )

            data_aug = HeteroData()
            data_aug[target_ntype].x = x_aug
            data_aug[target_ntype].y = y_aug
            data_aug[target_ntype].train_mask = train_mask_aug
            data_aug[target_ntype].val_mask = val_mask_aug
            data_aug[target_ntype].test_mask = test_mask_aug
            data_aug[target_ntype].is_synthetic = torch.cat(
                [
                    torch.zeros(n_original, dtype=torch.bool),
                    torch.ones(n_new, dtype=torch.bool),
                ],
                dim=0,
            )
            data_aug[(target_ntype, "imgagn", target_ntype)].edge_index = edge_index_aug

            st.session_state["balanced_graph"] = {
                "data": data_aug,
                "source": "ImGAGN",
                "params": {"target_ntype": target_ntype},
            }
            st.success(
                f"Balance ImGAGN completado. Nodos: {data_aug[target_ntype].num_nodes}"
            )


def _render_training_tab() -> None:
    st.subheader("Training (GNN)")
    loaded_graph = st.session_state.get("loaded_graph")
    balanced_graph = st.session_state.get("balanced_graph")

    if loaded_graph is None:
        st.warning("No hay grafo cargado en memoria. Use la pestaña Graph.")
        return

    use_balanced = False
    if balanced_graph is not None:
        use_balanced = st.checkbox(
            "Usar grafo balanceado",
            value=True,
            key="gnn_train_use_balanced",
        )
        if use_balanced:
            st.caption(
                f"Se entrenara con el grafo balanceado ({balanced_graph.get('source', 'N/A')})."
            )
        else:
            st.caption("Se entrenara con el grafo original cargado en memoria.")
    else:
        st.info("No hay grafo balanceado disponible. Se usara el grafo original.")

    graph_obj = dict(loaded_graph)
    graph_source_label = "Original"
    if use_balanced and balanced_graph is not None:
        graph_obj["data"] = balanced_graph.get("data")
        graph_source_label = f"Balanceado ({balanced_graph.get('source', 'N/A')})"
    if not graph_obj.get("filename"):
        graph_path = st.session_state.get("graph_path")
        if graph_path:
            graph_obj["filename"] = os.path.basename(graph_path)

    graph_data = graph_obj.get("data")
    if not isinstance(graph_data, HeteroData):
        st.warning("El grafo seleccionado no es HeteroData.")
        return
    if "pm" not in graph_data.node_types:
        st.warning("El grafo no contiene nodos 'pm'.")
        return

    required_edges = {("pm", "spatial", "pm"), ("pm", "temporal", "pm")}
    missing_edges = [e for e in required_edges if e not in graph_data.edge_types]
    if missing_edges:
        st.warning(
            "Faltan aristas necesarias para entrenar el GNN. "
            "Use el grafo original o un balanceo que preserve la topologia."
        )

    pm_nodes = int(graph_data["pm"].num_nodes)
    total_edges = int(
        sum(
            graph_data[edge_type].edge_index.shape[1]
            for edge_type in graph_data.edge_types
        )
    )
    synth_count = 0
    if hasattr(graph_data["pm"], "is_synthetic"):
        try:
            synth_count = int(graph_data["pm"].is_synthetic.sum().item())
        except Exception:
            synth_count = 0

    st.markdown(
        f"**Grafo seleccionado:** {graph_source_label}  \n"
        f"- Nodos PM: {pm_nodes}  \n"
        f"- Aristas: {total_edges}  \n"
        f"- Sinteticos: {synth_count}"
    )
    st.caption(
        "El entrenamiento usa las mascaras train/val/test del grafo para reportar metricas."
    )
    split_train_ratio = int(
        st.session_state.get("gnn_split_train_ratio", TRAIN_RATIO)
    )
    split_val_ratio = int(
        st.session_state.get("gnn_split_val_ratio", VAL_RATIO)
    )
    split_test_ratio = int(
        st.session_state.get(
            "gnn_split_test_ratio",
            100 - split_train_ratio - split_val_ratio,
        )
    )
    st.caption(
        f"Split configurado: train={split_train_ratio}%, "
        f"val={split_val_ratio}%, test={split_test_ratio}%."
    )

    network_cfg = st.session_state.get("gnn_network_config")
    default_smote = bool(network_cfg.get("use_graphsmote", False)) if network_cfg else False
    use_graphsmote_train = st.checkbox(
        "Usar GraphSMOTE durante entrenamiento",
        value=default_smote,
        key="gnn_train_use_graphsmote",
    )
    if use_graphsmote_train and use_balanced:
        st.caption(
            "GraphSMOTE se aplicara sobre un grafo ya balanceado. "
            "Considere desactivarlo si no necesita doble aumentacion."
        )
    elif use_graphsmote_train:
        st.caption(
            "GraphSMOTE generara nodos sinteticos adicionales durante el entrenamiento."
        )
    else:
        st.caption(
            "Entrenamiento directo sobre el grafo seleccionado (sin SMOTE en entrenamiento)."
        )

    if network_cfg:
        st.caption("Configuracion de Network disponible para este entrenamiento.")
    hp_files = _list_hpo_files_for_training(
        use_graphsmote=use_graphsmote_train, graph_obj=graph_obj
    )
    hp_options = ["Auto (mas reciente)", "Nueva busqueda"]
    network_option = None
    if network_cfg:
        network_option = "Network (config actual)"
        hp_options.append(network_option)
    hp_options += [os.path.basename(path) for path in hp_files]
    hp_choice = st.selectbox(
        "Hiperparametros",
        hp_options,
        index=0,
        key="gnn_train_hp_choice",
    )
    if hp_choice == "Nueva busqueda":
        st.caption(
            "Se ejecutara Optuna para buscar hiperparametros desde cero."
        )
    elif hp_choice == "Auto (mas reciente)":
        st.caption(
            "Se reutilizara el archivo de hiperparametros mas reciente compatible."
        )
    elif network_option and hp_choice == network_option:
        st.caption(
            "Se exportara la configuracion del tab Network y se usara en el entrenamiento."
        )
    else:
        st.caption(f"Se reutilizaran los hiperparametros: {hp_choice}.")

    device_opts = ["cpu"]
    if torch.cuda.is_available():
        device_opts.append("cuda")
    if torch.backends.mps.is_available():
        device_opts.append("mps")
    eval_device = st.selectbox(
        "Dispositivo para evaluacion",
        device_opts,
        index=0,
        key="gnn_train_eval_device",
    )
    st.caption(f"Evaluacion se ejecutara en {eval_device}.")

    if st.button("Entrenar modelo GNN", key="gnn_train_run"):
        try:
            from src import gnn_main as graph_main
        except Exception as exc:
            st.error(f"No se pudo cargar el entrenador GNN: {exc}")
            return

        pm_index_ref = st.session_state.get("loaded_graph", {}).get("pm_index")
        split_info = None
        if pm_index_ref is not None:
            split_info = _apply_temporal_split_to_graph(
                graph_data,
                pm_index_ref,
                split_train_ratio,
                split_val_ratio,
            )
        if split_info:
            st.caption(
                "Split aplicado: "
                f"train={split_info['train_count']}, "
                f"val={split_info['val_count']}, "
                f"test={split_info['test_count']}"
            )
        else:
            st.warning(
                "No se pudo aplicar el split temporal; se usaran las mascaras actuales."
            )

        hp_files_for_train = list(hp_files)
        network_hparams_path = None
        if network_option and hp_choice == network_option:
            network_hparams_path = _save_network_hparams(
                network_cfg or {}, use_graphsmote=use_graphsmote_train
            )
            if not network_hparams_path:
                st.error("No se pudo exportar la configuracion de Network.")
                return
            st.session_state["gnn_network_hparams_path"] = network_hparams_path
            hp_files_for_train = _list_hpo_files_for_training(
                use_graphsmote=use_graphsmote_train, graph_obj=graph_obj
            )

        def _find_hp_index(paths: List[str], basename: str) -> Optional[int]:
            for idx, path in enumerate(paths, 1):
                if os.path.basename(path) == basename:
                    return idx
            return None

        hp_choice_index = 0
        reuse_hparams = True
        if hp_choice == "Nueva busqueda":
            hp_choice_index = 0
            reuse_hparams = False
        elif hp_choice == "Auto (mas reciente)":
            hp_choice_index = len(hp_files_for_train) if hp_files_for_train else 0
        elif network_option and hp_choice == network_option:
            if not network_hparams_path:
                hp_choice_index = 0
                reuse_hparams = False
            else:
                idx = _find_hp_index(
                    hp_files_for_train, os.path.basename(network_hparams_path)
                )
                if idx is None:
                    st.error(
                        "No se encontro el archivo de Network en la lista de hiperparametros."
                    )
                    return
                hp_choice_index = idx
        else:
            idx = _find_hp_index(hp_files_for_train, hp_choice)
            if idx is None:
                st.warning(
                    "No se pudo ubicar el archivo seleccionado; se ejecutara nueva busqueda."
                )
                hp_choice_index = 0
                reuse_hparams = False
            else:
                hp_choice_index = idx

        original_input = builtins.input

        def _auto_input(prompt: str = "") -> str:
            norm = (
                unicodedata.normalize("NFKD", prompt)
                .encode("ascii", "ignore")
                .decode("ascii")
                .lower()
            )
            if "seleccione el numero del archivo" in norm:
                return str(max(hp_choice_index, 0))
            if "reutilizar estos" in norm:
                return "s" if reuse_hparams else "n"
            return "0"

        builtins.input = _auto_input
        with st.spinner("Entrenando modelo GNN..."):
            try:
                graph_main.run_gat_training(
                    graph_obj,
                    force_use_graphsmote=bool(use_graphsmote_train),
                )
            except Exception as exc:
                st.error(f"Error en entrenamiento: {exc}")
                return
            finally:
                builtins.input = original_input

        model_path = _select_latest_gat_model(
            use_graphsmote=use_graphsmote_train, graph_obj=graph_obj
        )
        if model_path:
            st.session_state["gnn_train_last_model"] = model_path
            st.success(f"Entrenamiento finalizado. Modelo: {os.path.basename(model_path)}")
        else:
            st.warning("Entrenamiento completado, pero no se encontro modelo guardado.")

    model_files = sorted(
        glob.glob(os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt")),
        key=os.path.getmtime,
    )
    model_choice = st.selectbox(
        "Modelo para evaluar",
        options=["(auto)"] + model_files,
        format_func=lambda x: "(auto)" if x == "(auto)" else os.path.basename(x),
        key="gnn_train_model_eval",
    )
    if model_choice == "(auto)":
        model_choice = st.session_state.get("gnn_train_last_model")
    if model_choice:
        st.caption(f"Modelo seleccionado: {os.path.basename(model_choice)}")
    else:
        st.caption("No hay modelos disponibles para evaluar.")

    if st.button("Evaluar modelo", key="gnn_train_eval"):
        if not model_choice:
            st.error("Seleccione un modelo valido para evaluar.")
            return

        try:
            from src.gat_model import HeteroGAT
            from src.temporal_head import TemporalAggregator
            from src import gnn_main as graph_main
        except Exception as exc:
            st.error(f"No se pudo cargar dependencias de evaluacion: {exc}")
            return

        pm_index_ref = st.session_state.get("loaded_graph", {}).get("pm_index")
        split_info = None
        if pm_index_ref is not None:
            split_info = _apply_temporal_split_to_graph(
                graph_data,
                pm_index_ref,
                split_train_ratio,
                split_val_ratio,
            )
        if split_info:
            st.caption(
                "Split aplicado: "
                f"train={split_info['train_count']}, "
                f"val={split_info['val_count']}, "
                f"test={split_info['test_count']}"
            )
        else:
            st.warning(
                "No se pudo aplicar el split temporal; se usaran las mascaras actuales."
            )

        try:
            model_obj = torch.load(
                model_choice, map_location=torch.device("cpu"), weights_only=False
            )
        except Exception as exc:
            st.error(f"No se pudo cargar el modelo: {exc}")
            return

        state_dict = None
        model = None
        if isinstance(model_obj, torch.nn.Module):
            model = model_obj
        elif isinstance(model_obj, dict):
            if isinstance(model_obj.get("model"), torch.nn.Module):
                model = model_obj.get("model")
            else:
                state_dict = model_obj.get("model_state") or model_obj.get("state_dict")
                if state_dict is None:
                    try:
                        if model_obj and all(
                            isinstance(k, str) for k in model_obj.keys()
                        ) and all(
                            isinstance(v, torch.Tensor)
                            for v in model_obj.values()
                        ):
                            state_dict = model_obj
                    except Exception:
                        state_dict = None
        if model is None and state_dict is None:
            st.error("El modelo cargado no es compatible.")
            return

        hparams = _load_hparams_for_model(model_choice)
        if state_dict is None and model is not None:
            state_dict = model.state_dict()
        if state_dict is None:
            st.error("No se pudo cargar el state_dict del modelo.")
            return

        arch_defaults = _infer_arch_from_state_dict(state_dict)
        in_channels = int(graph_data["pm"].x.shape[1])
        edge_feature_dim = _infer_edge_feature_dim(graph_data)
        num_classes = int(torch.unique(graph_data["pm"].y).numel())
        hidden_channels = int(hparams.get("hidden_channels", arch_defaults["hidden_channels"]))
        num_heads = int(hparams.get("num_heads", arch_defaults["num_heads"]))
        num_layers = int(hparams.get("num_layers", arch_defaults["num_layers"]))
        dropout = float(hparams.get("dropout", 0.2))
        aggr1 = hparams.get("aggr1", "sum")
        aggr2 = hparams.get("aggr2", "sum")
        use_checkpointing = bool(hparams.get("use_checkpointing", False))

        if model is None:
            model = HeteroGAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=num_classes,
                num_heads=num_heads,
                dropout=dropout,
                edge_feature_dim=edge_feature_dim,
                num_layers=num_layers,
                aggr1=aggr1,
                aggr2=aggr2,
                use_checkpointing=use_checkpointing,
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                st.warning(
                    f"Estado cargado con diferencias. missing={len(missing)} unexpected={len(unexpected)}"
                )

        device = torch.device(eval_device)
        model = model.to(device)
        model.eval()

        sequence_index = graph_obj.get("sequence_index")
        if sequence_index and getattr(sequence_index, "sequence_rows", None) is not None:
            if getattr(sequence_index.sequence_rows, "size", 0):
                seq_rows = torch.as_tensor(sequence_index.sequence_rows, dtype=torch.long)
                target_rows = torch.as_tensor(sequence_index.target_rows, dtype=torch.long)
                temporal_module = TemporalAggregator(
                    sequence_rows=seq_rows,
                    target_rows=target_rows,
                    num_nodes=graph_data["pm"].num_nodes,
                    embedding_dim=hidden_channels * num_heads,
                    sequence_length=sequence_index.sequence_rows.shape[1],
                    hidden_dim=hidden_channels * num_heads,
                    num_classes=num_classes,
                ).to(device)
                temporal_module.reset_cache()
                temporal_module.eval()
                model.temporal_head = temporal_module

        with st.spinner("Evaluando modelo..."):
            try:
                results_raw = graph_main.test(
                    model,
                    graph_data,
                    batch_size=BATCH_SIZE,
                    node_type="pm",
                    threshold=None,
                )
            except Exception as exc:
                st.error(f"Error al evaluar el modelo: {exc}")
                return

        threshold = 0.5
        if "val_mask" in results_raw:
            y_val = results_raw["val_mask"]["true"].numpy()
            y_prob = results_raw["val_mask"]["probs"][:, 1].numpy()
            if np.unique(y_val).size > 1:
                try:
                    threshold, _ = graph_main.pick_tau_fbeta(
                        y_val, y_prob, beta=1.0
                    )
                    st.caption(
                        f"Umbral optimizado en validacion: {threshold:.4f}"
                    )
                except Exception:
                    st.caption(
                        "No se pudo optimizar umbral en validacion; se usa 0.5."
                    )
                    threshold = 0.5
            else:
                st.caption(
                    "Validacion con una sola clase; se usa umbral 0.5."
                )
        else:
            st.caption("No hay val_mask disponible; se usa umbral 0.5.")

        try:
            results = graph_main.test(
                model,
                graph_data,
                batch_size=BATCH_SIZE,
                node_type="pm",
                threshold=float(threshold),
            )
        except Exception as exc:
            st.error(f"Error al evaluar con umbral: {exc}")
            return

        metrics_rows = []
        confusion_tables = {}
        for split_name, res in results.items():
            cm = np.array(res["cm"])
            metrics = _compute_binary_metrics_from_cm(cm)
            metrics.update(
                {
                    "split": split_name.replace("_mask", ""),
                    "auc": float(res.get("auc") or 0.0),
                    "auprc": float(res.get("auprc") or 0.0),
                    "mcc": float(res.get("mcc") or 0.0),
                }
            )
            metrics_rows.append(metrics)
            confusion_tables[split_name] = cm

        if metrics_rows:
            st.markdown("**Metricas por split**")
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, width="stretch")
        else:
            st.warning("No se pudieron calcular metricas.")

        if "test_mask" in confusion_tables:
            cm = confusion_tables["test_mask"]
            st.markdown("**Matriz de confusion (test)**")
            cm_df = pd.DataFrame(
                cm,
                index=["Real 0", "Real 1"],
                columns=["Pred 0", "Pred 1"],
            )
            st.dataframe(cm_df, width="stretch")
        else:
            st.info("No hay test_mask disponible para matriz de confusion.")


def _render_network_tab() -> None:
    st.subheader("Network (GNN)")

    loaded_graph = st.session_state.get("loaded_graph")
    if loaded_graph is None:
        st.warning("No hay grafo cargado en memoria. Use la pestaña Graph.")
        return

    graph_data = loaded_graph.get("data")
    if not isinstance(graph_data, HeteroData):
        st.warning("El grafo seleccionado no es HeteroData.")
        return
    if "pm" not in graph_data.node_types:
        st.warning("El grafo no contiene nodos 'pm'.")
        return

    in_channels = int(graph_data["pm"].x.shape[1])
    edge_feature_dim = _infer_edge_feature_dim(graph_data)
    num_classes = (
        int(torch.unique(graph_data["pm"].y).numel())
        if hasattr(graph_data["pm"], "y")
        else 2
    )
    pm_nodes = int(graph_data["pm"].num_nodes)
    total_edges = int(
        sum(
            graph_data[edge_type].edge_index.shape[1]
            for edge_type in graph_data.edge_types
        )
    )

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Input features", in_channels)
    metrics_col2.metric("PM nodes", pm_nodes)
    metrics_col3.metric("Edges", total_edges)
    st.caption(
        f"edge_attr_dim={edge_feature_dim} | clases={num_classes} | tipos de aristas={len(graph_data.edge_types)}"
    )

    existing_cfg = st.session_state.get("gnn_network_config", {})

    st.markdown("**Parametros de red**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        hidden_channels = st.number_input(
            "hidden_channels",
            min_value=8,
            value=int(existing_cfg.get("hidden_channels", 64)),
            step=8,
            key="gnn_net_hidden",
        )
        num_layers = st.number_input(
            "num_layers",
            min_value=1,
            value=int(existing_cfg.get("num_layers", 2)),
            step=1,
            key="gnn_net_layers",
        )
        dropout = st.number_input(
            "dropout",
            min_value=0.0,
            max_value=0.9,
            value=float(existing_cfg.get("dropout", 0.2)),
            step=0.05,
            key="gnn_net_dropout",
        )
    with col_b:
        num_heads = st.number_input(
            "num_heads",
            min_value=1,
            value=int(existing_cfg.get("num_heads", 4)),
            step=1,
            key="gnn_net_heads",
        )
        aggr1 = st.selectbox(
            "aggr1",
            ["sum", "mean", "max"],
            index=["sum", "mean", "max"].index(
                str(existing_cfg.get("aggr1", "sum"))
            ),
            key="gnn_net_aggr1",
        )
        aggr2 = st.selectbox(
            "aggr2",
            ["sum", "mean", "max"],
            index=["sum", "mean", "max"].index(
                str(existing_cfg.get("aggr2", "sum"))
            ),
            key="gnn_net_aggr2",
        )
    with col_c:
        use_checkpointing = st.checkbox(
            "use_checkpointing",
            value=bool(existing_cfg.get("use_checkpointing", False)),
            key="gnn_net_checkpoint",
        )
        batch_size = st.number_input(
            "batch_size",
            min_value=32,
            value=int(existing_cfg.get("batch_size", 512)),
            step=32,
            key="gnn_net_batch",
        )

    st.markdown("**Vecindario (NeighborLoader)**")

    def _parse_int_list(text: str, fallback: List[int]) -> List[int]:
        try:
            items = [int(v.strip()) for v in text.split(",") if v.strip()]
            return items or fallback
        except Exception:
            return fallback

    neighbor_default = existing_cfg.get("num_neighbors", [15, 10])
    if isinstance(neighbor_default, str):
        try:
            neighbor_default = json.loads(neighbor_default)
        except Exception:
            neighbor_default = [15, 10]
    neighbor_text = st.text_input(
        "num_neighbors (por capa, ej: 15,10)",
        value=",".join(str(v) for v in neighbor_default),
        key="gnn_net_neighbors",
    )
    num_neighbors = _parse_int_list(neighbor_text, [15, 10])
    st.caption(f"Perfil activo: {num_neighbors}")

    st.markdown("**Optimizador y loss**")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        optimizer_name = st.selectbox(
            "optimizer",
            ["Adam", "AdamW", "RAdam"],
            index=["Adam", "AdamW", "RAdam"].index(
                str(existing_cfg.get("optimizer", "Adam"))
            ),
            key="gnn_net_optimizer",
        )
        lr = st.number_input(
            "lr",
            min_value=1e-6,
            max_value=1.0,
            value=float(existing_cfg.get("lr", 1e-3)),
            format="%.6f",
            key="gnn_net_lr",
        )
    with col_opt2:
        weight_decay = st.number_input(
            "weight_decay",
            min_value=1e-8,
            max_value=1e-1,
            value=float(existing_cfg.get("weight_decay", 1e-6)),
            format="%.8f",
            key="gnn_net_wd",
        )
        loss_type = st.selectbox(
            "loss_type",
            ["CrossEntropy", "FocalLoss"],
            index=["CrossEntropy", "FocalLoss"].index(
                str(existing_cfg.get("loss_type", "CrossEntropy"))
            ),
            key="gnn_net_loss",
        )
    with col_opt3:
        lambda_l2_att = st.number_input(
            "lambda_l2_att",
            min_value=1e-6,
            max_value=1.0,
            value=float(existing_cfg.get("lambda_l2_att", 1e-4)),
            format="%.6f",
            key="gnn_net_l2",
        )
        lambda_H_mode = st.selectbox(
            "lambda_H_mode",
            ["fixed", "dynamic"],
            index=["fixed", "dynamic"].index(
                str(existing_cfg.get("lambda_H_mode", "fixed"))
            ),
            key="gnn_net_lambda_mode",
        )

    focal_gamma = float(existing_cfg.get("focal_gamma", 2.0))
    focal_alpha = float(existing_cfg.get("focal_alpha", 0.75))
    if loss_type == "FocalLoss":
        col_fg, col_fa = st.columns(2)
        with col_fg:
            focal_gamma = st.number_input(
                "focal_gamma",
                min_value=0.5,
                max_value=5.0,
                value=float(focal_gamma),
                step=0.1,
                key="gnn_net_focal_gamma",
            )
        with col_fa:
            focal_alpha = st.number_input(
                "focal_alpha",
                min_value=0.1,
                max_value=0.99,
                value=float(focal_alpha),
                step=0.05,
                key="gnn_net_focal_alpha",
            )

    lambda_H_fixed = float(existing_cfg.get("lambda_H_fixed", 1e-4))
    initial_lambda_H = float(existing_cfg.get("initial_lambda_H", 1e-4))
    final_lambda_H = float(existing_cfg.get("final_lambda_H", 1e-2))
    if lambda_H_mode == "fixed":
        lambda_H_fixed = st.number_input(
            "lambda_H_fixed",
            min_value=1e-6,
            max_value=1.0,
            value=float(lambda_H_fixed),
            format="%.6f",
            key="gnn_net_lambda_fixed",
        )
    else:
        col_lh1, col_lh2 = st.columns(2)
        with col_lh1:
            initial_lambda_H = st.number_input(
                "initial_lambda_H",
                min_value=1e-6,
                max_value=1.0,
                value=float(initial_lambda_H),
                format="%.6f",
                key="gnn_net_lambda_init",
            )
        with col_lh2:
            final_lambda_H = st.number_input(
                "final_lambda_H",
                min_value=1e-6,
                max_value=1.0,
                value=float(final_lambda_H),
                format="%.6f",
                key="gnn_net_lambda_final",
            )

    st.markdown("**Balance durante entrenamiento**")
    use_graphsmote = st.checkbox(
        "Usar GraphSMOTE en entrenamiento",
        value=bool(existing_cfg.get("use_graphsmote", False)),
        key="gnn_net_use_smote",
    )
    smote_k = int(existing_cfg.get("smote_k", 5))
    target_pos_ratio = float(existing_cfg.get("target_pos_ratio", 0.35))
    smote_every_n_epochs = int(existing_cfg.get("smote_every_n_epochs", 2))
    lambda_edge = float(existing_cfg.get("lambda_edge", 1e-6))
    if use_graphsmote:
        col_sm1, col_sm2, col_sm3, col_sm4 = st.columns(4)
        with col_sm1:
            smote_k = st.number_input(
                "smote_k",
                min_value=1,
                value=int(smote_k),
                step=1,
                key="gnn_net_smote_k",
            )
        with col_sm2:
            target_pos_ratio = st.number_input(
                "target_pos_ratio",
                min_value=0.1,
                max_value=0.9,
                value=float(target_pos_ratio),
                step=0.05,
                key="gnn_net_target_ratio",
            )
        with col_sm3:
            smote_every_n_epochs = st.number_input(
                "smote_every_n_epochs",
                min_value=1,
                value=int(smote_every_n_epochs),
                step=1,
                key="gnn_net_smote_every",
            )
        with col_sm4:
            lambda_edge = st.number_input(
                "lambda_edge",
                min_value=1e-8,
                max_value=1e-2,
                value=float(lambda_edge),
                format="%.8f",
                key="gnn_net_lambda_edge",
            )

    config_payload = {
        "hidden_channels": int(hidden_channels),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "aggr1": aggr1,
        "aggr2": aggr2,
        "use_checkpointing": bool(use_checkpointing),
        "num_neighbors": num_neighbors,
        "optimizer": optimizer_name,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "loss_type": loss_type,
        "focal_gamma": float(focal_gamma),
        "focal_alpha": float(focal_alpha),
        "lambda_l2_att": float(lambda_l2_att),
        "lambda_H_mode": lambda_H_mode,
        "lambda_H_fixed": float(lambda_H_fixed),
        "initial_lambda_H": float(initial_lambda_H),
        "final_lambda_H": float(final_lambda_H),
        "batch_size": int(batch_size),
        "use_graphsmote": bool(use_graphsmote),
        "smote_k": int(smote_k),
        "target_pos_ratio": float(target_pos_ratio),
        "smote_every_n_epochs": int(smote_every_n_epochs),
        "lambda_edge": float(lambda_edge),
    }

    st.session_state["gnn_network_config"] = config_payload

    col_save, col_export = st.columns(2)
    with col_save:
        if st.button("Guardar configuracion", key="gnn_net_save"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(
                RESULTADOS_DIR, f"network_config_{timestamp}.json"
            )
            try:
                with open(out_path, "w") as fh:
                    json.dump(config_payload, fh, indent=2)
                st.session_state["gnn_network_config_path"] = out_path
                st.success("Configuracion guardada.")
            except Exception as exc:
                st.error(f"No se pudo guardar la configuracion: {exc}")
    with col_export:
        if st.button("Exportar a Training", key="gnn_net_export"):
            export_path = _save_network_hparams(
                config_payload, use_graphsmote=use_graphsmote
            )
            if export_path:
                st.session_state["gnn_network_hparams_path"] = export_path
                st.success(
                    f"Hiperparametros exportados: {os.path.basename(export_path)}"
                )
            else:
                st.error("No se pudo exportar la configuracion.")

    st.markdown("**Visualizacion de capas**")
    layer_rows = []
    emb_dim = hidden_channels * num_heads
    for idx in range(1, int(num_layers) + 1):
        layer_rows.append(
            {
                "layer": idx,
                "type": "GAT",
                "heads": int(num_heads),
                "hidden": int(hidden_channels),
                "emb_dim": int(emb_dim),
            }
        )
    st.dataframe(pd.DataFrame(layer_rows), width="stretch")

    dot_lines = [
        "digraph G {",
        "rankdir=LR;",
        "node [shape=box style=\"rounded,filled\" fillcolor=\"#E8F1FF\" color=\"#2C3E50\" fontname=\"Helvetica\"];",
        f"input [label=\"Input\\nF={in_channels}\"];",
    ]
    for idx in range(1, int(num_layers) + 1):
        dot_lines.append(
            f"layer{idx} [label=\"GAT {idx}\\nheads={num_heads}\\nhidden={hidden_channels}\\nemb={emb_dim}\"];"
        )
    dot_lines.append(f"output [label=\"Output\\nclasses={num_classes}\"];")
    dot_lines.append("input -> layer1;")
    if int(num_layers) > 1:
        for idx in range(1, int(num_layers)):
            dot_lines.append(f"layer{idx} -> layer{idx+1};")
        dot_lines.append(f"layer{num_layers} -> output;")
    else:
        dot_lines.append("layer1 -> output;")
    dot_lines.append("}")
    st.graphviz_chart("\n".join(dot_lines))

    st.markdown("**TensorBoard**")
    st.code("tensorboard --logdir Resultados/runs_attention", language="bash")

    st.markdown("**Vista torchview**")
    view_options = [
        "Completo (GAT + salida)",
        "Backbone (todas las capas)",
        "Backbone hasta capa N",
        "Solo capa N",
        "Solo salida (Linear)",
    ]
    view_choice = st.selectbox(
        "Seccion del modelo",
        view_options,
        index=0,
        key="gnn_net_torchview_section",
    )
    selected_layer = 1
    if view_choice in {"Backbone hasta capa N", "Solo capa N"}:
        selected_layer = st.slider(
            "Capas a visualizar",
            min_value=1,
            max_value=int(num_layers),
            value=min(2, int(num_layers)),
            step=1,
            key="gnn_net_torchview_layers",
        )

    try:
        from src.gat_model import HeteroGAT
        from torchview import draw_graph

        class _GATView(torch.nn.Module):
            def __init__(
                self,
                model: HeteroGAT,
                *,
                max_layers: int,
                single_layer: Optional[int] = None,
                include_output: bool = False,
            ) -> None:
                super().__init__()
                self.model = model
                self.max_layers = max_layers
                self.single_layer = single_layer
                self.include_output = include_output

            def forward(
                self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
            ) -> Dict[str, torch.Tensor]:
                if edge_attr_dict is None:
                    edge_attr_dict = {}
                x_out = x_dict
                if self.single_layer is not None:
                    layer_indices = [self.single_layer]
                else:
                    layer_indices = list(range(max(0, self.max_layers)))
                for idx in layer_indices:
                    conv = self.model.convs[idx]
                    norm = self.model.norms[idx]
                    active_edge_types = conv.convs.keys()
                    active_eid = {
                        k: v for k, v in edge_index_dict.items()
                        if k in active_edge_types
                    }
                    active_ead: Dict[Tuple[str, str, str], torch.Tensor] = {}
                    for k in active_edge_types:
                        if k in edge_attr_dict and edge_attr_dict[k] is not None:
                            active_ead[k] = edge_attr_dict[k]
                        elif (
                            self.model.edge_feature_dim
                            and self.model.edge_feature_dim > 0
                            and k in active_eid
                        ):
                            num_e = active_eid[k].shape[1]
                            ref = next(iter(x_out.values()))
                            active_ead[k] = torch.zeros(
                                (num_e, self.model.edge_feature_dim),
                                dtype=ref.dtype,
                                device=ref.device,
                            )
                    x_out = conv(x_out, active_eid, active_ead)
                    x_out = {key: norm[key](val) for key, val in x_out.items()}
                    x_out = {key: torch.relu(val) for key, val in x_out.items()}
                    x_out = {
                        key: torch.nn.functional.dropout(
                            val,
                            p=self.model.dropout,
                            training=self.training,
                        )
                        for key, val in x_out.items()
                    }
                if self.include_output and "pm" in x_out:
                    x_out["pm"] = self.model.pm_lin(x_out["pm"])
                return x_out

        dummy_nodes = max(4, min(12, pm_nodes))
        input_dim = in_channels
        if view_choice == "Solo salida (Linear)":
            input_dim = hidden_channels * num_heads
        elif view_choice == "Solo capa N" and selected_layer > 1:
            input_dim = hidden_channels * num_heads
        x_dict = {"pm": torch.zeros((dummy_nodes, int(input_dim)))}
        edge_index = torch.tensor(
            [list(range(dummy_nodes - 1)), list(range(1, dummy_nodes))],
            dtype=torch.long,
        )
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index_dict = {
            ("pm", "spatial", "pm"): edge_index,
            ("pm", "temporal", "pm"): edge_index,
        }
        model = HeteroGAT(
            in_channels=in_channels,
            hidden_channels=int(hidden_channels),
            out_channels=int(num_classes),
            num_heads=int(num_heads),
            dropout=float(dropout),
            edge_feature_dim=int(edge_feature_dim),
            num_layers=int(num_layers),
            aggr1=aggr1,
            aggr2=aggr2,
            use_checkpointing=bool(use_checkpointing),
        )
        model.eval()

        if view_choice == "Completo (GAT + salida)":
            view_model = _GATView(
                model,
                max_layers=int(num_layers),
                include_output=True,
            )
        elif view_choice == "Backbone (todas las capas)":
            view_model = _GATView(
                model,
                max_layers=int(num_layers),
                include_output=False,
            )
        elif view_choice == "Backbone hasta capa N":
            view_model = _GATView(
                model,
                max_layers=int(selected_layer),
                include_output=False,
            )
        elif view_choice == "Solo capa N":
            view_model = _GATView(
                model,
                max_layers=int(num_layers),
                single_layer=int(selected_layer - 1),
                include_output=False,
            )
        else:
            view_model = _GATView(
                model,
                max_layers=0,
                include_output=True,
            )

        graph = draw_graph(
            view_model,
            input_data=(x_dict, edge_index_dict),
            expand_nested=True,
            device="cpu",
        )
        st.graphviz_chart(graph.visual_graph)
    except Exception:
        st.info(
            "torchview no esta disponible o no pudo generar la grafica. "
            "Se muestra la vista Graphviz."
        )


def _render_optuna_tab() -> None:
    st.subheader("Optuna (Balance + Training)")

    loaded_graph = st.session_state.get("loaded_graph")
    balanced_graph = st.session_state.get("balanced_graph")
    if loaded_graph is None:
        st.warning("No hay grafo cargado en memoria. Use la pestaña Graph.")
        return

    use_balanced = False
    if balanced_graph is not None:
        use_balanced = st.checkbox(
            "Usar grafo balanceado",
            value=True,
            key="gnn_optuna_use_balanced",
        )
        if use_balanced:
            st.caption(
                f"Se optimiza con grafo balanceado ({balanced_graph.get('source', 'N/A')})."
            )
        else:
            st.caption("Se optimiza con el grafo original.")
    else:
        st.info("No hay grafo balanceado disponible; se usara el grafo original.")

    graph_obj = dict(loaded_graph)
    graph_source_label = "Original"
    if use_balanced and balanced_graph is not None:
        graph_obj["data"] = balanced_graph.get("data")
        graph_source_label = f"Balanceado ({balanced_graph.get('source', 'N/A')})"
    if not graph_obj.get("filename"):
        graph_path = st.session_state.get("graph_path")
        if graph_path:
            graph_obj["filename"] = os.path.basename(graph_path)

    graph_data = graph_obj.get("data")
    if not isinstance(graph_data, HeteroData):
        st.warning("El grafo seleccionado no es HeteroData.")
        return
    if "pm" not in graph_data.node_types:
        st.warning("El grafo no contiene nodos 'pm'.")
        return
    if not hasattr(graph_data["pm"], "train_mask"):
        st.warning("El grafo no contiene train_mask.")
        return
    if not hasattr(graph_data["pm"], "y"):
        st.warning("El grafo no contiene etiquetas 'y'.")
        return

    st.markdown(
        f"**Grafo seleccionado:** {graph_source_label}  \n"
        f"- Nodos PM: {int(graph_data['pm'].num_nodes)}  \n"
        f"- Aristas: {int(sum(graph_data[edge_type].edge_index.shape[1] for edge_type in graph_data.edge_types))}"
    )
    st.caption(
        "Optuna busca hiperparametros para entrenamiento y, si se activa, "
        "incluye parametros de GraphSMOTE para balancear durante el entrenamiento."
    )

    st.markdown("### Estrategia de Balanceo")
    balancing_strategy = st.radio(
        "Seleccione la estrategia de balanceo:",
        [
            "Sin balancear",
            "Opt. balance con GraphSMOTE",
            "Opt. balance con ImGAGN",
        ],
        index=0,
        key="gnn_balancing_strategy",
    )

    # Variables de control para pasar a search
    use_graphsmote_search = (balancing_strategy == "Opt. balance con GraphSMOTE")
    use_imgagn_search = (balancing_strategy == "Opt. balance con ImGAGN")

    # Inicializar contenedores de parametros para evitar errores de referencia
    smote_k_min = smote_k_max = smote_k_step = 0
    target_pos_ratio = []
    smote_every_n_epochs = []
    lambda_edge_min = lambda_edge_max = 0.0

    imgagn_dz_min = imgagn_dz_max = imgagn_dz_step = 0
    imgagn_hidden_g_min = imgagn_hidden_g_max = imgagn_hidden_g_step = 0
    imgagn_topk_min = imgagn_topk_max = imgagn_topk_step = 0
    imgagn_lambda1_min = imgagn_lambda1_max = imgagn_lambda1_step = 0.0
    imgagn_lr_g_min = imgagn_lr_g_max = 0.0
    imgagn_lr_d_min = imgagn_lr_d_max = 0.0
    imgagn_epochs_min = imgagn_epochs_max = imgagn_epochs_step = 0
    imgagn_alpha_min = imgagn_alpha_max = 0.0
    imgagn_beta_min = imgagn_beta_max = 0.0

    if use_graphsmote_search:
        st.caption(
            "La busqueda explorara smote_k, target_pos_ratio, smote_every_n_epochs y lambda_edge."
        )
        with st.expander("Parametros GraphSMOTE"):
            smote_k_min = st.number_input(
                "smote_k min",
                min_value=1,
                value=3,
                step=1,
                key="gnn_optuna_smote_k_min",
            )
            smote_k_max = st.number_input(
                "smote_k max",
                min_value=1,
                value=7,
                step=1,
                key="gnn_optuna_smote_k_max",
            )
            smote_k_step = st.number_input(
                "smote_k step",
                min_value=1,
                value=2,
                step=1,
                key="gnn_optuna_smote_k_step",
            )

            def _parse_float_list(text: str, fallback: List[float]) -> List[float]:
                try:
                    items = [
                        float(v.strip())
                        for v in text.split(",")
                        if v.strip()
                    ]
                    return items or fallback
                except Exception:
                    return fallback

            target_ratio_txt = st.text_input(
                "target_pos_ratio opciones",
                value="0.25,0.35,0.5",
                key="gnn_optuna_target_ratio",
            )
            smote_every_txt = st.text_input(
                "smote_every_n_epochs opciones",
                value="2,4,6",
                key="gnn_optuna_smote_every",
            )
            lambda_edge_min = st.number_input(
                "lambda_edge min (log)",
                min_value=1e-8,
                max_value=1e-2,
                value=1e-7,
                format="%.8f",
                key="gnn_optuna_lambda_edge_min",
            )
            lambda_edge_max = st.number_input(
                "lambda_edge max (log)",
                min_value=1e-8,
                max_value=1e-2,
                value=1e-4,
                format="%.6f",
                key="gnn_optuna_lambda_edge_max",
            )
            target_pos_ratio = _parse_float_list(
                target_ratio_txt, [0.25, 0.35, 0.5]
            )
            smote_every_n_epochs = [
                int(v.strip())
                for v in smote_every_txt.split(",")
                if v.strip()
            ] or [2, 4, 6]
    elif use_imgagn_search:
        st.caption(
            "La busqueda explorara parametros del generador ImGAGN (dz, hidden, topk, lambda1) y su entrenamiento (lr, epochs)."
        )
        with st.expander("Parametros ImGAGN"):
            col_z_h, col_tk_l1 = st.columns(2)
            with col_z_h:
                st.caption("Generador (Structure)")
                imgagn_dz_min = st.number_input(
                    "dz min", min_value=16, max_value=256, value=64, step=16, key="imgagn_dz_min"
                )
                imgagn_dz_max = st.number_input(
                    "dz max", min_value=16, max_value=256, value=128, step=16, key="imgagn_dz_max"
                )
                imgagn_dz_step = st.number_input(
                    "dz step", min_value=1, max_value=64, value=32, step=1, key="imgagn_dz_step"
                )
                
                imgagn_hidden_g_min = st.number_input(
                    "hidden_g min", min_value=32, max_value=512, value=128, step=32, key="imgagn_hg_min"
                )
                imgagn_hidden_g_max = st.number_input(
                    "hidden_g max", min_value=32, max_value=512, value=256, step=32, key="imgagn_hg_max"
                )
                imgagn_hidden_g_step = st.number_input(
                    "hidden_g step", min_value=1, max_value=128, value=64, step=1, key="imgagn_hg_step"
                )

            with col_tk_l1:
                st.caption("Topologia y Ratio")
                imgagn_topk_min = st.number_input(
                    "topk min", min_value=1, max_value=20, value=3, step=1, key="imgagn_topk_min"
                )
                imgagn_topk_max = st.number_input(
                    "topk max", min_value=1, max_value=20, value=7, step=1, key="imgagn_topk_max"
                )
                imgagn_topk_step = st.number_input(
                    "topk step", min_value=1, max_value=5, value=2, step=1, key="imgagn_topk_step"
                )

                imgagn_lambda1_min = st.number_input(
                    "lambda1_ratio min", min_value=0.1, max_value=2.0, value=0.8, step=0.1, key="imgagn_l1_min"
                )
                imgagn_lambda1_max = st.number_input(
                    "lambda1_ratio max", min_value=0.1, max_value=2.0, value=1.2, step=0.1, key="imgagn_l1_max"
                )
                imgagn_lambda1_step = st.number_input(
                    "lambda1_ratio step", min_value=0.1, max_value=0.5, value=0.2, step=0.1, key="imgagn_l1_step"
                )

            st.markdown("---")
            col_lr, col_reg = st.columns(2)
            with col_lr:
                st.caption("Training (ImGAGN)")
                imgagn_epochs_min = st.number_input("epochs min", value=50, step=10, key="imgagn_ep_min")
                imgagn_epochs_max = st.number_input("epochs max", value=150, step=10, key="imgagn_ep_max")
                imgagn_epochs_step = st.number_input("epochs step", value=50, step=10, key="imgagn_ep_step")
                
                imgagn_lr_g_min = st.number_input("lr_g min (log)", value=1e-4, format="%.6f", key="imgagn_lrg_min")
                imgagn_lr_g_max = st.number_input("lr_g max (log)", value=1e-2, format="%.4f", key="imgagn_lrg_max")
                
                imgagn_lr_d_min = st.number_input("lr_d min (log)", value=1e-4, format="%.6f", key="imgagn_lrd_min")
                imgagn_lr_d_max = st.number_input("lr_d max (log)", value=1e-2, format="%.4f", key="imgagn_lrd_max")
            
            with col_reg:
                st.caption("Regularization (ImGAGN)")
                imgagn_alpha_min = st.number_input("alpha_reg min (log)", value=1e-6, format="%.6f", key="imgagn_areg_min")
                imgagn_alpha_max = st.number_input("alpha_reg max (log)", value=1e-3, format="%.4f", key="imgagn_areg_max")
                
                imgagn_beta_min = st.number_input("beta_reg min (log)", value=1e-6, format="%.6f", key="imgagn_breg_min")
                imgagn_beta_max = st.number_input("beta_reg max (log)", value=1e-3, format="%.4f", key="imgagn_breg_max")
    else:
        st.caption("La busqueda se centra solo en hiperparametros del GNN.")

    metric_options = [
        "F1",
        "Recall",
        "FAR",
        "Recall-FAR",
        "F0.5",
        "AUPRC",
        "MCC",
        "Accuracy",
    ]
    objective_metric = st.selectbox(
        "Objetivo de optimizacion",
        metric_options,
        index=0,
        key="gnn_optuna_metric",
    )
    threshold_beta = st.number_input(
        "Beta para el umbral (F-beta)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key="gnn_optuna_beta",
    )
    st.caption(
        "El umbral se optimiza en validacion usando F-beta; "
        "luego se calcula la metrica seleccionada."
    )

    st.markdown("**Split temporal (train/val/test)**")
    default_train = int(st.session_state.get("gnn_split_train_ratio", TRAIN_RATIO))
    default_val = int(st.session_state.get("gnn_split_val_ratio", VAL_RATIO))
    train_ratio = st.slider(
        "Train (%)",
        min_value=50,
        max_value=95,
        value=default_train,
        step=1,
        key="gnn_optuna_split_train",
    )
    max_val_allowed = max(1, 100 - train_ratio - 1)
    val_ratio = st.slider(
        "Val (%)",
        min_value=1,
        max_value=max_val_allowed,
        value=min(default_val, max_val_allowed),
        step=1,
        key="gnn_optuna_split_val",
    )
    test_ratio = 100 - train_ratio - val_ratio
    st.caption(f"Test (%): {test_ratio}")
    if test_ratio < 1:
        st.warning("El split deja menos de 1% para test.")

    st.session_state["gnn_split_train_ratio"] = int(train_ratio)
    st.session_state["gnn_split_val_ratio"] = int(val_ratio)
    st.session_state["gnn_split_test_ratio"] = int(test_ratio)

    pm_index_ref = loaded_graph.get("pm_index")
    ts_preview = None
    if pm_index_ref is not None:
        ts_preview = _extract_ts_min_from_pm_index(
            pm_index_ref, int(graph_data["pm"].num_nodes)
        )
    if ts_preview is None:
        st.warning("No se pudo calcular cortes temporales (pm_index no disponible).")
    else:
        cutoffs = _compute_temporal_cutoffs_from_ts(
            ts_preview, int(train_ratio), int(val_ratio)
        )
        if cutoffs is None:
            st.warning("No hay suficientes timestamps para dividir temporalmente.")
        else:
            train_cutoff, val_cutoff = cutoffs
            train_date = pd.to_datetime(train_cutoff, unit="m")
            val_date = pd.to_datetime(val_cutoff, unit="m")
            st.markdown(
                f"- Fin train: {train_date}  \n"
                f"- Fin val: {val_date}  \n"
                f"- Test: > {val_date}"
            )

    st.markdown("**Configuracion de Optuna**")
    col_trials, col_epochs, col_eval = st.columns(3)
    with col_trials:
        n_trials = st.number_input(
            "Numero de trials",
            min_value=1,
            value=int(N_TRIALS),
            step=1,
            key="gnn_optuna_trials",
        )
    with col_epochs:
        num_epochs = st.number_input(
            "Epochs por trial",
            min_value=1,
            value=int(NUM_EPOCHS_OPTUNA),
            step=1,
            key="gnn_optuna_epochs",
        )
    with col_eval:
        eval_every = st.number_input(
            "Evaluar cada N epochs",
            min_value=1,
            value=2,
            step=1,
            key="gnn_optuna_eval_every",
        )

    col_seed, col_pruner, col_device = st.columns(3)
    with col_seed:
        optuna_seed = st.number_input(
            "Seed Optuna",
            min_value=0,
            value=int(SEED),
            step=1,
            key="gnn_optuna_seed",
        )
    with col_pruner:
        pruner_type = st.selectbox(
            "Pruner",
            ["Hyperband", "Median", "Ninguno"],
            index=0,
            key="gnn_optuna_pruner",
        )
    with col_device:
        device_opts = ["cpu"]
        if torch.cuda.is_available():
            device_opts.append("cuda")
        if torch.backends.mps.is_available():
            device_opts.append("mps")
        optuna_device = st.selectbox(
            "Dispositivo",
            device_opts,
            index=0,
            key="gnn_optuna_device",
        )

    multivariate = st.checkbox(
        "Sampler multivariate",
        value=True,
        key="gnn_optuna_multivariate",
    )
    group = st.checkbox(
        "Sampler group",
        value=True,
        key="gnn_optuna_group",
    )

    if pruner_type == "Hyperband":
        col_min, col_max, col_red = st.columns(3)
        with col_min:
            min_resource = st.number_input(
                "min_resource",
                min_value=1,
                value=3,
                step=1,
                key="gnn_optuna_min_resource",
            )
        with col_max:
            max_resource = st.number_input(
                "max_resource",
                min_value=1,
                value=int(num_epochs),
                step=1,
                key="gnn_optuna_max_resource",
            )
        with col_red:
            reduction_factor = st.number_input(
                "reduction_factor",
                min_value=2,
                value=3,
                step=1,
                key="gnn_optuna_reduction",
            )
        n_warmup_steps = 0
    elif pruner_type == "Median":
        min_resource = 1
        max_resource = int(num_epochs)
        reduction_factor = 3
        n_warmup_steps = st.number_input(
            "n_warmup_steps",
            min_value=0,
            value=2,
            step=1,
            key="gnn_optuna_warmup",
        )
    else:
        min_resource = 1
        max_resource = int(num_epochs)
        reduction_factor = 3
        n_warmup_steps = 0

    st.markdown("**Espacio de busqueda**")
    with st.expander("Arquitectura"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hidden_min = st.number_input(
                "hidden_channels min",
                min_value=1,
                value=32,
                step=1,
                key="gnn_optuna_hidden_min",
            )
            hidden_max = st.number_input(
                "hidden_channels max",
                min_value=1,
                value=128,
                step=1,
                key="gnn_optuna_hidden_max",
            )
            hidden_step = st.number_input(
                "hidden_channels step",
                min_value=1,
                value=32,
                step=1,
                key="gnn_optuna_hidden_step",
            )
        with col2:
            heads_min = st.number_input(
                "num_heads min",
                min_value=1,
                value=2,
                step=1,
                key="gnn_optuna_heads_min",
            )
            heads_max = st.number_input(
                "num_heads max",
                min_value=1,
                value=8,
                step=1,
                key="gnn_optuna_heads_max",
            )
            heads_step = st.number_input(
                "num_heads step",
                min_value=1,
                value=2,
                step=1,
                key="gnn_optuna_heads_step",
            )
        with col3:
            layers_min = st.number_input(
                "num_layers min",
                min_value=1,
                value=2,
                step=1,
                key="gnn_optuna_layers_min",
            )
            layers_max = st.number_input(
                "num_layers max",
                min_value=1,
                value=5,
                step=1,
                key="gnn_optuna_layers_max",
            )
            layers_step = st.number_input(
                "num_layers step",
                min_value=1,
                value=1,
                step=1,
                key="gnn_optuna_layers_step",
            )

        col4, col5 = st.columns(2)
        with col4:
            dropout_min = st.number_input(
                "dropout min",
                min_value=0.0,
                max_value=0.9,
                value=0.05,
                step=0.01,
                key="gnn_optuna_dropout_min",
            )
            dropout_max = st.number_input(
                "dropout max",
                min_value=0.0,
                max_value=0.9,
                value=0.6,
                step=0.01,
                key="gnn_optuna_dropout_max",
            )
            dropout_step = st.number_input(
                "dropout step",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                key="gnn_optuna_dropout_step",
            )
        with col5:
            checkpoint_choices = st.multiselect(
                "use_checkpointing",
                [False, True],
                default=[False, True],
                key="gnn_optuna_checkpoint",
            )
        aggr_choices = ["sum", "mean", "max"]
        aggr1 = st.multiselect(
            "aggr1 opciones",
            aggr_choices,
            default=aggr_choices,
            key="gnn_optuna_aggr1",
        )
        aggr2 = st.multiselect(
            "aggr2 opciones",
            aggr_choices,
            default=aggr_choices,
            key="gnn_optuna_aggr2",
        )

    with st.expander("Perfiles de vecinos"):
        st.caption(
            "Define perfiles para NeighborLoader. Cada perfil es una lista por capa."
        )

        def _parse_int_list(text: str, fallback: List[int]) -> List[int]:
            try:
                items = [int(v.strip()) for v in text.split(",") if v.strip()]
                return items or fallback
            except Exception:
                return fallback

        profile_compact = st.text_input(
            "compact",
            value="15,10",
            key="gnn_optuna_neighbor_compact",
        )
        profile_focused = st.text_input(
            "focused",
            value="10,5",
            key="gnn_optuna_neighbor_focused",
        )
        profile_broad = st.text_input(
            "broad",
            value="25,15,10",
            key="gnn_optuna_neighbor_broad",
        )
        profile_wide = st.text_input(
            "wide",
            value="30,20",
            key="gnn_optuna_neighbor_wide",
        )
        neighbor_profiles = {
            "compact": _parse_int_list(profile_compact, [15, 10]),
            "focused": _parse_int_list(profile_focused, [10, 5]),
            "broad": _parse_int_list(profile_broad, [25, 15, 10]),
            "wide": _parse_int_list(profile_wide, [30, 20]),
        }
        neighbor_choices = st.multiselect(
            "Perfiles habilitados",
            list(neighbor_profiles.keys()),
            default=list(neighbor_profiles.keys()),
            key="gnn_optuna_neighbor_choices",
        )
        if not neighbor_choices:
            st.warning("Debe seleccionar al menos un perfil de vecinos.")

    with st.expander("Optimizador"):
        col_lr, col_wd = st.columns(2)
        with col_lr:
            lr_min = st.number_input(
                "lr min (log)",
                min_value=1e-6,
                max_value=1.0,
                value=5e-5,
                format="%.6f",
                key="gnn_optuna_lr_min",
            )
            lr_max = st.number_input(
                "lr max (log)",
                min_value=1e-6,
                max_value=1.0,
                value=1e-2,
                format="%.6f",
                key="gnn_optuna_lr_max",
            )
        with col_wd:
            wd_min = st.number_input(
                "weight_decay min (log)",
                min_value=1e-8,
                max_value=1e-1,
                value=1e-6,
                format="%.8f",
                key="gnn_optuna_wd_min",
            )
            wd_max = st.number_input(
                "weight_decay max (log)",
                min_value=1e-8,
                max_value=1e-1,
                value=5e-3,
                format="%.6f",
                key="gnn_optuna_wd_max",
            )
        optimizer_choices = st.multiselect(
            "Optimizadores",
            ["Adam", "AdamW", "RAdam"],
            default=["Adam", "AdamW", "RAdam"],
            key="gnn_optuna_optimizers",
        )

    with st.expander("Regularizacion"):
        lambda_l2_min = st.number_input(
            "lambda_l2_att min (log)",
            min_value=1e-6,
            max_value=1.0,
            value=1e-4,
            format="%.6f",
            key="gnn_optuna_l2_min",
        )
        lambda_l2_max = st.number_input(
            "lambda_l2_att max (log)",
            min_value=1e-6,
            max_value=1.0,
            value=1e-1,
            format="%.4f",
            key="gnn_optuna_l2_max",
        )
        lambda_H_modes = st.multiselect(
            "lambda_H_mode",
            ["fixed", "dynamic"],
            default=["fixed", "dynamic"],
            key="gnn_optuna_lambda_mode",
        )
        col_fixed, col_dyn = st.columns(2)
        with col_fixed:
            lambda_H_min = st.number_input(
                "lambda_H_fixed min (log)",
                min_value=1e-6,
                max_value=1.0,
                value=1e-4,
                format="%.6f",
                key="gnn_optuna_lambda_fixed_min",
            )
            lambda_H_max = st.number_input(
                "lambda_H_fixed max (log)",
                min_value=1e-6,
                max_value=1.0,
                value=5e-2,
                format="%.4f",
                key="gnn_optuna_lambda_fixed_max",
            )
        with col_dyn:
            lambda_init_min = st.number_input(
                "initial_lambda_H min (log)",
                min_value=1e-6,
                max_value=1.0,
                value=1e-4,
                format="%.6f",
                key="gnn_optuna_lambda_init_min",
            )
            lambda_init_max = st.number_input(
                "initial_lambda_H max (log)",
                min_value=1e-6,
                max_value=1.0,
                value=1e-2,
                format="%.6f",
                key="gnn_optuna_lambda_init_max",
            )
            lambda_final_min = st.number_input(
                "final_lambda_H min (log)",
                min_value=1e-6,
                max_value=1.0,
                value=1e-2,
                format="%.6f",
                key="gnn_optuna_lambda_final_min",
            )
            lambda_final_max = st.number_input(
                "final_lambda_H max (log)",
                min_value=1e-6,
                max_value=1.0,
                value=5e-2,
                format="%.4f",
                key="gnn_optuna_lambda_final_max",
            )

    with st.expander("Loss"):
        loss_types = st.multiselect(
            "loss_type",
            ["CrossEntropy", "FocalLoss"],
            default=["CrossEntropy", "FocalLoss"],
            key="gnn_optuna_loss_type",
        )
        col_fg, col_fa = st.columns(2)
        with col_fg:
            focal_gamma_min = st.number_input(
                "focal_gamma min",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="gnn_optuna_fg_min",
            )
            focal_gamma_max = st.number_input(
                "focal_gamma max",
                min_value=0.5,
                max_value=5.0,
                value=3.0,
                step=0.1,
                key="gnn_optuna_fg_max",
            )
            focal_gamma_step = st.number_input(
                "focal_gamma step",
                min_value=0.1,
                max_value=1.0,
                value=0.1,
                step=0.1,
                key="gnn_optuna_fg_step",
            )
        with col_fa:
            focal_alpha_min = st.number_input(
                "focal_alpha min",
                min_value=0.1,
                max_value=0.99,
                value=0.5,
                step=0.05,
                key="gnn_optuna_fa_min",
            )
            focal_alpha_max = st.number_input(
                "focal_alpha max",
                min_value=0.1,
                max_value=0.99,
                value=0.95,
                step=0.05,
                key="gnn_optuna_fa_max",
            )
            focal_alpha_step = st.number_input(
                "focal_alpha step",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                key="gnn_optuna_fa_step",
            )

    with st.expander("Batch"):
        batch_min = st.number_input(
            "batch_size min",
            min_value=1,
            value=512,
            step=1,
            key="gnn_optuna_batch_min",
        )
        batch_max = st.number_input(
            "batch_size max",
            min_value=1,
            value=4096,
            step=1,
            key="gnn_optuna_batch_max",
        )
        batch_step = st.number_input(
            "batch_size step",
            min_value=1,
            value=512,
            step=1,
            key="gnn_optuna_batch_step",
        )

    if st.button("Ejecutar Optuna", key="gnn_optuna_run"):
        if not neighbor_choices:
            st.error("Seleccione al menos un perfil de vecinos.")
            return
        if not aggr1 or not aggr2:
            st.error("Seleccione opciones de agregacion para aggr1 y aggr2.")
            return
        if not loss_types:
            st.error("Seleccione al menos un loss_type.")
            return
        if not optimizer_choices:
            st.error("Seleccione al menos un optimizador.")
            return
        if not checkpoint_choices:
            st.error("Seleccione al menos un valor para use_checkpointing.")
            return
        if not lambda_H_modes:
            st.error("Seleccione al menos un lambda_H_mode.")
            return

        split_info = None
        if pm_index_ref is not None:
            split_info = _apply_temporal_split_to_graph(
                graph_data,
                pm_index_ref,
                int(train_ratio),
                int(val_ratio),
            )
        if split_info:
            st.caption(
                "Split aplicado: "
                f"train={split_info['train_count']}, "
                f"val={split_info['val_count']}, "
                f"test={split_info['test_count']}"
            )
        else:
            st.warning(
                "No se pudo aplicar el split temporal; se usaran las mascaras actuales."
            )

        search_space = {
            "hidden_channels": {
                "min": hidden_min,
                "max": hidden_max,
                "step": hidden_step,
            },
            "num_heads": {
                "min": heads_min,
                "max": heads_max,
                "step": heads_step,
            },
            "num_layers": {
                "min": layers_min,
                "max": layers_max,
                "step": layers_step,
            },
            "dropout": {
                "min": dropout_min,
                "max": dropout_max,
                "step": dropout_step,
            },
            "aggr1": aggr1,
            "aggr2": aggr2,
            "use_checkpointing": checkpoint_choices,
            "neighbor_profiles": neighbor_profiles,
            "neighbor_choices": neighbor_choices,
            "lr": {"min": lr_min, "max": lr_max},
            "weight_decay": {"min": wd_min, "max": wd_max},
            "lambda_l2_att": {"min": lambda_l2_min, "max": lambda_l2_max},
            "lambda_H_mode": lambda_H_modes,
            "lambda_H_fixed": {
                "min": lambda_H_min,
                "max": lambda_H_max,
            },
            "initial_lambda_H": {
                "min": lambda_init_min,
                "max": lambda_init_max,
            },
            "final_lambda_H": {
                "min": lambda_final_min,
                "max": lambda_final_max,
            },
            "loss_type": loss_types,
            "focal_gamma": {
                "min": focal_gamma_min,
                "max": focal_gamma_max,
                "step": focal_gamma_step,
            },
            "focal_alpha": {
                "min": focal_alpha_min,
                "max": focal_alpha_max,
                "step": focal_alpha_step,
            },
            "batch_size": {
                "min": batch_min,
                "max": batch_max,
                "step": batch_step,
            },
            "optimizers": optimizer_choices,
            "smote_k": {
                "min": smote_k_min,
                "max": smote_k_max,
                "step": smote_k_step,
            },
            "target_pos_ratio": target_pos_ratio,
            "smote_every_n_epochs": smote_every_n_epochs,
            "lambda_edge": {
                "min": lambda_edge_min,
                "max": lambda_edge_max,
            },
            # ImGAGN parameters
            "imgagn_dz": {"min": imgagn_dz_min, "max": imgagn_dz_max, "step": imgagn_dz_step},
            "imgagn_hidden_g": {"min": imgagn_hidden_g_min, "max": imgagn_hidden_g_max, "step": imgagn_hidden_g_step},
            "imgagn_topk": {"min": imgagn_topk_min, "max": imgagn_topk_max, "step": imgagn_topk_step},
            "imgagn_lambda1": {"min": imgagn_lambda1_min, "max": imgagn_lambda1_max, "step": imgagn_lambda1_step},
            "imgagn_epochs": {"min": imgagn_epochs_min, "max": imgagn_epochs_max, "step": imgagn_epochs_step},
            "imgagn_lr_g": {"min": imgagn_lr_g_min, "max": imgagn_lr_g_max},
            "imgagn_lr_d": {"min": imgagn_lr_d_min, "max": imgagn_lr_d_max},
            "imgagn_alpha_reg": {"min": imgagn_alpha_min, "max": imgagn_alpha_max},
            "imgagn_beta_reg": {"min": imgagn_beta_min, "max": imgagn_beta_max},
        }
        optuna_settings = {
            "n_trials": n_trials,
            "epochs": num_epochs,
            "eval_every": eval_every,
            "seed": optuna_seed,
            "pruner": pruner_type,
            "min_resource": min_resource,
            "max_resource": max_resource,
            "reduction_factor": reduction_factor,
            "n_warmup_steps": n_warmup_steps,
            "multivariate": multivariate,
            "group": group,
        }
        objective_settings = {
            "metric": objective_metric,
            "threshold_beta": threshold_beta,
            "device": optuna_device,
        }

        with st.spinner("Ejecutando Optuna..."):
            try:
                # Create a container for logs
                with st.expander("Logs de Optuna (En vivo)", expanded=True):
                    log_container = st.empty()
                
                result = _run_optuna_search(
                    graph_obj=graph_obj,
                    balancing_strategy=balancing_strategy,
                    search_space=search_space,
                    optuna_settings=optuna_settings,
                    objective_settings=objective_settings,
                    log_container=log_container,
                )
            except Exception as exc:
                st.error(f"Error en Optuna: {exc}")
                return

        best_params = result["best_params"]
        st.session_state["optuna_last_file"] = result["best_path"]
        st.success(
            f"Optuna finalizado. Archivo: {os.path.basename(result['best_path'])}"
        )
        st.markdown("**Mejores hiperparametros**")
        st.dataframe(
            pd.DataFrame([best_params]),
            width="stretch",
        )
        st.caption(
            "Estos resultados se guardaron en Resultados y pueden usarse en Training y Balance."
        )


def _render_events_tab() -> None:
    st.subheader("Eventos (accidentes)")
    st.markdown(
        "Selecciona uno o varios archivos de eventos desde la carpeta Datos."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No se encontraron archivos de eventos en la carpeta Datos.")
        return

    selected_names = st.multiselect(
        "Archivos de eventos disponibles",
        [path.name for path in event_files],
        default=[path.name for path in event_files],
        key="gnn_events_files",
    )

    if st.button("Procesar eventos", key="gnn_events_process"):
        if not selected_names:
            st.warning("Seleccione al menos un archivo de eventos.")
            return

        try:
            porticos_df = load_porticos()
            porticos_source = "Datos/Porticos.csv"
        except FileNotFoundError:
            st.error(
                "No se encontro Porticos.csv en la carpeta Datos. "
                "Agreguelo antes de continuar."
            )
            return
        except Exception as exc:
            st.error(f"No se pudieron cargar los porticos: {exc}")
            return

        frames = []
        for name in selected_names:
            path = DATA_DIR / name
            try:
                frames.append(
                    pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
                )
            except UnicodeDecodeError:
                frames.append(
                    pd.read_csv(
                        path, sep=None, engine="python", encoding="latin-1"
                    )
                )
            except Exception as exc:
                st.error(f"No se pudo leer {name}: {exc}")
                return

        raw_df = pd.concat(frames, ignore_index=True)
        try:
            acc_df, excluded = process_accidentes_df(
                raw_df, porticos_df, return_excluded=True
            )
        except Exception as exc:
            st.error(f"No se pudieron procesar los eventos: {exc}")
            return

        st.session_state["df_acc"] = acc_df
        st.session_state["accidents_df"] = acc_df
        st.session_state["accident_files"] = selected_names
        st.session_state["porticos_source"] = porticos_source
        st.session_state["loaded_acc_path"] = "__events_tab__"

        st.success(
            f"Accidentes procesados: {len(acc_df):,} | "
            f"Excluidos sin portico: {len(excluded):,}"
        )

    accidents_df = st.session_state.get("df_acc")
    if accidents_df is None or accidents_df.empty:
        st.info("No hay accidentes cargados.")
        return

    st.caption(
        f"Archivos: {', '.join(st.session_state.get('accident_files', []))} | "
        f"Porticos: {st.session_state.get('porticos_source') or '-'}"
    )

    preview_rows = st.slider(
        "Filas de vista previa",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        key="gnn_events_preview_rows",
    )

    preview_cols = [
        col
        for col in [
            "accidente_time",
            "ultimo_portico",
            "proximo_portico",
            "duracion_accidente",
            "severidad",
        ]
        if col in accidents_df.columns
    ]
    st.dataframe(
        accidents_df[preview_cols].head(preview_rows),
        width="stretch",
    )

    def _clean_p(val):
        s = str(val).strip()
        return s[:-2] if s.endswith(".0") else s

    st.subheader("Secuencia de porticos")
    st.caption(
        "Secuencia ordenada por eje/calzada segun el archivo Porticos.csv."
    )
    try:
        porticos_df = load_porticos()
    except Exception as exc:
        st.warning(f"No se pudieron cargar los porticos: {exc}")
        return
    if porticos_df is None or porticos_df.empty:
        st.warning("No hay porticos disponibles.")
        return

    porticos = porticos_df.copy()
    porticos["orden_num"] = pd.to_numeric(porticos["orden"], errors="coerce")
    porticos["km_num"] = pd.to_numeric(porticos["km"], errors="coerce")
    porticos["eje_norm"] = (
        porticos["eje"].astype(str).str.strip().str.upper()
    )
    porticos["calzada_norm"] = (
        porticos["calzada"].astype(str).str.strip().str.upper()
    )
    porticos = porticos.dropna(
        subset=["orden_num", "km_num", "eje_norm", "calzada_norm"]
    )

    sequence_rows = []
    for _, group in porticos.groupby(["eje_norm", "calzada_norm"]):
        group = group.sort_values("orden_num")
        sequence = [
            f"{_clean_p(row['portico'])}({row['km_num']:g})"
            for _, row in group.iterrows()
        ]
        if not sequence:
            continue
        sequence_rows.append(
            {
                "Eje": group["eje"].iloc[0],
                "Calzada": group["calzada"].iloc[0],
                "Secuencia": " -> ".join(sequence),
            }
        )
    if not sequence_rows:
        st.info("No se pudo construir la secuencia de porticos.")
    else:
        sequence_df = (
            pd.DataFrame(sequence_rows)
            .sort_values(["Eje", "Calzada"])
            .reset_index(drop=True)
        )
        st.dataframe(sequence_df, width="stretch")

    st.subheader("Accidentes por tramo")
    st.caption(
        "Conteo de accidentes entre porticos consecutivos segun el orden."
    )

    km_col = _find_match_column(accidents_df, ["Km.", "Km", "Kilometro"])
    eje_col = _find_match_column(accidents_df, ["Eje"])
    calzada_col = _find_match_column(accidents_df, ["Calzada"])
    if km_col is None or eje_col is None or calzada_col is None:
        st.warning(
            "No se encontraron columnas de km/eje/calzada en accidentes."
        )
        return

    segments = []
    for _, group in porticos.groupby(["eje_norm", "calzada_norm"]):
        group = group.sort_values("orden_num")
        for i in range(len(group) - 1):
            start = group.iloc[i]
            end = group.iloc[i + 1]
            segments.append(
                {
                    "Eje": start["eje"],
                    "Calzada": start["calzada"],
                    "orden_inicio": int(start["orden_num"]),
                    "portico_inicio": str(start["portico"]).strip(),
                    "km_inicio": float(start["km_num"]),
                    "orden_fin": int(end["orden_num"]),
                    "portico_fin": str(end["portico"]).strip(),
                    "km_fin": float(end["km_num"]),
                }
            )
    segments_df = pd.DataFrame(segments)
    if segments_df.empty:
        st.info("No se pudieron construir los tramos de porticos.")
        return

    acc_seg = accidents_df[[eje_col, calzada_col, km_col]].copy()
    acc_seg = acc_seg.rename(
        columns={eje_col: "eje", calzada_col: "calzada", km_col: "km_acc"}
    )
    acc_seg["km_acc"] = pd.to_numeric(
        acc_seg["km_acc"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    acc_seg = acc_seg.dropna(subset=["km_acc", "eje", "calzada"])

    segment_keys = []
    assigned_indices = set()
    for row in acc_seg.itertuples():
        try:
            cand = find_candidate_porticos(
                acc_km=row.km_acc,
                porticos_df=porticos_df,
                eje=row.eje,
                calzada=row.calzada,
            )
        except Exception:
            continue
        posterior = cand.get("posterior")
        cercano = cand.get("cercano")
        if posterior is None or cercano is None:
            continue
        assigned_indices.add(row.Index)
        segment_keys.append(
            {
                "Eje": posterior["eje"],
                "Calzada": posterior["calzada"],
                "portico_inicio": _clean_p(posterior["portico"]),
                "portico_fin": _clean_p(cercano["portico"]),
            }
        )

    if segment_keys:
        counts_df = (
            pd.DataFrame(segment_keys)
            .groupby(
                ["Eje", "Calzada", "portico_inicio", "portico_fin"],
                dropna=False,
            )
            .size()
            .reset_index(name="accidentes")
        )
    else:
        counts_df = pd.DataFrame(
            columns=[
                "Eje",
                "Calzada",
                "portico_inicio",
                "portico_fin",
                "accidentes",
            ]
        )

    segments_df = segments_df.merge(
        counts_df,
        on=["Eje", "Calzada", "portico_inicio", "portico_fin"],
        how="left",
    )
    segments_df["accidentes"] = (
        segments_df["accidentes"].fillna(0).astype(int)
    )
    segments_df = segments_df.sort_values(
        ["Eje", "Calzada", "orden_inicio"]
    ).reset_index(drop=True)

    display_cols = [
        "Eje",
        "Calzada",
        "portico_inicio",
        "km_inicio",
        "portico_fin",
        "km_fin",
        "accidentes",
    ]
    st.dataframe(segments_df[display_cols], width="stretch")

    missing_info = int(len(accidents_df) - len(acc_seg))
    unassigned = int(len(acc_seg) - len(segment_keys))
    if missing_info > 0:
        st.caption(
            f"Accidentes sin km/eje/calzada: {missing_info:,}"
        )
    if unassigned > 0:
        st.caption(
            f"Accidentes sin tramo asignado: {unassigned:,}"
        )
        unassigned_indices = acc_seg.index.difference(assigned_indices)
        if not unassigned_indices.empty:
            st.markdown("**Detalle de accidentes sin tramo asignado**")
            cols_to_show = [
                col
                for col in [
                    "accidente_time",
                    eje_col,
                    calzada_col,
                    km_col,
                    "Descripcion",
                    "SubTipo",
                ]
                if col in accidents_df.columns
            ]
            st.dataframe(
                accidents_df.loc[unassigned_indices, cols_to_show],
                width="stretch",
            )

def _render_load_graph():
    st.markdown("### Cargar Grafo")
    st.write("Cargue un grafo previamente construido (.pt) desde la carpeta de Resultados.")
    
    # List available graphs
    graph_files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "highway_graph_*.pt")), key=os.path.getmtime, reverse=True)
    if not graph_files:
        st.warning(f"No se encontraron grafos en {RESULTADOS_DIR}")
        return

    selected_file = st.selectbox("Seleccione un archivo de grafo:", graph_files, format_func=os.path.basename)
    
    if st.button("Cargar Grafo"):
        if selected_file:
            with st.spinner("Cargando grafo..."):
                try:
                    if torch.backends.mps.is_available():
                        device = torch.device("mps")
                    elif torch.cuda.is_available():
                        device = torch.device("cuda")
                    else:
                        device = torch.device("cpu")
                    
                    loaded_obj = torch.load(selected_file, map_location=device, weights_only=False)
                    loaded_obj['filename'] = os.path.basename(selected_file)
                    
                    st.session_state.loaded_graph = loaded_obj
                    st.session_state.graph_path = selected_file
                    
                    st.success(f"Grafo cargado: {os.path.basename(selected_file)}")
                    
                    # Display basic info
                    if 'data' in loaded_obj:
                         st.write("Estructura:", loaded_obj['data'])
                except Exception as e:
                    st.error(f"Error al cargar: {e}")

def _render_create_graph():
    st.markdown("#### Selección de Datos")
    
    # LOAD PORTICOS
    if 'df_port' not in st.session_state or st.session_state.df_port is None:
        with st.spinner("Cargando pórticos automáticamente..."):
            df_port = load_porticos() # Tries Datos/Porticos.csv by default
            if df_port is not None:
                st.session_state.df_port = df_port
            else:
                st.error("No se pudo cargar 'Datos/Porticos.csv'. Verifique el archivo.")

    if st.session_state.df_port is not None:
        st.markdown(
            "Archivo de Pórticos cargado: `Datos/Porticos.csv`"
        )
        events_label = None
        if (
            isinstance(st.session_state.get("df_acc"), pd.DataFrame)
            and not st.session_state.df_acc.empty
        ):
            event_files = st.session_state.get("accident_files", [])
            if event_files:
                events_label = ", ".join(event_files)
            elif st.session_state.get("loaded_acc_path"):
                events_label = os.path.basename(
                    st.session_state.get("loaded_acc_path")
                )
        if events_label:
            st.markdown(f"Archivo de Eventos: `{events_label}`")
        # st.dataframe(st.session_state.df_port.head(), height=150) # Hidden by user request
    
    # TRAMO SELECTION
    # st.markdown("#### Definición del Grafo (Tramo)")
    
    tramo_mode = st.radio("Definición del Grafo (Tramo)", ["Manual", "Puntos Negros (Reporte)"], index=0)
    
    porticos_a_incluir = None
    preselected_time_choice = None
    
    if tramo_mode == "Puntos Negros (Reporte)":
        pn_reports = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "analisis_puntos_negros_*.csv")))
        pn_mode_options = ["Usar reporte existente", "Calcular nuevo"]
        if pn_reports:
            pn_mode = st.radio(
                "Puntos Negros",
                pn_mode_options,
                horizontal=True,
                key="pn_mode_select",
            )
        else:
            pn_mode = "Calcular nuevo"

        if pn_mode == "Calcular nuevo":
            if st.button("Calcular nuevo reporte", key="pn_calc_report"):
                acc_df = st.session_state.get("df_acc")
                if acc_df is None or acc_df.empty:
                    st.warning(
                        "Cargue accidentes en la pestaña Eventos."
                    )
                elif st.session_state.df_port is None:
                    st.warning("No hay pórticos cargados.")
                else:
                    stats = _compute_puntos_negros_report(
                        acc_df, st.session_state.df_port
                    )
                    if stats is None or stats.empty:
                        st.warning(
                            "No se pudo generar el reporte de puntos negros."
                        )
                    else:
                        timestamp = datetime.now().strftime("%d%m%Y_%H%M")
                        output_path = os.path.join(
                            RESULTADOS_DIR,
                            f"analisis_puntos_negros_{timestamp}.csv",
                        )
                        stats.to_csv(
                            output_path,
                            index=False,
                            sep=";",
                            decimal=",",
                        )
                        st.session_state["pn_report_select"] = (
                            output_path
                        )
                        pn_reports = sorted(
                            glob.glob(
                                os.path.join(
                                    RESULTADOS_DIR,
                                    "analisis_puntos_negros_*.csv",
                                )
                            )
                        )

        if not pn_reports:
            st.warning("No se encontraron reportes de puntos negros.")
        else:
            selected_report = st.selectbox(
                "Reporte de puntos negros",
                pn_reports,
                format_func=lambda x: os.path.basename(x),
                key="pn_report_select",
            )
            if selected_report:
                try:
                    df_pn = pd.read_csv(selected_report, sep=';', decimal=',')
                    if not df_pn.empty:
                        top = df_pn.iloc[0]
                        st.write("Top Punto Negro:", top)
                        tramo_str = str(top['tramo'])
                        franja = str(top.get('franja_horaria', ''))
                        
                        if '06-10' in franja: preselected_time_choice = '2'
                        elif '18-22' in franja: preselected_time_choice = '3'
                        else: preselected_time_choice = '1'
                        
                        p1_raw, p2_raw = [s.strip() for s in tramo_str.split('->')]
                        if st.session_state.df_port is not None:
                            porticos_a_incluir = reconstruct_portico_sequence(
                                df_porticos=st.session_state.df_port,
                                p1=p1_raw, p2=p2_raw, 
                                max_anterior=4, max_posterior=0,
                                prefer_family=True, deduplicate_slice=True
                            )
                            st.info(f"Pórticos seleccionados ({len(porticos_a_incluir)}): {porticos_a_incluir}")
                except Exception as e:
                    st.error(f"Error leyendo reporte: {e}")

    else: # Manual
        if st.session_state.df_port is not None:
            # We recreate a simple manual selector here instead of 'seleccionar_tramo_y_porticos' which uses input()
            df_port = st.session_state.df_port
            autopista = None
            if "autopista" in df_port.columns:
                autopista_options = list(
                    pd.Index(df_port["autopista"].dropna().unique())
                )
                AUTOPISTA_MAP = {
                    "C": "Ruta 5 (Central)",
                    "NO": "General Velásquez",
                }
                if autopista_options:
                    if (
                        "gnn_autopista" not in st.session_state
                        or st.session_state["gnn_autopista"]
                        not in autopista_options
                    ):
                        st.session_state["gnn_autopista"] = (
                            autopista_options[0]
                        )
                    autopista = st.selectbox(
                        "Autopista",
                        autopista_options,
                        format_func=lambda x: AUTOPISTA_MAP.get(x, x),
                        key="gnn_autopista",
                    )
                    mask_aut = df_port["autopista"] == autopista
                else:
                    mask_aut = pd.Series(True, index=df_port.index)
            else:
                mask_aut = pd.Series(True, index=df_port.index)

            calzada_options = list(
                pd.Index(df_port[mask_aut]["calzada"].dropna().unique())
            )
            if not calzada_options:
                st.warning(
                    "No hay calzadas disponibles para esta autopista."
                )
                calzada = None
                mask_calz = pd.Series(False, index=df_port.index)
                eje = None
                mask_eje = mask_calz
            else:
                if (
                    "gnn_calzada" not in st.session_state
                    or st.session_state["gnn_calzada"]
                    not in calzada_options
                ):
                    st.session_state["gnn_calzada"] = calzada_options[0]
                calzada = st.selectbox(
                    "Calzada", calzada_options, key="gnn_calzada"
                )
                mask_calz = mask_aut & (df_port["calzada"] == calzada)
                eje_options = list(
                    pd.Index(df_port[mask_calz]["eje"].dropna().unique())
                )
                if not eje_options:
                    st.warning("No hay ejes disponibles para esta calzada.")
                    eje = None
                    mask_eje = mask_calz
                elif len(eje_options) == 1:
                    eje = eje_options[0]
                    st.session_state["gnn_eje"] = eje
                    mask_eje = mask_calz & (df_port["eje"] == eje)
                else:
                    if (
                        "gnn_eje" not in st.session_state
                        or st.session_state["gnn_eje"] not in eje_options
                    ):
                        st.session_state["gnn_eje"] = eje_options[0]
                    eje = st.selectbox(
                        "Eje", eje_options, key="gnn_eje"
                    )
                    mask_eje = mask_calz & (df_port["eje"] == eje)
            
            df_sel = df_port[mask_eje].copy()
            if "orden" in df_sel.columns:
                df_sel["orden_num"] = pd.to_numeric(
                    df_sel["orden"], errors="coerce"
                )
            df_sel["km_num"] = pd.to_numeric(df_sel.get("km"), errors="coerce")
            sort_cols = []
            if "orden_num" in df_sel.columns and df_sel["orden_num"].notna().any():
                sort_cols.append("orden_num")
            if "km_num" in df_sel.columns and df_sel["km_num"].notna().any():
                sort_cols.append("km_num")
            if sort_cols:
                df_sel = df_sel.sort_values(sort_cols)
            else:
                df_sel = df_sel.sort_values("portico")
            portico_series = (
                df_sel["portico"].dropna().astype(str).str.strip()
            )
            porticos_disp = list(pd.Index(portico_series).drop_duplicates())
            
            if not porticos_disp:
                st.warning("No hay pórticos disponibles para esta calzada.")
                p_start = None
                p_end = None
            else:
                tramo_key = (autopista, calzada, eje)
                if st.session_state.get("gnn_tramo_key") != tramo_key:
                    st.session_state["gnn_tramo_key"] = tramo_key
                    st.session_state["gnn_portico_start"] = porticos_disp[0]
                    st.session_state["gnn_portico_end"] = porticos_disp[-1]
                    st.session_state.pop("manual_porticos", None)
                c1, c2 = st.columns(2)
                with c1:
                    if (
                        st.session_state.get("gnn_portico_start")
                        not in porticos_disp
                    ):
                        st.session_state["gnn_portico_start"] = (
                            porticos_disp[0]
                        )
                    p_start = st.selectbox(
                        "Pórtico Inicio",
                        porticos_disp,
                        key="gnn_portico_start",
                    )
                with c2:
                    try:
                        start_idx = porticos_disp.index(p_start)
                    except ValueError:
                        start_idx = 0
                    end_options = porticos_disp[start_idx + 1:]
                    if not end_options:
                        end_options = [p_start]
                    if (
                        st.session_state.get("gnn_portico_end")
                        not in end_options
                    ):
                        st.session_state["gnn_portico_end"] = (
                            end_options[-1]
                        )
                    p_end = st.selectbox(
                        "Pórtico Fin",
                        end_options,
                        key="gnn_portico_end",
                    )
            
            # Logic to get range
            if p_start and p_end and st.button("Confirmar Tramo"):
                # Simple logic to get slice between start and end based on KM
                # Assuming portico list is sorted by KM
                try:
                    p_list = porticos_disp
                    idx_s = p_list.index(p_start)
                    idx_e = p_list.index(p_end)
                    if idx_s > idx_e: idx_s, idx_e = idx_e, idx_s
                    porticos_a_incluir = p_list[idx_s : idx_e+1]
                    st.session_state.manual_porticos = porticos_a_incluir
                    st.success(f"Tramo manual seleccionado: {len(porticos_a_incluir)} pórticos")
                except ValueError:
                    st.error("Error calculando rango de pórticos")
            
            if 'manual_porticos' in st.session_state:
                porticos_a_incluir = st.session_state.manual_porticos
                st.write(f"Selección actual: {porticos_a_incluir}")

    # ACCIDENTS
    if 'df_acc' not in st.session_state:
        st.session_state.df_acc = None

    events_loaded = (
        isinstance(st.session_state.get("df_acc"), pd.DataFrame)
        and not st.session_state.df_acc.empty
        and st.session_state.get("loaded_acc_path") == "__events_tab__"
    )
    use_events = False
    if events_loaded:
        use_events = st.checkbox(
            "Usar accidentes cargados en Eventos",
            value=True,
            key="gnn_use_events_tab",
        )

    sel_acc_file = None
    if not use_events:
        # Find available accident files
        acc_files = []
        try:
            acc_files = sorted(
                glob.glob(os.path.join("Datos", "Eventos*.csv"))
            )
        except Exception:
            pass

        if not acc_files:
            st.warning(
                "No se encontraron archivos 'eventos*.csv' en la carpeta 'Datos'."
            )
            sel_acc_file = None
        else:
            # Default to selectbox
            sel_acc_file = st.selectbox(
                "Seleccione el archivo de Accidentes",
                acc_files,
                format_func=os.path.basename,
            )

    # FEATURE CONFIG PREVIEW
    feat_cfg = st.session_state.get("feature_config")
    if not feat_cfg:
        st.warning(
            "No hay configuracion de features. Configure en Feature Engineering."
        )

    # The actual feature loading happens in BUILD logic below, reading from st.session_state.feature_config

    # TIME FILTER
    st.markdown("#### Filtro Temporal")
    time_options = {
        '1': 'Sin filtro (Todos)',
        '2': 'Punta Mañana (06:00 - 10:00)',
        '3': 'Punta Tarde (18:00 - 22:00)'
    }
    def_idx = 0
    if preselected_time_choice == '2': def_idx = 1
    elif preselected_time_choice == '3': def_idx = 2
    
    time_sel_key = st.selectbox("Franja Horaria", list(time_options.keys()), 
                                format_func=lambda x: time_options[x], 
                                index=def_idx)
    
    # EDGE CONFIG
    st.markdown("#### Configuración de Aristas")
    c1, c2, c3, c4 = st.columns(4)
    with c1: build_temporal = st.checkbox("Temporales", value=True)
    with c2: build_spatial = st.checkbox("Espaciales", value=True)
    with c3: build_st_fwd = st.checkbox("Espacio-Temporal (Fwd)", value=False)
    with c4: build_spatial_back = st.checkbox("Espacial (Back)", value=False)
    debug_labels = st.checkbox(
        "Debug etiquetas (mostrar resumen)",
        value=False,
        key="gnn_debug_labels",
    )

    # BUILD BUTTON
    if st.button("Construir Grafo", type="primary"):
        if st.session_state.df_port is None:
            st.error("Faltan pórticos.")
            return
        if porticos_a_incluir is None:
            st.error("Falta seleccionar tramo.")
            return
        if use_events:
            if (
                st.session_state.df_acc is None
                or st.session_state.df_acc.empty
            ):
                st.error("No hay accidentes cargados en Eventos.")
                return
        elif not sel_acc_file:
            st.error("Falta seleccionar archivo de accidentes.")
            return

        # Auto-load Accidents if needed
        if not use_events:
            if (
                st.session_state.df_acc is None
                or st.session_state.get("loaded_acc_path") != sel_acc_file
            ):
                with st.spinner(
                    "Cargando accidentes desde "
                    f"{os.path.basename(sel_acc_file)}..."
                ):
                    loaded_acc = load_accidentes(file_path=sel_acc_file)
                    if loaded_acc is not None:
                        st.session_state.df_acc = loaded_acc
                        st.session_state.loaded_acc_path = sel_acc_file
                    else:
                        st.error("Falló la carga de accidentes.")
                        return
        
        # --- EXECUTION LOGIC (Ported from src/graph.py) ---
        status = st.empty()
        status.info("Iniciando construcción...")
        progress = st.progress(0)
        
        try:
            # 1. Prepare Data
            porticos_norm = _normalize_portico_series(
                pd.Series(porticos_a_incluir)
            )
            porticos_norm = list(pd.Index(porticos_norm).dropna().drop_duplicates())
            if not porticos_norm:
                st.error("No hay pórticos válidos en el tramo seleccionado.")
                return
            porticos_a_incluir = porticos_norm

            df_port_use = st.session_state.df_port.copy()
            df_port_use["portico"] = _normalize_portico_series(
                df_port_use["portico"]
            )
            df_port_use = df_port_use[
                df_port_use["portico"].isin(porticos_a_incluir)
            ].copy()

            df_acc_use = st.session_state.df_acc.copy()
            df_acc_use["ultimo_portico"] = _normalize_portico_series(
                df_acc_use["ultimo_portico"]
            )
            df_acc_use = df_acc_use[
                df_acc_use["ultimo_portico"].isin(porticos_a_incluir)
            ].copy()

            feature_config = st.session_state.get("feature_config", {})
            params = feature_config.get("params", {}) if isinstance(feature_config, dict) else {}
            snap_config = SnapshotConfig()
            if params:
                snap_config.dt_minutes = params.get("dt_minutes", snap_config.dt_minutes)
                snap_config.window_minutes = params.get("window_minutes", snap_config.window_minutes)
                snap_config.slow_speed_kmh = params.get("slow_speed_kmh", snap_config.slow_speed_kmh)

            try:
                snapshot_topology = build_static_topology(df_port_use, snap_config)
            except Exception as exc:
                snapshot_topology = None
                st.warning(f"No se pudo construir topologia: {exc}")
            
            # 2. Features
            df_pm = None
            
            # Check Cache first
            df_pm_cache = st.session_state.get("df_pm_cache")
            if isinstance(df_pm_cache, pd.DataFrame) and not df_pm_cache.empty:
                status.text("Usando features en memoria cache...")
                df_pm = df_pm_cache.copy()
            else:
                status.info("Generando features (no encontradas en cache)...")
                # Retrieve Feature Config
                feat_cfg = st.session_state.get(
                    "feature_config", {"source": "Crash prediction"}
                )
                df_pm = run_feature_generation_workflow(feat_cfg)
            
            if df_pm is None or df_pm.empty:
                st.error("No se pudieron obtener features. Verifique la configuración en 'Feature Engineering'.")
                return
            feature_mode = params.get("feature_mode")
            if not feature_mode:
                feature_mode = (
                    "Snapshot"
                    if _is_snapshot_features(df_pm)
                    else "Flow 5 min"
                )
            use_snapshot_labels = feature_mode == "Snapshot"
            if debug_labels:
                _debug_labels(
                    debug_labels,
                    f"feature_mode={feature_mode} use_snapshot_labels={use_snapshot_labels}",
                )
            if (
                FORCE_SNAPSHOT_FEATURES
                and use_snapshot_labels
                and not _is_snapshot_features(df_pm)
            ):
                st.error(
                    "Las features en memoria no corresponden a snapshots. "
                    "Genere nuevamente en modo snapshot."
                )
                return

            # Ensure we have correct sorting/reset index
            df_pm = df_pm.sort_values(["portico", "ts_min"]).reset_index(drop=True)
            if debug_labels and "ts_min" in df_pm.columns:
                _debug_labels(
                    debug_labels,
                    f"df_pm rows={len(df_pm)} ts_min_range=({df_pm['ts_min'].min()}, {df_pm['ts_min'].max()})",
                )
            
            # --- 2.3 Filter by Porticos (Tramo) ---
            # Now we filter the Big Feature Set by the selected Porticos
            portico_col = buscar_columna(df_pm, "portico")
            if portico_col:
                initial_len = len(df_pm)
                df_pm = df_pm[df_pm[portico_col].isin(porticos_a_incluir)].copy()
                status.text(f"Filtrado por tramo: {initial_len} -> {len(df_pm)} registros.")
            else:
                 st.warning("Columna de pórtico no encontrada en features.")

            if df_pm.empty:
                st.error("Features vacías tras filtrar por tramo. Verifique nombres de pórticos.")
                return

            # --- 2.4 Time Filtering (Applied here post-generation for full flexibility) ---
            # Although run_feature_generation didn't apply strict time filtering (06-10 etc), 
            # we apply it here for the Graph Construction specifically.
            if time_sel_key in ['2', '3']:
                status.text("Aplicando filtro temporal...")
                timestamps = pd.to_datetime(df_pm['ts_min'], unit='m', origin='unix')
                hours = timestamps.dt.hour
                if time_sel_key == '2': time_mask = (hours >= 6) & (hours < 10)
                else: time_mask = (hours >= 18) & (hours < 22)
                df_pm = df_pm[time_mask].copy()
                
            # 3. Target Labeling (Future Accident)
            # Need Accident Data
            
            # ... Resume existing logic ...
            if df_pm.empty:
                st.error("Dataset vacío después de filtros.")
                return
            
            df_pm = df_pm.reset_index(drop=True)
            df_pm['ts_min'] = df_pm['ts_min'].astype(int)
            df_pm['node_idx'] = df_pm.index
            
            # DT Inference
            dt_feat_minutes = params.get("dt_minutes")
            if not dt_feat_minutes or dt_feat_minutes <= 0:
                dt_feat_minutes = _infer_dt_minutes(df_pm, portico_col)
            dt_feat_minutes = int(dt_feat_minutes)
            if debug_labels:
                _debug_labels(
                    debug_labels,
                    f"dt_feat_minutes={dt_feat_minutes} portico_col={portico_col}",
                )

            # Align Accidents
            df_acc_use["ts_min"] = (
                df_acc_use["accidente_time"]
                .dt.floor(f"{int(dt_feat_minutes)}min")
                .astype("int64")
                // 60_000_000_000
            )
            if not use_snapshot_labels:
                df_acc_use["ts_min"] = (
                    df_acc_use["ts_min"] - int(dt_feat_minutes)
                )
            if debug_labels and "ts_min" in df_acc_use.columns:
                _debug_labels(
                    debug_labels,
                    f"accidents_rows={len(df_acc_use)} acc_ts_range=({df_acc_use['ts_min'].min()}, {df_acc_use['ts_min'].max()}) shift_prev_window={not use_snapshot_labels}",
                )
            # Cast types
            try:
                if np.issubdtype(df_pm[portico_col].dtype, np.number):
                    df_acc_use["ultimo_portico"] = pd.to_numeric(
                        df_acc_use["ultimo_portico"], errors="coerce"
                    ).astype("Int64")
                    df_pm[portico_col] = pd.to_numeric(
                        df_pm[portico_col], errors="coerce"
                    ).astype("Int64")
                else:
                    df_acc_use["ultimo_portico"] = df_acc_use[
                        "ultimo_portico"
                    ].astype(str)
                    df_pm[portico_col] = df_pm[portico_col].astype(str)
            except Exception:
                pass

            affected_pms = pd.merge(
                df_acc_use,
                df_pm[[portico_col, "ts_min", "node_idx"]],
                left_on=["ultimo_portico", "ts_min"],
                right_on=[portico_col, "ts_min"],
                how="inner",
            )
            if debug_labels:
                _debug_labels(
                    debug_labels,
                    f"affected_pms rows={len(affected_pms)} unique_nodes={affected_pms['node_idx'].nunique() if not affected_pms.empty else 0}",
                )

            seq_index = None
            sequence_config = None
            if use_snapshot_labels:
                # Sequence Index (snapshot labeling)
                seq_l = params.get("seq_length", SEQ_LENGTH)
                gb_min = params.get("guard_band_minutes", GUARD_BAND_MINUTES)
                hor_min = params.get("horizon_minutes", HORIZON_MINUTES)
                gb_effective = int(gb_min) + int(dt_feat_minutes)

                sequence_config = SequenceConfig(
                    sequence_length=seq_l,
                    guard_band_minutes=gb_effective,
                    horizon_minutes=hor_min,
                    include_downstream=bool(INCLUDE_DOWNSTREAM_IN_LABEL),
                )
                seq_index, labels_df = build_sequence_index(
                    df_pm,
                    df_acc_use[["ultimo_portico", "ts_min"]].copy(),
                    snapshot_topology,
                    sequence_config,
                    step_minutes=dt_feat_minutes,
                )
                labels_df = labels_df.sort_values("row_id")
                df_pm["future_label"] = labels_df["label"].to_numpy(dtype=int)

                sequence_mask_np = np.zeros(len(df_pm), dtype=bool)
                if seq_index.target_rows.size:
                    sequence_mask_np[seq_index.target_rows] = True
                sequence_mask = torch.from_numpy(sequence_mask_np)
                if debug_labels:
                    _debug_labels(
                        debug_labels,
                        f"snapshot_labels pos={int(df_pm['future_label'].sum())} seq_targets={int(seq_index.target_rows.size)}",
                    )
            else:
                # Flow labeling (5 min previous interval)
                df_pm["future_label"] = 0
                if not affected_pms.empty:
                    df_pm.loc[
                        affected_pms["node_idx"].unique(), "future_label"
                    ] = 1
                sequence_mask = torch.ones(len(df_pm), dtype=torch.bool)
                if debug_labels:
                    _debug_labels(
                        debug_labels,
                        f"flow_labels pos={int(df_pm['future_label'].sum())} total={len(df_pm)}",
                    )

            # Normalization
            status.text("Normalizando features...")
            feature_cols = [
                c
                for c in df_pm.columns
                if c
                not in [
                    portico_col,
                    "ts_min",
                    "node_idx",
                    "future_label",
                    "next_ts_min",
                ]
            ]
            if "selected_features" in st.session_state:
                selected_features = st.session_state.get("selected_features")
                if not selected_features:
                    st.error(
                        "No hay variables seleccionadas. "
                        "Configure Feature selection antes de construir el grafo."
                    )
                    return
                selected_in_df = [
                    feature
                    for feature in selected_features
                    if feature in feature_cols
                ]
                ignore_missing = {
                    "portico",
                    "ts_min",
                    "node_idx",
                    "future_label",
                    "next_ts_min",
                    "target",
                }
                missing = [
                    feature
                    for feature in selected_features
                    if feature not in feature_cols
                    and feature not in ignore_missing
                ]
                if missing:
                    st.warning(
                        "Variables seleccionadas no disponibles en features: "
                        + ", ".join(missing)
                    )
                if not selected_in_df:
                    st.error(
                        "Ninguna variable seleccionada existe en las features actuales."
                    )
                    return
                feature_cols = selected_in_df
                status.text(
                    f"Aplicando selección de variables: {len(feature_cols)}"
                )
            
            accident_ts_unique = affected_pms['ts_min'].unique().tolist()
            train_ts_cutoff, val_ts_cutoff = _compute_time_cutoffs(accident_ts_unique, TRAIN_RATIO, VAL_RATIO)
            
            # Compute stats on train split
            if np.isfinite(train_ts_cutoff):
                train_stats_mask = df_pm['ts_min'] <= train_ts_cutoff
            else:
                train_stats_mask = np.ones(len(df_pm), dtype=bool)
                
            train_stats_df = df_pm.loc[train_stats_mask, feature_cols]
            mu = train_stats_df.mean().values
            sigmasigma = train_stats_df.std(ddof=0).fillna(0.0).values + 1e-9
            
            norm_stats_path = _save_normalization_stats(feature_cols, mu, sigmasigma, train_ts_cutoff, val_ts_cutoff)
            normalization_metadata = {
                "feature_names": feature_cols,
                "mean": mu.tolist(),
                "std": sigmasigma.tolist(),
                "stats_path": norm_stats_path
            }
            
            # Apply Norm
            feat_mat = (df_pm[feature_cols].values - mu) / sigmasigma
            feat_mat = np.nan_to_num(feat_mat)
            df_pm[feature_cols] = feat_mat
            
            progress.progress(60)
            status.text("Vectorizando nodos y aristas...")
            
            # Hetero Data Construction
            # 1. PM Nodes
            float_type = getattr(torch, FLOAT)
            pm_feats = torch.tensor(df_pm[feature_cols].values, dtype=float_type)
            pm_y = torch.tensor(df_pm['future_label'].to_numpy(), dtype=torch.long)
            
            is_accident_pm = torch.zeros(len(df_pm), dtype=torch.bool)
            if not affected_pms.empty:
                is_accident_pm[affected_pms['node_idx'].unique()] = True
                
            data = HeteroData()
            data["pm"].x = pm_feats
            data["pm"].y = pm_y
            data["pm"].is_accident_pm = is_accident_pm
            
            # 2. Edges
            # Note: We skip the detailed edge construction logic here for brevity in this first pass 
            # OR we must implement it if we want it to work. 
            # USEFUL: We can reuse the logic. I will implement a simplified version or copy it.
            # I'll implement the temporal and spatial core logic.
            
            EXTRA_SPATIAL = 6

            # Temporal
            temporal_attr = None
            temporal_src, temporal_dst = [], []
            if build_temporal:
                df_pm["next_ts_min"] = df_pm["ts_min"] + int(dt_feat_minutes)
                t_edges = pd.merge(
                    df_pm[[portico_col, "ts_min", "node_idx", "next_ts_min"]],
                    df_pm[[portico_col, "ts_min", "node_idx"]],
                    left_on=[portico_col, "next_ts_min"],
                    right_on=[portico_col, "ts_min"],
                    suffixes=("_src", "_dst"),
                )
                temporal_src = t_edges["node_idx_src"].tolist()
                temporal_dst = t_edges["node_idx_dst"].tolist()

                delta = feat_mat[temporal_dst] - feat_mat[temporal_src]
                temporal_attr = np.concatenate(
                    [delta, np.zeros((len(delta), EXTRA_SPATIAL))],
                    axis=1,
                )

                data[("pm", "temporal", "pm")].edge_index = torch.tensor(
                    [temporal_src, temporal_dst], dtype=torch.long
                )
                data[("pm", "temporal", "pm")].edge_attr = torch.tensor(
                    temporal_attr, dtype=float_type
                )

            spatial_src: List[int] = []
            spatial_dst: List[int] = []
            spatial_attr: Optional[np.ndarray] = None
            spatial_back_src: List[int] = []
            spatial_back_dst: List[int] = []
            spatial_back_attr: Optional[np.ndarray] = None
            stfwd_src: List[int] = []
            stfwd_dst: List[int] = []
            stfwd_attr: Optional[np.ndarray] = None

            if build_spatial or build_spatial_back or build_st_fwd:
                portico_sequences = []
                df_port_s = df_port_use.sort_values(
                    ["calzada", "eje", "orden"],
                    ascending=[False, True, True],
                )
                for (calz, eje), g in df_port_s.groupby(
                    ["calzada", "eje"]
                ):
                    s_ports = g["portico"].tolist()
                    kms = (
                        g["km"].astype(float).tolist()
                        if "km" in g.columns
                        else [0] * len(s_ports)
                    )
                    for i in range(len(s_ports) - 1):
                        dkm = abs(kms[i + 1] - kms[i])
                        portico_sequences.append(
                            (s_ports[i], s_ports[i + 1], dkm)
                        )
                df_seq = pd.DataFrame(
                    portico_sequences,
                    columns=["portico_src", "portico_dst", "dist_km"],
                )

                if df_seq.empty:
                    st.warning(
                        "No se generaron secuencias de pórticos para aristas espaciales."
                    )
                else:
                    if build_spatial or build_spatial_back:
                        s_edges = pd.merge(
                            df_pm.rename(
                                columns={
                                    portico_col: "portico_src",
                                    "node_idx": "node_idx_src",
                                }
                            ),
                            df_seq,
                            on="portico_src",
                        )
                        s_edges = pd.merge(
                            s_edges,
                            df_pm.rename(
                                columns={
                                    portico_col: "portico_dst",
                                    "node_idx": "node_idx_dst",
                                }
                            ),
                            left_on=["ts_min", "portico_dst"],
                            right_on=["ts_min", "portico_dst"],
                            how="inner",
                        )

                        spatial_src = s_edges["node_idx_src"].tolist()
                        spatial_dst = s_edges["node_idx_dst"].tolist()

                        if spatial_src:
                            delta_s = (
                                feat_mat[spatial_dst]
                                - feat_mat[spatial_src]
                            )
                            extras = np.zeros(
                                (len(delta_s), EXTRA_SPATIAL)
                            )
                            if EXTRA_SPATIAL > 0:
                                dist_vals = (
                                    s_edges["dist_km"].to_numpy(dtype=float)
                                    if "dist_km" in s_edges.columns
                                    else np.zeros(len(delta_s))
                                )
                                extras[:, 0] = np.nan_to_num(
                                    dist_vals, nan=0.0
                                )
                            spatial_attr = np.concatenate(
                                [delta_s, extras], axis=1
                            )

                            if build_spatial:
                                data[
                                    ("pm", "spatial", "pm")
                                ].edge_index = torch.tensor(
                                    [spatial_src, spatial_dst],
                                    dtype=torch.long,
                                )
                                data[
                                    ("pm", "spatial", "pm")
                                ].edge_attr = torch.tensor(
                                    spatial_attr, dtype=float_type
                                )

                            if build_spatial_back:
                                spatial_back_src = spatial_dst
                                spatial_back_dst = spatial_src
                                delta_back = -delta_s
                                spatial_back_attr = np.concatenate(
                                    [delta_back, extras], axis=1
                                )
                                data[
                                    ("pm", "spatial_back", "pm")
                                ].edge_index = torch.tensor(
                                    [spatial_back_src, spatial_back_dst],
                                    dtype=torch.long,
                                )
                                data[
                                    ("pm", "spatial_back", "pm")
                                ].edge_attr = torch.tensor(
                                    spatial_back_attr, dtype=float_type
                                )

                    if build_st_fwd:
                        if "next_ts_min" not in df_pm.columns:
                            df_pm["next_ts_min"] = (
                                df_pm["ts_min"] + int(dt_feat_minutes)
                            )
                        st_edges = pd.merge(
                            df_pm.rename(
                                columns={
                                    portico_col: "portico_src",
                                    "node_idx": "node_idx_src",
                                }
                            ),
                            df_seq,
                            on="portico_src",
                        )
                        st_edges = pd.merge(
                            st_edges,
                            df_pm.rename(
                                columns={
                                    portico_col: "portico_dst",
                                    "node_idx": "node_idx_dst",
                                }
                            ),
                            left_on=["next_ts_min", "portico_dst"],
                            right_on=["ts_min", "portico_dst"],
                            how="inner",
                        )

                        stfwd_src = st_edges["node_idx_src"].tolist()
                        stfwd_dst = st_edges["node_idx_dst"].tolist()
                        if stfwd_src:
                            delta_st = (
                                feat_mat[stfwd_dst]
                                - feat_mat[stfwd_src]
                            )
                            extras = np.zeros(
                                (len(delta_st), EXTRA_SPATIAL)
                            )
                            if EXTRA_SPATIAL > 0:
                                dist_vals = (
                                    st_edges["dist_km"].to_numpy(dtype=float)
                                    if "dist_km" in st_edges.columns
                                    else np.zeros(len(delta_st))
                                )
                                extras[:, 0] = np.nan_to_num(
                                    dist_vals, nan=0.0
                                )
                            stfwd_attr = np.concatenate(
                                [
                                    delta_st,
                                    extras,
                                ],
                                axis=1,
                            )
                            data[
                                ("pm", "st_fwd", "pm")
                            ].edge_index = torch.tensor(
                                [stfwd_src, stfwd_dst], dtype=torch.long
                            )
                            data[
                                ("pm", "st_fwd", "pm")
                            ].edge_attr = torch.tensor(
                                stfwd_attr, dtype=float_type
                            )
            
            # Clean edges
            for store in data.edge_stores:
                _safe_clean_edges(store, data[store._key[0]].num_nodes, data[store._key[2]].num_nodes)
            
            # Masks with time split
            ts_array = df_pm['ts_min'].to_numpy()
            train_mask = (ts_array <= train_ts_cutoff)
            val_mask = (ts_array > train_ts_cutoff) & (ts_array <= val_ts_cutoff)
            test_mask = (ts_array > val_ts_cutoff)
            
            data['pm'].train_mask = torch.from_numpy(train_mask) & sequence_mask
            data['pm'].val_mask = torch.from_numpy(val_mask) & sequence_mask
            data['pm'].test_mask = torch.from_numpy(test_mask) & sequence_mask
            data['pm'].sequence_mask = sequence_mask

            # Save
            pm_map = df_pm.set_index([portico_col, 'ts_min'])['node_idx'].to_dict()
            pm_rev = {v: k for k, v in pm_map.items()}
            pm_index = PMIndex(pm_map, pm_rev)
            
            period_tag = "stream_build"
            timestamp = datetime.now().strftime('%d%m%Y_%H%M')
            filename = f"highway_graph_{period_tag}_{timestamp}.pt"
            output_path = os.path.join(RESULTADOS_DIR, filename)

            torch.save({
                "data": data,
                "pm_index": pm_index,
                "mu": mu,
                "sigma": sigmasigma,
                "feature_cols": feature_cols,
                "normalization": normalization_metadata,
                "sequence_index": seq_index,
                "sequence_config": sequence_config
            }, output_path)
            
            progress.progress(100)
            status.success(f"Grafo construido y guardado en: {filename}")
            st.session_state.loaded_graph = {
                "data": data,
                "pm_index": pm_index,
                "filename": filename
            }
            st.session_state.graph_path = output_path
            
        except Exception as e:
            status.error(f"Error construyendo grafo: {e}")
            st.exception(e)


def _init_state() -> None:
    st.session_state.setdefault("loaded_graph", None)
    st.session_state.setdefault("graph_path", None)
    st.session_state.setdefault("df_pm_cache", None)
    st.session_state.setdefault("df_port", None)
    st.session_state.setdefault("df_acc", None)
    st.session_state.setdefault("feature_config", {"source": "Generar nuevas"})


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    _init_state()
    if set_page_config:
        st.set_page_config(page_title="Graph Builder", layout="wide")

    st.title("Crash prediction whit Graph Neural Networks")
    render_graph_builder()

    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)


if __name__ == "__main__":
    main()
