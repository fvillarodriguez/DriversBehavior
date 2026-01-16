#!/usr/bin/env python3
"""
Streamlit app to run the SUMO simulation pipeline and related tools.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SUMO import SegmentFilter, SUMOResult, run_sumo_pipeline  # noqa: E402
from utils import (  # noqa: E402
    FLOW_TABLE_NAME,
    FlowColumns,
    get_flow_db_summary,
    load_porticos,
)

SIMULATION_DIR = ROOT_DIR / "simulación"
RESULTS_DIR = ROOT_DIR / "Resultados"

BREW_DYLD_DIRS = [
    Path("/opt/homebrew/opt/gl2ps/lib"),
    Path("/opt/homebrew/opt/open-scene-graph/lib"),
    Path("/opt/homebrew/opt/gdal/lib"),
    Path("/opt/homebrew/opt/ffmpeg/lib"),
]
_PYARROW_LIB_WARNING_SHOWN = False


def _init_state() -> None:
    st.session_state.setdefault("sumo_flujos_df", None)
    st.session_state.setdefault("sumo_porticos_df", None)
    st.session_state.setdefault("sumo_result", None)


def _format_ts(ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M")


def _format_range(df: pd.DataFrame, column: str) -> str:
    if df is None or df.empty or column not in df.columns:
        return "-"
    series = pd.to_datetime(df[column], errors="coerce")
    if series.isna().all():
        return "-"
    start = series.min()
    end = series.max()
    return f"{_format_ts(start)} -> {_format_ts(end)}"


def _normalize_dyld_chunks(values: List[str]) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for raw_value in values:
        if not raw_value:
            continue
        for chunk in raw_value.split(":"):
            path = chunk.strip()
            if not path or path in seen:
                continue
            if Path(path).expanduser().exists():
                seen.add(path)
                normalized.append(path)
    return normalized


def _pyarrow_lib_dir() -> Optional[Path]:
    override = os.environ.get("SUMO_PYARROW_LIB_DIR")
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(Path(virtual_env) / "lib" / py_version / "site-packages" / "pyarrow")
    home_py = (
        Path.home()
        / "Library"
        / "Python"
        / f"{sys.version_info.major}.{sys.version_info.minor}"
        / "lib/python/site-packages/pyarrow"
    )
    candidates.append(home_py)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _warn_missing_pyarrow_lib() -> None:
    global _PYARROW_LIB_WARNING_SHOWN
    if _PYARROW_LIB_WARNING_SHOWN:
        return
    st.warning(
        "pyarrow/libarrow was not found. Set SUMO_PYARROW_LIB_DIR if SUMO needs it."
    )
    _PYARROW_LIB_WARNING_SHOWN = True


def _build_sumo_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    if sys.platform != "darwin":
        return env
    dyld_parts: List[str] = []
    dyld_parts.extend(str(path) for path in BREW_DYLD_DIRS if path.exists())
    pyarrow_dir = _pyarrow_lib_dir()
    if pyarrow_dir:
        dyld_parts.append(str(pyarrow_dir))
    else:
        if "pyarrow" not in env.get("DYLD_LIBRARY_PATH", ""):
            _warn_missing_pyarrow_lib()
    extra = os.environ.get("SUMO_EXTRA_DYLD_PATHS")
    if extra:
        dyld_parts.append(extra)
    existing = env.get("DYLD_LIBRARY_PATH")
    if existing:
        dyld_parts.append(existing)
    normalized = _normalize_dyld_chunks(dyld_parts)
    if normalized:
        env["DYLD_LIBRARY_PATH"] = ":".join(normalized)
    return env


def find_sumo_binary(executable: str) -> Optional[Path]:
    candidates: List[Path] = []
    sumo_home = os.environ.get("SUMO_HOME")
    suffixes = [""]
    if os.name == "nt" and not executable.lower().endswith(".exe"):
        suffixes.append(".exe")
    if sumo_home:
        for suffix in suffixes:
            candidates.append(Path(sumo_home) / "bin" / f"{executable}{suffix}")
    for suffix in suffixes:
        which_path = shutil.which(f"{executable}{suffix}")
        if which_path:
            candidates.append(Path(which_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_flujos_from_db(
    date_start: Optional[pd.Timestamp],
    date_end: Optional[pd.Timestamp],
    row_limit: Optional[int],
) -> Optional[pd.DataFrame]:
    try:
        import duckdb  # type: ignore
    except ImportError:
        st.error("duckdb no esta instalado. Ejecute `pip install duckdb`.")
        return None

    try:
        summary = get_flow_db_summary()
    except ImportError as exc:
        st.error(str(exc))
        return None

    if summary.row_count == 0:
        st.warning("La base de flujos esta vacia. Importe datos primero.")
        return None

    conn = duckdb.connect(str(summary.db_path), read_only=True)
    try:
        query = f"""
            SELECT
                FECHA,
                VELOCIDAD,
                CATEGORIA,
                MATRICULA,
                PORTICO,
                CARRIL
            FROM {FLOW_TABLE_NAME}
        """
        params: List[object] = []
        if date_start is not None and date_end is not None:
            query += " WHERE FECHA >= ? AND FECHA <= ?\n"
            params.extend([date_start, date_end])
        query += " ORDER BY COALESCE(FECHA, TIMESTAMP '1970-01-01')"
        if row_limit is not None:
            query += " LIMIT ?"
            params.append(int(row_limit))
        df = conn.execute(query, params).df()
    finally:
        conn.close()

    if df.empty:
        st.warning("La consulta no devolvio filas.")
        return df

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["FECHA"] = df["FECHA"].dt.tz_localize(None)
    df["VELOCIDAD"] = pd.to_numeric(df["VELOCIDAD"], errors="coerce")
    df["CATEGORIA"] = pd.to_numeric(df["CATEGORIA"], errors="coerce")
    df = df.dropna(subset=["CATEGORIA"])
    df["CATEGORIA"] = df["CATEGORIA"].astype("Int64")
    return df.reset_index(drop=True)


def _render_flow_loader() -> Optional[pd.DataFrame]:
    try:
        summary = get_flow_db_summary()
    except ImportError as exc:
        st.error(str(exc))
        return None

    st.caption(f"DB: {summary.db_path}")
    cols = st.columns(3)
    cols[0].metric("Filas", f"{summary.row_count:,}")
    cols[1].metric("Fecha min", _format_ts(summary.min_timestamp))
    cols[2].metric("Fecha max", _format_ts(summary.max_timestamp))

    if summary.row_count == 0:
        st.warning("La base de flujos esta vacia. Use Flow database para cargar datos.")
        return None

    mode = st.radio("Muestreo", ["Todo", "Rango de fechas"], horizontal=True, key="sumo_sample_mode")
    date_start = None
    date_end = None
    if mode == "Rango de fechas":
        default_start = summary.min_timestamp.date() if summary.min_timestamp else datetime.today().date()
        default_end = summary.max_timestamp.date() if summary.max_timestamp else datetime.today().date()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Fecha inicio", value=default_start, key="sumo_start_date")
        with col2:
            end_date = st.date_input("Fecha fin", value=default_end, key="sumo_end_date")
        use_time = st.checkbox("Usar horas", value=False, key="sumo_use_time")
        if use_time:
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.time_input("Hora inicio", value=dt_time(0, 0), key="sumo_start_time")
            with col2:
                end_time = st.time_input("Hora fin", value=dt_time(23, 59), key="sumo_end_time")
        else:
            start_time = dt_time(0, 0)
            end_time = dt_time(23, 59, 59)
        date_start = pd.Timestamp(datetime.combine(start_date, start_time))
        date_end = pd.Timestamp(datetime.combine(end_date, end_time))
        if date_end <= date_start:
            st.error("La fecha final debe ser posterior a la fecha de inicio.")

    row_limit = st.number_input(
        "Limite de filas (0 = sin limite)",
        min_value=0,
        value=0,
        step=10000,
        key="sumo_row_limit",
    )

    if st.button("Cargar flujos", key="sumo_load_flujos"):
        if mode == "Rango de fechas" and date_end is not None and date_start is not None:
            if date_end <= date_start:
                return st.session_state.get("sumo_flujos_df")
        limit_value = int(row_limit) if row_limit > 0 else None
        with st.spinner("Cargando flujos..."):
            df = _load_flujos_from_db(date_start, date_end, limit_value)
        if df is not None:
            st.session_state["sumo_flujos_df"] = df
            st.session_state["sumo_result"] = None
            st.success(f"Flujos cargados: {len(df):,} filas.")
    return st.session_state.get("sumo_flujos_df")


def _render_porticos_loader() -> Optional[pd.DataFrame]:
    default_path = ROOT_DIR / "Datos" / "Porticos.csv"
    default_value = str(default_path) if default_path.exists() else ""
    path_value = st.text_input("Ruta Porticos.csv", value=default_value, key="sumo_porticos_path")
    if st.button("Cargar porticos", key="sumo_load_porticos"):
        try:
            df = load_porticos(path_value or None)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return st.session_state.get("sumo_porticos_df")
        except Exception as exc:
            st.error(f"No se pudo leer Porticos.csv: {exc}")
            return st.session_state.get("sumo_porticos_df")
        st.session_state["sumo_porticos_df"] = df
        st.session_state["sumo_result"] = None
        st.success(f"Porticos cargados: {len(df):,} filas.")
    return st.session_state.get("sumo_porticos_df")


def _build_segment_groups(
    porticos_df: pd.DataFrame,
) -> List[Tuple[Tuple[str, str], pd.DataFrame]]:
    required = {"portico", "km", "calzada", "orden", "eje"}
    if porticos_df is None or porticos_df.empty:
        return []
    if not required.issubset(set(porticos_df.columns)):
        return []
    catalog = porticos_df.copy()
    catalog["orden"] = pd.to_numeric(catalog["orden"], errors="coerce")
    catalog = catalog.dropna(subset=["orden"])
    catalog["eje"] = catalog["eje"].astype(str).str.strip()
    catalog["calzada"] = catalog["calzada"].astype(str).str.strip()
    groups = []
    for (eje, calzada), group in catalog.groupby(["eje", "calzada"], sort=True):
        ordered = group.sort_values("orden").reset_index(drop=True)
        if len(ordered) >= 2:
            groups.append(((eje, calzada), ordered))
    return groups


def _render_segment_filter(
    porticos_df: Optional[pd.DataFrame],
) -> Optional[SegmentFilter]:
    if porticos_df is None or porticos_df.empty:
        return None

    use_segment = st.checkbox(
        "Filtrar por tramo especifico",
        value=False,
        key="sumo_use_segment",
    )
    if not use_segment:
        return None

    missing_cols = {"portico", "km", "calzada", "orden", "eje"} - set(porticos_df.columns)
    if missing_cols:
        st.warning(
            "Porticos no tiene las columnas requeridas para tramos: "
            + ", ".join(sorted(missing_cols))
        )
        return None

    groups = _build_segment_groups(porticos_df)
    if not groups:
        st.warning("No se encontraron combinaciones validas de eje/calzada.")
        return None

    group_labels = [
        f"{eje} / {calzada} ({len(group)})"
        for (eje, calzada), group in groups
    ]
    selection = st.selectbox(
        "Eje / Calzada",
        options=group_labels,
        key="sumo_segment_group",
    )
    group_idx = group_labels.index(selection)
    (eje, calzada), group_df = groups[group_idx]
    group_df = group_df.reset_index(drop=True)

    def _label_for_idx(idx: int) -> str:
        row = group_df.loc[idx]
        km = row.get("km")
        km_label = f"km {km}" if pd.notna(km) else "km -"
        return f"[{idx}] {row['portico']} ({km_label})"

    start_idx = st.selectbox(
        "Portico inicio",
        options=list(range(len(group_df) - 1)),
        format_func=_label_for_idx,
        key="sumo_segment_start",
    )
    end_idx = st.selectbox(
        "Portico fin",
        options=list(range(start_idx + 1, len(group_df))),
        format_func=_label_for_idx,
        key="sumo_segment_end",
    )

    selected = (
        group_df.loc[start_idx:end_idx, "portico"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    return SegmentFilter(
        eje=eje,
        calzada=calzada,
        start_portico=str(group_df.loc[start_idx, "portico"]).strip(),
        end_portico=str(group_df.loc[end_idx, "portico"]).strip(),
        portico_ids=selected,
    )


def _render_sumo_summary(result: SUMOResult) -> None:
    clean_count = len(result.clean_events)
    trajectories_count = (
        result.trajectories["trip_id"].nunique()
        if not result.trajectories.empty and "trip_id" in result.trajectories.columns
        else 0
    )
    segments_count = len(result.segments)
    macro_count = len(result.macro_metrics)
    headway_count = len(result.headways)

    st.subheader("Resumen SUMO")
    if result.segment_filter is not None:
        st.caption(f"Tramo: {result.segment_filter.description()}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Detecciones limpias", f"{clean_count:,}")
    col2.metric("Viajes detectados", f"{trajectories_count:,}")
    col3.metric("Segmentos", f"{segments_count:,}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ventanas macro", f"{macro_count:,}")
    col2.metric("Headways", f"{headway_count:,}")
    col3.metric("Trips SUMO", f"{len(result.sumo_trips or []):,}")

    if result.sumo_trips_path:
        st.caption(f"Trips XML: {result.sumo_trips_path}")
    if result.depart_summary_path:
        st.caption(f"Depart summary: {result.depart_summary_path}")
    if result.sumo_warning:
        st.warning(result.sumo_warning)

    with st.expander("Ver datos (muestra)", expanded=False):
        st.write("Clean events")
        st.dataframe(result.clean_events.head(200), width="stretch")
        st.write("Trajectories")
        st.dataframe(result.trajectories.head(200), width="stretch")
        st.write("Segments")
        st.dataframe(result.segments.head(200), width="stretch")
        st.write("Macro metrics")
        st.dataframe(result.macro_metrics.head(200), width="stretch")
        st.write("Headways")
        st.dataframe(result.headways.head(200), width="stretch")


def _get_existing_sumo_trips_path(result: Optional[SUMOResult]) -> Optional[Path]:
    candidates: List[Path] = []
    if result and result.sumo_trips_path:
        candidates.append(Path(result.sumo_trips_path))
    candidates.append(SIMULATION_DIR / "sumo_trips.rou.xml")
    candidates.append(RESULTS_DIR / "sumo_trips.rou.xml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _render_pipeline_tab() -> None:
    st.subheader("Pipeline SUMO")

    with st.expander("Flujos (DuckDB)", expanded=True):
        flujos_df = _render_flow_loader()
        if flujos_df is not None and not flujos_df.empty:
            st.caption(f"Rango: {_format_range(flujos_df, 'FECHA')}")
            st.caption(f"Filas cargadas: {len(flujos_df):,}")

    with st.expander("Porticos", expanded=True):
        porticos_df = _render_porticos_loader()
        if porticos_df is not None and not porticos_df.empty:
            st.caption(f"Filas cargadas: {len(porticos_df):,}")

    porticos_df = st.session_state.get("sumo_porticos_df")
    segment_filter = _render_segment_filter(porticos_df)

    col1, col2 = st.columns([2, 1])
    with col1:
        output_dir = st.text_input(
            "Directorio de salida",
            value=str(SIMULATION_DIR),
            key="sumo_output_dir",
        )
    with col2:
        save_trips = st.checkbox(
            "Guardar trips XML",
            value=True,
            key="sumo_save_trips",
        )

    if st.button("Ejecutar pipeline SUMO", key="sumo_run_pipeline"):
        flujos_df = st.session_state.get("sumo_flujos_df")
        porticos_df = st.session_state.get("sumo_porticos_df")
        if flujos_df is None or flujos_df.empty:
            st.error("Cargue flujos antes de ejecutar el pipeline.")
            return
        if porticos_df is None or porticos_df.empty:
            st.error("Cargue porticos antes de ejecutar el pipeline.")
            return
        output_path = Path(output_dir).expanduser() if save_trips else None
        with st.spinner("Ejecutando pipeline SUMO..."):
            try:
                result = run_sumo_pipeline(
                    flujos_df,
                    porticos_df,
                    flow_cols=FlowColumns(),
                    output_dir=output_path,
                    segment_filter=segment_filter,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        st.session_state["sumo_result"] = result
        st.success("Pipeline SUMO completado.")

    result = st.session_state.get("sumo_result")
    if result is not None:
        _render_sumo_summary(result)


def _render_duarouter_tab() -> None:
    st.subheader("Duarouter")

    result = st.session_state.get("sumo_result")
    detected_trips = _get_existing_sumo_trips_path(result)
    trips_text = str(detected_trips) if detected_trips else ""
    trips_input = st.text_input("Trips XML", value=trips_text, key="sumo_trips_path")

    default_net = SIMULATION_DIR / "highway.net.xml"
    net_value = str(default_net) if default_net.exists() else ""
    net_path = st.text_input("Archivo .net.xml", value=net_value, key="sumo_net_path")

    default_routes = SIMULATION_DIR / "routes.rou.xml"
    routes_value = str(default_routes) if default_routes.exists() else str(ROOT_DIR / "routes.rou.xml")
    output_path = st.text_input("Salida routes.rou.xml", value=routes_value, key="sumo_routes_path")

    if st.button("Generar routes.rou.xml", key="sumo_run_duarouter"):
        if not trips_input:
            st.error("Ingrese la ruta de sumo_trips.rou.xml.")
            return
        trips_path = Path(trips_input).expanduser()
        if not trips_path.exists():
            st.error("No se encontro sumo_trips.rou.xml. Ejecute el pipeline primero.")
            return
        if not net_path:
            st.error("Ingrese la ruta del archivo .net.xml.")
            return
        net_file = Path(net_path).expanduser()
        if not net_file.exists():
            st.error("El archivo .net.xml no existe.")
            return
        out_file = Path(output_path).expanduser()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        duarouter_bin = find_sumo_binary("duarouter")
        if duarouter_bin is None:
            st.error("No se encontro 'duarouter'. Configure SUMO_HOME o el PATH.")
            return

        cmd = [
            str(duarouter_bin),
            "-n",
            str(net_file),
            "-r",
            str(trips_path),
            "-o",
            str(out_file),
            "--departlane",
            "best",
            "--unsorted-input",
            "true",
            "--ignore-errors",
        ]
        with st.spinner("Ejecutando duarouter..."):
            try:
                subprocess.run(cmd, check=True, env=_build_sumo_subprocess_env())
            except subprocess.CalledProcessError as exc:
                st.error(f"duarouter fallo (codigo {exc.returncode}).")
                return
        st.success(f"routes.rou.xml generado en: {out_file}")


def _render_sumo_run_tab() -> None:
    st.subheader("SUMO tripinfo")

    default_cfg = SIMULATION_DIR / "sample.sumocfg"
    cfg_value = str(default_cfg) if default_cfg.exists() else ""
    cfg_path = st.text_input("Archivo .sumocfg", value=cfg_value, key="sumo_cfg_path")

    default_tripinfo = SIMULATION_DIR / "tripinfo.xml"
    tripinfo_value = (
        str(default_tripinfo) if default_tripinfo.exists() else str(ROOT_DIR / "tripinfo.xml")
    )
    output_path = st.text_input("Salida tripinfo.xml", value=tripinfo_value, key="sumo_tripinfo_path")

    if st.button("Ejecutar SUMO", key="sumo_run_sumo"):
        if not cfg_path:
            st.error("Ingrese la ruta del archivo .sumocfg.")
            return
        cfg_file = Path(cfg_path).expanduser()
        if not cfg_file.exists():
            st.error("El archivo .sumocfg no existe.")
            return
        out_file = Path(output_path).expanduser()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        sumo_bin = find_sumo_binary("sumo")
        if sumo_bin is None:
            st.error("No se encontro el ejecutable 'sumo'. Configure SUMO_HOME o el PATH.")
            return
        cmd = [
            str(sumo_bin),
            "-c",
            str(cfg_file),
            "--tripinfo-output",
            str(out_file),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
        with st.spinner("Ejecutando SUMO..."):
            try:
                subprocess.run(cmd, check=True, env=_build_sumo_subprocess_env())
            except subprocess.CalledProcessError as exc:
                st.error(f"SUMO fallo (codigo {exc.returncode}).")
                return
        st.success(f"tripinfo.xml generado en: {out_file}")


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    _init_state()
    if set_page_config:
        st.set_page_config(page_title="SUMO Simulation", layout="wide")
    st.title("Simulacion SUMO")

    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)

    tab_pipeline, tab_duarouter, tab_sumo = st.tabs(
        ["Pipeline", "Duarouter", "SUMO tripinfo"]
    )
    with tab_pipeline:
        _render_pipeline_tab()
    with tab_duarouter:
        _render_duarouter_tab()
    with tab_sumo:
        _render_sumo_run_tab()


if __name__ == "__main__":
    main()
