#!/usr/bin/env python3
"""
Streamlit app to map events on a map with filters.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Datos"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

EVENTS_DB_PATH = DATA_DIR / "eventos.duckdb"
EVENTS_TABLE_NAME = "eventos"
EVENTS_DB_COLUMNS = [
    "evento_time",
    "tipo_evento",
    "eje",
    "calzada",
    "km",
    "ultimo_portico",
    "portico_inicio",
    "portico_fin",
    "Descripcion",
    "SubTipo",
    "lat",
    "lon",
]

from src.utils import (  # noqa: E402
    buscar_columna,
    find_candidate_porticos,
    load_porticos,
    process_accidentes_df,
)


def _init_state() -> None:
    st.session_state.setdefault("events_df", None)
    st.session_state.setdefault("event_files", [])
    st.session_state.setdefault("porticos_source", None)
    st.session_state.setdefault("highway_geo", None)


def _normalize_key(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _is_accident_type(value: object) -> bool:
    return "accidente" in _normalize_key(value)


def _find_column(
    df: pd.DataFrame, candidates: Sequence[str]
) -> Optional[str]:
    normalized = {_normalize_key(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_key(candidate)
        col = normalized.get(key)
        if col:
            return col
    return None


def _select_columns(
    df: pd.DataFrame, candidates: Sequence[str]
) -> List[str]:
    normalized = {_normalize_key(col): col for col in df.columns}
    selected: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize_key(candidate)
        col = normalized.get(key)
        if col and col not in seen:
            selected.append(col)
            seen.add(col)
    return selected


def _list_event_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    candidates = []
    for path in DATA_DIR.glob("*.csv"):
        if path.name.lower().startswith("eventos"):
            candidates.append(path)
    return sorted(candidates)


def _ensure_duckdb_available() -> None:
    if duckdb is None:  # pragma: no cover
        raise ImportError(
            "duckdb no está instalado. Ejecute `pip install duckdb` para habilitar la base de eventos."
        )


def _connect_events_db(read_only: bool = False):
    _ensure_duckdb_available()
    EVENTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ro_flag = read_only and EVENTS_DB_PATH.exists()
    return duckdb.connect(str(EVENTS_DB_PATH), read_only=ro_flag)


def _events_db_exists() -> bool:
    return EVENTS_DB_PATH.exists()


def _clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace({"": pd.NA})


def _prepare_events_db_frame(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=EVENTS_DB_COLUMNS)
    events_df = events_df.reset_index(drop=True)
    data: Dict[str, pd.Series] = {}
    for col in EVENTS_DB_COLUMNS:
        if col in events_df.columns:
            series = events_df[col]
        else:
            series = pd.Series([pd.NA] * len(events_df))
        if col == "evento_time":
            data[col] = pd.to_datetime(series, errors="coerce")
        elif col in {"km", "lat", "lon"}:
            data[col] = pd.to_numeric(series, errors="coerce")
        else:
            data[col] = _clean_text_series(series)
    return pd.DataFrame(data)


def _write_events_db(events_df: pd.DataFrame) -> int:
    if EVENTS_DB_PATH.exists():
        EVENTS_DB_PATH.unlink()
    db_frame = _prepare_events_db_frame(events_df)
    conn = _connect_events_db(read_only=False)
    try:
        conn.register("events_df", db_frame)
        conn.execute(f"CREATE TABLE {EVENTS_TABLE_NAME} AS SELECT * FROM events_df")
        inserted = conn.execute(
            f"SELECT COUNT(*) FROM {EVENTS_TABLE_NAME}"
        ).fetchone()[0]
        return int(inserted or 0)
    finally:
        conn.close()


def _load_events_db() -> pd.DataFrame:
    if not _events_db_exists():
        return pd.DataFrame(columns=EVENTS_DB_COLUMNS)
    conn = _connect_events_db(read_only=True)
    try:
        info = conn.execute(
            f"PRAGMA table_info('{EVENTS_TABLE_NAME}')"
        ).fetchall()
        if not info:
            return pd.DataFrame(columns=EVENTS_DB_COLUMNS)
        available_cols = [row[1] for row in info]
        selected = [col for col in EVENTS_DB_COLUMNS if col in available_cols]
        if not selected:
            return pd.DataFrame(columns=EVENTS_DB_COLUMNS)
        df = conn.execute(
            f"SELECT {', '.join(selected)} FROM {EVENTS_TABLE_NAME}"
        ).df()
    finally:
        conn.close()

    for col in EVENTS_DB_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["evento_time"] = pd.to_datetime(df["evento_time"], errors="coerce")
    for col in ("km", "lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [c for c in EVENTS_DB_COLUMNS if c not in {"evento_time", "km", "lat", "lon"}]:
        df[col] = _clean_text_series(df[col])
    return df[EVENTS_DB_COLUMNS].copy()


def _read_event_files(selected_names: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name in selected_names:
        path = DATA_DIR / name
        try:
            frames.append(
                pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
            )
        except UnicodeDecodeError:
            frames.append(
                pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_datetime(
    date_series: pd.Series, time_series: pd.Series
) -> pd.Series:
    combined = (
        date_series.astype(str).str.strip()
        + " "
        + time_series.astype(str).str.strip()
    )
    try:
        dt = pd.to_datetime(combined, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        dt = pd.to_datetime(combined, errors="coerce", dayfirst=True)
    try:
        return dt.dt.tz_localize(None)
    except TypeError:
        return dt.dt.tz_convert(None)


def _prepare_events_df(
    raw_df: pd.DataFrame,
    porticos_df: pd.DataFrame,
    *,
    allowed_via: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    tipo_col = buscar_columna(df, "Tipo")
    via_col = buscar_columna(df, "Via")
    allowed_via = [v.lower() for v in (allowed_via or ["expresa", "via expresa"])]
    df = df[
        df[via_col].astype(str).str.lower().isin(allowed_via)
    ].copy()
    if df.empty:
        return df

    calzada_col = buscar_columna(df, "Calzada")
    df[calzada_col] = (
        df[calzada_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"oriente": "Oriente", "poniente": "Poniente", "ambas": "Ambas"})
    )
    eje_col = buscar_columna(df, "Eje")
    km_col = buscar_columna(df, "Km.", aliases=["Km"])

    desc_col = _find_column(df, ["Descripcion", "Descripción"])
    subtipo_col = _find_column(df, ["SubTipo", "Sub Tipo", "Sub-Tipo"])

    fecha_inicio_col = buscar_columna(
        df, "Fechas Inicio", aliases=["Fecha Inicio"]
    )
    hora_inicio_col = buscar_columna(df, "Hora Inicio")
    df["evento_time"] = _parse_datetime(
        df[fecha_inicio_col], df[hora_inicio_col]
    )

    df[km_col] = (
        df[km_col].astype(str).str.replace(",", ".").pipe(pd.to_numeric, errors="coerce")
    )

    def _get_segment(row: pd.Series) -> Tuple[Optional[object], Optional[object]]:
        try:
            cand = find_candidate_porticos(
                acc_km=row[km_col],
                porticos_df=porticos_df,
                eje=row[eje_col],
                calzada=row[calzada_col],
            )
        except Exception:
            return None, None
        posterior = cand.get("posterior")
        cercano = cand.get("cercano")
        return (
            posterior["portico"] if posterior is not None else None,
            cercano["portico"] if cercano is not None else None,
        )

    segments = df.apply(_get_segment, axis=1, result_type="expand")
    segments.columns = ["portico_inicio", "portico_fin"]
    df = pd.concat([df, segments], axis=1)
    df["ultimo_portico"] = df["portico_inicio"]

    df["tipo_evento"] = df[tipo_col].astype(str).str.strip()
    df["eje"] = df[eje_col]
    df["calzada"] = df[calzada_col]
    df["km"] = df[km_col]
    df["Descripcion"] = (
        df[desc_col].astype("string").str.strip() if desc_col else ""
    )
    df["SubTipo"] = (
        df[subtipo_col].astype("string").str.strip() if subtipo_col else ""
    )

    try:
        acc_df, excluded = process_accidentes_df(
            raw_df, porticos_df, allowed_via=allowed_via, return_excluded=True
        )
    except Exception:
        acc_df = pd.DataFrame()
        excluded = pd.DataFrame()

    if not acc_df.empty:
        common_idx = df.index.intersection(acc_df.index)
        if not common_idx.empty:
            df.loc[common_idx, "portico_inicio"] = acc_df.loc[
                common_idx, "ultimo_portico"
            ].values
            df.loc[common_idx, "portico_fin"] = acc_df.loc[
                common_idx, "proximo_portico"
            ].values
            df.loc[common_idx, "ultimo_portico"] = acc_df.loc[
                common_idx, "ultimo_portico"
            ].values
    df.attrs["accidents_excluded"] = int(len(excluded))

    df["portico_inicio"] = df["portico_inicio"].fillna("").astype(str).str.strip()
    df["portico_fin"] = df["portico_fin"].fillna("").astype(str).str.strip()
    df["ultimo_portico"] = df["ultimo_portico"].astype(str).str.strip()
    return df


def _fill_missing_porticos(
    events_df: pd.DataFrame, porticos_df: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    if events_df is None or events_df.empty:
        return events_df, 0
    if porticos_df is None or porticos_df.empty:
        return events_df, 0

    df = events_df.copy()
    for col in ("portico_inicio", "portico_fin", "ultimo_portico"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].apply(_clean_portico_str)

    if "km" in df.columns:
        df["km"] = pd.to_numeric(df["km"], errors="coerce")

    if "eje" not in df.columns or "calzada" not in df.columns:
        return df, 0

    missing_mask = df["portico_inicio"].isna()
    if not missing_mask.any():
        return df, 0

    def _get_segment(row: pd.Series) -> Tuple[Optional[object], Optional[object]]:
        try:
            cand = find_candidate_porticos(
                acc_km=row["km"],
                porticos_df=porticos_df,
                eje=row["eje"],
                calzada=row["calzada"],
            )
        except Exception:
            return None, None
        posterior = cand.get("posterior")
        cercano = cand.get("cercano")
        return (
            posterior["portico"] if posterior is not None else None,
            cercano["portico"] if cercano is not None else None,
        )

    segments = df.loc[missing_mask].apply(
        _get_segment, axis=1, result_type="expand"
    )
    segments.columns = ["portico_inicio_new", "portico_fin_new"]
    df.loc[missing_mask, "portico_inicio"] = segments["portico_inicio_new"].values
    df.loc[missing_mask, "portico_fin"] = segments["portico_fin_new"].values
    df.loc[missing_mask, "ultimo_portico"] = df.loc[missing_mask, "portico_inicio"]

    fixed_mask = missing_mask & df["portico_inicio"].notna()
    return df, int(fixed_mask.sum())


def _build_segments_df(porticos_df: pd.DataFrame) -> pd.DataFrame:
    if porticos_df is None or porticos_df.empty:
        return pd.DataFrame()
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

    segments: List[Dict[str, object]] = []
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
                    "portico_fin": str(end["portico"]).strip(),
                    "km_fin": float(end["km_num"]),
                }
            )
    return pd.DataFrame(segments)


def _load_highway_geo() -> Optional[pd.DataFrame]:
    """Loads highway geometry (nodes/ways) from nodos_autopista.json."""
    json_path = DATA_DIR / "nodos_autopista.json"
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    elements = data.get("elements", [])
    paths = []
    
    for el in elements:
        if el.get("type") == "way" and "geometry" in el:
            # geometry is a list of {lat, lon} dicts
            geo = el["geometry"]
            # Extract coordinates as [lon, lat] for PyDeck PathLayer
            path_coords = [[pt["lon"], pt["lat"]] for pt in geo]
            
            # Simple metadata from tags
            tags = el.get("tags", {})
            name = tags.get("name", "Unknown")
            ref = tags.get("ref", "")
            
            paths.append({
                "path": path_coords,
                "name": name,
                "ref": ref,
                "color": [100, 100, 100], # Dark gray for roads
                # fields for tooltip
                "tipo_evento": "Tramo Autopista",
                "evento_time_str": ref if ref else "-",
                "Descripcion": name,
            })
            
    if not paths:
        return None
        
    return pd.DataFrame(paths)


def _render_filters(
    events_df: pd.DataFrame, porticos_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filtered = events_df.copy()

    tipo_options = sorted(
        [t for t in filtered["tipo_evento"].dropna().unique()]
    )
    default_types = [t for t in tipo_options if _is_accident_type(t)]
    if not default_types:
        default_types = tipo_options
    selected_types = st.multiselect(
        "Tipo de evento",
        options=tipo_options,
        default=default_types,
        key="events_type_filter",
    )
    if selected_types:
        filtered = filtered[filtered["tipo_evento"].isin(selected_types)]

    min_dt = filtered["evento_time"].min()
    max_dt = filtered["evento_time"].max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        st.info("No hay fechas validas para aplicar filtro temporal.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha inicio",
                value=min_dt.date(),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
                key="events_start_date",
            )
        with col2:
            end_date = st.date_input(
                "Fecha fin",
                value=max_dt.date(),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
                key="events_end_date",
            )
        start_ts = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
        end_ts = pd.Timestamp(datetime.combine(end_date, datetime.max.time()))
        filtered = filtered[
            (filtered["evento_time"] >= start_ts)
            & (filtered["evento_time"] <= end_ts)
        ]

    segments_df = _build_segments_df(porticos_df)
    if segments_df.empty:
        return filtered, segments_df

    # --- FIX: Ensure merge keys are string to avoid object vs float64 error ---
    segments_df["portico_inicio"] = segments_df["portico_inicio"].astype(str).str.strip()
    segments_df["portico_fin"] = segments_df["portico_fin"].astype(str).str.strip()
    
    filtered["portico_inicio"] = filtered["portico_inicio"].fillna("").astype(str).str.strip()
    filtered["portico_fin"] = filtered["portico_fin"].fillna("").astype(str).str.strip()
    # --------------------------------------------------------------------------

    counts_df = (
        filtered.groupby(
            ["eje", "calzada", "portico_inicio", "portico_fin"],
            dropna=False,
        )
        .size()
        .reset_index(name="eventos")
    )
    
    # Ensure counts_df also has string keys
    counts_df["portico_inicio"] = counts_df["portico_inicio"].astype(str).str.strip()
    counts_df["portico_fin"] = counts_df["portico_fin"].astype(str).str.strip()
    
    segments_df = segments_df.merge(
        counts_df,
        left_on=["Eje", "Calzada", "portico_inicio", "portico_fin"],
        right_on=["eje", "calzada", "portico_inicio", "portico_fin"],
        how="left",
    )
    segments_df["eventos"] = (
        segments_df["eventos"].fillna(0).astype(int)
    )
    segments_df = segments_df.sort_values(
        ["Eje", "Calzada", "orden_inicio"]
    ).reset_index(drop=True)

    tramo_options = ["Toda la autopista"]
    tramo_lookup: Dict[str, Tuple[str, str, str, str]] = {}
    for row in segments_df.itertuples(index=False):
        label = (
            f"{row.Eje} | {row.Calzada} | "
            f"{row.portico_inicio} -> {row.portico_fin} "
            f"({row.eventos} eventos)"
        )
        tramo_options.append(label)
        tramo_lookup[label] = (
            row.Eje,
            row.Calzada,
            row.portico_inicio,
            row.portico_fin,
        )

    tramo_choice = st.selectbox(
        "Tramo de autopista",
        options=tramo_options,
        key="events_tramo_filter",
    )
    if tramo_choice != "Toda la autopista":
        eje, calzada, portico_inicio, portico_fin = tramo_lookup[tramo_choice]
        filtered = filtered[
            (filtered["eje"] == eje)
            & (filtered["calzada"] == calzada)
            & (filtered["portico_inicio"] == portico_inicio)
            & (filtered["portico_fin"] == portico_fin)
        ]

    return filtered, segments_df


def _get_mapped_events_df(
    events_df: pd.DataFrame, porticos_df: pd.DataFrame, *, jitter: bool = True
) -> Tuple[pd.DataFrame, Optional[str]]:
    # Attempt to parse combined lat-lon column first if it exists
    if porticos_df is not None and not porticos_df.empty:
        lat_lon_col = _find_column(porticos_df, ["lat-lon", "lat_lon", "coordenadas"])
        if lat_lon_col and ("lat" not in porticos_df.columns or "lon" not in porticos_df.columns):
            try:
                split_coords = (
                    porticos_df[lat_lon_col].astype(str).str.split(",", expand=True)
                )
                if split_coords.shape[1] >= 2:
                    porticos_df["lat"] = pd.to_numeric(split_coords[0], errors="coerce")
                    porticos_df["lon"] = pd.to_numeric(split_coords[1], errors="coerce")
            except Exception:
                pass

    mapped = events_df.copy()
    has_event_coords = "lat" in mapped.columns and "lon" in mapped.columns
    all_missing = True
    if has_event_coords:
        all_missing = mapped[["lat", "lon"]].isna().all().all()

    if not has_event_coords or all_missing:
        if "lat" not in porticos_df.columns or "lon" not in porticos_df.columns:
            return (
                pd.DataFrame(),
                "Faltan coordenadas en Porticos.csv (lat/lon) para dibujar el mapa.",
            )
        mapped = _interpolate_event_coords(events_df, porticos_df)
    elif mapped[["lat", "lon"]].isna().any().any():
        if "lat" in porticos_df.columns and "lon" in porticos_df.columns:
            computed = _interpolate_event_coords(events_df, porticos_df)
            mapped["lat"] = mapped["lat"].fillna(computed["lat"])
            mapped["lon"] = mapped["lon"].fillna(computed["lon"])
    
    # If we have valid coords, try to snap them to highway to fix interpolation straight-line errors
    if "highway_geo" in st.session_state and st.session_state["highway_geo"] is not None:
        mapped = _snap_df_to_highway(mapped, st.session_state["highway_geo"])
        
    mapped = mapped.dropna(subset=["lat", "lon"])
    
    # Prepare display columns for tooltip
    if "evento_time" in mapped.columns:
        mapped["evento_time_str"] = mapped["evento_time"].astype(str)
    else:
        mapped["evento_time_str"] = ""
        
    if "Descripcion" not in mapped.columns:
        mapped["Descripcion"] = ""
    
    # Fallback to SubTipo if Descripcion is empty/NaN
    if "SubTipo" in mapped.columns:
        mapped["Descripcion"] = mapped["Descripcion"].astype("string")
        mapped["SubTipo"] = mapped["SubTipo"].astype("string")
        desc_clean = mapped["Descripcion"].str.strip().str.lower()
        mask_empty = (
            mapped["Descripcion"].isna()
            | desc_clean.isna()
            | desc_clean.isin({"", "nan", "<na>", "none", "null"})
        )
        mapped.loc[mask_empty, "Descripcion"] = mapped.loc[mask_empty, "SubTipo"]
        mapped["Descripcion"] = mapped["Descripcion"].fillna("").astype(str)
    else:
         mapped["Descripcion"] = mapped["Descripcion"].fillna("").astype(str)

    # Add display properties
    # Red color for events
    mapped["color"] = pd.Series([[200, 30, 0, 160]] * len(mapped))

    if jitter:
        # Add Jitter to avoid perfect stacking (approx 10-15m)
        # 0.0001 degree is roughly 11 meters
        rng = np.random.default_rng(42)
        mapped["lat"] = mapped["lat"] + rng.uniform(-0.00015, 0.00015, size=len(mapped))
        mapped["lon"] = mapped["lon"] + rng.uniform(-0.00015, 0.00015, size=len(mapped))
    
    return mapped, None


def _render_map(
    mapped_events: Optional[pd.DataFrame],
    heatmap_events: Optional[pd.DataFrame],
    highway_geo: Optional[pd.DataFrame],
    porticos_df: Optional[pd.DataFrame] = None,
    *,
    show_points: bool = True,
    show_heatmap: bool = False,
) -> None:
    layers = []
    
    # Calculate initial view state
    initial_view_state = pdk.ViewState(
        latitude=-33.45,
        longitude=-70.66,
        zoom=10,
        pitch=0,
    )

    # 1. Highway Layer (Background)
    if highway_geo is not None and not highway_geo.empty:
        highway_layer = pdk.Layer(
            "PathLayer",
            data=highway_geo,
            get_path="path",
            get_color="color",
            width_min_pixels=2,
            width_scale=1,
            pickable=True,
            opacity=0.5,
        )
        layers.append(highway_layer)

    # 2. Porticos Layer (Infrastructure - Blue)
    if porticos_df is not None and not porticos_df.empty:
        # Ensure we have coordinates
        if "lat" in porticos_df.columns and "lon" in porticos_df.columns:
            p_data = porticos_df.dropna(subset=["lat", "lon"]).copy()
            
            # Filter by 'aux' column if present
            if "aux" in p_data.columns:
                 # Convert to numeric to handle string/int mix safely
                 p_data["aux_numeric"] = (
                     p_data["aux"]
                     .astype(str)
                     .str.replace(",", ".")
                     .pipe(pd.to_numeric, errors="coerce")
                     .fillna(0)
                 )
                 p_data = p_data[p_data["aux_numeric"] == 0]

            if not p_data.empty:
                # Prepare tooltip columns to match the shared template
                # Template: <b>Evento:</b> {tipo_evento}<br><b>Hora:</b> {evento_time_str}<br><b>Desc:</b> {Descripcion}
                
                p_data["tipo_evento"] = ( "Pórtico " + p_data["portico"].astype(str) )
                p_data["evento_time_str"] = "Km " + p_data["km"].astype(str)
                p_data["Descripcion"] = (
                    " (" + p_data["eje"].astype(str) 
                    + " " + p_data["calzada"].astype(str) + ")"
                )

                # Blue color: [0, 100, 255, 200]
                p_data["color"] = pd.Series([[0, 100, 255, 200]] * len(p_data))
                
                porticos_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=p_data,
                    get_position=["lon", "lat"],
                    get_color="color",
                    get_radius=80, # Slightly smaller than events
                    pickable=True,
                    auto_highlight=True,
                    opacity=0.8,
                    radius_min_pixels=2,
                    radius_max_pixels=10,
                )
                layers.append(porticos_layer)

    # 3. Heatmap Layer (if enabled)
    if (
        show_heatmap
        and heatmap_events is not None
        and not heatmap_events.empty
    ):
        initial_view_state.latitude = heatmap_events["lat"].median()
        initial_view_state.longitude = heatmap_events["lon"].median()
        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=heatmap_events,
            get_position=["lon", "lat"],
            get_weight=1,
            radiusPixels=40,
            intensity=1.0,
            threshold=0.05,
        )
        layers.append(heat_layer)

    # 4. Events Layer (Red - Top)
    if show_points and mapped_events is not None and not mapped_events.empty:
        # Check median of events for better centering if available
        initial_view_state.latitude = mapped_events["lat"].median()
        initial_view_state.longitude = mapped_events["lon"].median()

        events_layer = pdk.Layer(
            "ScatterplotLayer",
            data=mapped_events,
            get_position=["lon", "lat"],
            get_color="color",
            get_radius=120,
            pickable=True,
            auto_highlight=True,
            opacity=0.9,
            radius_min_pixels=3,
            radius_max_pixels=15,
        )
        layers.append(events_layer)

    if not layers:
        st.info("No hay capas para mostrar en el mapa.")
        return

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=initial_view_state,
            layers=layers,
            tooltip={
                "html": "<b>Evento:</b> {tipo_evento}<br><b>Hora:</b> {evento_time_str}<br><b>Desc:</b> {Descripcion}",
                "style": {"color": "white"},
            },
        )
    )
    if show_points and mapped_events is not None:
         st.caption(f"Eventos mapeados: {len(mapped_events):,}")
    if show_heatmap and heatmap_events is not None:
         st.caption(f"Accidentes en mapa de calor: {len(heatmap_events):,}")


def _render_events_table(events_df: pd.DataFrame) -> None:
    detail_candidates = [
        "evento_time",
        "tipo_evento",
        "ultimo_portico",
        "portico_inicio",
        "portico_fin",
        "eje",
        "calzada",
        "km",
        "Descripcion",
        "SubTipo",
    ]
    detail_cols = _select_columns(events_df, detail_candidates)
    detail_cols = [col for col in detail_cols if col in events_df.columns]
    if not detail_cols:
        st.info("No hay columnas para mostrar en detalle.")
        return
    st.dataframe(events_df[detail_cols].head(500), width="stretch")

def _normalize_portico_val(val):
    if pd.isna(val):
        return val
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return str(val).strip()


def _clean_portico_str(val: object) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _ensure_porticos_coords(porticos_df: pd.DataFrame) -> pd.DataFrame:
    """Ensures porticos_df has 'lat' and 'lon' columns, loading from fallback if needed."""
    
    # helper to find existing coord columns
    lat_col = _find_column(porticos_df, ["lat", "latitude", "latitud", "y"])
    lon_col = _find_column(porticos_df, ["lon", "longitude", "longitud", "lng", "x"])
    
    # If already present, standardize and return
    if lat_col and lon_col:
        porticos_df["lat"] = pd.to_numeric(porticos_df[lat_col], errors="coerce")
        porticos_df["lon"] = pd.to_numeric(porticos_df[lon_col], errors="coerce")
        return porticos_df

    # Parse combined lat-lon column if present (avoid merge duplicates)
    latlon_col = _find_column(porticos_df, ["lat-lon", "lat_lon", "coordenadas"])
    if latlon_col:
        split_coords = (
            porticos_df[latlon_col].astype(str).str.split(",", expand=True)
        )
        if split_coords.shape[1] >= 2:
            porticos_df["lat"] = pd.to_numeric(split_coords[0], errors="coerce")
            porticos_df["lon"] = pd.to_numeric(split_coords[1], errors="coerce")
        return porticos_df

    # Load raw to get coord column
    csv_path = DATA_DIR / "Porticos.csv"
    if not csv_path.exists():
        return porticos_df
    
    try:
        raw = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        return porticos_df

    # Find lat-lon column
    coord_col = _find_column(raw, ["lat-lon", "lat_lon", "lat", "latitude"])
    if not coord_col:
        return porticos_df
    
    # Identify key column to merge on (cod_portico)
    key_col = _find_column(raw, ["cod_portico", "cod", "portico"])
    if not key_col:
        return porticos_df

    eje_col = _find_column(raw, ["Eje"])
    calzada_col = _find_column(raw, ["Calzada"])

    # Normalize key in raw to match porticos_df["portico"]
    raw["clean_key"] = raw[key_col].apply(_normalize_portico_val)
    if eje_col:
        raw["clean_eje"] = raw[eje_col].astype(str).str.strip().str.upper()
    if calzada_col:
        raw["clean_calzada"] = (
            raw[calzada_col].astype(str).str.strip().str.upper()
        )
    
    # Extract coordinates
    merge_cols = ["clean_key", coord_col]
    if eje_col and calzada_col:
        merge_cols.extend(["clean_eje", "clean_calzada"])
    to_merge = raw[merge_cols].copy()
    to_merge = to_merge.rename(columns={coord_col: "lat_lon_raw"})
    
    # Ensure join keys are strings to avoid int64 vs object merge error
    porticos_df["portico"] = porticos_df["portico"].astype(str).str.strip()
    to_merge["clean_key"] = to_merge["clean_key"].astype(str).str.strip()
    if eje_col and calzada_col:
        porticos_df["eje_norm"] = porticos_df["eje"].astype(str).str.strip().str.upper()
        porticos_df["calzada_norm"] = (
            porticos_df["calzada"].astype(str).str.strip().str.upper()
        )

    # Merge
    if eje_col and calzada_col:
        porticos_df = porticos_df.merge(
            to_merge,
            left_on=["portico", "eje_norm", "calzada_norm"],
            right_on=["clean_key", "clean_eje", "clean_calzada"],
            how="left",
        )
    else:
        porticos_df = porticos_df.merge(
            to_merge,
            left_on="portico",
            right_on="clean_key",
            how="left",
        )
    
    # Parse splitting
    if "lat_lon_raw" in porticos_df.columns:
         split_coords = porticos_df["lat_lon_raw"].astype(str).str.split(",", expand=True)
         if split_coords.shape[1] >= 2:
             porticos_df["lat"] = pd.to_numeric(split_coords[0], errors="coerce")
             porticos_df["lon"] = pd.to_numeric(split_coords[1], errors="coerce")
    
    return porticos_df


def _km_fraction(event_km: object, km_start: object, km_end: object) -> Optional[float]:
    try:
        event_val = float(event_km)
        start_val = float(km_start)
        end_val = float(km_end)
    except (TypeError, ValueError):
        return None
    if np.isclose(start_val, end_val):
        return 0.0
    ratio = (event_val - start_val) / (end_val - start_val)
    if not np.isfinite(ratio):
        return None
    return float(np.clip(ratio, 0.0, 1.0))


def _great_circle_interpolate(
    lat1: float, lon1: float, lat2: float, lon2: float, fraction: float
) -> Tuple[float, float]:
    if fraction <= 0:
        return lat1, lon1
    if fraction >= 1:
        return lat2, lon2
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    lambda1 = np.radians(lon1)
    lambda2 = np.radians(lon2)
    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    delta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    if np.isclose(delta, 0):
        return lat1, lon1
    sin_delta = np.sin(delta)
    factor_a = np.sin((1 - fraction) * delta) / sin_delta
    factor_b = np.sin(fraction * delta) / sin_delta
    x = factor_a * np.cos(phi1) * np.cos(lambda1) + factor_b * np.cos(phi2) * np.cos(lambda2)
    y = factor_a * np.cos(phi1) * np.sin(lambda1) + factor_b * np.cos(phi2) * np.sin(lambda2)
    z = factor_a * np.sin(phi1) + factor_b * np.sin(phi2)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    lambda_ = np.arctan2(y, x)
    return float(np.degrees(phi)), float(np.degrees(lambda_))

def _project_point_to_segment(p, a, b):
    """Project point p onto segment ab."""
    ap = p - a
    ab = b - a
    
    if np.all(ab == 0):
        return a
        
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    
    return a + t * ab

def _snap_df_to_highway(df: pd.DataFrame, highway_geo: pd.DataFrame) -> pd.DataFrame:
    """Snaps each point in df (lat, lon) to the nearest point on the highway network."""
    if df.empty or highway_geo is None or highway_geo.empty:
        return df
        
    # check standard coords
    if "lat" not in df.columns or "lon" not in df.columns:
        return df

    # Flatten highway geometry into segments
    # highway_geo["path"] is a list of [lon, lat] points
    segments = []
    for path in highway_geo["path"]:
        if len(path) < 2:
            continue
        path_arr = np.array(path) # shape (N, 2) -> columns are [lon, lat]
        # Create segments (start, end)
        for i in range(len(path_arr) - 1):
            segments.append((path_arr[i], path_arr[i+1]))
            
    if not segments:
        return df

    # Snap each point
    # This acts on [lon, lat] because pydeck uses that order in 'path'
    
    snapped_lats = []
    snapped_lons = []
    
    # Pre-convert segments to numpy array
    segments_arr = np.array(segments) 
    
    # Iterate
    # Brute force (O(N*M))
    for _, row in df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            snapped_lons.append(row.get("lon"))
            snapped_lats.append(row.get("lat"))
            continue
            
        p_lon = row["lon"]
        p_lat = row["lat"]
        p_vec = np.array([p_lon, p_lat])
        
        best_dist_sq = float("inf")
        best_point = p_vec
        
        for seg in segments_arr:
            a = seg[0]
            b = seg[1]
            proj = _project_point_to_segment(p_vec, a, b)
            dist_sq = np.sum((p_vec - proj)**2)
            
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_point = proj
                
        snapped_lons.append(best_point[0])
        snapped_lats.append(best_point[1])
        
    df = df.copy()
    df["lon"] = snapped_lons
    df["lat"] = snapped_lats
    
    return df

def _interpolate_event_coords(events_df: pd.DataFrame, porticos_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates event coordinates using ultimo/proximo portico + geodesic interpolation."""
    if events_df.empty or porticos_df.empty:
        return events_df

    if "lat" not in porticos_df.columns or "lon" not in porticos_df.columns:
        return events_df

    p_clean = porticos_df.copy()
    p_clean["portico"] = p_clean["portico"].astype(str).str.strip()
    p_clean["km"] = pd.to_numeric(p_clean["km"], errors="coerce")
    p_clean["lat"] = pd.to_numeric(p_clean["lat"], errors="coerce")
    p_clean["lon"] = pd.to_numeric(p_clean["lon"], errors="coerce")
    p_clean = p_clean.dropna(subset=["portico"])
    p_clean = p_clean.drop_duplicates(subset=["portico"], keep="first")

    portico_lookup = (
        p_clean.set_index("portico")[["lat", "lon", "km"]]
        .to_dict(orient="index")
    )
    porticos_by_group: Dict[Tuple[object, object], pd.DataFrame] = {}
    for group_key, group_df in p_clean.dropna(
        subset=["km", "lat", "lon"]
    ).groupby(["eje", "calzada"]):
        porticos_by_group[group_key] = group_df.sort_values("km").reset_index(drop=True)

    events_temp = events_df.copy()
    if "km" not in events_temp.columns:
        return events_df
    if events_temp["km"].dtype == object:
        events_temp["km"] = (
            events_temp["km"].astype(str).str.replace(",", ".").replace("nan", np.nan)
        )
    events_temp["km"] = pd.to_numeric(events_temp["km"], errors="coerce")

    for col in ("portico_inicio", "portico_fin", "ultimo_portico"):
        if col in events_temp.columns:
            events_temp[col] = events_temp[col].apply(_clean_portico_str)

    interpolated_lats: List[float] = []
    interpolated_lons: List[float] = []

    for row in events_temp.itertuples(index=False):
        start_id = getattr(row, "portico_inicio", None) or getattr(
            row, "ultimo_portico", None
        )
        end_id = getattr(row, "portico_fin", None)
        event_km = getattr(row, "km", np.nan)

        start = portico_lookup.get(start_id) if start_id else None
        end = portico_lookup.get(end_id) if end_id else None

        coords = None
        if start and end:
            ratio = _km_fraction(event_km, start.get("km"), end.get("km"))
            if ratio is not None:
                if pd.notna(start.get("lat")) and pd.notna(start.get("lon")) and pd.notna(
                    end.get("lat")
                ) and pd.notna(end.get("lon")):
                    coords = _great_circle_interpolate(
                        start["lat"], start["lon"], end["lat"], end["lon"], ratio
                    )

        if coords is None:
            if start and pd.notna(start.get("lat")) and pd.notna(start.get("lon")):
                coords = (start["lat"], start["lon"])
            elif end and pd.notna(end.get("lat")) and pd.notna(end.get("lon")):
                coords = (end["lat"], end["lon"])

        if coords is None and pd.notna(event_km):
            eje_key = getattr(row, "eje", None)
            calzada_key = getattr(row, "calzada", None)
            group = porticos_by_group.get((eje_key, calzada_key))
            if group is not None and not group.empty:
                dists = (group["km"] - event_km).abs()
                nearest = dists.argsort()[:2]
                p1 = group.iloc[int(nearest[0])]
                if len(nearest) == 1:
                    coords = (p1["lat"], p1["lon"])
                else:
                    p2 = group.iloc[int(nearest[1])]
                    ratio = _km_fraction(event_km, p1["km"], p2["km"])
                    if ratio is not None:
                        coords = _great_circle_interpolate(
                            p1["lat"], p1["lon"], p2["lat"], p2["lon"], ratio
                        )
                    else:
                        coords = (p1["lat"], p1["lon"])

        if coords is None:
            interpolated_lats.append(np.nan)
            interpolated_lons.append(np.nan)
        else:
            interpolated_lats.append(coords[0])
            interpolated_lons.append(coords[1])

    events_temp["lat"] = interpolated_lats
    events_temp["lon"] = interpolated_lons
    return events_temp


def _compute_event_coords(
    events_df: pd.DataFrame,
    porticos_df: pd.DataFrame,
    highway_geo: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return events_df
    coords_df = _interpolate_event_coords(events_df, porticos_df)
    if highway_geo is not None and not highway_geo.empty:
        coords_df = _snap_df_to_highway(coords_df, highway_geo)
    return coords_df


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    _init_state()
    if set_page_config:
        st.set_page_config(page_title="Events map", layout="wide")
    st.title("Events")

    if st.session_state.get("highway_geo") is None:
        st.session_state["highway_geo"] = _load_highway_geo()

    if st.session_state.get("events_df") is None and _events_db_exists():
        try:
            events_from_db = _load_events_db()
            if not events_from_db.empty:
                st.session_state["events_df"] = events_from_db
                st.session_state["event_files"] = ["Base eventos"]
                st.session_state["porticos_source"] = "Datos/Porticos.csv"
        except Exception as exc:
            st.warning(f"No se pudo cargar la base de eventos: {exc}")

    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)

    st.subheader("Carga de eventos")
    event_files = _list_event_files()
    selected_names: List[str] = []
    if event_files:
        selected_names = st.multiselect(
            "Archivos de eventos disponibles",
            [path.name for path in event_files],
            default=[path.name for path in event_files],
        )
    else:
        st.info("No se encontraron archivos de eventos en la carpeta Datos.")

    if st.button("Cargar eventos"):
        if not selected_names:
            st.warning("Seleccione al menos un archivo de eventos.")
            return
        # Barra de progreso y status
        progress_bar = st.progress(0, text="Iniciando carga de eventos...")
        
        try:
            # 1. Cargar Pórticos (10%)
            progress_bar.progress(10, text="Cargando y procesando pórticos...")
            try:
                porticos_df = load_porticos()
                # Try fallback augmentation
                porticos_df = _ensure_porticos_coords(porticos_df)
                porticos_source = "Datos/Porticos.csv"
            except Exception as exc:
                st.error(f"No se pudieron cargar los porticos: {exc}")
                progress_bar.empty()
                return

            # 2. Leer archivos de eventos (30%)
            progress_bar.progress(30, text="Leyendo archivos de eventos...")
            try:
                raw_df = _read_event_files(selected_names)
            except Exception as exc:
                st.error(f"No se pudieron leer los eventos: {exc}")
                progress_bar.empty()
                return

            highway_geo = st.session_state.get("highway_geo")
            
            # 3. Snap porticos (40%)
            if porticos_df is not None and highway_geo is not None:
                 progress_bar.progress(40, text="Ajustando pórticos a la autopista...")
                 porticos_df = _snap_df_to_highway(porticos_df, highway_geo)

            # 4. Preparar DataFrame de eventos (70%)
            progress_bar.progress(50, text="Procesando datos y segmentos...")
            events_df = _prepare_events_df(raw_df, porticos_df)
            excluded_accidents = int(events_df.attrs.get("accidents_excluded", 0) or 0)
            
            # 5. Calcular coordenadas (90%)
            progress_bar.progress(80, text="Calculando coordenadas geográficas...")
            events_df = _compute_event_coords(events_df, porticos_df, highway_geo)
            
            # 6. Guardar en DB (95%)
            progress_bar.progress(95, text="Actualizando base de datos local...")
            events_db_df = _prepare_events_db_frame(events_df)
            try:
                inserted = _write_events_db(events_db_df)
                st.success(
                    f"Eventos cargados: {len(events_db_df):,} | Base actualizada: {inserted:,}"
                )
            except Exception as exc:
                st.warning(f"No se pudo actualizar la base de eventos: {exc}")
                st.success(f"Eventos cargados: {len(events_db_df):,}")
            
            if excluded_accidents:
                st.warning(
                    f"Accidentes sin pórtico (excluidos del cálculo): {excluded_accidents:,}"
                )
            
            st.session_state["events_df"] = events_db_df
            st.session_state["event_files"] = selected_names
            st.session_state["porticos_source"] = porticos_source
            
            # Finalizar
            progress_bar.progress(100, text="¡Carga completa!")
            # Retraso breve o limpieza opcional? Dejamos que se vea el 100%
            
        except Exception as e:
            st.error(f"Error inesperado durante la carga: {e}")
        finally:
             # Opcional: limpiar la barra de progreso después de un momento si se desea
             # progress_bar.empty()
             pass

    events_df = st.session_state.get("events_df")
    highway_geo = st.session_state.get("highway_geo")
    
    # Allow showing map if highway_geo is present, even if no events
    if (events_df is None or events_df.empty) and (highway_geo is None or highway_geo.empty):
        st.info("Cargue eventos o nodos de autopista para visualizar.")
        return

    files_label = ", ".join(st.session_state.get("event_files", [])) or "-"
    st.caption(
        f"Archivos: {files_label} | "
        f"Porticos: {st.session_state.get('porticos_source') or '-'} | "
        f"Nodos Autopista: {'Cargados' if highway_geo is not None else 'No encontrados'} | "
        "Via: expresa"
    )

    try:
        porticos_df = load_porticos()
        porticos_df = _ensure_porticos_coords(porticos_df)
        if highway_geo is not None:
             porticos_df = _snap_df_to_highway(porticos_df, highway_geo)
    except Exception as exc:
        st.warning(f"No se pudieron cargar los porticos: {exc}")
        return

    if events_df is not None and not events_df.empty:
        events_df, fixed_count = _fill_missing_porticos(events_df, porticos_df)
        if fixed_count:
            events_df = _compute_event_coords(events_df, porticos_df, highway_geo)
            events_db_df = _prepare_events_db_frame(events_df)
            st.session_state["events_df"] = events_db_df
            events_df = events_db_df
            try:
                _write_events_db(events_db_df)
            except Exception as exc:
                st.warning(f"No se pudo actualizar la base de eventos: {exc}")
            st.info(
                f"Se recalcularon pórticos para {fixed_count:,} eventos sin pórtico."
            )
        else:
            st.session_state["events_df"] = events_df

        if "portico_inicio" in events_df.columns:
            missing_mask = events_df["portico_inicio"].isna()
            missing_df = events_df[missing_mask]
            
            # Check if any of the missing events are accidents
            missing_accidents = missing_df[
                missing_df["tipo_evento"].apply(_is_accident_type)
            ]
            
            if not missing_accidents.empty:
                st.warning(
                    f"Quedan {len(missing_accidents):,} accidentes sin pórtico. "
                    "Revise KM, Eje o Calzada en los datos."
                )

    filtered_df = pd.DataFrame()
    if events_df is not None and not events_df.empty:
        st.subheader("Filtros")
        filtered_df, _ = _render_filters(events_df, porticos_df)
        
        total_events = int(len(events_df))
        filtered_events = int(len(filtered_df))
        col1, col2 = st.columns(2)
        col1.metric("Eventos totales", f"{total_events:,}")
        col2.metric("Eventos filtrados", f"{filtered_events:,}")
    else:
        st.info("Visualizando solo estructura de autopista (sin eventos).")

    st.subheader("Mapa de eventos")
    mapped_df = None
    heat_df = pd.DataFrame()
    if not filtered_df.empty:
        mapped_df, map_error = _get_mapped_events_df(filtered_df, porticos_df, jitter=True)
        if map_error:
            st.warning(map_error)
            mapped_df = None
        accidents_df = filtered_df
        if "tipo_evento" in filtered_df.columns:
            accidents_df = filtered_df[
                filtered_df["tipo_evento"].apply(_is_accident_type)
            ]
        if not accidents_df.empty:
            heat_df, heat_error = _get_mapped_events_df(
                accidents_df, porticos_df, jitter=False
            )
            if heat_error:
                st.warning(heat_error)
    layer_options = [
        "Accidentes (puntos)",
        "Mapa de calor",
    ]
    selected_layers = st.multiselect(
        "Capas del mapa",
        options=layer_options,
        default=layer_options,
        key="events_map_layers",
    )
    show_points = "Accidentes (puntos)" in selected_layers
    show_heatmap = "Mapa de calor" in selected_layers

    _render_map(
        mapped_df,
        heat_df,
        highway_geo,
        porticos_df,
        show_points=show_points,
        show_heatmap=show_heatmap,
    )

    if not filtered_df.empty:
        st.subheader("Detalle de eventos filtrados")
        _render_events_table(filtered_df)


if __name__ == "__main__":
    main()
