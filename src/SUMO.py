#!/usr/bin/env python3
"""
SUMO.py
=======
Implementa la canalización de cálculo descrita para preparar datos de pórticos,
construir trayectorias discretas y derivar métricas para la calibración de SUMO.
"""
from __future__ import annotations

import math
import re
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import FlowColumns

DEFAULT_SPEED_MIN = 0
DEFAULT_SPEED_MAX = 200
MAX_TIME_GAP_S = 3600  # 60 minutos
MAX_DISTANCE_JUMP_KM = 50
MIN_IMPL_SPEED = 5
MAX_IMPL_SPEED = 160

DEFAULT_VTYPE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "car": {
        "accel": "0.8",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "4.5",
        "minGap": "2.5",
        "maxSpeed": "180",
        "color": "0.15,0.45,0.95",
        "vClass": "passenger",
        "guiShape": "passenger",
    },
    "bus": {
        "accel": "0.5",
        "decel": "4.0",
        "sigma": "0.5",
        "length": "12",
        "minGap": "2.5",
        "maxSpeed": "90",
        "color": "1.0,0.55,0.0",
        "vClass": "bus",
        "guiShape": "bus",
    },
    "truck": {
        "accel": "0.4",
        "decel": "4.0",
        "sigma": "0.5",
        "length": "10",
        "minGap": "2.5",
        "maxSpeed": "120",
        "color": "0.50,0.30,0.70",
        "vClass": "truck",
        "guiShape": "truck/semitrailer",
    },
    "motorcycle": {
        "accel": "1.0",
        "decel": "5.0",
        "sigma": "0.5",
        "length": "2.0",
        "minGap": "1.5",
        "maxSpeed": "180",
        "color": "0.0,0.75,0.45",
        "vClass": "motorcycle",
        "guiShape": "motorcycle",
    },
}

CLASS_TO_TEMPLATE = {
    1: "car",
    2: "bus",
    3: "truck",
    4: "motorcycle",
}


@dataclass
class SUMOResult:
    """
    Resultado agregado del pipeline SUMO.
    """

    clean_events: pd.DataFrame
    trajectories: pd.DataFrame
    segments: pd.DataFrame
    macro_metrics: pd.DataFrame
    headways: pd.DataFrame
    segment_filter: Optional["SegmentFilter"] = None
    sumo_trips: Optional[List["SumoTrip"]] = None
    sumo_trips_path: Optional[Path] = None
    depart_summary_path: Optional[Path] = None
    sumo_warning: Optional[str] = None
    vehicle_profiles: Optional[Dict[str, Dict[str, str]]] = None


@dataclass
class SegmentFilter:
    """
    Representa un tramo específico entre pórticos.
    """

    eje: str
    calzada: str
    start_portico: str
    end_portico: str
    portico_ids: List[str]

    def description(self) -> str:
        return (
            f"{self.eje} / {self.calzada}: "
            f"{self.start_portico} → {self.end_portico} "
            f"({len(self.portico_ids)} pórticos)"
        )


@dataclass
class SumoTrip:
    trip_id: str
    plate: str
    vehicle_type_id: Optional[int]
    depart_s: float
    from_edge: str
    to_edge: str
    via_edges: List[str]
    depart_lane: Optional[str]
    depart_pos: Optional[float]
    depart_speed: Optional[float]


def _normalize_plate(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.upper()
    )
    invalid_tokens = {"", "NAN", "NULL", "NONE"}
    mask = series.isna() | cleaned.isin(invalid_tokens)
    return cleaned.where(~mask, None)


def _prepare_portico_catalog(porticos_df: pd.DataFrame) -> pd.DataFrame:
    if porticos_df is None or porticos_df.empty:
        raise ValueError("No se proporcionaron datos de pórticos para SUMO.")
    catalog = porticos_df.copy()
    catalog["portico_id"] = catalog["portico"].astype(str).str.strip()
    catalog["km"] = pd.to_numeric(catalog["km"], errors="coerce")
    catalog["sentido"] = (
        catalog["eje"].astype(str).str.strip().str.upper()
        + "-"
        + catalog["calzada"].astype(str).str.strip().str.upper()
    )
    catalog = catalog.dropna(subset=["portico_id", "km"])
    catalog = catalog.drop_duplicates(subset="portico_id")
    return catalog[["portico_id", "km", "calzada", "eje", "sentido", "orden"]]


def _prepare_segment_groups(porticos_df: pd.DataFrame) -> List[Tuple[Tuple[str, str], pd.DataFrame]]:
    catalog = _prepare_portico_catalog(porticos_df).copy()
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


def build_portico_sumo_table(porticos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica que existan las columnas necesarias para mapear pórticos a la red SUMO.
    """
    required_cols = {"edge_id_sumo", "lane_id_sumo", "pos_m"}
    missing = required_cols - set(porticos_df.columns.str.lower())
    if missing:
        raise ValueError(
            "El archivo de pórticos no contiene las columnas necesarias para generar trips SUMO "
            "(se requieren edge_id_sumo, lane_id_sumo y pos_m)."
        )

    df = porticos_df.copy()
    # Acepta nombres con mayúsculas mezcladas
    for col in required_cols:
        matched = [c for c in df.columns if c.lower() == col]
        if not matched:
            raise ValueError(f"No se encontró la columna '{col}' en el archivo de pórticos.")
        if col != matched[0]:
            df.rename(columns={matched[0]: col}, inplace=True)

    df["portico_id"] = df["portico"].astype(str).str.strip()
    df["lane_id_sumo"] = df["lane_id_sumo"].astype(str).str.strip()
    df["pos_m"] = pd.to_numeric(df["pos_m"], errors="coerce")
    df = df.dropna(subset=["portico_id", "edge_id_sumo", "lane_id_sumo", "pos_m"])
    return df[
        [
            "portico_id",
            "edge_id_sumo",
            "lane_id_sumo",
            "pos_m",
            "km",
            "calzada",
            "eje",
        ]
    ].drop_duplicates(subset="portico_id")


def _prompt_int(prompt: str, min_value: int, max_value: int) -> int:
    while True:
        value = input(prompt).strip()
        try:
            number = int(value)
        except ValueError:
            print("Entrada inválida. Intente nuevamente.")
            continue
        if min_value <= number <= max_value:
            return number
        print(f"Ingrese un número entre {min_value} y {max_value}.")


def prompt_segment_filter(porticos_df: pd.DataFrame) -> Optional[SegmentFilter]:
    resp = input("\n¿Desea analizar un tramo específico entre pórticos? (s/n): ").strip().lower()
    if resp not in {"s", "si", "y", "yes"}:
        return None

    groups = _prepare_segment_groups(porticos_df)
    if not groups:
        print("⚠️ No se encontraron combinaciones válidas de eje/cazada para definir un tramo.")
        return None

    print("\nSeleccione el eje/calzada del tramo:")
    for idx, (key, _) in enumerate(groups, start=1):
        eje, calzada = key
        print(f"  [{idx}] {eje} / {calzada}")
    choice = _prompt_int("Opción: ", 1, len(groups))
    eje, calzada = groups[choice - 1][0]
    group_df = groups[choice - 1][1]

    print(f"\nPórticos disponibles para {eje} / {calzada}:")
    for idx, row in group_df.iterrows():
        print(f"  [{idx}] {row['portico_id']} (km {row['km']})")
    start_idx = _prompt_int("Índice de pórtico inicial: ", 0, len(group_df) - 2)
    end_idx = _prompt_int(
        f"Índice de pórtico final (>{start_idx}): ",
        start_idx + 1,
        len(group_df) - 1,
    )

    selected_porticos = (
        group_df.loc[start_idx:end_idx, "portico_id"].astype(str).str.strip().tolist()
    )
    segment = SegmentFilter(
        eje=eje,
        calzada=calzada,
        start_portico=str(group_df.loc[start_idx, "portico_id"]).strip(),
        end_portico=str(group_df.loc[end_idx, "portico_id"]).strip(),
        portico_ids=selected_porticos,
    )
    print(f"\nTramo seleccionado: {segment.description()}")
    return segment


def _vehicle_type_sumo_id(vehicle_type_id: Optional[float]) -> Optional[str]:
    if vehicle_type_id is None or (isinstance(vehicle_type_id, float) and math.isnan(vehicle_type_id)):
        return None
    try:
        value = int(vehicle_type_id)
    except (TypeError, ValueError):
        return None
    template_key = CLASS_TO_TEMPLATE.get(value)
    return template_key or None


def _sanitize_xml_id(value: str) -> str:
    """
    Genera un id seguro para XML a partir de patente/placa.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    cleaned = cleaned.strip("_") or "veh"
    if cleaned[0].isdigit():
        cleaned = f"veh_{cleaned}"
    if len(cleaned) > 60:
        cleaned = cleaned[:60]
    return cleaned


def _format_num(value: Optional[float], decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{decimals}f}"


def _base_template_for_class(vehicle_type_id: Optional[float]) -> Dict[str, str]:
    try:
        value = int(vehicle_type_id) if vehicle_type_id is not None else None
    except (TypeError, ValueError):
        value = None
    template_key = CLASS_TO_TEMPLATE.get(value, "car")
    return DEFAULT_VTYPE_TEMPLATES.get(template_key, DEFAULT_VTYPE_TEMPLATES["car"])


def _pick_quantile(series: pd.Series, q: float, default: float) -> float:
    if series is None or series.empty:
        return default
    try:
        quant = float(series.quantile(q))
        if math.isnan(quant):
            return default
        return quant
    except Exception:
        return default


def build_vehicle_type_profiles(
    events_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    flow_cols: FlowColumns,
) -> Dict[str, Dict[str, str]]:
    """
    Construye un vType por patente usando métricas observadas (accel/decel/maxSpeed).
    """
    if events_df is None or events_df.empty:
        return {}

    profiles: Dict[str, Dict[str, str]] = {}
    segments_grouped = (
        segments_df.groupby("plate", sort=False) if segments_df is not None and not segments_df.empty else None
    )

    for plate, group in events_df.groupby("__plate", sort=False):
        speeds = pd.to_numeric(group[flow_cols.speed_kmh], errors="coerce")
        max_speed_kmh = _pick_quantile(speeds, 0.95, float(DEFAULT_VTYPE_TEMPLATES["car"]["maxSpeed"]))

        vehicle_type_id = None
        if "vehicle_type_id" in group.columns:
            mode_val = group["vehicle_type_id"].mode(dropna=True)
            if not mode_val.empty:
                vehicle_type_id = mode_val.iloc[0]

        template = _base_template_for_class(vehicle_type_id)

        accel_val = float(template["accel"])
        decel_val = float(template["decel"])
        sigma_val = float(template["sigma"])
        lane_change_rate = 0.0

        seg_group = None
        if segments_grouped is not None:
            try:
                seg_group = segments_grouped.get_group(plate)
            except KeyError:
                seg_group = None
        if seg_group is not None and not seg_group.empty:
            pos_accel = seg_group.loc[seg_group["accel_m_s2"] > 0, "accel_m_s2"].dropna()
            neg_accel = seg_group.loc[seg_group["accel_m_s2"] < 0, "accel_m_s2"].dropna().abs()
            accel_val = _pick_quantile(pos_accel, 0.9, accel_val)
            decel_val = _pick_quantile(neg_accel, 0.9, decel_val)
            lane_change_rate = float(seg_group["lane_change"].mean())
            sigma_val = min(1.2, sigma_val + lane_change_rate * 0.4)

        vtype_id = f"vt_{_sanitize_xml_id(plate)}"
        profiles[plate] = {
            "id": vtype_id,
            "accel": _format_num(accel_val, 3),
            "decel": _format_num(decel_val, 3),
            "sigma": _format_num(sigma_val, 2),
            "length": template["length"],
            "minGap": template["minGap"],
            "maxSpeed": _format_num(max_speed_kmh, 2),
            "color": template["color"],
            "vClass": template["vClass"],
            "guiShape": template["guiShape"],
        }

    return profiles


def build_sumo_trips(
    traj_df: pd.DataFrame,
    flow_cols: FlowColumns,
    portico_sumo_df: pd.DataFrame,
) -> List[SumoTrip]:
    if traj_df.empty or portico_sumo_df.empty:
        return []

    traj_df = traj_df.copy()
    traj_df[flow_cols.timestamp] = pd.to_datetime(
        traj_df[flow_cols.timestamp], errors="coerce"
    )
    traj_df = traj_df[traj_df[flow_cols.timestamp].notna()]
    if traj_df.empty:
        return []

    base_time = traj_df[flow_cols.timestamp].min()
    mapping = (
        portico_sumo_df.drop_duplicates("portico_id")
        .set_index("portico_id")
        .to_dict(orient="index")
    )

    sumo_trips: List[SumoTrip] = []
    for trip_id, group in traj_df.groupby("trip_id", sort=False):
        ordered = group.sort_values(flow_cols.timestamp)
        if len(ordered) < 2:
            continue
        start_portico = str(ordered.iloc[0][flow_cols.portico]).strip()
        end_portico = str(ordered.iloc[-1][flow_cols.portico]).strip()
        start_row = mapping.get(start_portico)
        end_row = mapping.get(end_portico)
        if start_row is None or end_row is None:
            continue

        via_edges: List[str] = []
        skip_trip = False
        for portico in ordered[flow_cols.portico].iloc[1:-1]:
            portico = str(portico).strip()
            row = mapping.get(portico)
            if row is None:
                skip_trip = True
                break
            via_edges.append(row["edge_id_sumo"])
        if skip_trip:
            continue

        start_time = ordered.iloc[0][flow_cols.timestamp]
        depart_s = (start_time - base_time).total_seconds()
        if depart_s < 0:
            depart_s = 0.0

        start_lane = ordered.iloc[0].get("lane_numeric")
        depart_lane = None
        if not pd.isna(start_lane):
            lane_idx = int(start_lane) - 1
            if lane_idx < 0:
                lane_idx = 0
            depart_lane = str(lane_idx)

        try:
            depart_pos = float(start_row["pos_m"])
        except (TypeError, ValueError, KeyError):
            depart_pos = 0.0
        depart_speed = ordered.iloc[0].get(flow_cols.speed_kmh)
        if pd.notna(depart_speed):
            depart_speed = float(depart_speed) * (1000 / 3600)
        else:
            depart_speed = None

        sumo_trips.append(
            SumoTrip(
                trip_id=str(trip_id),
                plate=str(ordered.iloc[0]["__plate"]),
                vehicle_type_id=ordered.iloc[0].get("vehicle_type_id"),
                depart_s=float(depart_s),
                from_edge=str(start_row["edge_id_sumo"]),
                to_edge=str(end_row["edge_id_sumo"]),
                via_edges=via_edges,
                depart_lane=depart_lane,
                depart_pos=depart_pos,
                depart_speed=depart_speed,
            )
        )
    return sumo_trips


def export_sumo_trips_xml(
    trips: List[SumoTrip],
    path: Path,
    vehicle_profiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    root = ET.Element("routes")
    # Definir tipos base
    base_vtypes = [
        {"id": key, **attrs} for key, attrs in DEFAULT_VTYPE_TEMPLATES.items()
    ]
    for vt in base_vtypes:
        ET.SubElement(root, "vType", attrib=vt)

    profiles = vehicle_profiles or {}
    for profile in profiles.values():
        ET.SubElement(root, "vType", attrib=profile)

    for trip in trips:
        attrib = {
            "id": trip.trip_id,
            "from": trip.from_edge,
            "to": trip.to_edge,
            "depart": f"{trip.depart_s:.2f}",
        }
        profile = profiles.get(trip.plate)
        vtype = profile["id"] if profile else _vehicle_type_sumo_id(trip.vehicle_type_id)
        if vtype:
            attrib["type"] = vtype
        if trip.via_edges:
            attrib["via"] = " ".join(trip.via_edges)
        if trip.depart_lane is not None:
            attrib["departLane"] = trip.depart_lane
        if trip.depart_speed is not None:
            # Usar el límite de velocidad del carril para evitar errores de inserción
            # cuando la velocidad observada supera el máximo permitido en la red.
            attrib["departSpeed"] = "speedLimit"
        ET.SubElement(root, "trip", attrib=attrib)

    rough_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(rough_xml).toprettyxml(indent="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def export_sumo_depart_summary(trips: List[SumoTrip], path: Path) -> None:
    if not trips:
        return
    depart_values = [trip.depart_s for trip in trips]
    depart_min = min(depart_values)
    depart_max = max(depart_values)
    root = ET.Element("departSummary", attrib={
        "depart_min": f"{depart_min:.2f}",
        "depart_max": f"{depart_max:.2f}",
    })
    rough_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(rough_xml).toprettyxml(indent="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


def prepare_raw_events(
    flujos_df: pd.DataFrame,
    porticos_df: pd.DataFrame,
    flow_cols: FlowColumns,
    speed_min: int = DEFAULT_SPEED_MIN,
    speed_max: int = DEFAULT_SPEED_MAX,
    segment_filter: Optional[SegmentFilter] = None,
) -> pd.DataFrame:
    """
    Limpia y normaliza las detecciones crudas provenientes de los CSV de flujos.
    """
    if flujos_df is None or flujos_df.empty:
        raise ValueError("No hay datos de flujos para procesar.")

    catalog = _prepare_portico_catalog(porticos_df)
    df = flujos_df.copy()
    df[flow_cols.timestamp] = pd.to_datetime(df[flow_cols.timestamp], errors="coerce").dt.tz_localize(None)
    df[flow_cols.speed_kmh] = pd.to_numeric(df[flow_cols.speed_kmh], errors="coerce")
    df["__plate"] = _normalize_plate(df[flow_cols.plate_id])

    df = df[
        df[flow_cols.timestamp].notna()
        & df["__plate"].notna()
        & df[flow_cols.speed_kmh].between(speed_min, speed_max)
    ].copy()

    df[flow_cols.portico] = df[flow_cols.portico].astype(str).str.strip()
    df = df.merge(
        catalog,
        left_on=flow_cols.portico,
        right_on="portico_id",
        how="inner",
        suffixes=("", "_meta"),
    )

    df = df.dropna(subset=["km"])
    df = df.drop_duplicates(
        subset=["__plate", flow_cols.portico, flow_cols.timestamp],
        keep="first",
    )
    df["lane_numeric"] = pd.to_numeric(df[flow_cols.lane], errors="coerce")
    df["vehicle_type_id"] = pd.to_numeric(df[flow_cols.class_id], errors="coerce")
    if segment_filter is not None:
        df = df[df[flow_cols.portico].isin(segment_filter.portico_ids)]
        if df.empty:
            raise ValueError("El tramo seleccionado no contiene detecciones en el archivo de flujos.")
    ordered_df = df.sort_values(["__plate", flow_cols.timestamp]).reset_index(drop=True)
    return ordered_df


def build_discrete_trajectories(
    events_df: pd.DataFrame,
    flow_cols: FlowColumns,
) -> pd.DataFrame:
    """
    Calcula diferencias temporales y espaciales, y asigna IDs de viaje (trip_id)
    por placa según los umbrales definidos.
    """
    if events_df.empty:
        return events_df.copy()

    df = events_df.copy()
    group_key = "__plate"
    grouped = df.groupby(group_key, sort=False)

    df["delta_t_s"] = grouped[flow_cols.timestamp].diff().dt.total_seconds()
    df["delta_km"] = grouped["km"].diff()
    df["delta_portico"] = grouped[flow_cols.portico].shift()
    df["implicit_speed_kmh"] = (
        (df["delta_km"].abs() / df["delta_t_s"])
        * 3600
    )

    df["delta_km_sign"] = np.sign(df["delta_km"].fillna(0))
    prev_sign = grouped["delta_km_sign"].shift()
    sign_change = (
        (df["delta_km_sign"] != prev_sign)
        & prev_sign.notna()
        & (df["delta_km_sign"] != 0)
        & (prev_sign != 0)
    )

    break_mask = (
        df["delta_t_s"].isna()
        | (df["delta_t_s"] > MAX_TIME_GAP_S)
        | (df["delta_km"].abs() > MAX_DISTANCE_JUMP_KM)
        | (df["implicit_speed_kmh"] > MAX_IMPL_SPEED)
        | ((df["implicit_speed_kmh"] < MIN_IMPL_SPEED) & df["delta_t_s"].notna())
        | sign_change
    )

    first_row_mask = grouped.cumcount() == 0
    df["trip_break"] = break_mask | first_row_mask
    df["trip_local_id"] = df.groupby(group_key)["trip_break"].cumsum()
    df["trip_id"] = (
        df[group_key] + "_" + df["trip_local_id"].astype(int).astype(str)
    )
    return df


def build_segments_from_trajectories(
    traj_df: pd.DataFrame,
    flow_cols: FlowColumns,
) -> pd.DataFrame:
    """
    Reconstruye segmentos entre detecciones consecutivas dentro de cada viaje.
    """
    if traj_df.empty:
        return pd.DataFrame()

    df = traj_df.copy()
    group_cols = ["__plate", "trip_local_id"]
    grouped = df.groupby(group_cols, sort=False)

    df["prev_time"] = grouped[flow_cols.timestamp].shift()
    df["prev_speed"] = grouped[flow_cols.speed_kmh].shift()
    df["prev_km"] = grouped["km"].shift()
    df["prev_portico"] = grouped[flow_cols.portico].shift()
    df["prev_lane"] = grouped["lane_numeric"].shift()

    segments = df[df["prev_time"].notna()].copy()
    if segments.empty:
        return segments

    segments["delta_t_s"] = (segments[flow_cols.timestamp] - segments["prev_time"]).dt.total_seconds()
    segments = segments[segments["delta_t_s"] > 0]
    if segments.empty:
        return segments

    segments["delta_km"] = segments["km"] - segments["prev_km"]
    segments["distance_km_abs"] = segments["delta_km"].abs()
    segments["avg_speed_kmh"] = (
        segments["distance_km_abs"] / segments["delta_t_s"] * 3600
    )
    segments["travel_time_min"] = segments["delta_t_s"] / 60
    segments["speed_change_kmh"] = segments[flow_cols.speed_kmh] - segments["prev_speed"]

    start_speed = segments["prev_speed"]
    end_speed = segments[flow_cols.speed_kmh]
    distance = segments["distance_km_abs"]
    valid_mask = (
        start_speed.notna()
        & end_speed.notna()
        & distance.gt(0)
    )
    segments["accel_m_s2"] = np.nan
    if valid_mask.any():
        start_ms = start_speed[valid_mask] * (1000 / 3600)
        end_ms = end_speed[valid_mask] * (1000 / 3600)
        distance_m = distance[valid_mask] * 1000
        segments.loc[valid_mask, "accel_m_s2"] = (
            (end_ms**2 - start_ms**2) / (2 * distance_m)
        )
    segments["lane_change"] = (
        segments["prev_lane"].notna()
        & segments["lane_numeric"].notna()
        & (segments["prev_lane"] != segments["lane_numeric"])
    ).astype(int)
    segments["segment_id"] = (
        segments["prev_portico"].astype(str)
        + "→"
        + segments[flow_cols.portico].astype(str)
    )
    return segments[
        [
            "__plate",
            "trip_id",
            "segment_id",
            "prev_portico",
            flow_cols.portico,
            "prev_time",
            flow_cols.timestamp,
            "delta_t_s",
            "travel_time_min",
            "delta_km",
            "distance_km_abs",
            "avg_speed_kmh",
            "speed_change_kmh",
            "accel_m_s2",
            "prev_lane",
            "lane_numeric",
            "lane_change",
            "km",
            "prev_km",
            "sentido",
            "vehicle_type_id",
        ]
    ].rename(
        columns={
            "__plate": "plate",
            "prev_portico": "start_portico",
            flow_cols.portico: "end_portico",
            "prev_time": "start_time",
            flow_cols.timestamp: "end_time",
            "prev_lane": "start_lane",
            "lane_numeric": "end_lane",
            "km": "end_km",
            "prev_km": "start_km",
        }
    )


def compute_headways(events_df: pd.DataFrame, flow_cols: FlowColumns) -> pd.DataFrame:
    """
    Calcula headways temporales por pórtico y carril.
    """
    if events_df.empty:
        return pd.DataFrame()
    df = events_df.copy()
    df = df.sort_values([flow_cols.portico, flow_cols.lane, flow_cols.timestamp])
    grouped = df.groupby([flow_cols.portico, flow_cols.lane])
    df["headway_s"] = grouped[flow_cols.timestamp].diff().dt.total_seconds()
    df = df[df["headway_s"].notna()].copy()
    df["headway_m"] = (df["headway_s"] * df[flow_cols.speed_kmh]) / 3.6
    return df[
        [
            "__plate",
            flow_cols.portico,
            flow_cols.lane,
            flow_cols.timestamp,
            "headway_s",
            "headway_m",
            flow_cols.speed_kmh,
        ]
    ].rename(columns={"__plate": "plate"})


def _quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(series.quantile(q))


def _safe_density(flow_veh_per_window: int, avg_speed: float, window_minutes: int) -> float:
    if avg_speed is None or math.isnan(avg_speed) or avg_speed <= 0:
        return float("nan")
    flow_per_hour = flow_veh_per_window * (60 / window_minutes)
    return flow_per_hour / avg_speed


def aggregate_macro_metrics(
    segments_df: pd.DataFrame,
    window_minutes: int = 15,
) -> pd.DataFrame:
    if segments_df.empty:
        return pd.DataFrame()

    df = segments_df.copy()
    df["window_start"] = df["start_time"].dt.floor(f"{window_minutes}min")
    grouped = df.groupby(["start_portico", "end_portico", "window_start"])

    agg = grouped.agg(
        vehicle_count=("plate", "count"),
        mean_speed_kmh=("avg_speed_kmh", "mean"),
        std_speed_kmh=("avg_speed_kmh", "std"),
        p05_speed_kmh=("avg_speed_kmh", lambda s: _quantile(s, 0.05)),
        p50_speed_kmh=("avg_speed_kmh", "median"),
        p95_speed_kmh=("avg_speed_kmh", lambda s: _quantile(s, 0.95)),
        mean_travel_time_min=("travel_time_min", "mean"),
        p95_travel_time_min=("travel_time_min", lambda s: _quantile(s, 0.95)),
    ).reset_index()

    agg["density_veh_per_km"] = agg.apply(
        lambda row: _safe_density(row["vehicle_count"], row["mean_speed_kmh"], window_minutes),
        axis=1,
    )
    return agg


def compute_acceleration(start_speed_kmh: float, end_speed_kmh: float, distance_km: float) -> Optional[float]:
    """
    Reutiliza la fórmula de aceleración constante usada en main.py.
    """
    if any(pd.isna(x) for x in (start_speed_kmh, end_speed_kmh)) or distance_km <= 0:
        return None
    start_ms = start_speed_kmh * (1000 / 3600)
    end_ms = end_speed_kmh * (1000 / 3600)
    distance_m = distance_km * 1000
    if distance_m == 0:
        return None
    return (end_ms ** 2 - start_ms ** 2) / (2 * distance_m)


def run_sumo_pipeline(
    flujos_df: pd.DataFrame,
    porticos_df: pd.DataFrame,
    flow_cols: Optional[FlowColumns] = None,
    output_dir: Optional[Path] = None,
    segment_filter: Optional[SegmentFilter] = None,
) -> SUMOResult:
    """
    Ejecuta los pasos 1-5 descritos en la solicitud. Retorna los DataFrames
    intermedios y finales para ser usados en la calibración de SUMO.
    """
    flow_cols = flow_cols or FlowColumns()
    clean = prepare_raw_events(
        flujos_df,
        porticos_df,
        flow_cols,
        segment_filter=segment_filter,
    )
    trajectories = build_discrete_trajectories(clean, flow_cols)
    segments = build_segments_from_trajectories(trajectories, flow_cols)
    headways = compute_headways(clean, flow_cols)
    macro_metrics = aggregate_macro_metrics(segments)

    sumo_trips: Optional[List[SumoTrip]] = None
    sumo_trips_path: Optional[Path] = None
    sumo_warning: Optional[str] = None
    depart_summary_path = None
    vehicle_profiles: Dict[str, Dict[str, str]] = {}
    try:
        portico_sumo_table = build_portico_sumo_table(porticos_df)
        sumo_trips = build_sumo_trips(trajectories, flow_cols, portico_sumo_table)
        vehicle_profiles = build_vehicle_type_profiles(clean, segments, flow_cols)
        if sumo_trips and output_dir is not None:
            sumo_trips_path = output_dir / "sumo_trips.rou.xml"
            export_sumo_trips_xml(
                sumo_trips,
                sumo_trips_path,
                vehicle_profiles=vehicle_profiles,
            )
            depart_summary_path = output_dir / "sumo_depart_summary.xml"
            export_sumo_depart_summary(sumo_trips, depart_summary_path)
    except ValueError as exc:
        sumo_warning = str(exc)

    result = SUMOResult(
        clean_events=clean,
        trajectories=trajectories,
        segments=segments,
        macro_metrics=macro_metrics,
        headways=headways,
        segment_filter=segment_filter,
        sumo_trips=sumo_trips,
        sumo_trips_path=sumo_trips_path,
        depart_summary_path=depart_summary_path,
        sumo_warning=sumo_warning,
        vehicle_profiles=vehicle_profiles or None,
    )

 #   if output_dir is not None:
 #       output_dir.mkdir(parents=True, exist_ok=True)
 #       clean.to_csv(output_dir / "somu_clean_events.csv", index=False)
 #       trajectories.to_csv(output_dir / "somu_trajectories.csv", index=False)
 #       segments.to_csv(output_dir / "somu_segments.csv", index=False)
 #       macro_metrics.to_csv(output_dir / "somu_macro_metrics.csv", index=False)
 #       headways.to_csv(output_dir / "somu_headways.csv", index=False)

    return result
