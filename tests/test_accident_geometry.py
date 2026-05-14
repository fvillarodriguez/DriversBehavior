"""
Tests for accident-side geometric features added by process_accidentes_df:
- km_post, km_cerc: km of the upstream/downstream pórticos picked by
  find_candidate_porticos.
- dist_to_post_km, dist_to_cerc_km: |accident_km - portico_km| within the
  same (eje, calzada) — find_candidate_porticos filters by eje/calzada
  before searching, so km values are always on the same scale.
- pos_relativa: dist_to_post / (dist_to_post + dist_to_cerc) ∈ [0, 1].
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import process_accidentes_df


def _porticos_fixture() -> pd.DataFrame:
    """Dos ejes con km reiniciado, en forma canónica (lowercase)."""
    return pd.DataFrame(
        [
            # Ruta 5 Norte Poniente
            {"portico": "N11", "km": 5.0,  "orden": 1, "eje": "RUTA 5 NORTE", "calzada": "Poniente"},
            {"portico": "N12", "km": 10.0, "orden": 2, "eje": "RUTA 5 NORTE", "calzada": "Poniente"},
            {"portico": "N13", "km": 15.0, "orden": 3, "eje": "RUTA 5 NORTE", "calzada": "Poniente"},
            # Ruta 5 Sur Poniente (km empieza desde 0 otra vez)
            {"portico": "S14", "km": 0.0,  "orden": 1, "eje": "RUTA 5 SUR",   "calzada": "Poniente"},
            {"portico": "S13", "km": 5.0,  "orden": 2, "eje": "RUTA 5 SUR",   "calzada": "Poniente"},
            {"portico": "S12", "km": 12.0, "orden": 3, "eje": "RUTA 5 SUR",   "calzada": "Poniente"},
        ]
    )


def _build_accident_row(*, eje: str, calzada: str, km: float) -> dict:
    """Construye una fila mínima del CSV de eventos esperado por process_accidentes_df."""
    base = {
        "Tipo": "Accidente",
        "Via": "expresa",
        "Eje": eje,
        "Calzada": calzada,
        "SubTipo": "Choque",
        "Fechas Inicio": "01-01-2026",
        "Hora Inicio": "08:00:00",
        "Fecha Fin": "01-01-2026",
        "Hora Fin": "08:30:00",
        "Km.": km,
    }
    # severity classifier columns (all zero → severidad=0)
    for c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        base[c] = 0
    return base


def _accidents_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _build_accident_row(eje="RUTA 5 NORTE", calzada="Poniente", km=6.0),   # entre N11(5) y N12(10) → 1 km de N11
            _build_accident_row(eje="RUTA 5 NORTE", calzada="Poniente", km=13.0),  # entre N12(10) y N13(15) → 3 km de N12
            _build_accident_row(eje="RUTA 5 SUR",   calzada="Poniente", km=2.0),   # entre S14(0) y S13(5) → 2 km de S14
        ]
    )


def _processed() -> pd.DataFrame:
    return process_accidentes_df(_accidents_fixture(), _porticos_fixture())


def test_km_post_and_km_cerc_are_persisted():
    df = _processed()
    assert {"km_post", "km_cerc"}.issubset(df.columns)
    # Primer accidente: km=6, post=N11(km=5), cerc=N12(km=10).
    first = df.iloc[0]
    assert first["km_post"] == pytest.approx(5.0)
    assert first["km_cerc"] == pytest.approx(10.0)


def test_dist_to_post_and_cerc_are_within_same_eje():
    df = _processed()
    # km=6 entre 5 y 10 → 1 km y 4 km.
    a0 = df.iloc[0]
    assert a0["dist_to_post_km"] == pytest.approx(1.0)
    assert a0["dist_to_cerc_km"] == pytest.approx(4.0)
    # km=13 entre 10 y 15 → 3 km y 2 km.
    a1 = df.iloc[1]
    assert a1["dist_to_post_km"] == pytest.approx(3.0)
    assert a1["dist_to_cerc_km"] == pytest.approx(2.0)
    # km=2 en Sur entre 0 y 5 → 2 km y 3 km (NO 7 km vs Norte).
    a2 = df.iloc[2]
    assert a2["dist_to_post_km"] == pytest.approx(2.0)
    assert a2["dist_to_cerc_km"] == pytest.approx(3.0)


def test_pos_relativa_is_in_unit_interval_and_consistent():
    df = _processed()
    assert (df["pos_relativa"] >= 0).all()
    assert (df["pos_relativa"] <= 1).all()
    # km=6 → 1/(1+4)=0.2; km=13 → 3/(3+2)=0.6; km=2 (sur) → 2/(2+3)=0.4
    assert df.iloc[0]["pos_relativa"] == pytest.approx(0.2)
    assert df.iloc[1]["pos_relativa"] == pytest.approx(0.6)
    assert df.iloc[2]["pos_relativa"] == pytest.approx(0.4)


def test_eje_isolation_no_cross_axis_leakage():
    """El accidente en Ruta 5 Sur km=2 no se asigna a un pórtico de Ruta 5 Norte
    aunque haya pórticos del Norte numéricamente más cercanos."""
    df = _processed()
    sur = df.iloc[2]
    assert sur["ultimo_portico"] == "S14"
    assert sur["proximo_portico"] == "S13"


def test_pos_relativa_nan_when_no_downstream():
    """Accidente al final del eje (sin pórtico aguas abajo): pos_relativa = NaN."""
    porticos = _porticos_fixture()
    # Accidente después del último pórtico Norte (km=20, último N13 está en km=15).
    df_acc = pd.DataFrame(
        [_build_accident_row(eje="RUTA 5 NORTE", calzada="Poniente", km=20.0)]
    )
    out = process_accidentes_df(df_acc, porticos)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["ultimo_portico"] == "N13"
    assert pd.isna(row["proximo_portico"])
    assert pd.isna(row["km_cerc"])
    # dist_to_post bien definido; pos_relativa NaN porque falta cerc.
    assert row["dist_to_post_km"] == pytest.approx(5.0)
    assert pd.isna(row["pos_relativa"])
