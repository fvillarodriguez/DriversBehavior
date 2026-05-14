import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.portico_geometry import (
    attach_portico_geometry,
    attach_portico_geometry_polars,
    compute_portico_geometry,
)


def _two_eje_fixture() -> pd.DataFrame:
    """
    Caso realista: dos ejes en la misma calzada cuyo km se reinicia.
    Ej. pórticos 11-15 = RUTA 5 NORTE (km creciente),
        pórticos 14-11 = RUTA 5 SUR (km decreciente desde cero al reiniciarse).
    """
    return pd.DataFrame(
        [
            # Ruta 5 Norte
            {"portico": "N11", "km": 5.0,  "orden": 1, "eje": "R5N", "calzada": "P"},
            {"portico": "N12", "km": 10.0, "orden": 2, "eje": "R5N", "calzada": "P"},
            {"portico": "N13", "km": 15.0, "orden": 3, "eje": "R5N", "calzada": "P"},
            {"portico": "N14", "km": 20.0, "orden": 4, "eje": "R5N", "calzada": "P"},
            {"portico": "N15", "km": 25.0, "orden": 5, "eje": "R5N", "calzada": "P"},
            # Ruta 5 Sur (km se reinicia, orden empieza de nuevo)
            {"portico": "S14", "km": 0.0,  "orden": 1, "eje": "R5S", "calzada": "P"},
            {"portico": "S13", "km": 5.0,  "orden": 2, "eje": "R5S", "calzada": "P"},
            {"portico": "S12", "km": 12.0, "orden": 3, "eje": "R5S", "calzada": "P"},
            {"portico": "S11", "km": 30.0, "orden": 4, "eje": "R5S", "calzada": "P"},
        ]
    )


def test_no_cross_eje_neighbors():
    df = _two_eje_fixture()
    geom = compute_portico_geometry(df).set_index("portico")

    # Frontera N15 -> S14: el último de Norte y el primero de Sur deben
    # quedar marcados como extremos, y sus distancias rellenadas con sentinel.
    assert int(geom.loc["N15", "is_eje_last"]) == 1
    assert geom.loc["N15", "dist_to_downstream_km"] == 0.0
    assert int(geom.loc["S14", "is_eje_first"]) == 1
    assert geom.loc["S14", "dist_to_upstream_km"] == 0.0

    # El primer pórtico de Norte y el último de Sur también son extremos.
    assert int(geom.loc["N11", "is_eje_first"]) == 1
    assert int(geom.loc["S11", "is_eje_last"]) == 1


def test_distance_within_eje_is_correct():
    df = _two_eje_fixture()
    geom = compute_portico_geometry(df).set_index("portico")

    # Distancias internas en Ruta 5 Norte (5 km uniforme).
    for portico in ["N12", "N13", "N14"]:
        assert geom.loc[portico, "dist_to_upstream_km"] == pytest.approx(5.0)
        assert geom.loc[portico, "dist_to_downstream_km"] == pytest.approx(5.0)

    # Distancias variables en Ruta 5 Sur: 0,5,12,30 -> 5, 7, 18.
    assert geom.loc["S13", "dist_to_upstream_km"] == pytest.approx(5.0)
    assert geom.loc["S13", "dist_to_downstream_km"] == pytest.approx(7.0)
    assert geom.loc["S12", "dist_to_upstream_km"] == pytest.approx(7.0)
    assert geom.loc["S12", "dist_to_downstream_km"] == pytest.approx(18.0)


def test_km_norm_resets_per_eje():
    df = _two_eje_fixture()
    geom = compute_portico_geometry(df).set_index("portico")

    # km_norm_eje empieza en 0 y termina en 1 dentro de cada eje, independiente
    # del valor absoluto de km.
    assert geom.loc["N11", "km_norm_eje"] == pytest.approx(0.0)
    assert geom.loc["N15", "km_norm_eje"] == pytest.approx(1.0)
    assert geom.loc["S14", "km_norm_eje"] == pytest.approx(0.0)
    assert geom.loc["S11", "km_norm_eje"] == pytest.approx(1.0)


def test_orden_norm_resets_per_eje():
    df = _two_eje_fixture()
    geom = compute_portico_geometry(df).set_index("portico")
    assert geom.loc["N15", "orden_norm_eje"] == pytest.approx(1.0)
    assert geom.loc["S11", "orden_norm_eje"] == pytest.approx(1.0)


def test_one_hot_columns_present():
    df = _two_eje_fixture()
    geom = compute_portico_geometry(df)
    assert "eje_R5N" in geom.columns
    assert "eje_R5S" in geom.columns
    assert "calzada_P" in geom.columns
    # Suma por fila == 1 para cada categórico (un solo eje y una sola calzada).
    eje_cols = [c for c in geom.columns if c.startswith("eje_")]
    calzada_cols = [c for c in geom.columns if c.startswith("calzada_")]
    assert (geom[eje_cols].sum(axis=1) == 1).all()
    assert (geom[calzada_cols].sum(axis=1) == 1).all()


def test_stable_categories_keep_unused_columns():
    df = _two_eje_fixture()
    geom = compute_portico_geometry(
        df,
        eje_categories=["R5N", "R5S", "R68"],  # R68 no aparece en datos
        calzada_categories=["P", "O"],
    )
    assert "eje_R68" in geom.columns
    assert "calzada_O" in geom.columns
    # Columnas no presentes en datos -> todos ceros.
    assert (geom["eje_R68"] == 0).all()
    assert (geom["calzada_O"] == 0).all()


def test_single_portico_eje():
    df = pd.DataFrame(
        [
            {"portico": "X1", "km": 7.5, "orden": 1, "eje": "SOLO", "calzada": "P"},
        ]
    )
    geom = compute_portico_geometry(df).set_index("portico")
    # Sin span ni vecinos: rellenos en cero, ambos extremos.
    assert geom.loc["X1", "km_norm_eje"] == 0.0
    assert geom.loc["X1", "orden_norm_eje"] == 1.0
    assert int(geom.loc["X1", "is_eje_first"]) == 1
    assert int(geom.loc["X1", "is_eje_last"]) == 1
    assert geom.loc["X1", "dist_to_upstream_km"] == 0.0
    assert geom.loc["X1", "dist_to_downstream_km"] == 0.0


def test_calzada_split_within_same_eje():
    """Mismo eje, calzadas distintas, deben tratarse como grupos independientes."""
    df = pd.DataFrame(
        [
            {"portico": "P1", "km": 1.0, "orden": 1, "eje": "R5N", "calzada": "P"},
            {"portico": "P2", "km": 2.0, "orden": 2, "eje": "R5N", "calzada": "P"},
            {"portico": "O1", "km": 1.0, "orden": 1, "eje": "R5N", "calzada": "O"},
            {"portico": "O2", "km": 2.0, "orden": 2, "eje": "R5N", "calzada": "O"},
        ]
    )
    geom = compute_portico_geometry(df).set_index("portico")
    # P1 y O1 son ambos primeros de su grupo, no vecinos entre sí.
    assert int(geom.loc["P1", "is_eje_first"]) == 1
    assert int(geom.loc["O1", "is_eje_first"]) == 1
    assert geom.loc["P1", "dist_to_upstream_km"] == 0.0
    assert geom.loc["O1", "dist_to_upstream_km"] == 0.0


def test_empty_input_returns_empty_frame():
    geom = compute_portico_geometry(pd.DataFrame())
    assert geom.empty
    assert "portico" in geom.columns


def test_missing_columns_raises():
    bad = pd.DataFrame([{"portico": "X", "km": 1.0}])
    with pytest.raises(ValueError):
        compute_portico_geometry(bad)


def _features_fixture() -> pd.DataFrame:
    """Pequeño df_pm sintético compatible con el fixture de pórticos."""
    return pd.DataFrame(
        [
            {"portico": "N11", "ts_min": 0, "speed_mean": 60.0},
            {"portico": "N15", "ts_min": 0, "speed_mean": 55.0},
            {"portico": "S14", "ts_min": 0, "speed_mean": 70.0},
            {"portico": "S11", "ts_min": 0, "speed_mean": 50.0},
        ]
    )


def test_attach_geometry_adds_columns():
    porticos = _two_eje_fixture()
    features = _features_fixture()
    out = attach_portico_geometry(features, porticos)
    assert "km_norm_eje" in out.columns
    assert "is_eje_first" in out.columns
    assert "eje_R5N" in out.columns
    # El primer pórtico de Norte debe quedar marcado como inicio.
    row = out[out["portico"] == "N11"].iloc[0]
    assert int(row["is_eje_first"]) == 1
    assert row["km_norm_eje"] == pytest.approx(0.0)


def test_attach_geometry_is_idempotent():
    porticos = _two_eje_fixture()
    features = _features_fixture()
    once = attach_portico_geometry(features, porticos)
    twice = attach_portico_geometry(once, porticos)
    assert list(twice.columns) == list(once.columns)
    pd.testing.assert_frame_equal(once, twice)


def test_attach_geometry_handles_dtype_mismatch():
    """Pórtico como int en features, str en porticos: el join debe funcionar."""
    porticos = pd.DataFrame(
        [
            {"portico": "11", "km": 1.0, "orden": 1, "eje": "E", "calzada": "P"},
            {"portico": "12", "km": 2.0, "orden": 2, "eje": "E", "calzada": "P"},
        ]
    )
    features = pd.DataFrame(
        [
            {"portico": 11, "ts_min": 0, "speed_mean": 60.0},
            {"portico": 12, "ts_min": 0, "speed_mean": 55.0},
        ]
    )
    out = attach_portico_geometry(features, porticos)
    # No NaN en columnas de geometría: el join encontró ambos pórticos.
    assert out["km_norm_eje"].notna().all()
    assert int(out.loc[out["portico"] == 11, "is_eje_first"].iloc[0]) == 1


def test_attach_geometry_polars_adds_columns():
    porticos = _two_eje_fixture()
    features = pl.from_pandas(_features_fixture())
    out = attach_portico_geometry_polars(features, porticos)
    cols = out.columns
    assert "km_norm_eje" in cols
    assert "eje_R5N" in cols
    assert "eje_R5S" in cols


def test_attach_geometry_polars_is_idempotent():
    porticos = _two_eje_fixture()
    features = pl.from_pandas(_features_fixture())
    once = attach_portico_geometry_polars(features, porticos)
    twice = attach_portico_geometry_polars(once, porticos)
    assert once.columns == twice.columns
    assert once.shape == twice.shape


def test_attach_geometry_noop_on_non_canonical_porticos():
    """porticos_df sin schema canónico (ej. tests sintéticos) -> no-op."""
    features = _features_fixture()
    non_canonical = pd.DataFrame(
        {
            "PORTICO": ["N11", "N15"],
            "km_ruta": [10.0, 15.0],
            "latitud": [0.0, 0.1],
            "longitud": [0.0, 0.1],
        }
    )
    out = attach_portico_geometry(features, non_canonical)
    # df_pm queda intacto y no se levanta excepción.
    pd.testing.assert_frame_equal(features, out)


def test_attach_geometry_dedupes_repeated_portico_ids():
    """
    En datos reales, un mismo portico ID aparece en múltiples (eje, calzada)
    (ej. mismo número en calzada Norte y Sur). El attach debe colapsar a una
    única fila por ID para no inflar df_pm.
    """
    porticos = pd.DataFrame(
        [
            {"portico": "11", "km": 1.0, "orden": 1, "eje": "R5N", "calzada": "P"},
            {"portico": "11", "km": 5.0, "orden": 1, "eje": "R5S", "calzada": "O"},
            {"portico": "12", "km": 2.0, "orden": 2, "eje": "R5N", "calzada": "P"},
        ]
    )
    features = pd.DataFrame(
        [
            {"portico": "11", "ts_min": 0, "speed_mean": 60.0},
            {"portico": "11", "ts_min": 5, "speed_mean": 65.0},
            {"portico": "12", "ts_min": 0, "speed_mean": 55.0},
        ]
    )
    out = attach_portico_geometry(features, porticos)
    # No row explosion: misma cantidad de filas que features de entrada.
    assert len(out) == len(features)


def test_attach_geometry_polars_noop_on_non_canonical_porticos():
    features = pl.from_pandas(_features_fixture())
    non_canonical = pd.DataFrame(
        {
            "PORTICO": ["N11", "N15"],
            "km_ruta": [10.0, 15.0],
        }
    )
    out = attach_portico_geometry_polars(features, non_canonical)
    assert features.columns == out.columns
    assert features.shape == out.shape
