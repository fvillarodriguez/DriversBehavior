"""Pórtico geometry helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_portico_id(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def load_porticos_geometry(porticos_csv: Path) -> pd.DataFrame:
    """Load Porticos.csv with the conventions documented in README.

    The file uses ';' separator and contains at minimum cod_portico, Km,
    Calzada, Orden, Eje, plus optional lat/lon and SUMO mappings.
    """
    df = pd.read_csv(porticos_csv, sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "cod_portico" not in df.columns:
        raise ValueError("Porticos.csv must contain column 'cod_portico'.")
    df["cod_portico_norm"] = df["cod_portico"].map(normalize_portico_id)
    for col in ("Km", "lat", "lon", "pos_m"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
