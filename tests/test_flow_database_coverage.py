from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import flow_database_app as app  # noqa: E402


def test_build_flow_coverage_views():
    counts_year = pd.DataFrame({"anio": [2024, 2025], "total": [100, 80]})
    counts_month = pd.DataFrame(
        {
            "anio": [2024, 2024, 2025],
            "mes": [1, 3, 2],
            "total": [50, 50, 80],
        }
    )

    year_df = app._build_flow_coverage_year_table(
        counts_year,
        counts_month,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2025-03-31"),
    )
    matrix_df = app._build_flow_coverage_month_matrix(
        counts_month,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2025-03-31"),
    )

    assert year_df.to_dict(orient="records") == [
        {
            "Anio": 2024,
            "Detecciones": 100,
            "Meses activos": 2,
            "Primer mes con datos": "Ene",
            "Ultimo mes con datos": "Mar",
            "Meses esperados": 12,
            "Cobertura %": 16.7,
            "Meses faltantes": "Feb, Abr, May, Jun, Jul, Ago, Sep, Oct, Nov, Dic",
        },
        {
            "Anio": 2025,
            "Detecciones": 80,
            "Meses activos": 1,
            "Primer mes con datos": "Feb",
            "Ultimo mes con datos": "Feb",
            "Meses esperados": 3,
            "Cobertura %": 33.3,
            "Meses faltantes": "Ene, Mar",
        },
    ]

    assert matrix_df.to_dict(orient="records") == [
        {
            "Anio": "2024",
            "Ene": "50",
            "Feb": "0",
            "Mar": "50",
            "Abr": "0",
            "May": "0",
            "Jun": "0",
            "Jul": "0",
            "Ago": "0",
            "Sep": "0",
            "Oct": "0",
            "Nov": "0",
            "Dic": "0",
        },
        {
            "Anio": "2025",
            "Ene": "0",
            "Feb": "80",
            "Mar": "0",
            "Abr": "",
            "May": "",
            "Jun": "",
            "Jul": "",
            "Ago": "",
            "Sep": "",
            "Oct": "",
            "Nov": "",
            "Dic": "",
        },
    ]
