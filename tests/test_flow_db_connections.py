from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import FLOW_TABLE_NAME, _connect_duckdb, import_flujos_to_duckdb  # noqa: E402


def test_import_flujos_with_existing_read_intent_connection(tmp_path):
    pytest.importorskip("duckdb")
    db_path = tmp_path / "flows.duckdb"
    csv_path = tmp_path / "flows.csv"
    csv_path.write_text(
        (
            "FECHA,VELOCIDAD,CATEGORIA,MATRICULA,PORTICO,CARRIL\n"
            "2024-01-01T00:00:00Z,80,1,ABC123,P1,1\n"
        ),
        encoding="utf-8",
    )

    setup_conn = _connect_duckdb(read_only=False, db_path=db_path)
    try:
        setup_conn.execute("CREATE TABLE IF NOT EXISTS bootstrap(seed INTEGER)")
    finally:
        setup_conn.close()

    read_conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        inserted = import_flujos_to_duckdb(csv_path=str(csv_path), db_path=db_path)
    finally:
        read_conn.close()

    assert inserted == 1

    verify_conn = _connect_duckdb(read_only=False, db_path=db_path)
    try:
        count = verify_conn.execute(
            f"SELECT COUNT(*) FROM {FLOW_TABLE_NAME}"
        ).fetchone()[0]
    finally:
        verify_conn.close()

    assert count == 1
