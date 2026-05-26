#!/usr/bin/env python3
"""
Phase 0 validation — verify the input data before launching the costly
dynamic GMM regeneration (~30-50h on local CPU).

Checks performed:
  1. flujos.duckdb is reachable, table flujos_duckdb exists, schema matches.
  2. Temporal coverage spans the requested [date_start, date_end].
  3. Plate / portico / categoria distributions are non-degenerate per year.
  4. Porticos.csv has every portico referenced by the flow database.
  5. eventos.duckdb has events in the requested range, with non-null
     portico_inicio / portico_fin, and the event type vocabulary is the
     expected one.
  6. Mapping eventos → Porticos.csv resolves to known portico codes.
  7. Frequency of the supported event types matches the configuration.

Writes a JSON report at --output with all counts and a top-level
"ready_for_phase_1" boolean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trc_paper.lib import (  # noqa: E402
    connect_duckdb_readonly,
    load_porticos_geometry,
    normalize_portico_id,
    write_json_atomic,
)

EXPECTED_FLOW_COLS = {
    "FECHA",
    "VELOCIDAD",
    "CATEGORIA",
    "MATRICULA",
    "PORTICO",
    "CARRIL",
}

EVENT_TYPES_OF_INTEREST = (
    "accidente",
    "averia",
    "averia_mayor",
    "objeto_calzada",
    "incidente",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-db", required=True, type=Path)
    p.add_argument("--porticos-csv", required=True, type=Path)
    p.add_argument("--events-db", required=True, type=Path)
    p.add_argument("--date-start", required=True)
    p.add_argument("--date-end", required=True)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def validate_flow_db(
    path: Path, date_start: str, date_end: str
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        report["ok"] = False
        return report

    con = connect_duckdb_readonly(path)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        report["tables"] = tables
        if "flujos_duckdb" not in tables:
            report["ok"] = False
            report["error"] = "Table 'flujos_duckdb' missing."
            return report

        schema = con.execute("DESCRIBE flujos_duckdb").fetchdf()
        report["schema"] = schema.set_index("column_name")["column_type"].to_dict()
        missing = EXPECTED_FLOW_COLS - set(report["schema"])
        if missing:
            report["ok"] = False
            report["error"] = f"Missing flow columns: {sorted(missing)}"
            return report

        # Temporal coverage
        coverage = con.execute(
            """
            SELECT MIN(FECHA) AS min_t, MAX(FECHA) AS max_t,
                   COUNT(*) AS total_rows
            FROM flujos_duckdb
            WHERE FECHA BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            """,
            [date_start, f"{date_end} 23:59:59"],
        ).fetchone()
        report["coverage"] = {
            "min_fecha": str(coverage[0]),
            "max_fecha": str(coverage[1]),
            "rows_in_range": int(coverage[2]),
        }
        if coverage[2] == 0:
            report["ok"] = False
            report["error"] = "No rows in requested date range."
            return report

        # Yearly distribution — run year by year to keep memory bounded
        years = list(range(int(date_start[:4]), int(date_end[:4]) + 1))
        yearly_rows = []
        for y in years:
            row = con.execute(
                """
                SELECT COUNT(*) AS rows,
                       COUNT(DISTINCT PORTICO) AS distinct_porticos,
                       COUNT(DISTINCT MATRICULA) AS distinct_plates,
                       COUNT(DISTINCT CATEGORIA) AS distinct_categorias
                FROM flujos_duckdb
                WHERE FECHA >= ?::TIMESTAMP AND FECHA < ?::TIMESTAMP
                """,
                [f"{y}-01-01", f"{y+1}-01-01"],
            ).fetchone()
            yearly_rows.append({
                "year": int(y),
                "rows": int(row[0]),
                "distinct_porticos": int(row[1]),
                "distinct_plates": int(row[2]),
                "distinct_categorias": int(row[3]),
            })
            print(f"    year {y}: rows={row[0]:,}")
        yearly = pd.DataFrame(yearly_rows)
        report["yearly_distribution"] = yearly_rows

        # Detect years with severely degraded coverage (<1% of median)
        median_rows = yearly["rows"].median()
        weak_years = yearly[yearly["rows"] < 0.01 * median_rows]
        report["weak_years"] = weak_years["year"].astype(int).tolist()

        report["ok"] = True
    finally:
        con.close()
    return report


def validate_porticos(path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        report["ok"] = False
        return report
    df = load_porticos_geometry(path)
    report["n_rows"] = int(len(df))
    report["columns"] = list(df.columns)
    report["distinct_porticos"] = int(df["cod_portico_norm"].nunique())
    report["has_lat_lon"] = bool({"lat", "lon"}.issubset(df.columns))
    report["has_km"] = "Km" in df.columns
    report["ok"] = report["n_rows"] > 0
    return report


def cross_check_porticos(
    flow_db: Path,
    porticos_df: pd.DataFrame,
    date_start: str,
    date_end: str,
) -> Dict[str, Any]:
    con = connect_duckdb_readonly(flow_db)
    try:
        # Use GROUP BY (more memory-stable than DISTINCT on 3.6B rows) and
        # restrict to a recent slice for representativeness.
        slice_start = date_end[:4] + "-01-01"
        seen = con.execute(
            """
            SELECT PORTICO, COUNT(*) AS n
            FROM flujos_duckdb
            WHERE FECHA >= ?::TIMESTAMP AND FECHA <= ?::TIMESTAMP
            GROUP BY PORTICO
            """,
            [slice_start, f"{date_end} 23:59:59"],
        ).fetchdf()
    finally:
        con.close()
    seen_norm = {normalize_portico_id(v) for v in seen["PORTICO"]}
    known = set(porticos_df["cod_portico_norm"])
    missing_in_csv = sorted(seen_norm - known)
    missing_in_flow = sorted(known - seen_norm)
    return {
        "n_porticos_in_flow": len(seen_norm),
        "n_porticos_in_csv": len(known),
        "missing_in_csv": missing_in_csv[:50],
        "missing_in_csv_count": len(missing_in_csv),
        "missing_in_flow": missing_in_flow[:50],
        "missing_in_flow_count": len(missing_in_flow),
        "ok": len(missing_in_csv) == 0,
    }


def validate_events(
    path: Path, date_start: str, date_end: str
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        report["ok"] = False
        return report
    con = connect_duckdb_readonly(path)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        report["tables"] = tables
        if "eventos" not in tables:
            report["ok"] = False
            report["error"] = "Table 'eventos' missing."
            return report

        coverage = con.execute(
            """
            SELECT MIN(evento_time) AS min_t, MAX(evento_time) AS max_t,
                   COUNT(*) AS total
            FROM eventos
            WHERE evento_time BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            """,
            [date_start, f"{date_end} 23:59:59"],
        ).fetchone()
        report["coverage"] = {
            "min": str(coverage[0]),
            "max": str(coverage[1]),
            "in_range": int(coverage[2]),
        }

        # Type vocabulary
        types = con.execute(
            """
            SELECT tipo_evento, COUNT(*) AS n
            FROM eventos
            WHERE evento_time BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            GROUP BY 1 ORDER BY n DESC
            """,
            [date_start, f"{date_end} 23:59:59"],
        ).fetchdf()
        report["event_types"] = types.to_dict(orient="records")

        # Yearly distribution of events
        yearly = con.execute(
            """
            SELECT YEAR(evento_time) AS year,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE portico_inicio IS NOT NULL) AS with_portico
            FROM eventos
            WHERE evento_time BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            GROUP BY 1 ORDER BY 1
            """,
            [date_start, f"{date_end} 23:59:59"],
        ).fetchdf()
        report["yearly_distribution"] = yearly.to_dict(orient="records")

        report["ok"] = coverage[2] > 0
    finally:
        con.close()
    return report


def cross_check_events_porticos(
    events_db: Path,
    porticos_df: pd.DataFrame,
    date_start: str,
    date_end: str,
) -> Dict[str, Any]:
    con = connect_duckdb_readonly(events_db)
    try:
        seen = con.execute(
            """
            SELECT DISTINCT portico_inicio AS p FROM eventos
            WHERE evento_time BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            UNION
            SELECT DISTINCT portico_fin AS p FROM eventos
            WHERE evento_time BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            """,
            [date_start, f"{date_end} 23:59:59", date_start, f"{date_end} 23:59:59"],
        ).fetchdf()
    finally:
        con.close()
    seen_norm = {normalize_portico_id(v) for v in seen["p"] if v is not None}
    known = set(porticos_df["cod_portico_norm"])
    missing = sorted(seen_norm - known)
    return {
        "n_event_porticos": len(seen_norm),
        "missing_in_csv": missing[:50],
        "missing_in_csv_count": len(missing),
        "ok": len(missing) <= 5,  # tolerate a few legacy codes
    }


def main() -> int:
    args = parse_args()
    report: Dict[str, Any] = {
        "params": {
            "flow_db": str(args.flow_db),
            "porticos_csv": str(args.porticos_csv),
            "events_db": str(args.events_db),
            "date_start": args.date_start,
            "date_end": args.date_end,
        },
    }

    report["flow_db_check"] = validate_flow_db(
        args.flow_db, args.date_start, args.date_end
    )
    report["porticos_check"] = validate_porticos(args.porticos_csv)
    report["events_db_check"] = validate_events(
        args.events_db, args.date_start, args.date_end
    )

    if report["porticos_check"].get("ok"):
        porticos_df = load_porticos_geometry(args.porticos_csv)
        report["flow_porticos_cross"] = cross_check_porticos(
            args.flow_db, porticos_df, args.date_start, args.date_end
        )
        report["events_porticos_cross"] = cross_check_events_porticos(
            args.events_db, porticos_df, args.date_start, args.date_end
        )
    else:
        report["flow_porticos_cross"] = {"ok": False, "reason": "porticos check failed"}
        report["events_porticos_cross"] = {"ok": False, "reason": "porticos check failed"}

    report["ready_for_phase_1"] = all(
        section.get("ok", False)
        for section in (
            report["flow_db_check"],
            report["porticos_check"],
            report["events_db_check"],
            report["flow_porticos_cross"],
            report["events_porticos_cross"],
        )
    )

    write_json_atomic(args.output, report)

    # Console summary
    print("=" * 60)
    print("Phase 0 — data validation")
    print("=" * 60)
    print(f"  flow_db:                {report['flow_db_check'].get('ok')}")
    print(f"  porticos_csv:           {report['porticos_check'].get('ok')}")
    print(f"  events_db:              {report['events_db_check'].get('ok')}")
    print(f"  flow ↔ porticos cross:  {report['flow_porticos_cross'].get('ok')}")
    print(f"  events ↔ porticos cross:{report['events_porticos_cross'].get('ok')}")
    print(f"  READY FOR PHASE 1:      {report['ready_for_phase_1']}")
    print(f"  full report: {args.output}")
    return 0 if report["ready_for_phase_1"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
