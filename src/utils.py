#!/usr/bin/env python
"""
utils.py
========
Funciones auxiliares utilizadas por el pipeline principal para cargar y
normalizar datos de flujos y pórticos.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import fnmatch
import io
import os
import re

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


DEFAULT_DUCKDB_FILE = Path("Datos") / "flujos.duckdb"
FLOW_TABLE_NAME = "flujos_duckdb"
FLOW_TABLE_SCHEMA = """
    FECHA TIMESTAMP,
    VELOCIDAD DOUBLE,
    CATEGORIA INTEGER,
    MATRICULA VARCHAR,
    PORTICO VARCHAR,
    CARRIL VARCHAR
"""
DEFAULT_INTERVAL_MINUTES = 5


@dataclass(frozen=True)
class FlowColumns:
    """
    Nombres estándar para las columnas del CSV de flujos.
    """

    timestamp: str = "FECHA"
    speed_kmh: str = "VELOCIDAD"
    class_id: str = "CATEGORIA"
    plate_id: str = "MATRICULA"
    portico: str = "PORTICO"
    lane: str = "CARRIL"


@dataclass(frozen=True)
class FlowDBSummary:
    row_count: int
    min_timestamp: Optional[pd.Timestamp]
    max_timestamp: Optional[pd.Timestamp]
    db_path: Path


@dataclass(frozen=True)
class FlowSampleSelection:
    date_start: Optional[pd.Timestamp] = None
    date_end: Optional[pd.Timestamp] = None
    row_limit: Optional[int] = None


def choose_file(start_folder: str) -> Optional[str]:
    """
    Permite navegar por carpetas para seleccionar un archivo CSV.
    """

    start_folder = os.path.abspath(start_folder)
    current_folder = start_folder

    while True:
        print(f"\n📁 Carpeta actual: {current_folder}")
        items = os.listdir(current_folder)
        dirs = sorted(
            [f for f in items if os.path.isdir(os.path.join(current_folder, f))]
        )
        files = sorted(
            [
                f
                for f in items
                if os.path.isfile(os.path.join(current_folder, f))
                and f.lower().endswith(".csv")
            ]
        )
        opciones = dirs + files

        if current_folder == start_folder:
            print("  [0] 🔙 Volver/Salir")
            offset = 1
        else:
            print("  [0] 🔙 Subir un nivel")
            offset = 1

        for i, name in enumerate(opciones):
            icon = "📂" if name in dirs else "📄"
            print(f"  [{i+offset}] {icon} {name}")

        choice = input("Seleccione una opción: ").strip()
        if choice == "0":
            if current_folder == start_folder:
                print("Cancelado.")
                return None
            current_folder = os.path.dirname(current_folder)
            continue

        try:
            idx = int(choice) - offset
            if 0 <= idx < len(opciones):
                selected = opciones[idx]
                selected_path = os.path.join(current_folder, selected)
                if os.path.isdir(selected_path):
                    current_folder = selected_path
                else:
                    print(f"Archivo seleccionado: {selected_path}\n")
                    return selected_path
            else:
                print("Selección fuera de rango.")
        except ValueError:
            print("Entrada inválida.")


def choose_file_filtered(start_folder: str, patterns) -> Optional[str]:
    """
    Igual a choose_file, pero solo muestra archivos CSV que apliquen a los patrones glob.
    """

    if isinstance(patterns, str):
        patterns = [patterns]
    patterns_l = [p.lower() for p in patterns]

    start_folder = os.path.abspath(start_folder)
    current_folder = start_folder

    while True:
        print(f"\n📁 Carpeta actual: {current_folder}")
        items = os.listdir(current_folder)
        dirs = sorted(
            [f for f in items if os.path.isdir(os.path.join(current_folder, f))]
        )
        files = []
        for f in items:
            fp = os.path.join(current_folder, f)
            if os.path.isfile(fp) and f.lower().endswith(".csv"):
                name_l = f.lower()
                if any(fnmatch.fnmatch(name_l, p) for p in patterns_l):
                    files.append(f)
        files = sorted(files)

        opciones = dirs + files

        if current_folder == start_folder:
            print("  [0] 🔙 Volver/Salir")
            offset = 1
        else:
            print("  [0] 🔙 Subir un nivel")
            offset = 1

        if not files and not dirs:
            print("  (No hay carpetas ni archivos que coincidan con el patrón.)")

        print(f"  ↳ Mostrando solo archivos: {', '.join(patterns)}")
        for i, name in enumerate(opciones):
            icon = "📂" if name in dirs else "📄"
            print(f"  [{i+offset}] {icon} {name}")

        choice = input("Seleccione una opción: ").strip()
        if choice == "0":
            if current_folder == start_folder:
                print("Cancelado.")
                return None
            current_folder = os.path.dirname(current_folder)
            continue

        try:
            idx = int(choice) - offset
            if 0 <= idx < len(opciones):
                selected = opciones[idx]
                selected_path = os.path.join(current_folder, selected)
                if os.path.isdir(selected_path):
                    current_folder = selected_path
                else:
                    print(f"Archivo seleccionado: {selected_path}\n")
                    return selected_path
            else:
                print("Selección fuera de rango.")
        except ValueError:
            print("Entrada inválida.")


def read_csv_with_progress(
    path_csv: str,
    chunksize: int = 100_000,
    sep: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Lee un CSV en chunks mostrando una barra de progreso.
    Intenta leer con UTF-8 y si falla, reintenta con 'latin-1'.
    """

    def _read_chunks_with_progress(path, encoding, separator, chunk_size, row_limit):
        total_size = os.path.getsize(path)
        chunks = []
        rows_loaded = 0
        with open(path, "rb") as f:
            f_text = io.TextIOWrapper(f, encoding=encoding, errors="strict")
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Cargando CSV ({encoding})",
                disable=True,
            ) as pbar:
                reader = pd.read_csv(
                    f_text, sep=separator, chunksize=chunk_size, low_memory=False
                )
                last_position = f.tell()
                for chunk in reader:
                    if row_limit is not None:
                        remaining = row_limit - rows_loaded
                        if remaining <= 0:
                            break
                        chunk = chunk.iloc[:remaining]
                    if chunk.empty:
                        break
                    chunks.append(chunk)
                    rows_loaded += len(chunk)
                    current_position = f.tell()
                    pbar.update(current_position - last_position)
                    last_position = current_position
                    if row_limit is not None and rows_loaded >= row_limit:
                        break
        return pd.concat(chunks, ignore_index=True)

    detected_sep = sep
    if detected_sep is None:
        try:
            with open(path_csv, "r", encoding="utf-8") as f:
                sample = f.read(4096)
        except UnicodeDecodeError:
            with open(path_csv, "r", encoding="latin-1") as f:
                sample = f.read(4096)

        import csv

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            detected_sep = dialect.delimiter
            print(f"Separador detectado: '{detected_sep}'")
        except (csv.Error, UnicodeDecodeError):
            detected_sep = ","
            print(
                f"No se pudo detectar el separador, usando '{detected_sep}' por defecto."
            )

    try:
        #print("Intentando leer con codificación 'utf-8'...")
        return _read_chunks_with_progress(
            path_csv, "utf-8", detected_sep, chunksize, max_rows
        )
    except UnicodeDecodeError:
        print("⚠️  Lectura con UTF-8 fallida. Reintentando con 'latin-1'...")
        return _read_chunks_with_progress(
            path_csv, "latin-1", detected_sep, chunksize, max_rows
        )


def select_flujos_csv_path() -> Optional[str]:
    """
    Selecciona el archivo de flujos (FLUJO*.csv) desde la carpeta 'Datos'.
    """

    base_dir = "Datos"
    patterns = ["flujo*.csv"]
    try:
        top_files = [
            f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))
        ]
    except FileNotFoundError:
        top_files = []
    matches_top = [
        os.path.join(base_dir, f)
        for f in top_files
        if f.lower().endswith(".csv")
        and any(fnmatch.fnmatch(f.lower(), p) for p in patterns)
    ]
    if len(matches_top) == 1:
        file_path = matches_top[0]
        print(
            f"🔎 Detectado único archivo de flujos en '{base_dir}'. "
            f"Cargando automáticamente → {file_path}"
        )
        return file_path
    return choose_file_filtered(base_dir, patterns)


def _ensure_duckdb_available() -> None:
    if duckdb is None:  # pragma: no cover
        raise ImportError(
            "duckdb no está instalado. Ejecute `pip install duckdb` para habilitar la carga de flujos."
        )


def _get_duckdb_path(db_path: Optional[Path] = None) -> Path:
    path = Path(db_path) if db_path is not None else DEFAULT_DUCKDB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect_duckdb(read_only: bool = False, db_path: Optional[Path] = None):
    _ensure_duckdb_available()
    path = _get_duckdb_path(db_path)
    # DuckDB refuses to open the same file with mixed read_only settings inside
    # the same process. The Streamlit app interleaves reads and writes against
    # this database on reruns, so we keep a single writable configuration and
    # treat `read_only=True` as caller intent rather than a driver flag.
    return duckdb.connect(str(path), read_only=False)


def _create_flow_table(conn) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {FLOW_TABLE_NAME} ({FLOW_TABLE_SCHEMA})")

    info = conn.execute(f"PRAGMA table_info('{FLOW_TABLE_NAME}')").fetchall()
    column_names = [row[1] for row in info]



def _flow_date_parse_expr(column: str = "fecha_txt") -> str:
    """
    Construye una expresión SQL que intenta parsear múltiples formatos de fecha comunes.
    """

    return (
        "COALESCE("
        f"try_strptime({column}, '%d/%m/%Y %H:%M:%S'),"
        f"try_strptime({column}, '%d-%m-%Y %H:%M:%S'),"
        f"try_strptime({column}, '%Y-%m-%d %H:%M:%S'),"
        f"try_strptime({column}, '%Y/%m/%d %H:%M:%S'),"
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%SZ'),"
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%S.%fZ'),"
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%S%z'),"
        f"try_strptime({column}, '%d/%m/%Y %H:%M'),"
        f"try_strptime({column}, '%Y-%m-%d %H:%M')"
        ")"
    )


def _flow_csv_select_sql() -> str:
    fecha_expr = _flow_date_parse_expr()
    return f"""
        WITH raw AS (
            SELECT *
            FROM read_csv_auto(?,
                SAMPLE_SIZE=-1,
                HEADER=TRUE,
                IGNORE_ERRORS=TRUE,
                ALL_VARCHAR=TRUE
            )
        ),
        normalized AS (
            SELECT
                raw.*,
                NULLIF(TRIM(raw.FECHA), '') AS fecha_txt,
                NULLIF(TRIM(raw.MATRICULA), '') AS matricula_txt,
                NULLIF(TRIM(raw.PORTICO), '') AS portico_txt,
                NULLIF(TRIM(raw.CARRIL), '') AS carril_txt,
                TRY_CAST(NULLIF(TRIM(raw.CATEGORIA), '') AS INTEGER) AS categoria_num,
                TRY_CAST(REPLACE(NULLIF(TRIM(raw.VELOCIDAD), ''), ',', '.') AS DOUBLE) AS velocidad_num
            FROM raw
        ),
        prepared AS (
            SELECT
                {fecha_expr} AS FECHA,
                velocidad_num AS VELOCIDAD,
                categoria_num AS CATEGORIA,
                matricula_txt AS MATRICULA,
                portico_txt AS PORTICO,
                carril_txt AS CARRIL
            FROM normalized
            WHERE categoria_num IS NOT NULL
              AND {fecha_expr} IS NOT NULL
        )
        SELECT * FROM prepared
    """


ProgressCallback = Callable[[float, str], None]


def import_flujos_to_duckdb(
    csv_path: Optional[str] = None,
    *,
    replace: bool = False,
    db_path: Optional[Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> int:
    """
    Importa un archivo CSV de flujos hacia la base DuckDB.
    """

    csv_path = csv_path or select_flujos_csv_path()
    if csv_path is None:
        return 0

    csv_path = str(Path(csv_path).resolve())

    def _report(ratio: float, message: str) -> None:
        if progress_callback is None:
            return
        ratio_clamped = min(max(ratio, 0.0), 1.0)
        try:
            progress_callback(ratio_clamped, message)
        except Exception:
            pass

    def _stage(message: str, value: float) -> None:
        _report(value, message)

    conn = _connect_duckdb(read_only=False, db_path=db_path)
    select_sql = _flow_csv_select_sql()
    inserted = 0
    success = False
    try:
        _stage("Iniciando importación de flujos", 0.0)
        _stage("Validando archivo CSV seleccionado", 0.05)
        if replace:
            _stage("Eliminando tabla actual", 0.1)
            conn.execute(f"DROP TABLE IF EXISTS {FLOW_TABLE_NAME}")
            _stage("Leyendo CSV y construyendo tabla", 0.25)
            conn.execute(f"CREATE TABLE {FLOW_TABLE_NAME} AS {select_sql}", [csv_path])
            _stage("Contando filas importadas", 0.65)
            inserted = conn.execute(
                f"SELECT COUNT(*) FROM {FLOW_TABLE_NAME}"
            ).fetchone()[0]
            _stage("Tabla reemplazada satisfactoriamente", 0.85)
        else:
            _stage("Asegurando estructura de la tabla", 0.12)
            _create_flow_table(conn)
            _stage("Cargando CSV en tabla temporal", 0.3)
            conn.execute(
                f"CREATE OR REPLACE TEMP TABLE __new_flujos AS {select_sql}",
                [csv_path],
            )
            _stage("Contando filas nuevas", 0.55)
            inserted = conn.execute(
                "SELECT COUNT(*) FROM __new_flujos"
            ).fetchone()[0]
            if inserted:
                _stage("Insertando registros en la tabla principal", 0.7)
                conn.execute(
                    f"""
                    INSERT INTO {FLOW_TABLE_NAME} (FECHA, VELOCIDAD, CATEGORIA,
                        MATRICULA, PORTICO, CARRIL)
                    SELECT FECHA, VELOCIDAD, CATEGORIA,
                        MATRICULA, PORTICO, CARRIL
                    FROM __new_flujos
                    """
                )
            else:
                _stage("No se encontraron filas nuevas para insertar", 0.7)
            conn.execute("DROP TABLE __new_flujos")
            _stage("Tabla temporal eliminada", 0.8)
        success = True
        _stage("Validando resultado final", 0.95)
        return int(inserted or 0)
    finally:
        conn.close()
        if success:
            _stage("Importación completada", 1.0)
        else:
            _stage("Importación interrumpida", 1.0)


def clear_flow_table(db_path: Optional[Path] = None) -> int:
    """
    Elimina todos los registros de flujos de la base DuckDB.
    """

    conn = _connect_duckdb(read_only=False, db_path=db_path)
    try:
        _create_flow_table(conn)
        existing = conn.execute(
            f"SELECT COUNT(*) FROM {FLOW_TABLE_NAME}"
        ).fetchone()[0]
        conn.execute(f"DELETE FROM {FLOW_TABLE_NAME}")
        return int(existing or 0)
    finally:
        conn.close()


def delete_flow_rows_by_date_range(
    *,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    db_path: Optional[Path] = None,
) -> int:
    """
    Elimina registros de flujos entre dos fechas (inclusive).
    """

    if date_start is None or date_end is None:
        return 0

    conn = _connect_duckdb(read_only=False, db_path=db_path)
    try:
        _create_flow_table(conn)
        params = [date_start, date_end]
        existing = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {FLOW_TABLE_NAME}
            WHERE FECHA >= ? AND FECHA <= ?
            """,
            params,
        ).fetchone()[0]
        if existing:
            conn.execute(
                f"""
                DELETE FROM {FLOW_TABLE_NAME}
                WHERE FECHA >= ? AND FECHA <= ?
                """,
                params,
            )
        return int(existing or 0)
    finally:
        conn.close()


def _fetch_summary_row(conn) -> Tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            MIN(FECHA) AS min_ts,
            MAX(FECHA) AS max_ts
        FROM {FLOW_TABLE_NAME}
        """
    ).fetchone()
    return row


def get_flow_db_summary(db_path: Optional[Path] = None) -> FlowDBSummary:
    """
    Retorna métricas básicas de la tabla de flujos en DuckDB.
    """

    db_file = _get_duckdb_path(db_path)
    row: Optional[Tuple[int, Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None
    try:
        conn = _connect_duckdb(read_only=True, db_path=db_path)
        try:
            row = _fetch_summary_row(conn)
        finally:
            conn.close()
    except Exception as exc:
        catalog_exc = getattr(duckdb, "CatalogException", None)
        if catalog_exc is not None and isinstance(exc, catalog_exc):
            conn = _connect_duckdb(read_only=False, db_path=db_path)
            try:
                _create_flow_table(conn)
                row = _fetch_summary_row(conn)
            finally:
                conn.close()
        else:
            raise

    if row is None:
        raise RuntimeError("Unable to obtain flow table summary.")

    min_ts = pd.to_datetime(row[1]) if row[1] is not None else None
    max_ts = pd.to_datetime(row[2]) if row[2] is not None else None
    return FlowDBSummary(
        row_count=int(row[0] or 0),
        min_timestamp=min_ts,
        max_timestamp=max_ts,
        db_path=db_file,
    )


def ensure_flow_db_summary(db_path: Optional[Path] = None) -> Optional[FlowDBSummary]:
    """
    Obtiene el resumen de la base de flujos, importando un CSV si es necesario.
    """

    try:
        summary = get_flow_db_summary(db_path=db_path)
    except ImportError as exc:
        print(f"❌ {exc}")
        return None

    if summary.row_count == 0:
        print(f"⚠️ No se encontraron datos en {summary.db_path}.")
        resp = input("¿Desea importar un CSV de flujos ahora? (s/n): ").strip().lower()
        if resp not in {"s", "si", "y", "yes"}:
            return None
        inserted = import_flujos_to_duckdb(db_path=db_path)
        if inserted == 0:
            print("⚠️ No se importaron registros. Reintente la operación.")
            return None
        summary = get_flow_db_summary(db_path=db_path)

    return summary


def get_flow_table_columns(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Devuelve el esquema actual de la tabla de flujos.
    """

    conn = _connect_duckdb(read_only=False, db_path=db_path)
    try:
        _create_flow_table(conn)
        info = conn.execute(f"PRAGMA table_info('{FLOW_TABLE_NAME}')").fetchall()
    finally:
        conn.close()
    columns: List[Dict[str, Any]] = []
    for row in info:
        columns.append(
            {
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "default": row[4],
                "pk": bool(row[5]),
            }
        )
    return columns


def get_flow_table_sample(
    limit: int = 5, db_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Obtiene una muestra de la tabla de flujos.
    """

    if limit <= 0:
        return pd.DataFrame()
    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT *
            FROM {FLOW_TABLE_NAME}
            ORDER BY COALESCE(FECHA, TIMESTAMP '1970-01-01')
            LIMIT ?
        """
        return conn.execute(query, [int(limit)]).df()
    finally:
        conn.close()


def get_flow_counts_by_category(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Retorna el total de observaciones por categoría vehicular.
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                CATEGORIA AS categoria,
                COUNT(*) AS total
            FROM {FLOW_TABLE_NAME}
            WHERE CATEGORIA IS NOT NULL
            GROUP BY CATEGORIA
            ORDER BY total DESC, categoria
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_flow_counts_by_year(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Retorna el total de observaciones por año.
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                CAST(EXTRACT(year FROM FECHA) AS INTEGER) AS anio,
                COUNT(*) AS total
            FROM {FLOW_TABLE_NAME}
            WHERE FECHA IS NOT NULL
            GROUP BY anio
            ORDER BY anio
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_flow_counts_by_month(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Retorna el total de observaciones por mes.
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                CAST(EXTRACT(year FROM FECHA) AS INTEGER) AS anio,
                CAST(EXTRACT(month FROM FECHA) AS INTEGER) AS mes,
                COUNT(*) AS total
            FROM {FLOW_TABLE_NAME}
            WHERE FECHA IS NOT NULL
            GROUP BY anio, mes
            ORDER BY anio, mes
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_flow_days_by_year(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Retorna el total de días con observaciones por año.
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                CAST(EXTRACT(year FROM FECHA) AS INTEGER) AS anio,
                COUNT(DISTINCT CAST(FECHA AS DATE)) AS dias
            FROM {FLOW_TABLE_NAME}
            WHERE FECHA IS NOT NULL
            GROUP BY anio
            ORDER BY anio
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def _flow_time_bin_expr(interval_minutes: int) -> str:
    interval = max(1, int(interval_minutes))
    return (
        "DATE_TRUNC('minute', FECHA) - "
        f"(EXTRACT(minute FROM FECHA) % {interval}) * INTERVAL '1 minute'"
    )


def get_flow_basic_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Retorna métricas básicas de cobertura y escala de la tabla de flujos.
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                COUNT(*) AS total,
                MIN(FECHA) AS min_ts,
                MAX(FECHA) AS max_ts,
                COUNT(DISTINCT NULLIF(TRIM(PORTICO), '')) AS porticos,
                COUNT(DISTINCT NULLIF(TRIM(CARRIL), '')) AS carriles,
                COUNT(DISTINCT CATEGORIA) AS categorias,
                COUNT(DISTINCT NULLIF(UPPER(TRIM(MATRICULA)), '')) AS patentes,
                COUNT(DISTINCT CAST(FECHA AS DATE)) AS dias,
                COUNT(DISTINCT DATE_TRUNC('hour', FECHA)) AS horas
            FROM {FLOW_TABLE_NAME}
        """
        row = conn.execute(query).fetchone()
    finally:
        conn.close()

    if row is None:
        return {}

    min_ts = pd.to_datetime(row[1]) if row[1] is not None else None
    max_ts = pd.to_datetime(row[2]) if row[2] is not None else None
    return {
        "total": int(row[0] or 0),
        "min_ts": min_ts,
        "max_ts": max_ts,
        "porticos": int(row[3] or 0),
        "carriles": int(row[4] or 0),
        "categorias": int(row[5] or 0),
        "patentes": int(row[6] or 0),
        "dias": int(row[7] or 0),
        "horas": int(row[8] or 0),
    }


def get_flow_missingness_stats(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Retorna el conteo y porcentaje de missingness en columnas clave.
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN CARRIL IS NULL OR TRIM(CARRIL) = '' THEN 1 ELSE 0 END) AS miss_carril,
                SUM(CASE WHEN VELOCIDAD IS NULL THEN 1 ELSE 0 END) AS miss_velocidad,
                SUM(CASE WHEN CATEGORIA IS NULL THEN 1 ELSE 0 END) AS miss_categoria,
                SUM(CASE WHEN PORTICO IS NULL OR TRIM(PORTICO) = '' THEN 1 ELSE 0 END) AS miss_portico,
                SUM(CASE WHEN MATRICULA IS NULL OR TRIM(MATRICULA) = '' THEN 1 ELSE 0 END) AS miss_matricula
            FROM {FLOW_TABLE_NAME}
        """
        row = conn.execute(query).fetchone()
    finally:
        conn.close()

    if row is None:
        return pd.DataFrame()

    total = int(row[0] or 0)
    if total <= 0:
        return pd.DataFrame()

    def _pct(value: int) -> float:
        return round(value * 100.0 / total, 3)

    data = [
        {"Campo": "CARRIL", "Missing": int(row[1] or 0), "%": _pct(int(row[1] or 0))},
        {"Campo": "VELOCIDAD", "Missing": int(row[2] or 0), "%": _pct(int(row[2] or 0))},
        {"Campo": "CATEGORIA", "Missing": int(row[3] or 0), "%": _pct(int(row[3] or 0))},
        {"Campo": "PORTICO", "Missing": int(row[4] or 0), "%": _pct(int(row[4] or 0))},
        {"Campo": "MATRICULA", "Missing": int(row[5] or 0), "%": _pct(int(row[5] or 0))},
    ]
    return pd.DataFrame(data)


def get_flow_speed_stats(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Estadísticos de velocidad instantánea (global).
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                AVG(VELOCIDAD) AS mean,
                STDDEV_SAMP(VELOCIDAD) AS sd,
                QUANTILE_CONT(VELOCIDAD, 0.05) AS p5,
                QUANTILE_CONT(VELOCIDAD, 0.25) AS p25,
                QUANTILE_CONT(VELOCIDAD, 0.50) AS p50,
                QUANTILE_CONT(VELOCIDAD, 0.95) AS p95
            FROM {FLOW_TABLE_NAME}
            WHERE VELOCIDAD IS NOT NULL
        """
        row = conn.execute(query).fetchone()
    finally:
        conn.close()

    if row is None:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Variable": "Velocidad (km/h)",
                "Mean": row[0],
                "SD": row[1],
                "P5": row[2],
                "P25": row[3],
                "P50": row[4],
                "P95": row[5],
            }
        ]
    )


def get_flow_speed_stats_by_category(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Estadísticos de velocidad instantánea por categoría.
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            SELECT
                CASE WHEN CATEGORIA = 4 THEN 2 ELSE CATEGORIA END AS categoria,
                AVG(VELOCIDAD) AS mean,
                STDDEV_SAMP(VELOCIDAD) AS sd,
                QUANTILE_CONT(VELOCIDAD, 0.05) AS p5,
                QUANTILE_CONT(VELOCIDAD, 0.25) AS p25,
                QUANTILE_CONT(VELOCIDAD, 0.50) AS p50,
                QUANTILE_CONT(VELOCIDAD, 0.95) AS p95
            FROM {FLOW_TABLE_NAME}
            WHERE VELOCIDAD IS NOT NULL AND CATEGORIA IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_flow_plate_concentration(
    db_path: Optional[Path] = None, top_percent: float = 0.01
) -> Dict[str, Any]:
    """
    Retorna métricas de concentración de patentes (top % de patentes).
    """

    pct = max(0.0, float(top_percent))
    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            WITH plate_counts AS (
                SELECT
                    NULLIF(UPPER(TRIM(MATRICULA)), '') AS plate,
                    COUNT(*) AS cnt
                FROM {FLOW_TABLE_NAME}
                WHERE MATRICULA IS NOT NULL
                GROUP BY plate
            ),
            ordered AS (
                SELECT
                    cnt,
                    ROW_NUMBER() OVER (ORDER BY cnt DESC) AS rn,
                    COUNT(*) OVER () AS total_plates,
                    SUM(cnt) OVER () AS total_detections
                FROM plate_counts
                WHERE plate IS NOT NULL
            )
            SELECT
                SUM(cnt) FILTER (WHERE rn <= CEIL(total_plates * {pct})) AS top_detections,
                MAX(total_detections) AS total_detections,
                MAX(total_plates) AS total_plates
            FROM ordered
        """
        row = conn.execute(query).fetchone()
    finally:
        conn.close()

    if row is None:
        return {}

    top_detections = int(row[0] or 0)
    total_detections = int(row[1] or 0)
    total_plates = int(row[2] or 0)
    share = (top_detections / total_detections * 100.0) if total_detections else 0.0
    return {
        "top_percent": pct,
        "top_detections": top_detections,
        "total_detections": total_detections,
        "total_plates": total_plates,
        "top_share": share,
    }


def get_flow_window_stats(
    db_path: Optional[Path] = None, interval_minutes: int = DEFAULT_INTERVAL_MINUTES
) -> pd.DataFrame:
    """
    Estadísticos de variables a nivel pórtico-ventana.
    """

    interval = max(1, int(interval_minutes))
    flow_bin = _flow_time_bin_expr(interval)
    scale = 60.0 / interval
    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        query = f"""
            WITH lanes AS (
                SELECT
                    PORTICO,
                    COUNT(DISTINCT NULLIF(TRIM(CARRIL), '')) AS lanes
                FROM {FLOW_TABLE_NAME}
                WHERE PORTICO IS NOT NULL
                GROUP BY PORTICO
            ),
            windows AS (
                SELECT
                    PORTICO,
                    {flow_bin} AS ts_bin,
                    COUNT(*) AS n,
                    AVG(VELOCIDAD) AS vbar
                FROM {FLOW_TABLE_NAME}
                WHERE FECHA IS NOT NULL AND PORTICO IS NOT NULL
                GROUP BY PORTICO, ts_bin
            ),
            features AS (
                SELECT
                    n,
                    vbar,
                    CASE WHEN lanes > 0 THEN n * {scale} / lanes ELSE NULL END AS q,
                    CASE WHEN vbar > 0 AND lanes > 0 THEN (n * {scale} / lanes) / vbar ELSE NULL END AS k
                FROM windows
                LEFT JOIN lanes USING (PORTICO)
            )
            SELECT
                AVG(n) AS n_mean,
                STDDEV_SAMP(n) AS n_sd,
                QUANTILE_CONT(n, 0.05) AS n_p5,
                QUANTILE_CONT(n, 0.25) AS n_p25,
                QUANTILE_CONT(n, 0.50) AS n_p50,
                QUANTILE_CONT(n, 0.95) AS n_p95,
                AVG(vbar) AS v_mean,
                STDDEV_SAMP(vbar) AS v_sd,
                QUANTILE_CONT(vbar, 0.05) AS v_p5,
                QUANTILE_CONT(vbar, 0.25) AS v_p25,
                QUANTILE_CONT(vbar, 0.50) AS v_p50,
                QUANTILE_CONT(vbar, 0.95) AS v_p95,
                AVG(q) AS q_mean,
                STDDEV_SAMP(q) AS q_sd,
                QUANTILE_CONT(q, 0.05) AS q_p5,
                QUANTILE_CONT(q, 0.25) AS q_p25,
                QUANTILE_CONT(q, 0.50) AS q_p50,
                QUANTILE_CONT(q, 0.95) AS q_p95,
                AVG(k) AS k_mean,
                STDDEV_SAMP(k) AS k_sd,
                QUANTILE_CONT(k, 0.05) AS k_p5,
                QUANTILE_CONT(k, 0.25) AS k_p25,
                QUANTILE_CONT(k, 0.50) AS k_p50,
                QUANTILE_CONT(k, 0.95) AS k_p95
            FROM features
        """
        row = conn.execute(query).fetchone()
    finally:
        conn.close()

    if row is None:
        return pd.DataFrame()

    rows = [
        {
            "Variable": "n (veh/ventana)",
            "Mean": row[0],
            "SD": row[1],
            "P5": row[2],
            "P25": row[3],
            "P50": row[4],
            "P95": row[5],
        },
        {
            "Variable": "v̄ (km/h)",
            "Mean": row[6],
            "SD": row[7],
            "P5": row[8],
            "P25": row[9],
            "P50": row[10],
            "P95": row[11],
        },
        {
            "Variable": "q (veh/h/carril)",
            "Mean": row[12],
            "SD": row[13],
            "P5": row[14],
            "P25": row[15],
            "P50": row[16],
            "P95": row[17],
        },
        {
            "Variable": "k (veh/km/carril)",
            "Mean": row[18],
            "SD": row[19],
            "P5": row[20],
            "P25": row[21],
            "P50": row[22],
            "P95": row[23],
        },
    ]
    return pd.DataFrame(rows)


def get_flow_window_prevalence_stats(
    acc_windows: pd.DataFrame,
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    db_path: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Calcula el total de ventanas y las positivas en el set de flujos.
    """

    interval = max(1, int(interval_minutes))
    flow_bin = _flow_time_bin_expr(interval)
    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        acc_df = acc_windows.copy()
        if not acc_df.empty:
            acc_df = acc_df[["PORTICO", "ts_bin"]].dropna().drop_duplicates()
        if acc_df.empty:
            query_total = f"""
                SELECT COUNT(*) FROM (
                    SELECT PORTICO, {flow_bin} AS ts_bin
                    FROM {FLOW_TABLE_NAME}
                    WHERE FECHA IS NOT NULL AND PORTICO IS NOT NULL
                    GROUP BY PORTICO, ts_bin
                ) t
            """
            total = int(conn.execute(query_total).fetchone()[0] or 0)
            return {"total_windows": total, "positive_windows": 0}

        conn.register("acc_windows", acc_df)
        query = f"""
            WITH flow_windows AS (
                SELECT
                    PORTICO,
                    {flow_bin} AS ts_bin
                FROM {FLOW_TABLE_NAME}
                WHERE FECHA IS NOT NULL AND PORTICO IS NOT NULL
                GROUP BY PORTICO, ts_bin
            ),
            acc AS (
                SELECT DISTINCT PORTICO, ts_bin FROM acc_windows
            )
            SELECT
                (SELECT COUNT(*) FROM flow_windows) AS total_windows,
                (SELECT COUNT(*) FROM (
                    SELECT acc.PORTICO, acc.ts_bin
                    FROM acc
                    INNER JOIN flow_windows
                        ON acc.PORTICO = flow_windows.PORTICO
                       AND acc.ts_bin = flow_windows.ts_bin
                    GROUP BY acc.PORTICO, acc.ts_bin
                ) t) AS positive_windows
        """
        row = conn.execute(query).fetchone()
    finally:
        conn.close()

    if row is None:
        return {"total_windows": 0, "positive_windows": 0}
    return {
        "total_windows": int(row[0] or 0),
        "positive_windows": int(row[1] or 0),
    }


def buscar_columna(df: pd.DataFrame, nombre_esperado: str) -> str:
    """
    Busca una columna que se parezca a 'nombre_esperado'. Si no se encuentra, solicita al usuario.
    """

    columnas = list(df.columns)
    if nombre_esperado in columnas:
        return nombre_esperado
    columnas_norm = {col.strip().lower().replace(" ", ""): col for col in columnas}
    clave = nombre_esperado.strip().lower().replace(" ", "")
    if clave in columnas_norm:
        return columnas_norm[clave]
    print(f"No se encontró la columna '{nombre_esperado}'. Columnas disponibles:")
    for i, col in enumerate(columnas):
        print(f"  [{i}] {col}")
    idx = input(
        f"Ingrese el número correspondiente a la columna que represente '{nombre_esperado}': "
    ).strip()
    try:
        idx = int(idx)
        return columnas[idx]
    except Exception as exc:  # pragma: no cover - interacción manual
        raise ValueError("Selección inválida. Proceso abortado.") from exc


def find_column_case_insensitive(
    df: pd.DataFrame, nombre_esperado: str
) -> Optional[str]:
    """
    Busca una columna ignorando mayúsculas/minúsculas y espacios. Retorna None si no existe.
    """

    nombre_norm = nombre_esperado.strip().lower()
    mapping = {col.strip().lower(): col for col in df.columns}
    return mapping.get(nombre_norm)


def load_porticos() -> Optional[pd.DataFrame]:
    """
    Carga el CSV de pórticos y normaliza las columnas principales.
    """

    auto_dir = "Datos"
    auto_path = None
    try:
        if os.path.isdir(auto_dir):
            for fname in os.listdir(auto_dir):
                if fname.lower() == "porticos.csv":
                    auto_path = os.path.join(auto_dir, fname)
                    break
    except Exception:
        auto_path = None

    if auto_path is not None and os.path.isfile(auto_path):
        print(
            f"🔎 Detectado 'Porticos.csv' en '{auto_dir}'. "
            f"Cargando automáticamente → {auto_path}"
        )
        file_path = auto_path
    else:
        file_path = choose_file("Datos")
        if file_path is None:
            return None

    print("Cargando información de pórticos...")
    df = read_csv_with_progress(file_path, sep=";")

    cod_portico_col = buscar_columna(df, "cod_portico")
    km_col = buscar_columna(df, "Km")
    calzada_col = buscar_columna(df, "Calzada")
    orden_col = buscar_columna(df, "Orden")
    eje_col = buscar_columna(df, "Eje")
    optional_cols = {
        "edge_id_sumo": find_column_case_insensitive(df, "edge_id_sumo"),
        "lane_id_sumo": find_column_case_insensitive(df, "lane_id_sumo"),
        "pos_m": find_column_case_insensitive(df, "pos_m"),
        "lat-lon": find_column_case_insensitive(df, "lat-lon"),
        "lat": find_column_case_insensitive(df, "lat"),
        "lon": find_column_case_insensitive(df, "lon"),
        "aux": find_column_case_insensitive(df, "aux"),
    }

    df[km_col] = df[km_col].astype(str).str.replace(",", ".")
    df[km_col] = pd.to_numeric(df[km_col], errors="coerce")
    df[orden_col] = pd.to_numeric(df[orden_col], errors="coerce")

    def _clean_portico(val):
        if pd.isna(val):
            return val
        try:
            f = float(val)
            if f.is_integer():
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return str(val).strip()

    df[cod_portico_col] = df[cod_portico_col].apply(_clean_portico)

    keep_columns = [cod_portico_col, km_col, calzada_col, orden_col, eje_col]
    for col in optional_cols.values():
        if col:
            keep_columns.append(col)

    df_porticos = df[keep_columns].rename(
        columns={
            cod_portico_col: "portico",
            km_col: "km",
            calzada_col: "calzada",
            orden_col: "orden",
            eje_col: "eje",
            **{
                optional_col: optional
                for optional, optional_col in optional_cols.items()
                if optional_col
            },
        }
    )
    if "pos_m" in df_porticos.columns:
        df_porticos["pos_m"] = (
            df_porticos["pos_m"].astype(str).str.replace(",", ".").astype(float)
        )
    return df_porticos


def _format_flow_range(summary: FlowDBSummary) -> Optional[str]:
    if summary.min_timestamp or summary.max_timestamp:
        start = (
            summary.min_timestamp.strftime("%Y-%m-%d %H:%M")
            if summary.min_timestamp
            else "-"
        )
        end = (
            summary.max_timestamp.strftime("%Y-%m-%d %H:%M")
            if summary.max_timestamp
            else "-"
        )
        return f"{start} → {end}"
    return None


def _format_flow_range_from_series(series: pd.Series) -> Optional[str]:
    if series is None or series.empty:
        return None
    min_ts = series.min()
    max_ts = series.max()
    if not isinstance(min_ts, pd.Timestamp) and not pd.isna(min_ts):
        series = pd.to_datetime(series, errors="coerce")
        min_ts = series.min()
        max_ts = series.max()
    if pd.isna(min_ts) and pd.isna(max_ts):
        return None
    if isinstance(min_ts, pd.Timestamp) and min_ts.tzinfo is not None:
        min_ts = min_ts.tz_convert(None)
    if isinstance(max_ts, pd.Timestamp) and max_ts.tzinfo is not None:
        max_ts = max_ts.tz_convert(None)
    start = min_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(min_ts) else "-"
    end = max_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(max_ts) else "-"
    return f"{start} → {end}"


def _prompt_flow_date(label: str) -> tuple[pd.Timestamp, bool]:
    while True:
        raw = input(f"Ingrese {label} (AAAA-MM-DD o AAAA-MM-DD HH:MM): ").strip()
        if not raw:
            print("Entrada vacía. Intente nuevamente.")
            continue
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            print("Fecha inválida. Intente nuevamente.")
            continue
        if isinstance(parsed, pd.Timestamp) and parsed.tzinfo is not None:
            parsed = parsed.tz_convert(None)
        has_time = ":" in raw
        return parsed, has_time


def prompt_flow_sample_selection(summary: FlowDBSummary) -> FlowSampleSelection:
    sample_frac = 1.0
    row_limit = None
    date_start = None
    date_end = None
    use_sample = input("¿Desea cargar solo una muestra de la base? (s/n): ").strip().lower()
    if use_sample in {"s", "si", "y", "yes"}:
        use_range = input("¿Desea seleccionar un rango de fechas? (s/n): ").strip().lower()
        if use_range in {"s", "si", "y", "yes"}:
            range_label = _format_flow_range(summary)
            if range_label:
                print(f"Rango disponible en la base: {range_label}")
            else:
                print("Rango disponible en la base: no informado.")
            print("Si la fecha final no incluye hora, se tomará el día completo.")
            while True:
                start_ts, _ = _prompt_flow_date("fecha de inicio")
                end_ts, end_has_time = _prompt_flow_date("fecha final")
                if not end_has_time:
                    end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                if end_ts <= start_ts:
                    print("La fecha final debe ser posterior a la fecha de inicio.")
                    continue
                date_start = start_ts
                date_end = end_ts
                break
        else:
            while True:
                try:
                    percentage = float(
                        input("Ingrese el porcentaje a cargar (1-100): ").strip()
                    )
                    if not 1 <= percentage <= 100:
                        print("Por favor, ingrese un número entre 1 y 100.")
                        continue
                    sample_frac = percentage / 100.0
                    row_limit = max(1, int(summary.row_count * sample_frac))
                    print(
                        f"\nSe consultarán {row_limit:,} filas "
                        f"({sample_frac * 100:.2f}% de la tabla)."
                    )
                    break
                except ValueError:
                    print("Entrada inválida. Por favor, ingrese un número.")

    return FlowSampleSelection(
        date_start=date_start,
        date_end=date_end,
        row_limit=row_limit,
    )


def load_flujos(
    sample: Optional[FlowSampleSelection] = None,
    progress: Optional[object] = None,
) -> Optional[pd.DataFrame]:
    """
    Carga los flujos vehiculares directamente desde la base DuckDB.
    Si se proporciona sample, se aplican esos filtros sin volver a preguntar.
    progress: barra de progreso opcional (para interfaces visuales).
    """

    summary = ensure_flow_db_summary()
    if summary is None:
        return None

    if sample is None:
        sample = prompt_flow_sample_selection(summary)
    row_limit = sample.row_limit
    date_start = sample.date_start
    date_end = sample.date_end

    def _tick(label: str) -> None:
        if progress is None:
            return
        if hasattr(progress, "set_description"):
            progress.set_description(label)
        if hasattr(progress, "update"):
            progress.update(1)

    if progress is not None:
        _tick("Paso 1/4: Consultando base")

    conn = _connect_duckdb(read_only=True)
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
        query += """
            ORDER BY COALESCE(FECHA, TIMESTAMP '1970-01-01')
        """
        if row_limit is not None:
            query += " LIMIT ?"
            params.append(int(row_limit))
        df = conn.execute(query, params).df()
    finally:
        conn.close()

    if df.empty:
        print("⚠️ La consulta no devolvió filas.")
        return df

    row_count = len(df)
    range_label = None
    if "FECHA" in df.columns:
        range_label = _format_flow_range_from_series(df["FECHA"])
    print(
        f"\n📦 Base de flujos en {summary.db_path} | "
        f"registros: {row_count:,}"
    )
    if range_label:
        print(f"   Rango temporal: {range_label}")

    print("\n🔄 Aplicando transformaciones...")
    if progress is None:
        with tqdm(total=3, desc="Procesando flujos") as pbar:
            pbar.set_description("Paso 1/3: Parseando fechas")
            df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
            df["FECHA"] = df["FECHA"].dt.tz_localize(None)
            pbar.update(1)

            pbar.set_description("Paso 2/3: Normalizando velocidades y categorías")
            df["VELOCIDAD"] = pd.to_numeric(df["VELOCIDAD"], errors="coerce")
            df["CATEGORIA"] = pd.to_numeric(df["CATEGORIA"], errors="coerce")
            df = df.dropna(subset=["CATEGORIA"])
            df["CATEGORIA"] = df["CATEGORIA"].astype("Int64")
            pbar.update(1)

            pbar.set_description("Paso 3/3: Finalizando")
            print("\n📊 Distribución de categorías (sin modificaciones):")
            print(df["CATEGORIA"].value_counts().sort_index())
            pbar.update(1)
    else:
        _tick("Paso 2/4: Parseando fechas")
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
        df["FECHA"] = df["FECHA"].dt.tz_localize(None)

        _tick("Paso 3/4: Normalizando velocidades y categorias")
        df["VELOCIDAD"] = pd.to_numeric(df["VELOCIDAD"], errors="coerce")
        df["CATEGORIA"] = pd.to_numeric(df["CATEGORIA"], errors="coerce")
        df = df.dropna(subset=["CATEGORIA"])
        df["CATEGORIA"] = df["CATEGORIA"].astype("Int64")

        _tick("Paso 4/4: Finalizando")
        print("\n📊 Distribución de categorías (sin modificaciones):")
        print(df["CATEGORIA"].value_counts().sort_index())

    return df.reset_index(drop=True)


def load_flujos_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Carga los flujos vehiculares entre dos timestamps (inicio inclusivo, fin exclusivo).
    """

    conn = _connect_duckdb(read_only=True, db_path=db_path)
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
            WHERE FECHA >= ? AND FECHA < ?
            ORDER BY COALESCE(FECHA, TIMESTAMP '1970-01-01')
        """
        return conn.execute(query, [start, end]).df()
    finally:
        conn.close()


_PORTICOS_DF_CACHE: Optional[pd.DataFrame] = None

DEFAULT_LANES = 3
DEFAULT_INTERVAL_MINUTES = 5
DEFAULT_CATEGORY_REMAP = {1: 1, 2: 2, 3: 2, 4: 3}
DEFAULT_CATEGORY_LABELS = {1: "Light", 2: "Heavy", 3: "Motorcycles"}


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _sanitize_cluster_label(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("-"):
        text = "neg_" + text[1:]
    return _slugify(text)


def buscar_columna(
    df: pd.DataFrame,
    nombre_esperado: str,
    aliases: Optional[Sequence[str]] = None,
) -> str:
    """
    Busca una columna que se parezca a 'nombre_esperado'.
    """
    if nombre_esperado in df.columns:
        return nombre_esperado
    normalizadas = {_normalize_column_name(col): col for col in df.columns}
    candidates = [nombre_esperado] + list(aliases or [])
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalizadas:
            return normalizadas[key]
    available = ", ".join(df.columns.astype(str).tolist())
    raise ValueError(
        f"No se encontro la columna '{nombre_esperado}'. Columnas disponibles: {available}"
    )


def load_porticos_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza un DataFrame de porticos con columnas:
    cod_portico, Km, Calzada, Orden, Eje.
    """
    df = df.copy()
    cod_portico_col = buscar_columna(df, "cod_portico", aliases=["cod portico", "portico"])
    km_col = buscar_columna(df, "Km", aliases=["Km.", "kilometro", "kilometros"])
    calzada_col = buscar_columna(df, "Calzada")
    orden_col = buscar_columna(df, "Orden")
    eje_col = buscar_columna(df, "Eje")

    df[km_col] = df[km_col].astype(str).str.replace(",", ".")
    df[km_col] = pd.to_numeric(df[km_col], errors="coerce")
    df[orden_col] = pd.to_numeric(df[orden_col], errors="coerce")

    optional_map = {
        "aux": ["aux"],
        "autopista": ["autopista"],
        "edge_id_sumo": ["edge_id_sumo"],
        "lane_id_sumo": ["lane_id_sumo"],
        "pos_m": ["pos_m"],
        "lat-lon": ["lat-lon", "lat_lon", "coordenadas"],
        "lat": ["lat", "latitude", "latitud", "y"],
        "lon": ["lon", "longitude", "longitud", "lng", "x"],
    }
    normalized = {_normalize_column_name(col): col for col in df.columns}
    optional_cols: Dict[str, Optional[str]] = {}
    for key, aliases in optional_map.items():
        found = None
        for alias in aliases:
            candidate = normalized.get(_normalize_column_name(alias))
            if candidate:
                found = candidate
                break
        optional_cols[key] = found

    keep_columns = [cod_portico_col, km_col, calzada_col, orden_col, eje_col]
    for col in optional_cols.values():
        if col:
            keep_columns.append(col)

    renamed = {
        cod_portico_col: "portico",
        km_col: "km",
        calzada_col: "calzada",
        orden_col: "orden",
        eje_col: "eje",
    }
    for key, col in optional_cols.items():
        if col:
            renamed[col] = key

    df_porticos = df[keep_columns].rename(columns=renamed)
    if "pos_m" in df_porticos.columns:
        df_porticos["pos_m"] = (
            df_porticos["pos_m"].astype(str).str.replace(",", ".")
        )
        df_porticos["pos_m"] = pd.to_numeric(df_porticos["pos_m"], errors="coerce")
    return df_porticos


def load_porticos(path: Optional[str] = None) -> pd.DataFrame:
    """
    Carga Porticos.csv desde Datos o desde una ruta proporcionada.
    """
    global _PORTICOS_DF_CACHE
    if path is None and _PORTICOS_DF_CACHE is not None:
        return _PORTICOS_DF_CACHE.copy()

    if path is None:
        auto_path = Path("Datos") / "Porticos.csv"
        if not auto_path.exists():
            raise FileNotFoundError(
                "No se encontro Porticos.csv en la carpeta Datos."
            )
        path = str(auto_path)

    df = read_csv_with_progress(path, sep=";")
    porticos_df = load_porticos_from_df(df)
    _PORTICOS_DF_CACHE = porticos_df
    return porticos_df.copy()


def find_candidate_porticos(
    acc_km: object, porticos_df: pd.DataFrame, eje: object, calzada: object
) -> Dict[str, Optional[Dict[str, object]]]:
    """
    Retorna el portico anterior y el mas cercano al km del accidente.
    """
    try:
        accident_km = float(str(acc_km).replace(",", "."))
    except Exception as exc:
        raise ValueError("Valor de km del accidente invalido.") from exc

    df = porticos_df.copy()
    df["Eje_norm"] = df["eje"].astype(str).str.strip().str.upper()
    df["Calzada_norm"] = df["calzada"].astype(str).str.strip().str.upper()
    eje_filtro = str(eje).strip().upper()
    calzada_filtro = str(calzada).strip().upper()
    df_filtrado = df[
        (df["Eje_norm"] == eje_filtro) & (df["Calzada_norm"] == calzada_filtro)
    ].copy()

    if df_filtrado.empty:
        return {"posterior": None, "cercano": None}

    df_filtrado["km_num"] = (
        df_filtrado["km"].astype(str).str.replace(",", ".").astype(float)
    )
    df_filtrado = df_filtrado.sort_values(by="orden", ascending=True).reset_index(
        drop=True
    )

    if len(df_filtrado) >= 2:
        first_km = df_filtrado.loc[0, "km_num"]
        second_km = df_filtrado.loc[1, "km_num"]
        direction = "asc" if first_km <= second_km else "desc"
    else:
        return {"posterior": None, "cercano": df_filtrado.iloc[0].to_dict()}

    posterior_candidate = None
    cercano_candidate = None
    found = False
    for i in range(len(df_filtrado) - 1):
        km_current = df_filtrado.loc[i, "km_num"]
        km_next = df_filtrado.loc[i + 1, "km_num"]
        if direction == "asc":
            if km_current <= accident_km <= km_next:
                posterior_candidate = df_filtrado.loc[i].to_dict()
                cercano_candidate = df_filtrado.loc[i + 1].to_dict()
                found = True
                break
        else:
            if km_current >= accident_km >= km_next:
                posterior_candidate = df_filtrado.loc[i].to_dict()
                cercano_candidate = df_filtrado.loc[i + 1].to_dict()
                found = True
                break

    if not found:
        if direction == "asc":
            if accident_km < df_filtrado.loc[0, "km_num"]:
                posterior_candidate = None
                cercano_candidate = df_filtrado.loc[0].to_dict()
            elif accident_km > df_filtrado.loc[len(df_filtrado) - 1, "km_num"]:
                posterior_candidate = df_filtrado.loc[len(df_filtrado) - 1].to_dict()
                cercano_candidate = None
        else:
            if accident_km > df_filtrado.loc[0, "km_num"]:
                posterior_candidate = None
                cercano_candidate = df_filtrado.loc[0].to_dict()
            elif accident_km < df_filtrado.loc[len(df_filtrado) - 1, "km_num"]:
                posterior_candidate = df_filtrado.loc[len(df_filtrado) - 1].to_dict()
                cercano_candidate = None

    return {"posterior": posterior_candidate, "cercano": cercano_candidate}


def process_accidentes_df(
    df: pd.DataFrame,
    porticos_df: pd.DataFrame,
    *,
    allowed_via: Optional[Sequence[str]] = None,
    return_excluded: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Procesa eventos de accidentes y asigna el ultimo portico.
    """
    if df is None or df.empty:
        empty = pd.DataFrame()
        return (empty, empty) if return_excluded else empty

    df = df.copy()
    tipo_col = buscar_columna(df, "Tipo")
    via_col = buscar_columna(df, "Via")
    allowed_via = [v.lower() for v in (allowed_via or ["expresa", "via expresa"])]
    df = df[
        (df[tipo_col] == "Accidente")
        & (df[via_col].astype(str).str.lower().isin(allowed_via))
    ].copy()

    if df.empty:
        empty = pd.DataFrame()
        return (empty, empty) if return_excluded else empty

    calzada_col = buscar_columna(df, "Calzada")
    df[calzada_col] = (
        df[calzada_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"oriente": "Oriente", "poniente": "Poniente", "ambas": "Ambas"})
    )

    subtipo_col = buscar_columna(df, "SubTipo", aliases=["Sub Tipo", "Sub-Tipo"])
    df[subtipo_col] = df[subtipo_col].astype(str).str.replace(
        r"^(AC-0[1-6] -\s*)", "", regex=True
    )

    fecha_inicio_col = buscar_columna(
        df, "Fechas Inicio", aliases=["Fecha Inicio"]
    )
    hora_inicio_col = buscar_columna(df, "Hora Inicio")
    s_start = (
        df[fecha_inicio_col].astype(str).str.strip()
        + " "
        + df[hora_inicio_col].astype(str).str.strip()
    )
    try:
        dt = pd.to_datetime(s_start, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        dt = pd.to_datetime(s_start, errors="coerce", dayfirst=True)
    try:
        dt = dt.dt.tz_localize(None)
    except TypeError:
        dt = dt.dt.tz_convert(None)
    df["accidente_time"] = dt

    fecha_fin_col = buscar_columna(df, "Fecha Fin")
    hora_fin_col = buscar_columna(df, "Hora Fin")
    s_end = (
        df[fecha_fin_col].astype(str).str.strip()
        + " "
        + df[hora_fin_col].astype(str).str.strip()
    )
    try:
        end_dt = pd.to_datetime(s_end, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        end_dt = pd.to_datetime(s_end, errors="coerce", dayfirst=True)
    try:
        end_dt = end_dt.dt.tz_localize(None)
    except TypeError:
        end_dt = end_dt.dt.tz_convert(None)
    df["duracion_accidente"] = ((end_dt - dt).dt.total_seconds() / 60).round()

    km_col = buscar_columna(df, "Km.", aliases=["Km"])
    df[km_col] = (
        df[km_col].astype(str).str.replace(",", ".").pipe(pd.to_numeric, errors="coerce")
    )

    cols_leve = ["1", "2", "3", "4", "5", "6", "8", "9"]
    cols_grave = ["7"]
    if all(c in df.columns for c in cols_leve) and all(
        c in df.columns for c in cols_grave
    ):
        tmp_leve = df[cols_leve].apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0)
        tmp_grave = df[cols_grave].apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0)
        df["severidad"] = 0
        df.loc[(tmp_grave > 0).any(axis=1), "severidad"] = 1
    else:
        df["severidad"] = "Na"

    eje_col = buscar_columna(df, "Eje")

    def _get_porticos(row: pd.Series) -> pd.Series:
        cand = find_candidate_porticos(
            acc_km=row[km_col],
            porticos_df=porticos_df,
            eje=row[eje_col],
            calzada=row[calzada_col],
        )
        post = cand.get("posterior")
        cerc = cand.get("cercano")
        return pd.Series(
            [
                post["portico"] if post else None,
                cerc["portico"] if cerc else None,
            ]
        )

    df[["ultimo_portico", "proximo_portico"]] = df.apply(
        _get_porticos, axis=1, result_type="expand"
    )
    excluded = df[df["ultimo_portico"].isna()].copy()
    df = df[df["ultimo_portico"].notna()].copy()

    def _clean_portico_str(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s.lower() in ("nan", "none", "null"):
            return None
        if s.endswith(".0"):
            return s[:-2]
        return s

    df["ultimo_portico"] = df["ultimo_portico"].apply(_clean_portico_str)
    df["proximo_portico"] = df["proximo_portico"].apply(_clean_portico_str)

    return (df, excluded) if return_excluded else df


def load_accidentes_from_frames(
    frames: Sequence[pd.DataFrame],
    porticos_df: pd.DataFrame,
    *,
    allowed_via: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Une y procesa multiples archivos de eventos en un solo DataFrame.
    """
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return process_accidentes_df(df, porticos_df, allowed_via=allowed_via)


def load_accidentes(
    file_path: Optional[str] = None,
    *,
    allowed_via: Optional[Sequence[str]] = None,
    return_excluded: bool = False,
) -> Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Carga eventos de accidentes desde CSV y los procesa con los pórticos.
    """
    if file_path is None:
        base_dir = Path("Datos")
        if not base_dir.exists():
            print("No se encontro la carpeta Datos.")
            return None

        patterns = ["eventos*.csv", "accidentes*.csv"]
        candidates = []
        for name in os.listdir(base_dir):
            lower = name.lower()
            if any(fnmatch.fnmatch(lower, pattern) for pattern in patterns):
                candidates.append(base_dir / name)

        if not candidates:
            print("No se encontraron archivos de eventos en la carpeta Datos.")
            return None

        if len(candidates) == 1:
            file_path = str(candidates[0])
            print(f"Archivo de eventos detectado: {file_path}")
        else:
            file_path = choose_file_filtered(str(base_dir), patterns)
            if file_path is None:
                return None

    try:
        raw_df = read_csv_with_progress(file_path, sep=None)
    except Exception as exc:
        print(f"No se pudo leer el archivo de eventos: {exc}")
        return None

    try:
        porticos_df = load_porticos()
    except Exception as exc:
        print(f"No se pudieron cargar los pórticos: {exc}")
        return None

    try:
        return process_accidentes_df(
            raw_df,
            porticos_df,
            allowed_via=allowed_via,
            return_excluded=return_excluded,
        )
    except Exception as exc:
        print(f"No se pudieron procesar los eventos: {exc}")
        return None


def normalize_plate_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.upper()
    invalid_tokens = {"", "NAN", "NULL", "NONE"}
    invalid_mask = series.isna() | cleaned.isin(invalid_tokens)
    return cleaned.where(~invalid_mask, None)


def compute_flow_features(
    flows_df: pd.DataFrame,
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    lanes: int = DEFAULT_LANES,
    category_remap: Optional[Dict[int, int]] = None,
    category_labels: Optional[Dict[int, str]] = None,
    metrics: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    timestamp_col: str = "FECHA",
    speed_col: str = "VELOCIDAD",
    category_col: str = "CATEGORIA",
    portico_col: str = "PORTICO",
    progress: Optional[object] = None,
) -> pd.DataFrame:
    """
    Calcula Flow/Speed/Speed_std/Density por portico, intervalo y tipo de vehiculo.
    """
    if flows_df is None:
        return pd.DataFrame()
    if isinstance(flows_df, pl.LazyFrame):
        flows_df = flows_df.collect()
    if isinstance(flows_df, pd.DataFrame):
        flows_df = pl.from_pandas(flows_df)
    if not isinstance(flows_df, pl.DataFrame):
        raise ValueError("flows_df debe ser DataFrame (pandas o polars).")
    if flows_df.is_empty():
        return pd.DataFrame()
    return _compute_flow_features_polars(
        flows_df,
        interval_minutes=interval_minutes,
        lanes=lanes,
        category_remap=category_remap,
        category_labels=category_labels,
        metrics=metrics,
        categories=categories,
        timestamp_col=timestamp_col,
        speed_col=speed_col,
        category_col=category_col,
        portico_col=portico_col,
    )


def _compute_flow_features_polars(
    flows_df: pl.DataFrame,
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    lanes: int = DEFAULT_LANES,
    category_remap: Optional[Dict[int, int]] = None,
    category_labels: Optional[Dict[int, str]] = None,
    metrics: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    timestamp_col: str = "FECHA",
    speed_col: str = "VELOCIDAD",
    category_col: str = "CATEGORIA",
    portico_col: str = "PORTICO",
) -> pd.DataFrame:
    if flows_df is None or flows_df.is_empty():
        return pd.DataFrame()
    required = {timestamp_col, speed_col, category_col, portico_col}
    missing = [col for col in required if col not in flows_df.columns]
    if missing:
        raise ValueError(
            f"No se encontraron columnas requeridas: {sorted(missing)}"
        )

    def _time_expr() -> pl.Expr:
        dtype = flows_df.schema.get(timestamp_col)
        if dtype == pl.Utf8:
            return pl.col(timestamp_col).str.to_datetime(
                time_unit="ns", strict=False
            )
        return pl.col(timestamp_col).cast(pl.Datetime, strict=False)

    work = (
        flows_df.select([timestamp_col, speed_col, category_col, portico_col])
        .with_columns(
            [
                _time_expr().alias(timestamp_col),
                pl.col(speed_col).cast(pl.Float64, strict=False).alias(
                    speed_col
                ),
                pl.col(category_col)
                .cast(pl.Int64, strict=False)
                .alias(category_col),
                pl.col(portico_col)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .alias(portico_col),
            ]
        )
        .drop_nulls([timestamp_col, speed_col, category_col, portico_col])
    )
    if work.is_empty():
        return pd.DataFrame()

    remap = category_remap or DEFAULT_CATEGORY_REMAP
    labels = category_labels or DEFAULT_CATEGORY_LABELS

    work = work.with_columns(
        pl.col(category_col)
        .replace_strict(remap, default=None)
        .alias("category_id")
    ).with_columns(
        pl.col("category_id")
        .replace_strict(labels, default=None)
        .alias("category_label")
    )
    work = work.filter(pl.col("category_label").is_not_null())
    if categories:
        work = work.filter(pl.col("category_label").is_in(list(categories)))
    if work.is_empty():
        return pd.DataFrame()

    work = work.with_columns(
        pl.col(timestamp_col)
        .dt.truncate(f"{int(interval_minutes)}m")
        .alias("interval_start")
    )

    grouped = work.group_by(
        [portico_col, "interval_start", "category_label"]
    ).agg(
        [
            pl.len().alias("flow_count"),
            pl.col(speed_col).mean().alias("speed_mean"),
            pl.col(speed_col).std().alias("speed_std"),
        ]
    )
    if grouped.is_empty():
        return pd.DataFrame()

    interval_factor = 60.0 / max(1, interval_minutes)
    lanes_value = max(1, lanes)

    grouped = grouped.with_columns(
        [
            (
                pl.col("flow_count") * interval_factor / lanes_value
            ).alias("flow_per_hour"),
            pl.when(pl.col("speed_mean") <= 0)
            .then(0.0)
            .otherwise(
                (pl.col("flow_count") * interval_factor / lanes_value)
                / pl.col("speed_mean")
            )
            .alias("density"),
            pl.col("speed_std").fill_null(0.0).alias("speed_std"),
        ]
    ).with_columns(
        pl.col("density").fill_null(0.0).alias("density")
    )

    grouped = grouped.sort([portico_col, "category_label", "interval_start"])
    grouped = grouped.with_columns(
        [
            pl.col("speed_mean")
            .diff()
            .over([portico_col, "category_label"])
            .fill_null(0.0)
            .alias("delta_speed"),
            pl.col("density")
            .diff()
            .over([portico_col, "category_label"])
            .fill_null(0.0)
            .alias("delta_density"),
        ]
    )

    metric_map = {
        "flow": "flow_per_hour",
        "speed": "speed_mean",
        "speed_std": "speed_std",
        "density": "density",
        "delta_speed": "delta_speed",
        "delta_density": "delta_density",
    }
    metrics = metrics or ["flow", "speed", "density"]
    frames = []
    for metric in metrics:
        value_col = metric_map.get(metric)
        if value_col is None:
            continue
        pivoted = (
            grouped.pivot(
                index=[portico_col, "interval_start"],
                on="category_label",
                values=value_col,
                aggregate_function="first",
            )
            .fill_null(0.0)
        )
        rename_map = {}
        for col in pivoted.columns:
            if col in (portico_col, "interval_start"):
                continue
            rename_map[col] = f"{metric}_{_slugify(col)}"
        if rename_map:
            pivoted = pivoted.rename(rename_map)
        frames.append(pivoted)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(
            frame, on=[portico_col, "interval_start"], how="left"
        )

    result = result.rename({portico_col: "portico"})
    return result.to_pandas()


def compute_cluster_features(
    flows_df: pd.DataFrame,
    cluster_labels_df: pd.DataFrame,
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    include_counts: bool = False,
    include_entropy: bool = False,
    include_speed: bool = False,
    include_density: bool = False,
    include_delta_speed: bool = False,
    include_delta_density: bool = False,
    lanes: int = DEFAULT_LANES,
    timestamp_col: str = "FECHA",
    portico_col: str = "PORTICO",
    plate_col_flow: str = "MATRICULA",
    plate_col_cluster: str = "plate",
    cluster_col: str = "cluster_label",
    speed_col: str = "VELOCIDAD",
) -> pd.DataFrame:
    """
    Calcula proporciones de clusters por portico e intervalo, y Flow/Speed/Density/Delta.
    """
    def _is_empty_df(obj: object) -> bool:
        if obj is None:
            return True
        if isinstance(obj, pl.LazyFrame):
            return obj.collect().is_empty()
        if isinstance(obj, pl.DataFrame):
            return obj.is_empty()
        if hasattr(obj, "empty"):
            return bool(obj.empty)
        return False

    if isinstance(flows_df, pl.LazyFrame):
        flows_df = flows_df.collect()
    if isinstance(cluster_labels_df, pl.LazyFrame):
        cluster_labels_df = cluster_labels_df.collect()

    if isinstance(flows_df, pl.DataFrame):
        return _compute_cluster_features_polars(
            flows_df,
            cluster_labels_df,
            interval_minutes=interval_minutes,
            include_counts=include_counts,
            include_entropy=include_entropy,
            include_speed=include_speed,
            include_density=include_density,
            include_delta_speed=include_delta_speed,
            include_delta_density=include_delta_density,
            lanes=lanes,
            timestamp_col=timestamp_col,
            portico_col=portico_col,
            plate_col_flow=plate_col_flow,
            plate_col_cluster=plate_col_cluster,
            cluster_col=cluster_col,
            speed_col=speed_col,
        )
    if _is_empty_df(flows_df):
        return pd.DataFrame()
    if _is_empty_df(cluster_labels_df):
        return pd.DataFrame()
    if cluster_col not in cluster_labels_df.columns:
        raise ValueError("El archivo de clusters no contiene 'cluster_label'.")
    if plate_col_cluster not in cluster_labels_df.columns:
        raise ValueError("El archivo de clusters no contiene la columna de placas.")

    need_speed = (
        include_speed
        or include_density
        or include_delta_speed
        or include_delta_density
    )
    if need_speed and speed_col not in flows_df.columns:
        raise ValueError(f"El archivo de flujos no contiene '{speed_col}'.")
    flow_cols = [timestamp_col, portico_col, plate_col_flow]
    if need_speed:
        flow_cols.append(speed_col)
    df = flows_df[flow_cols].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[portico_col] = df[portico_col].astype(str).str.strip()
    if need_speed:
        df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce")
    df["plate_clean"] = normalize_plate_series(df[plate_col_flow])
    df = df.dropna(subset=[timestamp_col, portico_col, "plate_clean"])
    if df.empty:
        return pd.DataFrame()

    clusters = cluster_labels_df[[plate_col_cluster, cluster_col]].copy()
    clusters["plate_clean"] = normalize_plate_series(clusters[plate_col_cluster])
    clusters = clusters.dropna(subset=["plate_clean"])
    clusters = clusters.drop_duplicates(subset=["plate_clean"])
    if clusters.empty:
        return pd.DataFrame()

    merged = df.merge(
        clusters[["plate_clean", cluster_col]],
        on="plate_clean",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["interval_start"] = merged[timestamp_col].dt.floor(
        f"{interval_minutes}min"
    )
    grouped = merged.groupby(
        [portico_col, "interval_start", cluster_col]
    ).agg(count=("plate_clean", "size"))
    if need_speed:
        grouped["speed_mean"] = merged.groupby(
            [portico_col, "interval_start", cluster_col]
        )[speed_col].mean()
    grouped = grouped.reset_index()
    if grouped.empty:
        return pd.DataFrame()

    grouped["total"] = grouped.groupby([portico_col, "interval_start"])[
        "count"
    ].transform("sum")
    grouped["share"] = grouped["count"] / grouped["total"].replace(0, 1)
    interval_factor = 60.0 / max(1, interval_minutes)
    lanes_value = max(1, lanes)
    grouped["flow_per_hour"] = grouped["count"] * interval_factor / lanes_value
    need_density = include_density or include_delta_density
    if need_density:
        grouped["density"] = grouped["flow_per_hour"] / grouped["speed_mean"]
        grouped.loc[grouped["speed_mean"] <= 0, "density"] = 0
        grouped["density"] = grouped["density"].fillna(0)
    if include_delta_speed or include_delta_density:
        grouped = grouped.sort_values(
            [portico_col, cluster_col, "interval_start"]
        )
    if include_delta_speed:
        grouped["delta_speed"] = (
            grouped.groupby([portico_col, cluster_col])["speed_mean"]
            .diff()
            .fillna(0)
        )
    if include_delta_density:
        grouped["delta_density"] = (
            grouped.groupby([portico_col, cluster_col])["density"]
            .diff()
            .fillna(0)
        )

    share_pivot = grouped.pivot_table(
        index=[portico_col, "interval_start"],
        columns=cluster_col,
        values="share",
        fill_value=0,
    )
    share_pivot.columns = [
        f"cluster_share_{_sanitize_cluster_label(label)}"
        for label in share_pivot.columns
    ]
    frames = [share_pivot]

    entropy_series = None
    if include_entropy:
        share_values = share_pivot.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = np.where(share_values > 0, np.log(share_values), 0.0)
        entropy_values = -np.sum(share_values * log_values, axis=1)
        entropy_series = pd.Series(
            entropy_values, index=share_pivot.index, name="cluster_entropy"
        )

    metric_map = {
        "flow": ("flow_per_hour", "cluster_flow_"),
        "speed": ("speed_mean", "cluster_speed_"),
        "density": ("density", "cluster_density_"),
        "delta_speed": ("delta_speed", "cluster_delta_speed_"),
        "delta_density": ("delta_density", "cluster_delta_density_"),
    }
    if include_counts:
        value_col, prefix = metric_map["flow"]
        flow_pivot = grouped.pivot_table(
            index=[portico_col, "interval_start"],
            columns=cluster_col,
            values=value_col,
            fill_value=0,
        )
        flow_pivot.columns = [
            f"{prefix}{_sanitize_cluster_label(label)}"
            for label in flow_pivot.columns
        ]
        frames.append(flow_pivot)
    if include_speed:
        value_col, prefix = metric_map["speed"]
        speed_pivot = grouped.pivot_table(
            index=[portico_col, "interval_start"],
            columns=cluster_col,
            values=value_col,
            fill_value=0,
        )
        speed_pivot.columns = [
            f"{prefix}{_sanitize_cluster_label(label)}"
            for label in speed_pivot.columns
        ]
        frames.append(speed_pivot)
    if include_density:
        value_col, prefix = metric_map["density"]
        density_pivot = grouped.pivot_table(
            index=[portico_col, "interval_start"],
            columns=cluster_col,
            values=value_col,
            fill_value=0,
        )
        density_pivot.columns = [
            f"{prefix}{_sanitize_cluster_label(label)}"
            for label in density_pivot.columns
        ]
        frames.append(density_pivot)
    if include_delta_speed:
        value_col, prefix = metric_map["delta_speed"]
        delta_speed_pivot = grouped.pivot_table(
            index=[portico_col, "interval_start"],
            columns=cluster_col,
            values=value_col,
            fill_value=0,
        )
        delta_speed_pivot.columns = [
            f"{prefix}{_sanitize_cluster_label(label)}"
            for label in delta_speed_pivot.columns
        ]
        frames.append(delta_speed_pivot)
    if include_delta_density:
        value_col, prefix = metric_map["delta_density"]
        delta_density_pivot = grouped.pivot_table(
            index=[portico_col, "interval_start"],
            columns=cluster_col,
            values=value_col,
            fill_value=0,
        )
        delta_density_pivot.columns = [
            f"{prefix}{_sanitize_cluster_label(label)}"
            for label in delta_density_pivot.columns
        ]
        frames.append(delta_density_pivot)

    if include_entropy and entropy_series is not None:
        frames.append(entropy_series)

    result = pd.concat(frames, axis=1).reset_index()
    return result.rename(columns={portico_col: "portico"})


def _compute_cluster_features_polars(
    flows_df: pl.DataFrame,
    cluster_labels_df: Union[pd.DataFrame, pl.DataFrame],
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    include_counts: bool = False,
    include_entropy: bool = False,
    include_speed: bool = False,
    include_density: bool = False,
    include_delta_speed: bool = False,
    include_delta_density: bool = False,
    lanes: int = DEFAULT_LANES,
    timestamp_col: str = "FECHA",
    portico_col: str = "PORTICO",
    plate_col_flow: str = "MATRICULA",
    plate_col_cluster: str = "plate",
    cluster_col: str = "cluster_label",
    speed_col: str = "VELOCIDAD",
) -> pd.DataFrame:
    if flows_df is None or flows_df.is_empty():
        return pd.DataFrame()
    if isinstance(cluster_labels_df, pd.DataFrame):
        if cluster_labels_df.empty:
            return pd.DataFrame()
        labels = pl.from_pandas(cluster_labels_df)
    else:
        if cluster_labels_df.is_empty():
            return pd.DataFrame()
        labels = cluster_labels_df
    if cluster_col not in labels.columns:
        raise ValueError("El archivo de clusters no contiene 'cluster_label'.")
    if plate_col_cluster not in labels.columns:
        raise ValueError("El archivo de clusters no contiene la columna de placas.")

    need_speed = (
        include_speed
        or include_density
        or include_delta_speed
        or include_delta_density
    )
    if need_speed and speed_col not in flows_df.columns:
        raise ValueError(f"El archivo de flujos no contiene '{speed_col}'.")

    if timestamp_col not in flows_df.columns:
        raise ValueError(f"El archivo de flujos no contiene '{timestamp_col}'.")
    if portico_col not in flows_df.columns:
        raise ValueError(f"El archivo de flujos no contiene '{portico_col}'.")
    if plate_col_flow not in flows_df.columns:
        raise ValueError(f"El archivo de flujos no contiene '{plate_col_flow}'.")

    def _normalize_plate_expr(col_name: str) -> pl.Expr:
        cleaned = (
            pl.col(col_name)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.to_uppercase()
        )
        invalid_tokens = ["", "NAN", "NULL", "NONE"]
        return (
            pl.when(cleaned.is_null() | cleaned.is_in(invalid_tokens))
            .then(None)
            .otherwise(cleaned)
        )

    time_dtype = flows_df.schema.get(timestamp_col)
    if time_dtype == pl.Utf8:
        time_expr = pl.col(timestamp_col).str.to_datetime(
            time_unit="ns", strict=False
        )
    else:
        time_expr = pl.col(timestamp_col).cast(pl.Datetime, strict=False)

    flow_cols = [timestamp_col, portico_col, plate_col_flow]
    if need_speed:
        flow_cols.append(speed_col)

        work = flows_df.select(flow_cols).with_columns(
            [
                time_expr.alias(timestamp_col),
                pl.col(portico_col)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .alias(portico_col),
                _normalize_plate_expr(plate_col_flow).alias("plate_clean"),
            ]
        )
    if need_speed:
        work = work.with_columns(
            pl.col(speed_col).cast(pl.Float64, strict=False)
        )

    drop_cols = [timestamp_col, portico_col, "plate_clean"]
    work = work.drop_nulls(drop_cols)
    if work.is_empty():
        return pd.DataFrame()

    labels = labels.select([plate_col_cluster, cluster_col]).with_columns(
        _normalize_plate_expr(plate_col_cluster).alias("plate_clean")
    )
    labels = labels.drop_nulls([plate_col_cluster, cluster_col, "plate_clean"])
    labels = labels.unique(subset=["plate_clean"])
    if labels.is_empty():
        return pd.DataFrame()

    merged = work.join(
        labels.select(["plate_clean", cluster_col]),
        on="plate_clean",
        how="inner",
    )
    if merged.is_empty():
        return pd.DataFrame()

    merged = merged.with_columns(
        pl.col(timestamp_col)
        .dt.truncate(f"{int(interval_minutes)}m")
        .alias("interval_start")
    )

    agg_exprs = [pl.len().alias("count")]
    if need_speed:
        agg_exprs.append(pl.col(speed_col).mean().alias("speed_mean"))
    grouped = merged.group_by(
        [portico_col, "interval_start", cluster_col]
    ).agg(agg_exprs)
    if grouped.is_empty():
        return pd.DataFrame()

    interval_factor = 60.0 / max(1, interval_minutes)
    lanes_value = max(1, lanes)
    total_expr = pl.col("count").sum().over([portico_col, "interval_start"])
    grouped = grouped.with_columns(
        [
            total_expr.alias("total"),
            pl.when(total_expr == 0)
            .then(0.0)
            .otherwise(pl.col("count") / total_expr)
            .alias("share"),
            (pl.col("count") * interval_factor / lanes_value).alias(
                "flow_per_hour"
            ),
        ]
    )

    need_density = include_density or include_delta_density
    if need_density:
        grouped = grouped.with_columns(
            pl.when(
                pl.col("speed_mean").is_null() | (pl.col("speed_mean") <= 0)
            )
            .then(0.0)
            .otherwise(pl.col("flow_per_hour") / pl.col("speed_mean"))
            .alias("density")
        )

    if include_delta_speed or include_delta_density:
        grouped = grouped.sort(
            [portico_col, cluster_col, "interval_start"]
        )
        if include_delta_speed:
            grouped = grouped.with_columns(
                pl.col("speed_mean")
                .diff()
                .over([portico_col, cluster_col])
                .fill_null(0.0)
                .alias("delta_speed")
            )
        if include_delta_density:
            grouped = grouped.with_columns(
                pl.col("density")
                .diff()
                .over([portico_col, cluster_col])
                .fill_null(0.0)
                .alias("delta_density")
            )

    def _pivot_metric(value_col: str, prefix: str) -> pl.DataFrame:
        pivoted = grouped.pivot(
            index=[portico_col, "interval_start"],
            on=cluster_col,
            values=value_col,
            aggregate_function="first",
        ).fill_null(0.0)
        rename_map = {}
        for col in pivoted.columns:
            if col in (portico_col, "interval_start"):
                continue
            rename_map[col] = f"{prefix}{_sanitize_cluster_label(col)}"
        if rename_map:
            pivoted = pivoted.rename(rename_map)
        return pivoted

    share_pivot = _pivot_metric("share", "cluster_share_")
    frames = [share_pivot]

    if include_counts:
        frames.append(_pivot_metric("flow_per_hour", "cluster_flow_"))
    if include_speed:
        frames.append(_pivot_metric("speed_mean", "cluster_speed_"))
    if include_density:
        frames.append(_pivot_metric("density", "cluster_density_"))
    if include_delta_speed:
        frames.append(_pivot_metric("delta_speed", "cluster_delta_speed_"))
    if include_delta_density:
        frames.append(_pivot_metric("delta_density", "cluster_delta_density_"))

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(
            frame, on=[portico_col, "interval_start"], how="left"
        )

    if include_entropy:
        share_cols = [
            col
            for col in share_pivot.columns
            if col not in (portico_col, "interval_start")
        ]
        if share_cols:
            entropy_terms = [
                pl.when(pl.col(col) > 0)
                .then(pl.col(col) * pl.col(col).log())
                .otherwise(0.0)
                for col in share_cols
            ]
            result = result.with_columns(
                (-pl.sum_horizontal(entropy_terms)).alias("cluster_entropy")
            )

    metric_cols = [
        col for col in result.columns if col not in (portico_col, "interval_start")
    ]
    if metric_cols:
        result = result.with_columns(
            [pl.col(col).fill_null(0.0) for col in metric_cols]
        )

    result = result.rename({portico_col: "portico"})
    return result.to_pandas()


def add_accident_target(
    features_df: pd.DataFrame,
    accidents_df: pd.DataFrame,
    *,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    portico_col: str = "portico",
    interval_col: str = "interval_start",
    accident_time_col: str = "accidente_time",
    accident_portico_col: str = "ultimo_portico",
) -> pd.DataFrame:
    """
    Agrega columna target (1 si hay accidente en el portico/intervalo).
    """
    if features_df is None or features_df.empty:
        return pd.DataFrame()
    if accidents_df is None or accidents_df.empty:
        df = features_df.copy()
        df["target"] = 0
        return df

    df = features_df.copy()
    if interval_col in df.columns:
        df[interval_col] = pd.to_datetime(df[interval_col], errors="coerce")
    
    # Auto-detect portico column if default is not present
    if portico_col not in df.columns and "portico_last" in df.columns:
        portico_col = "portico_last"

    acc = accidents_df[[accident_portico_col, accident_time_col]].copy()
    acc[accident_time_col] = pd.to_datetime(acc[accident_time_col], errors="coerce")
    acc = acc.dropna(subset=[accident_portico_col, accident_time_col])
    if acc.empty:
        df["target"] = 0
        return df

    acc["interval_start"] = acc[accident_time_col].dt.floor(
        f"{interval_minutes}min"
    ) - pd.Timedelta(minutes=interval_minutes)
    acc[accident_portico_col] = acc[accident_portico_col].astype(str).str.strip()
    acc_pairs = acc[[accident_portico_col, "interval_start"]].drop_duplicates()
    acc_pairs = acc_pairs.rename(columns={accident_portico_col: portico_col})
    acc_pairs["target"] = 1

    df[portico_col] = df[portico_col].astype(str).str.strip()
    merged = df.merge(
        acc_pairs, how="left", on=[portico_col, "interval_start"]
    )
    merged["target"] = merged["target"].fillna(0).astype(int)
    return merged


def get_portico_segments(porticos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un DataFrame con tramos: P_last -> P_next.
    Columnas: eje, calzada, portico_last, km_last, portico_next, km_next.
    """
    if porticos_df is None or porticos_df.empty:
        return pd.DataFrame()

    df = porticos_df.copy()
    df["orden_num"] = pd.to_numeric(df["orden"], errors="coerce")
    df["km_num"] = pd.to_numeric(df["km"], errors="coerce")
    df["eje_norm"] = df["eje"].astype(str).str.strip().str.upper()
    df["calzada_norm"] = df["calzada"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["orden_num", "km_num", "eje_norm", "calzada_norm"])

    segments: List[Dict[str, object]] = []
    
    def _clean(val):
        s = str(val).strip()
        return s[:-2] if s.endswith(".0") else s

    for (_, _), group in df.groupby(["eje_norm", "calzada_norm"]):
        group = group.sort_values("orden_num")
        # Iterates on (i, i+1) pairs
        for i in range(len(group) - 1):
            start = group.iloc[i]
            end = group.iloc[i + 1]
            segments.append(
                {
                    "eje": start["eje"],
                    "calzada": start["calzada"],
                    "portico_last": _clean(start["portico"]),
                    "km_last": float(start["km_num"]),
                    "portico_next": _clean(end["portico"]),
                    "km_next": float(end["km_num"]),
                }
            )
    return pd.DataFrame(segments)


def limpiar_pantalla() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _eje_family(eje: str) -> str:
    """Normaliza el nombre del eje para agrupar NORTE/SUR en la misma familia."""
    eje = str(eje or "").strip()
    eje = re.sub(r"\b(SUR|NORTE)\b", "", eje, flags=re.IGNORECASE)
    eje = re.sub(r"\s+", " ", eje).strip()
    return eje.upper()


def normalize_portico_code(value: object) -> str:
    """Normaliza el codigo de portico para comparaciones estables."""
    try:
        num = float(str(value).strip())
        if abs(num - int(num)) < 1e-9:
            return str(int(num))
        return str(num)
    except Exception:
        return str(value).strip()


def reconstruct_portico_sequence(
    df_porticos: pd.DataFrame,
    p1: object,
    p2: object,
    *,
    max_anterior: int = 4,
    max_posterior: int = 0,
    prefer_family: bool = True,
    deduplicate_slice: bool = True,
) -> list:
    """
    Reconstruye la lista de porticos alrededor del par consecutivo (p1 -> p2).
    """
    required_cols = {"portico", "eje", "calzada", "orden"}
    if not required_cols.issubset(df_porticos.columns):
        raise ValueError(
            f"df_porticos debe contener columnas {sorted(required_cols)}"
        )

    p1n = normalize_portico_code(p1)
    p2n = normalize_portico_code(p2)

    df_sorted = df_porticos.sort_values(by=["calzada", "eje", "orden"])

    pair_ctx = None
    for (calzada, eje), group in df_sorted.groupby(["calzada", "eje"]):
        seq_raw = group["portico"].tolist()
        seq_norm = [normalize_portico_code(x) for x in seq_raw]
        for idx in range(len(seq_norm) - 1):
            if seq_norm[idx] == p1n and seq_norm[idx + 1] == p2n:
                pair_ctx = (calzada, eje, idx, seq_raw)
                break
        if pair_ctx is not None:
            break

    if pair_ctx is None:
        for (calzada, eje), group in df_sorted.groupby(["calzada", "eje"]):
            seq_raw = group["portico"].tolist()
            seq_norm = [normalize_portico_code(x) for x in seq_raw]
            if p1n in seq_norm:
                pair_ctx = (calzada, eje, seq_norm.index(p1n), seq_raw)
                break

    if pair_ctx is None:
        raise ValueError(
            "No se encontro p1 en las secuencias de porticos cargadas"
        )

    sel_calzada, sel_eje, i_local, seq_local_raw = pair_ctx

    if prefer_family:
        df_tmp = df_porticos.copy()
        df_tmp["eje_fam"] = df_tmp["eje"].apply(_eje_family)
        fam_key = (sel_calzada, _eje_family(sel_eje))
        fam_groups = df_tmp.sort_values(
            ["calzada", "eje_fam", "orden"]
        ).groupby(["calzada", "eje_fam"])
        if fam_key in fam_groups.groups:
            seq_fam_raw = fam_groups.get_group(fam_key)["portico"].tolist()
            seq_fam_norm = [normalize_portico_code(x) for x in seq_fam_raw]
            match_idx = None
            for idx in range(len(seq_fam_norm) - 1):
                if seq_fam_norm[idx] == p1n and seq_fam_norm[idx + 1] == p2n:
                    match_idx = idx
            if match_idx is not None:
                start_idx = max(0, match_idx - max_anterior)
                end_idx = min(
                    len(seq_fam_raw), (match_idx + 1) + max_posterior + 1
                )
                slice_seq = seq_fam_raw[start_idx:end_idx]
                if deduplicate_slice:
                    seen = set()
                    out = []
                    for item in reversed(slice_seq):
                        if item not in seen:
                            out.append(item)
                            seen.add(item)
                    return list(reversed(out))
                return slice_seq

    start_idx = max(0, i_local - max_anterior)
    end_idx = min(len(seq_local_raw), (i_local + 1) + max_posterior + 1)
    return seq_local_raw[start_idx:end_idx]


def seleccionar_tramo_y_porticos(
    df_porticos: pd.DataFrame,
    max_anterior: int = 4,
    max_posterior: int = 0,
) -> Optional[list]:
    """
    Permite seleccionar un tramo de forma interactiva y devuelve la lista
    de porticos asociados. Usa input en consola.
    """
    if df_porticos is None or df_porticos.empty:
        print("El DataFrame de porticos esta vacio.")
        return None

    df_sorted = df_porticos.sort_values(by=["calzada", "eje", "orden"])
    tramos = []
    secuencias = {}

    for (calzada, eje), group in df_sorted.groupby(["calzada", "eje"]):
        porticos = group["portico"].tolist()
        kms = group["km"].tolist()
        secuencias[(calzada, eje)] = porticos
        for idx in range(len(porticos) - 1):
            tramos.append(
                {
                    "calzada": calzada,
                    "eje": eje,
                    "tramo_str": (
                        f"Calzada {calzada} - Eje {eje}: "
                        f"{porticos[idx]} (km {kms[idx]}) -> "
                        f"{porticos[idx + 1]} (km {kms[idx + 1]})"
                    ),
                    "p1_idx": idx,
                }
            )

    if not tramos:
        print("No se pudieron generar tramos a partir de los porticos.")
        return None

    print("Seleccione el tramo de interes:")
    for idx, tramo in enumerate(tramos):
        print(f"  [{idx}] {tramo['tramo_str']}")

    choice_str = input(
        "\nIngrese el numero del tramo (Enter para cancelar): "
    ).strip()
    if not choice_str:
        print("Operacion cancelada.")
        return None
    try:
        choice = int(choice_str)
        if not (0 <= choice < len(tramos)):
            raise ValueError
    except ValueError:
        print("Seleccion invalida.")
        return None

    tramo_sel = tramos[choice]
    calzada = tramo_sel["calzada"]
    eje = tramo_sel["eje"]
    p1_idx = tramo_sel["p1_idx"]
    p1 = secuencias[(calzada, eje)][p1_idx]
    p2 = secuencias[(calzada, eje)][p1_idx + 1]

    return reconstruct_portico_sequence(
        df_porticos=df_porticos,
        p1=p1,
        p2=p2,
        max_anterior=max_anterior,
        max_posterior=max_posterior,
        prefer_family=True,
        deduplicate_slice=True,
    )
