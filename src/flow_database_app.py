#!/usr/bin/env python3
"""
Streamlit app to manage the flow database (DuckDB).
"""
from __future__ import annotations

import fnmatch
import importlib
import inspect
import json
import os
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import polars as pl
import streamlit as st

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import utils as utils_mod  # noqa: E402
from utils import (  # noqa: E402
    clear_flow_table,
    get_flow_basic_stats,
    get_flow_db_summary,
    get_flow_counts_by_category,
    get_flow_counts_by_month,
    get_flow_counts_by_year,
    get_flow_days_by_year,
    get_flow_missingness_stats,
    get_flow_plate_concentration,
    get_flow_speed_stats,
    get_flow_speed_stats_by_category,
    get_flow_table_columns,
    get_flow_table_sample,
    get_flow_window_prevalence_stats,
    get_flow_window_stats,
    import_flujos_to_duckdb,
    load_accidentes,
    buscar_columna,
    FLOW_TABLE_NAME,
)
from flow_stats_legacy import (  # noqa: E402
    build_weekday_flow_lines,
    generate_table_a_polars,
    process_monthly_and_store,
    process_monthly_and_store_duckdb,
    _compute_surface_duckdb,
    _compute_surface_polars,
    _plot_surface_by_category_interactive,
    _plot_weekday_flow_lines_interactive,
)

DATA_DIR = ROOT_DIR / "Datos"
DB_PATH = DATA_DIR / "flujos.duckdb"
FLOW_PATTERNS = ("flujo*.csv",)
EVENT_PATTERNS = ("eventos*.csv", "accidentes*.csv")
CATEGORY_LABELS = {1: "Light", 2: "Heavy", 3: "Motorcycles"}
CACHE_PATH = ROOT_DIR / "Resultados" / "flow_data_description_cache.json"


def _get_delete_flow_rows_fn():
    fn = getattr(utils_mod, "delete_flow_rows_by_date_range", None)
    if fn is None:
        try:
            importlib.reload(utils_mod)
        except Exception:
            return None
        fn = getattr(utils_mod, "delete_flow_rows_by_date_range", None)
    return fn


def _supports_progress() -> bool:
    try:
        sig = inspect.signature(import_flujos_to_duckdb)
    except (ValueError, TypeError):
        return False
    return "progress_callback" in sig.parameters


def _run_import(
    *,
    csv_path: Path,
    replace: bool,
    progress_callback: Optional[object],
) -> int:
    kwargs = {
        "csv_path": str(csv_path),
        "db_path": DB_PATH,
        "replace": replace,
    }
    if progress_callback is not None and _supports_progress():
        kwargs["progress_callback"] = progress_callback
    try:
        return import_flujos_to_duckdb(**kwargs)
    except TypeError as exc:
        if "progress_callback" in str(exc):
            kwargs.pop("progress_callback", None)
            return import_flujos_to_duckdb(**kwargs)
        raise


def _format_ts(value) -> str:
    if value is None:
        return "-"
    try:
        return value.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)


def _format_bytes(value: Optional[float]) -> str:
    if value is None:
        return "-"
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    return f"{size:,.2f} TB"


def _get_ram_usage() -> str:
    if psutil is not None:
        vm = psutil.virtual_memory()
        return f"RAM { _format_bytes(vm.used) } / { _format_bytes(vm.total) } ({vm.percent:.1f}%)"
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_bytes = float(rss)
        else:
            rss_bytes = float(rss) * 1024.0
        total = None
        if hasattr(os, "sysconf"):
            try:
                total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            except Exception:
                total = None
        if total:
            percent = rss_bytes / total * 100.0
            return f"RAM { _format_bytes(rss_bytes) } / { _format_bytes(total) } ({percent:.1f}%)"
        return f"RAM { _format_bytes(rss_bytes) } (proceso)"
    except Exception:
        return "RAM -"


class _ProgressTracker:
    def __init__(self) -> None:
        self.total = 0
        self.done: set[str] = set()
        self.registered: set[str] = set()
        self._bar = st.progress(0.0)
        self._text = st.empty()

    def register(self, key: str) -> None:
        if key in self.registered:
            return
        self.registered.add(key)
        self.total += 1
        self._update("Preparando...")

    def mark_done(self, key: str, label: str) -> None:
        if key not in self.registered:
            self.register(key)
        if key in self.done:
            return
        self.done.add(key)
        self._update(label)

    def _update(self, label: str) -> None:
        total = max(self.total, 1)
        ratio = min(len(self.done) / total, 1.0)
        self._bar.progress(ratio)
        self._text.caption(f"{label} | {_get_ram_usage()}")


def _pl_read_flow(columns: list[str]) -> pl.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb no está instalado.")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cols_sql = ", ".join(columns)
        query = f"SELECT {cols_sql} FROM {FLOW_TABLE_NAME}"
        return pl.read_database(query, conn)
    finally:
        conn.close()


def _pl_clean_string(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).str.strip_chars()


def _pl_flow_basic_stats() -> Dict[str, object]:
    df = _pl_read_flow(["FECHA", "PORTICO", "CARRIL", "CATEGORIA", "MATRICULA"])
    if df.is_empty():
        return {"total": 0}
    df = df.with_columns(
        [
            _pl_clean_string(pl.col("PORTICO")).alias("portico_clean"),
            _pl_clean_string(pl.col("CARRIL")).alias("carril_clean"),
            _pl_clean_string(pl.col("MATRICULA")).str.to_uppercase().alias(
                "matricula_clean"
            ),
        ]
    )
    total = df.height
    min_ts = df.select(pl.col("FECHA").min()).item()
    max_ts = df.select(pl.col("FECHA").max()).item()

    porticos = (
        df.filter(pl.col("portico_clean").is_not_null() & (pl.col("portico_clean") != ""))
        .select(pl.col("portico_clean").n_unique())
        .item()
    )
    carriles = (
        df.filter(pl.col("carril_clean").is_not_null() & (pl.col("carril_clean") != ""))
        .select(pl.col("carril_clean").n_unique())
        .item()
    )
    categorias = (
        df.filter(pl.col("CATEGORIA").is_not_null())
        .select(pl.col("CATEGORIA").n_unique())
        .item()
    )
    patentes = (
        df.filter(
            pl.col("matricula_clean").is_not_null()
            & (pl.col("matricula_clean") != "")
        )
        .select(pl.col("matricula_clean").n_unique())
        .item()
    )
    dias = (
        df.filter(pl.col("FECHA").is_not_null())
        .select(pl.col("FECHA").dt.date().n_unique())
        .item()
    )
    horas = (
        df.filter(pl.col("FECHA").is_not_null())
        .select(pl.col("FECHA").dt.truncate("1h").n_unique())
        .item()
    )
    return {
        "total": int(total),
        "min_ts": pd.to_datetime(min_ts) if min_ts is not None else None,
        "max_ts": pd.to_datetime(max_ts) if max_ts is not None else None,
        "porticos": int(porticos or 0),
        "carriles": int(carriles or 0),
        "categorias": int(categorias or 0),
        "patentes": int(patentes or 0),
        "dias": int(dias or 0),
        "horas": int(horas or 0),
    }


def _pl_flow_counts_by_year() -> pl.DataFrame:
    df = _pl_read_flow(["FECHA"])
    if df.is_empty():
        return pl.DataFrame(schema=["anio", "total"])
    df = df.filter(pl.col("FECHA").is_not_null()).with_columns(
        pl.col("FECHA").dt.year().alias("anio")
    )
    result = df.group_by("anio").len().rename({"len": "total"}).sort("anio")
    return result


def _pl_flow_days_by_year() -> pl.DataFrame:
    df = _pl_read_flow(["FECHA"])
    if df.is_empty():
        return pl.DataFrame(schema=["anio", "dias"])
    df = df.filter(pl.col("FECHA").is_not_null()).with_columns(
        [
            pl.col("FECHA").dt.year().alias("anio"),
            pl.col("FECHA").dt.date().alias("fecha_dia"),
        ]
    )
    result = (
        df.group_by("anio")
        .agg(pl.col("fecha_dia").n_unique().alias("dias"))
        .sort("anio")
    )
    return result


def _pl_flow_counts_by_month() -> pl.DataFrame:
    df = _pl_read_flow(["FECHA"])
    if df.is_empty():
        return pl.DataFrame(schema=["anio", "mes", "total"])
    df = df.filter(pl.col("FECHA").is_not_null()).with_columns(
        [
            pl.col("FECHA").dt.year().alias("anio"),
            pl.col("FECHA").dt.month().alias("mes"),
        ]
    )
    result = (
        df.group_by(["anio", "mes"])
        .len()
        .rename({"len": "total"})
        .sort(["anio", "mes"])
    )
    return result


def _pl_flow_missingness() -> pl.DataFrame:
    df = _pl_read_flow(["CARRIL", "VELOCIDAD", "CATEGORIA", "PORTICO", "MATRICULA"])
    if df.is_empty():
        return pl.DataFrame()
    df = df.with_columns(
        [
            _pl_clean_string(pl.col("CARRIL")).alias("carril_clean"),
            _pl_clean_string(pl.col("PORTICO")).alias("portico_clean"),
            _pl_clean_string(pl.col("MATRICULA")).alias("matricula_clean"),
        ]
    )
    total = df.height
    miss = df.select(
        [
            pl.len().alias("total"),
            (
                pl.col("carril_clean").is_null() | (pl.col("carril_clean") == "")
            )
            .sum()
            .alias("miss_carril"),
            pl.col("VELOCIDAD").is_null().sum().alias("miss_velocidad"),
            pl.col("CATEGORIA").is_null().sum().alias("miss_categoria"),
            (
                pl.col("portico_clean").is_null() | (pl.col("portico_clean") == "")
            )
            .sum()
            .alias("miss_portico"),
            (
                pl.col("matricula_clean").is_null()
                | (pl.col("matricula_clean") == "")
            )
            .sum()
            .alias("miss_matricula"),
        ]
    ).to_dicts()[0]

    def _pct(value: int) -> float:
        return round(value * 100.0 / total, 3) if total else 0.0

    data = [
        {"Campo": "CARRIL", "Missing": int(miss["miss_carril"]), "%": _pct(int(miss["miss_carril"]))},
        {"Campo": "VELOCIDAD", "Missing": int(miss["miss_velocidad"]), "%": _pct(int(miss["miss_velocidad"]))},
        {"Campo": "CATEGORIA", "Missing": int(miss["miss_categoria"]), "%": _pct(int(miss["miss_categoria"]))},
        {"Campo": "PORTICO", "Missing": int(miss["miss_portico"]), "%": _pct(int(miss["miss_portico"]))},
        {"Campo": "MATRICULA", "Missing": int(miss["miss_matricula"]), "%": _pct(int(miss["miss_matricula"]))},
    ]
    return pl.DataFrame(data)


def _pl_flow_speed_stats() -> pl.DataFrame:
    df = _pl_read_flow(["VELOCIDAD"])
    if df.is_empty():
        return pl.DataFrame()
    df = df.filter(pl.col("VELOCIDAD").is_not_null())
    if df.is_empty():
        return pl.DataFrame()
    row = df.select(
        [
            pl.col("VELOCIDAD").mean().alias("mean"),
            pl.col("VELOCIDAD").std().alias("sd"),
            pl.col("VELOCIDAD").quantile(0.05, interpolation="linear").alias("p5"),
            pl.col("VELOCIDAD").quantile(0.25, interpolation="linear").alias("p25"),
            pl.col("VELOCIDAD").quantile(0.50, interpolation="linear").alias("p50"),
            pl.col("VELOCIDAD").quantile(0.95, interpolation="linear").alias("p95"),
        ]
    ).to_dicts()[0]
    return pl.DataFrame(
        [
            {
                "Variable": "Velocidad (km/h)",
                "Mean": row.get("mean"),
                "SD": row.get("sd"),
                "P5": row.get("p5"),
                "P25": row.get("p25"),
                "P50": row.get("p50"),
                "P95": row.get("p95"),
            }
        ]
    )


def _pl_flow_speed_stats_by_category() -> pl.DataFrame:
    df = _pl_read_flow(["VELOCIDAD", "CATEGORIA"])
    if df.is_empty():
        return pl.DataFrame()
    df = df.filter(pl.col("VELOCIDAD").is_not_null() & pl.col("CATEGORIA").is_not_null())
    if df.is_empty():
        return pl.DataFrame()
    df = df.with_columns(
        pl.when(pl.col("CATEGORIA") == 4)
        .then(pl.lit(2))
        .otherwise(pl.col("CATEGORIA"))
        .alias("categoria_group")
    )
    result = (
        df.group_by("categoria_group")
        .agg(
            [
                pl.col("VELOCIDAD").mean().alias("mean"),
                pl.col("VELOCIDAD").std().alias("sd"),
                pl.col("VELOCIDAD").quantile(0.05, interpolation="linear").alias("p5"),
                pl.col("VELOCIDAD").quantile(0.25, interpolation="linear").alias("p25"),
                pl.col("VELOCIDAD").quantile(0.50, interpolation="linear").alias("p50"),
                pl.col("VELOCIDAD").quantile(0.95, interpolation="linear").alias("p95"),
            ]
        )
        .rename({"categoria_group": "categoria"})
        .sort("categoria")
    )
    return result


def _pl_flow_window_stats(interval_minutes: int) -> pl.DataFrame:
    df = _pl_read_flow(["FECHA", "PORTICO", "CARRIL", "VELOCIDAD"])
    if df.is_empty():
        return pl.DataFrame()
    interval = max(1, int(interval_minutes))
    df = df.filter(pl.col("FECHA").is_not_null() & pl.col("PORTICO").is_not_null())
    if df.is_empty():
        return pl.DataFrame()
    df = df.with_columns(
        [
            _pl_clean_string(pl.col("PORTICO")).alias("portico_clean"),
            _pl_clean_string(pl.col("CARRIL")).alias("carril_clean"),
            pl.col("FECHA").dt.truncate(f"{interval}m").alias("ts_bin"),
        ]
    )
    lanes = (
        df.filter(pl.col("portico_clean").is_not_null() & (pl.col("portico_clean") != ""))
        .group_by("portico_clean")
        .agg(
            pl.col("carril_clean")
            .filter(pl.col("carril_clean").is_not_null() & (pl.col("carril_clean") != ""))
            .n_unique()
            .alias("lanes")
        )
    )
    windows = (
        df.group_by(["portico_clean", "ts_bin"])
        .agg(
            [
                pl.len().alias("n"),
                pl.col("VELOCIDAD").mean().alias("vbar"),
            ]
        )
        .join(lanes, on="portico_clean", how="left")
    )
    scale = 60.0 / interval
    features = windows.with_columns(
        [
            pl.when(pl.col("lanes") > 0)
            .then(pl.col("n") * scale / pl.col("lanes"))
            .otherwise(None)
            .alias("q"),
            pl.when((pl.col("lanes") > 0) & (pl.col("vbar") > 0))
            .then((pl.col("n") * scale / pl.col("lanes")) / pl.col("vbar"))
            .otherwise(None)
            .alias("k"),
        ]
    )
    row = features.select(
        [
            pl.col("n").mean().alias("n_mean"),
            pl.col("n").std().alias("n_sd"),
            pl.col("n").quantile(0.05, interpolation="linear").alias("n_p5"),
            pl.col("n").quantile(0.25, interpolation="linear").alias("n_p25"),
            pl.col("n").quantile(0.50, interpolation="linear").alias("n_p50"),
            pl.col("n").quantile(0.95, interpolation="linear").alias("n_p95"),
            pl.col("vbar").mean().alias("v_mean"),
            pl.col("vbar").std().alias("v_sd"),
            pl.col("vbar").quantile(0.05, interpolation="linear").alias("v_p5"),
            pl.col("vbar").quantile(0.25, interpolation="linear").alias("v_p25"),
            pl.col("vbar").quantile(0.50, interpolation="linear").alias("v_p50"),
            pl.col("vbar").quantile(0.95, interpolation="linear").alias("v_p95"),
            pl.col("q").mean().alias("q_mean"),
            pl.col("q").std().alias("q_sd"),
            pl.col("q").quantile(0.05, interpolation="linear").alias("q_p5"),
            pl.col("q").quantile(0.25, interpolation="linear").alias("q_p25"),
            pl.col("q").quantile(0.50, interpolation="linear").alias("q_p50"),
            pl.col("q").quantile(0.95, interpolation="linear").alias("q_p95"),
            pl.col("k").mean().alias("k_mean"),
            pl.col("k").std().alias("k_sd"),
            pl.col("k").quantile(0.05, interpolation="linear").alias("k_p5"),
            pl.col("k").quantile(0.25, interpolation="linear").alias("k_p25"),
            pl.col("k").quantile(0.50, interpolation="linear").alias("k_p50"),
            pl.col("k").quantile(0.95, interpolation="linear").alias("k_p95"),
        ]
    ).to_dicts()[0]
    rows = [
        {
            "Variable": "n (veh/ventana)",
            "Mean": row.get("n_mean"),
            "SD": row.get("n_sd"),
            "P5": row.get("n_p5"),
            "P25": row.get("n_p25"),
            "P50": row.get("n_p50"),
            "P95": row.get("n_p95"),
        },
        {
            "Variable": "v̄ (km/h)",
            "Mean": row.get("v_mean"),
            "SD": row.get("v_sd"),
            "P5": row.get("v_p5"),
            "P25": row.get("v_p25"),
            "P50": row.get("v_p50"),
            "P95": row.get("v_p95"),
        },
        {
            "Variable": "q (veh/h/carril)",
            "Mean": row.get("q_mean"),
            "SD": row.get("q_sd"),
            "P5": row.get("q_p5"),
            "P25": row.get("q_p25"),
            "P50": row.get("q_p50"),
            "P95": row.get("q_p95"),
        },
        {
            "Variable": "k (veh/km/carril)",
            "Mean": row.get("k_mean"),
            "SD": row.get("k_sd"),
            "P5": row.get("k_p5"),
            "P25": row.get("k_p25"),
            "P50": row.get("k_p50"),
            "P95": row.get("k_p95"),
        },
    ]
    return pl.DataFrame(rows)


def _pl_flow_plate_concentration(top_percent: float = 0.01) -> Dict[str, object]:
    pct = max(0.0, float(top_percent))
    df = _pl_read_flow(["MATRICULA"])
    if df.is_empty():
        return {}
    df = df.with_columns(
        _pl_clean_string(pl.col("MATRICULA")).str.to_uppercase().alias("plate")
    ).filter(pl.col("plate").is_not_null() & (pl.col("plate") != ""))
    if df.is_empty():
        return {}
    counts = df.group_by("plate").agg(pl.len().alias("cnt")).sort("cnt", descending=True)
    total_detections = int(counts["cnt"].sum())
    total_plates = counts.height
    if total_plates == 0 or total_detections == 0:
        return {}
    top_n = int(np.ceil(total_plates * pct))
    top_detections = int(counts.head(top_n)["cnt"].sum()) if top_n > 0 else 0
    share = top_detections / total_detections * 100.0 if total_detections else 0.0
    return {
        "top_percent": pct,
        "top_detections": top_detections,
        "total_detections": total_detections,
        "total_plates": int(total_plates),
        "top_share": share,
    }


def _pl_flow_window_prevalence(
    acc_windows: object, interval_minutes: int
) -> Dict[str, int]:
    df = _pl_read_flow(["FECHA", "PORTICO"])
    if df.is_empty():
        return {"total_windows": 0, "positive_windows": 0}
    interval = max(1, int(interval_minutes))
    df = df.filter(pl.col("FECHA").is_not_null() & pl.col("PORTICO").is_not_null())
    if df.is_empty():
        return {"total_windows": 0, "positive_windows": 0}
    flow_windows = (
        df.with_columns(
            [
                _pl_clean_string(pl.col("PORTICO")).alias("portico_clean"),
                pl.col("FECHA").dt.truncate(f"{interval}m").alias("ts_bin"),
            ]
        )
        .select(["portico_clean", "ts_bin"])
        .unique()
    )
    total_windows = flow_windows.height
    if acc_windows is None:
        return {"total_windows": int(total_windows), "positive_windows": 0}
    if isinstance(acc_windows, pl.DataFrame):
        acc_pl = acc_windows
    else:
        acc_pl = pl.from_pandas(acc_windows)
    if acc_pl.is_empty():
        return {"total_windows": int(total_windows), "positive_windows": 0}
    acc_pl = (
        acc_pl.with_columns(
            [
                _pl_clean_string(pl.col("PORTICO")).alias("portico_clean"),
            ]
        )
        .select(["portico_clean", "ts_bin"])
        .unique()
    )
    positives = acc_pl.join(flow_windows, on=["portico_clean", "ts_bin"], how="inner")
    return {"total_windows": int(total_windows), "positive_windows": int(positives.height)}


def _ensure_cache_structure(cache: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(cache, dict):
        cache = {}
    if not isinstance(cache.get("meta"), dict):
        cache["meta"] = {}
    flow = cache.get("flow")
    if not isinstance(flow, dict):
        flow = {}
    if "polars" not in flow and "duckdb" not in flow and any(
        key in flow for key in ("basic", "counts_year", "days_year", "counts_month")
    ):
        flow = {"polars": flow}
    for engine in ("polars", "duckdb"):
        if not isinstance(flow.get(engine), dict):
            flow[engine] = {}
    cache["flow"] = flow
    accidents = cache.get("accidents")
    if not isinstance(accidents, dict):
        accidents = {}
    files = accidents.get("files")
    if not isinstance(files, dict):
        files = {}
    accidents["files"] = files
    cache["accidents"] = accidents
    prevalence = cache.get("prevalence")
    if not isinstance(prevalence, dict):
        prevalence = {}
    if "polars" not in prevalence and "duckdb" not in prevalence and prevalence:
        prevalence = {"polars": prevalence}
    for engine in ("polars", "duckdb"):
        if not isinstance(prevalence.get(engine), dict):
            prevalence[engine] = {}
    cache["prevalence"] = prevalence
    flow_stats = cache.get("flow_stats")
    if not isinstance(flow_stats, dict):
        flow_stats = {}
    for engine in ("polars", "duckdb"):
        if not isinstance(flow_stats.get(engine), dict):
            flow_stats[engine] = {}
    cache["flow_stats"] = flow_stats
    return cache


def _empty_desc_cache() -> Dict[str, object]:
    return _ensure_cache_structure({})


def _load_desc_cache() -> Dict[str, object]:
    if not CACHE_PATH.exists():
        return _empty_desc_cache()
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return _empty_desc_cache()
    return _ensure_cache_structure(data)


def _save_desc_cache(cache: Dict[str, object]) -> None:
    cache = _ensure_cache_structure(cache)
    cache["meta"]["updated_at"] = datetime.utcnow().isoformat()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=True)
    tmp_path.replace(CACHE_PATH)


def _df_to_records(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if df is None:
        return {"__columns__": [], "records": []}
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    local = df.copy()
    for col in local.columns:
        if pd.api.types.is_datetime64_any_dtype(local[col]):
            local[col] = local[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    local = local.where(pd.notna(local), None)
    local = local.applymap(lambda value: value.item() if isinstance(value, np.generic) else value)
    return {"__columns__": list(local.columns), "records": local.to_dict(orient="records")}


def _records_to_df(records: Optional[object]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    if isinstance(records, dict) and "records" in records:
        df = pd.DataFrame.from_records(records.get("records") or [])
        columns = records.get("__columns__") or []
        if columns:
            df = df.reindex(columns=columns)
        return df
    if isinstance(records, list):
        return pd.DataFrame.from_records(records)
    return pd.DataFrame(records)


def _serialize_table_list(tables: Optional[list[Dict[str, object]]]) -> list[Dict[str, object]]:
    payload: list[Dict[str, object]] = []
    for item in tables or []:
        payload.append(
            {
                "title": item.get("title", ""),
                "stats": _df_to_records(item.get("stats")),
            }
        )
    return payload


def _deserialize_table_list(payload: Optional[list[Dict[str, object]]]) -> list[Dict[str, object]]:
    results: list[Dict[str, object]] = []
    for item in payload or []:
        results.append(
            {
                "title": item.get("title", ""),
                "stats": _records_to_df(item.get("stats")),
            }
        )
    return results


def _flow_stats_cache_complete(entry: Optional[Dict[str, object]]) -> bool:
    if not isinstance(entry, dict):
        return False
    required = (
        "table_a",
        "weekday_flow",
        "hourly_profile",
        "surface",
    )
    return all(key in entry for key in required)


def _build_hourly_profile_plot_df(df_pl: pl.DataFrame) -> pd.DataFrame:
    if df_pl is None or df_pl.is_empty():
        return pd.DataFrame()
    df = df_pl.with_columns(
        [
            pl.col("FECHA").dt.hour().alias("hour"),
            pl.when(pl.col("FECHA").dt.weekday() >= 6)
            .then(pl.lit("Weekend"))
            .otherwise(pl.lit("Weekday"))
            .alias("day_type"),
        ]
    )
    plot_df = (
        df.group_by(["day_type", "hour"])
        .agg(
            [
                pl.col("flow").sum().alias("total_flow"),
                pl.col("mean_speed").mean().alias("avg_speed"),
            ]
        )
        .sort("hour")
    )
    return plot_df.to_pandas()


def _plot_hourly_profile_interactive(plot_df: pd.DataFrame):
    if plot_df is None or plot_df.empty:
        return None, None
    try:
        import plotly.express as px
    except Exception:
        return None, None
    fig_flow = px.line(
        plot_df,
        x="hour",
        y="total_flow",
        color="day_type",
        markers=True,
        labels={"hour": "Hora", "total_flow": "Flujo promedio"},
        template="plotly_white",
    )
    fig_speed = px.line(
        plot_df,
        x="hour",
        y="avg_speed",
        color="day_type",
        markers=True,
        labels={"hour": "Hora", "avg_speed": "Velocidad media (km/h)"},
        template="plotly_white",
    )
    return fig_flow, fig_speed


def _serialize_basic_stats(stats: Dict[str, object]) -> Dict[str, object]:
    if not stats:
        return {}
    return {
        **stats,
        "min_ts": stats["min_ts"].isoformat() if stats.get("min_ts") is not None else None,
        "max_ts": stats["max_ts"].isoformat() if stats.get("max_ts") is not None else None,
    }


def _deserialize_basic_stats(payload: Dict[str, object]) -> Dict[str, object]:
    if not payload:
        return {}
    data = dict(payload)
    if data.get("min_ts"):
        data["min_ts"] = pd.to_datetime(data["min_ts"], errors="coerce")
    else:
        data["min_ts"] = None
    if data.get("max_ts"):
        data["max_ts"] = pd.to_datetime(data["max_ts"], errors="coerce")
    else:
        data["max_ts"] = None
    return data


def _cache_file_key(path: Path) -> str:
    try:
        mtime = int(path.stat().st_mtime)
    except Exception:
        mtime = 0
    return f"{path}|{mtime}"


def _accident_cache_complete(entry: Optional[Dict[str, object]]) -> bool:
    if not isinstance(entry, dict):
        return False
    required = (
        "coverage",
        "time_dist",
        "weekday_dist",
        "severity_dist",
        "top_porticos",
        "mapped_events",
    )
    return all(key in entry for key in required)


def _flow_cache_complete(entry: Optional[Dict[str, object]]) -> bool:
    if not isinstance(entry, dict):
        return False
    required = (
        "basic",
        "counts_year",
        "days_year",
        "counts_month",
        "missingness",
        "speed",
        "speed_by_category",
    )
    return all(key in entry for key in required)


def _list_flow_csv_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    files = []
    for entry in base_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".csv":
            continue
        if any(fnmatch.fnmatch(entry.name.lower(), pattern) for pattern in FLOW_PATTERNS):
            files.append(entry)
    return sorted(files)


def _list_event_csv_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    files = []
    for entry in base_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".csv":
            continue
        if any(fnmatch.fnmatch(entry.name.lower(), pattern) for pattern in EVENT_PATTERNS):
            files.append(entry)
    return sorted(files)


def _compute_missing_months(
    month_df: pd.DataFrame, min_ts: Optional[pd.Timestamp], max_ts: Optional[pd.Timestamp]
) -> Dict[int, list[str]]:
    if month_df.empty or min_ts is None or max_ts is None:
        return {}
    observed = set(
        pd.Period(year=int(row["anio"]), month=int(row["mes"]), freq="M")
        for _, row in month_df.iterrows()
    )
    all_months = pd.period_range(min_ts.to_period("M"), max_ts.to_period("M"), freq="M")
    missing = sorted(set(all_months) - observed)
    missing_by_year: Dict[int, list[str]] = {}
    for period in missing:
        missing_by_year.setdefault(period.year, []).append(period.strftime("%m"))
    return missing_by_year


def _build_accident_stats(
    file_path: Path,
    cache_entry: Optional[Dict[str, object]] = None,
    save_cache: Optional[Callable[[], None]] = None,
    progress: Optional[_ProgressTracker] = None,
) -> Optional[Dict[str, object]]:
    cache_entry = cache_entry or {}
    stats: Dict[str, object] = {}
    required = [
        "coverage",
        "time_dist",
        "weekday_dist",
        "severity_dist",
        "top_porticos",
        "mapped_events",
    ]

    for key in required:
        if key in cache_entry:
            stats[key] = _records_to_df(cache_entry.get(key))
            if progress is not None:
                progress.mark_done(f"acc_{key}", f"Accidentes: {key} (cache)")

    if (
        "mapped_events" in stats
        and not stats["mapped_events"].empty
        and "accidente_time" in stats["mapped_events"].columns
    ):
        stats["mapped_events"]["accidente_time"] = pd.to_datetime(
            stats["mapped_events"]["accidente_time"], errors="coerce"
        )

    missing = [key for key in required if key not in cache_entry]
    if not missing:
        return stats

    result = load_accidentes(file_path=str(file_path), return_excluded=True)
    if result is None:
        return None

    if isinstance(result, tuple):
        mapped_df, excluded_df = result
    else:
        mapped_df, excluded_df = result, pd.DataFrame()

    mapped_df = mapped_df.copy()
    mapped_df["mapped"] = True
    excluded_df = excluded_df.copy()
    if not excluded_df.empty:
        excluded_df["mapped"] = False
        combined_df = pd.concat([mapped_df, excluded_df], ignore_index=True)
    else:
        combined_df = mapped_df.copy()

    def _set_cache(key: str, df: pl.DataFrame) -> None:
        cache_entry[key] = _df_to_records(df)
        stats[key] = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        if save_cache is not None:
            save_cache()
        if progress is not None:
            progress.mark_done(f"acc_{key}", f"Accidentes: {key}")

    if combined_df.empty:
        empty_df = pl.DataFrame()
        for key in missing:
            _set_cache(key, empty_df)
        return stats

    def _safe_col(df: pd.DataFrame, name: str, aliases=None) -> Optional[str]:
        try:
            return buscar_columna(df, name, aliases=aliases)
        except Exception:
            return None

    fecha_col = _safe_col(combined_df, "Fechas Inicio", aliases=["Fecha Inicio"])
    km_col = _safe_col(combined_df, "Km.", aliases=["Km"])
    eje_col = _safe_col(combined_df, "Eje")
    calzada_col = _safe_col(combined_df, "Calzada")

    needed_cols = {"accidente_time", "mapped", "severidad", "ultimo_portico"}
    if fecha_col:
        needed_cols.add(fecha_col)
    if km_col:
        needed_cols.add(km_col)
    if eje_col:
        needed_cols.add(eje_col)
    if calzada_col:
        needed_cols.add(calzada_col)
    subset_cols = [col for col in combined_df.columns if col in needed_cols]
    subset_df = combined_df[subset_cols].copy()
    obj_cols = subset_df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        subset_df[col] = subset_df[col].astype("string")
    pl_combined = pl.from_pandas(subset_df)

    if fecha_col:
        fecha_expr = pl.coalesce(
            [
                pl.col(fecha_col)
                .cast(pl.Utf8)
                .str.to_datetime(strict=False, exact=False),
                pl.col("accidente_time"),
            ]
        )
    else:
        fecha_expr = pl.col("accidente_time")

    pl_combined = pl_combined.with_columns(
        [
            fecha_expr.alias("fecha_base"),
            pl.col("accidente_time").is_not_null().alias("has_time"),
        ]
    )
    if km_col and eje_col and calzada_col:
        pl_combined = pl_combined.with_columns(
            (
                pl.col(km_col).is_not_null()
                & pl.col(eje_col).is_not_null()
                & pl.col(calzada_col).is_not_null()
            ).alias("has_loc")
        )
    else:
        pl_combined = pl_combined.with_columns(pl.lit(False).alias("has_loc"))

    pl_combined = pl_combined.with_columns(
        pl.col("fecha_base").dt.year().alias("year")
    )

    if "coverage" in missing:
        coverage = (
            pl_combined.group_by("year", maintain_order=True)
            .agg(
                [
                    pl.len().alias("Accidentes"),
                    pl.col("has_time").mean().alias("% con hora"),
                    pl.col("has_loc").mean().alias("% con localizacion"),
                    pl.col("mapped").mean().alias("% mapeado a portico"),
                ]
            )
            .with_columns(
                [
                    (pl.col("% con hora") * 100.0).round(2),
                    (pl.col("% con localizacion") * 100.0).round(2),
                    (pl.col("% mapeado a portico") * 100.0).round(2),
                ]
            )
            .rename({"year": "Año"})
        )
        coverage = coverage.with_columns(
            [
                pl.col("Año").cast(pl.Utf8),
                pl.col("Accidentes").cast(pl.Int64),
                pl.col("% con hora").cast(pl.Float64),
                pl.col("% con localizacion").cast(pl.Float64),
                pl.col("% mapeado a portico").cast(pl.Float64),
            ]
        )
        total_row = pl.DataFrame(
            {
                "Año": ["Total"],
                "Accidentes": [int(coverage["Accidentes"].sum())],
                "% con hora": [round(pl_combined["has_time"].mean() * 100.0, 2)],
                "% con localizacion": [
                    round(pl_combined["has_loc"].mean() * 100.0, 2)
                ],
                "% mapeado a portico": [
                    round(pl_combined["mapped"].mean() * 100.0, 2)
                ],
            }
        ).with_columns(
            [
                pl.col("Año").cast(pl.Utf8),
                pl.col("Accidentes").cast(pl.Int64),
                pl.col("% con hora").cast(pl.Float64),
                pl.col("% con localizacion").cast(pl.Float64),
                pl.col("% mapeado a portico").cast(pl.Float64),
            ]
        )
        coverage = pl.concat([coverage, total_row], how="vertical")
        _set_cache("coverage", coverage)

    time_df = pl_combined.filter(pl.col("accidente_time").is_not_null())
    if "time_dist" in missing or "weekday_dist" in missing:
        if time_df.is_empty():
            if "time_dist" in missing:
                _set_cache("time_dist", pl.DataFrame())
            if "weekday_dist" in missing:
                _set_cache("weekday_dist", pl.DataFrame())
        else:
            if "time_dist" in missing:
                time_dist = (
                    time_df.with_columns(
                        pl.when(
                            (pl.col("accidente_time").dt.hour() >= 7)
                            & (pl.col("accidente_time").dt.hour() < 10)
                        )
                        .then(pl.lit("AM peak (07-10)"))
                        .when(
                            (pl.col("accidente_time").dt.hour() >= 17)
                            & (pl.col("accidente_time").dt.hour() < 20)
                        )
                        .then(pl.lit("PM peak (17-20)"))
                        .otherwise(pl.lit("Otro"))
                        .alias("Nivel")
                    )
                    .group_by("Nivel")
                    .agg(pl.len().alias("Conteo"))
                    .with_columns(
                        (pl.col("Conteo") / pl.col("Conteo").sum() * 100.0)
                        .round(2)
                        .alias("%")
                    )
                    .with_columns(pl.lit("Franja horaria").alias("Categoria"))
                    .select(["Categoria", "Nivel", "Conteo", "%"])
                )
                _set_cache("time_dist", time_dist)
            if "weekday_dist" in missing:
                weekday_dist = (
                    time_df.with_columns(
                        pl.when(pl.col("accidente_time").dt.weekday() <= 5)
                        .then(pl.lit("Lun-Vie"))
                        .otherwise(pl.lit("Sab-Dom"))
                        .alias("Nivel")
                    )
                    .group_by("Nivel")
                    .agg(pl.len().alias("Conteo"))
                    .with_columns(
                        (pl.col("Conteo") / pl.col("Conteo").sum() * 100.0)
                        .round(2)
                        .alias("%")
                    )
                    .with_columns(pl.lit("Dia semana").alias("Categoria"))
                    .select(["Categoria", "Nivel", "Conteo", "%"])
                )
                _set_cache("weekday_dist", weekday_dist)

    if "severity_dist" in missing:
        severity_dist = pl.DataFrame()
        if "severidad" in pl_combined.columns:
            severity_dist = (
                pl_combined.with_columns(
                    pl.col("severidad").cast(pl.Utf8).alias("Severidad")
                )
                .group_by("Severidad")
                .agg(pl.len().alias("Conteo"))
                .with_columns(
                    (pl.col("Conteo") / pl.col("Conteo").sum() * 100.0)
                    .round(2)
                    .alias("%")
                )
            )
        _set_cache("severity_dist", severity_dist)

    if "top_porticos" in missing:
        top_porticos = pl.DataFrame()
        if "ultimo_portico" in mapped_df.columns and not mapped_df.empty:
            mapped_subset = mapped_df[["ultimo_portico"]].copy()
            if mapped_subset["ultimo_portico"].dtype == "object":
                mapped_subset["ultimo_portico"] = mapped_subset["ultimo_portico"].astype("string")
            pl_mapped = pl.from_pandas(mapped_subset)
            top_porticos = (
                pl_mapped.with_columns(
                    pl.col("ultimo_portico").cast(pl.Utf8).alias("Portico")
                )
                .group_by("Portico")
                .agg(pl.len().alias("Accidentes"))
                .sort("Accidentes", descending=True)
                .head(10)
            )
        _set_cache("top_porticos", top_porticos)

    if "mapped_events" in missing:
        mapped_events = pl.DataFrame()
        if "ultimo_portico" in mapped_df.columns and "accidente_time" in mapped_df.columns:
            mapped_subset = mapped_df[["ultimo_portico", "accidente_time"]].copy()
            if mapped_subset["ultimo_portico"].dtype == "object":
                mapped_subset["ultimo_portico"] = mapped_subset["ultimo_portico"].astype("string")
            pl_mapped = pl.from_pandas(mapped_subset)
            mapped_events = pl_mapped.select(["ultimo_portico", "accidente_time"])
        _set_cache("mapped_events", mapped_events)

    if (
        "mapped_events" in stats
        and not stats["mapped_events"].empty
        and "accidente_time" in stats["mapped_events"].columns
    ):
        stats["mapped_events"]["accidente_time"] = pd.to_datetime(
            stats["mapped_events"]["accidente_time"], errors="coerce"
        )

    return stats


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix or ".csv"
    for idx in range(1, 1000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{stem}_{stamp}{suffix}")


def _save_uploaded_csv(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_name = Path(uploaded_file.name or "flujos_upload.csv").name
    filename = raw_name if raw_name else "flujos_upload.csv"
    destination = _unique_path(target_dir / filename)
    with open(destination, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return destination


def _select_csv_source(key_prefix: str) -> Tuple[str, Optional[Path], Optional[object]]:
    source = st.radio(
        "Origen del CSV",
        ["Archivo en Datos/", "Subir CSV"],
        key=f"{key_prefix}_source",
        horizontal=True,
    )
    if source == "Archivo en Datos/":
        files = _list_flow_csv_files(DATA_DIR)
        if not files:
            st.warning("No se encontraron archivos flujo*.csv en Datos/.")
            return source, None, None
        names = [path.name for path in files]
        selected = st.selectbox(
            "Archivo CSV",
            names,
            key=f"{key_prefix}_file",
        )
        return source, DATA_DIR / selected, None
    uploaded = st.file_uploader(
        "Subir CSV de flujos",
        type=["csv"],
        key=f"{key_prefix}_upload",
    )
    return source, None, uploaded


def _resolve_csv_path(
    source: str, selected_path: Optional[Path], uploaded
) -> Optional[Path]:
    if source == "Archivo en Datos/":
        if selected_path is None:
            st.error("Seleccione un archivo CSV valido.")
            return None
        return selected_path
    if uploaded is None:
        st.error("Suba un archivo CSV para continuar.")
        return None
    saved_path = _save_uploaded_csv(uploaded, DATA_DIR)
    st.caption(f"Archivo guardado en: {saved_path}")
    return saved_path


def _render_summary() -> None:
    try:
        summary = get_flow_db_summary(db_path=DB_PATH)
    except ImportError as exc:
        st.error(str(exc))
        st.stop()
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{summary.row_count:,}")
    col2.metric("Fecha min", _format_ts(summary.min_timestamp))
    col3.metric("Fecha max", _format_ts(summary.max_timestamp))
    st.caption(f"Archivo DuckDB: {summary.db_path}")


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Flow database", layout="wide")
    st.title("Flow database (DuckDB)")

    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)

    summary_container = st.empty()
    with summary_container.container():
        _render_summary()

    tabs = st.tabs(
        [
            "Importar / Anexar",
            "Reemplazar",
            "Vaciar tabla",
            "Esquema y muestra",
            "Estadistica de flujos",
        ]
    )

    with tabs[0]:
        st.write("Importa un CSV y lo anexa a la tabla actual.")
        source, selected_path, uploaded = _select_csv_source("append")
        progress_placeholder = st.empty()
        progress_status = st.empty()
        if st.button("Importar y anexar", key="append_button"):
            csv_path = _resolve_csv_path(source, selected_path, uploaded)
            if csv_path is None:
                st.stop()
            progress_bar = progress_placeholder.progress(0.0)

            def _update_progress(ratio: float, message: str) -> None:
                progress_bar.progress(ratio)
                progress_status.caption(message)

            with st.spinner("Importando datos..."):
                inserted = _run_import(
                    csv_path=csv_path,
                    replace=False,
                    progress_callback=_update_progress,
                )
            progress_bar.progress(1.0)
            if inserted:
                st.success(f"Se agregaron {inserted:,} filas.")
            else:
                st.warning("No se agregaron filas.")
            with summary_container.container():
                _render_summary()

    with tabs[1]:
        st.write("Reemplaza la tabla completa con un CSV.")
        source, selected_path, uploaded = _select_csv_source("replace")
        confirm_text = st.text_input(
            "Escriba REPLACE para confirmar",
            key="replace_confirm",
        )
        can_replace = confirm_text.strip().lower() == "replace"
        if st.button(
            "Reemplazar tabla",
            key="replace_button",
            disabled=not can_replace,
        ):
            csv_path = _resolve_csv_path(source, selected_path, uploaded)
            if csv_path is None:
                st.stop()
            progress_placeholder = st.empty()
            progress_status = st.empty()
            progress_bar = progress_placeholder.progress(0.0)

            def _update_progress(ratio: float, message: str) -> None:
                progress_bar.progress(ratio)
                progress_status.caption(message)

            with st.spinner("Reemplazando datos..."):
                inserted = _run_import(
                    csv_path=csv_path,
                    replace=True,
                    progress_callback=_update_progress,
                )
            progress_bar.progress(1.0)
            st.success(f"Se cargaron {inserted:,} filas nuevas.")
            with summary_container.container():
                _render_summary()

    with tabs[2]:
        st.write("Elimina todos los registros de la tabla.")
        confirm_text = st.text_input(
            "Escriba DELETE para confirmar",
            key="delete_confirm",
        )
        can_delete = confirm_text.strip().upper() == "DELETE"
        if st.button(
            "Vaciar tabla",
            key="delete_button",
            disabled=not can_delete,
        ):
            with st.spinner("Eliminando registros..."):
                removed = clear_flow_table(db_path=DB_PATH)
            st.success(f"Se eliminaron {removed:,} filas.")
            with summary_container.container():
                _render_summary()

        st.divider()
        st.subheader("Eliminar por rango de fechas")
        try:
            summary = get_flow_db_summary(db_path=DB_PATH)
        except ImportError as exc:
            st.error(str(exc))
            st.stop()

        if summary.row_count <= 0:
            st.info("La tabla está vacía. No hay registros para eliminar.")
        else:
            range_label = "-"
            if summary.min_timestamp or summary.max_timestamp:
                start_label = (
                    summary.min_timestamp.strftime("%Y-%m-%d %H:%M")
                    if summary.min_timestamp
                    else "-"
                )
                end_label = (
                    summary.max_timestamp.strftime("%Y-%m-%d %H:%M")
                    if summary.max_timestamp
                    else "-"
                )
                range_label = f"{start_label} → {end_label}"
            st.caption(f"Rango disponible en la base: {range_label}")

            default_start_date = (
                summary.min_timestamp.date()
                if summary.min_timestamp
                else datetime.today().date()
            )
            default_end_date = (
                summary.max_timestamp.date()
                if summary.max_timestamp
                else datetime.today().date()
            )
            default_start_time = (
                summary.min_timestamp.time().replace(second=0, microsecond=0)
                if summary.min_timestamp
                else dt_time(0, 0)
            )
            default_end_time = (
                summary.max_timestamp.time().replace(second=0, microsecond=0)
                if summary.max_timestamp
                else dt_time(23, 59)
            )

            cols = st.columns(2)
            with cols[0]:
                start_date = st.date_input(
                    "Fecha inicio",
                    value=default_start_date,
                    key="delete_range_start_date",
                )
            with cols[1]:
                end_date = st.date_input(
                    "Fecha fin",
                    value=default_end_date,
                    key="delete_range_end_date",
                )

            cols = st.columns(2)
            with cols[0]:
                start_time = st.time_input(
                    "Hora inicio",
                    value=default_start_time,
                    key="delete_range_start_time",
                )
            with cols[1]:
                end_time = st.time_input(
                    "Hora fin",
                    value=default_end_time,
                    key="delete_range_end_time",
                )

            start_ts = pd.Timestamp(datetime.combine(start_date, start_time))
            end_ts = pd.Timestamp(datetime.combine(end_date, end_time))
            if end_ts < start_ts:
                st.error("La fecha/hora final debe ser posterior a la inicial.")
                can_delete_range = False
            else:
                st.caption(
                    f"Se eliminarán registros entre {start_ts} y {end_ts} (inclusive)."
                )
                can_delete_range = True

            confirm_range = st.text_input(
                "Escriba RANGE para confirmar",
                key="delete_range_confirm",
            )
            can_delete_range = (
                can_delete_range and confirm_range.strip().upper() == "RANGE"
            )
            if st.button(
                "Eliminar rango",
                key="delete_range_button",
                disabled=not can_delete_range,
            ):
                delete_fn = _get_delete_flow_rows_fn()
                if delete_fn is None:
                    st.error(
                        "No se pudo cargar la función de borrado por rango. "
                        "Reinicie la app."
                    )
                    st.stop()
                with st.spinner("Eliminando registros en rango..."):
                    removed = delete_fn(
                        date_start=start_ts,
                        date_end=end_ts,
                        db_path=DB_PATH,
                    )
                st.success(f"Se eliminaron {removed:,} filas en el rango.")
                with summary_container.container():
                    _render_summary()

    with tabs[3]:
        st.write("Explora el esquema y una muestra de la tabla.")
        try:
            columns = get_flow_table_columns(db_path=DB_PATH)
        except ImportError as exc:
            st.error(str(exc))
            st.stop()
        if columns:
            st.subheader("Columnas")
            st.dataframe(pd.DataFrame(columns), width="stretch")
        else:
            st.info("La tabla no tiene columnas definidas.")

        sample_limit = st.slider(
            "Filas de muestra",
            min_value=1,
            max_value=200,
            value=5,
            step=1,
        )
        try:
            sample_df = get_flow_table_sample(
                limit=int(sample_limit), db_path=DB_PATH
            )
        except ImportError as exc:
            st.error(str(exc))
            st.stop()
        if sample_df.empty:
            st.warning("La tabla esta vacia o no hay datos disponibles.")
        else:
            st.subheader("Muestra")
            st.dataframe(sample_df, width="stretch")

        st.subheader("Totales")
        col_totals_a, col_totals_b = st.columns(2)

        with col_totals_a:
            st.caption("Por tipo de vehiculo")
            try:
                category_df = get_flow_counts_by_category(db_path=DB_PATH)
            except ImportError as exc:
                st.error(str(exc))
                st.stop()
            if category_df.empty:
                st.info("No hay datos de categorias disponibles.")
            else:
                category_df = category_df.copy()
                category_df["tipo"] = category_df["categoria"].map(CATEGORY_LABELS)
                category_df["tipo"] = category_df["tipo"].fillna(
                    category_df["categoria"].map(lambda value: f"Categoria {value}")
                )
                category_df = category_df.rename(
                    columns={"tipo": "Tipo de vehiculo", "total": "Total"}
                )[["Tipo de vehiculo", "Total"]]
                st.dataframe(category_df, width="stretch")

        with col_totals_b:
            st.caption("Total por anio")
            try:
                year_df = get_flow_counts_by_year(db_path=DB_PATH)
            except ImportError as exc:
                st.error(str(exc))
                st.stop()
            if year_df.empty:
                st.info("No hay datos de anios disponibles.")
            else:
                year_df = year_df.rename(
                    columns={"anio": "Anio", "total": "Total"}
                )
                st.dataframe(year_df, width="stretch")

    with tabs[4]:
        st.write("Descripcion y estadistica de flujos/accidentes (dataset operativo).")
        params_cols = st.columns(3)
        engine_label = params_cols[0].selectbox(
            "Motor de calculo",
            ["DuckDB", "Polars (batching)"],
            key="flow_desc_engine",
        )
        interval_minutes = params_cols[1].number_input(
            "Intervalo de ventana (min)",
            min_value=1,
            max_value=60,
            value=5,
            step=1,
            key="flow_desc_interval",
        )
        lead_intervals = params_cols[2].number_input(
            "Lead time (ventanas hacia atras)",
            min_value=0,
            max_value=12,
            value=1,
            step=1,
            key="flow_desc_lead",
        )
        min_speed = st.number_input(
            "Min Speed (km/h)",
            min_value=1.0,
            max_value=150.0,
            value=1.0,
            step=1.0,
            key="flow_stats_min_speed",
        )

        engine_key = "duckdb" if engine_label.startswith("DuckDB") else "polars"
        current_params = {
            "engine": engine_key,
            "interval": int(interval_minutes),
            "lead": int(lead_intervals),
            "min_speed": float(min_speed),
        }
        if st.session_state.get("flow_desc_run_params") != current_params:
            st.session_state["flow_desc_run"] = False
        if st.button("Calcular", key="flow_desc_run_btn"):
            st.session_state["flow_desc_run"] = True
            st.session_state["flow_desc_run_params"] = current_params
        if not st.session_state.get("flow_desc_run"):
            st.info("Presione Calcular para cargar resultados o generar nuevos.")
        else:
            cache = _load_desc_cache()
            flow_mtime = int(DB_PATH.stat().st_mtime) if DB_PATH.exists() else None
            if flow_mtime is not None and cache["meta"].get("flow_db_mtime") != flow_mtime:
                cache["flow"] = {"polars": {}, "duckdb": {}}
                cache["prevalence"] = {"polars": {}, "duckdb": {}}
                cache["flow_stats"] = {"polars": {}, "duckdb": {}}
                cache["meta"]["flow_db_mtime"] = flow_mtime
                _save_desc_cache(cache)

            flow_cache = cache["flow"].setdefault(engine_key, {})
            acc_cache = cache["accidents"]["files"]
            prev_cache = cache["prevalence"].setdefault(engine_key, {})
            flow_stats_cache = cache["flow_stats"].setdefault(engine_key, {})
            flow_complete = False
            acc_complete = False
            flow_stats_complete = False
            accident_stats = None
            selected_event = None
            event_key = None
            progress = _ProgressTracker()

            def _get_cached_df(
                key: str, label: str, compute_fn: Callable[[], object]
            ) -> pd.DataFrame:
                task_key = f"{engine_key}_flow_{key}"
                progress.register(task_key)
                if key in flow_cache:
                    df_cached = _records_to_df(flow_cache.get(key))
                    progress.mark_done(task_key, f"{label} (cache)")
                    return df_cached
                df = compute_fn()
                flow_cache[key] = _df_to_records(df)
                _save_desc_cache(cache)
                df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
                progress.mark_done(task_key, label)
                return df_pd

            st.subheader("Traffic-flow (AVI) dataset")
            try:
                progress.register("flow_basic")
                if "basic" in flow_cache:
                    basic_stats = _deserialize_basic_stats(flow_cache.get("basic"))
                    progress.mark_done("flow_basic", "Flujos: resumen (cache)")
                else:
                    if engine_key == "polars":
                        basic_stats = _pl_flow_basic_stats()
                    else:
                        basic_stats = get_flow_basic_stats(db_path=DB_PATH)
                    flow_cache["basic"] = _serialize_basic_stats(basic_stats)
                    _save_desc_cache(cache)
                    progress.mark_done("flow_basic", "Flujos: resumen")
            except ImportError as exc:
                st.error(str(exc))
                st.stop()

            if not basic_stats or basic_stats.get("total", 0) == 0:
                st.warning("La tabla de flujos esta vacia o no tiene datos.")
            else:
                summary_rows = [
                    {
                        "Metrica": "Periodo",
                        "Valor": f"{_format_ts(basic_stats['min_ts'])} - {_format_ts(basic_stats['max_ts'])}",
                    },
                    {"Metrica": "Detecciones", "Valor": basic_stats["total"]},
                    {"Metrica": "Pórticos", "Valor": basic_stats["porticos"]},
                    {"Metrica": "Pistas", "Valor": basic_stats["carriles"]},
                    {"Metrica": "Clases vehiculares", "Valor": basic_stats["categorias"]},
                    {"Metrica": "Patentes únicas", "Valor": basic_stats["patentes"]},
                    {"Metrica": "Días observados", "Valor": basic_stats["dias"]},
                    {"Metrica": "Horas observadas", "Valor": basic_stats["horas"]},
                ]
                st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

                try:
                    counts_year = _get_cached_df(
                        "counts_year",
                        "Flujos: conteos por año",
                        _pl_flow_counts_by_year
                        if engine_key == "polars"
                        else lambda: get_flow_counts_by_year(db_path=DB_PATH),
                    )
                    days_year = _get_cached_df(
                        "days_year",
                        "Flujos: dias por año",
                        _pl_flow_days_by_year
                        if engine_key == "polars"
                        else lambda: get_flow_days_by_year(db_path=DB_PATH),
                    )
                    counts_month = _get_cached_df(
                        "counts_month",
                        "Flujos: conteos por mes",
                        _pl_flow_counts_by_month
                        if engine_key == "polars"
                        else lambda: get_flow_counts_by_month(db_path=DB_PATH),
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    st.stop()

                coverage_pl = (
                    pl.from_pandas(counts_year)
                    .join(pl.from_pandas(days_year), on="anio", how="left")
                    .rename(
                        {"anio": "Año", "total": "Detecciones", "dias": "Dias activos"}
                    )
                    .with_columns(pl.lit("-").alias("% removido (limpieza)"))
                )
                coverage = coverage_pl.to_pandas()
                missing_by_year = _compute_missing_months(
                    counts_month, basic_stats["min_ts"], basic_stats["max_ts"]
                )
                coverage["Notas"] = coverage["Año"].map(
                    lambda year: (
                        "Meses sin datos: " + ", ".join(missing_by_year.get(int(year), []))
                        if int(year) in missing_by_year and missing_by_year[int(year)]
                        else ""
                    )
                )
                total_row = pd.DataFrame(
                    [
                        {
                            "Año": "Total",
                            "Detecciones": basic_stats["total"],
                            "Dias activos": basic_stats["dias"],
                            "% removido (limpieza)": "-",
                            "Notas": f"{basic_stats['porticos']} pórticos, {basic_stats['carriles']} pistas",
                        }
                    ]
                )
                coverage = pd.concat([coverage, total_row], ignore_index=True)
                coverage["Año"] = coverage["Año"].astype(str)
                st.subheader("Cobertura anual")
                st.dataframe(coverage, width="stretch", hide_index=True)

                st.subheader("Calidad y missingness")
                try:
                    missing_df = _get_cached_df(
                        "missingness",
                        "Flujos: missingness",
                        _pl_flow_missingness
                        if engine_key == "polars"
                        else lambda: get_flow_missingness_stats(db_path=DB_PATH),
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    st.stop()
                if missing_df.empty:
                    st.info("No se pudo calcular missingness.")
                else:
                    st.dataframe(missing_df, width="stretch", hide_index=True)

                st.subheader("Velocidad instantanea (por deteccion)")
                try:
                    speed_df = _get_cached_df(
                        "speed",
                        "Flujos: velocidad",
                        _pl_flow_speed_stats
                        if engine_key == "polars"
                        else lambda: get_flow_speed_stats(db_path=DB_PATH),
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    st.stop()
                if speed_df.empty:
                    st.info("No hay datos de velocidad disponibles.")
                else:
                    st.dataframe(speed_df, width="stretch", hide_index=True)

                try:
                    speed_cat_df = _get_cached_df(
                        "speed_by_category",
                        "Flujos: velocidad por categoria",
                        _pl_flow_speed_stats_by_category
                        if engine_key == "polars"
                        else lambda: get_flow_speed_stats_by_category(db_path=DB_PATH),
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    st.stop()
                if not speed_cat_df.empty and "categoria" in speed_cat_df.columns:
                    if (speed_cat_df["categoria"] == 4).any():
                        try:
                            if engine_key == "polars":
                                speed_cat_df = _pl_flow_speed_stats_by_category()
                            else:
                                speed_cat_df = get_flow_speed_stats_by_category(
                                    db_path=DB_PATH
                                )
                            flow_cache["speed_by_category"] = _df_to_records(speed_cat_df)
                            _save_desc_cache(cache)
                            if isinstance(speed_cat_df, pl.DataFrame):
                                speed_cat_df = speed_cat_df.to_pandas()
                        except Exception:
                            pass
                if not speed_cat_df.empty:
                    speed_cat_df = speed_cat_df.copy()
                    speed_cat_df["Tipo"] = speed_cat_df["categoria"].map(CATEGORY_LABELS)
                    speed_cat_df["Tipo"] = speed_cat_df["Tipo"].fillna(
                        speed_cat_df["categoria"].map(lambda value: f"Categoria {value}")
                    )
                    speed_cat_df = speed_cat_df.rename(
                        columns={
                            "Tipo": "Tipo de vehiculo",
                            "mean": "Mean",
                            "sd": "SD",
                            "p5": "P5",
                            "p25": "P25",
                            "p50": "P50",
                            "p95": "P95",
                        }
                    )[["Tipo de vehiculo", "Mean", "SD", "P5", "P25", "P50", "P95"]]
                    st.dataframe(speed_cat_df, width="stretch", hide_index=True)

                st.subheader("Ventanas 5-min (pórtico x ventana)")
                interval_key = f"interval_{int(interval_minutes)}"
                window_cache = flow_cache.setdefault("window_stats", {})
                window_task = f"flow_window_{interval_key}"
                progress.register(f"{engine_key}_{window_task}")
                if interval_key in window_cache:
                    window_stats = _records_to_df(window_cache.get(interval_key))
                    progress.mark_done(
                        f"{engine_key}_{window_task}", "Flujos: ventanas 5-min (cache)"
                    )
                else:
                    with st.spinner("Calculando estadisticos 5-min..."):
                        if engine_key == "polars":
                            window_stats = _pl_flow_window_stats(int(interval_minutes))
                        else:
                            window_stats = get_flow_window_stats(
                                db_path=DB_PATH,
                                interval_minutes=int(interval_minutes),
                            )
                    window_cache[interval_key] = _df_to_records(window_stats)
                    _save_desc_cache(cache)
                    if isinstance(window_stats, pl.DataFrame):
                        window_stats = window_stats.to_pandas()
                    progress.mark_done(
                        f"{engine_key}_{window_task}", "Flujos: ventanas 5-min"
                    )
                if window_stats is not None:
                    if window_stats.empty:
                        st.info("No se pudieron calcular estadisticos de ventanas.")
                    else:
                        st.dataframe(window_stats, width="stretch", hide_index=True)

                plate_key = "plate_concentration_0.01"
                progress.register(f"{engine_key}_flow_plate")
                if plate_key in flow_cache:
                    plate_stats = flow_cache.get(plate_key)
                    progress.mark_done(
                        f"{engine_key}_flow_plate",
                        "Flujos: concentracion patentes (cache)",
                    )
                else:
                    with st.spinner("Calculando concentracion de patentes..."):
                        if engine_key == "polars":
                            plate_stats = _pl_flow_plate_concentration(top_percent=0.01)
                        else:
                            plate_stats = get_flow_plate_concentration(
                                db_path=DB_PATH, top_percent=0.01
                            )
                    flow_cache[plate_key] = plate_stats
                    _save_desc_cache(cache)
                    progress.mark_done(
                        f"{engine_key}_flow_plate", "Flujos: concentracion patentes"
                    )
                if not plate_stats:
                    st.info("No se pudieron calcular metricas de patentes.")
                else:
                    st.write(
                        f"Top 1% de patentes: {plate_stats['top_share']:.2f}% "
                        f"de las detecciones ({plate_stats['top_detections']:,} de "
                        f"{plate_stats['total_detections']:,})."
                    )

                flow_complete = _flow_cache_complete(flow_cache)

            st.subheader("Crash reports dataset")
            event_files = _list_event_csv_files(DATA_DIR)
            if not event_files:
                st.warning("No se encontraron archivos de eventos en Datos.")
                accident_stats = None
            else:
                selected_event = st.selectbox(
                    "Archivo de eventos",
                    options=event_files,
                    format_func=lambda p: p.name,
                )
                event_key = _cache_file_key(selected_event)
                acc_entry = acc_cache.get(event_key, {})
                if acc_entry:
                    accident_stats = {}
                    for key, value in acc_entry.items():
                        accident_stats[key] = _records_to_df(value)
                    if (
                        "mapped_events" in accident_stats
                        and not accident_stats["mapped_events"].empty
                        and "accidente_time" in accident_stats["mapped_events"].columns
                    ):
                        accident_stats["mapped_events"]["accidente_time"] = pd.to_datetime(
                            accident_stats["mapped_events"]["accidente_time"], errors="coerce"
                        )
                acc_complete = _accident_cache_complete(acc_entry)
                for acc_key in (
                    "coverage",
                    "time_dist",
                    "weekday_dist",
                    "severity_dist",
                    "top_porticos",
                    "mapped_events",
                ):
                    progress.register(f"acc_{acc_key}")
                if acc_entry and acc_complete:
                    for acc_key in acc_entry.keys():
                        progress.mark_done(
                            f"acc_{acc_key}", f"Accidentes: {acc_key} (cache)"
                        )
                if not acc_complete:
                    acc_cache[event_key] = acc_entry
                    with st.spinner("Procesando eventos..."):
                        accident_stats = _build_accident_stats(
                            selected_event,
                            cache_entry=acc_entry,
                            save_cache=lambda: _save_desc_cache(cache),
                            progress=progress,
                        )
                    acc_cache[event_key] = acc_entry
                    _save_desc_cache(cache)
                    acc_complete = _accident_cache_complete(acc_entry)

            if accident_stats is None:
                st.info("Calcule estadisticas para visualizar las tablas de accidentes.")
            else:
                coverage = accident_stats.get("coverage", pd.DataFrame())
                if coverage.empty:
                    st.info("No hay datos de accidentes para mostrar.")
                else:
                    if "Año" in coverage.columns:
                        coverage["Año"] = coverage["Año"].astype(str)
                    st.dataframe(coverage, width="stretch", hide_index=True)

                dist_frames = []
                time_dist = accident_stats.get("time_dist", pd.DataFrame())
                weekday_dist = accident_stats.get("weekday_dist", pd.DataFrame())
                if not time_dist.empty:
                    dist_frames.append(time_dist)
                if not weekday_dist.empty:
                    dist_frames.append(weekday_dist)
                if dist_frames:
                    dist_df = pd.concat(dist_frames, ignore_index=True)
                    st.subheader("Distribucion temporal (resumen)")
                    st.dataframe(dist_df, width="stretch", hide_index=True)

                severity_dist = accident_stats.get("severity_dist", pd.DataFrame())
                if not severity_dist.empty:
                    st.subheader("Distribucion por severidad")
                    st.dataframe(severity_dist, width="stretch", hide_index=True)

                top_porticos = accident_stats.get("top_porticos", pd.DataFrame())
                if not top_porticos.empty:
                    st.subheader("Top 10 porticos con mas accidentes")
                    st.dataframe(top_porticos, width="stretch", hide_index=True)

            st.subheader("Prevalencia del target (p,τ)")
            if accident_stats is None or accident_stats.get("mapped_events") is None:
                st.info("Necesitas cargar accidentes para calcular la prevalencia.")
            elif not basic_stats or basic_stats.get("total", 0) == 0:
                st.info("Necesitas flujos cargados para calcular la prevalencia.")
            else:
                prev_key = (
                    f"{event_key}|interval={int(interval_minutes)}|lead={int(lead_intervals)}"
                    f"|flow_mtime={flow_mtime or 0}"
                )
                progress.register(f"{engine_key}_prevalence")
                if prev_key in prev_cache:
                    prev_stats = prev_cache.get(prev_key)
                    progress.mark_done(
                        f"{engine_key}_prevalence", "Prevalencia (cache)"
                    )
                else:
                    prev_stats = None
                    acc_pl = pl.from_pandas(accident_stats["mapped_events"])
                    acc_pl = acc_pl.filter(pl.col("accidente_time").is_not_null())
                    if acc_pl.is_empty():
                        st.info("No hay accidentes con timestamp disponible.")
                        progress.mark_done(
                            f"{engine_key}_prevalence", "Prevalencia (sin datos)"
                        )
                    else:
                        acc_pl = acc_pl.with_columns(
                            pl.col("accidente_time")
                            .dt.truncate(f"{int(interval_minutes)}m")
                            .alias("ts_bin")
                        )
                        if lead_intervals:
                            acc_pl = acc_pl.with_columns(
                                (
                                    pl.col("ts_bin")
                                    - pl.duration(
                                        minutes=int(interval_minutes) * int(lead_intervals)
                                    )
                                ).alias("ts_bin")
                            )
                        flow_min = basic_stats.get("min_ts") if basic_stats else None
                        flow_max = basic_stats.get("max_ts") if basic_stats else None
                        if flow_min is not None and flow_max is not None:
                            acc_pl = acc_pl.filter(
                                (pl.col("ts_bin") >= pl.lit(flow_min))
                                & (pl.col("ts_bin") <= pl.lit(flow_max))
                            )
                        acc_windows = acc_pl.select(
                            [pl.col("ultimo_portico").alias("PORTICO"), pl.col("ts_bin")]
                        )
                        with st.spinner("Calculando prevalencia..."):
                            if engine_key == "polars":
                                prev_stats = _pl_flow_window_prevalence(
                                    acc_windows, int(interval_minutes)
                                )
                            else:
                                prev_stats = get_flow_window_prevalence_stats(
                                    acc_windows.to_pandas(),
                                    interval_minutes=int(interval_minutes),
                                    db_path=DB_PATH,
                                )
                        prev_cache[prev_key] = {
                            "total_windows": prev_stats.get("total_windows", 0),
                            "positive_windows": prev_stats.get("positive_windows", 0),
                            "interval": int(interval_minutes),
                            "lead": int(lead_intervals),
                        }
                        _save_desc_cache(cache)
                        progress.mark_done(f"{engine_key}_prevalence", "Prevalencia")

                if prev_stats:
                    total_windows = prev_stats.get("total_windows", 0)
                    positive_windows = prev_stats.get("positive_windows", 0)
                    prevalence = (
                        positive_windows / total_windows * 100.0 if total_windows else 0.0
                    )
                    ratio = (
                        round(total_windows / positive_windows, 2)
                        if positive_windows
                        else None
                    )
                    prevalence_df = pd.DataFrame(
                        [
                            {
                                "Cantidad": "Total ventanas (p,τ)",
                                "Valor": total_windows,
                                "Notas": "Despues de exclusiones y rango temporal",
                            },
                            {
                                "Cantidad": "Ventanas positivas",
                                "Valor": positive_windows,
                                "Notas": "Accidentes en ventana ajustada",
                            },
                            {
                                "Cantidad": "Prevalencia (%)",
                                "Valor": round(prevalence, 4),
                                "Notas": "Positivas / total",
                            },
                            {
                                "Cantidad": "Ratio (1 : N)",
                                "Valor": ratio if ratio is not None else "N/A",
                                "Notas": "Desbalance de clases",
                            },
                        ]
                    )
                    prevalence_df["Valor"] = prevalence_df["Valor"].astype(str)
                    st.dataframe(prevalence_df, width="stretch", hide_index=True)

            st.subheader("Estadisticas de flujos")
            temp_agg_path = ROOT_DIR / "Resultados" / "temp_agg.duckdb"
            temp_agg_path.parent.mkdir(parents=True, exist_ok=True)
            stats_key = f"min_speed={float(min_speed):.2f}"
            stats_entry = flow_stats_cache.setdefault(stats_key, {})
            table_task = f"{engine_key}_flow_stats_table_a"
            weekday_task = f"{engine_key}_flow_stats_weekday"
            hourly_task = f"{engine_key}_flow_stats_hourly"
            surface_task = f"{engine_key}_flow_stats_surface"
            progress.register(table_task)
            progress.register(weekday_task)
            progress.register(hourly_task)
            progress.register(surface_task)

            tables = _deserialize_table_list(stats_entry.get("table_a"))
            weekday_flow = _records_to_df(stats_entry.get("weekday_flow"))
            hourly_profile = _records_to_df(stats_entry.get("hourly_profile"))
            surface_df = _records_to_df(stats_entry.get("surface"))
            for col in ("dow_idx", "flow_avg"):
                if col in weekday_flow.columns:
                    weekday_flow[col] = pd.to_numeric(weekday_flow[col], errors="coerce")
            for col in ("hour", "total_flow", "avg_speed"):
                if col in hourly_profile.columns:
                    hourly_profile[col] = pd.to_numeric(hourly_profile[col], errors="coerce")
            for col in ("dow_idx", "minute_of_day", "flow_avg", "speed_avg", "density_avg"):
                if col in surface_df.columns:
                    surface_df[col] = pd.to_numeric(surface_df[col], errors="coerce")
            if "table_a" in stats_entry:
                progress.mark_done(table_task, "Tablas A (cache)")
            if "weekday_flow" in stats_entry:
                progress.mark_done(weekday_task, "Flujo semanal (cache)")
            if "hourly_profile" in stats_entry:
                progress.mark_done(hourly_task, "Perfil horario (cache)")
            if "surface" in stats_entry:
                progress.mark_done(surface_task, "Superficie 3D (cache)")

            needs_df_res = any(
                key not in stats_entry for key in ("table_a", "weekday_flow", "hourly_profile")
            )
            df_res = None
            if needs_df_res:
                with st.spinner("Procesando estadisticas de flujos..."):
                    if engine_key == "duckdb":
                        df_res = process_monthly_and_store_duckdb(
                            DB_PATH, float(min_speed), temp_agg_path
                        )
                    else:
                        df_res = process_monthly_and_store(
                            DB_PATH, float(min_speed), False, temp_agg_path
                        )
                if df_res is None or df_res.is_empty():
                    if "table_a" not in stats_entry:
                        stats_entry["table_a"] = []
                        progress.mark_done(table_task, "Tablas A (vacias)")
                    if "weekday_flow" not in stats_entry:
                        stats_entry["weekday_flow"] = _df_to_records(pd.DataFrame())
                        progress.mark_done(weekday_task, "Flujo semanal (vacio)")
                    if "hourly_profile" not in stats_entry:
                        stats_entry["hourly_profile"] = _df_to_records(pd.DataFrame())
                        progress.mark_done(hourly_task, "Perfil horario (vacio)")
                    _save_desc_cache(cache)
                else:
                    if "table_a" not in stats_entry:
                        tables = generate_table_a_polars(df_res)
                        stats_entry["table_a"] = _serialize_table_list(tables)
                        progress.mark_done(table_task, "Tablas A")
                    else:
                        progress.mark_done(table_task, "Tablas A (cache)")
                    if "weekday_flow" not in stats_entry:
                        weekday_flow = build_weekday_flow_lines(df_res)
                        stats_entry["weekday_flow"] = _df_to_records(weekday_flow)
                        progress.mark_done(weekday_task, "Flujo semanal")
                    else:
                        progress.mark_done(weekday_task, "Flujo semanal (cache)")
                    if "hourly_profile" not in stats_entry:
                        hourly_profile = _build_hourly_profile_plot_df(df_res)
                        stats_entry["hourly_profile"] = _df_to_records(hourly_profile)
                        progress.mark_done(hourly_task, "Perfil horario")
                    else:
                        progress.mark_done(hourly_task, "Perfil horario (cache)")
                    _save_desc_cache(cache)

            if "surface" not in stats_entry:
                with st.spinner("Calculando superficie 3D..."):
                    if engine_key == "duckdb":
                        surface_df = _compute_surface_duckdb(
                            DB_PATH, float(min_speed), 5, temp_agg_path
                        )
                    else:
                        surface_df = _compute_surface_polars(DB_PATH, float(min_speed), 5)
                stats_entry["surface"] = _df_to_records(surface_df)
                _save_desc_cache(cache)
                progress.mark_done(surface_task, "Superficie 3D")

            flow_stats_cache[stats_key] = stats_entry
            flow_stats_complete = _flow_stats_cache_complete(stats_entry)

            if tables:
                for table in tables:
                    title = table.get("title", "Tabla")
                    stats_df = table.get("stats", pd.DataFrame())
                    st.markdown(f"**{title}**")
                    if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
                        st.dataframe(stats_df, width="stretch", hide_index=True)
                    else:
                        st.info("Tabla sin datos.")
            else:
                st.info("No hay tablas de estadisticas disponibles.")

            if not hourly_profile.empty:
                fig_flow, fig_speed = _plot_hourly_profile_interactive(hourly_profile)
                if fig_flow is None:
                    st.info("Plotly no disponible para perfiles horarios.")
                else:
                    st.subheader("Perfil horario (flujo)")
                    st.plotly_chart(fig_flow, use_container_width=True)
                    st.subheader("Perfil horario (velocidad)")
                    st.plotly_chart(fig_speed, use_container_width=True)

            if not weekday_flow.empty:
                fig_line = _plot_weekday_flow_lines_interactive(weekday_flow)
                if fig_line is None:
                    st.info("Plotly no disponible para el grafico semanal.")
                else:
                    st.subheader("Flujo promedio por dia de la semana")
                    st.plotly_chart(fig_line, use_container_width=True)

            st.subheader("Superficie 3D (dia vs hora, resolucion 5-min)")
            axis_cols = st.columns(3)
            with axis_cols[0]:
                axis_x = st.selectbox(
                    "Eje X",
                    ["Dia de semana", "Hora del dia"],
                    index=0,
                    key="flow_stats_axis_x",
                )
            with axis_cols[1]:
                axis_y = st.selectbox(
                    "Eje Y",
                    ["Hora del dia", "Dia de semana"],
                    index=0,
                    key="flow_stats_axis_y",
                )
            with axis_cols[2]:
                axis_z = st.selectbox(
                    "Eje Z",
                    ["Flujo promedio", "Velocidad media", "Densidad media"],
                    index=0,
                    key="flow_stats_axis_z",
                )

            if surface_df is None or surface_df.empty:
                st.info("No hay datos suficientes para la superficie 3D.")
            else:
                if axis_x == axis_y:
                    st.info("Selecciona ejes distintos para graficar la superficie.")
                else:
                    figs = _plot_surface_by_category_interactive(
                        surface_df, x_var=axis_x, y_var=axis_y, z_var=axis_z
                    )
                    if not figs:
                        st.info("Plotly no disponible para las superficies.")
                    else:
                        for _, fig in figs:
                            st.plotly_chart(fig, use_container_width=True)

            cache_complete = (
                flow_complete
                and (acc_complete or not event_files)
                and flow_stats_complete
            )
            if cache_complete:
                if st.button("Recalcular (limpiar cache)", key="desc_recalc"):
                    cache = _empty_desc_cache()
                    _save_desc_cache(cache)
                    st.rerun()
            else:
                st.caption(
                    "Cache incompleto: se continuara desde el ultimo punto guardado."
                )


if __name__ == "__main__":
    main()
