"""I/O helpers shared across pipeline scripts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import yaml

# Project root is the directory that contains Datos/, Resultados/, and src/.
# This file lives at src/trc_paper/lib/io.py — three parents up reaches the root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_under_root(relative: str | Path) -> Path:
    """Resolve a config path string against the project root.

    Examples:
        resolve_under_root("Datos/flujos.duckdb") →
            /Volumes/felipe/Desktop/Tesis/Datos/flujos.duckdb
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Read a YAML configuration file. Path may be absolute or project-relative."""
    target = Path(path)
    if not target.is_absolute() and not target.exists():
        target = resolve_under_root(path)
    with open(target, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_config_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve every path declared under the top-level `paths:` key."""
    paths = cfg.get("paths", {})
    return {k: resolve_under_root(v) for k, v in paths.items()}


def connect_duckdb_readonly(
    path: Path,
    *,
    memory_limit: str = "6GB",
    threads: int = 2,
    temp_directory: str | None = None,
    max_temp_directory_size: str = "60GB",
) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection with memory-safe defaults.

    The flujos.duckdb table is ~39 GB and aggregations easily exhaust the
    default temp directory and per-thread memory. These pragmas keep large
    queries within bounds at the cost of single-query speed.
    """
    con = duckdb.connect(str(path), read_only=True)
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    con.execute("SET preserve_insertion_order=false")
    if temp_directory is not None:
        con.execute(f"SET temp_directory='{temp_directory}'")
    con.execute(f"SET max_temp_directory_size='{max_temp_directory_size}'")
    return con


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
