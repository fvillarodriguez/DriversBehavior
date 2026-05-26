"""Shared utilities for the TRC paper pipeline."""

from .io import (
    PROJECT_ROOT,
    connect_duckdb_readonly,
    load_yaml_config,
    resolve_config_paths,
    resolve_under_root,
    write_json_atomic,
)
from .portico import load_porticos_geometry, normalize_portico_id

__all__ = [
    "PROJECT_ROOT",
    "connect_duckdb_readonly",
    "load_yaml_config",
    "resolve_config_paths",
    "resolve_under_root",
    "write_json_atomic",
    "load_porticos_geometry",
    "normalize_portico_id",
]
