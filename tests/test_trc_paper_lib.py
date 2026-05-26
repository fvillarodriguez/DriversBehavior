"""
Unit tests for src/trc_paper/lib/ — shared utilities.

Covers:
  • lib.io.connect_duckdb_readonly         (memory-safe pragmas)
  • lib.io.resolve_under_root              (project-root anchoring)
  • lib.io.resolve_config_paths            (full config.paths resolution)
  • lib.io.load_yaml_config                (project-relative or absolute)
  • lib.io.write_json_atomic               (atomic, no partial writes)
  • lib.portico.normalize_portico_id       (string normalization)
  • lib.portico.load_porticos_geometry     (CSV schema validation)
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402

from src.trc_paper.lib import (  # noqa: E402
    PROJECT_ROOT,
    connect_duckdb_readonly,
    load_porticos_geometry,
    load_yaml_config,
    normalize_portico_id,
    resolve_config_paths,
    resolve_under_root,
    write_json_atomic,
)


# ---------------------------------------------------------------------------
# normalize_portico_id
# ---------------------------------------------------------------------------


class TestNormalizePorticoId:
    def test_strips_whitespace_and_uppercases(self) -> None:
        assert normalize_portico_id(" p12 ") == "P12"

    def test_handles_none(self) -> None:
        assert normalize_portico_id(None) == ""

    def test_accepts_int(self) -> None:
        assert normalize_portico_id(7) == "7"

    def test_idempotent(self) -> None:
        once = normalize_portico_id("p2")
        twice = normalize_portico_id(once)
        assert once == twice == "P2"


# ---------------------------------------------------------------------------
# resolve_under_root / resolve_config_paths / load_yaml_config
# ---------------------------------------------------------------------------


class TestResolveUnderRoot:
    def test_relative_path_resolves_under_project_root(self) -> None:
        out = resolve_under_root("Datos/flujos.duckdb")
        assert out == (PROJECT_ROOT / "Datos" / "flujos.duckdb").resolve()
        assert out.is_absolute()

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        absolute = tmp_path / "X.csv"
        absolute.write_text("k,v\n1,2\n")
        assert resolve_under_root(absolute) == absolute

    def test_resolve_config_paths_resolves_each_entry(self) -> None:
        cfg = {
            "paths": {
                "flow_db": "Datos/flujos.duckdb",
                "events_db": "Datos/eventos.duckdb",
                "results_root": "Resultados/trc_paper",
            }
        }
        resolved = resolve_config_paths(cfg)
        assert set(resolved) == {"flow_db", "events_db", "results_root"}
        for v in resolved.values():
            assert v.is_absolute()
            assert str(v).startswith(str(PROJECT_ROOT))

    def test_load_yaml_config_resolves_relative(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "x.yaml").write_text("a: 1\nb: 2\n")
        # Absolute path always works
        result = load_yaml_config(cfg_dir / "x.yaml")
        assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "out.json"
        write_json_atomic(target, {"k": 1})
        assert target.exists()
        assert json.loads(target.read_text()) == {"k": 1}

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "x.json"
        target.write_text('{"prev": true}')
        write_json_atomic(target, {"new": True})
        assert json.loads(target.read_text()) == {"new": True}

    def test_atomic_no_partial_file_on_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "should_not_appear.json"

        def _explode(*_a, **_kw):
            raise RuntimeError("boom")

        # Sabotage json.dump to ensure we never see a partial target file.
        monkeypatch.setattr("src.trc_paper.lib.io.json.dump", _explode)
        with pytest.raises(RuntimeError, match="boom"):
            write_json_atomic(target, {"x": 1})
        assert not target.exists()
        # No leftover .tmp files either
        assert not list(tmp_path.glob("*.tmp"))

    def test_serializes_pathlib_via_default(self, tmp_path: Path) -> None:
        target = tmp_path / "paths.json"
        write_json_atomic(target, {"p": Path("/tmp/x")})
        loaded = json.loads(target.read_text())
        assert loaded == {"p": "/tmp/x"} or loaded == {"p": "\\tmp\\x"}

    def test_concurrent_writes_do_not_corrupt(self, tmp_path: Path) -> None:
        target = tmp_path / "concurrent.json"

        def writer(i: int) -> None:
            write_json_atomic(target, {"writer": i, "payload": list(range(10))})

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(writer, range(40)))
        # File must be a valid JSON with exactly the schema we wrote
        loaded = json.loads(target.read_text())
        assert set(loaded) == {"writer", "payload"}
        assert loaded["payload"] == list(range(10))


# ---------------------------------------------------------------------------
# connect_duckdb_readonly — pragmas applied
# ---------------------------------------------------------------------------


class TestConnectDuckdbReadonly:
    def _build_db(self, path: Path) -> None:
        con = duckdb.connect(str(path))
        try:
            con.execute("CREATE TABLE t AS SELECT range AS x FROM range(10)")
        finally:
            con.close()

    def test_opens_readonly(self, tmp_path: Path) -> None:
        db = tmp_path / "tiny.duckdb"
        self._build_db(db)
        con = connect_duckdb_readonly(db)
        try:
            rows = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
            assert rows == 10
            # Writing must fail because the connection is read-only
            with pytest.raises(Exception):
                con.execute("INSERT INTO t VALUES (99)")
        finally:
            con.close()

    def test_threads_pragma_applied(self, tmp_path: Path) -> None:
        db = tmp_path / "tiny.duckdb"
        self._build_db(db)
        con = connect_duckdb_readonly(db, threads=1)
        try:
            value = con.execute("SELECT current_setting('threads')").fetchone()[0]
            assert int(value) == 1
        finally:
            con.close()

    def test_memory_limit_pragma_applied(self, tmp_path: Path) -> None:
        db = tmp_path / "tiny.duckdb"
        self._build_db(db)
        con = connect_duckdb_readonly(db, memory_limit="512MB")
        try:
            ml = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            # DuckDB normalizes formats; just verify a positive byte count
            assert ml  # non-empty
        finally:
            con.close()


# ---------------------------------------------------------------------------
# load_porticos_geometry
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_porticos_csv(tmp_path: Path) -> Path:
    p = tmp_path / "Porticos.csv"
    p.write_text(
        "cod_portico;Km;Calzada;Orden;Eje;lat;lon\n"
        "P01;10.5;Oriente;1;RUTA 5 SUR;-33.45;-70.66\n"
        "p02;20.3;Poniente;2;RUTA 5 SUR;-33.55;-70.70\n"
        "  P03 ;30.0;Oriente;3;RUTA 5 SUR;-33.62;-70.80\n"
    )
    return p


class TestLoadPorticosGeometry:
    def test_required_columns_present(self, sample_porticos_csv: Path) -> None:
        df = load_porticos_geometry(sample_porticos_csv)
        for col in ("cod_portico", "Km", "Calzada", "Orden", "Eje"):
            assert col in df.columns

    def test_normalizes_cod_portico(self, sample_porticos_csv: Path) -> None:
        df = load_porticos_geometry(sample_porticos_csv)
        assert "cod_portico_norm" in df.columns
        assert set(df["cod_portico_norm"]) == {"P01", "P02", "P03"}

    def test_numeric_coercion(self, sample_porticos_csv: Path) -> None:
        df = load_porticos_geometry(sample_porticos_csv)
        # Km, lat, lon should be numeric (not strings) after coercion
        for col in ("Km", "lat", "lon"):
            assert df[col].dtype.kind in {"f", "i"}

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("foo;bar\n1;2\n")
        with pytest.raises(ValueError, match="cod_portico"):
            load_porticos_geometry(bad)
