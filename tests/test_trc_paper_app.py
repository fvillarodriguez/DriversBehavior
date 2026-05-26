"""
Tests for src/trc_paper_app.py — the Streamlit page that drives the pipeline.

These tests deliberately avoid spawning Streamlit. They cover the data layer:
  • PIPELINE definition completeness
  • JobStatus dataclass and running detection
  • _persist_job / _load_job round-trip (atomic PID file)
  • _run_tag formatting
  • _results_root_for resolution
  • _format_age strings
  • _status_icon transitions
  • _tail_log returns trailing lines without loading full file
  • Each step's command builder returns a well-formed CLI list with absolute paths
  • _output_path matches what each builder declares
  • _load_yaml / _save_yaml round-trip
  • Variant K=5 vs K=8 produces different results roots
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.trc_paper_app as app  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


class TestPipelineDefinition:
    def test_pipeline_has_expected_steps(self) -> None:
        keys = [s.key for s in app.PIPELINE]
        expected = [
            "validate", "dynamic_gmm", "entropy", "markov",
            "homogeneity", "stationary", "covid", "events", "integration",
        ]
        assert keys == expected

    def test_every_step_has_builder(self) -> None:
        for step in app.PIPELINE:
            assert step.builder is not None, f"step {step.key} missing builder"

    def test_every_step_has_output_template(self) -> None:
        for step in app.PIPELINE:
            assert "{run_tag}" in step.output_template, f"step {step.key} missing run_tag placeholder"

    def test_long_running_flags_only_on_expected_steps(self) -> None:
        long_keys = {s.key for s in app.PIPELINE if s.long_running}
        # GMM regeneration and Markov bootstrap are the slow ones
        assert "dynamic_gmm" in long_keys
        assert "markov" in long_keys


class TestModuleConstants:
    def test_paths_under_project_root(self) -> None:
        assert app.PACKAGE_DIR.exists()
        assert app.CONFIG_DEFAULT.exists()
        assert app.CONFIG_K8.exists()
        assert app.MANUSCRIPT_DIR.exists()
        # Results root may not yet exist; just check that the path is under repo root
        assert str(app.RESULTS_ROOT).startswith(str(app.ROOT_DIR))
        assert str(app.RESULTS_ROOT_K8).startswith(str(app.ROOT_DIR))


# ---------------------------------------------------------------------------
# JobStatus / PID file management
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_running_false_when_pid_none(self) -> None:
        s = app.JobStatus(name="x")
        assert s.running is False

    def test_running_self_process_true(self) -> None:
        s = app.JobStatus(name="x", pid=os.getpid())
        assert s.running is True

    def test_running_false_for_unlikely_pid(self) -> None:
        s = app.JobStatus(name="x", pid=2**30)
        assert s.running is False


class TestJobPersistence:
    def test_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app, "PID_DIR", tmp_path)
        original = app.JobStatus(
            name="myjob",
            pid=os.getpid(),
            started_at=datetime(2026, 5, 25, 12, 0, 0),
            log_path=tmp_path / "x.log",
            cmd=["python", "x.py"],
        )
        app._persist_job(original)
        loaded = app._load_job("myjob")
        assert loaded is not None
        assert loaded.name == "myjob"
        assert loaded.pid == os.getpid()
        assert loaded.cmd == ["python", "x.py"]
        assert loaded.log_path == tmp_path / "x.log"
        assert loaded.started_at == datetime(2026, 5, 25, 12, 0, 0)

    def test_load_missing_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app, "PID_DIR", tmp_path)
        assert app._load_job("no_such_job") is None

    def test_load_corrupt_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app, "PID_DIR", tmp_path)
        (tmp_path / "bad.json").write_text("not json")
        assert app._load_job("bad") is None

    def test_dead_pid_recorded_as_unknown_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app, "PID_DIR", tmp_path)
        log = tmp_path / "x.log"
        log.write_text("done\n")
        s = app.JobStatus(
            name="dead",
            pid=2**30,  # virtually never assigned
            started_at=datetime.now() - timedelta(minutes=5),
            log_path=log,
        )
        app._persist_job(s)
        loaded = app._load_job("dead")
        assert loaded is not None
        assert loaded.exit_code == -1
        assert loaded.finished_at is not None


# ---------------------------------------------------------------------------
# Config / variant helpers
# ---------------------------------------------------------------------------


class TestVariantHelpers:
    def test_run_tag_basic(self) -> None:
        cfg = {"dynamic_gmm": {"k": 5, "date_start": "2018-01-01", "date_end": "2024-09-30"}}
        assert app._run_tag(cfg) == "k5_2018-01-01_2024-09-30"

    def test_run_tag_k8(self) -> None:
        cfg = {"dynamic_gmm": {"k": 8, "date_start": "2019-01-01", "date_end": "2020-12-31"}}
        assert app._run_tag(cfg) == "k8_2019-01-01_2020-12-31"

    def test_results_root_resolves_under_project(self) -> None:
        cfg = {"paths": {"results_root": "Resultados/trc_paper"}}
        out = app._results_root_for(cfg)
        assert out == (app.ROOT_DIR / "Resultados" / "trc_paper").resolve()


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


class TestYamlHelpers:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "x.yaml"
        data = {"dynamic_gmm": {"k": 5}, "paths": {"flow_db": "Datos/flujos.duckdb"}}
        app._save_yaml(path, data)
        loaded = app._load_yaml(path)
        assert loaded == data

    def test_load_missing_returns_empty_dict(self, tmp_path: Path) -> None:
        assert app._load_yaml(tmp_path / "missing.yaml") == {}


# ---------------------------------------------------------------------------
# Output path / age / status icon
# ---------------------------------------------------------------------------


class TestOutputPath:
    @pytest.mark.parametrize("step_key", [s.key for s in app.PIPELINE])
    def test_output_path_is_under_results(self, step_key: str) -> None:
        step = next(s for s in app.PIPELINE if s.key == step_key)
        cfg = {
            "dynamic_gmm": {"k": 5, "date_start": "2018-01-01", "date_end": "2024-09-30"},
            "paths": {"results_root": "Resultados/trc_paper"},
        }
        path = app._output_path(cfg, step)
        assert str(path).startswith(str(app._results_root_for(cfg)))
        assert "k5_2018-01-01_2024-09-30" in str(path)


class TestFormatAge:
    def test_missing_path(self, tmp_path: Path) -> None:
        assert app._format_age(tmp_path / "no.file") == "—"

    def test_recent_file(self, tmp_path: Path) -> None:
        p = tmp_path / "x"
        p.write_text("hi")
        age = app._format_age(p)
        assert age in {"ahora"} or age.startswith("hace ")


class TestStatusIcon:
    def test_running_overrides_everything(self) -> None:
        icon = app._status_icon(running=True, output_exists=True, exit_code=0)
        assert "hourglass" in icon

    def test_failure_marked(self) -> None:
        icon = app._status_icon(running=False, output_exists=False, exit_code=1)
        assert "error" in icon

    def test_success_when_output_exists(self) -> None:
        icon = app._status_icon(running=False, output_exists=True, exit_code=0)
        assert "check_circle" in icon

    def test_pending_when_no_output_and_no_run(self) -> None:
        icon = app._status_icon(running=False, output_exists=False, exit_code=None)
        assert "pending" in icon


# ---------------------------------------------------------------------------
# _tail_log
# ---------------------------------------------------------------------------


class TestTailLog:
    def test_empty_when_missing(self, tmp_path: Path) -> None:
        assert app._tail_log(tmp_path / "x.log") == ""

    def test_returns_last_n_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "x.log"
        p.write_text("\n".join(f"line {i}" for i in range(100)))
        tail = app._tail_log(p, n_lines=10)
        assert tail.splitlines()[-1] == "line 99"
        assert len(tail.splitlines()) == 10

    def test_handles_huge_file_efficiently(self, tmp_path: Path) -> None:
        # 200 KB synthetic log (>64KB window the implementation uses)
        p = tmp_path / "big.log"
        with open(p, "w") as fh:
            for i in range(20_000):
                fh.write(f"line {i:08d}\n")
        tail = app._tail_log(p, n_lines=3)
        last_line = tail.splitlines()[-1]
        assert last_line == "line 00019999"


# ---------------------------------------------------------------------------
# Command builders — they must produce well-formed CLI lists
# ---------------------------------------------------------------------------


@pytest.fixture
def base_cfg() -> dict:
    return {
        "dynamic_gmm": {
            "k": 5,
            "date_start": "2018-01-01",
            "date_end": "2024-09-30",
            "parallel_jobs": 4,
        },
        "paths": {
            "flow_db": "Datos/flujos.duckdb",
            "porticos_csv": "Datos/Porticos.csv",
            "events_db": "Datos/eventos.duckdb",
            "results_root": "Resultados/trc_paper",
        },
        "markov": {
            "step": "1W",
            "subpopulation": "frequent",
            "bootstrap": {"n_replicas": 1000},
        },
    }


class TestCommandBuilders:
    """Sanity-check that every builder emits a `python <script.py> --flag value …` CLI."""

    @pytest.mark.parametrize("step_key", [s.key for s in app.PIPELINE])
    def test_builder_returns_list_with_script_path(self, base_cfg: dict, step_key: str) -> None:
        step = next(s for s in app.PIPELINE if s.key == step_key)
        run_tag = app._run_tag(base_cfg)
        log_path = app.LOGS_ROOT / f"{step_key}_{run_tag}.log"
        cmd = step.builder(base_cfg, {}, log_path, run_tag)
        assert isinstance(cmd, list)
        assert cmd[0] == app.PYTHON_BIN
        # Second element is the script path under src/trc_paper/
        script_path = Path(cmd[1])
        assert script_path.name == step.script
        assert script_path.exists(), f"{script_path} missing"

    def test_validate_cmd_includes_required_flags(self, base_cfg: dict) -> None:
        step = next(s for s in app.PIPELINE if s.key == "validate")
        cmd = step.builder(base_cfg, {}, Path("/tmp/x.log"), app._run_tag(base_cfg))
        for flag in ("--flow-db", "--porticos-csv", "--events-db",
                     "--date-start", "--date-end", "--output"):
            assert flag in cmd

    def test_dynamic_gmm_cmd_carries_k_and_config(self, base_cfg: dict) -> None:
        step = next(s for s in app.PIPELINE if s.key == "dynamic_gmm")
        cmd = step.builder(base_cfg, {}, Path("/tmp/x.log"), app._run_tag(base_cfg))
        k_idx = cmd.index("--k")
        assert cmd[k_idx + 1] == "5"
        config_idx = cmd.index("--config")
        assert Path(cmd[config_idx + 1]).name == "default.yaml"

    def test_dynamic_gmm_cmd_uses_k8_config_when_k8(self, base_cfg: dict) -> None:
        base_cfg["dynamic_gmm"]["k"] = 8
        step = next(s for s in app.PIPELINE if s.key == "dynamic_gmm")
        cmd = step.builder(base_cfg, {}, Path("/tmp/x.log"), app._run_tag(base_cfg))
        config_idx = cmd.index("--config")
        assert Path(cmd[config_idx + 1]).name == "k8_sensitivity.yaml"

    def test_markov_cmd_includes_step_and_bootstrap(self, base_cfg: dict) -> None:
        step = next(s for s in app.PIPELINE if s.key == "markov")
        cmd = step.builder(base_cfg, {}, Path("/tmp/x.log"), app._run_tag(base_cfg))
        assert cmd[cmd.index("--step") + 1] == "1W"
        assert cmd[cmd.index("--subpopulation") + 1] == "frequent"
        assert cmd[cmd.index("--bootstrap-replicas") + 1] == "1000"

    @pytest.mark.parametrize("step_key", [s.key for s in app.PIPELINE])
    def test_builder_paths_are_absolute(self, base_cfg: dict, step_key: str) -> None:
        step = next(s for s in app.PIPELINE if s.key == step_key)
        cmd = step.builder(base_cfg, {}, Path("/tmp/x.log"), app._run_tag(base_cfg))
        # Find any --output or --output-* flags and verify their values are absolute
        for i, token in enumerate(cmd):
            if token.startswith("--output") and i + 1 < len(cmd):
                value = cmd[i + 1]
                assert Path(value).is_absolute(), f"{token}={value} should be absolute"


# ---------------------------------------------------------------------------
# K=8 variant produces distinct results root
# ---------------------------------------------------------------------------


class TestVariantRouting:
    def test_k8_uses_separate_results_root(self) -> None:
        cfg_k8 = app._load_yaml(app.CONFIG_K8) or {}
        assert cfg_k8["paths"]["results_root"] == "Resultados/trc_paper_k8"
        assert cfg_k8["dynamic_gmm"]["k"] == 8
