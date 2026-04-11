import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.experiments.drift_sweep as drift_sweep
import src.neural_drift_experiments as nde
import src.Neural_drift_app as neural_drift_app


def _experiment_bundle() -> dict:
    timestamps = pd.to_datetime(
        [
            *pd.date_range("2018-01-01", "2024-09-01", freq="30D").tolist(),
            pd.Timestamp("2024-09-30 23:55:00"),
        ]
    )
    df = pd.DataFrame(
        {
            "interval_start": timestamps,
            "flow_light": np.linspace(10.0, 20.0, len(timestamps)),
            "flow_heavy": np.linspace(5.0, 15.0, len(timestamps)),
            "target": np.asarray(([0, 1] * (len(timestamps) // 2 + 1))[: len(timestamps)], dtype=int),
        }
    )
    return {
        "source": "test",
        "df": df,
        "feature_cols": ["flow_light", "flow_heavy"],
        "selection_metadata": {},
        "feature_export_path": "",
    }


def _records_for_protocol(seed: int) -> pd.DataFrame:
    rows = []
    month_starts = pd.date_range("2019-01-01", "2024-09-01", freq="MS")
    for idx, month_start in enumerate(month_starts):
        rows.append(
            {
                "timestamp": month_start + pd.Timedelta(hours=1),
                "y_true": 0,
                "score": 0.10,
                "prediction": 0,
                "decision_threshold": 0.5,
                "action_taken": "none",
            }
        )
        rows.append(
            {
                "timestamp": month_start + pd.Timedelta(hours=2),
                "y_true": 1,
                "score": min(0.99, 0.80 + 0.001 * float(seed) + 0.0001 * idx),
                "prediction": 1,
                "decision_threshold": 0.5,
                "action_taken": "retrain" if idx % 6 == 0 else "none",
            }
        )
    return pd.DataFrame(rows)


def _fake_baseline_seed_run(seed: int) -> dict:
    records = _records_for_protocol(seed)
    payload = nde._evaluate_records(records)
    return {
        "seed": int(seed),
        "config": {"seed": int(seed)},
        "dev": payload["dev"],
        "holdout": payload["holdout"],
        "records": payload["records"],
    }


def test_validate_experiment_dataset_accepts_full_protocol():
    context = nde._validate_experiment_dataset(_experiment_bundle())

    assert context["full_coverage"] is True
    assert context["rows_total"] > 0
    assert pd.Timestamp(context["min_timestamp"]) <= nde.BASE_START
    assert pd.Timestamp(context["max_timestamp"]).normalize() >= nde.HOLDOUT_END.normalize()


def test_monthly_metrics_and_score_use_calendar_months():
    records = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2019-01-05 00:00:00"),
                "y_true": 0,
                "score": 0.1,
                "prediction": 0,
                "decision_threshold": 0.5,
                "action_taken": "none",
            },
            {
                "timestamp": pd.Timestamp("2019-01-05 01:00:00"),
                "y_true": 1,
                "score": 0.9,
                "prediction": 1,
                "decision_threshold": 0.5,
                "action_taken": "retrain",
            },
            {
                "timestamp": pd.Timestamp("2019-02-05 00:00:00"),
                "y_true": 0,
                "score": 0.2,
                "prediction": 0,
                "decision_threshold": 0.5,
                "action_taken": "none",
            },
            {
                "timestamp": pd.Timestamp("2019-02-05 01:00:00"),
                "y_true": 1,
                "score": 0.8,
                "prediction": 1,
                "decision_threshold": 0.5,
                "action_taken": "none",
            },
        ]
    )

    monthly = nde._monthly_metrics_from_records(records)
    score = nde._build_experiment_score(monthly)

    assert monthly["month"].dt.strftime("%Y-%m").tolist() == ["2019-01", "2019-02"]
    assert monthly["n_actions"].tolist() == [1, 0]
    assert score["monthly_pr_auc_median"] == pytest.approx(1.0)
    assert score["action_cost"] == pytest.approx(0.5)
    assert score["stability_penalty"] == pytest.approx(0.0)
    assert score["score"] == pytest.approx(0.975)


def test_pairwise_stat_row_passes_gate_for_consistent_holdout_gain():
    months = pd.date_range("2023-01-01", periods=18, freq="MS")
    left = pd.DataFrame({"month": months, "pr_auc": np.full(len(months), 0.62)})
    right = pd.DataFrame({"month": months, "pr_auc": np.full(len(months), 0.56)})

    row = nde._pairwise_stat_row("neural", "cumulative", left, right)

    assert row["n_months"] == 18
    assert row["wilcoxon_p"] < nde.PAIRWISE_ALPHA
    assert row["delta_median"] == pytest.approx(0.06)
    assert row["ci_low"] > 0.0
    assert row["passes_gate"] is True


def test_run_experiment_plan_persists_and_resumes_baseline_only(tmp_path, monkeypatch: pytest.MonkeyPatch):
    bundle = _experiment_bundle()
    progress_events: list[dict] = []
    monkeypatch.setattr(nde, "NEURAL_DRIFT_EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(nde, "_evaluate_cumulative_baseline", lambda dataset_bundle, balance_mode, seed, fixed_params=None: _fake_baseline_seed_run(int(seed)))

    result = nde.run_experiment_plan(
        bundle,
        base_config={"balance_modes": ["none"]},
        balance_mode="none",
        selected_study=nde.STUDY_CUMULATIVE,
        selected_phase="all",
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )

    manifest_path = Path(result["manifest_path"])
    assert result["run_id"]
    assert manifest_path.exists()
    assert (manifest_path.parent / "artifacts" / "leaderboard_dev.csv").exists()
    assert result["leaderboard_dev"]["label"].tolist() == ["cumulative"]
    assert progress_events
    assert progress_events[-1]["label"] == "Experimento completado"
    assert progress_events[-1]["progress_ratio"] == pytest.approx(1.0)

    resumed = nde.run_experiment_plan(
        bundle,
        base_config={"balance_modes": ["none"]},
        balance_mode="none",
        selected_study=nde.STUDY_CUMULATIVE,
        selected_phase="all",
        resume_run_id=str(result["run_id"]),
    )

    persisted = nde._load_persisted_run(manifest_path)
    listed = nde._list_persisted_runs(tmp_path)

    assert resumed["run_id"] == result["run_id"]
    assert persisted["leaderboard_dev"]["label"].tolist() == ["cumulative"]
    assert listed[0]["run_id"] == result["run_id"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["result_status"] == "success"


def test_drift_sweep_run_cli_resolves_dataset_and_returns_paths(monkeypatch: pytest.MonkeyPatch):
    bundle = _experiment_bundle()
    captured: dict = {}

    def fake_resolve_dataset_from_context(context):
        captured["context"] = dict(context)
        return bundle

    def fake_run_experiment_plan(dataset_bundle, *, base_config, balance_mode, selected_study, selected_phase, resume_run_id=None):
        captured["dataset_bundle"] = dataset_bundle
        captured["base_config"] = dict(base_config)
        captured["balance_mode"] = balance_mode
        captured["selected_study"] = selected_study
        captured["selected_phase"] = selected_phase
        captured["resume_run_id"] = resume_run_id
        return {
            "run_id": "run_cli_demo",
            "manifest_path": "/tmp/run_cli_demo/manifest.json",
        }

    monkeypatch.setattr(neural_drift_app, "resolve_dataset_from_context", fake_resolve_dataset_from_context)
    monkeypatch.setattr(nde, "run_experiment_plan", fake_run_experiment_plan)

    payload = drift_sweep.run_cli(
        study=nde.STUDY_CUMULATIVE,
        phase="all",
        balance_mode="none",
        feature_export="Resultados/drift_flow_features_20260320_165122.duckdb",
        resume_run_id="run_resume",
    )

    assert captured["context"]["feature_export_path"].endswith("drift_flow_features_20260320_165122.duckdb")
    assert captured["dataset_bundle"] is bundle
    assert captured["base_config"]["balance_modes"] == list(neural_drift_app.DEFAULT_CONFIG["balance_modes"])
    assert captured["selected_study"] == nde.STUDY_CUMULATIVE
    assert captured["resume_run_id"] == "run_resume"
    assert payload["run_id"] == "run_cli_demo"
    assert payload["artifacts_dir"].endswith("/tmp/run_cli_demo/artifacts")
