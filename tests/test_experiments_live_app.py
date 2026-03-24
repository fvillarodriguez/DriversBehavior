import json
from pathlib import Path

import pandas as pd

import src.experiments_live_app as live_app


def test_build_live_sources_includes_drift_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(live_app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_app,
        "DRIFT_RUNS_DIR",
        tmp_path / "drift_recalibration_runs",
    )

    run_dir = live_app.DRIFT_RUNS_DIR / "run_abc123"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run_abc123",
        "status": "running",
        "updated_at": "2026-03-22T10:00:00",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    sources = live_app._build_live_sources()

    assert len(sources) == 1
    assert sources[0]["type"] == "drift_recalibration"
    assert "run_abc123" in str(sources[0]["label"])


def test_read_drift_run_builds_partial_monitoring_frames(tmp_path):
    run_dir = tmp_path / "drift_recalibration_runs" / "run_xyz789"
    blocks_dir = run_dir / "blocks"
    tuning_dir = run_dir / "tuning"
    blocks_dir.mkdir(parents=True)
    tuning_dir.mkdir(parents=True)

    manifest = {
        "run_id": "run_xyz789",
        "status": "running",
        "started_at": "2026-03-22T10:00:00",
        "updated_at": "2026-03-22T10:05:00",
        "progress": {
            "completed_units": 1.5,
            "total_units": 4,
            "completed_tuning_tasks": 1,
            "total_tuning_tasks": 2,
            "completed_blocks": 1,
            "total_blocks": 2,
        },
        "block_index": {
            "block_a": {
                "status": "completed",
                "strategy": "static",
                "model": "Random Forest",
                "balance_mode": "none",
                "run_seed": 11,
                "run_order": 1,
            },
            "block_b": {
                "status": "pending",
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "smote",
                "run_seed": 11,
                "run_order": 1,
            },
        },
        "tuning_index": {
            "tune_a": {
                "status": "completed",
                "filename": "tune_a.json",
                "model_name": "Random Forest",
                "balance_mode": "none",
            }
        },
        "smote_index": {"smote_a": {"status": "available"}},
        "global_execution_log": [
            {
                "order": 1,
                "timestamp": "2026-03-22T10:00:00",
                "phase": "global_tuning_start",
                "status": "started",
                "rss_before_mb": 512.0,
            }
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    block_payload = {
        "saved_at": "2026-03-22T10:04:00",
        "block_id": "block_a",
        "block": {
            "strategy": "static",
            "model": "Random Forest",
            "balance_mode": "none",
        },
        "run_seed": 11,
        "run_order": 1,
        "yearly_rows": [
            {
                "strategy": "static",
                "model": "Random Forest",
                "balance_mode": "none",
                "auc": 0.81,
                "sensitivity": 0.72,
                "specificity": 0.78,
                "error_rate": 0.19,
                "training_time_sec": 0.4,
                "run_seed": 11,
            }
        ],
        "adaptive_rows": [
            {
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "smote",
                "auc": 0.79,
                "sensitivity": 0.70,
                "specificity": 0.76,
                "error_rate": 0.21,
                "training_time_sec": 0.6,
                "run_seed": 11,
            }
        ],
        "execution_log": [
            {
                "timestamp": "2026-03-22T10:04:00",
                "phase": "block_complete",
                "status": "ok",
                "training_time_sec": 0.4,
                "rss_after_mb": 620.0,
            }
        ],
        "roc_payload": [
            {
                "strategy": "static",
                "model": "Random Forest",
                "balance_mode": "none",
                "segment": "2019",
                "y_true": [0, 1],
                "scores": [0.1, 0.9],
                "run_seed": 11,
                "run_order": 1,
            }
        ],
        "tuning_refs": ["tune_a"],
        "smote_refs": [],
    }
    (blocks_dir / "block_a.json").write_text(
        json.dumps(block_payload),
        encoding="utf-8",
    )

    tuning_artifact = {
        "study_id": "study_rf_none",
        "model_name": "Random Forest",
        "balance_mode": "none",
        "stage": "global_tuning",
        "trials": [
            {
                "trial_number": 0,
                "state": "COMPLETE",
                "cv_auc": 0.74,
            },
            {
                "trial_number": 1,
                "state": "COMPLETE",
                "cv_auc": 0.81,
            },
        ],
    }
    (tuning_dir / "tune_a.json").write_text(
        json.dumps(tuning_artifact),
        encoding="utf-8",
    )

    live_status = {
        "timestamp": "2026-03-22T10:05:00",
        "status": "running",
        "completed_units": 1.5,
        "total_units": 4,
        "progress_ratio": 0.375,
        "label": "Ejecutando bloque de experimentos...",
        "detail": "Seed 11 | Strategy adaptive_adwin",
        "context": {
            "phase": "block_running",
            "strategy": "adaptive_adwin",
            "model": "Random Forest",
            "run_seed": 11,
        },
    }
    (run_dir / "live_status.json").write_text(
        json.dumps(live_status),
        encoding="utf-8",
    )
    (run_dir / "live_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-22T10:00:00",
                        "completed_units": 0.0,
                        "total_units": 4,
                        "progress_ratio": 0.0,
                        "label": "Iniciando",
                        "detail": "",
                        "context": {},
                    }
                ),
                json.dumps(live_status),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = live_app._read_drift_run(manifest_path)

    assert isinstance(payload["block_df"], pd.DataFrame)
    assert len(payload["block_df"]) == 2
    assert not payload["summary_df"].empty
    assert set(payload["summary_df"]["strategy"].unique()) == {"static", "adaptive_adwin"}
    assert "n_segments" in payload["summary_df"].columns
    assert "n_repetitions" in payload["summary_df"].columns
    assert not payload["tuning_trials_df"].empty
    assert not payload["memory_trace_df"].empty
    assert not payload["live_events_df"].empty
    assert not payload["average_roc_df"].empty
    assert payload["live_events_df"]["progress_pct"].iloc[-1] == 37.5


def test_build_drift_live_result_tables_fill_pending_slots():
    manifest = {
        "run_manifest": {
            "strategies": ["static", "adaptive_adwin"],
            "models": ["Random Forest"],
            "balance_modes": ["none", "smote"],
            "repetition_seeds": [11],
            "base_year": 2018,
            "prediction_years": [2019, 2020],
        }
    }
    yearly_df = pd.DataFrame(
        [
            {
                "strategy": "static",
                "iteration": 1,
                "training_year": "2018",
                "prediction_year": 2019,
                "model": "Random Forest",
                "balance_mode": "none",
                "auc": 0.81,
                "sensitivity": 0.72,
                "specificity": 0.78,
                "error_rate": 0.19,
                "training_time_sec": 0.4,
                "threshold": 0.5,
                "n_train": 10,
                "n_test": 10,
                "run_seed": 11,
                "run_order": 1,
            }
        ]
    )
    adaptive_df = pd.DataFrame()
    block_df = pd.DataFrame(
        [
            {
                "status": "pending",
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "smote",
                "detector_variant": "",
                "run_seed": 11,
                "run_order": 1,
            }
        ]
    )

    tables = live_app._build_drift_live_result_tables(
        manifest,
        block_df,
        yearly_df,
        adaptive_df,
    )

    assert len(tables["A.6"]) == 4
    assert tables["A.6"]["status"].tolist().count("Completado") == 1
    assert tables["A.6"]["status"].tolist().count("Pendiente") == 3
    pending_row = tables["A.6"].loc[
        (tables["A.6"]["prediction_year"] == 2020)
        & (tables["A.6"]["balance_mode"] == "none")
    ].iloc[0]
    assert pending_row["auc"] == "Pendiente"
    assert pending_row["training_year"] == "2018"

    assert len(tables["A.9"]) == 1
    assert tables["A.9"].iloc[0]["status"] == "Pendiente"
    assert tables["A.9"].iloc[0]["model"] == "Random Forest"
