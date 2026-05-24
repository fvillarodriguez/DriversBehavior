import gzip
import json

import pandas as pd
import pytest

from scripts.run_drift_yearly_variance_repetitions import (
    validate_yearly_roc_payload,
    write_variance_outputs,
)


def _roc_payload_pair(y_second=None):
    return [
        {
            "strategy": "static",
            "model": "XGBoost",
            "balance_mode": "none",
            "segment": "2019",
            "run_seed": 42,
            "run_order": 1,
            "y_true": [0, 1],
            "calibrated_scores": [0.2, 0.8],
        },
        {
            "strategy": "static",
            "model": "XGBoost",
            "balance_mode": "none",
            "segment": "2019",
            "run_seed": 43,
            "run_order": 2,
            "y_true": [0, 1] if y_second is None else y_second,
            "calibrated_scores": [0.4, 0.6],
        },
    ]


def test_validate_yearly_roc_payload_reports_nonzero_variance_for_repeated_seeds():
    report = validate_yearly_roc_payload(_roc_payload_pair(), expected_seeds=(42, 43))

    assert report["errors"] == []
    assert report["group_count"] == 1
    assert report["nonzero_variance_groups"] == 1
    assert report["groups"][0]["brier_score"] == pytest.approx(0.10)
    assert report["groups"][0]["bias2"] == pytest.approx(0.09)
    assert report["groups"][0]["variance"] == pytest.approx(0.01)
    assert report["groups"][0]["additive_residual"] == pytest.approx(0.0)


def test_validate_yearly_roc_payload_rejects_misaligned_y_true():
    report = validate_yearly_roc_payload(
        _roc_payload_pair(y_second=[1, 0]),
        expected_seeds=(42, 43),
    )

    assert any("y_true differs across seeds" in error for error in report["errors"])


def test_write_variance_outputs_persists_mean_tables_and_validation_report(tmp_path):
    mean_table = pd.DataFrame(
        [
            {
                "iteration": 1,
                "training_year": "2018",
                "prediction_year": 2019,
                "model": "XGBoost",
                "balance_mode": "none",
                "brier_score": 0.10,
                "bias2": 0.09,
                "variance": 0.01,
                "noise": 0.0,
                "n_repetitions": 2,
                "seed_list": "42,43",
            }
        ]
    )
    outputs = {
        "yearly_results": mean_table,
        "summary": mean_table,
        "appendix_tables": {"A.6": mean_table, "A.7": mean_table, "A.8": mean_table},
        "appendix_tables_mean": {"A.6": mean_table, "A.7": mean_table, "A.8": mean_table},
        "roc_payload": _roc_payload_pair(),
        "optuna_json_path": "/tmp/run.json",
        "checkpoint_run_dir": "/tmp/run",
    }

    report = write_variance_outputs(
        outputs,
        output_dir=tmp_path,
        seeds=(42, 43),
        source_json_path=tmp_path / "source.json",
        cache_report={"copied_tuning_artifacts": 0},
    )

    assert report["status"] == "ok"
    assert (tmp_path / "A6_5seeds.csv").exists()
    assert (tmp_path / "A6_by_seed_5seeds.csv").exists()
    assert (tmp_path / "summary_5seeds.csv").exists()
    assert (tmp_path / "validation_report.json").exists()
    with gzip.open(tmp_path / "roc_payload_5seeds.json.gz", "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert len(payload) == 2
