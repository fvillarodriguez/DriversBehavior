from __future__ import annotations

import sys
import types

optuna_stub = types.ModuleType("optuna")
optuna_stub.Study = object
optuna_stub.Trial = object
optuna_stub.TrialPruned = RuntimeError
optuna_stub.pruners = types.SimpleNamespace(
    BasePruner=object,
    NopPruner=object,
    HyperbandPruner=object,
    MedianPruner=object,
)
optuna_stub.samplers = types.SimpleNamespace(BaseSampler=object, TPESampler=object)
optuna_stub.trial = types.SimpleNamespace(
    FrozenTrial=object,
    TrialState=types.SimpleNamespace(
        COMPLETE="COMPLETE",
        PRUNED="PRUNED",
        FAIL="FAIL",
        RUNNING="RUNNING",
        WAITING="WAITING",
    ),
)
optuna_stub.create_study = lambda *args, **kwargs: None
sys.modules.setdefault("optuna", optuna_stub)

imblearn_stub = types.ModuleType("imblearn")
imblearn_over_sampling_stub = types.ModuleType("imblearn.over_sampling")
imblearn_over_sampling_stub.SMOTE = object
sys.modules.setdefault("imblearn", imblearn_stub)
sys.modules.setdefault("imblearn.over_sampling", imblearn_over_sampling_stub)

import src.cluster_accident_app as app


def test_history_model_metrics_dataframe_normalizes_flat_manifest_metrics():
    record = {
        "id": 10,
        "stage": "Modelos",
        "created_at": "2026-04-23T09:27:19",
        "model_name": "XGBoost",
        "threshold_objective": "far",
        "calibration_method": "sigmoid",
        "balance_strategy": "none",
        "tramo_label": "RUTA 5 SUR | Oriente | 3 -> 5",
        "features_path": "/tmp/features.duckdb",
        "metrics": {
            "mcc": 0.12,
            "pr_auc": 0.034,
            "recall": 0.56,
            "far": 0.14,
            "confusion_matrix": [[42, 7], [3, 5]],
        },
        "metadata": {
            "run_id": "20260423_092719_78e0fcab",
            "manifest": {
                "bundle_dir": "/tmp/model_history/20260423_092719_78e0fcab/base_cluster",
            },
        },
    }

    df = app._history_model_metrics_dataframe([record])

    assert len(df) == 1
    assert df.loc[0, "feature_set"] == "Base+Cluster"
    assert df.loc[0, "mcc"] == 0.12
    assert df.loc[0, "run_id"] == "20260423_092719_78e0fcab"
    assert isinstance(df.loc[0, "confusion_matrix"], str)


def test_history_model_metrics_dataframe_normalizes_grouped_feature_sets():
    record = {
        "id": 11,
        "stage": "Modelos",
        "created_at": "2026-04-23T10:00:00",
        "model_name": "Random Forest",
        "metrics": {
            "Base": {"mcc": 0.1, "pr_auc": 0.2},
            "Cluster": {"mcc": 0.3, "recall": 0.4},
            "Base + Cluster": {"mcc": 0.5, "far": 0.6},
        },
    }

    df = app._history_model_metrics_dataframe([record])

    assert set(df["feature_set"]) == {"Base", "Cluster", "Base+Cluster"}
    by_group = df.set_index("feature_set")
    assert by_group.loc["Base", "pr_auc"] == 0.2
    assert by_group.loc["Cluster", "recall"] == 0.4
    assert by_group.loc["Base+Cluster", "far"] == 0.6


def test_history_model_metrics_dataframe_normalizes_batch_candidates():
    record = {
        "id": 12,
        "stage": "Modelos",
        "created_at": "2026-04-23T11:00:00",
        "model_name": "SVM",
        "threshold_objective": "mcc",
        "calibration_method": "none",
        "balance_strategy": "optuna_batch",
        "metrics": {
            "subrun_cluster": {
                "candidate_a": {
                    "robust": {
                        "mcc": 0.27,
                        "recall": 0.71,
                        "threshold_protocol": "robust",
                    }
                }
            }
        },
        "metadata": {
            "history_entry": {
                "subruns": [
                    {
                        "subrun_id": "subrun_cluster",
                        "feature_set_label": "Cluster",
                        "candidates": [
                            {
                                "candidate_id": "candidate_a",
                                "candidate_label": "trial 7",
                                "feature_cols": ["cluster_speed_0"],
                            }
                        ],
                    }
                ]
            }
        },
    }

    df = app._history_model_metrics_dataframe([record])

    assert len(df) == 1
    assert df.loc[0, "feature_set"] == "Cluster"
    assert df.loc[0, "subrun"] == "subrun_cluster"
    assert df.loc[0, "candidate"] == "candidate_a"
    assert df.loc[0, "protocol"] == "robust"
    assert df.loc[0, "mcc"] == 0.27


def test_history_model_metrics_dataframe_reads_protocol_results_from_metadata():
    record = {
        "id": 314,
        "stage": "Modelos",
        "created_at": "2026-04-23T22:34:29",
        "model_name": "XGBoost",
        "threshold_objective": "far",
        "calibration_method": "isotonic",
        "balance_strategy": "optuna_batch",
        "metrics": {},
        "metadata": {
            "run_id": "20260423_223429_batch",
            "history_entry": {
                "subruns": [
                    {
                        "subrun_id": "optuna_record_308",
                        "feature_set_label": "Base",
                        "candidates": [
                            {
                                "candidate_id": "pareto_1",
                                "candidate_label": "Pareto 1",
                                "feature_cols": ["speed_light", "density_light"],
                            }
                        ],
                    }
                ],
                "protocol_results": {
                    "optuna_record_308": {
                        "pareto_1": {
                            "conservative": {
                                "mcc": 0.028,
                                "pr_auc": 0.015,
                                "recall": 0.47,
                                "threshold_protocol": "conservative",
                            },
                            "robust": {
                                "mcc": 0.027,
                                "pr_auc": 0.032,
                                "recall": 0.43,
                                "threshold_protocol": "robust",
                            },
                        }
                    }
                },
            },
        },
    }

    df = app._history_model_metrics_dataframe([record])

    assert len(df) == 2
    assert set(df["feature_set"]) == {"Base"}
    assert set(df["protocol"]) == {"conservative", "robust"}
    assert set(df["candidate"]) == {"pareto_1"}
    assert df["pr_auc"].max() == 0.032
