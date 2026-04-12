from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.cluster_accident_app as app
from src.model_training import train_model
from src.model_xai import compute_xai_report
from tests.pipeline_helpers import build_synthetic_base_df


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}


def test_record_experiment_history_persists_base_cluster_xai_bundle(
    tmp_path, monkeypatch
):
    pytest.importorskip("streamlit")
    pytest.importorskip("shap")
    base_df, feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "flow_features_path": str(tmp_path / "flow_features.duckdb"),
            "flow_features_source": "duckdb",
            "cluster_features_path": str(tmp_path / "cluster_features.duckdb"),
            "cluster_features_source": "calculadas",
            "cluster_features_df": features_df,
            "cluster_choice": "cluster_kmeans_k2.csv",
            "selected_features": feature_cols,
            "feature_selection_store": {},
            "feature_importances_df": None,
            "optuna_results_store": {},
            "optuna_active_key": None,
            "balanced_base_df": None,
            "balanced_cluster_df": None,
            "balance_last_stats": None,
            "balance_last_params": None,
            "flow_features_tramo": None,
            "flow_features_tramo_label": None,
            "acc_flow_metrics": ["flow", "speed", "density"],
            "acc_flow_categories": ["Light", "Heavy"],
            "acc_flow_lanes": 2,
            "acc_flow_include_cluster_vars": True,
            "acc_flow_cluster_vars": ["share", "flow", "speed"],
            "history_entries": [],
        }
    )

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(app, "MODELS_DIR", tmp_path / "model_history")
    monkeypatch.setattr(app, "HISTORY_PATH", tmp_path / "experiment_history.jsonl")

    base_result = train_model(
        base_df,
        base_cols,
        "Random Forest",
        {"n_estimators": 20, "max_depth": 3},
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        random_state=42,
    )
    cluster_result = train_model(
        base_df,
        feature_cols,
        "Random Forest",
        {"n_estimators": 20, "max_depth": 3},
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        random_state=42,
    )

    entry = app._record_experiment_history(
        base_df=base_df,
        features_df=features_df,
        balanced_df=None,
        base_feature_cols=base_cols,
        base_result=base_result,
        cluster_feature_cols=feature_cols,
        cluster_result=cluster_result,
        model_choice="Random Forest",
        model_params_base={"n_estimators": 20, "max_depth": 3},
        model_params_cluster={"n_estimators": 20, "max_depth": 3},
        random_state=42,
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        use_balanced=False,
    )

    cluster_entry = entry["models"]["Base + Cluster"]
    bundle_path = Path(cluster_entry["xai_bundle_path"])
    assert bundle_path.exists()
    assert (bundle_path / "manifest.json").exists()
    assert entry["xai"]["base_cluster"]["available"] is True

    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cluster_features_path"] == str(tmp_path / "cluster_features.duckdb")
    assert manifest["cluster_features_source"] == "calculadas"
    assert manifest["feature_cols"] == feature_cols
    report = compute_xai_report(bundle_path)
    assert isinstance(report["beeswarm_points"], pd.DataFrame)
    assert not report["beeswarm_points"].empty


def test_resolve_base_cluster_xai_info_handles_legacy_entry():
    bundle_path, bundle_error = app._resolve_base_cluster_xai_info(
        {"run_id": "legacy", "models": {"Base + Cluster": {"model_path": "legacy.joblib"}}}
    )
    assert bundle_path is None
    assert bundle_error is None
