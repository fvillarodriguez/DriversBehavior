from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import src.model_xai as model_xai
import src.xgboost as local_xgboost_module
from src.model_training import train_model
from src.model_xai import compute_xai_report, load_xai_bundle, save_xai_bundle
from tests.pipeline_helpers import build_synthetic_base_df


def _train_rf_result(tmp_path: Path):
    pytest.importorskip("sklearn")
    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(
        tmp_path
    )
    result = train_model(
        base_df,
        feature_cols,
        "Random Forest",
        {"n_estimators": 20, "max_depth": 3},
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        random_state=42,
    )
    return base_df, feature_cols, result


def test_save_xai_bundle_persists_manifest_and_tables(tmp_path):
    _base_df, feature_cols, result = _train_rf_result(tmp_path)
    bundle_dir = tmp_path / "model_history" / "run_1" / "base_cluster"
    manifest = save_xai_bundle(
        bundle_dir,
        model=result["model"],
        feature_cols=feature_cols,
        xai_payload=result["xai_payload"],
        manifest={
            "run_id": "run_1",
            "model_name": "Random Forest",
            "cluster_features_path": "Resultados/cluster_features.duckdb",
            "cluster_features_source": "calculadas",
        },
    )

    assert (bundle_dir / "model.joblib").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "background.parquet").exists()
    assert (bundle_dir / "explain_rows.parquet").exists()
    assert manifest["cluster_features_path"] == "Resultados/cluster_features.duckdb"
    assert manifest["cluster_features_source"] == "calculadas"
    assert manifest["feature_cols"] == feature_cols

    loaded = load_xai_bundle(bundle_dir)
    assert loaded["manifest"]["run_id"] == "run_1"
    assert isinstance(loaded["background_df"], pd.DataFrame)
    assert isinstance(loaded["explain_rows_df"], pd.DataFrame)
    assert not loaded["background_df"].empty
    assert not loaded["explain_rows_df"].empty


def test_compute_xai_report_returns_global_and_local_sections(tmp_path):
    pytest.importorskip("shap")
    _base_df, feature_cols, result = _train_rf_result(tmp_path)
    bundle_dir = tmp_path / "model_history" / "run_2" / "base_cluster"
    save_xai_bundle(
        bundle_dir,
        model=result["model"],
        feature_cols=feature_cols,
        xai_payload=result["xai_payload"],
        manifest={
            "run_id": "run_2",
            "model_name": "Random Forest",
            "cluster_features_path": "Resultados/cluster_features.duckdb",
            "cluster_features_source": "calculadas",
        },
    )

    report = compute_xai_report(bundle_dir)
    global_df = report["global_importance"]
    group_df = report["group_summary"]
    beeswarm_df = report["beeswarm_points"]

    assert isinstance(global_df, pd.DataFrame)
    assert {"feature", "mean_abs_shap", "feature_group"}.issubset(global_df.columns)
    assert isinstance(group_df, pd.DataFrame)
    assert {"feature_group", "total_mean_abs_shap", "share"}.issubset(group_df.columns)
    assert isinstance(beeswarm_df, pd.DataFrame)
    assert {
        "feature",
        "feature_rank",
        "feature_group",
        "sample_id",
        "shap_value",
        "abs_shap",
        "feature_value",
        "score",
        "pred",
        "target",
    }.issubset(beeswarm_df.columns)
    assert not beeswarm_df.empty
    assert set(beeswarm_df["feature"].unique()).issubset(set(global_df["feature"]))
    rank_map = global_df.set_index("feature")["rank"].to_dict()
    beeswarm_rank_map = beeswarm_df.groupby("feature")["feature_rank"].first().to_dict()
    assert beeswarm_rank_map == {
        feature: rank_map[feature] for feature in beeswarm_rank_map
    }
    assert isinstance(report["local_cases"], list)


def test_compute_xai_report_marks_next_and_last_cluster_features_as_cluster(
    tmp_path, monkeypatch
):
    feature_cols = [
        "next_speed_heavy",
        "next_cluster_speed_0",
        "last_cluster_density_1",
        "cluster_share_0",
    ]
    background_df = pd.DataFrame(
        {
            "next_speed_heavy": [85.0, 82.0],
            "next_cluster_speed_0": [70.0, 65.0],
            "last_cluster_density_1": [0.12, 0.18],
            "cluster_share_0": [0.45, 0.35],
        }
    )
    explain_rows_df = background_df.copy()
    explain_rows_df["target"] = [0, 1]
    explain_rows_df["score"] = [0.25, 0.82]
    explain_rows_df["pred"] = [0, 1]
    explain_rows_df["threshold"] = [0.5, 0.5]
    explain_rows_df["case_hint"] = ["baseline", "true_positive"]

    class _FakeTreeExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, X):
            return np.array(
                [
                    [0.40, 0.15, -0.10, 0.04],
                    [0.30, 0.12, -0.18, 0.03],
                ],
                dtype=float,
            )

    class _FakeShap:
        TreeExplainer = _FakeTreeExplainer

    monkeypatch.setattr(
        model_xai,
        "load_xai_bundle",
        lambda _bundle_dir: {
            "bundle_dir": str(tmp_path / "bundle"),
            "manifest": {
                "feature_cols": feature_cols,
                "model_name": "Random Forest",
            },
            "background_df": background_df,
            "explain_rows_df": explain_rows_df,
            "model": object(),
        },
    )
    monkeypatch.setattr(model_xai, "_require_shap", lambda: _FakeShap())

    report = model_xai.compute_xai_report(tmp_path / "bundle")

    global_df = report["global_importance_full"].set_index("feature")
    assert global_df.loc["next_speed_heavy", "feature_group"] == "Base"
    assert global_df.loc["next_cluster_speed_0", "feature_group"] == "Cluster"
    assert global_df.loc["last_cluster_density_1", "feature_group"] == "Cluster"
    assert global_df.loc["cluster_share_0", "feature_group"] == "Cluster"

    group_df = report["group_summary"].set_index("feature_group")
    assert group_df.loc["Cluster", "total_mean_abs_shap"] > 0
    assert group_df.loc["Base", "total_mean_abs_shap"] > 0

    cluster_top_features = set(report["cluster_top"]["feature"].astype(str))
    assert {
        "next_cluster_speed_0",
        "last_cluster_density_1",
        "cluster_share_0",
    }.issubset(cluster_top_features)

    assert report["local_cases"]
    local_detail = report["local_cases"][0]["all_contributions"].set_index("feature")
    assert local_detail.loc["next_cluster_speed_0", "feature_group"] == "Cluster"
    assert local_detail.loc["last_cluster_density_1", "feature_group"] == "Cluster"


def test_compute_xai_report_prefers_external_xgboost_when_local_module_shadows(
    tmp_path, monkeypatch
):
    pytest.importorskip("xgboost")

    feature_cols = ["flow", "speed", "cluster_share"]
    background_df = pd.DataFrame(
        {
            "flow": [10.0, 12.0, 15.0],
            "speed": [80.0, 78.0, 74.0],
            "cluster_share": [0.1, 0.2, 0.3],
        }
    )
    explain_rows_df = background_df.iloc[[0, 2]].reset_index(drop=True).copy()
    explain_rows_df["target"] = [0, 1]
    explain_rows_df["score"] = [0.2, 0.8]
    explain_rows_df["pred"] = [0, 1]
    explain_rows_df["threshold"] = [0.5, 0.5]
    explain_rows_df["case_hint"] = ["highest_score", "true_positive"]

    class _FakeTreeExplainer:
        def __init__(self, model):
            import xgboost as xgb  # type: ignore

            self._module_file = str(getattr(xgb, "__file__", "") or "")
            assert hasattr(xgb, "core")

        def shap_values(self, X):
            return np.array(
                [
                    [0.3, -0.1, 0.2],
                    [0.4, -0.2, 0.1],
                ],
                dtype=float,
            )

    class _FakeShap:
        TreeExplainer = _FakeTreeExplainer

    monkeypatch.syspath_prepend(str(Path(model_xai.__file__).resolve().parent))
    monkeypatch.setitem(sys.modules, "xgboost", local_xgboost_module)
    monkeypatch.setattr(
        model_xai,
        "load_xai_bundle",
        lambda _bundle_dir: {
            "bundle_dir": str(tmp_path / "bundle"),
            "manifest": {
                "feature_cols": feature_cols,
                "model_name": "XGBoost",
            },
            "background_df": background_df,
            "explain_rows_df": explain_rows_df,
            "model": object(),
        },
    )
    monkeypatch.setattr(model_xai, "_require_shap", lambda: _FakeShap())

    report = model_xai.compute_xai_report(tmp_path / "bundle")

    assert report["explainer_name"] == "TreeExplainer"
    assert not report["global_importance"].empty
    imported_xgboost = sys.modules["xgboost"]
    assert Path(str(imported_xgboost.__file__)).name == "__init__.py"
    assert hasattr(imported_xgboost, "core")


def test_import_external_xgboost_attaches_core_module_when_missing_on_package(
    monkeypatch,
):
    class _FakePackage:
        __file__ = "/tmp/site-packages/xgboost/__init__.py"

    class _FakeCoreModule:
        Booster = object

    fake_package = _FakePackage()
    fake_core = _FakeCoreModule()

    def _fake_import(name):
        if name == "xgboost":
            return fake_package
        if name == "xgboost.core":
            return fake_core
        raise ImportError(name)

    monkeypatch.setattr(model_xai.importlib, "import_module", _fake_import)
    monkeypatch.setitem(sys.modules, "xgboost", local_xgboost_module)

    imported = model_xai._import_external_xgboost()

    assert imported is fake_package
    assert imported.core is fake_core
