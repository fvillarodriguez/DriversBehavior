from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("optuna")
pytest.importorskip("imblearn")

import src.cluster_accident_app as app
import src.crash_prediction_history_store as history_store
from src.model_training import train_model
from src.model_xai import compute_xai_report
from tests.pipeline_helpers import build_synthetic_base_df


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}


class _SelectorFakeStreamlit:
    def __init__(self, selected: str) -> None:
        self.session_state: dict = {}
        self.selected = selected
        self.options: list[str] = []
        self.captions: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def selectbox(self, _label, options=None, **_kwargs):
        self.options = list(options or [])
        return self.selected

    def caption(self, message: str) -> None:
        self.captions.append(str(message))

    def info(self, message: str) -> None:
        self.infos.append(str(message))

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))


class _FakeTab:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _TabsFakeStreamlit(_FakeStreamlit):
    def __init__(self) -> None:
        super().__init__()
        self.titles: list[str] = []
        self.tabs_calls: list[list[str]] = []

    def title(self, message: str) -> None:
        self.titles.append(str(message))

    def tabs(self, labels):
        label_list = list(labels)
        self.tabs_calls.append(label_list)
        return [_FakeTab(label) for label in label_list]

    def radio(self, *args, **kwargs):
        raise AssertionError("Crash prediction main navigation must use st.tabs")


def _make_optuna_batch_record(
    *,
    record_id: int,
    feature_key: str,
    dataset_fingerprint: str,
    feature_cols,
    feature_set_label: str,
    balance_mode: str = "none",
    calibration_method: str = "sigmoid",
    threshold_objective: str = "far",
    model_name: str = "XGBoost",
    objective_mode: str = app.CALIBRATION_SWEEP_OBJECTIVE_MODE_SCALAR,
    best_model_params: dict | None = None,
    best_smote_params: dict | None = None,
    pareto_front_csv: str | None = None,
    extra_settings: dict | None = None,
) -> dict:
    settings = {
        "feature_set_label": feature_set_label,
        "threshold_objective": threshold_objective,
        "calibration_method": calibration_method,
        "objective_mode": objective_mode,
        "best_feature_cols": list(feature_cols),
        "ranked_cols": list(feature_cols),
        "best_top_k": len(list(feature_cols)),
        "test_size": 0.2,
        "val_size": 0.2,
        "far_target": 0.2,
        "alerts_per_day": 5.0,
        "fn_cost": 10.0,
        "fp_cost": 1.0,
        "robust_folds": 3,
        "random_state": 42,
        "objective_label": "MCC",
    }
    if extra_settings:
        settings.update(extra_settings)

    variant = {
        "model_choice": model_name,
        "balance_mode": balance_mode,
        "calibration_method": calibration_method,
        "best_model_params": dict(best_model_params or {"n_estimators": 120}),
        "best_smote_params": dict(best_smote_params or {}),
        "best_trial_number": 7,
        "objective_mode": objective_mode,
        "optuna_settings": settings,
    }
    if pareto_front_csv:
        variant["pareto_front_csv"] = str(pareto_front_csv)

    return {
        "id": int(record_id),
        "record_uid": f"optuna-record-{record_id}",
        "created_at": f"2026-04-23T12:{record_id:02d}:00",
        "model_name": model_name,
        "balance_strategy": balance_mode,
        "calibration_method": calibration_method,
        "threshold_objective": threshold_objective,
        "metadata": {
            "feature_cols": list(feature_cols),
            "feature_set_label": feature_set_label,
            "store_entry": {
                "feature_key": feature_key,
                "dataset_fingerprint": dataset_fingerprint,
                "feature_cols": list(feature_cols),
            },
            "variant": variant,
        },
    }


def test_build_tramo_selector_offers_combined_feature_engineering_option(monkeypatch):
    fake_st = _SelectorFakeStreamlit(app.COMBINED_TRAMO_LABEL)
    monkeypatch.setattr(app, "st", fake_st)

    selected = app._build_tramo_selector(
        None,
        date_start=None,
        date_end=None,
        key="test_combined_tramo",
        include_combined_tramo=True,
    )

    assert selected == app.COMBINED_TRAMO_SELECTION
    assert app.COMBINED_TRAMO_LABEL in fake_st.options
    assert fake_st.captions == [f"Filtro activo: {app.COMBINED_TRAMO_LABEL}"]


def test_combined_tramo_duckdb_filter_expands_to_three_segment_pairs():
    clauses, params, filter_ok = app._build_tramo_duckdb_filters(
        app.COMBINED_TRAMO_SELECTION,
        {"portico_last", "portico_next"},
    )

    assert filter_ok is True
    assert len(clauses) == 1
    assert clauses[0].count("portico_last = ? AND portico_next = ?") == 3
    assert " OR " in clauses[0]
    assert params == ["15", "14", "14", "12", "12", "11"]


def test_apply_combined_tramo_filter_matches_requested_accident_pairs():
    df = pd.DataFrame(
        {
            "ultimo_portico": ["15", "14", "12", "15", "11"],
            "proximo_portico": ["14", "12", "11", "12", "9"],
            "value": [1, 2, 3, 4, 5],
        }
    )

    filtered, filter_ok = app._apply_tramo_filter_df(
        df,
        app.COMBINED_TRAMO_SELECTION,
    )

    assert filter_ok is True
    assert list(filtered["value"]) == [1, 2, 3]


def test_combined_tramo_portico_codes_limit_flow_query_to_chain():
    assert app._tramo_portico_codes(app.COMBINED_TRAMO_SELECTION) == (
        "15",
        "14",
        "12",
        "11",
    )


def test_filter_segments_for_combined_tramo_returns_chain_order_without_duplicates():
    segments_df = pd.DataFrame(
        {
            "eje": ["NS", "NS", "NS", "NS", "NS"],
            "calzada": ["Poniente"] * 5,
            "portico_last": ["15", "15", "14", "12", "14"],
            "portico_next": ["14", "14", "12", "11", "15"],
            "marker": ["old_duplicate", "selected", "middle", "end", "reverse"],
        }
    )

    filtered = app._filter_segments_for_tramo(
        segments_df,
        app.COMBINED_TRAMO_SELECTION,
    )

    assert list(zip(filtered["portico_last"], filtered["portico_next"])) == [
        ("15", "14"),
        ("14", "12"),
        ("12", "11"),
    ]
    assert list(filtered["marker"]) == ["selected", "middle", "end"]


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


def test_apply_optuna_model_params_to_state_targets_base_cluster(tmp_path, monkeypatch):
    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    features_path = str(tmp_path / "flow_features.duckdb")
    feature_key = app._feature_selection_key(features_path, "duckdb", features_df)
    optuna_key = app._optuna_result_key(feature_key, feature_cols)
    optuna_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 3.6,
        "reg_lambda": 8.3,
        "gamma": 0.0,
    }
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "flow_features_path": features_path,
            "flow_features_source": "duckdb",
            "selected_features": feature_cols,
            "optuna_results_store": {
                optuna_key: {
                    "results": {
                        "XGBoost": {
                            "model_choice": "XGBoost",
                            "by_balance_mode": {
                                "smote": {
                                    "by_calibration_method": {
                                        "sigmoid": {
                                            "model_choice": "XGBoost",
                                            "best_model_params": optuna_params,
                                            "best_smote_params": {
                                                "smote_k_neighbors": 5,
                                                "smote_sampling_strategy": 0.5,
                                            },
                                            "optuna_settings": {
                                                "balance_mode": "smote",
                                                "balance_mode_label": "Con SMOTE",
                                                "calibration_method": "sigmoid",
                                                "calibration_method_label": "Platt scaling (sigmoid)",
                                            },
                                            "search_space": {},
                                        }
                                    }
                                }
                            },
                        }
                    }
                }
            },
            "optuna_model_params_applied_signatures": {},
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    status = app._apply_optuna_model_params_to_state(
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        base_df=base_df,
        features_df=features_df,
    )

    assert "Parametros Optuna (Con SMOTE | Platt scaling (sigmoid))" in status
    assert "Base + Cluster" in status
    assert fake_st.session_state["cluster_model_xgb_n_estimators"] == 200
    assert fake_st.session_state["cluster_model_xgb_max_depth"] == 4
    assert fake_st.session_state["cluster_model_xgb_learning_rate"] == pytest.approx(0.01)
    assert fake_st.session_state["cluster_model_xgb_subsample"] == pytest.approx(0.9)
    assert fake_st.session_state["cluster_model_xgb_colsample"] == pytest.approx(0.8)
    assert fake_st.session_state["cluster_model_xgb_reg_alpha"] == pytest.approx(3.6)
    assert fake_st.session_state["cluster_model_xgb_reg_lambda"] == pytest.approx(8.3)
    assert "cluster_model_xgb_n_jobs" not in fake_st.session_state
    assert "base_model_xgb_n_estimators" not in fake_st.session_state
    assert fake_st.session_state["optuna_model_params_applied_signatures"]


def test_apply_optuna_model_params_respects_calibration_fallback_opt_in(
    tmp_path, monkeypatch
):
    """Sin opt-in (default), _apply_optuna_model_params_to_state no debe aplicar
    los best_model_params de Optuna cuando la calibración guardada difiere de
    la pedida. Con opt-in activado, debe aplicarlos usando fallback."""
    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    features_path = str(tmp_path / "flow_features.duckdb")
    feature_key = app._feature_selection_key(features_path, "duckdb", features_df)
    optuna_key = app._optuna_result_key(feature_key, feature_cols)
    optuna_params = {
        "n_estimators": 321,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "gamma": 0.0,
    }

    # Optuna guardó `sigmoid`, Modelos pide `isotonic`.
    def _build_state(fallback_opt_in: bool) -> "_FakeStreamlit":
        fake_st = _FakeStreamlit()
        fake_st.session_state.update(
            {
                "flow_features_path": features_path,
                "flow_features_source": "duckdb",
                "selected_features": feature_cols,
                "allow_optuna_calibration_fallback": fallback_opt_in,
                "optuna_results_store": {
                    optuna_key: {
                        "results": {
                            "XGBoost": {
                                "model_choice": "XGBoost",
                                "by_balance_mode": {
                                    "smote": {
                                        "by_calibration_method": {
                                            "sigmoid": {
                                                "model_choice": "XGBoost",
                                                "best_model_params": optuna_params,
                                                "best_smote_params": {},
                                                "optuna_settings": {
                                                    "balance_mode": "smote",
                                                    "calibration_method": "sigmoid",
                                                },
                                                "search_space": {},
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    }
                },
                "optuna_model_params_applied_signatures": {},
            }
        )
        return fake_st

    # Sin opt-in: no debe aplicar los params (cae a "sin match").
    fake_st = _build_state(fallback_opt_in=False)
    monkeypatch.setattr(app, "st", fake_st)
    status = app._apply_optuna_model_params_to_state(
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="isotonic",
        base_df=base_df,
        features_df=features_df,
    )
    assert status is not None
    assert "sin resultados" in status.lower() or "sin match" in status.lower()
    assert "cluster_model_xgb_n_estimators" not in fake_st.session_state

    # Con opt-in: debe aplicar los params de sigmoid como fallback.
    fake_st = _build_state(fallback_opt_in=True)
    monkeypatch.setattr(app, "st", fake_st)
    status = app._apply_optuna_model_params_to_state(
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="isotonic",
        base_df=base_df,
        features_df=features_df,
    )
    assert status is not None
    assert "aplicados" in status.lower()
    assert fake_st.session_state["cluster_model_xgb_n_estimators"] == 321


def test_apply_optuna_model_params_skips_groups_outside_last_optuna_filter(
    tmp_path,
    monkeypatch,
):
    base_df, feature_cols, _base_cols, cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    features_path = str(tmp_path / "flow_features.duckdb")
    feature_key = app._feature_selection_key(features_path, "duckdb", features_df)
    dataset_fingerprint = app._dataset_content_fingerprint(features_df)
    cluster_key = app._optuna_result_key(feature_key, cluster_cols)
    base_cluster_key = app._optuna_result_key(feature_key, feature_cols)

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "flow_features_path": features_path,
            "flow_features_source": "duckdb",
            "selected_features": feature_cols,
            "model_feature_source": "optuna",
            "allow_optuna_calibration_fallback": False,
            "optuna_last_optimized_feature_sets": ["Base", "Base + Cluster"],
            "optuna_last_optimized_feature_key": feature_key,
            "optuna_last_optimized_dataset_fingerprint": dataset_fingerprint,
            "optuna_results_store": {
                cluster_key: {
                    "results": {
                        "XGBoost": {
                            "model_choice": "XGBoost",
                            "by_balance_mode": {
                                "smote": {
                                    "by_calibration_method": {
                                        "sigmoid": {
                                            "model_choice": "XGBoost",
                                            "best_model_params": {"n_estimators": 111},
                                            "best_smote_params": {},
                                            "optuna_settings": {
                                                "balance_mode": "smote",
                                                "calibration_method": "sigmoid",
                                            },
                                            "search_space": {},
                                        }
                                    }
                                }
                            },
                        }
                    }
                },
                base_cluster_key: {
                    "results": {
                        "XGBoost": {
                            "model_choice": "XGBoost",
                            "by_balance_mode": {
                                "smote": {
                                    "by_calibration_method": {
                                        "sigmoid": {
                                            "model_choice": "XGBoost",
                                            "best_model_params": {"n_estimators": 222},
                                            "best_smote_params": {},
                                            "optuna_settings": {
                                                "balance_mode": "smote",
                                                "calibration_method": "sigmoid",
                                            },
                                            "search_space": {},
                                        }
                                    }
                                }
                            },
                        }
                    }
                },
            },
            "optuna_model_params_applied_signatures": {},
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    status = app._apply_optuna_model_params_to_state(
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        base_df=base_df,
        features_df=features_df,
    )

    assert status is not None
    assert "filtrados: Cluster" in status
    assert fake_st.session_state["cluster_model_xgb_n_estimators"] == 222
    assert "cluster_only_model_xgb_n_estimators" not in fake_st.session_state


def test_apply_optuna_model_params_ignores_last_optuna_filter_in_feature_selection_mode(
    tmp_path,
    monkeypatch,
):
    base_df, feature_cols, _base_cols, cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    features_path = str(tmp_path / "flow_features.duckdb")
    feature_key = app._feature_selection_key(features_path, "duckdb", features_df)
    cluster_key = app._optuna_result_key(feature_key, cluster_cols)

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "flow_features_path": features_path,
            "flow_features_source": "duckdb",
            "selected_features": feature_cols,
            "model_feature_source": "feature_selection",
            "optuna_last_optimized_feature_sets": ["Base", "Base + Cluster"],
            "optuna_last_optimized_feature_key": feature_key,
            "optuna_last_optimized_dataset_fingerprint": app._dataset_content_fingerprint(
                features_df
            ),
            "optuna_results_store": {
                cluster_key: {
                    "results": {
                        "XGBoost": {
                            "model_choice": "XGBoost",
                            "by_balance_mode": {
                                "smote": {
                                    "by_calibration_method": {
                                        "sigmoid": {
                                            "model_choice": "XGBoost",
                                            "best_model_params": {"n_estimators": 333},
                                            "best_smote_params": {},
                                            "optuna_settings": {
                                                "balance_mode": "smote",
                                                "calibration_method": "sigmoid",
                                            },
                                            "search_space": {},
                                        }
                                    }
                                }
                            },
                        }
                    }
                }
            },
            "optuna_model_params_applied_signatures": {},
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    status = app._apply_optuna_model_params_to_state(
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        base_df=base_df,
        features_df=features_df,
    )

    assert status is not None
    assert "Cluster" in status
    assert fake_st.session_state["cluster_only_model_xgb_n_estimators"] == 333


def test_resolve_model_optuna_feature_set_filter_excludes_cluster_for_pareto(
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "model_feature_source": "optuna",
            "optuna_last_optimized_feature_sets": ["Base", "Base + Cluster"],
            "optuna_last_optimized_feature_key": "/tmp/features.duckdb",
            "optuna_last_optimized_dataset_fingerprint": "fp_1",
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    filter_info = app._resolve_model_optuna_feature_set_filter(
        current_feature_key="/tmp/features.duckdb",
        current_dataset_fingerprint="fp_1",
        available_feature_sets=["Base", "Cluster", "Base + Cluster"],
    )
    pareto_candidates = {
        "Base": {"candidates": [1]},
        "Cluster": {"candidates": [1]},
        "Base + Cluster": {"candidates": [1]},
    }
    available_pareto_feature_sets = [
        label
        for label in app.MODEL_FEATURE_SET_ORDER
        if label in pareto_candidates
        and label in set(filter_info["allowed_feature_sets"])
    ]

    assert filter_info["applies"] is True
    assert available_pareto_feature_sets == ["Base", "Base + Cluster"]


def test_resolve_model_optuna_feature_set_filter_ignores_mismatched_context(
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "model_feature_source": "optuna",
            "optuna_last_optimized_feature_sets": ["Base", "Base + Cluster"],
            "optuna_last_optimized_feature_key": "/tmp/other_features.duckdb",
            "optuna_last_optimized_dataset_fingerprint": "fp_old",
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    filter_info = app._resolve_model_optuna_feature_set_filter(
        current_feature_key="/tmp/features.duckdb",
        current_dataset_fingerprint="fp_new",
        available_feature_sets=["Base", "Cluster", "Base + Cluster"],
    )

    assert filter_info["applies"] is False
    assert filter_info["allowed_feature_sets"] == [
        "Base",
        "Cluster",
        "Base + Cluster",
    ]
    assert filter_info["reason"] == "feature_key_mismatch"


def test_lookup_optuna_best_feature_cols_falls_back_within_same_balance_mode(
    tmp_path,
):
    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    features_path = str(tmp_path / "flow_features.duckdb")
    feature_key = app._feature_selection_key(features_path, "duckdb", features_df)
    optuna_key = app._optuna_result_key(feature_key, feature_cols)

    store = {
        optuna_key: {
            "results": {
                "XGBoost": {
                    "model_choice": "XGBoost",
                    "by_balance_mode": {
                        "smote": {
                            "by_calibration_method": {
                                "sigmoid": {
                                    "model_choice": "XGBoost",
                                    "best_model_params": {"n_estimators": 100},
                                    "best_smote_params": {
                                        "smote_k_neighbors": 5,
                                        "smote_sampling_strategy": 0.5,
                                    },
                                    "optuna_settings": {
                                        "balance_mode": "smote",
                                        "balance_mode_label": "Con SMOTE",
                                        "calibration_method": "sigmoid",
                                        "calibration_method_label": "Platt scaling (sigmoid)",
                                        "best_feature_cols": feature_cols,
                                    },
                                    "search_space": {},
                                }
                            }
                        }
                    },
                }
            }
        }
    }

    exact = app._lookup_optuna_best_feature_cols(
        store=store,
        feature_key=feature_key,
        cols=feature_cols,
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        allow_calibration_fallback=True,
    )
    fallback = app._lookup_optuna_best_feature_cols(
        store=store,
        feature_key=feature_key,
        cols=feature_cols,
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="isotonic",
        allow_calibration_fallback=True,
    )

    assert exact is not None
    assert exact["used_fallback"] is False
    assert exact["calibration_method"] == "sigmoid"
    assert fallback is not None
    assert fallback["used_fallback"] is True
    assert fallback["calibration_method"] == "sigmoid"
    assert fallback["calibration_method_label"] == "Platt scaling (sigmoid)"


def test_feature_selection_key_disambiguates_memory_datasets_by_fingerprint():
    """Sin features_path, dos datasets distintos deben producir keys distintos
    aunque compartan shape. Antes del fingerprint, colisionaban silenciosamente
    y Optuna viejo parecía válido para un dataset nuevo."""
    df_a = pd.DataFrame(
        {
            "flow_light": [1.0, 2.0, 3.0],
            "speed_mean": [60.0, 65.0, 70.0],
            "target": [0, 1, 0],
        }
    )
    df_b = pd.DataFrame(
        {
            "density_heavy": [0.1, 0.2, 0.3],
            "entropy": [0.5, 0.4, 0.6],
            "target": [1, 0, 1],
        }
    )

    key_a = app._feature_selection_key(None, "memory", df_a)
    key_b = app._feature_selection_key(None, "memory", df_b)

    # Mismo shape (3 filas x 3 cols), distintos esquemas → keys distintos.
    assert df_a.shape == df_b.shape
    assert key_a != key_b
    # Fingerprint explícito también debe diferenciarlos.
    assert app._dataset_content_fingerprint(df_a) != app._dataset_content_fingerprint(df_b)
    # Y debe ser estable: llamar dos veces sobre el mismo df da el mismo fingerprint.
    assert app._dataset_content_fingerprint(df_a) == app._dataset_content_fingerprint(df_a)


def test_feature_selection_key_preserves_path_behavior():
    """Con features_path, el key sigue siendo el path resuelto (sin modificación),
    para no invalidar resultados Optuna guardados en disco."""
    features_df = pd.DataFrame({"flow_light": [1.0, 2.0]})
    key = app._feature_selection_key("/tmp/example.duckdb", "duckdb", features_df)
    assert key.endswith("example.duckdb")
    # No se agrega fingerprint cuando hay path.
    assert app._dataset_content_fingerprint(features_df) not in key


def test_diagnose_optuna_key_mismatch_detects_selected_features_change():
    """Cuando el feature_key coincide pero las variables seleccionadas
    cambiaron, el diagnóstico debe identificarlo explícitamente."""
    feature_key = "/path/to/features.duckdb"
    sig_a = app._feature_list_signature(["a", "b", "c"])
    sig_b = app._feature_list_signature(["a", "b"])
    expected_key = f"{feature_key}|{sig_b}"  # lo que el usuario pide ahora
    active_key = f"{feature_key}|{sig_a}"    # lo que Optuna tiene guardado
    store = {active_key: {"dataset_fingerprint": "fp1"}}

    diag = app._diagnose_optuna_key_mismatch(
        store=store,
        expected_key=expected_key,
        active_key=active_key,
        current_fingerprint="fp1",
    )
    assert diag["has_match"] is False
    assert any("variables seleccionadas" in r for r in diag["reasons"])


def test_diagnose_optuna_key_mismatch_detects_dataset_change():
    """Cuando el feature_key cambia (archivo/schema distinto), el diagnóstico
    debe explicarlo en lugar de decir genéricamente 'no coinciden'."""
    sig = app._feature_list_signature(["a", "b"])
    expected_key = f"/path/A|{sig}"
    active_key = f"/path/B|{sig}"
    store = {active_key: {"dataset_fingerprint": "fp"}}

    diag = app._diagnose_optuna_key_mismatch(
        store=store,
        expected_key=expected_key,
        active_key=active_key,
        current_fingerprint="fp",
    )
    assert diag["has_match"] is False
    assert any("dataset activo" in r for r in diag["reasons"])


def test_diagnose_optuna_key_mismatch_detects_content_drift_with_same_key():
    """Cuando el key es exactamente el mismo pero el fingerprint del dataset
    cambió (ej. el archivo se regeneró), `dataset_drift` debe marcarse."""
    sig = app._feature_list_signature(["a", "b"])
    key = f"/path/A|{sig}"
    store = {key: {"dataset_fingerprint": "fp_old"}}

    diag = app._diagnose_optuna_key_mismatch(
        store=store,
        expected_key=key,
        active_key=key,
        current_fingerprint="fp_new",
    )
    assert diag["has_match"] is True
    assert diag["dataset_drift"] is True
    assert any("contenido del dataset" in r for r in diag["reasons"])


def test_diagnose_optuna_key_mismatch_when_no_active_result():
    """Sin active_key (nunca se corrió Optuna en la sesión), el diagnóstico
    debe decirlo explícitamente."""
    diag = app._diagnose_optuna_key_mismatch(
        store={},
        expected_key="/path|sig",
        active_key=None,
        current_fingerprint="fp",
    )
    assert diag["has_match"] is False
    assert any("no hay resultado" in r.lower() for r in diag["reasons"])


def test_lookup_optuna_best_feature_cols_returns_dataset_fingerprint(tmp_path):
    """El lookup debe propagar el `dataset_fingerprint` del entry para que
    Modelos pueda detectar drift sin consultar el store dos veces."""
    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    features_path = str(tmp_path / "flow_features.duckdb")
    feature_key = app._feature_selection_key(features_path, "duckdb", features_df)
    optuna_key = app._optuna_result_key(feature_key, feature_cols)

    store = {
        optuna_key: {
            "dataset_fingerprint": "fingerprint_abc123",
            "results": {
                "XGBoost": {
                    "model_choice": "XGBoost",
                    "by_balance_mode": {
                        "smote": {
                            "by_calibration_method": {
                                "sigmoid": {
                                    "model_choice": "XGBoost",
                                    "best_model_params": {"n_estimators": 100},
                                    "best_smote_params": {},
                                    "optuna_settings": {
                                        "balance_mode": "smote",
                                        "calibration_method": "sigmoid",
                                        "best_feature_cols": feature_cols,
                                    },
                                    "search_space": {},
                                }
                            }
                        }
                    },
                }
            },
        }
    }

    match = app._lookup_optuna_best_feature_cols(
        store=store,
        feature_key=feature_key,
        cols=feature_cols,
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        allow_calibration_fallback=True,
    )
    assert match is not None
    assert match["dataset_fingerprint"] == "fingerprint_abc123"


def _make_optuna_store_with_smote_and_none(optuna_key: str, feature_cols):
    """Helper local que construye un store con variantes SMOTE y none.

    Útil para los tests de ``_get_active_optuna_best``: incluye resultados
    para ``balance_mode in {smote, none}`` y ``calibration in {sigmoid,
    isotonic}`` para poder chequear el orden de preferencia.
    """
    return {
        optuna_key: {
            "dataset_fingerprint": "fingerprint_abc123",
            "results": {
                "XGBoost": {
                    "model_choice": "XGBoost",
                    "by_balance_mode": {
                        "smote": {
                            "by_calibration_method": {
                                "sigmoid": {
                                    "model_choice": "XGBoost",
                                    "best_model_params": {"n_estimators": 111},
                                    "best_smote_params": {
                                        "smote_k_neighbors": 5,
                                        "smote_sampling_strategy": 0.4,
                                    },
                                    "best_score": 0.91,
                                    "trials_df": None,
                                    "optuna_settings": {
                                        "balance_mode": "smote",
                                        "calibration_method": "sigmoid",
                                        "best_feature_cols": list(feature_cols),
                                    },
                                    "search_space": {"model": {"n_estimators": {}}},
                                },
                                "isotonic": {
                                    "model_choice": "XGBoost",
                                    "best_model_params": {"n_estimators": 222},
                                    "best_smote_params": {
                                        "smote_k_neighbors": 7,
                                        "smote_sampling_strategy": 0.5,
                                    },
                                    "best_score": 0.88,
                                    "trials_df": None,
                                    "optuna_settings": {
                                        "balance_mode": "smote",
                                        "calibration_method": "isotonic",
                                        "best_feature_cols": list(feature_cols),
                                    },
                                    "search_space": {},
                                },
                            },
                        },
                        "none": {
                            "by_calibration_method": {
                                "sigmoid": {
                                    "model_choice": "XGBoost",
                                    "best_model_params": {"n_estimators": 333},
                                    "best_score": 0.80,
                                    "trials_df": None,
                                    "optuna_settings": {
                                        "balance_mode": "none",
                                        "calibration_method": "sigmoid",
                                        "best_feature_cols": list(feature_cols),
                                    },
                                    "search_space": {},
                                },
                            },
                        },
                    },
                }
            },
        }
    }


def test_get_active_optuna_best_prefers_smote_over_none(tmp_path, monkeypatch):
    """El helper debe devolver la variante SMOTE cuando coexisten SMOTE y
    none (mismo contrato que el disk-reloader legacy)."""
    base_df, feature_cols, _b, _c = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    feature_key = app._feature_selection_key(
        str(tmp_path / "flow_features.duckdb"), "duckdb", features_df
    )
    optuna_key = app._optuna_result_key(feature_key, feature_cols)
    store = _make_optuna_store_with_smote_and_none(optuna_key, feature_cols)

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "optuna_results_store": store,
            "optuna_active_key": optuna_key,
            "optuna_model_choice": "XGBoost",
            "optuna_calibration_method": "sigmoid",
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    result = app._get_active_optuna_best()

    assert result is not None
    assert result["balance_mode"] == "smote"
    assert result["best_model_params"] == {"n_estimators": 111}
    assert result["best_smote_params"] == {
        "smote_k_neighbors": 5,
        "smote_sampling_strategy": 0.4,
    }
    assert result["best_score"] == 0.91
    assert result["active_key"] == optuna_key
    assert result["model_choice"] == "XGBoost"


def test_get_active_optuna_best_returns_none_smote_params_when_only_none_mode(
    tmp_path, monkeypatch
):
    """Si solo existe ``balance_mode=none``, el helper promueve ese variante
    pero deja ``best_smote_params=None`` (mirror del disk-reloader)."""
    base_df, feature_cols, _b, _c = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    feature_key = app._feature_selection_key(
        str(tmp_path / "flow_features.duckdb"), "duckdb", features_df
    )
    optuna_key = app._optuna_result_key(feature_key, feature_cols)

    none_only_store = {
        optuna_key: {
            "results": {
                "XGBoost": {
                    "model_choice": "XGBoost",
                    "by_balance_mode": {
                        "none": {
                            "by_calibration_method": {
                                "sigmoid": {
                                    "model_choice": "XGBoost",
                                    "best_model_params": {"n_estimators": 333},
                                    "best_score": 0.80,
                                    "optuna_settings": {
                                        "balance_mode": "none",
                                        "calibration_method": "sigmoid",
                                        "best_feature_cols": list(feature_cols),
                                    },
                                    "search_space": {},
                                },
                            },
                        },
                    },
                }
            }
        }
    }

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "optuna_results_store": none_only_store,
            "optuna_active_key": optuna_key,
            "optuna_model_choice": "XGBoost",
            "optuna_calibration_method": "sigmoid",
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    result = app._get_active_optuna_best()

    assert result is not None
    assert result["balance_mode"] == "none"
    assert result["best_smote_params"] is None
    assert result["best_model_params"] == {"n_estimators": 333}


def test_get_active_optuna_best_returns_none_without_active_key(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "optuna_results_store": {"foo": {}},
            "optuna_active_key": None,
            "optuna_model_choice": "XGBoost",
            "optuna_calibration_method": "sigmoid",
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    assert app._get_active_optuna_best() is None


def test_get_active_optuna_best_matches_legacy_top_level_semantics(
    tmp_path, monkeypatch
):
    """Test de paridad: los campos devueltos por el helper deben coincidir
    uno a uno con los keys top-level ``optuna_best_*`` que escribe el
    disk-reloader en L9262-9286 de ``_render_optuna_tab``.

    Esto asegura que migrar un consumidor del key legacy al helper produce
    el mismo valor.
    """
    base_df, feature_cols, _b, _c = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    feature_key = app._feature_selection_key(
        str(tmp_path / "flow_features.duckdb"), "duckdb", features_df
    )
    optuna_key = app._optuna_result_key(feature_key, feature_cols)
    store = _make_optuna_store_with_smote_and_none(optuna_key, feature_cols)

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "optuna_results_store": store,
            "optuna_active_key": optuna_key,
            "optuna_model_choice": "XGBoost",
            "optuna_calibration_method": "sigmoid",
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    # Replica EXACTAMENTE la lógica del disk-reloader (L9243-9286) para
    # calcular cómo se habrían escrito los keys top-level.
    entry = store[optuna_key]
    model_result_legacy = app._get_optuna_model_result_variant(
        entry.get("results"),
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
        fallback_modes=["none"],
    )
    smote_result_legacy = app._get_optuna_model_result_variant(
        entry.get("results"),
        model_choice="XGBoost",
        balance_mode="smote",
        calibration_method="sigmoid",
    )
    legacy_top_level = {
        "optuna_best_smote_params": (
            smote_result_legacy.get("best_smote_params")
            if isinstance(smote_result_legacy, dict)
            else None
        ),
        "optuna_best_model_params": model_result_legacy.get("best_model_params"),
        "optuna_best_score": model_result_legacy.get("best_score"),
        "optuna_best_model_choice": "XGBoost",
        "optuna_best_settings": model_result_legacy.get("optuna_settings"),
        "optuna_best_search_space": model_result_legacy.get("search_space"),
    }

    result = app._get_active_optuna_best()

    assert result is not None
    assert result["best_smote_params"] == legacy_top_level["optuna_best_smote_params"]
    assert result["best_model_params"] == legacy_top_level["optuna_best_model_params"]
    assert result["best_score"] == legacy_top_level["optuna_best_score"]
    assert result["model_choice"] == legacy_top_level["optuna_best_model_choice"]
    assert result["optuna_settings"] == legacy_top_level["optuna_best_settings"]
    assert result["search_space"] == legacy_top_level["optuna_best_search_space"]


def test_history_optuna_batch_options_preserve_all_subruns(tmp_path, monkeypatch):
    base_df, feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "flow_features_path": str(tmp_path / "flow_features.duckdb"),
            "flow_features_source": "duckdb",
            "flow_features_tramo_label": "Toda la autopista",
            "accident_files": ["accidentes.csv"],
        }
    )
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(app, "HISTORY_PATH", tmp_path / "empty_history.jsonl")

    context = app._history_context_details(features_df, selected_features=None)
    db_path = app._history_db_path()
    history_store.set_meta(db_path, "legacy_seed_v1", "done")
    feature_key = app._feature_selection_key(
        fake_st.session_state["flow_features_path"],
        "duckdb",
        features_df,
    )
    current_fp = app._dataset_content_fingerprint(features_df)

    first_record = _make_optuna_batch_record(
        record_id=101,
        feature_key=feature_key,
        dataset_fingerprint=current_fp,
        feature_cols=base_cols,
        feature_set_label="Base",
    )
    second_record = _make_optuna_batch_record(
        record_id=102,
        feature_key=feature_key,
        dataset_fingerprint=current_fp,
        feature_cols=feature_cols,
        feature_set_label="Base + Cluster",
        balance_mode="smote",
        best_smote_params={
            "smote_k_neighbors": 5,
            "smote_sampling_strategy": 0.5,
        },
    )

    for record in (first_record, second_record):
        history_store.insert_record(
            db_path,
            stage="Optuna",
            record_uid=str(record["record_uid"]),
            created_at=str(record["created_at"]),
            feature_context_key=str(context["feature_context_key"]),
            batch_key="batch-optuna-001",
            model_name=str(record["model_name"]),
            threshold_objective=str(record["threshold_objective"]),
            calibration_method=str(record["calibration_method"]),
            balance_strategy=str(record["balance_strategy"]),
            metadata=record["metadata"],
        )

    options = app._history_optuna_batch_options_for_model_tab(
        feature_context_key=str(context["feature_context_key"])
    )

    assert len(options) == 1
    assert options[0]["subrun_count"] == 2
    assert len(options[0]["records"]) == 2
    assert len(options[0]["record_ids"]) == 2
    assert set(options[0]["feature_set_labels"]) == {"Base", "Base + Cluster"}


def test_build_model_optuna_batch_contract_expands_all_subruns_and_blocks_incompatible(
    tmp_path,
):
    base_df, feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    current_fp = app._dataset_content_fingerprint(features_df)
    feature_key = app._feature_selection_key(
        str(tmp_path / "flow_features.duckdb"),
        "duckdb",
        features_df,
    )

    option = {
        "token": "batch-001",
        "batch_key": "batch-001",
        "label": "Batch 001",
        "record_ids": [11, 12],
        "subrun_count": 2,
        "feature_set_labels": ["Base", "Base + Cluster"],
        "records": [
            _make_optuna_batch_record(
                record_id=11,
                feature_key=feature_key,
                dataset_fingerprint=current_fp,
                feature_cols=base_cols,
                feature_set_label="Base",
            ),
            _make_optuna_batch_record(
                record_id=12,
                feature_key=feature_key,
                dataset_fingerprint="fingerprint-distinto",
                feature_cols=feature_cols,
                feature_set_label="Base + Cluster",
                balance_mode="smote",
                best_smote_params={
                    "smote_k_neighbors": 5,
                    "smote_sampling_strategy": 0.5,
                },
            ),
        ],
    }

    contract = app._build_model_optuna_batch_contract(
        option,
        current_feature_key=feature_key,
        current_dataset_fingerprint=current_fp,
        base_df=base_df,
        cluster_df=base_df,
    )

    assert len(contract["subruns"]) == 2
    assert contract["subruns"][0]["compatible"] is True
    assert contract["subruns"][1]["compatible"] is False
    assert contract["compatible"] is False
    assert any(
        "fingerprint del dataset no coincide" in reason
        for reason in contract["reasons"]
    )


def test_build_model_optuna_batch_contract_rejects_heterogeneous_shared_metadata(
    tmp_path,
):
    base_df, _feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    current_fp = app._dataset_content_fingerprint(features_df)
    feature_key = app._feature_selection_key(
        str(tmp_path / "flow_features.duckdb"),
        "duckdb",
        features_df,
    )

    option = {
        "token": "batch-hetero",
        "batch_key": "batch-hetero",
        "label": "Batch heterogéneo",
        "records": [
            _make_optuna_batch_record(
                record_id=21,
                feature_key=feature_key,
                dataset_fingerprint=current_fp,
                feature_cols=base_cols,
                feature_set_label="Base",
                calibration_method="sigmoid",
            ),
            _make_optuna_batch_record(
                record_id=22,
                feature_key=feature_key,
                dataset_fingerprint=current_fp,
                feature_cols=base_cols,
                feature_set_label="Base",
                calibration_method="isotonic",
            ),
        ],
    }

    contract = app._build_model_optuna_batch_contract(
        option,
        current_feature_key=feature_key,
        current_dataset_fingerprint=current_fp,
        base_df=base_df,
        cluster_df=base_df,
    )

    assert contract["compatible"] is False
    assert any("batch heterogéneo" in reason for reason in contract["reasons"])


def test_optuna_subrun_candidates_from_variant_returns_all_pareto_candidates(
    tmp_path,
):
    pareto_front_path = tmp_path / "pareto_front.csv"
    pd.DataFrame(
        [
            {
                "trial_number": 3,
                "params_xgb_n_estimators": 150,
                "params_top_k": 2,
                "values_0": 0.61,
                "values_1": 0.18,
                "selected_trial": True,
            },
            {
                "trial_number": 5,
                "params_xgb_n_estimators": 220,
                "params_top_k": 3,
                "values_0": 0.58,
                "values_1": 0.14,
                "selected_trial": False,
            },
        ]
    ).to_csv(pareto_front_path, index=False)

    entry = {"feature_cols": ["f1", "f2", "f3"]}
    variant = {
        "best_model_params": {"n_estimators": 120},
        "best_smote_params": {"smote_k_neighbors": 5},
        "optuna_settings": {
            "objective_mode": app.CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
            "ranked_cols": ["f1", "f2", "f3"],
            "best_feature_cols": ["f1", "f2", "f3"],
            "best_top_k": 3,
        },
        "pareto_front_csv": str(pareto_front_path),
    }

    candidates, error = app._optuna_subrun_candidates_from_variant(
        entry=entry,
        variant=variant,
        model_choice="XGBoost",
        feature_cols_fallback=["f1", "f2", "f3"],
        best_model_params={"n_estimators": 120},
        best_smote_params={"smote_k_neighbors": 5},
    )

    assert error is None
    assert len(candidates) == 2
    assert candidates[0]["candidate_kind"] == "pareto"
    assert candidates[0]["feature_cols"] == ["f1", "f2"]
    assert candidates[1]["feature_cols"] == ["f1", "f2", "f3"]


def test_apply_model_history_record_to_state_restores_optuna_batch_selection(
    monkeypatch,
):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    record = {
        "model_name": "XGBoost",
        "threshold_objective": "far",
        "calibration_method": "sigmoid",
        "protocols": ["conservative", "robust"],
        "params": {
            "source_mode": "optuna_batch",
            "test_size": 0.2,
            "optuna_batch": {
                "batch_ref": {"token": "batch-token-77"},
            },
        },
    }

    app._apply_model_history_record_to_state(record)

    assert fake_st.session_state["model_feature_source"] == "optuna"
    assert fake_st.session_state["model_feature_source_radio"] == "Optuna (batch explícito)"
    assert fake_st.session_state["model_optuna_batch_token"] == "batch-token-77"
    assert fake_st.session_state["model_choice"] == "XGBoost"
    assert fake_st.session_state["model_threshold_protocols"] == [
        "Conservador",
        "Robusto",
    ]
    assert fake_st.session_state["test_size"] == 0.2


# =============================================================================
# Tests para los contratos de estado por tab (punto 5)
# =============================================================================


def test_main_navigation_uses_streamlit_tabs(monkeypatch):
    fake_st = _TabsFakeStreamlit()
    rendered_sections: list[str] = []
    expected_labels = [
        "Eventos",
        "Feature engineering",
        "Match",
        "Feature selection",
        "Optuna",
        "Balance",
        "Modelos",
        "History",
        "Experiments",
    ]
    renderers = {
        "_render_event_tab": "Eventos",
        "_render_variables_tab": "Feature engineering",
        "_render_match_tab": "Match",
        "_render_feature_selection_tab": "Feature selection",
        "_render_optuna_tab": "Optuna",
        "_render_balance_tab": "Balance",
        "_render_model_tab": "Modelos",
        "_render_history_tab": "History",
        "_render_experiments_tab": "Experiments",
    }

    monkeypatch.setattr(app, "st", fake_st)
    for function_name, label in renderers.items():
        monkeypatch.setattr(
            app,
            function_name,
            lambda section=label: rendered_sections.append(section),
        )

    app.main(set_page_config=False, show_exit_button=False)

    assert fake_st.tabs_calls == [expected_labels]
    assert rendered_sections == expected_labels
    assert fake_st.session_state.get("crash_prediction_active_section") is None


def test_validate_tab_state_unknown_tab_returns_error():
    issues = app._validate_tab_state("foobar", session_state={})
    assert len(issues) == 1
    assert issues[0]["level"] == "error"
    assert issues[0]["key"] == "__contract__"


def test_validate_tab_state_optuna_flags_missing_required():
    """Sin accidents_df / flow_features_df / selected_features se deben
    reportar 3 errores."""
    issues = app._validate_tab_state("optuna", session_state={})
    errors = [i for i in issues if i["level"] == "error"]
    keys = {i["key"] for i in errors}
    assert "accidents_df" in keys
    assert "flow_features_df" in keys
    assert "selected_features" in keys


def test_validate_tab_state_balance_tolerates_missing_selected_features():
    """Balance no exige ``selected_features`` — la tab puede operar con
    el dataset balanceado cargado de disco."""
    fake_state = {
        "accidents_df": pd.DataFrame({"x": [1]}),
        "flow_features_df": pd.DataFrame({"y": [1]}),
        # selected_features AUSENTE
    }
    issues = app._validate_tab_state("balance", session_state=fake_state)
    errors = [i for i in issues if i["level"] == "error"]
    assert errors == []


def test_validate_tab_state_flags_empty_dataframe_as_error():
    fake_state = {
        "accidents_df": pd.DataFrame(),
        "flow_features_df": pd.DataFrame({"x": [1]}),
        "selected_features": ["x"],
    }
    issues = app._validate_tab_state("optuna", session_state=fake_state)
    errors = [i for i in issues if i["level"] == "error"]
    empty_errors = [i for i in errors if i["key"] == "accidents_df"]
    assert empty_errors, "se esperaba un error por `accidents_df` vacío"
    assert "vacío" in empty_errors[0]["message"].lower()


def test_validate_tab_state_flags_empty_selected_features():
    fake_state = {
        "accidents_df": pd.DataFrame({"x": [1]}),
        "flow_features_df": pd.DataFrame({"x": [1]}),
        "selected_features": [],
    }
    issues = app._validate_tab_state("optuna", session_state=fake_state)
    sel_errors = [
        i for i in issues if i["key"] == "selected_features" and i["level"] == "error"
    ]
    assert sel_errors


def test_validate_tab_state_warns_on_orphan_optuna_active_key():
    """Si ``optuna_active_key`` no está en el store, se emite un warning."""
    fake_state = {
        "accidents_df": pd.DataFrame({"x": [1]}),
        "flow_features_df": pd.DataFrame({"x": [1]}),
        "selected_features": ["x"],
        "optuna_results_store": {"some_other_key": {}},
        "optuna_active_key": "missing_key",
    }
    issues = app._validate_tab_state("optuna", session_state=fake_state)
    warnings = [i for i in issues if i["level"] == "warning"]
    active_warns = [w for w in warnings if w["key"] == "optuna_active_key"]
    assert active_warns
    assert "missing_key" in active_warns[0]["message"]


def test_validate_tab_state_modelos_warns_when_no_balanced_data():
    fake_state = {
        "accidents_df": pd.DataFrame({"x": [1]}),
        "flow_features_df": pd.DataFrame({"x": [1]}),
        "balanced_base_df": None,
    }
    issues = app._validate_tab_state("modelos", session_state=fake_state)
    warns = [i for i in issues if i["key"] == "balanced_base_df"]
    assert warns
    assert warns[0]["level"] == "warning"


def test_reset_tab_state_keys_only_removes_produced_keys():
    """El reset solo borra keys ``produces``; los ``required`` quedan intactos
    para que el usuario no pierda los datasets cargados."""
    accidents = pd.DataFrame({"x": [1]})
    features = pd.DataFrame({"y": [1]})
    fake_state = {
        # required — no se tocan
        "accidents_df": accidents,
        "flow_features_df": features,
        "selected_features": ["y"],
        # produces — deben desaparecer
        "optuna_results_store": {"k": {}},
        "optuna_active_key": "k",
        "optuna_best_model_params": {"n_estimators": 100},
        "optuna_trials_df": pd.DataFrame(),
    }
    removed = app._reset_tab_state_keys("optuna", session_state=fake_state)

    assert "optuna_results_store" in removed
    assert "optuna_active_key" in removed
    assert "optuna_best_model_params" in removed
    assert "optuna_trials_df" in removed

    # required intactos
    assert fake_state["accidents_df"] is accidents
    assert fake_state["flow_features_df"] is features
    assert fake_state["selected_features"] == ["y"]
    # produces borrados
    assert "optuna_results_store" not in fake_state
    assert "optuna_active_key" not in fake_state


def test_reset_tab_state_keys_unknown_tab_is_noop():
    fake_state = {"foo": "bar"}
    removed = app._reset_tab_state_keys("unknown_tab", session_state=fake_state)
    assert removed == []
    assert fake_state == {"foo": "bar"}


def test_tab_state_contracts_are_disjoint_required_vs_produces():
    """Un key declarado como ``required`` no debería estar también en
    ``produces`` del MISMO contrato — eso significaría que el reset borraría
    un dato que la misma tab necesita para operar."""
    for tab, contract in app._TAB_STATE_CONTRACTS.items():
        required = set(contract.get("required", []))
        produces = set(contract.get("produces", []))
        overlap = required & produces
        assert not overlap, (
            f"tab {tab!r}: keys en required ∩ produces = {overlap} "
            "(el reset borraría un key requerido)"
        )


def test_controlled_feature_date_bounds_and_loader_filter_interval(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    features_path = tmp_path / "features.duckdb"
    raw_df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                ["2024-01-01 08:00:00", "2024-01-02 08:00:00", "2024-01-03 08:00:00"]
            ),
            "eje": ["N", "N", "N"],
            "calzada": ["Oriente", "Oriente", "Oriente"],
            "portico_last": ["1", "1", "1"],
            "portico_next": ["2", "2", "2"],
            "flow_light": [10.0, 11.0, 12.0],
            "cluster_count_0": [1.0, 2.0, 3.0],
        }
    )
    con = duckdb.connect(str(features_path))
    try:
        con.register("features_view", raw_df)
        con.execute("CREATE TABLE flow_features AS SELECT * FROM features_view")
    finally:
        con.close()

    bounds = app._controlled_feature_timestamp_bounds(features_path)
    assert bounds is not None
    assert bounds[0] == pd.Timestamp("2024-01-01 08:00:00")
    assert bounds[1] == pd.Timestamp("2024-01-03 08:00:00")

    filtered = app._load_controlled_features_df(
        features_path,
        ("N", "Oriente", "1", "2"),
        date_start=pd.Timestamp("2024-01-02 00:00:00"),
        date_end=pd.Timestamp("2024-01-02 23:59:59"),
    )

    assert len(filtered) == 1
    assert filtered["interval_start"].iloc[0] == pd.Timestamp("2024-01-02 08:00:00")
    assert filtered["flow_light"].iloc[0] == pytest.approx(11.0)


def test_load_feature_segment_catalog_uses_duckdb_metadata_and_porticos(
    tmp_path,
    monkeypatch,
):
    duckdb = pytest.importorskip("duckdb")
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    features_path = tmp_path / "segments.duckdb"
    raw_df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 08:05:00",
                    "2024-01-03 08:00:00",
                ]
            ),
            "portico_last": ["1", "1", "2"],
            "portico_next": ["2", "2", "3"],
            "cluster_count_0": [1.0, 2.0, 3.0],
        }
    )
    con = duckdb.connect(str(features_path))
    try:
        con.register("features_view", raw_df)
        con.execute("CREATE TABLE flow_features AS SELECT * FROM features_view")
    finally:
        con.close()

    porticos_df = pd.DataFrame(
        {
            "eje": ["N", "N", "N"],
            "calzada": ["Oriente", "Oriente", "Oriente"],
            "orden": [1, 2, 3],
            "km": [10.0, 11.0, 12.0],
            "portico": ["1", "2", "3"],
        }
    )
    monkeypatch.setattr(app, "load_porticos", lambda: porticos_df.copy())

    segments = app._load_feature_segment_catalog(
        features_path,
        date_start=pd.Timestamp("2024-01-01 00:00:00"),
        date_end=pd.Timestamp("2024-01-02 23:59:59"),
    )

    assert len(segments) == 1
    assert segments.loc[0, "portico_last"] == "1"
    assert segments.loc[0, "portico_next"] == "2"
    assert segments.loc[0, "eje"] == "N"
    assert segments.loc[0, "calzada"] == "Oriente"
    assert segments.loc[0, "km_last"] == pytest.approx(10.0)
    assert segments.loc[0, "km_next"] == pytest.approx(11.0)


def test_load_accidents_for_event_can_skip_session_cache(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)

    processed_df = pd.DataFrame(
        {
            "accidente_time": pd.to_datetime(["2024-01-01 08:00:00"]),
            "ultimo_portico": ["1"],
            "proximo_portico": ["2"],
        }
    )
    monkeypatch.setattr(
        app,
        "read_csv_with_progress",
        lambda _path: pd.DataFrame({"raw": [1]}),
    )
    monkeypatch.setattr(
        app,
        "load_porticos",
        lambda: pd.DataFrame({"portico": ["1", "2"]}),
    )
    monkeypatch.setattr(
        app,
        "process_accidentes_df",
        lambda *_args, **_kwargs: (processed_df.copy(), pd.DataFrame()),
    )

    loaded = app._load_accidents_for_event(
        Path("/tmp/fake_events.csv"),
        cache_in_session=False,
    )

    assert loaded is not None
    assert loaded.equals(processed_df)
    assert fake_st.session_state["accidents_by_event_cache"] == {}


def test_prepare_controlled_comparison_base_df_reuses_prefetched_features(
    monkeypatch,
):
    tramo_tuple = ("N", "Oriente", "1", "2")
    accidents_df = pd.DataFrame(
        {
            "accidente_time": pd.to_datetime(
                ["2024-01-01 08:00:00", "2024-01-01 08:05:00"]
            ),
            "ultimo_portico": ["1", "1"],
            "proximo_portico": ["2", "2"],
        }
    )
    features_df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                ["2024-01-01 08:00:00", "2024-01-01 08:05:00"]
            ),
            "portico_last": ["1", "1"],
            "portico_next": ["2", "2"],
            "cluster_count_0": [1.0, 2.0],
        }
    )

    def _fail_loader(*_args, **_kwargs):
        raise AssertionError("No debería recargar features si ya vienen prefetched.")

    def _fake_add_accident_target(prefetched_features, _accidents_segment):
        result = prefetched_features.copy()
        result["target"] = [0, 1]
        return result

    monkeypatch.setattr(app, "_load_controlled_features_df", _fail_loader)
    monkeypatch.setattr(app, "add_accident_target", _fake_add_accident_target)

    base_df = app._prepare_controlled_comparison_base_df(
        accidents_df_for_tramo=accidents_df,
        selected_features_path=Path("/tmp/unused.duckdb"),
        tramo_tuple=tramo_tuple,
        features_df=features_df,
    )

    assert list(base_df["cluster_count_0"]) == [1.0, 2.0]
    assert list(base_df["target"]) == [0, 1]


def test_controlled_comparison_metric_options_include_extended_metrics():
    df = pd.DataFrame(
        [
            {
                "val_objective_score": 0.81,
                "test_objective_score": 0.79,
                "test_accuracy": 0.95,
                "test_pr_auc": 0.22,
                "test_brier_score": 0.14,
                "test_mcc": 0.11,
                "test_false_negatives": 3,
                "decision_threshold": 0.45,
            }
        ]
    )

    options = app._controlled_comparison_metric_options(
        df,
        objective_label="ROC-AUC",
    )
    labels_by_col = {column: label for label, column in options}

    assert labels_by_col["val_objective_score"] == "Validación ROC-AUC (objetivo)"
    assert labels_by_col["test_accuracy"] == "Test Accuracy"
    assert labels_by_col["test_pr_auc"] == "Test PR-AUC"
    assert labels_by_col["test_brier_score"] == "Test Brier"
    assert labels_by_col["test_mcc"] == "Test MCC"
    assert labels_by_col["test_false_negatives"] == "Test Falsos Negativos"
    assert labels_by_col["decision_threshold"] == "Threshold de decisión"


def test_optuna_objective_options_include_brier_and_mcc():
    options = app._optuna_objective_options()

    assert options["MCC"] == {"key": "mcc", "direction": "maximize"}
    assert options["Brier (menor es mejor)"] == {
        "key": "brier_score",
        "direction": "minimize",
    }


def test_prepare_controlled_comparison_detail_display_formats_all_values():
    detail_df = pd.DataFrame(
        [
            {
                "model_name": "XGBoost",
                "feature_set": "Base",
                "balance_mode": "none",
                "k": 25,
                "status": "completed",
                "selected_features": '["f1","f2"]',
                "best_params": '{"max_depth": 4}',
                "smote_params": '{"sampling_strategy": 0.5}',
                "test_confusion_matrix": "[[10, 2], [3, 4]]",
                "val_confusion_matrix": [[8, 1], [2, 5]],
                "test_accuracy": 0.91,
            }
        ]
    )

    display_df = app._prepare_controlled_comparison_detail_display(detail_df)

    assert list(display_df.columns[:4]) == [
        "model_name",
        "feature_set",
        "balance_mode",
        "k",
    ]
    row = display_df.iloc[0]
    assert row["selected_features"] == "f1, f2"
    assert "max_depth" in row["best_params"]
    assert "sampling_strategy" in row["smote_params"]
    assert row["test_confusion_matrix"] == "[[10, 2], [3, 4]]"
    assert row["val_confusion_matrix"] == "[[8, 1], [2, 5]]"


def test_prepare_controlled_comparison_detail_display_keeps_ablation_columns():
    detail_df = pd.DataFrame(
        [
            {
                "model_name": "XGBoost",
                "feature_set": "Base + Cluster",
                "ablation_phase": "cross_eval",
                "params_source_feature_set": "Base",
                "target_feature_set": "Base + Cluster",
                "source_combo_id": "ablation_source",
                "frozen_tuning": True,
                "threshold_freeze_policy": "recalibrate_per_target",
                "balance_mode": "smote",
                "threshold_protocol": "conservative",
                "k": 2,
                "effective_k": 2,
                "selected_base_feature_count": 1,
                "selected_cluster_feature_count": 1,
                "selected_features": '["flow","cluster_count_0"]',
                "best_params": '{"source": "Base"}',
                "effective_model_params": '{"source": "Base", "n_jobs": 1}',
                "smote_params": '{"sampling_strategy": 0.5}',
                "status": "completed",
            }
        ]
    )

    display_df = app._prepare_controlled_comparison_detail_display(detail_df)
    expected_cols = [
        "model_name",
        "feature_set",
        "ablation_phase",
        "params_source_feature_set",
        "target_feature_set",
        "source_combo_id",
        "frozen_tuning",
        "threshold_freeze_policy",
    ]

    assert list(display_df.columns[: len(expected_cols)]) == expected_cols
    row = display_df.iloc[0]
    assert row["selected_features"] == "flow, cluster_count_0"
    assert "source" in row["best_params"]
    assert "n_jobs" in row["effective_model_params"]
    assert "sampling_strategy" in row["smote_params"]


def test_calibration_sweep_best_feature_cols_prefers_effective_variables():
    best_summary_df = pd.DataFrame(
        [
            {
                "rank": 1,
                "selected_features": json.dumps(["pool_a", "pool_b", "pool_c"]),
                "best_feature_cols": json.dumps(["pool_b", "pool_c"]),
            }
        ]
    )
    leaderboard_df = pd.DataFrame()
    grid_results_df = pd.DataFrame()

    result_state = {"protocol": {"selected_features": ["pool_a", "pool_b", "pool_c"]}}
    assert app._calibration_sweep_selected_features(
        result_state,
        pd.DataFrame({"selected_features": [json.dumps(["other"])]}),
    ) == ["pool_a", "pool_b", "pool_c"]

    assert app._calibration_sweep_best_feature_cols(
        best_summary_df=best_summary_df,
        leaderboard_df=leaderboard_df,
        grid_results_df=grid_results_df,
    ) == ["pool_b", "pool_c"]

    fallback_df = best_summary_df.drop(columns=["best_feature_cols"])
    assert app._calibration_sweep_best_feature_cols(
        best_summary_df=fallback_df,
        leaderboard_df=leaderboard_df,
        grid_results_df=grid_results_df,
    ) == ["pool_a", "pool_b", "pool_c"]


def test_prepare_dataframe_for_streamlit_stringifies_mixed_metadata_value_column():
    df = pd.DataFrame(
        [
            {"grupo": "Corrida", "campo": "run_id", "valor": "calibration_001"},
            {"grupo": "Progreso", "campo": "completed_steps", "valor": 7},
            {"grupo": "Protocolo", "campo": "far_target", "valor": 0.2},
            {"grupo": "Corrida", "campo": "last_error", "valor": None},
        ]
    )

    safe_df = app._prepare_dataframe_for_streamlit(df)

    assert safe_df["valor"].tolist() == ["calibration_001", "7", "0.2", None]


def test_seed_controlled_comparison_live_db_backfills_checkpoint_results(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    db_path = app._init_experiment_db(
        "Controlled comparison",
        {"dataset_name": "events.csv", "features_name": "features.duckdb"},
    )
    assert db_path is not None

    checkpoint_run_dir = tmp_path / "controlled_run"
    results_dir = checkpoint_run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "combo_id": "combo__rf__base__none__k5",
                "run_id": "controlled_001",
                "model_name": "Random Forest",
                "feature_set": "Base",
                "balance_mode": "none",
                "k": 5,
                "status": "completed",
                "val_objective_score": 0.73,
            }
        ]
    ).to_csv(results_dir / "grid_results.csv", index=False)

    seeded = app._seed_controlled_comparison_live_db(
        db_path,
        checkpoint_run_dir=checkpoint_run_dir,
        dataset_name="events.csv",
        features_name="features.duckdb",
        segment_info={"segment_label": "P1 -> P2"},
    )

    assert seeded == 1

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT payload_json FROM results ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    assert len(rows) == 1
    payload = json.loads(rows[0][0])
    assert payload["experiment"] == "Controlled comparison"
    assert payload["dataset_name"] == "events.csv"
    assert payload["features_name"] == "features.duckdb"
    assert payload["segment_info"]["segment_label"] == "P1 -> P2"
    assert payload["combo_id"] == "combo__rf__base__none__k5"


def test_load_controlled_comparison_result_frames_reads_paths(tmp_path):
    summary_df = pd.DataFrame([{"model_name": "Random Forest", "k_optimo": 5}])
    curves_df = pd.DataFrame([{"model_name": "Random Forest", "k": 5}])
    detail_df = pd.DataFrame([{"combo_id": "combo__rf__base__none__k5"}])
    deltas_df = pd.DataFrame([{"effect_type": "feature_effect", "k": 5}])

    summary_path = tmp_path / "summary.csv"
    curves_path = tmp_path / "curves.csv"
    detail_path = tmp_path / "detail.csv"
    deltas_path = tmp_path / "deltas.csv"
    summary_df.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    deltas_df.to_csv(deltas_path, index=False)

    loaded_summary, loaded_curves, loaded_detail, loaded_deltas = (
        app._load_controlled_comparison_result_frames(
            {
                "summary_path": str(summary_path),
                "curves_path": str(curves_path),
                "detail_path": str(detail_path),
                "ablation_deltas_path": str(deltas_path),
            }
        )
    )

    pd.testing.assert_frame_equal(loaded_summary, summary_df)
    pd.testing.assert_frame_equal(loaded_curves, curves_df)
    pd.testing.assert_frame_equal(loaded_detail, detail_df)
    pd.testing.assert_frame_equal(loaded_deltas, deltas_df)


def test_load_controlled_comparison_result_frames_tolerates_legacy_without_deltas(tmp_path):
    summary_df = pd.DataFrame([{"model_name": "Random Forest", "k_optimo": 5}])
    summary_path = tmp_path / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    loaded_summary, loaded_curves, loaded_detail, loaded_deltas = (
        app._load_controlled_comparison_result_frames(
            {
                "summary_path": str(summary_path),
            }
        )
    )

    pd.testing.assert_frame_equal(loaded_summary, summary_df)
    assert loaded_curves.empty
    assert loaded_detail.empty
    assert loaded_deltas.empty


def test_history_protocol_results_summary_keeps_training_curves():
    summary = app._history_protocol_results_summary(
        {
            "Base": {
                "conservative": {
                    "metrics": {"f1": 0.42},
                    "training_curves": {
                        "epochs": [1, 2],
                        "train_loss": [0.8, 0.5],
                        "val_loss": [0.9, 0.6],
                    },
                }
            }
        }
    )

    assert summary["Base"]["conservative"]["training_curves"]["train_loss"] == [
        0.8,
        0.5,
    ]
