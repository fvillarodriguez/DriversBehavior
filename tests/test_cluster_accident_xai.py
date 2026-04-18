from __future__ import annotations

import json
import sqlite3
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


# =============================================================================
# Tests para los contratos de estado por tab (punto 5)
# =============================================================================


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
