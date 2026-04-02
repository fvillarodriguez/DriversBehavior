from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.nlp_severity_app as nlp_app
from src.nlp_severity_app import (
    _candidate_feature_caps,
    _classification_metrics,
    _paper_candidate_k_values,
    _paper_compare_routes,
    _paper_normalize_cv_folds,
    _paper_normalize_k_grid,
    _paper_protocol_config,
    _prepare_holdout_split_with_ids,
    _project_transformer_hidden_state,
    _annotate_events_with_flow_coverage,
    _sample_accidents_for_feature_engineering,
    build_severity_feature_dataset,
    generate_text_embeddings,
)
from src.utils import FLOW_TABLE_NAME, FlowSampleSelection


def _build_flow_db(path: Path) -> Path:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            f"""
            CREATE TABLE {FLOW_TABLE_NAME} (
                FECHA TIMESTAMP,
                VELOCIDAD DOUBLE,
                CATEGORIA INTEGER,
                MATRICULA VARCHAR,
                PORTICO VARCHAR,
                CARRIL VARCHAR
            )
            """
        )
        flow_rows = [
            ("2024-01-01 10:04:30", 60.0, 1, "AA11", "P1", "1"),
            ("2024-01-01 10:00:30", 50.0, 1, "AA12", "P1", "1"),
            ("2024-01-01 10:05:20", 40.0, 2, "BB21", "P2", "1"),
            ("2024-01-01 10:06:20", 30.0, 3, "BB22", "P2", "1"),
        ]
        con.executemany(
            f"INSERT INTO {FLOW_TABLE_NAME} VALUES (?, ?, ?, ?, ?, ?)",
            flow_rows,
        )
    finally:
        con.close()
    return path


def test_build_severity_feature_dataset_from_duckdb(tmp_path: Path):
    flow_db = _build_flow_db(tmp_path / "flows.duckdb")
    accidents_df = pd.DataFrame(
        [
            {
                "accidente_time": pd.Timestamp("2024-01-01 10:05:00"),
                "Km.": 10.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Prueba controlada",
                "duracion_accidente": 15,
                "severidad": 1,
                "severity_target": 1,
                "ultimo_portico": "P1",
                "proximo_portico": "P2",
                "source_files": "test.csv",
            }
        ]
    )

    features_df, granular_df, ranking_df = build_severity_feature_dataset(
        accidents_df,
        flow_db_path=flow_db,
        windows_before=5,
        windows_after=5,
        window_size_minutes=1,
        top_k_ranking=5,
        text_columns=["accidente_time", "km", "eje", "calzada", "subtipo", "descripcion"],
    )

    assert len(features_df) == 1
    assert not granular_df.empty
    assert "flow_light_ultimo_before_min1" in features_df.columns
    assert "flow_heavy_proximo_after_min1" in features_df.columns
    assert features_df.loc[0, "flow_light_ultimo_before_min1"] == 1
    assert features_df.loc[0, "speed_mean_light_ultimo_before_min1"] == 60.0
    assert features_df.loc[0, "flow_heavy_proximo_after_min1"] == 1
    assert features_df.loc[0, "flow_heavy_proximo_after_min2"] == 1
    assert "text_bert" in features_df.columns
    assert "full_text_bert" not in features_df.columns
    assert features_df.loc[0, "text_bert"]
    assert "Lunes 10:05" in features_df.loc[0, "text_bert"]
    assert "km=10.5" in features_df.loc[0, "text_bert"]
    assert "subtipo=Choque" in features_df.loc[0, "text_bert"]
    assert "descripcion=Prueba controlada" in features_df.loc[0, "text_bert"]
    assert "flow_light_ultimo_before_min1" not in features_df.loc[0, "text_bert"]
    assert ranking_df is not None


def test_build_severity_feature_dataset_custom_window_size(tmp_path: Path):
    flow_db = _build_flow_db(tmp_path / "flows_custom.duckdb")
    accidents_df = pd.DataFrame(
        [
            {
                "accidente_time": pd.Timestamp("2024-01-01 10:05:00"),
                "Km.": 10.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Prueba controlada",
                "duracion_accidente": 15,
                "severidad": 1,
                "severity_target": 1,
                "ultimo_portico": "P1",
                "proximo_portico": "P2",
                "source_files": "test.csv",
            }
        ]
    )

    features_df, granular_df, _ = build_severity_feature_dataset(
        accidents_df,
        flow_db_path=flow_db,
        windows_before=3,
        windows_after=2,
        window_size_minutes=2,
        selected_metrics=["flow"],
        include_deltas=False,
        text_columns=["accidente_time", "descripcion"],
    )

    assert "flow_light_ultimo_before_w2m_1" in features_df.columns
    assert "flow_heavy_proximo_after_w2m_1" in features_df.columns
    assert "speed_mean_light_ultimo_before_w2m_1" not in features_df.columns
    assert not any(col.startswith("delta_") for col in features_df.columns)
    assert features_df.loc[0, "flow_light_ultimo_before_w2m_1"] == 1
    assert features_df.loc[0, "flow_heavy_proximo_after_w2m_1"] == 2
    assert "window_size_minutes" in granular_df.columns
    assert granular_df["window_size_minutes"].eq(2).all()
    assert list(granular_df.columns) == [
        "accident_id",
        "anchor",
        "direction",
        "minute_idx",
        "window_size_minutes",
        "category_label",
        "flow",
    ]


def test_build_severity_feature_dataset_excludes_accidents_without_flow_coverage(tmp_path: Path):
    flow_db = _build_flow_db(tmp_path / "flows_coverage.duckdb")
    accidents_df = pd.DataFrame(
        [
            {
                "accidente_time": pd.Timestamp("2024-01-01 10:05:00"),
                "Km.": 10.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Con cobertura",
                "duracion_accidente": 15,
                "severidad": 1,
                "severity_target": 1,
                "ultimo_portico": "P1",
                "proximo_portico": "P2",
                "source_files": "test.csv",
            },
            {
                "accidente_time": pd.Timestamp("2024-01-01 12:00:00"),
                "Km.": 11.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Sin cobertura",
                "duracion_accidente": 20,
                "severidad": 0,
                "severity_target": 0,
                "ultimo_portico": "P1",
                "proximo_portico": "P2",
                "source_files": "test.csv",
            },
        ]
    )

    features_df, granular_df, _ = build_severity_feature_dataset(
        accidents_df,
        flow_db_path=flow_db,
        windows_before=5,
        windows_after=5,
        window_size_minutes=1,
        text_columns=["accidente_time", "descripcion"],
    )

    assert len(features_df) == 1
    assert len(granular_df["accident_id"].unique()) == 1
    assert features_df.iloc[0]["descripcion"] == "Con cobertura"
    assert "Sin cobertura" not in features_df["descripcion"].tolist()


def test_annotate_events_with_flow_coverage_marks_rows(tmp_path: Path):
    flow_db = _build_flow_db(tmp_path / "flows_events.duckdb")
    accidents_df = pd.DataFrame(
        [
            {
                "accidente_time": pd.Timestamp("2024-01-01 10:05:00"),
                "Km.": 10.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Con cobertura",
                "duracion_accidente": 15,
                "severidad": 1,
                "severity_target": 1,
                "ultimo_portico": "P1",
                "proximo_portico": "P2",
                "source_files": "test.csv",
            },
            {
                "accidente_time": pd.Timestamp("2024-01-01 12:00:00"),
                "Km.": 11.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Sin cobertura",
                "duracion_accidente": 20,
                "severidad": 0,
                "severity_target": 0,
                "ultimo_portico": "P1",
                "proximo_portico": "P2",
                "source_files": "test.csv",
            },
        ]
    )

    marked_df, meta = _annotate_events_with_flow_coverage(
        accidents_df,
        flow_db_path=str(flow_db),
        windows_before=5,
        windows_after=5,
        window_size_minutes=1,
    )

    assert marked_df["has_flow_coverage"].tolist() == [True, False]
    assert marked_df["flow_coverage_label"].tolist() == ["Con datos de flujo", "Sin datos de flujo"]
    assert meta["coverage_evaluated"] is True
    assert meta["covered_events"] == 1
    assert meta["uncovered_events"] == 1


def test_annotate_events_with_flow_coverage_respects_before_after_ranges(tmp_path: Path):
    flow_db = _build_flow_db(tmp_path / "flows_directional.duckdb")
    accidents_df = pd.DataFrame(
        [
            {
                "accidente_time": pd.Timestamp("2024-01-01 10:05:00"),
                "Km.": 10.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Cobertura antes",
                "duracion_accidente": 15,
                "severidad": 1,
                "severity_target": 1,
                "ultimo_portico": "P1",
                "proximo_portico": None,
                "source_files": "test.csv",
            },
            {
                "accidente_time": pd.Timestamp("2024-01-01 10:05:00"),
                "Km.": 11.5,
                "Eje": "Ruta 1",
                "Calzada": "Oriente",
                "Tipo": "Accidente",
                "SubTipo": "Choque",
                "Descripcion": "Cobertura despues",
                "duracion_accidente": 20,
                "severidad": 0,
                "severity_target": 0,
                "ultimo_portico": None,
                "proximo_portico": "P2",
                "source_files": "test.csv",
            },
        ]
    )

    before_df, _ = _annotate_events_with_flow_coverage(
        accidents_df,
        flow_db_path=str(flow_db),
        windows_before=1,
        windows_after=0,
        window_size_minutes=1,
    )
    after_df, _ = _annotate_events_with_flow_coverage(
        accidents_df,
        flow_db_path=str(flow_db),
        windows_before=0,
        windows_after=2,
        window_size_minutes=1,
    )

    assert before_df["has_flow_coverage"].tolist() == [True, False]
    assert after_df["has_flow_coverage"].tolist() == [False, True]


def test_generate_text_embeddings_tfidf_svd():
    df = pd.DataFrame(
        {
            "accident_id": ["a1", "a2", "a3", "a4"],
            "text_bert": [
                "choque leve en portico uno",
                "accidente severo con corte de pista",
                "falla mecanica y colision menor",
                "impacto severo con lesionados",
            ],
            "severity_target": [0, 1, 0, 1],
        }
    )

    embedded_df, embed_cols, meta = generate_text_embeddings(
        df,
        text_col="text_bert",
        method="tfidf_svd",
        n_components=3,
        max_features=200,
    )

    assert len(embedded_df) == len(df)
    assert len(embed_cols) >= 2
    assert all(col in embedded_df.columns for col in embed_cols)
    assert meta["method"] == "tfidf_svd"


def test_candidate_feature_caps_respects_requested_cap():
    assert _candidate_feature_caps(8, max_features=100) == [8]
    assert _candidate_feature_caps(240, max_features=100) == [100]


def test_prepare_holdout_split_with_ids_returns_shared_test_ids():
    df = pd.DataFrame(
        [
            {
                "accident_id": f"a{i}",
                "accidente_time": pd.Timestamp("2024-01-01 10:00:00") + pd.Timedelta(minutes=i),
                "flow_a": float(i),
                "flow_b": float(i % 3),
                "severity_target": int(i % 2),
            }
            for i in range(10)
        ]
    )

    X_train, X_test, y_train, y_test, test_ids, meta = _prepare_holdout_split_with_ids(
        df,
        ["flow_a", "flow_b"],
        test_size=0.2,
        random_state=42,
        split_mode="Estratificado",
    )

    assert len(X_test) == len(y_test) == len(test_ids)
    assert meta["split_mode"] == "Estratificado"
    assert meta["comparison_id_field"] == "accident_id"
    assert meta["train_rows"] + meta["test_rows"] == len(df)
    assert set(test_ids.tolist()).issubset(set(df["accident_id"].tolist()))


def test_sample_accidents_for_feature_engineering_modes():
    accidents_df = pd.DataFrame(
        {
            "accidente_time": pd.date_range("2024-01-01 00:00:00", periods=10, freq="h"),
            "severity_target": [0, 1] * 5,
        }
    )

    range_sample = FlowSampleSelection(
        date_start=pd.Timestamp("2024-01-01 02:00:00"),
        date_end=pd.Timestamp("2024-01-01 05:00:00"),
        row_limit=None,
    )
    range_df = _sample_accidents_for_feature_engineering(
        accidents_df,
        range_sample,
        mode="Rango de fechas",
    )
    assert len(range_df) == 4
    assert range_df["accidente_time"].min() == pd.Timestamp("2024-01-01 02:00:00")
    assert range_df["accidente_time"].max() == pd.Timestamp("2024-01-01 05:00:00")

    percent_sample = FlowSampleSelection(date_start=None, date_end=None, row_limit=3)
    percent_df = _sample_accidents_for_feature_engineering(
        accidents_df,
        percent_sample,
        mode="Porcentaje",
        sample_seed=7,
    )
    assert len(percent_df) == 3
    assert percent_df["accidente_time"].is_monotonic_increasing


def test_hf_model_api_url_preserves_namespace_separator():
    api_url = nlp_app._hf_model_api_url("dccuchile/bert-base-spanish-wwm-cased")

    assert api_url == "https://huggingface.co/api/models/dccuchile/bert-base-spanish-wwm-cased"
    assert "%2F" not in api_url


def test_fetch_hf_model_siblings_uses_plain_hf_repo_path(monkeypatch: pytest.MonkeyPatch):
    requested: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"siblings": [{"rfilename": "pytorch_model.bin"}]}).encode("utf-8")

    def fake_urlopen(request, timeout=0):
        requested["url"] = request.full_url
        requested["timeout"] = timeout
        return _FakeResponse()

    nlp_app._fetch_hf_model_siblings.clear()
    monkeypatch.setattr(nlp_app, "urlopen", fake_urlopen)

    result = nlp_app._fetch_hf_model_siblings("dccuchile/bert-base-spanish-wwm-cased")

    assert requested["url"] == "https://huggingface.co/api/models/dccuchile/bert-base-spanish-wwm-cased"
    assert requested["timeout"] == 6
    assert result == {"ok": True, "siblings": ["pytorch_model.bin"]}


def test_transformer_model_options_include_existing_local_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    local_dir = tmp_path / "local_beto_safetensors"
    local_dir.mkdir()
    monkeypatch.setattr(
        nlp_app,
        "LOCAL_TRANSFORMER_MODEL_LOCATIONS",
        (("Local · BETO safetensors", local_dir),),
    )

    options = nlp_app._transformer_model_options()

    assert options["Local · BETO safetensors"] == str(local_dir)
    assert "dccuchile/bert-base-spanish-wwm-cased" in options


def _paper_compare_fixture_payload() -> dict:
    expected = copy.deepcopy(nlp_app.PAPER_EXPECTED_COUNTS)
    class_metrics = {
        "0": {"precision": 0.80, "recall": 0.70, "f1_score": 0.75, "support": 94},
        "1": {"precision": 0.92, "recall": 0.95, "f1_score": 0.93, "support": 320},
    }
    model_results = []
    for model_code, selected_k in [("M1", 70), ("M2", 150), ("M3", 300)]:
        model_results.append(
            {
                "model_code": model_code,
                "selected_k": selected_k,
                "metrics": {
                    "accuracy": 0.90,
                    "precision": 0.91,
                    "recall": 0.95,
                    "f1_score": 0.93,
                    "roc_auc": 0.96,
                    "false_negatives_positive_class": 16,
                    "class_metrics": copy.deepcopy(class_metrics),
                },
            }
        )
    return {
        "status": "ok",
        "route_name": "fixture",
        "dataset_validation": {
            "rows": expected["rows"],
            "flow_features": expected["flow_features"],
            "embedding_features": expected["embedding_features"],
            "total_features": expected["total_features"],
            "train_rows": expected["train_rows"],
            "test_rows": expected["test_rows"],
            "train_class_counts": copy.deepcopy(expected["train_class_counts"]),
            "test_class_counts": copy.deepcopy(expected["test_class_counts"]),
        },
        "model_results": model_results,
    }


def _paper_route_stub_payload(route_name: str) -> dict:
    expected = copy.deepcopy(nlp_app.PAPER_EXPECTED_COUNTS)
    return {
        "status": "ok",
        "status_message": "",
        "route_name": route_name,
        "route_metadata": {},
        "dataset_validation": {
            "rows": expected["rows"],
            "flow_features": expected["flow_features"],
            "embedding_features": expected["embedding_features"],
            "total_features": expected["total_features"],
            "train_rows": expected["train_rows"],
            "test_rows": expected["test_rows"],
            "train_class_counts": copy.deepcopy(expected["train_class_counts"]),
            "test_class_counts": copy.deepcopy(expected["test_class_counts"]),
            "is_valid": True,
        },
        "model_results": [],
        "comparison_df": pd.DataFrame([{"model_code": "M3", "accuracy": 0.9, "selected_k": 200}]),
        "metricas_df": pd.DataFrame(
            [{"class_label": "All", "metric": "False negatives", "M1": 1, "M2": 1, "M3": 1}]
        ),
        "predictions_df": pd.DataFrame(),
        "m3_grid_df": pd.DataFrame(
            [
                {
                    "k": 200,
                    "accuracy": 0.9,
                    "f1_score": 0.9,
                    "false_negatives_pct": 1.0,
                    "validation_score": 0.9,
                    "training_time_sec": 1.0,
                }
            ]
        ),
    }


def test_transformer_finetuned_cls_projection_uses_first_token():
    if nlp_app.torch is None:
        pytest.skip("torch no esta disponible en el entorno de tests.")
    hidden_state = nlp_app.torch.tensor(
        [
            [[10.0, 20.0], [1.0, 2.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]],
        ]
    )
    attention_mask = nlp_app.torch.tensor([[1, 1, 1], [1, 1, 0]])

    cls_projection = _project_transformer_hidden_state(
        hidden_state,
        attention_mask,
        projection="cls",
    )
    mean_projection = _project_transformer_hidden_state(
        hidden_state,
        attention_mask,
        projection="mean",
    )

    assert cls_projection.tolist() == [[10.0, 20.0], [7.0, 8.0]]
    assert cls_projection.tolist() != mean_projection.tolist()


def test_classification_metrics_include_per_class_stats():
    metrics = _classification_metrics(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 0, 1]),
        [0.1, 0.8, 0.2, 0.9],
    )

    assert metrics["sample_size"] == 4
    assert metrics["false_negatives_positive_class"] == 1
    assert metrics["false_negative_rate_positive_class"] == pytest.approx(0.25)
    assert metrics["class_metrics"]["0"]["precision"] == pytest.approx(0.5)
    assert metrics["class_metrics"]["0"]["recall"] == pytest.approx(0.5)
    assert metrics["class_metrics"]["1"]["f1_score"] == pytest.approx(0.5)


def test_paper_protocol_config_is_locked_to_article_setup():
    protocol = _paper_protocol_config()

    assert protocol["split_mode"] == "Estratificado"
    assert protocol["test_size"] == pytest.approx(0.2)
    assert protocol["random_state"] == 42
    assert protocol["model_family"] == "XGBoost"
    assert protocol["feature_groups"] == {"M1": "Solo flujo", "M2": "Solo embeddings", "M3": "Todo"}
    assert protocol["comparison_tolerance"] == pytest.approx(0.001)


def test_paper_candidate_k_values_clip_to_available_features():
    assert _paper_candidate_k_values(12) == [10]
    assert _paper_candidate_k_values(200) == [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200]
    assert _paper_candidate_k_values(120, k_grid=[10, 70, 150]) == [10, 70]
    assert _paper_candidate_k_values(8, k_grid=[10, 15]) == [8]


def test_paper_normalize_k_grid_enforces_ui_limits():
    assert _paper_normalize_k_grid([200, 10, 50], enforce_limits=True) == [10, 50, 200]
    with pytest.raises(ValueError, match="entre 2 y 5"):
        _paper_normalize_k_grid([10], enforce_limits=True)
    with pytest.raises(ValueError, match="entre 2 y 5"):
        _paper_normalize_k_grid([10, 15, 20, 25, 30, 40], enforce_limits=True)


def test_paper_normalize_cv_folds_enforces_range():
    assert _paper_normalize_cv_folds(5) == 5
    with pytest.raises(ValueError, match="K de folds"):
        _paper_normalize_cv_folds(1)
    with pytest.raises(ValueError, match="K de folds"):
        _paper_normalize_cv_folds(11)


def test_optimize_xgb_classifier_supports_optuna_backend(monkeypatch: pytest.MonkeyPatch):
    class DummyEstimator:
        def __init__(self, params: dict[str, object]):
            self.params = dict(params)
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True
            return self

    monkeypatch.setattr(
        nlp_app,
        "build_model",
        lambda model_name, params, random_state: DummyEstimator(params),
    )

    def fake_cross_validate(model, X, y, scoring, cv, n_jobs, error_score):
        score = 0.5 + float(model.params.get("learning_rate", 0.0))
        if int(model.params.get("max_depth", 0)) >= 5:
            score += 0.05
        return {"test_score": np.array([score, score], dtype=float)}

    monkeypatch.setattr(nlp_app, "cross_validate", fake_cross_validate)

    X_train = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4]})
    y_train = pd.Series([0, 1, 0, 1])

    model, best_params, best_score, search_df, search_meta = nlp_app._optimize_xgb_classifier(
        X_train,
        y_train,
        random_state=42,
        tune_hyperparameters=True,
        tuning_folds=2,
        tuning_profile="Rapida",
        optimization_backend="optuna",
        optuna_trials=2,
    )

    assert isinstance(model, DummyEstimator)
    assert model.fitted is True
    assert search_meta["requested_backend"] == "optuna"
    assert search_meta["backend"] == "optuna"
    assert search_meta["optuna_trials_requested"] == 2
    assert search_meta["optuna_trials_completed"] == 2
    assert best_score is not None
    assert not search_df.empty
    assert set(search_df["optimization_backend"].astype(str)) == {"optuna"}
    assert "learning_rate" in best_params


def test_paper_compare_routes_blocks_on_strict_tolerance():
    frozen_payload = _paper_compare_fixture_payload()
    raw_payload = copy.deepcopy(frozen_payload)

    aligned = _paper_compare_routes(frozen_payload, raw_payload, tolerance=0.001)
    assert aligned["passed"] is True
    assert aligned["status"] == "ok"

    numeric_drift = copy.deepcopy(frozen_payload)
    numeric_drift["model_results"][0]["metrics"]["accuracy"] = 0.903
    numeric_result = _paper_compare_routes(frozen_payload, numeric_drift, tolerance=0.001)
    assert numeric_result["passed"] is False
    assert numeric_result["status"] == "blocked"
    assert not numeric_result["numeric_failures"].empty

    discrete_drift = copy.deepcopy(frozen_payload)
    discrete_drift["model_results"][2]["selected_k"] = 301
    discrete_result = _paper_compare_routes(frozen_payload, discrete_drift, tolerance=0.001)
    assert discrete_result["passed"] is False
    assert not discrete_result["discrete_failures"].empty


def test_paper_checkpoint_manifest_roundtrip_and_compatibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nlp_app, "PAPER_REPLICATION_DIR", tmp_path / "paper_replication")
    execution_context = {
        "protocol_snapshot": {"split_mode": "Estratificado"},
        "input_fingerprints": {
            "protocol_version": nlp_app.PAPER_PROTOCOL_VERSION,
            "frozen_dataset": {"sha256": "abc"},
            "raw_source": {"kind": "session_accidents", "rows": 2},
            "transformer_model": {"kind": "transformer_finetuned", "model_label": "m1"},
        },
        "computed_run_id": "paper_replication_testhash",
    }
    run_dir = nlp_app._paper_run_dir("paper_replication_testhash", checkpoint_root=tmp_path / "paper_replication")
    paths = nlp_app._paper_run_paths(run_dir)
    nlp_app._ensure_paper_run_dirs(paths)
    manifest = nlp_app._paper_initial_manifest(
        run_id="paper_replication_testhash",
        computed_run_id=execution_context["computed_run_id"],
        protocol_snapshot=execution_context["protocol_snapshot"],
        input_fingerprints=execution_context["input_fingerprints"],
    )
    nlp_app._paper_persist_manifest(paths["manifest"], manifest)

    loaded_manifest = nlp_app._paper_load_manifest(paths["manifest"])
    assert loaded_manifest is not None
    assert loaded_manifest["computed_run_id"] == execution_context["computed_run_id"]

    preview = nlp_app._paper_preview_checkpoint_run(
        "paper_replication_testhash",
        execution_context=execution_context,
        checkpoint_root=tmp_path / "paper_replication",
    )
    assert preview["compatible"] is True
    assert preview["can_resume"] is True

    manifest["input_fingerprints"]["raw_source"]["rows"] = 3
    nlp_app._paper_persist_manifest(paths["manifest"], manifest)
    incompatible_preview = nlp_app._paper_preview_checkpoint_run(
        "paper_replication_testhash",
        execution_context=execution_context,
        checkpoint_root=tmp_path / "paper_replication",
    )
    assert incompatible_preview["compatible"] is False
    assert "fingerprints" in incompatible_preview["incompatibility_reason"]


def test_persist_paper_replication_payload_is_idempotent_when_registry_sync_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = tmp_path / "paper_replication" / "paper_replication_test"
    paths = nlp_app._paper_run_paths(run_dir)
    nlp_app._ensure_paper_run_dirs(paths)
    manifest = nlp_app._paper_initial_manifest(
        run_id="paper_replication_test",
        computed_run_id="paper_replication_test",
        protocol_snapshot={"split_mode": "Estratificado"},
        input_fingerprints={"protocol_version": nlp_app.PAPER_PROTOCOL_VERSION},
    )
    manifest["registry_sync"] = {"completed": True, "completed_at": "2026-03-30T12:00:00"}
    manifest["status"] = "completed"
    manifest["result_status"] = "ok"
    nlp_app._paper_persist_manifest(paths["manifest"], manifest)

    calls: list[str] = []
    monkeypatch.setattr(nlp_app, "_persist_artifact", lambda *args, **kwargs: calls.append("artifact"))
    monkeypatch.setattr(nlp_app, "_record_model_result", lambda *args, **kwargs: calls.append("model"))
    monkeypatch.setattr(nlp_app, "_log_action", lambda *args, **kwargs: calls.append("log"))

    nlp_app._persist_paper_replication_payload(
        {
            "run_id": "paper_replication_test",
            "checkpoint_manifest_path": str(paths["manifest"]),
            "frozen": {},
            "raw": {},
            "compare": {},
            "candidate_paths": {},
            "promoted_paths": {},
            "latex_promoted": False,
        }
    )
    assert calls == []


def test_paper_execution_context_changes_with_route_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nlp_app, "_paper_file_fingerprint", lambda path: {"sha256": "frozen-sha"})
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_raw_source",
        lambda accidents_df: {
            "source_df": None,
            "source_kind": "session_accidents",
            "source_metadata": {"rows": 2},
            "source_fingerprint": {"kind": "session_accidents", "rows": 2},
        },
    )
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_transformer_model",
        lambda: pd.Series(
            {
                "model_label": "bert-paper",
                "output_dir_resolved": "/tmp/model",
                "created_at": "2026-03-30T10:00:00",
            }
        ),
    )

    frozen_only = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": True, "run_raw": False},
    )
    raw_only = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": False, "run_raw": True},
    )

    assert frozen_only["computed_run_id"] != raw_only["computed_run_id"]
    assert frozen_only["protocol_snapshot"]["route_options"] == {"run_frozen": True, "run_raw": False, "run_update_embeddings": False}
    assert raw_only["protocol_snapshot"]["route_options"] == {"run_frozen": False, "run_raw": True, "run_update_embeddings": False}
    assert frozen_only["input_fingerprints"]["raw_source"] == {"skipped": True}
    assert raw_only["input_fingerprints"]["frozen_dataset"] == {"skipped": True}


def test_paper_protocol_config_supports_optuna_backend():
    default_protocol = _paper_protocol_config()
    optuna_protocol = _paper_protocol_config(
        k_grid=[10, 50, 200],
        cv_folds=4,
        optimization_backend="optuna",
        optuna_trials=17,
    )

    assert default_protocol["optimization_backend"] == "gridsearch"
    assert default_protocol["optuna_trials"] == 0
    assert default_protocol["k_grid"] == nlp_app.PAPER_K_GRID
    assert default_protocol["cv_folds"] == nlp_app.PAPER_CV_FOLDS_DEFAULT
    assert optuna_protocol["optimization_backend"] == "optuna"
    assert optuna_protocol["optuna_trials"] == 17
    assert optuna_protocol["k_grid"] == [10, 50, 200]
    assert optuna_protocol["cv_folds"] == 4


def test_paper_execution_context_changes_with_optimization_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nlp_app, "_paper_file_fingerprint", lambda path: {"sha256": "frozen-sha"})
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_raw_source",
        lambda accidents_df: {
            "source_df": None,
            "source_kind": "session_accidents",
            "source_metadata": {"rows": 2},
            "source_fingerprint": {"kind": "session_accidents", "rows": 2},
        },
    )
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_transformer_model",
        lambda: pd.Series(
            {
                "model_label": "bert-paper",
                "output_dir_resolved": "/tmp/model",
                "created_at": "2026-03-30T10:00:00",
            }
        ),
    )

    grid_context = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": True, "run_raw": True},
        optimization_backend="gridsearch",
    )
    optuna_context = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": True, "run_raw": True},
        optimization_backend="optuna",
        optuna_trials=11,
    )

    assert grid_context["computed_run_id"] != optuna_context["computed_run_id"]
    assert grid_context["protocol_snapshot"]["optimization_backend"] == "gridsearch"
    assert grid_context["protocol_snapshot"]["optuna_trials"] == 0
    assert optuna_context["protocol_snapshot"]["optimization_backend"] == "optuna"
    assert optuna_context["protocol_snapshot"]["optuna_trials"] == 11


def test_paper_execution_context_changes_with_k_grid_selection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nlp_app, "_paper_file_fingerprint", lambda path: {"sha256": "frozen-sha"})
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_raw_source",
        lambda accidents_df: {
            "source_df": None,
            "source_kind": "session_accidents",
            "source_metadata": {"rows": 2},
            "source_fingerprint": {"kind": "session_accidents", "rows": 2},
        },
    )
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_transformer_model",
        lambda: pd.Series(
            {
                "model_label": "bert-paper",
                "output_dir_resolved": "/tmp/model",
                "created_at": "2026-03-30T10:00:00",
            }
        ),
    )

    small_grid = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": True, "run_raw": True},
        k_grid=[10, 50],
    )
    wide_grid = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": True, "run_raw": True},
        k_grid=[10, 50, 100, 200],
    )

    assert small_grid["computed_run_id"] != wide_grid["computed_run_id"]
    assert small_grid["protocol_snapshot"]["k_grid"] == [10, 50]
    assert wide_grid["protocol_snapshot"]["k_grid"] == [10, 50, 100, 200]


def test_paper_execution_context_changes_with_cv_folds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nlp_app, "_paper_file_fingerprint", lambda path: {"sha256": "frozen-sha"})
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_raw_source",
        lambda accidents_df: {
            "source_df": None,
            "source_kind": "session_accidents",
            "source_metadata": {"rows": 2},
            "source_fingerprint": {"kind": "session_accidents", "rows": 2},
        },
    )
    monkeypatch.setattr(
        nlp_app,
        "_paper_resolve_transformer_model",
        lambda: pd.Series(
            {
                "model_label": "bert-paper",
                "output_dir_resolved": "/tmp/model",
                "created_at": "2026-03-30T10:00:00",
            }
        ),
    )

    folds_3 = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": True, "run_raw": True},
        cv_folds=3,
    )
    folds_5 = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": True, "run_raw": True},
        cv_folds=5,
    )

    assert folds_3["computed_run_id"] != folds_5["computed_run_id"]
    assert folds_3["protocol_snapshot"]["cv_folds"] == 3
    assert folds_5["protocol_snapshot"]["cv_folds"] == 5


def test_paper_execution_context_uses_selected_raw_overrides(monkeypatch: pytest.MonkeyPatch):
    selected_features_df = pd.DataFrame(
        {
            "accident_id": ["a1"],
            "severity_target": [1],
            "text_bert": ["evento de prueba"],
            "flow_feature": [2.5],
        }
    )
    selected_granular_df = pd.DataFrame({"accident_id": ["a1"], "metric": ["flow"]})
    feature_row = pd.Series(
        {
            "artifact_id": "feat-1",
            "run_id": "run-feat",
            "created_at": "2026-03-30T12:00:00",
            "db_path": "/tmp/features.duckdb",
            "table_name": "severity_features",
            "row_count": 1,
        }
    )
    transformer_row = pd.Series(
        {
            "model_label": "manual-transformer",
            "output_dir_resolved": "/tmp/model",
            "created_at": "2026-03-30T13:00:00",
        }
    )
    monkeypatch.setattr(
        nlp_app,
        "_load_feature_bundle_from_catalog_row",
        lambda row: (
            selected_features_df.copy(),
            selected_granular_df.copy(),
            {"artifact_id": "feat-1", "run_id": "run-feat"},
        ),
    )
    monkeypatch.setattr(nlp_app, "_paper_file_fingerprint", lambda path: {"sha256": "frozen-sha"})

    context = nlp_app._paper_build_execution_context(
        None,
        route_options={"run_frozen": False, "run_raw": True},
        raw_features_artifact_row=feature_row,
        transformer_model_row_override=transformer_row,
    )

    assert context["raw_source_kind"] == "precomputed_features_artifact"
    assert isinstance(context["raw_features_df"], pd.DataFrame)
    assert context["raw_features_df"].equals(selected_features_df)
    assert context["input_fingerprints"]["raw_features"]["artifact_id"] == "feat-1"
    assert context["transformer_model_row"]["model_label"] == "manual-transformer"


def test_run_paper_replication_loads_completed_checkpoint_without_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(nlp_app, "PAPER_REPLICATION_DIR", tmp_path / "paper_replication")
    run_dir = nlp_app._paper_run_dir("paper_replication_completed", checkpoint_root=tmp_path / "paper_replication")
    paths = nlp_app._paper_run_paths(run_dir)
    nlp_app._ensure_paper_run_dirs(paths)
    execution_context = {
        "protocol_snapshot": {"split_mode": "Estratificado"},
        "input_fingerprints": {
            "protocol_version": nlp_app.PAPER_PROTOCOL_VERSION,
            "frozen_dataset": {"sha256": "abc"},
            "raw_source": {"kind": "session_accidents", "rows": 2},
            "transformer_model": {"kind": "transformer_finetuned", "model_label": "m1"},
        },
        "computed_run_id": "paper_replication_completed",
        "raw_source_df": None,
        "transformer_model_row": None,
    }
    manifest = nlp_app._paper_initial_manifest(
        run_id="paper_replication_completed",
        computed_run_id=execution_context["computed_run_id"],
        protocol_snapshot=execution_context["protocol_snapshot"],
        input_fingerprints=execution_context["input_fingerprints"],
    )
    manifest["status"] = "completed"
    manifest["result_status"] = "blocked"
    nlp_app._paper_persist_manifest(paths["manifest"], manifest)

    frozen_payload = _paper_route_stub_payload("frozen")
    raw_payload = {
        **_paper_route_stub_payload("raw"),
        "status": "blocked",
        "status_message": "raw blocked",
    }
    compare_payload = {
        "status": "blocked",
        "reason": "Diferencias detectadas.",
        "passed": False,
        "max_numeric_diff": 0.01,
        "tolerance": 0.001,
        "diff_df": pd.DataFrame(),
    }
    nlp_app._paper_persist_route_payload(frozen_payload, nlp_app._paper_route_paths(paths, "frozen"))
    nlp_app._paper_persist_route_payload(raw_payload, nlp_app._paper_route_paths(paths, "raw"))
    nlp_app._paper_persist_compare_payload(compare_payload, nlp_app._paper_compare_paths(paths))
    nlp_app._paper_persist_export_payload(
        {
            "run_id": "paper_replication_completed",
            "run_dir": str(run_dir),
            "frozen": frozen_payload,
            "raw": raw_payload,
            "compare": compare_payload,
            "candidate_paths": {"metrics.png": str(run_dir / "export" / "latex_candidate" / "metrics.png")},
            "promoted_paths": {},
            "latex_promoted": False,
            "result_status": "blocked",
        },
        nlp_app._paper_export_paths(paths),
    )

    monkeypatch.setattr(nlp_app, "_paper_build_execution_context", lambda accidents_df, **kwargs: execution_context)
    monkeypatch.setattr(
        nlp_app,
        "_paper_run_route",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no deberia recomputar rutas")),
    )
    monkeypatch.setattr(
        nlp_app,
        "_paper_build_raw_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no deberia recomputar raw")),
    )

    payload = nlp_app.run_paper_replication(accidents_df=None)
    assert payload["loaded_from_checkpoint"] is True
    assert payload["run_id"] == "paper_replication_completed"
    assert payload["result_status"] == "blocked"
    assert payload["latex_promoted"] is False


def test_train_rf_xgb_holdout_threads_optimization_backend(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame(
        {
            "severity_target": [0, 1, 0, 1],
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1, 2, 3, 4],
        }
    )
    X_train = df[["f1", "f2"]].iloc[:2].reset_index(drop=True)
    X_test = df[["f1", "f2"]].iloc[2:].reset_index(drop=True)
    y_train = pd.Series([0, 1])
    y_test = pd.Series([0, 1])
    captured: list[tuple[str | None, int | None]] = []

    class DummyModel:
        def predict(self, X):
            return np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

    monkeypatch.setattr(
        nlp_app,
        "_prepare_holdout_split",
        lambda *args, **kwargs: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
            {"split_mode": "Estratificado"},
        ),
    )
    monkeypatch.setattr(nlp_app, "_fit_imputer", lambda X_tr, X_te: (X_tr.copy(), X_te.copy(), None))
    monkeypatch.setattr(
        nlp_app,
        "_maybe_balance_training_data",
        lambda X_tr, y_tr, random_state: (X_tr.copy(), y_tr.copy(), {"balanced": False}),
    )
    monkeypatch.setattr(
        nlp_app,
        "_rf_rank_features",
        lambda X_tr, y_tr, random_state: pd.DataFrame({"variable": ["f1", "f2"], "importance": [0.9, 0.8]}),
    )

    def fake_optimize(X_train_arg, y_train_arg, **kwargs):
        captured.append((kwargs.get("optimization_backend"), kwargs.get("optuna_trials")))
        return DummyModel(), {"max_depth": 5}, 0.91, pd.DataFrame({"trial": [1]}), {
            "requested_backend": "optuna",
            "backend": "optuna",
            "optuna_trials_requested": 7,
            "optuna_trials_effective": 7,
        }

    monkeypatch.setattr(nlp_app, "_optimize_xgb_classifier", fake_optimize)

    payload = nlp_app.train_rf_xgb_holdout(
        df,
        feature_group="Solo flujo",
        test_size=0.2,
        random_state=42,
        split_mode="Estratificado",
        top_k=2,
        tune_hyperparameters=True,
        tuning_folds=3,
        tuning_profile="Rapida",
        optimization_backend="optuna",
        optuna_trials=7,
    )

    assert captured == [("optuna", 7)]
    assert payload["xgb_optimization"]["backend"] == "optuna"
    assert payload["results"][0]["params"]["optimization_backend"] == "optuna"
    assert payload["results"][0]["params"]["optuna_trials_requested"] == 7


def test_paper_build_raw_dataset_uses_precomputed_features_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = tmp_path / "paper_replication" / "paper_precomputed_raw"
    paths = nlp_app._paper_run_paths(run_dir)
    nlp_app._ensure_paper_run_dirs(paths)
    manifest = nlp_app._paper_initial_manifest(
        run_id="paper_precomputed_raw",
        computed_run_id="paper_precomputed_raw",
        protocol_snapshot={"split_mode": "Estratificado"},
        input_fingerprints={"protocol_version": nlp_app.PAPER_PROTOCOL_VERSION},
    )
    nlp_app._paper_persist_manifest(paths["manifest"], manifest)

    features_df = pd.DataFrame(
        {
            "accident_id": ["a1", "a2"],
            "severity_target": [0, 1],
            "text_bert": ["uno", "dos"],
            "flow_feature": [1.0, 2.0],
        }
    )
    granular_df = pd.DataFrame({"accident_id": ["a1", "a2"], "metric": ["flow", "speed"]})
    execution_context = {
        "raw_source_df": None,
        "raw_features_df": features_df.copy(),
        "raw_granular_df": granular_df.copy(),
        "raw_feature_artifact": {"artifact_id": "feat-1"},
        "transformer_model_row": pd.Series(
            {
                "output_dir_resolved": "/tmp/manual-model",
                "model_label": "manual-transformer",
            }
        ),
    }

    monkeypatch.setattr(
        nlp_app,
        "build_severity_feature_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no deberia reconstruir features")),
    )
    monkeypatch.setattr(
        nlp_app,
        "generate_text_embeddings",
        lambda df, **kwargs: (
            df.assign(emb_000=[0.1, 0.2], emb_001=[0.3, 0.4]),
            ["emb_000", "emb_001"],
            {"model_name": "manual-transformer"},
        ),
    )
    monkeypatch.setattr(
        nlp_app,
        "run_embedding_rf_analysis",
        lambda df, embed_cols: pd.DataFrame(
            {"variable": ["emb_001", "emb_000"], "importance": [0.9, 0.8]}
        ),
    )

    payload = nlp_app._paper_build_raw_dataset(
        accidents_df=None,
        paths=paths,
        manifest=manifest,
        execution_context=execution_context,
    )

    assert payload["features_df"].equals(features_df)
    assert payload["granular_df"].equals(granular_df)
    assert payload["selected_embedding_cols"] == ["emb_001", "emb_000"]
    assert "emb_000" in payload["dataset_df"].columns
    assert "emb_001" in payload["dataset_df"].columns


def test_run_paper_replication_can_skip_raw_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frozen_path = tmp_path / "resultado.pkl"
    pd.DataFrame(
        {
            "accident_id": ["a1", "a2"],
            "severity_target": [0, 1],
            "flow_feature": [1.0, 2.0],
            "emb_feature": [0.1, 0.2],
        }
    ).to_pickle(frozen_path)

    monkeypatch.setattr(nlp_app, "PAPER_FROZEN_DATASET_PATH", frozen_path)
    monkeypatch.setattr(nlp_app, "PAPER_REPLICATION_DIR", tmp_path / "paper_replication")
    monkeypatch.setattr(nlp_app, "PAPER_LATEX_IMAGES_DIR", tmp_path / "latex_images")
    monkeypatch.setattr(nlp_app, "PAPER_LATEX_GENERATED_DIR", tmp_path / "latex_generated")

    route_calls: list[str] = []

    def fake_run_route(*, route_name: str, dataset_df: pd.DataFrame, **kwargs):
        route_calls.append(route_name)
        return _paper_route_stub_payload(route_name)

    monkeypatch.setattr(nlp_app, "_paper_run_route", fake_run_route)
    monkeypatch.setattr(
        nlp_app,
        "_paper_build_raw_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw no deberia ejecutarse")),
    )
    monkeypatch.setattr(
        nlp_app,
        "_paper_stage_latex_candidates",
        lambda frozen_payload, *, output_dir: {
            "metrics.png": (output_dir / "metrics.png"),
        },
    )

    def fake_stage_candidates(frozen_payload: dict, *, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        staged = {}
        for name in ["metrics.png", "gridsearch_k.tex", "metricas_modelos.tex"]:
            path = output_dir / name
            path.write_text("stub", encoding="utf-8")
            staged[name] = path
        return staged

    monkeypatch.setattr(nlp_app, "_paper_stage_latex_candidates", fake_stage_candidates)
    monkeypatch.setattr(
        nlp_app,
        "_paper_promote_latex_assets",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no deberia promover assets")),
    )

    payload = nlp_app.run_paper_replication(
        accidents_df=None,
        run_id="paper_skip_raw",
        run_frozen=True,
        run_raw=False,
    )

    assert route_calls == ["frozen"]
    assert payload["route_options"] == {"run_frozen": True, "run_raw": False, "run_update_embeddings": False}
    assert payload["frozen"]["status"] == "ok"
    assert payload["raw"]["status"] == "skipped"
    assert payload["compare"]["status"] == "skipped"
    assert payload["latex_promoted"] is False
    assert payload["candidate_paths"]


def test_run_paper_replication_can_skip_frozen_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nlp_app, "PAPER_REPLICATION_DIR", tmp_path / "paper_replication")
    monkeypatch.setattr(nlp_app, "PAPER_LATEX_IMAGES_DIR", tmp_path / "latex_images")
    monkeypatch.setattr(nlp_app, "PAPER_LATEX_GENERATED_DIR", tmp_path / "latex_generated")

    route_calls: list[str] = []

    def fake_run_route(*, route_name: str, dataset_df: pd.DataFrame, **kwargs):
        route_calls.append(route_name)
        return _paper_route_stub_payload(route_name)

    def fake_build_raw_dataset(*, accidents_df=None, **kwargs):
        return {
            "dataset_df": pd.DataFrame(
                {
                    "accident_id": ["a1", "a2"],
                    "severity_target": [0, 1],
                    "flow_feature": [1.0, 2.0],
                    "emb_feature": [0.1, 0.2],
                }
            ),
            "embedding_meta": {},
            "selected_embedding_cols": ["emb_feature"],
        }

    monkeypatch.setattr(nlp_app, "_paper_run_route", fake_run_route)
    monkeypatch.setattr(nlp_app, "_paper_build_raw_dataset", fake_build_raw_dataset)
    monkeypatch.setattr(
        nlp_app,
        "_paper_stage_latex_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no deberia generar assets frozen")),
    )
    monkeypatch.setattr(
        nlp_app,
        "_paper_promote_latex_assets",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no deberia promover assets")),
    )

    payload = nlp_app.run_paper_replication(
        accidents_df=None,
        run_id="paper_skip_frozen",
        run_frozen=False,
        run_raw=True,
    )

    assert route_calls == ["raw"]
    assert payload["route_options"] == {"run_frozen": False, "run_raw": True, "run_update_embeddings": False}
    assert payload["frozen"]["status"] == "skipped"
    assert payload["raw"]["status"] == "ok"
    assert payload["compare"]["status"] == "skipped"
    assert payload["candidate_paths"] == {}
    assert payload["latex_promoted"] is False


def test_train_model_comparison_holdout_includes_xgb_optimization_protocol(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame(
        {
            "accident_id": ["a1", "a2", "a3", "a4"],
            "severity_target": [0, 1, 0, 1],
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1, 2, 3, 4],
        }
    )
    X_train = df[["f1", "f2"]].iloc[:2].reset_index(drop=True)
    X_test = df[["f1", "f2"]].iloc[2:].reset_index(drop=True)
    y_train = pd.Series([0, 1])
    y_test = pd.Series([0, 1])
    test_ids = pd.Series(["a3", "a4"])
    captured: list[tuple[str | None, int | None]] = []

    monkeypatch.setattr(nlp_app, "_resolve_feature_group", lambda df_arg, feature_group: ["f1", "f2"])
    monkeypatch.setattr(
        nlp_app,
        "_prepare_holdout_split_with_ids",
        lambda *args, **kwargs: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
            test_ids.copy(),
            {"split_mode": "Estratificado", "comparison_id_field": "accident_id", "test_rows": 2},
        ),
    )

    def fake_rf_xgb(*args, **kwargs):
        captured.append((kwargs.get("optimization_backend"), kwargs.get("optuna_trials")))
        return {
            "model_name": "RF + XGBoost",
            "feature_strategy": "RF top-2",
            "selected_cols": ["f1", "f2"],
            "ranking_df": pd.DataFrame({"variable": ["f1", "f2"], "importance": [0.9, 0.8]}),
            "balancing_meta": {"balanced": False},
            "search_df": pd.DataFrame({"optimization_backend": ["optuna"]}),
            "params": {"optimization_backend": "optuna", "optuna_trials_requested": 9},
            "metrics": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.95, "false_negatives_global": 0},
            "predictions": np.array([0, 1]),
            "scores": np.array([0.1, 0.9]),
            "optimization": {"backend": "optuna", "optuna_trials_requested": 9},
        }

    def fake_other(model_name: str):
        return {
            "model_name": model_name,
            "feature_strategy": model_name,
            "selected_cols": ["f1"],
            "ranking_df": pd.DataFrame({"variable": ["f1"], "importance": [0.5]}),
            "balancing_meta": {"balanced": False},
            "search_df": pd.DataFrame(),
            "params": {},
            "metrics": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8, "roc_auc": 0.85, "false_negatives_global": 0},
            "predictions": np.array([0, 1]),
            "scores": np.array([0.2, 0.8]),
        }

    monkeypatch.setattr(nlp_app, "_train_rf_xgb_shared_holdout", fake_rf_xgb)
    monkeypatch.setattr(nlp_app, "_train_elastic_net_shared_holdout", lambda *args, **kwargs: fake_other("Elastic Net"))
    monkeypatch.setattr(nlp_app, "_train_svm_rfe_shared_holdout", lambda *args, **kwargs: fake_other("SVM + RFE"))

    payload = nlp_app.train_model_comparison_holdout(
        df,
        feature_group="Solo flujo",
        test_size=0.2,
        random_state=42,
        split_mode="Estratificado",
        max_features_per_model=2,
        xgb_tuning_profile="Rapida",
        xgb_optimization_backend="optuna",
        xgb_optuna_trials=9,
        tuning_folds=3,
    )

    assert captured == [("optuna", 9)]
    assert payload["protocol"]["xgb_optimization_backend"] == "optuna"
    assert payload["protocol"]["xgb_optuna_trials"] == 9


def test_paper_build_model_result_threads_optimization_config(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame(
        {
            "accident_id": [f"a{i}" for i in range(6)],
            "severity_target": [0, 1, 0, 1, 0, 1],
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "f2": [1, 2, 3, 4, 5, 6],
            "f3": [6, 5, 4, 3, 2, 1],
        }
    )
    X_train = df[["f1", "f2", "f3"]].iloc[:4].reset_index(drop=True)
    X_test = df[["f1", "f2", "f3"]].iloc[4:].reset_index(drop=True)
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1])
    test_ids = pd.Series(["a4", "a5"])

    nested_calls: list[tuple[str | None, int | None, int | None, int | None]] = []
    final_calls: list[tuple[str | None, int | None, int | None]] = []

    monkeypatch.setattr(nlp_app, "_resolve_feature_group", lambda df_arg, feature_group: ["f1", "f2", "f3"])
    monkeypatch.setattr(
        nlp_app,
        "_prepare_holdout_split_with_ids",
        lambda *args, **kwargs: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
            test_ids.copy(),
            {"train_rows": 4, "test_rows": 2},
        ),
    )
    monkeypatch.setattr(nlp_app, "_fit_imputer", lambda X_tr, X_te: (X_tr.copy(), X_te.copy(), None))
    monkeypatch.setattr(
        nlp_app,
        "_maybe_balance_training_data",
        lambda X_tr, y_tr, random_state: (X_tr.copy(), y_tr.copy(), {"balanced": False}),
    )
    monkeypatch.setattr(
        nlp_app,
        "_rf_rank_features",
        lambda X_tr, y_tr, random_state: pd.DataFrame({"variable": ["f1", "f2", "f3"], "importance": [0.9, 0.8, 0.7]}),
    )
    monkeypatch.setattr(nlp_app, "_paper_candidate_k_values", lambda total_features: [1, 2])
    monkeypatch.setattr(nlp_app, "_paper_select_k_from_search", lambda search_df, epsilon=0.001: 2)

    def fake_nested(X, y, **kwargs):
        nested_calls.append(
            (
                kwargs.get("optimization_backend"),
                kwargs.get("optuna_trials"),
                kwargs.get("inner_folds"),
                kwargs.get("outer_folds"),
            )
        )
        size = len(X.columns)
        return {
            "summary": {
                "accuracy": 0.8 + size * 0.01,
                "precision": 0.8,
                "recall": 0.8,
                "f1_score": 0.8 + size * 0.01,
                "roc_auc": 0.9,
                "false_negatives_pct": 5.0,
                "validation_score": 0.8 + size * 0.01,
                "training_time_sec": 0.1,
                "optimization_backend": "optuna",
                "requested_optimization_backend": "optuna",
            }
        }

    def fake_final(*args, **kwargs):
        final_calls.append((kwargs.get("optimization_backend"), kwargs.get("optuna_trials"), kwargs.get("inner_folds")))
        return {
            "metrics": {
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1_score": 0.9,
                "roc_auc": 0.95,
                "false_negatives_positive_class": 0,
                "class_metrics": {},
            },
            "best_params": {"max_depth": 3},
            "best_cv_score": 0.9,
            "search_df": pd.DataFrame(),
            "predictions_df": pd.DataFrame({"severity_target": [0, 1], "prediction": [0, 1]}),
            "balancing_meta": {"balanced": False},
            "optimization": {"requested_backend": "optuna", "backend": "optuna", "optuna_trials_requested": 9},
        }

    monkeypatch.setattr(nlp_app, "_paper_nested_xgb_validation", fake_nested)
    monkeypatch.setattr(nlp_app, "_paper_fit_final_xgb_model", fake_final)

    result = nlp_app._paper_build_model_result(
        df,
        model_code="M1",
        feature_group="Solo flujo",
        cv_folds=3,
        random_state=42,
        optimization_backend="optuna",
        optuna_trials=9,
        route_name="frozen",
    )

    assert nested_calls == [("optuna", 9, 3, 3), ("optuna", 9, 3, 3)]
    assert final_calls == [("optuna", 9, 3)]
    assert result["optimization"]["backend"] == "optuna"
    assert result["optimization"]["optuna_trials_requested"] == 9


def test_paper_build_model_result_resumes_from_next_missing_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = tmp_path / "paper_replication" / "paper_replication_resume"
    paths = nlp_app._paper_run_paths(run_dir)
    nlp_app._ensure_paper_run_dirs(paths)
    manifest = nlp_app._paper_initial_manifest(
        run_id="paper_replication_resume",
        computed_run_id="paper_replication_resume",
        protocol_snapshot={"split_mode": "Estratificado"},
        input_fingerprints={"protocol_version": nlp_app.PAPER_PROTOCOL_VERSION},
    )
    nlp_app._paper_persist_manifest(paths["manifest"], manifest)

    df = pd.DataFrame(
        {
            "accident_id": [f"a{i}" for i in range(6)],
            "severity_target": [0, 1, 0, 1, 0, 1],
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "f2": [1, 2, 3, 4, 5, 6],
            "f3": [6, 5, 4, 3, 2, 1],
        }
    )
    X_train = df[["f1", "f2", "f3"]].iloc[:4].reset_index(drop=True)
    X_test = df[["f1", "f2", "f3"]].iloc[4:].reset_index(drop=True)
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1])
    test_ids = pd.Series(["a4", "a5"])

    monkeypatch.setattr(nlp_app, "_resolve_feature_group", lambda df_arg, feature_group: ["f1", "f2", "f3"])
    monkeypatch.setattr(
        nlp_app,
        "_prepare_holdout_split_with_ids",
        lambda *args, **kwargs: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
            test_ids.copy(),
            {"train_rows": 4, "test_rows": 2},
        ),
    )
    monkeypatch.setattr(nlp_app, "_fit_imputer", lambda X_tr, X_te: (X_tr.copy(), X_te.copy(), None))
    monkeypatch.setattr(
        nlp_app,
        "_maybe_balance_training_data",
        lambda X_tr, y_tr, random_state: (X_tr.copy(), y_tr.copy(), {"balanced": False}),
    )
    monkeypatch.setattr(
        nlp_app,
        "_rf_rank_features",
        lambda X_tr, y_tr, random_state: pd.DataFrame({"variable": ["f1", "f2", "f3"], "importance": [0.9, 0.8, 0.7]}),
    )
    monkeypatch.setattr(nlp_app, "_paper_candidate_k_values", lambda total_features: [1, 2])

    nested_calls: list[int] = []

    def fake_nested(X, y, **kwargs):
        nested_calls.append(len(X.columns))
        if nested_calls == [1, 2]:
            raise RuntimeError("boom on second k")
        size = len(X.columns)
        return {
            "summary": {
                "accuracy": 0.8 + size * 0.01,
                "precision": 0.8,
                "recall": 0.8,
                "f1_score": 0.8 + size * 0.01,
                "roc_auc": 0.9,
                "false_negatives_pct": 5.0,
                "validation_score": 0.8 + size * 0.01,
                "training_time_sec": 0.1,
            }
        }

    monkeypatch.setattr(nlp_app, "_paper_nested_xgb_validation", fake_nested)
    monkeypatch.setattr(
        nlp_app,
        "_paper_fit_final_xgb_model",
        lambda *args, **kwargs: {
            "metrics": {
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1_score": 0.9,
                "roc_auc": 0.95,
                "false_negatives_positive_class": 0,
                "class_metrics": {},
            },
            "best_params": {"max_depth": 3},
            "best_cv_score": 0.9,
            "search_df": pd.DataFrame(),
            "predictions_df": pd.DataFrame({"severity_target": [0, 1], "prediction": [0, 1]}),
            "balancing_meta": {"balanced": False},
        },
    )

    with pytest.raises(RuntimeError, match="boom on second k"):
        nlp_app._paper_build_model_result(
            df,
            model_code="M1",
            feature_group="Solo flujo",
            random_state=42,
            route_name="frozen",
            paths=paths,
            manifest=manifest,
        )

    result = nlp_app._paper_build_model_result(
        df,
        model_code="M1",
        feature_group="Solo flujo",
        random_state=42,
        route_name="frozen",
        paths=paths,
        manifest=manifest,
    )
    assert nested_calls == [1, 2, 2]
    assert result["selected_k"] in {1, 2}
    assert nlp_app._paper_is_step_completed(manifest, "frozen.M1.k.1") is True
    assert nlp_app._paper_is_step_completed(manifest, "frozen.M1.final") is True


def test_paper_run_route_uses_m3_selected_k_for_m1_and_m2(monkeypatch: pytest.MonkeyPatch):
    dataset_df = pd.DataFrame(
        {
            "accident_id": ["a1", "a2"],
            "severity_target": [0, 1],
            "flow_1": [1.0, 2.0],
            "flow_2": [3.0, 4.0],
            "emb_1": [0.1, 0.2],
            "emb_2": [0.3, 0.4],
            "emb_3": [0.5, 0.6],
        }
    )
    monkeypatch.setattr(nlp_app, "_ensure_paper_dataset_columns", lambda df, source_name: df.copy())
    monkeypatch.setattr(
        nlp_app,
        "_paper_dataset_validation_report",
        lambda work, route_name: {"rows": int(len(work)), "is_valid": True},
    )

    build_calls: list[dict[str, object]] = []

    def fake_build_model_result(
        df,
        *,
        model_code,
        feature_group,
        k_grid=None,
        forced_selected_k=None,
        **kwargs,
    ):
        build_calls.append(
            {
                "model_code": str(model_code),
                "feature_group": str(feature_group),
                "k_grid": None if k_grid is None else list(k_grid),
                "forced_selected_k": forced_selected_k,
            }
        )
        selected_k = 2 if str(model_code) == "M3" else int(forced_selected_k)
        return {
            "model_code": str(model_code),
            "model_title": str(model_code),
            "feature_group": str(feature_group),
            "candidate_feature_count": 3,
            "selected_k": selected_k,
            "selected_cols": [],
            "split_meta": {},
            "ranking_df": pd.DataFrame(),
            "k_search_df": pd.DataFrame([{"k": selected_k}]),
            "metrics": {
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1_score": 0.9,
                "roc_auc": 0.95,
                "false_negatives_positive_class": 0,
                "class_metrics": {},
            },
            "best_params": {},
            "best_cv_score": 0.9,
            "search_df": pd.DataFrame(),
            "predictions_df": pd.DataFrame(
                {
                    "accident_id": ["a1", "a2"],
                    "model_code": [str(model_code), str(model_code)],
                    "severity_target": [0, 1],
                    "prediction": [0, 1],
                }
            ),
            "balancing_meta": {},
            "optimization": {"backend": "gridsearch"},
        }

    monkeypatch.setattr(nlp_app, "_paper_build_model_result", fake_build_model_result)

    payload = nlp_app._paper_run_route(route_name="frozen", dataset_df=dataset_df, k_grid=[10, 15, 20])

    assert [call["model_code"] for call in build_calls] == ["M3", "M1", "M2"]
    assert build_calls[0]["k_grid"] == [2]
    assert build_calls[0]["forced_selected_k"] is None
    assert build_calls[1]["forced_selected_k"] == 2
    assert build_calls[2]["forced_selected_k"] == 2
    assert payload["shared_k"] == 2
    assert payload["shared_k_grid"] == [2]
    assert [result["model_code"] for result in payload["model_results"]] == ["M1", "M2", "M3"]
    assert [result["selected_k"] for result in payload["model_results"]] == [2, 2, 2]


def test_run_paper_replication_emits_detailed_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frozen_path = tmp_path / "resultado.pkl"
    pd.DataFrame(
        {
            "accident_id": ["a1", "a2"],
            "severity_target": [0, 1],
            "flow_feature": [1.0, 2.0],
            "emb_feature": [0.1, 0.2],
        }
    ).to_pickle(frozen_path)

    monkeypatch.setattr(nlp_app, "PAPER_FROZEN_DATASET_PATH", frozen_path)
    monkeypatch.setattr(nlp_app, "PAPER_REPLICATION_DIR", tmp_path / "paper_replication")
    monkeypatch.setattr(nlp_app, "PAPER_LATEX_IMAGES_DIR", tmp_path / "latex_images")
    monkeypatch.setattr(nlp_app, "PAPER_LATEX_GENERATED_DIR", tmp_path / "latex_generated")

    def fake_run_route(*, route_name: str, dataset_df: pd.DataFrame, route_metadata=None, progress_callback=None, **kwargs):
        if progress_callback is not None:
            progress_callback(0, f"{route_name}: inicio")
            progress_callback(50, f"{route_name}: mitad")
            progress_callback(100, f"{route_name}: fin")
        return _paper_route_stub_payload(route_name)

    def fake_build_raw_dataset(*, accidents_df=None, progress_callback=None, **kwargs):
        if progress_callback is not None:
            progress_callback(10, "raw build inicio")
            progress_callback(100, "raw build fin")
        return {
            "dataset_df": pd.DataFrame(
                {
                    "accident_id": ["a1", "a2"],
                    "severity_target": [0, 1],
                    "flow_feature": [1.0, 2.0],
                    "emb_feature": [0.1, 0.2],
                }
            ),
            "embedding_meta": {},
            "selected_embedding_cols": ["emb_feature"],
        }

    def fake_compare_routes(*args, **kwargs):
        return {
            "status": "ok",
            "reason": "Rutas coinciden.",
            "passed": True,
            "max_numeric_diff": 0.0,
            "tolerance": 0.001,
            "diff_df": pd.DataFrame(),
        }

    def fake_stage_latex_candidates(frozen_payload: dict, *, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in [
            "accuracy_vs_k.png",
            "f1_score_vs_k.png",
            "false_negatives_pct_vs_k.png",
            "validation_score_vs_k.png",
            "metrics.png",
            "gridsearch_k.tex",
            "metricas_modelos.tex",
        ]:
            path = output_dir / name
            path.write_text("stub", encoding="utf-8")
            paths[name] = path
        return paths

    def fake_promote(candidate_paths: dict):
        return {name: str(path) for name, path in candidate_paths.items()}

    monkeypatch.setattr(nlp_app, "_paper_run_route", fake_run_route)
    monkeypatch.setattr(nlp_app, "_paper_build_raw_dataset", fake_build_raw_dataset)
    monkeypatch.setattr(nlp_app, "_paper_compare_routes", fake_compare_routes)
    monkeypatch.setattr(nlp_app, "_paper_stage_latex_candidates", fake_stage_latex_candidates)
    monkeypatch.setattr(nlp_app, "_paper_promote_latex_assets", fake_promote)

    progress_events: list[tuple[int, str]] = []
    payload = nlp_app.run_paper_replication(
        accidents_df=None,
        run_id="paper_progress_test",
        progress_callback=lambda value, message: progress_events.append((int(value), str(message))),
    )

    assert payload["latex_promoted"] is True
    assert len(progress_events) >= 10
    progress_values = [value for value, _ in progress_events]
    progress_messages = [message for _, message in progress_events]
    assert progress_values[0] == 5
    assert progress_values[-1] == 100
    assert any("Frozen |" in message for message in progress_messages)
    assert any("Raw build |" in message for message in progress_messages)
    assert any("Raw |" in message for message in progress_messages)
    assert any(8 <= value <= 38 for value in progress_values)
    assert any(40 <= value <= 58 for value in progress_values)
    assert any(59 <= value <= 78 for value in progress_values)
