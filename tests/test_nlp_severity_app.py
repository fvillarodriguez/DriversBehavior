from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.nlp_severity_app import (
    _candidate_feature_caps,
    _prepare_holdout_split_with_ids,
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
