#!/usr/bin/env python3
"""
Streamlit app for the "NLP in Severity" pipeline.
"""
from __future__ import annotations

import json
import itertools
import math
import random
import re
import shutil
import traceback
import uuid
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

try:
    from bertopic import BERTopic
except ImportError:  # pragma: no cover
    BERTopic = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

try:
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoModelForMaskedLM = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    DataCollatorForLanguageModeling = None
    EarlyStoppingCallback = None
    Trainer = None
    TrainingArguments = None

from sklearn import __version__ as SKLEARN_VERSION
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover
    SMOTE = None

from src.model_training import build_model
from src.utils import (
    DEFAULT_CATEGORY_LABELS,
    DEFAULT_CATEGORY_REMAP,
    FLOW_TABLE_NAME,
    FlowSampleSelection,
    buscar_columna,
    get_flow_db_summary,
    load_porticos,
    process_accidentes_df,
    read_csv_with_progress,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Datos"
RESULTS_DIR = ROOT_DIR / "Resultados"
MODULE_RESULTS_DIR = RESULTS_DIR / "nlp_in_severity"
REGISTRY_DB = MODULE_RESULTS_DIR / "registry.duckdb"
LOCAL_TRANSFORMER_MODEL_DIR = ROOT_DIR / "NLP" / "bert_base_chile_text_bert_128t"

WEEKDAY_ES = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miercoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sabado",
    "Sunday": "Domingo",
}
MODEL_EXCLUDE_COLUMNS = {
    "severity_target",
    "severidad",
    "accidente_time",
    "interval_start",
    "duracion_accidente",
    "duracion",
    "km",
    "Mes",
}
TEXT_SOURCE_BASE_COLUMNS = [
    "accidente_time",
    "km",
    "eje",
    "calzada",
    "subtipo",
    "descripcion",
]
TEXT_SOURCE_EXCLUDE_COLUMNS = {
    "accident_id",
    "severity_target",
    "severidad",
    "source_files",
}
GRANULAR_METRIC_NAMES = ["flow", "speed_mean", "speed_std", "density"]
GRANULAR_METRIC_LABELS = {
    "flow": "Flow",
    "speed_mean": "Speed mean",
    "speed_std": "Speed std",
    "density": "Density",
}

STATE_DEFAULTS: Dict[str, object] = {
    "nlp_sev_accidents_df": None,
    "nlp_sev_excluded_df": None,
    "nlp_sev_event_files": [],
    "nlp_sev_events_artifact": None,
    "nlp_sev_event_coverage_meta": None,
    "nlp_sev_feature_source": "Calcular nuevas",
    "nlp_sev_features_df": None,
    "nlp_sev_granular_df": None,
    "nlp_sev_feature_ranking_df": None,
    "nlp_sev_features_artifact": None,
    "nlp_sev_granular_artifact": None,
    "nlp_sev_coverage_preview": None,
    "nlp_sev_language_df": None,
    "nlp_sev_language_artifact": None,
    "nlp_sev_embeddings_df": None,
    "nlp_sev_embedding_cols": [],
    "nlp_sev_embeddings_artifact": None,
    "nlp_sev_embedding_meta": None,
    "nlp_sev_embedding_rf_df": None,
    "nlp_sev_selected_embedding_cols": [],
    "nlp_sev_transformer_search_trials_df": None,
    "nlp_sev_transformer_search_confirm_df": None,
    "nlp_sev_transformer_search_summary_df": None,
    "nlp_sev_transformer_search_result": None,
    "nlp_sev_transformer_active_preset": None,
    "nlp_sev_model_results": [],
    "nlp_sev_topic_df": None,
    "nlp_sev_topic_meta": None,
}


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "artifact"


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    return str(value)


def _ts_now() -> str:
    return datetime.now().isoformat()


def _stamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _new_run_id(prefix: str) -> str:
    return f"{_slug(prefix)}_{_stamp_now()}_{uuid.uuid4().hex[:8]}"


def _normalize_portico(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _format_spanish_datetime(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    weekday = WEEKDAY_ES.get(ts.strftime("%A"), "")
    return f"{weekday} {ts:%H:%M}".strip()


def _safe_number(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _list_event_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    paths = []
    for path in DATA_DIR.glob("*.csv"):
        lower = path.name.lower()
        if lower.startswith("eventos") or lower.startswith("accidentes"):
            paths.append(path)
    return sorted(paths)


def _ensure_registry_db() -> None:
    if duckdb is None:  # pragma: no cover
        raise RuntimeError("duckdb no esta instalado.")
    MODULE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(REGISTRY_DB))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS action_log (
                action_id VARCHAR,
                run_id VARCHAR,
                created_at TIMESTAMP,
                stage VARCHAR,
                action VARCHAR,
                payload_json TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id VARCHAR,
                run_id VARCHAR,
                created_at TIMESTAMP,
                stage VARCHAR,
                artifact_name VARCHAR,
                db_path VARCHAR,
                table_name VARCHAR,
                row_count BIGINT,
                metadata_json TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS model_results (
                result_id VARCHAR,
                run_id VARCHAR,
                created_at TIMESTAMP,
                stage VARCHAR,
                model_name VARCHAR,
                feature_group VARCHAR,
                metrics_json TEXT,
                params_json TEXT,
                metadata_json TEXT
            )
            """
        )
    finally:
        con.close()


def _log_action(
    stage: str,
    action: str,
    payload: Dict[str, object],
    *,
    run_id: Optional[str] = None,
) -> str:
    _ensure_registry_db()
    action_id = uuid.uuid4().hex
    con = duckdb.connect(str(REGISTRY_DB))
    try:
        con.execute(
            """
            INSERT INTO action_log
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                action_id,
                run_id,
                _ts_now(),
                stage,
                action,
                json.dumps(payload, ensure_ascii=True, default=_json_default),
            ],
        )
    finally:
        con.close()
    return action_id


def _write_df_to_duckdb(df: pd.DataFrame, path: Path, table_name: str) -> None:
    if duckdb is None:  # pragma: no cover
        raise RuntimeError("duckdb no esta instalado.")
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.register("tmp_df", df)
        table_ref = _quote_identifier(table_name)
        con.execute(f"CREATE OR REPLACE TABLE {table_ref} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    finally:
        con.close()


def _persist_artifact(
    df: pd.DataFrame,
    *,
    stage: str,
    artifact_name: str,
    run_id: Optional[str],
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    _ensure_registry_db()
    artifact_id = uuid.uuid4().hex
    table_name = f"{_slug(artifact_name)}_{_stamp_now()}"
    db_path = MODULE_RESULTS_DIR / f"{table_name}.duckdb"
    _write_df_to_duckdb(df, db_path, table_name)
    payload = {
        "artifact_id": artifact_id,
        "artifact_name": artifact_name,
        "db_path": str(db_path),
        "table_name": table_name,
        "row_count": int(len(df)),
        "columns": int(len(df.columns)),
    }
    if metadata:
        payload["metadata"] = metadata
    con = duckdb.connect(str(REGISTRY_DB))
    try:
        con.execute(
            """
            INSERT INTO artifacts
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                artifact_id,
                run_id,
                _ts_now(),
                stage,
                artifact_name,
                str(db_path),
                table_name,
                int(len(df)),
                json.dumps(metadata or {}, ensure_ascii=True, default=_json_default),
            ],
        )
    finally:
        con.close()
    _log_action(stage, f"persist_{_slug(artifact_name)}", payload, run_id=run_id)
    return payload


def _record_model_result(
    *,
    run_id: str,
    stage: str,
    model_name: str,
    feature_group: str,
    metrics: Dict[str, object],
    params: Dict[str, object],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    _ensure_registry_db()
    result_id = uuid.uuid4().hex
    con = duckdb.connect(str(REGISTRY_DB))
    try:
        con.execute(
            """
            INSERT INTO model_results
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result_id,
                run_id,
                _ts_now(),
                stage,
                model_name,
                feature_group,
                json.dumps(metrics, ensure_ascii=True, default=_json_default),
                json.dumps(params, ensure_ascii=True, default=_json_default),
                json.dumps(metadata or {}, ensure_ascii=True, default=_json_default),
            ],
        )
    finally:
        con.close()


def _load_model_results() -> pd.DataFrame:
    if duckdb is None or not REGISTRY_DB.exists():
        return pd.DataFrame()
    con = duckdb.connect(str(REGISTRY_DB), read_only=True)
    try:
        df = con.execute(
            """
            SELECT created_at, stage, model_name, feature_group, metrics_json, params_json, metadata_json, run_id
            FROM model_results
            ORDER BY created_at DESC
            """
        ).df()
    finally:
        con.close()
    if df.empty:
        return df
    records: List[Dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        metrics = json.loads(row.pop("metrics_json") or "{}")
        params = json.loads(row.pop("params_json") or "{}")
        metadata = json.loads(row.pop("metadata_json") or "{}")
        flat = dict(row)
        for key, value in metrics.items():
            flat[key] = value
        flat["params"] = params
        flat["metadata"] = metadata
        records.append(flat)
    return pd.DataFrame(records)


def _list_transformer_search_presets() -> pd.DataFrame:
    results_df = _load_model_results()
    if results_df.empty:
        return results_df
    mask = (
        results_df["stage"].astype(str).eq("language_modeling")
        & results_df["model_name"].astype(str).str.startswith("Transformers Search")
    )
    presets_df = results_df.loc[mask].copy()
    if presets_df.empty:
        return presets_df

    def _format_preset_label(row: pd.Series) -> str:
        metadata = row.get("metadata") or {}
        search_summary = metadata.get("search_summary") or {}
        base_model = metadata.get("base_model") or search_summary.get("best_model_name") or "modelo?"
        text_col = metadata.get("text_col") or "text?"
        mode = metadata.get("mode") or "mode?"
        objective = metadata.get("objective_metric") or "objective?"
        metric_value = _resolve_transformer_objective(
            {key: row.get(key) for key in row.index},
            objective_metric=str(objective),
        )
        metric_text = "-" if pd.isna(metric_value) else f"{float(metric_value):.4f}"
        return (
            f"{row.get('created_at')} | {mode} | {text_col} | {base_model} | "
            f"{objective}={metric_text}"
        )

    presets_df["preset_label"] = presets_df.apply(_format_preset_label, axis=1)
    return presets_df


def _transformer_preset_from_model_result_row(row: pd.Series) -> Dict[str, object]:
    metadata = row.get("metadata") or {}
    search_summary = metadata.get("search_summary") or {}
    params = row.get("params") or {}
    return {
        "run_id": row.get("run_id"),
        "created_at": row.get("created_at"),
        "source_model_name": row.get("model_name"),
        "text_col": metadata.get("text_col"),
        "mode": metadata.get("mode"),
        "base_model": metadata.get("base_model") or search_summary.get("best_model_name"),
        "objective_metric": metadata.get("objective_metric"),
        "output_dir": metadata.get("best_output_dir") or search_summary.get("best_model_output_dir"),
        "params": params,
        "metrics": {key: row.get(key) for key in row.index if isinstance(key, str)},
        "metadata": metadata,
        "label": row.get("preset_label") or str(row.get("model_name") or "preset"),
    }


def _list_transformer_finetuned_models() -> pd.DataFrame:
    def _is_reusable_transformer_dir(path: object) -> bool:
        model_dir = Path(str(path))
        if not model_dir.exists() or not model_dir.is_dir():
            return False
        required_files = [
            model_dir / "config.json",
            model_dir / "training_summary.json",
        ]
        has_weights = any(
            (model_dir / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        has_tokenizer = any(
            (model_dir / filename).exists()
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "spiece.model",
                "sentencepiece.bpe.model",
            )
        )
        return all(path.exists() for path in required_files) and has_weights and has_tokenizer

    results_df = _load_model_results()
    if results_df.empty:
        return results_df
    mask = (
        results_df["stage"].astype(str).eq("language_modeling")
        & results_df["model_name"].astype(str).str.startswith("Transformers (")
        & ~results_df["model_name"].astype(str).str.startswith("Transformers Search")
    )
    models_df = results_df.loc[mask].copy()
    if models_df.empty:
        return models_df
    models_df["output_dir_resolved"] = models_df["metadata"].apply(
        lambda meta: str((meta or {}).get("output_dir") or "")
    )
    models_df = models_df[models_df["output_dir_resolved"].astype(str).str.strip() != ""].copy()
    models_df = models_df[models_df["output_dir_resolved"].apply(_is_reusable_transformer_dir)].copy()
    if models_df.empty:
        return models_df

    def _format_model_label(row: pd.Series) -> str:
        metadata = row.get("metadata") or {}
        base_model = metadata.get("base_model") or "modelo?"
        text_col = metadata.get("text_col") or "text?"
        mode = metadata.get("mode") or "mode?"
        return (
            f"{row.get('created_at')} | {mode} | {text_col} | "
            f"{base_model} | {Path(str(row.get('output_dir_resolved'))).name}"
        )

    models_df["model_label"] = models_df.apply(_format_model_label, axis=1)
    return models_df.sort_values("created_at", ascending=False, ignore_index=True)


def _load_artifact_catalog(
    *,
    stage: Optional[str] = None,
    artifact_name: Optional[str] = None,
) -> pd.DataFrame:
    if duckdb is None or not REGISTRY_DB.exists():
        return pd.DataFrame()
    clauses: List[str] = []
    params: List[object] = []
    if stage:
        clauses.append("stage = ?")
        params.append(stage)
    if artifact_name:
        clauses.append("artifact_name = ?")
        params.append(artifact_name)
    query = """
        SELECT artifact_id, run_id, created_at, stage, artifact_name, db_path, table_name, row_count, metadata_json
        FROM artifacts
    """
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC"
    con = duckdb.connect(str(REGISTRY_DB), read_only=True)
    try:
        df = con.execute(query, params).df()
    finally:
        con.close()
    if df.empty:
        return df
    df["metadata"] = df["metadata_json"].apply(
        lambda raw: json.loads(raw) if isinstance(raw, str) and raw else {}
    )
    return df


def _read_artifact_df(db_path: object, table_name: object) -> pd.DataFrame:
    if duckdb is None:  # pragma: no cover
        raise RuntimeError("duckdb no esta instalado.")
    path = Path(str(db_path))
    if not path.exists():
        raise FileNotFoundError(f"No existe el artefacto: {path}")
    con = duckdb.connect(str(path), read_only=True)
    try:
        table_ref = _quote_identifier(str(table_name))
        return con.execute(f"SELECT * FROM {table_ref}").df()
    finally:
        con.close()


def _list_feature_engineering_artifacts() -> pd.DataFrame:
    catalog = _load_artifact_catalog(stage="feature_engineering", artifact_name="severity_features")
    if catalog.empty:
        return catalog
    catalog = catalog.copy()
    catalog["label"] = catalog.apply(
        lambda row: (
            f"{row['created_at']} | {int(row['row_count'] or 0):,} filas | "
            f"{Path(str(row['db_path'])).name}"
        ),
        axis=1,
    )
    return catalog


def _load_feature_bundle_from_catalog_row(row: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, object]]]:
    features_df = _read_artifact_df(row["db_path"], row["table_name"])
    granular_df = pd.DataFrame()
    granular_artifact: Optional[Dict[str, object]] = None
    run_id = row.get("run_id")
    if run_id:
        granular_catalog = _load_artifact_catalog(
            stage="feature_engineering",
            artifact_name="severity_granular",
        )
        if not granular_catalog.empty:
            granular_match = granular_catalog.loc[granular_catalog["run_id"] == run_id]
            if not granular_match.empty:
                g_row = granular_match.iloc[0]
                granular_df = _read_artifact_df(g_row["db_path"], g_row["table_name"])
                granular_artifact = {
                    "artifact_id": g_row.get("artifact_id"),
                    "artifact_name": g_row.get("artifact_name"),
                    "db_path": str(g_row.get("db_path")),
                    "table_name": g_row.get("table_name"),
                    "row_count": int(g_row.get("row_count") or 0),
                    "metadata": g_row.get("metadata") or {},
                    "run_id": run_id,
                }
    feature_artifact = {
        "artifact_id": row.get("artifact_id"),
        "artifact_name": row.get("artifact_name"),
        "db_path": str(row.get("db_path")),
        "table_name": row.get("table_name"),
        "row_count": int(row.get("row_count") or 0),
        "metadata": row.get("metadata") or {},
        "run_id": run_id,
    }
    return features_df, granular_df, feature_artifact if not granular_artifact else {
        **feature_artifact,
        "paired_granular": granular_artifact,
    }


def _feature_date_defaults(accidents_df: Optional[pd.DataFrame]) -> Tuple[datetime.date, datetime.date]:
    today = datetime.today().date()
    if accidents_df is None or accidents_df.empty or "accidente_time" not in accidents_df.columns:
        return today, today
    times = pd.to_datetime(accidents_df["accidente_time"], errors="coerce").dropna()
    if times.empty:
        return today, today
    return times.min().date(), times.max().date()


def _build_feature_sample_mode_selector(key_prefix: str) -> str:
    return st.radio(
        "Muestreo",
        ["Todo", "Rango de fechas", "Porcentaje"],
        horizontal=True,
        key=f"{key_prefix}_sample_mode",
    )


def _build_feature_sample_inputs(
    accidents_df: pd.DataFrame,
    mode: str,
    *,
    key_prefix: str,
) -> Tuple[FlowSampleSelection, bool, Optional[int]]:
    date_start = None
    date_end = None
    row_limit = None
    range_valid = True
    sample_seed: Optional[int] = None

    if mode == "Rango de fechas":
        default_start, default_end = _feature_date_defaults(accidents_df)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha inicio",
                value=default_start,
                key=f"{key_prefix}_start_date",
            )
        with col2:
            end_date = st.date_input(
                "Fecha fin",
                value=default_end,
                key=f"{key_prefix}_end_date",
            )
        use_time = st.checkbox(
            "Usar horas en el rango",
            value=False,
            key=f"{key_prefix}_use_time",
        )
        if use_time:
            col3, col4 = st.columns(2)
            with col3:
                start_time = st.time_input(
                    "Hora inicio",
                    value=dt_time(0, 0),
                    key=f"{key_prefix}_start_time",
                )
            with col4:
                end_time = st.time_input(
                    "Hora fin",
                    value=dt_time(23, 59),
                    key=f"{key_prefix}_end_time",
                )
        else:
            start_time = dt_time(0, 0)
            end_time = dt_time(23, 59, 59)
        start_ts = pd.Timestamp(datetime.combine(start_date, start_time))
        end_ts = pd.Timestamp(datetime.combine(end_date, end_time))
        if end_ts <= start_ts:
            st.error("La fecha final debe ser posterior a la fecha de inicio.")
            range_valid = False
        else:
            date_start = start_ts
            date_end = end_ts
    elif mode == "Porcentaje":
        percent = st.slider(
            "Porcentaje de eventos",
            min_value=1,
            max_value=100,
            value=10,
            key=f"{key_prefix}_percent",
        )
        sample_seed = int(
            st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1,
                key=f"{key_prefix}_percent_seed",
            )
        )
        row_limit = max(1, int(round(len(accidents_df) * (percent / 100.0))))
        st.caption(f"Se muestrearan {row_limit:,} eventos de forma reproducible.")

    sample = FlowSampleSelection(
        date_start=date_start,
        date_end=date_end,
        row_limit=row_limit,
    )
    return sample, range_valid, sample_seed


def _sample_accidents_for_feature_engineering(
    accidents_df: pd.DataFrame,
    sample: FlowSampleSelection,
    *,
    mode: str,
    sample_seed: Optional[int] = None,
) -> pd.DataFrame:
    if accidents_df is None or accidents_df.empty:
        return pd.DataFrame()
    work = accidents_df.copy()
    work["accidente_time"] = pd.to_datetime(work["accidente_time"], errors="coerce")
    work = work.dropna(subset=["accidente_time"]).reset_index(drop=True)
    if mode == "Rango de fechas" and sample.date_start is not None and sample.date_end is not None:
        work = work[
            (work["accidente_time"] >= sample.date_start)
            & (work["accidente_time"] <= sample.date_end)
        ].reset_index(drop=True)
    elif mode == "Porcentaje" and sample.row_limit is not None and len(work) > sample.row_limit:
        work = (
            work.sample(n=int(sample.row_limit), random_state=int(sample_seed or 42))
            .sort_values("accidente_time")
            .reset_index(drop=True)
        )
    return work


def _build_coverage_preview_signature(
    *,
    sample_mode: str,
    sample_seed: Optional[int],
    sampled_events: int,
    total_events: int,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
    sampling_date_start: Optional[object],
    sampling_date_end: Optional[object],
) -> Dict[str, object]:
    start_ts = pd.to_datetime(sampling_date_start, errors="coerce")
    end_ts = pd.to_datetime(sampling_date_end, errors="coerce")
    return {
        "sample_mode": sample_mode,
        "sample_seed": None if sample_seed is None else int(sample_seed),
        "sampled_events": int(sampled_events),
        "total_events": int(total_events),
        "windows_before": int(windows_before),
        "windows_after": int(windows_after),
        "window_size_minutes": int(window_size_minutes),
        "sampling_date_start": None if pd.isna(start_ts) else start_ts.isoformat(),
        "sampling_date_end": None if pd.isna(end_ts) else end_ts.isoformat(),
    }


def _init_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _numeric_feature_columns(df: pd.DataFrame, *, include_embeddings: bool = True) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [col for col in numeric_cols if col not in MODEL_EXCLUDE_COLUMNS]
    if not include_embeddings:
        cols = [col for col in cols if not col.startswith("emb_")]
    return cols


def _flow_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in _numeric_feature_columns(df) if not col.startswith("emb_")]


def _embedding_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in _numeric_feature_columns(df) if col.startswith("emb_")]


def _severity_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("severity_target"), errors="coerce")


def _has_binary_target(df: pd.DataFrame) -> bool:
    target = _severity_series(df).dropna()
    return target.nunique() == 2


def _compute_relevant_feature_ranking(
    df: pd.DataFrame,
    top_k: int,
    *,
    candidate_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if candidate_cols is None:
        numeric_cols = _flow_feature_columns(df)
    else:
        numeric_cols = [col for col in candidate_cols if col in df.columns]
    if not numeric_cols:
        return pd.DataFrame(columns=["variable", "importance"])
    target = _severity_series(df)
    work = df[numeric_cols].copy()
    valid_mask = target.notna()
    work = work.loc[valid_mask].replace([np.inf, -np.inf], np.nan)
    target = target.loc[valid_mask].astype(int)
    if work.empty or target.nunique() < 2:
        return pd.DataFrame({"variable": numeric_cols, "importance": np.zeros(len(numeric_cols))}).head(top_k)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(work)
    model = RandomForestClassifier(
        n_estimators=max(200, min(800, len(numeric_cols) * 8)),
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, target)
    ranking = pd.DataFrame(
        {
            "variable": numeric_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return ranking.head(max(1, int(top_k)))


def _window_bucket_suffix(window_size_minutes: int, bucket_idx: int) -> str:
    size = max(1, int(window_size_minutes))
    idx = max(1, int(bucket_idx))
    if size == 1:
        return f"min{idx}"
    return f"w{size}m_{idx}"


def _normalize_selected_metrics(selected_metrics: Optional[Sequence[str]]) -> List[str]:
    if selected_metrics is None:
        return GRANULAR_METRIC_NAMES.copy()
    return [metric for metric in selected_metrics if metric in GRANULAR_METRIC_NAMES]


def _emit_progress(
    progress_callback: Optional[Callable[[int, str], None]],
    value: int,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(max(0, min(100, int(value))), str(message))


def _project_granular_feature_columns(
    *,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
    selected_metrics: Optional[Sequence[str]] = None,
    include_deltas: bool = True,
) -> List[str]:
    metric_names = _normalize_selected_metrics(selected_metrics)
    label_keys = sorted({_slug(label) for label in DEFAULT_CATEGORY_LABELS.values()})
    feature_columns: List[str] = []
    for anchor in ("ultimo", "proximo"):
        for direction, total_windows in (("before", windows_before), ("after", windows_after)):
            for bucket_idx in range(1, max(0, int(total_windows)) + 1):
                suffix = _window_bucket_suffix(window_size_minutes, bucket_idx)
                for label in label_keys:
                    for metric in metric_names:
                        feature_columns.append(f"{metric}_{label}_{anchor}_{direction}_{suffix}")
    if include_deltas:
        for anchor in ("ultimo", "proximo"):
            for direction, total_windows in (("before", windows_before), ("after", windows_after)):
                for label in label_keys:
                    for metric in metric_names:
                        for bucket_idx in range(2, max(0, int(total_windows)) + 1):
                            suffix = _window_bucket_suffix(window_size_minutes, bucket_idx)
                            feature_columns.append(f"delta_{metric}_{label}_{anchor}_{direction}_{suffix}")
    return feature_columns


def _project_text_source_columns(
    *,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
) -> List[str]:
    return TEXT_SOURCE_BASE_COLUMNS.copy()


def _text_selectable_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in TEXT_SOURCE_BASE_COLUMNS if col in df.columns]


def _build_text_columns(
    df: pd.DataFrame,
    *,
    selected_columns: Sequence[str],
    text_prefix: str = "",
    include_target: bool = False,
) -> pd.DataFrame:
    work = df.copy()
    ordered_columns = [col for col in TEXT_SOURCE_BASE_COLUMNS if col in selected_columns]
    text_values: List[str] = []
    for row in work.to_dict(orient="records"):
        parts: List[str] = []
        for col in ordered_columns:
            if col not in row:
                continue
            value = row.get(col)
            if col == "accidente_time":
                rendered = _format_spanish_datetime(value)
                if rendered:
                    parts.append(rendered)
                continue
            rendered = _safe_number(value)
            label = "descripcion" if col == "descripcion" else col
            parts.append(f"{label}={rendered}")
        if include_target:
            parts.append(f"severidad={_safe_number(row.get('severity_target'))}")
        text_values.append(" | ".join([part for part in parts if part]))

    base_col = f"{text_prefix}text_bert" if text_prefix else "text_bert"
    work[base_col] = text_values
    return work


def _prepare_event_frames(file_names: Sequence[str]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    frames: List[pd.DataFrame] = []
    for file_name in file_names:
        path = DATA_DIR / file_name
        frames.append(read_csv_with_progress(str(path), sep=None))
    raw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return raw_df, frames


def load_events_for_severity(file_names: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not file_names:
        return pd.DataFrame(), pd.DataFrame()
    porticos_df = load_porticos()
    raw_df, _ = _prepare_event_frames(file_names)
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    accidents_df, excluded_df = process_accidentes_df(
        raw_df,
        porticos_df,
        return_excluded=True,
    )
    if accidents_df.empty:
        return accidents_df, excluded_df
    accidents_df = accidents_df.copy()
    accidents_df["severity_target"] = pd.to_numeric(accidents_df.get("severidad"), errors="coerce")
    accidents_df["source_files"] = ", ".join(file_names)
    return accidents_df, excluded_df


def _make_feature_accident_ids(work: pd.DataFrame) -> List[str]:
    if "_feature_event_id" in work.columns:
        event_ids = pd.to_numeric(work["_feature_event_id"], errors="coerce")
        if event_ids.notna().all():
            return [f"evt_{int(value):06d}" for value in event_ids.tolist()]
    return [f"acc_{idx:06d}" for idx in range(len(work))]


def _normalize_accidents_for_feature_engineering(accidents_df: pd.DataFrame) -> pd.DataFrame:
    if accidents_df is None or accidents_df.empty:
        return pd.DataFrame()

    work = accidents_df.copy()
    work["accidente_time"] = pd.to_datetime(work["accidente_time"], errors="coerce")
    work = work.dropna(subset=["accidente_time"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    km_col = buscar_columna(work, "Km.", aliases=["Km"])
    eje_col = buscar_columna(work, "Eje")
    calzada_col = buscar_columna(work, "Calzada")
    tipo_col = buscar_columna(work, "Tipo")
    subtipo_col = buscar_columna(work, "SubTipo", aliases=["Sub Tipo", "Sub-Tipo"])
    desc_col = buscar_columna(
        work,
        "Descripcion",
        aliases=["Descripcion", "Descripcion del evento", "Descripcion Evento", "Descripción"],
    )

    return pd.DataFrame(
        {
            "accident_id": _make_feature_accident_ids(work),
            "accidente_time": work["accidente_time"],
            "km": pd.to_numeric(work[km_col], errors="coerce"),
            "eje": work[eje_col].astype(str),
            "calzada": work[calzada_col].astype(str),
            "tipo": work[tipo_col].astype(str),
            "subtipo": work[subtipo_col].astype(str),
            "descripcion": work[desc_col].astype(str),
            "duracion_accidente": pd.to_numeric(work.get("duracion_accidente"), errors="coerce"),
            "severidad": pd.to_numeric(work.get("severidad"), errors="coerce"),
            "severity_target": pd.to_numeric(work.get("severity_target"), errors="coerce"),
            "ultimo_portico": work["ultimo_portico"].map(_normalize_portico),
            "proximo_portico": work["proximo_portico"].map(_normalize_portico),
            "source_files": work.get("source_files", ""),
        }
    )


def _build_window_rows(
    accidents_df: pd.DataFrame,
    *,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    window_delta = pd.Timedelta(minutes=max(1, int(window_size_minutes)))
    for row in accidents_df.to_dict(orient="records"):
        accident_id = row["accident_id"]
        accident_time = pd.to_datetime(row["accidente_time"], errors="coerce")
        if pd.isna(accident_time):
            continue
        for anchor, portico in (
            ("ultimo", _normalize_portico(row.get("ultimo_portico"))),
            ("proximo", _normalize_portico(row.get("proximo_portico"))),
        ):
            if portico is None:
                continue
            for minute_idx in range(1, max(0, int(windows_before)) + 1):
                window_end = accident_time - window_delta * (minute_idx - 1)
                window_start = window_end - window_delta
                rows.append(
                    {
                        "accident_id": accident_id,
                        "anchor": anchor,
                        "direction": "before",
                        "minute_idx": minute_idx,
                        "window_size_minutes": int(window_delta / pd.Timedelta(minutes=1)),
                        "portico": portico,
                        "window_start": window_start,
                        "window_end": window_end,
                    }
                )
            for minute_idx in range(1, max(0, int(windows_after)) + 1):
                window_start = accident_time + window_delta * (minute_idx - 1)
                window_end = window_start + window_delta
                rows.append(
                    {
                        "accident_id": accident_id,
                        "anchor": anchor,
                        "direction": "after",
                        "minute_idx": minute_idx,
                        "window_size_minutes": int(window_delta / pd.Timedelta(minutes=1)),
                        "portico": portico,
                        "window_start": window_start,
                        "window_end": window_end,
                    }
                )
    return pd.DataFrame(rows)


def _category_case_sql() -> str:
    label_to_raw: Dict[str, List[int]] = {}
    for raw_value, mapped_value in DEFAULT_CATEGORY_REMAP.items():
        label = DEFAULT_CATEGORY_LABELS.get(mapped_value)
        if not label:
            continue
        label_key = _slug(label)
        label_to_raw.setdefault(label_key, []).append(int(raw_value))
    clauses = []
    for label, raw_values in label_to_raw.items():
        values_sql = ", ".join(str(int(value)) for value in sorted(set(raw_values)))
        clauses.append(
            f"WHEN TRY_CAST(f.CATEGORIA AS INTEGER) IN ({values_sql}) THEN '{label}'"
        )
    return "CASE " + " ".join(clauses) + " ELSE NULL END"


def _compute_granular_metrics_duckdb(
    accidents_df: pd.DataFrame,
    *,
    flow_db_path: Path,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
    selected_metrics: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> pd.DataFrame:
    if duckdb is None:  # pragma: no cover
        raise RuntimeError("duckdb no esta instalado.")
    metric_names = _normalize_selected_metrics(selected_metrics)
    _emit_progress(progress_callback, 20, "Construyendo ventanas temporales.")
    windows_df = _build_window_rows(
        accidents_df,
        windows_before=windows_before,
        windows_after=windows_after,
        window_size_minutes=window_size_minutes,
    )
    if windows_df.empty:
        return pd.DataFrame()
    _emit_progress(
        progress_callback,
        45,
        f"Consultando DuckDB y agregando metricas para {len(windows_df):,} ventanas.",
    )
    con = duckdb.connect()
    escaped_path = str(flow_db_path).replace("'", "''")
    try:
        con.register("windows_df", windows_df)
        con.execute(f"ATTACH '{escaped_path}' AS flow_db (READ_ONLY)")
        query = f"""
            WITH agg AS (
                SELECT
                    w.accident_id,
                    w.anchor,
                    w.direction,
                    w.minute_idx,
                    w.window_size_minutes,
                    {_category_case_sql()} AS category_label,
                    COUNT(f.FECHA) AS flow,
                    AVG(TRY_CAST(f.VELOCIDAD AS DOUBLE)) AS speed_mean,
                    COALESCE(STDDEV_SAMP(TRY_CAST(f.VELOCIDAD AS DOUBLE)), 0.0) AS speed_std,
                    CASE
                        WHEN AVG(TRY_CAST(f.VELOCIDAD AS DOUBLE)) > 0
                            THEN COUNT(f.FECHA) / AVG(TRY_CAST(f.VELOCIDAD AS DOUBLE))
                        ELSE 0.0
                    END AS density
                FROM windows_df w
                LEFT JOIN flow_db.{FLOW_TABLE_NAME} f
                    ON TRIM(CAST(f.PORTICO AS VARCHAR)) = w.portico
                   AND f.FECHA >= w.window_start
                   AND f.FECHA < w.window_end
                GROUP BY 1, 2, 3, 4, 5, 6
            )
            SELECT *
            FROM agg
            WHERE category_label IS NOT NULL
        """
        df = con.execute(query).df()
        if df.empty:
            return df
        base_cols = [
            "accident_id",
            "anchor",
            "direction",
            "minute_idx",
            "window_size_minutes",
            "category_label",
        ]
        _emit_progress(progress_callback, 65, "Metricas granulares obtenidas desde DuckDB.")
        return df[base_cols + metric_names]
    finally:
        con.close()


def _build_flow_coverage_rows(
    normalized_accidents_df: pd.DataFrame,
    *,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
) -> pd.DataFrame:
    if normalized_accidents_df is None or normalized_accidents_df.empty:
        return pd.DataFrame(columns=["accident_id", "portico", "window_start", "window_end"])

    total_before_minutes = max(0, int(windows_before)) * max(1, int(window_size_minutes))
    total_after_minutes = max(0, int(windows_after)) * max(1, int(window_size_minutes))
    if total_before_minutes <= 0 and total_after_minutes <= 0:
        return pd.DataFrame(columns=["accident_id", "portico", "window_start", "window_end"])

    work = normalized_accidents_df.copy()
    work["accidente_time"] = pd.to_datetime(work["accidente_time"], errors="coerce")
    work = work.dropna(subset=["accidente_time"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["accident_id", "portico", "window_start", "window_end"])

    before_delta = pd.Timedelta(minutes=total_before_minutes)
    after_delta = pd.Timedelta(minutes=total_after_minutes)
    coverage_frames: List[pd.DataFrame] = []

    for portico_col in ("ultimo_portico", "proximo_portico"):
        if portico_col not in work.columns:
            continue
        frame = work[["accident_id", "accidente_time", portico_col]].rename(
            columns={portico_col: "portico"}
        )
        frame["portico"] = frame["portico"].map(_normalize_portico)
        frame = frame.dropna(subset=["portico"]).copy()
        if frame.empty:
            continue
        frame["window_start"] = frame["accidente_time"] - before_delta
        frame["window_end"] = frame["accidente_time"] + after_delta
        coverage_frames.append(frame[["accident_id", "portico", "window_start", "window_end"]])

    if not coverage_frames:
        return pd.DataFrame(columns=["accident_id", "portico", "window_start", "window_end"])

    coverage_df = pd.concat(coverage_frames, ignore_index=True)
    coverage_df = coverage_df[coverage_df["window_end"] > coverage_df["window_start"]]
    if coverage_df.empty:
        return pd.DataFrame(columns=["accident_id", "portico", "window_start", "window_end"])
    return coverage_df.drop_duplicates(ignore_index=True)


@st.cache_data(show_spinner=False)
def _find_accident_ids_with_flow_coverage(
    normalized_accidents_df: pd.DataFrame,
    *,
    flow_db_path: str,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
) -> List[str]:
    if duckdb is None:  # pragma: no cover
        raise RuntimeError("duckdb no esta instalado.")
    if normalized_accidents_df is None or normalized_accidents_df.empty:
        return []

    coverage_df = _build_flow_coverage_rows(
        normalized_accidents_df,
        windows_before=windows_before,
        windows_after=windows_after,
        window_size_minutes=window_size_minutes,
    )
    if coverage_df.empty:
        return []

    con = duckdb.connect()
    escaped_path = str(flow_db_path).replace("'", "''")
    try:
        con.register("coverage_df", coverage_df)
        con.execute(f"ATTACH '{escaped_path}' AS flow_db (READ_ONLY)")
        coverage_query = f"""
            WITH relevant_flows AS (
                SELECT f.FECHA, f.PORTICO, f.CATEGORIA
                FROM flow_db.{FLOW_TABLE_NAME} f
                WHERE f.FECHA >= (SELECT MIN(window_start) FROM coverage_df)
                  AND f.FECHA < (SELECT MAX(window_end) FROM coverage_df)
                  AND TRIM(CAST(f.PORTICO AS VARCHAR)) IN (SELECT DISTINCT portico FROM coverage_df)
                  AND {_category_case_sql()} IS NOT NULL
            )
            SELECT DISTINCT c.accident_id
            FROM coverage_df c
            JOIN relevant_flows f
              ON TRIM(CAST(f.PORTICO AS VARCHAR)) = c.portico
             AND f.FECHA >= c.window_start
             AND f.FECHA < c.window_end
            ORDER BY 1
        """
        coverage_df = con.execute(coverage_query).df()
    finally:
        con.close()
    if coverage_df.empty:
        return []
    return coverage_df["accident_id"].astype(str).tolist()


def _annotate_events_with_flow_coverage(
    accidents_df: pd.DataFrame,
    *,
    flow_db_path: str,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    work = accidents_df.copy()
    if work.empty:
        return work, {
            "coverage_evaluated": True,
            "covered_events": 0,
            "uncovered_events": 0,
            "window_size_minutes": int(window_size_minutes),
            "windows_before": int(windows_before),
            "windows_after": int(windows_after),
        }

    indexed = work.reset_index(drop=False).rename(columns={"index": "_feature_event_id"})
    indexed["__coverage_accident_id"] = _make_feature_accident_ids(indexed)
    normalized = _normalize_accidents_for_feature_engineering(indexed)
    covered_ids = set(
        _find_accident_ids_with_flow_coverage(
            normalized,
            flow_db_path=str(flow_db_path),
            windows_before=int(windows_before),
            windows_after=int(windows_after),
            window_size_minutes=int(window_size_minutes),
        )
    )

    indexed["has_flow_coverage"] = indexed["__coverage_accident_id"].isin(covered_ids)
    indexed["flow_coverage_label"] = np.where(
        indexed["has_flow_coverage"],
        "Con datos de flujo",
        "Sin datos de flujo",
    )

    result = indexed.drop(columns=["_feature_event_id", "__coverage_accident_id"]).copy()
    result["has_flow_coverage"] = result["has_flow_coverage"].astype("boolean")
    covered_events = int(result["has_flow_coverage"].fillna(False).sum())
    uncovered_events = int(len(result) - covered_events)
    return result, {
        "coverage_evaluated": True,
        "covered_events": covered_events,
        "uncovered_events": uncovered_events,
        "window_size_minutes": int(window_size_minutes),
        "windows_before": int(windows_before),
        "windows_after": int(windows_after),
    }


def _wide_granular_features(
    accidents_df: pd.DataFrame,
    granular_df: pd.DataFrame,
    *,
    windows_before: int,
    windows_after: int,
    window_size_minutes: int,
    selected_metrics: Optional[Sequence[str]] = None,
    include_deltas: bool = True,
) -> pd.DataFrame:
    metric_names = _normalize_selected_metrics(selected_metrics)
    label_keys = sorted({_slug(label) for label in DEFAULT_CATEGORY_LABELS.values()})
    work = accidents_df.copy()
    work = work.set_index("accident_id", drop=False)
    row_lookup = {accident_id: idx for idx, accident_id in enumerate(work.index.tolist())}
    feature_data: Dict[str, np.ndarray] = {}

    for anchor in ("ultimo", "proximo"):
        for direction, total_windows in (("before", windows_before), ("after", windows_after)):
            for minute_idx in range(1, max(0, int(total_windows)) + 1):
                suffix = _window_bucket_suffix(window_size_minutes, minute_idx)
                for label in label_keys:
                    for metric in metric_names:
                        col_name = f"{metric}_{label}_{anchor}_{direction}_{suffix}"
                        feature_data[col_name] = np.zeros(len(work), dtype=float)

    for row in granular_df.to_dict(orient="records"):
        accident_id = row.get("accident_id")
        row_idx = row_lookup.get(accident_id)
        if row_idx is None:
            continue
        label = _slug(row.get("category_label"))
        anchor = _slug(row.get("anchor"))
        direction = _slug(row.get("direction"))
        minute_idx = int(row.get("minute_idx") or 0)
        if minute_idx <= 0:
            continue
        suffix = _window_bucket_suffix(window_size_minutes, minute_idx)
        for metric in metric_names:
            col = f"{metric}_{label}_{anchor}_{direction}_{suffix}"
            if col in feature_data:
                feature_data[col][row_idx] = float(row.get(metric) or 0.0)

    delta_data: Dict[str, np.ndarray] = {}
    if include_deltas:
        for anchor in ("ultimo", "proximo"):
            for direction, total_windows in (("before", windows_before), ("after", windows_after)):
                for label in label_keys:
                    for metric in metric_names:
                        prev_col = f"{metric}_{label}_{anchor}_{direction}_{_window_bucket_suffix(window_size_minutes, 1)}"
                        for minute_idx in range(2, max(0, int(total_windows)) + 1):
                            suffix = _window_bucket_suffix(window_size_minutes, minute_idx)
                            current_col = f"{metric}_{label}_{anchor}_{direction}_{suffix}"
                            delta_col = f"delta_{metric}_{label}_{anchor}_{direction}_{suffix}"
                            delta_data[delta_col] = feature_data[current_col] - feature_data[prev_col]
                            prev_col = current_col

    if feature_data:
        work = pd.concat([work, pd.DataFrame(feature_data, index=work.index)], axis=1)
    if delta_data:
        work = pd.concat([work, pd.DataFrame(delta_data, index=work.index)], axis=1)
    work["interval_start"] = pd.to_datetime(work["accidente_time"], errors="coerce")
    return work.reset_index(drop=True)


def _filter_to_accidents_with_flow_coverage(
    accidents_df: pd.DataFrame,
    granular_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if accidents_df is None or accidents_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if granular_df is None or granular_df.empty or "accident_id" not in granular_df.columns:
        return accidents_df.iloc[0:0].copy(), pd.DataFrame(columns=getattr(granular_df, "columns", None))

    covered_ids = granular_df["accident_id"].dropna().astype(str).unique().tolist()
    if not covered_ids:
        return accidents_df.iloc[0:0].copy(), granular_df.iloc[0:0].copy()

    filtered_accidents = (
        accidents_df[accidents_df["accident_id"].astype(str).isin(covered_ids)]
        .copy()
        .reset_index(drop=True)
    )
    filtered_granular = (
        granular_df[granular_df["accident_id"].astype(str).isin(covered_ids)]
        .copy()
        .reset_index(drop=True)
    )
    return filtered_accidents, filtered_granular


def build_severity_feature_dataset(
    accidents_df: pd.DataFrame,
    *,
    flow_db_path: Path,
    windows_before: int = 5,
    windows_after: int = 5,
    window_size_minutes: int = 1,
    top_k_ranking: Optional[int] = None,
    selected_metrics: Optional[Sequence[str]] = None,
    include_deltas: bool = True,
    text_columns: Optional[Sequence[str]] = None,
    include_target_in_text: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if accidents_df is None or accidents_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    _emit_progress(progress_callback, 5, "Preparando eventos seleccionados.")
    _emit_progress(progress_callback, 12, "Normalizando columnas base del dataset.")
    normalized = _normalize_accidents_for_feature_engineering(accidents_df)
    if normalized.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    granular_df = _compute_granular_metrics_duckdb(
        normalized,
        flow_db_path=flow_db_path,
        windows_before=windows_before,
        windows_after=windows_after,
        window_size_minutes=window_size_minutes,
        selected_metrics=selected_metrics,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, 72, "Filtrando accidentes sin cobertura de flujo.")
    normalized, granular_df = _filter_to_accidents_with_flow_coverage(normalized, granular_df)
    if normalized.empty:
        _emit_progress(progress_callback, 100, "No se encontraron accidentes con cobertura de flujo.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    _emit_progress(progress_callback, 80, "Consolidando variables granulares por accidente.")
    wide_df = _wide_granular_features(
        normalized,
        granular_df,
        windows_before=windows_before,
        windows_after=windows_after,
        window_size_minutes=window_size_minutes,
        selected_metrics=selected_metrics,
        include_deltas=include_deltas,
    )
    ranking_limit = (
        max(1, len(_flow_feature_columns(wide_df)))
        if top_k_ranking is None
        else max(1, int(top_k_ranking))
    )
    _emit_progress(progress_callback, 90, "Calculando ranking de variables relevantes.")
    ranking_df = _compute_relevant_feature_ranking(wide_df, top_k=ranking_limit)
    selected_text_columns = list(text_columns) if text_columns is not None else _text_selectable_columns(wide_df)
    _emit_progress(progress_callback, 96, "Generando columnas de texto.")
    wide_df = _build_text_columns(
        wide_df,
        selected_columns=selected_text_columns,
        include_target=include_target_in_text,
    )
    _emit_progress(progress_callback, 100, "Dataset granular generado.")
    return wide_df, granular_df, ranking_df


def generate_text_embeddings(
    df: pd.DataFrame,
    *,
    text_col: str,
    method: str,
    n_components: int,
    max_features: int,
    random_state: int = 42,
    transformer_model_path: Optional[str] = None,
    transformer_batch_size: int = 16,
    transformer_max_length: int = 256,
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    if df is None or df.empty:
        return pd.DataFrame(), [], {}
    if text_col not in df.columns:
        raise ValueError(f"La columna '{text_col}' no existe.")

    texts = df[text_col].fillna("").astype(str).str.strip()
    if texts.eq("").all():
        raise ValueError("No hay textos disponibles para generar embeddings.")

    work = df.copy()
    meta: Dict[str, object] = {"method": method, "text_col": text_col}

    if method == "sentence_transformer":
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers no esta instalado.")
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(model_name)
        matrix = model.encode(texts.tolist(), show_progress_bar=False)
        meta["model_name"] = model_name
    elif method == "transformer_finetuned":
        if torch is None or AutoTokenizer is None or AutoModel is None:
            raise ImportError("Se requiere torch y transformers para usar modelos fine-tuneados.")
        if not transformer_model_path:
            raise ValueError("Seleccione un modelo fine-tuneado para generar embeddings.")
        model_path = Path(str(transformer_model_path))
        if not model_path.exists():
            raise FileNotFoundError(f"No existe el modelo fine-tuneado: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        model = AutoModel.from_pretrained(str(model_path))
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        model.to(device)
        model.eval()
        batch_size = max(1, int(transformer_batch_size))
        all_batches: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts.iloc[start : start + batch_size].tolist()
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max(8, int(transformer_max_length)),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded, return_dict=True)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).detach().cpu().numpy()
            all_batches.append(batch_embeddings)
        matrix = np.vstack(all_batches) if all_batches else np.zeros((len(work), 1), dtype=float)
        meta["model_name"] = str(model_path.name)
        meta["model_path"] = str(model_path)
        meta["batch_size"] = int(batch_size)
        meta["max_length"] = int(transformer_max_length)
        meta["projection"] = "transformer_mean_pooling"
    else:
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            max_features=max(100, int(max_features)),
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf = vectorizer.fit_transform(texts.tolist())
        max_rank = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if max_rank < 1:
            matrix = tfidf.toarray().astype(float)
            meta["projection"] = "none"
        else:
            components = min(max_rank, max(2, int(n_components)))
            svd = TruncatedSVD(n_components=components, random_state=random_state)
            matrix = svd.fit_transform(tfidf)
            meta["projection"] = "tfidf_svd"
            meta["explained_variance"] = float(np.sum(svd.explained_variance_ratio_))
            meta["svd_components"] = int(components)

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    embed_cols = [f"emb_{idx:03d}" for idx in range(matrix.shape[1])]
    embed_df = pd.DataFrame(matrix, columns=embed_cols, index=work.index)
    for col in embed_cols:
        work[col] = embed_df[col].astype(float)
    meta["embedding_dims"] = int(len(embed_cols))
    meta["rows"] = int(len(work))
    return work, embed_cols, meta


def run_embedding_rf_analysis(df: pd.DataFrame, embed_cols: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty or not embed_cols:
        return pd.DataFrame()
    target = _severity_series(df)
    valid_mask = target.notna()
    if valid_mask.sum() == 0:
        return pd.DataFrame()
    X = df.loc[valid_mask, list(embed_cols)].replace([np.inf, -np.inf], np.nan)
    y = target.loc[valid_mask].astype(int)
    if y.nunique() < 2:
        return pd.DataFrame()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_imp, y)
    return pd.DataFrame(
        {
            "variable": list(embed_cols),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def _select_top_embedding_features(
    ranking_df: pd.DataFrame,
    *,
    top_k: int,
) -> List[str]:
    if not isinstance(ranking_df, pd.DataFrame) or ranking_df.empty or "variable" not in ranking_df.columns:
        return []
    return [
        str(value)
        for value in ranking_df["variable"].head(max(1, int(top_k))).tolist()
        if isinstance(value, str) and value.startswith("emb_")
    ]


def _build_train_dataset_with_selected_embeddings(
    features_df: pd.DataFrame,
    embeddings_df: Optional[pd.DataFrame],
    *,
    selected_embedding_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if features_df is None or features_df.empty:
        base_df = pd.DataFrame()
    else:
        base_df = features_df.copy()
    if embeddings_df is None or embeddings_df.empty:
        return base_df

    all_embedding_cols = _embedding_feature_columns(embeddings_df)
    if not all_embedding_cols:
        return base_df if not base_df.empty else embeddings_df.copy()

    requested_cols = list(selected_embedding_cols or all_embedding_cols)
    selected_cols = [col for col in requested_cols if col in all_embedding_cols]
    if not selected_cols:
        selected_cols = list(all_embedding_cols)

    if not base_df.empty and "accident_id" in base_df.columns and "accident_id" in embeddings_df.columns:
        merge_cols = ["accident_id"] + list(selected_cols)
        if "severity_target" in embeddings_df.columns and "severity_target" not in base_df.columns:
            merge_cols.append("severity_target")
        embed_subset = embeddings_df[merge_cols].drop_duplicates(subset=["accident_id"])
        return base_df.merge(embed_subset, on="accident_id", how="left")

    drop_cols = [col for col in all_embedding_cols if col not in selected_cols]
    return embeddings_df.drop(columns=drop_cols, errors="ignore").copy()


def _classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    unique_labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    average = "binary" if len(unique_labels) <= 2 else "macro"
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_labels)
    false_negatives_by_class = {
        str(label): int(cm[idx, :].sum() - cm[idx, idx])
        for idx, label in enumerate(unique_labels)
    }
    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": unique_labels,
        "false_negatives_global": int(sum(false_negatives_by_class.values())),
        "false_negatives_by_class": false_negatives_by_class,
    }
    if y_score is not None and len(unique_labels) <= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, np.asarray(y_score, dtype=float)))
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def _render_confusion_matrix_summary(model_name: str, metrics: Dict[str, object]) -> None:
    labels = list(metrics.get("labels") or [])
    raw_matrix = metrics.get("confusion_matrix") or []
    if not labels or not raw_matrix:
        return
    matrix_df = pd.DataFrame(
        raw_matrix,
        index=[f"real_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    fn_by_class = metrics.get("false_negatives_by_class") or {}
    fn_rows = pd.DataFrame(
        [
            {
                "clase_real": str(label),
                "fn": int(fn_by_class.get(str(label), 0)),
            }
            for label in labels
        ]
    )
    st.markdown(f"#### Matriz de confusion · {model_name}")
    c1, c2 = st.columns(2)
    c1.metric("FN global", f"{int(metrics.get('false_negatives_global') or 0):,}")
    c2.metric("Clases", f"{len(labels):,}")
    st.dataframe(matrix_df, width="stretch")
    st.dataframe(fn_rows, width="stretch")


def _render_controlled_comparison_protocol(
    *,
    feature_group: str,
    feature_count: int,
    feature_count_per_model: int,
    split_mode: str,
    test_size: float,
    random_state: int,
    xgb_tuning_profile: str,
    tuning_folds: int,
    protocol: Optional[Dict[str, object]] = None,
) -> None:
    with st.expander("Protocolo de comparacion", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "1. Se define un unico `holdout` con el `feature group` seleccionado. Ese test queda congelado y se reutiliza en los tres modelos.",
                    "2. El `split` se genera una sola vez con el mismo `random_state`. Si el split temporal no deja ambas clases en train/test, se degrada a estratificado.",
                    "3. RF + XGBoost usa solo train: imputacion mediana, SMOTE si aplica, ranking RF, seleccion del numero exacto de variables definido en la UI y luego busqueda XGBoost.",
                    "4. Elastic Net usa solo train: imputacion, escalado, SMOTE si aplica, `GridSearchCV` sobre `C in {0.01, 0.1, 1, 10}` y `l1_ratio in {0.1, 0.5, 0.9}`, y luego se conservan exactamente las variables con mayor `|coef|` hasta el numero definido.",
                    "5. SVM + RFE usa solo train: imputacion, escalado, SMOTE si aplica, `RFE` con el numero exacto de variables definido y `GridSearchCV` del SVM con `C in {0.1, 1, 10}` y `kernel in {linear, rbf}`.",
                    "6. Las matrices de confusion finales se calculan sobre exactamente las mismas filas y etiquetas de test.",
                ]
            )
        )
        st.caption(
            "Configuracion solicitada: "
            f"grupo={feature_group} | variables disponibles={feature_count} | "
            f"numero por modelo={feature_count_per_model} | split={split_mode} | "
            f"test_size={float(test_size):.2f} | random_state={int(random_state)}"
        )
        st.caption(
            "Ajuste interno: "
            f"XGBoost={xgb_tuning_profile} | "
            f"folds internos compartidos={int(tuning_folds)} | "
            f"variables por modelo={feature_count_per_model}"
        )
        st.caption(f"Detalle XGBoost: {_xgb_search_strategy_help(xgb_tuning_profile)}")
        if protocol:
            st.json(protocol)


def _prepare_holdout_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, object]]:
    if not feature_cols:
        raise ValueError("No hay columnas numericas para entrenar.")

    target = _severity_series(df)
    work = df[list(feature_cols)].copy()
    work["severity_target"] = target
    if "accidente_time" in df.columns:
        work["accidente_time"] = pd.to_datetime(df["accidente_time"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["severity_target"]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No hay filas validas para entrenar.")
    y = work.pop("severity_target").astype(int)
    time_col = work.pop("accidente_time") if "accidente_time" in work.columns else None
    if y.nunique() < 2:
        raise ValueError("La variable objetivo debe ser binaria.")

    if split_mode == "Temporal" and time_col is not None and time_col.notna().sum() >= 2:
        order = np.argsort(pd.to_datetime(time_col, errors="coerce").fillna(pd.Timestamp.min).to_numpy())
        X_sorted = work.iloc[order].reset_index(drop=True)
        y_sorted = y.iloc[order].reset_index(drop=True)
        n_test = max(1, int(round(len(X_sorted) * float(test_size))))
        if n_test >= len(X_sorted):
            n_test = len(X_sorted) - 1
        X_train = X_sorted.iloc[:-n_test].reset_index(drop=True)
        X_test = X_sorted.iloc[-n_test:].reset_index(drop=True)
        y_train = y_sorted.iloc[:-n_test].reset_index(drop=True)
        y_test = y_sorted.iloc[-n_test:].reset_index(drop=True)
        if y_train.nunique() >= 2 and y_test.nunique() >= 2:
            return X_train, X_test, y_train, y_test, {"split_mode": "Temporal"}

    X_train, X_test, y_train, y_test = train_test_split(
        work,
        y,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        {"split_mode": "Estratificado"},
    )


def _fit_imputer(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    return X_train_imp, X_test_imp, imputer


def _maybe_balance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    if SMOTE is None:
        return X_train, y_train, {"balancing": "none", "reason": "smote_no_instalado"}
    class_counts = y_train.value_counts(dropna=False)
    if class_counts.empty or int(class_counts.min()) < 2:
        return X_train, y_train, {"balancing": "none", "reason": "clase_minoritaria_insuficiente"}
    sampler = SMOTE(random_state=int(random_state))
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    X_bal = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_bal = pd.Series(y_resampled, name=y_train.name)
    return X_bal, y_bal, {
        "balancing": "smote",
        "rows_before": int(len(X_train)),
        "rows_after": int(len(X_bal)),
    }


def _rf_rank_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
) -> pd.DataFrame:
    if X_train is None or X_train.empty:
        return pd.DataFrame(columns=["variable", "importance"])
    selector = RandomForestClassifier(
        n_estimators=max(300, min(1000, len(X_train.columns) * 8)),
        random_state=int(random_state),
        class_weight="balanced",
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    return pd.DataFrame(
        {
            "variable": list(X_train.columns),
            "importance": selector.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def _default_xgb_param_grid(profile: str) -> Dict[str, List[object]]:
    profile = str(profile or "Rapida")
    if profile == "GridSearch original":
        return {
            "max_depth": [3, 5, 7, 10, 15, 20],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 250, 500, 750, 1000],
            "subsample": [0.5, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.8, 1.0],
        }
    if profile == "Amplia":
        return {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [150, 300, 500],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    return {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [150, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def _xgb_search_strategy_help(strategy: str) -> str:
    strategy = str(strategy or "Rapida")
    if strategy == "GridSearch original":
        return (
            "Replica la grilla original de NLP/main.py. Explora 810 combinaciones "
            "de hiperparametros y puede tardar bastante."
        )
    if strategy == "Amplia":
        return (
            "Busqueda intermedia. Explora una grilla mas grande que la rapida "
            "para mejorar ajuste sin llegar al costo del GridSearch original."
        )
    return (
        "Busqueda compacta para iterar rapido. Usa una grilla reducida y sirve "
        "para obtener una buena base antes de ampliar la exploracion."
    )


def _optimize_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    tune_hyperparameters: bool,
    tuning_folds: int,
    tuning_profile: str,
) -> Tuple[object, Dict[str, object], Optional[float], pd.DataFrame]:
    base_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    if not tune_hyperparameters:
        model = build_model("XGBoost", base_params, random_state=int(random_state))
        model.fit(X_train, y_train)
        return model, dict(base_params), None, pd.DataFrame()

    min_class = int(y_train.value_counts().min())
    cv_folds = min(max(2, int(tuning_folds)), min_class)
    if cv_folds < 2:
        model = build_model("XGBoost", base_params, random_state=int(random_state))
        model.fit(X_train, y_train)
        fallback_params = {
            **base_params,
            "tuning_fallback": "clase_minoritaria_insuficiente",
        }
        return model, fallback_params, None, pd.DataFrame()

    base_model = build_model("XGBoost", base_params, random_state=int(random_state))
    scoring = "f1" if pd.Series(y_train).nunique() <= 2 else "f1_macro"
    search = GridSearchCV(
        base_model,
        param_grid=_default_xgb_param_grid(tuning_profile),
        scoring=scoring,
        cv=cv_folds,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score",
        ascending=True,
        ignore_index=True,
    )
    best_params = {
        key: value
        for key, value in search.best_params_.items()
    }
    return search.best_estimator_, best_params, float(search.best_score_), cv_results_df


def _predict_model_scores(model: object, X: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba_arr = np.asarray(proba)
        if proba_arr.ndim == 2 and proba_arr.shape[1] > 1:
            return proba_arr[:, 1]
        return proba_arr.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores, dtype=float).ravel()
    return None


def _elastic_net_penalty_is_deprecated() -> bool:
    version_match = re.match(r"^(\d+)\.(\d+)", str(SKLEARN_VERSION or ""))
    if not version_match:
        return False
    major, minor = int(version_match.group(1)), int(version_match.group(2))
    return (major, minor) >= (1, 8)


def _build_elastic_net_logistic_regression(
    *,
    random_state: int,
    max_iter: int,
    C: Optional[float] = None,
    l1_ratio: Optional[float] = None,
) -> LogisticRegression:
    kwargs: Dict[str, object] = {
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    if C is not None:
        kwargs["C"] = float(C)
    if l1_ratio is not None:
        kwargs["l1_ratio"] = float(l1_ratio)
    if not _elastic_net_penalty_is_deprecated():
        kwargs["penalty"] = "elasticnet"
    return LogisticRegression(**kwargs)


def _training_scoring_name(y_train: pd.Series) -> str:
    return "f1" if pd.Series(y_train).nunique() <= 2 else "f1_macro"


def _effective_cv_folds(y_train: pd.Series, requested_folds: int) -> int:
    class_counts = pd.Series(y_train).value_counts(dropna=False)
    if class_counts.empty:
        return 0
    min_class = int(class_counts.min())
    if min_class < 2:
        return 0
    return min(max(2, int(requested_folds)), min_class)


def _candidate_feature_caps(total_features: int, *, max_features: int) -> List[int]:
    capped_total = max(1, min(int(total_features), int(max_features)))
    return [capped_total]


def _prepare_holdout_split_with_ids(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, object]]:
    if not feature_cols:
        raise ValueError("No hay columnas numericas para entrenar.")

    target = _severity_series(df)
    work = df[list(feature_cols)].copy()
    work["severity_target"] = target
    id_field = "accident_id" if "accident_id" in df.columns else "__row_index__"
    if id_field == "accident_id":
        work[id_field] = df["accident_id"]
    else:
        work[id_field] = pd.Series(df.index, index=df.index)
    if "accidente_time" in df.columns:
        work["accidente_time"] = pd.to_datetime(df["accidente_time"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["severity_target"]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No hay filas validas para entrenar.")

    y = work.pop("severity_target").astype(int)
    row_ids = work.pop(id_field).astype(str).reset_index(drop=True)
    time_col = work.pop("accidente_time") if "accidente_time" in work.columns else None
    if y.nunique() < 2:
        raise ValueError("La variable objetivo debe ser binaria.")

    def _build_meta(applied_split_mode: str, y_tr: pd.Series, y_te: pd.Series) -> Dict[str, object]:
        return {
            "split_mode": applied_split_mode,
            "requested_split_mode": str(split_mode),
            "train_rows": int(len(y_tr)),
            "test_rows": int(len(y_te)),
            "train_class_counts": {
                str(label): int(count)
                for label, count in y_tr.value_counts().sort_index().to_dict().items()
            },
            "test_class_counts": {
                str(label): int(count)
                for label, count in y_te.value_counts().sort_index().to_dict().items()
            },
            "comparison_id_field": id_field,
        }

    if split_mode == "Temporal" and time_col is not None and time_col.notna().sum() >= 2:
        order = np.argsort(pd.to_datetime(time_col, errors="coerce").fillna(pd.Timestamp.min).to_numpy())
        X_sorted = work.iloc[order].reset_index(drop=True)
        y_sorted = y.iloc[order].reset_index(drop=True)
        ids_sorted = row_ids.iloc[order].reset_index(drop=True)
        n_test = max(1, int(round(len(X_sorted) * float(test_size))))
        if n_test >= len(X_sorted):
            n_test = len(X_sorted) - 1
        X_train = X_sorted.iloc[:-n_test].reset_index(drop=True)
        X_test = X_sorted.iloc[-n_test:].reset_index(drop=True)
        y_train = y_sorted.iloc[:-n_test].reset_index(drop=True)
        y_test = y_sorted.iloc[-n_test:].reset_index(drop=True)
        test_ids = ids_sorted.iloc[-n_test:].reset_index(drop=True)
        if y_train.nunique() >= 2 and y_test.nunique() >= 2:
            return X_train, X_test, y_train, y_test, test_ids, _build_meta("Temporal", y_train, y_test)

    X_train, X_test, y_train, y_test, _, test_ids = train_test_split(
        work,
        y,
        row_ids,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train,
        y_test,
        pd.Series(test_ids).reset_index(drop=True).astype(str),
        _build_meta("Estratificado", y_train, y_test),
    )


def _train_rf_xgb_shared_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int,
    max_features: int,
    tuning_profile: str,
    tuning_folds: int,
) -> Dict[str, object]:
    X_train_imp, X_test_imp, _ = _fit_imputer(X_train, X_test)
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_imp,
        y_train,
        random_state=int(random_state),
    )
    ranking_df = _rf_rank_features(
        X_train_bal,
        y_train_bal,
        random_state=int(random_state),
    )
    selected_cols = ranking_df["variable"].head(max(1, int(max_features))).tolist()
    if not selected_cols:
        raise ValueError("No hay variables disponibles para RF + XGBoost.")
    best_model, best_params, best_score, search_df = _optimize_xgb_classifier(
        X_train_bal[selected_cols],
        y_train_bal,
        random_state=int(random_state),
        tune_hyperparameters=True,
        tuning_folds=int(tuning_folds),
        tuning_profile=str(tuning_profile),
    )
    if isinstance(search_df, pd.DataFrame) and not search_df.empty:
        search_df = search_df.copy()
        search_df.insert(0, "rf_top_k", int(len(selected_cols)))

    y_pred = best_model.predict(X_test_imp[selected_cols])
    y_score = _predict_model_scores(best_model, X_test_imp[selected_cols])
    metrics = _classification_metrics(y_test, y_pred, y_score)
    return {
        "model_name": "RF + XGBoost",
        "feature_strategy": f"RF top-{int(len(selected_cols))}",
        "selected_cols": selected_cols,
        "ranking_df": ranking_df,
        "balancing_meta": balancing_meta,
        "search_df": search_df if isinstance(search_df, pd.DataFrame) else pd.DataFrame(),
        "params": {
            **best_params,
            "rf_top_k": int(len(selected_cols)),
            "tuning_profile": str(tuning_profile),
            "tuning_folds": int(tuning_folds),
        },
        "metrics": metrics,
        "predictions": np.asarray(y_pred, dtype=int),
        "scores": y_score,
    }


def _train_elastic_net_shared_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int,
    max_features: int,
    tuning_folds: int,
) -> Dict[str, object]:
    X_train_imp, X_test_imp, _ = _fit_imputer(X_train, X_test)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns)
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_scaled,
        y_train,
        random_state=int(random_state),
    )

    base_model = _build_elastic_net_logistic_regression(
        random_state=int(random_state),
        max_iter=7000,
    )
    cv_folds = _effective_cv_folds(y_train_bal, int(tuning_folds))
    scoring = _training_scoring_name(y_train_bal)
    search_df = pd.DataFrame()
    if cv_folds >= 2:
        search = GridSearchCV(
            base_model,
            param_grid={
                "C": [0.01, 0.1, 1.0, 10.0],
                "l1_ratio": [0.1, 0.5, 0.9],
            },
            scoring=scoring,
            cv=cv_folds,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train_bal, y_train_bal)
        tuned_model = search.best_estimator_
        best_params = dict(search.best_params_)
        search_df = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score",
            ascending=True,
            ignore_index=True,
        )
        best_score = float(search.best_score_)
    else:
        tuned_model = base_model.fit(X_train_bal, y_train_bal)
        best_params = {"search_fallback": "clase_minoritaria_insuficiente"}
        best_score = None

    coef_df = pd.DataFrame(
        {
            "variable": list(X_train_scaled.columns),
            "coef": np.asarray(tuned_model.coef_).ravel(),
        }
    )
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False, ignore_index=True)

    non_zero_cols = coef_df.loc[coef_df["abs_coef"] > 1e-8, "variable"].tolist()
    if not non_zero_cols:
        non_zero_cols = coef_df["variable"].tolist()
    selected_cols = non_zero_cols[: max(1, min(int(max_features), len(non_zero_cols)))]

    final_params = {
        "C": float(best_params.get("C", 1.0)),
        "l1_ratio": float(best_params.get("l1_ratio", 0.5)),
    }
    final_model = _build_elastic_net_logistic_regression(
        random_state=int(random_state),
        max_iter=7000,
        **final_params,
    )
    final_model.fit(X_train_bal[selected_cols], y_train_bal)
    y_pred = final_model.predict(X_test_scaled[selected_cols])
    y_score = _predict_model_scores(final_model, X_test_scaled[selected_cols])
    metrics = _classification_metrics(y_test, y_pred, y_score)
    return {
        "model_name": "Elastic Net",
        "feature_strategy": "Elastic Net shrinkage",
        "selected_cols": selected_cols,
        "ranking_df": coef_df,
        "balancing_meta": balancing_meta,
        "search_df": search_df,
        "params": {
            **best_params,
            "selected_feature_count_target": int(max_features),
            "selected_features_final": int(len(selected_cols)),
            "non_zero_features": int(len(non_zero_cols)),
            "tuning_folds": int(tuning_folds),
            "best_cv_score": best_score,
        },
        "metrics": metrics,
        "predictions": np.asarray(y_pred, dtype=int),
        "scores": y_score,
    }


def _train_svm_rfe_shared_holdout(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int,
    max_features: int,
    tuning_folds: int,
) -> Dict[str, object]:
    feature_cols = list(X_train.columns)
    X_train_imp, X_test_imp, _ = _fit_imputer(X_train, X_test)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_cols)
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_scaled,
        y_train,
        random_state=int(random_state),
    )

    scoring = _training_scoring_name(y_train_bal)
    cv_folds = _effective_cv_folds(y_train_bal, int(tuning_folds))
    selected_feature_count = max(1, min(int(max_features), len(feature_cols)))
    selector = RFE(
        estimator=LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=6000,
            random_state=int(random_state),
        ),
        n_features_to_select=selected_feature_count,
        step=max(1, len(feature_cols) // 10),
    )
    selector.fit(X_train_bal, y_train_bal)
    selected_cols = X_train_bal.columns[selector.support_].tolist()
    if not selected_cols:
        raise ValueError("No se pudo seleccionar variables con SVM + RFE.")

    ranking_df = pd.DataFrame(
        {
            "variable": feature_cols,
            "ranking_rfe": selector.ranking_,
        }
    ).sort_values(["ranking_rfe", "variable"], ignore_index=True)

    search_df = pd.DataFrame()
    if cv_folds >= 2:
        search = GridSearchCV(
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=int(random_state),
            ),
            param_grid={
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
            },
            scoring=scoring,
            cv=cv_folds,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train_bal[selected_cols], y_train_bal)
        best_model = search.best_estimator_
        best_params = dict(search.best_params_)
        best_score = float(search.best_score_)
        search_df = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score",
            ascending=True,
            ignore_index=True,
        )
        search_df.insert(0, "rfe_top_k", int(len(selected_cols)))
    else:
        best_model = SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=int(random_state),
        )
        best_model.fit(X_train_bal[selected_cols], y_train_bal)
        best_params = {"C": 1.0, "kernel": "rbf", "search_fallback": "clase_minoritaria_insuficiente"}
        best_score = None

    y_pred = best_model.predict(X_test_scaled[selected_cols])
    y_score = _predict_model_scores(best_model, X_test_scaled[selected_cols])
    metrics = _classification_metrics(y_test, y_pred, y_score)
    return {
        "model_name": "SVM + RFE",
        "feature_strategy": f"RFE top-{int(len(selected_cols))}",
        "selected_cols": selected_cols,
        "ranking_df": ranking_df,
        "balancing_meta": balancing_meta,
        "search_df": search_df,
        "params": {
            **best_params,
            "rfe_top_k": int(len(selected_cols)),
            "tuning_folds": int(tuning_folds),
            "best_cv_score": best_score,
        },
        "metrics": metrics,
        "predictions": np.asarray(y_pred, dtype=int),
        "scores": y_score,
    }


def train_model_comparison_holdout(
    df: pd.DataFrame,
    *,
    feature_group: str,
    test_size: float,
    random_state: int,
    split_mode: str,
    max_features_per_model: int,
    xgb_tuning_profile: str,
    tuning_folds: int,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    feature_cols = _resolve_feature_group(df, feature_group)
    if not feature_cols:
        raise ValueError("No hay variables disponibles para entrenar.")
    max_features = _candidate_feature_caps(
        len(feature_cols),
        max_features=min(int(max_features_per_model), 100),
    )[0]
    if progress_callback:
        progress_callback(5, "Preparando split holdout compartido...")

    X_train, X_test, y_train, y_test, test_ids, split_meta = _prepare_holdout_split_with_ids(
        df,
        feature_cols,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    if progress_callback:
        progress_callback(25, "Entrenando RF + XGBoost...")

    rf_xgb_result = _train_rf_xgb_shared_holdout(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=int(random_state),
        max_features=max_features,
        tuning_profile=str(xgb_tuning_profile),
        tuning_folds=int(tuning_folds),
    )
    if progress_callback:
        progress_callback(55, "Entrenando Elastic Net...")
    elastic_result = _train_elastic_net_shared_holdout(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=int(random_state),
        max_features=max_features,
        tuning_folds=int(tuning_folds),
    )
    if progress_callback:
        progress_callback(80, "Entrenando SVM + RFE...")
    svm_result = _train_svm_rfe_shared_holdout(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=int(random_state),
        max_features=max_features,
        tuning_folds=int(tuning_folds),
    )
    if progress_callback:
        progress_callback(95, "Armando resumen comparativo...")

    results = [rf_xgb_result, elastic_result, svm_result]
    comparison_rows: List[Dict[str, object]] = []
    params_rows: List[Dict[str, object]] = []
    for result in results:
        metrics = result.get("metrics") or {}
        comparison_rows.append(
            {
                "model_name": result["model_name"],
                "feature_strategy": result.get("feature_strategy"),
                "selected_features": int(len(result.get("selected_cols") or [])),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "roc_auc": metrics.get("roc_auc"),
                "false_negatives_global": metrics.get("false_negatives_global"),
            }
        )
        params_rows.append(
            {
                "model_name": result["model_name"],
                "feature_strategy": result.get("feature_strategy"),
                "selected_features": int(len(result.get("selected_cols") or [])),
                "params": json.dumps(result.get("params") or {}, ensure_ascii=True, default=_json_default),
            }
        )

    predictions_df = pd.DataFrame(
        {
            str(split_meta.get("comparison_id_field") or "test_id"): test_ids.tolist(),
            "severity_target": y_test.astype(int).tolist(),
            "pred_rf_xgb": rf_xgb_result["predictions"].tolist(),
            "pred_elastic_net": elastic_result["predictions"].tolist(),
            "pred_svm_rfe": svm_result["predictions"].tolist(),
        }
    )
    if rf_xgb_result.get("scores") is not None:
        predictions_df["score_rf_xgb"] = np.asarray(rf_xgb_result["scores"], dtype=float)
    if elastic_result.get("scores") is not None:
        predictions_df["score_elastic_net"] = np.asarray(elastic_result["scores"], dtype=float)
    if svm_result.get("scores") is not None:
        predictions_df["score_svm_rfe"] = np.asarray(svm_result["scores"], dtype=float)

    protocol = {
        **split_meta,
        "feature_group": str(feature_group),
        "total_available_features": int(len(feature_cols)),
        "feature_count_per_model": int(max_features),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "xgb_tuning_profile": str(xgb_tuning_profile),
        "tuning_folds": int(tuning_folds),
    }
    protocol_df = pd.DataFrame([{**_flatten_scalar_payload(protocol, prefix="protocol_")}])
    comparison_df = pd.DataFrame(comparison_rows)
    params_df = pd.DataFrame(params_rows)
    if progress_callback:
        progress_callback(100, "Comparacion controlada completada.")
    return {
        "protocol": protocol,
        "protocol_df": protocol_df,
        "comparison_df": comparison_df,
        "params_df": params_df,
        "predictions_df": predictions_df,
        "results": results,
    }


def _transformer_model_options() -> Dict[str, str]:
    options: Dict[str, str] = {}
    if LOCAL_TRANSFORMER_MODEL_DIR.exists():
        options[f"Local · {LOCAL_TRANSFORMER_MODEL_DIR.name}"] = str(LOCAL_TRANSFORMER_MODEL_DIR)
    for model_name in [
        "dccuchile/bert-base-spanish-wwm-cased",
        "PlanTL-GOB-ES/roberta-base-bne",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "xlm-roberta-large",
        "bertin-project/bertin-roberta-base",
        "answerdotai/ModernBERT-base",
    ]:
        options[model_name] = model_name
    return options


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.clip(exp.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return exp / denom


def _dedupe_preserve_order(values: Sequence[object]) -> List[object]:
    seen: set[str] = set()
    out: List[object] = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, default=_json_default)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(value)
    return out


def _flatten_scalar_payload(payload: Dict[str, object], *, prefix: str) -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in payload.items():
        column = f"{prefix}{key}"
        if isinstance(value, (str, bool, int, float)) or value is None:
            flat[column] = value
        else:
            flat[column] = json.dumps(value, ensure_ascii=True, default=_json_default)
    return flat


class TransformerSearchDebugError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        search_result: Optional[Dict[str, object]] = None,
        trials_df: Optional[pd.DataFrame] = None,
        confirm_df: Optional[pd.DataFrame] = None,
        confirm_summary_df: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__(message)
        self.search_result = search_result or {}
        self.trials_df = (
            trials_df.copy()
            if isinstance(trials_df, pd.DataFrame)
            else pd.DataFrame()
        )
        self.confirm_df = (
            confirm_df.copy()
            if isinstance(confirm_df, pd.DataFrame)
            else pd.DataFrame()
        )
        self.confirm_summary_df = (
            confirm_summary_df.copy()
            if isinstance(confirm_summary_df, pd.DataFrame)
            else pd.DataFrame()
        )


def _summarize_transformer_search_errors(
    trials_df: pd.DataFrame,
    *,
    top_n: int = 5,
) -> Dict[str, object]:
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return {
            "failed_trials": 0,
            "unique_error_groups": 0,
            "top_errors": [],
            "sample_failures": [],
        }
    error_df = trials_df.copy()
    if "status" in error_df.columns:
        error_df = error_df[error_df["status"].astype(str) == "error"].copy()
    if error_df.empty:
        return {
            "failed_trials": 0,
            "unique_error_groups": 0,
            "top_errors": [],
            "sample_failures": [],
        }
    error_df["error_type"] = (
        error_df["error_type"]
        if "error_type" in error_df.columns
        else pd.Series([""] * len(error_df), index=error_df.index)
    )
    error_df["error"] = (
        error_df["error"]
        if "error" in error_df.columns
        else pd.Series([""] * len(error_df), index=error_df.index)
    )
    error_df["error_type"] = error_df["error_type"].fillna("").astype(str)
    error_df["error"] = error_df["error"].fillna("").astype(str)
    grouped_df = (
        error_df.groupby(["error_type", "error"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "error_type", "error"], ascending=[False, True, True], ignore_index=True)
    )
    top_errors = [
        {
            "count": int(row["count"]),
            "error_type": str(row["error_type"] or "Error"),
            "error": str(row["error"])[:500],
        }
        for row in grouped_df.head(max(1, int(top_n))).to_dict(orient="records")
    ]
    sample_columns = [
        column
        for column in [
            "trial_index",
            "model_name",
            "num_train_epochs",
            "batch_size",
            "max_length",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "freeze_layers",
            "error_type",
            "error",
            "output_dir",
        ]
        if column in error_df.columns
    ]
    sample_failures = error_df[sample_columns].head(max(1, int(top_n))).to_dict(orient="records")
    return {
        "failed_trials": int(len(error_df)),
        "unique_error_groups": int(len(grouped_df)),
        "top_errors": top_errors,
        "sample_failures": sample_failures,
    }


def _compute_effective_warmup_steps(
    *,
    train_rows: int,
    batch_size: int,
    num_train_epochs: float,
    warmup_steps: int,
    warmup_ratio: Optional[float] = None,
) -> Tuple[int, int]:
    steps_per_epoch = max(1, int(math.ceil(max(1, int(train_rows)) / max(1, int(batch_size)))))
    total_steps = max(1, int(math.ceil(steps_per_epoch * float(num_train_epochs))))
    if warmup_ratio is not None:
        effective = int(round(total_steps * max(0.0, float(warmup_ratio))))
    else:
        effective = int(max(0, int(warmup_steps)))
    if total_steps <= 1:
        return 0, total_steps
    return min(max(0, effective), total_steps - 1), total_steps


def _resolve_transformer_objective(
    metrics: Dict[str, object],
    *,
    objective_metric: str,
) -> float:
    objective_metric = str(objective_metric).strip()
    candidate_keys = [objective_metric]
    if objective_metric.startswith("eval_"):
        candidate_keys.append(objective_metric[5:])
    else:
        candidate_keys.insert(0, f"eval_{objective_metric}")
    for key in candidate_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)
    return float("nan")


def _parse_version_tuple(raw_version: object) -> Tuple[int, int, int]:
    parts = re.findall(r"\d+", str(raw_version or ""))
    values = [int(part) for part in parts[:3]]
    while len(values) < 3:
        values.append(0)
    return tuple(values[:3])


def _torch_requires_safetensors() -> bool:
    if torch is None:
        return False
    return _parse_version_tuple(getattr(torch, "__version__", "")) < (2, 6, 0)


def _transformer_pretrained_load_kwargs(model_name: object) -> Dict[str, object]:
    model_ref = str(model_name or "")
    model_path = Path(model_ref)
    if model_path.exists():
        return {}
    if _torch_requires_safetensors():
        return {"use_safetensors": True}
    return {}


def _raise_transformer_load_error(model_name: object, exc: Exception) -> None:
    message = str(exc)
    model_ref = str(model_name or "")
    if (
        "serious vulnerability issue in torch.load" in message
        or "require users to upgrade torch to at least v2.6" in message
    ):
        raise ValueError(
            "El entorno actual usa torch<2.6 y este modelo base no puede cargarse "
            "con pesos PyTorch inseguros. Actualice torch a >=2.6 o use un modelo "
            f"con safetensors. Modelo: {model_ref}"
        ) from exc
    if "safetensors" in message.lower() and _torch_requires_safetensors():
        raise ValueError(
            "El entorno actual usa torch<2.6, por lo que solo se pueden cargar "
            f"modelos remotos con safetensors. El modelo '{model_ref}' no expone "
            "pesos safetensors compatibles."
        ) from exc
    raise exc


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_hf_model_siblings(model_ref: str) -> Dict[str, object]:
    api_url = f"https://huggingface.co/api/models/{quote(str(model_ref), safe='')}"
    request = Request(
        api_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "SUMO-nlp-severity-app/1.0",
        },
    )
    try:
        with urlopen(request, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return {"ok": False, "error": f"El repositorio '{model_ref}' no existe en Hugging Face."}
        if exc.code in {401, 403}:
            return {"ok": False, "error": f"El repositorio '{model_ref}' requiere autenticacion o esta restringido."}
        return {"ok": False, "error": f"Hugging Face devolvio HTTP {exc.code} para '{model_ref}'."}
    except URLError as exc:
        return {"ok": False, "error": f"No se pudo consultar Hugging Face para '{model_ref}': {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": f"No se pudo inspeccionar '{model_ref}': {exc}"}
    siblings = []
    for item in payload.get("siblings", []) or []:
        if isinstance(item, dict) and item.get("rfilename"):
            siblings.append(str(item["rfilename"]))
    return {"ok": True, "siblings": siblings}


def _transformer_model_validation(model_name: object) -> Dict[str, object]:
    model_ref = str(model_name or "")
    model_path = Path(model_ref)
    requires_safetensors = _torch_requires_safetensors()

    def _has_safetensors(paths: Sequence[str]) -> bool:
        return any(
            path.endswith(".safetensors") or path.endswith(".safetensors.index.json")
            for path in paths
        )

    def _has_pytorch_bin(paths: Sequence[str]) -> bool:
        return any(
            path.endswith("pytorch_model.bin") or path.endswith("pytorch_model.bin.index.json")
            for path in paths
        )

    if model_path.exists():
        local_files = [path.name for path in model_path.glob("*") if path.is_file()]
        has_safetensors = _has_safetensors(local_files)
        has_pytorch_bin = _has_pytorch_bin(local_files)
        if requires_safetensors and not has_safetensors:
            return {
                "status": "incompatible",
                "message": (
                    f"Modelo local incompatible con torch<2.6: '{model_ref}' no contiene "
                    "pesos safetensors."
                ),
                "requires_safetensors": True,
                "has_safetensors": False,
                "has_pytorch_bin": has_pytorch_bin,
                "checked_via": "local_files",
            }
        return {
            "status": "compatible",
            "message": (
                "Modelo local compatible."
                if not requires_safetensors
                else "Modelo local compatible: contiene pesos safetensors."
            ),
            "requires_safetensors": bool(requires_safetensors),
            "has_safetensors": bool(has_safetensors),
            "has_pytorch_bin": bool(has_pytorch_bin),
            "checked_via": "local_files",
        }

    if not requires_safetensors:
        return {
            "status": "compatible",
            "message": "Compatible con el runtime actual.",
            "requires_safetensors": False,
            "has_safetensors": None,
            "has_pytorch_bin": None,
            "checked_via": "runtime",
        }

    remote_info = _fetch_hf_model_siblings(model_ref)
    if not remote_info.get("ok"):
        return {
            "status": "unknown",
            "message": str(remote_info.get("error") or "No se pudo validar el modelo."),
            "requires_safetensors": True,
            "has_safetensors": None,
            "has_pytorch_bin": None,
            "checked_via": "huggingface_api",
        }

    siblings = [str(item) for item in remote_info.get("siblings") or []]
    has_safetensors = _has_safetensors(siblings)
    has_pytorch_bin = _has_pytorch_bin(siblings)
    if has_safetensors:
        return {
            "status": "compatible",
            "message": "Compatible con torch<2.6: el repositorio publica pesos safetensors.",
            "requires_safetensors": True,
            "has_safetensors": True,
            "has_pytorch_bin": bool(has_pytorch_bin),
            "checked_via": "huggingface_api",
        }
    if has_pytorch_bin:
        return {
            "status": "incompatible",
            "message": (
                "Incompatible con torch<2.6: el repositorio solo expone "
                "`pytorch_model.bin`."
            ),
            "requires_safetensors": True,
            "has_safetensors": False,
            "has_pytorch_bin": True,
            "checked_via": "huggingface_api",
        }
    return {
        "status": "unknown",
        "message": "No se detectaron archivos de pesos safetensors ni pytorch_model.bin.",
        "requires_safetensors": True,
        "has_safetensors": False,
        "has_pytorch_bin": False,
        "checked_via": "huggingface_api",
    }


def _transformer_model_status_badge(validation: Dict[str, object]) -> str:
    status = str(validation.get("status") or "unknown")
    if status == "compatible":
        return "compatible"
    if status == "incompatible":
        return "incompatible"
    return "sin_validar"


def _sort_transformer_trials_df(
    df: pd.DataFrame,
    *,
    objective_col: str,
    greater_is_better: bool,
) -> pd.DataFrame:
    if df is None or df.empty or objective_col not in df.columns:
        return pd.DataFrame() if df is None else df
    work = df.copy()
    return work.sort_values(
        by=[objective_col, "trial_index"] if "trial_index" in work.columns else [objective_col],
        ascending=[not greater_is_better, True] if "trial_index" in work.columns else [not greater_is_better],
        na_position="last",
        ignore_index=True,
    )


def _enumerate_transformer_search_configs(
    search_space: Dict[str, Sequence[object]],
) -> List[Dict[str, object]]:
    keys = list(search_space.keys())
    values_by_key = [_dedupe_preserve_order(search_space[key]) for key in keys]
    for key, values in zip(keys, values_by_key):
        if not values:
            raise ValueError(f"La busqueda requiere al menos un valor para '{key}'.")
    return [
        {key: value for key, value in zip(keys, values)}
        for values in itertools.product(*values_by_key)
    ]


def _sample_transformer_search_configs(
    configs: Sequence[Dict[str, object]],
    *,
    max_trials: int,
    random_state: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    config_list = list(configs)
    total_candidates = len(config_list)
    if total_candidates == 0:
        return [], {"sampling_mode": "empty", "total_candidates": 0, "executed_trials": 0}
    trial_budget = max(1, int(max_trials))
    if total_candidates <= trial_budget:
        return config_list, {
            "sampling_mode": "grid",
            "total_candidates": int(total_candidates),
            "executed_trials": int(total_candidates),
        }
    rng = random.Random(int(random_state))
    sampled_indices = rng.sample(range(total_candidates), k=trial_budget)
    sampled = [config_list[idx] for idx in sampled_indices]
    return sampled, {
        "sampling_mode": "random_without_replacement",
        "total_candidates": int(total_candidates),
        "executed_trials": int(len(sampled)),
    }


def _freeze_transformer_base_layers(model: object) -> bool:
    base_prefix = getattr(model, "base_model_prefix", "")
    base_model = getattr(model, base_prefix, None)
    if base_model is None:
        return False

    froze_any = False
    embeddings = getattr(base_model, "embeddings", None)
    if embeddings is not None:
        for param in embeddings.parameters():
            param.requires_grad = False
            froze_any = True

    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is not None:
        half = max(0, len(layers) // 2)
        for layer in layers[:half]:
            for param in layer.parameters():
                param.requires_grad = False
                froze_any = True
    return froze_any


class _TransformerTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: object,
        *,
        max_length: int,
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def run_transformers_finetune(
    df: pd.DataFrame,
    *,
    text_col: str,
    mode: str,
    model_name: str,
    output_dir: Path,
    num_train_epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    warmup_ratio: Optional[float],
    random_state: int,
    split_random_state: Optional[int],
    trainer_random_state: Optional[int],
    freeze_layers: bool,
    test_size: float,
    mlm_probability: float,
) -> Dict[str, object]:
    missing_deps = []
    if torch is None:
        missing_deps.append("torch")
    if AutoTokenizer is None or Trainer is None or TrainingArguments is None:
        missing_deps.append("transformers")
    if missing_deps:
        raise ImportError(
            "Faltan dependencias para finetuning con Transformers: "
            + ", ".join(sorted(set(missing_deps)))
            + ". Instale `transformers` y `accelerate` en el entorno."
        )
    if text_col not in df.columns:
        raise ValueError(f"La columna '{text_col}' no existe.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_output_dir = output_dir / "trainer_output"
    trainer_output_dir.mkdir(parents=True, exist_ok=True)

    work = df[[text_col]].copy()
    if mode == "classification":
        work["severity_target"] = _severity_series(df)
    work[text_col] = work[text_col].fillna("").astype(str).str.strip()
    work = work.loc[work[text_col] != ""].reset_index(drop=True)
    if work.empty:
        raise ValueError("No hay textos validos para entrenar.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    callbacks = []
    data_collator = None
    label_mapping: Dict[str, int] = {}
    pretrained_load_kwargs = _transformer_pretrained_load_kwargs(model_name)

    if mode == "classification":
        work = work.dropna(subset=["severity_target"]).reset_index(drop=True)
        if work.empty:
            raise ValueError("No hay etiquetas de severidad validas para clasificacion.")
        labels_raw = work["severity_target"].astype(int)
        unique_labels = sorted(labels_raw.unique().tolist())
        if len(unique_labels) < 2:
            raise ValueError("La variable objetivo debe tener al menos dos clases.")
        min_class = int(labels_raw.value_counts().min())
        if min_class < 2:
            raise ValueError("Cada clase necesita al menos 2 filas para entrenar y validar.")
        label_mapping = {str(label): idx for idx, label in enumerate(unique_labels)}
        labels = labels_raw.map(lambda value: label_mapping[str(int(value))]).astype(int)
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            work[text_col].tolist(),
            labels.tolist(),
            test_size=float(test_size),
            stratify=labels.tolist(),
            random_state=int(split_random_state if split_random_state is not None else random_state),
        )
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(unique_labels),
                id2label={idx: label for label, idx in label_mapping.items()},
                label2id=label_mapping,
                **pretrained_load_kwargs,
            )
        except Exception as exc:
            _raise_transformer_load_error(model_name, exc)
        train_dataset = _TransformerTextDataset(
            train_texts,
            tokenizer,
            max_length=int(max_length),
            labels=train_labels,
        )
        eval_dataset = _TransformerTextDataset(
            eval_texts,
            tokenizer,
            max_length=int(max_length),
            labels=eval_labels,
        )

        def _trainer_metrics(eval_pred: object) -> Dict[str, float]:
            logits, labels_true = eval_pred
            y_true = np.asarray(labels_true, dtype=int)
            y_pred = np.asarray(logits).argmax(axis=1)
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
            if len(label_mapping) == 2:
                probs = _softmax_logits(np.asarray(logits))[:, 1]
                metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
                metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
                except ValueError:
                    metrics["roc_auc"] = float("nan")
            else:
                metrics["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
                metrics["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            return metrics

        compute_metrics = _trainer_metrics
    else:
        texts = work[text_col].tolist()
        if len(texts) < 2:
            raise ValueError("Se requieren al menos 2 textos para MLM.")
        train_texts, eval_texts = train_test_split(
            texts,
            test_size=float(test_size),
            random_state=int(split_random_state if split_random_state is not None else random_state),
        )
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_name, **pretrained_load_kwargs)
        except Exception as exc:
            _raise_transformer_load_error(model_name, exc)
        train_dataset = _TransformerTextDataset(
            train_texts,
            tokenizer,
            max_length=int(max_length),
        )
        eval_dataset = _TransformerTextDataset(
            eval_texts,
            tokenizer,
            max_length=int(max_length),
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=float(mlm_probability),
        )
        compute_metrics = None

    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    froze_layers = bool(freeze_layers and _freeze_transformer_base_layers(model))
    if EarlyStoppingCallback is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    logging_steps = max(1, int(math.ceil(len(train_dataset) / max(1, int(batch_size)))))
    effective_warmup_steps, total_train_steps = _compute_effective_warmup_steps(
        train_rows=len(train_dataset),
        batch_size=int(batch_size),
        num_train_epochs=float(num_train_epochs),
        warmup_steps=int(warmup_steps),
        warmup_ratio=warmup_ratio,
    )
    metric_for_best = "f1" if mode == "classification" else "eval_loss"
    greater_is_better = mode == "classification"
    training_args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        num_train_epochs=float(num_train_epochs),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        warmup_steps=int(effective_warmup_steps),
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=int(logging_steps),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        seed=int(trainer_random_state if trainer_random_state is not None else random_state),
        report_to="none",
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    train_output = trainer.train()
    eval_metrics = trainer.evaluate()
    history_df = pd.DataFrame(trainer.state.log_history)

    if mode == "classification":
        predictions = trainer.predict(eval_dataset)
        logits = np.asarray(predictions.predictions)
        y_true = np.asarray(predictions.label_ids, dtype=int)
        y_pred = logits.argmax(axis=1)
        y_score = _softmax_logits(logits)[:, 1] if len(label_mapping) == 2 else None
        metrics = _classification_metrics(y_true, y_pred, y_score)
    else:
        metrics = {}
        eval_loss = eval_metrics.get("eval_loss")
        if eval_loss is not None:
            metrics["eval_loss"] = float(eval_loss)
            if float(eval_loss) < 20:
                metrics["perplexity"] = float(math.exp(float(eval_loss)))

    for key, value in (train_output.metrics or {}).items():
        if isinstance(value, (int, float)) and key not in metrics:
            metrics[key] = float(value)
    for key, value in (eval_metrics or {}).items():
        if isinstance(value, (int, float)) and key not in metrics:
            metrics[key] = float(value)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "model_name": model_name,
        "mode": mode,
        "text_col": text_col,
        "output_dir": str(output_dir),
        "rows_train": int(len(train_dataset)),
        "rows_eval": int(len(eval_dataset)),
        "metrics": metrics,
        "params": {
            "epochs": int(num_train_epochs),
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "warmup_steps": int(effective_warmup_steps),
            "warmup_ratio": None if warmup_ratio is None else float(warmup_ratio),
            "total_train_steps": int(total_train_steps),
            "test_size": float(test_size),
            "random_state": int(random_state),
            "split_random_state": int(split_random_state if split_random_state is not None else random_state),
            "trainer_random_state": int(trainer_random_state if trainer_random_state is not None else random_state),
            "freeze_layers": bool(froze_layers),
            "mlm_probability": float(mlm_probability),
        },
        "label_mapping": label_mapping,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=True, indent=2, default=_json_default)

    return {
        **summary,
        "history_df": history_df,
    }


def run_transformers_hyperparameter_search(
    df: pd.DataFrame,
    *,
    text_col: str,
    mode: str,
    output_dir: Path,
    search_space: Dict[str, Sequence[object]],
    max_trials: int,
    objective_metric: str,
    split_random_state: int,
    trainer_seed_base: int,
    confirm_top_k: int,
    confirm_seed_count: int,
    keep_trial_artifacts: bool,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    if mode not in {"classification", "mlm"}:
        raise ValueError(f"Modo de busqueda no soportado: {mode}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_trials_dir = output_dir / "search_trials"
    confirm_trials_dir = output_dir / "confirmation_trials"
    final_model_dir = output_dir / "best_model"
    search_trials_dir.mkdir(parents=True, exist_ok=True)
    confirm_trials_dir.mkdir(parents=True, exist_ok=True)

    all_configs = _enumerate_transformer_search_configs(search_space)
    trial_configs, sampling_meta = _sample_transformer_search_configs(
        all_configs,
        max_trials=int(max_trials),
        random_state=int(split_random_state),
    )
    if not trial_configs:
        raise ValueError("La busqueda no tiene configuraciones para evaluar.")

    objective_metric = str(objective_metric).strip()
    greater_is_better = objective_metric not in {"eval_loss", "loss", "perplexity"}
    _emit_progress(
        progress_callback,
        1,
        f"Preparando {len(trial_configs):,} trials sobre {sampling_meta.get('total_candidates', 0):,} configuraciones.",
    )

    config_lookup: Dict[str, Dict[str, object]] = {}
    trial_rows: List[Dict[str, object]] = []
    trial_output_dirs: List[Path] = []

    for trial_index, config in enumerate(trial_configs, start=1):
        config_payload = json.dumps(config, sort_keys=True, default=_json_default)
        config_id = uuid.uuid5(uuid.NAMESPACE_DNS, config_payload).hex[:12]
        config_lookup[config_id] = dict(config)
        trial_dir = search_trials_dir / f"trial_{trial_index:03d}_{config_id}"
        trial_output_dirs.append(trial_dir)
        progress_start = 5 + int((trial_index - 1) * 55 / max(1, len(trial_configs)))
        _emit_progress(
            progress_callback,
            progress_start,
            f"Trial {trial_index}/{len(trial_configs)}: {config.get('model_name')} | lr={config.get('learning_rate')}.",
        )
        try:
            result = run_transformers_finetune(
                df,
                text_col=text_col,
                mode=mode,
                model_name=str(config["model_name"]),
                output_dir=trial_dir,
                num_train_epochs=int(config["num_train_epochs"]),
                batch_size=int(config["batch_size"]),
                max_length=int(config["max_length"]),
                learning_rate=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
                warmup_steps=0,
                warmup_ratio=float(config["warmup_ratio"]),
                random_state=int(trainer_seed_base),
                split_random_state=int(split_random_state),
                trainer_random_state=int(trainer_seed_base),
                freeze_layers=bool(config["freeze_layers"]),
                test_size=float(config["test_size"]),
                mlm_probability=float(config["mlm_probability"]),
            )
            objective_value = _resolve_transformer_objective(
                result["metrics"],
                objective_metric=objective_metric,
            )
            row = {
                "phase": "search",
                "trial_index": int(trial_index),
                "config_id": config_id,
                "status": "ok",
                "objective_metric": objective_metric,
                "objective": objective_value,
                "model_name": str(config["model_name"]),
                "num_train_epochs": int(config["num_train_epochs"]),
                "batch_size": int(config["batch_size"]),
                "max_length": int(config["max_length"]),
                "learning_rate": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
                "warmup_ratio": float(config["warmup_ratio"]),
                "freeze_layers": bool(config["freeze_layers"]),
                "test_size": float(config["test_size"]),
                "mlm_probability": float(config["mlm_probability"]),
                "rows_train": int(result["rows_train"]),
                "rows_eval": int(result["rows_eval"]),
                "output_dir": str(result["output_dir"]),
            }
            row.update(_flatten_scalar_payload(result["metrics"], prefix="metric_"))
            row.update(_flatten_scalar_payload(result["params"], prefix="param_"))
        except Exception as exc:
            row = {
                "phase": "search",
                "trial_index": int(trial_index),
                "config_id": config_id,
                "status": "error",
                "objective_metric": objective_metric,
                "objective": float("nan"),
                "model_name": str(config.get("model_name")),
                "num_train_epochs": int(config.get("num_train_epochs") or 0),
                "batch_size": int(config.get("batch_size") or 0),
                "max_length": int(config.get("max_length") or 0),
                "learning_rate": float(config.get("learning_rate") or 0.0),
                "weight_decay": float(config.get("weight_decay") or 0.0),
                "warmup_ratio": float(config.get("warmup_ratio") or 0.0),
                "freeze_layers": bool(config.get("freeze_layers")),
                "test_size": float(config.get("test_size") or 0.0),
                "mlm_probability": float(config.get("mlm_probability") or 0.0),
                "rows_train": 0,
                "rows_eval": 0,
                "output_dir": str(trial_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "error_traceback": traceback.format_exc(limit=10),
            }
        trial_rows.append(row)

    trials_df = pd.DataFrame(trial_rows)
    successful_trials = trials_df[trials_df["status"] == "ok"].copy()
    if successful_trials.empty:
        failure_summary = _summarize_transformer_search_errors(trials_df)
        failed_trials_csv = output_dir / "search_trials_failed.csv"
        failure_summary_json = output_dir / "search_failure_summary.json"
        trials_df.to_csv(failed_trials_csv, index=False)
        failure_summary = {
            **failure_summary,
            "search_root": str(output_dir),
            "failed_trials_csv": str(failed_trials_csv),
            "failure_summary_json": str(failure_summary_json),
        }
        with failure_summary_json.open("w", encoding="utf-8") as fh:
            json.dump(failure_summary, fh, ensure_ascii=True, indent=2, default=_json_default)
        search_result = {
            "status": "failed",
            "search_summary": {
                "mode": mode,
                "text_col": text_col,
                "objective_metric": objective_metric,
                "greater_is_better": bool(greater_is_better),
                "sampling_meta": sampling_meta,
                "split_random_state": int(split_random_state),
                "trainer_seed_base": int(trainer_seed_base),
                "confirm_top_k": int(confirm_top_k),
                "confirm_seed_count": int(confirm_seed_count),
                "keep_trial_artifacts": bool(keep_trial_artifacts),
            },
            "best_result": {},
            "failure_summary": failure_summary,
        }
        top_errors = failure_summary.get("top_errors") or []
        error_preview = " | ".join(
            f"{item.get('error_type')}: {str(item.get('error') or '')[:160]} (x{int(item.get('count') or 0)})"
            for item in top_errors[:3]
        )
        message = "Ningun trial finalizo correctamente en la busqueda de Transformers."
        if error_preview:
            message += f" Top errores: {error_preview}"
        raise TransformerSearchDebugError(
            message,
            search_result=search_result,
            trials_df=trials_df,
        )
    ranked_trials_df = _sort_transformer_trials_df(
        successful_trials,
        objective_col="objective",
        greater_is_better=greater_is_better,
    )

    confirm_seed_values = [
        int(trainer_seed_base + offset) for offset in range(max(1, int(confirm_seed_count)))
    ]
    selected_config_ids = ranked_trials_df["config_id"].drop_duplicates().head(max(1, int(confirm_top_k))).tolist()
    confirm_rows: List[Dict[str, object]] = []
    confirm_output_dirs: List[Path] = []
    _emit_progress(
        progress_callback,
        62,
        f"Confirmando top-{len(selected_config_ids)} con {len(confirm_seed_values)} seeds.",
    )

    for config_rank, config_id in enumerate(selected_config_ids, start=1):
        config = config_lookup[config_id]
        search_row = ranked_trials_df.loc[ranked_trials_df["config_id"] == config_id].iloc[0]
        for seed_idx, trainer_seed in enumerate(confirm_seed_values, start=1):
            confirm_index = (config_rank - 1) * len(confirm_seed_values) + seed_idx
            confirm_total = max(1, len(selected_config_ids) * len(confirm_seed_values))
            progress_value = 62 + int(confirm_index * 23 / confirm_total)
            _emit_progress(
                progress_callback,
                progress_value,
                f"Confirmacion {confirm_index}/{confirm_total}: cfg {config_rank} seed {trainer_seed}.",
            )
            confirm_dir = confirm_trials_dir / f"cfg_{config_rank:02d}_{config_id}" / f"seed_{trainer_seed}"
            confirm_output_dirs.append(confirm_dir)
            try:
                result = run_transformers_finetune(
                    df,
                    text_col=text_col,
                    mode=mode,
                    model_name=str(config["model_name"]),
                    output_dir=confirm_dir,
                    num_train_epochs=int(config["num_train_epochs"]),
                    batch_size=int(config["batch_size"]),
                    max_length=int(config["max_length"]),
                    learning_rate=float(config["learning_rate"]),
                    weight_decay=float(config["weight_decay"]),
                    warmup_steps=0,
                    warmup_ratio=float(config["warmup_ratio"]),
                    random_state=int(trainer_seed),
                    split_random_state=int(split_random_state),
                    trainer_random_state=int(trainer_seed),
                    freeze_layers=bool(config["freeze_layers"]),
                    test_size=float(config["test_size"]),
                    mlm_probability=float(config["mlm_probability"]),
                )
                objective_value = _resolve_transformer_objective(
                    result["metrics"],
                    objective_metric=objective_metric,
                )
                row = {
                    "phase": "confirmation",
                    "config_rank": int(config_rank),
                    "config_id": config_id,
                    "status": "ok",
                    "trainer_seed": int(trainer_seed),
                    "objective_metric": objective_metric,
                    "objective": objective_value,
                    "search_objective": float(search_row["objective"]),
                    "model_name": str(config["model_name"]),
                    "num_train_epochs": int(config["num_train_epochs"]),
                    "batch_size": int(config["batch_size"]),
                    "max_length": int(config["max_length"]),
                    "learning_rate": float(config["learning_rate"]),
                    "weight_decay": float(config["weight_decay"]),
                    "warmup_ratio": float(config["warmup_ratio"]),
                    "freeze_layers": bool(config["freeze_layers"]),
                    "test_size": float(config["test_size"]),
                    "mlm_probability": float(config["mlm_probability"]),
                    "rows_train": int(result["rows_train"]),
                    "rows_eval": int(result["rows_eval"]),
                    "output_dir": str(result["output_dir"]),
                }
                row.update(_flatten_scalar_payload(result["metrics"], prefix="metric_"))
            except Exception as exc:
                row = {
                    "phase": "confirmation",
                    "config_rank": int(config_rank),
                    "config_id": config_id,
                    "status": "error",
                    "trainer_seed": int(trainer_seed),
                    "objective_metric": objective_metric,
                    "objective": float("nan"),
                    "search_objective": float(search_row["objective"]),
                    "model_name": str(config["model_name"]),
                    "num_train_epochs": int(config["num_train_epochs"]),
                    "batch_size": int(config["batch_size"]),
                    "max_length": int(config["max_length"]),
                    "learning_rate": float(config["learning_rate"]),
                    "weight_decay": float(config["weight_decay"]),
                    "warmup_ratio": float(config["warmup_ratio"]),
                    "freeze_layers": bool(config["freeze_layers"]),
                    "test_size": float(config["test_size"]),
                    "mlm_probability": float(config["mlm_probability"]),
                    "rows_train": 0,
                    "rows_eval": 0,
                    "output_dir": str(confirm_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_traceback": traceback.format_exc(limit=10),
                }
            confirm_rows.append(row)

    confirm_df = pd.DataFrame(confirm_rows)
    confirm_ok_df = confirm_df[confirm_df["status"] == "ok"].copy()
    if confirm_ok_df.empty:
        best_config_id = str(ranked_trials_df.iloc[0]["config_id"])
        confirm_summary_df = pd.DataFrame()
    else:
        confirm_summary_df = (
            confirm_ok_df.groupby("config_id", dropna=False)
            .agg(
                model_name=("model_name", "first"),
                num_train_epochs=("num_train_epochs", "first"),
                batch_size=("batch_size", "first"),
                max_length=("max_length", "first"),
                learning_rate=("learning_rate", "first"),
                weight_decay=("weight_decay", "first"),
                warmup_ratio=("warmup_ratio", "first"),
                freeze_layers=("freeze_layers", "first"),
                test_size=("test_size", "first"),
                mlm_probability=("mlm_probability", "first"),
                search_objective=("search_objective", "first"),
                confirm_runs=("objective", "count"),
                confirm_objective_mean=("objective", "mean"),
                confirm_objective_std=("objective", lambda values: float(np.std(values, ddof=0))),
                confirm_objective_min=("objective", "min"),
                confirm_objective_max=("objective", "max"),
            )
            .reset_index()
        )
        confirm_summary_df = confirm_summary_df.sort_values(
            by=[
                "confirm_objective_mean",
                "confirm_objective_std",
                "search_objective",
            ],
            ascending=[not greater_is_better, True, not greater_is_better],
            na_position="last",
            ignore_index=True,
        )
        best_config_id = str(confirm_summary_df.iloc[0]["config_id"])

    best_config = config_lookup[best_config_id]
    _emit_progress(progress_callback, 90, "Entrenando configuracion final seleccionada.")
    best_result = run_transformers_finetune(
        df,
        text_col=text_col,
        mode=mode,
        model_name=str(best_config["model_name"]),
        output_dir=final_model_dir,
        num_train_epochs=int(best_config["num_train_epochs"]),
        batch_size=int(best_config["batch_size"]),
        max_length=int(best_config["max_length"]),
        learning_rate=float(best_config["learning_rate"]),
        weight_decay=float(best_config["weight_decay"]),
        warmup_steps=0,
        warmup_ratio=float(best_config["warmup_ratio"]),
        random_state=int(trainer_seed_base),
        split_random_state=int(split_random_state),
        trainer_random_state=int(trainer_seed_base),
        freeze_layers=bool(best_config["freeze_layers"]),
        test_size=float(best_config["test_size"]),
        mlm_probability=float(best_config["mlm_probability"]),
    )
    best_history_df = best_result.pop("history_df", pd.DataFrame())

    trials_csv = output_dir / "search_trials.csv"
    confirm_csv = output_dir / "confirmation_trials.csv"
    confirm_summary_csv = output_dir / "confirmation_summary.csv"
    ranked_trials_df.to_csv(trials_csv, index=False)
    confirm_df.to_csv(confirm_csv, index=False)
    if not confirm_summary_df.empty:
        confirm_summary_df.to_csv(confirm_summary_csv, index=False)

    search_summary = {
        "mode": mode,
        "text_col": text_col,
        "objective_metric": objective_metric,
        "greater_is_better": bool(greater_is_better),
        "sampling_meta": sampling_meta,
        "split_random_state": int(split_random_state),
        "trainer_seed_base": int(trainer_seed_base),
        "confirm_top_k": int(confirm_top_k),
        "confirm_seed_count": int(len(confirm_seed_values)),
        "keep_trial_artifacts": bool(keep_trial_artifacts),
        "best_config_id": best_config_id,
        "best_model_name": best_result.get("model_name"),
        "best_model_output_dir": best_result["output_dir"],
        "best_metrics": best_result["metrics"],
        "best_params": best_result["params"],
        "search_trials_csv": str(trials_csv),
        "confirmation_trials_csv": str(confirm_csv),
        "confirmation_summary_csv": str(confirm_summary_csv) if confirm_summary_df is not None and not confirm_summary_df.empty else None,
    }
    with (output_dir / "search_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(search_summary, fh, ensure_ascii=True, indent=2, default=_json_default)

    if not keep_trial_artifacts:
        for path in trial_output_dirs + confirm_output_dirs:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    _emit_progress(progress_callback, 100, "Busqueda de hiperparametros completada.")
    return {
        "search_summary": search_summary,
        "trials_df": ranked_trials_df,
        "confirm_df": confirm_df,
        "confirm_summary_df": confirm_summary_df,
        "best_result": best_result,
        "best_history_df": best_history_df,
    }


def execute_transformer_finetune_from_preset(
    df: pd.DataFrame,
    *,
    preset: Dict[str, object],
    output_dir: Path,
    run_id: str,
    result_model_name: str,
    action_name: str,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    params = preset.get("params") or {}
    text_col = str(preset.get("text_col") or "")
    mode = str(preset.get("mode") or "")
    base_model = str(preset.get("base_model") or "")
    output_dir = Path(output_dir)
    result = run_transformers_finetune(
        df,
        text_col=text_col,
        mode=mode,
        model_name=base_model,
        output_dir=output_dir,
        num_train_epochs=int(params.get("epochs") or 3),
        batch_size=int(params.get("batch_size") or 8),
        max_length=int(params.get("max_length") or 128),
        learning_rate=float(params.get("learning_rate") or 5e-5),
        weight_decay=float(params.get("weight_decay") or 0.01),
        warmup_steps=int(params.get("warmup_steps") or 0),
        warmup_ratio=(
            float(params.get("warmup_ratio"))
            if params.get("warmup_ratio") is not None
            else None
        ),
        random_state=int(
            params.get("trainer_random_state")
            or params.get("random_state")
            or 42
        ),
        split_random_state=int(
            params.get("split_random_state")
            or params.get("random_state")
            or 42
        ),
        trainer_random_state=int(
            params.get("trainer_random_state")
            or params.get("random_state")
            or 42
        ),
        freeze_layers=bool(params.get("freeze_layers") or False),
        test_size=float(params.get("test_size") or 0.2),
        mlm_probability=float(params.get("mlm_probability") or 0.15),
    )
    history_df = result.pop("history_df", pd.DataFrame())
    metadata = {
        "text_col": text_col,
        "mode": mode,
        "base_model": base_model,
        "output_dir": result["output_dir"],
        "rows_train": result["rows_train"],
        "rows_eval": result["rows_eval"],
        "source_preset_run_id": preset.get("run_id"),
        "source_preset_created_at": preset.get("created_at"),
        "source_preset_label": preset.get("label"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    _record_model_result(
        run_id=run_id,
        stage="language_modeling",
        model_name=result_model_name,
        feature_group=f"{text_col} [{mode}]",
        metrics=result["metrics"],
        params=result["params"],
        metadata=metadata,
    )
    if isinstance(history_df, pd.DataFrame) and not history_df.empty:
        _persist_artifact(
            history_df,
            stage="language_modeling",
            artifact_name="transformers_finetune_history",
            run_id=run_id,
            metadata={
                "text_col": text_col,
                "mode": mode,
                "base_model": base_model,
                "output_dir": result["output_dir"],
                "source_preset_run_id": preset.get("run_id"),
            },
        )
    _log_action(
        "language_modeling",
        action_name,
        {
            "text_col": text_col,
            "mode": mode,
            "base_model": base_model,
            "output_dir": result["output_dir"],
            "rows_train": result["rows_train"],
            "rows_eval": result["rows_eval"],
            "metrics": result["metrics"],
            "params": result["params"],
            "source_preset_run_id": preset.get("run_id"),
            "source_preset_label": preset.get("label"),
        },
        run_id=run_id,
    )
    return {
        **result,
        "history_df": history_df,
    }


def train_rf_xgb_holdout(
    df: pd.DataFrame,
    *,
    feature_group: str,
    test_size: float,
    random_state: int,
    split_mode: str,
    top_k: int,
    tune_hyperparameters: bool,
    tuning_folds: int,
    tuning_profile: str,
) -> Dict[str, object]:
    feature_cols = _resolve_feature_group(df, feature_group)
    if not feature_cols:
        raise ValueError("No hay variables disponibles para entrenar.")
    X_train, X_test, y_train, y_test, split_meta = _prepare_holdout_split(
        df,
        feature_cols,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    X_train_imp, X_test_imp, _ = _fit_imputer(X_train, X_test)
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_imp,
        y_train,
        random_state=int(random_state),
    )
    ranking_df = _rf_rank_features(
        X_train_bal,
        y_train_bal,
        random_state=int(random_state),
    )
    selected_cols = ranking_df["variable"].head(max(1, int(top_k))).tolist()
    if not selected_cols:
        raise ValueError("No se pudieron seleccionar variables con Random Forest.")

    xgb_model, best_params, best_score, search_df = _optimize_xgb_classifier(
        X_train_bal[selected_cols],
        y_train_bal,
        random_state=int(random_state),
        tune_hyperparameters=bool(tune_hyperparameters),
        tuning_folds=int(tuning_folds),
        tuning_profile=str(tuning_profile),
    )
    xgb_pred = xgb_model.predict(X_test_imp[selected_cols])
    xgb_score = xgb_model.predict_proba(X_test_imp[selected_cols])[:, 1]
    xgb_metrics = _classification_metrics(y_test, xgb_pred, xgb_score)

    return {
        "feature_group": feature_group,
        "selected_cols": selected_cols,
        "ranking_df": ranking_df,
        "balancing_meta": balancing_meta,
        "split_meta": split_meta,
        "xgb_search_df": search_df,
        "xgb_best_score": best_score,
        "results": [
            {
                "model_name": "XGBoost",
                "metrics": xgb_metrics,
                "params": best_params,
            },
        ],
    }


def train_rf_xgb_cv(
    df: pd.DataFrame,
    *,
    feature_group: str,
    random_state: int,
    folds: int,
) -> pd.DataFrame:
    feature_cols = _resolve_feature_group(df, feature_group)
    if not feature_cols:
        raise ValueError("No hay columnas numericas para CV.")
    work = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = _severity_series(df).dropna()
    valid_mask = _severity_series(df).notna()
    work = work.loc[valid_mask].reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)
    if y.nunique() < 2:
        raise ValueError("La variable objetivo debe ser binaria.")
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(work), columns=feature_cols)
    min_class = int(y.value_counts().min())
    n_splits = min(max(2, int(folds)), min_class)
    if n_splits < 2:
        raise ValueError("No hay suficientes ejemplos por clase para validacion cruzada.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    configs = [
        ("Random Forest", build_model("Random Forest", {"n_estimators": 400, "max_depth": None}, random_state)),
        (
            "XGBoost",
            build_model(
                "XGBoost",
                {
                    "n_estimators": 300,
                    "max_depth": 5,
                    "learning_rate": 0.08,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                },
                random_state,
            ),
        ),
    ]
    rows: List[Dict[str, object]] = []
    for model_name, model in configs:
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            "model_name": model_name,
            "feature_group": feature_group,
            "folds": int(n_splits),
        }
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows)


def train_elastic_net_holdout(
    df: pd.DataFrame,
    *,
    feature_group: str,
    test_size: float,
    random_state: int,
    split_mode: str,
) -> Dict[str, object]:
    feature_cols = _resolve_feature_group(df, feature_group)
    X_train, X_test, y_train, y_test, split_meta = _prepare_holdout_split(
        df,
        feature_cols,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                _build_elastic_net_logistic_regression(
                    random_state=int(random_state),
                    max_iter=8000,
                ),
            ),
        ]
    )
    min_class = int(y_train.value_counts().min())
    if min_class >= 2:
        search = GridSearchCV(
            pipe,
            param_grid={
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            scoring="f1",
            cv=min(3, min_class),
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        best_model = pipe.fit(X_train, y_train)
        best_params = {"fallback": "sin_grid_search_por_clase_minoritaria"}
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]
    metrics = _classification_metrics(y_test, y_pred, y_score)
    return {
        "model_name": "Elastic Net",
        "metrics": metrics,
        "best_params": best_params,
        "split_meta": split_meta,
    }


def train_svm_rfe_holdout(
    df: pd.DataFrame,
    *,
    feature_group: str,
    test_size: float,
    random_state: int,
    split_mode: str,
    k_features: int,
) -> Dict[str, object]:
    feature_cols = _resolve_feature_group(df, feature_group)
    X_train, X_test, y_train, y_test, split_meta = _prepare_holdout_split(
        df,
        feature_cols,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
    )
    X_train_imp, X_test_imp, _ = _fit_imputer(X_train, X_test)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_cols)

    selector = RFE(
        estimator=LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            random_state=int(random_state),
            max_iter=6000,
        ),
        n_features_to_select=max(1, min(int(k_features), len(feature_cols))),
        step=max(1, len(feature_cols) // 10),
    )
    selector.fit(X_train_scaled, y_train)
    selected_cols = X_train_scaled.columns[selector.support_].tolist()

    model = SVC(
        C=1.0,
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=int(random_state),
    )
    model.fit(X_train_scaled[selected_cols], y_train)
    y_pred = model.predict(X_test_scaled[selected_cols])
    y_score = model.predict_proba(X_test_scaled[selected_cols])[:, 1]
    metrics = _classification_metrics(y_test, y_pred, y_score)
    ranking_df = pd.DataFrame(
        {
            "variable": feature_cols,
            "ranking_rfe": selector.ranking_,
        }
    ).sort_values(["ranking_rfe", "variable"], ignore_index=True)
    return {
        "model_name": "SVM + RFE",
        "metrics": metrics,
        "selected_cols": selected_cols,
        "ranking_df": ranking_df,
        "split_meta": split_meta,
    }


def _resolve_feature_group(df: pd.DataFrame, feature_group: str) -> List[str]:
    flow_cols = _flow_feature_columns(df)
    emb_cols = _embedding_feature_columns(df)
    if feature_group == "Solo embeddings":
        return emb_cols
    if feature_group == "Todo":
        return flow_cols + emb_cols
    return flow_cols


def evaluate_predictive_variables(df: pd.DataFrame, *, feature_group: str) -> pd.DataFrame:
    feature_cols = _resolve_feature_group(df, feature_group)
    if not feature_cols:
        return pd.DataFrame()
    ranking = _compute_relevant_feature_ranking(
        df[feature_cols + ["severity_target"]],
        top_k=len(feature_cols),
        candidate_cols=feature_cols,
    )
    ranking["feature_group"] = feature_group
    return ranking


def build_granular_visualization_df(granular_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    if granular_df is None or granular_df.empty or features_df is None or features_df.empty:
        return pd.DataFrame()
    merged = granular_df.merge(
        features_df[["accident_id", "severity_target"]],
        on="accident_id",
        how="left",
    )
    merged["severity_target"] = pd.to_numeric(merged["severity_target"], errors="coerce")
    window_size = pd.to_numeric(merged.get("window_size_minutes"), errors="coerce").fillna(1).astype(int)
    merged["time_offset"] = np.where(
        merged["direction"].astype(str).eq("before"),
        -merged["minute_idx"].astype(int) * window_size,
        merged["minute_idx"].astype(int) * window_size,
    )
    return merged


def run_topic_analysis(
    df: pd.DataFrame,
    *,
    text_col: str,
    n_topics: int,
    top_terms: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if text_col not in df.columns:
        raise ValueError(f"La columna '{text_col}' no existe.")
    work = df[[text_col, "severity_target"]].copy()
    work[text_col] = work[text_col].fillna("").astype(str).str.strip()
    work["severity_target"] = _severity_series(work)
    work = work.loc[work[text_col] != ""].dropna(subset=["severity_target"]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No hay textos validos para analisis de topicos.")
    texts = work[text_col].tolist()

    if BERTopic is not None:
        topic_model = BERTopic(language="spanish", min_topic_size=max(2, int(len(texts) / max(2, n_topics))), verbose=False)
        topics, _ = topic_model.fit_transform(texts)
        label_map: Dict[int, str] = {}
        for topic_id in sorted(set(topics)):
            if topic_id == -1:
                label_map[topic_id] = "Ruido"
                continue
            topic_terms_info = topic_model.get_topic(topic_id) or []
            terms = [term for term, _ in topic_terms_info[: max(1, int(top_terms))]]
            label_map[topic_id] = ", ".join(terms)
        work["topic_id"] = topics
        work["topic_label"] = work["topic_id"].map(label_map)
        meta = {"method": "BERTopic", "text_col": text_col, "n_topics_detected": int(work["topic_id"].nunique())}
        return work, meta

    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(texts)
    max_topics = min(tfidf.shape[0], tfidf.shape[1])
    if max_topics < 2:
        raise ValueError("No hay suficientes terminos para aproximar topicos.")
    n_topics_final = min(max(2, int(n_topics)), max_topics - 1 if max_topics > 2 else 2)
    nmf = NMF(n_components=n_topics_final, init="nndsvda", random_state=random_state, max_iter=500)
    topic_weights = nmf.fit_transform(tfidf)
    topic_ids = topic_weights.argmax(axis=1)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    topic_labels: Dict[int, str] = {}
    for topic_id, component in enumerate(nmf.components_):
        terms = vocab[np.argsort(component)[::-1][: max(1, int(top_terms))]]
        topic_labels[topic_id] = ", ".join(terms.tolist())
    work["topic_id"] = topic_ids
    work["topic_label"] = work["topic_id"].map(topic_labels)
    meta = {"method": "NMF fallback", "text_col": text_col, "n_topics_detected": int(work["topic_id"].nunique())}
    return work, meta


def _render_metric_cards(df: pd.DataFrame, severity_col: str = "severity_target") -> None:
    total = len(df)
    target = pd.to_numeric(df.get(severity_col), errors="coerce")
    valid = target.notna().sum()
    severe = int(target.fillna(0).sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{total:,}")
    col2.metric("Severidad valida", f"{valid:,}")
    col3.metric("Casos severos", f"{severe:,}")


def _render_registry_caption() -> None:
    st.caption(f"Registro reproducible: {REGISTRY_DB}")


def _reset_nlp_sev_language_state() -> None:
    st.session_state["nlp_sev_language_df"] = None
    st.session_state["nlp_sev_language_artifact"] = None
    st.session_state["nlp_sev_embeddings_df"] = None
    st.session_state["nlp_sev_embedding_cols"] = []
    st.session_state["nlp_sev_embeddings_artifact"] = None
    st.session_state["nlp_sev_embedding_meta"] = None
    st.session_state["nlp_sev_embedding_rf_df"] = None
    st.session_state["nlp_sev_selected_embedding_cols"] = []
    st.session_state["nlp_sev_transformer_search_trials_df"] = None
    st.session_state["nlp_sev_transformer_search_confirm_df"] = None
    st.session_state["nlp_sev_transformer_search_summary_df"] = None
    st.session_state["nlp_sev_transformer_search_result"] = None
    st.session_state["nlp_sev_transformer_active_preset"] = None
    st.session_state["nlp_sev_topic_df"] = None
    st.session_state["nlp_sev_topic_meta"] = None


def _render_feature_engineering_output() -> None:
    features_df = st.session_state.get("nlp_sev_features_df")
    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        return

    _render_metric_cards(features_df)

    artifact = st.session_state.get("nlp_sev_features_artifact") or {}
    if artifact:
        st.caption(f"Features DuckDB: {artifact.get('db_path')} :: {artifact.get('table_name')}")
        metadata = artifact.get("metadata") or {}
        excluded_no_flow = int(metadata.get("excluded_without_flow_coverage") or 0)
        covered_events = metadata.get("covered_events")
        if covered_events is not None or excluded_no_flow > 0:
            st.caption(
                f"Accidentes con cobertura de flujo: {int(covered_events or len(features_df)):,} | "
                f"Excluidos sin cobertura: {excluded_no_flow:,}"
            )

    granular_artifact = st.session_state.get("nlp_sev_granular_artifact") or {}
    if granular_artifact:
        st.caption(
            f"Granular DuckDB: {granular_artifact.get('db_path')} :: {granular_artifact.get('table_name')}"
        )
    feature_cols = _flow_feature_columns(features_df)
    preview_base_cols = [
        col
        for col in [
            "accident_id",
            "accidente_time",
            "km",
            "eje",
            "calzada",
            "subtipo",
            "severidad",
            "ultimo_portico",
            "proximo_portico",
            "source_files",
        ]
        if col in features_df.columns
    ]
    if feature_cols:
        numeric_features = features_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        nonzero_counts = (numeric_features != 0).sum(axis=0).sort_values(ascending=False)
        signal_df = pd.DataFrame(
            {
                "variable": nonzero_counts.index,
                "filas_no_cero": nonzero_counts.values.astype(int),
            }
        )
        signal_df["cobertura_pct"] = (
            signal_df["filas_no_cero"] / max(1, len(features_df)) * 100.0
        ).round(2)
        row_signal = numeric_features.abs().sum(axis=1)
        signal_cols = [
            col
            for col in signal_df.loc[signal_df["filas_no_cero"] > 0, "variable"].tolist()
            if col not in preview_base_cols
        ][:8]

        s1, s2, s3 = st.columns(3)
        s1.metric("Features con senal", f"{int((nonzero_counts > 0).sum()):,} / {len(feature_cols):,}")
        s2.metric("Filas con senal", f"{int((row_signal > 0).sum()):,} / {len(features_df):,}")
        s3.metric("Cobertura max", f"{signal_df['cobertura_pct'].max():.1f}%")

        preview_df = features_df.loc[row_signal.sort_values(ascending=False).index].copy()
        preview_df["feature_signal_total"] = row_signal.loc[preview_df.index].round(3)
        visible_cols = list(dict.fromkeys(preview_base_cols + signal_cols + ["feature_signal_total"]))
        st.dataframe(preview_df[visible_cols].head(50), width="stretch")

        with st.expander("Cobertura no-cero por feature", expanded=False):
            st.dataframe(signal_df.head(50), width="stretch")
        with st.expander("Preview completo del dataset", expanded=False):
            st.dataframe(features_df.head(50), width="stretch")
    else:
        st.dataframe(features_df.head(50), width="stretch")

    ranking_df = st.session_state.get("nlp_sev_feature_ranking_df")
    if isinstance(ranking_df, pd.DataFrame) and not ranking_df.empty:
        with st.expander("Ranking de variables relevantes", expanded=False):
            max_k = int(len(ranking_df))
            top_k_view = st.slider(
                "Top K a mostrar",
                min_value=1,
                max_value=max_k,
                value=min(20, max_k),
                step=1,
                key="nlp_sev_ranking_top_k_view",
            )
            st.dataframe(ranking_df.head(int(top_k_view)), width="stretch")

    _render_registry_caption()


def _render_transformer_search_output() -> None:
    result = st.session_state.get("nlp_sev_transformer_search_result") or {}
    if not result:
        return

    status = str(result.get("status") or "ok")
    summary = result.get("search_summary") or {}
    best_result = result.get("best_result") or {}
    failure_summary = result.get("failure_summary") or {}
    trials_df = st.session_state.get("nlp_sev_transformer_search_trials_df")
    confirm_summary_df = st.session_state.get("nlp_sev_transformer_search_summary_df")

    st.markdown("#### Resultado de busqueda robusta")
    if status == "failed":
        error_rows = (
            trials_df[trials_df["status"].astype(str) == "error"].copy()
            if isinstance(trials_df, pd.DataFrame) and not trials_df.empty and "status" in trials_df.columns
            else pd.DataFrame()
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Trials ejecutados", f"{int(len(trials_df) if isinstance(trials_df, pd.DataFrame) else 0):,}")
        c2.metric("Trials fallidos", f"{int(len(error_rows)):,}")
        c3.metric("Errores unicos", f"{int(failure_summary.get('unique_error_groups') or 0):,}")
        st.error("La busqueda robusta fallo antes de encontrar un trial valido.")
        st.caption(
            f"Objetivo={summary.get('objective_metric') or '-'} | "
            f"Split seed={summary.get('split_random_state')} | "
            f"Train seed base={summary.get('trainer_seed_base')}."
        )
        if failure_summary:
            st.json(failure_summary)
        if isinstance(error_rows, pd.DataFrame) and not error_rows.empty:
            error_columns = [
                column
                for column in [
                    "trial_index",
                    "model_name",
                    "num_train_epochs",
                    "batch_size",
                    "max_length",
                    "learning_rate",
                    "weight_decay",
                    "warmup_ratio",
                    "freeze_layers",
                    "error_type",
                    "error",
                ]
                if column in error_rows.columns
            ]
            with st.expander("Detalle de errores por trial", expanded=True):
                st.dataframe(error_rows[error_columns].head(50), width="stretch")
            if "error_traceback" in error_rows.columns:
                with st.expander("Traceback de los primeros errores", expanded=False):
                    for row in error_rows.head(3).to_dict(orient="records"):
                        st.markdown(
                            f"**Trial {row.get('trial_index')} | {row.get('model_name')} | {row.get('error_type')}**"
                        )
                        st.code(str(row.get("error_traceback") or ""))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Trials evaluados", f"{int(len(trials_df) if isinstance(trials_df, pd.DataFrame) else 0):,}")
    c2.metric("Objetivo", str(summary.get("objective_metric") or "-"))
    best_metric_value = _resolve_transformer_objective(
        best_result.get("metrics") or {},
        objective_metric=str(summary.get("objective_metric") or ""),
    )
    c3.metric(
        "Mejor valor",
        "-" if pd.isna(best_metric_value) else f"{float(best_metric_value):.4f}",
    )
    st.caption(
        f"Split seed={summary.get('split_random_state')} | "
        f"Train seed base={summary.get('trainer_seed_base')} | "
        f"Top-K confirmado={summary.get('confirm_top_k')} x {summary.get('confirm_seed_count')} seeds."
    )
    if best_result:
        st.json(
            {
                "output_dir": best_result.get("output_dir"),
                "metrics": best_result.get("metrics"),
                "params": best_result.get("params"),
            }
        )
    if isinstance(confirm_summary_df, pd.DataFrame) and not confirm_summary_df.empty:
        with st.expander("Resumen de confirmacion", expanded=False):
            st.dataframe(confirm_summary_df.head(20), width="stretch")
    if isinstance(trials_df, pd.DataFrame) and not trials_df.empty:
        with st.expander("Ranking de trials", expanded=False):
            st.dataframe(trials_df.head(50), width="stretch")


def _render_transformer_preset_summary(preset: Dict[str, object]) -> None:
    if not preset:
        return
    params = preset.get("params") or {}
    st.json(
        {
            "label": preset.get("label"),
            "text_col": preset.get("text_col"),
            "mode": preset.get("mode"),
            "base_model": preset.get("base_model"),
            "objective_metric": preset.get("objective_metric"),
            "output_dir": preset.get("output_dir"),
            "params": params,
        }
    )


def _render_events_tab() -> None:
    st.subheader("Eventos")
    event_files = _list_event_files()
    if not event_files:
        st.warning("No se encontraron CSV de eventos en Datos.")
        return
    selected_names = st.multiselect(
        "Archivos de eventos",
        [path.name for path in event_files],
        default=[path.name for path in event_files],
        key="nlp_sev_event_files_select",
    )
    if st.button("Procesar eventos", key="nlp_sev_process_events"):
        run_id = _new_run_id("events")
        coverage_meta: Dict[str, object] = {
            "coverage_evaluated": False,
            "coverage_deferred": True,
        }
        try:
            with st.spinner("Procesando eventos..."):
                accidents_df, excluded_df = load_events_for_severity(selected_names)
        except Exception as exc:
            st.error(f"No se pudieron procesar los eventos: {exc}")
        else:
            st.session_state["nlp_sev_accidents_df"] = accidents_df
            st.session_state["nlp_sev_excluded_df"] = excluded_df
            st.session_state["nlp_sev_event_files"] = list(selected_names)
            st.session_state["nlp_sev_event_coverage_meta"] = coverage_meta
            st.session_state["nlp_sev_features_df"] = None
            st.session_state["nlp_sev_granular_df"] = None
            st.session_state["nlp_sev_coverage_preview"] = None
            if accidents_df is not None and not accidents_df.empty:
                artifact = _persist_artifact(
                    accidents_df,
                    stage="events",
                    artifact_name="processed_events",
                    run_id=run_id,
                    metadata={
                        "files": list(selected_names),
                        "excluded_rows": int(len(excluded_df)),
                        "coverage_evaluated": bool(coverage_meta.get("coverage_evaluated")),
                        "coverage_deferred": bool(coverage_meta.get("coverage_deferred")),
                    },
                )
                st.session_state["nlp_sev_events_artifact"] = artifact
            _log_action(
                "events",
                "process_events",
                {
                    "files": list(selected_names),
                    "processed_rows": int(len(accidents_df)),
                    "excluded_rows": int(len(excluded_df)),
                    "coverage_evaluated": bool(coverage_meta.get("coverage_evaluated")),
                    "coverage_deferred": bool(coverage_meta.get("coverage_deferred")),
                },
                run_id=run_id,
            )
            st.success(
                f"Eventos procesados: {len(accidents_df):,} | Excluidos sin portico: {len(excluded_df):,}"
            )
            st.caption("La cobertura de flujo se valida al generar el dataset en Feature engineering.")

    accidents_df = st.session_state.get("nlp_sev_accidents_df")
    if accidents_df is None or accidents_df.empty:
        st.info("No hay eventos procesados en memoria.")
        return

    coverage_meta = st.session_state.get("nlp_sev_event_coverage_meta") or {}
    _render_metric_cards(accidents_df)
    if not coverage_meta.get("coverage_evaluated"):
        st.caption("La cobertura de flujo no se valida en esta pestaña. Se calcula en Feature engineering.")

    artifact = st.session_state.get("nlp_sev_events_artifact") or {}
    if artifact:
        st.caption(f"Artifact: {artifact.get('db_path')} :: {artifact.get('table_name')}")
    preview_cols = [
        col
        for col in [
            "accidente_time",
            "flow_coverage_label",
            "has_flow_coverage",
            "ultimo_portico",
            "proximo_portico",
            "duracion_accidente",
            "severidad",
        ]
        if col in accidents_df.columns
    ]
    st.dataframe(accidents_df[preview_cols].head(100), width="stretch")
    excluded_df = st.session_state.get("nlp_sev_excluded_df")
    if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty:
        with st.expander("Eventos excluidos"):
            st.dataframe(excluded_df.head(100), width="stretch")
    _render_registry_caption()


def _render_feature_engineering_tab() -> None:
    st.subheader("Feature engineering")
    accidents_df = st.session_state.get("nlp_sev_accidents_df")
    if accidents_df is None or accidents_df.empty:
        st.info("Primero procese eventos en la pestana Eventos.")
        return

    has_memory = isinstance(st.session_state.get("nlp_sev_features_df"), pd.DataFrame) and not st.session_state.get(
        "nlp_sev_features_df"
    ).empty
    source_options = ["Cargar existentes", "Calcular nuevas", "En memoria"]
    if st.session_state.get("nlp_sev_feature_source") not in source_options:
        st.session_state["nlp_sev_feature_source"] = "En memoria" if has_memory else "Calcular nuevas"

    source = st.radio(
        "Fuente",
        source_options,
        horizontal=True,
        key="nlp_sev_feature_source",
    )

    if source == "En memoria":
        if not has_memory:
            st.info("No hay variables en memoria.")
            _render_registry_caption()
            return
        _render_feature_engineering_output()
        return

    if source == "Cargar existentes":
        catalog = _list_feature_engineering_artifacts()
        if catalog.empty:
            st.warning("No se encontraron datasets persistidos de feature engineering.")
        else:
            selected_idx = st.selectbox(
                "Dataset persistido",
                options=catalog.index.tolist(),
                format_func=lambda idx: str(catalog.loc[idx, "label"]),
                key="nlp_sev_feature_artifact_select",
            )
            selected_row = catalog.loc[selected_idx]
            metadata = selected_row.get("metadata") or {}
            st.caption(
                f"Run: {selected_row.get('run_id') or '-'} | "
                f"Ventana: {metadata.get('window_size_minutes', 1)} min | "
                f"Antes: {metadata.get('windows_before', metadata.get('minutes_before', '-'))} | "
                f"Despues: {metadata.get('windows_after', metadata.get('minutes_after', '-'))} | "
                f"Metricas: {', '.join(metadata.get('selected_metrics', GRANULAR_METRIC_NAMES))} | "
                f"Deltas: {'Si' if metadata.get('include_deltas', True) else 'No'} | "
                f"Muestreo: {metadata.get('sampling_mode', '-')}"
            )
            if st.button("Cargar dataset persistido", key="nlp_sev_load_existing_features"):
                try:
                    features_df, granular_df, artifact_bundle = _load_feature_bundle_from_catalog_row(selected_row)
                except Exception as exc:
                    st.error(f"No se pudo cargar el dataset persistido: {exc}")
                else:
                    if features_df.empty:
                        st.warning("El dataset seleccionado no contiene features.")
                    else:
                        ranking_df = _compute_relevant_feature_ranking(
                            features_df,
                            top_k=max(1, len(_flow_feature_columns(features_df))),
                        )
                        st.session_state["nlp_sev_features_df"] = features_df
                        st.session_state["nlp_sev_granular_df"] = granular_df
                        st.session_state["nlp_sev_feature_ranking_df"] = ranking_df
                        st.session_state["nlp_sev_features_artifact"] = artifact_bundle
                        st.session_state["nlp_sev_granular_artifact"] = (
                            artifact_bundle.get("paired_granular")
                            if isinstance(artifact_bundle, dict)
                            else None
                        )
                        _reset_nlp_sev_language_state()
                        _log_action(
                            "feature_engineering",
                            "load_existing_features",
                            {
                                "artifact_id": selected_row.get("artifact_id"),
                                "run_id": selected_row.get("run_id"),
                                "db_path": str(selected_row.get("db_path")),
                                "table_name": selected_row.get("table_name"),
                                "rows": int(len(features_df)),
                                "granular_rows": int(len(granular_df)),
                                "sampling_mode": metadata.get("sampling_mode"),
                                "sampling_seed": metadata.get("sampling_seed"),
                                "sampled_events": metadata.get("sampled_events"),
                            },
                            run_id=selected_row.get("run_id"),
                        )
                        st.success(f"Dataset cargado: {len(features_df):,} accidentes con features.")
        _render_feature_engineering_output()
        return

    try:
        summary = get_flow_db_summary()
    except Exception as exc:
        st.error(f"No se pudo leer la base de flujos DuckDB: {exc}")
        _render_feature_engineering_output()
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows DuckDB", f"{summary.row_count:,}")
    c2.metric("Fecha min", summary.min_timestamp.strftime("%Y-%m-%d %H:%M") if summary.min_timestamp else "-")
    c3.metric("Fecha max", summary.max_timestamp.strftime("%Y-%m-%d %H:%M") if summary.max_timestamp else "-")

    accidents_for_sampling = accidents_df.reset_index(drop=False).rename(columns={"index": "_feature_event_id"})

    sample_mode = _build_feature_sample_mode_selector("nlp_sev")
    st.caption(
        "El muestreo se aplica sobre los eventos procesados; las ventanas de flujo se recalculan desde DuckDB."
    )
    sample, range_valid, sample_seed = _build_feature_sample_inputs(
        accidents_for_sampling,
        sample_mode,
        key_prefix="nlp_sev",
    )
    sampled_accidents = _sample_accidents_for_feature_engineering(
        accidents_for_sampling,
        sample,
        mode=sample_mode,
        sample_seed=sample_seed,
    )
    config_cols = st.columns(3)
    with config_cols[0]:
        window_size_minutes = int(
            st.selectbox(
                "Tamaño de ventana (min)",
                options=[1, 2, 3, 5, 10, 15],
                index=0,
                key="nlp_sev_window_size_minutes",
            )
        )
    with config_cols[1]:
        windows_before = int(
            st.selectbox(
                "Ventanas antes",
                options=list(range(0, 16)),
                index=5,
                key="nlp_sev_windows_before",
            )
        )
    with config_cols[2]:
        windows_after = int(
            st.selectbox(
                "Ventanas despues",
                options=list(range(0, 16)),
                index=5,
                key="nlp_sev_windows_after",
            )
        )

    sampled_event_count = int(len(sampled_accidents))
    total_event_count = int(len(accidents_for_sampling))
    coverage_preview_signature = _build_coverage_preview_signature(
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        sampled_events=sampled_event_count,
        total_events=total_event_count,
        windows_before=windows_before,
        windows_after=windows_after,
        window_size_minutes=window_size_minutes,
        sampling_date_start=sample.date_start,
        sampling_date_end=sample.date_end,
    )
    coverage_preview = st.session_state.get("nlp_sev_coverage_preview")

    if sampled_accidents.empty:
        st.warning("El muestreo seleccionado no contiene eventos para procesar.")

    if coverage_preview and coverage_preview.get("signature") == coverage_preview_signature:
        covered_events_preview = int(coverage_preview.get("covered_events") or 0)
        st.caption(
            f"Eventos a procesar: {covered_events_preview:,} de {sampled_event_count:,} "
            "seleccionados con cobertura de flujo."
        )
    else:
        st.caption(
            f"Eventos seleccionados: {sampled_event_count:,} de {total_event_count:,}. "
            "La cobertura exacta de flujo se valida al generar el dataset."
        )

    if range_valid and sampled_event_count > 0 and windows_before == 0 and windows_after == 0:
        st.warning("Seleccione al menos una ventana temporal antes o despues del accidente.")

    if sampled_event_count > 0:
        _render_metric_cards(sampled_accidents)

    selected_metric_labels = st.multiselect(
        "Features a calcular",
        options=[GRANULAR_METRIC_LABELS[metric] for metric in GRANULAR_METRIC_NAMES],
        default=[GRANULAR_METRIC_LABELS[metric] for metric in GRANULAR_METRIC_NAMES],
        key="nlp_sev_selected_metric_labels",
    )
    selected_metrics = [
        metric
        for metric, label in GRANULAR_METRIC_LABELS.items()
        if label in selected_metric_labels
    ]
    include_deltas = st.checkbox(
        "Calcular deltas entre ventanas temporales",
        value=True,
        key="nlp_sev_include_deltas",
    )

    text_candidate_columns = _project_text_source_columns(
        windows_before=windows_before,
        windows_after=windows_after,
        window_size_minutes=window_size_minutes,
    )
    current_text_columns = st.session_state.get("nlp_sev_feature_text_columns")
    if (
        not isinstance(current_text_columns, list)
        or any(col not in text_candidate_columns for col in current_text_columns)
    ):
        st.session_state["nlp_sev_feature_text_columns"] = text_candidate_columns.copy()
    selected_text_columns = st.multiselect(
        "Columnas para concatenar en text_bert",
        options=text_candidate_columns,
        default=text_candidate_columns,
        key="nlp_sev_feature_text_columns",
    )

    if st.button(
        "Generar dataset granular de severidad",
        key="nlp_sev_build_features",
        disabled=(
            (not range_valid)
            or sampled_accidents.empty
            or (windows_before == 0 and windows_after == 0)
            or not selected_metrics
            or not selected_text_columns
        ),
    ):
        if not selected_metrics:
            st.warning("Seleccione al menos una feature a calcular.")
            return
        if not selected_text_columns:
            st.warning("Seleccione al menos una columna para text_bert.")
            return
        run_id = _new_run_id("feature_engineering")
        progress_bar = st.progress(0)
        progress_status = st.empty()

        def _update_feature_progress(value: int, message: str) -> None:
            progress_bar.progress(int(value))
            progress_status.caption(message)

        try:
            features_df, granular_df, ranking_df = build_severity_feature_dataset(
                sampled_accidents,
                flow_db_path=summary.db_path,
                windows_before=windows_before,
                windows_after=windows_after,
                window_size_minutes=window_size_minutes,
                selected_metrics=selected_metrics,
                include_deltas=bool(include_deltas),
                text_columns=selected_text_columns,
                progress_callback=_update_feature_progress,
            )
        except Exception as exc:
            st.session_state["nlp_sev_coverage_preview"] = None
            progress_bar.empty()
            progress_status.empty()
            st.error(f"No se pudo generar el feature engineering: {exc}")
        else:
            if features_df.empty:
                st.session_state["nlp_sev_coverage_preview"] = None
                progress_bar.empty()
                progress_status.empty()
                st.warning("No se genero dataset con cobertura de flujo.")
            else:
                covered_events = int(len(features_df))
                excluded_without_flow_coverage = max(0, int(len(sampled_accidents)) - covered_events)
                metadata = {
                    "windows_before": int(windows_before),
                    "windows_after": int(windows_after),
                    "window_size_minutes": int(window_size_minutes),
                    "selected_metrics": list(selected_metrics),
                    "include_deltas": bool(include_deltas),
                    "text_columns": list(selected_text_columns),
                    "sampling_mode": sample_mode,
                    "sampling_seed": sample_seed,
                    "sampled_events": int(len(sampled_accidents)),
                    "covered_events": covered_events,
                    "excluded_without_flow_coverage": excluded_without_flow_coverage,
                    "sampling_date_start": sample.date_start,
                    "sampling_date_end": sample.date_end,
                    "flow_db_path": str(summary.db_path),
                }
                st.session_state["nlp_sev_coverage_preview"] = {
                    "signature": coverage_preview_signature,
                    "covered_events": covered_events,
                    "excluded_without_flow_coverage": excluded_without_flow_coverage,
                }
                st.session_state["nlp_sev_features_df"] = features_df
                st.session_state["nlp_sev_granular_df"] = granular_df
                st.session_state["nlp_sev_feature_ranking_df"] = ranking_df
                _reset_nlp_sev_language_state()
                features_artifact = _persist_artifact(
                    features_df,
                    stage="feature_engineering",
                    artifact_name="severity_features",
                    run_id=run_id,
                    metadata=metadata,
                )
                granular_artifact = _persist_artifact(
                    granular_df,
                    stage="feature_engineering",
                    artifact_name="severity_granular",
                    run_id=run_id,
                    metadata={
                        "source": "granular_long",
                        "windows_before": int(windows_before),
                        "windows_after": int(windows_after),
                        "window_size_minutes": int(window_size_minutes),
                        "selected_metrics": list(selected_metrics),
                        "include_deltas": bool(include_deltas),
                        "sampling_mode": sample_mode,
                        "sampling_seed": sample_seed,
                        "sampled_events": int(len(sampled_accidents)),
                        "covered_events": covered_events,
                        "excluded_without_flow_coverage": excluded_without_flow_coverage,
                        "sampling_date_start": sample.date_start,
                        "sampling_date_end": sample.date_end,
                    },
                )
                st.session_state["nlp_sev_features_artifact"] = features_artifact
                st.session_state["nlp_sev_granular_artifact"] = granular_artifact
                _log_action(
                    "feature_engineering",
                    "build_feature_dataset",
                    {
                        "rows": int(len(features_df)),
                        "granular_rows": int(len(granular_df)),
                        "flow_db_path": str(summary.db_path),
                        "window_size_minutes": int(window_size_minutes),
                        "windows_before": int(windows_before),
                        "windows_after": int(windows_after),
                        "selected_metrics": list(selected_metrics),
                        "include_deltas": bool(include_deltas),
                        "text_columns": list(selected_text_columns),
                        "sampling_mode": sample_mode,
                        "sampling_seed": sample_seed,
                        "sampled_events": int(len(sampled_accidents)),
                        "covered_events": covered_events,
                        "excluded_without_flow_coverage": excluded_without_flow_coverage,
                        "sampling_date_start": sample.date_start,
                        "sampling_date_end": sample.date_end,
                    },
                    run_id=run_id,
                )
                progress_bar.progress(100)
                progress_status.caption("Dataset granular de severidad listo.")
                st.success(
                    f"Dataset generado: {covered_events:,} accidentes con cobertura de flujo | "
                    f"Excluidos sin cobertura: {excluded_without_flow_coverage:,}."
                )

    _render_feature_engineering_output()


def _render_language_modeling_tab() -> None:
    st.subheader("Language modeling")
    base_df = st.session_state.get("nlp_sev_features_df")
    if base_df is None or base_df.empty:
        st.info("Ejecute Feature engineering primero.")
        return

    source_df = st.session_state.get("nlp_sev_language_df")
    if source_df is None or source_df.empty:
        source_df = base_df

    sub_tabs = st.tabs(
        [
            "Textos",
            "Finetune LLM",
            "Embeddings",
            "RF embeddings",
        ]
    )

    with sub_tabs[0]:
        candidate_text_cols = _text_selectable_columns(base_df)
        current_text_cols = st.session_state.get("nlp_sev_text_relevant_cols")
        if not isinstance(current_text_cols, list) or any(col not in candidate_text_cols for col in current_text_cols):
            st.session_state["nlp_sev_text_relevant_cols"] = candidate_text_cols.copy()
        selected_text_cols = st.multiselect(
            "Columnas para concatenar en text_bert_lm",
            candidate_text_cols,
            default=candidate_text_cols,
            key="nlp_sev_text_relevant_cols",
        )
        include_target = st.checkbox(
            "Incluir severidad en el texto (no recomendado)",
            value=False,
            key="nlp_sev_text_include_target",
        )
        if st.button("Generar version textual para language modeling", key="nlp_sev_regen_texts"):
            run_id = _new_run_id("language_texts")
            lm_df = _build_text_columns(
                base_df,
                selected_columns=selected_text_cols,
                text_prefix="lm_",
                include_target=bool(include_target),
            )
            st.session_state["nlp_sev_language_df"] = lm_df
            artifact = _persist_artifact(
                lm_df,
                stage="language_modeling",
                artifact_name="language_texts",
                run_id=run_id,
                metadata={
                    "selected_text_cols": list(selected_text_cols),
                    "include_target": bool(include_target),
                },
            )
            st.session_state["nlp_sev_language_artifact"] = artifact
            _log_action(
                "language_modeling",
                "regenerate_texts",
                {
                    "rows": int(len(lm_df)),
                    "selected_text_cols": list(selected_text_cols),
                    "include_target": bool(include_target),
                },
                run_id=run_id,
            )
            st.success("Version textual regenerada.")

        active_df = st.session_state.get("nlp_sev_language_df")
        if active_df is None or active_df.empty:
            active_df = base_df
        text_cols = [col for col in active_df.columns if "text_bert" in col]
        st.caption(f"Columnas textuales disponibles: {', '.join(text_cols)}")
        preview_cols = [col for col in text_cols if col in active_df.columns]
        st.dataframe(active_df[preview_cols].head(20), width="stretch")

    with sub_tabs[1]:
        text_df = st.session_state.get("nlp_sev_language_df")
        if text_df is None or text_df.empty:
            text_df = base_df
        text_cols = [col for col in text_df.columns if "text_bert" in col]
        model_options = _transformer_model_options()
        model_labels = list(model_options.keys())
        transformers_ready = not (
            AutoTokenizer is None
            or Trainer is None
            or TrainingArguments is None
            or torch is None
            or AutoModel is None
        )
        if not transformers_ready:
            st.warning(
                "El entorno actual no tiene el stack de Transformers instalado. "
                "Se requiere `transformers`, `accelerate` y `torch`."
            )
        elif _torch_requires_safetensors():
            st.warning(
                "El entorno actual usa `torch < 2.6`. En este modo, los modelos remotos "
                "solo se podran cargar si publican pesos `safetensors`. Si un modelo base "
                "solo tiene `pytorch_model.bin`, la busqueda/fine-tune va a fallar hasta "
                "actualizar `torch` a >= 2.6."
            )
        st.caption(
            "Pipeline recomendado: 1) seleccionar hiperparametros guardados o buscar nuevos, "
            "2) aplicar el ajuste final con esos hiperparametros, 3) reutilizar el modelo en Embeddings."
        )

        st.markdown("#### Paso 1 · Hiperparametros")
        hyperparam_strategy = st.radio(
            "Fuente de hiperparametros",
            ["Buscar nuevos", "Usar guardados"],
            horizontal=True,
            key="nlp_sev_tf_hparam_strategy",
        )
        preset_catalog = _list_transformer_search_presets()

        if hyperparam_strategy == "Usar guardados":
            if preset_catalog.empty:
                st.info("No hay resultados guardados de busqueda robusta. Ejecute una nueva busqueda.")
                st.session_state["nlp_sev_transformer_active_preset"] = None
            else:
                selected_preset_idx = st.selectbox(
                    "Resultado robusto",
                    options=preset_catalog.index.tolist(),
                    format_func=lambda idx: str(preset_catalog.loc[idx, "preset_label"]),
                    key="nlp_sev_tf_saved_search_select",
                )
                active_preset = _transformer_preset_from_model_result_row(
                    preset_catalog.loc[selected_preset_idx]
                )
                st.session_state["nlp_sev_transformer_active_preset"] = active_preset
                _render_transformer_preset_summary(active_preset)
        else:
            current_search_payload = st.session_state.get("nlp_sev_transformer_search_result") or {}
            current_best_result = current_search_payload.get("best_result") or {}
            if not current_best_result:
                st.session_state["nlp_sev_transformer_active_preset"] = None
            if not text_cols:
                st.info("No hay columnas textuales disponibles. Genere textos primero.")
                st.session_state["nlp_sev_transformer_active_preset"] = None
            else:
                st.caption(
                    "Defina la busqueda primero. Cuando termine, el mejor preset quedara listo para el ajuste final."
                )
                with st.expander("Busqueda robusta de hiperparametros", expanded=True):
                    st.caption(
                        "Fase 1: explora configuraciones sobre un split fijo. "
                        "Fase 2: confirma el top-K con multiples train seeds. "
                        "Fase 3: deja guardado el mejor preset para el finetune final."
                    )
                    cfg1, cfg2 = st.columns([1, 1])
                    with cfg1:
                        selected_text_col = st.selectbox(
                            "Columna de texto",
                            text_cols,
                            key="nlp_sev_text_baseline_col",
                        )
                    with cfg2:
                        mode_label = st.radio(
                            "Modo",
                            ["Clasificador", "Masked Language Modeling (MLM)"],
                            horizontal=True,
                            key="nlp_sev_transformers_mode",
                        )
                    mode = "classification" if mode_label == "Clasificador" else "mlm"

                    severity_values = (
                        pd.to_numeric(text_df["severity_target"], errors="coerce")
                        if "severity_target" in text_df.columns
                        else pd.Series(dtype=float)
                    )
                    objective_options = ["f1", "accuracy"] if mode == "classification" else ["eval_loss"]
                    if mode == "classification" and severity_values.dropna().nunique() == 2:
                        objective_options.append("roc_auc")

                    cfg3, cfg4, cfg5 = st.columns(3)
                    with cfg3:
                        objective_metric = st.selectbox(
                            "Metrica objetivo",
                            objective_options,
                            index=0,
                            key="nlp_sev_tf_search_objective",
                        )
                    with cfg4:
                        validation_size = st.slider(
                            "Validation size",
                            min_value=0.1,
                            max_value=0.4,
                            value=0.2,
                            step=0.05,
                            key="nlp_sev_tf_validation_size",
                        )
                    with cfg5:
                        search_output_folder = st.text_input(
                            "Carpeta de salida de la busqueda",
                            value=(
                                f"transformers_search_"
                                f"{'clf' if mode == 'classification' else 'mlm'}_"
                                f"{_slug(selected_text_col)}"
                            ),
                            key="nlp_sev_tf_search_output_folder",
                        )

                    model_validation_map = {
                        label: _transformer_model_validation(model_options[label])
                        for label in model_labels
                    }

                    def _format_model_option(label: str) -> str:
                        validation = model_validation_map.get(label) or {}
                        badge = _transformer_model_status_badge(validation)
                        return f"{label} [{badge}]"

                    selected_search_model_label = st.selectbox(
                        "Modelo de lenguaje base",
                        model_labels,
                        format_func=_format_model_option,
                        key="nlp_sev_tf_search_primary_model",
                    )
                    selected_search_validation = model_validation_map.get(selected_search_model_label) or {}
                    if selected_search_validation.get("status") == "incompatible":
                        st.error(str(selected_search_validation.get("message") or "Modelo incompatible."))
                    elif selected_search_validation.get("status") == "unknown":
                        st.warning(str(selected_search_validation.get("message") or "No se pudo validar el modelo."))
                    else:
                        st.caption(str(selected_search_validation.get("message") or "Modelo compatible."))

                    additional_candidate_labels = [
                        label
                        for label in model_labels
                        if label != selected_search_model_label
                        and str((model_validation_map.get(label) or {}).get("status") or "") != "incompatible"
                    ]
                    additional_model_labels = st.multiselect(
                        "Modelos adicionales a explorar",
                        additional_candidate_labels,
                        default=[],
                        format_func=_format_model_option,
                        key="nlp_sev_tf_search_models",
                    )
                    model_candidate_labels = _dedupe_preserve_order(
                        [selected_search_model_label] + list(additional_model_labels)
                    )
                    incompatible_labels = [
                        label
                        for label in model_labels
                        if str((model_validation_map.get(label) or {}).get("status") or "") == "incompatible"
                    ]
                    if incompatible_labels:
                        st.caption(
                            "Modelos excluidos por incompatibilidad del runtime actual: "
                            + ", ".join(incompatible_labels)
                        )
                    with st.expander("Compatibilidad detectada por modelo", expanded=False):
                        compatibility_df = pd.DataFrame(
                            [
                                {
                                    "model_label": label,
                                    "status": (model_validation_map.get(label) or {}).get("status"),
                                    "requires_safetensors": (model_validation_map.get(label) or {}).get("requires_safetensors"),
                                    "has_safetensors": (model_validation_map.get(label) or {}).get("has_safetensors"),
                                    "has_pytorch_bin": (model_validation_map.get(label) or {}).get("has_pytorch_bin"),
                                    "checked_via": (model_validation_map.get(label) or {}).get("checked_via"),
                                    "message": (model_validation_map.get(label) or {}).get("message"),
                                }
                                for label in model_labels
                            ]
                        )
                        st.dataframe(compatibility_df, width="stretch")

                    search_col1, search_col2, search_col3 = st.columns(3)
                    with search_col1:
                        max_trials = int(
                            st.number_input(
                                "Max trials",
                                min_value=1,
                                max_value=64,
                                value=8,
                                step=1,
                                key="nlp_sev_tf_search_max_trials",
                            )
                        )
                    with search_col2:
                        confirm_top_k = int(
                            st.number_input(
                                "Top-K a confirmar",
                                min_value=1,
                                max_value=10,
                                value=3,
                                step=1,
                                key="nlp_sev_tf_search_confirm_top_k",
                            )
                        )
                    with search_col3:
                        confirm_seed_count = int(
                            st.number_input(
                                "Seeds de confirmacion",
                                min_value=1,
                                max_value=10,
                                value=3,
                                step=1,
                                key="nlp_sev_tf_search_confirm_seed_count",
                            )
                        )

                    search_col4, search_col5, search_col6 = st.columns(3)
                    with search_col4:
                        split_seed = int(
                            st.number_input(
                                "Split seed fijo",
                                min_value=0,
                                max_value=9999,
                                value=42,
                                step=1,
                                key="nlp_sev_tf_search_split_seed",
                            )
                        )
                    with search_col5:
                        trainer_seed_base = int(
                            st.number_input(
                                "Train seed base",
                                min_value=0,
                                max_value=9999,
                                value=42,
                                step=1,
                                key="nlp_sev_tf_search_train_seed",
                            )
                        )
                    with search_col6:
                        keep_trial_artifacts = st.checkbox(
                            "Conservar carpetas intermedias de trials",
                            value=False,
                            key="nlp_sev_tf_search_keep_artifacts",
                        )

                    epoch_options = [1, 2, 3, 4, 5, 6, 8, 10]
                    batch_options = [4, 8, 16, 32]
                    max_length_options = [64, 128, 256, 384, 512]
                    lr_options = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4]
                    wd_options = [0.0, 0.01, 0.05, 0.1]
                    warmup_ratio_options = [0.0, 0.05, 0.1]
                    freeze_label_map = {"No": False, "Si": True}
                    mlm_candidate_options = [0.10, 0.15, 0.20]

                    cand1, cand2, cand3 = st.columns(3)
                    with cand1:
                        search_epochs = st.multiselect(
                            "Epocas candidatas",
                            epoch_options,
                            default=[2, 3, 4],
                            key="nlp_sev_tf_search_epochs",
                        )
                    with cand2:
                        search_batches = st.multiselect(
                            "Batch sizes",
                            batch_options,
                            default=[8, 16],
                            key="nlp_sev_tf_search_batches",
                        )
                    with cand3:
                        search_max_lengths = st.multiselect(
                            "Max tokens candidatos",
                            max_length_options,
                            default=[128, 256],
                            key="nlp_sev_tf_search_max_lengths",
                        )

                    cand4, cand5, cand6 = st.columns(3)
                    with cand4:
                        search_learning_rates = st.multiselect(
                            "Learning rates",
                            lr_options,
                            default=[1e-5, 2e-5, 5e-5],
                            key="nlp_sev_tf_search_lrs",
                            format_func=lambda value: f"{float(value):.0e}",
                        )
                    with cand5:
                        search_weight_decays = st.multiselect(
                            "Weight decays",
                            wd_options,
                            default=[0.0, 0.01],
                            key="nlp_sev_tf_search_wds",
                        )
                    with cand6:
                        selected_freeze_labels = st.multiselect(
                            "Congelar capas inferiores",
                            list(freeze_label_map.keys()),
                            default=["No", "Si"],
                            key="nlp_sev_tf_search_freeze",
                        )

                    cand7, cand8 = st.columns(2)
                    with cand7:
                        search_warmup_ratios = st.multiselect(
                            "Warmup ratios",
                            warmup_ratio_options,
                            default=warmup_ratio_options,
                            key="nlp_sev_tf_search_warmup_ratios",
                        )
                    with cand8:
                        if mode == "mlm":
                            search_mlm_probs = st.multiselect(
                                "MLM probability candidata",
                                mlm_candidate_options,
                                default=[0.15],
                                key="nlp_sev_tf_search_mlm_probs",
                            )
                        else:
                            search_mlm_probs = [0.15]

                    search_ready = (
                        str(selected_search_validation.get("status") or "") != "incompatible"
                        and
                        bool(model_candidate_labels)
                        and bool(search_epochs)
                        and bool(search_batches)
                        and bool(search_max_lengths)
                        and bool(search_learning_rates)
                        and bool(search_weight_decays)
                        and bool(search_warmup_ratios)
                        and bool(selected_freeze_labels)
                        and bool(search_mlm_probs)
                    )

                    if st.button(
                        "Ejecutar busqueda robusta",
                        key="nlp_sev_train_text_search",
                        disabled=not transformers_ready or not search_ready,
                    ):
                        run_id = _new_run_id("language_search")
                        search_progress = st.progress(0)
                        search_status = st.empty()

                        def _update_search_progress(value: int, message: str) -> None:
                            search_progress.progress(int(value))
                            search_status.caption(message)

                        search_space = {
                            "model_name": [model_options[label] for label in model_candidate_labels],
                            "num_train_epochs": [int(value) for value in search_epochs],
                            "batch_size": [int(value) for value in search_batches],
                            "max_length": [int(value) for value in search_max_lengths],
                            "learning_rate": [float(value) for value in search_learning_rates],
                            "weight_decay": [float(value) for value in search_weight_decays],
                            "warmup_ratio": [float(value) for value in search_warmup_ratios],
                            "freeze_layers": [freeze_label_map[label] for label in selected_freeze_labels],
                            "test_size": [float(validation_size)],
                            "mlm_probability": [float(value) for value in search_mlm_probs],
                        }
                        try:
                            search_root = MODULE_RESULTS_DIR / "models" / _slug(search_output_folder) / run_id
                            search_result = run_transformers_hyperparameter_search(
                                text_df,
                                text_col=selected_text_col,
                                mode=mode,
                                output_dir=search_root,
                                search_space=search_space,
                                max_trials=int(max_trials),
                                objective_metric=str(objective_metric),
                                split_random_state=int(split_seed),
                                trainer_seed_base=int(trainer_seed_base),
                                confirm_top_k=int(confirm_top_k),
                                confirm_seed_count=int(confirm_seed_count),
                                keep_trial_artifacts=bool(keep_trial_artifacts),
                                progress_callback=_update_search_progress,
                            )
                        except Exception as exc:
                            search_progress.empty()
                            search_status.empty()
                            st.session_state["nlp_sev_transformer_search_trials_df"] = None
                            st.session_state["nlp_sev_transformer_search_confirm_df"] = None
                            st.session_state["nlp_sev_transformer_search_summary_df"] = None
                            st.session_state["nlp_sev_transformer_search_result"] = None
                            st.session_state["nlp_sev_transformer_active_preset"] = None
                            if isinstance(exc, TransformerSearchDebugError):
                                st.session_state["nlp_sev_transformer_search_trials_df"] = exc.trials_df
                                st.session_state["nlp_sev_transformer_search_confirm_df"] = exc.confirm_df
                                st.session_state["nlp_sev_transformer_search_summary_df"] = exc.confirm_summary_df
                                st.session_state["nlp_sev_transformer_search_result"] = exc.search_result
                            st.error(f"No se pudo ejecutar la busqueda de hiperparametros: {exc}")
                        else:
                            search_progress.empty()
                            search_status.empty()
                            trials_df = search_result.get("trials_df", pd.DataFrame())
                            confirm_df = search_result.get("confirm_df", pd.DataFrame())
                            confirm_summary_df = search_result.get("confirm_summary_df", pd.DataFrame())
                            best_result = search_result.get("best_result", {})
                            best_history_df = search_result.get("best_history_df", pd.DataFrame())
                            search_summary = search_result.get("search_summary", {})

                            st.session_state["nlp_sev_transformer_search_trials_df"] = trials_df
                            st.session_state["nlp_sev_transformer_search_confirm_df"] = confirm_df
                            st.session_state["nlp_sev_transformer_search_summary_df"] = confirm_summary_df
                            st.session_state["nlp_sev_transformer_search_result"] = {
                                "search_summary": search_summary,
                                "best_result": best_result,
                            }

                            _record_model_result(
                                run_id=run_id,
                                stage="language_modeling",
                                model_name=f"Transformers Search ({mode_label})",
                                feature_group=f"{selected_text_col} [{objective_metric}]",
                                metrics=best_result.get("metrics") or {},
                                params=best_result.get("params") or {},
                                metadata={
                                    "text_col": selected_text_col,
                                    "mode": mode,
                                    "objective_metric": objective_metric,
                                    "base_model": best_result.get("model_name"),
                                    "best_output_dir": best_result.get("output_dir"),
                                    "search_summary": search_summary,
                                },
                            )
                            if isinstance(trials_df, pd.DataFrame) and not trials_df.empty:
                                _persist_artifact(
                                    trials_df,
                                    stage="language_modeling",
                                    artifact_name="transformers_search_trials",
                                    run_id=run_id,
                                    metadata={
                                        "text_col": selected_text_col,
                                        "mode": mode,
                                        "objective_metric": objective_metric,
                                    },
                                )
                            if isinstance(confirm_summary_df, pd.DataFrame) and not confirm_summary_df.empty:
                                _persist_artifact(
                                    confirm_summary_df,
                                    stage="language_modeling",
                                    artifact_name="transformers_search_confirmation",
                                    run_id=run_id,
                                    metadata={
                                        "text_col": selected_text_col,
                                        "mode": mode,
                                        "objective_metric": objective_metric,
                                    },
                                )
                            if isinstance(best_history_df, pd.DataFrame) and not best_history_df.empty:
                                _persist_artifact(
                                    best_history_df,
                                    stage="language_modeling",
                                    artifact_name="transformers_search_best_history",
                                    run_id=run_id,
                                    metadata={
                                        "text_col": selected_text_col,
                                        "mode": mode,
                                        "objective_metric": objective_metric,
                                        "output_dir": best_result.get("output_dir"),
                                    },
                                )
                            _log_action(
                                "language_modeling",
                                "transformers_hparam_search",
                                {
                                    "text_col": selected_text_col,
                                    "mode": mode,
                                    "objective_metric": objective_metric,
                                    "executed_trials": int(len(trials_df)),
                                    "confirmed_configs": int(len(confirm_summary_df)) if isinstance(confirm_summary_df, pd.DataFrame) else 0,
                                    "best_output_dir": best_result.get("output_dir"),
                                    "best_metrics": best_result.get("metrics"),
                                    "best_params": best_result.get("params"),
                                },
                                run_id=run_id,
                            )
                            st.session_state["nlp_sev_transformer_active_preset"] = {
                                "run_id": run_id,
                                "created_at": _ts_now(),
                                "source_model_name": f"Transformers Search ({mode_label})",
                                "text_col": selected_text_col,
                                "mode": mode,
                                "base_model": best_result.get("model_name"),
                                "objective_metric": objective_metric,
                                "output_dir": best_result.get("output_dir"),
                                "params": best_result.get("params") or {},
                                "metrics": best_result.get("metrics") or {},
                                "metadata": {"search_summary": search_summary},
                                "label": (
                                    f"Busqueda actual | {mode} | {selected_text_col} | "
                                    f"{best_result.get('model_name')}"
                                ),
                            }
                            st.success(
                                "Busqueda completada. Hiperparametros listos para el ajuste final."
                            )

            _render_transformer_search_output()

        st.markdown("#### Paso 2 · Ajuste final del modelo")
        active_preset = st.session_state.get("nlp_sev_transformer_active_preset") or {}
        if not active_preset:
            st.info("Primero busque nuevos hiperparametros o seleccione un resultado guardado.")
        else:
            preset_ready = True
            preset_text_col = str(active_preset.get("text_col") or "")
            preset_mode = str(active_preset.get("mode") or "")
            preset_model_name = str(active_preset.get("base_model") or "")
            if preset_text_col not in text_df.columns:
                preset_ready = False
                st.warning(
                    f"La columna '{preset_text_col}' del preset no existe en el dataset textual actual."
                )
            if preset_mode not in {"classification", "mlm"}:
                preset_ready = False
                st.warning("El preset activo no contiene un modo valido.")
            if not preset_model_name:
                preset_ready = False
                st.warning("El preset activo no contiene el modelo base ganador.")
            _render_transformer_preset_summary(active_preset)
            final_output_folder = st.text_input(
                "Carpeta base del modelo fine-tuneado",
                value=f"transformers_final_{_slug(preset_text_col)}",
                key="nlp_sev_tf_final_output_folder",
            )
            st.caption(
                "Cada ajuste final se guarda en una subcarpeta unica con su run_id para que el "
                "modelo quede disponible despues en la tab Embeddings."
            )
            if st.button(
                "Aplicar finetune con hiperparametros seleccionados",
                key="nlp_sev_apply_transformer_preset",
                disabled=not transformers_ready or not preset_ready,
            ):
                run_id = _new_run_id("language_finetune")
                try:
                    model_output_dir = (
                        MODULE_RESULTS_DIR
                        / "models"
                        / _slug(final_output_folder)
                        / run_id
                    )
                    with st.spinner("Aplicando finetune final con hiperparametros seleccionados..."):
                        result = execute_transformer_finetune_from_preset(
                            text_df,
                            preset=active_preset,
                            output_dir=model_output_dir,
                            run_id=run_id,
                            result_model_name="Transformers (Fine-tuned)",
                            action_name="transformers_finetune_from_selected_preset",
                            extra_metadata={"selection_strategy": hyperparam_strategy},
                        )
                except Exception as exc:
                    st.error(f"No se pudo ejecutar el finetune final: {exc}")
                else:
                    history_df = result.pop("history_df", pd.DataFrame())
                    st.success(
                        "Finetune final completado. "
                        f"Modelo guardado en {result['output_dir']}"
                    )
                    st.json({key: value for key, value in result.items() if key != "history_df"})
                    if isinstance(history_df, pd.DataFrame) and not history_df.empty:
                        st.dataframe(history_df.tail(20), width="stretch")

    with sub_tabs[2]:
        embed_source_df = st.session_state.get("nlp_sev_language_df")
        if embed_source_df is None or embed_source_df.empty:
            embed_source_df = base_df
        text_cols = [col for col in embed_source_df.columns if "text_bert" in col]
        if not text_cols:
            st.info("No hay columnas textuales disponibles. Genere textos primero.")
        else:
            method_options = ["tfidf_svd"]
            if SentenceTransformer is not None:
                method_options.append("sentence_transformer")
            finetuned_model_catalog = _list_transformer_finetuned_models()
            if (
                AutoModel is not None
                and AutoTokenizer is not None
                and torch is not None
                and not finetuned_model_catalog.empty
            ):
                method_options.append("transformer_finetuned")
            method = st.selectbox("Metodo", method_options, key="nlp_sev_embeddings_method")
            selected_text_col = None
            transformer_model_path = None
            transformer_batch_size = 16
            transformer_max_length = 256
            if method == "transformer_finetuned":
                selected_model_idx = st.selectbox(
                    "Modelo fine-tuneado",
                    options=finetuned_model_catalog.index.tolist(),
                    format_func=lambda idx: str(finetuned_model_catalog.loc[idx, "model_label"]),
                    key="nlp_sev_embeddings_finetuned_model",
                )
                model_row = finetuned_model_catalog.loc[selected_model_idx]
                model_metadata = model_row.get("metadata") or {}
                suggested_text_col = str(model_metadata.get("text_col") or "")
                if suggested_text_col in text_cols:
                    selected_text_col = suggested_text_col
                    st.caption(f"Usando la columna textual del modelo: `{selected_text_col}`")
                else:
                    selected_text_col = st.selectbox(
                        "Columna de texto para embeddings",
                        text_cols,
                        key="nlp_sev_embeddings_col_finetuned",
                    )
                transformer_model_path = str(model_row.get("output_dir_resolved"))
                transformer_batch_size = int(
                    st.selectbox(
                        "Batch size inferencia",
                        options=[4, 8, 16, 32],
                        index=2,
                        key="nlp_sev_embeddings_ft_batch_size",
                    )
                )
                transformer_max_length = int(
                    st.selectbox(
                        "Max tokens inferencia",
                        options=[64, 128, 256, 384, 512],
                        index=2,
                        key="nlp_sev_embeddings_ft_max_length",
                    )
                )
                st.caption(f"Modelo seleccionado: {transformer_model_path}")
                max_features = 0
                dims = 0
            else:
                selected_text_col = st.selectbox(
                    "Columna de texto para embeddings",
                    text_cols,
                    key="nlp_sev_embeddings_col",
                )
                max_features = st.slider("Max TF-IDF features", 500, 10000, 4000, 500, key="nlp_sev_embeddings_max_features")
                dims = st.slider("Dimensiones", 2, 128, 32, 2, key="nlp_sev_embeddings_dims")
            if st.button("Generar embeddings", key="nlp_sev_generate_embeddings"):
                run_id = _new_run_id("embeddings")
                try:
                    embeddings_df, embed_cols, meta = generate_text_embeddings(
                        embed_source_df,
                        text_col=selected_text_col,
                        method=method,
                        n_components=int(dims),
                        max_features=int(max_features),
                        transformer_model_path=transformer_model_path,
                        transformer_batch_size=int(transformer_batch_size),
                        transformer_max_length=int(transformer_max_length),
                    )
                except Exception as exc:
                    st.error(f"No se pudieron generar embeddings: {exc}")
                else:
                    st.session_state["nlp_sev_embeddings_df"] = embeddings_df
                    st.session_state["nlp_sev_embedding_cols"] = embed_cols
                    st.session_state["nlp_sev_embedding_meta"] = meta
                    st.session_state["nlp_sev_embedding_rf_df"] = None
                    st.session_state["nlp_sev_selected_embedding_cols"] = []
                    artifact = _persist_artifact(
                        embeddings_df,
                        stage="language_modeling",
                        artifact_name="text_embeddings",
                        run_id=run_id,
                        metadata=meta,
                    )
                    st.session_state["nlp_sev_embeddings_artifact"] = artifact
                    _log_action(
                        "language_modeling",
                        "generate_embeddings",
                        meta,
                        run_id=run_id,
                    )
                    st.success(f"Embeddings generados con {len(embed_cols)} dimensiones.")

        embeddings_df = st.session_state.get("nlp_sev_embeddings_df")
        embed_cols = st.session_state.get("nlp_sev_embedding_cols") or []
        if isinstance(embeddings_df, pd.DataFrame) and embed_cols:
            meta = st.session_state.get("nlp_sev_embedding_meta") or {}
            st.json(meta)
            matrix = embeddings_df[embed_cols].to_numpy(dtype=float)
            if matrix.shape[1] >= 2:
                coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
                plot_df = pd.DataFrame(
                    {
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "severity_target": pd.to_numeric(embeddings_df["severity_target"], errors="coerce"),
                    }
                )
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color=plot_df["severity_target"].astype("Int64").astype(str),
                    title="PCA de embeddings",
                )
                st.plotly_chart(fig, width="stretch")
            st.dataframe(embeddings_df[["accident_id"] + list(embed_cols[:8])].head(20), width="stretch")

    with sub_tabs[3]:
        embeddings_df = st.session_state.get("nlp_sev_embeddings_df")
        embed_cols = st.session_state.get("nlp_sev_embedding_cols") or []
        if embeddings_df is None or embeddings_df.empty or not embed_cols:
            st.info("Genere embeddings primero.")
        else:
            selected_embedding_top_k = int(
                st.number_input(
                    "Embeddings a conservar para Train",
                    min_value=1,
                    max_value=max(1, len(embed_cols)),
                    value=min(200, max(1, len(embed_cols))),
                    step=1,
                    key="nlp_sev_embedding_rf_top_k",
                )
            )
            if st.button("Analizar embeddings con Random Forest", key="nlp_sev_rf_embeddings"):
                run_id = _new_run_id("embedding_rf")
                ranking_df = run_embedding_rf_analysis(embeddings_df, embed_cols)
                st.session_state["nlp_sev_embedding_rf_df"] = ranking_df
                selected_embedding_cols = _select_top_embedding_features(
                    ranking_df,
                    top_k=selected_embedding_top_k,
                )
                st.session_state["nlp_sev_selected_embedding_cols"] = selected_embedding_cols
                if not ranking_df.empty:
                    _persist_artifact(
                        ranking_df,
                        stage="language_modeling",
                        artifact_name="embedding_rf_ranking",
                        run_id=run_id,
                        metadata={
                            "embedding_cols": list(embed_cols),
                            "selected_top_k": int(selected_embedding_top_k),
                            "selected_embedding_cols": list(selected_embedding_cols),
                        },
                    )
                _log_action(
                    "language_modeling",
                    "rf_embeddings",
                    {
                        "rows": int(len(ranking_df)),
                        "embedding_cols": len(embed_cols),
                        "selected_top_k": int(selected_embedding_top_k),
                        "selected_embedding_cols": list(selected_embedding_cols),
                    },
                    run_id=run_id,
                )
                st.success(
                    f"Analisis RF ejecutado. Se seleccionaron {len(selected_embedding_cols)} embeddings para Train."
                )
            ranking_df = st.session_state.get("nlp_sev_embedding_rf_df")
            if isinstance(ranking_df, pd.DataFrame) and not ranking_df.empty:
                selected_embedding_cols = [
                    col
                    for col in (st.session_state.get("nlp_sev_selected_embedding_cols") or [])
                    if col in set(ranking_df["variable"].astype(str).tolist())
                ]
                if selected_embedding_cols:
                    st.caption(
                        f"Train usara {len(selected_embedding_cols)} embeddings seleccionados por RF."
                    )
                    st.code(", ".join(selected_embedding_cols[:20]) + (" ..." if len(selected_embedding_cols) > 20 else ""))
                st.dataframe(ranking_df.head(50), width="stretch")

    _render_registry_caption()


def _append_session_model_result(result: Dict[str, object]) -> None:
    current = st.session_state.get("nlp_sev_model_results", [])
    st.session_state["nlp_sev_model_results"] = current + [result]


def _render_train_tab() -> None:
    st.subheader("Train")
    features_df = st.session_state.get("nlp_sev_features_df")
    if features_df is None or features_df.empty:
        st.info("Ejecute Feature engineering primero.")
        return

    embeddings_df = st.session_state.get("nlp_sev_embeddings_df")
    selected_embedding_cols = [
        col
        for col in (st.session_state.get("nlp_sev_selected_embedding_cols") or [])
        if isinstance(col, str)
        and isinstance(embeddings_df, pd.DataFrame)
        and col in embeddings_df.columns
    ]
    active_df = _build_train_dataset_with_selected_embeddings(
        features_df,
        embeddings_df,
        selected_embedding_cols=selected_embedding_cols,
    )
    if active_df is None or active_df.empty:
        active_df = features_df

    feature_group_options = ["Solo flujo"]
    if _embedding_feature_columns(active_df):
        feature_group_options.extend(["Solo embeddings", "Todo"])

    if _embedding_feature_columns(active_df):
        if selected_embedding_cols:
            st.caption(
                f"Dataset de train cargado con {len(_embedding_feature_columns(active_df))} embeddings seleccionados por Analisis RF."
            )
        else:
            st.caption(
                f"Dataset de train cargado con todas las dimensiones de embeddings disponibles ({len(_embedding_feature_columns(active_df))})."
            )

    sub_tabs = st.tabs(
        [
            "RF + XGBoost",
            "RF + XGBoost + CV",
            "Elastic Net",
            "SVM + RFE",
            "Comparacion controlada",
        ]
    )

    with sub_tabs[0]:
        feature_group = st.selectbox(
            "Feature group",
            feature_group_options,
            key="nlp_sev_holdout_feature_group",
        )
        feature_count = max(1, len(_resolve_feature_group(active_df, feature_group)))
        st.caption(f"Variables disponibles para este entrenamiento: {feature_count}")
        split_mode = st.selectbox("Split", ["Temporal", "Estratificado"], key="nlp_sev_holdout_split_mode")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="nlp_sev_holdout_test_size")
        top_k = st.slider(
            "Top K por RF",
            1,
            feature_count,
            min(100, feature_count),
            1,
            key="nlp_sev_holdout_top_k",
        )
        random_state = st.number_input("Random state", 0, 9999, 42, 1, key="nlp_sev_holdout_random_state")
        tune_xgb = st.checkbox(
            "Optimizar hiperparametros XGBoost",
            value=True,
            key="nlp_sev_holdout_tune_xgb",
        )
        tune_col1, tune_col2 = st.columns(2)
        with tune_col1:
            tuning_profile = st.selectbox(
                "Estrategia de busqueda XGBoost",
                ["Rapida", "Amplia", "GridSearch original"],
                index=0,
                key="nlp_sev_holdout_xgb_tuning_profile",
                disabled=not tune_xgb,
            )
        with tune_col2:
            tuning_folds = int(
                st.slider(
                    "Folds para tuning",
                    2,
                    5,
                    3,
                    1,
                    key="nlp_sev_holdout_xgb_tuning_folds",
                    disabled=not tune_xgb,
                )
            )
        if tune_xgb:
            st.caption(_xgb_search_strategy_help(tuning_profile))
        if st.button("Entrenar RF y XGBoost", key="nlp_sev_train_holdout"):
            run_id = _new_run_id("train_holdout")
            try:
                payload = train_rf_xgb_holdout(
                    active_df,
                    feature_group=feature_group,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    split_mode=split_mode,
                    top_k=int(top_k),
                    tune_hyperparameters=bool(tune_xgb),
                    tuning_folds=int(tuning_folds),
                    tuning_profile=str(tuning_profile),
                )
            except Exception as exc:
                st.error(f"No se pudo entrenar RF/XGBoost: {exc}")
            else:
                if isinstance(payload.get("ranking_df"), pd.DataFrame):
                    _persist_artifact(
                        payload["ranking_df"],
                        stage="train",
                        artifact_name="rf_feature_ranking",
                        run_id=run_id,
                        metadata={
                            "feature_group": feature_group,
                            "selected_top_k": int(top_k),
                            "selected_features": payload.get("selected_cols") or [],
                        },
                    )
                if isinstance(payload.get("xgb_search_df"), pd.DataFrame) and not payload["xgb_search_df"].empty:
                    _persist_artifact(
                        payload["xgb_search_df"],
                        stage="train",
                        artifact_name="xgb_hyperparameter_search",
                        run_id=run_id,
                        metadata={
                            "feature_group": feature_group,
                            "selected_top_k": int(top_k),
                            "tuning_profile": str(tuning_profile),
                            "tuning_folds": int(tuning_folds),
                        },
                    )
                for result in payload["results"]:
                    _record_model_result(
                        run_id=run_id,
                        stage="train",
                        model_name=result["model_name"],
                        feature_group=feature_group,
                        metrics=result["metrics"],
                        params=result["params"],
                        metadata={
                            "split_mode": payload["split_meta"].get("split_mode"),
                            "selected_features": payload["selected_cols"],
                            "selected_top_k": int(top_k),
                            "balancing_meta": payload.get("balancing_meta"),
                            "xgb_best_score": payload.get("xgb_best_score"),
                            "tuning_profile": str(tuning_profile) if tune_xgb else "Sin busqueda",
                        },
                    )
                    _append_session_model_result(result)
                st.success("RF selector + XGBoost entrenados.")
                st.caption(
                    f"Variables seleccionadas por RF: {len(payload.get('selected_cols') or [])} de {feature_count}."
                )
                if payload.get("xgb_best_score") is not None:
                    st.caption(f"Mejor score de validacion XGBoost: {float(payload['xgb_best_score']):.4f}")
                st.json(
                    {
                        "selected_features_preview": (payload.get("selected_cols") or [])[:30],
                        "xgb_params": (payload.get("results") or [{}])[0].get("params"),
                        "balancing_meta": payload.get("balancing_meta"),
                    }
                )
                st.dataframe(payload["ranking_df"].head(30), width="stretch")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "model_name": item["model_name"],
                                **{
                                    key: value
                                    for key, value in (item.get("metrics") or {}).items()
                                    if isinstance(value, (int, float, str)) and key not in {"confusion_matrix", "labels"}
                                },
                            }
                            for item in payload["results"]
                        ]
                    ),
                    width="stretch",
                )
                for result in payload["results"]:
                    _render_confusion_matrix_summary(
                        str(result.get("model_name") or "Modelo"),
                        result.get("metrics") or {},
                    )

    with sub_tabs[1]:
        feature_group = st.selectbox("Feature group", feature_group_options, key="nlp_sev_cv_feature_group")
        st.caption(
            f"Variables disponibles para este entrenamiento: {len(_resolve_feature_group(active_df, feature_group))}"
        )
        folds = st.slider("Folds", 2, 8, 5, 1, key="nlp_sev_cv_folds")
        random_state = st.number_input("Random state", 0, 9999, 42, 1, key="nlp_sev_cv_random_state")
        if st.button("Ejecutar validacion cruzada", key="nlp_sev_run_cv"):
            run_id = _new_run_id("train_cv")
            try:
                cv_df = train_rf_xgb_cv(
                    active_df,
                    feature_group=feature_group,
                    random_state=int(random_state),
                    folds=int(folds),
                )
            except Exception as exc:
                st.error(f"No se pudo ejecutar la validacion cruzada: {exc}")
            else:
                _persist_artifact(
                    cv_df,
                    stage="train",
                    artifact_name="rf_xgb_cv",
                    run_id=run_id,
                    metadata={"feature_group": feature_group, "folds": int(folds)},
                )
                for row in cv_df.to_dict(orient="records"):
                    metrics = {
                        key: row[key]
                        for key in row
                        if key.endswith("_mean") or key.endswith("_std")
                    }
                    _record_model_result(
                        run_id=run_id,
                        stage="train_cv",
                        model_name=str(row["model_name"]),
                        feature_group=feature_group,
                        metrics=metrics,
                        params={"folds": int(folds)},
                        metadata={"cv": True},
                    )
                st.success("Validacion cruzada finalizada.")
                st.dataframe(cv_df, width="stretch")

    with sub_tabs[2]:
        feature_group = st.selectbox("Feature group", feature_group_options, key="nlp_sev_elastic_group")
        st.caption(
            f"Variables disponibles para este entrenamiento: {len(_resolve_feature_group(active_df, feature_group))}"
        )
        split_mode = st.selectbox("Split", ["Temporal", "Estratificado"], key="nlp_sev_elastic_split")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="nlp_sev_elastic_test")
        random_state = st.number_input("Random state", 0, 9999, 42, 1, key="nlp_sev_elastic_rs")
        if st.button("Entrenar Elastic Net", key="nlp_sev_train_elastic"):
            run_id = _new_run_id("elastic_net")
            try:
                result = train_elastic_net_holdout(
                    active_df,
                    feature_group=feature_group,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    split_mode=split_mode,
                )
            except Exception as exc:
                st.error(f"No se pudo entrenar Elastic Net: {exc}")
            else:
                _record_model_result(
                    run_id=run_id,
                    stage="train",
                    model_name=result["model_name"],
                    feature_group=feature_group,
                    metrics=result["metrics"],
                    params=result["best_params"],
                    metadata=result["split_meta"],
                )
                _append_session_model_result(result)
                st.success("Elastic Net entrenado.")
                st.json(result)

    with sub_tabs[3]:
        feature_group = st.selectbox("Feature group", feature_group_options, key="nlp_sev_svm_group")
        st.caption(
            f"Variables disponibles para este entrenamiento: {len(_resolve_feature_group(active_df, feature_group))}"
        )
        split_mode = st.selectbox("Split", ["Temporal", "Estratificado"], key="nlp_sev_svm_split")
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="nlp_sev_svm_test")
        max_k = max(2, len(_resolve_feature_group(active_df, feature_group)))
        k_features = st.slider("RFE top K", 2, max_k, min(20, max_k), 1, key="nlp_sev_svm_k")
        random_state = st.number_input("Random state", 0, 9999, 42, 1, key="nlp_sev_svm_rs")
        if st.button("Entrenar SVM + RFE", key="nlp_sev_train_svm"):
            run_id = _new_run_id("svm_rfe")
            try:
                result = train_svm_rfe_holdout(
                    active_df,
                    feature_group=feature_group,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    split_mode=split_mode,
                    k_features=int(k_features),
                )
            except Exception as exc:
                st.error(f"No se pudo entrenar SVM + RFE: {exc}")
            else:
                _persist_artifact(
                    result["ranking_df"],
                    stage="train",
                    artifact_name="svm_rfe_ranking",
                    run_id=run_id,
                    metadata={"feature_group": feature_group, "k_features": int(k_features)},
                )
                _record_model_result(
                    run_id=run_id,
                    stage="train",
                    model_name=result["model_name"],
                    feature_group=feature_group,
                    metrics=result["metrics"],
                    params={"k_features": int(k_features)},
                    metadata={
                        "split_mode": result["split_meta"].get("split_mode"),
                        "selected_cols": result["selected_cols"],
                    },
                )
                _append_session_model_result(result)
                st.success("SVM + RFE entrenado.")
                st.dataframe(result["ranking_df"].head(30), width="stretch")
                st.json(result["metrics"])

    with sub_tabs[4]:
        feature_group = st.selectbox(
            "Feature group",
            feature_group_options,
            key="nlp_sev_compare_group",
        )
        feature_count = max(1, len(_resolve_feature_group(active_df, feature_group)))
        max_feature_cap = min(100, feature_count)
        split_mode = st.selectbox(
            "Split compartido",
            ["Temporal", "Estratificado"],
            key="nlp_sev_compare_split_mode",
        )
        test_size = st.slider(
            "Test size compartido",
            0.1,
            0.4,
            0.2,
            0.05,
            key="nlp_sev_compare_test_size",
        )
        max_features_per_model = st.slider(
            "Numero de variables por modelo",
            1,
            max_feature_cap,
            max_feature_cap,
            1,
            key="nlp_sev_compare_max_features",
        )
        random_state = st.number_input(
            "Random state compartido",
            0,
            9999,
            42,
            1,
            key="nlp_sev_compare_random_state",
        )
        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            xgb_tuning_profile = st.selectbox(
                "Busqueda XGBoost",
                ["Rapida", "Amplia", "GridSearch original"],
                index=0,
                key="nlp_sev_compare_xgb_profile",
            )
        with compare_col2:
            tuning_folds = int(
                st.slider(
                    "Folds internos compartidos",
                    2,
                    5,
                    3,
                    1,
                    key="nlp_sev_compare_common_folds",
                )
            )
        _render_controlled_comparison_protocol(
            feature_group=feature_group,
            feature_count=feature_count,
            feature_count_per_model=int(max_features_per_model),
            split_mode=split_mode,
            test_size=float(test_size),
            random_state=int(random_state),
            xgb_tuning_profile=str(xgb_tuning_profile),
            tuning_folds=int(tuning_folds),
        )
        if st.button("Ejecutar comparacion controlada", key="nlp_sev_train_comparison"):
            run_id = _new_run_id("train_comparison")
            comparison_progress = st.progress(0)
            comparison_status = st.empty()

            def _update_comparison_progress(value: int, message: str) -> None:
                comparison_progress.progress(int(value))
                comparison_status.caption(message)

            try:
                payload = train_model_comparison_holdout(
                    active_df,
                    feature_group=feature_group,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    split_mode=split_mode,
                    max_features_per_model=int(max_features_per_model),
                    xgb_tuning_profile=str(xgb_tuning_profile),
                    tuning_folds=int(tuning_folds),
                    progress_callback=_update_comparison_progress,
                )
            except Exception as exc:
                comparison_progress.empty()
                comparison_status.empty()
                st.error(f"No se pudo ejecutar la comparacion controlada: {exc}")
            else:
                protocol = payload.get("protocol") or {}
                _persist_artifact(
                    payload["comparison_df"],
                    stage="train",
                    artifact_name="controlled_model_comparison",
                    run_id=run_id,
                    metadata=protocol,
                )
                _persist_artifact(
                    payload["predictions_df"],
                    stage="train",
                    artifact_name="controlled_model_predictions",
                    run_id=run_id,
                    metadata=protocol,
                )
                if isinstance(payload.get("protocol_df"), pd.DataFrame) and not payload["protocol_df"].empty:
                    _persist_artifact(
                        payload["protocol_df"],
                        stage="train",
                        artifact_name="controlled_model_protocol",
                        run_id=run_id,
                        metadata=protocol,
                    )

                artifact_specs = [
                    ("rf_feature_ranking", payload["results"][0].get("ranking_df")),
                    ("xgb_hyperparameter_search", payload["results"][0].get("search_df")),
                    ("elastic_net_coefficients", payload["results"][1].get("ranking_df")),
                    ("elastic_net_hyperparameter_search", payload["results"][1].get("search_df")),
                    ("svm_rfe_ranking", payload["results"][2].get("ranking_df")),
                    ("svm_rfe_hyperparameter_search", payload["results"][2].get("search_df")),
                ]
                for artifact_name, artifact_df in artifact_specs:
                    if isinstance(artifact_df, pd.DataFrame) and not artifact_df.empty:
                        _persist_artifact(
                            artifact_df,
                            stage="train",
                            artifact_name=artifact_name,
                            run_id=run_id,
                            metadata=protocol,
                        )

                for result in payload["results"]:
                    metadata = {
                        **protocol,
                        "comparison_protocol": True,
                        "selected_cols": result.get("selected_cols") or [],
                        "feature_strategy": result.get("feature_strategy"),
                        "balancing_meta": result.get("balancing_meta"),
                    }
                    _record_model_result(
                        run_id=run_id,
                        stage="train_comparison",
                        model_name=str(result["model_name"]),
                        feature_group=feature_group,
                        metrics=result["metrics"],
                        params=result["params"],
                        metadata=metadata,
                    )
                    _append_session_model_result(result)

                _log_action(
                    "train",
                    "controlled_model_comparison",
                    {
                        "feature_group": feature_group,
                        "protocol": protocol,
                        "models": [result["model_name"] for result in payload["results"]],
                    },
                    run_id=run_id,
                )

                st.success("Comparacion controlada finalizada.")
                _render_controlled_comparison_protocol(
                    feature_group=feature_group,
                    feature_count=feature_count,
                    feature_count_per_model=int(max_features_per_model),
                    split_mode=split_mode,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    xgb_tuning_profile=str(xgb_tuning_profile),
                    tuning_folds=int(tuning_folds),
                    protocol=protocol,
                )
                st.caption(
                    "Todas las matrices de confusion usan la misma base de test: "
                    f"{int(protocol.get('test_rows') or 0)} filas | "
                    f"clases test={protocol.get('test_class_counts') or {}}"
                )
                st.dataframe(payload["comparison_df"], width="stretch")
                st.dataframe(payload["params_df"], width="stretch")
                for result in payload["results"]:
                    st.caption(
                        f"{result['model_name']}: {len(result.get('selected_cols') or [])} variables seleccionadas."
                    )
                    _render_confusion_matrix_summary(
                        str(result["model_name"]),
                        result.get("metrics") or {},
                    )

    _render_registry_caption()


def _render_experiments_tab() -> None:
    st.subheader("Experiments")
    features_df = st.session_state.get("nlp_sev_features_df")
    if features_df is None or features_df.empty:
        st.info("Ejecute Feature engineering primero.")
        return

    active_df = st.session_state.get("nlp_sev_embeddings_df")
    if active_df is None or active_df.empty:
        active_df = features_df

    feature_group_options = ["Solo flujo"]
    if _embedding_feature_columns(active_df):
        feature_group_options.extend(["Solo embeddings", "Todo"])

    exp_tabs = st.tabs(
        [
            "Variables predictoras",
            "Datos granulares",
            "Comparativa",
            "BERTopic",
        ]
    )

    with exp_tabs[0]:
        feature_group = st.selectbox("Feature group", feature_group_options, key="nlp_sev_exp_feature_group")
        if st.button("Evaluar variables predictoras", key="nlp_sev_eval_vars"):
            run_id = _new_run_id("predictive_variables")
            ranking_df = evaluate_predictive_variables(active_df, feature_group=feature_group)
            if ranking_df.empty:
                st.warning("No se pudo generar ranking.")
            else:
                _persist_artifact(
                    ranking_df,
                    stage="experiments",
                    artifact_name="predictive_variables",
                    run_id=run_id,
                    metadata={"feature_group": feature_group},
                )
                _log_action(
                    "experiments",
                    "predictive_variables",
                    {"feature_group": feature_group, "rows": int(len(ranking_df))},
                    run_id=run_id,
                )
                st.dataframe(ranking_df.head(50), width="stretch")

    with exp_tabs[1]:
        granular_df = st.session_state.get("nlp_sev_granular_df")
        viz_df = build_granular_visualization_df(granular_df, features_df)
        if viz_df.empty:
            st.info("No hay datos granulares.")
        else:
            available_metrics = [metric for metric in GRANULAR_METRIC_NAMES if metric in viz_df.columns]
            if not available_metrics:
                st.info("No hay metricas disponibles para visualizar.")
            else:
                metric = st.selectbox("Metrica", available_metrics, key="nlp_sev_granular_metric")
                category = st.selectbox(
                    "Categoria",
                    sorted(viz_df["category_label"].dropna().astype(str).unique().tolist()),
                    key="nlp_sev_granular_category",
                )
                anchor = st.selectbox(
                    "Portico",
                    sorted(viz_df["anchor"].dropna().astype(str).unique().tolist()),
                    key="nlp_sev_granular_anchor",
                )
                plot_df = viz_df[
                    (viz_df["category_label"].astype(str) == category)
                    & (viz_df["anchor"].astype(str) == anchor)
                ].copy()
                if plot_df.empty:
                    st.info("No hay datos para la combinacion seleccionada.")
                else:
                    grouped = (
                        plot_df.groupby(["severity_target", "time_offset"], dropna=False)[metric]
                        .mean()
                        .reset_index()
                        .sort_values(["severity_target", "time_offset"])
                    )
                    grouped["severity_label"] = grouped["severity_target"].map({0: "No severe", 1: "Severe"}).fillna("NA")
                    fig = px.line(
                        grouped,
                        x="time_offset",
                        y=metric,
                        color="severity_label",
                        markers=True,
                        title=f"{metric} por desfase temporal | {anchor} | {category}",
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(grouped, width="stretch")

    with exp_tabs[2]:
        comparison_df = _load_model_results()
        if comparison_df.empty:
            st.info("No hay resultados persistidos.")
        else:
            st.dataframe(comparison_df, width="stretch")

    with exp_tabs[3]:
        topic_source_df = st.session_state.get("nlp_sev_language_df")
        if topic_source_df is None or topic_source_df.empty:
            topic_source_df = features_df
        text_cols = [col for col in topic_source_df.columns if "text_bert" in col or col == "descripcion"]
        text_col = st.selectbox("Columna de texto", text_cols, key="nlp_sev_topic_text_col")
        n_topics = st.slider("Numero de topicos", 2, 12, 5, 1, key="nlp_sev_topic_n")
        top_terms = st.slider("Top terms", 1, 8, 3, 1, key="nlp_sev_topic_terms")
        if BERTopic is None:
            st.caption("BERTopic no esta instalado. Se usara un fallback NMF reproducible.")
        if st.button("Ejecutar BERTopic / fallback", key="nlp_sev_run_topic_model"):
            run_id = _new_run_id("topic_model")
            try:
                topic_df, meta = run_topic_analysis(
                    topic_source_df,
                    text_col=text_col,
                    n_topics=int(n_topics),
                    top_terms=int(top_terms),
                )
            except Exception as exc:
                st.error(f"No se pudo ejecutar el analisis de topicos: {exc}")
            else:
                st.session_state["nlp_sev_topic_df"] = topic_df
                st.session_state["nlp_sev_topic_meta"] = meta
                _persist_artifact(
                    topic_df,
                    stage="experiments",
                    artifact_name="topic_model",
                    run_id=run_id,
                    metadata=meta,
                )
                _log_action("experiments", "topic_model", meta, run_id=run_id)
                st.success("Analisis de topicos completado.")
        topic_df = st.session_state.get("nlp_sev_topic_df")
        meta = st.session_state.get("nlp_sev_topic_meta") or {}
        if isinstance(topic_df, pd.DataFrame) and not topic_df.empty:
            st.json(meta)
            grouped = (
                topic_df.groupby(["topic_label", "severity_target"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values(["topic_label", "severity_target"])
            )
            grouped["severity_label"] = grouped["severity_target"].map({0: "No severe", 1: "Severe"}).fillna("NA")
            fig = px.bar(
                grouped,
                x="topic_label",
                y="count",
                color="severity_label",
                barmode="group",
                title="Distribucion de topicos por severidad",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(grouped, width="stretch")

    _render_registry_caption()


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    _init_state()
    if set_page_config:
        st.set_page_config(page_title="NLP in Severity", layout="wide")
    st.title("NLP in Severity")
    st.caption(
        "Migracion del experimento NLP legacy a la interfaz actual, con feature engineering granular "
        "a 1 minuto, persistencia DuckDB y logging reproducible."
    )

    if show_exit_button and st.sidebar.button("Cerrar app", key="nlp_sev_close_app"):
        raise SystemExit(0)

    tabs = st.tabs(
        [
            "Eventos",
            "Feature engineering",
            "Language modeling",
            "Train",
            "Experiments",
        ]
    )
    with tabs[0]:
        _render_events_tab()
    with tabs[1]:
        _render_feature_engineering_tab()
    with tabs[2]:
        _render_language_modeling_tab()
    with tabs[3]:
        _render_train_tab()
    with tabs[4]:
        _render_experiments_tab()


if __name__ == "__main__":
    main()
