#!/usr/bin/env python3
"""
Streamlit app for the "NLP in Severity" pipeline.
"""
from __future__ import annotations

import hashlib
import json
import itertools
import math
import os
import random
import re
import shutil
import tempfile
import time
import traceback
import uuid
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:  # pragma: no cover
    optuna = None
    TPESampler = None

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
LEGACY_LOCAL_TRANSFORMER_MODEL_DIR = ROOT_DIR / "NLP" / "bert_base_chile_text_bert_128t"
LOCAL_TRANSFORMER_MODEL_LOCATIONS: Tuple[Tuple[str, Path], ...] = (
    (
        "Local · dccuchile/bert-base-spanish-wwm-cased (safetensors)",
        ROOT_DIR / "NLP" / "hf_models" / "dccuchile_bert_base_spanish_wwm_cased_safetensors",
    ),
    (
        f"Local · {LEGACY_LOCAL_TRANSFORMER_MODEL_DIR.name}",
        LEGACY_LOCAL_TRANSFORMER_MODEL_DIR,
    ),
)
PAPER_REPLICATION_DIR = MODULE_RESULTS_DIR / "paper_replication"
PAPER_FROZEN_DATASET_PATH = ROOT_DIR / "NLP" / "Dataframes" / "resultado.pkl"
PAPER_LATEX_DIR = ROOT_DIR / "NLP" / "Latex"
PAPER_LATEX_IMAGES_DIR = PAPER_LATEX_DIR / "images"
PAPER_LATEX_GENERATED_DIR = PAPER_LATEX_DIR / "generated"

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
PAPER_K_GRID = [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500, 632]
PAPER_K_GRID_SELECTION_MIN = 2
PAPER_K_GRID_SELECTION_MAX = 5
PAPER_K_GRID_DEFAULT_SELECTION = [10, 50, 100, 200, 632]
PAPER_CLASS_LABELS = {0: "No-MARC", 1: "MARC"}
PAPER_PROTOCOL = {
    "split_mode": "Temporal",
    "test_size": 0.2,
    "random_state": 42,
    "model_family": "XGBoost",
    "feature_groups": {
        "M1": "Solo flujo",
        "M2": "Solo embeddings",
        "M3": "Todo",
    },
}
PAPER_EXPECTED_COUNTS = {
    "rows": 2070,
    "flow_features": 432,
    "embedding_features": 200,
    "total_features": 632,
    "train_rows": 1656,
    "test_rows": 414,
    "train_class_counts": {"0": 376, "1": 1280},
    "test_class_counts": {"0": 94, "1": 320},
}
PAPER_COMPARISON_TOLERANCE = 0.001
PAPER_VALIDATION_ALPHA = 0.3
PAPER_PROTOCOL_VERSION = "paper_replication_checkpoint_v2"
PAPER_ROUTE_SELECTION_DEFAULTS = {
    "run_frozen": True,
    "run_raw": True,
    "run_update_embeddings": False,
}
PAPER_UPDATE_EMB_TOP_K = 200
PAPER_CV_FOLDS_DEFAULT = 5
PAPER_CV_FOLDS_MIN = 2
PAPER_CV_FOLDS_MAX = 10
PAPER_OPTIMIZATION_BACKEND_DEFAULT = "gridsearch"
PAPER_OPTUNA_TRIALS_DEFAULT = 24
PAPER_SCORING_METRICS = ["f1", "roc_auc", "recall", "precision", "accuracy"]
PAPER_SCORING_METRIC_DEFAULT = "f1"
PAPER_SCORING_METRIC_LABELS = {
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "recall": "Recall",
    "precision": "Precision",
    "accuracy": "Accuracy",
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
    "nlp_sev_paper_replication_payload": None,
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


def _to_json_safe(value: object) -> object:
    return json.loads(json.dumps(value, default=_json_default, ensure_ascii=True))


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_to_json_safe(payload), handle, ensure_ascii=True, indent=2)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _load_json_file(path: Path, *, default: object = None) -> object:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_pickle(payload: object, path: Path) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target_path.stem}_", suffix=".tmp", dir=str(target_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pd.to_pickle(payload, tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _load_pickle_file(path: Path, *, default: object = None) -> object:
    file_path = Path(path)
    if not file_path.exists():
        return default
    return pd.read_pickle(file_path)


def _hash_series_signature(series: pd.Series) -> bytes:
    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64, copy=False)
    return hashed.tobytes()


def _frame_signature(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    include_index: bool = True,
) -> str:
    if df is None or df.empty:
        return "empty"
    sha = hashlib.sha256()
    work = df.copy()
    if include_index:
        try:
            index_arr = np.asarray(work.index.to_numpy(), dtype=np.int64)
            sha.update(index_arr.tobytes())
        except Exception:
            sha.update(str(list(work.index)).encode("utf-8"))
    for col in columns:
        if col not in work.columns:
            sha.update(f"missing::{col}".encode("utf-8"))
            continue
        series = work[col]
        if isinstance(series.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(series):
            normalized = pd.to_datetime(series, errors="coerce").astype("string").fillna("<NA>")
        else:
            normalized = series.astype("string").fillna("<NA>")
        sha.update(str(col).encode("utf-8"))
        sha.update(_hash_series_signature(normalized))
    sha.update(str(len(work)).encode("utf-8"))
    return sha.hexdigest()


class PaperReplicationBlockedError(RuntimeError):
    """Expected business block during paper replication."""


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


def _paper_fingerprint_feature_artifact_row(row: Optional[pd.Series]) -> Dict[str, object]:
    if row is None:
        return {"kind": "rebuild_from_events"}
    return {
        "kind": "severity_features_artifact",
        "artifact_id": str(row.get("artifact_id") or ""),
        "run_id": str(row.get("run_id") or ""),
        "created_at": str(row.get("created_at") or ""),
        "db_path": str(row.get("db_path") or ""),
        "table_name": str(row.get("table_name") or ""),
        "row_count": int(row.get("row_count") or 0),
    }


def _paper_resolve_feature_bundle_override(
    feature_artifact_row: Optional[pd.Series],
) -> Dict[str, object]:
    if feature_artifact_row is None:
        return {
            "features_df": None,
            "granular_df": pd.DataFrame(),
            "feature_artifact": {},
            "fingerprint": {"kind": "rebuild_from_events"},
        }
    features_df, granular_df, feature_artifact = _load_feature_bundle_from_catalog_row(feature_artifact_row)
    return {
        "features_df": features_df,
        "granular_df": granular_df if isinstance(granular_df, pd.DataFrame) else pd.DataFrame(),
        "feature_artifact": feature_artifact or {},
        "fingerprint": _paper_fingerprint_feature_artifact_row(feature_artifact_row),
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


def _subprogress_callback(
    progress_callback: Optional[Callable[[int, str], None]],
    *,
    start: int,
    end: int,
    prefix: str = "",
) -> Optional[Callable[[int, str], None]]:
    if progress_callback is None:
        return None
    start_value = int(start)
    end_value = int(end)
    span = max(0, end_value - start_value)

    def _callback(value: int, message: str) -> None:
        clipped = max(0, min(100, int(value)))
        scaled = start_value + int(round(span * (clipped / 100.0)))
        rendered = f"{prefix}{message}" if prefix else str(message)
        _emit_progress(progress_callback, scaled, rendered)

    return _callback


def _paper_checkpoint_root(*, checkpoint_root: Optional[Path] = None) -> Path:
    return Path(checkpoint_root) if checkpoint_root is not None else PAPER_REPLICATION_DIR


def _paper_run_dir(run_id: str, *, checkpoint_root: Optional[Path] = None) -> Path:
    return _paper_checkpoint_root(checkpoint_root=checkpoint_root) / str(run_id)


def _paper_run_paths(run_dir: Path) -> Dict[str, Path]:
    run_path = Path(run_dir)
    return {
        "run_dir": run_path,
        "manifest": run_path / "manifest.json",
        "live_status": run_path / "live_status.json",
        "live_events": run_path / "live_events.jsonl",
        "frozen_dir": run_path / "frozen",
        "raw_build_dir": run_path / "raw_build",
        "raw_dir": run_path / "raw",
        "update_emb_build_dir": run_path / "update_emb_build",
        "update_emb_dir": run_path / "update_emb",
        "compare_dir": run_path / "compare",
        "export_dir": run_path / "export",
    }


def _ensure_paper_run_dirs(paths: Dict[str, Path]) -> None:
    for key in ["run_dir", "frozen_dir", "raw_build_dir", "raw_dir", "update_emb_build_dir", "update_emb_dir", "compare_dir", "export_dir"]:
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


def _paper_file_fingerprint(path: object) -> Dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "path": str(file_path),
            "exists": False,
            "size": None,
            "mtime": None,
            "sha256": None,
        }
    sha = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256": sha.hexdigest(),
    }


def _paper_model_dir(paths: Dict[str, Path], route_name: str, model_code: str) -> Path:
    if str(route_name) == "frozen":
        base_dir = Path(paths["frozen_dir"])
    elif str(route_name) == "update_emb":
        base_dir = Path(paths["update_emb_dir"])
    else:
        base_dir = Path(paths["raw_dir"])
    return base_dir / "models" / str(model_code)


def _paper_k_result_path(model_dir: Path, k_value: int) -> Path:
    return Path(model_dir) / "k_results" / f"k_{int(k_value):03d}.json"


def _paper_model_result_paths(model_dir: Path) -> Dict[str, Path]:
    model_path = Path(model_dir)
    return {
        "dir": model_path,
        "summary": model_path / "final_summary.json",
        "ranking": model_path / "ranking.pkl",
        "k_search": model_path / "k_search.pkl",
        "search": model_path / "search.pkl",
        "predictions": model_path / "predictions.pkl",
        "metrics": model_path / "metrics.json",
    }


def _paper_raw_build_paths(paths: Dict[str, Path]) -> Dict[str, Path]:
    base_dir = Path(paths["raw_build_dir"])
    return {
        "dir": base_dir,
        "features": base_dir / "features.pkl",
        "granular": base_dir / "granular.pkl",
        "feature_ranking": base_dir / "feature_ranking.pkl",
        "embeddings": base_dir / "embeddings.pkl",
        "embedding_ranking": base_dir / "embedding_ranking.pkl",
        "selected_embedding_cols": base_dir / "selected_embedding_cols.json",
        "dataset": base_dir / "dataset.pkl",
        "payload": base_dir / "payload.pkl",
        "meta": base_dir / "meta.json",
    }


def _paper_update_emb_build_paths(paths: Dict[str, Path]) -> Dict[str, Path]:
    base_dir = Path(paths["update_emb_build_dir"])
    return {
        "dir": base_dir,
        "frozen_base": base_dir / "frozen_base.pkl",
        "features_source": base_dir / "features_source.pkl",
        "embeddings": base_dir / "embeddings.pkl",
        "embedding_ranking": base_dir / "embedding_ranking.pkl",
        "selected_embedding_cols": base_dir / "selected_embedding_cols.json",
        "dataset": base_dir / "dataset.pkl",
        "payload": base_dir / "payload.pkl",
        "meta": base_dir / "meta.json",
    }


def _paper_route_paths(paths: Dict[str, Path], route_name: str) -> Dict[str, Path]:
    if str(route_name) == "frozen":
        route_dir = Path(paths["frozen_dir"])
    elif str(route_name) == "update_emb":
        route_dir = Path(paths["update_emb_dir"])
    else:
        route_dir = Path(paths["raw_dir"])
    return {
        "dir": route_dir,
        "summary_json": route_dir / "summary.json",
        "dataset_validation": route_dir / "dataset_validation.json",
        "comparison_csv": route_dir / "comparison.csv",
        "metricas_csv": route_dir / "metricas.csv",
        "predictions_csv": route_dir / "predictions.csv",
        "m3_grid_csv": route_dir / "m3_grid.csv",
        "payload": route_dir / "route_payload.pkl",
    }


def _paper_compare_paths(paths: Dict[str, Path]) -> Dict[str, Path]:
    compare_dir = Path(paths["compare_dir"])
    return {
        "dir": compare_dir,
        "summary": compare_dir / "summary.json",
        "diff": compare_dir / "diff.csv",
        "payload": compare_dir / "payload.pkl",
    }


def _paper_export_paths(paths: Dict[str, Path]) -> Dict[str, Path]:
    export_dir = Path(paths["export_dir"])
    return {
        "dir": export_dir,
        "candidate_paths": export_dir / "candidate_paths.json",
        "promoted_paths": export_dir / "promoted_paths.json",
        "payload": export_dir / "payload.json",
        "final_payload": export_dir / "final_payload.pkl",
        "latex_candidate_dir": export_dir / "latex_candidate",
    }


def _paper_persist_model_result(model_result: Dict[str, object], model_dir: Path) -> Dict[str, str]:
    model_paths = _paper_model_result_paths(model_dir)
    Path(model_paths["dir"]).mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "model_code": model_result.get("model_code"),
        "model_title": model_result.get("model_title"),
        "feature_group": model_result.get("feature_group"),
        "candidate_feature_count": model_result.get("candidate_feature_count"),
        "selected_k": model_result.get("selected_k"),
        "selected_cols": model_result.get("selected_cols") or [],
        "split_meta": model_result.get("split_meta") or {},
        "best_params": model_result.get("best_params") or {},
        "best_cv_score": model_result.get("best_cv_score"),
        "balancing_meta": model_result.get("balancing_meta") or {},
        "optimization": model_result.get("optimization") or {},
    }
    _atomic_write_json(model_paths["summary"], summary_payload)
    _atomic_write_json(model_paths["metrics"], model_result.get("metrics") or {})
    _atomic_write_pickle(model_result.get("ranking_df"), model_paths["ranking"])
    _atomic_write_pickle(model_result.get("k_search_df"), model_paths["k_search"])
    _atomic_write_pickle(model_result.get("search_df"), model_paths["search"])
    _atomic_write_pickle(model_result.get("predictions_df"), model_paths["predictions"])
    return {key: str(value) for key, value in model_paths.items() if key != "dir"}


def _paper_load_model_result(model_dir: Path) -> Dict[str, object]:
    model_paths = _paper_model_result_paths(model_dir)
    summary_payload = _load_json_file(model_paths["summary"], default={})
    metrics_payload = _load_json_file(model_paths["metrics"], default={})
    ranking_df = _load_pickle_file(model_paths["ranking"], default=pd.DataFrame())
    k_search_df = _load_pickle_file(model_paths["k_search"], default=pd.DataFrame())
    search_df = _load_pickle_file(model_paths["search"], default=pd.DataFrame())
    predictions_df = _load_pickle_file(model_paths["predictions"], default=pd.DataFrame())
    return {
        "model_code": summary_payload.get("model_code"),
        "model_title": summary_payload.get("model_title"),
        "feature_group": summary_payload.get("feature_group"),
        "candidate_feature_count": int(summary_payload.get("candidate_feature_count") or 0),
        "selected_k": int(summary_payload.get("selected_k") or 0),
        "selected_cols": list(summary_payload.get("selected_cols") or []),
        "split_meta": summary_payload.get("split_meta") or {},
        "ranking_df": ranking_df if isinstance(ranking_df, pd.DataFrame) else pd.DataFrame(),
        "k_search_df": k_search_df if isinstance(k_search_df, pd.DataFrame) else pd.DataFrame(),
        "metrics": metrics_payload if isinstance(metrics_payload, dict) else {},
        "best_params": summary_payload.get("best_params") or {},
        "best_cv_score": summary_payload.get("best_cv_score"),
        "search_df": search_df if isinstance(search_df, pd.DataFrame) else pd.DataFrame(),
        "predictions_df": predictions_df if isinstance(predictions_df, pd.DataFrame) else pd.DataFrame(),
        "balancing_meta": summary_payload.get("balancing_meta") or {},
        "optimization": summary_payload.get("optimization") or {},
    }


def _paper_persist_route_payload(route_payload: Dict[str, object], route_paths: Dict[str, Path]) -> Dict[str, str]:
    route_dir = Path(route_paths["dir"])
    route_dir.mkdir(parents=True, exist_ok=True)
    _paper_write_json(route_paths["dataset_validation"], route_payload.get("dataset_validation") or {})
    comparison_df = route_payload.get("comparison_df")
    if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
        _paper_write_csv(comparison_df, route_paths["comparison_csv"])
    metricas_df = route_payload.get("metricas_df")
    if isinstance(metricas_df, pd.DataFrame) and not metricas_df.empty:
        _paper_write_csv(metricas_df, route_paths["metricas_csv"])
    predictions_df = route_payload.get("predictions_df")
    if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
        _paper_write_csv(predictions_df, route_paths["predictions_csv"])
    m3_grid_df = route_payload.get("m3_grid_df")
    if isinstance(m3_grid_df, pd.DataFrame) and not m3_grid_df.empty:
        _paper_write_csv(m3_grid_df, route_paths["m3_grid_csv"])
    summary_payload = {
        "route_name": route_payload.get("route_name"),
        "status": route_payload.get("status"),
        "status_message": route_payload.get("status_message"),
        "dataset_validation": route_payload.get("dataset_validation") or {},
        "route_metadata": route_payload.get("route_metadata") or {},
        "raw_build": route_payload.get("raw_build") or {},
    }
    _paper_write_json(route_paths["summary_json"], summary_payload)
    _atomic_write_pickle(route_payload, route_paths["payload"])
    return {
        "summary_json": str(route_paths["summary_json"]),
        "dataset_validation": str(route_paths["dataset_validation"]),
        "payload": str(route_paths["payload"]),
    }


def _paper_load_route_payload(route_paths: Dict[str, Path]) -> Optional[Dict[str, object]]:
    payload = _load_pickle_file(route_paths["payload"], default=None)
    return payload if isinstance(payload, dict) else None


def _paper_skipped_route_payload(
    route_name: str,
    *,
    reason: str,
    route_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "status": "skipped",
        "status_message": str(reason),
        "route_name": str(route_name),
        "route_metadata": route_metadata or {},
        "dataset_validation": {},
        "model_results": [],
        "comparison_df": pd.DataFrame(),
        "metricas_df": pd.DataFrame(),
        "predictions_df": pd.DataFrame(),
        "m3_grid_df": pd.DataFrame(),
    }


def _paper_persist_compare_payload(compare_payload: Dict[str, object], compare_paths: Dict[str, Path]) -> Dict[str, str]:
    compare_dir = Path(compare_paths["dir"])
    compare_dir.mkdir(parents=True, exist_ok=True)
    _paper_write_json(
        compare_paths["summary"],
        {
            "status": compare_payload.get("status"),
            "reason": compare_payload.get("reason"),
            "passed": compare_payload.get("passed"),
            "max_numeric_diff": compare_payload.get("max_numeric_diff"),
            "tolerance": compare_payload.get("tolerance"),
        },
    )
    diff_df = compare_payload.get("diff_df")
    if isinstance(diff_df, pd.DataFrame) and not diff_df.empty:
        _paper_write_csv(diff_df, compare_paths["diff"])
    _atomic_write_pickle(compare_payload, compare_paths["payload"])
    return {
        "summary": str(compare_paths["summary"]),
        "payload": str(compare_paths["payload"]),
    }


def _paper_load_compare_payload(compare_paths: Dict[str, Path]) -> Optional[Dict[str, object]]:
    payload = _load_pickle_file(compare_paths["payload"], default=None)
    return payload if isinstance(payload, dict) else None


def _paper_persist_raw_build_payload(
    raw_build_payload: Dict[str, object],
    raw_paths: Dict[str, Path],
) -> Dict[str, str]:
    raw_dir = Path(raw_paths["dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, artifact_path in [
        ("dataset_df", raw_paths["dataset"]),
        ("features_df", raw_paths["features"]),
        ("granular_df", raw_paths["granular"]),
        ("feature_ranking_df", raw_paths["feature_ranking"]),
        ("embeddings_df", raw_paths["embeddings"]),
        ("embedding_ranking_df", raw_paths["embedding_ranking"]),
    ]:
        frame = raw_build_payload.get(key)
        if isinstance(frame, pd.DataFrame):
            _atomic_write_pickle(frame, artifact_path)
    _atomic_write_json(raw_paths["selected_embedding_cols"], raw_build_payload.get("selected_embedding_cols") or [])
    _atomic_write_json(raw_paths["meta"], raw_build_payload.get("embedding_meta") or {})
    _atomic_write_pickle(raw_build_payload, raw_paths["payload"])
    return {key: str(value) for key, value in raw_paths.items() if key != "dir"}


def _paper_load_raw_build_payload(raw_paths: Dict[str, Path]) -> Optional[Dict[str, object]]:
    payload = _load_pickle_file(raw_paths["payload"], default=None)
    if isinstance(payload, dict):
        return payload
    dataset_df = _load_pickle_file(raw_paths["dataset"], default=None)
    if not isinstance(dataset_df, pd.DataFrame):
        return None
    return {
        "dataset_df": dataset_df,
        "features_df": _load_pickle_file(raw_paths["features"], default=pd.DataFrame()),
        "granular_df": _load_pickle_file(raw_paths["granular"], default=pd.DataFrame()),
        "feature_ranking_df": _load_pickle_file(raw_paths["feature_ranking"], default=pd.DataFrame()),
        "embeddings_df": _load_pickle_file(raw_paths["embeddings"], default=pd.DataFrame()),
        "embedding_ranking_df": _load_pickle_file(raw_paths["embedding_ranking"], default=pd.DataFrame()),
        "selected_embedding_cols": _load_json_file(raw_paths["selected_embedding_cols"], default=[]),
        "embedding_meta": _load_json_file(raw_paths["meta"], default={}),
    }


def _paper_persist_export_payload(payload: Dict[str, object], export_paths: Dict[str, Path]) -> Dict[str, str]:
    export_dir = Path(export_paths["dir"])
    export_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(export_paths["candidate_paths"], payload.get("candidate_paths") or {})
    _atomic_write_json(export_paths["promoted_paths"], payload.get("promoted_paths") or {})
    _atomic_write_json(
        export_paths["payload"],
        {
            "latex_promoted": bool(payload.get("latex_promoted")),
            "result_status": payload.get("result_status"),
            "candidate_paths": payload.get("candidate_paths") or {},
            "promoted_paths": payload.get("promoted_paths") or {},
        },
    )
    _atomic_write_pickle(payload, export_paths["final_payload"])
    return {
        "candidate_paths": str(export_paths["candidate_paths"]),
        "promoted_paths": str(export_paths["promoted_paths"]),
        "payload": str(export_paths["payload"]),
        "final_payload": str(export_paths["final_payload"]),
    }


def _paper_load_export_payload(export_paths: Dict[str, Path]) -> Dict[str, object]:
    payload = _load_pickle_file(export_paths["final_payload"], default=None)
    if isinstance(payload, dict):
        return payload
    serialized = _load_json_file(export_paths["payload"], default={})
    return {
        "candidate_paths": serialized.get("candidate_paths") or {},
        "promoted_paths": serialized.get("promoted_paths") or {},
        "latex_promoted": bool(serialized.get("latex_promoted")),
        "result_status": serialized.get("result_status"),
    }


def _paper_assemble_payload_from_checkpoint(
    *,
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    auto_resumed: bool,
    loaded_from_checkpoint: bool,
) -> Dict[str, object]:
    export_paths = _paper_export_paths(paths)
    final_payload = _load_pickle_file(export_paths["final_payload"], default=None)
    if isinstance(final_payload, dict):
        payload = dict(final_payload)
    else:
        frozen_payload = _paper_load_route_payload(_paper_route_paths(paths, "frozen")) or {}
        raw_payload = _paper_load_route_payload(_paper_route_paths(paths, "raw")) or {}
        update_emb_payload = _paper_load_route_payload(_paper_route_paths(paths, "update_emb")) or {}
        compare_payload = _paper_load_compare_payload(_paper_compare_paths(paths)) or {}
        export_payload = _paper_load_export_payload(export_paths)
        manifest_protocol = manifest.get("protocol") or {}
        route_options = (manifest_protocol.get("route_options") or PAPER_ROUTE_SELECTION_DEFAULTS)
        manifest_backend = _paper_normalize_optimization_backend(manifest_protocol.get("optimization_backend"))
        payload = {
            "run_id": str(manifest.get("run_id") or ""),
            "run_dir": str(paths["run_dir"]),
            "route_options": dict(route_options),
            "k_grid": list(manifest_protocol.get("k_grid") or _paper_normalize_k_grid()),
            "cv_folds": int(manifest_protocol.get("cv_folds") or PAPER_CV_FOLDS_DEFAULT),
            "optimization_backend": str(manifest_backend),
            "optuna_trials": int(
                manifest_protocol.get("optuna_trials")
                or (PAPER_OPTUNA_TRIALS_DEFAULT if manifest_backend == "optuna" else 0)
            ),
            "frozen": frozen_payload,
            "raw": raw_payload,
            "update_emb": update_emb_payload,
            "compare": compare_payload,
            "candidate_paths": export_payload.get("candidate_paths") or {},
            "promoted_paths": export_payload.get("promoted_paths") or {},
            "latex_promoted": bool(export_payload.get("latex_promoted")),
            "result_status": export_payload.get("result_status") or manifest.get("result_status"),
        }
    if "route_options" not in payload:
        payload["route_options"] = dict(
            ((manifest.get("protocol") or {}).get("route_options") or PAPER_ROUTE_SELECTION_DEFAULTS)
        )
    if "optimization_backend" not in payload:
        payload["optimization_backend"] = str(
            _paper_normalize_optimization_backend(
                (manifest.get("protocol") or {}).get("optimization_backend")
            )
        )
    if "k_grid" not in payload:
        payload["k_grid"] = list(((manifest.get("protocol") or {}).get("k_grid") or _paper_normalize_k_grid()))
    if "cv_folds" not in payload:
        payload["cv_folds"] = int(((manifest.get("protocol") or {}).get("cv_folds") or PAPER_CV_FOLDS_DEFAULT))
    if "optuna_trials" not in payload:
        manifest_backend = _paper_normalize_optimization_backend(
            ((manifest.get("protocol") or {}).get("optimization_backend"))
        )
        payload["optuna_trials"] = int(
            ((manifest.get("protocol") or {}).get("optuna_trials"))
            or (PAPER_OPTUNA_TRIALS_DEFAULT if manifest_backend == "optuna" else 0)
        )
    payload["checkpoint_manifest"] = manifest
    payload["checkpoint_manifest_path"] = str(paths["manifest"])
    payload["checkpoint_run_dir"] = str(paths["run_dir"])
    payload["auto_resumed"] = bool(auto_resumed)
    payload["loaded_from_checkpoint"] = bool(loaded_from_checkpoint)
    payload["computed_run_id"] = str(manifest.get("computed_run_id") or "")
    payload["result_status"] = payload.get("result_status") or manifest.get("result_status")
    return payload


def _paper_fingerprint_source_df(source_df: pd.DataFrame) -> Dict[str, object]:
    normalized = _normalize_accidents_for_feature_engineering(source_df)
    signature_cols = [
        col
        for col in ["accident_id", "accidente_time", "severity_target", "ultimo_portico", "proximo_portico", "text_bert"]
        if col in normalized.columns
    ]
    return {
        "kind": "session_accidents",
        "rows": int(len(normalized)),
        "signature": _frame_signature(normalized, columns=signature_cols, include_index=True),
        "signature_columns": signature_cols,
    }


def _paper_load_latest_processed_events_entry() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    catalog = _load_artifact_catalog(stage="events", artifact_name="processed_events")
    if catalog.empty:
        return None, None
    selected = catalog.iloc[0]
    return _read_artifact_df(selected.get("db_path"), selected.get("table_name")), selected


def _paper_resolve_raw_source(
    accidents_df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    if isinstance(accidents_df, pd.DataFrame) and not accidents_df.empty:
        return {
            "source_df": accidents_df.copy(),
            "source_kind": "session_accidents",
            "source_metadata": {"rows": int(len(accidents_df))},
            "source_fingerprint": _paper_fingerprint_source_df(accidents_df),
        }

    source_df, artifact_row = _paper_load_latest_processed_events_entry()
    if isinstance(source_df, pd.DataFrame) and not source_df.empty and artifact_row is not None:
        return {
            "source_df": source_df,
            "source_kind": "processed_events_artifact",
            "source_metadata": {
                "artifact_id": str(artifact_row.get("artifact_id") or ""),
                "db_path": str(artifact_row.get("db_path") or ""),
                "table_name": str(artifact_row.get("table_name") or ""),
                "row_count": int(artifact_row.get("row_count") or 0),
            },
            "source_fingerprint": {
                "kind": "processed_events_artifact",
                "artifact_id": str(artifact_row.get("artifact_id") or ""),
                "db_path": str(artifact_row.get("db_path") or ""),
                "table_name": str(artifact_row.get("table_name") or ""),
                "row_count": int(artifact_row.get("row_count") or 0),
            },
        }

    return {
        "source_df": None,
        "source_kind": "missing",
        "source_metadata": {},
        "source_fingerprint": {"kind": "missing", "reason": "no_raw_source"},
    }


def _paper_fingerprint_transformer_model(model_row: Optional[pd.Series]) -> Dict[str, object]:
    if model_row is None:
        return {"kind": "missing", "reason": "no_transformer_model"}
    return {
        "kind": "transformer_finetuned",
        "model_label": str(model_row.get("model_label") or ""),
        "output_dir_resolved": str(model_row.get("output_dir_resolved") or ""),
        "created_at": str(model_row.get("created_at") or ""),
    }


def _paper_build_computed_run_id(
    *,
    protocol_snapshot: Dict[str, object],
    input_fingerprints: Dict[str, object],
) -> str:
    payload = {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "protocol": _to_json_safe(protocol_snapshot),
        "input_fingerprints": _to_json_safe(input_fingerprints),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"paper_replication_{digest[:16]}"


def _paper_route_options(
    *,
    run_frozen: bool = True,
    run_raw: bool = True,
    run_update_embeddings: bool = False,
) -> Dict[str, bool]:
    return {
        "run_frozen": bool(run_frozen),
        "run_raw": bool(run_raw),
        "run_update_embeddings": bool(run_update_embeddings),
    }


def _paper_build_execution_context(
    accidents_df: Optional[pd.DataFrame],
    *,
    route_options: Optional[Dict[str, object]] = None,
    k_grid: Optional[Sequence[object]] = None,
    cv_folds: Optional[object] = None,
    raw_features_artifact_row: Optional[pd.Series] = None,
    transformer_model_row_override: Optional[pd.Series] = None,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
) -> Dict[str, object]:
    effective_route_options = _paper_route_options(
        run_frozen=bool((route_options or {}).get("run_frozen", PAPER_ROUTE_SELECTION_DEFAULTS["run_frozen"])),
        run_raw=bool((route_options or {}).get("run_raw", PAPER_ROUTE_SELECTION_DEFAULTS["run_raw"])),
        run_update_embeddings=bool((route_options or {}).get("run_update_embeddings", PAPER_ROUTE_SELECTION_DEFAULTS["run_update_embeddings"])),
    )
    protocol_snapshot = {
        **_paper_protocol_config(
            k_grid=k_grid,
            cv_folds=cv_folds,
            optimization_backend=optimization_backend,
            optuna_trials=optuna_trials,
            scoring_metric=scoring_metric,
        ),
        "route_options": effective_route_options,
    }
    raw_source = {
        "source_df": None,
        "source_kind": "skipped",
        "source_metadata": {"skipped": True},
        "source_fingerprint": {"skipped": True},
    }
    feature_bundle_override = {
        "features_df": None,
        "granular_df": pd.DataFrame(),
        "feature_artifact": {},
        "fingerprint": {"skipped": True},
    }
    transformer_model_row = None
    needs_transformer = effective_route_options["run_raw"] or effective_route_options["run_update_embeddings"]
    if effective_route_options["run_raw"]:
        feature_bundle_override = _paper_resolve_feature_bundle_override(raw_features_artifact_row)
        if feature_bundle_override.get("features_df") is None:
            raw_source = _paper_resolve_raw_source(accidents_df)
        else:
            raw_source = {
                "source_df": None,
                "source_kind": "precomputed_features_artifact",
                "source_metadata": feature_bundle_override.get("feature_artifact") or {},
                "source_fingerprint": {"kind": "skipped", "reason": "precomputed_features_artifact"},
            }
    if needs_transformer:
        if transformer_model_row_override is not None:
            transformer_model_row = transformer_model_row_override.copy()
        else:
            try:
                transformer_model_row = _paper_resolve_transformer_model()
            except Exception:
                transformer_model_row = None
    input_fingerprints = {
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "frozen_dataset": (
            _paper_file_fingerprint(PAPER_FROZEN_DATASET_PATH)
            if effective_route_options["run_frozen"] or effective_route_options["run_update_embeddings"]
            else {"skipped": True}
        ),
        "raw_source": raw_source.get("source_fingerprint") or {},
        "transformer_model": (
            _paper_fingerprint_transformer_model(transformer_model_row)
            if needs_transformer
            else {"skipped": True}
        ),
        "raw_features": (
            feature_bundle_override.get("fingerprint") or {}
            if effective_route_options["run_raw"]
            else {"skipped": True}
        ),
    }
    computed_run_id = _paper_build_computed_run_id(
        protocol_snapshot=protocol_snapshot,
        input_fingerprints=input_fingerprints,
    )
    return {
        "protocol_snapshot": protocol_snapshot,
        "input_fingerprints": input_fingerprints,
        "computed_run_id": computed_run_id,
        "raw_source_df": raw_source.get("source_df"),
        "raw_source_kind": raw_source.get("source_kind"),
        "raw_source_metadata": raw_source.get("source_metadata") or {},
        "transformer_model_row": transformer_model_row,
        "raw_features_df": feature_bundle_override.get("features_df"),
        "raw_granular_df": feature_bundle_override.get("granular_df"),
        "raw_feature_artifact": feature_bundle_override.get("feature_artifact") or {},
        "route_options": effective_route_options,
    }


def _paper_refresh_manifest_progress(manifest: Dict[str, object]) -> None:
    steps_index = manifest.setdefault("steps_index", {})
    step_sequence = manifest.setdefault("step_sequence", list(steps_index.keys()))
    completed_steps = int(
        sum(1 for step_id in step_sequence if str((steps_index.get(step_id) or {}).get("status")) in {"completed", "blocked"})
    )
    running_step_id = next(
        (
            step_id
            for step_id in reversed(step_sequence)
            if str((steps_index.get(step_id) or {}).get("status")) == "running"
        ),
        None,
    )
    progress = dict(manifest.get("progress") or {})
    progress["completed_steps"] = completed_steps
    progress["total_steps"] = int(len(step_sequence))
    progress["completed_units"] = float(completed_steps)
    progress["total_units"] = float(len(step_sequence))
    progress["current_step_id"] = running_step_id
    if running_step_id:
        running_step = steps_index.get(running_step_id) or {}
        progress["current_stage"] = str(running_step.get("stage") or "")
    elif completed_steps == len(step_sequence) and step_sequence:
        progress["current_stage"] = "completed"
    manifest["progress"] = progress


def _paper_persist_manifest(path: Path, manifest: Dict[str, object]) -> None:
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _paper_refresh_manifest_progress(manifest)
    _atomic_write_json(path, manifest)


def _paper_persist_live_status(path: Path, payload: Dict[str, object]) -> None:
    _atomic_write_json(path, payload)


def _paper_append_live_event(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json_safe(payload), ensure_ascii=True))
        handle.write("\n")


def _paper_reset_live_artifacts(paths: Dict[str, Path]) -> None:
    for key in ["live_status", "live_events"]:
        live_path = Path(paths[key])
        if not live_path.exists():
            continue
        try:
            live_path.unlink()
        except Exception:
            pass


def _paper_initial_manifest(
    *,
    run_id: str,
    computed_run_id: str,
    protocol_snapshot: Dict[str, object],
    input_fingerprints: Dict[str, object],
    checkpoint_run_id_override: Optional[str] = None,
) -> Dict[str, object]:
    created_at = datetime.now().isoformat(timespec="seconds")
    return {
        "run_id": str(run_id),
        "computed_run_id": str(computed_run_id),
        "checkpoint_run_id_override": None if checkpoint_run_id_override is None else str(checkpoint_run_id_override),
        "status": "running",
        "result_status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "protocol": _to_json_safe(protocol_snapshot),
        "input_fingerprints": _to_json_safe(input_fingerprints),
        "progress": {
            "current_stage": "",
            "current_step_id": None,
            "completed_steps": 0,
            "total_steps": 0,
            "completed_units": 0.0,
            "total_units": 0.0,
        },
        "steps_index": {},
        "step_sequence": [],
        "resume": {
            "auto_resumed": False,
            "checkpoint_status": None,
            "checkpoint_mode": "fresh",
        },
        "registry_sync": {
            "completed": False,
            "completed_at": None,
        },
        "last_error": None,
    }


def _paper_step_artifacts_exist(payload: object) -> bool:
    if isinstance(payload, dict):
        if not payload:
            return False
        return all(_paper_step_artifacts_exist(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        if not payload:
            return False
        return all(_paper_step_artifacts_exist(value) for value in payload)
    if isinstance(payload, str):
        return Path(payload).exists()
    return bool(payload)


def _paper_register_step(
    manifest: Dict[str, object],
    step_id: str,
    *,
    stage: str,
    description: str,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    steps_index = manifest.setdefault("steps_index", {})
    step_sequence = manifest.setdefault("step_sequence", [])
    if step_id not in steps_index:
        step_sequence.append(str(step_id))
        steps_index[str(step_id)] = {
            "step_id": str(step_id),
            "stage": str(stage),
            "description": str(description),
            "status": "pending",
            "order": int(len(step_sequence)),
            "artifact_paths": {},
            "metadata": {},
            "error": None,
        }
    entry = steps_index[str(step_id)]
    entry["stage"] = str(stage)
    entry["description"] = str(description)
    if metadata:
        current_metadata = dict(entry.get("metadata") or {})
        current_metadata.update(_to_json_safe(metadata))
        entry["metadata"] = current_metadata
    _paper_refresh_manifest_progress(manifest)
    return entry


def _paper_reset_step_entry(step_entry: Dict[str, object]) -> None:
    step_entry["status"] = "pending"
    step_entry["artifact_paths"] = {}
    step_entry["error"] = None
    step_entry["started_at"] = None
    step_entry["completed_at"] = None
    step_entry["last_message"] = None


def _paper_invalidate_from_step(manifest: Dict[str, object], step_id: str) -> None:
    steps_index = manifest.setdefault("steps_index", {})
    step_sequence = list(manifest.get("step_sequence") or [])
    if step_id not in step_sequence:
        return
    start_idx = step_sequence.index(step_id)
    for invalid_step_id in step_sequence[start_idx:]:
        step_entry = steps_index.get(invalid_step_id)
        if not isinstance(step_entry, dict):
            continue
        _paper_reset_step_entry(step_entry)
    manifest["status"] = "running"
    manifest["result_status"] = "running"
    manifest["completed_at"] = None
    manifest["last_error"] = None
    _paper_refresh_manifest_progress(manifest)


def _paper_reconcile_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(manifest, dict):
        return {}
    manifest.setdefault("steps_index", {})
    manifest.setdefault("step_sequence", list((manifest.get("steps_index") or {}).keys()))
    manifest.setdefault("registry_sync", {"completed": False, "completed_at": None})
    for step_id in list(manifest.get("step_sequence") or []):
        step_entry = (manifest.get("steps_index") or {}).get(step_id) or {}
        status = str(step_entry.get("status") or "pending")
        if status == "completed" and not _paper_step_artifacts_exist(step_entry.get("artifact_paths") or {}):
            _paper_invalidate_from_step(manifest, step_id)
            break
    _paper_refresh_manifest_progress(manifest)
    return manifest


def _paper_load_manifest(path: Path) -> Optional[Dict[str, object]]:
    payload = _load_json_file(path, default=None)
    if not isinstance(payload, dict):
        return None
    return _paper_reconcile_manifest(payload)


def _paper_persist_step_event(
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    *,
    step_id: str,
    status: str,
    message: str,
) -> None:
    progress = dict(manifest.get("progress") or {})
    live_payload = {
        "run_id": manifest.get("run_id"),
        "status": str(manifest.get("status") or ""),
        "result_status": str(manifest.get("result_status") or ""),
        "step_id": str(step_id),
        "step_status": str(status),
        "message": str(message),
        "progress": progress,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _paper_persist_live_status(Path(paths["live_status"]), live_payload)
    _paper_append_live_event(
        Path(paths["live_events"]),
        {
            **live_payload,
            "event_type": "step_status",
        },
    )


def _paper_mark_step_running(
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    step_id: str,
    *,
    stage: str,
    description: str,
    message: str,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    entry = _paper_register_step(
        manifest,
        step_id,
        stage=stage,
        description=description,
        metadata=metadata,
    )
    entry["status"] = "running"
    entry["started_at"] = entry.get("started_at") or datetime.now().isoformat(timespec="seconds")
    entry["last_message"] = str(message)
    entry["error"] = None
    manifest["status"] = "running"
    manifest["result_status"] = "running"
    progress = dict(manifest.get("progress") or {})
    progress["current_stage"] = str(stage)
    progress["current_step_id"] = str(step_id)
    manifest["progress"] = progress
    _paper_persist_manifest(Path(paths["manifest"]), manifest)
    _paper_persist_step_event(paths, manifest, step_id=step_id, status="running", message=message)
    return entry


def _paper_mark_step_completed(
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    step_id: str,
    *,
    stage: str,
    description: str,
    message: str,
    artifact_paths: Dict[str, object],
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    entry = _paper_register_step(
        manifest,
        step_id,
        stage=stage,
        description=description,
        metadata=metadata,
    )
    entry["status"] = "completed"
    entry["completed_at"] = datetime.now().isoformat(timespec="seconds")
    entry["last_message"] = str(message)
    entry["artifact_paths"] = _to_json_safe(artifact_paths)
    entry["error"] = None
    _paper_persist_manifest(Path(paths["manifest"]), manifest)
    _paper_persist_step_event(paths, manifest, step_id=step_id, status="completed", message=message)
    return entry


def _paper_mark_step_blocked(
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    step_id: str,
    *,
    stage: str,
    description: str,
    message: str,
    artifact_paths: Optional[Dict[str, object]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    entry = _paper_register_step(
        manifest,
        step_id,
        stage=stage,
        description=description,
        metadata=metadata,
    )
    entry["status"] = "blocked"
    entry["completed_at"] = datetime.now().isoformat(timespec="seconds")
    entry["last_message"] = str(message)
    entry["artifact_paths"] = _to_json_safe(artifact_paths or {})
    entry["error"] = str(message)
    _paper_persist_manifest(Path(paths["manifest"]), manifest)
    _paper_persist_step_event(paths, manifest, step_id=step_id, status="blocked", message=message)
    return entry


def _paper_mark_step_failed(
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    step_id: str,
    *,
    stage: str,
    description: str,
    message: str,
    error: object,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    entry = _paper_register_step(
        manifest,
        step_id,
        stage=stage,
        description=description,
        metadata=metadata,
    )
    entry["status"] = "failed"
    entry["completed_at"] = datetime.now().isoformat(timespec="seconds")
    entry["last_message"] = str(message)
    entry["error"] = str(error)
    manifest["status"] = "failed"
    manifest["result_status"] = "failed"
    manifest["completed_at"] = None
    manifest["last_error"] = str(error)
    _paper_persist_manifest(Path(paths["manifest"]), manifest)
    _paper_persist_step_event(paths, manifest, step_id=step_id, status="failed", message=message)
    return entry


def _paper_is_step_completed(
    manifest: Optional[Dict[str, object]],
    step_id: str,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    step_entry = ((manifest.get("steps_index") or {}).get(step_id) or {})
    return str(step_entry.get("status") or "") == "completed" and _paper_step_artifacts_exist(
        step_entry.get("artifact_paths") or {}
    )


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


def _project_transformer_hidden_state(
    last_hidden_state: object,
    attention_mask: object,
    *,
    projection: str,
) -> np.ndarray:
    if torch is None:
        raise ImportError("Se requiere torch para proyectar embeddings del transformer.")

    hidden = last_hidden_state
    mask = attention_mask
    if not isinstance(hidden, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("La salida del transformer debe incluir tensores validos.")

    projection_key = str(projection or "cls").strip().lower()
    if projection_key == "cls":
        if hidden.ndim != 3 or hidden.shape[1] < 1:
            raise ValueError("No se pudo extraer el token [CLS] del transformer.")
        return hidden[:, 0, :].detach().cpu().numpy()

    attention_mask_expanded = mask.unsqueeze(-1)
    sum_embeddings = torch.sum(hidden * attention_mask_expanded, dim=1)
    sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return (sum_embeddings / sum_mask).detach().cpu().numpy()


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
    transformer_projection: str = "cls",
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
            batch_embeddings = _project_transformer_hidden_state(
                outputs.last_hidden_state,
                encoded["attention_mask"],
                projection=transformer_projection,
            )
            all_batches.append(batch_embeddings)
        matrix = np.vstack(all_batches) if all_batches else np.zeros((len(work), 1), dtype=float)
        meta["model_name"] = str(model_path.name)
        meta["model_path"] = str(model_path)
        meta["batch_size"] = int(batch_size)
        meta["max_length"] = int(transformer_max_length)
        meta["projection"] = f"transformer_{str(transformer_projection).strip().lower()}"
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
    class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(unique_labels):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        support = float(cm[idx, :].sum())
        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_cls = (
            2.0 * precision_cls * recall_cls / (precision_cls + recall_cls)
            if (precision_cls + recall_cls) > 0
            else 0.0
        )
        class_metrics[str(label)] = {
            "precision": float(precision_cls),
            "recall": float(recall_cls),
            "f1_score": float(f1_cls),
            "support": int(support),
        }
    positive_label = 1 if 1 in unique_labels else unique_labels[-1]
    false_negatives_positive = int(false_negatives_by_class.get(str(positive_label), 0))
    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": unique_labels,
        "sample_size": int(len(y_true_arr)),
        "false_negatives_global": int(sum(false_negatives_by_class.values())),
        "false_negatives_by_class": false_negatives_by_class,
        "false_negatives_positive_class": false_negatives_positive,
        "false_negative_rate_positive_class": (
            float(false_negatives_positive) / float(len(y_true_arr))
            if len(y_true_arr) > 0
            else 0.0
        ),
        "class_metrics": class_metrics,
    }
    if y_score is not None and len(unique_labels) <= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, np.asarray(y_score, dtype=float)))
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def _paper_normalize_optimization_backend(backend: Optional[object]) -> str:
    normalized = str(backend or PAPER_OPTIMIZATION_BACKEND_DEFAULT).strip().lower()
    if normalized in {"gridsearchcv", "grid_search", "grid-search"}:
        return "gridsearch"
    if normalized in {"optuna", "tpe"}:
        return "optuna"
    return PAPER_OPTIMIZATION_BACKEND_DEFAULT


def _paper_optimization_backend_label(backend: Optional[object]) -> str:
    normalized = _paper_normalize_optimization_backend(backend)
    return {
        "gridsearch": "GridSearchCV",
        "optuna": "Optuna (TPE)",
    }.get(normalized, str(backend or "GridSearchCV"))


def _paper_protocol_config(
    *,
    k_grid: Optional[Sequence[object]] = None,
    cv_folds: Optional[object] = None,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
) -> Dict[str, object]:
    normalized_backend = _paper_normalize_optimization_backend(optimization_backend)
    resolved_trials = (
        max(1, int(optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT))
        if normalized_backend == "optuna"
        else 0
    )
    resolved_scoring = str(scoring_metric or PAPER_SCORING_METRIC_DEFAULT)
    if resolved_scoring not in PAPER_SCORING_METRICS:
        resolved_scoring = PAPER_SCORING_METRIC_DEFAULT
    return {
        **PAPER_PROTOCOL,
        "k_grid": _paper_normalize_k_grid(k_grid, enforce_limits=k_grid is not None),
        "cv_folds": _paper_normalize_cv_folds(cv_folds),
        "validation_alpha": float(PAPER_VALIDATION_ALPHA),
        "marginal_epsilon": 0.001,
        "comparison_tolerance": float(PAPER_COMPARISON_TOLERANCE),
        "optimization_backend": normalized_backend,
        "optuna_trials": resolved_trials,
        "scoring_metric": resolved_scoring,
    }


def _paper_normalize_k_grid(
    k_grid: Optional[Sequence[object]] = None,
    *,
    enforce_limits: bool = False,
) -> List[int]:
    if k_grid is None:
        return list(PAPER_K_GRID)
    values: List[int] = []
    for value in k_grid:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            continue
        integer_value = int(numeric)
        if integer_value in PAPER_K_GRID:
            values.append(integer_value)
    normalized = sorted(set(values))
    if not normalized:
        raise ValueError("Seleccione al menos una grilla valida de k.")
    if enforce_limits and (
        len(normalized) < int(PAPER_K_GRID_SELECTION_MIN) or len(normalized) > int(PAPER_K_GRID_SELECTION_MAX)
    ):
        raise ValueError(
            f"La grilla de k debe contener entre {int(PAPER_K_GRID_SELECTION_MIN)} y {int(PAPER_K_GRID_SELECTION_MAX)} valores."
        )
    return normalized


def _paper_normalize_cv_folds(cv_folds: Optional[object] = None) -> int:
    numeric = pd.to_numeric(cv_folds, errors="coerce")
    if pd.isna(numeric):
        return int(PAPER_CV_FOLDS_DEFAULT)
    value = int(numeric)
    if value < int(PAPER_CV_FOLDS_MIN) or value > int(PAPER_CV_FOLDS_MAX):
        raise ValueError(
            f"El K de folds debe estar entre {int(PAPER_CV_FOLDS_MIN)} y {int(PAPER_CV_FOLDS_MAX)}."
        )
    return value


def _paper_candidate_k_values(total_features: int, *, k_grid: Optional[Sequence[object]] = None) -> List[int]:
    max_features = max(1, int(total_features))
    clipped = sorted(
        {
            value
            for value in _paper_normalize_k_grid(k_grid, enforce_limits=k_grid is not None)
            if int(value) <= max_features
        }
    )
    if not clipped:
        return [max_features]
    return clipped


def _paper_shared_k_search_grid(
    df: pd.DataFrame,
    *,
    k_grid: Optional[Sequence[object]] = None,
) -> List[int]:
    feature_counts = [
        len(_resolve_feature_group(df, str(feature_group)))
        for feature_group in PAPER_PROTOCOL["feature_groups"].values()
    ]
    positive_counts = [count for count in feature_counts if int(count) > 0]
    if not positive_counts:
        raise ValueError("No hay variables disponibles para calcular un K compartido entre M1, M2 y M3.")
    common_cap = min(int(count) for count in positive_counts)
    return _paper_candidate_k_values(common_cap, k_grid=k_grid)


def _paper_feature_group_to_model_code(feature_group: str) -> str:
    normalized = str(feature_group or "").strip()
    for model_code, expected_group in PAPER_PROTOCOL["feature_groups"].items():
        if normalized == expected_group:
            return str(model_code)
    return normalized or "paper_model"


def _paper_model_title(model_code: str) -> str:
    return {
        "M1": "M1 · Solo flujo",
        "M2": "M2 · Solo embeddings",
        "M3": "M3 · Fusion multimodal",
    }.get(str(model_code), str(model_code))


def _paper_route_dir(run_id: str, route_name: str) -> Path:
    return _paper_run_dir(run_id) / _slug(route_name)


def _paper_write_json(path: Path, payload: Dict[str, object]) -> None:
    _atomic_write_json(path, payload)


def _paper_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _ensure_paper_dataset_columns(
    df: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"No hay datos disponibles para la ruta '{source_name}'.")
    work = df.copy()
    if "severity_target" not in work.columns or pd.to_numeric(work.get("severity_target"), errors="coerce").isna().all():
        if "severidad" in work.columns:
            work["severity_target"] = pd.to_numeric(work.get("severidad"), errors="coerce")
        else:
            raise ValueError(f"El dataset '{source_name}' no contiene `severity_target` ni `severidad`.")
    else:
        work["severity_target"] = pd.to_numeric(work.get("severity_target"), errors="coerce")
    if "accident_id" not in work.columns:
        work["accident_id"] = [f"evt_{idx:06d}" for idx in range(len(work))]
    else:
        accident_ids = work["accident_id"].astype(str).str.strip()
        empty_mask = accident_ids.eq("") | accident_ids.eq("nan") | accident_ids.eq("None")
        if empty_mask.any():
            replacement = pd.Series(
                [f"evt_{idx:06d}" for idx in range(len(work))],
                index=work.index,
            )
            accident_ids = accident_ids.mask(empty_mask, replacement)
        work["accident_id"] = accident_ids
    if "accidente_time" in work.columns:
        work["accidente_time"] = pd.to_datetime(work["accidente_time"], errors="coerce")
    return work


def _paper_dataset_validation_report(
    df: pd.DataFrame,
    *,
    route_name: str,
) -> Dict[str, object]:
    work = _ensure_paper_dataset_columns(df, source_name=route_name)
    flow_cols = _flow_feature_columns(work)
    emb_cols = _embedding_feature_columns(work)
    feature_cols = flow_cols + emb_cols
    report: Dict[str, object] = {
        "route_name": route_name,
        "rows": int(len(work)),
        "flow_features": int(len(flow_cols)),
        "embedding_features": int(len(emb_cols)),
        "total_features": int(len(feature_cols)),
        "class_counts": {
            str(label): int(count)
            for label, count in (
                pd.to_numeric(work["severity_target"], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            )
        },
        "mismatches": [],
    }
    if feature_cols:
        _, _, y_train, y_test, _, split_meta = _prepare_holdout_split_with_ids(
            work,
            feature_cols,
            test_size=float(PAPER_PROTOCOL["test_size"]),
            random_state=int(PAPER_PROTOCOL["random_state"]),
            split_mode=str(PAPER_PROTOCOL["split_mode"]),
        )
        report["train_rows"] = int(split_meta.get("train_rows") or len(y_train))
        report["test_rows"] = int(split_meta.get("test_rows") or len(y_test))
        report["train_class_counts"] = split_meta.get("train_class_counts") or {}
        report["test_class_counts"] = split_meta.get("test_class_counts") or {}
        report["split_mode_applied"] = split_meta.get("split_mode")
    else:
        report["train_rows"] = 0
        report["test_rows"] = 0
        report["train_class_counts"] = {}
        report["test_class_counts"] = {}
        report["split_mode_applied"] = None

    for key, expected in PAPER_EXPECTED_COUNTS.items():
        actual = report.get(key)
        if actual != expected:
            report["mismatches"].append({"field": key, "expected": expected, "actual": actual})
    report["is_valid"] = not bool(report["mismatches"])
    return report


def _paper_validation_score(
    metrics: Dict[str, object],
    *,
    best_cv_score: Optional[float] = None,
    scoring_metric: Optional[str] = None,
    alpha: float = PAPER_VALIDATION_ALPHA,
) -> float:
    # validation_score is the GridSearchCV best_score_ for the selected
    # scoring metric (defaults to F1 for backward compat).
    if best_cv_score is not None and not (isinstance(best_cv_score, float) and math.isnan(best_cv_score)):
        return float(best_cv_score)
    # Fallback when best_cv_score is unavailable (e.g. manual params).
    fallback_key = str(scoring_metric or "f1_score").strip()
    if fallback_key == "f1":
        fallback_key = "f1_score"
    return float(metrics.get(fallback_key) or metrics.get("f1_score") or 0.0)


def _paper_select_k_from_search(search_df: pd.DataFrame, *, epsilon: float = 0.001) -> int:
    if search_df is None or search_df.empty or "k" not in search_df.columns:
        raise ValueError("No hay resultados de grid search para seleccionar k.")
    ordered = search_df.sort_values("k", ascending=True).reset_index(drop=True)
    previous_row = ordered.iloc[0]
    for idx in range(1, len(ordered)):
        current_row = ordered.iloc[idx]
        previous_score = float(previous_row.get("validation_score") or 0.0)
        current_score = float(current_row.get("validation_score") or 0.0)
        if (current_score - previous_score) < float(epsilon):
            return int(previous_row["k"])
        previous_row = current_row
    return int(ordered.iloc[-1]["k"])


def _paper_load_latest_processed_events() -> Optional[pd.DataFrame]:
    catalog = _load_artifact_catalog(stage="events", artifact_name="processed_events")
    if catalog.empty:
        return None
    selected = catalog.iloc[0]
    return _read_artifact_df(selected.get("db_path"), selected.get("table_name"))


def _paper_resolve_transformer_model() -> pd.Series:
    catalog = _list_transformer_finetuned_models()
    if catalog.empty:
        raise PaperReplicationBlockedError("No se encontraron modelos fine-tuneados reutilizables para la ruta raw.")
    work = catalog.copy()
    work["mode_value"] = work["metadata"].apply(lambda meta: str((meta or {}).get("mode") or ""))
    work["text_col_value"] = work["metadata"].apply(lambda meta: str((meta or {}).get("text_col") or ""))
    work = work[work["mode_value"].eq("classification")].copy()
    if work.empty:
        raise PaperReplicationBlockedError("No hay modelos fine-tuneados de clasificacion disponibles para la ruta raw.")
    preferred = work[work["text_col_value"].eq("text_bert")].copy()
    if not preferred.empty:
        work = preferred
    return work.sort_values("created_at", ascending=False, ignore_index=True).iloc[0]


def _paper_metric_rows_for_compare(route_payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    validation = route_payload.get("dataset_validation") or {}
    for field in [
        "rows",
        "flow_features",
        "embedding_features",
        "total_features",
        "train_rows",
        "test_rows",
    ]:
        rows.append(
            {
                "scope": "dataset",
                "model_code": "dataset",
                "metric": field,
                "value": validation.get(field),
                "is_discrete": True,
            }
        )
    for class_key, count in (validation.get("train_class_counts") or {}).items():
        rows.append(
            {
                "scope": "dataset",
                "model_code": "dataset",
                "metric": f"train_class_count_{class_key}",
                "value": count,
                "is_discrete": True,
            }
        )
    for class_key, count in (validation.get("test_class_counts") or {}).items():
        rows.append(
            {
                "scope": "dataset",
                "model_code": "dataset",
                "metric": f"test_class_count_{class_key}",
                "value": count,
                "is_discrete": True,
            }
        )
    for result in route_payload.get("model_results") or []:
        model_code = str(result.get("model_code") or "")
        metrics = result.get("metrics") or {}
        rows.extend(
            [
                {"scope": "model", "model_code": model_code, "metric": "selected_k", "value": result.get("selected_k"), "is_discrete": True},
                {"scope": "model", "model_code": model_code, "metric": "accuracy", "value": metrics.get("accuracy"), "is_discrete": False},
                {"scope": "model", "model_code": model_code, "metric": "precision", "value": metrics.get("precision"), "is_discrete": False},
                {"scope": "model", "model_code": model_code, "metric": "recall", "value": metrics.get("recall"), "is_discrete": False},
                {"scope": "model", "model_code": model_code, "metric": "f1_score", "value": metrics.get("f1_score"), "is_discrete": False},
                {"scope": "model", "model_code": model_code, "metric": "roc_auc", "value": metrics.get("roc_auc"), "is_discrete": False},
                {
                    "scope": "model",
                    "model_code": model_code,
                    "metric": "false_negatives_positive_class",
                    "value": metrics.get("false_negatives_positive_class"),
                    "is_discrete": True,
                },
            ]
        )
        for class_key, class_metrics in (metrics.get("class_metrics") or {}).items():
            for metric_name in ("precision", "recall", "f1_score"):
                rows.append(
                    {
                        "scope": "class",
                        "model_code": model_code,
                        "metric": f"class_{class_key}_{metric_name}",
                        "value": (class_metrics or {}).get(metric_name),
                        "is_discrete": False,
                    }
                )
    return rows


def _paper_compare_routes(
    frozen_payload: Dict[str, object],
    raw_payload: Dict[str, object],
    *,
    tolerance: float = PAPER_COMPARISON_TOLERANCE,
) -> Dict[str, object]:
    if str(frozen_payload.get("status") or "") != "ok":
        return {
            "status": "blocked",
            "reason": "La ruta frozen no se pudo completar.",
            "passed": False,
            "diff_df": pd.DataFrame(),
        }
    if str(raw_payload.get("status") or "") != "ok":
        return {
            "status": "blocked",
            "reason": str(raw_payload.get("status_message") or "La ruta raw no se pudo completar."),
            "passed": False,
            "diff_df": pd.DataFrame(),
        }

    frozen_rows = pd.DataFrame(_paper_metric_rows_for_compare(frozen_payload))
    raw_rows = pd.DataFrame(_paper_metric_rows_for_compare(raw_payload))
    diff_df = frozen_rows.merge(
        raw_rows,
        on=["scope", "model_code", "metric", "is_discrete"],
        how="outer",
        suffixes=("_frozen", "_raw"),
    )
    diff_df["value_frozen"] = pd.to_numeric(diff_df["value_frozen"], errors="coerce")
    diff_df["value_raw"] = pd.to_numeric(diff_df["value_raw"], errors="coerce")
    diff_df["abs_diff"] = (diff_df["value_frozen"] - diff_df["value_raw"]).abs()
    diff_df["discrete_match"] = np.where(
        diff_df["is_discrete"].fillna(False),
        diff_df["value_frozen"].eq(diff_df["value_raw"]),
        np.nan,
    )
    discrete_failures = diff_df.loc[
        diff_df["is_discrete"].fillna(False) & ~diff_df["value_frozen"].eq(diff_df["value_raw"])
    ].copy()
    numeric_df = diff_df.loc[~diff_df["is_discrete"].fillna(False)].copy()
    max_numeric_diff = float(numeric_df["abs_diff"].max()) if not numeric_df.empty else 0.0
    numeric_failures = numeric_df.loc[numeric_df["abs_diff"] > float(tolerance)].copy()
    passed = discrete_failures.empty and numeric_failures.empty
    reason = "Rutas coinciden bajo tolerancia estricta." if passed else "Diferencias detectadas entre frozen y raw."
    return {
        "status": "ok" if passed else "blocked",
        "reason": reason,
        "passed": bool(passed),
        "max_numeric_diff": max_numeric_diff,
        "tolerance": float(tolerance),
        "discrete_failures": discrete_failures,
        "numeric_failures": numeric_failures,
        "diff_df": diff_df.sort_values(["scope", "model_code", "metric"]).reset_index(drop=True),
    }


def _paper_manifest_signature(payload: object) -> str:
    return json.dumps(_to_json_safe(payload), sort_keys=True, ensure_ascii=True)


def _paper_manifest_is_compatible(
    manifest: Optional[Dict[str, object]],
    execution_context: Optional[Dict[str, object]],
) -> Tuple[bool, str]:
    if not isinstance(manifest, dict):
        return False, "Checkpoint inexistente."
    if not isinstance(execution_context, dict):
        return False, "No hay contexto de ejecucion disponible."
    if str(manifest.get("protocol_version") or "") != str(PAPER_PROTOCOL_VERSION):
        return False, "La version del protocolo del checkpoint no coincide."
    if str(manifest.get("computed_run_id") or "") != str(execution_context.get("computed_run_id") or ""):
        return False, "El computed_run_id del checkpoint no coincide."
    manifest_fingerprints = manifest.get("input_fingerprints") or {}
    context_fingerprints = execution_context.get("input_fingerprints") or {}
    if _paper_manifest_signature(manifest_fingerprints) != _paper_manifest_signature(context_fingerprints):
        return False, "Los fingerprints de insumos del checkpoint no coinciden."
    return True, ""


def _paper_preview_payload_from_manifest(
    manifest: Optional[Dict[str, object]],
    *,
    run_dir: Path,
    manifest_path: Path,
    compatible: bool,
    incompatibility_reason: str,
) -> Dict[str, object]:
    manifest_dict = manifest or {}
    progress = dict(manifest_dict.get("progress") or {})
    status = str(manifest_dict.get("status") or "missing")
    checkpoint_available = bool(manifest)
    result_status = str(manifest_dict.get("result_status") or status)
    return {
        "run_id": str(manifest_dict.get("run_id") or run_dir.name),
        "computed_run_id": str(manifest_dict.get("computed_run_id") or ""),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "status": status,
        "result_status": result_status,
        "checkpoint_available": checkpoint_available,
        "compatible": bool(checkpoint_available and compatible),
        "incompatibility_reason": str(incompatibility_reason or ""),
        "can_resume": bool(checkpoint_available and compatible and status != "completed"),
        "can_load_completed": bool(checkpoint_available and compatible and status == "completed"),
        "updated_at": manifest_dict.get("updated_at"),
        "completed_steps": int(progress.get("completed_steps", 0)),
        "total_steps": int(progress.get("total_steps", 0)),
        "current_step_id": progress.get("current_step_id"),
        "manifest": manifest,
    }


def _paper_preview_checkpoint(
    accidents_df: Optional[pd.DataFrame],
    *,
    execution_context: Optional[Dict[str, object]] = None,
    checkpoint_root: Optional[Path] = None,
) -> Dict[str, object]:
    context = execution_context or _paper_build_execution_context(accidents_df)
    checkpoints_df = _list_paper_checkpoints(
        accidents_df,
        execution_context=context,
        checkpoint_root=checkpoint_root,
    )
    compatible_df = checkpoints_df.loc[checkpoints_df["compatible"].fillna(False)].copy() if not checkpoints_df.empty else pd.DataFrame()
    if compatible_df.empty:
        run_dir = _paper_run_dir(str(context.get("computed_run_id") or ""), checkpoint_root=checkpoint_root)
        paths = _paper_run_paths(run_dir)
        return _paper_preview_payload_from_manifest(
            None,
            run_dir=run_dir,
            manifest_path=Path(paths["manifest"]),
            compatible=False,
            incompatibility_reason="No existe checkpoint compatible.",
        )
    selected = compatible_df.iloc[0]
    return _paper_preview_checkpoint_run(
        str(selected["run_id"]),
        accidents_df=accidents_df,
        execution_context=context,
        checkpoint_root=checkpoint_root,
    )


def _paper_preview_checkpoint_run(
    run_id: str,
    *,
    accidents_df: Optional[pd.DataFrame] = None,
    execution_context: Optional[Dict[str, object]] = None,
    checkpoint_root: Optional[Path] = None,
) -> Dict[str, object]:
    resolved_run_id = str(run_id).strip()
    run_dir = _paper_run_dir(resolved_run_id, checkpoint_root=checkpoint_root)
    paths = _paper_run_paths(run_dir)
    manifest = _paper_load_manifest(Path(paths["manifest"]))
    context = execution_context or _paper_build_execution_context(accidents_df)
    compatible, incompatibility_reason = _paper_manifest_is_compatible(manifest, context)
    return _paper_preview_payload_from_manifest(
        manifest,
        run_dir=run_dir,
        manifest_path=Path(paths["manifest"]),
        compatible=compatible,
        incompatibility_reason=incompatibility_reason,
    )


def _list_paper_checkpoints(
    accidents_df: Optional[pd.DataFrame],
    *,
    execution_context: Optional[Dict[str, object]] = None,
    checkpoint_root: Optional[Path] = None,
) -> pd.DataFrame:
    root = _paper_checkpoint_root(checkpoint_root=checkpoint_root)
    if not root.exists():
        return pd.DataFrame()
    context = execution_context
    rows: List[Dict[str, object]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _paper_load_manifest(manifest_path)
        if not isinstance(manifest, dict):
            continue
        if context is None:
            context = _paper_build_execution_context(accidents_df)
        compatible, incompatibility_reason = _paper_manifest_is_compatible(manifest, context)
        progress = dict(manifest.get("progress") or {})
        rows.append(
            {
                "run_id": str(manifest.get("run_id") or manifest_path.parent.name),
                "computed_run_id": str(manifest.get("computed_run_id") or ""),
                "status": str(manifest.get("status") or ""),
                "result_status": str(manifest.get("result_status") or ""),
                "updated_at": str(manifest.get("updated_at") or ""),
                "compatible": bool(compatible),
                "incompatibility_reason": str(incompatibility_reason or ""),
                "completed_steps": int(progress.get("completed_steps", 0)),
                "total_steps": int(progress.get("total_steps", 0)),
                "current_step_id": progress.get("current_step_id"),
                "manifest_path": str(manifest_path),
                "run_dir": str(manifest_path.parent),
            }
        )
    if not rows:
        return pd.DataFrame()
    checkpoints_df = pd.DataFrame(rows)
    return checkpoints_df.sort_values(["compatible", "updated_at"], ascending=[False, False], ignore_index=True)


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


def _paper_route_summary_df(route_payload: Dict[str, object]) -> pd.DataFrame:
    validation = route_payload.get("dataset_validation") or {}
    return pd.DataFrame(
        [
            {
                "route_name": str(route_payload.get("route_name") or ""),
                "status": str(route_payload.get("status") or ""),
                "status_message": str(route_payload.get("status_message") or ""),
                "rows": validation.get("rows"),
                "flow_features": validation.get("flow_features"),
                "embedding_features": validation.get("embedding_features"),
                "total_features": validation.get("total_features"),
                "train_rows": validation.get("train_rows"),
                "test_rows": validation.get("test_rows"),
                "is_valid": validation.get("is_valid"),
            }
        ]
    )


def _compare_summary_df(compare_payload: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": str(compare_payload.get("status") or ""),
                "passed": bool(compare_payload.get("passed")),
                "reason": str(compare_payload.get("reason") or ""),
                "max_numeric_diff": compare_payload.get("max_numeric_diff"),
                "tolerance": compare_payload.get("tolerance"),
                "discrete_failures": int(
                    len(compare_payload.get("discrete_failures"))
                    if isinstance(compare_payload.get("discrete_failures"), pd.DataFrame)
                    else 0
                ),
                "numeric_failures": int(
                    len(compare_payload.get("numeric_failures"))
                    if isinstance(compare_payload.get("numeric_failures"), pd.DataFrame)
                    else 0
                ),
            }
        ]
    )


def _paper_export_manifest_df(payload: Dict[str, object]) -> pd.DataFrame:
    candidate_paths = payload.get("candidate_paths") or {}
    promoted_paths = payload.get("promoted_paths") or {}
    rows = [
        {
            "asset_name": str(asset_name),
            "candidate_path": str(candidate_path),
            "promoted": str(asset_name) in promoted_paths,
            "promoted_path": str(promoted_paths.get(str(asset_name)) or ""),
        }
        for asset_name, candidate_path in candidate_paths.items()
    ]
    if not rows:
        rows = [
            {
                "asset_name": "paper_replication",
                "candidate_path": "",
                "promoted": False,
                "promoted_path": "",
            }
        ]
    return pd.DataFrame(rows)


def _persist_paper_replication_payload(payload: Dict[str, object], *, run_id: Optional[str] = None) -> None:
    manifest_path_raw = payload.get("checkpoint_manifest_path")
    manifest_path = Path(str(manifest_path_raw)) if manifest_path_raw else None
    manifest = _paper_load_manifest(manifest_path) if manifest_path is not None else None
    if isinstance(manifest, dict) and bool((manifest.get("registry_sync") or {}).get("completed")):
        return

    resolved_run_id = str(
        run_id
        or payload.get("run_id")
        or (manifest.get("run_id") if isinstance(manifest, dict) else "")
    )
    if not resolved_run_id:
        raise ValueError("No hay run_id disponible para persistir la replica del paper.")

    for route_key, stage_name in (
        ("frozen", "paper_replication_frozen"),
        ("raw", "paper_replication_raw"),
        ("update_emb", "paper_replication_update_emb"),
    ):
        route_payload = payload.get(route_key) or {}
        summary_df = _paper_route_summary_df(route_payload)
        _persist_artifact(
            summary_df,
            stage=stage_name,
            artifact_name=f"{route_key}_summary",
            run_id=resolved_run_id,
            metadata={"route_name": route_key},
        )
        for artifact_name, frame in (
            (f"{route_key}_comparison", route_payload.get("comparison_df")),
            (f"{route_key}_metricas", route_payload.get("metricas_df")),
            (f"{route_key}_predictions", route_payload.get("predictions_df")),
            (f"{route_key}_m3_grid", route_payload.get("m3_grid_df")),
        ):
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                _persist_artifact(
                    frame,
                    stage=stage_name,
                    artifact_name=artifact_name,
                    run_id=resolved_run_id,
                    metadata={"route_name": route_key},
                )
        build_key = "raw_build" if route_key == "raw" else ("update_emb_build" if route_key == "update_emb" else None)
        if build_key:
            route_build = route_payload.get(build_key) or {}
            embedding_meta = route_build.get("embedding_meta") or {}
            selected_embedding_cols = route_build.get("selected_embedding_cols") or []
            _log_action(
                stage_name,
                f"paper_replication_{route_key}_build",
                {
                    "embedding_meta": embedding_meta,
                    "selected_embedding_count": int(len(selected_embedding_cols)),
                },
                run_id=resolved_run_id,
            )
        for result in route_payload.get("model_results") or []:
            _record_model_result(
                run_id=resolved_run_id,
                stage=stage_name,
                model_name=str(result.get("model_title") or result.get("model_code") or "paper_model"),
                feature_group=str(result.get("feature_group") or ""),
                metrics=result.get("metrics") or {},
                params=result.get("best_params") or {},
                metadata={
                    "route_name": route_key,
                    "model_code": result.get("model_code"),
                    "selected_k": result.get("selected_k"),
                    "selected_cols": result.get("selected_cols") or [],
                    "split_meta": result.get("split_meta") or {},
                    "best_cv_score": result.get("best_cv_score"),
                },
            )

    compare_payload = payload.get("compare") or {}
    _persist_artifact(
        _compare_summary_df(compare_payload),
        stage="paper_replication_compare",
        artifact_name="frozen_raw_compare_summary",
        run_id=resolved_run_id,
        metadata={"passed": bool(compare_payload.get("passed"))},
    )
    diff_df = compare_payload.get("diff_df")
    if isinstance(diff_df, pd.DataFrame) and not diff_df.empty:
        _persist_artifact(
            diff_df,
            stage="paper_replication_compare",
            artifact_name="frozen_raw_compare_diff",
            run_id=resolved_run_id,
            metadata={
                "passed": bool(compare_payload.get("passed")),
                "tolerance": compare_payload.get("tolerance"),
            },
        )

    export_manifest_df = _paper_export_manifest_df(payload)
    _persist_artifact(
        export_manifest_df,
        stage="paper_replication_export",
        artifact_name="latex_asset_manifest",
        run_id=resolved_run_id,
        metadata={
            "latex_promoted": bool(payload.get("latex_promoted")),
            "candidate_count": int(len(payload.get("candidate_paths") or {})),
            "promoted_count": int(len(payload.get("promoted_paths") or {})),
        },
    )
    _log_action(
        "paper_replication_export",
        "paper_replication_completed",
        {
            "latex_promoted": bool(payload.get("latex_promoted")),
            "compare_status": compare_payload.get("status"),
            "compare_reason": compare_payload.get("reason"),
            "run_dir": payload.get("run_dir"),
        },
        run_id=resolved_run_id,
    )
    if isinstance(manifest, dict) and manifest_path is not None:
        registry_sync = dict(manifest.get("registry_sync") or {})
        registry_sync["completed"] = True
        registry_sync["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["registry_sync"] = registry_sync
        _paper_persist_manifest(manifest_path, manifest)


def _render_paper_replication_subtab(*, accidents_df: Optional[pd.DataFrame]) -> None:
    previous_payload = st.session_state.get("nlp_sev_paper_replication_payload")
    st.markdown("**Rutas a ejecutar**")
    route_cols = st.columns(3)
    with route_cols[0]:
        run_frozen = st.checkbox(
            "Ejecutar frozen",
            value=True,
            key="nlp_sev_paper_route_frozen",
            help="Carga `NLP/Dataframes/resultado.pkl` y replica M1/M2/M3 sobre el dataset congelado del paper.",
        )
    with route_cols[1]:
        run_raw = st.checkbox(
            "Ejecutar raw",
            value=True,
            key="nlp_sev_paper_route_raw",
            help="Reconstruye features + embeddings desde eventos y luego replica M1/M2/M3.",
        )
    with route_cols[2]:
        run_update_embeddings = st.checkbox(
            "Actualizar embeddings",
            value=False,
            key="nlp_sev_paper_route_update_emb",
            help=(
                "Usa el dataset congelado (2070 filas con features de flujo), "
                "busca los textos en features ya calculadas, regenera embeddings "
                "con el transformer fine-tuneado actual y selecciona los 200 mas "
                "importantes via RF. Luego ejecuta M1/M2/M3."
            ),
        )
    route_options = _paper_route_options(run_frozen=run_frozen, run_raw=run_raw, run_update_embeddings=run_update_embeddings)
    if not any(route_options.values()):
        st.warning("Seleccione al menos una ruta para ejecutar la replica.")
    st.markdown("**Grilla de k**")
    selected_k_grid = st.multiselect(
        "Valores de k a evaluar",
        options=list(PAPER_K_GRID),
        default=list(PAPER_K_GRID_DEFAULT_SELECTION),
        key="nlp_sev_paper_k_grid",
        help=(
            "Seleccione entre 2 y 5 valores de la grilla del paper. "
            "La grilla se recorta automaticamente al total de features disponible en M1, M2 y M3."
        ),
    )
    k_grid_error: Optional[str] = None
    try:
        normalized_k_grid = _paper_normalize_k_grid(selected_k_grid)
    except Exception as exc:
        normalized_k_grid = []
        k_grid_error = str(exc)
    if k_grid_error:
        st.warning(k_grid_error)
    else:
        st.caption(f"Grilla seleccionada: {normalized_k_grid}")
    selected_cv_folds = int(
        st.number_input(
            "K de folds para cross validation",
            min_value=int(PAPER_CV_FOLDS_MIN),
            max_value=int(PAPER_CV_FOLDS_MAX),
            value=int(PAPER_CV_FOLDS_DEFAULT),
            step=1,
            key="nlp_sev_paper_cv_folds",
            help="Se usa como K compartido para la nested CV por k y para la CV interna del ajuste final.",
        )
    )
    st.markdown("**Optimizacion XGBoost**")
    optimization_cols = st.columns(2)
    with optimization_cols[0]:
        selected_optimization_backend = st.selectbox(
            "Metodo de optimizacion",
            options=["gridsearch", "optuna"],
            index=0,
            format_func=_paper_optimization_backend_label,
            key="nlp_sev_paper_optimization_backend",
            help="GridSearchCV recorre la grilla completa; Optuna explora la misma grilla discreta con TPE.",
        )
    with optimization_cols[1]:
        selected_optuna_trials = st.number_input(
            "Trials de Optuna",
            min_value=1,
            max_value=256,
            value=int(PAPER_OPTUNA_TRIALS_DEFAULT),
            step=1,
            key="nlp_sev_paper_optuna_trials",
            disabled=_paper_normalize_optimization_backend(selected_optimization_backend) != "optuna",
            help="Solo aplica cuando se selecciona Optuna.",
        )
    selected_scoring_metric = st.selectbox(
        "Metrica de seleccion de k (best_cv_score)",
        options=PAPER_SCORING_METRICS,
        index=PAPER_SCORING_METRICS.index(PAPER_SCORING_METRIC_DEFAULT),
        format_func=lambda m: PAPER_SCORING_METRIC_LABELS.get(m, m),
        key="nlp_sev_paper_scoring_metric",
        help=(
            "Metrica usada como scoring en GridSearchCV/Optuna y como "
            "validation_score para la regla marginal de seleccion de k*."
        ),
    )
    if _paper_normalize_optimization_backend(selected_optimization_backend) == "optuna" and optuna is None:
        st.warning(
            "Optuna no esta disponible en el entorno activo. La replica caera automaticamente a GridSearchCV."
        )
    if k_grid_error:
        protocol = {
            **PAPER_PROTOCOL,
            "k_grid": list(selected_k_grid),
            "cv_folds": int(selected_cv_folds),
            "validation_alpha": float(PAPER_VALIDATION_ALPHA),
            "marginal_epsilon": 0.001,
            "comparison_tolerance": float(PAPER_COMPARISON_TOLERANCE),
            "optimization_backend": _paper_normalize_optimization_backend(selected_optimization_backend),
            "optuna_trials": (
                max(1, int(selected_optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT))
                if _paper_normalize_optimization_backend(selected_optimization_backend) == "optuna"
                else 0
            ),
        }
    else:
        protocol = _paper_protocol_config(
            k_grid=normalized_k_grid,
            cv_folds=selected_cv_folds,
            optimization_backend=selected_optimization_backend,
            optuna_trials=selected_optuna_trials,
        )
    selected_raw_features_artifact_row: Optional[pd.Series] = None
    selected_transformer_model_row: Optional[pd.Series] = None
    raw_features_mode = "Reconstruir desde eventos"
    if route_options["run_raw"]:
        st.markdown("**Configuracion de Raw**")
        raw_cfg_cols = st.columns(2)
        with raw_cfg_cols[0]:
            raw_features_mode = st.radio(
                "Fuente de features raw",
                ["Reconstruir desde eventos", "Usar features ya calculadas"],
                horizontal=True,
                key="nlp_sev_paper_raw_features_mode",
            )
        feature_catalog = _list_feature_engineering_artifacts()
        if raw_features_mode == "Usar features ya calculadas":
            if feature_catalog.empty:
                st.warning("No hay artifacts `severity_features` disponibles en el registry.")
            else:
                selected_feature_idx = st.selectbox(
                    "Artifact de features raw",
                    options=feature_catalog.index.tolist(),
                    format_func=lambda idx: str(feature_catalog.loc[idx, "label"]),
                    key="nlp_sev_paper_raw_features_artifact",
                )
                selected_raw_features_artifact_row = feature_catalog.loc[selected_feature_idx]
                st.caption(
                    "Se reutilizaran las features del artifact seleccionado y la ruta raw solo recalculara "
                    "embeddings, seleccion supervisada y M1/M2/M3."
                )
        transformer_catalog = _list_transformer_finetuned_models()
        if transformer_catalog.empty:
            st.warning("No hay modelos fine-tuneados reutilizables para la ruta raw.")
        else:
            transformer_options: List[object] = ["AUTO"] + transformer_catalog.index.tolist()
            selected_transformer_option = st.selectbox(
                "Modelo de lenguaje para raw",
                options=transformer_options,
                format_func=lambda option: (
                    "Automatico (mas reciente compatible)"
                    if option == "AUTO"
                    else str(transformer_catalog.loc[option, "model_label"])
                ),
                key="nlp_sev_paper_raw_transformer_model",
            )
            if selected_transformer_option != "AUTO":
                selected_transformer_model_row = transformer_catalog.loc[selected_transformer_option]
                st.caption(
                    f"Modelo raw seleccionado: {selected_transformer_model_row.get('model_label')}"
                )
    execution_context: Optional[Dict[str, object]] = None
    checkpoint_preview: Dict[str, object] = {}
    checkpoints_df = pd.DataFrame()
    checkpoint_preview_error: Optional[str] = None
    try:
        execution_context = _paper_build_execution_context(
            accidents_df,
            route_options=route_options,
            k_grid=selected_k_grid,
            cv_folds=selected_cv_folds,
            raw_features_artifact_row=selected_raw_features_artifact_row,
            transformer_model_row_override=selected_transformer_model_row,
            optimization_backend=selected_optimization_backend,
            optuna_trials=selected_optuna_trials,
        )
        checkpoint_preview = _paper_preview_checkpoint(
            accidents_df,
            execution_context=execution_context,
        )
        checkpoints_df = _list_paper_checkpoints(
            accidents_df,
            execution_context=execution_context,
        )
    except Exception as exc:
        checkpoint_preview_error = str(exc)

    with st.expander("Protocolo bloqueado del paper", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "1. `M1` usa solo flujo, `M2` usa solo embeddings narrativos y `M3` usa fusion multimodal.",
                    "2. El `holdout` final es estratificado `80/20` con `random_state=42` y se congela antes del ranking, la seleccion de `k` y el tuning.",
                    "3. La replica usa solo `XGBoost` con nested CV en train, una subgrilla de `k` seleccionada por el usuario y un `K` configurable para cross validation.",
                    "4. La ruta `frozen` carga `NLP/Dataframes/resultado.pkl`; la ruta `raw` reconstruye desde eventos + flujos y embeddings fine-tuneados con proyeccion `[CLS]`.",
                    "5. El backend de optimizacion puede ser `GridSearchCV` u `Optuna`, pero ambos usan la misma grilla discreta de hiperparametros XGBoost.",
                    "6. Solo se promocionan assets hacia `NLP/Latex/` si frozen y raw coinciden con tolerancia `<= 0.001` y conteos discretos identicos.",
                    (
                        f"7. Seleccion actual: frozen={route_options['run_frozen']} | raw={route_options['run_raw']} | "
                        f"k_grid={normalized_k_grid or 'invalida'} | cv_folds={int(selected_cv_folds)} | "
                        f"optimizacion={_paper_optimization_backend_label(selected_optimization_backend)}"
                        + (
                            f" ({int(selected_optuna_trials)} trials)"
                            if _paper_normalize_optimization_backend(selected_optimization_backend) == "optuna"
                            else ""
                        )
                        + "."
                    ),
                ]
            )
        )
        st.json(protocol)
        st.caption(
            "Chequeos esperados del dataset congelado: "
            f"{PAPER_EXPECTED_COUNTS['rows']} filas | "
            f"{PAPER_EXPECTED_COUNTS['flow_features']} flujo | "
            f"{PAPER_EXPECTED_COUNTS['embedding_features']} embeddings | "
            f"split {PAPER_EXPECTED_COUNTS['train_rows']}/{PAPER_EXPECTED_COUNTS['test_rows']} "
            f"con test={PAPER_EXPECTED_COUNTS['test_class_counts']}"
        )
        if accidents_df is None or accidents_df.empty:
            st.caption(
                "La ruta raw intentara reutilizar el ultimo artifact `processed_events` del registry."
            )
        else:
            st.caption(f"Ruta raw disponible con {len(accidents_df):,} eventos en memoria.")
        if route_options["run_raw"]:
            st.caption(
                "Features raw: "
                + (
                    "artifact precomputado seleccionado"
                    if selected_raw_features_artifact_row is not None
                    else "reconstruccion desde eventos"
                )
            )
            if selected_transformer_model_row is not None:
                st.caption(
                    f"Modelo raw seleccionado manualmente: {selected_transformer_model_row.get('model_label')}"
                )
        st.caption(
            "Optimizacion XGBoost: "
            + _paper_optimization_backend_label(selected_optimization_backend)
            + (
                f" con {int(selected_optuna_trials)} trials"
                if _paper_normalize_optimization_backend(selected_optimization_backend) == "optuna"
                else f" sobre {_paper_xgb_search_space_size(_paper_xgb_param_grid())} combinaciones"
            )
        )
        st.caption(f"Grilla de k activa: {normalized_k_grid or 'invalida'}")
        st.caption(f"K de folds activo: {int(selected_cv_folds)}")

    if checkpoint_preview_error:
        st.warning(f"No se pudo construir el preview de checkpoints: {checkpoint_preview_error}")

    execution_mode: Optional[str] = None
    execution_checkpoint_run_id_override: Optional[str] = None
    selected_checkpoint_preview: Dict[str, object] = {}
    if isinstance(checkpoints_df, pd.DataFrame) and not checkpoints_df.empty:
        options = [""] + checkpoints_df["run_id"].astype(str).tolist()
        selected_run_id = st.selectbox(
            "Historial de checkpoints",
            options=options,
            index=0,
            key="nlp_sev_paper_checkpoint_selector",
            format_func=lambda value: "Sin seleccion" if value == "" else str(value),
        )
        if selected_run_id:
            try:
                selected_checkpoint_preview = _paper_preview_checkpoint_run(
                    str(selected_run_id),
                    accidents_df=accidents_df,
                    execution_context=execution_context,
                )
                st.caption(
                    f"Run `{selected_checkpoint_preview.get('run_id')}` | "
                    f"estado {selected_checkpoint_preview.get('status')} | "
                    f"compatible={bool(selected_checkpoint_preview.get('compatible'))} | "
                    f"ultimo paso={selected_checkpoint_preview.get('current_step_id') or 'N/A'}"
                )
                if not bool(selected_checkpoint_preview.get("compatible")):
                    st.caption(
                        f"Incompatibilidad: {selected_checkpoint_preview.get('incompatibility_reason') or 'desconocida'}"
                    )
            except Exception as exc:
                st.caption(f"No se pudo leer el checkpoint seleccionado: {exc}")

    if bool(checkpoint_preview.get("checkpoint_available")):
        st.markdown("**Compatible checkpoint**" if bool(checkpoint_preview.get("compatible")) else "**Checkpoint preview**")
        st.caption(
            f"Run `{checkpoint_preview.get('run_id', '')}` | "
            f"estado {checkpoint_preview.get('status', '')} | "
            f"actualizado {checkpoint_preview.get('updated_at') or 'desconocido'} | "
            f"pasos {int(checkpoint_preview.get('completed_steps', 0))}/{int(checkpoint_preview.get('total_steps', 0))} | "
            f"ultimo paso {checkpoint_preview.get('current_step_id') or 'N/A'}"
        )
        st.caption(f"Manifest: {checkpoint_preview.get('manifest_path', '')}")
        if bool(checkpoint_preview.get("compatible")):
            if bool(checkpoint_preview.get("can_load_completed")):
                st.info("Existe una corrida compatible ya completada. Puedes cargarla o forzar una corrida fresca.")
            else:
                st.info("Existe un checkpoint compatible recuperable. Puedes reanudarlo o forzar una corrida fresca.")
            preview_cols = st.columns(2)
            with preview_cols[0]:
                resume_label = (
                    "Load completed checkpoint"
                    if bool(checkpoint_preview.get("can_load_completed"))
                    else "Resume compatible checkpoint"
                )
                if st.button(
                    resume_label,
                    key="nlp_sev_resume_paper_checkpoint",
                    disabled=bool(k_grid_error),
                ):
                    execution_mode = "resume"
                    execution_checkpoint_run_id_override = str(checkpoint_preview.get("run_id") or "")
            with preview_cols[1]:
                if st.button(
                    "Start fresh ignoring checkpoint",
                    key="nlp_sev_run_paper_replication_fresh",
                    disabled=bool(k_grid_error),
                ):
                    execution_mode = "fresh"
        else:
            st.warning(str(checkpoint_preview.get("incompatibility_reason") or "El checkpoint no es compatible."))

    selected_preview_run_id = str(selected_checkpoint_preview.get("run_id") or "")
    compatible_preview_run_id = str(checkpoint_preview.get("run_id") or "")
    if selected_preview_run_id and selected_preview_run_id != compatible_preview_run_id:
        st.markdown("**Loaded Checkpoint Actions**")
        if bool(selected_checkpoint_preview.get("compatible")):
            loaded_cols = st.columns(2)
            with loaded_cols[0]:
                loaded_label = (
                    "Load loaded checkpoint"
                    if bool(selected_checkpoint_preview.get("can_load_completed"))
                    else "Resume loaded checkpoint"
                )
                if st.button(
                    loaded_label,
                    key="nlp_sev_resume_loaded_paper_checkpoint",
                    disabled=bool(k_grid_error),
                ):
                    execution_mode = "resume"
                    execution_checkpoint_run_id_override = selected_preview_run_id
            with loaded_cols[1]:
                if st.button(
                    "Start fresh with loaded configuration",
                    key="nlp_sev_fresh_loaded_paper_checkpoint",
                    disabled=bool(k_grid_error),
                ):
                    execution_mode = "fresh"
        else:
            st.caption(
                f"El checkpoint seleccionado no es compatible: {selected_checkpoint_preview.get('incompatibility_reason') or 'desconocido'}"
            )

    if execution_mode is None and st.button(
        "Replicar paper M1/M2/M3",
        key="nlp_sev_run_paper_replication",
        disabled=(not any(route_options.values())) or bool(k_grid_error),
    ):
        execution_mode = "resume"

    if execution_mode is not None:
        progress_bar = st.progress(0)
        status_box = st.empty()

        def _update_progress(value: int, message: str) -> None:
            progress_bar.progress(int(value))
            status_box.caption(message)

        try:
            payload = run_paper_replication(
                accidents_df=accidents_df,
                run_frozen=route_options["run_frozen"],
                run_raw=route_options["run_raw"],
                run_update_embeddings=route_options["run_update_embeddings"],
                features_source_df=st.session_state.get("nlp_sev_features_df"),
                k_grid=normalized_k_grid,
                cv_folds=selected_cv_folds,
                raw_features_artifact_row=selected_raw_features_artifact_row,
                transformer_model_row_override=selected_transformer_model_row,
                optimization_backend=selected_optimization_backend,
                optuna_trials=selected_optuna_trials,
                scoring_metric=selected_scoring_metric,
                progress_callback=_update_progress,
                auto_resume=execution_mode != "fresh",
                checkpoint_run_id_override=execution_checkpoint_run_id_override,
                start_fresh=execution_mode == "fresh",
            )
            st.session_state["nlp_sev_paper_replication_payload"] = payload
            _persist_paper_replication_payload(payload)
        except Exception as exc:
            progress_bar.empty()
            status_box.empty()
            st.error(f"No se pudo ejecutar la replica del paper: {exc}")
        else:
            st.success("Replica del paper ejecutada.")
            st.caption(f"Run reproducible: {payload.get('run_dir')}")

    payload = st.session_state.get("nlp_sev_paper_replication_payload")
    if not isinstance(payload, dict) or not payload:
        payload = previous_payload
    if not isinstance(payload, dict) or not payload:
        st.info("Ejecute `Replicar paper M1/M2/M3` para generar los artifacts del paper.")
        return

    frozen_payload = payload.get("frozen") or {}
    raw_payload = payload.get("raw") or {}
    update_emb_payload = payload.get("update_emb") or {}
    compare_payload = payload.get("compare") or {}
    payload_route_options = _paper_route_options(
        run_frozen=bool((payload.get("route_options") or {}).get("run_frozen", True)),
        run_raw=bool((payload.get("route_options") or {}).get("run_raw", True)),
        run_update_embeddings=bool((payload.get("route_options") or {}).get("run_update_embeddings", False)),
    )
    st.caption(
        f"Checkpoint `{payload.get('run_id')}` | "
        f"computed_run_id={payload.get('computed_run_id') or 'N/A'} | "
        f"auto_resumed={bool(payload.get('auto_resumed'))} | "
        f"loaded_from_checkpoint={bool(payload.get('loaded_from_checkpoint'))}"
    )
    st.caption(
        "Optimizacion: "
        + _paper_optimization_backend_label(payload.get("optimization_backend"))
        + (
            f" | trials={int(payload.get('optuna_trials') or PAPER_OPTUNA_TRIALS_DEFAULT)}"
            if _paper_normalize_optimization_backend(payload.get("optimization_backend")) == "optuna"
            else ""
        )
    )
    if payload.get("checkpoint_manifest_path"):
        st.caption(f"Manifest: {payload.get('checkpoint_manifest_path')}")
    status_cols = st.columns(5)
    status_cols[0].metric("Frozen", str(frozen_payload.get("status") or ""))
    status_cols[1].metric("Raw", str(raw_payload.get("status") or ""))
    update_emb_status_str = str(update_emb_payload.get("status") or "n/a")
    status_cols[2].metric("Update Emb", update_emb_status_str)
    compare_status = str(compare_payload.get("status") or "")
    compare_metric = "PASS" if compare_payload.get("passed") else "SKIP" if compare_status == "skipped" else "BLOCK"
    latex_metric = "Promovido" if payload.get("latex_promoted") else "Staging" if payload.get("candidate_paths") else "Omitido"
    status_cols[3].metric("Compare", compare_metric)
    status_cols[4].metric("LaTeX", latex_metric)
    st.caption(
        f"Rutas ejecutadas: frozen={payload_route_options['run_frozen']} | "
        f"raw={payload_route_options['run_raw']} | "
        f"update_emb={payload_route_options['run_update_embeddings']}"
    )
    if compare_payload.get("passed"):
        st.success(str(compare_payload.get("reason") or "Rutas alineadas."))
    elif compare_status == "skipped":
        st.info(str(compare_payload.get("reason") or "Comparacion omitida por configuracion de rutas."))
    else:
        st.warning(str(compare_payload.get("reason") or "Replica bloqueada por desalineacion."))

    route_tab_names = ["Frozen", "Raw", "Actualizar Emb", "Compare/Export"]
    route_tabs = st.tabs(route_tab_names)
    with route_tabs[0]:
        st.dataframe(_paper_route_summary_df(frozen_payload), width="stretch")
        if str(frozen_payload.get("status") or "") == "skipped":
            st.info(str(frozen_payload.get("status_message") or "La ruta frozen fue omitida."))
        if isinstance(frozen_payload.get("comparison_df"), pd.DataFrame) and not frozen_payload["comparison_df"].empty:
            st.dataframe(frozen_payload["comparison_df"], width="stretch")
        if isinstance(frozen_payload.get("metricas_df"), pd.DataFrame) and not frozen_payload["metricas_df"].empty:
            st.dataframe(frozen_payload["metricas_df"], width="stretch")
        for result in frozen_payload.get("model_results") or []:
            st.caption(
                f"{result.get('model_title')}: k*={int(result.get('selected_k') or 0)} | "
                f"variables candidatas={int(result.get('candidate_feature_count') or 0)}"
            )
            _render_confusion_matrix_summary(
                str(result.get("model_title") or result.get("model_code") or "paper_model"),
                result.get("metrics") or {},
            )

    with route_tabs[1]:
        st.dataframe(_paper_route_summary_df(raw_payload), width="stretch")
        if str(raw_payload.get("status") or "") == "skipped":
            st.info(str(raw_payload.get("status_message") or "La ruta raw fue omitida."))
        elif str(raw_payload.get("status") or "") != "ok":
            st.warning(str(raw_payload.get("status_message") or "La ruta raw quedo bloqueada."))
        if isinstance(raw_payload.get("comparison_df"), pd.DataFrame) and not raw_payload["comparison_df"].empty:
            st.dataframe(raw_payload["comparison_df"], width="stretch")
        if isinstance(raw_payload.get("metricas_df"), pd.DataFrame) and not raw_payload["metricas_df"].empty:
            st.dataframe(raw_payload["metricas_df"], width="stretch")
        raw_build = raw_payload.get("raw_build") or {}
        if raw_build:
            st.json(raw_build)
        for result in raw_payload.get("model_results") or []:
            st.caption(
                f"{result.get('model_title')}: k*={int(result.get('selected_k') or 0)} | "
                f"variables candidatas={int(result.get('candidate_feature_count') or 0)}"
            )
            _render_confusion_matrix_summary(
                str(result.get("model_title") or result.get("model_code") or "paper_model"),
                result.get("metrics") or {},
            )

    with route_tabs[2]:
        if not update_emb_payload:
            st.info("La ruta 'Actualizar embeddings' no fue ejecutada en este checkpoint.")
        else:
            st.dataframe(_paper_route_summary_df(update_emb_payload), width="stretch")
            ue_status = str(update_emb_payload.get("status") or "")
            if ue_status == "skipped":
                st.info(str(update_emb_payload.get("status_message") or "La ruta update-emb fue omitida."))
            elif ue_status == "blocked":
                st.warning(str(update_emb_payload.get("status_message") or "La ruta update-emb quedo bloqueada."))
            if isinstance(update_emb_payload.get("comparison_df"), pd.DataFrame) and not update_emb_payload["comparison_df"].empty:
                st.dataframe(update_emb_payload["comparison_df"], width="stretch")
            if isinstance(update_emb_payload.get("metricas_df"), pd.DataFrame) and not update_emb_payload["metricas_df"].empty:
                st.dataframe(update_emb_payload["metricas_df"], width="stretch")
            ue_build = update_emb_payload.get("update_emb_build") or {}
            if ue_build:
                st.json(ue_build)
            for result in update_emb_payload.get("model_results") or []:
                st.caption(
                    f"{result.get('model_title')}: k*={int(result.get('selected_k') or 0)} | "
                    f"variables candidatas={int(result.get('candidate_feature_count') or 0)}"
                )
                _render_confusion_matrix_summary(
                    str(result.get("model_title") or result.get("model_code") or "paper_model"),
                    result.get("metrics") or {},
                )

    with route_tabs[3]:
        st.dataframe(_compare_summary_df(compare_payload), width="stretch")
        diff_df = compare_payload.get("diff_df")
        if isinstance(diff_df, pd.DataFrame) and not diff_df.empty:
            st.dataframe(diff_df, width="stretch")
        export_manifest_df = _paper_export_manifest_df(payload)
        st.dataframe(export_manifest_df, width="stretch")
        if payload.get("promoted_paths"):
            st.json(payload.get("promoted_paths"))


def _render_controlled_comparison_protocol(
    *,
    feature_group: str,
    feature_count: int,
    feature_count_per_model: int,
    split_mode: str,
    test_size: float,
    random_state: int,
    xgb_optimization_backend: str,
    xgb_optuna_trials: int,
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
            f"XGBoost={_paper_optimization_backend_label(xgb_optimization_backend)} / {xgb_tuning_profile} | "
            f"folds internos compartidos={int(tuning_folds)} | "
            f"variables por modelo={feature_count_per_model}"
        )
        xgb_detail = _xgb_search_strategy_help(xgb_tuning_profile)
        if _paper_normalize_optimization_backend(xgb_optimization_backend) == "optuna":
            xgb_detail = (
                f"Optuna (TPE) explora la misma grilla discreta configurada por el perfil "
                f"`{xgb_tuning_profile}` con {int(xgb_optuna_trials)} trials. {xgb_detail}"
            )
        st.caption(f"Detalle XGBoost: {xgb_detail}")
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
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
) -> Tuple[object, Dict[str, object], Optional[float], pd.DataFrame, Dict[str, object]]:
    base_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    requested_backend = _paper_normalize_optimization_backend(optimization_backend)
    resolved_optuna_trials = (
        max(1, int(optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT))
        if requested_backend == "optuna"
        else 0
    )
    param_grid = _default_xgb_param_grid(tuning_profile)
    search_space_size = _paper_xgb_search_space_size(param_grid)
    effective_optuna_trials = (
        min(int(resolved_optuna_trials), max(1, int(search_space_size)))
        if requested_backend == "optuna"
        else 0
    )
    search_meta: Dict[str, object] = {
        "requested_backend": requested_backend,
        "backend": requested_backend if bool(tune_hyperparameters) else "disabled",
        "search_space_size": int(search_space_size),
        "optuna_trials_requested": int(resolved_optuna_trials),
        "optuna_trials_effective": int(effective_optuna_trials),
        "optuna_trials_completed": 0,
        "tuning_profile": str(tuning_profile),
    }
    if not tune_hyperparameters:
        model = build_model("XGBoost", base_params, random_state=int(random_state))
        model.fit(X_train, y_train)
        return model, dict(base_params), None, pd.DataFrame(), search_meta

    min_class = int(y_train.value_counts().min())
    cv_folds = min(max(2, int(tuning_folds)), min_class)
    search_meta["effective_inner_folds"] = int(cv_folds)
    if cv_folds < 2:
        model = build_model("XGBoost", base_params, random_state=int(random_state))
        model.fit(X_train, y_train)
        fallback_params = {
            **base_params,
            "tuning_fallback": "clase_minoritaria_insuficiente",
        }
        return model, fallback_params, None, pd.DataFrame(), search_meta

    scoring = "f1" if pd.Series(y_train).nunique() <= 2 else "f1_macro"
    if requested_backend == "optuna" and optuna is not None:
        sampler = TPESampler(seed=int(random_state)) if TPESampler is not None else None
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial: object) -> float:
            params = {
                key: trial.suggest_categorical(str(key), list(values))
                for key, values in param_grid.items()
            }
            model = build_model(
                "XGBoost",
                {**base_params, **params},
                random_state=int(random_state),
            )
            scores = cross_validate(
                model,
                X_train,
                y_train,
                scoring=scoring,
                cv=int(cv_folds),
                n_jobs=1,
                error_score="raise",
            )
            return float(np.mean(scores["test_score"]))

        study.optimize(
            _objective,
            n_trials=int(effective_optuna_trials),
            show_progress_bar=False,
        )
        search_meta["optuna_trials_completed"] = int(len(study.trials))
        best_params = dict(study.best_trial.params or {})
        best_model = build_model(
            "XGBoost",
            {**base_params, **best_params},
            random_state=int(random_state),
        )
        best_model.fit(X_train, y_train)
        cv_results_df = _paper_optuna_search_df(study)
        best_score = float(study.best_value)
    else:
        if requested_backend == "optuna" and optuna is None:
            search_meta["backend"] = "gridsearch"
            search_meta["fallback_reason"] = "optuna_no_instalado"
        base_model = build_model("XGBoost", base_params, random_state=int(random_state))
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
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
        best_model = search.best_estimator_
        best_score = float(search.best_score_)
    if isinstance(cv_results_df, pd.DataFrame) and not cv_results_df.empty:
        cv_results_df = cv_results_df.copy()
        cv_results_df.insert(0, "optimization_backend", str(search_meta.get("backend") or requested_backend))
        cv_results_df.insert(1, "requested_optimization_backend", str(requested_backend))
    return best_model, best_params, best_score, cv_results_df, search_meta


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


def _paper_xgb_param_grid() -> Dict[str, List[object]]:
    return {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [150, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def _paper_xgb_base_params() -> Dict[str, object]:
    return {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }


def _paper_xgb_search_space_size(param_grid: Dict[str, List[object]]) -> int:
    if not param_grid:
        return 0
    total = 1
    for values in param_grid.values():
        total *= max(1, len(list(values)))
    return int(total)


def _paper_optuna_search_df(study: object) -> pd.DataFrame:
    if optuna is None or not hasattr(study, "trials"):
        return pd.DataFrame()
    records: List[Dict[str, object]] = []
    for trial in study.trials:
        record: Dict[str, object] = {
            "trial_number": int(trial.number),
            "value": float(trial.value) if trial.value is not None else np.nan,
            "state": str(trial.state),
        }
        for key, value in sorted((trial.params or {}).items()):
            record[f"param_{key}"] = value
        records.append(record)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        by=["value", "trial_number"],
        ascending=[False, True],
        ignore_index=True,
    )


def _paper_optimize_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    param_grid: Optional[Dict[str, List[object]]] = None,
    inner_folds: int = 5,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    param_grid = param_grid or _paper_xgb_param_grid()
    base_params = _paper_xgb_base_params()
    resolved_scoring = str(scoring_metric or "").strip()
    if resolved_scoring and resolved_scoring in PAPER_SCORING_METRICS:
        scoring = resolved_scoring
    else:
        scoring = _training_scoring_name(y_train)
    effective_inner_folds = _effective_cv_folds(y_train, int(inner_folds))
    requested_backend = _paper_normalize_optimization_backend(optimization_backend)
    resolved_optuna_trials = max(1, int(optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT))
    search_space_size = _paper_xgb_search_space_size(param_grid)
    effective_optuna_trials = min(int(resolved_optuna_trials), max(1, int(search_space_size)))
    search_meta: Dict[str, object] = {
        "requested_backend": requested_backend,
        "backend": requested_backend,
        "search_space_size": int(search_space_size),
        "effective_inner_folds": int(effective_inner_folds),
        "optuna_trials_requested": int(resolved_optuna_trials if requested_backend == "optuna" else 0),
        "optuna_trials_effective": int(effective_optuna_trials if requested_backend == "optuna" else 0),
        "optuna_trials_completed": 0,
    }
    if effective_inner_folds < 2:
        _emit_progress(progress_callback, 15, "Sin CV interna suficiente; ajustando fallback del modelo.")
        best_model = build_model("XGBoost", base_params, random_state=int(random_state))
        best_model.fit(X_train, y_train)
        return {
            "model": best_model,
            "best_params": {**base_params, "search_fallback": "clase_minoritaria_insuficiente"},
            "best_cv_score": np.nan,
            "search_df": pd.DataFrame(),
            "search_meta": search_meta,
        }

    if requested_backend == "optuna" and optuna is not None:
        _emit_progress(
            progress_callback,
            5,
            f"Optuna sobre {search_meta['search_space_size']} combinaciones discretas ({effective_optuna_trials} trials efectivos).",
        )
        sampler = TPESampler(seed=int(random_state)) if TPESampler is not None else None
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial: object) -> float:
            params = {
                key: trial.suggest_categorical(str(key), list(values))
                for key, values in param_grid.items()
            }
            model = build_model(
                "XGBoost",
                {**base_params, **params},
                random_state=int(random_state),
            )
            scores = cross_validate(
                model,
                X_train,
                y_train,
                scoring=scoring,
                cv=int(effective_inner_folds),
                n_jobs=1,
                error_score="raise",
            )
            return float(np.mean(scores["test_score"]))

        def _progress_callback(study_obj: object, trial: object) -> None:
            completed_trials = int(len(getattr(study_obj, "trials", [])))
            search_meta["optuna_trials_completed"] = completed_trials
            pct = min(100, max(10, int(round((completed_trials / max(1, effective_optuna_trials)) * 100))))
            _emit_progress(
                progress_callback,
                pct,
                f"Optuna trial {completed_trials}/{effective_optuna_trials} completado.",
            )

        study.optimize(
            _objective,
            n_trials=int(effective_optuna_trials),
            callbacks=[_progress_callback],
            show_progress_bar=False,
        )
        best_params = dict(getattr(study, "best_trial").params or {})
        best_cv_score = float(getattr(study, "best_value"))
        best_model = build_model(
            "XGBoost",
            {**base_params, **best_params},
            random_state=int(random_state),
        )
        best_model.fit(X_train, y_train)
        search_df = _paper_optuna_search_df(study)
    else:
        if requested_backend == "optuna" and optuna is None:
            search_meta["backend"] = "gridsearch"
            search_meta["fallback_reason"] = "optuna_no_instalado"
        _emit_progress(
            progress_callback,
            5,
            f"{_paper_optimization_backend_label(search_meta['backend'])} sobre {search_meta['search_space_size']} combinaciones.",
        )
        search = GridSearchCV(
            build_model("XGBoost", base_params, random_state=int(random_state)),
            param_grid=param_grid,
            scoring=scoring,
            cv=int(effective_inner_folds),
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = dict(search.best_params_)
        best_cv_score = float(search.best_score_)
        search_df = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score",
            ascending=True,
            ignore_index=True,
        )

    if isinstance(search_df, pd.DataFrame) and not search_df.empty:
        search_df = search_df.copy()
        search_df.insert(0, "optimization_backend", str(search_meta.get("backend") or requested_backend))
        search_df.insert(1, "requested_optimization_backend", str(requested_backend))
    _emit_progress(progress_callback, 100, "Optimizacion de hiperparametros completada.")
    return {
        "model": best_model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "search_df": search_df if isinstance(search_df, pd.DataFrame) else pd.DataFrame(),
        "search_meta": search_meta,
    }


def _paper_nested_xgb_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    param_grid: Optional[Dict[str, List[object]]] = None,
    outer_folds: int = 5,
    inner_folds: int = 5,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    feature_cols = list(X.columns)
    if not feature_cols:
        raise ValueError("No hay variables para la validacion nested de XGBoost.")
    outer_cv_folds = _effective_cv_folds(y, int(outer_folds))
    if outer_cv_folds < 2:
        raise ValueError("No hay suficientes casos por clase para la validacion nested.")
    param_grid = param_grid or _paper_xgb_param_grid()
    outer_cv = StratifiedKFold(
        n_splits=int(outer_cv_folds),
        shuffle=True,
        random_state=int(random_state),
    )
    outer_records: List[Dict[str, object]] = []
    start_time = time.time()
    _emit_progress(progress_callback, 0, "Iniciando nested CV.")

    for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        fold_start = int(round(((outer_fold - 1) / max(1, outer_cv_folds)) * 100))
        fold_end = int(round((outer_fold / max(1, outer_cv_folds)) * 100))
        _emit_progress(
            progress_callback,
            fold_start,
            f"Nested CV fold {outer_fold}/{outer_cv_folds}.",
        )
        X_train_outer = X.iloc[train_idx].reset_index(drop=True)
        X_val_outer = X.iloc[val_idx].reset_index(drop=True)
        y_train_outer = y.iloc[train_idx].reset_index(drop=True)
        y_val_outer = y.iloc[val_idx].reset_index(drop=True)

        X_train_imp, X_val_imp, _ = _fit_imputer(X_train_outer, X_val_outer)
        X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
            X_train_imp,
            y_train_outer,
            random_state=int(random_state),
        )
        optimization = _paper_optimize_xgb_model(
            X_train_bal,
            y_train_bal,
            random_state=int(random_state),
            param_grid=param_grid,
            inner_folds=int(inner_folds),
            optimization_backend=optimization_backend,
            optuna_trials=optuna_trials,
            scoring_metric=scoring_metric,
            progress_callback=_subprogress_callback(
                progress_callback,
                start=max(fold_start, min(95, fold_start + 5)),
                end=max(fold_start + 10, min(95, fold_end - 5)),
                prefix=f"Nested CV fold {outer_fold}/{outer_cv_folds} | ",
            ),
        )
        best_model = optimization["model"]
        best_params = dict(optimization.get("best_params") or {})
        best_cv_score = float(optimization.get("best_cv_score")) if not pd.isna(optimization.get("best_cv_score")) else np.nan
        search_meta = dict(optimization.get("search_meta") or {})

        y_pred = best_model.predict(X_val_imp[feature_cols])
        y_score = _predict_model_scores(best_model, X_val_imp[feature_cols])
        metrics = _classification_metrics(y_val_outer, y_pred, y_score)
        outer_records.append(
            {
                "outer_fold": int(outer_fold),
                "accuracy": float(metrics.get("accuracy") or 0.0),
                "precision": float(metrics.get("precision") or 0.0),
                "recall": float(metrics.get("recall") or 0.0),
                "f1_score": float(metrics.get("f1_score") or 0.0),
                "roc_auc": float(metrics.get("roc_auc")) if not pd.isna(metrics.get("roc_auc")) else np.nan,
                "false_negatives_positive_class": int(metrics.get("false_negatives_positive_class") or 0),
                "false_negatives_pct": float(metrics.get("false_negative_rate_positive_class") or 0.0) * 100.0,
                "validation_score": _paper_validation_score(metrics, best_cv_score=best_cv_score),
                "inner_best_score": best_cv_score,
                "optimization_backend": str(search_meta.get("backend") or _paper_normalize_optimization_backend(optimization_backend)),
                "requested_optimization_backend": str(search_meta.get("requested_backend") or _paper_normalize_optimization_backend(optimization_backend)),
                "best_params": json.dumps(best_params, ensure_ascii=True, default=_json_default),
                "balancing_meta": json.dumps(balancing_meta, ensure_ascii=True, default=_json_default),
            }
        )
        _emit_progress(
            progress_callback,
            fold_end,
            f"Nested CV fold {outer_fold}/{outer_cv_folds} completado.",
        )

    folds_df = pd.DataFrame(outer_records)
    _emit_progress(progress_callback, 100, "Nested CV completado.")
    return {
        "folds_df": folds_df,
        "summary": {
            "accuracy": float(folds_df["accuracy"].mean()),
            "precision": float(folds_df["precision"].mean()),
            "recall": float(folds_df["recall"].mean()),
            "f1_score": float(folds_df["f1_score"].mean()),
            "roc_auc": float(folds_df["roc_auc"].mean()) if "roc_auc" in folds_df.columns else np.nan,
            "false_negatives_pct": float(folds_df["false_negatives_pct"].mean()),
            "validation_score": float(folds_df["validation_score"].mean()),
            "training_time_sec": float(time.time() - start_time),
            "optimization_backend": (
                str(folds_df["optimization_backend"].iloc[0])
                if "optimization_backend" in folds_df.columns and not folds_df.empty
                else _paper_normalize_optimization_backend(optimization_backend)
            ),
            "requested_optimization_backend": (
                str(folds_df["requested_optimization_backend"].iloc[0])
                if "requested_optimization_backend" in folds_df.columns and not folds_df.empty
                else _paper_normalize_optimization_backend(optimization_backend)
            ),
        },
    }


def _paper_fit_final_xgb_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    random_state: int,
    param_grid: Optional[Dict[str, List[object]]] = None,
    inner_folds: int = 5,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    feature_cols = list(X_train.columns)
    _emit_progress(progress_callback, 5, "Preparando datos del modelo final.")
    X_train_imp, X_test_imp, imputer = _fit_imputer(X_train, X_test)
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_imp,
        y_train,
        random_state=int(random_state),
    )
    param_grid = param_grid or _paper_xgb_param_grid()
    start_time = time.time()
    optimization = _paper_optimize_xgb_model(
        X_train_bal,
        y_train_bal,
        random_state=int(random_state),
        param_grid=param_grid,
        inner_folds=int(inner_folds),
        optimization_backend=optimization_backend,
        optuna_trials=optuna_trials,
        scoring_metric=scoring_metric,
        progress_callback=_subprogress_callback(
            progress_callback,
            start=20,
            end=85,
            prefix="Modelo final | ",
        ),
    )
    best_model = optimization["model"]
    best_params = dict(optimization.get("best_params") or {})
    best_cv_score = float(optimization.get("best_cv_score")) if not pd.isna(optimization.get("best_cv_score")) else np.nan
    search_df = optimization.get("search_df")
    search_meta = dict(optimization.get("search_meta") or {})
    training_time_sec = float(time.time() - start_time)
    _emit_progress(progress_callback, 90, "Evaluando holdout final del paper.")
    y_pred = best_model.predict(X_test_imp[feature_cols])
    y_score = _predict_model_scores(best_model, X_test_imp[feature_cols])
    metrics = _classification_metrics(y_test, y_pred, y_score)
    metrics["training_time_sec"] = training_time_sec
    metrics["validation_score"] = _paper_validation_score(metrics, best_cv_score=best_cv_score)
    metrics["optimization_backend"] = str(
        search_meta.get("backend") or _paper_normalize_optimization_backend(optimization_backend)
    )
    metrics["requested_optimization_backend"] = str(
        search_meta.get("requested_backend") or _paper_normalize_optimization_backend(optimization_backend)
    )
    predictions_df = pd.DataFrame(
        {
            "severity_target": y_test.astype(int).tolist(),
            "prediction": np.asarray(y_pred, dtype=int).tolist(),
        }
    )
    if y_score is not None:
        predictions_df["score"] = np.asarray(y_score, dtype=float)
    _emit_progress(progress_callback, 100, "Modelo final listo.")
    return {
        "model": best_model,
        "metrics": metrics,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "search_df": search_df if isinstance(search_df, pd.DataFrame) else pd.DataFrame(),
        "predictions_df": predictions_df,
        "imputer": imputer,
        "balancing_meta": balancing_meta,
        "optimization": search_meta,
    }


def _paper_build_model_result(
    df: pd.DataFrame,
    *,
    model_code: str,
    feature_group: str,
    k_grid: Optional[Sequence[object]] = None,
    forced_selected_k: Optional[object] = None,
    cv_folds: Optional[object] = None,
    random_state: int,
    inner_folds: int = 5,
    outer_folds: int = 5,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
    route_name: str = "paper",
    paths: Optional[Dict[str, Path]] = None,
    manifest: Optional[Dict[str, object]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    feature_cols = _resolve_feature_group(df, feature_group)
    if not feature_cols:
        raise ValueError(f"No hay variables disponibles para {model_code}.")
    model_dir = _paper_model_dir(paths, route_name, model_code) if paths is not None else None
    final_step_id = f"{route_name}.{model_code}.final"
    if (
        paths is not None
        and manifest is not None
        and model_dir is not None
        and _paper_is_step_completed(manifest, final_step_id)
    ):
        _emit_progress(progress_callback, 100, f"{model_code}: cargado desde checkpoint.")
        return _paper_load_model_result(model_dir)

    _emit_progress(progress_callback, 0, f"{model_code}: preparando split holdout.")
    X_train, X_test, y_train, y_test, test_ids, split_meta = _prepare_holdout_split_with_ids(
        df,
        feature_cols,
        test_size=float(PAPER_PROTOCOL["test_size"]),
        random_state=int(random_state),
        split_mode=str(PAPER_PROTOCOL["split_mode"]),
    )
    X_train_imp, _, _ = _fit_imputer(X_train, X_test)
    X_train_bal, y_train_bal, ranking_balancing_meta = _maybe_balance_training_data(
        X_train_imp,
        y_train,
        random_state=int(random_state),
    )
    ranking_df = _rf_rank_features(
        X_train_bal,
        y_train_bal,
        random_state=int(random_state),
    )
    if ranking_df.empty:
        raise ValueError(f"No se pudo rankear variables para {model_code}.")

    resolved_forced_k = None if forced_selected_k is None else int(forced_selected_k)
    if resolved_forced_k is not None and resolved_forced_k < 1:
        raise ValueError(f"K compartido invalido para {model_code}: {resolved_forced_k}")
    _emit_progress(
        progress_callback,
        5,
        (
            f"{model_code}: ranking RF listo, usando K compartido={resolved_forced_k} definido por M3."
            if resolved_forced_k is not None
            else f"{model_code}: ranking RF listo, iniciando grilla de k."
        ),
    )
    k_records: List[Dict[str, object]] = []
    resolved_inner_folds = _paper_normalize_cv_folds(
        cv_folds if cv_folds is not None else inner_folds
    )
    resolved_outer_folds = _paper_normalize_cv_folds(
        cv_folds if cv_folds is not None else outer_folds
    )
    if resolved_forced_k is None:
        candidate_k_values = (
            _paper_candidate_k_values(len(feature_cols))
            if k_grid is None
            else _paper_candidate_k_values(len(feature_cols), k_grid=k_grid)
        )
        total_k = max(1, len(candidate_k_values))
        if model_dir is not None:
            (Path(model_dir) / "k_results").mkdir(parents=True, exist_ok=True)
        for idx, k_value in enumerate(candidate_k_values, start=1):
            k_step_id = f"{route_name}.{model_code}.k.{int(k_value)}"
            k_result_path = _paper_k_result_path(model_dir, int(k_value)) if model_dir is not None else None
            if (
                paths is not None
                and manifest is not None
                and k_result_path is not None
                and _paper_is_step_completed(manifest, k_step_id)
            ):
                k_payload = _load_json_file(k_result_path, default={})
                if isinstance(k_payload, dict) and k_payload:
                    k_records.append(k_payload)
                    continue
                _paper_invalidate_from_step(manifest, k_step_id)
                _paper_persist_manifest(Path(paths["manifest"]), manifest)
            k_start_pct = 5 + int(round(((idx - 1) / total_k) * 75))
            k_end_pct = 5 + int(round((idx / total_k) * 75))
            _emit_progress(
                progress_callback,
                k_start_pct,
                f"{model_code}: evaluando k={int(k_value)} ({idx}/{total_k}).",
            )
            if paths is not None and manifest is not None:
                _paper_mark_step_running(
                    paths,
                    manifest,
                    k_step_id,
                    stage=str(route_name),
                    description=f"{model_code} nested CV k={int(k_value)}",
                    message=f"{model_code}: nested CV para k={int(k_value)}.",
                    metadata={"model_code": model_code, "k": int(k_value)},
                )
            selected_cols = ranking_df["variable"].head(int(k_value)).tolist()
            k_start = time.time()
            nested_payload = _paper_nested_xgb_validation(
                X_train[selected_cols],
                y_train,
                random_state=int(random_state),
                outer_folds=int(resolved_outer_folds),
                inner_folds=int(resolved_inner_folds),
                optimization_backend=optimization_backend,
                optuna_trials=optuna_trials,
                scoring_metric=scoring_metric,
                progress_callback=_subprogress_callback(
                    progress_callback,
                    start=k_start_pct,
                    end=k_end_pct,
                    prefix=f"{model_code} | k={int(k_value)} | ",
                ),
            )
            summary = nested_payload["summary"]
            k_record = {
                "model_code": model_code,
                "feature_group": feature_group,
                "k": int(k_value),
                "accuracy": float(summary.get("accuracy") or 0.0),
                "precision": float(summary.get("precision") or 0.0),
                "recall": float(summary.get("recall") or 0.0),
                "f1_score": float(summary.get("f1_score") or 0.0),
                "roc_auc": float(summary.get("roc_auc")) if not pd.isna(summary.get("roc_auc")) else np.nan,
                "false_negatives_pct": float(summary.get("false_negatives_pct") or 0.0),
                "validation_score": float(summary.get("validation_score") or 0.0),
                "training_time_sec": max(float(summary.get("training_time_sec") or 0.0), float(time.time() - k_start)),
                "optimization_backend": str(
                    summary.get("optimization_backend") or _paper_normalize_optimization_backend(optimization_backend)
                ),
                "requested_optimization_backend": str(
                    summary.get("requested_optimization_backend") or _paper_normalize_optimization_backend(optimization_backend)
                ),
            }
            if k_result_path is not None:
                _atomic_write_json(k_result_path, k_record)
            if paths is not None and manifest is not None and k_result_path is not None:
                _paper_mark_step_completed(
                    paths,
                    manifest,
                    k_step_id,
                    stage=str(route_name),
                    description=f"{model_code} nested CV k={int(k_value)}",
                    message=f"{model_code}: k={int(k_value)} persistido.",
                    artifact_paths={"k_result": str(k_result_path)},
                    metadata={"model_code": model_code, "k": int(k_value)},
                )
            k_records.append(k_record)
        k_search_df = pd.DataFrame(k_records).sort_values("k", ascending=True).reset_index(drop=True)
        _emit_progress(progress_callback, 82, f"{model_code}: seleccionando k* con la regla marginal del paper.")
        selected_k = _paper_select_k_from_search(k_search_df, epsilon=0.001)
    else:
        if resolved_forced_k > len(feature_cols):
            raise ValueError(
                f"K compartido={resolved_forced_k} excede las variables disponibles para {model_code} ({len(feature_cols)})."
            )
        selected_k = int(resolved_forced_k)
        k_search_df = pd.DataFrame(
            [
                {
                    "model_code": model_code,
                    "feature_group": feature_group,
                    "k": int(selected_k),
                    "selection_mode": "shared_from_m3",
                }
            ]
        )
        _emit_progress(progress_callback, 82, f"{model_code}: reutilizando k*={int(selected_k)} definido por M3.")
    selected_cols = ranking_df["variable"].head(int(selected_k)).tolist()
    if paths is not None and manifest is not None:
        _paper_mark_step_running(
            paths,
            manifest,
            final_step_id,
            stage=str(route_name),
            description=f"{model_code} final fit",
            message=f"{model_code}: ajustando modelo final con k*={int(selected_k)}.",
            metadata={"model_code": model_code, "selected_k": int(selected_k)},
        )
    final_payload = _paper_fit_final_xgb_model(
        X_train[selected_cols],
        X_test[selected_cols],
        y_train,
        y_test,
        random_state=int(random_state),
        inner_folds=int(resolved_inner_folds),
        optimization_backend=optimization_backend,
        optuna_trials=optuna_trials,
        scoring_metric=scoring_metric,
        progress_callback=_subprogress_callback(
            progress_callback,
            start=85,
            end=98,
            prefix=f"{model_code} | ",
        ),
    )
    predictions_df = final_payload["predictions_df"].copy()
    predictions_df.insert(0, "accident_id", test_ids.astype(str).tolist())
    predictions_df.insert(1, "model_code", str(model_code))
    _emit_progress(progress_callback, 100, f"{model_code}: ruta completada con k*={int(selected_k)}.")
    model_result = {
        "model_code": str(model_code),
        "model_title": _paper_model_title(model_code),
        "feature_group": str(feature_group),
        "candidate_feature_count": int(len(feature_cols)),
        "selected_k": int(selected_k),
        "selected_cols": selected_cols,
        "split_meta": split_meta,
        "ranking_df": ranking_df,
        "k_search_df": k_search_df,
        "metrics": final_payload["metrics"],
        "best_params": final_payload["best_params"],
        "best_cv_score": final_payload["best_cv_score"],
        "search_df": final_payload["search_df"],
        "predictions_df": predictions_df,
        "balancing_meta": {
            "ranking": ranking_balancing_meta,
            "final_fit": final_payload["balancing_meta"],
        },
        "optimization": final_payload.get("optimization") or {
            "requested_backend": _paper_normalize_optimization_backend(optimization_backend),
            "backend": _paper_normalize_optimization_backend(optimization_backend),
            "optuna_trials_requested": int(
                (optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT)
                if _paper_normalize_optimization_backend(optimization_backend) == "optuna"
                else 0
            ),
        },
        "k_selection_strategy": "shared_from_m3" if resolved_forced_k is not None else "searched_on_this_model",
    }
    if paths is not None and manifest is not None and model_dir is not None:
        artifact_paths = _paper_persist_model_result(model_result, model_dir)
        _paper_mark_step_completed(
            paths,
            manifest,
            final_step_id,
            stage=str(route_name),
            description=f"{model_code} final fit",
            message=f"{model_code}: resultado final persistido.",
            artifact_paths=artifact_paths,
            metadata={"model_code": model_code, "selected_k": int(selected_k)},
        )
    return model_result


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
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
) -> Dict[str, object]:
    X_train_imp, X_test_imp, _ = _fit_imputer(X_train, X_test)
    # Fix: RF ranking on original (pre-SMOTE) data to avoid leakage from
    # synthetic samples inflating importance of minority-correlated features.
    ranking_df = _rf_rank_features(
        X_train_imp,
        y_train,
        random_state=int(random_state),
    )
    selected_cols = ranking_df["variable"].head(max(1, int(max_features))).tolist()
    if not selected_cols:
        raise ValueError("No hay variables disponibles para RF + XGBoost.")
    # Balance *after* feature selection so XGBoost trains on balanced data.
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_imp[selected_cols],
        y_train,
        random_state=int(random_state),
    )
    best_model, best_params, best_score, search_df, search_meta = _optimize_xgb_classifier(
        X_train_bal[selected_cols],
        y_train_bal,
        random_state=int(random_state),
        tune_hyperparameters=True,
        tuning_folds=int(tuning_folds),
        tuning_profile=str(tuning_profile),
        optimization_backend=optimization_backend,
        optuna_trials=optuna_trials,
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
            "optimization_backend": str(search_meta.get("backend") or _paper_normalize_optimization_backend(optimization_backend)),
            "requested_optimization_backend": str(search_meta.get("requested_backend") or _paper_normalize_optimization_backend(optimization_backend)),
            "optuna_trials_requested": int(search_meta.get("optuna_trials_requested") or 0),
            "optuna_trials_effective": int(search_meta.get("optuna_trials_effective") or 0),
        },
        "metrics": metrics,
        "predictions": np.asarray(y_pred, dtype=int),
        "scores": y_score,
        "optimization": search_meta,
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

    # Fix: GridSearchCV and coef-based feature selection on original
    # (pre-SMOTE) data to avoid synthetic samples biasing coefficient
    # magnitudes and the resulting feature ranking.
    base_model = _build_elastic_net_logistic_regression(
        random_state=int(random_state),
        max_iter=7000,
    )
    cv_folds = _effective_cv_folds(y_train, int(tuning_folds))
    scoring = _training_scoring_name(y_train)
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
        search.fit(X_train_scaled, y_train)
        tuned_model = search.best_estimator_
        best_params = dict(search.best_params_)
        search_df = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score",
            ascending=True,
            ignore_index=True,
        )
        best_score = float(search.best_score_)
    else:
        tuned_model = base_model.fit(X_train_scaled, y_train)
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

    # Balance *after* feature selection, then re-train final model on
    # balanced data with the selected hyperparameters and features.
    final_params = {
        "C": float(best_params.get("C", 1.0)),
        "l1_ratio": float(best_params.get("l1_ratio", 0.5)),
    }
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_scaled[selected_cols],
        y_train,
        random_state=int(random_state),
    )
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

    # Fix: RFE on original (pre-SMOTE) data so feature elimination
    # reflects real data distribution, not synthetic sample influence.
    scoring = _training_scoring_name(y_train)
    cv_folds = _effective_cv_folds(y_train, int(tuning_folds))
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
    selector.fit(X_train_scaled, y_train)
    selected_cols = X_train_scaled.columns[selector.support_].tolist()
    if not selected_cols:
        raise ValueError("No se pudo seleccionar variables con SVM + RFE.")

    ranking_df = pd.DataFrame(
        {
            "variable": feature_cols,
            "ranking_rfe": selector.ranking_,
        }
    ).sort_values(["ranking_rfe", "variable"], ignore_index=True)

    # Balance *after* feature selection for final SVM training and GridSearchCV.
    X_train_bal, y_train_bal, balancing_meta = _maybe_balance_training_data(
        X_train_scaled[selected_cols],
        y_train,
        random_state=int(random_state),
    )

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
    xgb_optimization_backend: Optional[object] = None,
    xgb_optuna_trials: Optional[object] = None,
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
        optimization_backend=xgb_optimization_backend,
        optuna_trials=xgb_optuna_trials,
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
        "xgb_optimization_backend": str(
            _paper_normalize_optimization_backend(xgb_optimization_backend)
        ),
        "xgb_optuna_trials": int(
            max(1, int(xgb_optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT))
            if _paper_normalize_optimization_backend(xgb_optimization_backend) == "optuna"
            else 0
        ),
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


def _paper_model_summary_df(model_results: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for result in model_results:
        metrics = result.get("metrics") or {}
        class_metrics = metrics.get("class_metrics") or {}
        rows.append(
            {
                "model_code": result.get("model_code"),
                "model_title": result.get("model_title"),
                "feature_group": result.get("feature_group"),
                "selected_k": int(result.get("selected_k") or 0),
                "accuracy": float(metrics.get("accuracy") or 0.0),
                "precision": float(metrics.get("precision") or 0.0),
                "recall": float(metrics.get("recall") or 0.0),
                "f1_score": float(metrics.get("f1_score") or 0.0),
                "roc_auc": float(metrics.get("roc_auc")) if not pd.isna(metrics.get("roc_auc")) else np.nan,
                "false_negatives_positive_class": int(metrics.get("false_negatives_positive_class") or 0),
                "no_marc_precision": float((class_metrics.get("0") or {}).get("precision") or 0.0),
                "no_marc_recall": float((class_metrics.get("0") or {}).get("recall") or 0.0),
                "no_marc_f1": float((class_metrics.get("0") or {}).get("f1_score") or 0.0),
                "marc_precision": float((class_metrics.get("1") or {}).get("precision") or 0.0),
                "marc_recall": float((class_metrics.get("1") or {}).get("recall") or 0.0),
                "marc_f1": float((class_metrics.get("1") or {}).get("f1_score") or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _paper_metricas_table_df(model_results: Sequence[Dict[str, object]]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    summary_df = _paper_model_summary_df(model_results)
    if summary_df.empty:
        return pd.DataFrame()
    code_to_row = {
        str(row["model_code"]): row
        for row in summary_df.to_dict(orient="records")
    }
    records.append(
        {
            "class_label": "All",
            "metric": "False negatives",
            "M1": int(code_to_row.get("M1", {}).get("false_negatives_positive_class", 0)),
            "M2": int(code_to_row.get("M2", {}).get("false_negatives_positive_class", 0)),
            "M3": int(code_to_row.get("M3", {}).get("false_negatives_positive_class", 0)),
        }
    )
    metric_keys = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
    ]
    for class_key, class_label, prefix in [("0", "No-MARC", "no_marc"), ("1", "MARC", "marc")]:
        for metric_label, suffix in metric_keys:
            records.append(
                {
                    "class_label": class_label,
                    "metric": metric_label,
                    "M1": float(code_to_row.get("M1", {}).get(f"{prefix}_{suffix}", 0.0)),
                    "M2": float(code_to_row.get("M2", {}).get(f"{prefix}_{suffix}", 0.0)),
                    "M3": float(code_to_row.get("M3", {}).get(f"{prefix}_{suffix}", 0.0)),
                }
            )
    return pd.DataFrame(records)


def _paper_merge_predictions(model_results: Sequence[Dict[str, object]]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for result in model_results:
        model_code = str(result.get("model_code") or "").strip().lower()
        predictions_df = result.get("predictions_df")
        if not isinstance(predictions_df, pd.DataFrame) or predictions_df.empty:
            continue
        rename_map = {
            "prediction": f"pred_{model_code}",
            "score": f"score_{model_code}",
        }
        current = predictions_df.drop(columns=["model_code"], errors="ignore").rename(columns=rename_map)
        if merged is None:
            merged = current.copy()
        else:
            merged = merged.merge(
                current,
                on=["accident_id", "severity_target"],
                how="inner",
            )
    return merged if isinstance(merged, pd.DataFrame) else pd.DataFrame()


def _paper_gridsearch_tex(grid_df: pd.DataFrame) -> str:
    if grid_df is None or grid_df.empty:
        return "% No hay resultados para gridsearch_k.tex\n"
    ordered = grid_df.sort_values("k", ascending=True).reset_index(drop=True)
    lines = [
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "$k$ & Accuracy & F1 global & FNR (\\%) & Score$_{val}$ & Training time (s) \\\\",
        "\\midrule",
    ]
    for row in ordered.to_dict(orient="records"):
        lines.append(
            f"{int(row['k'])} & "
            f"{float(row['accuracy']):.3f} & "
            f"{float(row['f1_score']):.3f} & "
            f"{float(row['false_negatives_pct']):.2f} & "
            f"{float(row['validation_score']):.3f} & "
            f"{int(round(float(row['training_time_sec'])))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _paper_metricas_tex(metricas_df: pd.DataFrame) -> str:
    if metricas_df is None or metricas_df.empty:
        return "% No hay resultados para metricas_modelos.tex\n"
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\textbf{Class} & \\textbf{Metrics} & \\textbf{M1} & \\textbf{M2} & \\textbf{M3} \\\\",
        "\\midrule",
    ]
    wrote_class_separator = False
    for row in metricas_df.to_dict(orient="records"):
        class_label = str(row.get("class_label") or "")
        metric = str(row.get("metric") or "")
        m1_value = row.get("M1")
        m2_value = row.get("M2")
        m3_value = row.get("M3")
        if metric == "False negatives":
            lines.append(
                f"\\multicolumn{{2}}{{l}}{{False negatives}} & "
                f"{int(m1_value)} & {int(m2_value)} & {int(m3_value)} \\\\"
            )
            continue
        if not wrote_class_separator:
            lines.append("\\addlinespace")
            lines.append("\\multicolumn{5}{l}{\\emph{Performance by class}}\\\\")
            wrote_class_separator = True
        lines.append(
            f"{class_label} & {metric} & "
            f"{float(m1_value):.2f} & {float(m2_value):.2f} & {float(m3_value):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _paper_plot_k_search(grid_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_map: Dict[str, Path] = {}
    metric_specs = [
        ("accuracy", "Accuracy", "accuracy_vs_k.png"),
        ("f1_score", "F1 global", "f1_score_vs_k.png"),
        ("false_negatives_pct", "False Negative Rate (%)", "false_negatives_pct_vs_k.png"),
        ("validation_score", "Validation score", "validation_score_vs_k.png"),
    ]
    for metric, ylabel, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grid_df["k"], grid_df[metric], marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs k")
        ax.grid(alpha=0.25)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        file_map[filename] = path
    return file_map


def _paper_plot_metrics(metricas_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = metricas_df[metricas_df["metric"] != "False negatives"].copy()
    if filtered.empty:
        raise ValueError("No hay metricas por clase para graficar.")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    model_cols = ["M1", "M2", "M3"]
    x_labels = filtered["metric"].drop_duplicates().tolist()
    width = 0.22
    x_base = np.arange(len(x_labels))
    for ax_idx, (class_label, ax) in enumerate(zip(["No-MARC", "MARC"], axes)):
        subset = filtered[filtered["class_label"] == class_label].reset_index(drop=True)
        for model_idx, model_col in enumerate(model_cols):
            ax.bar(
                x_base + (model_idx - 1) * width,
                subset[model_col].to_numpy(dtype=float),
                width=width,
                label=model_col if ax_idx == 0 else None,
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(class_label)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(x_labels)
    axes[0].legend(loc="lower right")
    fig.suptitle("Performance metrics by class")
    path = output_dir / "metrics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _paper_build_raw_dataset(
    *,
    accidents_df: Optional[pd.DataFrame],
    paths: Optional[Dict[str, Path]] = None,
    manifest: Optional[Dict[str, object]] = None,
    execution_context: Optional[Dict[str, object]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    raw_paths = _paper_raw_build_paths(paths) if paths is not None else None
    final_step_id = "raw.build.dataset"
    if (
        paths is not None
        and manifest is not None
        and raw_paths is not None
        and _paper_is_step_completed(manifest, final_step_id)
    ):
        cached_payload = _paper_load_raw_build_payload(raw_paths)
        if isinstance(cached_payload, dict):
            _emit_progress(progress_callback, 100, "Ruta raw cargada desde checkpoint.")
            return cached_payload

    context = execution_context or _paper_build_execution_context(accidents_df)
    source_df = context.get("raw_source_df")
    precomputed_features_df = context.get("raw_features_df")
    precomputed_granular_df = context.get("raw_granular_df")
    precomputed_feature_artifact = context.get("raw_feature_artifact") or {}
    has_precomputed_features = isinstance(precomputed_features_df, pd.DataFrame) and not precomputed_features_df.empty
    if not has_precomputed_features and (not isinstance(source_df, pd.DataFrame) or source_df.empty):
        source_df = accidents_df
    if not has_precomputed_features and (source_df is None or source_df.empty):
        source_df = _paper_load_latest_processed_events()
    if not has_precomputed_features and (source_df is None or source_df.empty):
        error = PaperReplicationBlockedError("No hay eventos procesados disponibles para reconstruir la ruta raw.")
        if paths is not None and manifest is not None:
            _paper_mark_step_blocked(
                paths,
                manifest,
                "raw.build.features",
                stage="raw_build",
                description="Reconstruccion de features raw",
                message=str(error),
            )
        raise error

    flow_summary = get_flow_db_summary()
    feature_step_id = "raw.build.features"
    if (
        paths is not None
        and manifest is not None
        and raw_paths is not None
        and _paper_is_step_completed(manifest, feature_step_id)
    ):
        features_df = _load_pickle_file(raw_paths["features"], default=pd.DataFrame())
        granular_df = _load_pickle_file(raw_paths["granular"], default=pd.DataFrame())
        ranking_df = _load_pickle_file(raw_paths["feature_ranking"], default=pd.DataFrame())
    elif isinstance(precomputed_features_df, pd.DataFrame) and not precomputed_features_df.empty:
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths,
                manifest,
                feature_step_id,
                stage="raw_build",
                description="Carga de features raw precomputadas",
                message="Cargando features raw desde artifact seleccionado.",
            )
        _emit_progress(progress_callback, 10, "Cargando features raw precomputadas.")
        features_df = precomputed_features_df.copy()
        granular_df = (
            precomputed_granular_df.copy()
            if isinstance(precomputed_granular_df, pd.DataFrame)
            else pd.DataFrame()
        )
        ranking_df = _compute_relevant_feature_ranking(
            features_df,
            top_k=max(1, len(_flow_feature_columns(features_df))),
        )
        if raw_paths is not None:
            _atomic_write_pickle(features_df, raw_paths["features"])
            _atomic_write_pickle(granular_df, raw_paths["granular"])
            _atomic_write_pickle(ranking_df, raw_paths["feature_ranking"])
        if paths is not None and manifest is not None and raw_paths is not None:
            _paper_mark_step_completed(
                paths,
                manifest,
                feature_step_id,
                stage="raw_build",
                description="Carga de features raw precomputadas",
                message="Features raw precomputadas persistidas.",
                artifact_paths={
                    "features": str(raw_paths["features"]),
                    "granular": str(raw_paths["granular"]),
                    "feature_ranking": str(raw_paths["feature_ranking"]),
                },
                metadata={
                    "rows": int(len(features_df)),
                    "source": "precomputed_features_artifact",
                    "artifact": precomputed_feature_artifact,
                },
            )
    else:
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths,
                manifest,
                feature_step_id,
                stage="raw_build",
                description="Reconstruccion de features raw",
                message="Generando features raw desde eventos + flujo.",
            )
        _emit_progress(progress_callback, 10, "Reconstruyendo dataset raw con preset del paper.")
        features_df, granular_df, ranking_df = build_severity_feature_dataset(
            source_df,
            flow_db_path=flow_summary.db_path,
            windows_before=5,
            windows_after=5,
            window_size_minutes=1,
            selected_metrics=list(GRANULAR_METRIC_NAMES),
            include_deltas=True,
            text_columns=list(TEXT_SOURCE_BASE_COLUMNS),
            progress_callback=lambda value, message: _emit_progress(
                progress_callback,
                min(60, 10 + int(value * 0.5)),
                message,
            ),
        )
        if features_df.empty:
            error = PaperReplicationBlockedError("La reconstruccion raw no genero features con cobertura de flujo.")
            if paths is not None and manifest is not None:
                _paper_mark_step_blocked(
                    paths,
                    manifest,
                    feature_step_id,
                    stage="raw_build",
                    description="Reconstruccion de features raw",
                    message=str(error),
                )
            raise error
        if raw_paths is not None:
            _atomic_write_pickle(features_df, raw_paths["features"])
            _atomic_write_pickle(granular_df, raw_paths["granular"])
            _atomic_write_pickle(ranking_df, raw_paths["feature_ranking"])
        if paths is not None and manifest is not None and raw_paths is not None:
            _paper_mark_step_completed(
                paths,
                manifest,
                feature_step_id,
                stage="raw_build",
                description="Reconstruccion de features raw",
                message="Features raw persistidos.",
                artifact_paths={
                    "features": str(raw_paths["features"]),
                    "granular": str(raw_paths["granular"]),
                    "feature_ranking": str(raw_paths["feature_ranking"]),
                },
                metadata={"rows": int(len(features_df))},
            )

    embedding_step_id = "raw.build.embeddings"
    embedding_meta = _load_json_file(raw_paths["meta"], default={}) if raw_paths is not None else {}
    if (
        paths is not None
        and manifest is not None
        and raw_paths is not None
        and _paper_is_step_completed(manifest, embedding_step_id)
    ):
        embeddings_df = _load_pickle_file(raw_paths["embeddings"], default=pd.DataFrame())
        embed_cols = list((embedding_meta or {}).get("embedding_feature_columns") or [])
    else:
        model_row = context.get("transformer_model_row")
        if model_row is None:
            error = PaperReplicationBlockedError("No se encontraron modelos fine-tuneados reutilizables para la ruta raw.")
            if paths is not None and manifest is not None:
                _paper_mark_step_blocked(
                    paths,
                    manifest,
                    embedding_step_id,
                    stage="raw_build",
                    description="Extraccion de embeddings fine-tuneados",
                    message=str(error),
                )
            raise error
        model_path = str(model_row.get("output_dir_resolved") or "")
        if not model_path:
            error = PaperReplicationBlockedError("El modelo fine-tuneado no tiene `output_dir_resolved` valido.")
            if paths is not None and manifest is not None:
                _paper_mark_step_blocked(
                    paths,
                    manifest,
                    embedding_step_id,
                    stage="raw_build",
                    description="Extraccion de embeddings fine-tuneados",
                    message=str(error),
                )
            raise error
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths,
                manifest,
                embedding_step_id,
                stage="raw_build",
                description="Extraccion de embeddings fine-tuneados",
                message="Generando embeddings [CLS].",
                metadata={"transformer_model_path": model_path},
            )
        _emit_progress(progress_callback, 65, "Generando embeddings [CLS] con el modelo fine-tuneado.")
        embeddings_df, embed_cols, generated_embedding_meta = generate_text_embeddings(
            features_df,
            text_col="text_bert",
            method="transformer_finetuned",
            n_components=0,
            max_features=0,
            transformer_model_path=model_path,
            transformer_batch_size=16,
            transformer_max_length=128,
            transformer_projection="cls",
        )
        embedding_meta = {
            **generated_embedding_meta,
            "embedding_feature_columns": list(embed_cols),
            "transformer_model_path": model_path,
            "transformer_model_label": str(model_row.get("model_label") or ""),
        }
        if raw_paths is not None:
            _atomic_write_pickle(embeddings_df, raw_paths["embeddings"])
            _atomic_write_json(raw_paths["meta"], embedding_meta)
        if paths is not None and manifest is not None and raw_paths is not None:
            _paper_mark_step_completed(
                paths,
                manifest,
                embedding_step_id,
                stage="raw_build",
                description="Extraccion de embeddings fine-tuneados",
                message="Embeddings raw persistidos.",
                artifact_paths={
                    "embeddings": str(raw_paths["embeddings"]),
                    "meta": str(raw_paths["meta"]),
                },
                metadata={"embedding_count": int(len(embed_cols))},
            )

    selection_step_id = "raw.build.embedding_selection"
    if (
        paths is not None
        and manifest is not None
        and raw_paths is not None
        and _paper_is_step_completed(manifest, selection_step_id)
    ):
        ranking_embed_df = _load_pickle_file(raw_paths["embedding_ranking"], default=pd.DataFrame())
        selected_embedding_cols = list(_load_json_file(raw_paths["selected_embedding_cols"], default=[]))
    else:
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths,
                manifest,
                selection_step_id,
                stage="raw_build",
                description="Seleccion supervisada de embeddings",
                message="Rankeando embeddings con RF.",
            )
        ranking_embed_df = run_embedding_rf_analysis(embeddings_df, embed_cols)
        selected_embedding_cols = _select_top_embedding_features(
            ranking_embed_df,
            top_k=int(PAPER_EXPECTED_COUNTS["embedding_features"]),
        )
        if not selected_embedding_cols:
            error = PaperReplicationBlockedError("No se pudieron seleccionar embeddings supervisados para la ruta raw.")
            if paths is not None and manifest is not None:
                _paper_mark_step_blocked(
                    paths,
                    manifest,
                    selection_step_id,
                    stage="raw_build",
                    description="Seleccion supervisada de embeddings",
                    message=str(error),
                )
            raise error
        if raw_paths is not None:
            _atomic_write_pickle(ranking_embed_df, raw_paths["embedding_ranking"])
            _atomic_write_json(raw_paths["selected_embedding_cols"], list(selected_embedding_cols))
        if paths is not None and manifest is not None and raw_paths is not None:
            _paper_mark_step_completed(
                paths,
                manifest,
                selection_step_id,
                stage="raw_build",
                description="Seleccion supervisada de embeddings",
                message="Seleccion de embeddings persistida.",
                artifact_paths={
                    "embedding_ranking": str(raw_paths["embedding_ranking"]),
                    "selected_embedding_cols": str(raw_paths["selected_embedding_cols"]),
                },
                metadata={"selected_embedding_count": int(len(selected_embedding_cols))},
            )

    if paths is not None and manifest is not None:
        _paper_mark_step_running(
            paths,
            manifest,
            final_step_id,
            stage="raw_build",
            description="Dataset raw final",
            message="Consolidando dataset raw final.",
        )
    _emit_progress(progress_callback, 85, "Reduciendo embeddings al bloque textual supervisado del paper.")
    dataset_df = _build_train_dataset_with_selected_embeddings(
        features_df,
        embeddings_df,
        selected_embedding_cols=selected_embedding_cols,
    )
    drop_cols = [
        col
        for col in _embedding_feature_columns(dataset_df)
        if col not in set(selected_embedding_cols)
    ]
    dataset_df = dataset_df.drop(columns=drop_cols, errors="ignore")
    dataset_df = _ensure_paper_dataset_columns(dataset_df, source_name="raw")
    raw_build_payload = {
        "dataset_df": dataset_df,
        "features_df": features_df,
        "granular_df": granular_df if isinstance(granular_df, pd.DataFrame) else pd.DataFrame(),
        "feature_ranking_df": ranking_df if isinstance(ranking_df, pd.DataFrame) else pd.DataFrame(),
        "embeddings_df": embeddings_df if isinstance(embeddings_df, pd.DataFrame) else pd.DataFrame(),
        "embedding_ranking_df": ranking_embed_df if isinstance(ranking_embed_df, pd.DataFrame) else pd.DataFrame(),
        "selected_embedding_cols": list(selected_embedding_cols),
        "embedding_meta": {
            **(embedding_meta or {}),
            "selected_embedding_cols": list(selected_embedding_cols),
            "selected_embedding_count": int(len(selected_embedding_cols)),
        },
    }
    artifact_paths = _paper_persist_raw_build_payload(raw_build_payload, raw_paths) if raw_paths is not None else {}
    if paths is not None and manifest is not None and raw_paths is not None:
        _paper_mark_step_completed(
            paths,
            manifest,
            final_step_id,
            stage="raw_build",
            description="Dataset raw final",
            message="Dataset raw final persistido.",
            artifact_paths=artifact_paths,
            metadata={"rows": int(len(dataset_df))},
        )
    _emit_progress(progress_callback, 100, "Ruta raw reconstruida.")
    return raw_build_payload


def _paper_build_update_embeddings_dataset(
    *,
    paths: Optional[Dict[str, Path]] = None,
    manifest: Optional[Dict[str, object]] = None,
    execution_context: Optional[Dict[str, object]] = None,
    features_source_df: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    """Build dataset for 'Actualizar embeddings' route.

    Takes the frozen dataset (2070 rows with flow features), matches text_bert
    from the Feature Engineering features_df via accident_id, generates fresh
    embeddings with the current fine-tuned transformer, selects top 200 via
    RF importance, and assembles the final dataset (flow + new top-200 embeddings).
    """
    ue_paths = _paper_update_emb_build_paths(paths) if paths is not None else None
    final_step_id = "update_emb.build.dataset"

    # ── checkpoint fast-path ────────────────────────────────────────────
    if (
        paths is not None
        and manifest is not None
        and ue_paths is not None
        and _paper_is_step_completed(manifest, final_step_id)
    ):
        cached_payload = _load_pickle_file(ue_paths["payload"], default=None)
        if isinstance(cached_payload, dict):
            _emit_progress(progress_callback, 100, "Update-emb cargado desde checkpoint.")
            return cached_payload

    context = execution_context or {}

    # ── Step 1: Load frozen base (flow features) ────────────────────────
    frozen_step_id = "update_emb.build.frozen_base"
    if (
        paths is not None
        and manifest is not None
        and ue_paths is not None
        and _paper_is_step_completed(manifest, frozen_step_id)
    ):
        frozen_df = _load_pickle_file(ue_paths["frozen_base"], default=pd.DataFrame())
    else:
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths, manifest, frozen_step_id,
                stage="update_emb_build",
                description="Carga de dataset congelado para update-emb",
                message="Cargando dataset congelado (frozen).",
            )
        _emit_progress(progress_callback, 5, "Cargando dataset congelado para actualizar embeddings.")
        frozen_df = pd.read_pickle(PAPER_FROZEN_DATASET_PATH)
        frozen_df = _ensure_paper_dataset_columns(frozen_df, source_name="update_emb_frozen_base")
        if ue_paths is not None:
            _atomic_write_pickle(frozen_df, ue_paths["frozen_base"])
        if paths is not None and manifest is not None and ue_paths is not None:
            _paper_mark_step_completed(
                paths, manifest, frozen_step_id,
                stage="update_emb_build",
                description="Carga de dataset congelado para update-emb",
                message="Dataset congelado cargado.",
                artifact_paths={"frozen_base": str(ue_paths["frozen_base"])},
                metadata={"rows": int(len(frozen_df))},
            )

    # ── Step 2: Match text_bert from features_df ────────────────────────
    text_step_id = "update_emb.build.text_match"
    if (
        paths is not None
        and manifest is not None
        and ue_paths is not None
        and _paper_is_step_completed(manifest, text_step_id)
    ):
        features_with_text = _load_pickle_file(ue_paths["features_source"], default=pd.DataFrame())
    else:
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths, manifest, text_step_id,
                stage="update_emb_build",
                description="Match de textos desde features",
                message="Vinculando text_bert desde features calculadas.",
            )
        _emit_progress(progress_callback, 10, "Vinculando text_bert desde features calculadas.")

        # Resolve source of text: prefer explicit features_source_df, then session state
        source_df = features_source_df
        if source_df is None or source_df.empty:
            source_df = st.session_state.get("nlp_sev_features_df")
        if source_df is None or source_df.empty:
            raise PaperReplicationBlockedError(
                "No hay features calculadas disponibles para obtener text_bert. "
                "Ejecute primero Feature Engineering para generar features con textos."
            )

        # Ensure accident_id in both dataframes
        if "accident_id" not in frozen_df.columns:
            raise PaperReplicationBlockedError("El dataset congelado no tiene columna 'accident_id'.")
        if "accident_id" not in source_df.columns:
            raise PaperReplicationBlockedError("El dataframe de features no tiene columna 'accident_id'.")
        if "text_bert" not in source_df.columns:
            raise PaperReplicationBlockedError("El dataframe de features no tiene columna 'text_bert'.")

        # Match text_bert via accident_id
        text_lookup = (
            source_df[["accident_id", "text_bert"]]
            .drop_duplicates(subset=["accident_id"])
            .set_index("accident_id")["text_bert"]
        )
        frozen_ids = frozen_df["accident_id"].astype(str).str.strip()
        matched_text = frozen_ids.map(text_lookup)
        coverage = int(matched_text.notna().sum())
        total = int(len(frozen_df))
        if coverage == 0:
            raise PaperReplicationBlockedError(
                f"No se encontro text_bert para ningun accidente del frozen (0/{total}). "
                "Verifique que los accident_id coincidan."
            )

        features_with_text = frozen_df.copy()
        features_with_text["text_bert"] = matched_text.values
        # Fill missing texts with empty string (will produce zero embeddings)
        features_with_text["text_bert"] = features_with_text["text_bert"].fillna("")

        if ue_paths is not None:
            _atomic_write_pickle(features_with_text, ue_paths["features_source"])
        if paths is not None and manifest is not None and ue_paths is not None:
            _paper_mark_step_completed(
                paths, manifest, text_step_id,
                stage="update_emb_build",
                description="Match de textos desde features",
                message=f"Textos vinculados: {coverage}/{total} accidentes con text_bert.",
                artifact_paths={"features_source": str(ue_paths["features_source"])},
                metadata={"coverage": coverage, "total": total, "coverage_pct": round(100.0 * coverage / max(1, total), 2)},
            )
        _emit_progress(progress_callback, 20, f"Textos vinculados: {coverage}/{total} accidentes.")

    # ── Step 3: Generate new embeddings ─────────────────────────────────
    embedding_step_id = "update_emb.build.embeddings"
    embedding_meta: Dict[str, object] = {}
    if (
        paths is not None
        and manifest is not None
        and ue_paths is not None
        and _paper_is_step_completed(manifest, embedding_step_id)
    ):
        embeddings_df = _load_pickle_file(ue_paths["embeddings"], default=pd.DataFrame())
        embedding_meta = _load_json_file(ue_paths["meta"], default={})
        embed_cols = list((embedding_meta or {}).get("embedding_feature_columns") or [])
    else:
        model_row = context.get("transformer_model_row")
        if model_row is None:
            raise PaperReplicationBlockedError(
                "No se encontro modelo fine-tuneado para generar embeddings en update-emb."
            )
        model_path = str(model_row.get("output_dir_resolved") or "")
        if not model_path:
            raise PaperReplicationBlockedError(
                "El modelo fine-tuneado no tiene output_dir_resolved valido."
            )
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths, manifest, embedding_step_id,
                stage="update_emb_build",
                description="Generacion de embeddings con transformer fine-tuneado",
                message="Generando embeddings [CLS] con el modelo fine-tuneado.",
                metadata={"transformer_model_path": model_path},
            )
        _emit_progress(progress_callback, 30, "Generando embeddings [CLS] con el modelo fine-tuneado.")
        embeddings_df, embed_cols, generated_meta = generate_text_embeddings(
            features_with_text,
            text_col="text_bert",
            method="transformer_finetuned",
            n_components=0,
            max_features=0,
            transformer_model_path=model_path,
            transformer_batch_size=16,
            transformer_max_length=128,
            transformer_projection="cls",
        )
        embedding_meta = {
            **generated_meta,
            "embedding_feature_columns": list(embed_cols),
            "transformer_model_path": model_path,
            "transformer_model_label": str(model_row.get("model_label") or ""),
        }
        if ue_paths is not None:
            _atomic_write_pickle(embeddings_df, ue_paths["embeddings"])
            _atomic_write_json(ue_paths["meta"], embedding_meta)
        if paths is not None and manifest is not None and ue_paths is not None:
            _paper_mark_step_completed(
                paths, manifest, embedding_step_id,
                stage="update_emb_build",
                description="Generacion de embeddings con transformer fine-tuneado",
                message="Embeddings generados y persistidos.",
                artifact_paths={
                    "embeddings": str(ue_paths["embeddings"]),
                    "meta": str(ue_paths["meta"]),
                },
                metadata={"embedding_count": int(len(embed_cols))},
            )
        _emit_progress(progress_callback, 60, f"Embeddings generados: {len(embed_cols)} dimensiones.")

    # ── Step 4: Select top-200 embeddings via RF importance ─────────────
    selection_step_id = "update_emb.build.embedding_selection"
    if (
        paths is not None
        and manifest is not None
        and ue_paths is not None
        and _paper_is_step_completed(manifest, selection_step_id)
    ):
        ranking_embed_df = _load_pickle_file(ue_paths["embedding_ranking"], default=pd.DataFrame())
        selected_embedding_cols = list(_load_json_file(ue_paths["selected_embedding_cols"], default=[]))
    else:
        if paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths, manifest, selection_step_id,
                stage="update_emb_build",
                description="Seleccion supervisada de embeddings (top-200)",
                message="Rankeando embeddings con RF para seleccionar top-200.",
            )
        _emit_progress(progress_callback, 65, "Rankeando embeddings con RF para seleccionar top-200.")
        ranking_embed_df = run_embedding_rf_analysis(embeddings_df, embed_cols)
        selected_embedding_cols = _select_top_embedding_features(
            ranking_embed_df,
            top_k=int(PAPER_UPDATE_EMB_TOP_K),
        )
        if not selected_embedding_cols:
            raise PaperReplicationBlockedError(
                "No se pudieron seleccionar embeddings supervisados para la ruta update-emb."
            )
        if ue_paths is not None:
            _atomic_write_pickle(ranking_embed_df, ue_paths["embedding_ranking"])
            _atomic_write_json(ue_paths["selected_embedding_cols"], list(selected_embedding_cols))
        if paths is not None and manifest is not None and ue_paths is not None:
            _paper_mark_step_completed(
                paths, manifest, selection_step_id,
                stage="update_emb_build",
                description="Seleccion supervisada de embeddings (top-200)",
                message=f"Top-{len(selected_embedding_cols)} embeddings seleccionados.",
                artifact_paths={
                    "embedding_ranking": str(ue_paths["embedding_ranking"]),
                    "selected_embedding_cols": str(ue_paths["selected_embedding_cols"]),
                },
                metadata={"selected_embedding_count": int(len(selected_embedding_cols))},
            )
        _emit_progress(progress_callback, 75, f"Top-{len(selected_embedding_cols)} embeddings seleccionados.")

    # ── Step 5: Assemble final dataset (frozen flow + new top-200 emb) ──
    if paths is not None and manifest is not None:
        _paper_mark_step_running(
            paths, manifest, final_step_id,
            stage="update_emb_build",
            description="Dataset update-emb final",
            message="Ensamblando dataset final (flow congelado + embeddings nuevos).",
        )
    _emit_progress(progress_callback, 80, "Ensamblando dataset final (flow congelado + embeddings nuevos).")

    # Start from frozen flow features (drop any old emb_ columns)
    old_emb_cols = [col for col in frozen_df.columns if col.startswith("emb_")]
    base_df = frozen_df.drop(columns=old_emb_cols, errors="ignore")

    # Merge new selected embeddings via accident_id
    dataset_df = _build_train_dataset_with_selected_embeddings(
        base_df,
        embeddings_df,
        selected_embedding_cols=selected_embedding_cols,
    )
    # Drop any extra embedding columns not in selected set
    drop_cols = [
        col for col in _embedding_feature_columns(dataset_df)
        if col not in set(selected_embedding_cols)
    ]
    dataset_df = dataset_df.drop(columns=drop_cols, errors="ignore")
    dataset_df = _ensure_paper_dataset_columns(dataset_df, source_name="update_emb")

    update_emb_build_payload = {
        "dataset_df": dataset_df,
        "frozen_base_df": frozen_df,
        "embeddings_df": embeddings_df if isinstance(embeddings_df, pd.DataFrame) else pd.DataFrame(),
        "embedding_ranking_df": ranking_embed_df if isinstance(ranking_embed_df, pd.DataFrame) else pd.DataFrame(),
        "selected_embedding_cols": list(selected_embedding_cols),
        "embedding_meta": {
            **(embedding_meta or {}),
            "selected_embedding_cols": list(selected_embedding_cols),
            "selected_embedding_count": int(len(selected_embedding_cols)),
        },
    }
    if ue_paths is not None:
        _atomic_write_pickle(dataset_df, ue_paths["dataset"])
        _atomic_write_pickle(update_emb_build_payload, ue_paths["payload"])
    if paths is not None and manifest is not None and ue_paths is not None:
        _paper_mark_step_completed(
            paths, manifest, final_step_id,
            stage="update_emb_build",
            description="Dataset update-emb final",
            message="Dataset update-emb final persistido.",
            artifact_paths={
                "dataset": str(ue_paths["dataset"]),
                "payload": str(ue_paths["payload"]),
            },
            metadata={
                "rows": int(len(dataset_df)),
                "flow_features": int(len(_flow_feature_columns(dataset_df))),
                "embedding_features": int(len([c for c in dataset_df.columns if c.startswith("emb_")])),
            },
        )
    _emit_progress(progress_callback, 100, "Dataset update-emb ensamblado.")
    return update_emb_build_payload


def _paper_run_route(
    *,
    route_name: str,
    dataset_df: pd.DataFrame,
    route_metadata: Optional[Dict[str, object]] = None,
    k_grid: Optional[Sequence[object]] = None,
    cv_folds: Optional[object] = None,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
    paths: Optional[Dict[str, Path]] = None,
    manifest: Optional[Dict[str, object]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    route_paths = _paper_route_paths(paths, route_name) if paths is not None else None
    dataset_step_id = f"{route_name}.dataset_validation"
    final_step_ids = [f"{route_name}.{model_code}.final" for model_code in PAPER_PROTOCOL["feature_groups"]]
    if (
        route_paths is not None
        and manifest is not None
        and _paper_is_step_completed(manifest, dataset_step_id)
        and all(_paper_is_step_completed(manifest, step_id) for step_id in final_step_ids)
    ):
        cached_payload = _paper_load_route_payload(route_paths)
        if isinstance(cached_payload, dict):
            _emit_progress(progress_callback, 100, f"{route_name}: cargado desde checkpoint.")
            return cached_payload

    _emit_progress(progress_callback, 0, f"{route_name}: normalizando dataset.")
    work = _ensure_paper_dataset_columns(dataset_df, source_name=route_name)
    if (
        route_paths is not None
        and manifest is not None
        and _paper_is_step_completed(manifest, dataset_step_id)
    ):
        dataset_validation = _load_json_file(route_paths["dataset_validation"], default={})
    else:
        if route_paths is not None and manifest is not None:
            _paper_mark_step_running(
                paths,
                manifest,
                dataset_step_id,
                stage=str(route_name),
                description=f"{route_name} dataset validation",
                message=f"{route_name}: validando dataset del paper.",
            )
        dataset_validation = _paper_dataset_validation_report(work, route_name=route_name)
        if route_paths is not None:
            _paper_write_json(route_paths["dataset_validation"], dataset_validation)
        if route_paths is not None and manifest is not None:
            _paper_mark_step_completed(
                paths,
                manifest,
                dataset_step_id,
                stage=str(route_name),
                description=f"{route_name} dataset validation",
                message=f"{route_name}: dataset validado.",
                artifact_paths={"dataset_validation": str(route_paths["dataset_validation"])},
                metadata={"rows": int(dataset_validation.get("rows") or 0)},
            )
    _emit_progress(
        progress_callback,
        5,
        f"{route_name}: dataset validado ({int(dataset_validation.get('rows') or 0)} filas).",
    )
    feature_group_map = {str(model_code): str(feature_group) for model_code, feature_group in PAPER_PROTOCOL["feature_groups"].items()}
    shared_k_grid = _paper_shared_k_search_grid(work, k_grid=k_grid)
    execution_order = [model_code for model_code in ["M3", "M1", "M2"] if model_code in feature_group_map]
    execution_order.extend(
        model_code
        for model_code in feature_group_map
        if model_code not in execution_order
    )
    total_models = max(1, len(execution_order))
    model_results_by_code: Dict[str, Dict[str, object]] = {}
    shared_selected_k: Optional[int] = None
    for idx, model_code in enumerate(execution_order, start=1):
        feature_group = feature_group_map[str(model_code)]
        model_start = 5 + int(round(((idx - 1) / total_models) * 90))
        model_end = 5 + int(round((idx / total_models) * 90))
        _emit_progress(
            progress_callback,
            model_start,
            f"{route_name}: ejecutando {model_code} ({idx}/{total_models}).",
        )
        model_result = _paper_build_model_result(
            work,
            model_code=str(model_code),
            feature_group=str(feature_group),
            k_grid=shared_k_grid if str(model_code) == "M3" else None,
            forced_selected_k=shared_selected_k if str(model_code) != "M3" else None,
            cv_folds=cv_folds,
            random_state=int(PAPER_PROTOCOL["random_state"]),
            optimization_backend=optimization_backend,
            optuna_trials=optuna_trials,
            scoring_metric=scoring_metric,
            route_name=str(route_name),
            paths=paths,
            manifest=manifest,
            progress_callback=_subprogress_callback(
                progress_callback,
                start=model_start,
                end=model_end,
                prefix=f"{route_name} | ",
            ),
        )
        model_results_by_code[str(model_code)] = model_result
        if str(model_code) == "M3":
            shared_selected_k = int(model_result.get("selected_k") or 0)
    model_results = [
        model_results_by_code[str(model_code)]
        for model_code in PAPER_PROTOCOL["feature_groups"]
        if str(model_code) in model_results_by_code
    ]
    _emit_progress(progress_callback, 96, f"{route_name}: consolidando tablas y predicciones.")
    predictions_df = _paper_merge_predictions(model_results)
    _emit_progress(progress_callback, 100, f"{route_name}: completado.")
    route_payload = {
        "status": "ok",
        "status_message": "",
        "route_name": route_name,
        "route_metadata": route_metadata or {},
        "dataset_validation": dataset_validation,
        "model_results": model_results,
        "comparison_df": _paper_model_summary_df(model_results),
        "metricas_df": _paper_metricas_table_df(model_results),
        "predictions_df": predictions_df,
        "optimization": {
            "backend": _paper_normalize_optimization_backend(optimization_backend),
            "optuna_trials": int(
                (optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT)
                if _paper_normalize_optimization_backend(optimization_backend) == "optuna"
                else 0
            ),
        },
        "shared_k": int(shared_selected_k or 0),
        "shared_k_grid": list(shared_k_grid),
        "m3_grid_df": next(
            (
                result.get("k_search_df")
                for result in model_results
                if str(result.get("model_code")) == "M3"
            ),
            pd.DataFrame(),
        ),
    }
    if route_paths is not None:
        _paper_persist_route_payload(route_payload, route_paths)
    return route_payload


def _paper_stage_route_payload(
    route_payload: Dict[str, object],
    *,
    route_dir: Path,
) -> None:
    route_paths = {
        "dir": route_dir,
        "summary_json": route_dir / "summary.json",
        "dataset_validation": route_dir / "dataset_validation.json",
        "comparison_csv": route_dir / "comparison.csv",
        "metricas_csv": route_dir / "metricas.csv",
        "predictions_csv": route_dir / "predictions.csv",
        "m3_grid_csv": route_dir / "m3_grid.csv",
        "payload": route_dir / "route_payload.pkl",
    }
    _paper_persist_route_payload(route_payload, route_paths)


def _paper_stage_latex_candidates(
    frozen_payload: Dict[str, object],
    *,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    m3_grid_df = frozen_payload.get("m3_grid_df")
    if not isinstance(m3_grid_df, pd.DataFrame):
        m3_grid_df = pd.DataFrame()
    metricas_df = frozen_payload.get("metricas_df")
    if not isinstance(metricas_df, pd.DataFrame):
        metricas_df = pd.DataFrame()
    image_paths = _paper_plot_k_search(m3_grid_df, output_dir)
    metrics_path = _paper_plot_metrics(metricas_df, output_dir)
    image_paths[metrics_path.name] = metrics_path
    grid_tex_path = output_dir / "gridsearch_k.tex"
    metricas_tex_path = output_dir / "metricas_modelos.tex"
    grid_tex_path.write_text(
        _paper_gridsearch_tex(m3_grid_df),
        encoding="utf-8",
    )
    metricas_tex_path.write_text(
        _paper_metricas_tex(metricas_df),
        encoding="utf-8",
    )
    image_paths["gridsearch_k.tex"] = grid_tex_path
    image_paths["metricas_modelos.tex"] = metricas_tex_path
    return image_paths


def _paper_promote_latex_assets(candidate_paths: Dict[str, Path]) -> Dict[str, str]:
    PAPER_LATEX_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_LATEX_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    promoted: Dict[str, str] = {}
    for name in [
        "accuracy_vs_k.png",
        "f1_score_vs_k.png",
        "false_negatives_pct_vs_k.png",
        "validation_score_vs_k.png",
        "metrics.png",
    ]:
        source_path = candidate_paths.get(name)
        if not source_path:
            continue
        target_path = PAPER_LATEX_IMAGES_DIR / name
        shutil.copy2(source_path, target_path)
        promoted[name] = str(target_path)
    for name in ["gridsearch_k.tex", "metricas_modelos.tex"]:
        source_path = candidate_paths.get(name)
        if not source_path:
            continue
        target_path = PAPER_LATEX_GENERATED_DIR / name
        shutil.copy2(source_path, target_path)
        promoted[name] = str(target_path)
    return promoted


def run_paper_replication(
    *,
    accidents_df: Optional[pd.DataFrame],
    run_id: Optional[str] = None,
    run_frozen: bool = True,
    run_raw: bool = True,
    run_update_embeddings: bool = False,
    features_source_df: Optional[pd.DataFrame] = None,
    k_grid: Optional[Sequence[object]] = None,
    cv_folds: Optional[object] = None,
    raw_features_artifact_row: Optional[pd.Series] = None,
    transformer_model_row_override: Optional[pd.Series] = None,
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
    scoring_metric: Optional[str] = None,
    auto_resume: bool = True,
    checkpoint_run_id_override: Optional[str] = None,
    start_fresh: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    route_options = _paper_route_options(run_frozen=run_frozen, run_raw=run_raw, run_update_embeddings=run_update_embeddings)
    if not any(route_options.values()):
        raise ValueError("Seleccione al menos una ruta para ejecutar la replica del paper.")
    resolved_optimization_backend = _paper_normalize_optimization_backend(optimization_backend)
    resolved_optuna_trials = (
        max(1, int(optuna_trials or PAPER_OPTUNA_TRIALS_DEFAULT))
        if resolved_optimization_backend == "optuna"
        else 0
    )
    resolved_scoring_metric = str(scoring_metric or PAPER_SCORING_METRIC_DEFAULT)
    if resolved_scoring_metric not in PAPER_SCORING_METRICS:
        resolved_scoring_metric = PAPER_SCORING_METRIC_DEFAULT
    execution_context = _paper_build_execution_context(
        accidents_df,
        route_options=route_options,
        k_grid=k_grid,
        cv_folds=cv_folds,
        raw_features_artifact_row=raw_features_artifact_row,
        transformer_model_row_override=transformer_model_row_override,
        optimization_backend=resolved_optimization_backend,
        optuna_trials=resolved_optuna_trials,
        scoring_metric=resolved_scoring_metric,
    )
    computed_run_id = str(execution_context.get("computed_run_id") or "")
    checkpoint_preview: Optional[Dict[str, object]] = None
    if checkpoint_run_id_override:
        checkpoint_preview = _paper_preview_checkpoint_run(
            str(checkpoint_run_id_override),
            accidents_df=accidents_df,
            execution_context=execution_context,
        )
    elif bool(auto_resume) and not bool(start_fresh):
        checkpoint_preview = _paper_preview_checkpoint(
            accidents_df,
            execution_context=execution_context,
        )

    use_checkpoint = bool(checkpoint_preview and checkpoint_preview.get("compatible")) and not bool(start_fresh)
    if use_checkpoint:
        effective_run_id = str(checkpoint_preview.get("run_id") or computed_run_id)
    elif start_fresh:
        if run_id:
            effective_run_id = str(run_id)
        else:
            effective_run_id = f"{computed_run_id}_{_stamp_now()}_{uuid.uuid4().hex[:8]}"
    else:
        effective_run_id = str(run_id or computed_run_id)

    run_dir = _paper_run_dir(effective_run_id)
    paths = _paper_run_paths(run_dir)
    _ensure_paper_run_dirs(paths)
    compare_paths = _paper_compare_paths(paths)
    export_paths = _paper_export_paths(paths)
    raw_route_paths = _paper_route_paths(paths, "raw")

    auto_resumed = False
    loaded_from_checkpoint = False
    if use_checkpoint:
        manifest = _paper_load_manifest(Path(paths["manifest"])) or _paper_initial_manifest(
            run_id=effective_run_id,
            computed_run_id=computed_run_id,
            protocol_snapshot=execution_context.get("protocol_snapshot") or {},
            input_fingerprints=execution_context.get("input_fingerprints") or {},
            checkpoint_run_id_override=checkpoint_run_id_override,
        )
        checkpoint_status = str(manifest.get("status") or "")
        checkpoint_mode = "load_completed" if checkpoint_status == "completed" else "resume"
        manifest["resume"] = {
            "auto_resumed": checkpoint_mode == "resume",
            "checkpoint_status": checkpoint_status,
            "checkpoint_mode": checkpoint_mode,
        }
        if checkpoint_status == "completed":
            loaded_from_checkpoint = True
            _emit_progress(progress_callback, 100, "Cargando checkpoint completado de la replica.")
            return _paper_assemble_payload_from_checkpoint(
                paths=paths,
                manifest=manifest,
                auto_resumed=False,
                loaded_from_checkpoint=True,
            )
        auto_resumed = True
        manifest["status"] = "running"
        manifest["result_status"] = "running"
        manifest["completed_at"] = None
        manifest["last_error"] = None
        _paper_reset_live_artifacts(paths)
        _paper_persist_manifest(Path(paths["manifest"]), manifest)
    else:
        manifest = _paper_initial_manifest(
            run_id=effective_run_id,
            computed_run_id=computed_run_id,
            protocol_snapshot=execution_context.get("protocol_snapshot") or {},
            input_fingerprints=execution_context.get("input_fingerprints") or {},
            checkpoint_run_id_override=checkpoint_run_id_override,
        )
        manifest["resume"] = {
            "auto_resumed": False,
            "checkpoint_status": None,
            "checkpoint_mode": "fresh" if start_fresh else "new",
        }
        _paper_reset_live_artifacts(paths)
        _paper_persist_manifest(Path(paths["manifest"]), manifest)

    try:
        result_status = "ok"
        if route_options["run_frozen"]:
            _emit_progress(progress_callback, 5, "Cargando dataset congelado del paper.")
            frozen_df = pd.read_pickle(PAPER_FROZEN_DATASET_PATH)
            _emit_progress(progress_callback, 8, "Dataset congelado cargado. Ejecutando protocolo M1/M2/M3.")
            frozen_payload = _paper_run_route(
                route_name="frozen",
                dataset_df=frozen_df,
                route_metadata={
                    "dataset_path": str(PAPER_FROZEN_DATASET_PATH),
                    "enabled": True,
                },
                k_grid=execution_context.get("protocol_snapshot", {}).get("k_grid"),
                cv_folds=execution_context.get("protocol_snapshot", {}).get("cv_folds"),
                optimization_backend=resolved_optimization_backend,
                optuna_trials=resolved_optuna_trials,
                scoring_metric=resolved_scoring_metric,
                paths=paths,
                manifest=manifest,
                progress_callback=_subprogress_callback(
                    progress_callback,
                    start=8,
                    end=38,
                    prefix="Frozen | ",
                ),
            )
        else:
            _emit_progress(progress_callback, 38, "Ruta frozen omitida por configuracion.")
            frozen_payload = _paper_skipped_route_payload(
                "frozen",
                reason="Ruta frozen omitida por configuracion del usuario.",
                route_metadata={"enabled": False},
            )
            _paper_persist_route_payload(frozen_payload, _paper_route_paths(paths, "frozen"))

        if route_options["run_raw"]:
            try:
                _emit_progress(progress_callback, 40, "Reconstruyendo la ruta raw desde insumos y modelo fine-tuneado.")
                raw_build = _paper_build_raw_dataset(
                    accidents_df=accidents_df,
                    paths=paths,
                    manifest=manifest,
                    execution_context=execution_context,
                    progress_callback=_subprogress_callback(
                        progress_callback,
                        start=40,
                        end=58,
                        prefix="Raw build | ",
                    ),
                )
                _emit_progress(progress_callback, 59, "Ruta raw reconstruida. Ejecutando protocolo M1/M2/M3.")
                raw_payload = _paper_run_route(
                    route_name="raw",
                    dataset_df=raw_build["dataset_df"],
                    route_metadata={
                        "embedding_meta": raw_build.get("embedding_meta") or {},
                        "selected_embedding_cols": raw_build.get("selected_embedding_cols") or [],
                        "enabled": True,
                    },
                    k_grid=execution_context.get("protocol_snapshot", {}).get("k_grid"),
                    cv_folds=execution_context.get("protocol_snapshot", {}).get("cv_folds"),
                    optimization_backend=resolved_optimization_backend,
                    optuna_trials=resolved_optuna_trials,
                    scoring_metric=resolved_scoring_metric,
                    paths=paths,
                    manifest=manifest,
                    progress_callback=_subprogress_callback(
                        progress_callback,
                        start=59,
                        end=78,
                        prefix="Raw | ",
                    ),
                )
                raw_payload["raw_build"] = {
                    "embedding_meta": raw_build.get("embedding_meta") or {},
                    "selected_embedding_cols": raw_build.get("selected_embedding_cols") or [],
                }
                _paper_persist_route_payload(raw_payload, raw_route_paths)
            except PaperReplicationBlockedError as exc:
                result_status = "blocked"
                raw_payload = {
                    "status": "blocked",
                    "status_message": str(exc),
                    "route_name": "raw",
                    "route_metadata": {"enabled": True},
                    "raw_build": {},
                    "dataset_validation": {},
                    "model_results": [],
                    "comparison_df": pd.DataFrame(),
                    "metricas_df": pd.DataFrame(),
                    "predictions_df": pd.DataFrame(),
                    "m3_grid_df": pd.DataFrame(),
                }
                _paper_persist_route_payload(raw_payload, raw_route_paths)
        else:
            _emit_progress(progress_callback, 78, "Ruta raw omitida por configuracion.")
            raw_payload = _paper_skipped_route_payload(
                "raw",
                reason="Ruta raw omitida por configuracion del usuario.",
                route_metadata={"enabled": False},
            )
            raw_payload["raw_build"] = {}
            _paper_persist_route_payload(raw_payload, raw_route_paths)

        update_emb_route_paths = _paper_route_paths(paths, "update_emb")
        if route_options["run_update_embeddings"]:
            try:
                _emit_progress(progress_callback, 79, "Construyendo dataset update-emb (frozen flow + embeddings nuevos).")
                update_emb_build = _paper_build_update_embeddings_dataset(
                    paths=paths,
                    manifest=manifest,
                    execution_context=execution_context,
                    features_source_df=features_source_df,
                    progress_callback=_subprogress_callback(
                        progress_callback,
                        start=79,
                        end=85,
                        prefix="Update-emb build | ",
                    ),
                )
                _emit_progress(progress_callback, 86, "Dataset update-emb listo. Ejecutando M1/M2/M3.")
                update_emb_payload = _paper_run_route(
                    route_name="update_emb",
                    dataset_df=update_emb_build["dataset_df"],
                    route_metadata={
                        "embedding_meta": update_emb_build.get("embedding_meta") or {},
                        "selected_embedding_cols": update_emb_build.get("selected_embedding_cols") or [],
                        "enabled": True,
                    },
                    k_grid=execution_context.get("protocol_snapshot", {}).get("k_grid"),
                    cv_folds=execution_context.get("protocol_snapshot", {}).get("cv_folds"),
                    optimization_backend=resolved_optimization_backend,
                    optuna_trials=resolved_optuna_trials,
                    scoring_metric=resolved_scoring_metric,
                    paths=paths,
                    manifest=manifest,
                    progress_callback=_subprogress_callback(
                        progress_callback,
                        start=86,
                        end=92,
                        prefix="Update-emb | ",
                    ),
                )
                update_emb_payload["update_emb_build"] = {
                    "embedding_meta": update_emb_build.get("embedding_meta") or {},
                    "selected_embedding_cols": update_emb_build.get("selected_embedding_cols") or [],
                }
                _paper_persist_route_payload(update_emb_payload, update_emb_route_paths)
            except PaperReplicationBlockedError as exc:
                update_emb_payload = {
                    "status": "blocked",
                    "status_message": str(exc),
                    "route_name": "update_emb",
                    "route_metadata": {"enabled": True},
                    "update_emb_build": {},
                    "dataset_validation": {},
                    "model_results": [],
                    "comparison_df": pd.DataFrame(),
                    "metricas_df": pd.DataFrame(),
                    "predictions_df": pd.DataFrame(),
                    "m3_grid_df": pd.DataFrame(),
                }
                _paper_persist_route_payload(update_emb_payload, update_emb_route_paths)
        else:
            update_emb_payload = _paper_skipped_route_payload(
                "update_emb",
                reason="Ruta update-emb omitida por configuracion del usuario.",
                route_metadata={"enabled": False},
            )
            update_emb_payload["update_emb_build"] = {}
            _paper_persist_route_payload(update_emb_payload, update_emb_route_paths)

        compare_step_id = "compare.routes"
        if route_options["run_frozen"] and route_options["run_raw"]:
            if _paper_is_step_completed(manifest, compare_step_id):
                compare_payload = _paper_load_compare_payload(compare_paths) or {}
            else:
                _paper_mark_step_running(
                    paths,
                    manifest,
                    compare_step_id,
                    stage="compare",
                    description="Comparacion frozen vs raw",
                    message="Comparando frozen y raw bajo tolerancia estricta.",
                )
                _emit_progress(progress_callback, 80, "Comparando frozen y raw bajo tolerancia estricta.")
                compare_payload = _paper_compare_routes(
                    frozen_payload,
                    raw_payload,
                    tolerance=float(PAPER_COMPARISON_TOLERANCE),
                )
                compare_artifacts = _paper_persist_compare_payload(compare_payload, compare_paths)
                _paper_mark_step_completed(
                    paths,
                    manifest,
                    compare_step_id,
                    stage="compare",
                    description="Comparacion frozen vs raw",
                    message=str(compare_payload.get("reason") or "Comparacion finalizada."),
                    artifact_paths=compare_artifacts,
                    metadata={"passed": bool(compare_payload.get("passed"))},
                )
        else:
            compare_payload = {
                "status": "skipped",
                "reason": "Comparacion omitida: se requiere ejecutar simultaneamente frozen y raw.",
                "passed": False,
                "max_numeric_diff": np.nan,
                "tolerance": float(PAPER_COMPARISON_TOLERANCE),
                "diff_df": pd.DataFrame(),
            }
            compare_artifacts = _paper_persist_compare_payload(compare_payload, compare_paths)
            if not _paper_is_step_completed(manifest, compare_step_id):
                _paper_mark_step_completed(
                    paths,
                    manifest,
                    compare_step_id,
                    stage="compare",
                    description="Comparacion frozen vs raw",
                    message=str(compare_payload.get("reason") or "Comparacion omitida."),
                    artifact_paths=compare_artifacts,
                    metadata={"passed": False, "skipped": True},
                )

        candidate_step_id = "export.latex_candidates"
        export_payload = _paper_load_export_payload(export_paths) if _paper_is_step_completed(manifest, candidate_step_id) else {}
        candidate_paths: Dict[str, object] = dict(export_payload.get("candidate_paths") or {})
        if not route_options["run_frozen"]:
            candidate_paths = {}
            if not _paper_is_step_completed(manifest, candidate_step_id):
                export_artifacts = _paper_persist_export_payload(
                    {
                        "candidate_paths": {},
                        "promoted_paths": {},
                        "latex_promoted": False,
                        "result_status": result_status,
                        "route_options": route_options,
                    },
                    export_paths,
                )
                _paper_mark_step_completed(
                    paths,
                    manifest,
                    candidate_step_id,
                    stage="export",
                    description="Generacion de assets candidatos LaTeX",
                    message="Assets candidatos omitidos: frozen no fue ejecutado.",
                    artifact_paths=export_artifacts,
                    metadata={"candidate_count": 0, "skipped": True},
                )
        elif not candidate_paths:
            _paper_mark_step_running(
                paths,
                manifest,
                candidate_step_id,
                stage="export",
                description="Generacion de assets candidatos LaTeX",
                message="Generando assets candidatos para LaTeX.",
            )
            _emit_progress(progress_callback, 90, "Generando assets candidatos para LaTeX.")
            staged_candidate_paths = _paper_stage_latex_candidates(
                frozen_payload,
                output_dir=export_paths["latex_candidate_dir"],
            )
            candidate_paths = {key: str(value) for key, value in staged_candidate_paths.items()}
            partial_export_payload = {
                "candidate_paths": candidate_paths,
                "promoted_paths": {},
                "latex_promoted": False,
                "result_status": result_status,
            }
            export_artifacts = _paper_persist_export_payload(partial_export_payload, export_paths)
            _paper_mark_step_completed(
                paths,
                manifest,
                candidate_step_id,
                stage="export",
                description="Generacion de assets candidatos LaTeX",
                message="Assets candidatos persistidos.",
                artifact_paths=export_artifacts,
                metadata={"candidate_count": int(len(candidate_paths))},
            )

        promote_step_id = "export.latex_promote"
        promoted_paths: Dict[str, str] = {}
        if bool(compare_payload.get("passed")):
            export_payload = _paper_load_export_payload(export_paths) if _paper_is_step_completed(manifest, promote_step_id) else {}
            promoted_paths = {
                str(key): str(value)
                for key, value in (export_payload.get("promoted_paths") or {}).items()
            }
            if not promoted_paths:
                _paper_mark_step_running(
                    paths,
                    manifest,
                    promote_step_id,
                    stage="export",
                    description="Promocion de assets LaTeX",
                    message="Promoviendo assets a LaTeX.",
                )
                _emit_progress(progress_callback, 96, "Comparacion aprobada. Promoviendo assets a LaTeX.")
                promoted_paths = _paper_promote_latex_assets(
                    {str(name): Path(path) for name, path in candidate_paths.items()}
                )
                export_payload = {
                    "candidate_paths": candidate_paths,
                    "promoted_paths": promoted_paths,
                    "latex_promoted": bool(promoted_paths),
                    "result_status": result_status,
                }
                export_artifacts = _paper_persist_export_payload(export_payload, export_paths)
                _paper_mark_step_completed(
                    paths,
                    manifest,
                    promote_step_id,
                    stage="export",
                    description="Promocion de assets LaTeX",
                    message="Assets LaTeX promovidos.",
                    artifact_paths=export_artifacts,
                    metadata={"promoted_count": int(len(promoted_paths))},
                )
        else:
            if str(compare_payload.get("status") or "") == "blocked":
                result_status = "blocked"
                _emit_progress(progress_callback, 96, "Comparacion bloqueada. Assets quedan en staging.")
            else:
                _emit_progress(progress_callback, 96, "Promocion LaTeX omitida por configuracion de rutas.")
            if not _paper_is_step_completed(manifest, promote_step_id):
                export_artifacts = _paper_persist_export_payload(
                    {
                        "candidate_paths": candidate_paths,
                        "promoted_paths": {},
                        "latex_promoted": False,
                        "result_status": result_status,
                        "route_options": route_options,
                    },
                    export_paths,
                )
                _paper_mark_step_completed(
                    paths,
                    manifest,
                    promote_step_id,
                    stage="export",
                    description="Promocion de assets LaTeX",
                    message=str(compare_payload.get("reason") or "Promocion omitida."),
                    artifact_paths=export_artifacts,
                    metadata={"promoted_count": 0, "skipped": True},
                )

        payload = {
            "run_id": str(effective_run_id),
            "run_dir": str(run_dir),
            "route_options": route_options,
            "k_grid": list(execution_context.get("protocol_snapshot", {}).get("k_grid") or []),
            "cv_folds": int(execution_context.get("protocol_snapshot", {}).get("cv_folds") or PAPER_CV_FOLDS_DEFAULT),
            "optimization_backend": resolved_optimization_backend,
            "optuna_trials": resolved_optuna_trials,
            "frozen": frozen_payload,
            "raw": raw_payload,
            "update_emb": update_emb_payload,
            "compare": compare_payload,
            "candidate_paths": {str(key): str(value) for key, value in candidate_paths.items()},
            "promoted_paths": {str(key): str(value) for key, value in promoted_paths.items()},
            "latex_promoted": bool(promoted_paths),
            "result_status": result_status,
        }
        _paper_persist_export_payload(payload, export_paths)
        manifest["status"] = "completed"
        manifest["result_status"] = str(result_status)
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["last_error"] = None
        _paper_persist_manifest(Path(paths["manifest"]), manifest)
        _emit_progress(progress_callback, 100, "Replica del paper finalizada.")
        return _paper_assemble_payload_from_checkpoint(
            paths=paths,
            manifest=manifest,
            auto_resumed=auto_resumed,
            loaded_from_checkpoint=loaded_from_checkpoint,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["result_status"] = "failed"
        manifest["completed_at"] = None
        manifest["last_error"] = str(exc)
        _paper_persist_manifest(Path(paths["manifest"]), manifest)
        raise


def _transformer_model_options() -> Dict[str, str]:
    options: Dict[str, str] = {}
    for label, model_dir in LOCAL_TRANSFORMER_MODEL_LOCATIONS:
        if model_dir.exists():
            options[label] = str(model_dir)
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


def _hf_model_api_url(model_ref: object) -> str:
    normalized_ref = str(model_ref or "").strip()
    return f"https://huggingface.co/api/models/{quote(normalized_ref, safe='/')}"


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_hf_model_siblings(model_ref: str) -> Dict[str, object]:
    api_url = _hf_model_api_url(model_ref)
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
    early_stopping_patience: Optional[int] = 2,
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
    if EarlyStoppingCallback is not None and early_stopping_patience is not None and int(early_stopping_patience) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience)))

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
            # Early stopping with patience=1 during search to save compute;
            # confirmation and final training use default patience=2.
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
                early_stopping_patience=1,
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
                confirm_objective_std=("objective", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                confirm_objective_min=("objective", "min"),
                confirm_objective_max=("objective", "max"),
            )
            .reset_index()
        )
        # Fix: use stability-adjusted score (mean - k*std) so configs with
        # slightly lower mean but much lower variance are preferred.
        # For loss-type metrics (lower is better) the penalty flips sign.
        _stability_k = 0.5
        if greater_is_better:
            confirm_summary_df["confirm_adjusted_score"] = (
                confirm_summary_df["confirm_objective_mean"]
                - _stability_k * confirm_summary_df["confirm_objective_std"]
            )
        else:
            confirm_summary_df["confirm_adjusted_score"] = (
                confirm_summary_df["confirm_objective_mean"]
                + _stability_k * confirm_summary_df["confirm_objective_std"]
            )
        confirm_summary_df = confirm_summary_df.sort_values(
            by=[
                "confirm_adjusted_score",
                "search_objective",
            ],
            ascending=[not greater_is_better, not greater_is_better],
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
    optimization_backend: Optional[object] = None,
    optuna_trials: Optional[object] = None,
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

    xgb_model, best_params, best_score, search_df, search_meta = _optimize_xgb_classifier(
        X_train_bal[selected_cols],
        y_train_bal,
        random_state=int(random_state),
        tune_hyperparameters=bool(tune_hyperparameters),
        tuning_folds=int(tuning_folds),
        tuning_profile=str(tuning_profile),
        optimization_backend=optimization_backend,
        optuna_trials=optuna_trials,
    )
    xgb_pred = xgb_model.predict(X_test_imp[selected_cols])
    xgb_score = xgb_model.predict_proba(X_test_imp[selected_cols])[:, 1]
    xgb_metrics = _classification_metrics(y_test, xgb_pred, xgb_score)

    predictions_df = pd.DataFrame(
        {
            "severity_target": y_test.astype(int).tolist(),
            "pred_xgboost": np.asarray(xgb_pred, dtype=int).tolist(),
        }
    )
    predictions_df["score_xgboost"] = np.asarray(xgb_score, dtype=float)

    return {
        "feature_group": feature_group,
        "selected_cols": selected_cols,
        "ranking_df": ranking_df,
        "balancing_meta": balancing_meta,
        "split_meta": split_meta,
        "xgb_search_df": search_df,
        "xgb_best_score": best_score,
        "xgb_optimization": search_meta,
        "predictions_df": predictions_df,
        "results": [
            {
                "model_name": "XGBoost",
                "metrics": xgb_metrics,
                "params": {
                    **best_params,
                    "optimization_backend": str(
                        search_meta.get("backend") or _paper_normalize_optimization_backend(optimization_backend)
                    ),
                    "requested_optimization_backend": str(
                        search_meta.get("requested_backend") or _paper_normalize_optimization_backend(optimization_backend)
                    ),
                    "optuna_trials_requested": int(search_meta.get("optuna_trials_requested") or 0),
                    "optuna_trials_effective": int(search_meta.get("optuna_trials_effective") or 0),
                    "tuning_profile": str(tuning_profile),
                    "tuning_folds": int(tuning_folds),
                },
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
    search_df = pd.DataFrame()
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
        search_df = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score", ascending=True, ignore_index=True,
        )
    else:
        best_model = pipe.fit(X_train, y_train)
        best_params = {"fallback": "sin_grid_search_por_clase_minoritaria"}
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]
    metrics = _classification_metrics(y_test, y_pred, y_score)

    # Extract coefficient ranking from the fitted pipeline.
    model_step = best_model.named_steps["model"]
    coef_df = pd.DataFrame(
        {
            "variable": list(feature_cols),
            "coef": np.asarray(model_step.coef_).ravel(),
        }
    )
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False, ignore_index=True)

    predictions_df = pd.DataFrame(
        {
            "severity_target": y_test.astype(int).tolist(),
            "pred_elastic_net": np.asarray(y_pred, dtype=int).tolist(),
        }
    )
    predictions_df["score_elastic_net"] = np.asarray(y_score, dtype=float)

    return {
        "model_name": "Elastic Net",
        "metrics": metrics,
        "best_params": best_params,
        "split_meta": split_meta,
        "ranking_df": coef_df,
        "search_df": search_df,
        "predictions_df": predictions_df,
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

    predictions_df = pd.DataFrame(
        {
            "severity_target": y_test.astype(int).tolist(),
            "pred_svm_rfe": np.asarray(y_pred, dtype=int).tolist(),
        }
    )
    predictions_df["score_svm_rfe"] = np.asarray(y_score, dtype=float)

    return {
        "model_name": "SVM + RFE",
        "metrics": metrics,
        "selected_cols": selected_cols,
        "ranking_df": ranking_df,
        "split_meta": split_meta,
        "predictions_df": predictions_df,
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
    has_train_dataset = isinstance(features_df, pd.DataFrame) and not features_df.empty
    accidents_df = st.session_state.get("nlp_sev_accidents_df")

    embeddings_df = st.session_state.get("nlp_sev_embeddings_df")
    selected_embedding_cols = [
        col
        for col in (st.session_state.get("nlp_sev_selected_embedding_cols") or [])
        if isinstance(col, str)
        and isinstance(embeddings_df, pd.DataFrame)
        and col in embeddings_df.columns
    ]
    active_df = pd.DataFrame()
    if has_train_dataset:
        active_df = _build_train_dataset_with_selected_embeddings(
            features_df,
            embeddings_df,
            selected_embedding_cols=selected_embedding_cols,
        )
        if active_df is None or active_df.empty:
            active_df = features_df

    feature_group_options = ["Solo flujo"]
    if has_train_dataset and _embedding_feature_columns(active_df):
        feature_group_options.extend(["Solo embeddings", "Todo"])

    if has_train_dataset and _embedding_feature_columns(active_df):
        if selected_embedding_cols:
            st.caption(
                f"Dataset de train cargado con {len(_embedding_feature_columns(active_df))} embeddings seleccionados por Analisis RF."
            )
        else:
            st.caption(
                f"Dataset de train cargado con todas las dimensiones de embeddings disponibles ({len(_embedding_feature_columns(active_df))})."
            )
    else:
        st.info(
            "Los entrenamientos legacy requieren `Feature engineering`, "
            "pero la replica del paper puede ejecutarse desde esta misma seccion."
        )

    tab_labels = ["Paper replication"]
    if has_train_dataset:
        tab_labels = [
            "RF + XGBoost",
            "RF + XGBoost + CV",
            "Elastic Net",
            "SVM + RFE",
            "Comparacion controlada",
            "Paper replication",
        ]
    sub_tabs = dict(zip(tab_labels, st.tabs(tab_labels)))

    if has_train_dataset:
        with sub_tabs["RF + XGBoost"]:
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
            tune_col1, tune_col2, tune_col3 = st.columns(3)
            with tune_col1:
                optimization_backend = st.selectbox(
                    "Backend XGBoost",
                    ["gridsearch", "optuna"],
                    index=0,
                    format_func=_paper_optimization_backend_label,
                    key="nlp_sev_holdout_xgb_optimization_backend",
                    disabled=not tune_xgb,
                )
            with tune_col2:
                tuning_profile = st.selectbox(
                    "Estrategia de busqueda XGBoost",
                    ["Rapida", "Amplia", "GridSearch original"],
                    index=0,
                    key="nlp_sev_holdout_xgb_tuning_profile",
                    disabled=not tune_xgb,
                )
            with tune_col3:
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
            optuna_trials = int(
                st.number_input(
                    "Trials de Optuna",
                    min_value=1,
                    max_value=256,
                    value=int(PAPER_OPTUNA_TRIALS_DEFAULT),
                    step=1,
                    key="nlp_sev_holdout_xgb_optuna_trials",
                    disabled=(
                        not tune_xgb
                        or _paper_normalize_optimization_backend(optimization_backend) != "optuna"
                    ),
                )
            )
            if tune_xgb:
                if _paper_normalize_optimization_backend(optimization_backend) == "optuna" and optuna is None:
                    st.warning("Optuna no esta disponible en el entorno activo. Se usara GridSearchCV como fallback.")
                if _paper_normalize_optimization_backend(optimization_backend) == "optuna":
                    st.caption(
                        f"Optuna (TPE) explorara la grilla `{tuning_profile}` con {int(optuna_trials)} trials."
                    )
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
                        optimization_backend=optimization_backend,
                        optuna_trials=optuna_trials,
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
                                "optimization_backend": _paper_normalize_optimization_backend(optimization_backend),
                                "optuna_trials": int(
                                    optuna_trials
                                    if _paper_normalize_optimization_backend(optimization_backend) == "optuna"
                                    else 0
                                ),
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
                                "optimization_backend": str(
                                    (payload.get("xgb_optimization") or {}).get("backend")
                                    or _paper_normalize_optimization_backend(optimization_backend)
                                ),
                                "optuna_trials_requested": int(
                                    (payload.get("xgb_optimization") or {}).get("optuna_trials_requested") or 0
                                ),
                            },
                        )
                        _append_session_model_result(result)
                    if isinstance(payload.get("predictions_df"), pd.DataFrame) and not payload["predictions_df"].empty:
                        _persist_artifact(
                            payload["predictions_df"],
                            stage="train",
                            artifact_name="rf_xgb_holdout_predictions",
                            run_id=run_id,
                            metadata={
                                "feature_group": feature_group,
                                "split_mode": split_mode,
                                "selected_top_k": int(top_k),
                            },
                        )
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
                            "xgb_optimization": payload.get("xgb_optimization"),
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

        with sub_tabs["RF + XGBoost + CV"]:
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
                        _append_session_model_result({
                            "model_name": f"{row['model_name']} (CV {int(folds)}f)",
                            "metrics": metrics,
                        })
                    st.success("Validacion cruzada finalizada.")
                    st.dataframe(cv_df, width="stretch")

        with sub_tabs["Elastic Net"]:
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
                    if isinstance(result.get("ranking_df"), pd.DataFrame) and not result["ranking_df"].empty:
                        _persist_artifact(
                            result["ranking_df"],
                            stage="train",
                            artifact_name="elastic_net_coefficients",
                            run_id=run_id,
                            metadata={"feature_group": feature_group, "split_mode": split_mode},
                        )
                    if isinstance(result.get("search_df"), pd.DataFrame) and not result["search_df"].empty:
                        _persist_artifact(
                            result["search_df"],
                            stage="train",
                            artifact_name="elastic_net_hyperparameter_search",
                            run_id=run_id,
                            metadata={"feature_group": feature_group, "split_mode": split_mode},
                        )
                    if isinstance(result.get("predictions_df"), pd.DataFrame) and not result["predictions_df"].empty:
                        _persist_artifact(
                            result["predictions_df"],
                            stage="train",
                            artifact_name="elastic_net_holdout_predictions",
                            run_id=run_id,
                            metadata={"feature_group": feature_group, "split_mode": split_mode},
                        )
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

        with sub_tabs["SVM + RFE"]:
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
                    if isinstance(result.get("predictions_df"), pd.DataFrame) and not result["predictions_df"].empty:
                        _persist_artifact(
                            result["predictions_df"],
                            stage="train",
                            artifact_name="svm_rfe_holdout_predictions",
                            run_id=run_id,
                            metadata={
                                "feature_group": feature_group,
                                "split_mode": split_mode,
                                "k_features": int(k_features),
                            },
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

        with sub_tabs["Comparacion controlada"]:
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
            compare_col1, compare_col2, compare_col3 = st.columns(3)
            with compare_col1:
                xgb_optimization_backend = st.selectbox(
                    "Backend XGBoost",
                    ["gridsearch", "optuna"],
                    index=0,
                    format_func=_paper_optimization_backend_label,
                    key="nlp_sev_compare_xgb_backend",
                )
            with compare_col2:
                xgb_tuning_profile = st.selectbox(
                    "Busqueda XGBoost",
                    ["Rapida", "Amplia", "GridSearch original"],
                    index=0,
                    key="nlp_sev_compare_xgb_profile",
                )
            with compare_col3:
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
            xgb_optuna_trials = int(
                st.number_input(
                    "Trials de Optuna para XGBoost",
                    min_value=1,
                    max_value=256,
                    value=int(PAPER_OPTUNA_TRIALS_DEFAULT),
                    step=1,
                    key="nlp_sev_compare_xgb_optuna_trials",
                    disabled=_paper_normalize_optimization_backend(xgb_optimization_backend) != "optuna",
                )
            )
            if _paper_normalize_optimization_backend(xgb_optimization_backend) == "optuna":
                if optuna is None:
                    st.warning("Optuna no esta disponible en el entorno activo. Se usara GridSearchCV como fallback.")
                st.caption(
                    f"Optuna (TPE) explorara la grilla `{xgb_tuning_profile}` con {int(xgb_optuna_trials)} trials."
                )
            _render_controlled_comparison_protocol(
                feature_group=feature_group,
                feature_count=feature_count,
                feature_count_per_model=int(max_features_per_model),
                split_mode=split_mode,
                test_size=float(test_size),
                random_state=int(random_state),
                xgb_optimization_backend=str(xgb_optimization_backend),
                xgb_optuna_trials=int(xgb_optuna_trials),
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
                        xgb_optimization_backend=str(xgb_optimization_backend),
                        xgb_optuna_trials=int(xgb_optuna_trials),
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
                        xgb_optimization_backend=str(xgb_optimization_backend),
                        xgb_optuna_trials=int(xgb_optuna_trials),
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

    with sub_tabs["Paper replication"]:
        _render_paper_replication_subtab(accidents_df=accidents_df)

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


def _render_historial_tab() -> None:
    """Consolidated view of all persisted model results and artifacts."""
    st.subheader("Historial de resultados")

    results_df = _load_model_results()
    if results_df.empty:
        st.info("No hay resultados registrados. Ejecute entrenamientos desde el tab Train.")
        return

    # ── 1. Tabla resumen de todos los modelos ──
    st.markdown("### Resultados de modelos")
    display_cols = [
        col for col in [
            "created_at", "run_id", "stage", "model_name", "feature_group",
            "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "false_negatives_global",
        ]
        if col in results_df.columns
    ]
    st.dataframe(
        results_df[display_cols].head(200),
        width="stretch",
        height=min(400, 35 * min(len(results_df), 12) + 38),
    )
    st.caption(f"Total de registros: {len(results_df):,}")

    # ── 2. Filtros interactivos ──
    st.markdown("### Filtrar y comparar")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        stage_options = sorted(results_df["stage"].dropna().unique().tolist())
        selected_stages = st.multiselect(
            "Etapa",
            stage_options,
            default=stage_options,
            key="nlp_sev_hist_stage_filter",
        )
    with filter_col2:
        model_options = sorted(results_df["model_name"].dropna().unique().tolist())
        selected_models = st.multiselect(
            "Modelo",
            model_options,
            default=model_options,
            key="nlp_sev_hist_model_filter",
        )
    with filter_col3:
        group_options = sorted(results_df["feature_group"].dropna().unique().tolist())
        selected_groups = st.multiselect(
            "Feature group",
            group_options,
            default=group_options,
            key="nlp_sev_hist_group_filter",
        )

    filtered_df = results_df.copy()
    if selected_stages:
        filtered_df = filtered_df[filtered_df["stage"].isin(selected_stages)]
    if selected_models:
        filtered_df = filtered_df[filtered_df["model_name"].isin(selected_models)]
    if selected_groups:
        filtered_df = filtered_df[filtered_df["feature_group"].isin(selected_groups)]

    if filtered_df.empty:
        st.warning("No hay resultados que coincidan con los filtros seleccionados.")
        return

    # ── 3. Tabla comparativa filtrada ──
    compare_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "false_negatives_global"]
    available_metrics = [m for m in compare_metrics if m in filtered_df.columns]
    compare_cols = ["created_at", "run_id", "stage", "model_name", "feature_group"] + available_metrics
    compare_cols = [c for c in compare_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[compare_cols],
        width="stretch",
        height=min(500, 35 * min(len(filtered_df), 15) + 38),
    )

    # ── 4. Grafico comparativo ──
    if len(filtered_df) >= 2 and available_metrics:
        st.markdown("### Comparacion visual")
        chart_metric = st.selectbox(
            "Metrica a comparar",
            available_metrics,
            index=available_metrics.index("f1_score") if "f1_score" in available_metrics else 0,
            key="nlp_sev_hist_chart_metric",
        )
        chart_df = filtered_df[["model_name", "feature_group", "stage", chart_metric, "created_at"]].copy()
        chart_df[chart_metric] = pd.to_numeric(chart_df[chart_metric], errors="coerce")
        chart_df = chart_df.dropna(subset=[chart_metric])
        chart_df["label"] = (
            chart_df["model_name"].astype(str) + " | "
            + chart_df["feature_group"].astype(str) + " | "
            + chart_df["stage"].astype(str)
        )
        if not chart_df.empty:
            fig = px.bar(
                chart_df.sort_values(chart_metric, ascending=False).head(30),
                x="label",
                y=chart_metric,
                color="stage",
                title=f"{chart_metric} por modelo y etapa",
                labels={"label": "Modelo | Grupo | Etapa", chart_metric: chart_metric},
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig, width="stretch")

    # ── 5. Detalle de un run_id seleccionado ──
    st.markdown("### Detalle por ejecucion")
    run_ids = filtered_df["run_id"].dropna().unique().tolist()
    if run_ids:
        selected_run = st.selectbox(
            "Seleccionar run_id",
            run_ids,
            key="nlp_sev_hist_run_select",
        )
        run_results = filtered_df[filtered_df["run_id"] == selected_run]
        for _, row in run_results.iterrows():
            with st.expander(f"{row.get('model_name', '?')} · {row.get('feature_group', '?')} · {row.get('stage', '?')}"):
                metric_cols = [c for c in available_metrics if c in row.index and pd.notna(row[c])]
                if metric_cols:
                    cols = st.columns(min(len(metric_cols), 4))
                    for idx, m in enumerate(metric_cols):
                        value = row[m]
                        fmt = f"{int(value):,}" if m == "false_negatives_global" else f"{float(value):.4f}"
                        cols[idx % len(cols)].metric(m, fmt)
                params = row.get("params")
                if isinstance(params, dict) and params:
                    st.json(params)
                metadata = row.get("metadata")
                if isinstance(metadata, dict) and metadata:
                    st.json(metadata)

    # ── 6. Catalogo de artifacts ──
    st.markdown("### Artifacts persistidos")
    catalog_df = _load_artifact_catalog()
    if catalog_df.empty:
        st.info("No hay artifacts persistidos.")
    else:
        catalog_display = [
            c for c in [
                "created_at", "run_id", "stage", "artifact_name", "row_count", "db_path",
            ]
            if c in catalog_df.columns
        ]
        st.dataframe(
            catalog_df[catalog_display].head(100),
            width="stretch",
            height=min(350, 35 * min(len(catalog_df), 10) + 38),
        )
        st.caption(f"Total de artifacts: {len(catalog_df):,}")

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
            "Historial",
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
    with tabs[5]:
        _render_historial_tab()


if __name__ == "__main__":
    main()
