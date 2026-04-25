from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCHEMA_VERSION = "2"


def _json_dumps(value: object) -> str:
    return json.dumps(
        value if value is not None else {},
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )


def _json_loads(value: Optional[str], fallback: object) -> object:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _utc_now_text() -> str:
    return datetime.now().isoformat()


def _normalize_path(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _table_columns(con: sqlite3.Connection, table_name: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def _ensure_history_records_schema(con: sqlite3.Connection) -> None:
    columns = _table_columns(con, "history_records")
    if "batch_key" not in columns:
        con.execute("ALTER TABLE history_records ADD COLUMN batch_key TEXT")


def init_db(db_path: Path) -> None:
    with _connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS history_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS history_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_uid TEXT UNIQUE NOT NULL,
                stage TEXT NOT NULL,
                created_at TEXT NOT NULL,
                starred INTEGER NOT NULL DEFAULT 0,
                context_key TEXT,
                feature_context_key TEXT,
                optuna_context_key TEXT,
                model_context_key TEXT,
                batch_key TEXT,
                event_files_json TEXT,
                features_path TEXT,
                features_source TEXT,
                features_date_min TEXT,
                features_date_max TEXT,
                tramo_label TEXT,
                feature_signature TEXT,
                model_name TEXT,
                optuna_objective TEXT,
                threshold_objective TEXT,
                calibration_method TEXT,
                balance_strategy TEXT,
                protocols_json TEXT,
                params_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                legacy_ref TEXT
            )
            """
        )
        _ensure_history_records_schema(con)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS history_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL REFERENCES history_records(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                role TEXT,
                generated INTEGER NOT NULL DEFAULT 0,
                delete_on_record_delete INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_stage ON history_records(stage)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_starred ON history_records(starred)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_context ON history_records(context_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_feature_context ON history_records(feature_context_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_optuna_context ON history_records(optuna_context_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_model_context ON history_records(model_context_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_batch ON history_records(batch_key)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_model ON history_records(model_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_calibration ON history_records(calibration_method)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_threshold ON history_records(threshold_objective)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_optuna_objective ON history_records(optuna_objective)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_records_balance ON history_records(balance_strategy)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_artifacts_record ON history_artifacts(record_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_history_artifacts_path ON history_artifacts(path)")
        con.execute(
            "INSERT OR REPLACE INTO history_meta (key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
        con.commit()


def get_meta(db_path: Path, key: str) -> Optional[str]:
    init_db(db_path)
    with _connect(db_path) as con:
        row = con.execute("SELECT value FROM history_meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_meta(db_path: Path, key: str, value: object) -> None:
    init_db(db_path)
    with _connect(db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO history_meta (key, value) VALUES (?, ?)",
            (key, str(value)),
        )
        con.commit()


def feature_signature(features: Optional[Sequence[object]]) -> str:
    values = [str(item) for item in list(features or []) if str(item).strip()]
    if not values:
        return "none"
    joined = "|".join(sorted(values))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def build_context_key(
    *,
    event_files: Optional[Sequence[object]] = None,
    features_path: Optional[object] = None,
    features_source: Optional[object] = None,
    features_date_min: Optional[object] = None,
    features_date_max: Optional[object] = None,
    tramo_label: Optional[object] = None,
    features_rows: Optional[int] = None,
    features_cols: Optional[int] = None,
    dataset_fingerprint: Optional[object] = None,
    selected_features: Optional[Sequence[object]] = None,
) -> str:
    payload = {
        "event_files": sorted(str(item) for item in list(event_files or [])),
        "features_path": str(features_path or ""),
        "features_source": str(features_source or ""),
        "features_date_min": str(features_date_min or ""),
        "features_date_max": str(features_date_max or ""),
        "tramo_label": str(tramo_label or ""),
        "features_rows": int(features_rows or 0),
        "features_cols": int(features_cols or 0),
        "dataset_fingerprint": str(dataset_fingerprint or ""),
        "feature_signature": feature_signature(selected_features),
    }
    return hashlib.md5(_json_dumps(payload).encode("utf-8")).hexdigest()


def insert_record(
    db_path: Path,
    *,
    stage: str,
    record_uid: Optional[str] = None,
    created_at: Optional[str] = None,
    context_key: Optional[str] = None,
    feature_context_key: Optional[str] = None,
    optuna_context_key: Optional[str] = None,
    model_context_key: Optional[str] = None,
    batch_key: Optional[object] = None,
    event_files: Optional[Sequence[object]] = None,
    features_path: Optional[object] = None,
    features_source: Optional[object] = None,
    features_date_min: Optional[object] = None,
    features_date_max: Optional[object] = None,
    tramo_label: Optional[object] = None,
    feature_signature_value: Optional[str] = None,
    model_name: Optional[object] = None,
    optuna_objective: Optional[object] = None,
    threshold_objective: Optional[object] = None,
    calibration_method: Optional[object] = None,
    balance_strategy: Optional[object] = None,
    protocols: Optional[Sequence[object]] = None,
    params: Optional[Dict[str, object]] = None,
    metrics: Optional[Dict[str, object]] = None,
    metadata: Optional[Dict[str, object]] = None,
    artifacts: Optional[Iterable[Dict[str, object]]] = None,
    legacy_ref: Optional[object] = None,
) -> int:
    init_db(db_path)
    created = created_at or _utc_now_text()
    uid = record_uid or hashlib.md5(
        f"{stage}|{created}|{context_key}|{legacy_ref}|{_json_dumps(metadata)}".encode("utf-8")
    ).hexdigest()
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT id FROM history_records WHERE record_uid = ?",
            (uid,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
        cur = con.execute(
            """
            INSERT INTO history_records (
                record_uid, stage, created_at, starred, context_key,
                feature_context_key, optuna_context_key, model_context_key, batch_key,
                event_files_json, features_path, features_source,
                features_date_min, features_date_max, tramo_label,
                feature_signature, model_name, optuna_objective,
                threshold_objective, calibration_method, balance_strategy,
                protocols_json, params_json, metrics_json, metadata_json, legacy_ref
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                str(stage),
                created,
                context_key,
                feature_context_key,
                optuna_context_key,
                model_context_key,
                str(batch_key) if batch_key is not None else None,
                _json_dumps(list(event_files or [])),
                str(features_path) if features_path is not None else None,
                str(features_source) if features_source is not None else None,
                str(features_date_min) if features_date_min is not None else None,
                str(features_date_max) if features_date_max is not None else None,
                str(tramo_label) if tramo_label is not None else None,
                feature_signature_value,
                str(model_name) if model_name is not None else None,
                str(optuna_objective) if optuna_objective is not None else None,
                str(threshold_objective) if threshold_objective is not None else None,
                str(calibration_method) if calibration_method is not None else None,
                str(balance_strategy) if balance_strategy is not None else None,
                _json_dumps(list(protocols or [])),
                _json_dumps(params or {}),
                _json_dumps(metrics or {}),
                _json_dumps(metadata or {}),
                str(legacy_ref) if legacy_ref is not None else None,
            ),
        )
        record_id = int(cur.lastrowid)
        for artifact in list(artifacts or []):
            path = artifact.get("path")
            if not path:
                continue
            con.execute(
                """
                INSERT INTO history_artifacts (
                    record_id, path, role, generated, delete_on_record_delete
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    str(path),
                    str(artifact.get("role") or ""),
                    1 if bool(artifact.get("generated")) else 0,
                    1 if bool(artifact.get("delete_on_record_delete")) else 0,
                ),
            )
        con.commit()
    return record_id


def maybe_insert_generation_record(
    db_path: Path,
    *,
    generated: bool,
    **kwargs: object,
) -> Optional[int]:
    if not generated:
        return None
    return insert_record(db_path, **kwargs)  # type: ignore[arg-type]


def _row_to_record(row: sqlite3.Row, artifacts: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
    item = dict(row)
    item["starred"] = bool(item.get("starred"))
    item["event_files"] = _json_loads(item.pop("event_files_json", None), [])
    item["protocols"] = _json_loads(item.pop("protocols_json", None), [])
    item["params"] = _json_loads(item.pop("params_json", None), {})
    item["metrics"] = _json_loads(item.pop("metrics_json", None), {})
    item["metadata"] = _json_loads(item.pop("metadata_json", None), {})
    item["artifacts"] = artifacts or []
    return item


SUMMARY_COLUMNS = (
    "id",
    "record_uid",
    "stage",
    "created_at",
    "starred",
    "context_key",
    "feature_context_key",
    "optuna_context_key",
    "model_context_key",
    "batch_key",
    "features_path",
    "features_source",
    "features_date_min",
    "features_date_max",
    "tramo_label",
    "feature_signature",
    "model_name",
    "optuna_objective",
    "threshold_objective",
    "calibration_method",
    "balance_strategy",
    "legacy_ref",
)


def _row_to_summary(row: sqlite3.Row) -> Dict[str, object]:
    item = dict(row)
    item["starred"] = bool(item.get("starred"))
    return item


def _record_filter_clauses(
    *,
    stage: Optional[str] = None,
    starred: Optional[bool] = None,
    feature_context_key: Optional[str] = None,
    optuna_context_key: Optional[str] = None,
    model_context_key: Optional[str] = None,
    model_name: Optional[str] = None,
    optuna_objective: Optional[str] = None,
    threshold_objective: Optional[str] = None,
    calibration_method: Optional[str] = None,
    balance_strategy: Optional[str] = None,
    features_path: Optional[str] = None,
    tramo_label: Optional[str] = None,
) -> tuple[List[str], List[object]]:
    clauses: List[str] = []
    params: List[object] = []
    filters = {
        "stage": stage,
        "feature_context_key": feature_context_key,
        "optuna_context_key": optuna_context_key,
        "model_context_key": model_context_key,
        "model_name": model_name,
        "optuna_objective": optuna_objective,
        "threshold_objective": threshold_objective,
        "calibration_method": calibration_method,
        "balance_strategy": balance_strategy,
        "features_path": features_path,
        "tramo_label": tramo_label,
    }
    for column, value in filters.items():
        if value in (None, "", "Todos"):
            continue
        clauses.append(f"{column} = ?")
        params.append(value)
    if starred is not None:
        clauses.append("starred = ?")
        params.append(1 if starred else 0)
    return clauses, params


def _records_query(
    *,
    columns: str,
    stage: Optional[str] = None,
    starred: Optional[bool] = None,
    feature_context_key: Optional[str] = None,
    optuna_context_key: Optional[str] = None,
    model_context_key: Optional[str] = None,
    model_name: Optional[str] = None,
    optuna_objective: Optional[str] = None,
    threshold_objective: Optional[str] = None,
    calibration_method: Optional[str] = None,
    balance_strategy: Optional[str] = None,
    features_path: Optional[str] = None,
    tramo_label: Optional[str] = None,
    limit: Optional[int] = None,
) -> tuple[str, List[object]]:
    clauses, params = _record_filter_clauses(
        stage=stage,
        starred=starred,
        feature_context_key=feature_context_key,
        optuna_context_key=optuna_context_key,
        model_context_key=model_context_key,
        model_name=model_name,
        optuna_objective=optuna_objective,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        balance_strategy=balance_strategy,
        features_path=features_path,
        tramo_label=tramo_label,
    )
    query = f"SELECT {columns} FROM history_records"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY starred DESC, created_at DESC, id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))
    return query, params


def list_record_summaries(
    db_path: Path,
    *,
    stage: Optional[str] = None,
    starred: Optional[bool] = None,
    feature_context_key: Optional[str] = None,
    optuna_context_key: Optional[str] = None,
    model_context_key: Optional[str] = None,
    model_name: Optional[str] = None,
    optuna_objective: Optional[str] = None,
    threshold_objective: Optional[str] = None,
    calibration_method: Optional[str] = None,
    balance_strategy: Optional[str] = None,
    features_path: Optional[str] = None,
    tramo_label: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    init_db(db_path)
    columns = ", ".join(SUMMARY_COLUMNS)
    query, params = _records_query(
        columns=columns,
        stage=stage,
        starred=starred,
        feature_context_key=feature_context_key,
        optuna_context_key=optuna_context_key,
        model_context_key=model_context_key,
        model_name=model_name,
        optuna_objective=optuna_objective,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        balance_strategy=balance_strategy,
        features_path=features_path,
        tramo_label=tramo_label,
        limit=limit,
    )
    with _connect(db_path) as con:
        rows = con.execute(query, params).fetchall()
    return [_row_to_summary(row) for row in rows]


def get_record(db_path: Path, record_id: int) -> Optional[Dict[str, object]]:
    init_db(db_path)
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT * FROM history_records WHERE id = ?",
            (int(record_id),),
        ).fetchone()
        if row is None:
            return None
        artifacts = _artifacts_for_record(con, int(record_id))
    return _row_to_record(row, artifacts)


def _artifacts_for_record(
    con: sqlite3.Connection,
    record_id: int,
) -> List[Dict[str, object]]:
    artifact_rows = con.execute(
        "SELECT * FROM history_artifacts WHERE record_id = ? ORDER BY id",
        (int(record_id),),
    ).fetchall()
    artifacts = [dict(artifact_row) for artifact_row in artifact_rows]
    for artifact in artifacts:
        artifact["generated"] = bool(artifact.get("generated"))
        artifact["delete_on_record_delete"] = bool(
            artifact.get("delete_on_record_delete")
        )
    return artifacts


def list_records(
    db_path: Path,
    *,
    stage: Optional[str] = None,
    starred: Optional[bool] = None,
    feature_context_key: Optional[str] = None,
    optuna_context_key: Optional[str] = None,
    model_context_key: Optional[str] = None,
    model_name: Optional[str] = None,
    optuna_objective: Optional[str] = None,
    threshold_objective: Optional[str] = None,
    calibration_method: Optional[str] = None,
    balance_strategy: Optional[str] = None,
    features_path: Optional[str] = None,
    tramo_label: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    init_db(db_path)
    query, params = _records_query(
        columns="*",
        stage=stage,
        starred=starred,
        feature_context_key=feature_context_key,
        optuna_context_key=optuna_context_key,
        model_context_key=model_context_key,
        model_name=model_name,
        optuna_objective=optuna_objective,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        balance_strategy=balance_strategy,
        features_path=features_path,
        tramo_label=tramo_label,
        limit=limit,
    )
    with _connect(db_path) as con:
        rows = con.execute(query, params).fetchall()
        records: List[Dict[str, object]] = []
        for row in rows:
            artifacts = _artifacts_for_record(con, int(row["id"]))
            records.append(_row_to_record(row, artifacts))
    return records


def distinct_values(db_path: Path, column: str) -> List[str]:
    allowed = {
        "stage",
        "features_path",
        "tramo_label",
        "model_name",
        "optuna_objective",
        "threshold_objective",
        "calibration_method",
        "balance_strategy",
    }
    if column not in allowed:
        raise ValueError(f"Unsupported distinct column: {column}")
    init_db(db_path)
    with _connect(db_path) as con:
        rows = con.execute(
            f"SELECT DISTINCT {column} AS value FROM history_records "
            f"WHERE {column} IS NOT NULL AND {column} != '' ORDER BY value"
        ).fetchall()
    return [str(row["value"]) for row in rows]


def set_starred(db_path: Path, record_id: int, starred: bool) -> bool:
    init_db(db_path)
    with _connect(db_path) as con:
        cur = con.execute(
            "UPDATE history_records SET starred = ? WHERE id = ?",
            (1 if starred else 0, int(record_id)),
        )
        con.commit()
        return cur.rowcount > 0


def _path_reference_count(con: sqlite3.Connection, path: str, *, excluding_record_id: int) -> int:
    normalized = _normalize_path(path)
    count = 0
    rows = con.execute(
        "SELECT record_id, path FROM history_artifacts WHERE record_id != ?",
        (int(excluding_record_id),),
    ).fetchall()
    for row in rows:
        if _normalize_path(row["path"]) == normalized:
            count += 1
    return count


def delete_record(db_path: Path, record_id: int) -> Dict[str, object]:
    init_db(db_path)
    deleted_paths: List[str] = []
    skipped_paths: List[Dict[str, object]] = []
    with _connect(db_path) as con:
        record = con.execute(
            "SELECT id FROM history_records WHERE id = ?",
            (int(record_id),),
        ).fetchone()
        if record is None:
            return {"deleted": False, "deleted_paths": [], "skipped_paths": []}
        artifacts = con.execute(
            """
            SELECT * FROM history_artifacts
            WHERE record_id = ? AND generated = 1 AND delete_on_record_delete = 1
            """,
            (int(record_id),),
        ).fetchall()
        for artifact in artifacts:
            path_text = str(artifact["path"])
            if _path_reference_count(con, path_text, excluding_record_id=int(record_id)) > 0:
                skipped_paths.append({"path": path_text, "reason": "referenced_by_other_record"})
                continue
            path = Path(path_text)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    deleted_paths.append(path_text)
                elif path.exists():
                    path.unlink()
                    deleted_paths.append(path_text)
            except Exception as exc:
                skipped_paths.append({"path": path_text, "reason": str(exc)})
        con.execute("DELETE FROM history_artifacts WHERE record_id = ?", (int(record_id),))
        con.execute("DELETE FROM history_records WHERE id = ?", (int(record_id),))
        con.commit()
    return {
        "deleted": True,
        "deleted_paths": deleted_paths,
        "skipped_paths": skipped_paths,
    }


def query_previous_optuna(
    db_path: Path,
    *,
    feature_context_key: str,
    feature_signature_value: Optional[str] = None,
    model_name: Optional[str] = None,
    optuna_objective: Optional[str] = None,
    calibration_method: Optional[str] = None,
    threshold_objective: Optional[str] = None,
) -> List[Dict[str, object]]:
    records = list_records(
        db_path,
        stage="Optuna",
        feature_context_key=feature_context_key,
        model_name=model_name,
        optuna_objective=optuna_objective,
        calibration_method=calibration_method,
        threshold_objective=threshold_objective,
    )
    if feature_signature_value in (None, "", "Todos"):
        return records
    return [
        record
        for record in records
        if str(record.get("feature_signature") or "") == str(feature_signature_value)
    ]


def query_previous_models(
    db_path: Path,
    *,
    feature_context_key: str,
    optuna_context_key: Optional[str] = None,
    model_name: Optional[str] = None,
    threshold_objective: Optional[str] = None,
    calibration_method: Optional[str] = None,
    balance_strategy: Optional[str] = None,
    protocols: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    records = list_records(
        db_path,
        stage="Modelos",
        feature_context_key=feature_context_key,
        optuna_context_key=optuna_context_key,
        model_name=model_name,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        balance_strategy=balance_strategy,
    )
    requested_protocols = {str(item) for item in list(protocols or []) if str(item)}
    if not requested_protocols:
        return records
    return [
        record
        for record in records
        if requested_protocols.intersection({str(item) for item in list(record.get("protocols") or [])})
    ]
