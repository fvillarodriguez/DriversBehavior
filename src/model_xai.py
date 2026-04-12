"""
Persistence and SHAP-based explainability helpers for crash-prediction models.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


XAI_BUNDLE_VERSION = 1
MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"
BACKGROUND_FILENAME = "background.parquet"
EXPLAIN_ROWS_FILENAME = "explain_rows.parquet"

_META_COLUMNS = {
    "target",
    "score",
    "pred",
    "threshold",
    "case_hint",
    "source_index",
}


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    return value


def _require_joblib():
    try:
        import joblib  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "No se pudo importar `joblib`, requerido para guardar/cargar bundles XAI."
        ) from exc
    return joblib


def _require_shap():
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "No se pudo importar `shap`. Instale la dependencia para usar XAI."
        ) from exc
    return shap


def _import_external_xgboost():
    src_dir = Path(__file__).resolve().parent
    local_module_path = (src_dir / "xgboost.py").resolve()
    original_sys_path = list(sys.path)
    existing_module = sys.modules.get("xgboost")

    if existing_module is not None:
        module_file = Path(str(getattr(existing_module, "__file__", "") or ""))
        try:
            module_path = module_file.resolve()
        except Exception:
            module_path = module_file
        if module_path == local_module_path:
            sys.modules.pop("xgboost", None)

    try:
        sys.path = [
            entry
            for entry in original_sys_path
            if str(Path(entry or ".").resolve()) != str(src_dir)
        ]
        xgb = importlib.import_module("xgboost")  # type: ignore
    except Exception as exc:
        raise ImportError(
            "No se pudo importar el paquete externo `xgboost`. "
            "Instale `xgboost` o renombre `src/xgboost.py` para evitar sombreado."
        ) from exc
    finally:
        sys.path = original_sys_path

    module_file = Path(str(getattr(xgb, "__file__", "") or ""))
    try:
        module_path = module_file.resolve()
    except Exception:
        module_path = module_file
    if module_path == local_module_path:
        raise ImportError(
            "Se importo el modulo local `src/xgboost.py` en lugar del paquete externo `xgboost`."
        )

    try:
        core_module = importlib.import_module("xgboost.core")  # type: ignore
    except Exception as exc:
        raise ImportError(
            "No se pudo importar `xgboost.core` desde el paquete externo `xgboost`. "
            f"Modulo cargado: {module_path}"
        ) from exc
    if not hasattr(xgb, "core"):
        setattr(xgb, "core", core_module)
    if not hasattr(core_module, "Booster"):
        raise ImportError(
            "El modulo `xgboost.core` cargado no expone `Booster`. "
            f"Modulo cargado: {module_path}"
        )

    # Dejamos el paquete externo registrado para que SHAP reutilice el modulo correcto.
    sys.modules["xgboost"] = xgb
    return xgb


def _coerce_feature_frame(
    df: Optional[pd.DataFrame], feature_cols: Sequence[str]
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=list(feature_cols))
    work = df.copy()
    missing = [col for col in feature_cols if col not in work.columns]
    for col in missing:
        work[col] = 0.0
    work = work[list(feature_cols)].fillna(0)
    return work.reset_index(drop=True)


def _coerce_explain_rows(
    df: Optional[pd.DataFrame], feature_cols: Sequence[str]
) -> pd.DataFrame:
    if df is None:
        cols = list(feature_cols) + sorted(_META_COLUMNS)
        return pd.DataFrame(columns=cols)
    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    for col in _META_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA
    feature_frame = work[list(feature_cols)].fillna(0)
    other_cols = [
        col for col in work.columns if col not in feature_cols or col in _META_COLUMNS
    ]
    ordered = pd.concat([feature_frame, work[other_cols]], axis=1)
    return ordered.reset_index(drop=True)


def bundle_file_map(bundle_dir: Path) -> Dict[str, Path]:
    return {
        "bundle_dir": bundle_dir,
        "model": bundle_dir / MODEL_FILENAME,
        "manifest": bundle_dir / MANIFEST_FILENAME,
        "background": bundle_dir / BACKGROUND_FILENAME,
        "explain_rows": bundle_dir / EXPLAIN_ROWS_FILENAME,
    }


def save_xai_bundle(
    bundle_dir: Path,
    *,
    model: object,
    feature_cols: Sequence[str],
    xai_payload: Optional[Dict[str, object]],
    manifest: Dict[str, object],
) -> Dict[str, object]:
    paths = bundle_file_map(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(feature_cols)
    background_df = _coerce_feature_frame(
        xai_payload.get("background_rows") if isinstance(xai_payload, dict) else None,
        feature_cols,
    )
    explain_rows_df = _coerce_explain_rows(
        xai_payload.get("explain_rows") if isinstance(xai_payload, dict) else None,
        feature_cols,
    )

    background_df.to_parquet(paths["background"], index=False)
    explain_rows_df.to_parquet(paths["explain_rows"], index=False)

    joblib = _require_joblib()
    joblib.dump(model, paths["model"])

    payload = dict(manifest)
    payload["bundle_version"] = int(XAI_BUNDLE_VERSION)
    payload["feature_cols"] = feature_cols
    payload["bundle_dir"] = str(bundle_dir)
    payload["background_path"] = str(paths["background"])
    payload["explain_rows_path"] = str(paths["explain_rows"])
    payload["model_path"] = str(paths["model"])
    payload["background_rows"] = int(len(background_df))
    payload["explain_rows"] = int(len(explain_rows_df))
    payload["files"] = {
        "model": MODEL_FILENAME,
        "manifest": MANIFEST_FILENAME,
        "background": BACKGROUND_FILENAME,
        "explain_rows": EXPLAIN_ROWS_FILENAME,
    }

    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, default=_json_default)
    return payload


def load_xai_bundle(bundle_dir: Path) -> Dict[str, object]:
    paths = bundle_file_map(bundle_dir)
    if not paths["manifest"].exists():
        raise FileNotFoundError(
            f"No se encontro el manifest del bundle XAI: {paths['manifest']}"
        )
    with paths["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    background_df = pd.read_parquet(paths["background"])
    explain_rows_df = pd.read_parquet(paths["explain_rows"])
    if str(manifest.get("model_name", "") or "") == "XGBoost":
        _import_external_xgboost()
    joblib = _require_joblib()
    model = joblib.load(paths["model"])

    return {
        "bundle_dir": str(bundle_dir),
        "manifest": manifest,
        "background_df": background_df,
        "explain_rows_df": explain_rows_df,
        "model": model,
    }


def _model_predict_positive(model: object, feature_cols: Sequence[str], data) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        X = data[list(feature_cols)].copy()
    else:
        X = pd.DataFrame(np.asarray(data), columns=list(feature_cols))
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X), dtype=float)
        if probs.ndim == 1:
            return probs.astype(float)
        return probs[:, -1].astype(float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _normalize_shap_values(raw_values, *, n_samples: int, n_features: int) -> np.ndarray:
    if hasattr(raw_values, "values"):
        raw_values = raw_values.values
    if isinstance(raw_values, list):
        if not raw_values:
            return np.zeros((n_samples, n_features), dtype=float)
        raw_values = raw_values[-1]
    arr = np.asarray(raw_values, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            arr = arr[:, :, -1]
        elif arr.shape[0] in {1, 2} and arr.shape[1] == n_samples:
            arr = arr[-1, :, :]
        elif arr.shape[-1] == n_features and arr.shape[0] in {1, 2}:
            arr = arr[-1, :, :]
        elif arr.shape[0] == n_samples and arr.shape[-1] == n_features:
            arr = arr[:, -1, :]
        else:
            raise ValueError(f"Forma SHAP no soportada: {arr.shape}")

    if arr.shape[0] != n_samples and arr.shape[1] == n_samples:
        arr = arr.T
    if arr.shape != (n_samples, n_features):
        raise ValueError(
            f"SHAP normalizado con forma inesperada: {arr.shape}, esperado {(n_samples, n_features)}"
        )
    return arr


def select_representative_case_rows(explain_rows_df: pd.DataFrame) -> List[Dict[str, object]]:
    if explain_rows_df.empty or "score" not in explain_rows_df.columns:
        return []

    rows = explain_rows_df.reset_index(drop=True).copy()
    cases: List[Tuple[str, str, pd.Series]] = []
    seen: set[int] = set()

    def _append_case(label: str, key: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        idx = int(frame.index[0])
        if idx in seen:
            return
        seen.add(idx)
        cases.append((label, key, rows.loc[idx]))

    score_desc = rows.sort_values("score", ascending=False)
    _append_case("Mayor score", "highest_score", score_desc)
    if {"target", "pred"}.issubset(rows.columns):
        tp = rows[(rows["target"] == 1) & (rows["pred"] == 1)].sort_values(
            "score", ascending=False
        )
        fp = rows[(rows["target"] == 0) & (rows["pred"] == 1)].sort_values(
            "score", ascending=False
        )
        fn = rows[(rows["target"] == 1) & (rows["pred"] == 0)].sort_values(
            "score", ascending=False
        )
        _append_case("TP mas fuerte", "true_positive", tp)
        _append_case("FP mas fuerte", "false_positive", fp)
        _append_case("FN mas relevante", "false_negative", fn)

    result: List[Dict[str, object]] = []
    for label, key, row in cases:
        row_payload = row.to_dict()
        row_payload["case_position"] = int(row.name)
        row_payload["case_label"] = label
        row_payload["case_key"] = key
        result.append(row_payload)
    return result


def _build_case_contributions(
    row: pd.Series,
    shap_row: np.ndarray,
    *,
    feature_cols: Sequence[str],
    top_n: int,
) -> Dict[str, object]:
    detail_df = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "value": pd.to_numeric(row[list(feature_cols)], errors="coerce"),
            "shap_value": np.asarray(shap_row, dtype=float),
        }
    )
    detail_df["abs_shap"] = detail_df["shap_value"].abs()
    detail_df["feature_group"] = np.where(
        detail_df["feature"].str.startswith("cluster_"), "Cluster", "Base"
    )
    top_positive = detail_df[detail_df["shap_value"] > 0].sort_values(
        "shap_value", ascending=False
    ).head(top_n)
    top_negative = detail_df[detail_df["shap_value"] < 0].sort_values(
        "shap_value", ascending=True
    ).head(top_n)

    meta = {
        key: row.get(key)
        for key in row.index
        if key not in feature_cols and key not in {"case_label", "case_key"}
    }
    meta["case_label"] = row.get("case_label")
    meta["case_key"] = row.get("case_key")
    return {
        "meta": meta,
        "top_positive": top_positive.reset_index(drop=True),
        "top_negative": top_negative.reset_index(drop=True),
        "all_contributions": detail_df.sort_values(
            "abs_shap", ascending=False
        ).reset_index(drop=True),
    }


def _scale_series_0_1(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(0.5, index=series.index, dtype=float)
    min_val = float(valid.min())
    max_val = float(valid.max())
    if np.isclose(min_val, max_val):
        return pd.Series(0.5, index=series.index, dtype=float)
    scaled = (numeric - min_val) / (max_val - min_val)
    return scaled.fillna(0.5).astype(float)


def _build_beeswarm_points(
    explain_rows_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_cols: Sequence[str],
    global_df: pd.DataFrame,
    *,
    max_features: int,
) -> pd.DataFrame:
    base_columns = [
        "feature",
        "feature_rank",
        "feature_group",
        "sample_id",
        "shap_value",
        "abs_shap",
        "feature_value",
        "feature_value_scaled",
        "score",
        "pred",
        "target",
        "threshold",
        "case_hint",
        "jitter",
    ]
    if explain_rows_df.empty or global_df.empty:
        return pd.DataFrame(columns=base_columns)

    top_features_df = global_df.head(max_features).copy()
    feature_to_index = {feature: idx for idx, feature in enumerate(feature_cols)}
    sample_ids = np.arange(len(explain_rows_df), dtype=int)
    parts: List[pd.DataFrame] = []

    for feature_row in top_features_df.itertuples(index=False):
        feature_name = str(feature_row.feature)
        shap_idx = feature_to_index.get(feature_name)
        if shap_idx is None:
            continue
        feature_values = pd.to_numeric(
            explain_rows_df[feature_name], errors="coerce"
        ).astype(float)
        score_series = explain_rows_df.get(
            "score", pd.Series(np.nan, index=explain_rows_df.index)
        )
        pred_series = explain_rows_df.get(
            "pred", pd.Series(pd.NA, index=explain_rows_df.index)
        )
        target_series = explain_rows_df.get(
            "target", pd.Series(pd.NA, index=explain_rows_df.index)
        )
        threshold_series = explain_rows_df.get(
            "threshold", pd.Series(np.nan, index=explain_rows_df.index)
        )
        part = pd.DataFrame(
            {
                "feature": feature_name,
                "feature_rank": int(getattr(feature_row, "rank", 0)),
                "feature_group": str(getattr(feature_row, "feature_group", "Base")),
                "sample_id": sample_ids,
                "shap_value": np.asarray(shap_values[:, shap_idx], dtype=float),
                "feature_value": feature_values,
                "score": pd.to_numeric(score_series, errors="coerce").astype(float),
                "pred": pd.to_numeric(pred_series, errors="coerce").astype("Int64"),
                "target": pd.to_numeric(target_series, errors="coerce").astype("Int64"),
                "threshold": pd.to_numeric(threshold_series, errors="coerce").astype(float),
                "case_hint": explain_rows_df.get("case_hint", pd.Series("", index=explain_rows_df.index))
                .fillna("")
                .astype(str),
            }
        )
        part["abs_shap"] = part["shap_value"].abs()
        part["feature_value_scaled"] = _scale_series_0_1(part["feature_value"])

        jitter = np.linspace(-0.35, 0.35, num=len(part), dtype=float)
        order = np.argsort(part["shap_value"].to_numpy(dtype=float), kind="mergesort")
        part["jitter"] = 0.0
        part.loc[part.index[order], "jitter"] = jitter
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=base_columns)
    beeswarm_df = pd.concat(parts, ignore_index=True)
    return beeswarm_df[base_columns]


def compute_xai_report(
    bundle_dir: Path,
    *,
    max_display_features: int = 12,
    local_top_n: int = 10,
) -> Dict[str, object]:
    bundle = load_xai_bundle(bundle_dir)
    manifest = bundle["manifest"]
    feature_cols = list(manifest.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("El manifest del bundle XAI no contiene `feature_cols`.")

    background_df = _coerce_feature_frame(bundle["background_df"], feature_cols)
    explain_rows_df = _coerce_explain_rows(bundle["explain_rows_df"], feature_cols)
    model = bundle["model"]
    model_name = str(manifest.get("model_name", "") or "")
    if model_name == "XGBoost":
        _import_external_xgboost()
    shap = _require_shap()

    X_explain = explain_rows_df[feature_cols].fillna(0)
    if X_explain.empty:
        raise ValueError("El bundle XAI no contiene filas de explicacion.")

    explainer_name = "KernelExplainer"
    if model_name in {"Random Forest", "XGBoost"}:
        explainer = shap.TreeExplainer(model)
        explainer_name = "TreeExplainer"
        raw_values = explainer.shap_values(X_explain)
    else:
        predict_fn = lambda data: _model_predict_positive(model, feature_cols, data)
        explainer = shap.KernelExplainer(predict_fn, background_df)
        raw_values = explainer.shap_values(X_explain)

    shap_values = _normalize_shap_values(
        raw_values,
        n_samples=int(len(X_explain)),
        n_features=int(len(feature_cols)),
    )

    global_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
        }
    )
    global_df["feature_group"] = np.where(
        global_df["feature"].str.startswith("cluster_"), "Cluster", "Base"
    )
    global_df = global_df.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)
    global_df["rank"] = np.arange(1, len(global_df) + 1)

    group_df = (
        global_df.groupby("feature_group", dropna=False)["mean_abs_shap"]
        .sum()
        .reset_index(name="total_mean_abs_shap")
    )
    total_contrib = float(group_df["total_mean_abs_shap"].sum())
    if total_contrib > 0:
        group_df["share"] = group_df["total_mean_abs_shap"] / total_contrib
    else:
        group_df["share"] = 0.0
    group_df = group_df.sort_values("total_mean_abs_shap", ascending=False).reset_index(
        drop=True
    )

    cluster_top_df = global_df[global_df["feature_group"] == "Cluster"].head(
        max_display_features
    )
    representative_rows = select_representative_case_rows(explain_rows_df)
    local_cases = []
    for case in representative_rows:
        case_row = pd.Series(case)
        case_position = int(case_row.get("case_position", 0))
        if case_position < 0 or case_position >= len(explain_rows_df):
            continue
        report = _build_case_contributions(
            explain_rows_df.iloc[case_position],
            shap_values[case_position],
            feature_cols=feature_cols,
            top_n=local_top_n,
        )
        report["meta"]["case_label"] = case_row.get("case_label")
        report["meta"]["case_key"] = case_row.get("case_key")
        local_cases.append(report)

    beeswarm_points = _build_beeswarm_points(
        explain_rows_df,
        shap_values,
        feature_cols,
        global_df,
        max_features=max_display_features,
    )

    return {
        "manifest": manifest,
        "explainer_name": explainer_name,
        "global_importance": global_df.head(max_display_features).reset_index(
            drop=True
        ),
        "global_importance_full": global_df,
        "group_summary": group_df,
        "cluster_top": cluster_top_df.reset_index(drop=True),
        "beeswarm_points": beeswarm_points,
        "local_cases": local_cases,
        "rows_explained": int(len(explain_rows_df)),
    }
