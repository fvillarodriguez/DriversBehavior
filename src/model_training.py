"""
Shared model training and evaluation logic for the Crash Prediction App.
"""
from typing import Dict, List, Optional, Tuple, Any
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

XAI_BACKGROUND_MAX_ROWS = 128
XAI_EXPLAIN_MAX_ROWS = 64


def _import_external_xgboost():
    src_dir = str(Path(__file__).resolve().parent)
    original_sys_path = list(sys.path)
    existing_module = sys.modules.get("xgboost")
    removed_local_module = None
    try:
        if existing_module is not None:
            module_file = Path(str(getattr(existing_module, "__file__", "") or "")).resolve()
            if module_file == (Path(src_dir) / "xgboost.py").resolve():
                removed_local_module = sys.modules.pop("xgboost")
        sys.path = [
            entry
            for entry in original_sys_path
            if str(Path(entry or ".").resolve()) != src_dir
        ]
        xgb = importlib.import_module("xgboost")  # type: ignore
    finally:
        sys.path = original_sys_path
        if removed_local_module is not None:
            sys.modules["xgboost"] = removed_local_module

    module_path = Path(str(getattr(xgb, "__file__", "") or "")).resolve()
    if module_path == (Path(src_dir) / "xgboost.py").resolve():
        raise ImportError(
            "Se importo el modulo local `src/xgboost.py` en lugar del paquete externo `xgboost`. "
            "Revise el entorno o renombre el modulo local para evitar sombreado."
        )
    if not hasattr(xgb, "XGBClassifier"):
        raise ImportError(
            "El paquete `xgboost` importado no expone `XGBClassifier`. "
            f"Modulo cargado: {module_path}"
        )
    return xgb


def build_model(model_name: str, params: Dict[str, object], random_state: int):
    if model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=random_state,
            class_weight="balanced",
            n_jobs=params.get("n_jobs"),
        )

    if model_name == "XGBoost":
        try:
            xgb = _import_external_xgboost()
        except ImportError as exc:
            raise ImportError(
                "No se pudo cargar el paquete externo `xgboost`. "
                "Instale `xgboost` o corrija el sombreado del modulo local."
            ) from exc

        return xgb.XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            gamma=float(params.get("gamma", 0.0)),
            n_jobs=int(params.get("n_jobs", 1)),
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    if model_name == "SVM":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(params["C"]),
                        kernel=str(params["kernel"]),
                        gamma=params.get("gamma", "scale"),
                        degree=int(params.get("degree", 3)),
                        coef0=float(params.get("coef0", 0.0)),
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Modelo no soportado: {model_name}")


def temporal_train_test_split(
    df: pd.DataFrame,
    *,
    time_col: str = "interval_start",
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if time_col not in df.columns:
        raise ValueError(
            f"No se encontro la columna '{time_col}' para split temporal."
        )
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    work_df = df.copy()
    work_df["_split_time"] = pd.to_datetime(
        work_df[time_col], errors="coerce"
    )
    work_df = work_df.dropna(subset=["_split_time"])
    if work_df.empty:
        raise ValueError("No hay timestamps validos para split temporal.")

    unique_times = np.sort(work_df["_split_time"].unique())
    if len(unique_times) < 2:
        raise ValueError("No hay suficientes timestamps para split temporal.")

    test_count = max(1, int(round(len(unique_times) * float(test_size))))
    if test_count >= len(unique_times):
        test_count = len(unique_times) - 1
    split_idx = len(unique_times) - test_count
    train_times = unique_times[:split_idx]
    test_times = unique_times[split_idx:]

    train_df = work_df[work_df["_split_time"].isin(train_times)].drop(
        columns=["_split_time"]
    )
    test_df = work_df[work_df["_split_time"].isin(test_times)].drop(
        columns=["_split_time"]
    )
    if train_df.empty or test_df.empty:
        raise ValueError("No hay suficientes datos para split temporal.")
    return train_df, test_df


def get_model_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def far_and_sensitivity(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(far), float(sens)


def select_threshold_for_far(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    far_target: float = 0.20,
    *,
    mode: str = "max_sens_under_far",
) -> Dict[str, object]:
    y_val = np.asarray(y_val).astype(int)
    scores_val = np.asarray(scores_val).astype(float)

    if np.unique(y_val).size < 2:
        return {
            "threshold": 0.5,
            "far_val": np.nan,
            "sens_val": np.nan,
            "note": "Validacion con una sola clase.",
        }

    fpr, tpr, thr = roc_curve(y_val, scores_val)
    far_target = float(np.clip(far_target, 0.0, 1.0))

    if mode == "closest_far":
        idx = int(np.argmin(np.abs(fpr - far_target)))
        threshold = float(thr[idx])
    else:
        mask = fpr <= (far_target + 1e-12)
        if np.any(mask):
            idx_local = int(np.argmax(tpr[mask]))
            threshold = float(thr[mask][idx_local])
        else:
            idx = int(np.argmin(np.abs(fpr - far_target)))
            threshold = float(thr[idx])

    yhat_val = (scores_val >= threshold).astype(int)
    far_val, sens_val = far_and_sensitivity(y_val, yhat_val)

    return {
        "threshold": threshold,
        "far_val": far_val,
        "sens_val": sens_val,
        "note": "",
    }


def split_train_val_for_threshold(
    train_df: pd.DataFrame,
    *,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    synthetic_mask = (
        train_df["synthetic"].astype(bool)
        if "synthetic" in train_df.columns
        else pd.Series(False, index=train_df.index)
    )
    real_df = train_df.loc[~synthetic_mask].copy()
    synthetic_df = train_df.loc[synthetic_mask].copy()
    if real_df.empty:
        raise ValueError("No hay datos reales para validacion.")

    try:
        train_real, val_df = temporal_train_test_split(
            real_df, time_col="interval_start", test_size=val_size
        )
        # Sinteticos solo en train.
        train_df_final = pd.concat(
            [train_real, synthetic_df], ignore_index=True
        )
        return train_df_final, val_df
    except ValueError:
        from sklearn.model_selection import train_test_split

        stratify = real_df["target"] if real_df["target"].nunique() > 1 else None
        train_real, val_df = train_test_split(
            real_df,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_df_final = pd.concat(
            [train_real, synthetic_df], ignore_index=True
        )
        return train_df_final, val_df


def _sample_frame_rows(df: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if df.empty or len(df) <= max_rows:
        return df.reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def _build_xai_background_rows(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    synthetic_mask = (
        train_df["synthetic"].astype(bool)
        if "synthetic" in train_df.columns
        else pd.Series(False, index=train_df.index)
    )
    real_train_df = train_df.loc[~synthetic_mask].copy()
    if real_train_df.empty:
        real_train_df = train_df.copy()
    feature_df = real_train_df[feature_cols].fillna(0)
    return _sample_frame_rows(feature_df, max_rows=XAI_BACKGROUND_MAX_ROWS)


def _build_xai_explain_rows(
    test_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    scores_test: np.ndarray,
    preds: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    work = test_df.copy().reset_index(drop=False).rename(
        columns={"index": "source_index"}
    )
    work[feature_cols] = work[feature_cols].fillna(0)
    work["target"] = work["target"].astype(int)
    work["score"] = np.asarray(scores_test, dtype=float)
    work["pred"] = np.asarray(preds, dtype=int)
    work["threshold"] = float(threshold)
    work["case_hint"] = ""

    selected_idx: List[int] = []

    def _append_first(mask: pd.Series, label: str) -> None:
        if not mask.any():
            return
        idx = int(
            work.loc[mask]
            .sort_values("score", ascending=False)
            .index[0]
        )
        if idx in selected_idx:
            return
        selected_idx.append(idx)
        work.loc[idx, "case_hint"] = label

    _append_first(pd.Series(True, index=work.index), "highest_score")
    _append_first((work["target"] == 1) & (work["pred"] == 1), "true_positive")
    _append_first((work["target"] == 0) & (work["pred"] == 1), "false_positive")
    _append_first((work["target"] == 1) & (work["pred"] == 0), "false_negative")

    score_ranked = list(
        work.sort_values("score", ascending=False).index.astype(int)
    )
    for idx in score_ranked:
        if idx in selected_idx:
            continue
        selected_idx.append(int(idx))
        if len(selected_idx) >= XAI_EXPLAIN_MAX_ROWS:
            break

    explain_df = work.loc[selected_idx].copy()
    return explain_df.reset_index(drop=True)


def _build_xai_payload(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scores_test: np.ndarray,
    preds: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    return {
        "background_rows": _build_xai_background_rows(train_df, feature_cols),
        "explain_rows": _build_xai_explain_rows(
            test_df,
            feature_cols=feature_cols,
            scores_test=scores_test,
            preds=preds,
            threshold=threshold,
        ),
        "split_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "xai_limits": {
            "background_max_rows": int(XAI_BACKGROUND_MAX_ROWS),
            "explain_max_rows": int(XAI_EXPLAIN_MAX_ROWS),
        },
    }


def train_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    test_size: float,
    val_size: float,
    far_target: float,
    random_state: int,
) -> Dict[str, object]:
    y = df["target"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Solo existe una clase en el target.")
    train_val_df, test_df = temporal_train_test_split(
        df, time_col="interval_start", test_size=test_size
    )
    train_df, val_df = temporal_train_test_split(
        train_val_df, time_col="interval_start", test_size=val_size
    )
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].astype(int)
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError(
            "El split temporal dejo una sola clase en train. "
            "Ajuste el rango o el test_size."
        )
    if y_test.nunique() < 2:
        raise ValueError(
            "El split temporal dejo una sola clase en test. "
            "Ajuste el rango o el test_size."
        )

    model = build_model(model_name, model_params, random_state)
    model.fit(X_train, y_train)

    scores_val = get_model_scores(model, X_val)
    thr_info = select_threshold_for_far(
        y_val.to_numpy(), scores_val, far_target=float(far_target)
    )
    threshold = float(thr_info["threshold"])

    scores_test = get_model_scores(model, X_test)
    preds = (scores_test >= threshold).astype(int)
    far_test, sens_test = far_and_sensitivity(
        y_test.to_numpy(), preds
    )

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "sensitivity": sens_test,
        "far": far_test,
        "threshold": threshold,
        "far_val": float(thr_info.get("far_val", np.nan)),
        "sens_val": float(thr_info.get("sens_val", np.nan)),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, scores_test)
    else:
        metrics["roc_auc"] = float("nan")
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    split_info = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "model": model,
        "split_info": split_info,
        "xai_payload": _build_xai_payload(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            scores_test=scores_test,
            preds=preds,
            threshold=threshold,
        ),
    }


def train_model_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    val_size: float,
    far_target: float,
    random_state: int,
) -> Dict[str, object]:
    train_df, val_df = split_train_val_for_threshold(
        train_df, val_size=val_size, random_state=random_state
    )
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].astype(int)
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target"].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Solo existe una clase en el train.")
    if y_test.nunique() < 2:
        raise ValueError("Solo existe una clase en el test.")

    model = build_model(model_name, model_params, random_state)
    model.fit(X_train, y_train)

    scores_val = get_model_scores(model, X_val)
    thr_info = select_threshold_for_far(
        y_val.to_numpy(), scores_val, far_target=float(far_target)
    )
    threshold = float(thr_info["threshold"])

    scores_test = get_model_scores(model, X_test)
    preds = (scores_test >= threshold).astype(int)
    far_test, sens_test = far_and_sensitivity(
        y_test.to_numpy(), preds
    )

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "sensitivity": sens_test,
        "far": far_test,
        "threshold": threshold,
        "far_val": float(thr_info.get("far_val", np.nan)),
        "sens_val": float(thr_info.get("sens_val", np.nan)),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, scores_test)
    else:
        metrics["roc_auc"] = float("nan")
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    split_info = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "model": model,
        "split_info": split_info,
        "xai_payload": _build_xai_payload(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            scores_test=scores_test,
            preds=preds,
            threshold=threshold,
        ),
    }
