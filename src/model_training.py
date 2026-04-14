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
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

XAI_BACKGROUND_MAX_ROWS = 128
XAI_EXPLAIN_MAX_ROWS = 64
THRESHOLD_PROTOCOL_LABELS = {
    "conservative": "Conservador",
    "robust": "Robusto",
}
THRESHOLD_OBJECTIVE_LABELS = {
    "far": "FAR",
    "f1": "F1",
    "balanced_f1": "Balanced F1",
    "mcc": "MCC",
    "recall_at_alerts_per_day": "Recall@N alertas/dia",
    "operational_cost": "Costo operacional",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
}
CALIBRATION_METHOD_LABELS = {
    "none": "Sin calibracion",
    "sigmoid": "Sigmoid",
    "isotonic": "Isotonic",
}
BALANCE_STRATEGY_LABELS = {
    "none": "Sin balance interno",
    "class_weight": "Class weight / scale_pos_weight",
    "smote": "SMOTE interno",
}


def normalize_threshold_protocol(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "conservador": "conservative",
        "conservative": "conservative",
        "robusto": "robust",
        "robust": "robust",
    }
    return aliases.get(text, "conservative")


def normalize_threshold_objective(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "far": "far",
        "fpr": "far",
        "f1": "f1",
        "balanced f1": "balanced_f1",
        "balanced_f1": "balanced_f1",
        "macro_f1": "balanced_f1",
        "f1_global": "balanced_f1",
        "mcc": "mcc",
        "recall@n": "recall_at_alerts_per_day",
        "recall_at_n": "recall_at_alerts_per_day",
        "recall_at_alerts_per_day": "recall_at_alerts_per_day",
        "recall_alerts": "recall_at_alerts_per_day",
        "operational_cost": "operational_cost",
        "cost": "operational_cost",
        "costo operacional": "operational_cost",
        "pr_auc": "pr_auc",
        "pr-auc": "pr_auc",
        "average_precision": "pr_auc",
        "roc_auc": "roc_auc",
        "roc-auc": "roc_auc",
        "auc": "roc_auc",
    }
    return aliases.get(text, "far")


def normalize_calibration_method(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "sin calibracion": "none",
        "no": "none",
        "sigmoid": "sigmoid",
        "platt": "sigmoid",
        "isotonic": "isotonic",
        "isotonic_regression": "isotonic",
    }
    return aliases.get(text, "none")


def normalize_balance_strategy(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "sin balance interno": "none",
        "class_weight": "class_weight",
        "class weight": "class_weight",
        "scale_pos_weight": "class_weight",
        "class_weight/scale_pos_weight": "class_weight",
        "smote": "smote",
        "smote interno": "smote",
    }
    return aliases.get(text, "none")


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


def _normalize_class_weight(value: object) -> Optional[object]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "false"}:
        return None
    if text == "balanced":
        return "balanced"
    return value


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
            class_weight=_normalize_class_weight(
                params.get("class_weight", "balanced")
            ),
            n_jobs=params.get("n_jobs"),
        )

    if model_name == "Balanced Random Forest":
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier
        except ImportError as exc:
            raise ImportError(
                "Balanced Random Forest requiere `imbalanced-learn` instalado."
            ) from exc

        return BalancedRandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params.get("max_depth"),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=params.get("n_jobs"),
            replacement=bool(params.get("replacement", False)),
        )

    if model_name == "XGBoost":
        try:
            xgb = _import_external_xgboost()
        except ImportError as exc:
            raise ImportError(
                "No se pudo cargar el paquete externo `xgboost`. "
                "Instale `xgboost` o corrija el sombreado del modulo local."
            ) from exc

        xgb_params = {
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "min_child_weight": float(params.get("min_child_weight", 1.0)),
            "reg_alpha": float(params.get("reg_alpha", 0.0)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "gamma": float(params.get("gamma", 0.0)),
            "n_jobs": int(params.get("n_jobs", 1)),
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        if params.get("scale_pos_weight") is not None:
            xgb_params["scale_pos_weight"] = float(params["scale_pos_weight"])
        if params.get("max_delta_step") is not None:
            xgb_params["max_delta_step"] = float(params["max_delta_step"])

        return xgb.XGBClassifier(**xgb_params)

    if model_name == "SVM":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        probability = bool(params.get("probability", True))
        cache_size = float(params.get("cache_size", 200.0))

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
                        probability=probability,
                        cache_size=cache_size,
                        class_weight=_normalize_class_weight(
                            params.get("class_weight")
                        ),
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
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
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


def _maybe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_arr = np.asarray(y_true).astype(int)
    if np.unique(y_arr).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_arr, np.asarray(scores, dtype=float)))
    except Exception:
        return float("nan")


def _maybe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_arr = np.asarray(y_true).astype(int)
    if np.unique(y_arr).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_arr, np.asarray(scores, dtype=float)))
    except Exception:
        return float("nan")


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.5], dtype=float)
    candidates = np.unique(arr)
    if candidates.size > 256:
        candidates = np.unique(
            np.round(np.quantile(candidates, np.linspace(0.0, 1.0, 257)), 10)
        )
    span = float(np.nanmax(candidates) - np.nanmin(candidates))
    eps = max(1e-12, span * 1e-9)
    candidates = np.append(candidates, [float(np.nanmax(candidates) + eps)])
    candidates = np.append(candidates, [float(np.nanmin(candidates) - eps)])
    if not np.any(np.isclose(candidates, 0.5)):
        candidates = np.append(candidates, 0.5)
    return np.sort(np.unique(candidates.astype(float)))


def estimate_frame_days(df: Optional[pd.DataFrame], row_count: int) -> float:
    if isinstance(df, pd.DataFrame) and "interval_start" in df.columns:
        times = pd.to_datetime(df["interval_start"], errors="coerce").dropna()
        if not times.empty:
            span_days = (
                times.max() - times.min()
            ).total_seconds() / 86400.0
            if span_days > 0:
                return max(span_days, 1.0 / 288.0)
    return max(float(row_count) / 288.0, 1.0 / 288.0)


def _event_groups_from_frame(
    y_true: np.ndarray,
    eval_df: Optional[pd.DataFrame],
) -> List[np.ndarray]:
    y_arr = np.asarray(y_true).astype(int)
    pos_idx = np.flatnonzero(y_arr == 1)
    if pos_idx.size == 0:
        return []
    if not isinstance(eval_df, pd.DataFrame) or "interval_start" not in eval_df.columns:
        groups: List[List[int]] = [[int(pos_idx[0])]]
        for idx in pos_idx[1:]:
            idx_int = int(idx)
            if idx_int == groups[-1][-1] + 1:
                groups[-1].append(idx_int)
            else:
                groups.append([idx_int])
        return [np.asarray(group, dtype=int) for group in groups]

    work = eval_df.reset_index(drop=True).copy()
    work["_event_pos"] = np.arange(len(work))
    work["_event_time"] = pd.to_datetime(work["interval_start"], errors="coerce")
    work["_event_target"] = y_arr
    segment_cols = [
        col
        for col in ["eje", "calzada", "portico", "ultimo_portico", "segment_id"]
        if col in work.columns
    ]
    sort_cols = ["_event_time"] + segment_cols
    work = work.sort_values(sort_cols).reset_index(drop=True)

    groups_out: List[List[int]] = []
    current: List[int] = []
    current_key: Tuple[object, ...] = tuple()
    last_time: Optional[pd.Timestamp] = None
    expected_step = pd.Timedelta(minutes=5)
    for _, row in work.iterrows():
        if int(row["_event_target"]) != 1:
            continue
        time_value = row["_event_time"]
        key = tuple(row[col] for col in segment_cols)
        pos = int(row["_event_pos"])
        contiguous = False
        if current and key == current_key and pd.notna(time_value) and last_time is not None:
            delta = pd.Timestamp(time_value) - pd.Timestamp(last_time)
            contiguous = pd.Timedelta(0) <= delta <= expected_step
        if not current or not contiguous:
            if current:
                groups_out.append(current)
            current = [pos]
            current_key = key
        else:
            current.append(pos)
        last_time = pd.Timestamp(time_value) if pd.notna(time_value) else None
    if current:
        groups_out.append(current)
    return [np.asarray(group, dtype=int) for group in groups_out]


def event_recall_approx(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_df: Optional[pd.DataFrame] = None,
) -> float:
    groups = _event_groups_from_frame(y_true, eval_df)
    if not groups:
        return float("nan")
    preds = np.asarray(y_pred).astype(int)
    detected = sum(1 for group in groups if np.any(preds[group] == 1))
    return float(detected / len(groups))


def compute_extended_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    eval_df: Optional[pd.DataFrame] = None,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> Dict[str, object]:
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    preds = (scores_arr >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()
    far_val, sens_val = far_and_sensitivity(y_arr, preds)
    f1_by_class = f1_score(
        y_arr,
        preds,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    days = estimate_frame_days(eval_df, len(y_arr))
    alert_count = int(tp + fp)
    false_alarm_count = int(fp)
    operational_cost = float(fn_cost) * float(fn) + float(fp_cost) * float(fp)
    recall_at_info = recall_at_alerts_per_day(
        y_arr,
        scores_arr,
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
    )
    return {
        "accuracy": float(accuracy_score(y_arr, preds)),
        "precision": float(precision_score(y_arr, preds, zero_division=0)),
        "recall": float(recall_score(y_arr, preds, zero_division=0)),
        "sensitivity": float(sens_val),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "far": float(far_val),
        "roc_auc": _maybe_roc_auc(y_arr, scores_arr),
        "pr_auc": _maybe_pr_auc(y_arr, scores_arr),
        "f1": float(f1_score(y_arr, preds, zero_division=0)),
        "balanced_f1": float(f1_score(y_arr, preds, average="macro", zero_division=0)),
        "f1_global": float(f1_score(y_arr, preds, average="macro", zero_division=0)),
        "f1_class_0": float(f1_by_class[0]),
        "f1_class_1": float(f1_by_class[1]),
        "mcc": float(matthews_corrcoef(y_arr, preds)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "alerts": int(alert_count),
        "alerts_per_day": float(alert_count / days),
        "false_alarms_per_day": float(false_alarm_count / days),
        "event_recall_approx": event_recall_approx(y_arr, preds, eval_df),
        "operational_cost": float(operational_cost),
        "cost_per_day": float(operational_cost / days),
        "recall_at_alerts_per_day": float(recall_at_info["recall"]),
        "threshold_at_alerts_per_day": float(recall_at_info["threshold"]),
        "alerts_per_day_budget": float(alerts_per_day),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def recall_at_alerts_per_day(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    eval_df: Optional[pd.DataFrame] = None,
    alerts_per_day: float = 5.0,
) -> Dict[str, float]:
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    days = estimate_frame_days(eval_df, len(y_arr))
    max_alerts = max(1, int(np.floor(float(alerts_per_day) * days)))
    if scores_arr.size == 0:
        return {"recall": float("nan"), "threshold": 0.5, "alerts": 0.0}
    order = np.argsort(scores_arr)[::-1]
    selected = order[: min(max_alerts, len(order))]
    preds = np.zeros_like(y_arr, dtype=int)
    preds[selected] = 1
    recall_val = recall_score(y_arr, preds, zero_division=0)
    threshold = float(scores_arr[selected].min()) if selected.size else float("inf")
    return {
        "recall": float(recall_val),
        "threshold": float(threshold),
        "alerts": float(selected.size),
    }


def _metric_value_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    objective: str,
    eval_df: Optional[pd.DataFrame],
    alerts_per_day: float,
    fn_cost: float,
    fp_cost: float,
) -> float:
    metrics = compute_extended_metrics(
        y_true,
        scores,
        threshold=float(threshold),
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    if objective == "operational_cost":
        return -float(metrics.get("operational_cost", float("inf")))
    if objective == "recall_at_alerts_per_day":
        return float(metrics.get("recall", 0.0))
    if objective == "balanced_f1":
        return float(metrics.get("balanced_f1", 0.0))
    if objective in {"f1", "mcc"}:
        return float(metrics.get(objective, 0.0))
    if objective == "far":
        return float(metrics.get("sensitivity", 0.0))
    if objective == "pr_auc":
        return float(metrics.get("pr_auc", float("nan")))
    if objective == "roc_auc":
        return float(metrics.get("roc_auc", float("nan")))
    return float(metrics.get("f1", 0.0))


def _orientation_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str,
) -> float:
    objective_key = normalize_threshold_objective(objective)
    if objective_key == "pr_auc":
        return _maybe_pr_auc(y_true, scores)
    if objective_key == "roc_auc":
        return _maybe_roc_auc(y_true, scores)
    info = select_threshold_for_metric(
        y_true,
        scores,
        objective=objective_key,
    )
    return float(info.get("objective_score", float("nan")))


def orient_scores_for_objective(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str = "roc_auc",
) -> Tuple[np.ndarray, float]:
    raw_scores = np.asarray(scores, dtype=float)
    y_arr = np.asarray(y_true).astype(int)
    if raw_scores.size == 0 or np.unique(y_arr).size < 2:
        return raw_scores, 1.0
    pos_score = _orientation_score(y_arr, raw_scores, objective=objective)
    neg_score = _orientation_score(y_arr, -raw_scores, objective=objective)
    if pd.isna(pos_score) and pd.isna(neg_score):
        return raw_scores, 1.0
    if pd.isna(pos_score):
        return -raw_scores, -1.0
    if pd.isna(neg_score):
        return raw_scores, 1.0
    if float(neg_score) > float(pos_score) + 1e-12:
        return -raw_scores, -1.0
    return raw_scores, 1.0


def select_threshold_for_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    objective: str = "far",
    eval_df: Optional[pd.DataFrame] = None,
    far_target: float = 0.20,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> Dict[str, object]:
    objective_key = normalize_threshold_objective(objective)
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    if np.unique(y_arr).size < 2:
        return {
            "threshold": 0.5,
            "objective": objective_key,
            "objective_score": float("nan"),
            "note": "Validacion con una sola clase.",
        }
    if objective_key in {"pr_auc", "roc_auc"}:
        score = _maybe_pr_auc(y_arr, scores_arr) if objective_key == "pr_auc" else _maybe_roc_auc(y_arr, scores_arr)
        return {
            "threshold": 0.5,
            "objective": objective_key,
            "objective_score": float(score),
            "note": "Metrica de ranking; threshold operativo 0.5.",
        }
    if objective_key == "far":
        info = select_threshold_for_far(
            y_arr,
            scores_arr,
            far_target=float(far_target),
        )
        threshold = float(info["threshold"])
        score = _metric_value_at_threshold(
            y_arr,
            scores_arr,
            threshold=threshold,
            objective=objective_key,
            eval_df=eval_df,
            alerts_per_day=alerts_per_day,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        info.update(
            {
                "objective": objective_key,
                "objective_score": float(score),
            }
        )
        return info

    best_score = float("-inf")
    best_threshold = 0.5
    best_alerts_per_day = float("inf")
    for threshold in threshold_candidates(scores_arr):
        metrics = compute_extended_metrics(
            y_arr,
            scores_arr,
            threshold=float(threshold),
            eval_df=eval_df,
            alerts_per_day=float(alerts_per_day),
            fn_cost=float(fn_cost),
            fp_cost=float(fp_cost),
        )
        if objective_key == "recall_at_alerts_per_day" and float(
            metrics.get("alerts_per_day", float("inf"))
        ) > float(alerts_per_day) + 1e-12:
            continue
        score = _metric_value_at_threshold(
            y_arr,
            scores_arr,
            threshold=float(threshold),
            objective=objective_key,
            eval_df=eval_df,
            alerts_per_day=alerts_per_day,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        if pd.isna(score):
            continue
        alerts_value = float(metrics.get("alerts_per_day", float("inf")))
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)
            best_alerts_per_day = alerts_value
            continue
        if abs(score - best_score) <= 1e-12:
            if alerts_value < best_alerts_per_day - 1e-12:
                best_threshold = float(threshold)
                best_alerts_per_day = alerts_value
            elif abs(alerts_value - best_alerts_per_day) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)

    if best_score == float("-inf"):
        best_threshold = float(recall_at_alerts_per_day(
            y_arr,
            scores_arr,
            eval_df=eval_df,
            alerts_per_day=alerts_per_day,
        )["threshold"])
        best_score = _metric_value_at_threshold(
            y_arr,
            scores_arr,
            threshold=best_threshold,
            objective=objective_key,
            eval_df=eval_df,
            alerts_per_day=alerts_per_day,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
    metrics = compute_extended_metrics(
        y_arr,
        scores_arr,
        threshold=float(best_threshold),
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    return {
        "threshold": float(best_threshold),
        "objective": objective_key,
        "objective_score": float(best_score),
        "far_val": float(metrics.get("far", np.nan)),
        "sens_val": float(metrics.get("sensitivity", np.nan)),
        "alerts_per_day": float(metrics.get("alerts_per_day", np.nan)),
        "operational_cost": float(metrics.get("operational_cost", np.nan)),
        "note": "",
    }


class ScoreCalibrator:
    def __init__(self, method: str = "none", model: Optional[object] = None):
        self.method = normalize_calibration_method(method)
        self.model = model

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float)
        if self.model is None or self.method == "none":
            return scores_arr
        if self.method == "sigmoid":
            return self.model.predict_proba(scores_arr.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.model.predict(scores_arr)
        return scores_arr


def fit_score_calibrator(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    method: str = "none",
) -> ScoreCalibrator:
    method_key = normalize_calibration_method(method)
    y_arr = np.asarray(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    if method_key == "none" or np.unique(y_arr).size < 2:
        return ScoreCalibrator("none", None)
    if method_key == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
        model.fit(scores_arr.reshape(-1, 1), y_arr)
        return ScoreCalibrator("sigmoid", model)
    if method_key == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores_arr, y_arr)
        return ScoreCalibrator("isotonic", model)
    return ScoreCalibrator("none", None)


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


def _scale_pos_weight_from_y(y_train: pd.Series) -> float:
    y_arr = pd.Series(y_train).astype(int)
    positives = int((y_arr == 1).sum())
    negatives = int((y_arr == 0).sum())
    if positives <= 0:
        return 1.0
    return max(1.0, float(negatives / positives))


def _resolve_training_model_params(
    model_name: str,
    model_params: Dict[str, object],
    y_train: pd.Series,
    *,
    balance_strategy: str,
) -> Dict[str, object]:
    params = dict(model_params or {})
    balance_key = normalize_balance_strategy(balance_strategy)
    if model_name == "XGBoost":
        raw_scale = params.get("scale_pos_weight", None)
        if raw_scale in {"auto", "Auto", "AUTO"} or (
            raw_scale is None and balance_key == "class_weight"
        ):
            params["scale_pos_weight"] = _scale_pos_weight_from_y(y_train)
        elif raw_scale is not None:
            params["scale_pos_weight"] = float(raw_scale)
    elif model_name in {"Random Forest", "SVM"}:
        if balance_key == "class_weight" and params.get("class_weight") is None:
            params["class_weight"] = "balanced"
    return params


def _apply_internal_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    smote_params: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("SMOTE requiere `imbalanced-learn` instalado.") from exc

    y_series = pd.Series(y_train).astype(int)
    counts = y_series.value_counts()
    if counts.empty or int(counts.min()) < 2:
        raise ValueError("SMOTE requiere al menos dos muestras en la clase minoritaria.")
    cfg = dict(smote_params or {})
    max_k = max(1, int(counts.min()) - 1)
    k_neighbors = min(max_k, int(cfg.get("k_neighbors", max(1, min(5, max_k)))))
    sampling_strategy = float(cfg.get("sampling_strategy", 1.0))
    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_series)
    return (
        pd.DataFrame(X_res, columns=X_train.columns),
        pd.Series(y_res, name=y_series.name),
    )


def _fit_protocol_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    random_state: int,
    balance_strategy: str,
    smote_params: Optional[Dict[str, object]] = None,
) -> object:
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target"].astype(int)
    params = _resolve_training_model_params(
        model_name,
        model_params,
        y_train,
        balance_strategy=balance_strategy,
    )
    X_fit = X_train
    y_fit = y_train
    if normalize_balance_strategy(balance_strategy) == "smote":
        X_fit, y_fit = _apply_internal_smote(
            X_train,
            y_train,
            random_state=random_state,
            smote_params=smote_params,
        )
    model = build_model(model_name, params, random_state)
    model.fit(X_fit, y_fit)
    return model


def _temporal_oof_splits(
    df: pd.DataFrame,
    *,
    n_splits: int,
    time_col: str = "interval_start",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if time_col not in df.columns:
        return []
    work = df.reset_index(drop=True).copy()
    work["_split_time"] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=["_split_time"])
    if work.empty:
        return []
    unique_times = np.sort(work["_split_time"].unique())
    if len(unique_times) < max(3, int(n_splits) + 1):
        return []
    blocks = [block for block in np.array_split(unique_times, int(n_splits) + 1) if len(block)]
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(1, len(blocks)):
        train_times = np.concatenate(blocks[:fold_idx])
        val_times = blocks[fold_idx]
        train_idx = work.index[work["_split_time"].isin(train_times)].to_numpy()
        val_idx = work.index[work["_split_time"].isin(val_times)].to_numpy()
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        y_train = df.iloc[train_idx]["target"].astype(int)
        y_val = df.iloc[val_idx]["target"].astype(int)
        if y_train.nunique() < 2 or y_val.nunique() < 2:
            continue
        splits.append((train_idx.astype(int), val_idx.astype(int)))
    return splits


def train_model_with_protocol(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_params: Dict[str, object],
    *,
    threshold_protocol: str = "conservative",
    threshold_objective: str = "far",
    calibration_method: str = "none",
    far_target: float = 0.20,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    robust_folds: int = 3,
    balance_strategy: str = "none",
    smote_params: Optional[Dict[str, object]] = None,
    random_state: int,
) -> Dict[str, object]:
    protocol_key = normalize_threshold_protocol(threshold_protocol)
    objective_key = normalize_threshold_objective(threshold_objective)
    calibration_key = normalize_calibration_method(calibration_method)
    balance_key = normalize_balance_strategy(balance_strategy)

    if train_df["target"].astype(int).nunique() < 2:
        raise ValueError("Solo existe una clase en train.")
    if test_df["target"].astype(int).nunique() < 2:
        raise ValueError("Solo existe una clase en test.")

    note = ""
    if protocol_key == "robust":
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        splits = _temporal_oof_splits(
            train_val_df,
            n_splits=max(2, int(robust_folds)),
        )
        if not splits:
            note = "Robusto sin folds validos; se uso protocolo conservador."
            protocol_key = "conservative"
        else:
            oof_scores = np.full(len(train_val_df), np.nan, dtype=float)
            oof_mask = np.zeros(len(train_val_df), dtype=bool)
            for fold_number, (fold_train_idx, fold_val_idx) in enumerate(splits):
                fold_model = _fit_protocol_model(
                    train_val_df.iloc[fold_train_idx],
                    feature_cols,
                    model_name,
                    model_params,
                    random_state=int(random_state) + int(fold_number),
                    balance_strategy=balance_key,
                    smote_params=smote_params,
                )
                fold_X_val = train_val_df.iloc[fold_val_idx][feature_cols].fillna(0)
                oof_scores[fold_val_idx] = get_model_scores(fold_model, fold_X_val)
                oof_mask[fold_val_idx] = True

            oof_df = train_val_df.loc[oof_mask].reset_index(drop=True)
            oof_y = oof_df["target"].astype(int).to_numpy()
            raw_oof_scores = oof_scores[oof_mask]
            if oof_y.size == 0 or np.unique(oof_y).size < 2:
                note = "Robusto sin OOF de dos clases; se uso protocolo conservador."
                protocol_key = "conservative"
            else:
                score_orientation = 1.0
                if model_name == "SVM":
                    raw_oof_scores, score_orientation = orient_scores_for_objective(
                        oof_y,
                        raw_oof_scores,
                        objective=objective_key,
                    )
                calibrator = fit_score_calibrator(
                    oof_y,
                    raw_oof_scores,
                    method=calibration_key,
                )
                oof_scores_cal = calibrator.transform(raw_oof_scores)
                thr_info = select_threshold_for_metric(
                    oof_y,
                    oof_scores_cal,
                    objective=objective_key,
                    eval_df=oof_df,
                    far_target=float(far_target),
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                )
                threshold = float(thr_info["threshold"])
                final_model = _fit_protocol_model(
                    train_val_df,
                    feature_cols,
                    model_name,
                    model_params,
                    random_state=int(random_state),
                    balance_strategy=balance_key,
                    smote_params=smote_params,
                )
                X_test = test_df[feature_cols].fillna(0)
                y_test = test_df["target"].astype(int).to_numpy()
                raw_test_scores = (
                    get_model_scores(final_model, X_test) * float(score_orientation)
                )
                scores_test = calibrator.transform(raw_test_scores)
                preds_test = (scores_test >= threshold).astype(int)
                validation_metrics = compute_extended_metrics(
                    oof_y,
                    oof_scores_cal,
                    threshold=threshold,
                    eval_df=oof_df,
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                )
                test_metrics = compute_extended_metrics(
                    y_test,
                    scores_test,
                    threshold=threshold,
                    eval_df=test_df,
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                )
                metrics = dict(test_metrics)
                metrics.update(
                    {
                        "threshold": threshold,
                        "far_val": float(validation_metrics.get("far", np.nan)),
                        "sens_val": float(validation_metrics.get("sensitivity", np.nan)),
                        "threshold_protocol": protocol_key,
                        "threshold_objective": objective_key,
                        "calibration_method": calibration_key,
                        "balance_strategy": balance_key,
                        "threshold_objective_score": float(
                            thr_info.get("objective_score", np.nan)
                        ),
                    }
                )
                split_info = {
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "test_rows": int(len(test_df)),
                    "oof_rows": int(len(oof_df)),
                    "robust_folds": int(len(splits)),
                }
                return {
                    "metrics": metrics,
                    "validation_metrics": validation_metrics,
                    "threshold_info": thr_info,
                    "confusion_matrix": test_metrics["confusion_matrix"],
                    "model": final_model,
                    "calibrator": calibrator,
                    "split_info": split_info,
                    "threshold_protocol": protocol_key,
                    "threshold_objective": objective_key,
                    "calibration_method": calibration_key,
                    "balance_strategy": balance_key,
                    "note": note,
                    "xai_payload": _build_xai_payload(
                        train_df=train_val_df,
                        val_df=oof_df,
                        test_df=test_df,
                        feature_cols=feature_cols,
                        scores_test=scores_test,
                        preds=preds_test,
                        threshold=threshold,
                    ),
                }

    model = _fit_protocol_model(
        train_df,
        feature_cols,
        model_name,
        model_params,
        random_state=int(random_state),
        balance_strategy=balance_key,
        smote_params=smote_params,
    )
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target"].astype(int).to_numpy()
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"].astype(int).to_numpy()
    raw_val_scores = get_model_scores(model, X_val)
    score_orientation = 1.0
    if model_name == "SVM":
        raw_val_scores, score_orientation = orient_scores_for_objective(
            y_val,
            raw_val_scores,
            objective=objective_key,
        )
    calibrator = fit_score_calibrator(
        y_val,
        raw_val_scores,
        method=calibration_key,
    )
    scores_val = calibrator.transform(raw_val_scores)
    thr_info = select_threshold_for_metric(
        y_val,
        scores_val,
        objective=objective_key,
        eval_df=val_df,
        far_target=float(far_target),
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    threshold = float(thr_info["threshold"])
    scores_test = calibrator.transform(
        get_model_scores(model, X_test) * float(score_orientation)
    )
    preds_test = (scores_test >= threshold).astype(int)
    validation_metrics = compute_extended_metrics(
        y_val,
        scores_val,
        threshold=threshold,
        eval_df=val_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    test_metrics = compute_extended_metrics(
        y_test,
        scores_test,
        threshold=threshold,
        eval_df=test_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    metrics = dict(test_metrics)
    metrics.update(
        {
            "threshold": threshold,
            "far_val": float(validation_metrics.get("far", np.nan)),
            "sens_val": float(validation_metrics.get("sensitivity", np.nan)),
            "threshold_protocol": protocol_key,
            "threshold_objective": objective_key,
            "calibration_method": calibration_key,
            "balance_strategy": balance_key,
            "threshold_objective_score": float(thr_info.get("objective_score", np.nan)),
        }
    )
    split_info = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "robust_folds": 0,
    }
    return {
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "threshold_info": thr_info,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "model": model,
        "calibrator": calibrator,
        "split_info": split_info,
        "threshold_protocol": protocol_key,
        "threshold_objective": objective_key,
        "calibration_method": calibration_key,
        "balance_strategy": balance_key,
        "note": note,
        "xai_payload": _build_xai_payload(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            scores_test=scores_test,
            preds=preds_test,
            threshold=threshold,
        ),
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
    threshold_protocol: str = "conservative",
    threshold_objective: str = "far",
    calibration_method: str = "none",
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    robust_folds: int = 3,
    balance_strategy: str = "none",
    smote_params: Optional[Dict[str, object]] = None,
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

    return train_model_with_protocol(
        train_df,
        val_df,
        test_df,
        feature_cols,
        model_name,
        model_params,
        threshold_protocol=threshold_protocol,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        far_target=float(far_target),
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        robust_folds=int(robust_folds),
        balance_strategy=balance_strategy,
        smote_params=smote_params,
        random_state=int(random_state),
    )


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
    threshold_protocol: str = "conservative",
    threshold_objective: str = "far",
    calibration_method: str = "none",
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    robust_folds: int = 3,
    balance_strategy: str = "none",
    smote_params: Optional[Dict[str, object]] = None,
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

    return train_model_with_protocol(
        train_df,
        val_df,
        test_df,
        feature_cols,
        model_name,
        model_params,
        threshold_protocol=threshold_protocol,
        threshold_objective=threshold_objective,
        calibration_method=calibration_method,
        far_target=float(far_target),
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
        robust_folds=int(robust_folds),
        balance_strategy=balance_strategy,
        smote_params=smote_params,
        random_state=int(random_state),
    )
