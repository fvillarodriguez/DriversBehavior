"""
Shared model training and evaluation logic for the Crash Prediction App.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def build_model(model_name: str, params: Dict[str, object], random_state: int):
    if model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params.get("max_depth"),
            random_state=random_state,
            class_weight="balanced",
        )

    if model_name == "XGBoost":
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "xgboost no esta instalado. Ejecute `pip install xgboost`."
            ) from exc

        return xgb.XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
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
    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "model": model,
        "split_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
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
    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "model": model,
        "split_info": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
    }
