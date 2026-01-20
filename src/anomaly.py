import os
import glob
import re
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
def _get_git_commit():
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit or None
    except Exception:
        return None
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from src.config import RESULTADOS_DIR, SEED, TRAIN_RATIO, VAL_RATIO, DT_MINUTES, F_BETA_THRESHOLD
from src.utils import buscar_columna, load_accidentes


def _latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load_latest_features_df() -> pd.DataFrame:
    """
    Carga el último features_{números}.csv (excluye features_{texto} y features_selected_names_*).
    """
    candidates = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "features_*.csv")), key=os.path.getmtime)
    candidates = [
        p for p in candidates
        if not os.path.basename(p).startswith("features_selected_names_")
        and re.match(r"^features_\d", os.path.basename(p))
    ]
    if not candidates:
        raise FileNotFoundError("No se encontraron archivos 'features_{números}.csv' en Resultados.")
    path = candidates[-1]
    print(f"▶️ Usando features desde: {os.path.basename(path)}")
    return pd.read_csv(path)


def build_target(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Devuelve features_df con columna 'target' (accidente=1) y los nombres de columna de pórtico y timestamp.
    """
    df = features_df.copy()

    # Identificar columnas clave
    portico_col = buscar_columna(df, "portico")
    ts_col = buscar_columna(df, "ts_min")

    # Cargar accidentes e identificar (pórtico, ts) con accidente → y=1
    df_acc = load_accidentes()
    if df_acc is None or df_acc.empty:
        raise RuntimeError("No se pudieron cargar los accidentes para construir el target.")

    df_acc['ts_min'] = (df_acc["accidente_time"].dt.floor(f'{int(DT_MINUTES)}min').astype("int64") // 60_000_000_000)
    accident_events = set(zip(df_acc['ultimo_portico'], df_acc['ts_min']))

    df['target'] = df.apply(lambda row: 1 if (row[portico_col], row[ts_col]) in accident_events else 0, axis=1)
    n_pos = int(df['target'].sum())
    print(f"  - Accidentes cruzados con features: {n_pos}")
    if n_pos == 0:
        print("⚠️ Advertencia: No se encontraron positivos tras el cruce features-accidentes.")

    return df, portico_col, ts_col


def _standardize_fit_transform(X_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_df.values)
    return X_all, scaler


def _train_val_test_split(n: int, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(n)
    train_size = float(TRAIN_RATIO) / 100.0
    val_size = float(VAL_RATIO) / 100.0
    test_size = max(0.0, 1.0 - train_size - val_size)

    if test_size > 0:
        idx_tmp, idx_test, y_tmp, y_test = train_test_split(all_idx, y, test_size=test_size, random_state=SEED, stratify=y)
    else:
        idx_tmp, idx_test, y_tmp, y_test = all_idx, None, y, None

    val_rel = val_size / max(1e-9, (train_size + val_size))
    if val_size > 0:
        idx_train, idx_val, y_train, y_val = train_test_split(idx_tmp, y_tmp, test_size=val_rel, random_state=SEED, stratify=y_tmp)
    else:
        idx_train, idx_val, y_train, y_val = idx_tmp, None, y_tmp, None

    return idx_train, idx_val, idx_test


def _pick_threshold_from_val(y_true_val: np.ndarray, anomaly_scores_val: np.ndarray, *, beta: float = 0.5) -> Tuple[float, Dict[str, float]]:
    """
    Selecciona umbral sobre scores de anomalía maximizando F_beta en validación.
    Devuelve (tau, métricas_en_tau).
    """
    y_true_val = y_true_val.astype(int).ravel()
    s_val = anomaly_scores_val.astype(float).ravel()
    prec, rec, thr = precision_recall_curve(y_true_val, s_val)
    if len(thr) == 0:
        return float(np.quantile(s_val, 0.99)), {"precision": 0.0, "recall": 0.0, "fbeta": 0.0}
    prec_, rec_, thr_ = prec[1:], rec[1:], thr
    b2 = beta**2
    fbeta = (1 + b2) * prec_ * rec_ / np.clip(b2 * prec_ + rec_, 1e-12, None)
    i = int(np.nanargmax(fbeta))
    return float(thr_[i]), {"precision": float(prec_[i]), "recall": float(rec_[i]), "fbeta": float(fbeta[i])}


def _evaluate_split(y_true: np.ndarray, scores: np.ndarray, tau: float, *, split_name: str) -> Dict[str, float]:
    y_true = y_true.astype(int)
    scores = scores.astype(float)
    y_pred = (scores >= tau).astype(int)
    auprc = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float('nan')
    auroc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float('nan')
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {split_name} ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Matriz de Confusión:")
    print(cm)
    return {
        'auprc': float(auprc) if auprc == auprc else float('nan'),
        'auroc': float(auroc) if auroc == auroc else float('nan'),
        'precision_1': rep.get('1', {}).get('precision', None),
        'recall_1': rep.get('1', {}).get('recall', None),
        'f1_1': rep.get('1', {}).get('f1-score', None),
        'accuracy': rep.get('accuracy', None),
    }


def _fit_isolation_forest(X_train_norm: np.ndarray, *, random_state: int) -> IsolationForest:
    # contamination='auto' usa heurística, pero preferimos determinar tau vía validación
    return IsolationForest(n_estimators=300, max_samples='auto', contamination='auto', random_state=random_state, n_jobs=-1)


def _fit_one_class_svm(X_train_norm: np.ndarray) -> OneClassSVM:
    # rbf suele funcionar bien en anomalías; nu controla proporción superior de outliers teóricos
    return OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)


def _fit_lof(X_train_norm: np.ndarray) -> LocalOutlierFactor:
    # LOF con novelty=True para permitir score en datos nuevos
    return LocalOutlierFactor(n_neighbors=min(35, max(5, X_train_norm.shape[0]-1)), novelty=True)


def _scores_from_model(model, X: np.ndarray, *, algo: str) -> np.ndarray:
    if algo == 'IF':
        # score_samples: mayor → más normal. Usamos anomalía = -score
        return -model.score_samples(X)
    elif algo == 'OCSVM':
        # decision_function: mayor → más normal
        s = model.decision_function(X)
        return -s
    elif algo == 'LOF':
        # score_samples: mayor → más normal
        s = model.score_samples(X)
        return -s
    else:
        raise ValueError(algo)


def run_anomaly_pipeline():
    """
    Pipeline de detección de anomalías (no supervisado) sobre features tabulares por pórtico-minuto.
    - Entrena con NORMALIDAD (y=0) en TRAIN únicamente.
    - Usa validación para elegir umbral por F_beta (beta configurable).
    - Modelos: IsolationForest, OneClassSVM, LOF; reporta métricas y guarda predicciones.
    """
    # 1) Cargar features + target (accidente=1)
    df_feat = load_latest_features_df()
    df_with_target, portico_col, ts_col = build_target(df_feat)

    # 2) Seleccionar columnas numéricas como features
    feature_cols = [c for c in df_with_target.columns if c not in {portico_col, ts_col, 'target'}]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_with_target[c])]
    X_df = df_with_target[feature_cols].copy().fillna(df_with_target[feature_cols].median())
    y = df_with_target['target'].astype(int).to_numpy()

    # 3) Escalado + splits
    X_all, scaler = _standardize_fit_transform(X_df)
    n = X_all.shape[0]
    idx_train, idx_val, idx_test = _train_val_test_split(n, y)

    def split_take(idxs):
        return X_all[idxs], y[idxs], idxs

    X_train, y_train, idx_tr = split_take(idx_train)
    X_val, y_val, idx_va = split_take(idx_val) if idx_val is not None else (None, None, None)
    X_test, y_test, idx_te = split_take(idx_test) if idx_test is not None else (None, None, None)

    # 4) Subconjunto de entrenamiento: SOLO normales (y=0)
    X_train_norm = X_train[y_train == 0]
    if X_train_norm.shape[0] == 0:
        raise RuntimeError("No hay muestras normales en TRAIN para entrenar un detector de anomalías.")

    # 5) Entrenar modelos (con barra de progreso)
    models = {}
    algo_order = ['IF', 'OCSVM', 'LOF']
    steps_total = 0
    # fits por modelo disponible
    steps_total += len(algo_order)
    # selección de umbral
    steps_total += 1
    # export/eval por split (train, val?, test?)
    n_splits = 1 + (1 if X_val is not None and y_val is not None else 0) + (1 if X_test is not None and y_test is not None else 0)
    steps_total += n_splits  # contabilizar export+eval por split juntos
    # guardado de resultados
    steps_total += 1

    pbar = tqdm(total=steps_total, desc='Anomalias (tabular)', leave=True)
    # Isolation Forest
    if_model = _fit_isolation_forest(X_train_norm, random_state=SEED)
    pbar.set_description('Entrenando IF')
    if_model.fit(X_train_norm)
    models['IF'] = if_model
    pbar.update(1)
    # One-Class SVM
    try:
        oc_model = _fit_one_class_svm(X_train_norm)
        pbar.set_description('Entrenando OCSVM')
        oc_model.fit(X_train_norm)
        models['OCSVM'] = oc_model
    except Exception as e:
        print(f"⚠️ OCSVM no se pudo entrenar: {e}")
    else:
        pbar.update(1)
    # LOF (novelty)
    try:
        lof_model = _fit_lof(X_train_norm)
        pbar.set_description('Entrenando LOF')
        lof_model.fit(X_train_norm)
        models['LOF'] = lof_model
    except Exception as e:
        print(f"⚠️ LOF no se pudo entrenar: {e}")
    else:
        pbar.update(1)

    # 6) Seleccionar mejor modelo por AUPRC en validación y umbral por F_beta
    results = []
    taus: Dict[str, float] = {}
    for algo, m in models.items():
        # scores en val (si no hay val, usar parte de train para simular)
        if X_val is None or y_val is None:
            Xv, yv = X_train, y_train
            split_name = 'train-as-val'
        else:
            Xv, yv = X_val, y_val
            split_name = 'val'
        s_val = _scores_from_model(m, Xv, algo=algo)
        auprc = average_precision_score(yv, s_val) if len(np.unique(yv)) > 1 else 0.0
        tau, info = _pick_threshold_from_val(yv, s_val, beta=F_BETA_THRESHOLD)
        taus[algo] = tau
        print(f"[{algo}] {split_name}: AUPRC={auprc:.4f} | tau={tau:.6f} (P={info['precision']:.3f}, R={info['recall']:.3f})")
        results.append((algo, auprc))
    # selección de umbral completa
    pbar.set_description('Seleccionando umbral')
    pbar.update(1)

    if not results:
        print("❌ No hay modelos entrenados para evaluar.")
        return

    best_algo = sorted(results, key=lambda t: t[1], reverse=True)[0][0]
    best_model = models[best_algo]
    best_tau = taus[best_algo]
    # Guardar el mejor detector y el escalador como artefactos (único + alias)
    try:
        ts_model = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path_unique = os.path.join(RESULTADOS_DIR, f"anomaly_best_model_{best_algo}_{ts_model}.joblib")
        alias_path = os.path.join(RESULTADOS_DIR, "anomaly_best_model.joblib")
        payload = {
            'algo': best_algo,
            'model': best_model,
            'scaler_mean': scaler.mean_.copy(),
            'scaler_scale': scaler.scale_.copy(),
            'feature_cols': feature_cols,
            'threshold': float(best_tau),
            'git_commit': _get_git_commit(),
        }
        joblib.dump(payload, model_path_unique)
        try:
            joblib.dump(payload, alias_path)
        except Exception:
            pass
        print(f"💾 Detector de anomalías guardado → {model_path_unique} (alias: {alias_path})")
    except Exception as e:
        print(f"⚠️ No se pudo guardar el detector de anomalías: {e}")
    print(f"\n✅ Mejor detector: {best_algo} (tau={best_tau:.6f})")

    # 7) Evaluar y guardar resultados/predicciones
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    unified_rows = []

    def export_preds(idxs, ytrue, scores, name: str):
        if idxs is None:
            return
        keys_df = df_with_target.iloc[idxs][[portico_col, ts_col]].reset_index(drop=True)
        df_out = keys_df.copy()
        df_out['y_true'] = ytrue.astype(int)
        df_out['anomaly_score'] = scores.astype(float)
        df_out['y_pred'] = (scores >= best_tau).astype(int)
        df_out['split'] = name
        out_path = os.path.join(RESULTADOS_DIR, f"preds_anomaly_{name}_{timestamp}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"📄 Predicciones ({name}) → {out_path}")

    # train
    s_train = _scores_from_model(best_model, X_train, algo=best_algo)
    export_preds(idx_tr, y_train, s_train, 'train')
    metrics = _evaluate_split(y_train, s_train, best_tau, split_name='train')
    unified_rows.append({'model': best_algo, 'split': 'train', 'threshold': best_tau, **metrics})
    pbar.set_description('Evaluando train')
    pbar.update(1)

    # val
    if X_val is not None and y_val is not None:
        s_val = _scores_from_model(best_model, X_val, algo=best_algo)
        export_preds(idx_va, y_val, s_val, 'val')
        metrics = _evaluate_split(y_val, s_val, best_tau, split_name='val')
        unified_rows.append({'model': best_algo, 'split': 'val', 'threshold': best_tau, **metrics})
        pbar.set_description('Evaluando val')
        pbar.update(1)

    # test
    if X_test is not None and y_test is not None:
        s_test = _scores_from_model(best_model, X_test, algo=best_algo)
        export_preds(idx_te, y_test, s_test, 'test')
        metrics = _evaluate_split(y_test, s_test, best_tau, split_name='test')
        unified_rows.append({'model': best_algo, 'split': 'test', 'threshold': best_tau, **metrics})
        pbar.set_description('Evaluando test')
        pbar.update(1)

    # Guardar resultados unificados
    res_path = os.path.join(RESULTADOS_DIR, f"results_anomaly_{timestamp}.csv")
    pd.DataFrame(unified_rows).to_csv(res_path, index=False)
    print(f"📄 Resultados (unificados) guardados → {res_path}")
    pbar.set_description('Guardando resultados')
    pbar.update(1)
    pbar.close()

    # Resumen en terminal (post-export)
    print("\n--- Resumen de evaluación (Anomalias - Tabular) ---")
    try:
        for r in unified_rows:
            print(
                f"[{r.get('split','')}] "
                f"AUPRC={r.get('auprc', float('nan')):.4f} "
                f"AUROC={r.get('auroc', float('nan')):.4f} "
                f"P1={r.get('precision_1', float('nan')):.4f} "
                f"R1={r.get('recall_1', float('nan')):.4f} "
                f"F1_1={r.get('f1_1', float('nan')):.4f} "
                f"ACC={r.get('accuracy', float('nan')):.4f} "
                f"tau={r.get('threshold', float('nan')):.4f}"
            )
    except Exception:
        pass

    # Guardar metadata del pipeline
    meta = {
        'model': best_algo,
        'threshold': float(best_tau),
        'feature_cols': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }
    meta_path = os.path.join(RESULTADOS_DIR, f"anomaly_meta_{timestamp}.json")
    try:
        import json
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        print(f"💾 Metadata guardada → {meta_path}")
    except Exception:
        pass

    print("\n✅ Pipeline de anomalías finalizado.")
