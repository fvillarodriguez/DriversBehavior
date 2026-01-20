import os
import glob
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from src.config import RESULTADOS_DIR, DT_MINUTES, TRAIN_RATIO, VAL_RATIO, SEED
from src.utils import (
    load_porticos,
    load_accidentes,
    buscar_columna,
    seleccionar_tramo_y_porticos,
    reconstruct_portico_sequence,
)


def _latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def _load_latest_features_df() -> pd.DataFrame:
    """Carga el último features_{números}.csv (excluye features_{texto} y features_selected_names_*)."""
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


def _load_selected_feature_names() -> Optional[List[str]]:
    path = _latest_file(os.path.join(RESULTADOS_DIR, "features_selected_names_*.csv"))
    if not path:
        print("⚠️ No se encontró archivo de nombres de features seleccionadas. Se usarán todas las columnas numéricas.")
        return None
    print(f"▶️ Usando selección de features desde: {os.path.basename(path)}")
    df = pd.read_csv(path)
    col = buscar_columna(df, "feature_name")
    names = df[col].dropna().astype(str).tolist()
    return names


def _select_porticos_with_pn_or_manual(df_porticos: pd.DataFrame) -> Tuple[List[str], Optional[str]]:
    """
    Reproduce la lógica de build_graph para usar el último reporte de Puntos Negros
    o selección manual de tramo. Devuelve (lista_porticos, time_choice) donde time_choice ∈ {'1','2','3'} o None.
    """
    porticos_a_incluir: Optional[List[str]] = None
    preselected_time_choice: Optional[str] = None

    pn_reports = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "analisis_puntos_negros_*.csv")))
    use_pn = False
    if pn_reports:
        print("\n▶️ Se encontró un reporte de Puntos Negros.")
        print("  [1] Usar Punto Negro (tramo y franja del último reporte)")
        print("  [2] Selección manual de tramo y franja horaria")
        sel = input("Elija una opción [1/2]: ").strip() or '1'
        use_pn = (sel == '1')

    if use_pn:
        pn_path = pn_reports[-1]
        try:
            df_pn = pd.read_csv(pn_path, sep=';', decimal=',')
            if df_pn.empty:
                raise ValueError("Reporte vacío")
            top = df_pn.iloc[0]
            tramo_str = str(top['tramo'])
            franja = str(top['franja_horaria']) if 'franja_horaria' in df_pn.columns else None

            if franja and '06-10' in franja:
                preselected_time_choice = '2'
            elif franja and '18-22' in franja:
                preselected_time_choice = '3'
            else:
                preselected_time_choice = '1'

            p1_raw, p2_raw = [s.strip() for s in tramo_str.split('->')]
            porticos_a_incluir = reconstruct_portico_sequence(
                df_porticos=df_porticos,
                p1=p1_raw,
                p2=p2_raw,
                max_anterior=4,
                max_posterior=0,
                prefer_family=True,
                deduplicate_slice=True,
            )

            print("\n✅ Usando Punto Negro del último reporte:")
            print(f"   - Tramo: {tramo_str}")
            print(f"   - Franja horaria: {franja or 'Sin filtro'}")
            print(f"   - Pórticos incluidos: {porticos_a_incluir}")
        except Exception as e:
            print(f"⚠️ No se pudo usar el reporte de Puntos Negros ({e}). Se solicitará selección manual.")
            porticos_a_incluir = None

    if porticos_a_incluir is None:
        porticos_a_incluir = seleccionar_tramo_y_porticos(df_porticos)
        if porticos_a_incluir is None:
            raise RuntimeError("Operación cancelada. No se seleccionaron pórticos.")

    return porticos_a_incluir, preselected_time_choice


def _filter_time_slot(df: pd.DataFrame, ts_col: str, preselected_choice: Optional[str]) -> pd.DataFrame:
    print("\n▶️ Selección de tramo horario...")
    if preselected_choice is None:
        print("  [1] Sin filtro horario (todos los datos)")
        print("  [2] Horario punta mañana (6:00 - 10:00)")
        print("  [3] Horario punta tarde (18:00 - 22:00)")
        time_choice = input("Elija una opción de tramo horario [1/2/3]: ").strip() or '1'
    else:
        time_choice = preselected_choice
        label = {
            '1': 'Sin filtro horario (todos los datos)',
            '2': 'Horario punta mañana (6:00 - 10:00)',
            '3': 'Horario punta tarde (18:00 - 22:00)'
        }.get(time_choice, 'Sin filtro horario (todos los datos)')
        print(f"  ↳ Usando franja preseleccionada por Punto Negro: {label}")

    if time_choice in ['2', '3']:
        timestamps = pd.to_datetime(df[ts_col], unit='m', origin='unix')
        hours = timestamps.dt.hour
        if time_choice == '2':
            mask = (hours >= 6) & (hours < 10)
            print("✅ Filtrando: punta mañana (6:00 - 10:00)")
        else:
            mask = (hours >= 18) & (hours < 22)
            print("✅ Filtrando: punta tarde (18:00 - 22:00)")
        initial = len(df)
        df = df[mask].copy()
        if df.empty:
            raise RuntimeError(f"Sin datos tras filtro horario (de {initial} filas).")
        print(f"ℹ️ Se conservan {len(df)} de {initial} filas tras filtro horario.")
    else:
        print("✅ Usando todos los datos sin filtro horario.")
    return df


def _build_target_from_accidents(features_df: pd.DataFrame, df_acc: pd.DataFrame, portico_col: str) -> pd.DataFrame:
    df = features_df.copy()
    if df_acc is None or df_acc.empty:
        raise RuntimeError("No se pudieron cargar los accidentes para construir el target.")
    df_acc = df_acc.copy()
    df_acc['ts_min'] = (df_acc["accidente_time"].dt.floor(f'{int(DT_MINUTES)}min').astype("int64") // 60_000_000_000)
    accident_events = set(zip(df_acc['ultimo_portico'], df_acc['ts_min']))
    ts_col = buscar_columna(df, "ts_min")
    df['target'] = df.apply(lambda row: 1 if (row[portico_col], row[ts_col]) in accident_events else 0, axis=1)
    n_pos = int(df['target'].sum())
    print(f"  - Accidentes encontrados en features: {n_pos}")
    if n_pos == 0:
        print("⚠️ Advertencia: No se encontraron positivos tras el cruce features-accidentes.")
    return df


def _train_val_test_split_indices(n: int, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
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


def _standardize_fit_transform(X_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_df.values)
    return X_all, {"mean": scaler.mean_.copy(), "scale": scaler.scale_.copy()}


def _smote_with_ratio(X: np.ndarray, y: np.ndarray, *, target_pos_ratio: float, k_neighbors: int, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE simple (similar a _simple_smote del MLP) pero permitiendo fijar una razón objetivo de positivos.
    Trabaja en espacio estandarizado. Solo sintetiza la clase minoritaria.
    """
    rng = np.random.default_rng(random_state)
    y = y.astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y
    pos, neg = 1, 0
    n_pos, n_neg = int((y == pos).sum()), int((y == neg).sum())
    if n_pos == 0 or n_neg == 0:
        return X, y
    current_ratio = n_pos / (n_pos + n_neg)
    target_pos_ratio = float(np.clip(target_pos_ratio, current_ratio, 0.9))
    # Resolver n_new en: (n_pos + n_new) / (n_pos + n_neg + n_new) = target_pos_ratio
    total0 = n_pos + n_neg
    numerator = target_pos_ratio * total0 - n_pos
    denom = 1.0 - target_pos_ratio
    n_new = int(np.ceil(max(0.0, numerator / max(denom, 1e-9))))
    if n_new <= 0:
        return X, y
    X_min = X[y == pos]
    k = max(1, min(k_neighbors, X_min.shape[0] - 1))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X_min)
    dist, idx = nn.kneighbors(X_min, return_distance=True)
    idx = idx[:, 1:]
    base_idx = rng.integers(0, X_min.shape[0], size=n_new)
    nbr_pos = rng.integers(0, idx.shape[1], size=n_new)
    nbr_idx = idx[base_idx, nbr_pos]
    alpha = rng.random(size=n_new)
    X_new = (1 - alpha)[:, None] * X_min[base_idx] + alpha[:, None] * X_min[nbr_idx]
    y_new = np.full(n_new, pos, dtype=y.dtype)
    X_bal = np.vstack([X, X_new])
    y_bal = np.concatenate([y, y_new])
    return X_bal, y_bal


def _export_balanced_train_csv(Xb: np.ndarray, yb: np.ndarray, feature_names: List[str], prefix: str = "xgb") -> str:
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%d%m%Y_%H%M')
    path = os.path.join(RESULTADOS_DIR, f"{prefix}_balanced_train_{ts}.csv")
    df = pd.DataFrame(Xb, columns=feature_names)
    df['target'] = yb.astype(int)
    df.to_csv(path, index=False)
    print(f"💾 Train balanceado (SMOTE) exportado → {path}")
    return path


def _pick_threshold_from_val(y_true_val: np.ndarray, y_prob1_val: np.ndarray, beta: float = 0.5) -> Tuple[float, Dict[str, float]]:
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true_val.astype(int), y_prob1_val.astype(float))
    prec_, rec_, thr_ = prec[1:], rec[1:], thr
    beta2 = beta ** 2
    if len(prec_) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "fbeta": 0.0}
    fbeta = (1 + beta2) * prec_ * rec_ / np.clip(beta2 * prec_ + rec_, 1e-12, None)
    i = int(np.nanargmax(fbeta))
    tau = float(thr_[i])
    return tau, {"precision": float(prec_[i]), "recall": float(rec_[i]), "fbeta": float(fbeta[i])}


def _evaluate(model, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import numpy as np
    prob1 = model.predict_proba(X)[:, 1]
    return prob1, y.astype(int)


def run_xgboost_pipeline():
    """
    Pipeline XGBoost (Baseline):
      - Carga pórticos, accidentes y features
      - Selección de tramo por Puntos Negros o manual
      - Búsqueda del mejor balance SMOTE (ratio, k)
      - SMOTE solo en TRAIN (se exporta)
      - Entrena XGBoost y genera reporte de evaluación (train/val/test)
    """
    # 0) Importar XGBoost
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        print("❌ No se pudo importar xgboost. Instale 'xgboost' (pip install xgboost) e inténtelo nuevamente.")
        print(f"Detalle: {e}")
        return

    # 1) Cargar pórticos y seleccionar tramo
    print("\n▶️ Cargando pórticos…")
    df_port = load_porticos()
    if df_port is None or df_port.empty:
        print("❌ No se pudieron cargar pórticos.")
        return
    try:
        porticos_sel, pre_time_choice = _select_porticos_with_pn_or_manual(df_port)
    except RuntimeError as e:
        print(str(e))
        return

    # 2) Cargar accidentes y filtrar por pórticos seleccionados
    print("\n▶️ Cargando accidentes…")
    df_acc = load_accidentes()
    if df_acc is None or df_acc.empty:
        print("❌ No se pudieron cargar accidentes.")
        return
    df_acc = df_acc[df_acc['ultimo_portico'].isin(porticos_sel)].copy()
    print(f"✅ Accidentes en pórticos seleccionados: {len(df_acc)}")

    # 3) Cargar features y filtrar por pórticos + franja horaria
    print("\n▶️ Cargando features…")
    df_feat = _load_latest_features_df()
    portico_col = buscar_columna(df_feat, "portico")
    ts_col = buscar_columna(df_feat, "ts_min")
    df_feat = df_feat[df_feat[portico_col].isin(porticos_sel)].copy()
    if df_feat.empty:
        print("❌ Features vacías tras filtrar pórticos seleccionados.")
        return
    try:
        df_feat = _filter_time_slot(df_feat, ts_col, pre_time_choice)
    except RuntimeError as e:
        print(str(e))
        return

    # 4) Construir target a partir de accidentes
    df_with_target = _build_target_from_accidents(df_feat, df_acc, portico_col)

    # 5) Selección de columnas de features
    selected = _load_selected_feature_names()
    if selected:
        feature_cols = [c for c in selected if c in df_with_target.columns]
    else:
        feature_cols = [c for c in df_with_target.columns if c not in {portico_col, ts_col, 'target'} and pd.api.types.is_numeric_dtype(df_with_target[c])]
    if not feature_cols:
        print("❌ No hay columnas de features numéricas disponibles.")
        return

    # 6) Imputación + estandarización
    X_df = df_with_target[feature_cols].copy().fillna(df_with_target[feature_cols].median())
    X_all, scaler_state = _standardize_fit_transform(X_df)
    y_all = df_with_target['target'].astype(int).to_numpy()

    # 7) Splits estratificados por índices
    n = X_all.shape[0]
    idx_train, idx_val, idx_test = _train_val_test_split_indices(n, y_all)
    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val, y_val = (X_all[idx_val], y_all[idx_val]) if idx_val is not None else (None, None)
    X_test, y_test = (X_all[idx_test], y_all[idx_test]) if idx_test is not None else (None, None)

    # 8) HPO ligero para SMOTE (ratio, k) usando AUPRC en validación
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(SEED)
    current_ratio = float((y_train == 1).mean()) if y_train.size > 0 else 0.0
    ratio_grid = sorted(set([
        float(np.clip(r, current_ratio + 1e-6, 0.9)) for r in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    ]))
    k_grid = [3, 5, 7]
    best_cfg = None
    best_val_auprc = -np.inf

    if X_val is None or y_val is None or len(np.unique(y_val)) < 2:
        # Si no hay val, hacemos un split interno sobre train
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    else:
        X_tr, y_tr = X_train, y_train
        X_va, y_va = X_val, y_val

    for ratio in ratio_grid:
        for k in k_grid:
            try:
                Xb, yb = _smote_with_ratio(X_tr, y_tr, target_pos_ratio=ratio, k_neighbors=k, random_state=SEED)
                # Entrenar XGBoost rápido
                clf = XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    max_depth=4,
                    learning_rate=0.1,
                    n_estimators=150,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    tree_method='hist'
                )
                clf.fit(Xb, yb)
                prob_va = clf.predict_proba(X_va)[:, 1]
                auprc = average_precision_score(y_va, prob_va) if len(np.unique(y_va)) > 1 else 0.0
                if auprc > best_val_auprc + 1e-9:
                    best_val_auprc = auprc
                    best_cfg = {"ratio": ratio, "k": k}
            except Exception:
                continue

    if best_cfg is None:
        print("⚠️ No se pudo optimizar SMOTE; se usará balance simple (ratio=0.5, k=5).")
        best_cfg = {"ratio": max(0.5, current_ratio), "k": 5}
    print(f"✅ Mejor SMOTE: ratio={best_cfg['ratio']:.3f}, k={best_cfg['k']} (val AUPRC={best_val_auprc:.4f})")

    # 9) Aplicar SMOTE final sobre TRAIN completo y exportar CSV
    Xb_train, yb_train = _smote_with_ratio(X_train, y_train, target_pos_ratio=best_cfg['ratio'], k_neighbors=best_cfg['k'], random_state=SEED)
    _export_balanced_train_csv(Xb_train, yb_train, feature_cols, prefix="xgb")

    # 10) Entrenar XGBoost final
    final_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=5,
        learning_rate=0.08,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method='hist'
    )
    final_clf.fit(Xb_train, yb_train)

    # 11) Seleccionar umbral óptimo (F0.5) en validación
    if X_val is None or y_val is None or len(np.unique(y_val)) < 2:
        # usar split interno
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    else:
        X_va, y_va = X_val, y_val
    prob_va, yva = _evaluate(final_clf, X_va, y_va)
    tau, info = _pick_threshold_from_val(yva, prob_va, beta=0.5)
    print(f"🔧 Umbral seleccionado (val): tau={tau:.6f} → P={info['precision']:.3f} R={info['recall']:.3f}")

    # 12) Evaluación en splits
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
    results = {}

    def eval_split(Xs, ys, idxs, name: str, ts_stamp: str):
        if Xs is None or ys is None:
            return
        prob, yt = _evaluate(final_clf, Xs, ys)
        yp = (prob >= tau).astype(int)
        auroc = roc_auc_score(yt, prob) if len(np.unique(yt)) > 1 else float('nan')
        auprc = average_precision_score(yt, prob) if len(np.unique(yt)) > 1 else float('nan')
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        print(f"\n--- {name} ---")
        rep_txt = classification_report(
            yt,
            yp,
            labels=[0, 1],
            target_names=['No (0)', 'Sí (1)'],
            digits=4,
            zero_division=0
        )
        print(rep_txt)
        rep = classification_report(yt, yp, labels=[0, 1], target_names=['No (0)', 'Sí (1)'], output_dict=True, zero_division=0)
        print("Matriz de Confusión:")
        print(cm)
        results[name] = {
            'auroc': float(auroc) if auroc == auroc else float('nan'),
            'auprc': float(auprc) if auprc == auprc else float('nan'),
            'threshold': float(tau),
            'support_pos': int((yt == 1).sum()),
            'support_neg': int((yt == 0).sum()),
            'precision_1': rep.get('1', {}).get('precision', None),
            'recall_1': rep.get('1', {}).get('recall', None),
            'f1_1': rep.get('1', {}).get('f1-score', None),
            'accuracy': rep.get('accuracy', None),
        }
        # Guardar predicciones por split con claves
        if idxs is not None and name != 'train':
            keys_df = df_with_target.iloc[idxs][[portico_col, ts_col]].reset_index(drop=True)
            out_df = keys_df.copy()
            out_df['y_true'] = yt
            out_df['prob1'] = prob
            out_df['y_pred'] = yp
            out_df['split'] = name
            os.makedirs(RESULTADOS_DIR, exist_ok=True)
            preds_path = os.path.join(RESULTADOS_DIR, f"preds_xgb_{name}_{ts_stamp}.csv")
            out_df.to_csv(preds_path, index=False)
            print(f"📄 Predicciones ({name}) → {preds_path}")

    ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_split(Xb_train, yb_train, idx_train, 'train', ts_stamp)
    eval_split(X_val, y_val, idx_val, 'val', ts_stamp)
    eval_split(X_test, y_test, idx_test, 'test', ts_stamp)

    # 13) Guardar resultados y modelo
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    res_path = os.path.join(RESULTADOS_DIR, f"results_xgb_{ts_stamp}.csv")
    rows = []
    for split, d in results.items():
        row = {'model': 'xgboost', 'split': split}
        row.update(d)
        rows.append(row)
    pd.DataFrame(rows).to_csv(res_path, index=False)
    print(f"📄 Resultados (unificados) guardados → {res_path}")

    try:
        # Guardar modelo en JSON (portable) + scaler
        model_json_path = os.path.join(RESULTADOS_DIR, f"xgb_best_model_{ts_stamp}.json")
        final_clf.get_booster().save_model(model_json_path)
        scaler_path = os.path.join(RESULTADOS_DIR, f"xgb_scaler_{ts_stamp}.npz")
        np.savez(scaler_path, mean=scaler_state['mean'], scale=scaler_state['scale'], feature_cols=np.array(feature_cols, dtype=object))
        # Alias estables (últimos mejores)
        model_json_alias = os.path.join(RESULTADOS_DIR, "xgb_best_model.json")
        try:
            final_clf.get_booster().save_model(model_json_alias)
        except Exception:
            pass
        scaler_alias = os.path.join(RESULTADOS_DIR, "xgb_scaler.npz")
        try:
            np.savez(scaler_alias, mean=scaler_state['mean'], scale=scaler_state['scale'], feature_cols=np.array(feature_cols, dtype=object))
        except Exception:
            pass
        print(f"💾 Modelo guardado → {model_json_path} (alias: {model_json_alias})")
        print(f"💾 Escalador guardado → {scaler_path} (alias: {scaler_alias})")

        # Sidecar meta (único + alias)
        try:
            import json as _json
            meta = {
                'threshold': float(tau),
                'feature_cols': feature_cols,
                'smote_ratio': float(best_cfg['ratio']) if isinstance(best_cfg, dict) and 'ratio' in best_cfg else None,
                'smote_k': int(best_cfg['k']) if isinstance(best_cfg, dict) and 'k' in best_cfg else None,
                'results': results,
                'git_commit': _get_git_commit(),
            }
            meta_unique = os.path.join(RESULTADOS_DIR, f"xgb_best_model_meta_{ts_stamp}.json")
            with open(meta_unique, 'w') as f:
                _json.dump(meta, f)
            meta_alias = os.path.join(RESULTADOS_DIR, "xgb_best_model_meta.json")
            try:
                with open(meta_alias, 'w') as f:
                    _json.dump(meta, f)
            except Exception:
                pass
            print(f"📝 Meta guardada → {meta_unique} (alias: {meta_alias})")
        except Exception as e:
            print(f"⚠️ No se pudo guardar meta JSON: {e}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar el modelo o escalador: {e}")

def _get_git_commit() -> Optional[str]:
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit or None
    except Exception:
        return None
    print("\n✅ Flujo XGBoost finalizado.")
