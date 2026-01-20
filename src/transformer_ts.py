import os
import glob
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve

from src.utils import buscar_columna, load_accidentes
from src.config import RESULTADOS_DIR, DT_MINUTES, TRAIN_RATIO, VAL_RATIO, SEED

def _get_git_commit() -> Optional[str]:
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit or None
    except Exception:
        return None


def _pick_features_csv() -> Optional[str]:
    """Permite elegir un features_{números}.csv (excluye features_{texto} y features_selected_names_*)."""
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "features_*.csv")))
    csv_files = [
        p for p in all_files
        if not os.path.basename(p).startswith("features_selected_names_")
        and re.match(r"^features_\d", os.path.basename(p))
    ]
    if not csv_files:
        print(f"❌ No se encontraron archivos 'features_{{números}}.csv' en '{RESULTADOS_DIR}'.")
        return None
    if len(csv_files) == 1:
        print(f"✅ Usando automáticamente: {os.path.basename(csv_files[0])}")
        return csv_files[0]
    print("Archivos de features disponibles:")
    for i, f in enumerate(csv_files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    try:
        sel = int(input("Seleccione el número de archivo CSV: ").strip()) - 1
        if 0 <= sel < len(csv_files):
            return csv_files[sel]
    except Exception:
        pass
    print("Selección inválida.")
    return None


def _pick_selected_feature_names() -> Optional[List[str]]:
    pattern = os.path.join(RESULTADOS_DIR, "features_selected_names_*.csv")
    name_files = sorted(glob.glob(pattern))
    if not name_files:
        print("⚠️ No se encontraron archivos de nombres de features seleccionadas. Se usarán todas las columnas numéricas.")
        return None
    if len(name_files) == 1:
        df = pd.read_csv(name_files[0])
        feats = df.iloc[:, 0].astype(str).tolist()
        print(f"✅ Usando lista de features: {os.path.basename(name_files[0])} ({len(feats)} columnas)")
        return feats
    print("Listas de features disponibles:")
    for i, f in enumerate(name_files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    try:
        sel = int(input("Seleccione el número de archivo con nombres de features (o 0 para usar todas): ").strip())
        if sel == 0:
            return None
        if 1 <= sel <= len(name_files):
            df = pd.read_csv(name_files[sel - 1])
            feats = df.iloc[:, 0].astype(str).tolist()
            print(f"✅ Usando lista de features: {os.path.basename(name_files[sel - 1])} ({len(feats)} columnas)")
            return feats
    except Exception:
        pass
    print("Selección inválida. Se usarán todas las columnas numéricas.")
    return None


def _time_based_split_unique_ts(df: pd.DataFrame) -> Tuple[set, set, set]:
    ts = np.sort(df['ts_min'].unique())
    n = len(ts)
    n_train = int(np.floor(n * TRAIN_RATIO / 100.0))
    n_val = int(np.floor(n * VAL_RATIO / 100.0))
    n_val = max(1, n_val)
    n_train = max(1, n_train)
    train_ts = set(ts[:n_train])
    val_ts = set(ts[n_train:n_train + n_val])
    test_ts = set(ts[n_train + n_val:])
    return train_ts, val_ts, test_ts


def _generate_target_from_accidents(df_features: pd.DataFrame) -> pd.DataFrame:
    print("Generando variable objetivo (target) desde accidentes...")
    df_acc = load_accidentes()
    if df_acc is None or df_acc.empty:
        raise RuntimeError("No se pudieron cargar los accidentes.")
    df_acc['ts_min'] = (df_acc["accidente_time"].dt.floor(f'{DT_MINUTES}min').astype("int64") // 60_000_000_000)
    portico_col = buscar_columna(df_features, "portico")
    accident_events = set(zip(df_acc['ultimo_portico'], df_acc['ts_min']))
    df_features = df_features.copy()
    df_features['target'] = df_features.apply(
        lambda row: 1 if (row[portico_col], row['ts_min']) in accident_events else 0, axis=1
    )
    n_pos = int(df_features['target'].sum())
    print(f"  - Accidentes encontrados en features: {n_pos}")
    if n_pos == 0:
        print("⚠️ No se encontraron cruces (pórtico, ts) con accidentes.")
    return df_features


def _build_windows(df: pd.DataFrame, feature_cols: List[str], seq_len: int,
                   train_ts: set, val_ts: set, test_ts: set,
                   portico_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X_tr, y_tr = [], []
    X_va, y_va = [], []
    X_te, y_te = [], []
    # Para tracking opcional de los últimos ts de cada ventana
    end_ts_tr, end_ts_va, end_ts_te = [], [], []
    for p, g in df.groupby(portico_col):
        g = g.sort_values('ts_min')
        A = g[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y = g['target'].astype(int).to_numpy()
        ts = g['ts_min'].to_numpy()
        if len(A) < seq_len:
            continue
        for i in range(seq_len - 1, len(A)):
            win = A[i - seq_len + 1:i + 1]
            t_end = int(ts[i])
            if t_end in train_ts:
                X_tr.append(win); y_tr.append(y[i]); end_ts_tr.append(t_end)
            elif t_end in val_ts:
                X_va.append(win); y_va.append(y[i]); end_ts_va.append(t_end)
            elif t_end in test_ts:
                X_te.append(win); y_te.append(y[i]); end_ts_te.append(t_end)
    def to_arr(Xl, yl):
        return (np.asarray(Xl, dtype=np.float32) if Xl else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
                np.asarray(yl, dtype=np.int64) if yl else np.zeros((0,), dtype=np.int64))
    X_tr, y_tr = to_arr(X_tr, y_tr)
    X_va, y_va = to_arr(X_va, y_va)
    X_te, y_te = to_arr(X_te, y_te)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, feature_cols


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        else:
            self.register_buffer('pe', pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) if batch_first else (L, B, d_model)
        if self.batch_first:
            L = x.size(1)
            return x + self.pe[:, :L]
        else:
            L = x.size(0)
            return x + self.pe[:L]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = self.input_proj(x)          # (B, L, d_model)
        x = self.posenc(x)              # (B, L, d_model)
        x = self.transformer(x)         # (B, L, d_model)
        x_last = x[:, -1, :]            # (B, d_model)
        x_last = self.dropout(x_last)
        return self.cls(x_last)         # (B, 2)


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer, device, class_weights=None) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, weight=class_weights)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / max(1, len(loader.dataset))


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: DataLoader, device) -> Dict:
    model.eval()
    ys, yps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob1 = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        yps.append(prob1)
        ys.append(yb.numpy())
    y_true = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
    y_prob = np.concatenate(yps) if yps else np.zeros((0,), dtype=np.float32)
    if len(y_true) == 0:
        return {"report": {"Sí (1)": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0}, "accuracy": 0.0},
                "threshold": 0.5, "y_true": y_true, "y_prob": y_prob}
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec_, rec_, thr_ = prec[1:], rec[1:], thr
    beta2 = 1.0
    fbeta = (1 + beta2) * prec_ * rec_ / np.clip(beta2 * prec_ + rec_, 1e-12, None)
    i = int(np.nanargmax(fbeta)) if len(fbeta) > 0 else 0
    tau = float(thr_[i]) if len(thr_) > 0 else 0.5
    y_pred = (y_prob >= tau).astype(int)
    rep = classification_report(y_true, y_pred, labels=[0, 1], target_names=['No (0)', 'Sí (1)'], output_dict=True, zero_division=0)
    return {"report": rep, "threshold": tau, "y_true": y_true, "y_prob": y_prob}


def run_transformer_timeseries():
    print("\n=== Entrenamiento Transformer (Series Temporales) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    csv_path = _pick_features_csv()
    if not csv_path:
        print("❌ No se seleccionó CSV de features.")
        return
    df = pd.read_csv(csv_path)
    if 'ts_min' not in df.columns:
        raise RuntimeError("El CSV de features debe contener la columna 'ts_min'.")
    portico_col = buscar_columna(df, 'portico')
    df = _generate_target_from_accidents(df)
    selected = _pick_selected_feature_names()
    if selected is None:
        feature_cols = [c for c in df.columns if c not in {portico_col, 'ts_min', 'target'} and np.issubdtype(df[c].dtype, np.number)]
    else:
        feature_cols = [c for c in selected if c in df.columns]
        missing = [c for c in selected if c not in df.columns]
        if missing:
            print(f"⚠️ Features seleccionadas no presentes y serán omitidas: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if not feature_cols:
        print("❌ No hay features válidas.")
        return

    # Ventana temporal
    try:
        seq_len = int(input("Largo de ventana temporal (ej. 12): ").strip())
        if seq_len < 2:
            seq_len = 12
    except Exception:
        seq_len = 12
    print(f"Usando ventana temporal L={seq_len}")

    # Split por tiempo
    train_ts, val_ts, test_ts = _time_based_split_unique_ts(df)
    X_tr, y_tr, X_va, y_va, X_te, y_te, feature_cols = _build_windows(df, feature_cols, seq_len, train_ts, val_ts, test_ts, portico_col)

    if X_tr.shape[0] == 0 or X_va.shape[0] == 0 or X_te.shape[0] == 0:
        print("⚠️ Con las configuraciones actuales no se pudieron generar ventanas suficientes para train/val/test.")
        print(f"Tamaños → train:{X_tr.shape}, val:{X_va.shape}, test:{X_te.shape}")
        return

    # Escalado (fit en TRAIN sólo)
    scaler = StandardScaler()
    Btr, L, D = X_tr.shape
    scaler.fit(X_tr.reshape(Btr * L, D))
    def _tfm(X):
        B, L, D = X.shape
        Z = scaler.transform(X.reshape(B * L, D)).reshape(B, L, D)
        return Z.astype(np.float32)
    X_tr, X_va, X_te = _tfm(X_tr), _tfm(X_va), _tfm(X_te)

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    def make_loader(X, y, bs, sh):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=bs, shuffle=sh)

    # HPO con optuna (si disponible)
    try:
        import optuna  # type: ignore
    except Exception:
        optuna = None  # type: ignore

    def run_cfg(cfg):
        model = TimeSeriesTransformer(
            input_dim=len(feature_cols), d_model=cfg['d_model'], nhead=cfg['nhead'],
            num_layers=cfg['num_layers'], dim_feedforward=cfg['ff'], dropout=cfg['dropout']
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        # pesos de clase desde TRAIN
        pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
        cw = None
        if pos > 0 and neg > 0:
            w0 = pos / (pos + neg)
            w1 = neg / (pos + neg)
            cw = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        tr_loader = make_loader(X_tr, y_tr, cfg['bs'], True)
        va_loader = make_loader(X_va, y_va, 256, False)
        best_f1, patience, pat = -1.0, 10, 0
        for _ in range(cfg['epochs']):
            _train_epoch(model, tr_loader, opt, device, class_weights=cw)
            val = _eval_epoch(model, va_loader, device)
            f1 = val['report']['Sí (1)']['f1-score']
            if f1 > best_f1 + 1e-6:
                best_f1, pat = f1, 0
            else:
                pat += 1
            if pat >= patience:
                break
        return best_f1, model

    if optuna is not None:
        def space(trial):
            return dict(
                d_model=trial.suggest_categorical('d_model', [64, 128, 256]),
                nhead=trial.suggest_categorical('nhead', [2, 4, 8]),
                num_layers=trial.suggest_int('num_layers', 1, 3),
                ff=trial.suggest_categorical('ff', [128, 256, 512, 1024]),
                dropout=trial.suggest_float('dropout', 0.0, 0.4),
                lr=trial.suggest_float('lr', 1e-4, 5e-3, log=True),
                wd=trial.suggest_float('wd', 1e-7, 1e-3, log=True),
                bs=trial.suggest_categorical('bs', [64, 128, 256]),
                epochs=trial.suggest_int('epochs', 10, 50),
            )
        def objective(trial):
            cfg = space(trial)
            f1, _ = run_cfg(cfg)
            return f1
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10, show_progress_bar=True)
        best_params = study.best_params
        best_cfg = dict(
            d_model=best_params.get('d_model', 128),
            nhead=best_params.get('nhead', 4),
            num_layers=best_params.get('num_layers', 2),
            ff=best_params.get('ff', 256),
            dropout=best_params.get('dropout', 0.1),
            lr=best_params.get('lr', 1e-3),
            wd=best_params.get('wd', 1e-5),
            bs=best_params.get('bs', 128),
            epochs=best_params.get('epochs', 30),
        )
        best_f1, best_model = run_cfg(best_cfg)
    else:
        candidates = [
            dict(d_model=128, nhead=4, num_layers=2, ff=256, dropout=0.1, lr=1e-3, wd=1e-5, bs=128, epochs=30),
            dict(d_model=256, nhead=8, num_layers=2, ff=512, dropout=0.2, lr=1e-3, wd=1e-5, bs=64, epochs=30),
        ]
        best_model, best_cfg, best_f1 = None, None, -1.0
        for cfg in candidates:
            f1, model = run_cfg(cfg)
            if f1 > best_f1:
                best_model, best_cfg, best_f1 = model, cfg, f1

    # Evaluar en VALIDACIÓN y TEST
    va_loader = DataLoader(TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()), batch_size=256)
    te_loader = DataLoader(TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long()), batch_size=256)
    val_res = _eval_epoch(best_model, va_loader, device)
    test_res = _eval_epoch(best_model, te_loader, device)
    print("\n--- Reporte Validación (Transformer) ---")
    rep_v = val_res['report']
    print(f"F1-Accidente (1): {rep_v['Sí (1)']['f1-score']:.4f} | Precision: {rep_v['Sí (1)']['precision']:.4f} | Recall: {rep_v['Sí (1)']['recall']:.4f}")
    print(f"Accuracy: {rep_v['accuracy']:.4f}")
    print("--- Reporte Test ---")
    rep_t = test_res['report']
    print(f"F1-Accidente (1): {rep_t['Sí (1)']['f1-score']:.4f} | Precision: {rep_t['Sí (1)']['precision']:.4f} | Recall: {rep_t['Sí (1)']['recall']:.4f}")
    print(f"Accuracy: {rep_t['accuracy']:.4f}")

    # Guardar
    ts = datetime.now().strftime('%d%m%Y_%H%M')
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTADOS_DIR, f"transformer_ts_best_{ts}.pt")
    payload = {
        'state_dict': best_model.state_dict(),
        'feature_cols': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'seq_len': seq_len,
        'best_cfg': best_cfg,
    }
    torch.save(payload, model_path)
    # Alias estable (último mejor)
    alias_path = os.path.join(RESULTADOS_DIR, "transformer_ts_best.pt")
    try:
        torch.save(payload, alias_path)
    except Exception:
        pass
    print(f"💾 Modelo guardado en → {model_path} (alias: {alias_path})")

    # Sidecar meta (único + alias)
    try:
        import json as _json
        meta = {
            'threshold': float(val_res.get('threshold', 0.5)),
            'val_report': val_res.get('report', {}),
            'test_report': test_res.get('report', {}),
            'feature_cols': feature_cols,
            'seq_len': int(seq_len),
            'best_cfg': best_cfg,
            'git_commit': _get_git_commit(),
        }
        meta_unique = os.path.join(RESULTADOS_DIR, f"transformer_ts_best_meta_{ts}.json")
        with open(meta_unique, 'w') as f:
            _json.dump(meta, f)
        meta_alias = os.path.join(RESULTADOS_DIR, "transformer_ts_best_meta.json")
        try:
            with open(meta_alias, 'w') as f:
                _json.dump(meta, f)
        except Exception:
            pass
        print(f"📝 Meta guardada → {meta_unique} (alias: {meta_alias})")
    except Exception as e:
        print(f"⚠️ No se pudo guardar meta JSON: {e}")

    summary_path = os.path.join(RESULTADOS_DIR, f"transformer_results_{ts}.csv")
    pd.DataFrame([
        {
            'val_f1_acc_1': rep_v['Sí (1)']['f1-score'],
            'val_precision_1': rep_v['Sí (1)']['precision'],
            'val_recall_1': rep_v['Sí (1)']['recall'],
            'val_accuracy': rep_v['accuracy'],
            'test_f1_acc_1': rep_t['Sí (1)']['f1-score'],
            'test_precision_1': rep_t['Sí (1)']['precision'],
            'test_recall_1': rep_t['Sí (1)']['recall'],
            'test_accuracy': rep_t['accuracy'],
        }
    ]).to_csv(summary_path, index=False)
    print(f"📄 Resumen guardado en → {summary_path}")


def run_transformer_ts_pipeline():
    """Alias para menú principal; ejecuta el flujo de entrenamiento del Transformer."""
    return run_transformer_timeseries()

if __name__ == "__main__":
    run_transformer_timeseries()
