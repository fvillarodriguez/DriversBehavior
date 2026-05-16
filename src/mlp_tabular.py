import os
import glob
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from src.config import RESULTADOS_DIR, SEED, TRAIN_RATIO, VAL_RATIO, DEBUG, DT_MINUTES, F_BETA_THRESHOLD
from src.utils import buscar_columna, load_accidentes


def _latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def load_latest_features_df() -> pd.DataFrame:
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


def load_selected_feature_names() -> Optional[List[str]]:
    path = _latest_file(os.path.join(RESULTADOS_DIR, "features_selected_names_*.csv"))
    if not path:
        print("⚠️ No se encontró archivo de nombres de features seleccionadas. Se usarán todas las columnas numéricas.")
        return None
    print(f"▶️ Usando selección de features desde: {os.path.basename(path)}")
    df = pd.read_csv(path)
    col = buscar_columna(df, "feature_name")
    names = df[col].dropna().astype(str).tolist()
    return names


def build_target(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Devuelve features_df con columna 'target' y el nombre de columna de pórtico."""
    df = features_df.copy()

    # Identificar columnas clave
    portico_col = buscar_columna(df, "portico")
    ts_col = buscar_columna(df, "ts_min")

    # Cargar accidentes (interactivo) y mapear a ventanas
    df_acc = load_accidentes()
    if df_acc is None or df_acc.empty:
        raise RuntimeError("No se pudieron cargar los accidentes para construir el target.")

    # Crear ts_min para accidentes y set de lookup
    df_acc['ts_min'] = (df_acc["accidente_time"].dt.floor(f'{int(DT_MINUTES)}min').astype("int64") // 60_000_000_000)
    accident_events = set(zip(df_acc['ultimo_portico'], df_acc['ts_min']))

    df['target'] = df.apply(lambda row: 1 if (row[portico_col], row[ts_col]) in accident_events else 0, axis=1)
    if df['target'].sum() == 0:
        print("⚠️ Advertencia: No se encontraron positivos tras el cruce features-accidentes.")
    return df, portico_col


def _simple_smote(X: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE mínimo con sklearn NearestNeighbors. Solo oversample de la clase minoritaria hasta
    igualar la mayoritaria. Trabaja en espacio estandarizado.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y
    maj = classes[np.argmax(counts)]
    min_ = classes[np.argmin(counts)]
    n_maj, n_min = int(counts.max()), int(counts.min())
    if n_min == 0:
        return X, y
    n_new = n_maj - n_min
    if n_new <= 0:
        return X, y

    X_min = X[y == min_]
    # k=5 vecinos (sin incluir self)
    k = min(5, max(1, X_min.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X_min)
    dist, idx = nn.kneighbors(X_min, return_distance=True)
    idx = idx[:, 1:]

    base_idx = rng.integers(0, X_min.shape[0], size=n_new)
    nbr_pos = rng.integers(0, idx.shape[1], size=n_new)
    nbr_idx = idx[base_idx, nbr_pos]
    alpha = rng.random(size=n_new)
    X_new = (1 - alpha)[:, None] * X_min[base_idx] + alpha[:, None] * X_min[nbr_idx]
    y_new = np.full(n_new, min_, dtype=y.dtype)
    X_bal = np.vstack([X, X_new])
    y_bal = np.concatenate([y, y_new])
    return X_bal, y_bal


def _pick_threshold_from_val(y_true_val: np.ndarray, y_prob1_val: np.ndarray, beta: float = 0.5) -> Tuple[float, Dict[str, float]]:
    prec, rec, thr = precision_recall_curve(y_true_val.astype(int), y_prob1_val.astype(float))
    prec_, rec_, thr_ = prec[:-1], rec[:-1], thr
    beta2 = beta ** 2
    # Manejo de casos degenerados (p.ej., validación con una sola clase)
    if len(prec_) == 0 or len(thr_) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "fbeta": 0.0}
    fbeta = (1 + beta2) * prec_ * rec_ / np.clip(beta2 * prec_ + rec_, 1e-12, None)
    i = int(np.nanargmax(fbeta))
    tau = float(thr_[i])
    return tau, {"precision": float(prec_[i]), "recall": float(rec_[i]), "fbeta": float(fbeta[i])}


@dataclass
class MLPConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 30
    early_stopping: int = 5


class MLPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 2,
        use_batch_norm: bool = False,
        hidden_activation: str = "relu",
    ):
        super().__init__()
        layers: List[nn.Module] = []
        dim_in = in_dim
        for i in range(max(0, num_layers)):
            layers.append(nn.Linear(dim_in, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._build_activation(hidden_activation))
            layers.append(nn.Dropout(dropout))
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, num_classes))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        key = str(name or "relu").strip().lower().replace("-", "_")
        if key == "gelu":
            return nn.GELU()
        if key == "elu":
            return nn.ELU()
        if key == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        if key == "tanh":
            return nn.Tanh()
        return nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_one(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * yb.size(0)
        n += yb.size(0)
    return total / max(1, n)


@torch.no_grad()
def _eval_collect(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs1, ys = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        p = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        probs1.append(p)
        ys.append(yb.numpy())
    return np.concatenate(probs1), np.concatenate(ys)


def _export_balanced_train_csv(Xb: np.ndarray, yb: np.ndarray, feature_names: List[str]) -> str:
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%d%m%Y_%H%M')
    path = os.path.join(RESULTADOS_DIR, f"mlp_balanced_train_{ts}.csv")
    df = pd.DataFrame(Xb, columns=feature_names)
    df['target'] = yb.astype(int)
    df.to_csv(path, index=False)
    print(f"💾 Train balanceado (SMOTE) exportado → {path}")
    return path


def prepare_tabular_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # X_train, y_train, idx_train
    np.ndarray, np.ndarray, np.ndarray,  # X_val, y_val, idx_val
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],  # X_test, y_test, idx_test
    List[str], StandardScaler, pd.DataFrame, str, str
]:
    """
    Carga features y accidentes, construye target y genera splits estratificados.
    Aplica estandarización feature-wise. Devuelve X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler
    """
    df_feat = load_latest_features_df()
    selected = load_selected_feature_names()
    df_with_target, portico_col = build_target(df_feat)
    ts_col = buscar_columna(df_with_target, "ts_min")

    # Seleccionar columnas numéricas (o seleccionadas si hay archivo)
    if selected:
        feature_cols = [c for c in selected if c in df_with_target.columns]
    else:
        # todas menos claves y target
        feature_cols = [c for c in df_with_target.columns if c not in {portico_col, ts_col, 'target'}]
        # mantener solo numéricas
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_with_target[c])]

    X_df = df_with_target[feature_cols].copy()
    y = df_with_target['target'].astype(int).to_numpy()

    # Imputación simple: medianas
    X_df = X_df.fillna(X_df.median())

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_df.values)
    n = X_all.shape[0]
    all_idx = np.arange(n)

    # Ratios desde config (porcentaje enteros)
    train_size = float(TRAIN_RATIO) / 100.0
    val_size = float(VAL_RATIO) / 100.0
    test_size = max(0.0, 1.0 - train_size - val_size)

    if test_size > 0:
        X_tmp, X_test, y_tmp, y_test, idx_tmp, idx_test = train_test_split(
            X_all, y, all_idx, test_size=test_size, random_state=SEED, stratify=y
        )
    else:
        X_tmp, X_test, y_tmp, y_test, idx_tmp, idx_test = X_all, None, y, None, all_idx, None

    val_rel = val_size / max(1e-9, (train_size + val_size))
    if val_size > 0:
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_tmp, y_tmp, idx_tmp, test_size=val_rel, random_state=SEED, stratify=y_tmp
        )
    else:
        X_train, X_val, y_train, y_val, idx_train, idx_val = X_tmp, None, y_tmp, None, idx_tmp, None

    return (
        X_train, y_train, idx_train,
        X_val, y_val, idx_val,
        X_test, y_test, idx_test,
        feature_cols, scaler, df_with_target, portico_col, ts_col,
    )


def run_mlp_tabular_pipeline():
    """Flujo completo: preparar datos, SMOTE en train, HPO simple, entrenamiento y reporte."""
    # Dispositivo
    # Semillas locales para reproducibilidad del pipeline
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    (
        X_train, y_train, idx_train,
        X_val, y_val, idx_val,
        X_test, y_test, idx_test,
        feature_cols, scaler, df_with_target, portico_col, ts_col,
    ) = prepare_tabular_data()

    # SMOTE en TRAIN únicamente
    Xb_train, yb_train = _simple_smote(X_train, y_train, random_state=SEED)
    _export_balanced_train_csv(Xb_train, yb_train, feature_cols)

    # DataLoaders
    def make_loader(X, y, bs, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)

    # HPO (ligero)
    try:
        import optuna  # type: ignore
    except Exception:
        optuna = None  # noqa: F841

    def suggest_configs(n_trials: int = 10) -> List[MLPConfig]:
        rng = np.random.default_rng(SEED)
        cfgs: List[MLPConfig] = []
        for _ in range(n_trials):
            cfgs.append(MLPConfig(
                hidden_dim=int(rng.integers(64, 513)),
                num_layers=int(rng.integers(1, 4)),
                dropout=float(rng.uniform(0.0, 0.5)),
                lr=float(10 ** rng.uniform(-4.5, -2.5)),
                weight_decay=float(10 ** rng.uniform(-6.0, -3.0)),
                batch_size=int(rng.choice([256, 512, 1024, 2048])),
                epochs=30,
                early_stopping=5,
            ))
        return cfgs

    best = None
    best_val = -np.inf
    best_state = None
    in_dim = Xb_train.shape[1]

    # Dividir validación si no existe
    have_val = X_val is not None and y_val is not None
    if not have_val:
        Xb_train, X_val, yb_train, y_val = train_test_split(
            Xb_train, yb_train, test_size=0.2, random_state=SEED, stratify=yb_train
        )
    else:
        yb_train = yb_train

    trial_cfgs = suggest_configs(12)
    for cfg in trial_cfgs:
        model = MLPNet(in_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        train_loader = make_loader(Xb_train, yb_train, cfg.batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, 4096, shuffle=False)

        best_epoch_val = -np.inf
        no_improve = 0
        for epoch in range(1, cfg.epochs + 1):
            _ = _train_one(model, train_loader, opt, device)
            # Validación por AUPRC
            probs_val, yv = _eval_collect(model, val_loader, device)
            auprc = average_precision_score(yv, probs_val) if len(np.unique(yv)) > 1 else 0.0
            if auprc > best_epoch_val + 1e-6:
                best_epoch_val = auprc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cfg.early_stopping:
                break

        if best_epoch_val > best_val:
            best_val = best_epoch_val
            best = cfg
            best_state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in model.state_dict().items()}

    if best is None:
        print("❌ No se pudo encontrar una configuración válida para MLP.")
        return

    # Entrenar final con mejor config y evaluar
    print(f"✅ Mejor configuración MLP: hidden={best.hidden_dim}, layers={best.num_layers}, dropout={best.dropout:.2f}, lr={best.lr:.4g}, wd={best.weight_decay:.1e}, bs={best.batch_size}")
    model = MLPNet(in_dim, best.hidden_dim, best.num_layers, best.dropout).to(device)
    model.load_state_dict(best_state)

    # Si existe split de test
    results = {}
    # Umbral óptimo por F0.5 desde validación
    val_loader = make_loader(X_val, y_val, 4096, shuffle=False)
    probs_val, yv = _eval_collect(model, val_loader, device)
    tau, info = _pick_threshold_from_val(yv, probs_val, beta=F_BETA_THRESHOLD)
    print(f"🔧 Umbral seleccionado (val) [Fβ, β={F_BETA_THRESHOLD}]: tau={tau:.6f} → P={info['precision']:.3f} R={info['recall']:.3f}")

    def eval_split(Xs, ys, idxs, name: str, ts_stamp: str):
        if Xs is None or ys is None:
            return
        loader = make_loader(Xs, ys, 4096, shuffle=False)
        probs, ytrue = _eval_collect(model, loader, device)
        ypred = (probs >= tau).astype(int)
        auroc = roc_auc_score(ytrue, probs) if len(np.unique(ytrue)) > 1 else float('nan')
        auprc = average_precision_score(ytrue, probs) if len(np.unique(ytrue)) > 1 else float('nan')
        cm = confusion_matrix(ytrue, ypred)
        print(f"\n--- {name} ---")
        rep_txt = classification_report(ytrue, ypred, digits=4)
        print(rep_txt)
        rep = classification_report(ytrue, ypred, output_dict=True, zero_division=0)
        print("Matriz de Confusión:")
        print(cm)
        results[name] = {
            'auroc': float(auroc) if auroc == auroc else float('nan'),
            'auprc': float(auprc) if auprc == auprc else float('nan'),
            'threshold': float(tau),
            'support_pos': int((ytrue == 1).sum()),
            'support_neg': int((ytrue == 0).sum()),
            'precision_1': rep.get('1', {}).get('precision', None),
            'recall_1': rep.get('1', {}).get('recall', None),
            'f1_1': rep.get('1', {}).get('f1-score', None),
            'accuracy': rep.get('accuracy', None),
        }
        # Guardar predicciones por split con claves
        if idxs is not None and name != 'train':
            keys_df = df_with_target.iloc[idxs][[portico_col, ts_col]].reset_index(drop=True)
            out_df = keys_df.copy()
            out_df['y_true'] = ytrue
            out_df['prob1'] = probs
            out_df['y_pred'] = ypred
            out_df['split'] = name
            os.makedirs(RESULTADOS_DIR, exist_ok=True)
            preds_path = os.path.join(RESULTADOS_DIR, f"preds_mlp_{name}_{ts_stamp}.csv")
            out_df.to_csv(preds_path, index=False)
            print(f"📄 Predicciones ({name}) → {preds_path}")

    ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_split(Xb_train, yb_train, idx_train, 'train', ts_stamp)
    eval_split(X_val, y_val, idx_val, 'val', ts_stamp)
    eval_split(X_test, y_test, idx_test, 'test', ts_stamp)

    # Guardar resultados y modelo
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    ts = ts_stamp
    res_path = os.path.join(RESULTADOS_DIR, f"results_mlp_{ts}.csv")
    rows = []
    for split, d in results.items():
        row = {'model': 'mlp', 'split': split}
        row.update(d)
        rows.append(row)
    pd.DataFrame(rows).to_csv(res_path, index=False)
    print(f"📄 Resultados (unificados) guardados → {res_path}")

    ts2 = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path_unique = os.path.join(RESULTADOS_DIR, f"mlp_best_model_{ts2}.pt")
    model_payload = {
        'state_dict': model.state_dict(),
        'in_dim': in_dim,
        'cfg': best.__dict__,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': feature_cols,
    }
    torch.save(model_payload, model_path_unique)
    # Alias estable para compatibilidad
    model_path_alias = os.path.join(RESULTADOS_DIR, "mlp_best_model.pt")
    try:
        torch.save(model_payload, model_path_alias)
    except Exception:
        pass
    print(f"💾 Modelo guardado → {model_path_unique} (alias: {model_path_alias})")

    print("\n✅ Flujo MLP finalizado.")
