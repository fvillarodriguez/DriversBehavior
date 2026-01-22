#!/usr/bin/env python
# tensorboard --logdir=Resultados/runs_attention
import os, math, glob, gc, sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # --- FIX for MPS FallBck ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", # Asignador seguro de CUDA
  "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8")

from typing import Callable, Optional, Union, List, Dict, Tuple

import pandas as pd
import re
from datetime import datetime
import numpy as np
import torch 
from torch_geometric.loader import NeighborLoader 
import torch.nn.functional as F 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, matthews_corrcoef
import optuna 
import logging
from torch.utils.tensorboard import SummaryWriter
import warnings
from optuna import TrialPruned
from optuna.exceptions import ExperimentalWarning
import hashlib
import shutil
from src.legacy_utils import (
    limpiar_pantalla,
    load_graph,
    visualize_acc_subgraph,
    visualize_graph_overview,
    export_graph_to_csv,
    fusionar_grafos,
    select_features,
    menu_utilidades_csv,
    load_flujos,
)
from src.visual import run_visual_server
import json
try:
    import joblib  # para cargar artefactos de anomalías
except Exception:
    joblib = None
from src.graph import build_graph
from src.features import compute_pm_features, compute_pm_features_streaming
from src.train_pretrain import train_minibatch, pretrain_edge_generator
from src.gat_model import HeteroGAT
from src.temporal_head import TemporalAggregator
from src.graphsmote import (RelEdgeGen,train_z2x_decoders,augment_graph_offline_once, refresh_synthetics_online)
from src.imgagn import ImGAGNConfig, train_imgagn
from src.config import (SEED,DT_MINUTES,MAX_EPOCHS,EARLY_STOPPING_PATIENCE,EARLY_STOPPING_MIN_DELTA,RESULTADOS_DIR,
                        BATCH_SIZE,NUM_NEIGHBORS,N_TRIALS,DEBUG,NUM_EPOCHS_OPTUNA,
                        GRAPHSMOTE_MODE, TARGET_POS_RATIO,
                        GRAPHSMOTE_K, PRETRAIN_EDGE_EPOCHS, SMOTE_EVERY_N_EPOCHS,
                        GS_SEED, SAVE_AUG_GRAPH_PATH, DECODER_EPOCHS, F_BETA_THRESHOLD, XAI,
                        EXPORT_LEGACY_GAT_CSV, SAVE_GAT_ALIASES, AUTO_IMGAGN_PRETRAIN,
                        AUTOCALIBRATE_PROBS)
from src.mlp_tabular import run_mlp_tabular_pipeline
from src.transformer_ts import run_transformer_ts_pipeline
from src.xgboost import run_xgboost_pipeline
from src.anomaly import run_anomaly_pipeline

sequence_index_global = None
sequence_config_global = None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

# --- Helpers: model path discovery with variant tags ---
def _has_imgagn(loaded_obj) -> bool:
    try:
        if loaded_obj.get('imgagn_best_params'):
            return True
        fn = str(loaded_obj.get('filename', ''))
        if 'ImGAGN' in fn or '_imgagn_' in fn.lower():
            return True
        return False
    except Exception:
        return False

def _model_tag_suffix(use_graphsmote: bool, loaded_obj) -> str:
    tags = []
    # If not explicitly forced by caller, check if the graph filename already implies GraphSMOTE
    fn = str(loaded_obj.get('filename', '')).lower()
    is_smote_graph = 'graphsmote' in fn or 'graph_aug' in fn
    
    if use_graphsmote or is_smote_graph:
        tags.append('GraphSMOTE')
    if _has_imgagn(loaded_obj):
        tags.append('ImGAGN')
    return ("_" + "_".join(tags)) if tags else ""

def _variant_tags(use_graphsmote: bool, loaded_obj) -> str:
    """Return a variant tag for filenames: _GraphSMOTE or _Base, plus optional _ImGAGN."""
    parts = []
    parts.append('GraphSMOTE' if use_graphsmote else 'Base')
    try:
        if _has_imgagn(loaded_obj):
            parts.append('ImGAGN')
    except Exception:
        pass
    return "_" + "_".join(parts) if parts else ""

def _find_best_model_path(use_graphsmote: bool, loaded_obj, best_params: Optional[dict] = None) -> Optional[str]:
    # Try exact tag match first
    tag = _model_tag_suffix(use_graphsmote, loaded_obj)
    exact = os.path.join(RESULTADOS_DIR, f"gat_model_BEST{tag}.pt")
    if os.path.exists(exact):
        return exact
    # Fallback: filter among any BEST files
    files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt")), key=os.path.getmtime)
    if not files:
        base = os.path.join(RESULTADOS_DIR, "gat_model_BEST.pt")
        return base if os.path.exists(base) else None
    preferred = []
    for f in files:
        name = os.path.basename(f)
        if use_graphsmote and "_GraphSMOTE" not in name:
            continue
        if (not use_graphsmote) and "_GraphSMOTE" in name:
            continue
        if _has_imgagn(loaded_obj) and "_ImGAGN" not in name:
            continue
        preferred.append(f)
    if preferred:
        candidates = preferred
    else:
        candidates = files

    # Si no se especifican best_params, devolver el más reciente compatible
    if not best_params:
        return candidates[-1] if candidates else None

    # Intentar emparejar por sidecar de hparams (hidden_channels/num_heads/num_layers)
    want_hidden = int(best_params.get('hidden_channels', -1))
    want_heads = int(best_params.get('num_heads', -1))
    want_layers = int(best_params.get('num_layers', -1))
    for f in reversed(candidates):  # más recientes primero
        base, _ = os.path.splitext(f)
        meta_path = base + "_hparams.json"
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r') as fh:
                meta = json.load(fh)
            ok = True
            if want_hidden != -1 and int(meta.get('hidden_channels', want_hidden)) != want_hidden:
                ok = False
            if want_heads != -1 and int(meta.get('num_heads', want_heads)) != want_heads:
                ok = False
            if want_layers != -1 and int(meta.get('num_layers', want_layers)) != want_layers:
                ok = False
            if ok:
                return f
        except Exception:
            continue
    # Fallback si no hubo coincidencia exacta
    return candidates[-1] if candidates else None

def _infer_arch_from_state_dict(sd: dict) -> dict:
    """Infer num_layers, num_heads, hidden_channels from a saved state_dict."""
    num_layers = 0
    num_heads = None
    hidden = None
    # Count layers by convs.<i> occurrences
    for k in sd.keys():
        if k.startswith('convs.'):
            try:
                idx = int(k.split('.')[1])
                num_layers = max(num_layers, idx + 1)
            except Exception:
                pass
        if k.endswith('.att_src') and num_heads is None:
            t = sd[k]
            # shape approx [1, heads, out_channels]
            if t.dim() >= 3:
                num_heads = int(t.shape[1])
                hidden = int(t.shape[2])
    # Fallback: try norms.0.pm.weight length = hidden*num_heads
    if (hidden is None or num_heads is None) and ('norms.0.pm.weight' in sd):
        d = int(sd['norms.0.pm.weight'].shape[0])
        # try common head counts
        for h in (8, 6, 4, 2, 1):
            if d % h == 0:
                num_heads = h
                hidden = d // h
                break
    return {
        'num_layers': int(num_layers if num_layers else 2),
        'num_heads': int(num_heads if num_heads else 4),
        'hidden_channels': int(hidden if hidden else 32),
    }

def _gather_gat_models_listing() -> list[dict]:
    files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt")), key=os.path.getmtime)
    # Deduplicar: si existe una versión única con timestamp/hash, ocultar alias sin timestamp
    # Ej.: gat_model_BEST_GraphSMOTE_YYYYmmdd_HHMMSS_abcd1234.pt → oculta gat_model_BEST_GraphSMOTE.pt
    import re
    base_from_unique: set[str] = set()
    unique_re = re.compile(r"^(gat_model_BEST(?:_[A-Za-z0-9]+)*)_\d{8}_\d{6}_[0-9a-fA-F]{8}\.pt$")
    alias_re  = re.compile(r"^(gat_model_BEST(?:_[A-Za-z0-9]+)*)\.pt$")
    for f in files:
        m = unique_re.match(os.path.basename(f))
        if m:
            base_from_unique.add(m.group(1))

    filtered = []
    for f in files:
        name = os.path.basename(f)
        if unique_re.match(name):
            filtered.append(f)
            continue
        m = alias_re.match(name)
        if m and m.group(1) in base_from_unique:
            # Alias duplicado de una versión con timestamp: ocultar
            continue
        filtered.append(f)

    out = []
    for f in filtered:
        base, _ = os.path.splitext(f)
        meta_path = base + "_hparams.json"
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {}
        variant = []
        name = os.path.basename(f)
        if "_GraphSMOTE" in name:
            variant.append('GraphSMOTE')
        # propósito desde meta
        purpose = meta.get('purpose') if isinstance(meta, dict) else None
        if purpose and str(purpose).lower().startswith('anomaly'):
            variant.insert(0, 'Anomalias')
        else:
            # si no hay propósito, inferir 'Base'
            if "_Base" in name or "_GraphSMOTE" in name or True:
                variant.insert(0, 'Base')
        if "_ImGAGN" in name:
            variant.append('ImGAGN')
        # Arch from meta or infer from state dict
        arch = {}
        if all(k in meta for k in ('hidden_channels','num_heads','num_layers')):
            arch = {
                'hidden_channels': int(meta.get('hidden_channels')),
                'num_heads': int(meta.get('num_heads')),
                'num_layers': int(meta.get('num_layers')),
            }
        else:
            try:
                sd = torch.load(f, map_location='cpu')
                arch = _infer_arch_from_state_dict(sd)
            except Exception:
                arch = {'hidden_channels': None, 'num_heads': None, 'num_layers': None}
        ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
        out.append({
            'path': f,
            'name': name,
            'variant': "+".join(variant),
            'time': ts,
            'meta': meta,
            'arch': arch,
            'score': meta.get('best_val_f1'),
            'epoch': meta.get('best_epoch'),
        })
    return out

def _detect_edge_feature_dim(data, node_type: str = 'pm') -> int:
    """Infer the dimensionality of edge attributes for the given node type."""
    try:
        spatial_key = (node_type, 'spatial', node_type)
        if hasattr(data, 'edge_types'):
            if spatial_key in data.edge_types:
                edge_attr = getattr(data[spatial_key], 'edge_attr', None)
                if edge_attr is not None:
                    return int(edge_attr.shape[1])
            for edge_type in data.edge_types:
                edge_attr = getattr(data[edge_type], 'edge_attr', None)
                if edge_attr is not None:
                    return int(edge_attr.shape[1])
    except Exception:
        pass
    return 0

def _resolve_num_neighbors(value, default_value, edge_types) -> dict:
    """Return a NeighborLoader num_neighbors dict based on config/best_params."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = default_value
    parsed = value
    if isinstance(parsed, str):
        parsed = parsed.strip()
        try:
            parsed = json.loads(parsed)
        except Exception:
            if '-' in parsed:
                try:
                    parsed = [int(tok) for tok in parsed.split('-') if tok.strip()]
                except Exception:
                    parsed = None
            elif ',' in parsed and '[' not in parsed:
                try:
                    parsed = [int(tok) for tok in parsed.split(',') if tok.strip()]
                except Exception:
                    parsed = None
    if isinstance(parsed, (int, float)):
        parsed = [int(parsed)]

    def _normalize_profile(val, fallback):
        if val is None:
            return fallback
        if isinstance(val, (int, float)):
            return [int(val)]
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        return fallback

    def _pick_default(source, fallback):
        if isinstance(source, dict):
            for v in source.values():
                prof = _normalize_profile(v, None)
                if prof:
                    return prof
        prof = _normalize_profile(source, None)
        if prof:
            return prof
        return fallback

    if isinstance(parsed, (list, tuple)):
        parsed_list = [int(x) for x in parsed]
        return {edge_type: parsed_list for edge_type in edge_types}
    if isinstance(parsed, dict):
        fallback = _pick_default(default_value, [15, 10])
        default_profile = _pick_default(parsed, fallback)
        return {
            edge_type: _normalize_profile(parsed.get(edge_type), default_profile)
            for edge_type in edge_types
        }
    fallback = _pick_default(default_value, [15, 10])
    return {edge_type: fallback for edge_type in edge_types}

def _safe_cast(value, cast_type, default):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return cast_type(value)
    except Exception:
        return default

def _get_repo_version() -> Optional[str]:
    """Devuelve el commit corto de git si disponible, si no None."""
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit or None
    except Exception:
        return None

def _archive_legacy_gnn_anomaly_files(keep: int = 3):
    """Move older legacy gnn_anomaly results/preds to Resultados/legacy/ keeping last N.
    This avoids clutter and 'gradual removal' of legacy outputs without deleting them.
    """
    try:
        os.makedirs(RESULTADOS_DIR, exist_ok=True)
        legacy_dir = os.path.join(RESULTADOS_DIR, 'legacy')
        os.makedirs(legacy_dir, exist_ok=True)

        patterns = [
            os.path.join(RESULTADOS_DIR, 'results_gnn_anomaly_*.csv'),
            os.path.join(RESULTADOS_DIR, 'preds_gnn_anomaly_*.csv'),
        ]
        for pat in patterns:
            files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
            if len(files) <= keep:
                continue
            for f in files[keep:]:
                try:
                    base = os.path.basename(f)
                    dest = os.path.join(legacy_dir, base)
                    if os.path.abspath(f) != os.path.abspath(dest):
                        shutil.move(f, dest)
                except Exception:
                    # Ignore move errors silently (no hard failure during report)
                    pass
    except Exception:
        pass

def run_gnn_anomaly_pipeline(loaded_obj):
    """
    Reporte de evaluación del modelo GAT entrenado sobre el grafo cargado.
    - Selecciona umbral por F_beta en validación (beta=F_BETA_THRESHOLD).
    - Exporta CSVs de predicciones por split y un CSV unificado de métricas.
    Nota: Por compatibilidad histórica, los archivos también se exportan con el
    prefijo 'gnn_anomaly_' aunque no se trate de un pipeline estrictamente de
    anomalías.
    """
    # Detección automática de variante GraphSMOTE según modelos disponibles
    model_candidates = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt")), key=os.path.getmtime)
    if not model_candidates and os.path.exists(os.path.join(RESULTADOS_DIR, "gat_model_BEST.pt")):
        model_candidates = [os.path.join(RESULTADOS_DIR, "gat_model_BEST.pt")]
    if model_candidates:
        latest = os.path.basename(model_candidates[-1])
        use_graphsmote = ("_GraphSMOTE" in latest)
    else:
        use_graphsmote = False

    # Dispositivo
    # Selection automática de dispositivo
    device = get_auto_device()
    print(f"▶️ Usando dispositivo: {device}")

    data = loaded_obj['data'].to(device)

    # Selección de modelo entrenado disponible (no mostrar HPs)
    models = _gather_gat_models_listing()
    if not models:
        print("❌ No se encontraron modelos GAT en 'Resultados/'. Entrene un modelo primero (opción 8).")
        return
    print("\nModelos GAT disponibles para evaluar:")
    print("  idx | variante           | capas | heads | hidden | F1(val) | epoch | fecha")
    for i, m in enumerate(models, 1):
        arch = m['arch']; meta = m['meta']
        f1 = meta.get('best_val_f1', float('nan'))
        ep = meta.get('best_epoch', '—')
        print(f"  [{i:>2}] {m['variant']:<18} {arch.get('num_layers','?'):>5} {arch.get('num_heads','?'):>6} {arch.get('hidden_channels','?'):>7}  {f1 if f1 is not None else float('nan'):.4f}  {ep!s:>5}  {m['time']}")
    try:
        sel = input("Seleccione un modelo a evaluar (Enter=último): ").strip()
        if sel:
            idx = int(sel) - 1
            if not (0 <= idx < len(models)):
                raise ValueError
            chosen = models[idx]
        else:
            chosen = models[-1]
    except Exception:
        chosen = models[-1]

    # Instanciar modelo con la arquitectura del modelo elegido
    num_classes = 2
    edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')
    in_channels = data['pm'].x.shape[1]
    arch = chosen['arch']
    meta = chosen['meta']
    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=int(arch.get('hidden_channels') or meta.get('hidden_channels', 32)),
        out_channels=num_classes,
        num_heads=int(arch.get('num_heads') or meta.get('num_heads', 4)),
        dropout=float(meta.get('dropout', 0.2)),
        edge_feature_dim=edge_feature_dim,
        num_layers=int(arch.get('num_layers') or meta.get('num_layers', len(NUM_NEIGHBORS))),
        use_checkpointing=False,
        aggr1=meta.get('aggr1', 'sum'),
        aggr2=meta.get('aggr2', 'sum')
    ).to(device)

    # Cargar pesos (elige el modelo correspondiente a la variante seleccionada)
    best_model_path = chosen['path']
    if not best_model_path or not os.path.exists(best_model_path):
        print("❌ No se encontró un modelo GAT compatible. Entrene el modelo primero (g).")
        return
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    except Exception as e:
        print(f"❌ No se pudo cargar el modelo: {e}")
        print("Sugerencia: el modelo no coincide con la arquitectura inferida/meta. Entrene nuevamente [Menú → 8].")
        return

    # Pase inicial sin umbral para recolectar probas en val
    initial = test(model, data, node_type='pm', threshold=None)
    if not initial:
        print("❌ Test inicial vacío.")
        return

    mask_key = 'val_mask' if 'val_mask' in initial else 'train_mask'
    y_true_val = initial[mask_key]['true'].cpu().numpy().ravel()
    y_prob1_val_raw = initial[mask_key]['probs'][:, 1].cpu().numpy().ravel()
    y_prob1_val, platt_model = _platt_scale_probabilities(y_true_val, y_prob1_val_raw)

    # Elegir umbral por F_beta
    tau, info = pick_threshold_from_val(y_true_val, y_prob1_val, mode="fbeta", beta=float(F_BETA_THRESHOLD))
    print(f"🔧 Umbral seleccionado (Fβ, β={F_BETA_THRESHOLD}): tau={tau:.6f} → P={info.get('precision', float('nan')):.3f}, R={info.get('recall', float('nan')):.3f}")

    # Pase final con umbral
    final = test(model, data, node_type='pm', threshold=tau, calibration_model=platt_model)
    if not final:
        print("❌ Test final vacío.")
        return

    # Exportar predicciones por split
    pm_index = loaded_obj.get('pm_index')
    ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Etiquetas de configuración para nombres basadas en el modelo elegido
    tag_parts = []
    # Propósito (Anomalias) si corresponde
    try:
        if isinstance(meta, dict) and str(meta.get('purpose','')).lower().startswith('anomaly'):
            tag_parts.append('Anomalias')
    except Exception:
        pass
    # Variante Base/GraphSMOTE a partir del nombre del archivo
    name = os.path.basename(best_model_path)
    if "_GraphSMOTE" in name:
        tag_parts.append('GraphSMOTE')
    else:
        tag_parts.append('Base')
    # Sufijo ImGAGN si corresponde
    if "_ImGAGN" in name:
        tag_parts.append('ImGAGN')
    tag_suffix = ("_" + "_".join(tag_parts)) if tag_parts else ""
    # Prefijo adicional más claro para usuarios (reporte GAT)
    report_prefix = "gat_report"
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    unified = []

    def export_split(split_key: str, label: str):
        if split_key not in final:
            return
        res = final[split_key]
        node_idx = res.get('node_idx')
        if node_idx is None:
            print(f"⚠️ No hay node_idx para {label}; no se exportarán claves.")
            return
        node_idx = node_idx.cpu().numpy().ravel()
        def _lookup_pm(idx: int):
            if pm_index and hasattr(pm_index, '_rev'):
                rev = pm_index._rev
                if isinstance(rev, dict):
                    return rev.get(idx, (None, None))
                try:
                    if 0 <= idx < len(rev):
                        return rev[idx]
                except Exception:
                    pass
            return (None, None)

        keys = [_lookup_pm(int(i)) for i in node_idx]
        porticos = [k[0] for k in keys]
        tsmins = [k[1] for k in keys]
        probs = res['probs'][:, 1].cpu().numpy().ravel()
        ytrue = res['true'].cpu().numpy().ravel()
        ypred = res['preds'].cpu().numpy().ravel()
        out_df = pd.DataFrame({'portico': porticos, 'ts_min': tsmins, 'y_true': ytrue, 'prob1': probs, 'y_pred': ypred})
        out_df['split'] = label
        # Exportar con nombre más claro (gat_report_*)
        out_path = os.path.join(RESULTADOS_DIR, f"preds_{report_prefix}_{label}{tag_suffix}_{ts_stamp}.csv")
        try:
            out_df.to_csv(out_path, index=False)
        except Exception:
            # Fallback a legacy si falla por cualquier motivo
            out_path = os.path.join(RESULTADOS_DIR, f"preds_gnn_anomaly_{label}{tag_suffix}_{ts_stamp}.csv")
            out_df.to_csv(out_path, index=False)
        # Opcional: compatibilidad histórica (gnn_anomaly_*) que duplica archivos
        if EXPORT_LEGACY_GAT_CSV:
            try:
                out_path_legacy = os.path.join(RESULTADOS_DIR, f"preds_gnn_anomaly_{label}{tag_suffix}_{ts_stamp}.csv")
                if os.path.abspath(out_path_legacy) != os.path.abspath(out_path):
                    out_df.to_csv(out_path_legacy, index=False)
            except Exception:
                pass
        print(f"📄 Predicciones ({label}) → {out_path}")

        # Métricas unificadas
        unified.append({
            'model': 'GAT',
            'split': label,
            'threshold': float(tau),
            'auprc': float(res.get('auprc') or float('nan')),
            'auc': float(res.get('auc') or float('nan')),
            'precision_1': res['report'].get('Accidente (1)', {}).get('precision', None),
            'recall_1': res['report'].get('Accidente (1)', {}).get('recall', None),
            'f1_1': res['report'].get('Accidente (1)', {}).get('f1-score', None),
            'accuracy': res['report'].get('accuracy', None),
        })

    export_split('train_mask', 'train')
    export_split('val_mask', 'val')
    export_split('test_mask', 'test')

    # Guardar resultados unificados
    res_path = os.path.join(RESULTADOS_DIR, f"results_{report_prefix}{tag_suffix}_{ts_stamp}.csv")
    try:
        pd.DataFrame(unified).to_csv(res_path, index=False)
    except Exception:
        res_path = os.path.join(RESULTADOS_DIR, f"results_gnn_anomaly{tag_suffix}_{ts_stamp}.csv")
        pd.DataFrame(unified).to_csv(res_path, index=False)
    # Opcional: también escribir con prefijo legacy (duplica archivos)
    if EXPORT_LEGACY_GAT_CSV:
        try:
            res_path_legacy = os.path.join(RESULTADOS_DIR, f"results_gnn_anomaly{tag_suffix}_{ts_stamp}.csv")
            if os.path.abspath(res_path_legacy) != os.path.abspath(res_path):
                pd.DataFrame(unified).to_csv(res_path_legacy, index=False)
        except Exception:
            pass
    print(f"📄 Resultados (unificados) guardados → {res_path}")

    # Mostrar reporte en terminal (después de exportar)
    if 'train_mask' in final:
        print_evaluation_report(final['train_mask'], "Train")
    if 'val_mask' in final:
        print_evaluation_report(final['val_mask'], "Validation")
    if 'test_mask' in final:
        print_evaluation_report(final['test_mask'], "Test")

    # Archivar archivos legacy gnn_anomaly para evitar sobresaturación
    _archive_legacy_gnn_anomaly_files(keep=3)

    # Mensaje más claro según variante (basado en modelo elegido)
    variant_flags = []
    if 'Anomalias' in tag_parts:
        variant_flags.append('Anomalias')
    if '_GraphSMOTE' in name or 'GraphSMOTE' in tag_parts:
        variant_flags.append('GraphSMOTE')
    if '_ImGAGN' in name or 'ImGAGN' in tag_parts:
        variant_flags.append('ImGAGN')
    if not variant_flags:
        variant_flags.append('Base')
    variant_msg = 'GAT ' + '+'.join(variant_flags)
    print(f"\n✅ Reporte GAT finalizado. Variante: {variant_msg}.")

def _list_artifacts():
    print("\n=== Artefactos en Resultados/ ===")
    patterns = {
        'GAT (variante)': ["gat_model_BEST*.pt"],
        'XGBoost': ["xgb_best_model_*.json", "xgb_scaler_*.npz"],
        'Transformer': ["transformer_ts_best_*.pt"],
        'Anomalias': ["anomaly_best_model_*.joblib"],
    }
    # Utilizado para deduplicar alias de GAT y para anotar variantes
    import re
    unique_re = re.compile(r"^(gat_model_BEST(?:_[A-Za-z0-9]+)*)_\d{8}_\d{6}_[0-9a-fA-F]{8}\.pt$")
    alias_re  = re.compile(r"^(gat_model_BEST(?:_[A-Za-z0-9]+)*)\.pt$")

    for label, pats in patterns.items():
        files = []
        for pat in pats:
            files.extend(glob.glob(os.path.join(RESULTADOS_DIR, pat)))
        files = sorted(files, key=os.path.getmtime, reverse=True)

        # Dedupe solo para GAT: ocultar alias sin timestamp si hay versión única equivalente
        if label.startswith('GAT'):
            base_from_unique: set[str] = set()
            for f in files:
                m = unique_re.match(os.path.basename(f))
                if m:
                    base_from_unique.add(m.group(1))
            deduped = []
            for f in files:
                name = os.path.basename(f)
                if unique_re.match(name):
                    deduped.append(f)
                    continue
                m = alias_re.match(name)
                if m and m.group(1) in base_from_unique:
                    # Alias duplicado: ocultar
                    continue
                deduped.append(f)
            files = deduped

        print(f"\n[{label}] ({len(files)})")
        for f in files[:20]:
            try:
                ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
                size_mb = os.path.getsize(f)/1e6
                line = f"  - {os.path.basename(f)}  ({size_mb:.2f} MB, {ts})"
                # Para GAT, anotar variante/purpose si hay meta
                if label.startswith('GAT') and f.endswith('.pt'):
                    base, _ = os.path.splitext(f)
                    meta_path = base + "_hparams.json"
                    tag = None
                    try:
                        meta = {}
                        if os.path.exists(meta_path):
                            with open(meta_path, 'r') as fh:
                                meta = json.load(fh)
                        purpose = (meta.get('purpose') or '').lower()
                        if purpose.startswith('anomaly'):
                            tag = 'Anomalias'
                        elif '_GraphSMOTE' in os.path.basename(f):
                            tag = 'GraphSMOTE'
                        else:
                            tag = 'Base'
                    except Exception:
                        pass
                    if tag:
                        line += f" [{tag}]"
                print(line)
            except Exception:
                print(f"  - {os.path.basename(f)}")

def _show_alias_meta():
    print("\n=== Metadatos (alias) ===")
    # GAT
    meta_gat = os.path.join(RESULTADOS_DIR, "gat_model_BEST_hparams.json")
    if os.path.exists(meta_gat):
        try:
            with open(meta_gat, 'r') as f:
                data = json.load(f)
            print("[GAT]", {k: data.get(k) for k in ['best_val_f1','best_epoch','use_graphsmote','graphsmote_mode','graph_hash','git_commit']})
        except Exception as e:
            print(f"[GAT] no se pudo leer meta: {e}")
    else:
        print("[GAT] alias sin meta disponible")
    # XGB
    meta_xgb = os.path.join(RESULTADOS_DIR, "xgb_best_model_meta.json")
    if os.path.exists(meta_xgb):
        try:
            with open(meta_xgb, 'r') as f:
                data = json.load(f)
            print("[XGB]", {k: data.get(k) for k in ['threshold','smote_ratio','smote_k']})
        except Exception as e:
            print(f"[XGB] no se pudo leer meta: {e}")
    else:
        print("[XGB] alias sin meta disponible")
    # Transformer
    meta_tr = os.path.join(RESULTADOS_DIR, "transformer_ts_best_meta.json")
    if os.path.exists(meta_tr):
        try:
            with open(meta_tr, 'r') as f:
                data = json.load(f)
            print("[Transformer]", {k: data.get(k) for k in ['threshold','seq_len']})
        except Exception as e:
            print(f"[Transformer] no se pudo leer meta: {e}")
    else:
        print("[Transformer] alias sin meta disponible")
    # Anomalías
    alias_anom = os.path.join(RESULTADOS_DIR, "anomaly_best_model.joblib")
    if joblib and os.path.exists(alias_anom):
        try:
            payload = joblib.load(alias_anom)
            print("[Anomalias]", {k: payload.get(k) for k in ['algo','threshold']})
        except Exception as e:
            print(f"[Anomalias] no se pudo leer alias: {e}")
    else:
        print("[Anomalias] alias no disponible")

def _archive_old_artifacts(method: str, keep: int):
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    archive_dir = os.path.join(RESULTADOS_DIR, 'archive')
    os.makedirs(archive_dir, exist_ok=True)
    method_map = {
        'gat': ["gat_model_BEST*.pt"],
        'xgb': ["xgb_best_model_*.json", "xgb_scaler_*.npz", "xgb_best_model_meta_*.json"],
        'transformer': ["transformer_ts_best_*.pt", "transformer_ts_best_meta_*.json"],
        'anomaly': ["anomaly_best_model_*.joblib"],
    }
    pats = method_map.get(method.lower())
    if not pats:
        print("❌ Método no reconocido. Opciones: gat, xgb, transformer, anomaly")
        return
    files = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(RESULTADOS_DIR, pat)))
    files = sorted(files, key=os.path.getmtime, reverse=True)
    if len(files) <= keep:
        print("No hay suficientes artefactos para archivar.")
        return
    to_move = files[keep:]
    print(f"Archivar {len(to_move)} artefactos a 'archive/' (mantener {keep})? (s/n)")
    ok = input("> ").strip().lower() in ('s','si','y','yes')
    if not ok:
        return
    import shutil
    for f in to_move:
        try:
            dest = os.path.join(archive_dir, os.path.basename(f))
            shutil.move(f, dest)
        except Exception as e:
            print(f"⚠️ No se pudo archivar {os.path.basename(f)}: {e}")
    print("✅ Archivado finalizado.")

def artifacts_menu():
    while True:
        print("\n===== Gestor de Artefactos =====")
        print("  [1] Listar artefactos (histórico)")
        print("  [2] Ver meta de alias por método")
        print("  [3] Archivar históricos (mantener N)")
        print("  [4] Inicializar (borrar todos los artefactos/resultados)")
        print("  [0] Volver")
        choice = input("Elige una opción: ").strip()
        if choice == '1':
            _list_artifacts()
        elif choice == '2':
            _show_alias_meta()
        elif choice == '3':
            method = input("Método [gat|xgb|transformer|anomaly]: ").strip()
            try:
                keep = int(input("¿Cuántos artefactos recientes quieres mantener? (ej. 3): ").strip())
            except Exception:
                keep = 3
            _archive_old_artifacts(method, keep)
        elif choice == '4':
            _initialize_artifacts()
        elif choice == '0':
            break
        else:
            print("❌ Opción no reconocida.")

def _initialize_artifacts():
    print("\n⚠️ Inicializar eliminará artefactos y resultados en 'Resultados/'.")
    print("Esto NO toca la carpeta 'Datos/'.")
    incl_graphs = input("¿Borrar también grafos guardados (highway_graph_*.pt)? (s/n): ").strip().lower() in ('s','si','y','yes')
    print("Para confirmar, escribe exactamente: BORRAR TODO")
    conf = input("> ").strip()
    if conf != 'BORRAR TODO':
        print("Operación cancelada.")
        return
    patterns = [
        # GAT
        "gat_model_BEST*.pt", "gat_model_BEST*_hparams.json", "gat_model_BEST_hparams.json",
        # XGB
        "xgb_best_model*.json", "xgb_scaler*.npz", "xgb_best_model_meta*.json",
        # Transformer
        "transformer_ts_best*.pt", "transformer_ts_best_meta*.json", "transformer_results_*.csv",
        # Anomaly
        "anomaly_best_model*.joblib", "results_anomaly_*.csv", "preds_anomaly_*.csv",
        # GAT report outputs (nuevo y legacy)
        "results_gat_report_*.csv", "preds_gat_report_*.csv",
        "results_gnn_anomaly_*.csv", "preds_gnn_anomaly_*.csv",
        # Baselines
        "results_mlp_*.csv", "preds_mlp_*.csv",
        "results_xgb_*.csv", "preds_xgb_*.csv",
        # HPO
        "optuna_hyperparams_*.csv", "optuna_full_study_*.csv",
        "imgagn_hyperparams_*.csv", "imgagn_full_study_*.csv",
        # ImGAGN augmented test/train graphs dumped under Resultados
        "highway_graph_ImGAGN_*.pt", "graph_aug.pt",
    ]
    if incl_graphs:
        patterns.extend(["highway_graph_*.pt"])  # incluir snapshots de grafo
    total = 0
    errors = 0
    for pat in patterns:
        files = glob.glob(os.path.join(RESULTADOS_DIR, pat))
        for f in files:
            try:
                os.remove(f)
                total += 1
            except Exception as e:
                errors += 1
                print(f"⚠️ No se pudo eliminar {os.path.basename(f)}: {e}")
    # Limpiar carpetas legacy/archive (solo contenidos conocidos)
    for sub in ("legacy", "archive"):
        subdir = os.path.join(RESULTADOS_DIR, sub)
        if os.path.isdir(subdir):
            try:
                for root, _, files in os.walk(subdir):
                    for fn in files:
                        try:
                            os.remove(os.path.join(root, fn))
                            total += 1
                        except Exception:
                            errors += 1
                # Intentar eliminar directorios vacíos
                try:
                    os.rmdir(subdir)
                except Exception:
                    pass
            except Exception:
                pass
    # Eliminar directorios auxiliares completos (decoders, runs, particiones y cache)
    dirs_to_remove = [
        os.path.join(RESULTADOS_DIR, 'z2x_decoders'),
        os.path.join(RESULTADOS_DIR, 'runs_attention'),
        os.path.join('.', 'cache', 'graphsmote'),
    ]
    # Particiones generadas por compute_pm_features_streaming
    try:
        parts = [d for d in os.listdir(RESULTADOS_DIR) if d.startswith('particiones_')]
        for d in parts:
            dirs_to_remove.append(os.path.join(RESULTADOS_DIR, d))
    except Exception:
        pass
    for d in dirs_to_remove:
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
                total += 1
            except Exception as e:
                errors += 1
                print(f"⚠️ No se pudo eliminar la carpeta {d}: {e}")
    print(f"\n✅ Inicialización completada. Archivos eliminados: {total}. Errores: {errors}.")

def run_gnn_anomaly_hpo_then_train(loaded_obj):
    """
    Búsqueda de hiperparámetros para GNN (Anomalías) y entrenamiento automático posterior.
    Respeta la elección de GraphSMOTE y la aplica tanto a la búsqueda como al entrenamiento.
    """
    try:
        use_smote_in = input("¿Incluir GraphSMOTE en la búsqueda/entrenamiento? (s/n): ").strip().lower()
    except Exception:
        use_smote_in = 'n'
    use_graphsmote = use_smote_in in ('s', 'si', 'y', 'yes')

    # Búsqueda (forzada con el flag elegido)
    best_params = search_hyperparameters(loaded_obj, use_graphsmote_search=use_graphsmote)
    if not best_params:
        print("❌ No se obtuvieron hiperparámetros.")
        return
    print("\n▶️ Iniciando entrenamiento con los hiperparámetros encontrados…")
    # Entrenamiento forzando el mismo flag de GraphSMOTE
    # Entrenamiento forzando el mismo flag de GraphSMOTE y etiquetando propósito 'Anomaly'
    if isinstance(loaded_obj, dict):
        loaded_obj['purpose'] = 'Anomaly'
    run_gat_training(loaded_obj, force_use_graphsmote=use_graphsmote, purpose='Anomaly')

# --------------------------------------------------------------------------- #
# SETUP LOGGING
# --------------------------------------------------------------------------- #
def setup_logging(log_dir=os.path.join(RESULTADOS_DIR,"logs")):
    """Configura el logging para guardar en archivo y mostrar en consola."""
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    log_filename = os.path.join(log_dir, f"training_log_{timestamp}.log")

    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File Handler con codificación UTF-8
    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# Inicializar el logger globalmente
logger = setup_logging()

# --------------------------------------------------------------------------- #
# CAPTURE WARNINGS
# --------------------------------------------------------------------------- #
logging.captureWarnings(True)
warnings.filterwarnings("once", category=UserWarning, module="torch_geometric.index")
warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", message="expandable_segments not supported on this platform")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True\\. Gradients will be None",
    category=UserWarning,
    module="torch.utils.checkpoint"
)

# Asegurarse de que las semillas se apliquen al inicio
torch.manual_seed(SEED)
np.random.seed(SEED)

def make_epoch_seeds(g, node_type='pm', pos_to_neg_ratio=3.0, strategy='random',
                     model=None, device=None, topk_hard=None):
    """
    Devuelve índices (semillas) para entrenar esta época: todos los positivos + un subconjunto de negativos.
    - strategy = 'random'   → negativos al azar
    - strategy = 'hard'     → negativos más 'difíciles' (p predicha alta para clase 1)
      Requiere model y device; topk_hard puede ser None (usa ratio) o un entero.
    """
    y = g[node_type].y.cpu()
    tr = g[node_type].train_mask.cpu().bool()

    pos_idx = torch.nonzero(tr & (y == 1), as_tuple=True)[0]
    neg_idx = torch.nonzero(tr & (y == 0), as_tuple=True)[0]

    if pos_idx.numel() == 0:
        # sin positivos → usa todos los train
        return torch.nonzero(tr, as_tuple=True)[0]

    # cuántos negativos mantener
    if topk_hard is not None:
        n_neg_keep = min(int(topk_hard), neg_idx.numel())
    else:
        n_neg_keep = min(int(pos_idx.numel() * float(pos_to_neg_ratio)), neg_idx.numel())

    if strategy == 'hard':
        assert model is not None and device is not None, "Para 'hard' necesitas model y device"
        model.eval()
        g_cpu = g.cpu()
        # scores para negativos (prob clase 1)
        from torch_geometric.loader import NeighborLoader
        loader = NeighborLoader(
            g_cpu, input_nodes=(node_type, neg_idx),
            num_neighbors={k:[15,10] for k in g_cpu.edge_types} if hasattr(g_cpu, 'edge_types') else [15,10],
            batch_size=2048, shuffle=False
        )
        scores = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                edge_attr_dict = {et: batch[et].edge_attr for et in batch.edge_types if 'edge_attr' in batch[et]}
                logits_dict, _, _ = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
                logits = logits_dict[node_type][:batch[node_type].batch_size]
                prob1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
                scores.append(prob1)
        scores = torch.cat(scores, dim=0)
        # mayores probabilidades primero → más difíciles
        order = torch.argsort(scores, descending=True)
        neg_keep = neg_idx[order[:n_neg_keep]]
    else:
        perm = torch.randperm(neg_idx.numel())
        neg_keep = neg_idx[perm[:n_neg_keep]]

    seeds = torch.cat([pos_idx, neg_keep], dim=0)
    seeds = seeds[torch.randperm(seeds.numel())]
    return seeds

# --------------------------------------------------------------------------- #
# ÍNDICE ENTERO PARA NODOS PÓRTICO
# --------------------------------------------------------------------------- #
class PMIndex:
    """Contenedor para mapeos de nodos PM."""
    def __init__(self, p_map, r_map):
        self._map = p_map
        self._rev = r_map
    def __len__(self):
        return len(self._rev)

# --------------------------------------------------------------------------- #
# FUNCIONES PRINCIPALES
# --------------------------------------------------------------------------- #

class FocalLoss(torch.nn.Module):
    """
    Implementación de Focal Loss para manejar desbalance de clases extremo,
    con soporte para ponderación de clases (alpha).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calcular Cross Entropy sin reducción para obtener la pérdida por elemento.
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # `pt` es la probabilidad estimada para la clase correcta.
        pt = torch.exp(-ce_loss)
        
        # Esta es la fórmula de Focal Loss: (1-pt)^gamma * ce_loss
        focal_loss = (1 - pt)**self.gamma * ce_loss

        # Si se proporcionan pesos (alpha), aplicarlos.
        if self.alpha is not None:
            # Mover los pesos al mismo dispositivo que los targets
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device)
            
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)

            # Aplicar el peso correspondiente a la clase de cada muestra.
            focal_loss = self.alpha[targets] * focal_loss
        
        # Aplicar la reducción final (mean, sum, or none).
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def prior_shift_adjust(p_train, p_real, p_hat):
    # p_train: prevalencia en entrenamiento; p_real: prevalencia real
    # p_hat: prob entrenada
    odds = p_hat / np.clip(1 - p_hat, 1e-9, None)
    adj = odds * (p_real/(1-p_real)) / (p_train/(1-p_train))
    return adj / (1 + adj)

def pick_threshold_from_val(y_true_val, y_prob1_val, *, mode="fbeta", beta=0.5, min_precision=None, top_k=None):
    """
    y_true_val: array-like binario (0/1) de validación
    y_prob1_val: probabilidades de la clase 1 en validación
    mode:
      - "fbeta": maximiza F_beta (beta<1 prioriza precisión)
      - "precision_at": exige precisión mínima (min_precision)
      - "topk": fija tau al score del k-ésimo (top_k)
    Devuelve: tau (float), dict con métricas del punto elegido
    """
    y_true_val = np.asarray(y_true_val).astype(int).ravel()
    y_prob1_val = np.asarray(y_prob1_val).astype(float).ravel()

    # Curva PR
    prec, rec, thr = precision_recall_curve(y_true_val, y_prob1_val)
    # Alinear: thr tiene len = len(prec)-1 = len(rec)-1
    prec_, rec_, thr_ = prec[1:], rec[1:], thr

    if top_k is not None and mode == "topk":
        order = np.argsort(-y_prob1_val)
        k = max(1, min(int(top_k), len(order)))
        tau = float(y_prob1_val[order[k-1]])
        # Métricas en ese tau
        pred = (y_prob1_val >= tau).astype(int)
        P = (pred[y_true_val==1].sum()) / max(pred.sum(), 1)
        R = (pred & (y_true_val==1)).sum() / max((y_true_val==1).sum(), 1)
        return tau, {"precision": float(P), "recall": float(R)}

    if min_precision is not None and mode == "precision_at":
        ok = np.where(prec_ >= float(min_precision))[0]
        if ok.size:
            # entre los que cumplen precisión, elige el de mayor recall
            i = ok[np.argmax(rec_[ok])]
            tau = float(thr_[i])
            return tau, {"precision": float(prec_[i]), "recall": float(rec_[i])}
        # si no hay ninguno que cumpla, cae a fbeta

    # por defecto: F_beta
    beta2 = beta**2
    fbeta = (1+beta2) * prec_ * rec_ / np.clip(beta2*prec_ + rec_, 1e-12, None)
    i = int(np.nanargmax(fbeta))
    tau = float(thr_[i])
    return tau, {"precision": float(prec_[i]), "recall": float(rec_[i]), "fbeta": float(fbeta[i])}

@torch.no_grad()
def test(model, data, batch_size=BATCH_SIZE, node_type='pm', threshold=None, calibration_model=None):
    """
    Evalúa el rendimiento de un modelo usando mini-batches para evitar OOM.
    Detecta dinámicamente las máscaras disponibles y calcula métricas adicionales.
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = {}
    
    # Determinar clases y nombres de target
    num_classes = 2
    labels = [0, 1]
    target_names = ['No Accidente (0)', 'Accidente (1)']

    # Mover datos a la CPU para NeighborLoader
    data_cpu = data.cpu()

    temporal_module = getattr(model, 'temporal_head', None)
    if temporal_module is not None:
        temporal_module.eval()
        temporal_module.reset_cache()

    # Detectar máscaras dinámicamente (ej. train_mask, val_mask, test_mask)
    available_masks = [key for key in data_cpu[node_type].keys() if key.endswith('_mask')]

    for mask_name in available_masks:
        mask = data_cpu[node_type][mask_name]
        if mask.sum() == 0:
            logger.info(f"Skipping '{mask_name}' as it is empty.")
            continue

        node_indices = mask.nonzero(as_tuple=False).view(-1)
        
        num_neighbors_cfg = _resolve_num_neighbors(
            NUM_NEIGHBORS, NUM_NEIGHBORS, data_cpu.edge_types
        )
        loader = NeighborLoader(
            data_cpu,
            input_nodes=(node_type, node_indices),
            num_neighbors=num_neighbors_cfg,
            batch_size=batch_size,
            shuffle=False,
        )

        all_preds, all_probs, all_true = [], [], []

        if temporal_module is not None:
            temporal_module.reset_cache()

        for batch in loader:
            batch = batch.to(device)
            
            # FIX: Reconstruir edge_attr_dict desde el batch para asegurar el slicing correcto
            edge_attr_dict = {
                et: batch[et].edge_attr 
                for et in batch.edge_types 
                if 'edge_attr' in batch[et]
            }

            # Obtener predicciones del modelo
            logits_dict, embeddings_dict, _ = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
            pm_embeddings = embeddings_dict[node_type]
            pm_logits_all = logits_dict[node_type]

            target_bs = batch[node_type].batch_size
            if temporal_module is not None:
                logits_target = temporal_module(pm_embeddings, batch[node_type].n_id, target_bs)
            else:
                logits_target = pm_logits_all[:target_bs]
            
            # Calcular probabilidades y predicciones
            probs = F.softmax(logits_target, dim=1)
            prob1 = probs[:, 1]
            if calibration_model is not None:
                prob1_np = prob1.detach().cpu().numpy()
                prob1_cal_np = _apply_platt_model(prob1_np, calibration_model)
                prob1_cal = torch.from_numpy(prob1_cal_np).to(prob1.device)
                probs = probs.clone()
                probs[:, 1] = prob1_cal
                probs[:, 0] = torch.clamp(1.0 - prob1_cal, min=0.0, max=1.0)
                prob1 = probs[:, 1]
            if threshold is None:
                preds = probs.argmax(dim=1)  # comportamiento antiguo
            else:
                preds = (prob1 >= threshold).long()  # UMBRAL DE VALIDACIÓN
            true = batch[node_type].y[:batch[node_type].batch_size]
            
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_true.append(true.cpu())

        y_pred = torch.cat(all_preds)
        y_prob = torch.cat(all_probs)
        y_true = torch.cat(all_true)

        # Calcular métricas
        report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Calcular AUC de forma segura
        auc_score = None
        if len(torch.unique(y_true)) > 1:
            try:
                if num_classes == 2:
                    auc_score = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except ValueError as e:
                logger.warning(f"No se pudo calcular el AUC para {mask_name} a pesar de que hay múltiples clases: {e}")
        else:
            logger.warning(f"Saltando cálculo de AUC para {mask_name} porque solo hay una clase en y_true.")

        # AUPRC (promedio de precisión) y MCC
        auprc = None
        try:
            if num_classes == 2 and len(torch.unique(y_true)) > 1:
                auprc = average_precision_score(y_true, y_prob[:, 1])
        except Exception as e:
            logger.warning(f"No se pudo calcular AUPRC: {e}")

        mcc = None
        try:
            if len(torch.unique(y_pred)) > 1:
                mcc = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            logger.warning(f"No se pudo calcular MCC: {e}")

        results[mask_name] = {
            'report': report,
            'cm': cm,
            'preds': y_pred,
            'probs': y_prob,
            'true': y_true,
            'auc': auc_score,
            'auprc': auprc,
            'mcc': mcc,
            'node_idx': node_indices,  # para exportar claves
        }
        
    return results

def print_evaluation_report(results, dataset_name):
    """
    Imprime un reporte de evaluación formateado a partir de los resultados de la función test.
    """
    report = results['report']
    cm = results['cm']
    
    print(f"--- Reporte de Evaluación: Conjunto de {dataset_name} ---")
    print(f"Clase 'No Accidente (0)':")
    print(f"  - Precisión: {report['No Accidente (0)']['precision']:.4f}")
    print(f"  - Recall:    {report['No Accidente (0)']['recall']:.4f}")
    print(f"  - F1-Score:  {report['No Accidente (0)']['f1-score']:.4f}")
    
    print(f"Clase 'Accidente (1)':")
    print(f"  - Precisión: {report['Accidente (1)']['precision']:.4f}")
    print(f"  - Recall:    {report['Accidente (1)']['recall']:.4f}")
    print(f"  - F1-Score:  {report['Accidente (1)']['f1-score']:.4f}")

    print("Globales:")    
    print(f"  - Accuracy:  {report['accuracy']:.4f}")    
    print(f"  - Precision: {report['macro avg']['precision']:.4f}")    
    print(f"  - Recall:    {report['macro avg']['recall']:.4f}")    
    print(f"  - F1-Score:  {report['macro avg']['f1-score']:.4f}")
    print(f"  - F1-Macro: {results.get('f1_macro', 0.0):.4f}")
    print(f"  - AUC: {results.get('auc') or 0.0:.4f}")
    print(f"  - AUPRC (clase 1): {results.get('auprc') or 0.0:.4f}")
    print(f"  - MCC: {results.get('mcc') or 0.0:.4f}")
    
    print("Matriz de Confusión:")    
    print(cm)    
    print("--------------------------------------------------")

def _calculate_graph_hash(graph_filename):
    """Calcula el hash SHA256 del contenido de un archivo de grafo."""
    if not graph_filename:
        logger.error("No se proporcionó un nombre de archivo de grafo para calcular el hash.")
        return None

    graph_path = os.path.join(RESULTADOS_DIR, graph_filename)
    if not os.path.exists(graph_path):
        logger.warning(f"No se pudo encontrar el archivo del grafo en {graph_path} para calcular el hash.")
        return None
    
    hasher = hashlib.sha256()
    try:
        with open(graph_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logger.error(f"Error al leer el archivo del grafo para hashear: {e}")
        return None

def run_gat_training(
    loaded_obj,
    force_use_graphsmote: Optional[bool] = None,
    purpose: Optional[str] = None,
    *,
    early_stop: Optional[bool] = None,
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: Optional[float] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    max_epochs: Optional[int] = None,
    smote_num_neighbors: Optional[object] = None,
    optimizer_overrides: Optional[dict] = None,
):
    """
    Entrenamiento GAT completo con:
      - Selección o búsqueda de hiperparámetros.
      - Registro de hparams en TensorBoard.
      - Serialización JSON de la mejor configuración junto al .pt.
      - Integración de GraphSMOTE (offline, online, o ninguno).
    """
    # 0) Dispositivo
    device = get_auto_device()
    logger.info(f"Usando dispositivo: {device}")

    # 1) Cargar datos y estadísticas
    data = loaded_obj['data']

    if AUTO_IMGAGN_PRETRAIN and not _has_imgagn(loaded_obj):
        logger.info("AUTO_IMGAGN_PRETRAIN habilitado: ejecutando ImGAGN previo al entrenamiento GAT.")
        try:
            augmented_obj = run_imgagn_pipeline(dict(loaded_obj), retrain_gat=False)
            if isinstance(augmented_obj, dict) and 'data' in augmented_obj:
                loaded_obj = augmented_obj
                data = loaded_obj['data']
                logger.info("Grafo sustituido por versión aumentada mediante ImGAGN.")
        except Exception as exc:
            logger.warning(f"No se pudo ejecutar ImGAGN automático: {exc}")

    # Preguntar al usuario si desea usar GraphSMOTE (o forzar desde el caller)
    if force_use_graphsmote is None:
        use_graphsmote_input = input("¿Desea utilizar aumentación de nodos sintéticos (GraphSMOTE)? (s/n): ").strip().lower()
        use_graphsmote = use_graphsmote_input in ('s', 'si', 'y', 'yes')
    else:
        use_graphsmote = bool(force_use_graphsmote)
    
    # GRAPHSMOTE_MODE ahora se controla desde config.py, pero la activación general sigue aquí
    if use_graphsmote:
        logger.info(f"GraphSMOTE habilitado para este entrenamiento (Modo: {GRAPHSMOTE_MODE}).")
    else:
        logger.info("GraphSMOTE deshabilitado para este entrenamiento.")
    if purpose:
        logger.info(f"Propósito de entrenamiento: {purpose}")

    # 2) Selección o búsqueda de hiperparámetros
    all_hp_files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "optuna_hyperparams_*.csv")), key=os.path.getmtime)

    # Filtrar por GraphSMOTE e ImGAGN (preferencias) y ordenar por coincidencia + fecha
    want_imgagn = _has_imgagn(loaded_obj)
    def _score_hp(path: str) -> tuple:
        name = os.path.basename(path)
        has_smote = ("_GraphSMOTE" in name)
        has_imgagn = ("_ImGAGN" in name)
        has_base = ("_Base" in name)
        # Coincidencia con selección actual
        ok_smote = (has_smote == use_graphsmote)
        ok_imgagn = (has_imgagn == want_imgagn)
        # Preferir _Base explícito cuando no se usa SMOTE
        base_pref = (has_base and not use_graphsmote)
        return (int(ok_smote), int(ok_imgagn), int(base_pref), os.path.getmtime(path))

    hp_files_sorted = sorted(all_hp_files, key=_score_hp)
    # Mantener solo los que coinciden en smote; de preferencia también imgagn
    hp_files = [f for f in hp_files_sorted if ("_GraphSMOTE" in os.path.basename(f)) == use_graphsmote]

    best_params = None
    if not hp_files:
        logger.info("No se encontraron archivos de hiperparámetros compatibles. Iniciando nueva búsqueda...")
        best_params = search_hyperparameters(loaded_obj, use_graphsmote_search=use_graphsmote, optimizer_overrides=optimizer_overrides)
    else:
        print("--- Selección de Hiperparámetros ---")
        variant_lbl = "GraphSMOTE" if use_graphsmote else "Base"
        if _has_imgagn(loaded_obj):
            variant_lbl += " + ImGAGN"
        print(f"Archivos de hiperparámetros disponibles (filtrados por variante: {variant_lbl}):")
        for i, f in enumerate(hp_files, 1):
            print(f"  [{i}] {os.path.basename(f)}")
        print("  [0] Iniciar nueva búsqueda de hiperparámetros")
        
        try:
            sel_str = input("Seleccione el número del archivo a usar (o 0 para una nueva búsqueda): ").strip()
            sel = int(sel_str)

            if sel == 0:
                logger.info("Iniciando nueva búsqueda de hiperparámetros...")
                best_params = search_hyperparameters(loaded_obj, use_graphsmote_search=use_graphsmote, optimizer_overrides=optimizer_overrides)
            elif 1 <= sel <= len(hp_files):
                hp_path = hp_files[sel - 1]
                logger.info(f"Cargando hiperparámetros desde: {os.path.basename(hp_path)}")
                df_hp = pd.read_csv(hp_path)

                if any(c.startswith('params_') for c in df_hp.columns):
                    df_hp.rename(columns=lambda c: c.replace('params_', ''), inplace=True)

                best_params = df_hp.loc[df_hp['value'].idxmax()].to_dict()
                
                for k, v in list(best_params.items()):
                    if isinstance(v, str) and v.lower() in ('true','false'):
                        best_params[k] = (v.lower() == 'true')
                    else:
                        try:
                            if isinstance(v, str) and v.replace('.','',1).isdigit():
                                best_params[k] = float(v) if ('.' in v) else int(v)
                        except Exception:
                            pass
            else:
                raise ValueError("Selección fuera de rango.")

        except (ValueError, IndexError) as e:
            logger.error(f"Selección inválida: {e}. Abortando entrenamiento.")
            return

    if not best_params:
        logger.error("No se pudieron obtener los hiperparámetros. Abortando.")
        return

    if smote_num_neighbors is None:
        smote_num_neighbors = best_params.get("smote_num_neighbors")
    if smote_num_neighbors is None:
        smote_num_neighbors = best_params.get("num_neighbors")
    if smote_num_neighbors is None:
        smote_num_neighbors = EMB_NUM_NEIGHBORS
    if isinstance(smote_num_neighbors, (int, float)):
        layers = int(best_params.get("num_layers", 2))
        smote_num_neighbors = [int(smote_num_neighbors)] * max(layers, 1)

    if early_stop is not None:
        best_params["early_stop"] = bool(early_stop)
    if early_stop_patience is not None:
        best_params["early_stop_patience"] = int(early_stop_patience)
    if early_stop_min_delta is not None:
        best_params["early_stop_min_delta"] = float(early_stop_min_delta)

    data.edge_attr_dict = { et: getattr(data[et], 'edge_attr', None) for et in data.edge_types }

    # 3) SummaryWriter + hparams
    writer = SummaryWriter(log_dir=os.path.join(RESULTADOS_DIR, "runs_attention"), flush_secs=30)
    hparam_payload = {}
    for key, value in best_params.items():
        if isinstance(value, (list, tuple, dict)):
            try:
                hparam_payload[key] = json.dumps(value)
            except Exception:
                hparam_payload[key] = str(value)
        else:
            hparam_payload[key] = value
    writer.add_hparams(hparam_payload, {'hparam/val_f1': 0.0})

    # 4) Modelo y optimizador
    num_classes = len(torch.unique(data['pm'].y))
    
    edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')

    in_channels = data['pm'].x.shape[1]
    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=int(best_params['hidden_channels']),
        out_channels=num_classes,
        num_heads=int(best_params['num_heads']),
        dropout=float(best_params['dropout']),
        edge_feature_dim=edge_feature_dim,
        num_layers=int(best_params.get('num_layers', 2)),
        aggr1=best_params.get('aggr1', 'sum'),
        aggr2=best_params.get('aggr2', 'sum'),
        use_checkpointing=bool(best_params.get('use_checkpointing', False)), 
    ).to(device)

    sequence_index = loaded_obj.get('sequence_index')
    temporal_module = None
    if sequence_index and getattr(sequence_index, "sequence_rows", None) is not None:
        if sequence_index.sequence_rows.size:
            seq_rows_tensor = torch.as_tensor(sequence_index.sequence_rows, dtype=torch.long)
            target_rows_tensor = torch.as_tensor(sequence_index.target_rows, dtype=torch.long)
            temporal_module = TemporalAggregator(
                sequence_rows=seq_rows_tensor,
                target_rows=target_rows_tensor,
                num_nodes=data['pm'].num_nodes,
                embedding_dim=int(best_params['hidden_channels']) * int(best_params['num_heads']),
                sequence_length=sequence_index.sequence_rows.shape[1],
                hidden_dim=int(best_params['hidden_channels']) * int(best_params['num_heads']),
                num_classes=num_classes,
            ).to(device)
            temporal_module.reset_cache()
            temporal_module.train()
            model.temporal_head = temporal_module
    elif hasattr(model, 'temporal_head'):
        delattr(model, 'temporal_head')

    # Rutas y directorios por modelo (usar mismo ID para z2x + modelo)
    tag_suffix = _model_tag_suffix(use_graphsmote, loaded_obj)
    ts_stamp_save = datetime.now().strftime('%Y%m%d_%H%M%S')
    graph_hash = _calculate_graph_hash(loaded_obj.get('filename'))
    hash_tag8 = f"_{graph_hash[:8]}" if graph_hash else ""
    
    if use_graphsmote:
        # User requested specific naming for GraphSMOTE models
        # We might still want ImGAGN tags if present, but the primary prefix changes.
        # Let's keep tag_suffix just in case it has other info, or strip _GraphSMOTE from it explicitly?
        # tag_suffix string likely contains "_GraphSMOTE" already.
        # User pattern: GraphSMOTE_embeddings_model_*.pt
        # We will use that as base.
        base_prefix = "GraphSMOTE_embeddings_model"
        # Clean tag_suffix to avoid double "GraphSMOTE" if we want, or just append. 
        # Simpler: just use the requested prefix + timestamp/hash.
        best_model_path = os.path.join(RESULTADOS_DIR, f"{base_prefix}.pt")
        best_model_path_unique = os.path.join(
            RESULTADOS_DIR,
            f"{base_prefix}_{ts_stamp_save}{hash_tag8}.pt",
        )
    else:
        best_model_path = os.path.join(RESULTADOS_DIR, f"gat_model_BEST{tag_suffix}.pt")
        best_model_path_unique = os.path.join(
            RESULTADOS_DIR,
            f"gat_model_BEST{tag_suffix}_{ts_stamp_save}{hash_tag8}.pt",
        )
    z2x_run_dir = os.path.join(
        RESULTADOS_DIR,
        "z2x_decoders",
        os.path.splitext(os.path.basename(best_model_path_unique))[0],
    )

    # --- Pre-entrenamiento de decodificadores si se usa GraphSMOTE ---
    z2x_decoders = None
    if use_graphsmote:
        logger.info("Entrenando decodificadores z->x para la aumentación...")
        model.use_checkpointing = False
        z2x_decoders = train_z2x_decoders(
            model,
            data,
            device=device,
            epochs=DECODER_EPOCHS,
            num_neighbors=smote_num_neighbors,
            save_dir=z2x_run_dir,
        )
        model.use_checkpointing = True

    # Edge generator
    z_dim_dict = {ntype: int(best_params['hidden_channels']) * int(best_params['num_heads']) for ntype in data.node_types}
    edge_gen = RelEdgeGen(z_dim_dict, data.edge_types).to(device) if use_graphsmote else None

    optimizer_name = str(best_params.get('optimizer', 'Adam'))
    optimizer_cls = getattr(torch.optim, optimizer_name, torch.optim.Adam)
    optimizer = optimizer_cls(
        model.parameters() if edge_gen is None else list(model.parameters()) + list(edge_gen.parameters()),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    
    # 6) Criterio
    # Detect if training data is ImGAGN-augmented to disable class weighting
    im_gagn_augmented = False
    try:
        im_gagn_augmented = ('imgagn_best_params' in loaded_obj) or \
                             (hasattr(data['pm'], 'is_synthetic') and data['pm'].is_synthetic.sum().item() > 0)
    except Exception:
        im_gagn_augmented = False

    loss_type = best_params.get('loss_type', 'CrossEntropy')
    if loss_type == 'FocalLoss':
        alpha_val = [1 - best_params.get('focal_alpha', 0.25), best_params.get('focal_alpha', 0.25)]
        if use_graphsmote:
            alpha_val = None
            logger.info("GraphSMOTE is active, so alpha weighting for FocalLoss is disabled.")
        else:
            logger.info(f"FocalLoss alpha weighting enabled. Alpha: {alpha_val}")

        criterion = FocalLoss(
            gamma=best_params.get('focal_gamma', 2.0),
            alpha=alpha_val
        )
    else:  # CrossEntropy
        weight = None
        if not use_graphsmote and not im_gagn_augmented:
            y_train = data['pm'].y[data['pm'].train_mask]
            counts = torch.bincount(y_train)
            if counts.numel() > 1:
                weight = (1.0 / counts.float())
                logger.info(f"Class weighting enabled for CrossEntropyLoss. Weights: {weight.cpu().numpy()}")
        else:
            reason = "GraphSMOTE active" if use_graphsmote else "ImGAGN-augmented graph detected"
            logger.info(f"Class weighting for CrossEntropyLoss disabled ({reason}).")

        criterion = torch.nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)

    # 7) Pre-entrenamiento del generador de aristas (si aplica)
    if use_graphsmote and edge_gen is not None and PRETRAIN_EDGE_EPOCHS > 0:
        logger.info(f"Pre-entrenando el generador de aristas por {PRETRAIN_EDGE_EPOCHS} épocas...")
        pretrain_edge_generator(
            encoder=model,
            edge_gen=edge_gen,
            data=data,
            device=device,
            pretrain_epochs=PRETRAIN_EDGE_EPOCHS,
            optimizer=torch.optim.Adam(edge_gen.parameters(), lr=1e-3),
            criterion=torch.nn.BCEWithLogitsLoss(),
            writer=writer
        )

    # 8) Selección de modo GraphSMOTE y preparación de grafos
    loader_num_neighbors = _resolve_num_neighbors(best_params.get('num_neighbors'), NUM_NEIGHBORS, data.edge_types)
    batch_size_hp = int(_safe_cast(best_params.get('batch_size'), int, BATCH_SIZE))
    target_pos_ratio_override = _safe_cast(best_params.get('target_pos_ratio'), float, TARGET_POS_RATIO)
    smote_every_override = max(1, _safe_cast(best_params.get('smote_every_n_epochs'), int, SMOTE_EVERY_N_EPOCHS))

    if use_graphsmote and GRAPHSMOTE_MODE == 'offline':
        logger.info("Modo Offline: Aumentando el grafo una sola vez.")
        model.use_checkpointing = False
        aug_data, _ = augment_graph_offline_once(
            model, data, device,
            z2x_decoders=z2x_decoders,
            target_pos_ratio=target_pos_ratio_override,
            k=GRAPHSMOTE_K,
            edge_gen=edge_gen,
            save_path=SAVE_AUG_GRAPH_PATH,
            seed=GS_SEED,
            num_neighbors=smote_num_neighbors,
        )
        model.use_checkpointing = True
        train_graph = aug_data.to(device)
        base_graph = data  # Para evaluación
        if writer and hasattr(train_graph['pm'], 'is_synthetic'):
            try:
                synth_count = int(train_graph['pm'].is_synthetic.sum().item())
                writer.add_scalar('GraphSMOTE/SyntheticNodes', synth_count, 0)
            except Exception:
                pass
    elif use_graphsmote and GRAPHSMOTE_MODE == 'online':
        logger.info("Modo Online: El grafo se refrescará periódicamente.")
        base_graph = data
        # La aumentación inicial se hace dentro del loop
        train_graph = base_graph 
    else:
        train_graph = data.to(device)
        base_graph = data

    # 9) Loader y Scheduler
    def create_loader(graph_to_load, use_undersampling=False, pos_to_neg_ratio=3.0,
                      strategy='random', model=None, device=None, topk_hard=None):
        """
        Si use_undersampling=True → usa semillas por época (positivos + subset de negativos).
        Si False → usa todos los train como semillas (comportamiento actual).
        """
        if use_undersampling:
            seeds = make_epoch_seeds(
                graph_to_load, node_type='pm',
                pos_to_neg_ratio=pos_to_neg_ratio,
                strategy=strategy, model=model, device=device, topk_hard=topk_hard
            )
            input_nodes = ('pm', seeds)
        else:
            train_idx = graph_to_load['pm'].train_mask.nonzero(as_tuple=False).view(-1)
            input_nodes = ('pm', train_idx)

        try:
            num_neighbors_cfg = loader_num_neighbors
        except Exception:
            num_neighbors_cfg = NUM_NEIGHBORS

        return NeighborLoader(
            graph_to_load.cpu(),
            input_nodes=input_nodes,
            num_neighbors=num_neighbors_cfg,
            batch_size=batch_size_hp,
            shuffle=True
        )

    raw_undersample = best_params.get('undersample', None)
    auto_enable_hard = (raw_undersample is None) or (isinstance(raw_undersample, str) and raw_undersample.lower() == 'auto')
    use_undersampling = bool(raw_undersample) if not auto_enable_hard else False
    pos_to_neg_ratio = float(_safe_cast(best_params.get('pos_to_neg_ratio'), float, 3.0))
    undersampling_strategy = str(best_params.get('undersampling_strategy', 'random'))
    topk_hard = best_params.get('topk_hard', None)
    if topk_hard is not None:
        topk_hard = int(topk_hard)

    hard_sampling_warmup = max(0, int(_safe_cast(best_params.get('hard_sampling_warmup'), int, 1)))

    if auto_enable_hard:
        try:
            train_mask = data['pm'].train_mask
            if train_mask.sum() > 0:
                pos_ratio_train = float((data['pm'].y[train_mask] == 1).float().mean().item())
            else:
                pos_ratio_train = 0.0
        except Exception:
            pos_ratio_train = 0.0
        if pos_ratio_train < 0.2:
            use_undersampling = True
            undersampling_strategy = 'hard'
            best_params['undersample'] = True
            best_params['undersampling_strategy'] = 'hard'
            if topk_hard is None:
                try:
                    pos_count = int((data['pm'].y[train_mask] == 1).sum().item())
                    topk_hard = max(int(pos_count * pos_to_neg_ratio), 128)
                except Exception:
                    topk_hard = 512

    def rebuild_train_loader(graph, strategy_override=None):
        effective_strategy = strategy_override or undersampling_strategy
        return create_loader(
            graph,
            use_undersampling=use_undersampling,
            pos_to_neg_ratio=pos_to_neg_ratio,
            strategy=effective_strategy,
            model=model if effective_strategy == 'hard' else None,
            device=device if effective_strategy == 'hard' else None,
            topk_hard=topk_hard
        )

    train_loader = rebuild_train_loader(train_graph)
    
    max_epochs = int(max_epochs) if max_epochs is not None else int(MAX_EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(best_params['lr']),
        steps_per_epoch=len(train_loader),
        epochs=max_epochs,
        cycle_momentum=isinstance(optimizer, (torch.optim.AdamW, torch.optim.NAdam))
    )

        # 10) Loop de entrenamiento
    best_val_f1 = 0.0
    best_val_auprc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop_enabled = bool(best_params.get("early_stop", True))
    patience = int(
        best_params.get("early_stop_patience", EARLY_STOPPING_PATIENCE)
    )
    min_delta = float(
        best_params.get("early_stop_min_delta", EARLY_STOPPING_MIN_DELTA)
    )
    if progress_callback is not None:
        try:
            progress_callback(
                epoch=0,
                total=max_epochs,
                val_f1=None,
                best_val_f1=best_val_f1,
                patience=patience,
                patience_counter=patience_counter,
            )
        except Exception:
            pass
    # best_model_path y best_model_path_unique ya definidos arriba
    
    # Automatic Mixed Precision (AMP)
    use_amp_param = best_params.get('use_amp', True)
    if use_amp_param and device.type != 'cuda':
        logger.warning("AMP con GradScaler solo es compatible con CUDA. Se desactivará AMP.")
        use_amp = False
    else:
        use_amp = use_amp_param

    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Automatic Mixed Precision (AMP) habilitado con GradScaler.")

    # Aviso único si se activó lambda_l2_att pero XAI está desactivado
    try:
        lambda_l2_att_global = float(best_params.get('lambda_l2_att', 0.0))
    except Exception:
        lambda_l2_att_global = 0.0
    suppress_missing_att_warning = False
    if lambda_l2_att_global > 0 and not bool(XAI):
        logger.info("XAI=0 (desactivado): no se capturarán atenciones. Se omitirá la regularización L2 de atenciones (lambda_l2_att>0) y se silenciarán avisos por época.")
        suppress_missing_att_warning = True

    for epoch in range(1, max_epochs + 1):
        if use_undersampling:
            strategy_now = undersampling_strategy
            if undersampling_strategy == 'hard' and epoch <= hard_sampling_warmup:
                strategy_now = 'random'
            train_loader = rebuild_train_loader(train_graph, strategy_override=strategy_now)
        # Refresco para modo ONLINE
        if use_graphsmote and GRAPHSMOTE_MODE == 'online' and epoch % smote_every_override == 0:
            logger.info(f"Epoch {epoch}: Refrescando nodos sintéticos (Online Mode)...")
            model.use_checkpointing = False
            aug_data, _ = refresh_synthetics_online(
                model, base_graph, device,
                target_pos_ratio=target_pos_ratio_override,
                k=GRAPHSMOTE_K,
                z2x_decoders=z2x_decoders,
                edge_gen=edge_gen,
                seed=GS_SEED + epoch,
                num_neighbors=smote_num_neighbors,
            )
            model.use_checkpointing = True
            train_graph = aug_data.to(device)
            if writer and hasattr(train_graph['pm'], 'is_synthetic'):
                try:
                    synth_count = int(train_graph['pm'].is_synthetic.sum().item())
                    writer.add_scalar('GraphSMOTE/SyntheticNodes', synth_count, epoch)
                except Exception:
                    pass
            strategy_now = undersampling_strategy
            if undersampling_strategy == 'hard' and epoch <= hard_sampling_warmup:
                strategy_now = 'random'
            train_loader = rebuild_train_loader(train_graph, strategy_override=strategy_now)  # Recargar el loader con configuración original

        # Rutina de entrenamiento
        model.train()
        
        # Agenda del regularizador lambda_H
        mode = best_params.get('lambda_H_mode', 'fixed')
        if mode == 'fixed':
            current_lambda_H = float(best_params.get('lambda_H_fixed', 0.0))
        else:
            lam0 = float(best_params.get('initial_lambda_H', 0.0))
            lam1 = float(best_params.get('final_lambda_H', 0.0))
            if mode == 'cosine':
                t = (epoch-1) / max(1, max_epochs-1)
                current_lambda_H = lam0 + 0.5*(lam1-lam0)*(1 - math.cos(math.pi*t))
            else:  # linear
                current_lambda_H = lam0 + (lam1 - lam0) * (epoch-1) / max(1, max_epochs-1)
        
        lambda_edge = float(best_params.get('lambda_edge', 1e-6))
        lambda_l2_att = float(best_params.get('lambda_l2_att', 0.0))
        
        loss, _, _, _ = train_minibatch(model, train_loader, optimizer, criterion,
                                  grad_clip_value=float(best_params.get('grad_clip', 1.0)),
                                  device=device, use_amp=use_amp, scaler=scaler,
                                  scheduler=scheduler, writer=writer, epoch=epoch,
                                  lambda_H=current_lambda_H, node_type='pm',
                                  edge_gen=edge_gen, lambda_edge=lambda_edge,
                                  lambda_l2_att=lambda_l2_att,
                                  suppress_missing_att_warning=suppress_missing_att_warning)
        writer.add_scalar("Loss/train", loss, epoch)

        # Validación (siempre sobre el grafo original)
        model.use_checkpointing = False
        val_res = test(model, base_graph, node_type='pm', batch_size=batch_size_hp)
        if temporal_module is not None:
            temporal_module.train()
        model.use_checkpointing = True
        
        val_f1 = 0.0
        if val_res and 'val_mask' in val_res:
            y_true_val = val_res['val_mask']['true'].numpy().ravel()
            y_prob1_val = val_res['val_mask']['probs'][:, 1].numpy().ravel()
            
            if len(np.unique(y_true_val)) > 1:
                # Usar beta=1.0 para F1-score, consistente con la métrica de early stopping
                tau, P, R, f1_score = pick_tau_fbeta(y_true_val, y_prob1_val, beta=1.0)
                val_f1 = f1_score
                if writer:
                    try:
                        writer.add_scalar('Calibration/Val_tau', tau, epoch)
                    except Exception:
                        pass
            else:
                # Fallback si solo hay una clase en el conjunto de validación
                report = val_res['val_mask']['report']
                val_f1 = report.get('Accidente (1)', {}).get('f1-score', 0.0)

            if writer:
                try:
                    val_report = val_res['val_mask']['report']
                    writer.add_scalar('Metrics/Val_F1_candidate', val_f1, epoch)
                    writer.add_scalar('Metrics/Val_F1_positive', val_report['Accidente (1)']['f1-score'], epoch)
                    writer.add_scalar('Metrics/Val_F1_macro', val_report['macro avg']['f1-score'], epoch)
                    writer.add_scalar('Metrics/Val_Precision_positive', val_report['Accidente (1)']['precision'], epoch)
                    writer.add_scalar('Metrics/Val_Recall_positive', val_report['Accidente (1)']['recall'], epoch)
                    writer.add_scalar('Metrics/Val_AUPRC', val_res['val_mask'].get('auprc') or 0.0, epoch)
                    writer.add_scalar('Metrics/Val_AUC', val_res['val_mask'].get('auc') or 0.0, epoch)
                except Exception:
                    pass

        if (val_f1 - best_val_f1) > min_delta:
            best_val_f1 = val_f1
            best_val_auprc = val_res['val_mask'].get('auprc', 0.0) if val_res else 0.0
            best_epoch = epoch
            patience_counter = 0
            # Guardar modelo: copia única y alias estable
            try:
                # Copia única (con timestamp/hash)
                torch.save(model.state_dict(), best_model_path_unique)
                if SAVE_GAT_ALIASES:
                    # Alias actualizado (último mejor para variante)
                    torch.save(model.state_dict(), best_model_path)
                    # Alias global (independiente de variante)
                    best_model_global = os.path.join(RESULTADOS_DIR, "gat_model_BEST.pt")
                    torch.save(model.state_dict(), best_model_global)
            except Exception as e:
                logger.error(f"No se pudo guardar el modelo en disco: {e}")
            logger.info(
                f"Epoch {epoch:03d}: New best F1 Accidente (1) on validation: {val_f1:.4f}.\n"
                f"  → Guardado: {os.path.basename(best_model_path_unique)} (alias variante: {os.path.basename(best_model_path)}, alias global: gat_model_BEST.pt)"
            )
            # Guardar sidecar con hiperparámetros y metadatos del entrenamiento (para ambas rutas)
            try:
                meta = dict(best_params)
                meta.update({
                    'best_val_f1': float(best_val_f1),
                    'best_val_auprc': float(best_val_auprc),
                    'best_epoch': int(best_epoch),
                    'use_graphsmote': bool(use_graphsmote),
                    'graphsmote_mode': str(GRAPHSMOTE_MODE),
                    'graph_hash': graph_hash,
                    'git_commit': _get_repo_version(),
                    'purpose': purpose or loaded_obj.get('purpose', 'General') if isinstance(loaded_obj, dict) else 'General',
                    'target_pos_ratio_used': float(target_pos_ratio_override),
                    'smote_every_n_epochs_used': int(smote_every_override),
                    'num_neighbors_effective': loader_num_neighbors,
                    'out_channels': int(num_classes),
                })
                meta = _json_safe(meta)
                # Para la copia única
                meta_path_unique = os.path.splitext(best_model_path_unique)[0] + "_hparams.json"
                import json as _json
                with open(meta_path_unique, 'w') as f:
                    _json.dump(meta, f)
                if SAVE_GAT_ALIASES:
                    # Para el alias por variante
                    try:
                        meta_path = os.path.splitext(best_model_path)[0] + "_hparams.json"
                        with open(meta_path, 'w') as f:
                            _json.dump(meta, f)
                    except Exception:
                        pass
                    # Sidecar para alias global
                    try:
                        meta_path_global = os.path.join(RESULTADOS_DIR, "gat_model_BEST_hparams.json")
                        with open(meta_path_global, 'w') as f:
                            _json.dump(meta, f)
                    except Exception:
                        pass
            except Exception as _e:
                logger.warning(f"No se pudo guardar hparams JSON: {_e}")
        else:
            patience_counter += 1

        if progress_callback is not None:
            try:
                progress_callback(
                    epoch=epoch,
                    total=max_epochs,
                    val_f1=val_f1,
                    best_val_f1=best_val_f1,
                    patience=patience,
                    patience_counter=patience_counter,
                )
            except Exception:
                pass

        if early_stop_enabled and patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d, min_delta=%g).",
                epoch,
                patience,
                min_delta,
            )
            break
    
    writer.close()
    logger.info(f"Training finished. Best F1: {best_val_f1:.4f} at epoch {best_epoch}.")

def pick_tau_fbeta(y_true, y_prob, beta=0.5):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec_, rec_, thr_ = prec[1:], rec[1:], thr
    beta2 = beta**2
    fbeta = (1+beta2) * prec_ * rec_ / np.clip(beta2*prec_ + rec_, 1e-12, None)
    i = int(np.nanargmax(fbeta))
    return float(thr_[i]), float(prec_[i]), float(rec_[i]), float(fbeta[i])

def _platt_scale_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, Optional[object]]:
    """Fit Platt scaling (logistic regression) and return calibrated probs + model."""
    if not AUTOCALIBRATE_PROBS:
        return y_prob, None
    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
        model.fit(y_prob.reshape(-1, 1), y_true.reshape(-1, 1).ravel())
        calibrated = model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return calibrated, model
    except Exception as exc:
        logger.debug(f"Platt scaling skipped (sklearn unavailable or failed): {exc}")
        return y_prob, None

def _apply_platt_model(y_prob: np.ndarray, model) -> np.ndarray:
    if model is None:
        return y_prob
    try:
        return model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    except Exception as exc:
        logger.debug(f"Error applying Platt scaler: {exc}")
        return y_prob

def objective(trial, device, use_graphsmote_search=False, optimizer_overrides=None):
    global data
    """
    Objetivo Optuna: entrena por pocas épocas, evalúa métricas robustas
    con umbral óptimo en validación y devuelve el mejor F0.5-score.
    """
    try:
        # --- Semillas para reproducibilidad del trial ---
        trial_seed = SEED + trial.number
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)

        # --- Espacio de búsqueda de hiperparámetros ---

        # Arquitectura
        hidden_channels = trial.suggest_int('hidden_channels', 32, 128, step=32)
        num_heads = trial.suggest_int('num_heads', 2, 8, step=2)
        dropout = trial.suggest_float('dropout', 0.05, 0.6)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        aggr1 = trial.suggest_categorical('aggr1', ['sum', 'mean', 'max'])
        aggr2 = trial.suggest_categorical('aggr2', ['sum', 'mean', 'max'])
        use_checkpointing_flag = trial.suggest_categorical('use_checkpointing', [False, True])

        neighbor_candidates = {
            'compact': [15, 10],
            'focused': [10, 5],
            'broad': [25, 15, 10],
            'wide': [30, 20],
        }
        neighbor_choice = trial.suggest_categorical('num_neighbors_choice', list(neighbor_candidates.keys()))
        neighbor_profile = neighbor_candidates[neighbor_choice]

        # Optimizador
        overrides = optimizer_overrides or {}
        
        lr_override = overrides.get('lr')
        if lr_override is not None:
             lr = trial.suggest_float('lr', float(lr_override), float(lr_override))
        else:
             lr = trial.suggest_float('lr', 5e-5, 1e-2, log=True)

        opt_name_override = overrides.get('optimizer')
        if opt_name_override:
            optimizer_name = trial.suggest_categorical('optimizer', [str(opt_name_override)])
        else:
            optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RAdam'])

        wd_override = overrides.get('weight_decay')
        if wd_override is not None:
            weight_decay = trial.suggest_float('weight_decay', float(wd_override), float(wd_override))
        else:
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 5e-3, log=True)
            
        grad_clip_override = overrides.get('grad_clip') # If we want to add this to search space in future
        # Currently grad_clip is not in default search space, but let's add it if requested?
        # The prompt only mentioned incorporating parameters. I'll stick to what is already searched + make them fixable.
        # Wait, usually grad_clip is NOT searched in objective? Let's check.
        # It seems grad_clip is pulled from best_params in training loop but usually default 1.0. 
        # I will leave grad_clip out of SEARCH unless valid demand, but user asked for "optimizer parameters".
        # I will inject it as a fixed param if provided, but I need to ensure it's returned.
        if grad_clip_override is not None:
            trial.set_user_attr('grad_clip', float(grad_clip_override))

        # Regularización
        lambda_l2_att = trial.suggest_float('lambda_l2_att', 1e-4, 1e-1, log=True)
        lambda_H_mode = trial.suggest_categorical('lambda_H_mode', ['fixed', 'dynamic'])
        if lambda_H_mode == 'fixed':
            initial_lambda_H = final_lambda_H = trial.suggest_float('lambda_H_fixed', 1e-4, 5e-2, log=True)
        else:
            initial_lambda_H = trial.suggest_float('initial_lambda_H', 1e-4, 1e-2, log=True)
            final_lambda_H = trial.suggest_float('final_lambda_H', 1e-2, 5e-2, log=True)

        # --- Lógica condicional para manejo de desbalance ---
        # En ambos modos permitimos explorar la función de pérdida; con GraphSMOTE
        # se desactiva alpha (ponderación de clases) pero se puede ajustar gamma.
        if use_graphsmote_search:
            loss_type = trial.suggest_categorical("loss_type", ["CrossEntropy", "FocalLoss"])
            smote_k = trial.suggest_int("smote_k", 3, 7, step=2)
            target_pos_ratio = trial.suggest_categorical("target_pos_ratio", [0.25, 0.35, 0.5])
            smote_every_n_epochs = trial.suggest_categorical("smote_every_n_epochs", [2, 4, 6])
            lambda_edge = trial.suggest_float('lambda_edge', 1e-7, 1e-4, log=True)
        else:
            loss_type = trial.suggest_categorical("loss_type", ["CrossEntropy", "FocalLoss"])
            lambda_edge = 0.0  # No aplica
            target_pos_ratio = TARGET_POS_RATIO
            smote_every_n_epochs = SMOTE_EVERY_N_EPOCHS

        batch_size_candidate = trial.suggest_int('batch_size', 512, 4096, step=512)
        trial.set_user_attr('num_neighbors', neighbor_profile)
        trial.set_user_attr('use_checkpointing', use_checkpointing_flag)

        # --- Construcción del Modelo ---
        num_classes = 2
        edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')
        in_channels = data['pm'].x.shape[1]

        model = HeteroGAT(
            in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes,
            num_heads=num_heads, dropout=dropout, edge_feature_dim=edge_feature_dim,
            num_layers=num_layers, aggr1=aggr1, aggr2=aggr2, use_checkpointing=use_checkpointing_flag
        ).to(device)

        temporal_module = None
        if sequence_index_global and getattr(sequence_index_global, "sequence_rows", None) is not None:
            if sequence_index_global.sequence_rows.size:
                seq_rows_tensor = torch.as_tensor(sequence_index_global.sequence_rows, dtype=torch.long)
                target_rows_tensor = torch.as_tensor(sequence_index_global.target_rows, dtype=torch.long)
                temporal_module = TemporalAggregator(
                    sequence_rows=seq_rows_tensor,
                    target_rows=target_rows_tensor,
                    num_nodes=data['pm'].num_nodes,
                    embedding_dim=hidden_channels * num_heads,
                    sequence_length=sequence_index_global.sequence_rows.shape[1],
                    hidden_dim=hidden_channels * num_heads,
                    num_classes=num_classes,
                ).to(device)
                temporal_module.reset_cache()
                temporal_module.train()
                model.temporal_head = temporal_module
        elif hasattr(model, 'temporal_head'):
            delattr(model, 'temporal_head')

        edge_gen = RelEdgeGen({ntype: hidden_channels * num_heads for ntype in data.node_types}, data.edge_types).to(device) if use_graphsmote_search else None
        
        optimizer = getattr(torch.optim, optimizer_name)(
            list(model.parameters()) + (list(edge_gen.parameters()) if edge_gen else []),
            lr=lr, weight_decay=weight_decay
        )

        # --- Criterio de Pérdida (condicional) ---
        criterion = None
        if loss_type == 'FocalLoss':
            alpha_val = None
            if not use_graphsmote_search:
                focal_alpha = trial.suggest_float('focal_alpha', 0.5, 0.95) # Alpha for the positive class
                alpha_val = [1 - focal_alpha, focal_alpha]
            criterion = FocalLoss(gamma=trial.suggest_float('focal_gamma', 1.0, 3.0), alpha=alpha_val)
        else: # CrossEntropy
            weight = None
            if not use_graphsmote_search:
                y_train = data['pm'].y[data['pm'].train_mask]
                counts = torch.bincount(y_train)
                if counts.numel() > 1:
                    weight = (1.0 / counts.float()).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

        # --- Preparación para el loop de entrenamiento ---
        base_graph = data
        train_graph = base_graph
        z2x_decoders = train_z2x_decoders(model, base_graph, device=device, epochs=DECODER_EPOCHS) if use_graphsmote_search else None
        
        num_neighbors_dict = {edge_type: neighbor_profile for edge_type in train_graph.edge_types}
        train_loader = NeighborLoader(
            train_graph.cpu(),
            input_nodes=('pm', train_graph['pm'].train_mask),
            num_neighbors=num_neighbors_dict,
            batch_size=batch_size_candidate,
            shuffle=True
        )
        
        best_f05 = -1.0
        best_tau = 0.5
        best_epoch = 0

        # --- Loop de Entrenamiento y Validación ---
        for epoch in range(1, NUM_EPOCHS_OPTUNA + 1):
            # ... (lógica de refresco de SMOTE si es online)
            if use_graphsmote_search and epoch % smote_every_n_epochs == 0:
                aug_data, _ = refresh_synthetics_online(
                    model, base_graph, device, target_pos_ratio=target_pos_ratio, k=smote_k,
                    z2x_decoders=z2x_decoders, edge_gen=edge_gen, seed=trial_seed + epoch
                )
                train_graph = aug_data.to(device)
                # Recargar loader con el grafo aumentado
                train_loader = NeighborLoader(
                    train_graph.cpu(),
                    input_nodes=('pm', train_graph['pm'].train_mask),
                    num_neighbors={edge_type: neighbor_profile for edge_type in train_graph.edge_types},
                    batch_size=batch_size_candidate,
                    shuffle=True
                )

            # Entrenamiento
            model.train()
            current_lambda_H = initial_lambda_H + (final_lambda_H - initial_lambda_H) * (epoch - 1) / max(NUM_EPOCHS_OPTUNA - 1, 1)
            train_minibatch(
                model, train_loader, optimizer, criterion, grad_clip_value=float(overrides.get('grad_clip', 1.0)), device=device,
                use_amp=False, scaler=None, scheduler=None, writer=None, epoch=epoch,
                lambda_H=current_lambda_H, node_type='pm', edge_gen=edge_gen, lambda_edge=lambda_edge,
                lambda_l2_att=lambda_l2_att
            )

            # Validación periódica
            if epoch % 2 == 0:
                val_results = test(model, base_graph, node_type='pm', batch_size=batch_size_candidate)
                if temporal_module is not None:
                    temporal_module.train()
                if not val_results or 'val_mask' not in val_results:
                    continue

                y_true_val = val_results['val_mask']['true'].numpy().ravel()
                y_prob1_val = val_results['val_mask']['probs'][:, 1].numpy().ravel()

                if len(np.unique(y_true_val)) < 2:
                    continue

                auprc = average_precision_score(y_true_val, y_prob1_val)
                tau, P, R, f05 = pick_tau_fbeta(y_true_val, y_prob1_val, beta=0.5)
                
                trial.report(auprc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if f05 > best_f05:
                    best_f05 = f05
                    best_tau = tau
                    best_epoch = epoch
                
                # Sanity Check
                pos_rate = (y_prob1_val >= tau).mean()
                if pos_rate > 0.6: # Si más del 60% es predicho como positivo, penalizar.
                    return best_f05 * 0.7 

        trial.set_user_attr("best_f0.5", best_f05)
        trial.set_user_attr("best_tau", best_tau)
        trial.set_user_attr("best_epoch", best_epoch)
        
        return best_f05

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        # Limpieza
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0.0

def search_hyperparameters(loaded_obj, use_graphsmote_search=None, optimizer_overrides=None):
    global data, sequence_index_global, sequence_config_global
    graph_filename = loaded_obj.get('filename')
    # Si la elección no viene desde el menú de entrenamiento, preguntar al usuario.
    if use_graphsmote_search is None:
        use_graphsmote_input = input("¿Desea que la búsqueda de hiperparámetros incluya GraphSMOTE? (s/n): ").strip().lower()
        use_graphsmote_search = use_graphsmote_input in ('s', 'si', 'y', 'yes')

    if use_graphsmote_search:
        logger.info("La búsqueda de hiperparámetros incluirá opciones de GraphSMOTE.")
    else:
        logger.info("La búsqueda de hiperparámetros NO incluirá opciones de GraphSMOTE.")

    # --- Carga inteligente de HPO ---
    graph_hash = _calculate_graph_hash(graph_filename)

    if graph_hash:
        # Buscar todos los HPO del mismo grafo y luego filtrar por variante
        pattern_any = os.path.join(RESULTADOS_DIR, f"optuna_hyperparams_*{graph_hash[:16]}*.csv")
        candidates = sorted(glob.glob(pattern_any), key=os.path.getmtime, reverse=True)
        if candidates:
            # Filtrar por SMOTE y por ImGAGN si aplica; preferir coincidencias exactas de variante
            want_smote = bool(use_graphsmote_search)
            want_imgagn = _has_imgagn(loaded_obj)
            preferred = []
            fallback = []
            for p in candidates:
                name = os.path.basename(p)
                has_smote = ("_GraphSMOTE" in name)
                has_imgagn = ("_ImGAGN" in name)
                has_base = ("_Base" in name)
                # Coincidencia exacta de ambos flags a la lista preferida
                if (has_smote == want_smote) and (has_imgagn == want_imgagn):
                    preferred.append(p)
                # En base sin etiqueta, aceptamos como fallback si coincide smote
                elif (has_smote == want_smote) and (not want_imgagn and not has_imgagn):
                    fallback.append(p)
                # También aceptar archivos antiguos sin _Base/_GraphSMOTE si smote=False
                elif (not want_smote) and ("_GraphSMOTE" not in name):
                    fallback.append(p)
            pick_list = preferred if preferred else fallback
            if pick_list:
                latest_hpo_file = pick_list[0]
                logger.info("Se encontraron archivos de HPO previos para este grafo y variante.")
                logger.info(f"Cargando el archivo más reciente: {os.path.basename(latest_hpo_file)}")
                try:
                    df_hp = pd.read_csv(latest_hpo_file)
                    best_params = df_hp.loc[0].to_dict()
                    reuse = input(f"¿Desea reutilizar estos hiperparámetros (F1: {best_params.get('value', 'N/A'):.4f})? (s/n, por defecto 's'): ").strip().lower()
                    if reuse in ('', 's', 'si', 'y', 'yes'):
                        logger.info("Reutilizando hiperparámetros existentes.")
                        return best_params
                    else:
                        logger.info("Descartando resultados previos. Iniciando nueva búsqueda.")
                except Exception as e:
                    logger.error(f"No se pudo cargar o procesar '{os.path.basename(latest_hpo_file)}': {e}. Iniciando nueva búsqueda.")

    # Dispositivo
    # Selection automática de dispositivo
    device = get_auto_device()
    logger.info(f"Usando dispositivo: {device}")

    data = loaded_obj['data'].to(device)
    sequence_index_global = loaded_obj.get('sequence_index')
    sequence_config_global = loaded_obj.get('sequence_config')

    # Sampler y Pruner
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True)
    pruner  = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=NUM_EPOCHS_OPTUNA, reduction_factor=3)

    # Crear estudio nuevo
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    logger.info(f"Optuna study creado: {study.study_name}")

    try:
        # Pasar el nombre de argumento correcto según la firma de objective()
        study.optimize(lambda tr: objective(tr, device, use_graphsmote_search=use_graphsmote_search, optimizer_overrides=optimizer_overrides), n_trials=N_TRIALS, show_progress_bar=True)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Limpieza de memoria de GPU finalizada.")

    # Mejor trial y guardado
    best = study.best_trial
    best_params = best.params.copy()
    best_params['value'] = best.value
    # Añadir los atributos de usuario al diccionario de parámetros
    for key, value in best.user_attrs.items():
        best_params[key] = value
    # Añadir los atributos de usuario al diccionario de parámetros
    for key, value in best.user_attrs.items():
        best_params[key] = value
    best_params['use_graphsmote'] = use_graphsmote_search
    best_params.setdefault('undersample', 'auto')
    try:
        best_params['use_imgagn_aug'] = bool(_has_imgagn(loaded_obj))
    except Exception:
        best_params['use_imgagn_aug'] = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Etiquetas de variante
    variant_tag = _variant_tags(use_graphsmote_search, loaded_obj)
    hash_tag = f"_{graph_hash[:16]}" if graph_hash else ""
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    best_params_path = os.path.join(RESULTADOS_DIR, f"optuna_hyperparams_{timestamp}{hash_tag}{variant_tag}.csv")
    pd.DataFrame([best_params]).to_csv(best_params_path, index=False)
    logger.info(f"Hiperparámetros top guardados en -> {os.path.basename(best_params_path)}")

    trials_df = study.trials_dataframe()
    trials_df['use_graphsmote'] = use_graphsmote_search
    trials_df['graph_hash'] = graph_hash
    full_study_path = os.path.join(RESULTADOS_DIR, f"optuna_full_study_{timestamp}{hash_tag}{variant_tag}.csv")
    trials_df.to_csv(full_study_path, index=False)
    logger.info(f"Estudio completo guardado en -> {os.path.basename(full_study_path)}")

    return best_params

def run_imgagn_hpo(loaded_obj):
    """
    Búsqueda de hiperparámetros dedicada a ImGAGN (sin reentrenar GAT).
    Optimiza la métrica proxy 'best_train_recall' reportada por ImGAGN.
    Guarda CSV con el mejor trial y el estudio completo en 'Resultados/'.
    """
    # Dispositivo
    # Selection automática de dispositivo
    device = get_auto_device()
    logger.info(f"Usando dispositivo: {device}")

    data = loaded_obj['data']
    node_type = 'pm'

    # Preparar etiquetas binarias 0/1 para el tipo de nodo objetivo
    y_orig = data[node_type].y.cpu()
    classes, counts = torch.unique(y_orig, return_counts=True)
    if classes.numel() == 2 and set(classes.tolist()) == {0, 1}:
        y_bin = y_orig.clone()
    else:
        # Mapear a 0/1 usando la clase minoritaria como 1
        min_idx = int(torch.argmin(counts))
        minority_label = int(classes[min_idx].item())
        y_bin = (y_orig == minority_label).long()
        logger.info(f"Mapeando etiquetas a binario: minoritaria={minority_label} -> 1")

    train_mask = data[node_type].train_mask.cpu().bool()
    if train_mask.sum().item() == 0:
        logger.error("La máscara de entrenamiento está vacía; no se puede ejecutar ImGAGN.")
        return

    graph_filename = loaded_obj.get('filename')
    graph_hash = _calculate_graph_hash(graph_filename)

    # Optuna: sampler y pruner razonables
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    logger.info(f"Optuna study (ImGAGN) creado: {study.study_name}")

    def objective(trial: optuna.Trial):
        try:
            cfg = ImGAGNConfig(
                # Generator
                dz=trial.suggest_int('dz', 64, 128, step=32),
                hidden_g=trial.suggest_int('hidden_g', 96, 256, step=32),
                n_hidden_g=trial.suggest_int('n_hidden_g', 1, 2),
                topk_links=trial.suggest_int('topk_links', 3, 10),
                # Discriminator/Encoder
                emb_dim=trial.suggest_int('emb_dim', 64, 192, step=64),
                hid_d=trial.suggest_int('hid_d', 64, 192, step=64),
                dropout=trial.suggest_float('dropout', 0.0, 0.3),
                # Training ratios/steps
                lambda1_ratio=trial.suggest_float('lambda1_ratio', 0.7, 1.2),
                d_steps=trial.suggest_int('d_steps', 10, 50, step=10),
                epochs=trial.suggest_int('epochs', 10, 50, step=5),
                # Optims
                lr_g=trial.suggest_float('lr_g', 1e-4, 5e-3, log=True),
                lr_d=trial.suggest_float('lr_d', 1e-4, 5e-3, log=True),
                wd=trial.suggest_float('wd', 1e-6, 5e-4, log=True),
                alpha_reg=trial.suggest_float('alpha_reg', 1e-6, 1e-3, log=True),
                beta_reg=trial.suggest_float('beta_reg', 1e-6, 1e-3, log=True),
                margin_mm=trial.suggest_float('margin_mm', 0.5, 2.0),
                grad_clip=2.0,
                device=str(device)
            )
            # Cap dinámico para nodos nuevos (memoria)
            try:
                N = int(loaded_obj['data']['pm'].x.size(0))
                # Más agresivo en HPO para acelerar
                cfg.max_new_nodes = max(200, min(5000, int(0.05 * N)))
            except Exception:
                cfg.max_new_nodes = 50000

            # Ejecutar entrenamiento ImGAGN
            res = train_imgagn(
                data=data,  # mantener en CPU para evitar OOM
                train_mask=train_mask.to(device),
                y_binary=y_bin.to(device),
                cfg=cfg,
                target_ntype=node_type
            )

            # Si no se generaron nodos (lambda1 muy bajo), devolvemos 0
            if 'best_train_recall' not in res:
                return 0.0

            recall = float(res['best_train_recall'].item())
            # Reporte intermedio para pruner
            trial.report(recall, step=cfg.epochs)
            if trial.should_prune():
                raise TrialPruned()
            return recall
        except TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial ImGAGN {trial.number} falló: {e}", exc_info=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0

    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Limpieza de memoria de GPU/CPU finalizada (ImGAGN HPO).")

    # Guardado de resultados
    best = study.best_trial
    best_params = best.params.copy()
    best_params['value'] = best.value
    best_params['node_type'] = node_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_tag = f"_{graph_hash[:16]}" if graph_hash else ""
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    best_path = os.path.join(RESULTADOS_DIR, f"imgagn_hyperparams_{timestamp}{hash_tag}.csv")
    pd.DataFrame([best_params]).to_csv(best_path, index=False)
    logger.info(f"[ImGAGN] Hiperparámetros top guardados en -> {os.path.basename(best_path)}")

    full_path = os.path.join(RESULTADOS_DIR, f"imgagn_full_study_{timestamp}{hash_tag}.csv")
    study.trials_dataframe().to_csv(full_path, index=False)
    logger.info(f"[ImGAGN] Estudio completo guardado en -> {os.path.basename(full_path)}")

    print("\n=== Mejor configuración ImGAGN ===")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")
    print("===============================\n")

    return best_params

def run_imgagn_pipeline(loaded_obj, retrain_gat: bool = True):
    """
    Encadena: (1) cargar o buscar mejores hparams de ImGAGN, (2) generar grafo aumentado,
    (3) guardar y (4) re-entrenar GAT sobre el grafo aumentado.
    """
    # 1) Intentar reutilizar HPO previo
    graph_filename = loaded_obj.get('filename')
    graph_hash = _calculate_graph_hash(graph_filename)
    hash_tag = f"_{graph_hash[:16]}" if graph_hash else ""
    existing = sorted(glob.glob(os.path.join(RESULTADOS_DIR, f"imgagn_hyperparams_*{hash_tag}.csv")))

    best_params = None
    if existing:
        latest = existing[-1]
        try:
            df_hp = pd.read_csv(latest)
            best_params = df_hp.iloc[0].to_dict()
            print(f"Usando hiperparámetros ImGAGN previos: {os.path.basename(latest)}")
        except Exception as e:
            logger.warning(f"No se pudo leer {os.path.basename(latest)}: {e}")

    # 2) Si no hay, lanzar HPO
    if not best_params:
        print("No hay HPO previo para ImGAGN. Ejecutando búsqueda...")
        best_params = run_imgagn_hpo(loaded_obj)
        if not best_params:
            logger.error("No se obtuvo configuración ImGAGN. Abortando pipeline.")
            return

    # 3) Construir cfg e invocar ImGAGN una vez
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        data = loaded_obj['data']
        node_type = 'pm'
        y_orig = data[node_type].y.cpu()
        # Asumimos binario; si no, min-class -> 1
        classes, counts = torch.unique(y_orig, return_counts=True)
        if classes.numel() == 2 and set(classes.tolist()) == {0, 1}:
            y_bin = y_orig.clone()
        else:
            min_idx = int(torch.argmin(counts))
            minority_label = int(classes[min_idx].item())
            y_bin = (y_orig == minority_label).long()

        train_mask = data[node_type].train_mask.cpu().bool()

        cfg = ImGAGNConfig(
            dz=int(best_params.get('dz', 100)),
            hidden_g=int(best_params.get('hidden_g', 200)),
            n_hidden_g=int(best_params.get('n_hidden_g', 1)),
            topk_links=int(best_params.get('topk_links', 5)),
            emb_dim=int(best_params.get('emb_dim', 128)),
            hid_d=int(best_params.get('hid_d', 128)),
            dropout=float(best_params.get('dropout', 0.2)),
            lambda1_ratio=float(best_params.get('lambda1_ratio', 1.0)),
            d_steps=int(best_params.get('d_steps', 50)),
            epochs=int(best_params.get('epochs', 60)),
            lr_g=float(best_params.get('lr_g', 1e-3)),
            lr_d=float(best_params.get('lr_d', 1e-3)),
            wd=float(best_params.get('wd', 5e-4)),
            alpha_reg=float(best_params.get('alpha_reg', 1e-4)),
            beta_reg=float(best_params.get('beta_reg', 1e-4)),
            margin_mm=float(best_params.get('margin_mm', 1.0)),
            grad_clip=2.0,
            device=str(device)
        )
        # Cap dinámico de nodos nuevos para evitar OOM
        try:
            N = int(loaded_obj['data']['pm'].x.size(0))
            cfg.max_new_nodes = max(1000, min(50000, int(0.3 * N)))
        except Exception:
            cfg.max_new_nodes = 50000

        logger.info("Entrenando ImGAGN con la mejor configuración...")
        res = train_imgagn(
            data=data,  # mantener en CPU para evitar OOM
            train_mask=train_mask.to(device),
            y_binary=y_bin.to(device),
            cfg=cfg,
            target_ntype=node_type
        )
    except Exception as e:
        logger.error(f"Fallo ImGAGN: {e}")
        return

    if 'x_aug' not in res or 'edge_index_aug' not in res:
        logger.warning("ImGAGN no generó nodos; entrenando GAT con grafo original.")
        run_gat_training(loaded_obj)
        return

    # 4) Volcar el grafo aumentado a HeteroData
    try:
        from torch_geometric.data import HeteroData
        x_aug = res['x_aug']  # CPU tensors
        e_hom_aug = res['edge_index_aug']
        N_old = loaded_obj['data']['pm'].x.size(0)
        ng = x_aug.size(0) - N_old
        if ng <= 0:
            logger.info("ImGAGN no añadió nodos; usando grafo original.")
            run_gat_training(loaded_obj)
            return

        data_aug = loaded_obj['data'].cpu().clone()
        # Extender features y etiquetas
        x_old = data_aug['pm'].x
        y_old = data_aug['pm'].y
        data_aug['pm'].x = x_aug  # [N_old+ng, F]
        # Etiquetas: asumimos binario y clase 1 para sintéticos
        y_new = torch.cat([y_old, torch.ones(ng, dtype=y_old.dtype)], dim=0)
        data_aug['pm'].y = y_new

        # Máscaras
        for m in ['train_mask', 'val_mask', 'test_mask']:
            if m in data_aug['pm']:
                mask_old = data_aug['pm'][m].bool().cpu()
                add = torch.zeros(ng, dtype=torch.bool)
                if m == 'train_mask':
                    add[:] = True  # sólo entreno
                data_aug['pm'][m] = torch.cat([mask_old, add], dim=0)

        # Flag sintético
        synth = torch.zeros(N_old + ng, dtype=torch.bool)
        synth[N_old:] = True
        data_aug['pm'].is_synthetic = synth

        # Edges nuevos: extraer los que tocan nodos nuevos
        idx_new_edges = (e_hom_aug[0] >= N_old) | (e_hom_aug[1] >= N_old)
        e_add = e_hom_aug[:, idx_new_edges]

        # Elegir relación destino para inyectar edges
        rel_candidates = []
        if ('pm', 'spatial', 'pm') in data_aug.edge_types:
            rel_candidates.append(('pm', 'spatial', 'pm'))
        if ('pm', 'temporal', 'pm') in data_aug.edge_types:
            rel_candidates.append(('pm', 'temporal', 'pm'))
        if not rel_candidates:
            logger.warning("No hay relaciones pm->pm para inyectar aristas; se actualizan solo features.")
        else:
            rel = rel_candidates[0]
            ei_old = data_aug[rel].edge_index.cpu()
            ei_new = torch.cat([ei_old, e_add.cpu()], dim=1)
            data_aug[rel].edge_index = ei_new

            # edge_attr si existe
            if 'edge_attr' in data_aug[rel] and data_aug[rel].edge_attr is not None:
                d_e = data_aug[rel].edge_attr.shape[1]
                zeros = torch.zeros(e_add.size(1), d_e, dtype=data_aug[rel].edge_attr.dtype)
                data_aug[rel].edge_attr = torch.cat([data_aug[rel].edge_attr.cpu(), zeros], dim=0)

        # 5) Guardar y entrenar GAT
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        filename = f"highway_graph_ImGAGN_AUG_{timestamp}.pt"
        out_path = os.path.join(RESULTADOS_DIR, filename)
        save_obj = dict(loaded_obj)
        save_obj['data'] = data_aug
        save_obj['filename'] = filename
        save_obj['imgagn_best_params'] = best_params
        torch.save(save_obj, out_path)
        logger.info(f"✅ Grafo ImGAGN aumentado guardado en -> {filename}")

        # Pasar a entrenamiento GAT
        if retrain_gat:
            print("\nRe-entrenando GAT sobre el grafo aumentado...\n")
            # Forzar GraphSMOTE desactivado durante el re-entrenamiento post-ImGAGN
            run_gat_training(save_obj)
        return save_obj
    except Exception as e:
        logger.error(f"Error al aplicar el grafo ImGAGN aumentado: {e}")
        return

def run_gat_testing(loaded_obj):
    """
    Carga un modelo GAT pre-entrenado y sus hiperparámetros para evaluarlo
    en el conjunto de datos de un grafo cargado.
    """
    # Preguntar al usuario si el modelo fue entrenado con GraphSMOTE para filtrar los hiperparámetros
    use_graphsmote_input = input("¿El modelo a testear fue entrenado con GraphSMOTE? (s/n): ").strip().lower()
    use_graphsmote = use_graphsmote_input in ('s', 'si', 'y', 'yes')

    # 1. Determinar dispositivo y cargar datos
    # Selection automática de dispositivo
    device = get_auto_device()
    print(f"▶️ Usando dispositivo: {device}")
    
    data = loaded_obj['data'].to(device)

    # 2. Cargar hiperparámetros
    print("--- Carga de Hiperparámetros para Testeo ---")
    all_hp_files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "optuna_hyperparams_*.csv"))) + sorted(glob.glob(os.path.join(RESULTADOS_DIR, "grid_search_results_*.csv")))
    
    # Filtrar archivos de hiperparámetros
    if use_graphsmote:
        hp_files = [f for f in all_hp_files if "_GraphSMOTE" in os.path.basename(f)]
    else:
        hp_files = [f for f in all_hp_files if "_GraphSMOTE" not in os.path.basename(f)]
    # Filtro ImGAGN
    want_imgagn = _has_imgagn(loaded_obj)
    hp_files = [f for f in hp_files if ("_ImGAGN" in os.path.basename(f)) == want_imgagn or (not want_imgagn and "_ImGAGN" not in os.path.basename(f))]

    if not hp_files:
        print(f"❌ No se encontraron archivos de hiperparámetros compatibles en '{RESULTADOS_DIR}'.")
        print("  Asegúrese de que su elección sobre GraphSMOTE sea correcta o entrene un modelo primero.")
        return

    print("Archivos de hiperparámetros disponibles (filtrados):")
    for i, f in enumerate(hp_files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    
    try:
        sel_str = input("Seleccione el número del archivo CSV de hiperparámetros (o presione Enter para usar el más reciente): ").strip()
        if not sel_str:
            sel = len(hp_files) - 1
            print(f"Usando el archivo más reciente: {os.path.basename(hp_files[sel])}")
        else:
            sel = int(sel_str) - 1

        if not (0 <= sel < len(hp_files)):
            raise ValueError("Selección fuera de rango.")
        
        hp_path = hp_files[sel]
        df_hp = pd.read_csv(hp_path)
        best_params = df_hp.to_dict(orient='records')[0]
        
        # Conversión de tipos (igual que en run_gat_training)
        if 'hidden_channels' in best_params:
            best_params['hidden_channels'] = int(best_params['hidden_channels'])
        if 'num_heads' in best_params:
            best_params['num_heads'] = int(best_params['num_heads'])
        if 'dropout' in best_params:
            best_params['dropout'] = float(best_params['dropout'])
        if 'lr' in best_params:
            best_params['lr'] = float(best_params['lr'])
        if 'weight_decay' in best_params:
            best_params['weight_decay'] = float(best_params['weight_decay'])
        if 'grad_clip' in best_params:
            best_params['grad_clip'] = float(best_params['grad_clip'])
        if 'use_checkpointing' in best_params:
            best_params['use_checkpointing'] = str(best_params.get('use_checkpointing', 'False')).lower() == 'true'
        if 'focal_gamma' in best_params:
            best_params['focal_gamma'] = float(best_params['focal_gamma'])
        if 'focal_alpha' in best_params:
            best_params['focal_alpha'] = float(best_params['focal_alpha'])
        if 'oversampling_ratio' in best_params:
            best_params['oversampling_ratio'] = int(best_params['oversampling_ratio'])
        
        # Handle lambda_H mode
        lambda_H_mode = best_params.get('lambda_H_mode', 'dynamic')
        if lambda_H_mode == 'fixed':
            if 'lambda_H_fixed' in best_params:
                best_params['lambda_H'] = float(best_params['lambda_H_fixed'])
        else:
            if 'initial_lambda_H' in best_params and 'final_lambda_H' in best_params:
                 best_params['lambda_H'] = (float(best_params['initial_lambda_H']) + float(best_params['final_lambda_H'])) / 2
            elif 'lambda_H' in best_params:
                 best_params['lambda_H'] = float(best_params['lambda_H'])

        print(f"✅ Hiperparámetros cargados desde '{os.path.basename(hp_path)}'.")

    except (ValueError, IndexError, KeyError) as e:
        print(f"❌ Selección inválida o error al leer el archivo: {e}.")
        return

    # 3. Determinar si el modelo es homogéneo y construirlo
    num_classes = 2

    print("▶️ Instanciando modelo GAT heterogéneo...")
    edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')

    in_channels = data['pm'].x.shape[1]
    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=best_params['hidden_channels'],
        out_channels=num_classes,
        num_heads=best_params['num_heads'],
        dropout=best_params['dropout'],
        edge_feature_dim=edge_feature_dim,
        num_layers=int(best_params.get('num_layers', len(NUM_NEIGHBORS))),
        use_checkpointing=True,
        aggr1=best_params.get('aggr1', 'sum'),
        aggr2=best_params.get('aggr2', 'sum')
    ).to(device)

    # 4. Cargar el estado del modelo entrenado
    best_model_path = _find_best_model_path(use_graphsmote, loaded_obj)
    if not best_model_path or not os.path.exists(best_model_path):
        print("❌ No se encontró un modelo GAT compatible. Por favor, entrene un modelo primero (opción g).")
        return
    
    print(f"Cargando el modelo entrenado desde: {os.path.basename(best_model_path)}")
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        print("  Asegúrese de que los hiperparámetros seleccionados coincidan con el modelo guardado.")
        return

    # 5. Ejecutar test e imprimir reporte
    print("▶️ Evaluación del modelo cargado...")
    model.use_checkpointing = False
    
    # 5.a) Determinar el umbral de decisión (tau)
    tau = None
    platt_model = None
    if 'best_tau' in best_params:
        tau = best_params['best_tau']
        print(f"ℹ️ Usando umbral (tau) pre-calculado de Optuna: {tau:.6f}")

    if tau is None:
        print("▶️ No se encontró `best_tau` en los hiperparámetros. Calculando umbral óptimo desde la validación...")
        # 1) Pase inicial para recolectar probabilidades en validación
        initial_results = test(model, data, node_type='pm', threshold=None)

    if tau is None:
        if 'val_mask' not in initial_results:
            print("⚠️ No hay val_mask; usaré train_mask para determinar umbral (menos ideal).")
            mask_key = 'train_mask'
        else:
            mask_key = 'val_mask'

        if mask_key not in initial_results:
            print(f"Error: La máscara '{mask_key}' no se encontró en los resultados iniciales. Abortando.")
            return

        y_true_val = initial_results[mask_key]['true'].cpu().numpy().ravel()
        y_prob1_val_raw = initial_results[mask_key]['probs'][:, 1].cpu().numpy().ravel()
        y_prob1_val, platt_model = _platt_scale_probabilities(y_true_val, y_prob1_val_raw)

        # --- Calibración por Prior Shift (Opcional) ---
        apply_prior_shift = input("¿Desea aplicar calibración por prior shift? (s/n, default: n): ").strip().lower()
        if apply_prior_shift in ('s', 'si', 'y', 'yes'):
            try:
                p_train = initial_results['train_mask']['true'].float().mean().item()
                p_real_str = input(f"Prevalencia en train (p_train) es {p_train:.4f}. Ingrese la prevalencia real esperada (p_real, default: 0.01): ").strip()
                p_real = float(p_real_str) if p_real_str else 0.01
                
                logger.info(f"Aplicando prior shift: p_train={p_train:.4f}, p_real={p_real:.4f}")
                y_prob1_val_adjusted = prior_shift_adjust(p_train, p_real, y_prob1_val)
                
                # Usar probabilidades ajustadas para elegir el umbral
                tau, info = pick_threshold_from_val(y_true_val, y_prob1_val_adjusted, mode="fbeta", beta=0.5)
                print(f"🔧 Umbral (ajustado por prior) seleccionado en validación: tau={tau:.6f} -> P={info.get('precision', float('nan')):.3f}, R={info.get('recall', float('nan')):.3f}")

            except Exception as e:
                logger.error(f"No se pudo aplicar el prior shift: {e}. Se usará el umbral sin ajustar.")
                tau, info = pick_threshold_from_val(y_true_val, y_prob1_val, mode="fbeta", beta=0.5)
                print(f"🔧 Umbral seleccionado en validación: tau={tau:.6f}  -> P={info.get('precision', float('nan')):.3f}, R={info.get('recall', float('nan')):.3f}")
        else:
            # 2) Elegir tau (sin ajuste)
            tau, info = pick_threshold_from_val(y_true_val, y_prob1_val, mode="fbeta", beta=0.5)
            print(f"🔧 Umbral seleccionado en validación: tau={tau:.6f}  -> P={info.get('precision', float('nan')):.3f}, R={info.get('recall', float('nan')):.3f}")

    # 3) Pase final con ese umbral aplicado en TODOS los splits
        final_results = test(model, data, node_type='pm', threshold=tau, calibration_model=platt_model)
    
    if not final_results:
        print("El test no produjo resultados. Verifique las máscaras de datos en el grafo.")
        return

    if 'train_mask' in final_results:
        print_evaluation_report(final_results['train_mask'], "Train")
    if 'val_mask' in final_results:
        print_evaluation_report(final_results['val_mask'], "Validation")
    if 'test_mask' in final_results:
        print_evaluation_report(final_results['test_mask'], "Test")
    
    print("✅ Testeo finalizado.")

    # Preguntar si se desea analizar las aristas
    analyze_choice = input("\n¿Desea analizar las aristas más relevantes basadas en la atención del modelo? (s/n): ").strip().lower()
    if analyze_choice in ('s', 'si', 'y', 'yes'):
        try:
            k_heads_str = input(f"Ingrese el número mínimo de cabezales de atención que deben coincidir (k, default: 3): ").strip()
            k_heads = int(k_heads_str) if k_heads_str else 3
            
            percentile_str = input(f"Ingrese el percentil de atención para el umbral (0.0 a 1.0, default: 0.95): ").strip()
            percentile = float(percentile_str) if percentile_str else 0.95

            analyze_and_save_relevant_edges(model, data, loaded_obj, device, k_heads=k_heads, percentile=percentile)
        except ValueError:
            logger.error("Entrada inválida. Abortando análisis de aristas.")
        except Exception as e:
            logger.error(f"Ocurrió un error inesperado durante el análisis de aristas: {e}")

def analyze_and_save_relevant_edges(model, data, loaded_obj, device, k_heads=3, percentile=0.95):
    """
    Analiza las atenciones del modelo entrenado, extrae las aristas más relevantes
    y las guarda en un archivo CSV. Usa checkpointing para reducir memoria.
    """
    logger.info("--- Iniciando análisis de aristas relevantes basado en atención ---")
    model.eval()
    
    # Guardar estado original y desactivar checkpointing, ya que no aporta beneficios
    # de memoria durante la inferencia (forward pass sin cálculo de gradientes).
    original_checkpointing_status = getattr(model, 'use_checkpointing', False)
    model.use_checkpointing = False
    logger.info(f"Checkpointing temporalmente desactivado para el análisis de inferencia.")

    data = data.to(device)

    try:
        # 1. Calcular atenciones y edge_index efectivos por capa/relación
        with torch.no_grad():
            # Reconstruir edge_attr_dict si no existe (robustez)
            edge_attr_dict = getattr(data, 'edge_attr_dict', None)
            if not isinstance(edge_attr_dict, dict) or not edge_attr_dict:
                edge_attr_dict = { et: getattr(data[et], 'edge_attr', None) for et in data.edge_types }

            # Copia de trabajo del dict de características por tipo de nodo
            x_dict = {k: v for k, v in data.x_dict.items()}

            def get_edge_attr_or_zeros(et, n_edges, ref_tensor):
                ea = None
                if isinstance(edge_attr_dict, dict):
                    ea = edge_attr_dict.get(et, None)
                if ea is None:
                    if hasattr(model, 'edge_feature_dim') and model.edge_feature_dim and model.edge_feature_dim > 0:
                        return torch.zeros((n_edges, model.edge_feature_dim), dtype=ref_tensor.dtype, device=ref_tensor.device)
                    return None
                return ea

            # Mapa: clave -> (edge_index_usado, alpha)
            attn_pairs = {}

            for i in range(getattr(model, 'num_layers', 1)):
                conv = model.convs[i]
                norm = model.norms[i]

                x_in = x_dict

                active_edge_types = list(conv.convs.keys())
                active_eid = {k: data.edge_index_dict[k] for k in data.edge_index_dict.keys() if k in active_edge_types}
                active_ead = {}
                for k in active_edge_types:
                    if k in edge_attr_dict and edge_attr_dict[k] is not None:
                        active_ead[k] = edge_attr_dict[k]
                    else:
                        if hasattr(model, 'edge_feature_dim') and model.edge_feature_dim and model.edge_feature_dim > 0 and k in active_eid:
                            num_e = active_eid[k].shape[1]
                            ref = next(iter(x_in.values()))
                            active_ead[k] = torch.zeros((num_e, model.edge_feature_dim), dtype=ref.dtype, device=ref.device)

                out_dict = conv(x_in, active_eid, active_ead)
                x_dict = {key: F.relu(norm[key](x)) for key, x in out_dict.items()}
                x_dict = {key: F.dropout(x, p=getattr(model, 'dropout', 0.0), training=False) for key, x in x_dict.items()}

                for edge_type, conv_layer in conv.convs.items():
                    src_ntype, rel_name, dst_ntype = edge_type
                    eidx = data.edge_index_dict.get(edge_type)
                    if eidx is None:
                        continue
                    x_src = x_in[src_ntype]
                    x_dst = x_in[dst_ntype]
                    n_edges = eidx.size(1)
                    ref = x_src if isinstance(x_src, torch.Tensor) else x_dst
                    eattr = get_edge_attr_or_zeros(edge_type, n_edges, ref)
                    try:
                        if src_ntype == dst_ntype:
                            _, (returned_eidx, alpha) = conv_layer(x_src, eidx, edge_attr=eattr, return_attention_weights=True)
                        else:
                            _, (returned_eidx, alpha) = conv_layer((x_src, x_dst), eidx, edge_attr=eattr, return_attention_weights=True)
                        key = f'conv{i+1}_{src_ntype}_{rel_name}_{dst_ntype}'
                        attn_pairs[key] = (returned_eidx.detach().cpu(), alpha.detach().cpu())
                    except Exception as ex:
                        logger.debug(f"No se pudo obtener atención para {edge_type} en capa {i+1}: {ex}")

    finally:
        # Restaurar el estado original del checkpointing
        model.use_checkpointing = original_checkpointing_status
        logger.info(f"Checkpointing restaurado a su estado original ({original_checkpointing_status}).")

    # 2. Seleccionar atenciones de la última capa disponible (preferir conv2)
    conv2_pairs = {k: v for k, v in attn_pairs.items() if k.startswith('conv2_')}
    selected_pairs = conv2_pairs
    if not selected_pairs:
        # Fallback: detectar la última capa 'conv{n}_' disponible
        layer_ids = []
        for k in attn_pairs.keys():
            m = re.match(r"conv(\d+)_", k)
            if m:
                try:
                    layer_ids.append(int(m.group(1)))
                except Exception:
                    pass
        if layer_ids:
            last_layer = max(layer_ids)
            logger.warning(f"No se encontraron pesos de 'conv2'. Usando capa 'conv{last_layer}'.")
            selected_pairs = {k: v for k, v in attn_pairs.items() if k.startswith(f'conv{last_layer}_')}
        else:
            logger.warning("No se encontraron pesos de atención en ninguna capa. Abortando análisis.")
            return

    all_relevant_edges = []
    pm_index = loaded_obj.get('pm_index')

    def _safe_pm_info(idx: int):
        if pm_index and hasattr(pm_index, '_rev'):
            rev = pm_index._rev
            if isinstance(rev, dict):
                return rev.get(idx, (None, None))
            try:
                if 0 <= idx < len(rev):
                    return rev[idx]
            except Exception:
                pass
        return (None, None)

    for rel, (eff_edge_index, alpha) in selected_pairs.items():
        # Claves generadas en el modelo: 'conv{i}_{src}_{rel}_{dst}'
        tokens = rel.split('_')
        if len(tokens) < 4:
            logger.warning(f"Formato de clave de atención inesperado: '{rel}'. Saltando.")
            continue
        _, src_ntype, relation_name, dst_ntype = tokens[0], tokens[1], tokens[2], tokens[3]

        # alpha y edge_index ya están alineados con lo que usó GAT (incluye self-loops).
        # Filtramos self-loops para reportar solo aristas reales del grafo.
        alpha_cpu = alpha
        if alpha_cpu.dim() == 1:
            alpha_cpu = alpha_cpu.unsqueeze(1)
        eff_eidx_cpu = eff_edge_index
        non_self_mask = eff_eidx_cpu[0] != eff_eidx_cpu[1]
        if non_self_mask.numel() == 0 or non_self_mask.sum().item() == 0:
            logger.info(f"No hay aristas no-triviales (sin self-loop) para la relación '{relation_name}'.")
            continue
        alpha_cpu = alpha_cpu[non_self_mask]
        eff_eidx_cpu = eff_eidx_cpu[:, non_self_mask]

        # Umbralizar sobre las atenciones filtradas
        threshold = torch.quantile(alpha_cpu, percentile)
        above_threshold = alpha_cpu > threshold
        robust_edges_mask = torch.sum(above_threshold, dim=1) >= k_heads
        relevant_indices = robust_edges_mask.nonzero(as_tuple=False).view(-1)
        
        if relevant_indices.numel() == 0:
            logger.info(f"No se encontraron aristas robustas para la relación '{relation_name}' con k={k_heads} y percentil={percentile}.")
            continue

        logger.info(f"Se encontraron {len(relevant_indices)} aristas relevantes para la relación '{relation_name}'.")

        # Mapeo de índices y recopilación de datos
        src_nodes, dst_nodes = eff_eidx_cpu[0], eff_eidx_cpu[1]

        for edge_idx in relevant_indices:
            src_node_idx = src_nodes[edge_idx].item()
            dst_node_idx = dst_nodes[edge_idx].item()

            src_info = _safe_pm_info(src_node_idx)
            dst_info = _safe_pm_info(dst_node_idx)

            all_relevant_edges.append({
                'relation': relation_name,
                'source_node_idx': src_node_idx,
                'dest_node_idx': dst_node_idx,
                'source_portico': src_info[0] if isinstance(src_info, tuple) else src_info,
                'source_ts': src_info[1] if isinstance(src_info, tuple) else 'N/A',
                'dest_portico': dst_info[0] if isinstance(dst_info, tuple) else dst_info,
                'dest_ts': dst_info[1] if isinstance(dst_info, tuple) else 'N/A',
                'mean_attention': alpha_cpu[edge_idx].mean().item(),
                'max_attention': alpha_cpu[edge_idx].max().item()
            })

    if all_relevant_edges:
        df_relevant = pd.DataFrame(all_relevant_edges)
        timestamp = datetime.now().strftime('%d%m%Y_%H%M')
        filename = f"relevant_edges_{timestamp}_k{k_heads}_p{int(percentile*100)}.csv"
        output_path = os.path.join(RESULTADOS_DIR, filename)
        df_relevant.to_csv(output_path, index=False)
        logger.info(f"✅ Aristas relevantes guardadas en -> {output_path}")
    else:
        logger.info("No se encontraron aristas relevantes en ninguna relación para guardar.")

def test_graphsmote(loaded_obj):
    """
    Aplica GraphSMOTE a un grafo cargado para generar nodos y aristas sintéticos,
    y guarda el grafo aumentado en un nuevo archivo .pt.
    Utiliza la nueva función 'augment_graph_offline_once'.
    """
    logger.info("--- Iniciando Test de GraphSMOTE (Modo Offline) ---")
    
    # 1. Extraer datos y determinar dispositivo
    data = loaded_obj['data']
    # Selection automática de dispositivo
    device = get_auto_device()
    logger.info(f"Usando dispositivo: {device}")

    # 2. Crear un modelo base para generar embeddings
    # ADVERTENCIA: Para resultados de alta calidad, se debería usar un modelo pre-entrenado.
    # Este modelo solo se usa para obtener embeddings representativos para el test.
    logger.warning("Creando un modelo GAT temporal para generar embeddings.")
    edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')

    in_channels = data['pm'].x.shape[1]
    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=len(data['pm'].y.unique()),
        num_heads=4,
        dropout=0.0,
        edge_feature_dim=edge_feature_dim,
        num_layers=2
    ).to(device)

    # 3. Parámetros interactivos para la aumentación
    # --- CORREGIDO: Calcular ratio sobre el conjunto de entrenamiento ---
    train_mask = data['pm'].train_mask
    if train_mask.sum() > 0:
        y_train = data['pm'].y[train_mask]
        actual_ratio = y_train.float().mean().item()
    else:
        actual_ratio = 0.0
        logger.warning("La máscara de entrenamiento está vacía. El ratio actual es 0.")

    try:
        ratio_str = input(f"Ingrese el ratio de positivos deseado (e.g., 0.4 para 40%, actual en train: {actual_ratio:.4f}): ").strip()
        target_ratio = float(ratio_str)
        k_str = input(f"Ingrese el número de vecinos para SMOTE (k, e.g., 5, default: {GRAPHSMOTE_K}): ").strip()
        k = int(k_str) if k_str else GRAPHSMOTE_K
    except ValueError:
        logger.error("Entrada inválida. Usando valores por defecto.")
        target_ratio = TARGET_POS_RATIO
        k = GRAPHSMOTE_K

    # --- AÑADIDO: Entrenar los decodificadores z->x necesarios ---
    logger.info("Entrenando decodificadores z->x para la síntesis de features...")
    z2x_decoders = train_z2x_decoders(model, data, device=device, epochs=DECODER_EPOCHS) # Controlado por config

    # 4. Ejecutar aumentación offline
    logger.info(f"Ejecutando augment_graph_offline_once con target_ratio={target_ratio} y k={k}...")
    
    # Edge generator (opcional pero recomendado)
    z_dim_dict = {ntype: 64 * 4 for ntype in data.node_types}
    edge_gen = RelEdgeGen(z_dim_dict, data.edge_types).to(device)

    augmented_data, registry = augment_graph_offline_once(
        model=model,
        data=data,
        device=device,
        target_pos_ratio=target_ratio,
        z2x_decoders=z2x_decoders, # <-- AÑADIDO: Pasar los decodificadores
        k=k,
        edge_gen=edge_gen,
        seed=GS_SEED
    )

    if registry and registry.get('pm', {}).get('n_new', 0) > 0:
        num_new_pm = registry['pm']['n_new']
        logger.info(f"Se generaron {num_new_pm} nodos 'pm' sintéticos.")

        # 5. Guardar el grafo aumentado
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        filename = f"highway_graph_SMOTE_TEST_{timestamp}.pt"
        output_path = os.path.join(RESULTADOS_DIR, filename)

        # El grafo aumentado ya tiene el flag 'is_synthetic'
        save_obj = {
            'data': augmented_data.cpu(),
            'pm_index': loaded_obj.get('pm_index'),
            'mu': loaded_obj.get('mu'),
            'sigma': loaded_obj.get('sigma'),
            'feature_cols': loaded_obj.get('feature_cols'),
            'filename': filename,
            'augmentation_registry': registry
        }

        torch.save(save_obj, output_path)
        logger.info(f"✅ Grafo aumentado guardado en -> {filename}")
        print(f"\n{augmented_data}")
        
        print("\n--- Resumen de Nodos Creados ---")
        print(f"Nodos 'pm' sintéticos:  {num_new_pm}")
        print("---------------------------------")
        return save_obj
    else:
        logger.info("No se generaron nodos sintéticos. El grafo no fue modificado.")
        return loaded_obj

def test_imgagn(loaded_obj):
    """
    Aplica ImGAGN al grafo cargado para generar nodos (minoría) y aristas
    sintéticas de forma interactiva, y guarda el grafo aumentado en un nuevo .pt.
    Similar a test_graphsmote pero usando el pipeline de ImGAGN directamente.
    """
    logger.info("--- Iniciando Test de ImGAGN ---")

    # 1) Dispositivo
    device = get_auto_device()
    logger.info(f"Usando dispositivo: {device}")

    data = loaded_obj['data']
    node_type = 'pm'

    # 2) Preparar etiquetas binarias 0/1 (minoría=1)
    y_orig = data[node_type].y.cpu()
    classes, counts = torch.unique(y_orig, return_counts=True)
    if classes.numel() == 2 and set(classes.tolist()) == {0, 1}:
        y_bin = y_orig.clone()
    else:
        min_idx = int(torch.argmin(counts))
        minority_label = int(classes[min_idx].item())
        y_bin = (y_orig == minority_label).long()
        logger.info(f"Mapeando etiquetas a binario: minoritaria={minority_label} -> 1")

    train_mask = data[node_type].train_mask.cpu().bool()
    if train_mask.sum().item() == 0:
        logger.error("La máscara de entrenamiento está vacía; no se puede ejecutar ImGAGN.")
        return loaded_obj

    # 3) Parámetros interactivos de ImGAGN (análogos a GraphSMOTE: ratio y k)
    #    lambda1_ratio controla cuántos nodos nuevos: (n_min + ng) / n_maj
    try:
        # Ratio deseado entre minoría y mayoría en el set de entrenamiento
        n_min = int(((train_mask == True) & (y_bin == 1)).sum().item())
        n_maj = int(((train_mask == True) & (y_bin == 0)).sum().item())
        cur_lambda = (n_min / max(1, n_maj)) if n_maj > 0 else 0.0
        ratio_str = input(f"Ingrese lambda1_ratio deseado (actual≈{cur_lambda:.3f}, ej. 1.0): ").strip()
        lambda1_ratio = float(ratio_str) if ratio_str else 1.0

        topk_str = input("Ingrese topk_links para nuevas aristas (ej. 5): ").strip()
        topk_links = int(topk_str) if topk_str else 5

        epochs_str = input("Epochs de entrenamiento (ej. 20, Enter=por defecto 30): ").strip()
        epochs = int(epochs_str) if epochs_str else 30

        dsteps_str = input("Pasos del Discriminador por epoch (ej. 20, Enter=por defecto 20): ").strip()
        d_steps = int(dsteps_str) if dsteps_str else 20
    except ValueError:
        logger.error("Entrada inválida. Usando parámetros por defecto para ImGAGN.")
        lambda1_ratio = 1.0
        topk_links = 5
        epochs = 30
        d_steps = 20

    # 4) Construir configuración y entrenar ImGAGN una vez
    cfg = ImGAGNConfig(
        lambda1_ratio=lambda1_ratio,
        topk_links=topk_links,
        epochs=epochs,
        d_steps=d_steps,
        device=str(device),
        # límites de seguridad para evitar OOM en tests rápidos
        max_new_nodes=max(1000, min(50000, int(0.3 * int(data[node_type].x.size(0)))))
    )

    logger.info(f"Ejecutando ImGAGN con lambda1_ratio={lambda1_ratio}, topk_links={topk_links}, epochs={epochs}, d_steps={d_steps}...")
    try:
        res = train_imgagn(
            data=data,  # mantener en CPU para seguridad de memoria
            train_mask=train_mask.to(device),
            y_binary=y_bin.to(device),
            cfg=cfg,
            target_ntype=node_type
        )
    except Exception as e:
        logger.error(f"Fallo ImGAGN: {e}")
        return loaded_obj

    if 'x_aug' not in res or 'edge_index_aug' not in res:
        logger.info("ImGAGN no generó nodos; el grafo no fue modificado.")
        return loaded_obj

    # 5) Aplicar los resultados al HeteroData original y guardar
    try:
        x_aug = res['x_aug']  # CPU tensors
        e_hom_aug = res['edge_index_aug']
        N_old = data[node_type].x.size(0)
        ng = int(x_aug.size(0) - N_old)
        if ng <= 0:
            logger.info("ImGAGN no añadió nodos; usando grafo original.")
            return loaded_obj

        data_aug = data.cpu().clone()

        # Extender features y etiquetas (minoría=1 para sintéticos)
        x_old = data_aug[node_type].x
        y_old = data_aug[node_type].y
        data_aug[node_type].x = x_aug  # [N_old+ng, F]
        y_new = torch.cat([y_old, torch.ones(ng, dtype=y_old.dtype)], dim=0)
        data_aug[node_type].y = y_new

        # Máscaras: nuevos nodos -> train
        for m in ['train_mask', 'val_mask', 'test_mask']:
            if m in data_aug[node_type]:
                mask_old = data_aug[node_type][m].bool().cpu()
                add = torch.zeros(ng, dtype=torch.bool)
                if m == 'train_mask':
                    add[:] = True
                data_aug[node_type][m] = torch.cat([mask_old, add], dim=0)

        # Flag sintético
        synth = torch.zeros(N_old + ng, dtype=torch.bool)
        synth[N_old:] = True
        data_aug[node_type].is_synthetic = synth

        # Aristas nuevas que tocan nodos generados (homogéneas -> relación pm->pm)
        idx_new_edges = (e_hom_aug[0] >= N_old) | (e_hom_aug[1] >= N_old)
        e_add = e_hom_aug[:, idx_new_edges]

        rel_candidates = []
        if ('pm', 'spatial', 'pm') in data_aug.edge_types:
            rel_candidates.append(('pm', 'spatial', 'pm'))
        if ('pm', 'temporal', 'pm') in data_aug.edge_types:
            rel_candidates.append(('pm', 'temporal', 'pm'))

        if rel_candidates:
            rel = rel_candidates[0]
            ei_old = data_aug[rel].edge_index.cpu()
            ei_new = torch.cat([ei_old, e_add.cpu()], dim=1)
            data_aug[rel].edge_index = ei_new

            # edge_attr si existe -> rellenar con ceros para nuevas aristas
            if 'edge_attr' in data_aug[rel] and data_aug[rel].edge_attr is not None:
                d_e = data_aug[rel].edge_attr.shape[1]
                zeros = torch.zeros(e_add.size(1), d_e, dtype=data_aug[rel].edge_attr.dtype)
                data_aug[rel].edge_attr = torch.cat([data_aug[rel].edge_attr.cpu(), zeros], dim=0)
        else:
            logger.warning("No hay relaciones pm->pm para inyectar aristas; solo se actualizan features y etiquetas.")

        # Guardar
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        filename = f"highway_graph_ImGAGN_TEST_{timestamp}.pt"
        out_path = os.path.join(RESULTADOS_DIR, filename)
        save_obj = dict(loaded_obj)
        save_obj['data'] = data_aug
        save_obj['filename'] = filename
        torch.save(save_obj, out_path)
        logger.info(f"✅ Grafo ImGAGN (TEST) aumentado guardado en -> {filename}")
        print(f"\n{data_aug}")

        print("\n--- Resumen de Nodos Creados (ImGAGN) ---")
        print(f"Nodos 'pm' sintéticos:  {ng}")
        print("-----------------------------------------")
        return save_obj
    except Exception as e:
        logger.error(f"Error al aplicar el grafo ImGAGN (TEST): {e}")
        return loaded_obj

def calculate_gini(alpha_h):
    if alpha_h.numel() == 0:
        return 0.0
    # Ordenar atenciones y calcular sumas acumuladas
    alpha_sorted = torch.sort(alpha_h).values
    n = len(alpha_sorted)
    cum_alpha = torch.cumsum(alpha_sorted, dim=0)
    # Fórmula del Gini
    return (n + 1 - 2 * torch.sum(cum_alpha) / cum_alpha[-1]) / n

def menu_utilidades_grafos():    
    limpiar_pantalla()
    while True:        
        print("===== Menú de Utilidades =====")        
        print("  [1] Fusionar grafos")        
        print("  [2] Selección de caracteristicas")        
        print(f"  [3] Agregar flujos [{DT_MINUTES} min x pórtico]") 
        print("  [4] Visualizar flujos agregados")
        print("  [5] Utilidades CSVs")
        print("  [6] Ejecutar Tests" )
        print("  [7] Gestor de Artefactos" )       
        print("")        
        print("  [0] Volver al menú principal")
        print("")        
        choice = input("Elige una opción: ").strip()        
        if choice == "1":                     
            fusionar_grafos()
        elif choice == "2":            
            select_features()
        elif choice == "3":
            print("▶️ Selecciona archivo de Flujos")
            try:
                resp_stream = input("¿Desea procesar por partes (archivos muy grandes)? (s/n): ").strip().lower()
            except Exception:
                resp_stream = 'n'
            if resp_stream in ('s','si','y','yes'):
                # Modo streaming de baja memoria: lee el CSV a chunks, particiona por pórtico y agrega por partes
                print(f"▶️ Procesamiento por partes (ventana {DT_MINUTES} min)")
                pivot = compute_pm_features_streaming(None)
                if pivot is not None:
                    print(f"✅ Features calculadas en memoria: {len(pivot)} filas")
                else:
                    print("ℹ️ Features calculadas y guardadas en 'Resultados/'.")
            else:
                # Modo en memoria (recomendado para archivos pequeños o muestras)
                df_flows = load_flujos()
                print(f"▶️ Agregando flujos por segmento temporal de {DT_MINUTES} min x pórtico…")
                compute_pm_features(df_flows)
        elif choice == "4":
            print("Lanzando la aplicación de visualización interactiva...")
            print("Cuando termines, puedes usar el botón 'Finalizar' en la aplicación web o presionar Ctrl+C en esta terminal.")
            input("Enter poara continuar....... ").strip()   
            os.system("streamlit run src/visualization.py")
            limpiar_pantalla()
        elif choice == "5":
            menu_utilidades_csv()
        elif choice == "6":
            print("Ejecutando tests con cobertura...")
            os.system("pytest -v tests")
        elif choice == "7":
            artifacts_menu()
        elif choice == "0":
            limpiar_pantalla()            
            break        
        else:            
            print("❌ Opción no reconocida.")

def menu_baseline():
    limpiar_pantalla()
    while True:
        print("===== Menú Baseline ======")
        print("  [1] Multilayer Perceptron")
        print("  [2] Transformer Series Temporales")
        print("  [3] XGBoost")
        print("  [4] Anomalias (tabular)")
        print("")
        print("  [5] Reportes de Evaluación")
        print("")
        print("  [0] Volver al menú principal")
        print("")
        choice = input("Elige una opción: ").strip()
        if choice == "1":
            run_mlp_tabular_pipeline()
        elif choice == "2":
            run_transformer_ts_pipeline()
        elif choice == "3":
            run_xgboost_pipeline()
        elif choice == "4":
            # Detección de anomalías tabular (pórtico-minuto y=1)
            run_anomaly_pipeline()
        elif choice == "5":
            baseline_reports_menu()
        elif choice == "0":
            limpiar_pantalla()
            break
        else:
            print("❌ Opción no reconocida.")

def _print_table(rows, headers):
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    def fmt_row(row):
        return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(row))
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))

def _report_generic_results(pattern: Union[str, List[str]], model_label: str):
    # Acepta uno o varios patrones y combina resultados
    patterns = pattern if isinstance(pattern, (list, tuple)) else [pattern]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(RESULTADOS_DIR, pat)))
    files = sorted(set(files), key=os.path.getmtime)
    if not files:
        pats_txt = ", ".join(patterns)
        print(f"❌ No se encontraron resultados para {model_label} (patrones: {pats_txt}).")
        return
    if len(files) == 1:
        path = files[0]
    else:
        print(f"Archivos de resultados ({model_label}) disponibles:")
        for i, f in enumerate(files, 1):
            print(f"  [{i}] {os.path.basename(f)}")
        try:
            sel = input("Seleccione el número de archivo a mostrar (Enter para el último): ").strip()
            if sel:
                idx = int(sel) - 1
                if not (0 <= idx < len(files)):
                    raise ValueError
                path = files[idx]
            else:
                path = files[-1]
        except Exception:
            path = files[-1]
    print(f"\n▶️ Mostrando → {os.path.basename(path)}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ No se pudo leer '{os.path.basename(path)}': {e}")
        return

    if 'split' in df.columns:
        headers = [
            'split','auroc','auprc','threshold','precision_1','recall_1','f1_1','accuracy','support_pos','support_neg'
        ]
        rows = []
        for _, r in df.iterrows():
            rows.append([
                r.get('split', ''),
                f"{r.get('auroc', float('nan')):.4f}" if pd.notna(r.get('auroc')) else 'NaN',
                f"{r.get('auprc', float('nan')):.4f}" if pd.notna(r.get('auprc')) else 'NaN',
                f"{r.get('threshold', float('nan')):.4f}" if pd.notna(r.get('threshold')) else 'NaN',
                f"{r.get('precision_1', float('nan')):.4f}" if pd.notna(r.get('precision_1')) else 'NaN',
                f"{r.get('recall_1', float('nan')):.4f}" if pd.notna(r.get('recall_1')) else 'NaN',
                f"{r.get('f1_1', float('nan')):.4f}" if pd.notna(r.get('f1_1')) else 'NaN',
                f"{r.get('accuracy', float('nan')):.4f}" if pd.notna(r.get('accuracy')) else 'NaN',
                int(r.get('support_pos', 0)) if pd.notna(r.get('support_pos')) else 0,
                int(r.get('support_neg', 0)) if pd.notna(r.get('support_neg')) else 0,
            ])
        _print_table(rows, headers)
    else:
        # Transformer summary file
        def g(key):
            v = df.iloc[0].get(key)
            if pd.isna(v):
                return 'NaN'
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)
        rows = [
            ['val', g('val_f1_acc_1'), g('val_precision_1'), g('val_recall_1'), g('val_accuracy')],
            ['test', g('test_f1_acc_1'), g('test_precision_1'), g('test_recall_1'), g('test_accuracy')],
        ]
        headers = ['split','f1_1','precision_1','recall_1','accuracy']
        _print_table(rows, headers)

def _show_top_anomalies_from_preds(pred_pattern: Union[str, List[str]], label: str, score_col: str = 'anomaly_score'):
    patterns = pred_pattern if isinstance(pred_pattern, (list, tuple)) else [pred_pattern]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(RESULTADOS_DIR, pat)))
    files = sorted(set(files), key=os.path.getmtime)
    if not files:
        pats_txt = ", ".join(patterns)
        print(f"❌ No se encontraron predicciones para {label} (patrones: {pats_txt}).")
        return
    if len(files) == 1:
        path = files[0]
    else:
        print(f"Archivos de predicciones ({label}) disponibles:")
        for i, f in enumerate(files, 1):
            print(f"  [{i}] {os.path.basename(f)}")
        try:
            sel = input("Seleccione el número de archivo a usar (Enter para el más reciente): ").strip()
            if sel:
                idx = int(sel) - 1
                if not (0 <= idx < len(files)):
                    raise ValueError
                path = files[idx]
            else:
                path = files[-1]
        except Exception:
            path = files[-1]

    print(f"\n▶️ Cargando → {os.path.basename(path)}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ No se pudo leer '{os.path.basename(path)}': {e}")
        return

    # Filtros
    splits = sorted(df['split'].dropna().astype(str).unique()) if 'split' in df.columns else []
    chosen_split = None
    if splits:
        print("Splits disponibles:", ", ".join(splits))
        s = input("Filtrar por split (Enter=Todos): ").strip()
        if s and s in splits:
            chosen_split = s
            df = df[df['split'] == s].copy()

    # Columnas esperadas
    if score_col not in df.columns and 'prob1' in df.columns:
        score_col = 'prob1'
    if score_col not in df.columns:
        print(f"❌ No se encontró la columna de score ('{score_col}').")
        return

    # Top-k
    try:
        k_str = input("Cantidad de top anomalías a mostrar (k, default: 20): ").strip()
        k = int(k_str) if k_str else 20
    except Exception:
        k = 20

    # Timestamp humano (si existe ts_min)
    if 'ts_min' in df.columns:
        try:
            ts_series = pd.to_datetime(df['ts_min'], unit='m', origin='unix')
            df['timestamp'] = ts_series.dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass

    dft = df.sort_values(by=score_col, ascending=False).head(k)
    headers = ['portico', 'ts_min']
    if 'timestamp' in dft.columns:
        headers.append('timestamp')
    headers += ['y_true', score_col, 'y_pred']
    rows = []
    for _, r in dft.iterrows():
        row = [
            r.get('portico', ''),
            r.get('ts_min', ''),
        ]
        if 'timestamp' in dft.columns:
            row.append(r.get('timestamp', ''))
        row += [
            int(r.get('y_true', 0)) if pd.notna(r.get('y_true')) else '',
            f"{float(r.get(score_col, float('nan'))):.6f}" if pd.notna(r.get(score_col)) else 'NaN',
            int(r.get('y_pred', 0)) if pd.notna(r.get('y_pred')) else '',
        ]
        rows.append(row)
    print(f"\nTop-{k} anomalías ({label})" + (f" [split={chosen_split}]" if chosen_split else ""))
    _print_table(rows, headers)

    # Exportar CSV con el Top-k
    try:
        os.makedirs(RESULTADOS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        import re as _re
        label_slug = _re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
        split_tag = chosen_split if chosen_split else 'all'
        out_path = os.path.join(RESULTADOS_DIR, f"topk_{label_slug}_{split_tag}_{ts}.csv")
        export_cols = []
        for c in ['portico', 'ts_min', 'timestamp', 'y_true', score_col, 'y_pred', 'split']:
            if c in dft.columns:
                export_cols.append(c)
        dft[export_cols].to_csv(out_path, index=False)
        print(f"\n💾 Top-{k} exportado → {out_path}")
    except Exception as e:
        print(f"⚠️ No se pudo exportar el Top-{k}: {e}")

def baseline_reports_menu():
    while True:
        print("\n===== Reportes Baseline ======")
        print("  [1] Multilayer Perceptron")
        print("  [2] Transformer Series Temporales")
        print("  [3] XGBoost")
        print("  [4] Anomalias (tabular)")
        print("  [5] GAT (Resultados)")
        print("  [6] Top-k Anomalias (tabular)")
        print("  [7] GAT (Top-k)")
        print("  [0] Volver")
        choice = input("Elige una opción: ").strip()
        if choice == '1':
            _report_generic_results("results_mlp_*.csv", "MLP")
        elif choice == '2':
            _report_generic_results("transformer_results_*.csv", "Transformer")
        elif choice == '3':
            _report_generic_results("results_xgb_*.csv", "XGBoost")
        elif choice == '4':
            _report_generic_results("results_anomaly_*.csv", "Anomalias (tabular)")
        elif choice == '5':
            _report_generic_results([
                "results_gnn_anomaly_*.csv",
                "results_gat_report_*.csv",
            ], "GAT (Resultados)")
        elif choice == '6':
            _show_top_anomalies_from_preds("preds_anomaly_*.csv", "Anomalias (tabular)", score_col='anomaly_score')
        elif choice == '7':
            _show_top_anomalies_from_preds([
                "preds_gnn_anomaly_*.csv",
                "preds_gat_report_*.csv",
            ], "GAT (Top-k)", score_col='prob1')
        elif choice == '0':
            break
        else:
            print("❌ Opción no reconocida.")

# --------------------------------------------------------------------------- #
# MENU PRINCIPAL 
# --------------------------------------------------------------------------- #

def main():
    
    limpiar_pantalla()
    loaded_obj = None
    while True:
        print("===== Menú ======")
        print("   🛠️ Grafos")
        print("  [1] Crear")
        print("  [2] Cargar")
        
        if loaded_obj:
            fname = loaded_obj.get('filename') if isinstance(loaded_obj, dict) else None
            print("")
            print(f"   🚀 Grafo: {fname or 'Ningún archivo (objeto en memoria)'}")
            print("  [4] Visualizar subgrafo de accidente")
            print("  [5] Exportar datos del grafo a CSV")
            print("  [6] Información del grafo")
            print("   ⚖️  GraphSMOTE")
            print("  [7] Búsqueda de Hiperparámetros")
            print("  [8] Entrenar GAT")
            print("  [9] Test aumento")
            print("   ⚖️  ImGAGN")
            print("  [10] Búsqueda de Hiperparámetros")
            print("  [11] Entrenar GAT")
            print("  [13] Test aumento")
            print("   🚨  Anomalias")
            print("  [14] Búsqueda de Hiperparámetros")
            print("  [15] Entrena GAT")
            
            
            has_any_model = bool(glob.glob(os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt"))) or os.path.isfile(os.path.join(RESULTADOS_DIR, "gat_model_BEST.pt"))
            if has_any_model:
                print("")
                print("   📊   Graph Attention Network (GAT)")
                print("  [r]  Reporte")

        print("")
        print("  [v] Reporte visual")
        print("  [b] Baseline")
        print("  [u] Utilidades")
        print("  [x] Salir")
        print("")
        if DEBUG:
            print("===DEBUG MODE ON===")
            print("")
        choice = input("Elige una opción: ").strip()

        if choice == "1":
            new_graph_path = build_graph()
            if new_graph_path:
                print(f"✅ Grafo recién creado en: {new_graph_path}")
                print("Cargando el nuevo grafo en memoria...")
                try:
                    # Selection automática de dispositivo
                    device = get_auto_device()
                    print(f"Usando dispositivo: {device}")
                    loaded_obj = torch.load(new_graph_path, map_location=device, weights_only=False)
                    loaded_obj['filename'] = os.path.basename(new_graph_path)
                    print("---------------------------------------")
                    print(f"✅ Grafo creado y cargado en memoria: {loaded_obj.get('filename', 'N/A')}")    
                except Exception as e:
                    print(f"❌ Error al cargar el grafo recién creado: {e}")
                    loaded_obj = None
        elif choice == "2":
            result = load_graph()
            if result:
                loaded_obj = result
                print(f"✅ Grafo cargado: {loaded_obj.get('filename', 'N/A')}")
                print(loaded_obj['data'])
        elif choice == "4" and loaded_obj:
            visualize_acc_subgraph(loaded_obj)
        elif choice == "5" and loaded_obj:
            export_graph_to_csv(loaded_obj)
        elif choice == "6" and loaded_obj:
            visualize_graph_overview(loaded_obj)
        elif choice == "8" and loaded_obj:
            run_gat_training(loaded_obj)
        elif choice == "r" and loaded_obj:
            # Reporte unificado con detección automática (Anomalías/GraphSMOTE/ImGAGN)
            run_gnn_anomaly_pipeline(loaded_obj)
        elif choice == "7" and loaded_obj:
            search_hyperparameters(loaded_obj)
        elif choice == "9" and loaded_obj:
            result = test_graphsmote(loaded_obj)
            if result:
                loaded_obj = result
                print(f"✅ Grafo aumentado y cargado en memoria: {loaded_obj.get('filename', 'N/A')}")
        elif choice == "10" and loaded_obj:
            run_imgagn_hpo(loaded_obj)
        elif choice == "11" and loaded_obj:
            result = run_imgagn_pipeline(loaded_obj)
            if result:
                loaded_obj = result
                print(f"✅ Grafo ImGAGN aumentado cargado en memoria: {loaded_obj.get('filename', 'N/A')}")
        elif choice == "13" and loaded_obj:
            result = test_imgagn(loaded_obj)
            if result:
                loaded_obj = result
                print(f"✅ Grafo ImGAGN (TEST) aumentado cargado en memoria: {loaded_obj.get('filename', 'N/A')}")
        elif choice == "15" and loaded_obj:
            # Entrenamiento de GAT (modo anomalías). Pregunta GraphSMOTE internamente.
            if isinstance(loaded_obj, dict):
                loaded_obj['purpose'] = 'Anomaly'
            run_gat_training(loaded_obj, purpose='Anomaly')
        elif choice == "14" and loaded_obj:
            run_gnn_anomaly_hpo_then_train(loaded_obj)
        elif choice == "v":
            # Lanza servidor dinámico del Reporte visual (logs y listas en vivo)
            print("Lanzando servidor dinámico del Reporte visual (Ctrl+C para salir)…")
            try:
                run_visual_server(open_browser=True)
            except Exception as e:
                print(f"❌ Error al iniciar el Reporte visual dinámico: {e}")
        elif choice == "b":
            menu_baseline()
        elif choice == "u":
            menu_utilidades_grafos()
        elif choice == "x":
            limpiar_pantalla()
            break
        else:
            print("❌ Opción no reconocida.")

if __name__ == '__main__':
    # Modo rápido: `python main.py -v` o `--visual` lanza el servidor visual
    try:
        args = sys.argv[1:]
        if any(a in ('-v', '--visual') for a in args):
            print("Lanzando servidor dinámico del Reporte visual (Ctrl+C para salir)…")
            run_visual_server(open_browser=True)
        else:
            main()
    except KeyboardInterrupt:
        pass
