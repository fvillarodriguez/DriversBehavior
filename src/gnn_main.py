#!/usr/bin/env python
# tensorboard --logdir=Resultados/runs_attention
import os, math, glob, gc, sys, time, uuid
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # --- FIX for MPS Fallback ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", # Asignador seguro de CUDA
  "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8")

from typing import Callable, Optional, Union, List, Dict, Tuple

import pandas as pd
import re
from datetime import datetime, timezone
import numpy as np
import torch 
from src.gnn_mps_scatter import install_gnn_mps_scatter_policy

install_gnn_mps_scatter_policy()

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import (
    NeighborLoader,
    ClusterData,
    ClusterLoader,
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
)
import torch.nn.functional as F 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, matthews_corrcoef
import optuna 
import logging
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
import warnings
from optuna import TrialPruned
from optuna.exceptions import ExperimentalWarning
import hashlib
import shutil
from src.visual import run_visual_server
import json
try:
    import joblib  # para cargar artefactos de anomalías
except Exception:
    joblib = None
from src.graph import build_graph
from src.features import compute_pm_features, compute_pm_features_streaming
from src.train_pretrain import train_minibatch, pretrain_edge_generator
from src.gnn_rl_sampler import (
    RLTopPHeteroLoader,
    RioGNNThresholdController,
    pretrain_label_aware_similarity,
    relation_max_degrees,
)
from src.gat_model import HeteroGAT, HeteroGATWithEdgeEncoder, HeteroEdgeAware
from src.temporal_head import TemporalAggregator
from src.temporal_heads import TemporalGRUHead, TemporalTransformerHead, TemporalAttnPoolHead
from src.graphsmote import (RelEdgeGen, train_z2x_decoders, train_edge_attr_decoders,
                            augment_graph_offline_once, refresh_synthetics_online, compute_epoch_embeddings)
from src.imgagn import ImGAGNConfig, train_imgagn
from src.optimizers import get_optimizer_cls
from src.config import (SEED,DT_MINUTES,MAX_EPOCHS,EARLY_STOPPING_PATIENCE,EARLY_STOPPING_MIN_DELTA,RESULTADOS_DIR,
                        BATCH_SIZE,NUM_NEIGHBORS,NUM_NEIGHBORS_OVERRIDE,N_TRIALS,DEBUG,NUM_EPOCHS_OPTUNA,ACCUMULATION_STEPS,
                        GRAPHSMOTE_MODE, TARGET_POS_RATIO,
                        GRAPHSMOTE_K, PRETRAIN_EDGE_EPOCHS, SMOTE_EVERY_N_EPOCHS,
                        GS_SEED, SAVE_AUG_GRAPH_PATH, DECODER_EPOCHS, F_BETA_THRESHOLD, XAI,
                        EXPORT_LEGACY_GAT_CSV, SAVE_GAT_ALIASES, AUTO_IMGAGN_PRETRAIN,
                        AUTOCALIBRATE_PROBS, GNN_VARIANT, GNN_TEMPORAL_CACHE_STRATEGY, get_auto_device)
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

def _json_clean(obj):
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj

_PERSISTED_TRAINING_EVENTS = {
    "train_start",
    "epoch",
    "test_result",
    "test_error",
    "train_end",
}


def _default_training_metrics_history_path(save_state_path: Optional[str]) -> Optional[str]:
    if not save_state_path:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(str(save_state_path))), "metrics_history.jsonl")


def _append_training_history_event(history_path: Optional[str], payload: Dict[str, object]) -> None:
    if not history_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(str(history_path))) or ".", exist_ok=True)
        clean_payload = _json_clean(_json_safe(payload))
        with open(str(history_path), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(clean_payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning(f"No se pudo persistir historial de entrenamiento GNN: {exc}")


def _reset_training_history(history_path: Optional[str]) -> None:
    if not history_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(str(history_path))) or ".", exist_ok=True)
        with open(str(history_path), "w", encoding="utf-8"):
            pass
    except Exception as exc:
        logger.warning(f"No se pudo reiniciar historial de entrenamiento GNN: {exc}")


def _emit_training_event(
    event: str,
    run_id: str,
    *,
    history_path: Optional[str] = None,
    **payload,
) -> None:
    try:
        data = {
            "scope": "gnn_training",
            "event": event,
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if history_path:
            data["metrics_history_path"] = str(history_path)
        data.update(payload)
        data = _json_clean(_json_safe(data))
        logger.info(json.dumps(data))
        if history_path and event in _PERSISTED_TRAINING_EVENTS:
            _append_training_history_event(history_path, data)
    except Exception:
        pass


def _training_stop_requested(should_stop: Optional[Callable[[], bool]]) -> bool:
    if should_stop is None:
        return False
    try:
        return bool(should_stop())
    except Exception as exc:
        logger.warning(f"No se pudo consultar solicitud de parada: {exc}")
        return False


def _training_test_requested(should_test: Optional[Callable[[], bool]]) -> bool:
    if should_test is None:
        return False
    try:
        return bool(should_test())
    except Exception as exc:
        logger.warning(f"No se pudo consultar solicitud de test intermedio: {exc}")
        return False


def _summarize_binary_test_result(
    result: Dict[str, object],
    *,
    epoch: int,
    best_epoch: int,
    threshold: Optional[float],
    checkpoint_path: Optional[str],
    eval_target: str = "best_checkpoint",
    automatic: bool = False,
) -> Dict[str, object]:
    report = result.get("report") or {}
    pos_report = report.get("Accidente (1)", {}) if isinstance(report, dict) else {}
    macro_report = report.get("macro avg", {}) if isinstance(report, dict) else {}
    far_value = result.get("far")
    if far_value is None:
        far_value = result.get("false_alarm_ratio")
    summary = {
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "eval_target": str(eval_target),
        "automatic": bool(automatic),
        "checkpoint_path": checkpoint_path,
        "threshold": float(threshold) if threshold is not None else None,
        "accuracy": report.get("accuracy") if isinstance(report, dict) else None,
        "precision": pos_report.get("precision") if isinstance(pos_report, dict) else None,
        "recall": pos_report.get("recall") if isinstance(pos_report, dict) else None,
        "f1_pos": pos_report.get("f1-score") if isinstance(pos_report, dict) else None,
        "f1_macro": macro_report.get("f1-score") if isinstance(macro_report, dict) else None,
        "auprc": result.get("auprc"),
        "auc": result.get("auc"),
        "mcc": result.get("mcc"),
        "far": far_value,
        "cm": result.get("cm"),
    }
    return _json_clean(_json_safe(summary))


def _test_best_checkpoint_during_training(
    *,
    model,
    best_checkpoint_path: str,
    base_graph,
    node_type: str,
    batch_size: int,
    num_neighbors,
    threshold: Optional[float],
    device: torch.device,
    epoch: int,
    best_epoch: int,
) -> Dict[str, object]:
    if not best_checkpoint_path or not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(
            f"No existe checkpoint BEST para test intermedio: {best_checkpoint_path}"
        )

    was_training = bool(model.training)
    previous_checkpointing = getattr(model, "use_checkpointing", None)
    current_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    try:
        best_state = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(best_state, dict) and (
            "model_state" in best_state or "state_dict" in best_state
        ):
            best_state = best_state.get("model_state") or best_state.get("state_dict")
        elif isinstance(best_state, torch.nn.Module):
            best_state = best_state.state_dict()
        if not isinstance(best_state, dict):
            raise TypeError("El checkpoint BEST no contiene un state_dict evaluable.")

        if previous_checkpointing is not None:
            model.use_checkpointing = False
        model.load_state_dict(best_state, strict=True)
        model.to(device)
        results = test(
            model,
            base_graph,
            node_type=node_type,
            batch_size=batch_size,
            masks=["test_mask"],
            threshold=threshold,
            num_neighbors=num_neighbors,
        )
        if not results or "test_mask" not in results:
            raise RuntimeError("No se obtuvieron resultados sobre test_mask.")
        return _summarize_binary_test_result(
            results["test_mask"],
            epoch=int(epoch),
            best_epoch=int(best_epoch),
            threshold=threshold,
            checkpoint_path=str(best_checkpoint_path),
            eval_target="best_checkpoint",
            automatic=False,
        )
    finally:
        try:
            model.load_state_dict(current_state, strict=True)
            model.to(device)
            if was_training:
                model.train()
            else:
                model.eval()
            if previous_checkpointing is not None:
                model.use_checkpointing = previous_checkpointing
        finally:
            del current_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _test_current_model_during_training(
    *,
    model,
    base_graph,
    node_type: str,
    batch_size: int,
    num_neighbors,
    threshold: Optional[float],
    device: torch.device,
    epoch: int,
    best_epoch: int,
    automatic: bool,
) -> Dict[str, object]:
    was_training = bool(model.training)
    previous_checkpointing = getattr(model, "use_checkpointing", None)
    try:
        if previous_checkpointing is not None:
            model.use_checkpointing = False
        model.to(device)
        results = test(
            model,
            base_graph,
            node_type=node_type,
            batch_size=batch_size,
            masks=["test_mask"],
            threshold=threshold,
            num_neighbors=num_neighbors,
        )
        if not results or "test_mask" not in results:
            raise RuntimeError("No se obtuvieron resultados sobre test_mask.")
        return _summarize_binary_test_result(
            results["test_mask"],
            epoch=int(epoch),
            best_epoch=int(best_epoch),
            threshold=threshold,
            checkpoint_path=None,
            eval_target="current_epoch",
            automatic=bool(automatic),
        )
    finally:
        try:
            if was_training:
                model.train()
            else:
                model.eval()
            if previous_checkpointing is not None:
                model.use_checkpointing = previous_checkpointing
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

def _normalize_gnn_variant(value: Optional[str] = None) -> str:
    raw = value
    if raw is None:
        raw = os.environ.get("SUMO_GNN_VARIANT") or os.environ.get("GNN_VARIANT") or GNN_VARIANT
    raw = str(raw or "gat_gru").strip().lower()
    raw = raw.replace("+", "_").replace("-", "_")
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    aliases = {
        "gat": "gat_snapshot",
        "baseline": "gat_snapshot",
        "snapshot": "gat_snapshot",
        "gat_temporal": "gat_gru",
        "temporal_gru": "gat_gru",
        "gru": "gat_gru",
        "temporal_lstm": "gat_gru",
        "lstm": "gat_gru",
        "transformer": "gat_transformer",
        "attnpool": "gat_attnpool",
        "attn_pool": "gat_attnpool",
        "gat_edge_mlp_temporal": "gat_edge_mlp_gru",
        "gat_edge_mlp_transformer_temporal": "gat_edge_mlp_transformer",
        "edge_aware": "gnn_edge_aware",
    }
    return aliases.get(raw, raw or "gat_gru")


def _parse_gnn_variant(value: Optional[str] = None) -> dict:
    name = _normalize_gnn_variant(value)
    encoder_kind = "gat"
    if name.startswith("gnn_edge_aware"):
        encoder_kind = "edge_aware"
    temporal_kind = "snapshot"
    for kind in ("attnpool", "transformer", "gru", "snapshot"):
        if name.endswith(f"_{kind}") or name == f"gat_{kind}":
            temporal_kind = kind
            break
    if name in {"gat_gru", "gat_transformer", "gat_attnpool"}:
        temporal_kind = name.split("_", 1)[1]
    if name in {"gat_edge_mlp", "gnn_edge_aware"}:
        temporal_kind = "snapshot"
    use_edge_mlp = "edge_mlp" in name
    return {
        "name": name,
        "encoder_kind": encoder_kind,
        "temporal_kind": temporal_kind,
        "use_edge_mlp": use_edge_mlp,
    }


def _gnn_variant_tag_component(gnn_variant: Optional[str] = None) -> str:
    return f"GNN_{_normalize_gnn_variant(gnn_variant)}"


def _variant_has_temporal_head(gnn_variant: Optional[str] = None) -> bool:
    return _parse_gnn_variant(gnn_variant)["temporal_kind"] != "snapshot"


def _model_tag_suffix(use_graphsmote: bool, loaded_obj, gnn_variant: Optional[str] = None) -> str:
    tags = []
    # If not explicitly forced by caller, check if the graph filename already implies GraphSMOTE
    fn = str(loaded_obj.get('filename', '')).lower()
    is_smote_graph = 'graphsmote' in fn or 'graph_aug' in fn
    
    if use_graphsmote or is_smote_graph:
        tags.append('GraphSMOTE')
    if _has_imgagn(loaded_obj):
        tags.append('ImGAGN')
    tags.append(_gnn_variant_tag_component(gnn_variant))
    return ("_" + "_".join(tags)) if tags else ""

def _variant_tags(use_graphsmote: bool, loaded_obj, gnn_variant: Optional[str] = None) -> str:
    """Return a variant tag for filenames: _GraphSMOTE/_ImGAGN, else _Base."""
    parts = []
    if use_graphsmote:
        parts.append("GraphSMOTE")
    has_imgagn = False
    try:
        has_imgagn = _has_imgagn(loaded_obj)
    except Exception:
        has_imgagn = False
    if has_imgagn:
        parts.append("ImGAGN")
    if not parts:
        parts.append("Base")
    parts.append(_gnn_variant_tag_component(gnn_variant))
    return "_" + "_".join(parts)

def _build_temporal_head_for_variant(
    *,
    gnn_variant: Optional[str],
    sequence_index,
    num_nodes: int,
    embedding_dim: int,
    num_classes: int,
    dropout: float = 0.0,
    cache_strategy: Optional[str] = None,
):
    variant_cfg = _parse_gnn_variant(gnn_variant)
    if variant_cfg["temporal_kind"] == "snapshot":
        return None
    if not sequence_index or getattr(sequence_index, "sequence_rows", None) is None:
        return None
    try:
        if int(np.size(sequence_index.sequence_rows)) == 0:
            return None
    except Exception:
        return None

    seq_rows_tensor = torch.as_tensor(sequence_index.sequence_rows, dtype=torch.long)
    target_rows_tensor = torch.as_tensor(sequence_index.target_rows, dtype=torch.long)
    seq_len = int(sequence_index.sequence_rows.shape[1])
    temporal_hidden_dim = embedding_dim
    cache_mode = str(cache_strategy or GNN_TEMPORAL_CACHE_STRATEGY or "incremental").strip().lower()

    temporal_kind = variant_cfg["temporal_kind"]
    if temporal_kind == "gru":
        head_cls = TemporalGRUHead
        extra_kwargs = {}
    elif temporal_kind == "transformer":
        head_cls = TemporalTransformerHead
        extra_kwargs = {
            "num_heads": 4,
            "num_layers": 2,
            "dropout": float(dropout),
        }
    elif temporal_kind == "attnpool":
        head_cls = TemporalAttnPoolHead
        extra_kwargs = {"dropout": float(dropout)}
    else:
        # Fallback conservador al head existente.
        head_cls = TemporalAggregator
        extra_kwargs = {}

    return head_cls(
        sequence_rows=seq_rows_tensor,
        target_rows=target_rows_tensor,
        num_nodes=int(num_nodes),
        embedding_dim=int(embedding_dim),
        sequence_length=seq_len,
        hidden_dim=int(temporal_hidden_dim),
        num_classes=int(num_classes),
        cache_strategy=cache_mode,
        **extra_kwargs,
    )


def _build_gnn_model(
    *,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_heads: int,
    dropout: float,
    edge_feature_dim: int,
    num_layers: int,
    use_checkpointing: bool = False,
    aggr1: str = "sum",
    aggr2: str = "sum",
    gnn_variant: Optional[str] = None,
    sequence_index=None,
    num_nodes: Optional[int] = None,
    device=None,
    use_residual: bool = False,
    use_relation_self_loops: bool = True,
    require_temporal_head: bool = False,
    edge_types=None,
    edge_feature_dims=None,
    edge_encoder_hidden_dims=None,
    edge_encoded_dims=None,
    edge_encoder_dropouts=None,
    edge_encoder_kinds=None,
):
    """
    `edge_feature_dim` (scalar) sigue siendo el contrato legacy.
    `edge_feature_dims` (dict {edge_type: int}) permite que cada relación tenga
    su propio `in_dim` raw — necesario tras eliminar el zero-padding entre
    aristas temporales y espaciales.

    `edge_encoder_hidden_dims`, `edge_encoded_dims`, `edge_encoder_dropouts`
    soportan el mismo contrato escalar-o-dict para hiperparámetros por tipo.

    `edge_encoder_kinds` (string o dict {edge_type: kind}) selecciona la clase
    del encoder por arista. Valores válidos: "mlp" (default), "mlp_residual",
    "layernorm_mlp", "time2vec". Permite, p.ej., usar Time2Vec solo en la
    arista temporal y LayerNormMLP en las espaciales.
    """
    variant_cfg = _parse_gnn_variant(gnn_variant)
    model_kwargs = dict(
        in_channels=in_channels,
        hidden_channels=int(hidden_channels),
        out_channels=int(out_channels),
        num_heads=int(num_heads),
        dropout=float(dropout),
        edge_feature_dim=int(edge_feature_dim),
        num_layers=int(num_layers),
        use_checkpointing=bool(use_checkpointing),
        aggr1=aggr1,
        aggr2=aggr2,
        use_residual=bool(use_residual),
        use_relation_self_loops=bool(use_relation_self_loops),
        edge_types=tuple(tuple(et) for et in edge_types) if edge_types is not None else None,
        edge_feature_dims=edge_feature_dims,
    )

    if variant_cfg["encoder_kind"] == "edge_aware":
        model = HeteroEdgeAware(**model_kwargs)
    elif variant_cfg["use_edge_mlp"]:
        raw_edge_dim = int(edge_feature_dim or 0)
        edge_hidden = max(8, raw_edge_dim * 2) if raw_edge_dim > 0 else 8
        # Si `edge_encoded_dims` no se especifica, igualamos al raw por tipo
        # (es el comportamiento histórico aplicado por relación).
        default_encoded_dims = edge_feature_dims if edge_encoded_dims is None else edge_encoded_dims
        model = HeteroGATWithEdgeEncoder(
            **model_kwargs,
            edge_encoder_hidden_dim=edge_hidden,
            edge_encoded_dim=raw_edge_dim,
            edge_encoder_dropout=float(dropout) * 0.5,
            edge_encoder_hidden_dims=edge_encoder_hidden_dims,
            edge_encoded_dims=default_encoded_dims,
            edge_encoder_dropouts=edge_encoder_dropouts,
            edge_encoder_kinds=edge_encoder_kinds,
        )
    else:
        model = HeteroGAT(**model_kwargs)

    model.gnn_variant = variant_cfg["name"]

    if _variant_has_temporal_head(variant_cfg["name"]):
        temporal_head = _build_temporal_head_for_variant(
            gnn_variant=variant_cfg["name"],
            sequence_index=sequence_index,
            num_nodes=int(num_nodes or 0),
            embedding_dim=int(hidden_channels) * int(num_heads),
            num_classes=int(out_channels),
            dropout=float(dropout),
        )
        if temporal_head is not None:
            model.temporal_head = temporal_head
            model.temporal_head.reset_cache()
            model.temporal_head.train()
        elif require_temporal_head:
            raise ValueError(
                f"La variante temporal '{variant_cfg['name']}' requiere SequenceIndex valido; "
                "reconstruye/carga el grafo con metadata temporal antes de entrenar o evaluar."
            )
    elif hasattr(model, 'temporal_head'):
        delattr(model, 'temporal_head')

    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def _prime_temporal_cache_if_needed(model, graph, node_type: str = "pm", context: str = "") -> bool:
    temporal_module = getattr(model, "temporal_head", None)
    if temporal_module is None:
        return False
    cache_strategy = str(getattr(temporal_module, "cache_strategy", "incremental")).strip().lower()
    if cache_strategy != "global_epoch":
        return False

    was_training = bool(model.training)
    prev_checkpointing = getattr(model, "use_checkpointing", None)
    graph_for_cache = graph
    cloned_graph_for_cache = False
    try:
        temporal_module.reset_cache()
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = torch.device("cpu")
        try:
            graph_device = graph[node_type].x.device
        except Exception:
            graph_device = model_device
        if graph_device != model_device:
            graph_for_cache = graph.clone().to(model_device)
            cloned_graph_for_cache = True
        graph_for_cache.edge_attr_dict = {
            et: getattr(graph_for_cache[et], "edge_attr", None)
            for et in getattr(graph_for_cache, "edge_types", [])
        }
        model.eval()
        if prev_checkpointing is not None:
            model.use_checkpointing = False
        z_dict = compute_epoch_embeddings(model, graph_for_cache)
        z_pm = z_dict.get(node_type)
        if z_pm is None:
            return False
        temporal_module.update_cache(z_pm)
        return True
    except Exception as exc:
        try:
            logger.warning(
                f"No se pudo primar el caché temporal global"
                f"{' (' + context + ')' if context else ''}: {exc}. Se usará caché incremental."
            )
        except Exception:
            pass
        return False
    finally:
        if prev_checkpointing is not None:
            try:
                model.use_checkpointing = prev_checkpointing
            except Exception:
                pass
        model.train(was_training)
        try:
            if temporal_module is not None:
                temporal_module.train(was_training)
        except Exception:
            pass
        if cloned_graph_for_cache:
            try:
                del graph_for_cache
            except Exception:
                pass
            try:
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _find_best_model_path(
    use_graphsmote: bool,
    loaded_obj,
    best_params: Optional[dict] = None,
    gnn_variant: Optional[str] = None,
) -> Optional[str]:
    # Try exact tag match first
    want_variant = gnn_variant
    if want_variant is None and isinstance(best_params, dict):
        want_variant = best_params.get("gnn_variant")
    tag = _model_tag_suffix(use_graphsmote, loaded_obj, gnn_variant=want_variant)
    exact = os.path.join(RESULTADOS_DIR, f"gat_model_BEST{tag}.pt")
    if os.path.exists(exact):
        return exact
    # Fallback: filter among any BEST files
    files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "gat_model_BEST*.pt")), key=os.path.getmtime)
    if not files:
        base = os.path.join(RESULTADOS_DIR, "gat_model_BEST.pt")
        return base if os.path.exists(base) else None
    preferred = []
    want_variant_component = _gnn_variant_tag_component(want_variant)
    for f in files:
        name = os.path.basename(f)
        if use_graphsmote and "_GraphSMOTE" not in name:
            continue
        if (not use_graphsmote) and "_GraphSMOTE" in name:
            continue
        if _has_imgagn(loaded_obj) and "_ImGAGN" not in name:
            continue
        if "_GNN_" in name and want_variant_component not in name:
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
        'use_residual': any(str(k).startswith("residual_lins.") for k in sd.keys()),
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
        gnn_variant_meta = None
        try:
            gnn_variant_meta = meta.get('gnn_variant')
        except Exception:
            gnn_variant_meta = None
        if gnn_variant_meta:
            variant.append(str(gnn_variant_meta))
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
    """Infer the maximum edge attribute width across relations. Mantiene el
    contrato escalar legacy; para la versión por tipo usar
    `_detect_edge_feature_dims`."""
    best = 0
    try:
        if hasattr(data, 'edge_types'):
            for edge_type in data.edge_types:
                edge_attr = getattr(data[edge_type], 'edge_attr', None)
                if edge_attr is not None:
                    best = max(best, int(edge_attr.shape[1]))
    except Exception:
        pass
    return int(best)


def _detect_edge_feature_dims(data) -> dict:
    """Per-edge-type raw `in_dim` map. Aristas sin `edge_attr` reportan 0."""
    dims: dict = {}
    if not hasattr(data, 'edge_types'):
        return dims
    for edge_type in data.edge_types:
        attr = getattr(data[edge_type], 'edge_attr', None)
        if attr is None:
            dims[tuple(edge_type)] = 0
        elif attr.dim() > 1:
            dims[tuple(edge_type)] = int(attr.shape[1])
        else:
            dims[tuple(edge_type)] = 1
    return dims


def _parse_edge_encoder_per_type(spec) -> dict:
    """Convert `edge_encoder_per_type` (que llega como JSON-string desde el
    CSV o como dict desde Python) en cuatro mapas `{edge_type: value}` para
    pasar a `_build_gnn_model` como `edge_encoder_hidden_dims`,
    `edge_encoded_dims`, `edge_encoder_dropouts`, `edge_encoder_kinds`."""
    if not spec:
        return {}
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception:
            return {}
    if not isinstance(spec, dict):
        return {}
    hidden_dims, encoded_dims, dropouts, kinds = {}, {}, {}, {}
    for key, cfg in spec.items():
        if not isinstance(cfg, dict):
            continue
        # Aceptar tanto tuple-keys como strings tipo "('pm', 'temporal', 'pm')".
        parsed_key = key
        if isinstance(key, str) and key.startswith("("):
            try:
                parts = [p.strip().strip("'").strip('"') for p in key.strip("()").split(",") if p.strip()]
                if len(parts) == 3:
                    parsed_key = tuple(parts)
            except Exception:
                parsed_key = key
        if "hidden_dim" in cfg and cfg["hidden_dim"] is not None:
            hidden_dims[parsed_key] = int(cfg["hidden_dim"])
        if "encoded_dim" in cfg and cfg["encoded_dim"] is not None:
            encoded_dims[parsed_key] = int(cfg["encoded_dim"])
        if "dropout" in cfg and cfg["dropout"] is not None:
            dropouts[parsed_key] = float(cfg["dropout"])
        if "kind" in cfg and cfg["kind"] is not None:
            kinds[parsed_key] = str(cfg["kind"]).strip().lower()
    return {
        "edge_encoder_hidden_dims": hidden_dims or None,
        "edge_encoded_dims": encoded_dims or None,
        "edge_encoder_dropouts": dropouts or None,
        "edge_encoder_kinds": kinds or None,
    }

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
        out = {}
        for edge_type in edge_types:
            prof = parsed.get(edge_type)
            # Fallback: lookup por nombre de relación (edge_type[1]) para soportar
            # dicts con claves string como {'temporal': [...], 'spatial': [...]}.
            # Esto permite que el perfil 'asymmetric' de Optuna sobreviva el roundtrip
            # a CSV (donde las tuplas se pierden) usando JSON con claves string.
            if prof is None and isinstance(edge_type, tuple) and len(edge_type) >= 2:
                prof = parsed.get(edge_type[1])
            out[edge_type] = _normalize_profile(prof, default_profile)
        return out
    fallback = _pick_default(default_value, [15, 10])
    return {edge_type: fallback for edge_type in edge_types}

def _safe_cast(value, cast_type, default):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return cast_type(value)
    except Exception:
        return default

def _safe_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if isinstance(value, float) and np.isnan(value):
                return bool(default)
        except Exception:
            pass
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "s", "si", "sí"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(default)

LR_SCHEDULER_CHOICES = (
    "one_cycle",
    "cosine_warm_restarts",
    "plateau_restart",
)

def _normalize_lr_scheduler_choice(value: object, default: str = "one_cycle") -> str:
    raw = str(value or default).strip().lower()
    key = raw.replace(" ", "_").replace("-", "_")
    aliases = {
        "onecycle": "one_cycle",
        "one_cycle": "one_cycle",
        "one_cycle_lr": "one_cycle",
        "onecyclelr": "one_cycle",
        "cosine": "cosine_warm_restarts",
        "cosine_restart": "cosine_warm_restarts",
        "cosine_restarts": "cosine_warm_restarts",
        "cosine_warm_restart": "cosine_warm_restarts",
        "cosine_warm_restarts": "cosine_warm_restarts",
        "cosineannealingwarmrestarts": "cosine_warm_restarts",
        "cosine_annealing_warm_restarts": "cosine_warm_restarts",
        "plateau": "plateau_restart",
        "plateau_restart": "plateau_restart",
        "reduce_on_plateau": "plateau_restart",
        "reduce_lr_on_plateau": "plateau_restart",
        "reducelronplateau": "plateau_restart",
    }
    normalized = aliases.get(key, default)
    return normalized if normalized in LR_SCHEDULER_CHOICES else default

def _optimizer_steps_per_epoch(loader, accumulation_steps: int = 1) -> int:
    try:
        loader_len = max(1, int(len(loader)))
    except Exception:
        loader_len = 1
    try:
        accum = max(1, int(accumulation_steps))
    except Exception:
        accum = 1
    return max(1, int(math.ceil(loader_len / accum)))

def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: object,
    *,
    max_lr: float,
    steps_per_epoch: int,
    epochs: int,
    monitor_mode: str = "min",
):
    scheduler_key = _normalize_lr_scheduler_choice(scheduler_name)
    steps = max(1, int(steps_per_epoch))
    total_epochs = max(1, int(epochs))
    lr_value = max(float(max_lr), 1e-12)
    if scheduler_key == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr_value,
            steps_per_epoch=steps,
            epochs=total_epochs,
            cycle_momentum=isinstance(optimizer, (torch.optim.AdamW, torch.optim.NAdam)),
        )
    if scheduler_key == "cosine_warm_restarts":
        restart_epochs = max(1, min(10, max(1, total_epochs // 3)))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, steps * restart_epochs),
            T_mult=2,
            eta_min=lr_value * 0.01,
        )
    mode = "min" if str(monitor_mode or "min").lower() == "min" else "max"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=0.5,
        patience=max(1, min(10, total_epochs // 10 or 1)),
        threshold=1e-4,
    )

def _lr_scheduler_steps_per_batch(scheduler_name: object) -> bool:
    return _normalize_lr_scheduler_choice(scheduler_name) in {
        "one_cycle",
        "cosine_warm_restarts",
    }

def _scheduler_lr_value(scheduler, optimizer: torch.optim.Optimizer) -> Optional[float]:
    try:
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            values = scheduler.get_last_lr()
            if values:
                return float(values[0])
    except Exception:
        pass
    try:
        return float(optimizer.param_groups[0].get("lr"))
    except Exception:
        return None

def _make_exhaustive_num_neighbors(edge_types, num_layers: int) -> dict:
    layers = max(int(num_layers), 1)
    profile = [-1] * layers
    return {edge_type: profile for edge_type in edge_types}

def _build_pm_homogeneous_view(graph, node_type: str = "pm") -> Optional[Data]:
    try:
        if node_type not in graph.node_types:
            return None
        num_nodes = int(graph[node_type].num_nodes)
        if num_nodes <= 0:
            return None
        edge_chunks = []
        for edge_type in graph.edge_types:
            src, _, dst = edge_type
            if src != node_type or dst != node_type:
                continue
            edge_index = getattr(graph[edge_type], "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                continue
            edge_chunks.append(edge_index.cpu())
        if not edge_chunks:
            return None
        edge_index = torch.cat(edge_chunks, dim=1)
        view = Data(num_nodes=num_nodes, edge_index=edge_index)
        view.orig_nid = torch.arange(num_nodes, dtype=torch.long)
        return view
    except Exception:
        return None

def _shutdown_torch_loader_iterator(iterator: object) -> None:
    if iterator is None:
        return
    try:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass

def _coerce_pm_node_ids(raw_nodes: object, total_nodes: int) -> torch.Tensor:
    if torch.is_tensor(raw_nodes):
        node_ids = raw_nodes.detach().long().view(-1).cpu()
    else:
        node_ids = torch.as_tensor(raw_nodes, dtype=torch.long).view(-1).cpu()
    if node_ids.numel() == 0:
        return node_ids

    filtered: List[int] = []
    seen = set()
    upper = int(total_nodes)
    for raw in node_ids.tolist():
        idx = int(raw)
        if idx < 0 or idx >= upper or idx in seen:
            continue
        seen.add(idx)
        filtered.append(idx)
    return torch.as_tensor(filtered, dtype=torch.long)

def _pm_induced_hetero_batch(
    graph_cpu: HeteroData,
    pm_nodes_raw: object,
    *,
    supervised_nodes: torch.Tensor,
    node_type: str = "pm",
) -> Tuple[Optional[HeteroData], Optional[str]]:
    if node_type not in graph_cpu.node_types:
        return None, f"No se encontró tipo de nodo '{node_type}'."
    try:
        total_pm = int(getattr(graph_cpu[node_type], "num_nodes", graph_cpu[node_type].x.size(0)))
    except Exception:
        return None, f"No se pudo resolver num_nodes para '{node_type}'."

    pm_nodes = _coerce_pm_node_ids(pm_nodes_raw, total_pm)
    if pm_nodes.numel() == 0:
        return None, "Subgrafo nativo sin nodos PM válidos."

    seed_nodes = _coerce_pm_node_ids(supervised_nodes, total_pm)
    if seed_nodes.numel() == 0:
        return None, "No hay semillas supervisadas para el sampler nativo."
    seed_mask = torch.zeros(total_pm, dtype=torch.bool)
    seed_mask[seed_nodes] = True
    supervised_mask = seed_mask[pm_nodes]
    supervised_count = int(supervised_mask.sum().item())
    if supervised_count <= 0:
        return None, "Subgrafo nativo sin nodos PM supervisados."

    if supervised_count < int(pm_nodes.numel()):
        pm_nodes = torch.cat([pm_nodes[supervised_mask], pm_nodes[~supervised_mask]], dim=0)

    out = HeteroData()
    pm_store = graph_cpu[node_type]
    out[node_type].x = pm_store.x[pm_nodes].cpu()
    if hasattr(pm_store, "y") and pm_store.y is not None:
        out[node_type].y = pm_store.y[pm_nodes].cpu()

    copied_node_attrs = {"x", "y"}
    try:
        node_items = list(pm_store.items())
    except Exception:
        node_items = []
    for attr_name, attr_value in node_items:
        if attr_name in copied_node_attrs:
            continue
        if torch.is_tensor(attr_value) and attr_value.size(0) >= total_pm:
            out[node_type][attr_name] = attr_value[pm_nodes].cpu()

    out[node_type].num_nodes = int(pm_nodes.numel())
    out[node_type].n_id = pm_nodes.clone()
    out[node_type].batch_size = int(supervised_count)
    supervised_batch_mask = torch.zeros(int(pm_nodes.numel()), dtype=torch.bool)
    supervised_batch_mask[:supervised_count] = True
    out[node_type].supervised_mask = supervised_batch_mask

    global_to_local = torch.full((total_pm,), -1, dtype=torch.long)
    global_to_local[pm_nodes] = torch.arange(pm_nodes.numel(), dtype=torch.long)

    pm_edge_types = [
        edge_type
        for edge_type in graph_cpu.edge_types
        if edge_type[0] == node_type and edge_type[2] == node_type
    ]
    for edge_type in pm_edge_types:
        edge_index = getattr(graph_cpu[edge_type], "edge_index", None)
        if edge_index is None:
            out[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            continue

        edge_index_cpu = edge_index.long().cpu()
        edge_attr = getattr(graph_cpu[edge_type], "edge_attr", None)
        if edge_index_cpu.numel() == 0:
            out[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            if torch.is_tensor(edge_attr) and edge_attr.dim() >= 2:
                out[edge_type].edge_attr = torch.zeros(
                    (0, int(edge_attr.size(1))),
                    dtype=edge_attr.dtype,
                )
            continue

        src_local = global_to_local[edge_index_cpu[0]]
        dst_local = global_to_local[edge_index_cpu[1]]
        keep = (src_local >= 0) & (dst_local >= 0)
        local_edge_index = torch.stack([src_local[keep], dst_local[keep]], dim=0)
        out[edge_type].edge_index = local_edge_index

        if not torch.is_tensor(edge_attr):
            continue
        edge_attr_cpu = edge_attr.cpu()
        if edge_attr_cpu.size(0) == edge_index_cpu.size(1):
            out[edge_type].edge_attr = edge_attr_cpu[keep]
        elif edge_attr_cpu.dim() >= 2:
            out[edge_type].edge_attr = torch.zeros(
                (int(local_edge_index.size(1)), int(edge_attr_cpu.size(1))),
                dtype=edge_attr_cpu.dtype,
            )

    try:
        out.graph_metadata = dict(getattr(graph_cpu, "graph_metadata", {}) or {})
    except Exception:
        pass
    return out, None

class _NativeSamplerAsHeteroLoader:
    def __init__(
        self,
        *,
        base_loader: object,
        graph_cpu: HeteroData,
        supervised_nodes: torch.Tensor,
        max_batches: Optional[int],
        deterministic_sampling: bool,
        sampling_seed_value: int,
        sampler_impl: str,
        node_type: str = "pm",
    ) -> None:
        self.base_loader = base_loader
        self.graph_cpu = graph_cpu
        self.supervised_nodes = supervised_nodes.detach().cpu().long().view(-1)
        self.max_batches = (
            int(max_batches)
            if max_batches is not None and int(max_batches) > 0
            else None
        )
        self.deterministic_sampling = bool(deterministic_sampling)
        self.sampling_seed_value = int(sampling_seed_value)
        self.sampler_impl = str(sampler_impl)
        self.node_type = str(node_type)
        self.seen = 0

    def __len__(self) -> int:
        try:
            base_len = int(len(self.base_loader))
        except Exception:
            base_len = 0
        if self.max_batches is None:
            return max(1, base_len) if base_len > 0 else 1
        if base_len <= 0:
            return max(1, int(self.max_batches))
        return max(1, min(base_len, int(self.max_batches)))

    def __iter__(self):
        self.seen = 0
        prev_cpu_state = torch.random.get_rng_state()
        if self.deterministic_sampling:
            torch.manual_seed(int(self.sampling_seed_value))
        base_it = iter(self.base_loader)
        try:
            for sampled in base_it:
                if self.max_batches is not None and self.seen >= int(self.max_batches):
                    break
                pm_nodes = getattr(sampled, "orig_nid", None)
                if pm_nodes is None:
                    pm_nodes = getattr(sampled, "n_id", None)
                if pm_nodes is None:
                    continue
                hetero_batch, _ = _pm_induced_hetero_batch(
                    self.graph_cpu,
                    pm_nodes,
                    supervised_nodes=self.supervised_nodes,
                    node_type=self.node_type,
                )
                if hetero_batch is None:
                    continue
                self.seen += 1
                yield hetero_batch
        finally:
            _shutdown_torch_loader_iterator(base_it)
            if self.deterministic_sampling:
                try:
                    torch.random.set_rng_state(prev_cpu_state)
                except Exception:
                    pass

def _normalize_train_sampler_mode(value: object) -> str:
    key = str(value or "neighbor").strip().lower().replace("-", "_")
    return {
        "neighbor": "neighbor",
        "neighborloader": "neighbor",
        "positive_aware": "positive_aware",
        "positiveaware": "positive_aware",
        "pos_aware": "positive_aware",
        "positive_aware_neighbor": "positive_aware",
        "positive_aware_neighborloader": "positive_aware",
        "cluster_gcn": "cluster_gcn",
        "clustergcn": "cluster_gcn",
        "graphsaint": "graphsaint",
        "rl_top_p": "rl_top_p",
        "riognn": "rl_top_p",
        "rio_gnn": "rl_top_p",
        "rsrl": "rl_top_p",
    }.get(key, "neighbor")


def _pm_index_to_lookup(
    pm_index: Optional[object],
    *,
    num_nodes: int,
) -> Tuple[List[Optional[str]], torch.Tensor, bool]:
    ports: List[Optional[str]] = [None] * int(num_nodes)
    ts_min = torch.full((int(num_nodes),), -1, dtype=torch.long)
    if pm_index is None:
        return ports, ts_min, False

    reverse = getattr(pm_index, "_rev", None)
    if reverse is None and isinstance(pm_index, dict):
        sample_keys = list(pm_index.keys())[:8]
        if sample_keys and all(isinstance(k, (int, np.integer)) for k in sample_keys):
            reverse = pm_index
        else:
            reverse = {}
            for key, idx in pm_index.items():
                try:
                    reverse[int(idx)] = key
                except Exception:
                    continue
    if reverse is None or not hasattr(reverse, "items"):
        return ports, ts_min, False

    found = False
    for idx_raw, value in reverse.items():
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        if idx < 0 or idx >= int(num_nodes):
            continue
        portico = None
        ts_value = None
        if isinstance(value, dict):
            portico = value.get("portico", value.get("portico_id", value.get("pm")))
            ts_value = value.get("ts_min", value.get("timestamp_min", value.get("t")))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            portico = value[0]
            ts_value = value[1]
        if portico is None or ts_value is None:
            continue
        try:
            ts_int = int(round(float(ts_value)))
        except Exception:
            continue
        ports[idx] = str(portico)
        ts_min[idx] = int(ts_int)
        found = True
    return ports, ts_min, bool(found)


def _shuffle_1d(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    values = values.detach().cpu().long().view(-1)
    if values.numel() <= 1:
        return values.clone()
    order = torch.randperm(int(values.numel()), generator=generator)
    return values[order]


def _repeat_shuffled_1d(
    values: torch.Tensor,
    needed: int,
    generator: torch.Generator,
) -> torch.Tensor:
    values = values.detach().cpu().long().view(-1)
    needed = int(max(0, needed))
    if needed <= 0 or values.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    chunks: List[torch.Tensor] = []
    remaining = int(needed)
    while remaining > 0:
        shuffled = _shuffle_1d(values, generator)
        take = min(int(remaining), int(shuffled.numel()))
        chunks.append(shuffled[:take])
        remaining -= take
    return torch.cat(chunks).long() if chunks else torch.empty(0, dtype=torch.long)


def _positive_aware_spatial_hard_negatives(
    graph_cpu: HeteroData,
    *,
    node_type: str,
    train_neg_mask: torch.Tensor,
    pos_mask: torch.Tensor,
    ts_min: torch.Tensor,
    has_pm_index: bool,
) -> torch.Tensor:
    candidates: List[torch.Tensor] = []
    for edge_type in graph_cpu.edge_types:
        if not isinstance(edge_type, tuple) or len(edge_type) < 3:
            continue
        src_type, rel, dst_type = edge_type
        if src_type != node_type or dst_type != node_type:
            continue
        if "spatial" not in str(rel).strip().lower():
            continue
        edge_index = getattr(graph_cpu[edge_type], "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            continue
        edge_index = edge_index.detach().cpu().long()
        src = edge_index[0]
        dst = edge_index[1]
        valid = torch.ones(src.numel(), dtype=torch.bool)
        if has_pm_index and ts_min.numel() > 0:
            src_ts = ts_min[src]
            dst_ts = ts_min[dst]
            valid = (src_ts >= 0) & (src_ts == dst_ts)
        src_pos_dst_neg = valid & pos_mask[src] & train_neg_mask[dst]
        dst_pos_src_neg = valid & pos_mask[dst] & train_neg_mask[src]
        if src_pos_dst_neg.any():
            candidates.append(dst[src_pos_dst_neg])
        if dst_pos_src_neg.any():
            candidates.append(src[dst_pos_src_neg])
    if not candidates:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat(candidates).long())


def _positive_aware_temporal_hard_negatives(
    *,
    pos_idx: torch.Tensor,
    neg_idx: torch.Tensor,
    ports: List[Optional[str]],
    ts_min: torch.Tensor,
    window_minutes: int,
) -> torch.Tensor:
    if pos_idx.numel() == 0 or neg_idx.numel() == 0 or ts_min.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    by_port: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    rows_by_port: Dict[str, List[Tuple[int, int]]] = {}
    for idx in neg_idx.detach().cpu().long().tolist():
        port = ports[int(idx)] if 0 <= int(idx) < len(ports) else None
        ts_value = int(ts_min[int(idx)].item()) if 0 <= int(idx) < int(ts_min.numel()) else -1
        if port is None or ts_value < 0:
            continue
        rows_by_port.setdefault(str(port), []).append((ts_value, int(idx)))
    for port, rows in rows_by_port.items():
        rows.sort(key=lambda item: item[0])
        by_port[port] = (
            np.asarray([item[0] for item in rows], dtype=np.int64),
            np.asarray([item[1] for item in rows], dtype=np.int64),
        )

    window = max(0, int(window_minutes))
    selected: List[np.ndarray] = []
    for pos in pos_idx.detach().cpu().long().tolist():
        pos_i = int(pos)
        port = ports[pos_i] if 0 <= pos_i < len(ports) else None
        ts_value = int(ts_min[pos_i].item()) if 0 <= pos_i < int(ts_min.numel()) else -1
        if port is None or ts_value < 0:
            continue
        pair = by_port.get(str(port))
        if pair is None:
            continue
        times, nodes = pair
        lo = int(np.searchsorted(times, ts_value - window, side="left"))
        hi = int(np.searchsorted(times, ts_value + window, side="right"))
        if hi > lo:
            selected.append(nodes[lo:hi])
    if not selected:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.as_tensor(np.concatenate(selected), dtype=torch.long))


def build_positive_aware_seed_order(
    graph_cpu: HeteroData,
    *,
    pm_index: Optional[object] = None,
    batch_size: int,
    sampling_seed: int,
    epoch: int = 1,
    target_positive_fraction: float = 0.02,
    hard_negative_window_minutes: int = 60,
    hard_negatives_per_positive: int = 4,
    node_type: str = "pm",
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Build a deterministic, positive-aware seed order for NeighborLoader."""
    if node_type not in graph_cpu.node_types:
        return torch.empty(0, dtype=torch.long), {"fallback_reason": "missing_node_type"}
    if not hasattr(graph_cpu[node_type], "train_mask"):
        return torch.empty(0, dtype=torch.long), {"fallback_reason": "missing_train_mask"}

    train_mask = graph_cpu[node_type].train_mask.detach().cpu().bool().view(-1)
    y = getattr(graph_cpu[node_type], "y", None)
    if y is None:
        train_idx = train_mask.nonzero(as_tuple=False).view(-1).long()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(sampling_seed) + int(epoch) * 9973)
        return _shuffle_1d(train_idx, gen), {
            "fallback_reason": "missing_labels",
            "train_seed_count": int(train_idx.numel()),
        }
    y = y.detach().cpu().view(-1).long()
    num_nodes = int(train_mask.numel())
    train_idx = train_mask.nonzero(as_tuple=False).view(-1).long()
    pos_idx = (train_mask & (y == 1)).nonzero(as_tuple=False).view(-1).long()
    neg_idx = (train_mask & (y != 1)).nonzero(as_tuple=False).view(-1).long()

    gen = torch.Generator(device="cpu")
    effective_seed = int(sampling_seed) + int(epoch) * 9973
    gen.manual_seed(int(effective_seed))

    batch_size = max(1, int(batch_size))
    target_fraction = float(target_positive_fraction)
    if not math.isfinite(target_fraction) or target_fraction <= 0.0:
        target_fraction = 0.02
    target_fraction = max(0.0, min(0.5, float(target_fraction)))
    pos_per_batch = max(1, int(round(float(batch_size) * target_fraction)))
    pos_per_batch = min(int(batch_size), int(pos_per_batch))
    neg_slots = max(0, int(batch_size) - int(pos_per_batch))

    stats: Dict[str, object] = {
        "sampler_impl": "positive_aware_neighbor",
        "sampling_seed": int(sampling_seed),
        "effective_seed": int(effective_seed),
        "epoch": int(epoch),
        "batch_size": int(batch_size),
        "target_positive_fraction": float(target_fraction),
        "pos_per_batch": int(pos_per_batch),
        "train_node_count": int(train_idx.numel()),
        "train_positive_count": int(pos_idx.numel()),
        "train_negative_count": int(neg_idx.numel()),
        "hard_negative_window_minutes": int(max(0, int(hard_negative_window_minutes))),
        "hard_negatives_per_positive": int(max(0, int(hard_negatives_per_positive))),
    }

    if train_idx.numel() == 0:
        stats["fallback_reason"] = "empty_train_mask"
        return train_idx, stats
    if pos_idx.numel() == 0:
        seed_order = _shuffle_1d(train_idx, gen)
        stats.update(
            {
                "fallback_reason": "no_train_positives",
                "batch_count": int(math.ceil(float(seed_order.numel()) / float(batch_size))),
                "seed_slot_count": int(seed_order.numel()),
                "positive_seed_slots": 0,
                "negative_seed_slots": int(seed_order.numel()),
            }
        )
        return seed_order, stats
    if neg_idx.numel() == 0 or neg_slots == 0:
        seed_order = _repeat_shuffled_1d(pos_idx, int(pos_idx.numel()), gen)
        stats.update(
            {
                "fallback_reason": "no_train_negatives",
                "batch_count": int(math.ceil(float(seed_order.numel()) / float(batch_size))),
                "seed_slot_count": int(seed_order.numel()),
                "positive_seed_slots": int(seed_order.numel()),
                "negative_seed_slots": 0,
            }
        )
        return seed_order, stats

    ports, ts_min, has_pm_index = _pm_index_to_lookup(pm_index, num_nodes=num_nodes)
    train_neg_mask = train_mask & (y != 1)
    pos_mask = train_mask & (y == 1)
    temporal_hard = (
        _positive_aware_temporal_hard_negatives(
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            ports=ports,
            ts_min=ts_min,
            window_minutes=int(hard_negative_window_minutes),
        )
        if has_pm_index
        else torch.empty(0, dtype=torch.long)
    )
    spatial_hard = (
        _positive_aware_spatial_hard_negatives(
            graph_cpu,
            node_type=node_type,
            train_neg_mask=train_neg_mask,
            pos_mask=pos_mask,
            ts_min=ts_min,
            has_pm_index=has_pm_index,
        )
        if has_pm_index
        else torch.empty(0, dtype=torch.long)
    )
    hard_idx = torch.unique(torch.cat([temporal_hard, spatial_hard]).long()) if (
        temporal_hard.numel() or spatial_hard.numel()
    ) else torch.empty(0, dtype=torch.long)
    if hard_idx.numel() > 0:
        hard_idx = hard_idx[train_neg_mask[hard_idx]]

    hard_slots = 0
    if hard_idx.numel() > 0:
        hard_slots = min(
            int(neg_slots),
            int(max(0, int(hard_negatives_per_positive))) * int(pos_per_batch),
        )
    easy_slots = int(neg_slots) - int(hard_slots)
    if easy_slots <= 0 and neg_idx.numel() > 0:
        if hard_slots > 0:
            hard_slots -= 1
        easy_slots = int(neg_slots) - int(hard_slots)

    is_hard = torch.zeros(num_nodes, dtype=torch.bool)
    if hard_idx.numel() > 0:
        is_hard[hard_idx] = True
    easy_idx = neg_idx[~is_hard[neg_idx]]
    if easy_idx.numel() == 0:
        easy_idx = neg_idx
    if hard_slots <= 0:
        hard_idx = torch.empty(0, dtype=torch.long)
        easy_slots = int(neg_slots)

    batches_from_neg = int(math.ceil(float(neg_idx.numel()) / float(max(1, neg_slots))))
    batches_from_pos = int(math.ceil(float(pos_idx.numel()) / float(max(1, pos_per_batch))))
    batch_count = max(1, batches_from_neg, batches_from_pos)
    total_pos_slots = int(batch_count) * int(pos_per_batch)
    total_hard_slots = int(batch_count) * int(hard_slots)
    total_easy_slots = int(batch_count) * int(max(0, easy_slots))

    pos_order = _repeat_shuffled_1d(pos_idx, total_pos_slots, gen)
    hard_order = _repeat_shuffled_1d(hard_idx, total_hard_slots, gen)
    easy_order = _repeat_shuffled_1d(easy_idx, total_easy_slots, gen)
    fill_order = _repeat_shuffled_1d(neg_idx, int(batch_count) * int(neg_slots), gen)

    batches: List[torch.Tensor] = []
    pos_ptr = 0
    hard_ptr = 0
    easy_ptr = 0
    fill_ptr = 0
    for _ in range(int(batch_count)):
        parts: List[torch.Tensor] = []
        if pos_per_batch > 0:
            parts.append(pos_order[pos_ptr: pos_ptr + pos_per_batch])
            pos_ptr += int(pos_per_batch)
        if hard_slots > 0:
            parts.append(hard_order[hard_ptr: hard_ptr + hard_slots])
            hard_ptr += int(hard_slots)
        if easy_slots > 0:
            parts.append(easy_order[easy_ptr: easy_ptr + easy_slots])
            easy_ptr += int(easy_slots)
        batch = torch.cat([part for part in parts if part.numel() > 0]).long()
        if batch.numel() < batch_size:
            need = int(batch_size) - int(batch.numel())
            if fill_ptr + need > int(fill_order.numel()):
                fill_order = torch.cat([fill_order, _repeat_shuffled_1d(neg_idx, need, gen)])
            batch = torch.cat([batch, fill_order[fill_ptr: fill_ptr + need].long()])
            fill_ptr += need
        if batch.numel() > 1:
            batch = batch[torch.randperm(int(batch.numel()), generator=gen)]
        batches.append(batch.long())
    seed_order = torch.cat(batches).long() if batches else train_idx.long()

    positive_slots = int((y[seed_order] == 1).sum().item()) if seed_order.numel() else 0
    negative_slots = int(seed_order.numel()) - positive_slots
    stats.update(
        {
            "pm_index_available": bool(has_pm_index),
            "temporal_hard_negative_candidates": int(temporal_hard.numel()),
            "spatial_hard_negative_candidates": int(spatial_hard.numel()),
            "hard_negative_candidates": int(hard_idx.numel()),
            "hard_negatives_per_batch": int(hard_slots),
            "random_negatives_per_batch": int(max(0, easy_slots)),
            "batch_count": int(batch_count),
            "seed_slot_count": int(seed_order.numel()),
            "unique_seed_count": int(torch.unique(seed_order).numel()) if seed_order.numel() else 0,
            "positive_seed_slots": int(positive_slots),
            "negative_seed_slots": int(negative_slots),
            "actual_positive_fraction": (
                float(positive_slots) / float(max(1, int(seed_order.numel())))
            ),
            "fallback_reason": None if bool(has_pm_index) else "missing_pm_index_hard_negatives_random",
        }
    )
    return seed_order, stats


def _build_native_sampler_loader(
    *,
    graph_cpu: HeteroData,
    sampler_config: Dict[str, object],
    batch_size: int,
    sampling_seed: int,
    base_seeds: Optional[torch.Tensor] = None,
    num_neighbors_cfg: Optional[object] = None,
    deterministic: Optional[bool] = None,
    node_type: str = "pm",
    scale_graphsaint_batch_with_loader_batch: bool = False,
) -> Tuple[Optional[object], Optional[str]]:
    if node_type not in graph_cpu.node_types:
        return None, f"No se encontró tipo de nodo '{node_type}'."
    if not hasattr(graph_cpu[node_type], "train_mask"):
        return None, f"No se encontró train_mask para '{node_type}'."

    cfg = dict(sampler_config or {})
    mode = _normalize_train_sampler_mode(cfg.get("train_sampler_mode", "neighbor"))
    if deterministic is None:
        deterministic = _safe_bool(cfg.get("deterministic_sampling"), True)
    seed = int(_safe_cast(cfg.get("sampling_seed"), int, int(sampling_seed)))

    if base_seeds is None:
        base_seeds = graph_cpu[node_type].train_mask.nonzero(as_tuple=False).view(-1).cpu()
    else:
        base_seeds = base_seeds.detach().cpu().long().view(-1)
    if base_seeds.numel() == 0:
        return None, "No hay semillas de train disponibles."

    try:
        resolved_neighbors = (
            num_neighbors_cfg
            if num_neighbors_cfg is not None
            else _resolve_num_neighbors(cfg.get("num_neighbors"), NUM_NEIGHBORS, graph_cpu.edge_types)
        )
    except Exception:
        resolved_neighbors = NUM_NEIGHBORS

    if mode == "rl_top_p":
        try:
            controller = cfg.get("rl_sampler_controller")
            if controller is None:
                num_layers_cfg = int(_safe_cast(cfg.get("num_layers"), int, 2))
                if isinstance(resolved_neighbors, dict):
                    layer_lengths = [
                        len(v)
                        for v in resolved_neighbors.values()
                        if isinstance(v, (list, tuple)) and len(v) > 0
                    ]
                    if layer_lengths:
                        num_layers_cfg = max(layer_lengths)
                elif isinstance(resolved_neighbors, (list, tuple)) and len(resolved_neighbors) > 0:
                    num_layers_cfg = len(resolved_neighbors)
                controller = RioGNNThresholdController(
                    edge_types=[
                        edge_type
                        for edge_type in graph_cpu.edge_types
                        if edge_type[0] == node_type and edge_type[2] == node_type
                    ],
                    num_layers=max(1, int(num_layers_cfg)),
                    in_channels=int(graph_cpu[node_type].x.size(1)),
                    max_degree_by_edge_type=relation_max_degrees(graph_cpu, node_type=node_type),
                    action_space=str(cfg.get("rl_action_space", "discrete")),
                    initial_p=float(_safe_cast(cfg.get("rl_initial_p"), float, 0.5)),
                    min_p=float(_safe_cast(cfg.get("rl_min_p"), float, 0.05)),
                    max_p=float(_safe_cast(cfg.get("rl_max_p"), float, 1.0)),
                    min_keep=int(_safe_cast(cfg.get("rl_min_keep"), int, 1)),
                    switch_patience=int(_safe_cast(cfg.get("rl_switch_patience"), int, 3)),
                    backtracking=_safe_bool(cfg.get("rl_backtracking"), True),
                    positive_only=_safe_bool(cfg.get("rl_positive_only"), True),
                    secondary_reward_weight=float(_safe_cast(cfg.get("rl_secondary_reward_weight"), float, 0.25)),
                    seed=int(seed),
                )
            loader = RLTopPHeteroLoader(
                graph_cpu,
                controller=controller,
                input_nodes=base_seeds,
                batch_size=max(1, int(batch_size)),
                num_neighbors_cfg=resolved_neighbors,
                node_type=node_type,
                shuffle=True,
                deterministic=bool(deterministic),
                sampling_seed=int(seed),
            )
            setattr(loader, "sampler_impl", "rl_top_p_rsrl")
            setattr(loader, "rl_sampler_controller", controller)
            return loader, None
        except Exception as exc:
            return None, f"RioGNN top-p RL falló: {exc}"

    if mode == "positive_aware":
        try:
            seed_order, sampler_stats = build_positive_aware_seed_order(
                graph_cpu,
                pm_index=cfg.get("pm_index"),
                batch_size=max(1, int(batch_size)),
                sampling_seed=int(seed),
                epoch=int(_safe_cast(cfg.get("positive_sampler_epoch"), int, 1)),
                target_positive_fraction=float(
                    _safe_cast(
                        cfg.get("positive_sampler_target_fraction"),
                        float,
                        0.02,
                    )
                ),
                hard_negative_window_minutes=int(
                    _safe_cast(
                        cfg.get("positive_sampler_hard_window_minutes"),
                        int,
                        60,
                    )
                ),
                hard_negatives_per_positive=int(
                    _safe_cast(
                        cfg.get("positive_sampler_hard_negatives_per_positive"),
                        int,
                        4,
                    )
                ),
                node_type=node_type,
            )
            if seed_order.numel() == 0:
                return None, "Positive-aware no encontró semillas de train disponibles."
            loader = NeighborLoader(
                graph_cpu,
                input_nodes=(node_type, seed_order.detach().cpu().long()),
                num_neighbors=resolved_neighbors,
                batch_size=max(1, int(batch_size)),
                shuffle=False,
            )
            try:
                setattr(loader, "sampler_impl", "positive_aware_neighbor")
                setattr(loader, "positive_sampler_stats", _json_safe(sampler_stats))
                setattr(loader, "positive_sampler_seed_count", int(seed_order.numel()))
            except Exception:
                pass
            return loader, None
        except Exception as exc:
            return None, f"Positive-aware NeighborLoader falló: {exc}"

    if mode == "neighbor":
        try:
            loader_gen = None
            if bool(deterministic):
                loader_gen = torch.Generator(device="cpu")
                loader_gen.manual_seed(int(seed))
            loader = NeighborLoader(
                graph_cpu,
                input_nodes=(node_type, base_seeds),
                num_neighbors=resolved_neighbors,
                batch_size=max(1, int(batch_size)),
                shuffle=True,
                generator=loader_gen,
            )
            try:
                setattr(loader, "sampler_impl", "neighbor_native")
            except Exception:
                pass
            return loader, None
        except Exception as exc:
            return None, str(exc)

    pm_view = _build_pm_homogeneous_view(graph_cpu, node_type=node_type)
    if pm_view is None:
        return None, "No se pudo construir vista homogénea PM para sampler nativo."

    if mode == "cluster_gcn":
        try:
            num_parts = max(2, int(cfg.get("cluster_gcn_num_parts", 64)))
            parts_per_epoch = int(cfg.get("cluster_gcn_parts_per_epoch", 0))
            cluster_data = ClusterData(
                pm_view,
                num_parts=int(num_parts),
                recursive=False,
                log=False,
            )
            avg_nodes_per_part = max(
                1,
                int(math.ceil(float(int(pm_view.num_nodes)) / float(int(num_parts)))),
            )
            target_nodes = max(1, int(batch_size))
            parts_per_batch = max(
                1,
                int(round(float(target_nodes) / float(avg_nodes_per_part))),
            )
            parts_per_batch = max(1, min(int(parts_per_batch), int(num_parts)))
            cluster_loader = ClusterLoader(
                cluster_data,
                batch_size=int(parts_per_batch),
                shuffle=not bool(deterministic),
            )
            max_cluster_batches = None
            if int(parts_per_epoch) > 0:
                max_cluster_batches = max(
                    1,
                    int(math.ceil(float(int(parts_per_epoch)) / float(int(parts_per_batch)))),
                )
            native_loader = _NativeSamplerAsHeteroLoader(
                base_loader=cluster_loader,
                graph_cpu=graph_cpu,
                supervised_nodes=base_seeds,
                max_batches=max_cluster_batches,
                deterministic_sampling=bool(deterministic),
                sampling_seed_value=int(seed),
                sampler_impl="cluster_gcn_native",
                node_type=node_type,
            )
            setattr(native_loader, "native_parts_per_batch", int(parts_per_batch))
            return native_loader, None
        except Exception as exc:
            return None, f"Cluster-GCN nativo falló: {exc}"

    try:
        saint_mode = str(cfg.get("graphsaint_mode", "node")).strip().lower().replace("-", "_")
        profile_batch_size = int(cfg.get("graphsaint_batch_size", 0))
        if profile_batch_size > 0:
            if bool(scale_graphsaint_batch_with_loader_batch):
                scale = float(max(1, int(batch_size))) / float(max(1, int(BATCH_SIZE)))
                effective_saint_batch = max(1, int(round(float(profile_batch_size) * scale)))
            else:
                effective_saint_batch = int(profile_batch_size)
        else:
            effective_saint_batch = max(1, int(batch_size))
        effective_saint_batch = min(
            int(max(1, int(pm_view.num_nodes))),
            int(effective_saint_batch),
        )
        saint_steps = max(1, int(cfg.get("graphsaint_num_steps", 8)))
        saint_walk = max(1, int(cfg.get("graphsaint_walk_length", 2)))

        if saint_mode == "edge":
            saint_loader = GraphSAINTEdgeSampler(
                pm_view,
                batch_size=int(effective_saint_batch),
                num_steps=int(saint_steps),
                log=False,
            )
        elif saint_mode in {"random_walk", "randomwalk", "rw"}:
            saint_mode = "random_walk"
            saint_loader = GraphSAINTRandomWalkSampler(
                pm_view,
                batch_size=int(effective_saint_batch),
                walk_length=int(saint_walk),
                num_steps=int(saint_steps),
                log=False,
            )
        else:
            saint_mode = "node"
            saint_loader = GraphSAINTNodeSampler(
                pm_view,
                batch_size=int(effective_saint_batch),
                num_steps=int(saint_steps),
                log=False,
            )
        native_loader = _NativeSamplerAsHeteroLoader(
            base_loader=saint_loader,
            graph_cpu=graph_cpu,
            supervised_nodes=base_seeds,
            max_batches=None,
            deterministic_sampling=bool(deterministic),
            sampling_seed_value=int(seed),
            sampler_impl=f"graphsaint_native_{saint_mode}",
            node_type=node_type,
        )
        setattr(native_loader, "native_graphsaint_batch_size", int(effective_saint_batch))
        return native_loader, None
    except Exception as exc:
        return None, f"GraphSAINT nativo falló: {exc}"

def _cluster_gcn_seed_order(
    pm_view: Data,
    base_seeds: torch.Tensor,
    *,
    num_parts: int,
    clusters_per_epoch: int,
    deterministic: bool,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    if base_seeds.numel() == 0:
        return base_seeds
    try:
        cluster_data = ClusterData(
            pm_view,
            num_parts=max(int(num_parts), 2),
            recursive=False,
            log=False,
        )
        partptr = cluster_data.partition.partptr.cpu()
        node_perm = cluster_data.partition.node_perm.cpu()
        n_parts = int(partptr.numel() - 1)
        if n_parts <= 0:
            return base_seeds

        part_ids = torch.arange(n_parts)
        if deterministic:
            gen = torch.Generator()
            gen.manual_seed(int(seed) + int(epoch))
            part_ids = part_ids[torch.randperm(n_parts, generator=gen)]
        else:
            part_ids = part_ids[torch.randperm(n_parts)]

        keep_parts = int(clusters_per_epoch) if int(clusters_per_epoch) > 0 else n_parts
        part_ids = part_ids[: min(keep_parts, n_parts)]

        seed_mask = torch.zeros(pm_view.num_nodes, dtype=torch.bool)
        seed_mask[base_seeds.cpu()] = True
        selected = []
        for part_id in part_ids.tolist():
            lo = int(partptr[part_id].item())
            hi = int(partptr[part_id + 1].item())
            nodes = node_perm[lo:hi]
            nodes = nodes[seed_mask[nodes]]
            if nodes.numel() > 0:
                selected.append(nodes)

        if not selected:
            return base_seeds
        return torch.cat(selected, dim=0).to(base_seeds.device)
    except Exception as exc:
        logger.warning(f"No se pudo aplicar sampler Cluster-GCN: {exc}. Se usará NeighborLoader.")
        return base_seeds

def _graphsaint_seed_sample(
    pm_view: Data,
    base_seeds: torch.Tensor,
    *,
    mode: str,
    batch_size: int,
    num_steps: int,
    walk_length: int,
    deterministic: bool,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    if base_seeds.numel() == 0:
        return base_seeds
    try:
        sample_mode = str(mode or "node").strip().lower()
        saint_data = pm_view.clone()
        saint_data.train_seed_mask = torch.zeros(saint_data.num_nodes, dtype=torch.bool)
        saint_data.train_seed_mask[base_seeds.cpu()] = True

        prev_cpu_state = torch.random.get_rng_state()
        if deterministic:
            torch.manual_seed(int(seed) + int(epoch))
        try:
            if sample_mode == "edge":
                saint_loader = GraphSAINTEdgeSampler(
                    saint_data,
                    batch_size=max(int(batch_size), 1),
                    num_steps=max(int(num_steps), 1),
                    log=False,
                )
            elif sample_mode in {"random_walk", "randomwalk", "rw"}:
                saint_loader = GraphSAINTRandomWalkSampler(
                    saint_data,
                    batch_size=max(int(batch_size), 1),
                    walk_length=max(int(walk_length), 1),
                    num_steps=max(int(num_steps), 1),
                    log=False,
                )
            else:
                saint_loader = GraphSAINTNodeSampler(
                    saint_data,
                    batch_size=max(int(batch_size), 1),
                    num_steps=max(int(num_steps), 1),
                    log=False,
                )

            seen = torch.zeros(saint_data.num_nodes, dtype=torch.bool)
            ordered = []
            for sampled in saint_loader:
                orig_nid = getattr(sampled, "orig_nid", None)
                mask = getattr(sampled, "train_seed_mask", None)
                if orig_nid is None or mask is None:
                    continue
                cand = orig_nid[mask.bool()].cpu()
                if cand.numel() == 0:
                    continue
                cand = cand[~seen[cand]]
                if cand.numel() == 0:
                    continue
                seen[cand] = True
                ordered.append(cand)

            if not ordered:
                return base_seeds
            return torch.cat(ordered, dim=0).to(base_seeds.device)
        finally:
            torch.random.set_rng_state(prev_cpu_state)
    except Exception as exc:
        logger.warning(f"No se pudo aplicar sampler GraphSAINT: {exc}. Se usará NeighborLoader.")
        return base_seeds

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

def run_gnn_anomaly_pipeline(
    loaded_obj,
    *,
    model_path: Optional[str] = None,
    model_index: Optional[int] = None,
):
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
    chosen = models[-1]
    if model_path:
        match = [m for m in models if os.path.abspath(str(m.get("path"))) == os.path.abspath(str(model_path))]
        if match:
            chosen = match[0]
        else:
            logger.warning(f"Modelo solicitado no encontrado en Resultados: {model_path}. Usando último disponible.")
    elif model_index is not None:
        try:
            idx = int(model_index) - 1
            if 0 <= idx < len(models):
                chosen = models[idx]
            else:
                logger.warning(f"model_index fuera de rango ({model_index}). Usando último disponible.")
        except Exception:
            logger.warning(f"model_index inválido ({model_index}). Usando último disponible.")

    # Instanciar modelo con la arquitectura del modelo elegido
    num_classes = 2
    edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')
    edge_feature_dims_per_type = _detect_edge_feature_dims(data)
    in_channels = data['pm'].x.shape[1]
    arch = chosen['arch']
    meta = chosen['meta']
    chosen_gnn_variant = meta.get('gnn_variant', GNN_VARIANT)
    encoder_overrides = _parse_edge_encoder_per_type(meta.get('edge_encoder_per_type'))
    model = _build_gnn_model(
        in_channels=in_channels,
        hidden_channels=int(arch.get('hidden_channels') or meta.get('hidden_channels', 32)),
        out_channels=num_classes,
        num_heads=int(arch.get('num_heads') or meta.get('num_heads', 4)),
        dropout=float(meta.get('dropout', 0.2)),
        edge_feature_dim=edge_feature_dim,
        num_layers=int(arch.get('num_layers') or meta.get('num_layers', len(NUM_NEIGHBORS))),
        use_checkpointing=False,
        aggr1=meta.get('aggr1', 'sum'),
        aggr2=meta.get('aggr2', 'sum'),
        gnn_variant=chosen_gnn_variant,
        sequence_index=loaded_obj.get('sequence_index') if isinstance(loaded_obj, dict) else None,
        num_nodes=data['pm'].num_nodes,
        device=device,
        use_residual=_safe_bool(meta.get("use_residual", arch.get("use_residual", False))),
        use_relation_self_loops=_safe_bool(meta.get("use_relation_self_loops"), True),
        require_temporal_head=_variant_has_temporal_head(chosen_gnn_variant),
        edge_types=tuple(data.edge_types) if hasattr(data, 'edge_types') else None,
        edge_feature_dims=edge_feature_dims_per_type,
        **encoder_overrides,
    )

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

    # Split VAL en val_thr (selección de umbral) y val_cal (fit Platt) — 50/50.
    # Evita el sesgo optimista de usar el mismo subconjunto para calibrar y
    # elegir el threshold; las métricas finales se reportan sobre TEST.
    if mask_key == 'val_mask' and y_true_val.size >= 4:
        idx_thr, idx_cal = _split_val_for_thr_cal(y_true_val.size, seed=int(SEED))
        y_true_cal = y_true_val[idx_cal]
        y_prob1_cal_raw = y_prob1_val_raw[idx_cal]
        # 1) Fit Platt en val_cal
        _, platt_model = _platt_scale_probabilities(y_true_cal, y_prob1_cal_raw)
        # 2) Aplica Platt y elige umbral en val_thr
        y_true_thr = y_true_val[idx_thr]
        y_prob1_thr_raw = y_prob1_val_raw[idx_thr]
        y_prob1_thr = _apply_platt_model(y_prob1_thr_raw, platt_model)
        tau, info = pick_threshold_from_val(y_true_thr, y_prob1_thr, mode="fbeta", beta=float(F_BETA_THRESHOLD))
        # Curva PR en val (sobre val_thr para coherencia con la selección).
        try:
            pr_curve_path = os.path.splitext(best_model_path)[0] + "_pr_curve_val.json"
            _save_pr_curve_artifact(y_true_thr, y_prob1_thr, pr_curve_path)
        except Exception:
            pass
    else:
        # Fallback: si no hay val_mask suficiente, mantén el comportamiento legacy.
        y_prob1_val, platt_model = _platt_scale_probabilities(y_true_val, y_prob1_val_raw)
        tau, info = pick_threshold_from_val(y_true_val, y_prob1_val, mode="fbeta", beta=float(F_BETA_THRESHOLD))
    print(f"🔧 Umbral seleccionado (Fβ, β={F_BETA_THRESHOLD}): tau={tau:.6f} → P={info.get('precision', float('nan')):.3f}, R={info.get('recall', float('nan')):.3f}")

    # Pase final con umbral + IC bootstrap sobre los splits (clave para clase rara).
    final = test(
        model, data, node_type='pm',
        threshold=tau, calibration_model=platt_model,
        compute_bootstrap_ci=True, bootstrap_n=1000, bootstrap_seed=int(SEED),
    )
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

def run_gnn_anomaly_hpo_then_train(loaded_obj, use_graphsmote: Optional[bool] = None):
    """
    Búsqueda de hiperparámetros para GNN (Anomalías) y entrenamiento automático posterior.
    Respeta la elección de GraphSMOTE y la aplica tanto a la búsqueda como al entrenamiento.
    """
    if use_graphsmote is None:
        use_graphsmote = False
        logger.info("GraphSMOTE no especificado para HPO+train de anomalías; usando False.")

    # Búsqueda (forzada con el flag elegido)
    best_params = search_hyperparameters(loaded_obj, use_graphsmote_search=bool(use_graphsmote))
    if not best_params:
        print("❌ No se obtuvieron hiperparámetros.")
        return
    print("\n▶️ Iniciando entrenamiento con los hiperparámetros encontrados…")
    # Entrenamiento forzando el mismo flag de GraphSMOTE
    # Entrenamiento forzando el mismo flag de GraphSMOTE y etiquetando propósito 'Anomaly'
    if isinstance(loaded_obj, dict):
        loaded_obj['purpose'] = 'Anomaly'
    run_gat_training(loaded_obj, force_use_graphsmote=bool(use_graphsmote), purpose='Anomaly')

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
                     model=None, device=None, topk_hard=None, generator: Optional[torch.Generator] = None):
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
        if generator is not None:
            perm = torch.randperm(neg_idx.numel(), generator=generator)
        else:
            perm = torch.randperm(neg_idx.numel())
        neg_keep = neg_idx[perm[:n_neg_keep]]

    seeds = torch.cat([pos_idx, neg_keep], dim=0)
    if generator is not None:
        seeds = seeds[torch.randperm(seeds.numel(), generator=generator)]
    else:
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

def _criterion_loss_sum_and_denominator(criterion, logits: torch.Tensor, targets: torch.Tensor):
    if criterion is None:
        return None

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        weight = getattr(criterion, "weight", None)
        ignore_index = int(getattr(criterion, "ignore_index", -100))
        label_smoothing = float(getattr(criterion, "label_smoothing", 0.0))
        per_item = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        ).view(-1)
        targets_flat = targets.view(-1)
        valid = targets_flat != ignore_index
        if not bool(valid.any().item()):
            return None
        loss_sum = per_item[valid].sum()
        if weight is not None:
            weight = weight.to(device=targets.device, dtype=per_item.dtype)
            denominator = weight[targets_flat[valid]].sum()
        else:
            denominator = valid.sum().to(device=per_item.device, dtype=per_item.dtype)
        if torch.isfinite(loss_sum) and torch.isfinite(denominator) and float(denominator.detach().cpu().item()) > 0:
            return loss_sum, denominator
        return None

    if isinstance(criterion, FocalLoss):
        original_reduction = criterion.reduction
        try:
            criterion.reduction = "none"
            per_item = criterion(logits, targets)
        finally:
            criterion.reduction = original_reduction
        if torch.is_tensor(per_item):
            per_item = per_item.view(-1)
            if per_item.numel() > 0:
                loss_sum = per_item.sum()
                denominator = torch.as_tensor(
                    per_item.numel(),
                    device=per_item.device,
                    dtype=per_item.dtype,
                )
                if torch.isfinite(loss_sum) and torch.isfinite(denominator):
                    return loss_sum, denominator

    try:
        loss_tensor = criterion(logits, targets)
    except Exception:
        return None
    if not torch.is_tensor(loss_tensor):
        return None
    if loss_tensor.numel() > 1:
        flat_loss = loss_tensor.reshape(-1)
        finite = torch.isfinite(flat_loss)
        if not bool(finite.any().item()):
            return None
        loss_sum = flat_loss[finite].sum()
        denominator = finite.sum().to(device=flat_loss.device, dtype=flat_loss.dtype)
        return loss_sum, denominator
    loss_tensor = loss_tensor.reshape(())
    if not torch.isfinite(loss_tensor):
        return None
    denominator = torch.as_tensor(
        max(int(targets.numel()), 1),
        device=loss_tensor.device,
        dtype=loss_tensor.dtype,
    )
    return loss_tensor * denominator, denominator

def _update_val_loss_monitor(
    *,
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    min_delta: float,
) -> Tuple[bool, float, int]:
    return _update_metric_monitor(
        monitor_value=val_loss,
        best_monitor_value=best_val_loss,
        patience_counter=patience_counter,
        min_delta=min_delta,
        monitor_mode="min",
    )

def _normalize_checkpoint_metric(metric: object) -> str:
    raw = str(metric or "val_objective_score").strip().lower()
    key = raw.replace(" ", "_").replace("-", "_")
    compact = key.replace("@", "_at_")
    if compact.startswith("val_recall_at_") or compact.startswith("val_precision_at_"):
        return compact
    if compact.startswith("recall_at_") or compact.startswith("precision_at_"):
        return f"val_{compact}"
    aliases = {
        "objective": "val_objective_score",
        "objective_score": "val_objective_score",
        "val_objective": "val_objective_score",
        "val_objective_score": "val_objective_score",
        "validation_objective": "val_objective_score",
        "auprc": "val_auprc",
        "average_precision": "val_auprc",
        "val_auprc": "val_auprc",
        "f1": "val_f1",
        "val_f1": "val_f1",
        "f0.5": "val_f05",
        "f05": "val_f05",
        "val_f0.5": "val_f05",
        "val_f05": "val_f05",
        "mcc": "val_mcc",
        "val_mcc": "val_mcc",
        "accuracy": "val_accuracy",
        "val_accuracy": "val_accuracy",
        "validation_loss": "val_loss",
        "val_loss": "val_loss",
        "loss": "val_loss",
    }
    return aliases.get(key, "val_objective_score")

def _monitor_mode_for_metric(metric: str) -> str:
    return "min" if _normalize_checkpoint_metric(metric) == "val_loss" else "max"

def _initial_monitor_value(monitor_mode: str) -> float:
    return float("inf") if monitor_mode == "min" else float("-inf")

def _update_metric_monitor(
    *,
    monitor_value: float,
    best_monitor_value: float,
    patience_counter: int,
    min_delta: float,
    monitor_mode: str,
) -> Tuple[bool, float, int]:
    try:
        previous = float(best_monitor_value)
    except Exception:
        previous = _initial_monitor_value(monitor_mode)
    try:
        current = float(monitor_value)
    except Exception:
        current = float("nan")
    if not math.isfinite(current):
        return False, previous, int(patience_counter) + 1
    delta = max(float(min_delta), 0.0)
    mode = str(monitor_mode or "max").lower()
    if not math.isfinite(previous):
        return True, current, 0
    if mode == "min" and (previous - current) > delta:
        return True, current, 0
    if mode != "min" and (current - previous) > delta:
        return True, current, 0
    return False, previous, int(patience_counter) + 1

def _metric_value_for_monitor(metric: str, values: Dict[str, Optional[float]]) -> float:
    normalized = _normalize_checkpoint_metric(metric)
    value = values.get(normalized)
    if normalized == "val_objective_score" and value is None:
        # Objective score can be absent in degenerate validation folds; F1 is the
        # closest legacy objective fallback for keeping training resumable.
        value = values.get("val_f1")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

def _safe_matthews_corrcoef(y_true, y_pred) -> Optional[float]:
    try:
        y_true_np = np.asarray(y_true.detach().cpu() if torch.is_tensor(y_true) else y_true).ravel()
        y_pred_np = np.asarray(y_pred.detach().cpu() if torch.is_tensor(y_pred) else y_pred).ravel()
        if y_true_np.size == 0 or y_true_np.size != y_pred_np.size:
            return None
        value = float(matthews_corrcoef(y_true_np, y_pred_np))
        return value if math.isfinite(value) else 0.0
    except Exception as e:
        logger.warning(f"No se pudo calcular MCC: {e}")
        return None

def prior_shift_adjust(p_train, p_real, p_hat):
    # p_train: prevalencia en entrenamiento; p_real: prevalencia real
    # p_hat: prob entrenada
    eps = 1e-6
    p_train = float(np.clip(p_train, eps, 1.0 - eps))
    p_real = float(np.clip(p_real, eps, 1.0 - eps))
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
    prec_, rec_, thr_ = prec[:-1], rec[:-1], thr

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


def _split_val_for_thr_cal(n_val: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits indices [0, n_val) en dos subsets disjuntos 50/50 de forma determinista.

    Devuelve (idx_thr, idx_cal) — el primero para selección de umbral, el segundo
    para fit de Platt. Evita el doble uso de val que sesga las métricas finales
    cuando umbral y calibración se eligen sobre el mismo conjunto.
    """
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(int(n_val))
    n_cal = int(n_val) // 2
    return perm[n_cal:], perm[:n_cal]  # (thr_idx, cal_idx)


def _compute_recall_precision_at_k(
    y_true,
    y_prob,
    k_values=(50, 100, 500, 1000),
) -> Dict[str, Optional[float]]:
    """
    Recall@K y Precision@K para clasificación binaria.

    Útil para clase rara: mide cuántos positivos atrapas en los top-K candidatos
    (ranking-aware), independiente del umbral de decisión.
    """
    y_true_np = np.asarray(y_true.detach().cpu() if torch.is_tensor(y_true) else y_true).astype(int).ravel()
    y_prob_np = np.asarray(y_prob.detach().cpu() if torch.is_tensor(y_prob) else y_prob).astype(float).ravel()
    out: Dict[str, Optional[float]] = {}
    n_total = int(y_true_np.size)
    n_pos = int(y_true_np.sum())
    if n_total == 0 or n_pos == 0:
        for k in k_values:
            out[f"recall@{k}"] = None
            out[f"precision@{k}"] = None
        return out

    order = np.argsort(-y_prob_np)
    sorted_true = y_true_np[order]
    cumsum_true = np.cumsum(sorted_true)
    for k in k_values:
        k_eff = min(int(k), n_total)
        if k_eff < 1:
            out[f"recall@{k}"] = None
            out[f"precision@{k}"] = None
            continue
        tp_at_k = int(cumsum_true[k_eff - 1])
        out[f"recall@{k}"] = float(tp_at_k / n_pos)
        out[f"precision@{k}"] = float(tp_at_k / k_eff)
    return out


def _bootstrap_metric_ci(
    y_true,
    y_prob,
    threshold: float,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Optional[object]]:
    """
    IC bootstrap (95% por defecto) para F1, F0.5, AUPRC, Recall y Precision (clase 1).

    Submuestrea con reemplazo `n_bootstrap` veces y computa percentiles
    [alpha/2, 1-alpha/2]. Crucial cuando el test set tiene pocos positivos
    (~39 con 0.3% de prevalencia): los puntos individuales tienen varianza alta.
    """
    y_true_np = np.asarray(y_true.detach().cpu() if torch.is_tensor(y_true) else y_true).astype(int).ravel()
    y_prob_np = np.asarray(y_prob.detach().cpu() if torch.is_tensor(y_prob) else y_prob).astype(float).ravel()
    n = int(y_true_np.size)
    out_keys = ("f1_ci", "f05_ci", "auprc_ci", "recall_ci", "precision_ci")
    if n == 0 or int(y_true_np.sum()) == 0:
        return {k: None for k in out_keys} | {"n_bootstrap_effective": 0}

    rng = np.random.RandomState(int(seed))
    f1s: List[float] = []
    f05s: List[float] = []
    auprcs: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.randint(0, n, size=n)
        yt = y_true_np[idx]
        yp = y_prob_np[idx]
        n_pos_b = int(yt.sum())
        if n_pos_b == 0 or n_pos_b == n:
            continue  # muestra degenerada; saltar
        pred = (yp >= float(threshold)).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        P = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        R = tp / max(tp + fn, 1)
        if (P + R) > 0:
            F1 = 2 * P * R / (P + R)
            denom_05 = 0.25 * P + R
            F05 = 1.25 * P * R / denom_05 if denom_05 > 0 else 0.0
        else:
            F1 = 0.0
            F05 = 0.0
        try:
            ap = float(average_precision_score(yt, yp))
        except Exception:
            continue
        f1s.append(F1)
        f05s.append(F05)
        auprcs.append(ap)
        recalls.append(R)
        precisions.append(P)

    if not f1s:
        return {k: None for k in out_keys} | {"n_bootstrap_effective": 0}

    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    def _pair(arr: List[float]) -> tuple:
        return (float(np.percentile(arr, lo_pct)), float(np.percentile(arr, hi_pct)))

    return {
        "f1_ci": _pair(f1s),
        "f05_ci": _pair(f05s),
        "auprc_ci": _pair(auprcs),
        "recall_ci": _pair(recalls),
        "precision_ci": _pair(precisions),
        "n_bootstrap_effective": len(f1s),
        "alpha": float(alpha),
    }


def _save_pr_curve_artifact(y_true, y_prob, save_path: str) -> bool:
    """Guarda la curva precision-recall completa como JSON: precision/recall/threshold arrays."""
    try:
        y_true_np = np.asarray(y_true.detach().cpu() if torch.is_tensor(y_true) else y_true).astype(int).ravel()
        y_prob_np = np.asarray(y_prob.detach().cpu() if torch.is_tensor(y_prob) else y_prob).astype(float).ravel()
        if y_true_np.size == 0 or int(y_true_np.sum()) == 0:
            return False
        prec, rec, thr = precision_recall_curve(y_true_np, y_prob_np)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as fh:
            json.dump(
                {
                    "precision": [float(v) for v in prec.tolist()],
                    "recall": [float(v) for v in rec.tolist()],
                    "thresholds": [float(v) for v in thr.tolist()],
                    "n_positives": int(y_true_np.sum()),
                    "n_total": int(y_true_np.size),
                },
                fh,
            )
        return True
    except Exception as exc:
        logger.warning(f"No se pudo guardar pr_curve: {exc}")
        return False


def _compute_binary_eval_extras(y_true, y_pred, y_prob) -> Dict[str, Optional[float]]:
    """Return operational binary metrics that are not in classification_report."""
    extras: Dict[str, Optional[float]] = {
        "false_alarm_ratio": None,
        "far": None,
        "brier_score": None,
        "brier": None,
    }
    try:
        y_true_np = np.asarray(y_true.detach().cpu() if torch.is_tensor(y_true) else y_true).astype(int).ravel()
        y_pred_np = np.asarray(y_pred.detach().cpu() if torch.is_tensor(y_pred) else y_pred).astype(int).ravel()
        y_prob_np = np.asarray(y_prob.detach().cpu() if torch.is_tensor(y_prob) else y_prob).astype(float)
    except Exception:
        return extras

    if y_true_np.size == 0:
        return extras

    try:
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
        tn, fp, _fn, _tp = cm.ravel()
        far = float(fp / (fp + tn)) if (fp + tn) else 0.0
        extras["false_alarm_ratio"] = far
        extras["far"] = far
    except Exception:
        pass

    try:
        if y_prob_np.ndim == 2 and y_prob_np.shape[1] > 1:
            prob1 = y_prob_np[:, 1]
        else:
            prob1 = y_prob_np.ravel()
        if prob1.size == y_true_np.size:
            brier = float(np.mean((np.clip(prob1, 0.0, 1.0) - y_true_np) ** 2))
            extras["brier_score"] = brier
            extras["brier"] = brier
    except Exception:
        pass

    return extras


@torch.no_grad()
def test(
    model,
    data,
    batch_size=BATCH_SIZE,
    node_type='pm',
    threshold=None,
    calibration_model=None,
    masks: Optional[List[str]] = None,
    node_indices: Optional[torch.Tensor] = None,
    mask_name: str = "subset_mask",
    num_neighbors=None,
    criterion=None,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    *,
    compute_at_k: bool = True,
    at_k_values: tuple = (50, 100, 500, 1000),
    compute_bootstrap_ci: bool = False,
    bootstrap_n: int = 1000,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: int = 0,
):
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

    # Mover datos a la CPU para NeighborLoader (evita copia si ya está en CPU)
    data_cpu = data
    try:
        sample_tensor = next(iter(data.x_dict.values()))
        if sample_tensor.device.type != "cpu":
            data_cpu = data.cpu()
    except Exception:
        data_cpu = data.cpu()

    temporal_module = getattr(model, 'temporal_head', None)
    if temporal_module is not None:
        temporal_module.eval()
        temporal_module.reset_cache()

    if node_indices is not None:
        if not torch.is_tensor(node_indices):
            node_indices = torch.as_tensor(node_indices, dtype=torch.long)
        node_indices = node_indices.view(-1).cpu()
        if node_indices.numel() == 0:
            return results
        eval_items = [(mask_name, node_indices)]
    else:
        # Detectar máscaras dinámicamente (ej. train_mask, val_mask, test_mask)
        available_masks = [
            key for key in data_cpu[node_type].keys() if key.endswith('_mask')
        ]
        if masks is not None:
            mask_set = set(masks)
            available_masks = [m for m in masks if m in mask_set and m in available_masks]
        eval_items = []
        for mname in available_masks:
            mask = data_cpu[node_type][mname]
            if mask.sum() == 0:
                logger.info(f"Skipping '{mname}' as it is empty.")
                continue
            idx = mask.nonzero(as_tuple=False).view(-1)
            eval_items.append((mname, idx))

    total_eval_items = max(len(eval_items), 1)

    def _emit_eval_progress(payload: Dict[str, object]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(payload)
        except Exception:
            logger.debug("GNN evaluation progress callback failed", exc_info=True)

    for mask_idx, (mask_name, node_indices) in enumerate(eval_items, start=1):
        num_neighbors_cfg = _resolve_num_neighbors(
            num_neighbors if num_neighbors is not None else NUM_NEIGHBORS,
            NUM_NEIGHBORS,
            data_cpu.edge_types,
        )
        loader = NeighborLoader(
            data_cpu,
            input_nodes=(node_type, node_indices),
            num_neighbors=num_neighbors_cfg,
            batch_size=batch_size,
            shuffle=False,
        )

        all_preds, all_probs, all_true = [], [], []
        loss_sum = 0.0
        loss_count = 0
        node_count = int(node_indices.numel())
        try:
            total_batches = int(math.ceil(node_count / max(int(batch_size), 1)))
        except Exception:
            total_batches = 1
        total_batches = max(total_batches, 1)
        _emit_eval_progress(
            {
                "event": "mask_start",
                "mask_name": mask_name,
                "mask_index": mask_idx,
                "mask_total": total_eval_items,
                "node_count": node_count,
                "batch_index": 0,
                "batch_total": total_batches,
                "processed_nodes": 0,
            }
        )

        if temporal_module is not None:
            temporal_module.reset_cache()
            _prime_temporal_cache_if_needed(model, data, node_type=node_type, context=f"test:{mask_name}")

        for batch_idx, batch in enumerate(loader, start=1):
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
                prob1_cal = _calibrated_probability_tensor(prob1, calibration_model)
                probs = probs.clone()
                probs[:, 1] = prob1_cal
                probs[:, 0] = torch.clamp(1.0 - prob1_cal, min=0.0, max=1.0)
                prob1 = probs[:, 1]
            if threshold is None:
                preds = probs.argmax(dim=1)  # comportamiento antiguo
            else:
                preds = (prob1 >= threshold).long()  # UMBRAL DE VALIDACIÓN
            true = batch[node_type].y[:batch[node_type].batch_size]
            if criterion is not None:
                try:
                    loss_parts = _criterion_loss_sum_and_denominator(
                        criterion,
                        logits_target,
                        true,
                    )
                    if loss_parts is not None:
                        loss_numerator, loss_denominator = loss_parts
                        loss_sum += float(loss_numerator.detach().cpu().item())
                        loss_count += float(loss_denominator.detach().cpu().item())
                except Exception:
                    pass
            
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_true.append(true.cpu())
            processed_nodes = min(int(batch_idx) * int(batch_size), node_count)
            _emit_eval_progress(
                {
                    "event": "batch_done",
                    "mask_name": mask_name,
                    "mask_index": mask_idx,
                    "mask_total": total_eval_items,
                    "node_count": node_count,
                    "batch_index": int(batch_idx),
                    "batch_total": total_batches,
                    "processed_nodes": processed_nodes,
                }
            )

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

        mcc = _safe_matthews_corrcoef(y_true, y_pred)

        extra_metrics = _compute_binary_eval_extras(y_true, y_pred, y_prob)

        # Métricas ranking-aware útiles para clase rara (~0.3%): cuántos positivos
        # quedan en el top-K. Independiente del umbral.
        at_k_metrics: Dict[str, Optional[float]] = {}
        if compute_at_k and num_classes == 2 and len(torch.unique(y_true)) > 1:
            at_k_metrics = _compute_recall_precision_at_k(
                y_true, y_prob[:, 1], k_values=at_k_values
            )

        # IC bootstrap (off por defecto: costoso). Útil al final, no por época.
        ci_metrics: Optional[Dict[str, object]] = None
        if (
            compute_bootstrap_ci
            and num_classes == 2
            and len(torch.unique(y_true)) > 1
            and threshold is not None
        ):
            ci_metrics = _bootstrap_metric_ci(
                y_true,
                y_prob[:, 1],
                threshold=float(threshold),
                n_bootstrap=int(bootstrap_n),
                alpha=float(bootstrap_alpha),
                seed=int(bootstrap_seed),
            )

        results[mask_name] = {
            'report': report,
            'cm': cm,
            'preds': y_pred,
            'probs': y_prob,
            'true': y_true,
            'auc': auc_score,
            'auprc': auprc,
            'mcc': mcc,
            'false_alarm_ratio': extra_metrics.get("false_alarm_ratio"),
            'far': extra_metrics.get("far"),
            'brier_score': extra_metrics.get("brier_score"),
            'brier': extra_metrics.get("brier"),
            'recall_at_k': {k: at_k_metrics.get(f"recall@{k}") for k in at_k_values} if at_k_metrics else {},
            'precision_at_k': {k: at_k_metrics.get(f"precision@{k}") for k in at_k_values} if at_k_metrics else {},
            'ci_95': ci_metrics,
            'node_idx': node_indices,  # para exportar claves
        }
        if criterion is not None and loss_count > 0:
            results[mask_name]['loss'] = float(loss_sum / float(loss_count))
        _emit_eval_progress(
            {
                "event": "mask_done",
                "mask_name": mask_name,
                "mask_index": mask_idx,
                "mask_total": total_eval_items,
                "node_count": node_count,
                "batch_index": total_batches,
                "batch_total": total_batches,
                "processed_nodes": node_count,
            }
        )
        
    return results

def _fmt_ci(ci_pair) -> str:
    """Formatea (lo, hi) como '[lo, hi]' a 3 decimales; '—' si no hay CI."""
    if ci_pair is None:
        return "—"
    try:
        lo, hi = float(ci_pair[0]), float(ci_pair[1])
        return f"[{lo:.3f}, {hi:.3f}]"
    except Exception:
        return "—"


def print_evaluation_report(results, dataset_name):
    """
    Imprime un reporte de evaluación formateado a partir de los resultados de la función test.

    Incluye, cuando están presentes:
      - Métricas por clase + globales (clásicas).
      - IC 95% bootstrap (si test() se llamó con compute_bootstrap_ci=True).
      - Recall@K y Precision@K (siempre que compute_at_k=True).
      - FAR / Brier para diagnóstico operacional.
    """
    report = results['report']
    cm = results['cm']
    ci = results.get('ci_95')
    rec_at_k = results.get('recall_at_k') or {}
    prec_at_k = results.get('precision_at_k') or {}

    print(f"--- Reporte de Evaluación: Conjunto de {dataset_name} ---")
    print(f"Clase 'No Accidente (0)':")
    print(f"  - Precisión: {report['No Accidente (0)']['precision']:.4f}")
    print(f"  - Recall:    {report['No Accidente (0)']['recall']:.4f}")
    print(f"  - F1-Score:  {report['No Accidente (0)']['f1-score']:.4f}")

    p1 = float(report['Accidente (1)']['precision'])
    r1 = float(report['Accidente (1)']['recall'])
    f1_1 = float(report['Accidente (1)']['f1-score'])
    print(f"Clase 'Accidente (1)':")
    if ci is not None:
        print(f"  - Precisión: {p1:.4f}  IC95={_fmt_ci(ci.get('precision_ci'))}")
        print(f"  - Recall:    {r1:.4f}  IC95={_fmt_ci(ci.get('recall_ci'))}")
        print(f"  - F1-Score:  {f1_1:.4f}  IC95={_fmt_ci(ci.get('f1_ci'))}")
        f05_ci = ci.get('f05_ci')
        if f05_ci is not None:
            print(f"  - F0.5 IC95: {_fmt_ci(f05_ci)}")
    else:
        print(f"  - Precisión: {p1:.4f}")
        print(f"  - Recall:    {r1:.4f}")
        print(f"  - F1-Score:  {f1_1:.4f}")

    print("Globales:")
    print(f"  - Accuracy:  {report['accuracy']:.4f}")
    print(f"  - Precision: {report['macro avg']['precision']:.4f}")
    print(f"  - Recall:    {report['macro avg']['recall']:.4f}")
    print(f"  - F1-Score:  {report['macro avg']['f1-score']:.4f}")
    print(f"  - F1-Macro: {results.get('f1_macro', 0.0):.4f}")
    auprc_val = results.get('auprc') or 0.0
    auc_val = results.get('auc') or 0.0
    if ci is not None:
        print(f"  - AUC:       {auc_val:.4f}")
        print(f"  - AUPRC (clase 1): {auprc_val:.4f}  IC95={_fmt_ci(ci.get('auprc_ci'))}")
    else:
        print(f"  - AUC: {auc_val:.4f}")
        print(f"  - AUPRC (clase 1): {auprc_val:.4f}")
    print(f"  - MCC: {results.get('mcc') or 0.0:.4f}")

    far_val = results.get('false_alarm_ratio') or results.get('far')
    if far_val is not None:
        print(f"  - FAR: {float(far_val):.4f}")
    brier_val = results.get('brier_score') or results.get('brier')
    if brier_val is not None:
        print(f"  - Brier: {float(brier_val):.4f}")

    # Ranking-aware @K (clave para clase rara): cuántos positivos hay en el top-K.
    if rec_at_k:
        ks = sorted(int(k) for k in rec_at_k.keys() if rec_at_k.get(k) is not None)
        if ks:
            print("Ranking @K (clase 1):")
            header = "  K        " + " ".join(f"{k:>8}" for k in ks)
            row_r = "  Recall   " + " ".join(
                f"{rec_at_k.get(k):>8.4f}" if rec_at_k.get(k) is not None else "    —   "
                for k in ks
            )
            row_p = "  Precision" + " ".join(
                f"{prec_at_k.get(k):>8.4f}" if prec_at_k.get(k) is not None else "    —   "
                for k in ks
            )
            print(header)
            print(row_r)
            print(row_p)

    if ci is not None and ci.get("n_bootstrap_effective"):
        print(f"  (IC95 bootstrap: N={ci['n_bootstrap_effective']} resamples efectivos)")

    print("Matriz de Confusión:")
    print(cm)
    print("--------------------------------------------------")

def _calculate_graph_hash(graph_filename=None, graph_path=None):
    """Calcula el hash SHA256 del contenido de un archivo de grafo."""
    if not graph_filename and not graph_path:
        logger.info("No se proporcionó ruta/nombre de grafo para calcular el hash.")
        return None

    candidates = []
    if graph_path:
        candidates.append(graph_path)
    if graph_filename:
        candidates.append(graph_filename)
        base_name = os.path.basename(graph_filename)
        candidates.append(os.path.join(RESULTADOS_DIR, base_name))
        if graph_filename != base_name:
            candidates.append(os.path.join(RESULTADOS_DIR, graph_filename))

    resolved_path = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            resolved_path = cand
            break

    if not resolved_path:
        logger.info(
            "No se encontró el archivo del grafo para calcular el hash "
            f"(filename={graph_filename}). Se omite el hash."
        )
        return None
    
    hasher = hashlib.sha256()
    try:
        with open(resolved_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logger.error(f"Error al leer el archivo del grafo para hashear: {e}")
        return None

def _normalize_graph_hash_value(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) >= 16 and re.fullmatch(r"[0-9a-f]+", text):
        return text
    return None

def _resolve_graph_identity(loaded_obj: object) -> Dict[str, Optional[str]]:
    """Resolve semantic graph identity and optional file-content hash.

    `graph_hash` is the semantic hash used for experiment identity when it is
    present in graph metadata. `graph_file_hash` is retained only for audit
    traceability and must not replace the semantic identifier.
    """
    if not isinstance(loaded_obj, dict):
        return {
            "graph_hash": None,
            "graph_file_hash": None,
            "graph_hash_source": None,
        }

    semantic_sources = []
    for key in ("graph_hash", "hash"):
        semantic_sources.append((loaded_obj.get(key), "semantic_metadata"))

    for meta_key in ("metadata", "meta"):
        meta = loaded_obj.get(meta_key)
        if isinstance(meta, dict):
            semantic_sources.append((meta.get("graph_hash"), "semantic_metadata"))

    data_obj = loaded_obj.get("data")
    for attr_name in ("graph_metadata", "metadata"):
        meta = getattr(data_obj, attr_name, None)
        if isinstance(meta, dict):
            semantic_sources.append((meta.get("graph_hash"), "semantic_metadata"))

    for attr_name in ("graph_hash", "hash"):
        semantic_sources.append((getattr(data_obj, attr_name, None), "semantic_metadata"))

    graph_hash = None
    graph_hash_source = None
    for raw, source in semantic_sources:
        normalized = _normalize_graph_hash_value(raw)
        if normalized:
            graph_hash = normalized
            graph_hash_source = source
            break

    graph_file_hash = _normalize_graph_hash_value(
        _calculate_graph_hash(
            graph_filename=loaded_obj.get("filename"),
            graph_path=loaded_obj.get("graph_path") or loaded_obj.get("path"),
        )
    )
    if graph_hash is None and graph_file_hash:
        graph_hash = graph_file_hash
        graph_hash_source = "file_sha256_fallback"

    return {
        "graph_hash": graph_hash,
        "graph_file_hash": graph_file_hash,
        "graph_hash_source": graph_hash_source,
    }

def _normalize_objective_metric(metric: object) -> str:
    raw = str(metric or "F1").strip()
    key = raw.lower().replace("_", "-")
    aliases = {
        "f1": "F1",
        "recall": "Recall",
        "far": "FAR",
        "recall-far": "Recall-FAR",
        "f0.5": "F0.5",
        "f05": "F0.5",
        "auprc": "AUPRC",
        "mcc": "MCC",
        "accuracy": "Accuracy",
    }
    return aliases.get(key, "F1")

def _score_from_objective_metrics(
    objective_metric: str,
    *,
    f1: Optional[float],
    recall: Optional[float],
    far: Optional[float],
    fbeta: Optional[float],
    auprc: Optional[float],
    mcc: Optional[float],
    accuracy: Optional[float],
) -> float:
    def _safe(v: Optional[float], default: float = 0.0) -> float:
        try:
            fv = float(v)
            return fv if math.isfinite(fv) else float(default)
        except Exception:
            return float(default)

    metric = _normalize_objective_metric(objective_metric)
    if metric == "F1":
        return _safe(f1)
    if metric == "Recall":
        return _safe(recall)
    if metric == "FAR":
        return 1.0 - _safe(far)
    if metric == "Recall-FAR":
        return _safe(recall) - _safe(far)
    if metric == "F0.5":
        return _safe(fbeta)
    if metric == "AUPRC":
        return _safe(auprc)
    if metric == "MCC":
        return _safe(mcc)
    if metric == "Accuracy":
        return _safe(accuracy)
    return _safe(f1)


def _coerce_hyperparam_value(value):
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower in ("true", "false"):
            return lower == "true"
        if not text:
            return value
        try:
            if re.fullmatch(r"[-+]?\d+", text):
                return int(text)
            if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text):
                return float(text)
        except Exception:
            return value
    return value


def _load_hyperparams_csv(path: str) -> dict:
    df_hp = pd.read_csv(path)
    if any(str(c).startswith("params_") for c in df_hp.columns):
        df_hp.rename(columns=lambda c: str(c).replace("params_", ""), inplace=True)
    if df_hp.empty:
        raise ValueError(f"Archivo de hiperparámetros vacío: {path}")
    if "value" in df_hp.columns:
        try:
            row = df_hp.loc[df_hp["value"].idxmax()].to_dict()
        except Exception:
            row = df_hp.iloc[0].to_dict()
    else:
        row = df_hp.iloc[0].to_dict()
    return {k: _coerce_hyperparam_value(v) for k, v in row.items()}


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
    train_decoders_only: bool = False,
    accumulation_steps: Optional[int] = None,
    resume_state_path: Optional[str] = None,
    save_state_path: Optional[str] = None,
    train_sampler_mode: Optional[str] = None,
    deterministic_sampling: Optional[bool] = None,
    sampling_seed: Optional[int] = None,
    disable_hard_undersampling: Optional[bool] = None,
    positive_sampler_target_fraction: Optional[float] = None,
    positive_sampler_hard_window_minutes: Optional[int] = None,
    positive_sampler_hard_negatives_per_positive: Optional[int] = None,
    cluster_gcn_num_parts: Optional[int] = None,
    cluster_gcn_parts_per_epoch: Optional[int] = None,
    graphsaint_mode: Optional[str] = None,
    graphsaint_batch_size: Optional[int] = None,
    graphsaint_num_steps: Optional[int] = None,
    graphsaint_walk_length: Optional[int] = None,
    rl_action_space: Optional[str] = None,
    rl_initial_p: Optional[float] = None,
    rl_min_p: Optional[float] = None,
    rl_max_p: Optional[float] = None,
    rl_min_keep: Optional[int] = None,
    rl_positive_only: Optional[bool] = None,
    rl_similarity_pretrain_epochs: Optional[int] = None,
    rl_lambda_simi: Optional[float] = None,
    rl_switch_patience: Optional[int] = None,
    rl_backtracking: Optional[bool] = None,
    eval_neighbors_mode: Optional[str] = None,
    eval_num_neighbors: Optional[object] = None,
    checkpoint_metric: Optional[str] = None,
    ranking_loss_mode: Optional[str] = None,
    ranking_loss_weight: Optional[float] = None,
    ranking_loss_margin: Optional[float] = None,
    ranking_loss_max_pairs: Optional[int] = None,
    metrics_history_path: Optional[str] = None,
    test_eval_interval_epochs: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    should_test: Optional[Callable[[], bool]] = None,
    hparams_path: Optional[str] = None,
    hparams_index: Optional[int] = None,
    reuse_hparams: bool = True,
    allow_hpo_search: bool = True,
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

    # Detect if graph is already balanced (synthetic nodes present)
    graph_has_synthetics = False
    try:
        if "pm" in data.node_types and hasattr(data["pm"], "is_synthetic"):
            graph_has_synthetics = bool(data["pm"].is_synthetic.sum().item() > 0)
    except Exception:
        graph_has_synthetics = False

    if force_use_graphsmote is None:
        use_graphsmote = bool(graph_has_synthetics)
        logger.info(
            "GraphSMOTE no fue especificado; usando detección automática del grafo "
            f"(synthetics={graph_has_synthetics})."
        )
    else:
        use_graphsmote = bool(force_use_graphsmote)
    skip_graphsmote_augment = bool(use_graphsmote and graph_has_synthetics and not train_decoders_only)
    enable_graphsmote_augment = bool(use_graphsmote and not skip_graphsmote_augment)
    
    # GRAPHSMOTE_MODE ahora se controla desde config.py, pero la activación general sigue aquí
    if use_graphsmote:
        logger.info(f"GraphSMOTE habilitado para este entrenamiento (Modo: {GRAPHSMOTE_MODE}).")
        if skip_graphsmote_augment:
            logger.info("Grafo ya balanceado (nodos sinteticos detectados). Se omite la aumentacion GraphSMOTE.")
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
    selected_hp_path = None
    if hparams_path:
        selected_hp_path = str(hparams_path)
        if not os.path.exists(selected_hp_path):
            logger.error(f"Archivo de hiperparámetros no encontrado: {selected_hp_path}")
            return
    elif hp_files and bool(reuse_hparams):
        if hparams_index is not None:
            if int(hparams_index) == 0:
                selected_hp_path = None
            elif 1 <= int(hparams_index) <= len(hp_files):
                selected_hp_path = hp_files[int(hparams_index) - 1]
            else:
                logger.error(f"hparams_index fuera de rango: {hparams_index}")
                return
        else:
            selected_hp_path = hp_files[-1]

    if selected_hp_path:
        try:
            logger.info(f"Cargando hiperparámetros desde: {os.path.basename(selected_hp_path)}")
            best_params = _load_hyperparams_csv(selected_hp_path)
        except Exception as exc:
            logger.error(f"No se pudo cargar '{os.path.basename(selected_hp_path)}': {exc}")
            return
    elif not hp_files:
        logger.info("No se encontraron archivos de hiperparámetros compatibles. Iniciando nueva búsqueda...")
        if not allow_hpo_search:
            logger.error("No hay hiperparámetros compatibles y allow_hpo_search=False.")
            return
        best_params = search_hyperparameters(
            loaded_obj,
            use_graphsmote_search=use_graphsmote,
            optimizer_overrides=optimizer_overrides,
            reuse_existing=bool(reuse_hparams),
        )
    else:
        variant_lbl = "GraphSMOTE" if use_graphsmote else "Base"
        if _has_imgagn(loaded_obj):
            variant_lbl += " + ImGAGN"
        if not allow_hpo_search:
            logger.error(f"reuse_hparams=False para variante {variant_lbl}, pero allow_hpo_search=False.")
            return
        logger.info(f"Iniciando nueva búsqueda de hiperparámetros para variante: {variant_lbl}.")
        best_params = search_hyperparameters(
            loaded_obj,
            use_graphsmote_search=use_graphsmote,
            optimizer_overrides=optimizer_overrides,
            reuse_existing=bool(reuse_hparams),
        )

    if not best_params:
        logger.error("No se pudieron obtener los hiperparámetros. Abortando.")
        return

    # Gradient accumulation (configurable)
    if accumulation_steps is None:
        accumulation_steps = best_params.get("accumulation_steps")
    if accumulation_steps is None:
        accumulation_steps = ACCUMULATION_STEPS
    try:
        accumulation_steps = max(1, int(accumulation_steps))
    except Exception:
        accumulation_steps = int(ACCUMULATION_STEPS)
    best_params["accumulation_steps"] = accumulation_steps
    best_params["lr_scheduler"] = _normalize_lr_scheduler_choice(
        best_params.get("lr_scheduler", best_params.get("lr_scheduler_choice", "one_cycle"))
    )

    # Override explícito de num_neighbors. Si NUM_NEIGHBORS_OVERRIDE está definido
    # en config.py, sobrescribe la elección de Optuna / valor cargado desde CSV.
    # Acepta lista plana, dict con tuple/string keys, o JSON string;
    # _resolve_num_neighbors normaliza el formato más adelante.
    if NUM_NEIGHBORS_OVERRIDE is not None:
        if isinstance(NUM_NEIGHBORS_OVERRIDE, dict):
            best_params["num_neighbors"] = json.dumps(
                {(k if isinstance(k, str) else (k[1] if isinstance(k, tuple) and len(k) >= 2 else str(k))): v
                 for k, v in NUM_NEIGHBORS_OVERRIDE.items()}
            )
        else:
            best_params["num_neighbors"] = NUM_NEIGHBORS_OVERRIDE
        logger.info(f"NUM_NEIGHBORS_OVERRIDE aplicado: {best_params['num_neighbors']}")

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

    sampler_raw = train_sampler_mode if train_sampler_mode is not None else best_params.get("train_sampler_mode", best_params.get("sampler_mode", "neighbor"))
    sampler_key = str(sampler_raw or "neighbor").strip().lower().replace("-", "_")
    sampler_aliases = {
        "neighbor": "neighbor",
        "neighborloader": "neighbor",
        "positive_aware": "positive_aware",
        "positiveaware": "positive_aware",
        "pos_aware": "positive_aware",
        "positive_aware_neighbor": "positive_aware",
        "positive_aware_neighborloader": "positive_aware",
        "cluster_gcn": "cluster_gcn",
        "clustergcn": "cluster_gcn",
        "graphsaint": "graphsaint",
        "rl_top_p": "rl_top_p",
        "riognn": "rl_top_p",
        "rio_gnn": "rl_top_p",
        "rsrl": "rl_top_p",
    }
    train_sampler_mode_resolved = sampler_aliases.get(sampler_key, "neighbor")

    deterministic_sampling_resolved = bool(
        deterministic_sampling if deterministic_sampling is not None else best_params.get("deterministic_sampling", False)
    )
    sampling_seed_resolved = int(
        _safe_cast(
            sampling_seed if sampling_seed is not None else best_params.get("sampling_seed"),
            int,
            SEED,
        )
    )
    disable_hard_undersampling_resolved = bool(
        disable_hard_undersampling
        if disable_hard_undersampling is not None
        else best_params.get("disable_hard_undersampling", False)
    )
    positive_sampler_target_fraction_resolved = float(
        _safe_cast(
            positive_sampler_target_fraction
            if positive_sampler_target_fraction is not None
            else best_params.get("positive_sampler_target_fraction"),
            float,
            0.02,
        )
    )
    if (
        not math.isfinite(float(positive_sampler_target_fraction_resolved))
        or positive_sampler_target_fraction_resolved <= 0.0
    ):
        positive_sampler_target_fraction_resolved = 0.02
    positive_sampler_target_fraction_resolved = max(
        0.0,
        min(0.5, float(positive_sampler_target_fraction_resolved)),
    )
    positive_sampler_hard_window_minutes_resolved = int(
        max(
            0,
            _safe_cast(
                positive_sampler_hard_window_minutes
                if positive_sampler_hard_window_minutes is not None
                else best_params.get("positive_sampler_hard_window_minutes"),
                int,
                60,
            ),
        )
    )
    positive_sampler_hard_negatives_per_positive_resolved = int(
        max(
            0,
            _safe_cast(
                positive_sampler_hard_negatives_per_positive
                if positive_sampler_hard_negatives_per_positive is not None
                else best_params.get("positive_sampler_hard_negatives_per_positive"),
                int,
                4,
            ),
        )
    )

    cluster_gcn_num_parts_resolved = int(
        _safe_cast(
            cluster_gcn_num_parts if cluster_gcn_num_parts is not None else best_params.get("cluster_gcn_num_parts"),
            int,
            64,
        )
    )
    cluster_gcn_parts_per_epoch_resolved = int(
        _safe_cast(
            cluster_gcn_parts_per_epoch if cluster_gcn_parts_per_epoch is not None else best_params.get("cluster_gcn_parts_per_epoch"),
            int,
            0,
        )
    )
    graphsaint_mode_resolved = str(
        graphsaint_mode if graphsaint_mode is not None else best_params.get("graphsaint_mode", "node")
    ).strip().lower()
    if graphsaint_mode_resolved not in {"node", "edge", "random_walk"}:
        graphsaint_mode_resolved = "node"
    graphsaint_batch_size_resolved = int(
        _safe_cast(
            graphsaint_batch_size if graphsaint_batch_size is not None else best_params.get("graphsaint_batch_size"),
            int,
            2048,
        )
    )
    graphsaint_num_steps_resolved = int(
        _safe_cast(
            graphsaint_num_steps if graphsaint_num_steps is not None else best_params.get("graphsaint_num_steps"),
            int,
            8,
        )
    )
    graphsaint_walk_length_resolved = int(
        _safe_cast(
            graphsaint_walk_length if graphsaint_walk_length is not None else best_params.get("graphsaint_walk_length"),
            int,
            2,
        )
    )

    rl_action_space_resolved = str(
        rl_action_space if rl_action_space is not None else best_params.get("rl_action_space", "discrete")
    ).strip().lower()
    if rl_action_space_resolved in {"continuous", "actor", "actor_critic"}:
        rl_action_space_resolved = "continuous_actor"
    if rl_action_space_resolved not in {"discrete", "continuous_actor"}:
        rl_action_space_resolved = "discrete"
    rl_initial_p_resolved = float(
        _safe_cast(rl_initial_p if rl_initial_p is not None else best_params.get("rl_initial_p"), float, 0.5)
    )
    rl_min_p_resolved = float(
        _safe_cast(rl_min_p if rl_min_p is not None else best_params.get("rl_min_p"), float, 0.05)
    )
    rl_max_p_resolved = float(
        _safe_cast(rl_max_p if rl_max_p is not None else best_params.get("rl_max_p"), float, 1.0)
    )
    rl_min_keep_resolved = int(
        _safe_cast(rl_min_keep if rl_min_keep is not None else best_params.get("rl_min_keep"), int, 1)
    )
    rl_positive_only_resolved = bool(
        rl_positive_only if rl_positive_only is not None else best_params.get("rl_positive_only", True)
    )
    rl_similarity_pretrain_epochs_resolved = int(
        _safe_cast(
            rl_similarity_pretrain_epochs
            if rl_similarity_pretrain_epochs is not None
            else best_params.get("rl_similarity_pretrain_epochs"),
            int,
            3,
        )
    )
    rl_lambda_simi_resolved = float(
        _safe_cast(rl_lambda_simi if rl_lambda_simi is not None else best_params.get("rl_lambda_simi"), float, 2.0)
    )
    rl_switch_patience_resolved = int(
        _safe_cast(
            rl_switch_patience if rl_switch_patience is not None else best_params.get("rl_switch_patience"),
            int,
            3,
        )
    )
    rl_backtracking_resolved = bool(
        rl_backtracking if rl_backtracking is not None else best_params.get("rl_backtracking", True)
    )

    eval_mode_raw = eval_neighbors_mode if eval_neighbors_mode is not None else best_params.get("eval_neighbors_mode", "same")
    eval_mode_key = str(eval_mode_raw or "same").strip().lower().replace("-", "_")
    eval_mode_aliases = {
        "same": "same",
        "train": "same",
        "same_as_train": "same",
        "exhaustive": "exhaustive",
        "full": "exhaustive",
        "custom": "custom",
    }
    eval_neighbors_mode_resolved = eval_mode_aliases.get(eval_mode_key, "same")
    eval_num_neighbors_resolved = eval_num_neighbors if eval_num_neighbors is not None else best_params.get("eval_num_neighbors")
    if eval_neighbors_mode_resolved == "custom" and eval_num_neighbors_resolved is None:
        eval_num_neighbors_resolved = best_params.get("num_neighbors")

    ranking_raw = (
        ranking_loss_mode
        if ranking_loss_mode is not None
        else best_params.get("ranking_loss_mode", "none")
    )
    ranking_key = str(ranking_raw or "none").strip().lower().replace("-", "_")
    ranking_aliases = {
        "": "none",
        "none": "none",
        "off": "none",
        "disabled": "none",
        "pairwise": "pairwise_softplus",
        "pairwise_softplus": "pairwise_softplus",
        "softplus": "pairwise_softplus",
        "topk": "topk_pairwise",
        "topk_pairwise": "topk_pairwise",
        "top_k_pairwise": "topk_pairwise",
    }
    ranking_loss_mode_resolved = ranking_aliases.get(ranking_key, "none")
    ranking_loss_weight_resolved = float(
        _safe_cast(
            ranking_loss_weight
            if ranking_loss_weight is not None
            else best_params.get("ranking_loss_weight"),
            float,
            0.0,
        )
    )
    if not math.isfinite(float(ranking_loss_weight_resolved)) or ranking_loss_weight_resolved < 0.0:
        ranking_loss_weight_resolved = 0.0
    if ranking_loss_mode_resolved == "none":
        ranking_loss_weight_resolved = 0.0
    ranking_loss_margin_resolved = float(
        _safe_cast(
            ranking_loss_margin
            if ranking_loss_margin is not None
            else best_params.get("ranking_loss_margin"),
            float,
            0.1,
        )
    )
    if not math.isfinite(float(ranking_loss_margin_resolved)):
        ranking_loss_margin_resolved = 0.1
    ranking_loss_max_pairs_resolved = int(
        max(
            1,
            _safe_cast(
                ranking_loss_max_pairs
                if ranking_loss_max_pairs is not None
                else best_params.get("ranking_loss_max_pairs"),
                int,
                4096,
            ),
        )
    )

    best_params["train_sampler_mode"] = train_sampler_mode_resolved
    best_params["deterministic_sampling"] = bool(deterministic_sampling_resolved)
    best_params["sampling_seed"] = int(sampling_seed_resolved)
    best_params["disable_hard_undersampling"] = bool(disable_hard_undersampling_resolved)
    best_params["positive_sampler_target_fraction"] = float(positive_sampler_target_fraction_resolved)
    best_params["positive_sampler_hard_window_minutes"] = int(positive_sampler_hard_window_minutes_resolved)
    best_params["positive_sampler_hard_negatives_per_positive"] = int(
        positive_sampler_hard_negatives_per_positive_resolved
    )
    best_params["cluster_gcn_num_parts"] = int(cluster_gcn_num_parts_resolved)
    best_params["cluster_gcn_parts_per_epoch"] = int(cluster_gcn_parts_per_epoch_resolved)
    best_params["graphsaint_mode"] = str(graphsaint_mode_resolved)
    best_params["graphsaint_batch_size"] = int(graphsaint_batch_size_resolved)
    best_params["graphsaint_num_steps"] = int(graphsaint_num_steps_resolved)
    best_params["graphsaint_walk_length"] = int(graphsaint_walk_length_resolved)
    best_params["rl_action_space"] = str(rl_action_space_resolved)
    best_params["rl_initial_p"] = float(rl_initial_p_resolved)
    best_params["rl_min_p"] = float(rl_min_p_resolved)
    best_params["rl_max_p"] = float(rl_max_p_resolved)
    best_params["rl_min_keep"] = int(rl_min_keep_resolved)
    best_params["rl_positive_only"] = bool(rl_positive_only_resolved)
    best_params["rl_similarity_pretrain_epochs"] = int(rl_similarity_pretrain_epochs_resolved)
    best_params["rl_lambda_simi"] = float(rl_lambda_simi_resolved)
    best_params["rl_switch_patience"] = int(rl_switch_patience_resolved)
    best_params["rl_backtracking"] = bool(rl_backtracking_resolved)
    best_params["eval_neighbors_mode"] = str(eval_neighbors_mode_resolved)
    best_params["eval_num_neighbors"] = eval_num_neighbors_resolved
    best_params["ranking_loss_mode"] = str(ranking_loss_mode_resolved)
    best_params["ranking_loss_weight"] = float(ranking_loss_weight_resolved)
    best_params["ranking_loss_margin"] = float(ranking_loss_margin_resolved)
    best_params["ranking_loss_max_pairs"] = int(ranking_loss_max_pairs_resolved)

    logger.info(
        "Sampling config | mode=%s | deterministic=%s | seed=%d | disable_hard=%s | "
        "positive_target=%.4f | positive_window_min=%d | positive_hard_per_pos=%d | "
        "cluster_parts=%d | cluster_parts_epoch=%d | saint_mode=%s | saint_batch=%d | saint_steps=%d | saint_walk=%d | "
        "eval_neighbors_mode=%s | eval_num_neighbors=%s",
        train_sampler_mode_resolved,
        deterministic_sampling_resolved,
        sampling_seed_resolved,
        disable_hard_undersampling_resolved,
        positive_sampler_target_fraction_resolved,
        positive_sampler_hard_window_minutes_resolved,
        positive_sampler_hard_negatives_per_positive_resolved,
        cluster_gcn_num_parts_resolved,
        cluster_gcn_parts_per_epoch_resolved,
        graphsaint_mode_resolved,
        graphsaint_batch_size_resolved,
        graphsaint_num_steps_resolved,
        graphsaint_walk_length_resolved,
        eval_neighbors_mode_resolved,
        eval_num_neighbors_resolved,
    )
    logger.info(
        "Ranking loss config | mode=%s | weight=%.4f | margin=%.4f | max_pairs=%d",
        ranking_loss_mode_resolved,
        ranking_loss_weight_resolved,
        ranking_loss_margin_resolved,
        ranking_loss_max_pairs_resolved,
    )

    data.edge_attr_dict = { et: getattr(data[et], 'edge_attr', None) for et in data.edge_types }

    # 3) SummaryWriter + hparams
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(RESULTADOS_DIR, "runs_attention"), flush_secs=30)
    else:
        logger.warning("TensorBoard no esta instalado; se omite SummaryWriter.")
    hparam_payload = {}
    for key, value in best_params.items():
        if isinstance(value, (list, tuple, dict)):
            try:
                hparam_payload[key] = json.dumps(value)
            except Exception:
                hparam_payload[key] = str(value)
        else:
            hparam_payload[key] = value
    if writer is not None:
        writer.add_hparams(hparam_payload, {'hparam/val_f1': 0.0})

    # 4) Modelo y optimizador
    num_classes = len(torch.unique(data['pm'].y))

    edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')
    edge_feature_dims_per_type = _detect_edge_feature_dims(data)
    encoder_overrides = _parse_edge_encoder_per_type(best_params.get('edge_encoder_per_type'))

    in_channels = data['pm'].x.shape[1]
    gnn_variant = best_params.get('gnn_variant', GNN_VARIANT)
    model = _build_gnn_model(
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
        gnn_variant=gnn_variant,
        sequence_index=loaded_obj.get('sequence_index'),
        num_nodes=data['pm'].num_nodes,
        device=device,
        use_residual=_safe_bool(best_params.get("use_residual"), False),
        use_relation_self_loops=_safe_bool(best_params.get("use_relation_self_loops"), True),
        require_temporal_head=_variant_has_temporal_head(gnn_variant),
        edge_types=tuple(data.edge_types) if hasattr(data, 'edge_types') else None,
        edge_feature_dims=edge_feature_dims_per_type,
        **encoder_overrides,
    )
    temporal_module = getattr(model, 'temporal_head', None)

    # Rutas y directorios por modelo (usar mismo ID para z2x + modelo)
    tag_suffix = _model_tag_suffix(use_graphsmote, loaded_obj, gnn_variant=gnn_variant)
    ts_stamp_save = datetime.now().strftime('%Y%m%d_%H%M%S')
    graph_identity = _resolve_graph_identity(loaded_obj)
    graph_hash = graph_identity.get("graph_hash")
    graph_file_hash = graph_identity.get("graph_file_hash")
    graph_hash_source = graph_identity.get("graph_hash_source")
    hash_tag8 = f"_{graph_hash[:8]}" if graph_hash else ""
    run_id = f"gnn_{ts_stamp_save}"
    if graph_hash:
        run_id += f"_{graph_hash[:8]}"
    run_id += f"_{uuid.uuid4().hex[:6]}"
    
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
    edge_attr_decoder = None
    if use_graphsmote and enable_graphsmote_augment:
        logger.info("Entrenando decodificadores z->x para la aumentación...")
        model.use_checkpointing = False
        z2x_decoders = train_z2x_decoders(
            model,
            data,
            device=device,
            epochs=DECODER_EPOCHS,
            num_neighbors=smote_num_neighbors,
            save_dir=z2x_run_dir,
            progress_callback=progress_callback,
        )
        # Decoder de atributos de arista: cierra el gap del zero-fill parcial
        # cuando edge_attr_dim > delta_feature_idx en GraphSMOTE.
        try:
            edge_attr_run_dir = os.path.join(
                os.path.dirname(z2x_run_dir),
                "edge_attr_decoders",
                os.path.basename(z2x_run_dir),
            )
            logger.info("Entrenando decodificador edge_attr para aristas sintéticas...")
            edge_attr_decoder = train_edge_attr_decoders(
                model,
                data,
                device=device,
                epochs=DECODER_EPOCHS,
                num_neighbors=smote_num_neighbors,
                save_dir=edge_attr_run_dir,
                seed=GS_SEED,
                show_progress=False,
            )
        except Exception as exc:
            logger.warning(f"No se pudo entrenar edge_attr_decoder: {exc}; SMOTE caerá a delta legacy.")
            edge_attr_decoder = None
        model.use_checkpointing = True
    elif use_graphsmote and not enable_graphsmote_augment:
        z2x_decoders = None
        edge_attr_decoder = None

    if train_decoders_only:
        if use_graphsmote:
            logger.info("✅ Entrenamiento de decodificadores (z->x) finalizado correctamente.")
            logger.info("El grafo aumentado ya está listo en memoria/disco para balanceo.")
            try:
                # Guardar modelo embeddings para reutilizar en GraphSMOTE (cargar existente)
                torch.save(model.state_dict(), best_model_path_unique)
                if SAVE_GAT_ALIASES:
                    torch.save(model.state_dict(), best_model_path)
                meta = dict(best_params)
                meta.update(
                    {
                        "best_val_f1": None,
                        "best_val_auprc": None,
                        "best_epoch": 0,
                        "use_graphsmote": True,
                        "graph_hash": graph_hash,
                        "graph_file_hash": graph_file_hash,
                        "graph_hash_source": graph_hash_source,
                        "git_commit": _get_repo_version(),
                        "purpose": "GraphSMOTE embeddings (decoders only)",
                        "train_decoders_only": True,
                        "num_neighbors_effective": smote_num_neighbors,
                        "out_channels": int(num_classes),
                        "in_channels": int(in_channels),
                        "edge_feature_dim": int(edge_feature_dim),
                    }
                )
                meta = _json_safe(meta)
                meta_path_unique = os.path.splitext(best_model_path_unique)[0] + "_hparams.json"
                import json as _json
                with open(meta_path_unique, 'w') as f:
                    _json.dump(meta, f)
                if SAVE_GAT_ALIASES:
                    try:
                        meta_path = os.path.splitext(best_model_path)[0] + "_hparams.json"
                        with open(meta_path, 'w') as f:
                            _json.dump(meta, f)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"No se pudo guardar modelo embeddings: {e}")
        else:
            logger.warning("⚠️ train_decoders_only=True pero GraphSMOTE no está habilitado.")
        return

    loader_num_neighbors = _resolve_num_neighbors(best_params.get('num_neighbors'), NUM_NEIGHBORS, data.edge_types)
    batch_size_hp = int(_safe_cast(best_params.get('batch_size'), int, BATCH_SIZE))
    target_pos_ratio_override = _safe_cast(best_params.get('target_pos_ratio'), float, TARGET_POS_RATIO)
    smote_every_override = max(1, _safe_cast(best_params.get('smote_every_n_epochs'), int, SMOTE_EVERY_N_EPOCHS))

    rl_sampler_controller = None
    if train_sampler_mode_resolved == "rl_top_p":
        rl_sampler_controller = RioGNNThresholdController(
            edge_types=[
                edge_type
                for edge_type in data.edge_types
                if edge_type[0] == "pm" and edge_type[2] == "pm"
            ],
            num_layers=int(best_params.get("num_layers", 2)),
            in_channels=int(in_channels),
            max_degree_by_edge_type=relation_max_degrees(data.cpu(), node_type="pm"),
            action_space=str(rl_action_space_resolved),
            initial_p=float(rl_initial_p_resolved),
            min_p=float(rl_min_p_resolved),
            max_p=float(rl_max_p_resolved),
            min_keep=int(rl_min_keep_resolved),
            switch_patience=int(rl_switch_patience_resolved),
            backtracking=bool(rl_backtracking_resolved),
            positive_only=bool(rl_positive_only_resolved),
            secondary_reward_weight=0.25,
            seed=int(sampling_seed_resolved),
        ).to(device)
        try:
            simi_losses = pretrain_label_aware_similarity(
                rl_sampler_controller,
                data.cpu(),
                node_type="pm",
                device=device,
                epochs=int(rl_similarity_pretrain_epochs_resolved),
                positive_only=bool(rl_positive_only_resolved),
            )
            if writer is not None and simi_losses:
                for pre_epoch, simi_loss in enumerate(simi_losses, start=1):
                    writer.add_scalar("RLTopP/SimilarityPretrainLoss", simi_loss, pre_epoch)
            logger.info(
                "RioGNN top-p RL sampler habilitado | action_space=%s | p0=%.3f | min=%.3f | max=%.3f | min_keep=%d",
                rl_action_space_resolved,
                rl_initial_p_resolved,
                rl_min_p_resolved,
                rl_max_p_resolved,
                rl_min_keep_resolved,
            )
        except Exception as exc:
            logger.warning(f"No se pudo preentrenar scorer label-aware RL: {exc}")

    # Edge generator
    z_dim_dict = {ntype: int(best_params['hidden_channels']) * int(best_params['num_heads']) for ntype in data.node_types}
    edge_gen = RelEdgeGen(z_dim_dict, data.edge_types).to(device) if use_graphsmote else None

    optimizer_name = str(best_params.get('optimizer', 'Adam'))
    optimizer_cls = get_optimizer_cls(optimizer_name)
    optimizer_params = list(model.parameters())
    if edge_gen is not None:
        optimizer_params += list(edge_gen.parameters())
    if rl_sampler_controller is not None:
        optimizer_params += list(rl_sampler_controller.parameters())
    optimizer = optimizer_cls(
        optimizer_params,
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
        # Default 0.95 calibrado para clase rara (~0.3%): peso fuerte a la clase positiva.
        alpha_val = [1 - best_params.get('focal_alpha', 0.95), best_params.get('focal_alpha', 0.95)]
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
                # Tempered class weights: sqrt(1/counts) en lugar de 1/counts.
                # Con 0.3% positivos el ratio crudo es ~333× y desestabiliza
                # gradientes; sqrt lo baja a ~18× sin perder señal de balance.
                weight = (1.0 / counts.float()).sqrt()
                logger.info(f"Class weighting (tempered sqrt) enabled for CE. Weights: {weight.cpu().numpy()}")
        else:
            reason = "GraphSMOTE active" if use_graphsmote else "ImGAGN-augmented graph detected"
            logger.info(f"Class weighting for CrossEntropyLoss disabled ({reason}).")

        criterion = torch.nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)

    # 7) Pre-entrenamiento del generador de aristas (si aplica)
    if enable_graphsmote_augment and use_graphsmote and edge_gen is not None and PRETRAIN_EDGE_EPOCHS > 0:
        logger.info(f"Pre-entrenando el generador de aristas por {PRETRAIN_EDGE_EPOCHS} épocas...")
        pretrain_edge_generator(
            encoder=model,
            edge_gen=edge_gen,
            data=data,
            device=device,
            pretrain_epochs=PRETRAIN_EDGE_EPOCHS,
            optimizer=torch.optim.Adam(edge_gen.parameters(), lr=1e-3),
            criterion=torch.nn.BCEWithLogitsLoss(),
            writer=writer,
            report_path=os.path.join(z2x_run_dir, "edge_gen_auc.json"),
        )

    # 8) Selección de modo GraphSMOTE y preparación de grafos
    loader_num_neighbors = _resolve_num_neighbors(best_params.get('num_neighbors'), NUM_NEIGHBORS, data.edge_types)
    batch_size_hp = int(_safe_cast(best_params.get('batch_size'), int, BATCH_SIZE))
    target_pos_ratio_override = _safe_cast(best_params.get('target_pos_ratio'), float, TARGET_POS_RATIO)
    smote_every_override = max(1, _safe_cast(best_params.get('smote_every_n_epochs'), int, SMOTE_EVERY_N_EPOCHS))

    if use_graphsmote and GRAPHSMOTE_MODE == 'offline':
        if enable_graphsmote_augment:
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
                edge_attr_decoder=edge_attr_decoder,
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
        else:
            logger.info("Modo Offline: Grafo ya balanceado, se usa directamente sin re-aumentar.")
            train_graph = data
            base_graph = data
    elif use_graphsmote and GRAPHSMOTE_MODE == 'online':
        logger.info("Modo Online: El grafo se refrescará periódicamente.")
        base_graph = data
        # La aumentación inicial se hace dentro del loop
        train_graph = base_graph 
    else:
        # Mantener el grafo en CPU para minimizar uso de memoria en modo sin balanceo.
        # NeighborLoader mueve los batches al device dentro de train_minibatch/test.
        train_graph = data
        base_graph = data

    # 9) Loader y Scheduler
    def _epoch_seed_generator(epoch_idx: int, offset: int = 0) -> Optional[torch.Generator]:
        if not deterministic_sampling_resolved:
            return None
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(sampling_seed_resolved) + int(epoch_idx) * 9973 + int(offset))
        return gen

    def _resolve_base_seeds(
        graph_cpu,
        *,
        use_undersampling: bool,
        strategy: str,
        model,
        device,
        topk_hard,
        epoch_idx: int,
        pos_to_neg_ratio: float,
    ) -> torch.Tensor:
        if use_undersampling:
            strategy_effective = str(strategy)
            if disable_hard_undersampling_resolved and strategy_effective == "hard":
                strategy_effective = "random"
            return make_epoch_seeds(
                graph_cpu,
                node_type="pm",
                pos_to_neg_ratio=pos_to_neg_ratio,
                strategy=strategy_effective,
                model=model if strategy_effective == "hard" else None,
                device=device if strategy_effective == "hard" else None,
                topk_hard=topk_hard,
                generator=_epoch_seed_generator(epoch_idx, offset=11),
            )
        return graph_cpu["pm"].train_mask.nonzero(as_tuple=False).view(-1)

    pm_index_for_sampler = loaded_obj.get("pm_index") if isinstance(loaded_obj, dict) else None

    def create_loader(
        graph_to_load,
        use_undersampling=False,
        pos_to_neg_ratio=3.0,
        strategy='random',
        model=None,
        device=None,
        topk_hard=None,
        epoch_idx: int = 1,
    ):
        """
        Construye loader por época con soporte de:
        - undersampling de semillas (random/hard)
        - sampler nativo completo para cluster_gcn/graphsaint
        - determinismo opcional (seed controlado por época)
        """
        graph_cpu = graph_to_load.cpu()
        base_seeds = _resolve_base_seeds(
            graph_cpu,
            use_undersampling=bool(use_undersampling),
            strategy=str(strategy),
            model=model,
            device=device,
            topk_hard=topk_hard,
            epoch_idx=int(epoch_idx),
            pos_to_neg_ratio=float(pos_to_neg_ratio),
        )

        try:
            num_neighbors_cfg = loader_num_neighbors
        except Exception:
            num_neighbors_cfg = NUM_NEIGHBORS

        if train_sampler_mode_resolved != "neighbor":
            sampler_seed_for_epoch = (
                int(sampling_seed_resolved)
                if train_sampler_mode_resolved == "positive_aware"
                else int(sampling_seed_resolved) + int(epoch_idx) * 9973
            )
            native_cfg = {
                "train_sampler_mode": str(train_sampler_mode_resolved),
                "pm_index": pm_index_for_sampler,
                "positive_sampler_epoch": int(epoch_idx),
                "positive_sampler_target_fraction": float(positive_sampler_target_fraction_resolved),
                "positive_sampler_hard_window_minutes": int(
                    positive_sampler_hard_window_minutes_resolved
                ),
                "positive_sampler_hard_negatives_per_positive": int(
                    positive_sampler_hard_negatives_per_positive_resolved
                ),
                "cluster_gcn_num_parts": int(cluster_gcn_num_parts_resolved),
                "cluster_gcn_parts_per_epoch": int(cluster_gcn_parts_per_epoch_resolved),
                "graphsaint_mode": str(graphsaint_mode_resolved),
                "graphsaint_batch_size": int(graphsaint_batch_size_resolved),
                "graphsaint_num_steps": int(graphsaint_num_steps_resolved),
                "graphsaint_walk_length": int(graphsaint_walk_length_resolved),
                "rl_sampler_controller": rl_sampler_controller,
                "rl_action_space": str(rl_action_space_resolved),
                "rl_initial_p": float(rl_initial_p_resolved),
                "rl_min_p": float(rl_min_p_resolved),
                "rl_max_p": float(rl_max_p_resolved),
                "rl_min_keep": int(rl_min_keep_resolved),
                "rl_positive_only": bool(rl_positive_only_resolved),
                "rl_switch_patience": int(rl_switch_patience_resolved),
                "rl_backtracking": bool(rl_backtracking_resolved),
                "num_layers": int(best_params.get("num_layers", 2)),
                "deterministic_sampling": bool(deterministic_sampling_resolved),
                "sampling_seed": int(sampler_seed_for_epoch),
            }
            native_loader, native_error = _build_native_sampler_loader(
                graph_cpu=graph_cpu,
                sampler_config=native_cfg,
                batch_size=int(batch_size_hp),
                sampling_seed=int(sampler_seed_for_epoch),
                base_seeds=base_seeds,
                num_neighbors_cfg=num_neighbors_cfg,
                deterministic=bool(deterministic_sampling_resolved),
            )
            if native_loader is None:
                raise RuntimeError(
                    f"No se pudo construir loader nativo para {train_sampler_mode_resolved}: "
                    f"{native_error or 'error desconocido'}"
                )
            return native_loader

        input_nodes = ('pm', base_seeds)

        loader_gen = _epoch_seed_generator(int(epoch_idx), offset=101)
        shuffle_batches = True

        return NeighborLoader(
            graph_cpu,
            input_nodes=input_nodes,
            num_neighbors=num_neighbors_cfg,
            batch_size=batch_size_hp,
            shuffle=shuffle_batches,
            generator=loader_gen,
        )

    raw_undersample = best_params.get('undersample', None)
    auto_enable_hard = (raw_undersample is None) or (isinstance(raw_undersample, str) and raw_undersample.lower() == 'auto')
    use_undersampling = bool(raw_undersample) if not auto_enable_hard else False
    pos_to_neg_ratio = float(_safe_cast(best_params.get('pos_to_neg_ratio'), float, 3.0))
    undersampling_strategy = str(best_params.get('undersampling_strategy', 'random'))
    if disable_hard_undersampling_resolved and undersampling_strategy == 'hard':
        undersampling_strategy = 'random'
    topk_hard = best_params.get('topk_hard', None)
    if topk_hard is not None:
        topk_hard = int(topk_hard)

    hard_sampling_warmup = max(0, int(_safe_cast(best_params.get('hard_sampling_warmup'), int, 1)))

    if auto_enable_hard and not disable_hard_undersampling_resolved:
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

    def rebuild_train_loader(graph, strategy_override=None, epoch_idx: int = 1):
        effective_strategy = strategy_override or undersampling_strategy
        return create_loader(
            graph,
            use_undersampling=use_undersampling,
            pos_to_neg_ratio=pos_to_neg_ratio,
            strategy=effective_strategy,
            model=model if effective_strategy == 'hard' else None,
            device=device if effective_strategy == 'hard' else None,
            topk_hard=topk_hard,
            epoch_idx=int(epoch_idx),
        )

    train_loader = rebuild_train_loader(train_graph, epoch_idx=1)
    train_sampler_impl = str(getattr(train_loader, "sampler_impl", "neighbor_native"))
    best_params["sampler_impl"] = train_sampler_impl
    positive_sampler_stats = getattr(train_loader, "positive_sampler_stats", None)
    if positive_sampler_stats is not None:
        best_params["positive_sampler_stats"] = _json_safe(positive_sampler_stats)

    max_epochs = int(max_epochs) if max_epochs is not None else int(MAX_EPOCHS)
    objective_metric = _normalize_objective_metric(best_params.get("objective_metric", "F1"))
    objective_beta_default = 0.5 if objective_metric == "F0.5" else 1.0
    objective_threshold_beta = float(
        _safe_cast(
            best_params.get("threshold_beta", best_params.get("f_beta_threshold")),
            float,
            objective_beta_default,
        )
    )
    best_params["objective_metric"] = objective_metric
    best_params["threshold_beta"] = float(objective_threshold_beta)
    monitor_metric = _normalize_checkpoint_metric(
        checkpoint_metric
        if checkpoint_metric is not None
        else best_params.get("checkpoint_metric", "val_objective_score")
    )
    monitor_mode = _monitor_mode_for_metric(monitor_metric)
    lr_scheduler = _normalize_lr_scheduler_choice(best_params.get("lr_scheduler", "one_cycle"))
    best_params["checkpoint_metric"] = monitor_metric
    best_params["monitor_metric"] = monitor_metric
    best_params["monitor_mode"] = monitor_mode
    best_params["lr_scheduler"] = lr_scheduler

    if save_state_path is None and resume_state_path:
        save_state_path = resume_state_path

    resume_metadata_ckpt = None
    if resume_state_path:
        try:
            if os.path.exists(resume_state_path):
                resume_metadata_ckpt = torch.load(
                    resume_state_path,
                    map_location="cpu",
                    weights_only=False,
                )
        except Exception:
            resume_metadata_ckpt = None
    if isinstance(resume_metadata_ckpt, dict):
        if resume_metadata_ckpt.get("run_id"):
            run_id = str(resume_metadata_ckpt.get("run_id"))
        if metrics_history_path is None and resume_metadata_ckpt.get("metrics_history_path"):
            metrics_history_path = str(resume_metadata_ckpt.get("metrics_history_path"))
    if metrics_history_path is None:
        metrics_history_path = _default_training_metrics_history_path(save_state_path)
    if metrics_history_path and not resume_state_path:
        _reset_training_history(metrics_history_path)

    try:
        test_eval_interval_epochs_resolved = max(0, int(test_eval_interval_epochs or 0))
    except Exception:
        test_eval_interval_epochs_resolved = 0

    has_val_mask = hasattr(base_graph["pm"], "val_mask")
    val_mask_count = int(base_graph["pm"].val_mask.sum().item()) if has_val_mask else 0
    if val_mask_count == 0:
        raise RuntimeError(
            "No se encontro un split de validacion valido (val_mask vacio o ausente). "
            "Aborta entrenamiento para evitar usar train_mask como validacion."
        )

    _emit_training_event(
        "train_start",
        run_id,
        history_path=metrics_history_path,
        total=max_epochs,
        resume_state_path=str(resume_state_path) if resume_state_path else None,
        save_state_path=str(save_state_path) if save_state_path else None,
        metrics_history_path=str(metrics_history_path) if metrics_history_path else None,
        test_eval_interval_epochs=int(test_eval_interval_epochs_resolved),
        device=str(device),
        gnn_variant=_normalize_gnn_variant(gnn_variant),
        variant_tag=_variant_tags(use_graphsmote, loaded_obj, gnn_variant=gnn_variant),
        purpose=purpose or (loaded_obj.get("purpose") if isinstance(loaded_obj, dict) else None),
        use_graphsmote=bool(use_graphsmote),
        graphsmote_mode=str(GRAPHSMOTE_MODE),
        train_decoders_only=bool(train_decoders_only),
        graph_hash=graph_hash,
        graph_file_hash=graph_file_hash,
        graph_hash_source=graph_hash_source,
        batch_size=int(batch_size_hp),
        num_neighbors=loader_num_neighbors,
        smote_every_n_epochs=int(smote_every_override),
        target_pos_ratio=float(target_pos_ratio_override),
        accumulation_steps=int(accumulation_steps),
        train_sampler_mode=str(train_sampler_mode_resolved),
        sampler_impl=str(train_sampler_impl),
        deterministic_sampling=bool(deterministic_sampling_resolved),
        sampling_seed=int(sampling_seed_resolved),
        disable_hard_undersampling=bool(disable_hard_undersampling_resolved),
        cluster_gcn_num_parts=int(cluster_gcn_num_parts_resolved),
        cluster_gcn_parts_per_epoch=int(cluster_gcn_parts_per_epoch_resolved),
        graphsaint_mode=str(graphsaint_mode_resolved),
        graphsaint_batch_size=int(graphsaint_batch_size_resolved),
        graphsaint_num_steps=int(graphsaint_num_steps_resolved),
        graphsaint_walk_length=int(graphsaint_walk_length_resolved),
        rl_action_space=str(rl_action_space_resolved),
        rl_initial_p=float(rl_initial_p_resolved),
        rl_min_p=float(rl_min_p_resolved),
        rl_max_p=float(rl_max_p_resolved),
        rl_min_keep=int(rl_min_keep_resolved),
        rl_lambda_simi=float(rl_lambda_simi_resolved) if rl_sampler_controller is not None else 0.0,
        positive_sampler_target_fraction=float(positive_sampler_target_fraction_resolved),
        positive_sampler_hard_window_minutes=int(positive_sampler_hard_window_minutes_resolved),
        positive_sampler_hard_negatives_per_positive=int(
            positive_sampler_hard_negatives_per_positive_resolved
        ),
        positive_sampler_stats=_json_safe(positive_sampler_stats),
        eval_neighbors_mode=str(eval_neighbors_mode_resolved),
        eval_num_neighbors=eval_num_neighbors_resolved,
        ranking_loss_mode=str(ranking_loss_mode_resolved),
        ranking_loss_weight=float(ranking_loss_weight_resolved),
        ranking_loss_margin=float(ranking_loss_margin_resolved),
        ranking_loss_max_pairs=int(ranking_loss_max_pairs_resolved),
        objective_metric=str(objective_metric),
        objective_threshold_beta=float(objective_threshold_beta),
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        lr_scheduler=str(lr_scheduler),
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        lr_scheduler,
        max_lr=float(best_params['lr']),
        steps_per_epoch=_optimizer_steps_per_epoch(train_loader, accumulation_steps),
        epochs=max_epochs,
        monitor_mode=monitor_mode,
    )
    batch_scheduler = scheduler if _lr_scheduler_steps_per_batch(lr_scheduler) else None

    start_epoch = 1
    resume_best_val_loss = _initial_monitor_value("min")
    resume_best_val_f1 = 0.0
    resume_best_val_auprc = 0.0
    resume_best_val_auc = None
    resume_best_val_mcc = None
    resume_best_val_far = None
    resume_best_val_f05 = None
    resume_best_val_tau = None
    resume_best_val_accuracy = None
    resume_best_val_recall_at_k = {}
    resume_best_val_precision_at_k = {}
    resume_best_val_objective = float("-inf")
    resume_best_epoch = 0
    resume_patience_counter = 0
    resume_best_monitor_value = _initial_monitor_value(monitor_mode)
    if resume_state_path:
        try:
            if os.path.exists(resume_state_path):
                ckpt = torch.load(resume_state_path, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    incoming_state = ckpt["model_state"]
                    partial_model_load = False
                    if isinstance(incoming_state, dict):
                        current_state = model.state_dict()
                        compatible_state = {}
                        skipped_model_keys = {}
                        for key, value in incoming_state.items():
                            current_value = current_state.get(key)
                            if current_value is not None and hasattr(value, "shape") and tuple(value.shape) == tuple(current_value.shape):
                                compatible_state[key] = value
                            else:
                                partial_model_load = True
                                skipped_model_keys[key] = {
                                    "checkpoint_shape": list(value.shape) if hasattr(value, "shape") else None,
                                    "model_shape": list(current_value.shape) if current_value is not None and hasattr(current_value, "shape") else None,
                                }
                        if skipped_model_keys:
                            logger.warning(
                                "Warm-start parcial: se omitieron %d tensor(es) incompatibles del checkpoint. Primeras claves: %s",
                                len(skipped_model_keys),
                                list(skipped_model_keys.keys())[:12],
                            )
                        incoming_state = compatible_state
                    load_info = model.load_state_dict(incoming_state, strict=False)
                    if getattr(load_info, "missing_keys", None) or getattr(load_info, "unexpected_keys", None):
                        logger.info(
                            "Warm-start load_state_dict: missing=%d unexpected=%d",
                            len(getattr(load_info, "missing_keys", []) or []),
                            len(getattr(load_info, "unexpected_keys", []) or []),
                        )
                    if ckpt.get("optimizer_state") and not partial_model_load:
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                    elif ckpt.get("optimizer_state") and partial_model_load:
                        logger.info("Warm-start parcial: no se restaura optimizer_state por incompatibilidad de modelo.")
                    if ckpt.get("scheduler_state") and not partial_model_load:
                        scheduler.load_state_dict(ckpt["scheduler_state"])
                    elif ckpt.get("scheduler_state") and partial_model_load:
                        logger.info("Warm-start parcial: no se restaura scheduler_state por incompatibilidad de modelo.")
                    if rl_sampler_controller is not None and ckpt.get("rl_sampler_state"):
                        try:
                            rl_sampler_controller.load_state_dict_serializable(ckpt.get("rl_sampler_state"))
                        except Exception as exc:
                            logger.warning(f"No se pudo restaurar estado RL top-p; se continúa en modo legacy: {exc}")
                    start_epoch = int(ckpt.get("epoch", 0)) + 1
                    try:
                        resume_best_val_loss = float(ckpt.get("best_val_loss"))
                    except Exception:
                        resume_best_val_loss = _initial_monitor_value("min")
                    if not math.isfinite(resume_best_val_loss):
                        resume_best_val_loss = _initial_monitor_value("min")
                    resume_best_val_f1 = float(ckpt.get("best_val_f1", 0.0))
                    resume_best_val_auprc = float(ckpt.get("best_val_auprc", 0.0))
                    resume_best_val_auc = ckpt.get("best_val_auc")
                    resume_best_val_mcc = ckpt.get("best_val_mcc")
                    resume_best_val_far = ckpt.get("best_val_far")
                    resume_best_val_f05 = ckpt.get("best_val_f05")
                    resume_best_val_tau = ckpt.get("best_val_tau")
                    resume_best_val_accuracy = ckpt.get("best_val_accuracy")
                    resume_best_val_recall_at_k = ckpt.get("best_val_recall_at_k") or {}
                    resume_best_val_precision_at_k = ckpt.get("best_val_precision_at_k") or {}
                    try:
                        resume_best_val_objective = float(
                            ckpt.get("best_val_objective_score", ckpt.get("best_val_f1", float("-inf")))
                        )
                    except Exception:
                        resume_best_val_objective = float("-inf")
                    resume_best_epoch = int(ckpt.get("best_epoch", 0))
                    ckpt_monitor_metric = _normalize_checkpoint_metric(
                        ckpt.get("monitor_metric", "val_loss")
                    )
                    ckpt_monitor_mode = str(
                        ckpt.get("monitor_mode", _monitor_mode_for_metric(ckpt_monitor_metric))
                    ).lower()
                    if ckpt_monitor_metric == monitor_metric and ckpt_monitor_mode == monitor_mode:
                        try:
                            resume_best_monitor_value = float(
                                ckpt.get("best_monitor_value", _initial_monitor_value(monitor_mode))
                            )
                        except Exception:
                            resume_best_monitor_value = _initial_monitor_value(monitor_mode)
                        resume_patience_counter = int(ckpt.get("patience_counter", 0))
                    else:
                        resume_best_monitor_value = _metric_value_for_monitor(
                            monitor_metric,
                            {
                                "val_loss": resume_best_val_loss,
                                "val_f1": resume_best_val_f1,
                                "val_auprc": resume_best_val_auprc,
                                "val_auc": resume_best_val_auc,
                                "val_mcc": resume_best_val_mcc,
                                "val_far": resume_best_val_far,
                                "val_f05": resume_best_val_f05,
                                "val_tau": resume_best_val_tau,
                                "val_accuracy": resume_best_val_accuracy,
                                "val_objective_score": resume_best_val_objective,
                            },
                        )
                        if not math.isfinite(resume_best_monitor_value):
                            resume_best_monitor_value = _initial_monitor_value(monitor_mode)
                        resume_patience_counter = 0
                        logger.info(
                            "Checkpoint previo usa monitor %s/%s; se reanuda con monitor %s/%s "
                            "y se reinicia paciencia.",
                            ckpt_monitor_metric,
                            ckpt_monitor_mode,
                            monitor_metric,
                            monitor_mode,
                        )
                    logger.info(
                        f"Reanudando entrenamiento desde epoch {start_epoch} "
                        f"(checkpoint: {os.path.basename(resume_state_path)})"
                    )
        except Exception as exc:
            logger.warning(f"No se pudo reanudar desde checkpoint: {exc}")

    # 10) Loop de entrenamiento
    best_val_loss = float(resume_best_val_loss)
    best_val_f1 = float(resume_best_val_f1)
    best_val_auprc = float(resume_best_val_auprc)
    best_val_auc = resume_best_val_auc
    best_val_mcc = resume_best_val_mcc
    best_val_far = resume_best_val_far
    best_val_f05 = resume_best_val_f05
    best_val_tau = resume_best_val_tau
    best_val_accuracy = resume_best_val_accuracy
    best_val_recall_at_k = dict(resume_best_val_recall_at_k or {})
    best_val_precision_at_k = dict(resume_best_val_precision_at_k or {})
    best_val_objective_score = float(resume_best_val_objective)
    if not math.isfinite(best_val_objective_score):
        best_val_objective_score = float("-inf")
    best_monitor_value = float(resume_best_monitor_value)
    if not math.isfinite(best_monitor_value):
        best_monitor_value = _initial_monitor_value(monitor_mode)
    best_epoch = int(resume_best_epoch)
    patience_counter = int(resume_patience_counter)
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
                val_loss=None,
                best_val_loss=best_val_loss,
                val_f1=None,
                best_val_f1=best_val_f1,
                best_val_objective_score=best_val_objective_score,
                objective_metric=objective_metric,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                monitor_value=None,
                best_monitor_value=best_monitor_value,
                patience=patience,
                patience_counter=patience_counter,
            )
        except Exception:
            pass
    batch_event_cb = None
    if progress_callback is not None:
        def _batch_event_cb(**payload):
            _emit_training_event(
                "train_batch",
                run_id,
                total=int(max_epochs),
                **payload,
            )
        batch_event_cb = _batch_event_cb
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

    def _resolve_eval_neighbors_cfg(graph_for_eval):
        if eval_neighbors_mode_resolved == "same":
            return loader_num_neighbors
        if eval_neighbors_mode_resolved == "exhaustive":
            return _make_exhaustive_num_neighbors(
                graph_for_eval.edge_types,
                int(best_params.get("num_layers", 2)),
            )
        return _resolve_num_neighbors(
            eval_num_neighbors_resolved,
            loader_num_neighbors,
            graph_for_eval.edge_types,
        )

    stopped_early = False
    stopped_by_user = False
    last_completed_epoch = int(start_epoch) - 1
    for epoch in range(start_epoch, max_epochs + 1):
        if _training_stop_requested(should_stop):
            logger.info(
                "Stop requested before epoch %d; preserving last checkpoint.",
                epoch,
            )
            stopped_by_user = True
            break
        epoch_start_time = time.time()
        if use_undersampling or train_sampler_mode_resolved != "neighbor":
            strategy_now = undersampling_strategy
            if undersampling_strategy == 'hard' and epoch <= hard_sampling_warmup:
                strategy_now = 'random'
            train_loader = rebuild_train_loader(train_graph, strategy_override=strategy_now, epoch_idx=epoch)
        positive_sampler_stats = getattr(train_loader, "positive_sampler_stats", None)
        if positive_sampler_stats is not None:
            best_params["positive_sampler_stats"] = _json_safe(positive_sampler_stats)
        # Refresco para modo ONLINE
        if enable_graphsmote_augment and use_graphsmote and GRAPHSMOTE_MODE == 'online' and epoch % smote_every_override == 0:
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
                edge_attr_decoder=edge_attr_decoder,
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
            train_loader = rebuild_train_loader(train_graph, strategy_override=strategy_now, epoch_idx=epoch)  # Recargar el loader con configuración original
            positive_sampler_stats = getattr(train_loader, "positive_sampler_stats", None)
            if positive_sampler_stats is not None:
                best_params["positive_sampler_stats"] = _json_safe(positive_sampler_stats)

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
        ranking_loss_mode = str(ranking_loss_mode_resolved)
        ranking_loss_weight = float(ranking_loss_weight_resolved)
        ranking_loss_margin = float(ranking_loss_margin_resolved)
        ranking_loss_max_pairs = int(ranking_loss_max_pairs_resolved)

        _prime_temporal_cache_if_needed(model, train_graph, node_type='pm', context=f"train_epoch_{epoch}")
        
        loss, cls_loss, edge_loss, l2_att_loss = train_minibatch(model, train_loader, optimizer, criterion,
                                  grad_clip_value=float(best_params.get('grad_clip', 1.0)),
                                  device=device, use_amp=use_amp, scaler=scaler,
                                  scheduler=batch_scheduler, writer=writer, epoch=epoch,
                                  lambda_H=current_lambda_H, node_type='pm',
                                          edge_gen=edge_gen, lambda_edge=lambda_edge,
                                          lambda_l2_att=lambda_l2_att,
                                          rl_sampler_controller=rl_sampler_controller,
                                          lambda_simi=(
                                              float(rl_lambda_simi_resolved)
                                              if rl_sampler_controller is not None
                                              else 0.0
                                          ),
                                          suppress_missing_att_warning=suppress_missing_att_warning,
                                          batch_callback=batch_event_cb,
                                          accumulation_steps=accumulation_steps,
                                          loss_weight_mode=str(best_params.get('loss_weight_mode', 'uniform')),
                                          ranking_loss_mode=ranking_loss_mode,
                                          ranking_loss_weight=ranking_loss_weight,
                                          ranking_loss_margin=ranking_loss_margin,
                                          ranking_loss_max_pairs=ranking_loss_max_pairs)
        train_ranking_loss = getattr(train_loader, "last_train_ranking_loss", None)
        if writer is not None:
            writer.add_scalar("Loss/train", loss, epoch)

        # Validación (siempre sobre el grafo original)
        model.use_checkpointing = False
        eval_neighbors_cfg = _resolve_eval_neighbors_cfg(base_graph)
        val_key = 'val_mask'
        val_res = test(
            model,
            base_graph,
            node_type='pm',
            batch_size=batch_size_hp,
            masks=[val_key],
            num_neighbors=eval_neighbors_cfg,
            criterion=criterion,
        )
        if not val_res or val_key not in val_res:
            raise RuntimeError(
                "No se pudieron obtener resultados sobre val_mask durante entrenamiento."
            )
        val_loss = val_res[val_key].get('loss')
        try:
            val_loss = float(val_loss)
        except (TypeError, ValueError):
            val_loss = float("nan")
        if not math.isfinite(val_loss):
            raise RuntimeError(
                "No se pudo calcular validation loss finita sobre val_mask durante entrenamiento."
            )
        if temporal_module is not None:
            temporal_module.train()
        model.use_checkpointing = True
        if writer is not None:
            try:
                writer.add_scalar("Loss/val", val_loss, epoch)
            except Exception:
                pass
        
        val_f1 = 0.0
        val_f1_pos = None
        val_f1_macro = None
        val_precision_pos = None
        val_recall_pos = None
        val_accuracy = None
        val_auc = None
        val_auprc = None
        val_mcc = None
        val_tau = None
        val_far = None
        val_f05 = None
        val_recall_at_k = {}
        val_precision_at_k = {}
        val_objective_score = None
        if val_res and val_key in val_res:
            y_true_val = val_res[val_key]['true'].numpy().ravel()
            y_prob1_val = val_res[val_key]['probs'][:, 1].numpy().ravel()

            score_f1 = None
            score_recall = None
            score_far = None
            score_accuracy = None
            score_f05 = None

            if len(np.unique(y_true_val)) > 1:
                # Métricas de referencia (F1 clásico con beta=1.0).
                tau, P, R, f1_score = pick_tau_fbeta(y_true_val, y_prob1_val, beta=1.0)
                val_f1 = f1_score
                val_tau = tau

                # Métricas alineadas al objetivo de HPO/final training.
                tau_obj, p_obj, r_obj, fbeta_obj = pick_tau_fbeta(
                    y_true_val, y_prob1_val, beta=float(objective_threshold_beta)
                )
                preds_obj = (y_prob1_val >= tau_obj).astype(int)
                cm_obj = confusion_matrix(y_true_val, preds_obj, labels=[0, 1])
                tn_obj, fp_obj, fn_obj, tp_obj = cm_obj.ravel()
                total_obj = tn_obj + fp_obj + fn_obj + tp_obj

                score_recall = float(tp_obj / (tp_obj + fn_obj)) if (tp_obj + fn_obj) else 0.0
                score_far = float(fp_obj / (fp_obj + tn_obj)) if (fp_obj + tn_obj) else 0.0
                score_accuracy = float((tp_obj + tn_obj) / total_obj) if total_obj else 0.0
                score_f1 = float(
                    2.0 * p_obj * r_obj / (p_obj + r_obj)
                ) if (p_obj + r_obj) else 0.0

                _, _, _, score_f05 = pick_tau_fbeta(y_true_val, y_prob1_val, beta=0.5)
                val_far = float(score_far)
                val_f05 = float(score_f05)

                if writer:
                    try:
                        writer.add_scalar('Calibration/Val_tau', tau, epoch)
                    except Exception:
                        pass
            else:
                # Fallback si solo hay una clase en el conjunto de validación
                report = val_res[val_key].get('report', {})
                val_f1 = report.get('Accidente (1)', {}).get('f1-score', 0.0)

            val_report = val_res[val_key].get('report', {})
            val_f1_pos = val_report.get('Accidente (1)', {}).get('f1-score')
            val_f1_macro = val_report.get('macro avg', {}).get('f1-score')
            val_precision_pos = val_report.get('Accidente (1)', {}).get('precision')
            val_recall_pos = val_report.get('Accidente (1)', {}).get('recall')
            val_accuracy = val_report.get('accuracy')
            val_auc = val_res[val_key].get('auc')
            val_auprc = val_res[val_key].get('auprc')
            val_mcc = val_res[val_key].get('mcc')
            val_recall_at_k = val_res[val_key].get('recall_at_k') or {}
            val_precision_at_k = val_res[val_key].get('precision_at_k') or {}

            if score_f1 is None:
                score_f1 = float(val_f1 or 0.0)
            if score_recall is None:
                score_recall = float(val_recall_pos or 0.0)
            if score_accuracy is None:
                score_accuracy = float(val_accuracy or 0.0)
            if score_f05 is None:
                p_pos = float(val_precision_pos or 0.0)
                r_pos = float(val_recall_pos or 0.0)
                beta2 = 0.5 ** 2
                score_f05 = (
                    float((1 + beta2) * p_pos * r_pos / (beta2 * p_pos + r_pos))
                    if (beta2 * p_pos + r_pos) > 0
                    else 0.0
                )
                val_f05 = float(score_f05)
            if score_far is None:
                cm_fallback = val_res[val_key].get('cm')
                if cm_fallback is not None and np.asarray(cm_fallback).shape == (2, 2):
                    tn_c, fp_c, _, _ = np.asarray(cm_fallback).ravel()
                    score_far = float(fp_c / (fp_c + tn_c)) if (fp_c + tn_c) else 0.0
                else:
                    score_far = 0.0
                val_far = float(score_far)

            val_objective_score = _score_from_objective_metrics(
                objective_metric,
                f1=score_f1,
                recall=score_recall,
                far=score_far,
                fbeta=score_f05,
                auprc=val_auprc,
                mcc=val_mcc,
                accuracy=score_accuracy,
            )

            if writer:
                try:
                    writer.add_scalar('Metrics/Val_F1_candidate', val_f1, epoch)
                    if val_f1_pos is not None:
                        writer.add_scalar('Metrics/Val_F1_positive', val_f1_pos, epoch)
                    if val_f1_macro is not None:
                        writer.add_scalar('Metrics/Val_F1_macro', val_f1_macro, epoch)
                    if val_precision_pos is not None:
                        writer.add_scalar('Metrics/Val_Precision_positive', val_precision_pos, epoch)
                    if val_recall_pos is not None:
                        writer.add_scalar('Metrics/Val_Recall_positive', val_recall_pos, epoch)
                    writer.add_scalar('Metrics/Val_AUPRC', val_auprc or 0.0, epoch)
                    writer.add_scalar('Metrics/Val_AUC', val_auc or 0.0, epoch)
                    if val_far is not None:
                        writer.add_scalar('Metrics/Val_FAR', val_far, epoch)
                    if val_f05 is not None:
                        writer.add_scalar('Metrics/Val_F0.5', val_f05, epoch)
                    if val_objective_score is not None:
                        writer.add_scalar('Metrics/Val_Objective', val_objective_score, epoch)
                except Exception:
                    pass

        rl_update_payload = None
        if rl_sampler_controller is not None:
            try:
                val_has_two_classes = False
                try:
                    val_has_two_classes = len(np.unique(y_true_val)) > 1
                except Exception:
                    val_has_two_classes = False
                rl_signal = val_auprc if val_has_two_classes else None
                rl_update_payload = rl_sampler_controller.update_after_validation(
                    val_auprc=rl_signal,
                    epoch=int(epoch),
                )
                rl_state_payload = rl_sampler_controller.state_dict_serializable()
                best_params["rl_sampler_state"] = rl_state_payload
                best_params["rl_thresholds"] = rl_state_payload.get("thresholds", {})
                best_params["rl_reward_history"] = rl_state_payload.get("reward_history", [])
                best_params["sampler_impl"] = "rl_top_p_rsrl"
                if writer is not None:
                    try:
                        thresholds = list((rl_state_payload.get("thresholds") or {}).values())
                        if thresholds:
                            writer.add_scalar("RLTopP/MeanThreshold", float(np.mean(thresholds)), epoch)
                        if isinstance(rl_update_payload, dict):
                            writer.add_scalar(
                                "RLTopP/PrimaryRewardDelta",
                                float(rl_update_payload.get("primary_delta", 0.0) or 0.0),
                                epoch,
                            )
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning(f"No se pudo actualizar controlador RL top-p: {exc}")

        current_monitor_values = {
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_auprc": val_auprc,
            "val_auc": val_auc,
            "val_mcc": val_mcc,
            "val_far": val_far,
            "val_f05": val_f05,
            "val_tau": val_tau,
            "val_accuracy": val_accuracy,
            "val_objective_score": val_objective_score,
        }
        for k_val, recall_val in (val_recall_at_k or {}).items():
            current_monitor_values[f"val_recall_at_{int(k_val)}"] = recall_val
        for k_val, precision_val in (val_precision_at_k or {}).items():
            current_monitor_values[f"val_precision_at_{int(k_val)}"] = precision_val
        monitor_value = _metric_value_for_monitor(monitor_metric, current_monitor_values)
        is_best, best_monitor_value, patience_counter = _update_metric_monitor(
            monitor_value=monitor_value,
            best_monitor_value=best_monitor_value,
            patience_counter=patience_counter,
            min_delta=min_delta,
            monitor_mode=monitor_mode,
        )
        if is_best:
            best_val_loss = float(val_loss)
            best_val_f1 = float(val_f1)
            best_val_objective_score = float(val_objective_score) if val_objective_score is not None else float("-inf")
            best_val_auprc = float(val_auprc) if val_auprc is not None else 0.0
            best_val_auc = float(val_auc) if val_auc is not None else None
            best_val_mcc = float(val_mcc) if val_mcc is not None else None
            best_val_far = float(val_far) if val_far is not None else None
            best_val_f05 = float(val_f05) if val_f05 is not None else None
            best_val_tau = float(val_tau) if val_tau is not None else None
            best_val_accuracy = float(val_accuracy) if val_accuracy is not None else None
            best_val_recall_at_k = dict(val_recall_at_k or {})
            best_val_precision_at_k = dict(val_precision_at_k or {})
            best_epoch = epoch
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
                f"Epoch {epoch:03d}: New best {monitor_metric} on validation: {best_monitor_value:.6f} "
                f"({objective_metric} ref={val_objective_score if val_objective_score is not None else float('nan'):.4f}, F1 ref={val_f1:.4f}).\n"
                f"  → Guardado: {os.path.basename(best_model_path_unique)} (alias variante: {os.path.basename(best_model_path)}, alias global: gat_model_BEST.pt)"
            )
            # Guardar sidecar con hiperparámetros y metadatos del entrenamiento (para ambas rutas)
            try:
                meta = dict(best_params)
                meta.update({
                    'gnn_variant': _normalize_gnn_variant(gnn_variant),
                    'variant_tag': _variant_tags(use_graphsmote, loaded_obj, gnn_variant=gnn_variant),
                    'monitor_metric': monitor_metric,
                    'monitor_mode': monitor_mode,
                    'lr_scheduler': str(lr_scheduler),
                    'monitor_value': float(monitor_value),
                    'best_monitor_value': float(best_monitor_value),
                    'best_val_loss': float(best_val_loss),
                    'best_val_f1': float(best_val_f1),
                    'best_val_objective_score': float(best_val_objective_score),
                    'best_val_auprc': float(best_val_auprc),
                    'best_val_auc': best_val_auc,
                    'best_val_mcc': best_val_mcc,
                    'best_val_far': best_val_far,
                    'best_val_f05': best_val_f05,
                    'best_val_tau': best_val_tau,
                    'best_val_accuracy': best_val_accuracy,
                    'best_val_recall_at_k': _json_safe(best_val_recall_at_k),
                    'best_val_precision_at_k': _json_safe(best_val_precision_at_k),
                    'best_epoch': int(best_epoch),
                    'objective_metric': str(objective_metric),
                    'objective_threshold_beta': float(objective_threshold_beta),
                    'use_graphsmote': bool(use_graphsmote),
                    'graphsmote_mode': str(GRAPHSMOTE_MODE),
                    'graph_hash': graph_hash,
                    'graph_file_hash': graph_file_hash,
                    'graph_hash_source': graph_hash_source,
                    'git_commit': _get_repo_version(),
                    'purpose': purpose or loaded_obj.get('purpose', 'General') if isinstance(loaded_obj, dict) else 'General',
                    'target_pos_ratio_used': float(target_pos_ratio_override),
                    'smote_every_n_epochs_used': int(smote_every_override),
                    'num_neighbors_effective': loader_num_neighbors,
                    'sampler_impl': str(best_params.get("sampler_impl", train_sampler_impl)),
                    'positive_sampler_stats': _json_safe(positive_sampler_stats),
                    'rl_sampler_state': (
                        rl_sampler_controller.state_dict_serializable()
                        if rl_sampler_controller is not None
                        else None
                    ),
                    'rl_thresholds': (
                        rl_sampler_controller.thresholds_serializable()
                        if rl_sampler_controller is not None
                        else None
                    ),
                    'rl_reward_history': (
                        list(rl_sampler_controller.reward_history)
                        if rl_sampler_controller is not None
                        else None
                    ),
                    'out_channels': int(num_classes),
                    'in_channels': int(in_channels),
                    'edge_feature_dim': int(edge_feature_dim),
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

        if scheduler is not None and batch_scheduler is None:
            try:
                scheduler_metric = float(monitor_value)
                if not math.isfinite(scheduler_metric):
                    scheduler_metric = float(val_loss)
                scheduler.step(scheduler_metric)
            except Exception:
                pass

        if progress_callback is not None:
            try:
                progress_callback(
                    epoch=epoch,
                    total=max_epochs,
                    val_loss=val_loss,
                    train_ranking_loss=train_ranking_loss,
                    best_val_loss=best_val_loss,
                    val_f1=val_f1,
                    best_val_f1=best_val_f1,
                    val_objective_score=val_objective_score,
                    best_val_objective_score=best_val_objective_score,
                    objective_metric=objective_metric,
                    monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode,
                    monitor_value=monitor_value,
                    best_monitor_value=best_monitor_value,
                    patience=patience,
                    patience_counter=patience_counter,
                )
            except Exception:
                pass

        current_lr = None
        try:
            current_lr = _scheduler_lr_value(scheduler, optimizer)
        except Exception:
            current_lr = None

        synth_count = None
        try:
            if hasattr(train_graph['pm'], 'is_synthetic'):
                synth_count = int(train_graph['pm'].is_synthetic.sum().item())
        except Exception:
            synth_count = None

        epoch_time_sec = time.time() - epoch_start_time
        _emit_training_event(
            "epoch",
            run_id,
            history_path=metrics_history_path,
            epoch=int(epoch),
            total=int(max_epochs),
            gnn_variant=_normalize_gnn_variant(gnn_variant),
            variant_tag=_variant_tags(use_graphsmote, loaded_obj, gnn_variant=gnn_variant),
            train_loss=loss,
            train_cls_loss=cls_loss,
            train_edge_loss=edge_loss,
            train_l2_att_loss=l2_att_loss,
            train_ranking_loss=train_ranking_loss,
            ranking_loss_mode=str(ranking_loss_mode_resolved),
            ranking_loss_weight=float(ranking_loss_weight_resolved),
            ranking_loss_margin=float(ranking_loss_margin_resolved),
            ranking_loss_max_pairs=int(ranking_loss_max_pairs_resolved),
            val_loss=val_loss,
            val_f1=val_f1,
            val_f1_pos=val_f1_pos,
            val_f1_macro=val_f1_macro,
            val_precision_pos=val_precision_pos,
            val_recall_pos=val_recall_pos,
            val_accuracy=val_accuracy,
            val_auc=val_auc,
            val_auprc=val_auprc,
            val_mcc=val_mcc,
            val_far=val_far,
            val_f05=val_f05,
            val_tau=val_tau,
            val_recall_at_k=_json_safe(val_recall_at_k),
            val_precision_at_k=_json_safe(val_precision_at_k),
            val_mask=val_key,
            best_val_loss=best_val_loss,
            best_val_f1=best_val_f1,
            best_val_objective_score=best_val_objective_score,
            best_val_auprc=best_val_auprc,
            best_val_auc=best_val_auc,
            best_val_mcc=best_val_mcc,
            best_val_far=best_val_far,
            best_val_f05=best_val_f05,
            best_val_tau=best_val_tau,
            best_val_accuracy=best_val_accuracy,
            best_val_recall_at_k=_json_safe(best_val_recall_at_k),
            best_val_precision_at_k=_json_safe(best_val_precision_at_k),
            best_epoch=best_epoch,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            lr_scheduler=str(lr_scheduler),
            monitor_value=monitor_value,
            best_monitor_value=best_monitor_value,
            objective_metric=objective_metric,
            objective_threshold_beta=objective_threshold_beta,
            val_objective_score=val_objective_score,
            is_best=is_best,
            patience=patience,
            patience_counter=patience_counter,
            early_stop_enabled=early_stop_enabled,
            lr=current_lr,
            lambda_H=current_lambda_H,
            lambda_edge=lambda_edge,
            lambda_l2_att=lambda_l2_att,
            train_sampler_mode=str(train_sampler_mode_resolved),
            sampler_impl=str(best_params.get("sampler_impl", train_sampler_impl)),
            positive_sampler_stats=_json_safe(positive_sampler_stats),
            rl_update=rl_update_payload,
            rl_thresholds=(
                rl_sampler_controller.thresholds_serializable()
                if rl_sampler_controller is not None
                else None
            ),
            eval_neighbors_mode=str(eval_neighbors_mode_resolved),
            graph_hash=graph_hash,
            graph_file_hash=graph_file_hash,
            graph_hash_source=graph_hash_source,
            smote_synth_count=synth_count,
            epoch_time_sec=epoch_time_sec,
        )

        if save_state_path:
            try:
                ckpt_payload = {
                    "epoch": int(epoch),
                    "max_epochs": int(max_epochs),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": float(best_val_loss),
                    "best_val_f1": float(best_val_f1),
                    "best_val_objective_score": float(best_val_objective_score),
                    "best_val_auprc": float(best_val_auprc),
                    "best_val_auc": float(best_val_auc) if best_val_auc is not None else None,
                    "best_val_mcc": float(best_val_mcc) if best_val_mcc is not None else None,
                    "best_val_far": float(best_val_far) if best_val_far is not None else None,
                    "best_val_f05": float(best_val_f05) if best_val_f05 is not None else None,
                    "best_val_tau": float(best_val_tau) if best_val_tau is not None else None,
                    "best_val_accuracy": float(best_val_accuracy) if best_val_accuracy is not None else None,
                    "best_val_recall_at_k": _json_safe(best_val_recall_at_k),
                    "best_val_precision_at_k": _json_safe(best_val_precision_at_k),
                    "best_epoch": int(best_epoch),
                    "monitor_metric": monitor_metric,
                    "monitor_mode": monitor_mode,
                    "lr_scheduler": str(lr_scheduler),
                    "monitor_value": float(monitor_value),
                    "best_monitor_value": float(best_monitor_value),
                    "patience_counter": int(patience_counter),
                    "run_id": run_id,
                    "metrics_history_path": str(metrics_history_path) if metrics_history_path else None,
                    "graph_hash": graph_hash,
                    "graph_file_hash": graph_file_hash,
                    "graph_hash_source": graph_hash_source,
                    "gnn_variant": _normalize_gnn_variant(gnn_variant),
                    "variant_tag": _variant_tags(use_graphsmote, loaded_obj, gnn_variant=gnn_variant),
                    "use_graphsmote": bool(use_graphsmote),
                    "objective_metric": str(objective_metric),
                    "objective_threshold_beta": float(objective_threshold_beta),
                    "train_sampler_mode": str(train_sampler_mode_resolved),
                    "sampler_impl": str(best_params.get("sampler_impl", train_sampler_impl)),
                    "positive_sampler_stats": _json_safe(positive_sampler_stats),
                    "ranking_loss_mode": str(ranking_loss_mode_resolved),
                    "ranking_loss_weight": float(ranking_loss_weight_resolved),
                    "ranking_loss_margin": float(ranking_loss_margin_resolved),
                    "ranking_loss_max_pairs": int(ranking_loss_max_pairs_resolved),
                    "rl_sampler_state": (
                        rl_sampler_controller.state_dict_serializable()
                        if rl_sampler_controller is not None
                        else None
                    ),
                    "deterministic_sampling": bool(deterministic_sampling_resolved),
                    "sampling_seed": int(sampling_seed_resolved),
                    "eval_neighbors_mode": str(eval_neighbors_mode_resolved),
                    "eval_num_neighbors": eval_num_neighbors_resolved,
                    "test_eval_interval_epochs": int(test_eval_interval_epochs_resolved),
                    "last_train_ranking_loss": (
                        float(train_ranking_loss)
                        if train_ranking_loss is not None and math.isfinite(float(train_ranking_loss))
                        else None
                    ),
                    "last_val_loss": float(val_loss),
                    "last_val_f1": float(val_f1) if val_f1 is not None else None,
                    "last_val_objective_score": float(val_objective_score) if val_objective_score is not None else None,
                    "last_val_auc": float(val_auc) if val_auc is not None else None,
                    "last_val_auprc": float(val_auprc) if val_auprc is not None else None,
                    "last_val_mcc": float(val_mcc) if val_mcc is not None else None,
                    "last_val_far": float(val_far) if val_far is not None else None,
                    "last_val_f05": float(val_f05) if val_f05 is not None else None,
                    "last_val_tau": float(val_tau) if val_tau is not None else None,
                    "last_val_accuracy": float(val_accuracy) if val_accuracy is not None else None,
                    "epoch_time_sec": float(epoch_time_sec),
                }
                torch.save(ckpt_payload, save_state_path)
            except Exception as exc:
                logger.warning(f"No se pudo guardar checkpoint: {exc}")

        last_completed_epoch = int(epoch)

        if (
            test_eval_interval_epochs_resolved > 0
            and int(epoch) % int(test_eval_interval_epochs_resolved) == 0
        ):
            test_threshold = None
            try:
                if val_tau is not None and math.isfinite(float(val_tau)):
                    test_threshold = float(val_tau)
            except Exception:
                test_threshold = None
            _emit_training_event(
                "test_start",
                run_id,
                epoch=int(epoch),
                total=int(max_epochs),
                best_epoch=int(best_epoch),
                eval_target="current_epoch",
                automatic=True,
                checkpoint_path=None,
                threshold=test_threshold,
            )
            try:
                test_summary = _test_current_model_during_training(
                    model=model,
                    base_graph=base_graph,
                    node_type="pm",
                    batch_size=int(batch_size_hp),
                    num_neighbors=eval_neighbors_cfg,
                    threshold=test_threshold,
                    device=device,
                    epoch=int(epoch),
                    best_epoch=int(best_epoch),
                    automatic=True,
                )
                _emit_training_event(
                    "test_result",
                    run_id,
                    history_path=metrics_history_path,
                    total=int(max_epochs),
                    **test_summary,
                )
            except Exception as exc:
                logger.warning(f"No se pudo ejecutar test automatico en epoch {epoch}: {exc}")
                _emit_training_event(
                    "test_error",
                    run_id,
                    history_path=metrics_history_path,
                    epoch=int(epoch),
                    total=int(max_epochs),
                    best_epoch=int(best_epoch),
                    eval_target="current_epoch",
                    automatic=True,
                    checkpoint_path=None,
                    error=str(exc),
                )

        if _training_test_requested(should_test):
            best_checkpoint_for_test = None
            for candidate_path in (best_model_path_unique, best_model_path):
                try:
                    if candidate_path and os.path.exists(candidate_path):
                        best_checkpoint_for_test = str(candidate_path)
                        break
                except Exception:
                    continue
            test_threshold = None
            try:
                if best_val_tau is not None and math.isfinite(float(best_val_tau)):
                    test_threshold = float(best_val_tau)
            except Exception:
                test_threshold = None
            _emit_training_event(
                "test_start",
                run_id,
                epoch=int(epoch),
                total=int(max_epochs),
                best_epoch=int(best_epoch),
                eval_target="best_checkpoint",
                automatic=False,
                checkpoint_path=best_checkpoint_for_test,
                threshold=test_threshold,
            )
            try:
                if best_checkpoint_for_test is None:
                    raise FileNotFoundError(
                        "Aun no hay checkpoint BEST disponible para test intermedio."
                    )
                test_summary = _test_best_checkpoint_during_training(
                    model=model,
                    best_checkpoint_path=best_checkpoint_for_test,
                    base_graph=base_graph,
                    node_type="pm",
                    batch_size=int(batch_size_hp),
                    num_neighbors=eval_neighbors_cfg,
                    threshold=test_threshold,
                    device=device,
                    epoch=int(epoch),
                    best_epoch=int(best_epoch),
                )
                _emit_training_event(
                    "test_result",
                    run_id,
                    history_path=metrics_history_path,
                    total=int(max_epochs),
                    **test_summary,
                )
            except Exception as exc:
                logger.warning(f"No se pudo ejecutar test intermedio: {exc}")
                _emit_training_event(
                    "test_error",
                    run_id,
                    history_path=metrics_history_path,
                    epoch=int(epoch),
                    total=int(max_epochs),
                    best_epoch=int(best_epoch),
                    eval_target="best_checkpoint",
                    automatic=False,
                    checkpoint_path=best_checkpoint_for_test,
                    error=str(exc),
                )

        if _training_stop_requested(should_stop):
            logger.info(
                "Stop requested after epoch %d; checkpoint preserved at %s.",
                epoch,
                save_state_path or "<sin checkpoint path>",
            )
            stopped_by_user = True
            _emit_training_event(
                "stop_requested",
                run_id,
                epoch=int(epoch),
                total=int(max_epochs),
                graph_hash=graph_hash,
                graph_file_hash=graph_file_hash,
                graph_hash_source=graph_hash_source,
                checkpoint_path=str(save_state_path) if save_state_path else None,
            )
            break

        if early_stop_enabled and patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d by %s (best=%g, last=%g, patience=%d, min_delta=%g).",
                epoch,
                monitor_metric,
                best_monitor_value,
                monitor_value,
                patience,
                min_delta,
            )
            stopped_early = True
            break

        # Limpieza de memoria por epoch (útil en grafos grandes)
        try:
            del val_res
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if writer is not None:
        writer.close()
    epochs_run = max(int(last_completed_epoch), 0)
    _emit_training_event(
        "train_end",
        run_id,
        history_path=metrics_history_path,
        epochs_run=epochs_run,
        total=int(max_epochs),
        metrics_history_path=str(metrics_history_path) if metrics_history_path else None,
        best_val_loss=best_val_loss,
        best_val_f1=best_val_f1,
        best_val_objective_score=best_val_objective_score,
        best_val_auprc=best_val_auprc,
        best_val_auc=best_val_auc,
        best_val_mcc=best_val_mcc,
        best_val_far=best_val_far,
        best_val_f05=best_val_f05,
        best_val_tau=best_val_tau,
        best_val_accuracy=best_val_accuracy,
        best_epoch=best_epoch,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        lr_scheduler=str(lr_scheduler),
        best_monitor_value=best_monitor_value,
        objective_metric=objective_metric,
        objective_threshold_beta=objective_threshold_beta,
        graph_hash=graph_hash,
        graph_file_hash=graph_file_hash,
        graph_hash_source=graph_hash_source,
        stopped_early=stopped_early,
        stopped_by_user=stopped_by_user,
        train_sampler_mode=str(train_sampler_mode_resolved),
        sampler_impl=str(best_params.get("sampler_impl", train_sampler_impl)),
        positive_sampler_stats=_json_safe(positive_sampler_stats),
        ranking_loss_mode=str(ranking_loss_mode_resolved),
        ranking_loss_weight=float(ranking_loss_weight_resolved),
        ranking_loss_margin=float(ranking_loss_margin_resolved),
        ranking_loss_max_pairs=int(ranking_loss_max_pairs_resolved),
    )
    if stopped_by_user:
        logger.info(
            f"Training stopped by user after {epochs_run} epochs. "
            f"Best {monitor_metric}: {best_monitor_value:.6f} "
            f"(best_val_loss={best_val_loss:.6f}) "
            f"({objective_metric} ref={best_val_objective_score:.4f}) "
            f"(F1 ref={best_val_f1:.4f}) at epoch {best_epoch}."
        )
    else:
        logger.info(
            f"Training finished. Best {monitor_metric}: {best_monitor_value:.6f} "
            f"(best_val_loss={best_val_loss:.6f}) "
            f"({objective_metric} ref={best_val_objective_score:.4f}) "
            f"(F1 ref={best_val_f1:.4f}) at epoch {best_epoch}."
        )

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

def _calibrated_probability_tensor(
    prob1: torch.Tensor,
    calibration_model,
) -> torch.Tensor:
    """Apply a CPU calibrator and return probabilities with the source tensor dtype/device."""
    prob1_np = prob1.detach().cpu().numpy()
    prob1_cal_np = _apply_platt_model(prob1_np, calibration_model)
    prob1_cal_np = np.asarray(prob1_cal_np, dtype=np.float32).reshape(prob1_np.shape)
    return torch.as_tensor(
        prob1_cal_np,
        dtype=prob1.dtype,
        device=prob1.device,
    )

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

        # Arquitectura — rangos calibrados para clase rara (~0.3%):
        #   hidden 64-128: capacidad razonable sin overfit en ~258 positivos
        #   heads 4-8:     atención multi-cabeza efectiva
        #   layers 2-3:    evita oversmoothing (clase rara se diluye con +profundidad)
        #   dropout 0.1-0.3: rango clásico, descarta valores patológicos
        hidden_channels = trial.suggest_int('hidden_channels', 64, 128, step=32)
        num_heads = trial.suggest_int('num_heads', 4, 8, step=2)
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        num_layers = trial.suggest_int('num_layers', 2, 3)
        aggr1 = trial.suggest_categorical('aggr1', ['sum', 'mean', 'max'])
        aggr2 = trial.suggest_categorical('aggr2', ['sum', 'mean', 'max'])
        use_checkpointing_flag = trial.suggest_categorical('use_checkpointing', [False, True])

        neighbor_candidates = {
            'compact':    [15, 10],
            'focused':    [10, 5],
            'broad':      [25, 15, 10],
            'wide':       [30, 20],
            # Fanout asimétrico por tipo de arista (claves string =
            # nombre de relación; _resolve_num_neighbors mapea a tuple).
            'asymmetric': {'temporal': [25, 25], 'spatial': [3, 3]},
        }
        neighbor_choice = trial.suggest_categorical('num_neighbors_choice', list(neighbor_candidates.keys()))
        neighbor_profile = neighbor_candidates[neighbor_choice]

        # Optimizador
        overrides = optimizer_overrides or {}
        gnn_variant = overrides.get('gnn_variant', GNN_VARIANT)
        trial.set_user_attr('gnn_variant', _normalize_gnn_variant(gnn_variant))
        use_residual = _safe_bool(overrides.get("use_residual"), True)
        use_relation_self_loops = _safe_bool(overrides.get("use_relation_self_loops"), False)
        trial.set_user_attr("use_residual", bool(use_residual))
        trial.set_user_attr("use_relation_self_loops", bool(use_relation_self_loops))
        accumulation_steps = int(overrides.get("accumulation_steps", ACCUMULATION_STEPS))
        
        lr_override = overrides.get('lr')
        if lr_override is not None:
             lr = trial.suggest_float('lr', float(lr_override), float(lr_override))
        else:
             lr = trial.suggest_float('lr', 5e-5, 1e-2, log=True)

        opt_name_override = overrides.get('optimizer')
        if opt_name_override:
            optimizer_name = trial.suggest_categorical('optimizer', [str(opt_name_override)])
        else:
            optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RAdam', 'Lion'])

        raw_scheduler_choices = overrides.get("lr_scheduler_choices")
        if raw_scheduler_choices is None:
            raw_scheduler_choices = overrides.get("lr_scheduler")
        if isinstance(raw_scheduler_choices, (list, tuple, set)):
            scheduler_choices = []
            for choice in raw_scheduler_choices:
                normalized_choice = _normalize_lr_scheduler_choice(choice)
                if normalized_choice not in scheduler_choices:
                    scheduler_choices.append(normalized_choice)
            scheduler_choices = scheduler_choices or ["one_cycle"]
        elif raw_scheduler_choices:
            scheduler_choices = [_normalize_lr_scheduler_choice(raw_scheduler_choices)]
        else:
            scheduler_choices = ["one_cycle"]
        lr_scheduler = trial.suggest_categorical("lr_scheduler", scheduler_choices)
        trial.set_user_attr("lr_scheduler", str(lr_scheduler))

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

        # Ponderación de la pérdida por distancia accidente→pórtico aguas arriba.
        # "uniform": cada nodo positivo pesa 1.0 (status quo).
        # "distance": peso = clip(1 - dist/5km, 0.2, 1.0) por nodo positivo.
        loss_weight_mode = trial.suggest_categorical(
            "loss_weight_mode", ["uniform", "distance"]
        )

        batch_size_candidate = trial.suggest_int('batch_size', 512, 4096, step=512)
        # Si el perfil es dict, serializar como JSON para sobrevivir CSV roundtrip
        # (pandas convierte dicts a repr Python con comillas simples y json.loads falla).
        nb_attr = json.dumps(neighbor_profile) if isinstance(neighbor_profile, dict) else neighbor_profile
        trial.set_user_attr('num_neighbors', nb_attr)
        trial.set_user_attr('use_checkpointing', use_checkpointing_flag)

        # --- Construcción del Modelo ---
        num_classes = 2
        edge_feature_dim = _detect_edge_feature_dim(data, node_type='pm')
        edge_feature_dims_per_type = _detect_edge_feature_dims(data)
        in_channels = data['pm'].x.shape[1]

        model = _build_gnn_model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_heads=num_heads,
            dropout=dropout,
            edge_feature_dim=edge_feature_dim,
            num_layers=num_layers,
            aggr1=aggr1,
            aggr2=aggr2,
            use_checkpointing=use_checkpointing_flag,
            gnn_variant=gnn_variant,
            sequence_index=sequence_index_global,
            num_nodes=data['pm'].num_nodes,
            device=device,
            use_residual=use_residual,
            use_relation_self_loops=use_relation_self_loops,
            require_temporal_head=_variant_has_temporal_head(gnn_variant),
            edge_types=tuple(data.edge_types) if hasattr(data, 'edge_types') else None,
            edge_feature_dims=edge_feature_dims_per_type,
        )
        temporal_module = getattr(model, 'temporal_head', None)

        edge_gen = RelEdgeGen({ntype: hidden_channels * num_heads for ntype in data.node_types}, data.edge_types).to(device) if use_graphsmote_search else None
        
        optimizer_cls = get_optimizer_cls(optimizer_name)
        optimizer = optimizer_cls(
            list(model.parameters()) + (list(edge_gen.parameters()) if edge_gen else []),
            lr=lr, weight_decay=weight_decay
        )

        # --- Criterio de Pérdida (condicional) ---
        criterion = None
        if loss_type == 'FocalLoss':
            alpha_val = None
            if not use_graphsmote_search:
                # Para clase rara (~0.3%) el alpha óptimo de la clase positiva
                # vive cerca de 1. Muestreamos (1-alpha) log-uniform sobre
                # [1e-3, 3e-1] para cubrir bien la cola: focal_alpha ∈ [0.7, 0.999].
                focal_alpha_complement = trial.suggest_float(
                    'focal_alpha_complement', 1e-3, 3e-1, log=True
                )
                focal_alpha = 1.0 - focal_alpha_complement
                trial.set_user_attr('focal_alpha', focal_alpha)
                alpha_val = [focal_alpha_complement, focal_alpha]
            criterion = FocalLoss(gamma=trial.suggest_float('focal_gamma', 1.0, 3.0), alpha=alpha_val)
        else: # CrossEntropy
            weight = None
            if not use_graphsmote_search:
                y_train = data['pm'].y[data['pm'].train_mask]
                counts = torch.bincount(y_train)
                if counts.numel() > 1:
                    # Tempered class weights: sqrt(1/counts) en lugar de 1/counts.
                    # Con 0.3% positivos el ratio crudo es ~333× y desestabiliza
                    # gradientes; sqrt lo baja a ~18× sin perder señal de balance.
                    weight = (1.0 / counts.float()).sqrt().to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)

        # --- Preparación para el loop de entrenamiento ---
        base_graph = data
        train_graph = base_graph
        z2x_decoders = train_z2x_decoders(model, base_graph, device=device, epochs=DECODER_EPOCHS) if use_graphsmote_search else None
        
        # _resolve_num_neighbors maneja perfiles list (compact/focused/...) y dict
        # (asymmetric, con claves string mapeadas a tuple por nombre de relación).
        num_neighbors_dict = _resolve_num_neighbors(neighbor_profile, NUM_NEIGHBORS, train_graph.edge_types)
        train_loader = NeighborLoader(
            train_graph.cpu(),
            input_nodes=('pm', train_graph['pm'].train_mask),
            num_neighbors=num_neighbors_dict,
            batch_size=batch_size_candidate,
            shuffle=True
        )
        
        try:
            scheduler = _build_lr_scheduler(
                optimizer,
                lr_scheduler,
                max_lr=float(lr),
                steps_per_epoch=_optimizer_steps_per_epoch(
                    train_loader,
                    accumulation_steps,
                ),
                epochs=int(NUM_EPOCHS_OPTUNA),
                monitor_mode="max",
            )
        except Exception as exc:
            logger.warning(f"No se pudo crear lr_scheduler={lr_scheduler}: {exc}")
            scheduler = None
        batch_scheduler = scheduler if _lr_scheduler_steps_per_batch(lr_scheduler) else None

        best_f05 = -1.0
        best_val_tau = 0.5
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
            _prime_temporal_cache_if_needed(model, train_graph, node_type='pm', context=f"hpo_epoch_{epoch}")
            train_minibatch(
                model, train_loader, optimizer, criterion, grad_clip_value=float(overrides.get('grad_clip', 1.0)), device=device,
                use_amp=False, scaler=None, scheduler=batch_scheduler, writer=None, epoch=epoch,
                lambda_H=current_lambda_H, node_type='pm', edge_gen=edge_gen, lambda_edge=lambda_edge,
                lambda_l2_att=lambda_l2_att,
                accumulation_steps=accumulation_steps,
                loss_weight_mode=loss_weight_mode,
            )

            # Validación periódica
            if epoch % 2 == 0:
                val_results = test(
                    model,
                    base_graph,
                    node_type='pm',
                    batch_size=batch_size_candidate,
                    masks=['val_mask'],
                )
                if temporal_module is not None:
                    temporal_module.train()
                if not val_results or 'val_mask' not in val_results:
                    raise RuntimeError("No se pudo evaluar val_mask durante HPO.")

                y_true_val = val_results['val_mask']['true'].numpy().ravel()
                y_prob1_val = val_results['val_mask']['probs'][:, 1].numpy().ravel()

                if len(np.unique(y_true_val)) < 2:
                    continue

                auprc = average_precision_score(y_true_val, y_prob1_val)
                tau, P, R, f05 = pick_tau_fbeta(y_true_val, y_prob1_val, beta=0.5)
                if scheduler is not None and batch_scheduler is None:
                    try:
                        scheduler.step(float(f05))
                    except Exception:
                        pass
                
                trial.report(f05, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if f05 > best_f05:
                    best_f05 = f05
                    best_val_tau = tau
                    best_epoch = epoch
                
                # Sanity Check
                pos_rate = (y_prob1_val >= tau).mean()
                if pos_rate > 0.6: # Si más del 60% es predicho como positivo, penalizar.
                    return best_f05 * 0.7 

        trial.set_user_attr("best_f0.5", best_f05)
        trial.set_user_attr("best_val_tau", best_val_tau)
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

def search_hyperparameters(
    loaded_obj,
    use_graphsmote_search=None,
    optimizer_overrides=None,
    reuse_existing: bool = True,
):
    global data, sequence_index_global, sequence_config_global
    if use_graphsmote_search is None:
        use_graphsmote_search = False
        logger.info("use_graphsmote_search no especificado; usando False por defecto no interactivo.")

    if use_graphsmote_search:
        logger.info("La búsqueda de hiperparámetros incluirá opciones de GraphSMOTE.")
    else:
        logger.info("La búsqueda de hiperparámetros NO incluirá opciones de GraphSMOTE.")

    # --- Carga inteligente de HPO ---
    graph_identity = _resolve_graph_identity(loaded_obj)
    graph_hash = graph_identity.get("graph_hash")
    graph_file_hash = graph_identity.get("graph_file_hash")
    graph_hash_source = graph_identity.get("graph_hash_source")

    if graph_hash:
        # Buscar todos los HPO del mismo grafo y luego filtrar por variante
        pattern_any = os.path.join(RESULTADOS_DIR, f"optuna_hyperparams_*{graph_hash[:16]}*.csv")
        candidates = sorted(glob.glob(pattern_any), key=os.path.getmtime, reverse=True)
        if candidates:
            # Filtrar por SMOTE y por ImGAGN si aplica; preferir coincidencias exactas de variante
            want_smote = bool(use_graphsmote_search)
            want_imgagn = _has_imgagn(loaded_obj)
            want_gnn_component = _gnn_variant_tag_component(GNN_VARIANT)
            preferred = []
            fallback = []
            for p in candidates:
                name = os.path.basename(p)
                has_smote = ("_GraphSMOTE" in name)
                has_imgagn = ("_ImGAGN" in name)
                has_base = ("_Base" in name)
                if "_GNN_" in name and want_gnn_component not in name:
                    continue
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
                    best_params = _load_hyperparams_csv(latest_hpo_file)
                    if bool(reuse_existing):
                        logger.info("Reutilizando hiperparámetros existentes.")
                        return best_params
                    logger.info("reuse_existing=False. Iniciando nueva búsqueda.")
                except Exception as e:
                    logger.error(f"No se pudo cargar o procesar '{os.path.basename(latest_hpo_file)}': {e}. Iniciando nueva búsqueda.")

    # Dispositivo
    # Selection automática de dispositivo
    device = get_auto_device()
    logger.info(f"Usando dispositivo: {device}")

    data = loaded_obj['data'].to(device)
    sequence_index_global = loaded_obj.get('sequence_index')
    sequence_config_global = loaded_obj.get('sequence_config')

    has_val_mask = hasattr(data["pm"], "val_mask")
    val_mask_count = int(data["pm"].val_mask.sum().item()) if has_val_mask else 0
    if val_mask_count == 0:
        raise RuntimeError(
            "No se puede ejecutar HPO sin val_mask valido. "
            "Cree un split temporal con validacion antes de lanzar Optuna."
        )

    # Sampler y Pruner
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True)
    pruner  = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=NUM_EPOCHS_OPTUNA, reduction_factor=3)

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    study_base = "gnn_optuna_main"
    if graph_hash:
        study_base = f"{study_base}_{graph_hash[:16]}"
    variant_tag = _variant_tags(use_graphsmote_search, loaded_obj, gnn_variant=GNN_VARIANT)
    study_base = f"{study_base}{variant_tag}"
    study_name = re.sub(r"[^A-Za-z0-9_\\-]", "_", study_base)
    storage_path = os.path.join(RESULTADOS_DIR, "optuna_studies.db")
    storage_url = f"sqlite:///{storage_path}"
    try:
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=60,
            grace_period=120,
        )
    except Exception:
        storage = storage_url

    # Crear o recuperar estudio
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    logger.info(f"Optuna study activo: {study.study_name}")

    try:
        try:
            done_states = {
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
            }
        except Exception:
            done_states = set()
        if done_states:
            done_trials = sum(1 for t in study.trials if t.state in done_states)
        else:
            done_trials = len(study.trials)
        remaining_trials = max(0, int(N_TRIALS) - int(done_trials))
        def _save_live(study_obj, _trial):
            try:
                live_path = os.path.join(
                    RESULTADOS_DIR, f"optuna_full_study_live_{study_obj.study_name}.csv"
                )
                study_obj.trials_dataframe().to_csv(live_path, index=False)
            except Exception:
                pass

        if remaining_trials <= 0:
            logger.info(
                f"Optuna reanudado: {done_trials} ensayos ya completados (objetivo {N_TRIALS})."
            )
            _save_live(study, None)
        else:
            # Pasar el nombre de argumento correcto según la firma de objective()
            study.optimize(
                lambda tr: objective(
                    tr,
                    device,
                    use_graphsmote_search=use_graphsmote_search,
                    optimizer_overrides=optimizer_overrides,
                ),
                n_trials=remaining_trials,
                show_progress_bar=True,
                callbacks=[_save_live],
            )
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
    best_params['gnn_variant'] = _normalize_gnn_variant(best_params.get('gnn_variant', GNN_VARIANT))
    best_params['variant_tag'] = _variant_tags(use_graphsmote_search, loaded_obj, gnn_variant=best_params['gnn_variant'])
    best_params.setdefault('undersample', 'auto')
    best_params['graph_hash'] = graph_hash
    best_params['graph_file_hash'] = graph_file_hash
    best_params['graph_hash_source'] = graph_hash_source
    try:
        best_params['use_imgagn_aug'] = bool(_has_imgagn(loaded_obj))
    except Exception:
        best_params['use_imgagn_aug'] = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Etiquetas de variante
    variant_tag = _variant_tags(use_graphsmote_search, loaded_obj, gnn_variant=best_params.get('gnn_variant', GNN_VARIANT))
    hash_tag = f"_{graph_hash[:16]}" if graph_hash else ""
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    best_params_path = os.path.join(RESULTADOS_DIR, f"optuna_hyperparams_{timestamp}{hash_tag}{variant_tag}.csv")
    pd.DataFrame([best_params]).to_csv(best_params_path, index=False)
    logger.info(f"Hiperparámetros top guardados en -> {os.path.basename(best_params_path)}")

    trials_df = study.trials_dataframe()
    trials_df['use_graphsmote'] = use_graphsmote_search
    trials_df['graph_hash'] = graph_hash
    trials_df['graph_file_hash'] = graph_file_hash
    trials_df['graph_hash_source'] = graph_hash_source
    trials_df['gnn_variant'] = _normalize_gnn_variant(best_params.get('gnn_variant', GNN_VARIANT))
    trials_df['variant_tag'] = variant_tag
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

    graph_identity = _resolve_graph_identity(loaded_obj)
    graph_hash = graph_identity.get("graph_hash")
    graph_file_hash = graph_identity.get("graph_file_hash")
    graph_hash_source = graph_identity.get("graph_hash_source")

    # Optuna: sampler y pruner razonables
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    study_base = "imgagn_optuna"
    if graph_hash:
        study_base = f"{study_base}_{graph_hash[:16]}"
    study_name = re.sub(r"[^A-Za-z0-9_\\-]", "_", study_base)
    storage_path = os.path.join(RESULTADOS_DIR, "optuna_studies.db")
    storage_url = f"sqlite:///{storage_path}"
    try:
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=60,
            grace_period=120,
        )
    except Exception:
        storage = storage_url
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    logger.info(f"Optuna study (ImGAGN) activo: {study.study_name}")

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
        try:
            done_states = {
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL,
            }
        except Exception:
            done_states = set()
        if done_states:
            done_trials = sum(1 for t in study.trials if t.state in done_states)
        else:
            done_trials = len(study.trials)
        remaining_trials = max(0, int(N_TRIALS) - int(done_trials))
        def _save_live(study_obj, _trial):
            try:
                live_path = os.path.join(
                    RESULTADOS_DIR, f"imgagn_full_study_live_{study_obj.study_name}.csv"
                )
                study_obj.trials_dataframe().to_csv(live_path, index=False)
            except Exception:
                pass

        if remaining_trials <= 0:
            logger.info(
                f"Optuna ImGAGN reanudado: {done_trials} ensayos ya completados (objetivo {N_TRIALS})."
            )
            _save_live(study, None)
        else:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                show_progress_bar=True,
                callbacks=[_save_live],
            )
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
    best_params['graph_hash'] = graph_hash
    best_params['graph_file_hash'] = graph_file_hash
    best_params['graph_hash_source'] = graph_hash_source
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_tag = f"_{graph_hash[:16]}" if graph_hash else ""
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    best_path = os.path.join(RESULTADOS_DIR, f"imgagn_hyperparams_{timestamp}{hash_tag}.csv")
    pd.DataFrame([best_params]).to_csv(best_path, index=False)
    logger.info(f"[ImGAGN] Hiperparámetros top guardados en -> {os.path.basename(best_path)}")

    full_path = os.path.join(RESULTADOS_DIR, f"imgagn_full_study_{timestamp}{hash_tag}.csv")
    trials_df = study.trials_dataframe()
    trials_df['graph_hash'] = graph_hash
    trials_df['graph_file_hash'] = graph_file_hash
    trials_df['graph_hash_source'] = graph_hash_source
    trials_df.to_csv(full_path, index=False)
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
    graph_identity = _resolve_graph_identity(loaded_obj)
    graph_hash = graph_identity.get("graph_hash")
    graph_file_hash = graph_identity.get("graph_file_hash")
    graph_hash_source = graph_identity.get("graph_hash_source")
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
    best_params["graph_hash"] = graph_hash
    best_params["graph_file_hash"] = graph_file_hash
    best_params["graph_hash_source"] = graph_hash_source

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

    # Trazabilidad: registrar recall, seed y config usados en ImGAGN
    best_recall = None
    try:
        if isinstance(res, dict) and "best_train_recall" in res:
            if torch.is_tensor(res["best_train_recall"]):
                best_recall = float(res["best_train_recall"].item())
            else:
                best_recall = float(res["best_train_recall"])
    except Exception:
        best_recall = None
    imgagn_meta = dict(best_params) if isinstance(best_params, dict) else {}
    imgagn_meta.update({
        "best_train_recall": best_recall,
        "seed": int(res.get("seed", SEED)) if isinstance(res, dict) else int(SEED),
        "config": res.get("config", cfg.__dict__ if hasattr(cfg, "__dict__") else cfg),
    })

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
        save_obj['imgagn_best_params'] = imgagn_meta
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

def run_gat_testing(
    loaded_obj,
    *,
    use_graphsmote: Optional[bool] = None,
    hparams_path: Optional[str] = None,
    hparams_index: Optional[int] = None,
    apply_prior_shift: bool = False,
    p_real: float = 0.01,
    analyze_xai: bool = False,
    xai_k_heads: int = 3,
    xai_percentile: float = 0.95,
):
    """
    Carga un modelo GAT pre-entrenado y sus hiperparámetros para evaluarlo
    en el conjunto de datos de un grafo cargado.
    """
    if use_graphsmote is None:
        try:
            use_graphsmote = bool(
                "pm" in loaded_obj["data"].node_types
                and hasattr(loaded_obj["data"]["pm"], "is_synthetic")
                and loaded_obj["data"]["pm"].is_synthetic.sum().item() > 0
            )
        except Exception:
            use_graphsmote = False
        logger.info(f"GraphSMOTE no especificado para test; usando detección automática: {use_graphsmote}.")

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

    try:
        hp_files = sorted(hp_files, key=os.path.getmtime)
        if hparams_path:
            hp_path = str(hparams_path)
        elif hparams_index is not None:
            sel = int(hparams_index) - 1
            if not (0 <= sel < len(hp_files)):
                raise ValueError("hparams_index fuera de rango.")
            hp_path = hp_files[sel]
        else:
            hp_path = hp_files[-1]
        best_params = _load_hyperparams_csv(hp_path)
        
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
    edge_feature_dims_per_type = _detect_edge_feature_dims(data)
    encoder_overrides = _parse_edge_encoder_per_type(best_params.get('edge_encoder_per_type'))

    in_channels = data['pm'].x.shape[1]
    gnn_variant = best_params.get('gnn_variant', GNN_VARIANT)
    model = _build_gnn_model(
        in_channels=in_channels,
        hidden_channels=best_params['hidden_channels'],
        out_channels=num_classes,
        num_heads=best_params['num_heads'],
        dropout=best_params['dropout'],
        edge_feature_dim=edge_feature_dim,
        num_layers=int(best_params.get('num_layers', len(NUM_NEIGHBORS))),
        use_checkpointing=True,
        aggr1=best_params.get('aggr1', 'sum'),
        aggr2=best_params.get('aggr2', 'sum'),
        gnn_variant=gnn_variant,
        sequence_index=loaded_obj.get('sequence_index'),
        num_nodes=data['pm'].num_nodes,
        device=device,
        use_residual=_safe_bool(best_params.get("use_residual"), False),
        use_relation_self_loops=_safe_bool(best_params.get("use_relation_self_loops"), True),
        require_temporal_head=_variant_has_temporal_head(gnn_variant),
        edge_types=tuple(data.edge_types) if hasattr(data, 'edge_types') else None,
        edge_feature_dims=edge_feature_dims_per_type,
        **encoder_overrides,
    )

    # 4. Cargar el estado del modelo entrenado
    best_model_path = _find_best_model_path(
        use_graphsmote,
        loaded_obj,
        best_params=best_params,
        gnn_variant=gnn_variant,
    )
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
    if 'best_val_tau' in best_params:
        tau = best_params['best_val_tau']
        print(f"ℹ️ Usando umbral (tau) pre-calculado de Optuna: {tau:.6f}")

    if tau is None:
        print("▶️ No se encontró `best_val_tau` en los hiperparámetros. Calculando umbral óptimo desde la validación...")
        # 1) Pase inicial para recolectar probabilidades en validación
        initial_results = test(model, data, node_type='pm', threshold=None, masks=['val_mask', 'train_mask'])

    if tau is None:
        if 'val_mask' not in initial_results:
            print("❌ No hay val_mask disponible para calibrar umbral. Abortando para evitar usar train_mask.")
            return
        mask_key = 'val_mask'

        if mask_key not in initial_results:
            print(f"Error: La máscara '{mask_key}' no se encontró en los resultados iniciales. Abortando.")
            return

        y_true_val = initial_results[mask_key]['true'].cpu().numpy().ravel()
        y_prob1_val_raw = initial_results[mask_key]['probs'][:, 1].cpu().numpy().ravel()

        # Split val_mask 50/50: val_cal para fit Platt, val_thr para selección
        # del umbral. Evita el doble uso del mismo subconjunto que sesga métricas.
        if y_true_val.size >= 4:
            idx_thr, idx_cal = _split_val_for_thr_cal(y_true_val.size, seed=int(SEED))
            _, platt_model = _platt_scale_probabilities(y_true_val[idx_cal], y_prob1_val_raw[idx_cal])
            y_true_val = y_true_val[idx_thr]
            y_prob1_val = _apply_platt_model(y_prob1_val_raw[idx_thr], platt_model)
        else:
            y_prob1_val, platt_model = _platt_scale_probabilities(y_true_val, y_prob1_val_raw)

        # --- Calibración por Prior Shift (opcional, controlada por parámetro) ---
        if bool(apply_prior_shift):
            try:
                p_train = initial_results['train_mask']['true'].float().mean().item()
                p_real = float(p_real)
                
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

        # Persistir curva PR (val_thr) para análisis posterior.
        try:
            best_path_for_artifact = locals().get('best_model_path')
            if best_path_for_artifact:
                pr_path = os.path.splitext(str(best_path_for_artifact))[0] + "_pr_curve_val.json"
                _save_pr_curve_artifact(y_true_val, y_prob1_val, pr_path)
        except Exception:
            pass

    # 3) Pase final con ese umbral aplicado en TODOS los splits + IC bootstrap.
    final_results = test(
        model, data, node_type='pm',
        threshold=tau, calibration_model=platt_model,
        compute_bootstrap_ci=True, bootstrap_n=1000, bootstrap_seed=int(SEED),
    )
    
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

    if bool(analyze_xai):
        try:
            analyze_and_save_relevant_edges(
                model,
                data,
                loaded_obj,
                device,
                k_heads=int(xai_k_heads),
                percentile=float(xai_percentile),
                threshold=tau,
                calibration_model=platt_model,
            )
        except Exception as e:
            logger.error(f"Ocurrió un error inesperado durante el análisis de aristas: {e}")

def analyze_and_save_relevant_edges(
    model,
    data,
    loaded_obj,
    device,
    k_heads=3,
    percentile=0.95,
    threshold: Optional[float] = None,
    calibration_model: Optional[object] = None,
    layer: str = "auto",
):
    """
    Analiza las atenciones del modelo entrenado, extrae las aristas más relevantes
    y las guarda en un archivo CSV. Usa checkpointing para reducir memoria.
    """
    logger.info("--- Iniciando análisis de aristas relevantes basado en atención ---")
    try:
        from src.gnn_xai import compute_gnn_xai_graph, save_gnn_xai_result

        result = compute_gnn_xai_graph(
            model=model,
            graph=data,
            pm_index=loaded_obj.get("pm_index") if isinstance(loaded_obj, dict) else None,
            mask_name="test_mask",
            batch_size=BATCH_SIZE,
            num_neighbors=NUM_NEIGHBORS,
            percentile=float(percentile),
            k_heads=int(k_heads),
            layer=layer,
            threshold=threshold,
            calibration_model=calibration_model,
            temporal_module=getattr(model, "temporal_head", None),
            device=device,
            model_path=None,
            graph_hash=(
                _resolve_graph_identity(loaded_obj).get("graph_hash")
                if isinstance(loaded_obj, dict)
                else None
            ),
        )
        output_dir = save_gnn_xai_result(result)
        logger.info(f"✅ XAI GNN guardado en -> {output_dir}")
        return result
    except Exception as exc:
        logger.error(f"No se pudo calcular XAI GNN reutilizable: {exc}", exc_info=True)
        return None

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

def test_graphsmote(
    loaded_obj,
    *,
    target_ratio: Optional[float] = None,
    k: Optional[int] = None,
):
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

    # 3. Parámetros explícitos para la aumentación
    # --- CORREGIDO: Calcular ratio sobre el conjunto de entrenamiento ---
    train_mask = data['pm'].train_mask
    if train_mask.sum() > 0:
        y_train = data['pm'].y[train_mask]
        actual_ratio = y_train.float().mean().item()
    else:
        actual_ratio = 0.0
        logger.warning("La máscara de entrenamiento está vacía. El ratio actual es 0.")

    if target_ratio is None:
        target_ratio = TARGET_POS_RATIO
        logger.info(f"target_ratio no especificado; usando default {target_ratio}. Ratio train actual={actual_ratio:.4f}.")
    if k is None:
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

def test_imgagn(
    loaded_obj,
    *,
    lambda1_ratio: float = 1.0,
    topk_links: int = 5,
    epochs: int = 30,
    d_steps: int = 20,
):
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

    # 3) Parámetros explícitos de ImGAGN (análogos a GraphSMOTE: ratio y k)
    #    lambda1_ratio controla cuántos nodos nuevos: (n_min + ng) / n_maj
    try:
        n_min = int(((train_mask == True) & (y_bin == 1)).sum().item())
        n_maj = int(((train_mask == True) & (y_bin == 0)).sum().item())
        cur_lambda = (n_min / max(1, n_maj)) if n_maj > 0 else 0.0
        logger.info(
            "Parámetros ImGAGN explícitos: "
            f"lambda1_ratio={lambda1_ratio}, topk_links={topk_links}, "
            f"epochs={epochs}, d_steps={d_steps}; lambda actual≈{cur_lambda:.3f}"
        )
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
