import logging
import os
import sys
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from src.graphsmote import compute_epoch_embeddings, evaluate_edge_generator_auc
from src.config import DEBUG
from src.config import (
    GRAPHSMOTE_MODE, TARGET_POS_RATIO, GRAPHSMOTE_K,
    PRETRAIN_EDGE_EPOCHS, SMOTE_EVERY_N_EPOCHS, GS_SEED,
    SAVE_AUG_GRAPH_PATH,
    NUM_NEIGHBORS,
    BATCH_SIZE
)
from torch_geometric.loader import NeighborLoader


logger = logging.getLogger(__name__)


def _mem_snapshot(tag: str) -> None:
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        try:
            import resource
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = rss_kb / (1024 ** 2) if sys.platform == "darwin" else rss_kb / 1024.0
        except Exception:
            rss = None
    parts = []
    if rss is not None:
        parts.append(f"RSS={rss:.1f}MB")
    if torch.cuda.is_available():
        try:
            parts.append(f"CUDA_alloc={torch.cuda.memory_allocated() / (1024 ** 2):.1f}MB")
            parts.append(f"CUDA_reserved={torch.cuda.memory_reserved() / (1024 ** 2):.1f}MB")
        except Exception:
            pass
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            parts.append(f"MPS_alloc={torch.mps.current_allocated_memory() / (1024 ** 2):.1f}MB")
        except Exception:
            pass
        try:
            parts.append(f"MPS_driver={torch.mps.driver_allocated_memory() / (1024 ** 2):.1f}MB")
        except Exception:
            pass
    if parts:
        logger.info(f"[MEM][train_minibatch] {tag} | " + " | ".join(parts))

# ================================
# Entrenamiento por mini-batch
# ================================
def _per_sample_classification_loss(criterion, logits, targets):
    """
    Devuelve la pérdida por muestra (sin reducir) para CrossEntropyLoss o
    cualquier criterion con atributo `reduction`. Devuelve None si no se puede.
    """
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        return F.cross_entropy(
            logits,
            targets,
            weight=getattr(criterion, "weight", None),
            ignore_index=int(getattr(criterion, "ignore_index", -100)),
            reduction="none",
            label_smoothing=float(getattr(criterion, "label_smoothing", 0.0)),
        )
    if hasattr(criterion, "reduction"):
        original = criterion.reduction
        try:
            criterion.reduction = "none"
            per_sample = criterion(logits, targets)
        finally:
            criterion.reduction = original
        if torch.is_tensor(per_sample) and per_sample.dim() >= 1:
            return per_sample
    return None


def _apply_distance_weighting(
    criterion,
    logits_m,
    y_m,
    batch_node_store,
    batch_size: int,
):
    """
    Si batch_node_store tiene `loss_weight`, devuelve cls_loss ponderada por
    distancia. Si no se puede (criterion incompatible o falta el atributo),
    cae al criterion estándar.
    """
    weights = getattr(batch_node_store, "loss_weight", None)
    if weights is None or not torch.is_tensor(weights):
        return criterion(logits_m, y_m)
    w = weights[:batch_size].to(dtype=logits_m.dtype, device=logits_m.device)
    if w.numel() != y_m.numel():
        return criterion(logits_m, y_m)
    per_sample = _per_sample_classification_loss(criterion, logits_m, y_m)
    if per_sample is None or per_sample.numel() != w.numel():
        return criterion(logits_m, y_m)
    denom = w.sum().clamp_min(1e-12)
    return (per_sample * w).sum() / denom


def _safe_get_logits(model, batch):
    """Compat: algunos modelos heterogéneos devuelven dict por tipo de nodo."""
    edge_attr_dict = {
        et: batch[et].edge_attr
        for et in batch.edge_index_dict.keys()
        if hasattr(batch[et], 'edge_attr') and batch[et].edge_attr is not None
    }
    logits, embeddings, attentions = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
    return logits, embeddings, attentions

def train_minibatch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    grad_clip_value: float = 1.0,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    writer=None,
    epoch: int = 1,
    lambda_H: float = 0.0,
    node_type: str = 'pm',
    edge_gen: Optional[torch.nn.Module] = None, # <-- Edge generator
    lambda_edge: float = 0.0, # <-- Peso para la loss de aristas
    lambda_l2_att: float = 0.0, # <-- Peso para la regularización L2 de la atención
    suppress_missing_att_warning: bool = False, # <-- Silencia aviso de atenciones ausentes
    batch_callback: Optional[Callable[..., None]] = None, # <-- Callback para progreso por batch
    batch_log_every: Optional[int] = None, # <-- Frecuencia de callback por batch
    accumulation_steps: int = 2, # <-- Acumulación de gradientes
    rl_sampler_controller: Optional[torch.nn.Module] = None,
    lambda_simi: float = 0.0,
    loss_weight_mode: str = "uniform", # <-- "uniform" | "distance"
) -> Tuple[float, float, float, float]: # Agregado avg_edge_loss
    """
    Entrena una época completa sobre un loader (vecindad) y devuelve:
      - avg_loss total
      - avg_cls_loss
      - avg_edge_loss
      - avg_l2_att_loss
    Realiza la tarea de clasificación sobre el `node_type` especificado.
    """
    model.train()
    if edge_gen is not None:
        edge_gen.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_edge_loss = 0.0
    total_l2_att_loss = 0.0
    total_h_loss = 0.0
    total_simi_loss = 0.0

    if DEBUG:
        logger.info(f"[train_minibatch] Epoch {epoch}: Iniciando entrenamiento de minibatch.")

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} (train)", leave=False)
    total_batches = len(loader)
    batch_log_step = None
    if batch_callback is not None:
        try:
            if batch_log_every is None:
                batch_log_step = max(1, int(total_batches // 50))
            else:
                batch_log_step = max(1, int(batch_log_every))
        except Exception:
            batch_log_step = 1
    # Acumulador de gradientes (evitar valores inválidos)
    try:
        accumulation_steps = max(1, int(accumulation_steps))
    except Exception:
        accumulation_steps = 1

    mem_debug = os.environ.get("GNN_MEM_DEBUG", "").lower() in ("1", "true", "yes", "y")
    mem_log_every = int(os.environ.get("GNN_MEM_DEBUG_EVERY", "200"))
    neigh_debug = os.environ.get("GNN_NEIGHBOR_DEBUG", "").lower() in ("1", "true", "yes", "y")
    
    optimizer.zero_grad(set_to_none=True)
    if DEBUG:
        logger.info(f"[train_minibatch] Epoch {epoch}: Gradientes iniciales puestos a cero.")

    accumulated_backward_batches = 0

    for i, batch in enumerate(progress_bar):
        if DEBUG:
            logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Iniciando procesamiento de batch.")
        if mem_debug and (i % mem_log_every == 0):
            _mem_snapshot(f"epoch={epoch} batch={i}")
        if neigh_debug and i == 0:
            try:
                pm_batch = batch[node_type].batch_size
                pm_nodes = batch[node_type].num_nodes
                edge_counts = {
                    str(et): int(store.edge_index.size(1))
                    for et, store in batch.edge_index_dict.items()
                }
                logger.info(
                    f"[NEIGHBOR] epoch={epoch} pm_batch={pm_batch} pm_nodes={pm_nodes} "
                    f"edge_counts={edge_counts}"
                )
            except Exception:
                pass
        batch = batch.to(device) if device is not None else batch

        def compute_loss():
            if DEBUG:
                logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Dentro de compute_loss.")
            logits_dict, embeddings_dict, attentions = _safe_get_logits(model, batch)
            pm_logits_all = logits_dict[node_type]
            pm_embeddings = embeddings_dict[node_type]
            batch_size = batch[node_type].batch_size
            y_m = batch[node_type].y[:batch_size]

            temporal_module = getattr(model, 'temporal_head', None)
            if temporal_module is not None:
                logits_m = temporal_module(pm_embeddings, batch[node_type].n_id, batch_size)
            else:
                logits_m = pm_logits_all[:batch_size]

            if loss_weight_mode == "distance":
                cls_loss = _apply_distance_weighting(
                    criterion,
                    logits_m,
                    y_m,
                    batch[node_type],
                    batch_size,
                )
            else:
                cls_loss = criterion(logits_m, y_m)

            edge_loss = torch.tensor(0.0, device=pm_embeddings.device)
            if edge_gen is not None and lambda_edge > 0:
                edge_loss_list = []
                # Edge reconstruction en mini-batch solo con nodos REALES
                for (src, rel, dst), store in batch.edge_index_dict.items():
                    # Solo procesar aristas conectadas al tipo de nodo principal
                    if src != node_type and dst != node_type:
                        continue
                        
                    key = f"{src}:{rel}:{dst}"
                    if key not in edge_gen.S: continue

                    # Asegurarse de que los embeddings para este tipo de arista existen en el batch
                    if src not in embeddings_dict or dst not in embeddings_dict:
                        if DEBUG:
                            logger.warning(f"Saltando tipo de arista ('{src}', '{dst}') por falta de embeddings en el batch.")
                        continue

                    # Solo usar nodos que están en el batch actual
                    zsrc = embeddings_dict[src]
                    zdst = embeddings_dict[dst]
                    
                    pos_src_idx = store[0]
                    pos_dst_idx = store[1]

                    # Filtrar para que solo se usen nodos del minibatch
                    mask_src = pos_src_idx < zsrc.size(0)
                    mask_dst = pos_dst_idx < zdst.size(0)
                    valid_mask = mask_src & mask_dst

                    pos_src_idx = pos_src_idx[valid_mask]
                    pos_dst_idx = pos_dst_idx[valid_mask]

                    if pos_src_idx.numel() == 0: continue

                    # Positivos
                    pos_logits = torch.sum((zsrc[pos_src_idx] @ edge_gen.S[key]) * zdst[pos_dst_idx], dim=1)
                    
                    # Negativos (muestreo)
                    num_neg_samples = pos_dst_idx.numel()
                    neg_dst_idx = torch.randint(0, zdst.size(0), (num_neg_samples,), device=zdst.device)
                    neg_logits = torch.sum((zsrc[pos_src_idx] @ edge_gen.S[key]) * zdst[neg_dst_idx], dim=1)
                    
                    edge_loss_list.append(F.binary_cross_entropy_with_logits(
                        torch.cat([pos_logits, neg_logits]),
                        torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
                    ))
                if edge_loss_list:
                    edge_loss = torch.stack(edge_loss_list).sum()

            # L2 regularization on attention weights
            l2_att_loss = torch.tensor(0.0, device=pm_embeddings.device)
            if lambda_l2_att > 0 and attentions:
                l2_att_loss_list = []
                for att in attentions.values():
                    if att is not None:
                        l2_att_loss_list.append(torch.mean(att**2))
                if l2_att_loss_list:
                    l2_att_loss = torch.stack(l2_att_loss_list).sum()
            elif lambda_l2_att > 0 and not attentions and i == 0 and not suppress_missing_att_warning:
                logger.warning("lambda_l2_att > 0, pero no se recibieron atenciones del modelo. Verifica que XAI esté activo y que el modelo guarde alpha.")

            # Entropy regularization over class probabilities (H).
            h_loss = torch.tensor(0.0, device=pm_embeddings.device)
            if lambda_H > 0:
                probs_m = F.softmax(logits_m, dim=1)
                entropy_per_node = -torch.sum(
                    probs_m * torch.log(torch.clamp(probs_m, min=1e-12)),
                    dim=1,
                )
                h_loss = torch.mean(entropy_per_node)

            simi_loss = torch.tensor(0.0, device=pm_embeddings.device)
            if rl_sampler_controller is not None and lambda_simi > 0:
                try:
                    simi_loss = rl_sampler_controller.similarity_loss(batch, node_type=node_type)
                    if not torch.isfinite(simi_loss):
                        simi_loss = torch.tensor(0.0, device=pm_embeddings.device)
                except Exception as exc:
                    if DEBUG and i == 0:
                        logger.warning(f"No se pudo calcular la similarity loss RL: {exc}")
                    simi_loss = torch.tensor(0.0, device=pm_embeddings.device)

            total_loss = (
                cls_loss
                + lambda_H * h_loss
                + lambda_edge * edge_loss
                + lambda_l2_att * l2_att_loss
                + lambda_simi * simi_loss
            )
            if DEBUG:
                grad_fn_info = total_loss.grad_fn.__class__.__name__ if total_loss.grad_fn else 'None'
                logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Loss calculada. grad_fn: {grad_fn_info}")
            return (
                total_loss,
                cls_loss.detach(),
                edge_loss.detach(),
                l2_att_loss.detach(),
                h_loss.detach(),
                simi_loss.detach(),
            )

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                raw_loss, cls_loss, edge_loss, l2_att_loss, h_loss, simi_loss = compute_loss()
                loss_for_backward = raw_loss / accumulation_steps
            
            if torch.isfinite(loss_for_backward):
                if DEBUG:
                    grad_fn_info = loss_for_backward.grad_fn.__class__.__name__ if loss_for_backward.grad_fn else 'None'
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Antes de backward (AMP). Loss grad_fn: {grad_fn_info}")
                scaler.scale(loss_for_backward).backward()
                accumulated_backward_batches += 1
                if DEBUG:
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Después de backward (AMP).")
            else:
                logger.warning("Loss is NaN, skipping backward pass.")

        else:
            raw_loss, cls_loss, edge_loss, l2_att_loss, h_loss, simi_loss = compute_loss()
            loss_for_backward = raw_loss / accumulation_steps
            if torch.isfinite(loss_for_backward):
                if DEBUG:
                    grad_fn_info = loss_for_backward.grad_fn.__class__.__name__ if loss_for_backward.grad_fn else 'None'
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Antes de backward. Loss grad_fn: {grad_fn_info}")
                loss_for_backward.backward()
                accumulated_backward_batches += 1
                if DEBUG:
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Después de backward.")
            else:
                logger.warning("Loss is NaN, skipping backward pass.")

        # --- PASO DE OPTIMIZACIÓN Y LIMPIEZA ---
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if DEBUG:
                logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Realizando paso de optimización.")
            optimizer_stepped = False
            if accumulated_backward_batches > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                    if edge_gen: torch.nn.utils.clip_grad_norm_(edge_gen.parameters(), grad_clip_value)
                    if rl_sampler_controller is not None:
                        torch.nn.utils.clip_grad_norm_(rl_sampler_controller.parameters(), grad_clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                    if edge_gen: torch.nn.utils.clip_grad_norm_(edge_gen.parameters(), grad_clip_value)
                    if rl_sampler_controller is not None:
                        torch.nn.utils.clip_grad_norm_(rl_sampler_controller.parameters(), grad_clip_value)
                    optimizer.step()
                optimizer_stepped = True
            else:
                logger.warning(
                    "No finite loss produced a backward pass in this accumulation window; "
                    "skipping optimizer step."
                )

            optimizer.zero_grad(set_to_none=True)
            accumulated_backward_batches = 0
            if DEBUG:
                logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Gradientes puestos a cero después del paso.")
            if scheduler is not None and optimizer_stepped:
                scheduler.step()

        if torch.isfinite(raw_loss):
            total_loss += float(raw_loss.item())
            total_cls_loss += float(cls_loss.item())
            total_edge_loss += float(edge_loss.item())
            total_l2_att_loss += float(l2_att_loss.item())
            total_h_loss += float(h_loss.item())
            total_simi_loss += float(simi_loss.item())

        progress_bar.set_postfix({
            'Loss': f"{raw_loss.item():.4f}" if torch.isfinite(raw_loss) else "nan",
            'CLS': f"{cls_loss.item():.4f}" if torch.isfinite(cls_loss) else "nan",
            'H': f"{h_loss.item():.4f}" if torch.isfinite(h_loss) else "nan",
            'Simi': f"{simi_loss.item():.4f}" if torch.isfinite(simi_loss) else "nan",
            'Edge': f"{edge_loss.item():.4f}" if torch.isfinite(edge_loss) else "nan",
            'L2_Att': f"{l2_att_loss.item():.4f}" if torch.isfinite(l2_att_loss) else "nan",
            'LR': f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else 'N/A'
        })
        if batch_callback is not None:
            is_last = (i + 1) >= total_batches
            if batch_log_step is None or (i % batch_log_step == 0) or is_last:
                lr_value = None
                try:
                    if scheduler is not None:
                        lr_value = float(scheduler.get_last_lr()[0])
                    elif optimizer.param_groups:
                        lr_value = float(optimizer.param_groups[0].get("lr"))
                except Exception:
                    lr_value = None
                try:
                    batch_callback(
                        epoch=int(epoch),
                        batch_idx=int(i + 1),
                        batch_total=int(total_batches),
                        train_loss=float(raw_loss.item()) if torch.isfinite(raw_loss) else None,
                        train_cls_loss=float(cls_loss.item()) if torch.isfinite(cls_loss) else None,
                        train_edge_loss=float(edge_loss.item()) if torch.isfinite(edge_loss) else None,
                        train_l2_att_loss=float(l2_att_loss.item()) if torch.isfinite(l2_att_loss) else None,
                        train_simi_loss=float(simi_loss.item()) if torch.isfinite(simi_loss) else None,
                        lr=lr_value,
                    )
                except Exception:
                    pass

    n_batches = max(1, total_batches)
    avg_loss = total_loss / n_batches
    avg_cls_loss = total_cls_loss / n_batches
    avg_edge_loss = total_edge_loss / n_batches
    avg_l2_att_loss = total_l2_att_loss / n_batches
    avg_h_loss = total_h_loss / n_batches
    avg_simi_loss = total_simi_loss / n_batches

    if writer is not None:
        writer.add_scalar('Loss/Train_Total', avg_loss, epoch)
        writer.add_scalar('Loss/Train_CLS', avg_cls_loss, epoch)
        writer.add_scalar('Loss/Train_H', avg_h_loss, epoch)
        writer.add_scalar('Loss/Train_Similarity', avg_simi_loss, epoch)
        writer.add_scalar('Loss/Train_Edge', avg_edge_loss, epoch)
        writer.add_scalar('Loss/Train_L2_Att', avg_l2_att_loss, epoch)
        if scheduler is not None:
            writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)

    return avg_loss, avg_cls_loss, avg_edge_loss, avg_l2_att_loss

# ================================
# Preentrenamiento de un generador de aristas
# ================================
def pretrain_edge_generator(
    encoder: torch.nn.Module,
    edge_gen: torch.nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    pretrain_epochs: int = 10,
    device: Optional[torch.device] = None,
    writer=None,
    report_path: Optional[str] = None,
) -> None:
    """
    Rutina ligera de preentrenamiento:
    - Congela encoder en modo eval para obtener embeddings (si se desea).
    - Ajusta edge_gen para aproximar adyacencias existentes (por tipo).
    Nota: Esta función es conservadora y no asume detalles de edge_gen.
    """
    if device is not None:
        data = data.to(device)
        encoder = encoder.to(device)
        edge_gen = edge_gen.to(device)

    encoder.eval()
    with torch.no_grad():
        if hasattr(encoder, "get_embeddings"):
            z_dict = encoder.get_embeddings(data)
        elif compute_epoch_embeddings is not None:
            z_dict = compute_epoch_embeddings(encoder, data)
        else:
            logger.warning("No compute_epoch_embeddings available; skipping pretrain_edge_generator.")
            return

    # --- Baseline AUC (debe rondar 0.5 con S inicializado pequeño) ---
    try:
        baseline = evaluate_edge_generator_auc(edge_gen, z_dict, data.edge_index_dict)
        macro = baseline.get('_macro')
        if macro is not None:
            logger.info(
                f"[EdgeGen eval] baseline (epoch 0): macro AUC={macro['auc']:.3f} "
                f"AP={macro['ap']:.3f} sobre {macro['n_relations']} relación(es)"
            )
            if writer is not None:
                writer.add_scalar('EdgeGen/AUC_macro', macro['auc'], 0)
                writer.add_scalar('EdgeGen/AP_macro', macro['ap'], 0)
    except Exception as exc:
        logger.warning(f"[EdgeGen eval] baseline AUC falló: {exc}")

    edge_gen.train()

    for epoch in range(1, pretrain_epochs + 1):
        total_loss = 0.0
        for et in data.edge_types:
            store = data[et]
            src_t, _, dst_t = et
            z_src = z_dict[src_t]
            z_dst = z_dict[dst_t]
            ei = store.edge_index  # [2, E]

            # Muestreo negativo simple: permuta destinos
            E = ei.size(1)
            if E == 0:
                continue
            neg_dst = ei[1][torch.randperm(E, device=ei.device)]
            # Positivos
            s_pos = z_src[ei[0]]
            d_pos = z_dst[ei[1]]
            # Negativos
            s_neg = z_src[ei[0]]
            d_neg = z_dst[neg_dst]

            # Puntajes (dot product como baseline si edge_gen no define forward específico)
            if hasattr(edge_gen, "score"):
                key = f"{et[0]}:{et[1]}:{et[2]}"
                if key not in edge_gen.S:
                    continue
                S = edge_gen.S[key]
                pos_logits = torch.sum((s_pos @ S) * d_pos, dim=1)
                neg_logits = torch.sum((s_neg @ S) * d_neg, dim=1)
            else:
                pos_logits = (s_pos * d_pos).sum(dim=-1)
                neg_logits = (s_neg * d_neg).sum(dim=-1)

            y = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
            logits = torch.cat([pos_logits, neg_logits], dim=0)

            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(edge_gen.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())

        logger.info(f"[EdgeGen pretrain] Epoch {epoch:03d} | Loss {total_loss:.4f}")
        if writer is not None:
            writer.add_scalar('Loss/pretrain_edge_gen', total_loss, epoch)

    # --- AUC final por relación + macro ---
    try:
        edge_gen.eval()
        report = evaluate_edge_generator_auc(edge_gen, z_dict, data.edge_index_dict)
        for key, m in report.items():
            if key == '_macro':
                continue
            logger.info(
                f"[EdgeGen eval] {key}: AUC={m['auc']:.3f} AP={m['ap']:.3f} "
                f"(n_pos={m['n_pos']}, n_neg={m['n_neg']})"
            )
        macro = report.get('_macro')
        if macro is not None:
            logger.info(
                f"[EdgeGen eval] post-pretrain: macro AUC={macro['auc']:.3f} "
                f"AP={macro['ap']:.3f}"
            )
            if writer is not None:
                writer.add_scalar('EdgeGen/AUC_macro', macro['auc'], int(pretrain_epochs))
                writer.add_scalar('EdgeGen/AP_macro', macro['ap'], int(pretrain_epochs))
            if macro['auc'] < 0.6:
                logger.warning(
                    f"[EdgeGen eval] AUC macro={macro['auc']:.3f} < 0.6; el generador "
                    "no aprendió señal de aristas — los sintéticos quedarán mal "
                    "conectados. Considera subir PRETRAIN_EDGE_EPOCHS o lambda_edge."
                )

        # Persistir reporte para que la UI/Streamlit lo renderice.
        if report_path:
            try:
                import json as _json
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                payload = {
                    "pretrain_epochs": int(pretrain_epochs),
                    "per_relation": {
                        k: v for k, v in report.items() if k != "_macro"
                    },
                    "macro": report.get("_macro"),
                }
                with open(report_path, "w") as f:
                    _json.dump(payload, f, indent=2)
                logger.info(f"[EdgeGen eval] reporte guardado en {report_path}")
            except Exception as exc:
                logger.warning(f"[EdgeGen eval] no se pudo guardar el reporte: {exc}")
    except Exception as exc:
        logger.warning(f"[EdgeGen eval] AUC final falló: {exc}")
