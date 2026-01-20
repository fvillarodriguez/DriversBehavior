import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from src.graphsmote import compute_epoch_embeddings
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

# ================================
# Entrenamiento por mini-batch
# ================================
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

    if DEBUG:
        logger.info(f"[train_minibatch] Epoch {epoch}: Iniciando entrenamiento de minibatch.")

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} (train)", leave=False)
    # Acumulador de gradientes
    accumulation_steps = 2
    
    optimizer.zero_grad()
    if DEBUG:
        logger.info(f"[train_minibatch] Epoch {epoch}: Gradientes iniciales puestos a cero.")

    for i, batch in enumerate(progress_bar):
        if DEBUG:
            logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Iniciando procesamiento de batch.")
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

            total_loss = cls_loss + lambda_edge * edge_loss + lambda_l2_att * l2_att_loss
            if DEBUG:
                grad_fn_info = total_loss.grad_fn.__class__.__name__ if total_loss.grad_fn else 'None'
                logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Loss calculada. grad_fn: {grad_fn_info}")
            return total_loss, cls_loss.detach(), edge_loss.detach(), l2_att_loss.detach()

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, cls_loss, edge_loss, l2_att_loss = compute_loss()
                loss = loss / accumulation_steps # Normalizar la loss
            
            if torch.isfinite(loss):
                if DEBUG:
                    grad_fn_info = loss.grad_fn.__class__.__name__ if loss.grad_fn else 'None'
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Antes de backward (AMP). Loss grad_fn: {grad_fn_info}")
                scaler.scale(loss).backward()
                if DEBUG:
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Después de backward (AMP).")
            else:
                logger.warning("Loss is NaN, skipping backward pass.")

        else:
            loss, cls_loss, edge_loss, l2_att_loss = compute_loss()
            loss = loss / accumulation_steps # Normalizar la loss
            if torch.isfinite(loss):
                if DEBUG:
                    grad_fn_info = loss.grad_fn.__class__.__name__ if loss.grad_fn else 'None'
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Antes de backward. Loss grad_fn: {grad_fn_info}")
                loss.backward()
                if DEBUG:
                    logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Después de backward.")
            else:
                logger.warning("Loss is NaN, skipping backward pass.")

        # --- PASO DE OPTIMIZACIÓN Y LIMPIEZA ---
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if DEBUG:
                logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Realizando paso de optimización.")
            if use_amp and scaler is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                if edge_gen: torch.nn.utils.clip_grad_norm_(edge_gen.parameters(), grad_clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                if edge_gen: torch.nn.utils.clip_grad_norm_(edge_gen.parameters(), grad_clip_value)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()
            if DEBUG:
                logger.info(f"[train_minibatch] Epoch {epoch}, Batch {i}: Gradientes puestos a cero después del paso.")

        if torch.isfinite(loss):
            total_loss += float(loss.item())
            total_cls_loss += float(cls_loss.item())
            total_edge_loss += float(edge_loss.item())
            total_l2_att_loss += float(l2_att_loss.item())

        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}" if torch.isfinite(loss) else "nan",
            'CLS': f"{cls_loss.item():.4f}" if torch.isfinite(cls_loss) else "nan",
            'Edge': f"{edge_loss.item():.4f}" if torch.isfinite(edge_loss) else "nan",
            'L2_Att': f"{l2_att_loss.item():.4f}" if torch.isfinite(l2_att_loss) else "nan",
            'LR': f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else 'N/A'
        })

    n_batches = max(1, len(loader))
    avg_loss = total_loss / n_batches
    avg_cls_loss = total_cls_loss / n_batches
    avg_edge_loss = total_edge_loss / n_batches
    avg_l2_att_loss = total_l2_att_loss / n_batches

    if writer is not None:
        writer.add_scalar('Loss/Train_Total', avg_loss, epoch)
        writer.add_scalar('Loss/Train_CLS', avg_cls_loss, epoch)
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
    writer=None
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
