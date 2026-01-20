from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TemporalAggregator(nn.Module):
    """
    Temporal head that consumes per-node embeddings gathered across a fixed-length
    sequence of snapshots (length L) and produces classification logits.

    During mini-batch training, it maintains a cache of the most recent
    embeddings for nodes that are not sampled together with the target node.
    """

    def __init__(
        self,
        sequence_rows: torch.Tensor,
        target_rows: torch.Tensor,
        num_nodes: int,
        embedding_dim: int,
        sequence_length: int,
        hidden_dim: Optional[int] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        if sequence_rows.ndim != 2:
            raise ValueError("sequence_rows must have shape [num_sequences, sequence_length].")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim else self.embedding_dim
        self.sequence_length = int(sequence_length)
        if self.sequence_length != sequence_rows.shape[1]:
            raise ValueError("sequence_length inconsistent with sequence_rows second dimension.")

        max_index = int(max(num_nodes, int(sequence_rows.max().item()) + 1 if sequence_rows.numel() else num_nodes))

        self.register_buffer("sequence_rows", sequence_rows.long())
        self.register_buffer("target_rows", target_rows.long())

        # Lookup table: node_id -> index in sequence_rows (or -1 if no sequence)
        lookup = torch.full((max_index,), -1, dtype=torch.long)
        if target_rows.numel():
            lookup[target_rows.long()] = torch.arange(target_rows.numel(), dtype=torch.long)
        self.register_buffer("target_lookup", lookup)

        # Mask of nodes with valid sequence/label
        seq_mask = torch.zeros(max_index, dtype=torch.bool)
        if target_rows.numel():
            seq_mask[target_rows.long()] = True
        self.register_buffer("sequence_mask", seq_mask)

        self.temporal = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

        self.register_buffer(
            "embedding_cache",
            torch.zeros(max_index, self.embedding_dim, dtype=torch.float32),
        )

    def reset_cache(self) -> None:
        """Reset cached embeddings (useful before evaluation passes)."""
        self.embedding_cache.zero_()

    def _ensure_cache_capacity(self, required_size: int) -> None:
        current = int(self.embedding_cache.size(0))
        if required_size <= current:
            return
        device = self.embedding_cache.device
        pad = required_size - current
        self.embedding_cache = torch.cat(
            [self.embedding_cache, torch.zeros(pad, self.embedding_dim, device=device, dtype=self.embedding_cache.dtype)],
            dim=0,
        )
        self.target_lookup = torch.cat(
            [self.target_lookup, torch.full((pad,), -1, dtype=torch.long, device=device)],
            dim=0,
        )
        self.sequence_mask = torch.cat(
            [self.sequence_mask, torch.zeros(pad, dtype=torch.bool, device=device)],
            dim=0,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        global_node_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor [num_nodes_in_subgraph, embedding_dim] from GAT.
            global_node_ids: Tensor of global node indices aligned with embeddings.
            batch_size: Number of target nodes (first entries) in the loader batch.
        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        if batch_size == 0:
            return torch.zeros(0, self.classifier.out_features, device=embeddings.device, dtype=embeddings.dtype)

        device = embeddings.device
        global_ids_cpu = global_node_ids.detach().cpu().tolist()
        max_gid = int(max(global_ids_cpu)) if global_ids_cpu else 0
        self._ensure_cache_capacity(max_gid + 1)

        # Update cache with current embeddings (detach to avoid leaking gradients)
        global_ids_device = global_node_ids.to(self.embedding_cache.device)
        self.embedding_cache[global_ids_device] = embeddings.detach().to(self.embedding_cache.device)

        # Mapping global id -> local index within current batch
        local_index_map = {gid: idx for idx, gid in enumerate(global_ids_cpu)}

        sequence_tensors = []
        for idx in range(batch_size):
            gid = global_ids_cpu[idx]
            seq_idx = -1
            if gid < self.target_lookup.size(0):
                seq_idx = int(self.target_lookup[gid])

            if seq_idx < 0:
                # No sequence available: repeat latest embedding
                filler = embeddings[idx].unsqueeze(0).expand(self.sequence_length, -1)
                sequence_tensors.append(filler)
                continue

            seq_global = self.sequence_rows[seq_idx].detach().cpu().tolist()
            self._ensure_cache_capacity(max(seq_global) + 1)
            seq_embs = []
            for sg in seq_global:
                local = local_index_map.get(sg)
                if local is not None:
                    seq_embs.append(embeddings[local])
                else:
                    seq_embs.append(self.embedding_cache[sg].to(device))
            seq_tensor = torch.stack(seq_embs, dim=0)
            sequence_tensors.append(seq_tensor)

        sequence_batch = torch.stack(sequence_tensors, dim=0)
        output, _ = self.temporal(sequence_batch)
        last_hidden = output[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits

