from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.temporal_head import TemporalAggregator


def _resolve_transformer_nheads(embedding_dim: int, requested_heads: int) -> int:
    heads = max(1, int(requested_heads))
    if embedding_dim % heads == 0:
        return heads
    for cand in (8, 6, 4, 3, 2, 1):
        if cand <= heads and embedding_dim % cand == 0:
            return cand
    for cand in range(min(heads, embedding_dim), 0, -1):
        if embedding_dim % cand == 0:
            return cand
    return 1


class TemporalGRUHead(TemporalAggregator):
    """GRU temporal head over snapshot embeddings for each target node."""

    def __init__(
        self,
        sequence_rows: torch.Tensor,
        target_rows: torch.Tensor,
        num_nodes: int,
        embedding_dim: int,
        sequence_length: int,
        hidden_dim: Optional[int] = None,
        num_classes: int = 2,
        cache_strategy: str = "incremental",
    ):
        super().__init__(
            sequence_rows=sequence_rows,
            target_rows=target_rows,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            cache_strategy=cache_strategy,
        )
        self.temporal = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)


class TemporalTransformerHead(TemporalAggregator):
    """Transformer encoder temporal head with learnable positional embeddings."""

    def __init__(
        self,
        sequence_rows: torch.Tensor,
        target_rows: torch.Tensor,
        num_nodes: int,
        embedding_dim: int,
        sequence_length: int,
        hidden_dim: Optional[int] = None,
        num_classes: int = 2,
        cache_strategy: str = "incremental",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
        pooling: str = "last",
    ):
        super().__init__(
            sequence_rows=sequence_rows,
            target_rows=target_rows,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim or embedding_dim,
            num_classes=num_classes,
            cache_strategy=cache_strategy,
        )
        model_dim = self.embedding_dim
        tf_heads = _resolve_transformer_nheads(model_dim, num_heads)
        self.pooling = str(pooling or "last").strip().lower()
        self.pos = nn.Parameter(torch.randn(self.sequence_length, model_dim) * 0.02)
        self.input_norm = nn.LayerNorm(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=tf_heads,
            dim_feedforward=max(model_dim, int(model_dim * ff_mult)),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.pool_score = nn.Linear(model_dim, 1) if self.pooling == "attn" else None
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
        global_node_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if batch_size == 0:
            return torch.zeros(
                0,
                self.classifier.out_features,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        seq = self._build_sequence_batch(embeddings, global_node_ids, batch_size)
        x = self.input_norm(seq + self.pos[: seq.size(1)].unsqueeze(0).to(seq.dtype))
        x = self.encoder(x)
        if self.pool_score is not None:
            scores = self.pool_score(torch.tanh(x)).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(weights * x, dim=1)
        else:
            pooled = x[:, -1, :]
        return self.classifier(pooled)


class TemporalAttnPoolHead(TemporalAggregator):
    """Lightweight temporal attention-pooling head (ablation-friendly)."""

    def __init__(
        self,
        sequence_rows: torch.Tensor,
        target_rows: torch.Tensor,
        num_nodes: int,
        embedding_dim: int,
        sequence_length: int,
        hidden_dim: Optional[int] = None,
        num_classes: int = 2,
        cache_strategy: str = "incremental",
        dropout: float = 0.1,
    ):
        super().__init__(
            sequence_rows=sequence_rows,
            target_rows=target_rows,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            cache_strategy=cache_strategy,
        )
        self.temporal_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.temporal_score = nn.Linear(self.hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
        global_node_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if batch_size == 0:
            return torch.zeros(
                0,
                self.classifier.out_features,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        seq = self._build_sequence_batch(embeddings, global_node_ids, batch_size)
        h = torch.tanh(self.temporal_proj(seq))
        scores = self.temporal_score(h).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(weights * h, dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

