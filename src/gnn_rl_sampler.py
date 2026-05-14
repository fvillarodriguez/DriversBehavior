"""RioGNN/CARE-GNN inspired top-p sampler for heterogeneous GNN training.

The implementation lives strictly in the sampling plane: relation/layer
thresholds choose the neighborhood delivered to the existing model. Learned
relation weights are persisted for auditability, but are not injected into the
message-passing layers here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData


EdgeType = Tuple[str, str, str]


def _edge_type_to_key(edge_type: EdgeType) -> str:
    return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"


def _edge_type_from_key(key: str) -> EdgeType:
    parts = str(key).split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid edge type key: {key!r}")
    return parts[0], parts[1], parts[2]


def _normalize_edge_type(edge_type: object) -> EdgeType:
    if isinstance(edge_type, tuple) and len(edge_type) == 3:
        return str(edge_type[0]), str(edge_type[1]), str(edge_type[2])
    if isinstance(edge_type, str) and "__" in edge_type:
        return _edge_type_from_key(edge_type)
    raise ValueError(f"Invalid edge type: {edge_type!r}")


def _clamp_float(value: object, low: float, high: float, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    return float(max(float(low), min(float(high), out)))


def _num_layers_from_neighbors(num_neighbors_cfg: Optional[object], fallback: int) -> int:
    if isinstance(num_neighbors_cfg, Mapping):
        lengths = []
        for value in num_neighbors_cfg.values():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                lengths.append(len(value))
        if lengths:
            return max(1, int(max(lengths)))
    if isinstance(num_neighbors_cfg, Sequence) and not isinstance(num_neighbors_cfg, (str, bytes)):
        return max(1, int(len(num_neighbors_cfg)))
    return max(1, int(fallback))


def _relation_degree(edge_index: Optional[Tensor], num_nodes: int) -> int:
    if edge_index is None or edge_index.numel() == 0:
        return 1
    deg = torch.zeros(int(num_nodes), dtype=torch.long)
    src = edge_index[0].detach().cpu().long()
    dst = edge_index[1].detach().cpu().long()
    one_src = torch.ones_like(src)
    one_dst = torch.ones_like(dst)
    deg.scatter_add_(0, src.clamp(0, max(int(num_nodes) - 1, 0)), one_src)
    deg.scatter_add_(0, dst.clamp(0, max(int(num_nodes) - 1, 0)), one_dst)
    return max(1, int(deg.max().item()))


def relation_max_degrees(graph: HeteroData, node_type: str = "pm") -> Dict[EdgeType, int]:
    """Return max incident degree for every same-node relation in ``graph``."""

    num_nodes = int(getattr(graph[node_type], "num_nodes", graph[node_type].x.size(0)))
    out: Dict[EdgeType, int] = {}
    for edge_type in graph.edge_types:
        if edge_type[0] != node_type or edge_type[2] != node_type:
            continue
        edge_index = getattr(graph[edge_type], "edge_index", None)
        out[_normalize_edge_type(edge_type)] = _relation_degree(edge_index, num_nodes)
    return out


def _resolve_relation_cap(
    num_neighbors_cfg: Optional[object],
    edge_type: EdgeType,
    layer_idx: int,
) -> Optional[int]:
    value: object = num_neighbors_cfg
    if isinstance(num_neighbors_cfg, Mapping):
        candidates = (
            edge_type,
            _edge_type_to_key(edge_type),
            edge_type[1],
            str(edge_type),
        )
        value = None
        for key in candidates:
            if key in num_neighbors_cfg:
                value = num_neighbors_cfg[key]
                break
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return None
        raw = value[min(int(layer_idx), len(value) - 1)]
    else:
        raw = value
    try:
        cap = int(raw)
    except Exception:
        return None
    if cap < 0:
        return None
    return max(0, cap)


def _unique_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for value in values:
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


class LabelAwareSimilarityScorer(nn.Module):
    """Small label predictor used to rank neighbors by label-aware similarity."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.1,
        out_channels: int = 2,
    ) -> None:
        super().__init__()
        hidden = max(4, int(hidden_channels))
        self.out_channels = max(2, int(out_channels))
        self.net = nn.Sequential(
            nn.Linear(int(in_channels), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.out_channels),
        )

    def forward_logits(self, x: Tensor) -> Tensor:
        return self.net(x)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.forward_logits(x)
        if logits.size(-1) >= 2:
            return torch.softmax(logits, dim=-1)[..., 1]
        return torch.sigmoid(logits.view(-1))

    def positive_probability(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def supervised_loss(
        self,
        x: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if mask is not None:
            mask = mask.bool()
            x = x[mask]
            y = y[mask]
        if x.numel() == 0 or y.numel() == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        logits = self.forward_logits(x.to(next(self.parameters()).device))
        return F.cross_entropy(logits, y.to(logits.device).long())

    @torch.no_grad()
    def similarity_for_edges(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.float32)
        param_device = next(self.parameters()).device
        probs = self.positive_probability(x.to(param_device)).detach().cpu()
        edge_index_cpu = edge_index.detach().cpu().long()
        src_p = probs[edge_index_cpu[0]]
        dst_p = probs[edge_index_cpu[1]]
        return (1.0 - torch.abs(src_p - dst_p)).clamp(0.0, 1.0)


class _ContinuousActorCritic(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        prev_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            self.actor = nn.Sequential(
                nn.Linear(6, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            self.critic = nn.Sequential(
                nn.Linear(6, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
            )
        finally:
            torch.random.set_rng_state(prev_state)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        return self.actor(state), self.critic(state)


@dataclass
class _ThresholdRecord:
    threshold: float
    low: float
    high: float
    best_threshold: float
    best_reward: float
    last_action: Optional[str] = None
    same_action_count: int = 0
    depth: int = 0
    converged: bool = False


class RioGNNThresholdController(nn.Module):
    """RSRL threshold controller for relation/layer top-p filtering."""

    def __init__(
        self,
        *,
        edge_types: Sequence[EdgeType],
        num_layers: int,
        in_channels: int,
        max_degree_by_edge_type: Optional[Mapping[EdgeType, int]] = None,
        action_space: str = "discrete",
        initial_p: float = 0.5,
        min_p: float = 0.05,
        max_p: float = 1.0,
        min_keep: int = 1,
        alpha: int = 10,
        switch_patience: int = 3,
        backtracking: bool = True,
        positive_only: bool = True,
        similarity_hidden_channels: int = 32,
        similarity_dropout: float = 0.1,
        secondary_reward_weight: float = 0.25,
        seed: int = 0,
        continuous_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.edge_types = [_normalize_edge_type(edge_type) for edge_type in edge_types]
        self.num_layers = max(1, int(num_layers))
        self.action_space = str(action_space or "discrete").strip().lower()
        if self.action_space in {"continuous", "actor", "actor_critic"}:
            self.action_space = "continuous_actor"
        if self.action_space not in {"discrete", "continuous_actor"}:
            self.action_space = "discrete"
        self.initial_p = _clamp_float(initial_p, min_p, max_p, 0.5)
        self.min_p = _clamp_float(min_p, 0.0, 1.0, 0.05)
        self.max_p = _clamp_float(max_p, self.min_p, 1.0, 1.0)
        self.min_keep = max(0, int(min_keep))
        self.alpha = max(2, int(alpha))
        self.switch_patience = max(1, int(switch_patience))
        self.backtracking = bool(backtracking)
        self.positive_only = bool(positive_only)
        self.secondary_reward_weight = float(secondary_reward_weight)
        self.seed = int(seed)

        self.scorer = LabelAwareSimilarityScorer(
            in_channels=int(in_channels),
            hidden_channels=int(similarity_hidden_channels),
            dropout=float(similarity_dropout),
            out_channels=2,
        )
        self.actor_critic = _ContinuousActorCritic(seed=int(seed))
        self._continuous_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=float(continuous_lr),
        )

        max_degree_by_edge_type = dict(max_degree_by_edge_type or {})
        self.max_depth_by_edge_type: Dict[str, int] = {}
        self.records: Dict[str, _ThresholdRecord] = {}
        for edge_type in self.edge_types:
            et_key = _edge_type_to_key(edge_type)
            max_degree = int(max_degree_by_edge_type.get(edge_type, max_degree_by_edge_type.get(et_key, 1)))
            max_depth = max(1, int(math.ceil(math.log(max(max_degree, 2), self.alpha))))
            self.max_depth_by_edge_type[et_key] = max_depth
            for layer_idx in range(self.num_layers):
                key = self._key(layer_idx, edge_type)
                self.records[key] = _ThresholdRecord(
                    threshold=float(self.initial_p),
                    low=float(self.min_p),
                    high=float(self.max_p),
                    best_threshold=float(self.initial_p),
                    best_reward=float("-inf"),
                )

        self.prev_val_auprc: Optional[float] = None
        self.prev_similarity_stats: Dict[str, float] = {}
        self._epoch_similarity_sum: Dict[str, float] = defaultdict(float)
        self._epoch_similarity_count: Dict[str, int] = defaultdict(int)
        self.reward_history: List[Dict[str, object]] = []
        self.threshold_history: List[Dict[str, object]] = []
        self.invalid_update_count = 0

    def _key(self, layer_idx: int, edge_type: EdgeType) -> str:
        return f"L{int(layer_idx)}::{_edge_type_to_key(edge_type)}"

    def get_threshold(self, layer_idx: int, edge_type: EdgeType) -> float:
        key = self._key(layer_idx, _normalize_edge_type(edge_type))
        if key not in self.records:
            self.records[key] = _ThresholdRecord(
                threshold=float(self.initial_p),
                low=float(self.min_p),
                high=float(self.max_p),
                best_threshold=float(self.initial_p),
                best_reward=float("-inf"),
            )
        return float(self.records[key].threshold)

    def set_threshold(self, layer_idx: int, edge_type: EdgeType, value: float) -> None:
        key = self._key(layer_idx, _normalize_edge_type(edge_type))
        if key not in self.records:
            self.get_threshold(layer_idx, edge_type)
        self.records[key].threshold = _clamp_float(value, self.min_p, self.max_p, self.initial_p)

    def record_similarity(
        self,
        layer_idx: int,
        edge_type: EdgeType,
        value: Optional[float],
        count: int = 1,
    ) -> None:
        if value is None:
            return
        try:
            f_value = float(value)
        except Exception:
            return
        if not math.isfinite(f_value):
            return
        key = self._key(layer_idx, _normalize_edge_type(edge_type))
        c = max(1, int(count))
        self._epoch_similarity_sum[key] += f_value * c
        self._epoch_similarity_count[key] += c

    def _current_similarity_stats(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, total in self._epoch_similarity_sum.items():
            count = max(1, int(self._epoch_similarity_count.get(key, 0)))
            out[key] = float(total) / float(count)
        return out

    def reset_epoch_stats(self) -> None:
        self._epoch_similarity_sum.clear()
        self._epoch_similarity_count.clear()

    def similarity_loss(self, batch: HeteroData, node_type: str = "pm") -> Tensor:
        if node_type not in batch.node_types:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        store = batch[node_type]
        if not hasattr(store, "x") or not hasattr(store, "y"):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        batch_size = int(getattr(store, "batch_size", store.x.size(0)))
        mask = torch.zeros(store.x.size(0), dtype=torch.bool, device=store.x.device)
        mask[:batch_size] = True
        return self.scorer.supervised_loss(store.x, store.y, mask=mask)

    def _valid_auprc(self, val_auprc: object) -> Optional[float]:
        try:
            value = float(val_auprc)
        except Exception:
            return None
        if not math.isfinite(value):
            return None
        return value

    def _state_tensor(self, key: str, record: _ThresholdRecord, reward: float) -> Tensor:
        layer_txt, edge_txt = key.split("::", 1)
        try:
            layer_idx = int(layer_txt[1:])
        except Exception:
            layer_idx = 0
        edge_hash_raw = sum((idx + 1) * ord(ch) for idx, ch in enumerate(edge_txt))
        edge_hash = float(edge_hash_raw % 1000) / 999.0
        state = torch.tensor(
            [
                float(record.threshold),
                float(record.low),
                float(record.high),
                float(reward),
                float(layer_idx) / float(max(1, self.num_layers - 1)),
                float(edge_hash),
            ],
            dtype=torch.float32,
            device=next(self.actor_critic.parameters()).device,
        )
        return state.view(1, -1)

    def _discrete_update_record(self, key: str, record: _ThresholdRecord, reward: float) -> Dict[str, object]:
        if record.converged:
            return {"action": "converged", "new_threshold": record.threshold}
        if reward > record.best_reward:
            record.best_reward = float(reward)
            record.best_threshold = float(record.threshold)
        elif reward < 0.0 and self.backtracking:
            record.threshold = float(record.best_threshold)

        width = max(record.high - record.low, 1e-8)
        step = width / float(max(1, self.alpha - 1))
        if reward >= 0.0:
            next_threshold = min(record.high, record.threshold + step)
            action = "increase"
        else:
            next_threshold = max(record.low, record.threshold - step)
            action = "decrease"
        if abs(next_threshold - record.threshold) <= 1e-12:
            action = "hold"

        if action == record.last_action:
            record.same_action_count += 1
        else:
            record.same_action_count = 1
        record.last_action = action
        record.threshold = float(max(self.min_p, min(self.max_p, next_threshold)))

        if record.same_action_count >= self.switch_patience:
            center = float(record.best_threshold if math.isfinite(record.best_reward) else record.threshold)
            half = max(width / float(self.alpha), 1e-6) / 2.0
            record.low = float(max(self.min_p, center - half))
            record.high = float(min(self.max_p, center + half))
            if record.low >= record.high:
                record.low = max(self.min_p, center - 1e-6)
                record.high = min(self.max_p, center + 1e-6)
            record.threshold = float(max(record.low, min(record.high, record.threshold)))
            record.depth += 1
            record.same_action_count = 0

        edge_key = key.split("::", 1)[1]
        if record.depth >= int(self.max_depth_by_edge_type.get(edge_key, 1)):
            record.converged = True

        return {
            "action": action,
            "new_threshold": record.threshold,
            "low": record.low,
            "high": record.high,
            "depth": record.depth,
            "converged": record.converged,
        }

    def _continuous_update_record(self, key: str, record: _ThresholdRecord, reward: float) -> Dict[str, object]:
        state = self._state_tensor(key, record, reward)
        actor_raw, value = self.actor_critic(state)
        bounded = self.min_p + (self.max_p - self.min_p) * actor_raw.view(())
        reward_tensor = torch.tensor(float(reward), dtype=torch.float32, device=value.device)
        advantage = reward_tensor - value.view(())
        actor_loss = -advantage.detach() * torch.log(torch.clamp(actor_raw.view(()), 1e-6, 1.0 - 1e-6))
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss
        self._continuous_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._continuous_optimizer.step()

        if reward > record.best_reward:
            record.best_reward = float(reward)
            record.best_threshold = float(record.threshold)
        next_threshold = float(bounded.detach().cpu().item())
        if reward < 0.0 and self.backtracking:
            next_threshold = 0.5 * next_threshold + 0.5 * float(record.best_threshold)
        record.threshold = _clamp_float(next_threshold, self.min_p, self.max_p, self.initial_p)
        record.depth += 1
        if record.depth >= self.switch_patience:
            record.converged = True
        return {
            "action": "actor",
            "new_threshold": record.threshold,
            "critic_value": float(value.detach().cpu().item()),
            "loss": float(loss.detach().cpu().item()),
            "converged": record.converged,
        }

    def update_after_validation(self, val_auprc: object, epoch: Optional[int] = None) -> Dict[str, object]:
        value = self._valid_auprc(val_auprc)
        current_sim = self._current_similarity_stats()
        if value is None:
            self.invalid_update_count += 1
            payload = {
                "epoch": epoch,
                "valid": False,
                "reason": "invalid_auprc",
                "thresholds": self.thresholds_serializable(),
            }
            self.reward_history.append(payload)
            self.reset_epoch_stats()
            return payload
        if self.prev_val_auprc is None:
            self.prev_val_auprc = float(value)
            self.prev_similarity_stats = current_sim
            payload = {
                "epoch": epoch,
                "valid": True,
                "warmup": True,
                "val_auprc": float(value),
                "thresholds": self.thresholds_serializable(),
            }
            self.reward_history.append(payload)
            self.threshold_history.append(payload)
            self.reset_epoch_stats()
            return payload

        denom = max(abs(float(self.prev_val_auprc)), 1e-6)
        primary_delta = (float(value) - float(self.prev_val_auprc)) / denom
        relation_updates: Dict[str, Dict[str, object]] = {}
        for key, record in self.records.items():
            sim_delta = float(current_sim.get(key, self.prev_similarity_stats.get(key, 0.0))) - float(
                self.prev_similarity_stats.get(key, current_sim.get(key, 0.0))
            )
            reward = float(primary_delta) + float(self.secondary_reward_weight) * float(sim_delta)
            if self.action_space == "continuous_actor":
                update = self._continuous_update_record(key, record, reward)
            else:
                update = self._discrete_update_record(key, record, reward)
            update.update(
                {
                    "reward": reward,
                    "primary_delta": float(primary_delta),
                    "similarity_delta": float(sim_delta),
                }
            )
            relation_updates[key] = update

        self.prev_val_auprc = float(value)
        self.prev_similarity_stats = current_sim
        payload = {
            "epoch": epoch,
            "valid": True,
            "warmup": False,
            "val_auprc": float(value),
            "primary_delta": float(primary_delta),
            "updates": relation_updates,
            "thresholds": self.thresholds_serializable(),
        }
        self.reward_history.append(payload)
        self.threshold_history.append({"epoch": epoch, "thresholds": self.thresholds_serializable()})
        self.reset_epoch_stats()
        return payload

    def thresholds_serializable(self) -> Dict[str, float]:
        return {key: float(record.threshold) for key, record in self.records.items()}

    def relation_weights_serializable(self) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        by_edge: Dict[str, List[float]] = defaultdict(list)
        for key, threshold in self.thresholds_serializable().items():
            edge_key = key.split("::", 1)[1]
            by_edge[edge_key].append(float(threshold))
        for edge_key, values in by_edge.items():
            weights[edge_key] = float(sum(values) / max(1, len(values)))
        return weights

    def state_dict_serializable(self) -> Dict[str, object]:
        return {
            "sampler_impl": "rl_top_p_rsrl",
            "action_space": self.action_space,
            "initial_p": float(self.initial_p),
            "min_p": float(self.min_p),
            "max_p": float(self.max_p),
            "min_keep": int(self.min_keep),
            "alpha": int(self.alpha),
            "switch_patience": int(self.switch_patience),
            "backtracking": bool(self.backtracking),
            "positive_only": bool(self.positive_only),
            "secondary_reward_weight": float(self.secondary_reward_weight),
            "num_layers": int(self.num_layers),
            "edge_types": [_edge_type_to_key(edge_type) for edge_type in self.edge_types],
            "thresholds": self.thresholds_serializable(),
            "relation_weights_audit": self.relation_weights_serializable(),
            "records": {
                key: {
                    "threshold": float(record.threshold),
                    "low": float(record.low),
                    "high": float(record.high),
                    "best_threshold": float(record.best_threshold),
                    "best_reward": float(record.best_reward),
                    "last_action": record.last_action,
                    "same_action_count": int(record.same_action_count),
                    "depth": int(record.depth),
                    "converged": bool(record.converged),
                }
                for key, record in self.records.items()
            },
            "reward_history": list(self.reward_history),
            "threshold_history": list(self.threshold_history),
            "prev_val_auprc": self.prev_val_auprc,
            "prev_similarity_stats": dict(self.prev_similarity_stats),
            "invalid_update_count": int(self.invalid_update_count),
        }

    def load_state_dict_serializable(self, payload: Optional[Mapping[str, object]]) -> None:
        if not payload:
            return
        records = payload.get("records") if isinstance(payload, Mapping) else None
        if isinstance(records, Mapping):
            for key, rec in records.items():
                if not isinstance(rec, Mapping):
                    continue
                record = self.records.get(str(key))
                if record is None:
                    continue
                record.threshold = _clamp_float(rec.get("threshold"), self.min_p, self.max_p, self.initial_p)
                record.low = _clamp_float(rec.get("low"), self.min_p, self.max_p, self.min_p)
                record.high = _clamp_float(rec.get("high"), self.min_p, self.max_p, self.max_p)
                record.best_threshold = _clamp_float(rec.get("best_threshold"), self.min_p, self.max_p, self.initial_p)
                try:
                    record.best_reward = float(rec.get("best_reward", float("-inf")))
                except Exception:
                    record.best_reward = float("-inf")
                record.last_action = rec.get("last_action") if rec.get("last_action") is not None else None
                record.same_action_count = int(rec.get("same_action_count", 0))
                record.depth = int(rec.get("depth", 0))
                record.converged = bool(rec.get("converged", False))
        else:
            thresholds = payload.get("thresholds") if isinstance(payload, Mapping) else None
            if isinstance(thresholds, Mapping):
                for key, value in thresholds.items():
                    if str(key) in self.records:
                        self.records[str(key)].threshold = _clamp_float(value, self.min_p, self.max_p, self.initial_p)
        try:
            self.reward_history = list(payload.get("reward_history", []))
            self.threshold_history = list(payload.get("threshold_history", []))
            self.prev_val_auprc = payload.get("prev_val_auprc")
            if self.prev_val_auprc is not None:
                self.prev_val_auprc = float(self.prev_val_auprc)
            self.prev_similarity_stats = dict(payload.get("prev_similarity_stats", {}) or {})
            self.invalid_update_count = int(payload.get("invalid_update_count", 0))
        except Exception:
            pass


def pretrain_label_aware_similarity(
    controller: RioGNNThresholdController,
    graph: HeteroData,
    *,
    node_type: str = "pm",
    device: Optional[torch.device] = None,
    epochs: int = 3,
    lr: float = 1e-3,
    positive_only: Optional[bool] = None,
) -> List[float]:
    """Pretrain the label-aware scorer on supervised train nodes."""

    epochs = max(0, int(epochs))
    if epochs <= 0 or node_type not in graph.node_types:
        return []
    store = graph[node_type]
    if not hasattr(store, "x") or not hasattr(store, "y") or not hasattr(store, "train_mask"):
        return []
    device = device or next(controller.parameters()).device
    controller.to(device)
    x = store.x.to(device)
    y = store.y.to(device).long()
    train_mask = store.train_mask.to(device).bool()
    if bool(controller.positive_only if positive_only is None else positive_only):
        pos_mask = train_mask & (y == 1)
        if bool(pos_mask.any()):
            train_mask = pos_mask
    if not bool(train_mask.any()):
        return []
    optimizer = torch.optim.Adam(controller.scorer.parameters(), lr=float(lr))
    losses: List[float] = []
    for _ in range(epochs):
        controller.scorer.train()
        optimizer.zero_grad(set_to_none=True)
        loss = controller.scorer.supervised_loss(x, y, mask=train_mask)
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    return losses


class RLTopPHeteroLoader:
    """Iterable loader that expands relation-wise top-p neighborhoods."""

    sampler_impl = "rl_top_p_rsrl"

    def __init__(
        self,
        graph_cpu: HeteroData,
        *,
        controller: RioGNNThresholdController,
        input_nodes: Tensor,
        batch_size: int,
        num_neighbors_cfg: Optional[object] = None,
        node_type: str = "pm",
        shuffle: bool = True,
        deterministic: bool = False,
        sampling_seed: int = 0,
    ) -> None:
        self.graph_cpu = graph_cpu.cpu()
        self.controller = controller
        self.input_nodes = input_nodes.detach().cpu().long().view(-1)
        self.batch_size = max(1, int(batch_size))
        self.num_neighbors_cfg = num_neighbors_cfg
        self.node_type = str(node_type)
        self.shuffle = bool(shuffle)
        self.deterministic = bool(deterministic)
        self.sampling_seed = int(sampling_seed)
        self.num_layers = _num_layers_from_neighbors(num_neighbors_cfg, controller.num_layers)
        self.sampler_impl = "rl_top_p_rsrl"

    def __len__(self) -> int:
        if self.input_nodes.numel() == 0:
            return 0
        return int(math.ceil(float(self.input_nodes.numel()) / float(self.batch_size)))

    def __iter__(self) -> Iterator[HeteroData]:
        if self.input_nodes.numel() == 0:
            return
        nodes = self.input_nodes
        if self.shuffle:
            gen = torch.Generator(device="cpu")
            if self.deterministic:
                gen.manual_seed(int(self.sampling_seed))
            else:
                gen.seed()
            nodes = nodes[torch.randperm(nodes.numel(), generator=gen)]
        for start in range(0, int(nodes.numel()), self.batch_size):
            seeds = nodes[start : start + self.batch_size]
            batch, err = self.build_batch(seeds)
            if batch is None:
                continue
            yield batch

    def build_batch(self, seeds: Tensor) -> Tuple[Optional[HeteroData], Optional[str]]:
        return build_rl_top_p_batch(
            self.graph_cpu,
            seeds,
            controller=self.controller,
            num_neighbors_cfg=self.num_neighbors_cfg,
            num_layers=self.num_layers,
            node_type=self.node_type,
        )


def build_rl_top_p_batch(
    graph_cpu: HeteroData,
    seeds: Tensor,
    *,
    controller: RioGNNThresholdController,
    num_neighbors_cfg: Optional[object] = None,
    num_layers: Optional[int] = None,
    node_type: str = "pm",
) -> Tuple[Optional[HeteroData], Optional[str]]:
    if node_type not in graph_cpu.node_types:
        return None, f"Missing node type {node_type!r}."
    store = graph_cpu[node_type]
    total_nodes = int(getattr(store, "num_nodes", store.x.size(0)))
    seed_nodes = [
        idx
        for idx in _unique_preserve_order(seeds.detach().cpu().long().view(-1).tolist())
        if 0 <= idx < total_nodes
    ]
    if not seed_nodes:
        return None, "No valid supervised seeds."

    train_mask = getattr(store, "train_mask", None)
    if torch.is_tensor(train_mask):
        seed_nodes = [idx for idx in seed_nodes if bool(train_mask.detach().cpu()[idx].item())]
        if not seed_nodes:
            return None, "No train seeds in batch."

    selected_nodes = list(seed_nodes)
    selected_set = set(selected_nodes)
    frontier = list(seed_nodes)
    selected_edges: Dict[EdgeType, List[int]] = defaultdict(list)
    selected_edge_sets: Dict[EdgeType, set] = defaultdict(set)
    num_layers = max(1, int(num_layers if num_layers is not None else controller.num_layers))

    pm_edge_types = [
        _normalize_edge_type(edge_type)
        for edge_type in graph_cpu.edge_types
        if edge_type[0] == node_type and edge_type[2] == node_type
    ]
    if not pm_edge_types:
        return _materialize_rl_batch(
            graph_cpu,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            batch_size=len(seed_nodes),
            node_type=node_type,
        )

    x_all = store.x.detach().cpu()
    stat_centers = set(seed_nodes)
    if bool(controller.positive_only) and hasattr(store, "y") and torch.is_tensor(store.y):
        y_cpu = store.y.detach().cpu().long()
        positive_seed_nodes = [idx for idx in seed_nodes if int(y_cpu[idx].item()) == 1]
        if positive_seed_nodes:
            stat_centers = set(positive_seed_nodes)
    for layer_idx in range(num_layers):
        if not frontier:
            break
        next_frontier_nodes: List[int] = []
        frontier_tensor = torch.as_tensor(frontier, dtype=torch.long)
        for edge_type in pm_edge_types:
            edge_index = getattr(graph_cpu[edge_type], "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                continue
            edge_index_cpu = edge_index.detach().cpu().long()
            src = edge_index_cpu[0]
            dst = edge_index_cpu[1]
            sim = controller.scorer.similarity_for_edges(x_all, edge_index_cpu)
            rel_cap = _resolve_relation_cap(num_neighbors_cfg, edge_type, layer_idx)
            threshold = controller.get_threshold(layer_idx, edge_type)
            kept_for_relation: List[int] = []
            kept_sim: List[float] = []
            for center in frontier_tensor.tolist():
                center = int(center)
                incident = torch.nonzero((src == center) | (dst == center), as_tuple=False).view(-1)
                if incident.numel() == 0:
                    continue
                incident_sim = sim[incident]
                order = torch.argsort(incident_sim, descending=True)
                ordered_idx = incident[order]
                keep_count = int(math.ceil(float(threshold) * float(ordered_idx.numel())))
                keep_count = max(int(controller.min_keep), keep_count)
                if rel_cap is not None:
                    keep_count = min(int(rel_cap), keep_count)
                keep_count = min(int(ordered_idx.numel()), max(0, keep_count))
                if keep_count <= 0:
                    continue
                chosen = ordered_idx[:keep_count].tolist()
                for edge_pos in chosen:
                    edge_pos = int(edge_pos)
                    if edge_pos in selected_edge_sets[edge_type]:
                        continue
                    selected_edge_sets[edge_type].add(edge_pos)
                    kept_for_relation.append(edge_pos)
                    selected_edges[edge_type].append(edge_pos)
                    s_idx = int(src[edge_pos].item())
                    d_idx = int(dst[edge_pos].item())
                    if s_idx not in selected_set:
                        selected_set.add(s_idx)
                        selected_nodes.append(s_idx)
                        next_frontier_nodes.append(s_idx)
                    if d_idx not in selected_set:
                        selected_set.add(d_idx)
                        selected_nodes.append(d_idx)
                        next_frontier_nodes.append(d_idx)
                    if center in stat_centers:
                        try:
                            kept_sim.append(float(sim[edge_pos].item()))
                        except Exception:
                            pass
            if kept_sim:
                controller.record_similarity(
                    layer_idx,
                    edge_type,
                    float(sum(kept_sim) / len(kept_sim)),
                    count=len(kept_sim),
                )
        frontier = _unique_preserve_order(next_frontier_nodes)

    return _materialize_rl_batch(
        graph_cpu,
        selected_nodes=selected_nodes,
        selected_edges=selected_edges,
        batch_size=len(seed_nodes),
        node_type=node_type,
    )


def _materialize_rl_batch(
    graph_cpu: HeteroData,
    *,
    selected_nodes: Sequence[int],
    selected_edges: Mapping[EdgeType, Sequence[int]],
    batch_size: int,
    node_type: str = "pm",
) -> Tuple[Optional[HeteroData], Optional[str]]:
    store = graph_cpu[node_type]
    total_nodes = int(getattr(store, "num_nodes", store.x.size(0)))
    pm_nodes = torch.as_tensor(list(selected_nodes), dtype=torch.long)
    if pm_nodes.numel() == 0:
        return None, "Empty node set."
    out = HeteroData()
    out[node_type].x = store.x[pm_nodes].cpu()
    if hasattr(store, "y") and store.y is not None:
        out[node_type].y = store.y[pm_nodes].cpu()
    copied_node_attrs = {"x", "y"}
    try:
        node_items = list(store.items())
    except Exception:
        node_items = []
    for attr_name, attr_value in node_items:
        if attr_name in copied_node_attrs:
            continue
        if torch.is_tensor(attr_value) and attr_value.size(0) >= total_nodes:
            out[node_type][attr_name] = attr_value[pm_nodes].cpu()

    out[node_type].num_nodes = int(pm_nodes.numel())
    out[node_type].n_id = pm_nodes.clone()
    out[node_type].batch_size = int(batch_size)
    supervised_mask = torch.zeros(int(pm_nodes.numel()), dtype=torch.bool)
    supervised_mask[: int(batch_size)] = True
    out[node_type].supervised_mask = supervised_mask

    global_to_local = torch.full((total_nodes,), -1, dtype=torch.long)
    global_to_local[pm_nodes] = torch.arange(pm_nodes.numel(), dtype=torch.long)
    for edge_type in graph_cpu.edge_types:
        if edge_type[0] != node_type or edge_type[2] != node_type:
            continue
        edge_type = _normalize_edge_type(edge_type)
        edge_index = getattr(graph_cpu[edge_type], "edge_index", None)
        if edge_index is None:
            out[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            continue
        edge_index_cpu = edge_index.detach().cpu().long()
        raw_idx = torch.as_tensor(list(selected_edges.get(edge_type, [])), dtype=torch.long)
        if raw_idx.numel() == 0:
            out[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = getattr(graph_cpu[edge_type], "edge_attr", None)
            if torch.is_tensor(edge_attr) and edge_attr.dim() >= 2:
                out[edge_type].edge_attr = torch.zeros((0, int(edge_attr.size(1))), dtype=edge_attr.dtype)
            continue
        raw_idx = raw_idx[(raw_idx >= 0) & (raw_idx < edge_index_cpu.size(1))]
        src_local = global_to_local[edge_index_cpu[0, raw_idx]]
        dst_local = global_to_local[edge_index_cpu[1, raw_idx]]
        keep = (src_local >= 0) & (dst_local >= 0)
        raw_idx = raw_idx[keep]
        local_edge_index = torch.stack([src_local[keep], dst_local[keep]], dim=0)
        out[edge_type].edge_index = local_edge_index
        edge_attr = getattr(graph_cpu[edge_type], "edge_attr", None)
        if torch.is_tensor(edge_attr):
            edge_attr_cpu = edge_attr.cpu()
            if edge_attr_cpu.size(0) == edge_index_cpu.size(1):
                out[edge_type].edge_attr = edge_attr_cpu[raw_idx]
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
