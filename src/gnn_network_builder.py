from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.nn import LayerNorm, Linear, ModuleDict, ModuleList
from torch_geometric.nn import HeteroConv

from src.gat_model import GATConvSaveAlpha, TransformerConvSaveAlpha


SCHEMA_VERSION = "1.0"

SUPPORTED_CONVS: Tuple[str, ...] = ("GATConv", "TransformerConv")
SUPPORTED_AGGREGATIONS: Tuple[str, ...] = ("sum", "mean", "max")
SUPPORTED_ACTIVATIONS: Tuple[str, ...] = (
    "relu",
    "leaky_relu",
    "elu",
    "gelu",
    "silu",
    "tanh",
)
SUPPORTED_TEMPORAL_HEADS: Tuple[str, ...] = (
    "snapshot",
    "gru",
    "transformer",
    "attnpool",
)
SUPPORTED_BLOCK_TYPES: Tuple[str, ...] = (
    "hetero_conv",
    "activation",
    "norm",
    "dropout",
    "temporal_head",
)
SUPPORTED_NORMS: Tuple[str, ...] = ("none", "layer_norm")

DEFAULT_EDGE_TYPES: Tuple[Tuple[str, str, str], ...] = (
    ("pm", "spatial", "pm"),
    ("pm", "temporal", "pm"),
    ("pm", "spatial_back", "pm"),
    ("pm", "st_fwd", "pm"),
)

ArchitectureLike = Union["NetworkArchitecture", Mapping[str, Any], str, Path]


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _slugify(value: str, fallback: str = "architecture") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_.-")
    return slug or fallback


def _normalize_conv_type(value: object) -> str:
    raw = str(value or "GATConv").strip().lower().replace("-", "_")
    aliases = {
        "gat": "GATConv",
        "gatconv": "GATConv",
        "gat_conv": "GATConv",
        "transformer": "TransformerConv",
        "transformerconv": "TransformerConv",
        "transformer_conv": "TransformerConv",
    }
    return aliases.get(raw, str(value or "GATConv").strip())


def _normalize_activation(value: object) -> str:
    raw = str(value or "relu").strip().lower().replace("-", "_")
    aliases = {
        "leakyrelu": "leaky_relu",
        "leaky_relu": "leaky_relu",
        "gelu": "gelu",
        "elu": "elu",
        "relu": "relu",
        "silu": "silu",
        "swish": "silu",
        "tanh": "tanh",
    }
    return aliases.get(raw, raw)


def _normalize_aggr(value: object) -> str:
    return str(value or "mean").strip().lower()


def _normalize_temporal(value: object) -> str:
    raw = str(value or "snapshot").strip().lower().replace("-", "_")
    aliases = {
        "none": "snapshot",
        "no": "snapshot",
        "attn_pool": "attnpool",
        "attention_pool": "attnpool",
    }
    return aliases.get(raw, raw)


def _safe_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y", "si", "s"}:
            return True
        if norm in {"0", "false", "no", "n"}:
            return False
    return bool(value)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(str(message))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def raise_for_errors(self) -> None:
        if not self.is_valid:
            raise ValueError("; ".join(self.errors))


@dataclass
class NetworkBlock:
    block_type: str = "hetero_conv"
    name: Optional[str] = None
    conv_type: Optional[str] = "GATConv"
    hidden_channels: Optional[int] = 64
    num_heads: Optional[int] = 4
    aggregation: Optional[str] = "mean"
    activation: Optional[str] = "relu"
    dropout: Optional[float] = 0.3
    residual: Optional[bool] = True
    norm: Optional[str] = "layer_norm"
    temporal_type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NetworkBlock":
        data = dict(payload or {})
        if "type" in data and "block_type" not in data:
            data["block_type"] = data.pop("type")
        if "aggr" in data and "aggregation" not in data:
            data["aggregation"] = data.pop("aggr")
        if "temporal_head" in data and "temporal_type" not in data:
            data["temporal_type"] = data.pop("temporal_head")
        allowed = set(cls.__dataclass_fields__.keys())
        params = dict(data.get("params") or {})
        extras = {k: v for k, v in data.items() if k not in allowed}
        params.update(extras)
        filtered = {k: v for k, v in data.items() if k in allowed}
        filtered["params"] = params
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        block_type = str(self.block_type or "").strip().lower()
        if block_type == "hetero_conv":
            keep = {
                "block_type",
                "name",
                "conv_type",
                "hidden_channels",
                "num_heads",
                "aggregation",
                "activation",
                "dropout",
                "residual",
                "norm",
                "params",
            }
        elif block_type == "activation":
            keep = {"block_type", "name", "activation", "params"}
        elif block_type == "norm":
            keep = {"block_type", "name", "norm", "params"}
        elif block_type == "dropout":
            keep = {"block_type", "name", "dropout", "params"}
        elif block_type == "temporal_head":
            keep = {"block_type", "name", "temporal_type", "params"}
        else:
            keep = set(asdict(self).keys())
        data = {k: v for k, v in asdict(self).items() if k in keep and v is not None}
        if not data.get("params"):
            data.pop("params", None)
        return data


@dataclass
class NetworkHead:
    name: str = "primary"
    head_type: str = "classifier"
    primary: bool = True
    hidden_channels: List[int] = field(default_factory=list)
    activation: str = "relu"
    dropout: float = 0.0
    out_channels: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NetworkHead":
        data = dict(payload or {})
        if "type" in data and "head_type" not in data:
            data["head_type"] = data.pop("type")
        allowed = set(cls.__dataclass_fields__.keys())
        params = dict(data.get("params") or {})
        extras = {k: v for k, v in data.items() if k not in allowed}
        params.update(extras)
        filtered = {k: v for k, v in data.items() if k in allowed}
        filtered["params"] = params
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        if not data.get("params"):
            data.pop("params", None)
        return data


@dataclass
class NetworkArchitecture:
    name: str
    blocks: List[NetworkBlock]
    heads: List[NetworkHead]
    schema_version: str = SCHEMA_VERSION
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    description: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    architecture_hash: Optional[str] = None
    favorite: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NetworkArchitecture":
        data = dict(payload or {})
        blocks = [NetworkBlock.from_dict(item) for item in data.get("blocks", [])]
        heads = [NetworkHead.from_dict(item) for item in data.get("heads", [])]
        data["blocks"] = blocks
        data["heads"] = heads
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        if "name" not in filtered:
            filtered["name"] = "Untitled architecture"
        if "schema_version" not in filtered:
            filtered["schema_version"] = SCHEMA_VERSION
        return cls(**filtered)

    def to_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        data["blocks"] = [block.to_dict() for block in self.blocks]
        data["heads"] = [head.to_dict() for head in self.heads]
        if include_hash:
            data["architecture_hash"] = self.architecture_hash or architecture_hash(self)
        else:
            data.pop("architecture_hash", None)
        return data


def _coerce_architecture(arch: ArchitectureLike) -> NetworkArchitecture:
    if isinstance(arch, NetworkArchitecture):
        return arch
    if isinstance(arch, Mapping):
        return NetworkArchitecture.from_dict(arch)
    if isinstance(arch, (str, Path)):
        return load_architecture(arch)
    raise TypeError(f"Unsupported architecture type: {type(arch)!r}")


def _canonical_architecture_payload(arch: ArchitectureLike) -> Dict[str, Any]:
    obj = _coerce_architecture(arch)
    return {
        "schema_version": obj.schema_version,
        "blocks": [block.to_dict() for block in obj.blocks],
        "heads": [head.to_dict() for head in obj.heads],
    }


def architecture_hash(arch: ArchitectureLike) -> str:
    payload = _canonical_architecture_payload(arch)
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def default_architecture(
    *,
    name: str = "Default configurable GAT",
    description: str = "Two-layer configurable hetero GAT baseline.",
    hidden_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.3,
    aggregation: str = "mean",
    activation: str = "relu",
    residual: bool = True,
    temporal_type: str = "snapshot",
) -> NetworkArchitecture:
    blocks: List[NetworkBlock] = []
    for idx in range(max(1, int(num_layers))):
        blocks.append(
            NetworkBlock(
                block_type="hetero_conv",
                name=f"conv_{idx + 1}",
                conv_type="GATConv",
                hidden_channels=int(hidden_channels),
                num_heads=int(num_heads),
                aggregation=str(aggregation),
                activation=str(activation),
                dropout=float(dropout),
                residual=bool(residual),
                norm="layer_norm",
            )
        )
    blocks.append(
        NetworkBlock(
            block_type="temporal_head",
            name="temporal",
            temporal_type=str(temporal_type),
            params={},
        )
    )
    arch = NetworkArchitecture(
        name=str(name),
        description=str(description),
        blocks=blocks,
        heads=[
            NetworkHead(
                name="primary",
                head_type="classifier",
                primary=True,
                hidden_channels=[],
                activation=str(activation),
                dropout=0.0,
                out_channels=None,
            )
        ],
    )
    arch.architecture_hash = architecture_hash(arch)
    return arch


def validate_architecture(
    arch: ArchitectureLike,
    graph_info: Optional[Mapping[str, Any]] = None,
) -> ValidationResult:
    obj = _coerce_architecture(arch)
    info = dict(graph_info or {})
    result = ValidationResult(is_valid=True)

    if str(obj.schema_version).split(".", 1)[0] != str(SCHEMA_VERSION).split(".", 1)[0]:
        result.add_error(
            f"Unsupported schema_version={obj.schema_version!r}; expected major version 1."
        )

    if not obj.blocks:
        result.add_error("Architecture must define at least one block.")
    if not obj.heads:
        result.add_error("Architecture must define at least one classifier head.")

    conv_count = 0
    temporal_count = 0
    current_dim = info.get("in_channels")
    for idx, block in enumerate(obj.blocks):
        block_type = str(block.block_type or "").strip().lower()
        if block_type not in SUPPORTED_BLOCK_TYPES:
            result.add_error(f"Block {idx}: unsupported block_type={block.block_type!r}.")
            continue

        if block_type == "hetero_conv":
            conv_count += 1
            conv_type = _normalize_conv_type(block.conv_type)
            if conv_type not in SUPPORTED_CONVS:
                result.add_error(f"Block {idx}: unsupported conv_type={block.conv_type!r}.")
            try:
                hidden = int(block.hidden_channels or 0)
            except Exception:
                hidden = 0
            try:
                heads = int(block.num_heads or 0)
            except Exception:
                heads = 0
            if hidden <= 0:
                result.add_error(f"Block {idx}: hidden_channels must be > 0.")
            if heads <= 0:
                result.add_error(f"Block {idx}: num_heads must be > 0.")
            aggr = _normalize_aggr(block.aggregation)
            if aggr not in SUPPORTED_AGGREGATIONS:
                result.add_error(f"Block {idx}: unsupported aggregation={block.aggregation!r}.")
            activation = _normalize_activation(block.activation)
            if activation not in SUPPORTED_ACTIVATIONS:
                result.add_error(f"Block {idx}: unsupported activation={block.activation!r}.")
            norm = str(block.norm or "layer_norm").strip().lower()
            if norm not in SUPPORTED_NORMS:
                result.add_error(f"Block {idx}: unsupported norm={block.norm!r}.")
            try:
                dropout = float(block.dropout or 0.0)
            except Exception:
                dropout = -1.0
            if dropout < 0.0 or dropout > 0.9:
                result.add_error(f"Block {idx}: dropout must be between 0.0 and 0.9.")
            if hidden > 0 and heads > 0:
                current_dim = hidden * heads

        elif block_type == "activation":
            activation = _normalize_activation(block.activation)
            if activation not in SUPPORTED_ACTIVATIONS:
                result.add_error(f"Block {idx}: unsupported activation={block.activation!r}.")

        elif block_type == "norm":
            norm = str(block.norm or "layer_norm").strip().lower()
            if norm not in SUPPORTED_NORMS:
                result.add_error(f"Block {idx}: unsupported norm={block.norm!r}.")
            if current_dim is None:
                result.add_warning(
                    f"Block {idx}: norm dimension will be inferred when the model is built."
                )

        elif block_type == "dropout":
            try:
                dropout = float(block.dropout or 0.0)
            except Exception:
                dropout = -1.0
            if dropout < 0.0 or dropout > 0.9:
                result.add_error(f"Block {idx}: dropout must be between 0.0 and 0.9.")

        elif block_type == "temporal_head":
            temporal_count += 1
            temporal = _normalize_temporal(block.temporal_type)
            if temporal not in SUPPORTED_TEMPORAL_HEADS:
                result.add_error(
                    f"Block {idx}: unsupported temporal_type={block.temporal_type!r}."
                )
            if (
                temporal != "snapshot"
                and "has_sequence_index" in info
                and not bool(info.get("has_sequence_index"))
            ):
                result.add_error(
                    f"Block {idx}: temporal_type={temporal!r} requires a SequenceIndex."
                )

    if conv_count == 0:
        result.add_error("Architecture must include at least one hetero_conv block.")
    if temporal_count > 1:
        result.add_error("Architecture can include at most one temporal_head block.")

    primary_count = 0
    seen_heads = set()
    out_channels = info.get("out_channels")
    for idx, head in enumerate(obj.heads):
        name = str(head.name or "").strip()
        if not name:
            result.add_error(f"Head {idx}: name is required.")
            continue
        key = _slugify(name, fallback=f"head_{idx}")
        if key in seen_heads:
            result.add_error(f"Head {idx}: duplicate head name={name!r}.")
        seen_heads.add(key)
        if str(head.head_type or "classifier").strip().lower() != "classifier":
            result.add_error(f"Head {idx}: only classifier heads are supported.")
        if _safe_bool(head.primary, False):
            primary_count += 1
        activation = _normalize_activation(head.activation)
        if activation not in SUPPORTED_ACTIVATIONS:
            result.add_error(f"Head {idx}: unsupported activation={head.activation!r}.")
        try:
            dropout = float(head.dropout or 0.0)
        except Exception:
            dropout = -1.0
        if dropout < 0.0 or dropout > 0.9:
            result.add_error(f"Head {idx}: dropout must be between 0.0 and 0.9.")
        for layer_idx, hidden in enumerate(head.hidden_channels or []):
            try:
                hidden_int = int(hidden)
            except Exception:
                hidden_int = 0
            if hidden_int <= 0:
                result.add_error(
                    f"Head {idx}: hidden_channels[{layer_idx}] must be > 0."
                )
        if out_channels is not None and head.out_channels is not None:
            try:
                if int(head.out_channels) != int(out_channels):
                    result.add_error(
                        f"Head {idx}: out_channels={head.out_channels} does not "
                        f"match graph out_channels={out_channels}."
                    )
            except Exception:
                result.add_error(f"Head {idx}: invalid out_channels={head.out_channels!r}.")

    if primary_count != 1:
        result.add_error("Exactly one head must be marked as primary.")

    return result


def load_architecture(path: Union[str, Path]) -> NetworkArchitecture:
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return NetworkArchitecture.from_dict(payload)


def save_architecture(
    arch: ArchitectureLike,
    path: Union[str, Path],
    *,
    validate: bool = True,
) -> str:
    obj = _coerce_architecture(arch)
    if validate:
        validate_architecture(obj).raise_for_errors()
    out = copy.deepcopy(obj)
    out.updated_at = _utc_now()
    out.architecture_hash = architecture_hash(out)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as fh:
        json.dump(out.to_dict(include_hash=True), fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    return str(path_obj)


def list_architectures(root_dir: Union[str, Path]) -> List[NetworkArchitecture]:
    root = Path(root_dir)
    if not root.exists():
        return []
    architectures: List[NetworkArchitecture] = []
    for path in sorted(root.glob("*.json")):
        try:
            arch = load_architecture(path)
            arch.metadata = dict(arch.metadata or {})
            arch.metadata.setdefault("path", str(path))
            architectures.append(arch)
        except Exception:
            continue
    architectures.sort(key=lambda item: (str(item.name).lower(), str(item.id)))
    return architectures


def _architecture_library_path(root_dir: Union[str, Path], arch: NetworkArchitecture) -> Path:
    slug = _slugify(arch.name)
    arch_id = _slugify(arch.id[:12], fallback=uuid.uuid4().hex[:12])
    return Path(root_dir) / f"{slug}_{arch_id}.json"


def duplicate_architecture(
    source: ArchitectureLike,
    root_dir: Union[str, Path],
    *,
    name: Optional[str] = None,
) -> str:
    arch = copy.deepcopy(_coerce_architecture(source))
    arch.id = uuid.uuid4().hex
    arch.name = str(name or f"{arch.name} copy")
    arch.created_at = _utc_now()
    arch.updated_at = arch.created_at
    arch.architecture_hash = architecture_hash(arch)
    return save_architecture(arch, _architecture_library_path(root_dir, arch))


def delete_architecture(path: Union[str, Path]) -> bool:
    path_obj = Path(path)
    if not path_obj.exists():
        return False
    path_obj.unlink()
    return True


def export_architecture(
    source: ArchitectureLike,
    destination_path: Union[str, Path],
) -> str:
    if isinstance(source, (str, Path)):
        src_path = Path(source)
        if src_path.exists():
            dest = Path(destination_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dest)
            return str(dest)
    return save_architecture(source, destination_path)


def import_architecture(
    source_path: Union[str, Path],
    root_dir: Union[str, Path],
    *,
    name: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    arch = load_architecture(source_path)
    if name:
        arch.name = str(name)
    if not arch.id:
        arch.id = uuid.uuid4().hex
    arch.architecture_hash = architecture_hash(arch)
    destination = _architecture_library_path(root_dir, arch)
    if destination.exists() and not overwrite:
        arch.id = uuid.uuid4().hex
        destination = _architecture_library_path(root_dir, arch)
    return save_architecture(arch, destination)


def _activation_module(name: object) -> torch.nn.Module:
    activation = _normalize_activation(name)
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=0.01)
    if activation == "elu":
        return torch.nn.ELU()
    if activation == "gelu":
        return torch.nn.GELU()
    if activation == "silu":
        return torch.nn.SiLU()
    if activation == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported activation: {name!r}")


def _module_key(value: object, fallback: str) -> str:
    return _slugify(str(value or fallback), fallback=fallback).replace(".", "_")


class _ClassifierHead(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        out_channels: int,
        hidden_channels: Sequence[int],
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        dim_in = int(input_dim)
        for hidden in hidden_channels:
            hidden_int = int(hidden)
            layers.append(Linear(dim_in, hidden_int))
            layers.append(_activation_module(activation))
            if float(dropout) > 0:
                layers.append(torch.nn.Dropout(float(dropout)))
            dim_in = hidden_int
        layers.append(Linear(dim_in, int(out_channels)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _HeteroConvBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        block: NetworkBlock,
        in_channels: int,
        edge_feature_dim: int,
        edge_types: Sequence[Tuple[str, str, str]],
        node_type: str,
        use_relation_self_loops: bool,
    ) -> None:
        super().__init__()
        conv_type = _normalize_conv_type(block.conv_type)
        hidden_channels = int(block.hidden_channels or 0)
        num_heads = int(block.num_heads or 1)
        self.output_dim = hidden_channels * num_heads
        self.edge_feature_dim = int(edge_feature_dim or 0)
        self.node_type = str(node_type)
        self.activation = _activation_module(block.activation)
        self.dropout = float(block.dropout or 0.0)
        self.use_residual = _safe_bool(block.residual, False)
        self.norm_name = str(block.norm or "layer_norm").strip().lower()
        edge_dim = self.edge_feature_dim if self.edge_feature_dim > 0 else None

        conv_dict = {}
        for edge_type in edge_types:
            if conv_type == "GATConv":
                conv_dict[edge_type] = GATConvSaveAlpha(
                    int(in_channels),
                    hidden_channels,
                    heads=num_heads,
                    add_self_loops=bool(use_relation_self_loops),
                    edge_dim=edge_dim,
                )
            elif conv_type == "TransformerConv":
                conv_dict[edge_type] = TransformerConvSaveAlpha(
                    int(in_channels),
                    hidden_channels,
                    heads=num_heads,
                    dropout=self.dropout,
                    edge_dim=edge_dim,
                )
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type!r}")

        self.conv = HeteroConv(conv_dict, aggr=_normalize_aggr(block.aggregation))
        self.norm = (
            LayerNorm(self.output_dim)
            if self.norm_name == "layer_norm"
            else torch.nn.Identity()
        )
        self.residual_lin = (
            Linear(int(in_channels), self.output_dim) if self.use_residual else None
        )
        self._edge_attr_mismatch_warned = set()

    def _active_edge_inputs(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Mapping[Tuple[str, str, str], Optional[torch.Tensor]],
    ) -> Tuple[
        Dict[Tuple[str, str, str], torch.Tensor],
        Dict[Tuple[str, str, str], torch.Tensor],
    ]:
        active_edge_types = list(self.conv.convs.keys())
        active_eid = {
            edge_type: edge_index
            for edge_type, edge_index in edge_index_dict.items()
            if edge_type in active_edge_types
        }
        active_ead: Dict[Tuple[str, str, str], torch.Tensor] = {}
        if not active_eid:
            return active_eid, active_ead
        ref = next(iter(x_dict.values()))
        for edge_type in active_edge_types:
            if edge_type not in active_eid:
                continue
            num_edges = int(active_eid[edge_type].shape[1])
            edge_attr = edge_attr_dict.get(edge_type)
            if edge_attr is not None:
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)
                if int(edge_attr.size(0)) != num_edges:
                    if self.edge_feature_dim > 0:
                        edge_attr = torch.zeros(
                            (num_edges, self.edge_feature_dim),
                            dtype=ref.dtype,
                            device=ref.device,
                        )
                    else:
                        edge_attr = None
                if edge_attr is not None:
                    active_ead[edge_type] = edge_attr
            elif self.edge_feature_dim > 0:
                active_ead[edge_type] = torch.zeros(
                    (num_edges, self.edge_feature_dim),
                    dtype=ref.dtype,
                    device=ref.device,
                )
        return active_eid, active_ead

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Mapping[Tuple[str, str, str], Optional[torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        active_eid, active_ead = self._active_edge_inputs(
            x_dict,
            edge_index_dict,
            edge_attr_dict,
        )
        out_dict = self.conv(dict(x_dict), active_eid, active_ead)
        if self.node_type in out_dict:
            x_pm = out_dict[self.node_type]
            if self.residual_lin is not None and self.node_type in x_dict:
                x_pm = x_pm + self.residual_lin(x_dict[self.node_type])
            x_pm = self.norm(x_pm)
            x_pm = self.activation(x_pm)
            if self.dropout > 0:
                x_pm = torch.nn.functional.dropout(
                    x_pm,
                    p=self.dropout,
                    training=self.training,
                )
            out_dict[self.node_type] = x_pm

        attentions = {}
        for edge_type, conv_layer in self.conv.convs.items():
            alpha = getattr(conv_layer, "_alpha", None)
            if alpha is not None:
                key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
                attentions[key] = alpha
                conv_layer._alpha = None
        return out_dict, attentions


class _ActivationStage(torch.nn.Module):
    def __init__(self, activation: str, node_type: str) -> None:
        super().__init__()
        self.activation = _activation_module(activation)
        self.node_type = str(node_type)

    def forward(self, x_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(x_dict)
        if self.node_type in out:
            out[self.node_type] = self.activation(out[self.node_type])
        return out


class _NormStage(torch.nn.Module):
    def __init__(self, dim: int, norm: str, node_type: str) -> None:
        super().__init__()
        self.node_type = str(node_type)
        self.norm = LayerNorm(int(dim)) if str(norm) == "layer_norm" else torch.nn.Identity()

    def forward(self, x_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(x_dict)
        if self.node_type in out:
            out[self.node_type] = self.norm(out[self.node_type])
        return out


class _DropoutStage(torch.nn.Module):
    def __init__(self, dropout: float, node_type: str) -> None:
        super().__init__()
        self.dropout = float(dropout or 0.0)
        self.node_type = str(node_type)

    def forward(self, x_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(x_dict)
        if self.node_type in out and self.dropout > 0:
            out[self.node_type] = torch.nn.functional.dropout(
                out[self.node_type],
                p=self.dropout,
                training=self.training,
            )
        return out


class ConfigurableHeteroGNN(torch.nn.Module):
    """
    Configurable hetero GNN encoder with classifier heads.

    The output contract matches the existing GNN stack:
    forward(...) returns logits_dict, embeddings_dict, attentions. The primary
    classifier is exposed as logits_dict[node_type]. Auxiliary heads are stored
    in self.aux_logits after each forward pass.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        edge_feature_dim: int,
        architecture: ArchitectureLike,
        edge_types: Optional[Sequence[Tuple[str, str, str]]] = None,
        node_type: str = "pm",
        use_relation_self_loops: bool = True,
        graph_info: Optional[Mapping[str, Any]] = None,
        validate: bool = True,
    ) -> None:
        super().__init__()
        self.architecture = _coerce_architecture(architecture)
        info = dict(graph_info or {})
        info.setdefault("in_channels", int(in_channels))
        info.setdefault("out_channels", int(out_channels))
        if validate:
            validate_architecture(self.architecture, info).raise_for_errors()

        self.node_type = str(node_type)
        self.edge_types = tuple(edge_types or DEFAULT_EDGE_TYPES)
        self.edge_feature_dim = int(edge_feature_dim or 0)
        self.out_channels = int(out_channels)
        self.gnn_variant = "configurable"
        self.architecture_hash = architecture_hash(self.architecture)
        self.temporal_block: Optional[NetworkBlock] = None
        self.aux_logits: Dict[str, torch.Tensor] = {}

        self.stages = ModuleList()
        self.stage_kinds: List[str] = []
        current_dim = int(in_channels)
        conv_layers = 0

        for block in self.architecture.blocks:
            block_type = str(block.block_type or "").strip().lower()
            if block_type == "hetero_conv":
                stage = _HeteroConvBlock(
                    block=block,
                    in_channels=current_dim,
                    edge_feature_dim=self.edge_feature_dim,
                    edge_types=self.edge_types,
                    node_type=self.node_type,
                    use_relation_self_loops=bool(use_relation_self_loops),
                )
                current_dim = int(stage.output_dim)
                self.stages.append(stage)
                self.stage_kinds.append("hetero_conv")
                conv_layers += 1
            elif block_type == "activation":
                self.stages.append(_ActivationStage(block.activation or "relu", self.node_type))
                self.stage_kinds.append("activation")
            elif block_type == "norm":
                norm_name = str(block.norm or "layer_norm").strip().lower()
                self.stages.append(_NormStage(current_dim, norm_name, self.node_type))
                self.stage_kinds.append("norm")
            elif block_type == "dropout":
                self.stages.append(_DropoutStage(float(block.dropout or 0.0), self.node_type))
                self.stage_kinds.append("dropout")
            elif block_type == "temporal_head":
                self.temporal_block = block

        self.embedding_dim = int(current_dim)
        self.hidden_channels = int(self.embedding_dim)
        self.num_heads = 1
        self.num_layers = int(conv_layers)
        self.dropout = max(
            [
                float(block.dropout or 0.0)
                for block in self.architecture.blocks
                if block.dropout is not None
            ]
            or [0.0]
        )

        self.heads = ModuleDict()
        self._head_key_to_name: Dict[str, str] = {}
        self.primary_head_key = ""
        for idx, head in enumerate(self.architecture.heads):
            key = _module_key(head.name, fallback=f"head_{idx}")
            if key in self.heads:
                key = f"{key}_{idx}"
            head_out = int(head.out_channels or self.out_channels)
            self.heads[key] = _ClassifierHead(
                input_dim=self.embedding_dim,
                out_channels=head_out,
                hidden_channels=list(head.hidden_channels or []),
                activation=head.activation,
                dropout=float(head.dropout or 0.0),
            )
            self._head_key_to_name[key] = str(head.name)
            if _safe_bool(head.primary, False):
                self.primary_head_key = key

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Mapping[Tuple[str, str, str], Optional[torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if edge_attr_dict is None:
            edge_attr_dict = {}

        out_dict = dict(x_dict)
        attentions: Dict[str, torch.Tensor] = {}
        conv_idx = 0
        for stage, kind in zip(self.stages, self.stage_kinds):
            if kind == "hetero_conv":
                conv_idx += 1
                out_dict, stage_attentions = stage(out_dict, edge_index_dict, edge_attr_dict)
                for key, value in stage_attentions.items():
                    attentions[f"conv{conv_idx}_{key}"] = value
            else:
                out_dict = stage(out_dict)

        embeddings_dict = {key: value.clone() for key, value in out_dict.items()}
        if self.node_type not in out_dict:
            raise RuntimeError(f"Model output does not contain node_type={self.node_type!r}.")
        if not self.primary_head_key:
            raise RuntimeError("No primary head is configured.")

        pm_embedding = out_dict[self.node_type]
        head_outputs: Dict[str, torch.Tensor] = {}
        for key, head in self.heads.items():
            head_name = self._head_key_to_name.get(key, key)
            head_outputs[head_name] = head(pm_embedding)

        primary_name = self._head_key_to_name.get(self.primary_head_key, self.primary_head_key)
        primary_logits = head_outputs[primary_name]
        self.aux_logits = {
            name: logits
            for name, logits in head_outputs.items()
            if name != primary_name
        }
        return {self.node_type: primary_logits}, embeddings_dict, attentions


__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_CONVS",
    "SUPPORTED_AGGREGATIONS",
    "SUPPORTED_ACTIVATIONS",
    "SUPPORTED_TEMPORAL_HEADS",
    "SUPPORTED_BLOCK_TYPES",
    "SUPPORTED_NORMS",
    "DEFAULT_EDGE_TYPES",
    "ValidationResult",
    "NetworkBlock",
    "NetworkHead",
    "NetworkArchitecture",
    "ConfigurableHeteroGNN",
    "architecture_hash",
    "default_architecture",
    "validate_architecture",
    "load_architecture",
    "save_architecture",
    "list_architectures",
    "duplicate_architecture",
    "delete_architecture",
    "export_architecture",
    "import_architecture",
]
