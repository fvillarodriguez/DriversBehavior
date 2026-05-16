import sys as _sys
import warnings
from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ModuleList
from src.gnn_mps_scatter import install_gnn_mps_scatter_policy

install_gnn_mps_scatter_policy()

from torch_geometric.nn import GATConv, HeteroConv, TransformerConv
from torch.utils.checkpoint import checkpoint
from src.config import DEBUG, XAI

# PyTorch's non-reentrant checkpoint path still enters torch.cpu.amp.autocast()
# internally on some 2.x releases. Keep the filter scoped to that upstream
# deprecation so GNN training logs stay readable without hiding local warnings.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated\. Please use `torch\.amp\.autocast\('cpu', args\.\.\.\)` instead\.",
    category=FutureWarning,
    module=r"torch\.utils\.checkpoint",
)

# Ensure module is available under the expected name for PyG inspector
_sys.modules.setdefault("src.gat_model", _sys.modules[__name__])

class GATConvSaveAlpha(GATConv):
    """
    Wrapper over PyG's GATConv that stores the last attention coefficients
    in `self._alpha` as a side-effect, while returning only the node features
    (so it remains compatible with HeteroConv expectations).
    """
    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=False):
        # Always request attention weights from super, but return as requested
        out = super().forward(x, edge_index, edge_attr=edge_attr, size=size, return_attention_weights=True)
        if isinstance(out, tuple):
            x_out, (eff_eidx, alpha) = out
            # Save attention weights for XAI/regularization
            self._alpha = alpha
            self._alpha_edge_index = eff_eidx
            if return_attention_weights:
                return x_out, (eff_eidx, alpha)
            else:
                return x_out
        # Fallback: if upstream signature changes
        self._alpha = None
        self._alpha_edge_index = None
        return out


class TransformerConvSaveAlpha(TransformerConv):
    """
    Wrapper over PyG's TransformerConv that stores the last attention coefficients
    in `self._alpha`, mirroring GATConvSaveAlpha.
    """
    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=False):
        out = super().forward(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        if isinstance(out, tuple):
            x_out, (eff_eidx, alpha) = out
            self._alpha = alpha
            self._alpha_edge_index = eff_eidx
            if return_attention_weights:
                return x_out, (eff_eidx, alpha)
            return x_out
        self._alpha = None
        self._alpha_edge_index = None
        return out


def _edge_type_module_key(edge_type):
    if isinstance(edge_type, tuple) and len(edge_type) == 3:
        return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
    return str(edge_type)


def _resolve_per_type_int(spec, edge_types, default):
    """Return dict {edge_type: int} from either a scalar, a dict (by tuple or by
    serialized key), or None. Missing entries fall back to `default`."""
    if isinstance(spec, dict):
        normalized = {}
        for k, v in spec.items():
            if isinstance(k, tuple) and len(k) == 3:
                key = k
            else:
                parsed = None
                txt = str(k)
                for sep in ("___", "__"):
                    if sep in txt:
                        parts = txt.split(sep)
                        if len(parts) == 3 and all(parts):
                            parsed = tuple(parts)
                            break
                key = parsed if parsed is not None else k
            try:
                normalized[key] = int(v) if v is not None else None
            except Exception:
                normalized[key] = None
        out = {}
        for et in edge_types:
            val = normalized.get(et, None)
            if val is None:
                val = normalized.get(_edge_type_module_key(et), None)
            out[et] = int(val) if val is not None else int(default or 0)
        return out
    if spec is None:
        return {et: int(default or 0) for et in edge_types}
    try:
        scalar = int(spec)
    except Exception:
        scalar = int(default or 0)
    return {et: scalar for et in edge_types}


def _resolve_per_type_str(spec, edge_types, default):
    """Same shape contract as `_resolve_per_type_int`, but for string values
    (e.g. `edge_encoder_kinds`). Missing entries fall back to `default`."""
    default_str = str(default or "")
    if isinstance(spec, dict):
        normalized = {}
        for k, v in spec.items():
            if isinstance(k, tuple) and len(k) == 3:
                key = k
            else:
                parsed = None
                txt = str(k)
                for sep in ("___", "__"):
                    if sep in txt:
                        parts = txt.split(sep)
                        if len(parts) == 3 and all(parts):
                            parsed = tuple(parts)
                            break
                key = parsed if parsed is not None else k
            normalized[key] = (str(v).strip().lower() if v is not None else None)
        out = {}
        for et in edge_types:
            val = normalized.get(et, None)
            if val is None:
                val = normalized.get(_edge_type_module_key(et), None)
            out[et] = (val if val else default_str)
        return out
    if spec is None:
        return {et: default_str for et in edge_types}
    scalar = str(spec).strip().lower()
    return {et: (scalar or default_str) for et in edge_types}


def _resolve_per_type_float(spec, edge_types, default):
    """Like _resolve_per_type_int for float-valued parameters (e.g. dropout)."""
    if isinstance(spec, dict):
        normalized = {}
        for k, v in spec.items():
            if isinstance(k, tuple) and len(k) == 3:
                key = k
            else:
                parsed = None
                txt = str(k)
                for sep in ("___", "__"):
                    if sep in txt:
                        parts = txt.split(sep)
                        if len(parts) == 3 and all(parts):
                            parsed = tuple(parts)
                            break
                key = parsed if parsed is not None else k
            try:
                normalized[key] = float(v) if v is not None else None
            except Exception:
                normalized[key] = None
        out = {}
        for et in edge_types:
            val = normalized.get(et, None)
            if val is None:
                val = normalized.get(_edge_type_module_key(et), None)
            out[et] = float(val) if val is not None else float(default or 0.0)
        return out
    if spec is None:
        return {et: float(default or 0.0) for et in edge_types}
    try:
        scalar = float(spec)
    except Exception:
        scalar = float(default or 0.0)
    return {et: scalar for et in edge_types}


# Edge types por defecto si no se pasa el parámetro al constructor.
# Los grafos nuevos usan `spatial` bidireccional y no crean relacion inversa separada.
# Grafos legacy con relaciones extra deben pasar edge_types=tuple(data.edge_types).
DEFAULT_EDGE_TYPES = (
    ('pm', 'spatial', 'pm'),
    ('pm', 'temporal', 'pm'),
    ('pm', 'st_fwd', 'pm'),
)


class EdgeAttrEncoder(torch.nn.Module):
    """Small MLP to learn a task-adapted representation of raw edge attributes.

    Es el encoder por defecto (`kind="mlp"`). Otros tipos comparten la misma
    interfaz `(in_dim, hidden_dim, out_dim, dropout)` → `forward(edge_attr) ->
    (E, out_dim)` para que el registry los pueda intercambiar sin tocar el GAT.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim, out_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            return edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        return self.net(edge_attr)


class MLPResidualEncoder(torch.nn.Module):
    """MLP + skip-connection. La proyección residual es lineal cuando
    `in_dim != out_dim` (no se puede sumar directo el raw)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim, out_dim),
        )
        self.skip = Linear(in_dim, out_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            return edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        return self.mlp(edge_attr) + self.skip(edge_attr)


class LayerNormMLPEncoder(torch.nn.Module):
    """LayerNorm sobre `edge_attr` raw, luego MLP. Útil cuando las features
    tienen escalas muy distintas (delta-features vs `dist_km` vs gradientes)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = LayerNorm(in_dim)
        self.net = torch.nn.Sequential(
            Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim, out_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            return edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        return self.net(self.norm(edge_attr))


class Time2VecEncoder(torch.nn.Module):
    """Time2Vec aplicado feature-wise sobre `edge_attr`.

    Para cada feature de entrada `x_i` produce `[w_i*x_i + b_i, sin(w_{i,1}*x_i
    + b_{i,1}), …, sin(w_{i,k-1}*x_i + b_{i,k-1})]` (un componente lineal +
    `k-1` periódicos), y luego una proyección lineal a `out_dim`.

    `hidden_dim` se interpreta como `k` (número de componentes Time2Vec por
    feature). El encoding total intermedio es `in_dim * k` antes de proyectar.
    Diseñado para variables continuas con estacionalidad o magnitudes muy
    variables (p.ej. `dt`).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        k = max(1, int(hidden_dim))
        self.in_dim = int(in_dim)
        self.k = k
        # Componente lineal por feature.
        self.linear_w = torch.nn.Parameter(torch.randn(self.in_dim))
        self.linear_b = torch.nn.Parameter(torch.zeros(self.in_dim))
        # Componentes periódicos: (in_dim, k-1).
        if k > 1:
            self.freq_w = torch.nn.Parameter(torch.randn(self.in_dim, k - 1) * 0.1)
            self.freq_b = torch.nn.Parameter(torch.zeros(self.in_dim, k - 1))
        else:
            # Registramos buffers vacíos para mantener la firma del state_dict
            # estable y simplificar la inferencia de `kind` desde checkpoint.
            self.register_parameter("freq_w", None)
            self.register_parameter("freq_b", None)
        self.proj = Linear(self.in_dim * k, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            return edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        # x: (E, in_dim)
        linear_part = self.linear_w * edge_attr + self.linear_b  # (E, in_dim)
        if self.freq_w is None:
            combined = linear_part
        else:
            # (E, in_dim, 1) * (in_dim, k-1) -> (E, in_dim, k-1)
            sin_part = torch.sin(
                edge_attr.unsqueeze(-1) * self.freq_w + self.freq_b
            )
            combined = torch.cat([linear_part.unsqueeze(-1), sin_part], dim=-1)
            combined = combined.reshape(edge_attr.shape[0], self.in_dim * self.k)
        return self.dropout(self.proj(combined))


# Registry de tipos de encoder por arista. Las claves se sirven a la UI y se
# serializan en `edge_encoder_per_type[<et>]["kind"]`.
EDGE_ENCODER_REGISTRY = {
    "mlp": EdgeAttrEncoder,
    "mlp_residual": MLPResidualEncoder,
    "layernorm_mlp": LayerNormMLPEncoder,
    "time2vec": Time2VecEncoder,
}


def _build_edge_encoder(
    kind: Optional[str],
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Factory: devuelve el encoder según `kind`. Si el nombre es desconocido
    cae al MLP por defecto (no rompe checkpoints con encoders que no estén en
    este build)."""
    normalized = str(kind or "mlp").strip().lower()
    cls = EDGE_ENCODER_REGISTRY.get(normalized, EdgeAttrEncoder)
    return cls(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=float(dropout))


class HeteroGAT(torch.nn.Module):
    """
    Modelo GAT Heterogéneo optimizado con HeteroConv, checkpointing y recuperación de atención.
    Soporta un número dinámico de capas.

    `edge_feature_dim` puede ser un int (mismo edge_dim para todas las relaciones,
    comportamiento legacy) o, vía `edge_feature_dims`, un dict {edge_type: int}
    que asigna edge_dim distinto por tipo de arista. Esto permite que cada relación
    consuma sus propias features sin necesidad de padding con ceros.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_heads,
        dropout,
        edge_feature_dim,
        num_layers,
        use_checkpointing=False,
        aggr1='sum',
        aggr2='sum',
        use_residual=False,
        use_relation_self_loops=True,
        edge_types=None,
        edge_feature_dims=None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing
        self.num_layers = num_layers
        self.use_residual = bool(use_residual)
        self.use_relation_self_loops = bool(use_relation_self_loops)
        # Edge types: si no se especifican, usa el contrato nuevo sin relacion inversa separada.
        # Grafos legacy con relaciones extra deben pasar edge_types=tuple(data.edge_types).
        self.edge_types_list = [tuple(et) for et in (edge_types if edge_types is not None else DEFAULT_EDGE_TYPES)]

        # Per-type edge_dim: si se pasa edge_feature_dims, cada relación recibe su
        # propio edge_dim. Si no, todas comparten edge_feature_dim (legacy).
        self.edge_feature_dims_per_type = _resolve_per_type_int(
            edge_feature_dims, self.edge_types_list, default=edge_feature_dim
        )
        # Compat: `edge_feature_dim` (scalar) sigue exponiéndose; cuando hay
        # heterogeneidad lo dejamos como el máximo (lo usan helpers externos como
        # el visualizador torchview para construir tensores dummy).
        try:
            self.edge_feature_dim = int(
                max(self.edge_feature_dims_per_type.values()) if self.edge_feature_dims_per_type else (edge_feature_dim or 0)
            )
        except Exception:
            self.edge_feature_dim = int(edge_feature_dim or 0)

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.residual_lins = ModuleList()

        for i in range(num_layers):
            conv_in_channels = in_channels if i == 0 else hidden_channels * num_heads

            conv_dict = {
                et: GATConvSaveAlpha(
                    conv_in_channels,
                    hidden_channels,
                    heads=num_heads,
                    add_self_loops=self.use_relation_self_loops,
                    edge_dim=int(self.edge_feature_dims_per_type.get(et, edge_feature_dim) or 0),
                )
                for et in self.edge_types_list
            }

            self.convs.append(HeteroConv(conv_dict, aggr=aggr1 if i == 0 else aggr2))
            if self.use_residual:
                self.residual_lins.append(Linear(conv_in_channels, hidden_channels * num_heads))

            norm_dict = torch.nn.ModuleDict()
            norm_dict['pm'] = LayerNorm(hidden_channels * num_heads)
            self.norms.append(norm_dict)

        # --- Capa Lineal Final ---
        # Fijamos la proyección final con la dimensión correcta desde __init__
        # para evitar cualquier re-inicialización durante forward.
        self.pm_lin = Linear(hidden_channels * num_heads, out_channels)
       
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        attentions = {}
        # Permitir omitir edge_attr_dict: rellenar si es necesario
        if edge_attr_dict is None:
            edge_attr_dict = {}

        if not hasattr(self, "_edge_attr_mismatch_warned"):
            self._edge_attr_mismatch_warned = set()
        
        if DEBUG:
            print("\n--- HeteroGAT Forward Pass ---")
            for key, value in x_dict.items():
                print(f"Input x_{key}: shape={value.shape}, is_finite={torch.isfinite(value).all()}")
            if isinstance(edge_attr_dict, dict):
                for key, value in edge_attr_dict.items():
                    if value is not None:
                        print(f"Input edge_attr_{key}: shape={value.shape}, is_finite={torch.isfinite(value).all()}")

        for i in range(self.num_layers):
            if DEBUG:
                print(f"\n[DEBUG] GAT Layer {i} - BEFORE conv")
                for k, v in x_dict.items():
                    print(f"  - x_dict['{k}'].shape: {v.shape}")

            conv = self.convs[i]
            norm = self.norms[i]
            residual = self.residual_lins[i] if self.use_residual else None

            if self.use_checkpointing and self.training:
                def _checkpointed_block(x_pm_tensor, x_dict_cap=x_dict, conv_cap=conv, norm_cap=norm, residual_cap=residual, edge_index_dict_cap=edge_index_dict, edge_attr_dict_cap=edge_attr_dict):
                    tmp_x_dict = {**x_dict_cap, 'pm': x_pm_tensor}

                    active_edge_types = conv_cap.convs.keys()
                    active_eid = {k: v for k, v in edge_index_dict_cap.items() if k in active_edge_types}
                    # Rellena atributos de aristas faltantes si edge_dim > 0
                    active_ead = {}
                    for k in active_edge_types:
                        k_dim = int(self.edge_feature_dims_per_type.get(k, self.edge_feature_dim) or 0)
                        if k in edge_attr_dict_cap and edge_attr_dict_cap[k] is not None:
                            ea = edge_attr_dict_cap[k]
                            if k in active_eid:
                                num_e = active_eid[k].shape[1]
                                if ea.size(0) != num_e:
                                    if k_dim > 0:
                                        ref = next(iter(x_dict_cap.values()))
                                        ea = torch.zeros(
                                            (num_e, k_dim),
                                            dtype=ref.dtype,
                                            device=ref.device,
                                        )
                                        if k not in self._edge_attr_mismatch_warned:
                                            print(
                                                f"[WARN] edge_attr mismatch for {k}: "
                                                f"edge_index={num_e}, edge_attr={edge_attr_dict_cap[k].size(0)}. "
                                                "Using zeros for this batch."
                                            )
                                            self._edge_attr_mismatch_warned.add(k)
                                    else:
                                        ea = None
                            if ea is not None:
                                active_ead[k] = ea
                        else:
                            if k_dim > 0 and k in active_eid:
                                num_e = active_eid[k].shape[1]
                                ref = next(iter(x_dict_cap.values()))
                                active_ead[k] = torch.zeros((num_e, k_dim), dtype=ref.dtype, device=ref.device)
                            # si edge_dim==0, GATConv ignora edge_attr
                    
                    out_dict = conv_cap(tmp_x_dict, active_eid, active_ead)
                    
                    processed_out_dict = {}
                    for key, x in out_dict.items():
                        if key == 'pm' and residual_cap is not None:
                            x = x + residual_cap(x_pm_tensor)
                        processed_out_dict[key] = F.relu(norm_cap[key](x))
                    
                    return processed_out_dict['pm']

                x_dict = {'pm': checkpoint(_checkpointed_block, x_dict['pm'], use_reentrant=False)}

            else:
                # Original path without checkpointing
                def _conv_step(xd, eid, ead):
                    active_edge_types = conv.convs.keys()
                    active_eid = {k: v for k, v in eid.items() if k in active_edge_types}
                    # Rellenar atributos de aristas faltantes
                    active_ead = {}
                    for k in active_edge_types:
                        k_dim = int(self.edge_feature_dims_per_type.get(k, self.edge_feature_dim) or 0)
                        if k in ead and ead[k] is not None:
                            ea = ead[k]
                            if k in active_eid:
                                num_e = active_eid[k].shape[1]
                                if ea.size(0) != num_e:
                                    if k_dim > 0:
                                        ref = next(iter(xd.values()))
                                        ea = torch.zeros(
                                            (num_e, k_dim),
                                            dtype=ref.dtype,
                                            device=ref.device,
                                        )
                                        if k not in self._edge_attr_mismatch_warned:
                                            print(
                                                f"[WARN] edge_attr mismatch for {k}: "
                                                f"edge_index={num_e}, edge_attr={ead[k].size(0)}. "
                                                "Using zeros for this batch."
                                            )
                                            self._edge_attr_mismatch_warned.add(k)
                                    else:
                                        ea = None
                            if ea is not None:
                                active_ead[k] = ea
                        else:
                            if k_dim > 0 and k in active_eid:
                                num_e = active_eid[k].shape[1]
                                ref = next(iter(xd.values()))
                                active_ead[k] = torch.zeros((num_e, k_dim), dtype=ref.dtype, device=ref.device)
                    return conv(xd, active_eid, active_ead)
                
                residual_input_pm = x_dict.get('pm')
                x_dict = _conv_step(x_dict, edge_index_dict, edge_attr_dict)
                if residual is not None and residual_input_pm is not None and 'pm' in x_dict:
                    x_dict['pm'] = x_dict['pm'] + residual(residual_input_pm)
                x_dict = {key: norm[key](x) for key, x in x_dict.items()}
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}


            if DEBUG:
                print(f"\n[DEBUG] GAT Layer {i} - AFTER conv")
                for k, v in x_dict.items():
                    print(f"  - x_dict['{k}'].shape: {v.shape}")

            if XAI:
                for edge_type, conv_layer in conv.convs.items():
                    if hasattr(conv_layer, '_alpha') and conv_layer._alpha is not None:
                        key = f'conv{i+1}_{edge_type[0]}_{edge_type[1]}_{edge_type[2]}'
                        attentions[key] = conv_layer._alpha
                        conv_layer._alpha = None
            
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        z_dict = {key: x.clone() for key, x in x_dict.items()}

        # --- Capa Final ---
        if 'pm' in x_dict:
            x_pm = x_dict['pm']
            if DEBUG:
                if x_pm.size(-1) != self.pm_lin.in_features:
                    print(f"[WARN] pm_lin in_features={self.pm_lin.in_features} but x_pm dim={x_pm.size(-1)}. Check model config.")
                print(f"Before final lin x_pm: shape={x_pm.shape}, is_finite={torch.isfinite(x_pm).all()}")
            x_dict['pm'] = self.pm_lin(x_pm)
            if DEBUG:
                print(f"After lin x_pm: shape={x_dict['pm'].shape}, is_finite={torch.isfinite(x_dict['pm']).all()}")
        
        if DEBUG:
            print("--- End HeteroGAT Forward Pass ---")

        return x_dict, z_dict, attentions


class HeteroGATWithEdgeEncoder(HeteroGAT):
    """
    HeteroGAT variant that first maps raw edge attributes through an MLP encoder.
    The encoded edge attributes are then consumed by GAT attention.

    Cada tipo de arista tiene su propio `EdgeAttrEncoder` con hiperparámetros
    potencialmente distintos. Cualquiera de los parámetros (`edge_feature_dims`,
    `edge_encoder_hidden_dims`, `edge_encoded_dims`, `edge_encoder_dropouts`)
    acepta un escalar (mismo valor para todos los tipos, comportamiento legacy)
    o un dict {edge_type: valor} para configurar por tipo.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_heads,
        dropout,
        edge_feature_dim,
        num_layers,
        use_checkpointing=False,
        aggr1='sum',
        aggr2='sum',
        edge_encoder_hidden_dim=None,
        edge_encoded_dim=None,
        edge_encoder_dropout=0.0,
        use_residual=False,
        use_relation_self_loops=True,
        edge_types=None,
        edge_feature_dims=None,
        edge_encoder_hidden_dims=None,
        edge_encoded_dims=None,
        edge_encoder_dropouts=None,
        edge_encoder_kinds=None,
        edge_encoder_kind="mlp",
    ):
        raw_edge_dim_scalar = int(edge_feature_dim or 0)
        encoded_edge_dim_scalar = int(
            edge_encoded_dim if edge_encoded_dim is not None else raw_edge_dim_scalar
        )

        et_list = [tuple(et) for et in (edge_types if edge_types is not None else DEFAULT_EDGE_TYPES)]

        # Per-type raw / encoded / hidden / dropout. Si el dict no se pasa, todos
        # los tipos comparten el escalar (comportamiento legacy).
        raw_dims = _resolve_per_type_int(edge_feature_dims, et_list, default=raw_edge_dim_scalar)
        encoded_dims = _resolve_per_type_int(edge_encoded_dims, et_list, default=encoded_edge_dim_scalar)
        hidden_dims = _resolve_per_type_int(
            edge_encoder_hidden_dims,
            et_list,
            default=(
                edge_encoder_hidden_dim
                if edge_encoder_hidden_dim is not None
                else max(raw_edge_dim_scalar, encoded_edge_dim_scalar, 8)
            ),
        )
        dropouts = _resolve_per_type_float(
            edge_encoder_dropouts, et_list, default=float(edge_encoder_dropout or 0.0)
        )
        kinds = _resolve_per_type_str(
            edge_encoder_kinds, et_list, default=str(edge_encoder_kind or "mlp")
        )

        # La GATConv subyacente debe ver el edge_dim *codificado* por tipo.
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            edge_feature_dim=encoded_edge_dim_scalar,
            num_layers=num_layers,
            use_checkpointing=use_checkpointing,
            aggr1=aggr1,
            aggr2=aggr2,
            use_residual=use_residual,
            use_relation_self_loops=use_relation_self_loops,
            edge_types=et_list,
            edge_feature_dims=encoded_dims,
        )

        self.raw_edge_feature_dim = raw_edge_dim_scalar
        self.encoded_edge_feature_dim = encoded_edge_dim_scalar
        self.edge_encoder_hidden_dim = int(
            edge_encoder_hidden_dim
            if edge_encoder_hidden_dim is not None
            else max(raw_edge_dim_scalar, encoded_edge_dim_scalar, 8)
        )
        # Diccionarios por tipo (para introspección/serialización).
        self.raw_edge_feature_dims_per_type = dict(raw_dims)
        self.encoded_edge_feature_dims_per_type = dict(encoded_dims)
        self.edge_encoder_hidden_dims_per_type = dict(hidden_dims)
        self.edge_encoder_dropouts_per_type = dict(dropouts)
        self.edge_encoder_kinds_per_type = dict(kinds)

        self.edge_attr_encoders = torch.nn.ModuleDict()
        try:
            module_edge_types = list(self.convs[0].convs.keys())
        except Exception:
            module_edge_types = list(et_list)
        for edge_type in module_edge_types:
            in_dim_et = int(raw_dims.get(edge_type, raw_edge_dim_scalar) or 0)
            out_dim_et = int(encoded_dims.get(edge_type, encoded_edge_dim_scalar) or 0)
            hidden_et = int(hidden_dims.get(edge_type, self.edge_encoder_hidden_dim) or 0)
            dropout_et = float(dropouts.get(edge_type, edge_encoder_dropout) or 0.0)
            kind_et = str(kinds.get(edge_type, "mlp") or "mlp")
            if in_dim_et > 0 and out_dim_et > 0:
                self.edge_attr_encoders[_edge_type_module_key(edge_type)] = _build_edge_encoder(
                    kind=kind_et,
                    in_dim=in_dim_et,
                    hidden_dim=hidden_et,
                    out_dim=out_dim_et,
                    dropout=dropout_et,
                )

    def _encode_edge_attr_dict(self, edge_attr_dict):
        if not isinstance(edge_attr_dict, dict) or not self.edge_attr_encoders:
            return edge_attr_dict
        encoded = {}
        for edge_type, edge_attr in edge_attr_dict.items():
            if edge_attr is None:
                continue
            enc_key = _edge_type_module_key(edge_type)
            if enc_key not in self.edge_attr_encoders:
                encoded[edge_type] = edge_attr
                continue
            encoded[edge_type] = self.edge_attr_encoders[enc_key](edge_attr)
        return encoded

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        edge_attr_encoded = self._encode_edge_attr_dict(edge_attr_dict)
        return super().forward(x_dict, edge_index_dict, edge_attr_encoded)


class HeteroEdgeAware(torch.nn.Module):
    """
    Alternative hetero GNN that swaps GATConv for TransformerConv while keeping
    the same output contract as HeteroGAT.

    Soporta `edge_feature_dims` per-type (mismo contrato que HeteroGAT).
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_heads,
        dropout,
        edge_feature_dim,
        num_layers,
        use_checkpointing=False,
        aggr1='sum',
        aggr2='sum',
        use_residual=False,
        use_relation_self_loops=True,
        edge_types=None,
        edge_feature_dims=None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing
        self.num_layers = num_layers
        self.use_residual = bool(use_residual)
        self.use_relation_self_loops = bool(use_relation_self_loops)
        self.edge_types_list = [tuple(et) for et in (edge_types if edge_types is not None else DEFAULT_EDGE_TYPES)]

        self.edge_feature_dims_per_type = _resolve_per_type_int(
            edge_feature_dims, self.edge_types_list, default=edge_feature_dim
        )
        try:
            self.edge_feature_dim = int(
                max(self.edge_feature_dims_per_type.values()) if self.edge_feature_dims_per_type else (edge_feature_dim or 0)
            )
        except Exception:
            self.edge_feature_dim = int(edge_feature_dim or 0)

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.residual_lins = ModuleList()

        for i in range(num_layers):
            conv_in_channels = in_channels if i == 0 else hidden_channels * num_heads
            conv_dict = {
                et: TransformerConvSaveAlpha(
                    conv_in_channels,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=int(self.edge_feature_dims_per_type.get(et, edge_feature_dim) or 0),
                )
                for et in self.edge_types_list
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr1 if i == 0 else aggr2))
            if self.use_residual:
                self.residual_lins.append(Linear(conv_in_channels, hidden_channels * num_heads))

            norm_dict = torch.nn.ModuleDict()
            norm_dict['pm'] = LayerNorm(hidden_channels * num_heads)
            self.norms.append(norm_dict)

        self.pm_lin = Linear(hidden_channels * num_heads, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Reuse the proven forward path from HeteroGAT by delegating to an adapter instance method.
        return HeteroGAT.forward(self, x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
