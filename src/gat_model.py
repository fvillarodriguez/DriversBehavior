import sys as _sys
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ModuleList
from torch_geometric.nn import GATConv, HeteroConv, TransformerConv
from torch.utils.checkpoint import checkpoint
from src.config import DEBUG, XAI

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
            if return_attention_weights:
                return x_out, (eff_eidx, alpha)
            else:
                return x_out
        # Fallback: if upstream signature changes
        self._alpha = None
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
            if return_attention_weights:
                return x_out, (eff_eidx, alpha)
            return x_out
        self._alpha = None
        return out


def _edge_type_module_key(edge_type):
    if isinstance(edge_type, tuple) and len(edge_type) == 3:
        return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
    return str(edge_type)


class EdgeAttrEncoder(torch.nn.Module):
    """Small MLP to learn a task-adapted representation of raw edge attributes."""

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


class HeteroGAT(torch.nn.Module):
    """
    Modelo GAT Heterogéneo optimizado con HeteroConv, checkpointing y recuperación de atención.
    Soporta un número dinámico de capas.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout, edge_feature_dim, num_layers, use_checkpointing=False,
                 aggr1='sum', aggr2='sum'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing
        self.num_layers = num_layers
        self.edge_feature_dim = edge_feature_dim

        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            conv_in_channels = in_channels if i == 0 else hidden_channels * num_heads
            
            conv_dict = {
                ('pm', 'spatial', 'pm'): GATConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, add_self_loops=True, edge_dim=edge_feature_dim),
                ('pm', 'temporal', 'pm'): GATConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, add_self_loops=True, edge_dim=edge_feature_dim),
                ('pm', 'spatial_back', 'pm'): GATConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, add_self_loops=True, edge_dim=edge_feature_dim),
                ('pm', 'st_fwd', 'pm'): GATConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, add_self_loops=True, edge_dim=edge_feature_dim),
            }
            
            self.convs.append(HeteroConv(conv_dict, aggr=aggr1 if i == 0 else aggr2))

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

            if self.use_checkpointing and self.training:
                def _checkpointed_block(x_pm_tensor, x_dict_cap=x_dict, conv_cap=conv, norm_cap=norm, edge_index_dict_cap=edge_index_dict, edge_attr_dict_cap=edge_attr_dict):
                    tmp_x_dict = {**x_dict_cap, 'pm': x_pm_tensor}
                    
                    active_edge_types = conv_cap.convs.keys()
                    active_eid = {k: v for k, v in edge_index_dict_cap.items() if k in active_edge_types}
                    # Rellena atributos de aristas faltantes si edge_dim > 0
                    active_ead = {}
                    for k in active_edge_types:
                        if k in edge_attr_dict_cap and edge_attr_dict_cap[k] is not None:
                            ea = edge_attr_dict_cap[k]
                            if k in active_eid:
                                num_e = active_eid[k].shape[1]
                                if ea.size(0) != num_e:
                                    if self.edge_feature_dim and self.edge_feature_dim > 0:
                                        ref = next(iter(x_dict_cap.values()))
                                        ea = torch.zeros(
                                            (num_e, self.edge_feature_dim),
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
                            if self.edge_feature_dim and self.edge_feature_dim > 0 and k in active_eid:
                                num_e = active_eid[k].shape[1]
                                ref = next(iter(x_dict_cap.values()))
                                active_ead[k] = torch.zeros((num_e, self.edge_feature_dim), dtype=ref.dtype, device=ref.device)
                            # si edge_feature_dim==0, GATConv ignora edge_attr
                    
                    out_dict = conv_cap(tmp_x_dict, active_eid, active_ead)
                    
                    processed_out_dict = {key: F.relu(norm_cap[key](x)) for key, x in out_dict.items()}
                    
                    return processed_out_dict['pm']

                x_dict = {'pm': checkpoint(_checkpointed_block, x_dict['pm'], use_reentrant=True)}

            else:
                # Original path without checkpointing
                def _conv_step(xd, eid, ead):
                    active_edge_types = conv.convs.keys()
                    active_eid = {k: v for k, v in eid.items() if k in active_edge_types}
                    # Rellenar atributos de aristas faltantes
                    active_ead = {}
                    for k in active_edge_types:
                        if k in ead and ead[k] is not None:
                            ea = ead[k]
                            if k in active_eid:
                                num_e = active_eid[k].shape[1]
                                if ea.size(0) != num_e:
                                    if self.edge_feature_dim and self.edge_feature_dim > 0:
                                        ref = next(iter(xd.values()))
                                        ea = torch.zeros(
                                            (num_e, self.edge_feature_dim),
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
                            if self.edge_feature_dim and self.edge_feature_dim > 0 and k in active_eid:
                                num_e = active_eid[k].shape[1]
                                ref = next(iter(xd.values()))
                                active_ead[k] = torch.zeros((num_e, self.edge_feature_dim), dtype=ref.dtype, device=ref.device)
                    return conv(xd, active_eid, active_ead)
                
                x_dict = _conv_step(x_dict, edge_index_dict, edge_attr_dict)
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
    ):
        raw_edge_dim = int(edge_feature_dim or 0)
        encoded_edge_dim = int(edge_encoded_dim if edge_encoded_dim is not None else raw_edge_dim)
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            edge_feature_dim=encoded_edge_dim,
            num_layers=num_layers,
            use_checkpointing=use_checkpointing,
            aggr1=aggr1,
            aggr2=aggr2,
        )
        self.raw_edge_feature_dim = raw_edge_dim
        self.encoded_edge_feature_dim = encoded_edge_dim
        self.edge_encoder_hidden_dim = int(
            edge_encoder_hidden_dim
            if edge_encoder_hidden_dim is not None
            else max(raw_edge_dim, encoded_edge_dim, 8)
        )
        self.edge_attr_encoders = torch.nn.ModuleDict()
        if raw_edge_dim > 0 and encoded_edge_dim > 0:
            try:
                edge_types = list(self.convs[0].convs.keys())
            except Exception:
                edge_types = []
            for edge_type in edge_types:
                self.edge_attr_encoders[_edge_type_module_key(edge_type)] = EdgeAttrEncoder(
                    in_dim=raw_edge_dim,
                    hidden_dim=self.edge_encoder_hidden_dim,
                    out_dim=encoded_edge_dim,
                    dropout=float(edge_encoder_dropout),
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
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout, edge_feature_dim, num_layers, use_checkpointing=False,
                 aggr1='sum', aggr2='sum'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing
        self.num_layers = num_layers
        self.edge_feature_dim = edge_feature_dim

        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            conv_in_channels = in_channels if i == 0 else hidden_channels * num_heads
            conv_dict = {
                ('pm', 'spatial', 'pm'): TransformerConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=edge_feature_dim),
                ('pm', 'temporal', 'pm'): TransformerConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=edge_feature_dim),
                ('pm', 'spatial_back', 'pm'): TransformerConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=edge_feature_dim),
                ('pm', 'st_fwd', 'pm'): TransformerConvSaveAlpha(conv_in_channels, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=edge_feature_dim),
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr1 if i == 0 else aggr2))

            norm_dict = torch.nn.ModuleDict()
            norm_dict['pm'] = LayerNorm(hidden_channels * num_heads)
            self.norms.append(norm_dict)

        self.pm_lin = Linear(hidden_channels * num_heads, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Reuse the proven forward path from HeteroGAT by delegating to an adapter instance method.
        return HeteroGAT.forward(self, x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
