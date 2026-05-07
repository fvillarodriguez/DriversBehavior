"""Canvas visual (streamlit-flow) para editar `NetworkArchitecture`.

Reemplaza el editor por acordeón de `_render_network_builder_tab` por un grafo
drag-and-drop. Mantiene el contrato existente: lee/escribe las mismas claves
numeradas en `st.session_state` que `_gnn_builder_arch_to_state` /
`_gnn_builder_build_architecture_from_state` usan, así que la biblioteca, los
botones (Guardar/Usar/Exportar) y la previsualización siguen funcionando sin
cambios.

Tipos de nodo soportados (paridad con el editor actual):

* ``input``      — placeholder con la metadata del grafo cargado.
* ``hetero_conv``— bloque GNN configurable (conv_type/hidden/heads/aggregation/
                   activation/dropout/residual/norm).
* ``temporal_head`` — head temporal (snapshot/gru/transformer/attnpool).
* ``classifier_head`` — head clasificador con MLP opcional.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.gnn_network_builder import (
    NetworkArchitecture,
    NetworkBlock,
    NetworkHead,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_AGGREGATIONS,
    SUPPORTED_CONVS,
    SUPPORTED_NORMS,
    SUPPORTED_TEMPORAL_HEADS,
    architecture_hash,
)


# ---------------------------------------------------------------------------
# Constantes de presentación (colores por tipo de nodo)
# ---------------------------------------------------------------------------

NODE_STYLES: Dict[str, Dict[str, str]] = {
    "input": {
        "backgroundColor": "#1f3346",
        "color": "#e8f1ff",
        "border": "1px solid #3b6ea8",
    },
    "hetero_conv": {
        "backgroundColor": "#1c3d2e",
        "color": "#e7fff0",
        "border": "1px solid #3da671",
    },
    "temporal_head": {
        "backgroundColor": "#3b2c1d",
        "color": "#fff1d6",
        "border": "1px solid #b9770e",
    },
    "classifier_head": {
        "backgroundColor": "#2f1d3b",
        "color": "#f3e1ff",
        "border": "1px solid #8e44ad",
    },
}

NODE_BASE_STYLE = {
    "borderRadius": "8px",
    "padding": "8px 12px",
    "fontFamily": "Helvetica, sans-serif",
    "fontSize": "12px",
    "minWidth": "140px",
    "textAlign": "center",
}


# ---------------------------------------------------------------------------
# Validación auxiliar (no duplica `validate_architecture`; es UX)
# ---------------------------------------------------------------------------

@dataclass
class CanvasIssue:
    severity: str  # "error" | "warning"
    message: str
    node_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Conversión NetworkArchitecture → flow state
# ---------------------------------------------------------------------------

def _xy(col: int, row: int = 0) -> Tuple[float, float]:
    return (float(col * 220), float(row * 110))


def _conv_label(block: NetworkBlock, idx: int) -> str:
    hidden = int(block.hidden_channels or 0)
    heads = int(block.num_heads or 0)
    return (
        f"{block.conv_type or 'GATConv'} #{idx}\n"
        f"hidden={hidden}  heads={heads}\n"
        f"emb={hidden * heads}  drop={float(block.dropout or 0):.2f}\n"
        f"act={block.activation or 'relu'}  norm={block.norm or 'layer_norm'}"
    )


def _temporal_label(block: NetworkBlock) -> str:
    return f"Temporal head\n{block.temporal_type or 'snapshot'}"


def _head_label(head: NetworkHead, idx: int) -> str:
    hidden_repr = ",".join(str(v) for v in (head.hidden_channels or [])) or "—"
    primary_tag = " (primary)" if head.primary else ""
    return (
        f"Head #{idx}{primary_tag}\n"
        f"{head.name or f'head_{idx}'}\n"
        f"mlp=[{hidden_repr}]  drop={float(head.dropout or 0):.2f}\n"
        f"act={head.activation or 'relu'}"
    )


def _input_label(graph_info: Optional[Dict[str, Any]]) -> str:
    info = graph_info or {}
    in_dim = info.get("in_channels") or "?"
    edge_dim = info.get("edge_feature_dim") or 0
    classes = info.get("out_channels") or "?"
    seq = "Sí" if info.get("has_sequence_index") else "No"
    return f"Input\nF={in_dim}  edge_dim={edge_dim}\nclasses={classes}  seq={seq}"


def architecture_to_flow_state(
    arch: NetworkArchitecture,
    *,
    graph_info: Optional[Dict[str, Any]] = None,
    selected_id: Optional[str] = None,
):
    """Construye un ``StreamlitFlowState`` a partir de la arquitectura.

    El nodo ``input`` es decorativo (no tiene contraparte en `NetworkBlock`),
    pero ayuda al usuario a ver dónde entran los datos en el grafo.
    """

    from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
    from streamlit_flow.state import StreamlitFlowState

    nodes: List[StreamlitFlowNode] = []
    edges: List[StreamlitFlowEdge] = []

    # 1) Input decorativo
    input_id = "input"
    nodes.append(StreamlitFlowNode(
        id=input_id,
        pos=_xy(0),
        data={"content": _input_label(graph_info)},
        node_type="input",
        source_position="right",
        target_position="left",
        draggable=True,
        selectable=True,
        connectable=False,
        deletable=False,
        style={**NODE_BASE_STYLE, **NODE_STYLES["input"]},
    ))

    # 2) Bloques (hetero_conv en orden, luego temporal_head si existe)
    conv_blocks = [
        (idx, block)
        for idx, block in enumerate(arch.blocks)
        if str(block.block_type or "").strip().lower() == "hetero_conv"
    ]
    temporal_blocks = [
        (idx, block)
        for idx, block in enumerate(arch.blocks)
        if str(block.block_type or "").strip().lower() == "temporal_head"
    ]

    prev_id = input_id
    col = 1
    for layer_idx, (_orig_idx, block) in enumerate(conv_blocks, start=1):
        node_id = f"layer_{layer_idx}"
        nodes.append(StreamlitFlowNode(
            id=node_id,
            pos=_xy(col),
            data={"content": _conv_label(block, layer_idx)},
            node_type="default",
            source_position="right",
            target_position="left",
            draggable=True,
            selectable=True,
            connectable=False,
            deletable=True,
            style={**NODE_BASE_STYLE, **NODE_STYLES["hetero_conv"]},
        ))
        edges.append(StreamlitFlowEdge(
            id=f"e_{prev_id}__{node_id}",
            source=prev_id,
            target=node_id,
            edge_type="smoothstep",
            animated=False,
            deletable=False,
        ))
        prev_id = node_id
        col += 1

    # 3) Temporal head (sólo el primero, igual que el editor actual)
    if temporal_blocks:
        node_id = "temporal"
        nodes.append(StreamlitFlowNode(
            id=node_id,
            pos=_xy(col),
            data={"content": _temporal_label(temporal_blocks[0][1])},
            node_type="default",
            source_position="right",
            target_position="left",
            draggable=True,
            selectable=True,
            connectable=False,
            deletable=False,
            style={**NODE_BASE_STYLE, **NODE_STYLES["temporal_head"]},
        ))
        edges.append(StreamlitFlowEdge(
            id=f"e_{prev_id}__{node_id}",
            source=prev_id,
            target=node_id,
            edge_type="smoothstep",
            animated=False,
            deletable=False,
        ))
        backbone_tail = node_id
        col += 1
    else:
        backbone_tail = prev_id

    # 4) Heads clasificadores en paralelo desde el backbone tail.
    heads = list(arch.heads or [])
    for head_idx, head in enumerate(heads, start=1):
        node_id = f"head_{head_idx}"
        # Distribuye verticalmente cuando hay múltiples heads.
        row_offset = head_idx - 1 - (len(heads) - 1) / 2.0
        nodes.append(StreamlitFlowNode(
            id=node_id,
            pos=(_xy(col)[0], _xy(col, row=int(row_offset * 1))[1]),
            data={"content": _head_label(head, head_idx)},
            node_type="output" if head_idx == len(heads) else "default",
            source_position="right",
            target_position="left",
            draggable=True,
            selectable=True,
            connectable=False,
            deletable=head_idx > 1,  # protege al primer head para evitar quedarse sin clasificador
            style={**NODE_BASE_STYLE, **NODE_STYLES["classifier_head"]},
        ))
        edges.append(StreamlitFlowEdge(
            id=f"e_{backbone_tail}__{node_id}",
            source=backbone_tail,
            target=node_id,
            edge_type="smoothstep",
            animated=False,
            deletable=False,
        ))

    return StreamlitFlowState(nodes=nodes, edges=edges, selected_id=selected_id)


# ---------------------------------------------------------------------------
# Conversión flow state → updates a session_state numerado
# ---------------------------------------------------------------------------

def _flow_node_kind(node_id: str) -> str:
    if node_id == "input":
        return "input"
    if node_id == "temporal":
        return "temporal_head"
    if node_id.startswith("layer_"):
        return "hetero_conv"
    if node_id.startswith("head_"):
        return "classifier_head"
    return "unknown"


def sync_flow_state_to_session(flow_state, *, current_arch: NetworkArchitecture) -> NetworkArchitecture:
    """Extrae los cambios del canvas (eliminación o reordenamiento de nodos)
    y los aplica al ``st.session_state`` numerado.

    El canvas en este MVP no permite añadir nodos vía drag (eso está en la
    paleta de botones), pero sí eliminarlos. Aquí ajustamos contadores y keys
    en consecuencia.
    """

    import streamlit as st

    if flow_state is None:
        return current_arch

    present_ids = {node.id for node in flow_state.nodes}

    # ----- HeteroConv layers -----
    surviving_layer_indices: List[int] = []
    for i in range(1, int(st.session_state.get("gnn_builder_num_layers", 0)) + 1):
        if f"layer_{i}" in present_ids:
            surviving_layer_indices.append(i)

    new_layer_count = max(1, len(surviving_layer_indices))
    if surviving_layer_indices != list(range(1, new_layer_count + 1)):
        # Compactamos las keys para que vayan 1..N sin huecos.
        snapshots = []
        for old_idx in surviving_layer_indices:
            prefix = f"gnn_builder_layer_{old_idx}"
            snapshots.append({
                "conv": st.session_state.get(f"{prefix}_conv", "GATConv"),
                "hidden": int(st.session_state.get(f"{prefix}_hidden", 64)),
                "heads": int(st.session_state.get(f"{prefix}_heads", 4)),
                "aggr": st.session_state.get(f"{prefix}_aggr", "mean"),
                "activation": st.session_state.get(f"{prefix}_activation", "relu"),
                "dropout": float(st.session_state.get(f"{prefix}_dropout", 0.0)),
                "residual": bool(st.session_state.get(f"{prefix}_residual", True)),
                "norm": st.session_state.get(f"{prefix}_norm", "layer_norm"),
            })
        # Limpia las keys viejas hasta el máximo conocido.
        max_old = max(int(st.session_state.get("gnn_builder_num_layers", 0)), len(snapshots))
        for old_idx in range(1, max_old + 1):
            for suffix in ("conv", "hidden", "heads", "aggr", "activation", "dropout", "residual", "norm"):
                st.session_state.pop(f"gnn_builder_layer_{old_idx}_{suffix}", None)
        for new_idx, snap in enumerate(snapshots, start=1):
            prefix = f"gnn_builder_layer_{new_idx}"
            for suffix, value in snap.items():
                st.session_state[f"{prefix}_{suffix}"] = value
        st.session_state["gnn_builder_num_layers"] = new_layer_count

    # ----- Classifier heads -----
    surviving_head_indices: List[int] = []
    for i in range(1, int(st.session_state.get("gnn_builder_num_classifier_heads", 0)) + 1):
        if f"head_{i}" in present_ids:
            surviving_head_indices.append(i)

    new_head_count = max(1, len(surviving_head_indices))
    if surviving_head_indices != list(range(1, new_head_count + 1)):
        snapshots = []
        for old_idx in surviving_head_indices:
            prefix = f"gnn_builder_head_{old_idx}"
            snapshots.append({
                "name": st.session_state.get(f"{prefix}_name", f"head_{old_idx}"),
                "hidden": st.session_state.get(f"{prefix}_hidden", ""),
                "activation": st.session_state.get(f"{prefix}_activation", "relu"),
                "dropout": float(st.session_state.get(f"{prefix}_dropout", 0.0)),
            })
        max_old = max(int(st.session_state.get("gnn_builder_num_classifier_heads", 0)), len(snapshots))
        for old_idx in range(1, max_old + 1):
            for suffix in ("name", "hidden", "activation", "dropout"):
                st.session_state.pop(f"gnn_builder_head_{old_idx}_{suffix}", None)
        for new_idx, snap in enumerate(snapshots, start=1):
            prefix = f"gnn_builder_head_{new_idx}"
            for suffix, value in snap.items():
                st.session_state[f"{prefix}_{suffix}"] = value
        st.session_state["gnn_builder_num_classifier_heads"] = new_head_count
        if int(st.session_state.get("gnn_builder_primary_head_idx", 0)) >= new_head_count:
            st.session_state["gnn_builder_primary_head_idx"] = 0

    # Selección persistente para el inspector lateral.
    if getattr(flow_state, "selected_id", None):
        st.session_state["gnn_builder_canvas_selected"] = flow_state.selected_id

    return current_arch  # el caller usa _gnn_builder_build_architecture_from_state después


# ---------------------------------------------------------------------------
# Inspector / paleta / palette de bloques
# ---------------------------------------------------------------------------

def _selected_node_id() -> Optional[str]:
    import streamlit as st
    return st.session_state.get("gnn_builder_canvas_selected")


def _render_inspector(arch: NetworkArchitecture) -> None:
    """Panel lateral derecho: edita los hiperparámetros del nodo seleccionado.

    Al modificar un widget escribimos directamente la key de session_state
    correspondiente (`gnn_builder_layer_{N}_*` o `gnn_builder_head_{N}_*`),
    de modo que ``_gnn_builder_build_architecture_from_state`` recoja los
    cambios en el siguiente ciclo de render.
    """

    import streamlit as st

    sel = _selected_node_id()
    st.markdown("**Bloque seleccionado**")
    if not sel or sel == "input":
        st.caption("Haz click en un nodo del canvas para editar sus hiperparámetros.")
        return

    kind = _flow_node_kind(sel)

    if kind == "hetero_conv":
        try:
            idx = int(sel.split("_", 1)[1])
        except Exception:
            st.warning("ID de nodo inválido.")
            return
        prefix = f"gnn_builder_layer_{idx}"
        st.caption(f"`{sel}` (HeteroConv)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.selectbox(
                "Conv",
                list(SUPPORTED_CONVS),
                index=_safe_index(SUPPORTED_CONVS, st.session_state.get(f"{prefix}_conv", "GATConv")),
                key=f"{prefix}_conv",
            )
            st.number_input(
                "hidden_channels",
                min_value=1, max_value=2048,
                value=int(st.session_state.get(f"{prefix}_hidden", 64)),
                step=8,
                key=f"{prefix}_hidden",
            )
            st.number_input(
                "num_heads",
                min_value=1, max_value=32,
                value=int(st.session_state.get(f"{prefix}_heads", 4)),
                step=1,
                key=f"{prefix}_heads",
            )
            st.number_input(
                "dropout",
                min_value=0.0, max_value=0.9,
                value=float(st.session_state.get(f"{prefix}_dropout", 0.0)),
                step=0.05,
                key=f"{prefix}_dropout",
            )
        with col_b:
            st.selectbox(
                "Activación",
                list(SUPPORTED_ACTIVATIONS),
                index=_safe_index(SUPPORTED_ACTIVATIONS, st.session_state.get(f"{prefix}_activation", "relu")),
                key=f"{prefix}_activation",
            )
            st.selectbox(
                "Norm",
                list(SUPPORTED_NORMS),
                index=_safe_index(SUPPORTED_NORMS, st.session_state.get(f"{prefix}_norm", "layer_norm")),
                key=f"{prefix}_norm",
            )
            st.selectbox(
                "Agregación",
                list(SUPPORTED_AGGREGATIONS),
                index=_safe_index(SUPPORTED_AGGREGATIONS, st.session_state.get(f"{prefix}_aggr", "mean")),
                key=f"{prefix}_aggr",
            )
            st.checkbox(
                "Residual",
                value=bool(st.session_state.get(f"{prefix}_residual", True)),
                key=f"{prefix}_residual",
            )
        return

    if kind == "temporal_head":
        st.caption("`temporal` (Temporal head)")
        st.selectbox(
            "Temporal type",
            list(SUPPORTED_TEMPORAL_HEADS),
            index=_safe_index(
                SUPPORTED_TEMPORAL_HEADS,
                st.session_state.get("gnn_builder_temporal_type", "snapshot"),
            ),
            key="gnn_builder_temporal_type",
        )
        return

    if kind == "classifier_head":
        try:
            idx = int(sel.split("_", 1)[1])
        except Exception:
            st.warning("ID de head inválido.")
            return
        prefix = f"gnn_builder_head_{idx}"
        st.caption(f"`{sel}` (Classifier head)")
        st.text_input(
            "Nombre",
            value=str(st.session_state.get(f"{prefix}_name", f"head_{idx}")),
            key=f"{prefix}_name",
        )
        st.text_input(
            "MLP hidden (csv)",
            value=str(st.session_state.get(f"{prefix}_hidden", "")),
            key=f"{prefix}_hidden",
            help="Ej: 64,32. Vacío deja un Linear directo a clases.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.selectbox(
                "Activación",
                list(SUPPORTED_ACTIVATIONS),
                index=_safe_index(
                    SUPPORTED_ACTIVATIONS,
                    st.session_state.get(f"{prefix}_activation", "relu"),
                ),
                key=f"{prefix}_activation",
            )
        with col_b:
            st.number_input(
                "dropout",
                min_value=0.0, max_value=0.9,
                value=float(st.session_state.get(f"{prefix}_dropout", 0.0)),
                step=0.05,
                key=f"{prefix}_dropout",
            )
        # Selector "head primario" sólo si hay más de uno.
        n_heads = int(st.session_state.get("gnn_builder_num_classifier_heads", 1))
        if n_heads > 1:
            st.radio(
                "Head primario",
                list(range(n_heads)),
                index=int(st.session_state.get("gnn_builder_primary_head_idx", 0)),
                format_func=lambda i: f"Head {i + 1}",
                key="gnn_builder_primary_head_idx",
                horizontal=True,
            )
        return

    st.info(f"Tipo de nodo `{kind}` sin inspector.")


def _safe_index(options: Sequence[str], value: object) -> int:
    try:
        return list(options).index(str(value))
    except ValueError:
        return 0


def _render_palette() -> None:
    """Botones para añadir bloques nuevos al final de la cadena / heads."""

    import streamlit as st

    cols = st.columns(3)
    if cols[0].button("+ Capa GNN", key="canvas_add_layer", use_container_width=True):
        n = int(st.session_state.get("gnn_builder_num_layers", 1))
        new_idx = n + 1
        prefix = f"gnn_builder_layer_{new_idx}"
        # Hereda config de la última capa para que el usuario sólo ajuste lo nuevo.
        last = f"gnn_builder_layer_{n}"
        st.session_state[f"{prefix}_conv"] = st.session_state.get(f"{last}_conv", "GATConv")
        st.session_state[f"{prefix}_hidden"] = int(st.session_state.get(f"{last}_hidden", 64))
        st.session_state[f"{prefix}_heads"] = int(st.session_state.get(f"{last}_heads", 4))
        st.session_state[f"{prefix}_aggr"] = st.session_state.get(f"{last}_aggr", "mean")
        st.session_state[f"{prefix}_activation"] = st.session_state.get(f"{last}_activation", "relu")
        st.session_state[f"{prefix}_dropout"] = float(st.session_state.get(f"{last}_dropout", 0.0))
        st.session_state[f"{prefix}_residual"] = bool(st.session_state.get(f"{last}_residual", True))
        st.session_state[f"{prefix}_norm"] = st.session_state.get(f"{last}_norm", "layer_norm")
        st.session_state["gnn_builder_num_layers"] = new_idx
        st.rerun()

    if cols[1].button("+ Head", key="canvas_add_head", use_container_width=True):
        n = int(st.session_state.get("gnn_builder_num_classifier_heads", 1))
        new_idx = n + 1
        prefix = f"gnn_builder_head_{new_idx}"
        st.session_state[f"{prefix}_name"] = f"aux_{new_idx}"
        st.session_state[f"{prefix}_hidden"] = ""
        st.session_state[f"{prefix}_activation"] = "relu"
        st.session_state[f"{prefix}_dropout"] = 0.0
        st.session_state["gnn_builder_num_classifier_heads"] = new_idx
        st.rerun()

    # Botón "Restaurar default" para resetear el state desde la default_architecture.
    if cols[2].button("↺ Reset default", key="canvas_reset_default", use_container_width=True):
        from src.gnn_network_builder import default_architecture
        # Limpieza laxa: borramos las keys numeradas y dejamos que init las repueble.
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and (
                key.startswith("gnn_builder_layer_")
                or key.startswith("gnn_builder_head_")
            ):
                st.session_state.pop(key, None)
        st.session_state.pop("gnn_builder_initialized", None)
        # Reinicializa con default_architecture y marca como listo.
        from src.graph_builder_app import _gnn_builder_arch_to_state
        _gnn_builder_arch_to_state(default_architecture())
        st.rerun()


# ---------------------------------------------------------------------------
# Entrada pública: render del editor con canvas
# ---------------------------------------------------------------------------

def render_canvas_editor(graph_info: Optional[Dict[str, Any]] = None) -> NetworkArchitecture:
    """Render principal del editor visual; devuelve el `NetworkArchitecture`
    actual leído desde `st.session_state`.
    """

    import streamlit as st
    from streamlit_flow import streamlit_flow

    from src.graph_builder_app import (
        _gnn_builder_arch_to_state,
        _gnn_builder_build_architecture_from_state,
    )

    st.markdown("#### Editor (canvas)")

    # Header de metadatos (no representable en el canvas).
    col_a, col_b = st.columns([0.7, 0.3])
    with col_a:
        st.text_input("Nombre", key="gnn_builder_name")
        st.text_area("Descripción", key="gnn_builder_description", height=80)
    with col_b:
        st.checkbox("Favorita", key="gnn_builder_favorite")

    # Construye arch actual desde session_state -> flow_state.
    current_arch = _gnn_builder_build_architecture_from_state()
    flow_state = architecture_to_flow_state(
        current_arch,
        graph_info=graph_info,
        selected_id=_selected_node_id(),
    )

    canvas_col, side_col = st.columns([0.65, 0.35])
    with canvas_col:
        _render_palette()
        new_state = streamlit_flow(
            key="gnn_builder_flow_canvas",
            state=flow_state,
            height=460,
            fit_view=True,
            show_controls=True,
            show_minimap=False,
            allow_new_edges=False,
            animate_new_edges=False,
            get_node_on_click=True,
            get_edge_on_click=False,
            enable_node_menu=True,
            enable_edge_menu=False,
            enable_pane_menu=False,
        )
        if new_state is not None:
            sync_flow_state_to_session(new_state, current_arch=current_arch)
    with side_col:
        # Refresca arch tras posibles eliminaciones del canvas.
        refreshed = _gnn_builder_build_architecture_from_state()
        _render_inspector(refreshed)

    final = _gnn_builder_build_architecture_from_state()
    final.architecture_hash = architecture_hash(final)
    return final


__all__ = [
    "architecture_to_flow_state",
    "render_canvas_editor",
    "sync_flow_state_to_session",
    "CanvasIssue",
]
