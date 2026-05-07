"""Tests para el canvas visual del Network Builder.

Cubre la lógica pura (sin Streamlit): conversión `NetworkArchitecture` →
`StreamlitFlowState` y la sincronización inversa cuando el usuario elimina
nodos del canvas.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.gnn_network_builder import (
    NetworkArchitecture,
    NetworkBlock,
    NetworkHead,
    default_architecture,
)
from src.gnn_network_builder_canvas import (
    NODE_STYLES,
    architecture_to_flow_state,
    sync_flow_state_to_session,
)


# ---------------------------------------------------------------------------
# architecture_to_flow_state
# ---------------------------------------------------------------------------

def test_default_architecture_produces_expected_node_topology():
    arch = default_architecture(num_layers=2)
    state = architecture_to_flow_state(
        arch,
        graph_info={
            "in_channels": 59,
            "out_channels": 2,
            "edge_feature_dim": 12,
            "has_sequence_index": True,
        },
    )
    node_ids = [n.id for n in state.nodes]
    assert node_ids == ["input", "layer_1", "layer_2", "temporal", "head_1"]
    edge_pairs = [(e.source, e.target) for e in state.edges]
    assert edge_pairs == [
        ("input", "layer_1"),
        ("layer_1", "layer_2"),
        ("layer_2", "temporal"),
        ("temporal", "head_1"),
    ]


def test_canvas_supports_multiple_layers_and_heads():
    arch = NetworkArchitecture(
        name="multi",
        blocks=[
            NetworkBlock(block_type="hetero_conv", name="conv_1", hidden_channels=32, num_heads=2),
            NetworkBlock(block_type="hetero_conv", name="conv_2", hidden_channels=64, num_heads=4),
            NetworkBlock(block_type="hetero_conv", name="conv_3", hidden_channels=128, num_heads=8),
            NetworkBlock(block_type="temporal_head", name="temporal", temporal_type="gru"),
        ],
        heads=[
            NetworkHead(name="primary", primary=True),
            NetworkHead(name="aux_2", primary=False),
            NetworkHead(name="aux_3", primary=False),
        ],
    )
    state = architecture_to_flow_state(arch)
    node_ids = [n.id for n in state.nodes]
    assert node_ids == ["input", "layer_1", "layer_2", "layer_3", "temporal", "head_1", "head_2", "head_3"]
    # Cada head sale del temporal head (backbone tail).
    head_targets = [(e.source, e.target) for e in state.edges if e.target.startswith("head_")]
    assert head_targets == [("temporal", "head_1"), ("temporal", "head_2"), ("temporal", "head_3")]


def test_canvas_without_temporal_head_routes_heads_from_last_layer():
    arch = NetworkArchitecture(
        name="snapshot-only",
        blocks=[
            NetworkBlock(block_type="hetero_conv", name="conv_1", hidden_channels=32, num_heads=2),
            NetworkBlock(block_type="hetero_conv", name="conv_2", hidden_channels=64, num_heads=4),
        ],
        heads=[NetworkHead(name="primary", primary=True)],
    )
    state = architecture_to_flow_state(arch)
    edges = [(e.source, e.target) for e in state.edges]
    assert ("layer_2", "head_1") in edges  # va directo al head sin pasar por temporal
    assert all(e.target != "temporal" for e in state.edges)


def test_node_styles_have_distinct_colors():
    # Sanity: cuatro tipos visualmente distinguibles.
    colors = {kind: spec["backgroundColor"] for kind, spec in NODE_STYLES.items()}
    assert len(set(colors.values())) == 4


# ---------------------------------------------------------------------------
# sync_flow_state_to_session — eliminación de nodos
# ---------------------------------------------------------------------------

def _make_session_state(num_layers: int, num_heads: int = 1) -> dict:
    """Simula `st.session_state` con keys numeradas para `num_layers` capas."""
    state = {
        "gnn_builder_num_layers": num_layers,
        "gnn_builder_num_classifier_heads": num_heads,
        "gnn_builder_primary_head_idx": 0,
        "gnn_builder_temporal_type": "gru",
        "gnn_builder_name": "demo",
        "gnn_builder_description": "",
        "gnn_builder_favorite": False,
    }
    for i in range(1, num_layers + 1):
        prefix = f"gnn_builder_layer_{i}"
        state[f"{prefix}_conv"] = "GATConv"
        state[f"{prefix}_hidden"] = 32 * i
        state[f"{prefix}_heads"] = i
        state[f"{prefix}_aggr"] = "mean"
        state[f"{prefix}_activation"] = "relu"
        state[f"{prefix}_dropout"] = 0.1
        state[f"{prefix}_residual"] = True
        state[f"{prefix}_norm"] = "layer_norm"
    for i in range(1, num_heads + 1):
        prefix = f"gnn_builder_head_{i}"
        state[f"{prefix}_name"] = f"head_{i}"
        state[f"{prefix}_hidden"] = ""
        state[f"{prefix}_activation"] = "relu"
        state[f"{prefix}_dropout"] = 0.0
    return state


def _flow_state_with_node_ids(node_ids):
    state = MagicMock()
    state.nodes = [MagicMock(id=nid) for nid in node_ids]
    state.selected_id = None
    return state


def test_sync_compacts_layer_indices_when_middle_layer_removed(monkeypatch):
    import streamlit as st

    fake_state = _make_session_state(num_layers=3, num_heads=1)
    monkeypatch.setattr(st, "session_state", fake_state)

    flow = _flow_state_with_node_ids(["input", "layer_1", "layer_3", "temporal", "head_1"])
    sync_flow_state_to_session(flow, current_arch=default_architecture())

    assert fake_state["gnn_builder_num_layers"] == 2
    # Layer 1 mantiene su config original (hidden=32).
    assert fake_state["gnn_builder_layer_1_hidden"] == 32
    # Lo que era layer_3 (hidden=96) ahora es layer_2.
    assert fake_state["gnn_builder_layer_2_hidden"] == 96
    # Las keys huérfanas se han eliminado.
    assert "gnn_builder_layer_3_hidden" not in fake_state


def test_sync_compacts_head_indices_when_middle_head_removed(monkeypatch):
    import streamlit as st

    fake_state = _make_session_state(num_layers=2, num_heads=3)
    fake_state["gnn_builder_head_1_name"] = "primary"
    fake_state["gnn_builder_head_2_name"] = "aux_2"
    fake_state["gnn_builder_head_3_name"] = "aux_3"
    fake_state["gnn_builder_primary_head_idx"] = 2  # primary apunta a head_3
    monkeypatch.setattr(st, "session_state", fake_state)

    flow = _flow_state_with_node_ids(["input", "layer_1", "layer_2", "temporal", "head_1", "head_3"])
    sync_flow_state_to_session(flow, current_arch=default_architecture())

    assert fake_state["gnn_builder_num_classifier_heads"] == 2
    # Head 3 sobrevive y se renumera a head_2.
    assert fake_state["gnn_builder_head_2_name"] == "aux_3"
    assert "gnn_builder_head_3_name" not in fake_state
    # primary_head_idx queda dentro del rango (era 2 ⇒ ahora se resetea a 0).
    assert fake_state["gnn_builder_primary_head_idx"] in (0, 1)


def test_sync_preserves_state_when_no_changes(monkeypatch):
    import streamlit as st

    fake_state = _make_session_state(num_layers=2, num_heads=1)
    snapshot = dict(fake_state)
    monkeypatch.setattr(st, "session_state", fake_state)

    flow = _flow_state_with_node_ids(["input", "layer_1", "layer_2", "temporal", "head_1"])
    sync_flow_state_to_session(flow, current_arch=default_architecture())

    # No se debió tocar ninguna key.
    assert fake_state == snapshot


def test_sync_persists_selected_id(monkeypatch):
    import streamlit as st

    fake_state = _make_session_state(num_layers=2, num_heads=1)
    monkeypatch.setattr(st, "session_state", fake_state)

    flow = _flow_state_with_node_ids(["input", "layer_1", "layer_2", "temporal", "head_1"])
    flow.selected_id = "layer_2"
    sync_flow_state_to_session(flow, current_arch=default_architecture())

    assert fake_state.get("gnn_builder_canvas_selected") == "layer_2"
