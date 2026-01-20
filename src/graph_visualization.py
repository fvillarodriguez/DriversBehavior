from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime
from itertools import cycle
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
import torch
from torch_geometric.data import HeteroData

from src.config import SEED


def _to_cpu(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu()
    return tensor


def _safe_num_nodes(store: object) -> int:
    if hasattr(store, "num_nodes") and store.num_nodes is not None:
        return int(store.num_nodes)
    if hasattr(store, "x") and store.x is not None:
        return int(store.x.shape[0])
    if hasattr(store, "y") and store.y is not None:
        return int(store.y.shape[0])
    return 0


def _build_summary_frames(loaded_obj: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = loaded_obj.get("data")
    feature_cols = loaded_obj.get("feature_cols", [])

    node_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    mask_rows: List[Dict[str, object]] = []

    if not isinstance(data, HeteroData):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for node_type in data.node_types:
        store = data[node_type]
        num_nodes = _safe_num_nodes(store)
        features = 0
        if "x" in store and store.x is not None and store.x.dim() > 1:
            features = int(store.x.shape[1])
        node_rows.append(
            {
                "node_type": node_type,
                "num_nodes": num_nodes,
                "features": features,
                "has_y": "y" in store,
                "has_is_accident_pm": "is_accident_pm" in store,
                "has_is_synthetic": hasattr(store, "is_synthetic"),
            }
        )

        mask_keys = [k for k in store.keys() if k.endswith("_mask") or k == "is_accident_pm"]
        for mask_name in sorted(mask_keys):
            mask = _to_cpu(store[mask_name]) if mask_name in store else None
            if mask is None:
                continue
            true_count = int(mask.sum().item())
            false_count = int(mask.numel() - true_count)
            mask_rows.append(
                {
                    "node_type": node_type,
                    "mask": mask_name,
                    "true": true_count,
                    "false": false_count,
                }
            )

    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        store = data[edge_type]
        num_edges = int(store.num_edges) if hasattr(store, "num_edges") else 0
        if num_edges == 0 and hasattr(store, "edge_index") and store.edge_index is not None:
            num_edges = int(store.edge_index.shape[1])
        edge_attr_dim = None
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            edge_attr_dim = int(store.edge_attr.shape[1]) if store.edge_attr.dim() > 1 else 1
        edge_rows.append(
            {
                "edge_type": f"{src}-{rel}-{dst}",
                "num_edges": num_edges,
                "edge_attr_dim": edge_attr_dim,
                "has_is_synthetic": hasattr(store, "is_synthetic"),
            }
        )

    node_df = pd.DataFrame(node_rows)
    edge_df = pd.DataFrame(edge_rows)
    mask_df = pd.DataFrame(mask_rows)

    if feature_cols:
        node_df.loc[node_df["node_type"] == "pm", "features"] = len(feature_cols)

    return node_df, edge_df, mask_df


def _collect_pm_candidates(data: HeteroData) -> Dict[str, List[int]]:
    candidates: Dict[str, List[int]] = {"accident": [], "synthetic": [], "all": []}
    if "pm" not in data.node_types:
        return candidates

    pm_store = data["pm"]
    pm_count = _safe_num_nodes(pm_store)
    candidates["all"] = list(range(pm_count))

    if hasattr(pm_store, "is_accident_pm"):
        mask = _to_cpu(pm_store.is_accident_pm)
        if mask is not None:
            candidates["accident"] = torch.where(mask)[0].tolist()

    if hasattr(pm_store, "is_synthetic"):
        mask = _to_cpu(pm_store.is_synthetic)
        if mask is not None:
            candidates["synthetic"] = torch.where(mask)[0].tolist()

    return candidates


def _lookup_pm_index(pm_index: object, idx: int) -> Optional[Tuple[str, float]]:
    rev = getattr(pm_index, "_rev", None)
    if rev is None:
        return None
    try:
        value = rev[idx]
    except Exception:
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return str(value[0]), float(value[1])


def _build_subgraph_nodes(
    data: HeteroData,
    start: Tuple[str, int],
    max_hops: int,
) -> set:
    temporal_rev_adj: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
    spatial_bi_adj: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)

    for edge_type in data.edge_types:
        src_t, rel, dst_t = edge_type
        store = data[edge_type]
        edge_index = getattr(store, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            continue
        edge_pairs = _to_cpu(edge_index).t().tolist()
        for src, dst in edge_pairs:
            a = (src_t, int(src))
            b = (dst_t, int(dst))
            if rel == "temporal":
                temporal_rev_adj[b].append(a)
            elif rel in ("spatial", "spatial_back"):
                spatial_bi_adj[a].append(b)
                spatial_bi_adj[b].append(a)

    visited = {start}
    q = deque([(start, 0)])

    while q:
        node, depth = q.popleft()
        for neigh in spatial_bi_adj.get(node, []):
            if neigh not in visited:
                visited.add(neigh)
                q.append((neigh, depth))
        if depth < max_hops:
            for pred in temporal_rev_adj.get(node, []):
                if pred not in visited:
                    visited.add(pred)
                    q.append((pred, depth + 1))

    return visited


def build_subgraph_figure(
    loaded_obj: Dict[str, object],
    start_node_idx: int,
    prev_nodes: int,
    spring_iters: int = 50,
) -> Optional[plt.Figure]:
    if not isinstance(loaded_obj, dict) or "data" not in loaded_obj:
        return None
    data = loaded_obj["data"]
    if not isinstance(data, HeteroData):
        return None

    pm_index = loaded_obj.get("pm_index")
    accident_indices = set()
    pm_syn_mask = None

    if "pm" in data.node_types:
        pm_store = data["pm"]
        if hasattr(pm_store, "is_accident_pm"):
            mask = _to_cpu(pm_store.is_accident_pm)
            if mask is not None:
                accident_indices = set(torch.where(mask)[0].tolist())
        if hasattr(pm_store, "is_synthetic"):
            pm_syn_mask = _to_cpu(pm_store.is_synthetic)

    start = ("pm", int(start_node_idx))
    max_hops = max(0, int(prev_nodes))
    sub_nodes = _build_subgraph_nodes(data, start, max_hops)

    if ("pm", start_node_idx) in sub_nodes:
        incoming_rel_types: List[Tuple[str, str, str]] = []
        if ("pm", "spatial", "pm") in data.edge_types:
            incoming_rel_types.append(("pm", "spatial", "pm"))
        if ("pm", "spatial_back", "pm") in data.edge_types:
            incoming_rel_types.append(("pm", "spatial_back", "pm"))
        for et in incoming_rel_types:
            edge_index = _to_cpu(data[et].edge_index)
            if edge_index is None or edge_index.numel() == 0:
                continue
            dst_eq_start = (edge_index[1] == int(start_node_idx)).nonzero(as_tuple=True)[0].tolist()
            for epos in dst_eq_start:
                src_idx = int(edge_index[0, epos].item())
                sub_nodes.add(("pm", src_idx))

    if len(sub_nodes) < 2:
        return None

    node_color_map = {
        "pm_accident": "salmon",
        "pm": "skyblue",
        "pm_syn": "#ffcc00",
    }
    size_map = {"pm_accident": 300, "pm": 150, "pm_syn": 250}

    palette = list(matplotlib.colormaps["tab10"].colors) + list(matplotlib.colormaps["Set2"].colors)
    node_color_cycle = cycle(palette)

    G = nx.DiGraph()
    custom_labels: Dict[str, str] = {}

    for ntype, idx in sub_nodes:
        node_id = f"{ntype}_{idx}"
        is_syn = (
            pm_syn_mask is not None
            and ntype == "pm"
            and idx < len(pm_syn_mask)
            and bool(pm_syn_mask[idx])
        )
        is_accident_node = idx in accident_indices if ntype == "pm" else False

        if ntype == "pm":
            tag = "pm_syn" if is_syn else ("pm_accident" if is_accident_node else "pm")
        else:
            tag = ntype
            if tag not in node_color_map:
                node_color_map[tag] = matplotlib.colors.to_hex(next(node_color_cycle))

        G.add_node(node_id, tag=tag)

        label_str = ""
        if "y" in data[ntype]:
            y_tensor = _to_cpu(data[ntype].y)
            if y_tensor is not None and idx < len(y_tensor):
                label_val = y_tensor[idx].item()
                label_str = f"\nLabel: {label_val}"

        if ntype == "pm" and pm_index is not None:
            info = _lookup_pm_index(pm_index, idx)
            if info:
                portico, ts_min = info
                dt = datetime.fromtimestamp(ts_min * 60)
                custom_labels[node_id] = f"P:{portico}\n{dt:%d-%m %H:%M}{label_str}"
                continue

        if is_syn:
            custom_labels[node_id] = f"{node_id}\n(Syn){label_str}"
        else:
            custom_labels[node_id] = f"{node_id}{label_str}"

    base_edge_colors = {
        "temporal": "green",
        "spatial": "blue",
        "spatial_back": "#9b59b6",
        "st_fwd": "#e67e22",
    }
    rels_present = [rel for (_, rel, _) in data.edge_types]
    edge_color_map: Dict[str, str] = {}
    for rel in rels_present:
        if rel in base_edge_colors:
            edge_color_map[rel] = base_edge_colors[rel]

    edge_palette = cycle(palette)
    for rel in rels_present:
        if rel not in edge_color_map:
            edge_color_map[rel] = matplotlib.colors.to_hex(next(edge_palette))

    edges_by_type: Dict[str, List[Tuple[str, str]]] = {rel: [] for rel in edge_color_map.keys()}
    syn_edges_by_type: Dict[str, List[Tuple[str, str]]] = {rel: [] for rel in edge_color_map.keys()}

    for edge_type in data.edge_types:
        src_t, rel, dst_t = edge_type
        store = data[edge_type]
        edge_index = _to_cpu(getattr(store, "edge_index", None))
        if edge_index is None or edge_index.numel() == 0:
            continue
        edge_syn_mask = _to_cpu(getattr(store, "is_synthetic", None))
        edge_pairs = edge_index.t().tolist()
        for k, (src, dst) in enumerate(edge_pairs):
            node_u = (src_t, int(src))
            node_v = (dst_t, int(dst))
            if node_u not in sub_nodes or node_v not in sub_nodes:
                continue
            u, v = f"{src_t}_{src}", f"{dst_t}_{dst}"
            G.add_edge(u, v)
            is_syn_edge = False
            if edge_syn_mask is not None and k < len(edge_syn_mask):
                is_syn_edge = bool(edge_syn_mask[k])
            if is_syn_edge:
                syn_edges_by_type[rel].append((u, v))
            else:
                edges_by_type[rel].append((u, v))

    fig = plt.figure(figsize=(18, 14))
    k_scale = 1.5 / math.sqrt(G.number_of_nodes()) if G.number_of_nodes() else 1
    pos = nx.spring_layout(G, iterations=spring_iters, seed=SEED, k=k_scale)

    for tag, color in node_color_map.items():
        node_list = [n for n, d in G.nodes(data=True) if d.get("tag") == tag]
        if not node_list:
            continue
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_list,
            node_color=color,
            node_size=size_map.get(tag, 120),
            label=tag,
        )

    for rel, color in edge_color_map.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges_by_type.get(rel, []),
            edge_color=color,
            arrowstyle="->",
            arrowsize=15,
            width=1.5,
            label=f"{rel} (Real)",
            connectionstyle="arc3,rad=0.1",
        )

    for rel, color in edge_color_map.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=syn_edges_by_type.get(rel, []),
            edge_color=color,
            arrowstyle="->",
            arrowsize=15,
            width=2.0,
            style="dashed",
            label=f"{rel} (Syn)",
            connectionstyle="arc3,rad=0.1",
        )

    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=8, font_color="black")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=f"Node: {t}", markerfacecolor=c, markersize=10)
        for t, c in node_color_map.items()
    ]
    legend_elements += [
        Line2D([0], [0], color=c, lw=2, label=f"Edge Real: {r}")
        for r, c in edge_color_map.items()
    ]
    legend_elements += [
        Line2D([0], [0], color=c, lw=2, linestyle="--", label=f"Edge Syn: {r}")
        for r, c in edge_color_map.items()
    ]

    plt.legend(handles=legend_elements, loc="upper right")
    plt.title(f"Influence subgraph for node PM {start_node_idx}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    return fig


def render_visual_graph_tab(loaded_obj: Optional[Dict[str, object]] = None) -> None:
    st.subheader("Visual Graph")

    if loaded_obj is None:
        loaded_obj = st.session_state.get("loaded_graph")
    if not loaded_obj:
        st.warning("No graph loaded. Create or load a graph in the Graph tab first.")
        return
    data = loaded_obj.get("data")
    if not isinstance(data, HeteroData):
        st.error("Loaded graph is missing valid data.")
        return

    node_df, edge_df, mask_df = _build_summary_frames(loaded_obj)

    with st.expander("Graph overview", expanded=False):
        if not node_df.empty:
            st.markdown("Nodes")
            st.dataframe(node_df, width="stretch")
        if not edge_df.empty:
            st.markdown("Edges")
            st.dataframe(edge_df, width="stretch")
        if not mask_df.empty:
            st.markdown("Masks")
            st.dataframe(mask_df, width="stretch")

        feature_cols = loaded_obj.get("feature_cols", [])
        if feature_cols:
            st.markdown("Feature columns (pm)")
            preview = ", ".join(feature_cols[:50])
            if len(feature_cols) > 50:
                preview += ", ..."
            st.caption(preview)

    st.markdown("---")
    st.markdown("Subgraph visualization")

    candidates = _collect_pm_candidates(data)
    modes = []
    if candidates["accident"]:
        modes.append("Accident PM")
    if candidates["synthetic"]:
        modes.append("Synthetic PM")
    if candidates["all"]:
        modes.append("Any PM")

    if not modes:
        st.warning("No PM nodes available to visualize.")
        return

    mode = st.radio("Node set", modes, horizontal=True)
    if mode == "Accident PM":
        pool = candidates["accident"]
    elif mode == "Synthetic PM":
        pool = candidates["synthetic"]
    else:
        pool = candidates["all"]

    if not pool:
        st.warning("No nodes available for the selected set.")
        return

    selected_idx = None
    if len(pool) <= 200:
        selected_idx = st.selectbox("Node index", pool)
    else:
        pos = st.number_input(
            "Position in list",
            min_value=0,
            max_value=max(len(pool) - 1, 0),
            step=1,
        )
        selected_idx = pool[int(pos)]

    pm_index = loaded_obj.get("pm_index")
    if pm_index is not None:
        info = _lookup_pm_index(pm_index, int(selected_idx))
        if info:
            portico, ts_min = info
            dt = datetime.fromtimestamp(ts_min * 60)
            st.caption(f"PM info: portico={portico}, ts={dt:%Y-%m-%d %H:%M}")

    prev_nodes = st.slider("Temporal hops", min_value=0, max_value=10, value=2, step=1)
    spring_iters = st.slider("Layout iterations", min_value=10, max_value=200, value=50, step=10)

    if st.button("Render subgraph", type="primary"):
        with st.spinner("Rendering subgraph..."):
            fig = build_subgraph_figure(
                loaded_obj=loaded_obj,
                start_node_idx=int(selected_idx),
                prev_nodes=int(prev_nodes),
                spring_iters=int(spring_iters),
            )
        if fig is None:
            st.info("Not enough nodes or edges to render.")
        else:
            st.pyplot(fig, clear_figure=True)
