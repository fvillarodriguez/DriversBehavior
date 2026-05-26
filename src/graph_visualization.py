from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime
from itertools import cycle
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch_geometric.data import HeteroData

from src.config import RESULTADOS_DIR, SEED
from src.gnn_artifacts import gnn_path

HISTORY_PATH = gnn_path("root", "gnn_history.jsonl", resultados_dir=RESULTADOS_DIR)


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


def _load_history_entries() -> List[Dict[str, object]]:
    if not HISTORY_PATH.exists():
        return []
    entries: List[Dict[str, object]] = []
    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
    except Exception:
        return []
    return entries


def _find_graph_build_log(loaded_obj: Dict[str, object]) -> Tuple[Optional[Dict[str, object]], str]:
    entries = _load_history_entries()
    if not entries:
        return None, "none"
    filename = loaded_obj.get("filename")
    match = None
    if filename:
        for entry in reversed(entries):
            if entry.get("type") != "Graph Build":
                continue
            graph_meta = entry.get("graph_build", {})
            if graph_meta.get("filename") == filename:
                match = entry
                break
    if match:
        return match, "filename"
    for entry in reversed(entries):
        if entry.get("type") == "Graph Build":
            return entry, "latest"
    return None, "none"


def _summarize_pm_index(pm_index: object) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    rev = getattr(pm_index, "_rev", None)
    if rev is None:
        return summary
    values: Iterable[object]
    if isinstance(rev, dict):
        values = rev.values()
    else:
        values = rev
    porticos: List[str] = []
    ts_vals: List[float] = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        portico = str(item[0])
        try:
            ts_min = float(item[1])
        except Exception:
            continue
        porticos.append(portico)
        ts_vals.append(ts_min)
    if not porticos or not ts_vals:
        return summary
    unique_ports = sorted(set(porticos))
    summary["portico_count"] = len(unique_ports)
    summary["porticos_preview"] = unique_ports[:20]
    summary["portico_min"] = unique_ports[0]
    summary["portico_max"] = unique_ports[-1]
    ts_min_val = min(ts_vals)
    ts_max_val = max(ts_vals)
    summary["ts_min"] = ts_min_val
    summary["ts_max"] = ts_max_val
    summary["date_min"] = datetime.fromtimestamp(ts_min_val * 60)
    summary["date_max"] = datetime.fromtimestamp(ts_max_val * 60)
    return summary


def _feature_names_for(node_type: str, dim: int, feature_cols: List[str]) -> List[str]:
    if node_type == "pm" and feature_cols and len(feature_cols) == dim:
        return list(feature_cols)
    return [f"f{i}" for i in range(dim)]


def _tensor_feature_stats(
    tensor: Optional[torch.Tensor],
    feature_names: List[str],
    *,
    max_rows: int,
    max_features: int,
) -> pd.DataFrame:
    if tensor is None:
        return pd.DataFrame()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.numel() == 0:
        return pd.DataFrame()
    rows = int(tensor.shape[0])
    if rows > max_rows:
        idx = torch.randint(0, rows, (max_rows,), device=tensor.device)
        tensor = tensor.index_select(0, idx)
    tensor = tensor.detach().float()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    dim = int(tensor.shape[1])
    if dim == 0:
        return pd.DataFrame()
    if max_features > 0 and dim > max_features:
        sel = list(range(max_features))
    else:
        sel = list(range(dim))
    names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in sel]
    mean = tensor[:, sel].mean(0).tolist()
    std = tensor[:, sel].std(0, unbiased=False).tolist()
    minv = tensor[:, sel].min(0).values.tolist()
    maxv = tensor[:, sel].max(0).values.tolist()
    return pd.DataFrame(
        {
            "feature": names,
            "mean": mean,
            "std": std,
            "min": minv,
            "max": maxv,
        }
    )


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


def _resolve_visual_graph_hash(loaded_obj: Dict[str, object]) -> Optional[str]:
    for key in ("graph_hash", "hash"):
        value = loaded_obj.get(key)
        if value:
            return str(value)
    metadata = loaded_obj.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("graph_hash") or metadata.get("hash")
        if value:
            return str(value)
    data = loaded_obj.get("data")
    for attr in ("graph_hash", "hash"):
        value = getattr(data, attr, None)
        if value:
            return str(value)
    return None


def _xai_manifest_compatible(manifest: Dict[str, object], loaded_obj: Dict[str, object]) -> bool:
    graph_hash = _resolve_visual_graph_hash(loaded_obj)
    result_hash = manifest.get("graph_hash")
    if graph_hash and result_hash and str(graph_hash) != str(result_hash):
        return False
    current_model = st.session_state.get("gnn_eval_model_path")
    result_model = manifest.get("model_path")
    if current_model and result_model and str(current_model) != str(result_model):
        return False
    return True


def _load_xai_result_for_visual_graph(loaded_obj: Dict[str, object]) -> Tuple[Optional[object], str]:
    session_result = st.session_state.get("gnn_xai_last_result")
    if session_result is not None and hasattr(session_result, "edges_df") and hasattr(session_result, "nodes_df"):
        if _xai_manifest_compatible(getattr(session_result, "manifest", {}) or {}, loaded_obj):
            return session_result, "sesión actual"
    try:
        from src.gnn_xai import find_latest_gnn_xai_result, load_gnn_xai_result

        result_dir = find_latest_gnn_xai_result(
            graph_hash=_resolve_visual_graph_hash(loaded_obj),
            model_path=st.session_state.get("gnn_eval_model_path"),
        )
        if result_dir is None:
            result_dir = find_latest_gnn_xai_result()
        if result_dir is None:
            return None, ""
        return load_gnn_xai_result(result_dir), str(result_dir)
    except Exception as exc:
        st.caption(f"No se pudo cargar XAI guardado: {exc}")
        return None, ""


def _filter_xai_edges(
    edges_df: pd.DataFrame,
    *,
    relation: str,
    layer: str,
    attention_percentile: float,
    error_type: str,
    metric: str,
) -> pd.DataFrame:
    if edges_df is None or edges_df.empty:
        return pd.DataFrame()
    out = edges_df.copy()
    if relation and relation != "Todas" and "relation" in out.columns:
        out = out[out["relation"].astype(str) == str(relation)]
    if layer and layer != "Todas" and "layer" in out.columns:
        out = out[out["layer"].astype(str) == str(layer)]
    if error_type and error_type != "Todos":
        src = out.get("source_error_type")
        dst = out.get("dest_error_type")
        if src is not None and dst is not None:
            out = out[(src.astype(str) == error_type) | (dst.astype(str) == error_type)]
    att_col = metric if metric in {"mean_attention", "max_attention"} else "mean_attention"
    if att_col in out.columns and not out.empty:
        threshold = out[att_col].quantile(float(attention_percentile))
        out = out[out[att_col] >= threshold]
    return out.reset_index(drop=True)


def _xai_node_color_values(nodes_df: pd.DataFrame, metric: str) -> Tuple[Dict[int, object], Dict[str, str]]:
    if nodes_df is None or nodes_df.empty:
        return {}, {}
    lookup: Dict[int, object] = {}
    palette = {
        "TP": "#2e7d32",
        "TN": "#607d8b",
        "FP": "#f57c00",
        "FN": "#c62828",
        "unknown": "#9e9e9e",
    }
    if metric == "error_type":
        for _, row in nodes_df.iterrows():
            lookup[int(row["node_idx"])] = str(row.get("error_type", "unknown"))
        return lookup, palette
    for _, row in nodes_df.iterrows():
        try:
            lookup[int(row["node_idx"])] = float(row.get("prob1", np.nan))
        except Exception:
            lookup[int(row["node_idx"])] = np.nan
    return lookup, palette


def build_xai_graph_figure(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    metric: str,
    max_edges: int,
    spring_iters: int = 80,
) -> Optional[plt.Figure]:
    if edges_df is None or edges_df.empty:
        return None
    sort_col = metric if metric in edges_df.columns and metric in {"mean_attention", "max_attention"} else "mean_attention"
    plot_edges = edges_df.sort_values(sort_col, ascending=False).head(max(1, int(max_edges))).copy()
    if plot_edges.empty:
        return None

    graph = nx.DiGraph()
    for _, row in plot_edges.iterrows():
        src = int(row["source_node_idx"])
        dst = int(row["dest_node_idx"])
        graph.add_node(src)
        graph.add_node(dst)
        graph.add_edge(
            src,
            dst,
            relation=str(row.get("relation", "")),
            mean_attention=float(row.get("mean_attention", 0.0) or 0.0),
            max_attention=float(row.get("max_attention", 0.0) or 0.0),
        )
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None

    pos = nx.spring_layout(graph, seed=SEED, iterations=int(spring_iters), weight="mean_attention")
    fig, ax = plt.subplots(figsize=(12, 8))
    color_values, palette = _xai_node_color_values(nodes_df, metric)
    node_ids = list(graph.nodes())
    if metric == "error_type":
        node_colors = [palette.get(str(color_values.get(node, "unknown")), "#9e9e9e") for node in node_ids]
    else:
        numeric = np.array(
            [
                float(color_values.get(node, np.nan))
                if color_values.get(node, np.nan) is not None
                else np.nan
                for node in node_ids
            ]
        )
        node_colors = np.nan_to_num(numeric, nan=0.0)

    edge_weights = np.array(
        [float(graph.edges[edge].get("mean_attention", 0.0)) for edge in graph.edges()]
    )
    if edge_weights.size == 0:
        edge_weights = np.array([1.0])
    denom = max(float(edge_weights.max() - edge_weights.min()), 1e-9)
    widths = 1.0 + 5.0 * ((edge_weights - edge_weights.min()) / denom)

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_weights,
        edge_cmap=plt.cm.magma,
        width=widths,
        arrows=True,
        arrowsize=14,
        alpha=0.82,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        cmap=plt.cm.viridis if metric != "error_type" else None,
        node_size=360,
        linewidths=0.8,
        edgecolors="#263238",
        alpha=0.95,
    )
    if graph.number_of_nodes() <= 80:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

    if metric == "error_type":
        handles = [
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=color,
                markersize=8,
            )
            for label, color in palette.items()
            if label != "unknown" or any(color_values.get(node) == "unknown" for node in node_ids)
        ]
        ax.legend(handles=handles, loc="best", frameon=True)
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="prob1")
    edge_sm = plt.cm.ScalarMappable(
        cmap=plt.cm.magma,
        norm=plt.Normalize(vmin=float(edge_weights.min()), vmax=float(edge_weights.max())),
    )
    edge_sm.set_array([])
    fig.colorbar(edge_sm, ax=ax, fraction=0.03, pad=0.08, label="mean_attention")
    ax.set_title("XAI GNN: atención de aristas + predicción por nodo")
    ax.axis("off")
    plt.tight_layout()
    return fig


def _render_xai_explanations(explanations_df: pd.DataFrame, max_rows: int) -> None:
    if explanations_df is None or explanations_df.empty:
        return
    st.markdown("**Explicaciones por perturbacion**")
    st.caption(
        "delta_prob1 mide el cambio de riesgo al intervenir relaciones o features. "
        "Estas mediciones auditan fidelidad del modelo; no prueban causalidad fisica por si solas."
    )
    sort_col = "score" if "score" in explanations_df.columns else "delta_prob1"
    view = explanations_df.sort_values(sort_col, ascending=False).head(max(1, int(max_rows)))
    st.dataframe(view, width="stretch")
    group_cols = [col for col in ("method", "relation", "feature_name", "error_type") if col in explanations_df.columns]
    if group_cols and "score" in explanations_df.columns:
        summary = (
            explanations_df.groupby(group_cols, dropna=False)
            .agg(
                explanation_count=("score", "count"),
                mean_score=("score", "mean"),
                mean_delta_prob1=("delta_prob1", "mean"),
                crossed_threshold=("crossed_threshold", "sum")
                if "crossed_threshold" in explanations_df.columns
                else ("score", "count"),
            )
            .reset_index()
            .sort_values("mean_score", ascending=False)
        )
        st.markdown("**Resumen de perturbaciones**")
        st.dataframe(summary, width="stretch")


def _render_xai_visual_graph_mode(loaded_obj: Dict[str, object]) -> None:
    result, source = _load_xai_result_for_visual_graph(loaded_obj)
    if result is None:
        st.info("No hay resultados XAI guardados para visualizar. Ejecute Evaluación Modelo con XAI activado.")
        return
    nodes_df = getattr(result, "nodes_df", pd.DataFrame())
    edges_df = getattr(result, "edges_df", pd.DataFrame())
    summary_df = getattr(result, "summary_df", pd.DataFrame())
    explanations_df = getattr(result, "explanations_df", pd.DataFrame())
    manifest = getattr(result, "manifest", {}) or {}
    st.caption(
        f"Resultado XAI: {source} | mask={manifest.get('mask_name', 'N/A')} "
        f"| nodos={len(nodes_df):,} | aristas={len(edges_df):,} "
        f"| explicaciones={len(explanations_df):,}"
    )
    if edges_df.empty:
        st.warning("El resultado XAI no contiene aristas relevantes.")
        _render_xai_explanations(explanations_df, max_rows=500)
        return

    relations = (
        ["Todas"] + sorted(edges_df["relation"].dropna().astype(str).unique().tolist())
        if "relation" in edges_df
        else ["Todas"]
    )
    layers = (
        ["Todas"] + sorted(edges_df["layer"].dropna().astype(str).unique().tolist())
        if "layer" in edges_df
        else ["Todas"]
    )
    errors = ["Todos"]
    for col in ("source_error_type", "dest_error_type"):
        if col in edges_df:
            errors.extend(edges_df[col].dropna().astype(str).unique().tolist())
    errors = ["Todos"] + sorted({e for e in errors if e != "Todos"})

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        relation = st.selectbox(
            "Relación",
            relations,
            key="viz_xai_relation",
            help="Filtra aristas XAI por tipo de relación del grafo. Usar 'Todas' evita sesgar el análisis a una sola fuente.",
        )
    with col_b:
        layer = st.selectbox(
            "Capa",
            layers,
            key="viz_xai_layer",
            help="Filtra por capa GNN ya calculada. Cambiarla no recalcula XAI; solo inspecciona capas guardadas.",
        )
    with col_c:
        attention_percentile = st.slider(
            "Percentil visual",
            min_value=0.0,
            max_value=0.99,
            value=0.0,
            step=0.01,
            key="viz_xai_percentile",
            help="Filtro adicional sobre las aristas ya calculadas. 0 muestra todo; valores altos conservan solo atención extrema.",
        )
    with col_d:
        error_type = st.selectbox(
            "Tipo de error",
            errors,
            key="viz_xai_error_type",
            help="Muestra aristas conectadas a nodos TP/TN/FP/FN. Útil para auditoría; no debe leerse como causalidad.",
        )

    col_e, col_f, col_g = st.columns(3)
    with col_e:
        metric = st.selectbox(
            "Métrica visual",
            ["mean_attention", "max_attention", "prob1", "error_type"],
            key="viz_xai_metric",
            help="Controla color de nodos y orden de aristas. Atención escala aristas; prob1/error_type colorea nodos.",
        )
    with col_f:
        max_edges = st.number_input(
            "Máximo aristas",
            min_value=10,
            max_value=20000,
            value=min(500, max(10, int(len(edges_df)))),
            step=50,
            key="viz_xai_max_edges",
            help="Limita el número de aristas dibujadas y listadas. Valores altos pueden saturar el layout y ocultar patrones.",
        )
    with col_g:
        spring_iters = st.slider(
            "Layout iterations",
            min_value=20,
            max_value=250,
            value=80,
            step=10,
            key="viz_xai_spring_iters",
            help="Iteraciones del layout de red. Más iteraciones ordenan mejor grafos pequeños, pero pueden ralentizar grafos densos.",
        )

    filtered = _filter_xai_edges(
        edges_df,
        relation=str(relation),
        layer=str(layer),
        attention_percentile=float(attention_percentile),
        error_type=str(error_type),
        metric=str(metric),
    )
    if filtered.empty:
        st.info("Sin aristas luego de aplicar filtros.")
        return
    fig = build_xai_graph_figure(
        nodes_df,
        filtered,
        metric=str(metric),
        max_edges=int(max_edges),
        spring_iters=int(spring_iters),
    )
    if fig is not None:
        st.pyplot(fig, clear_figure=True)

    st.markdown("**Aristas relevantes**")
    sort_col = str(metric) if str(metric) in filtered.columns else "mean_attention"
    st.dataframe(
        filtered.sort_values(sort_col, ascending=False).head(int(max_edges)),
        width="stretch",
    )
    if not summary_df.empty:
        st.markdown("**Resumen por relación**")
        st.dataframe(summary_df, width="stretch")
    _render_xai_explanations(explanations_df, max_rows=int(max_edges))


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
    feature_cols = loaded_obj.get("feature_cols", [])

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

        if feature_cols:
            st.markdown("Feature columns (pm)")
            preview = ", ".join(feature_cols[:50])
            if len(feature_cols) > 50:
                preview += ", ..."
            st.caption(preview)

    pm_index = loaded_obj.get("pm_index")
    pm_summary = _summarize_pm_index(pm_index) if pm_index is not None else {}
    log_entry, log_match = _find_graph_build_log(loaded_obj)

    with st.expander("Metadata (fechas, porticos y logs)", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Desde el grafo**")
            if pm_summary:
                st.write(
                    f"Rango fechas: {pm_summary['date_min']:%Y-%m-%d %H:%M} -> "
                    f"{pm_summary['date_max']:%Y-%m-%d %H:%M}"
                )
                st.write(
                    f"Porticos unicos: {pm_summary['portico_count']} "
                    f"(min={pm_summary['portico_min']}, max={pm_summary['portico_max']})"
                )
                preview_ports = ", ".join(pm_summary.get("porticos_preview", []))
                if preview_ports:
                    st.caption(f"Ejemplo porticos: {preview_ports}")
            else:
                st.info("No hay informacion de porticos/fechas en pm_index.")
        with col_b:
            st.markdown("**Desde logs (History)**")
            if log_entry:
                graph_meta = log_entry.get("graph_build", {})
                date_min = graph_meta.get("date_min")
                date_max = graph_meta.get("date_max")
                if date_min and date_max:
                    st.write(f"Rango fechas (log): {date_min} -> {date_max}")
                port_count = graph_meta.get("tramo_porticos_count")
                if port_count is not None:
                    st.write(f"Porticos en tramo (log): {port_count}")
                temporal_filter = graph_meta.get("temporal_filter")
                if temporal_filter:
                    st.write(f"Filtro temporal: {temporal_filter}")
                selected_vars = graph_meta.get("selected_vars", [])
                if selected_vars:
                    preview = ", ".join(selected_vars[:30])
                    if len(selected_vars) > 30:
                        preview += ", ..."
                    st.caption(f"Features seleccionadas: {preview}")
                edges_cfg = graph_meta.get("edges", {})
                if edges_cfg:
                    st.caption(
                        "Edges: "
                        + ", ".join([k for k, v in edges_cfg.items() if v and k != "physical_features"])
                    )
                    phys = edges_cfg.get("physical_features", [])
                    if phys:
                        st.caption(f"Features fisicas (edges): {', '.join(phys)}")
                    delta_feats = edges_cfg.get("delta_features", [])
                    if delta_feats:
                        st.caption(f"Features delta (edges): {', '.join(delta_feats)}")
                if log_match != "filename":
                    st.caption("Log asociado por ultimo build (no se encontro filename exacto).")
            else:
                st.info("No hay logs de Graph Build en History.")

    with st.expander("Detalle de features por tipo (nodos y aristas)", expanded=False):
        tab_nodes, tab_edges = st.tabs(["Nodos", "Aristas"])
        with tab_nodes:
            node_types = list(data.node_types)
            if not node_types:
                st.info("No hay tipos de nodos.")
            else:
                ntype = st.selectbox("Tipo de nodo", node_types, key="viz_node_type")
                store = data[ntype]
                x = getattr(store, "x", None)
                if x is None:
                    st.info("Este tipo de nodo no tiene features.")
                else:
                    dim_x = int(x.shape[1]) if x.dim() > 1 else 1
                    min_feats = 1 if dim_x < 5 else 5
                    max_rows = st.number_input(
                        "Muestras para estadistica",
                        min_value=1000,
                        max_value=200000,
                        value=50000,
                        step=5000,
                        key="viz_node_stats_rows",
                    )
                    max_feats = st.number_input(
                        "Maximo de features a mostrar",
                        min_value=min_feats,
                        max_value=max(1, min(200, dim_x)),
                        value=min(40, dim_x),
                        step=1,
                        key="viz_node_stats_feats",
                    )
                    if st.button("Calcular stats (nodos)", key="viz_node_stats_btn"):
                        names = _feature_names_for(ntype, dim_x, feature_cols)
                        df_stats = _tensor_feature_stats(
                            x,
                            names,
                            max_rows=int(max_rows),
                            max_features=int(max_feats),
                        )
                        if df_stats.empty:
                            st.info("Sin stats disponibles.")
                        else:
                            st.dataframe(df_stats, width="stretch")
        with tab_edges:
            edge_types = list(data.edge_types)
            if not edge_types:
                st.info("No hay tipos de aristas.")
            else:
                edge_labels = [f"{s}-{r}-{d}" for (s, r, d) in edge_types]
                sel = st.selectbox("Tipo de arista", list(range(len(edge_types))), format_func=lambda i: edge_labels[i], key="viz_edge_type")
                etype = edge_types[int(sel)]
                store = data[etype]
                edge_attr = getattr(store, "edge_attr", None)
                if edge_attr is None:
                    st.info("Este tipo de arista no tiene edge_attr.")
                else:
                    dim_e = int(edge_attr.shape[1]) if edge_attr.dim() > 1 else 1
                    min_feats = 1 if dim_e < 5 else 5
                    max_rows = st.number_input(
                        "Muestras para estadistica (aristas)",
                        min_value=1000,
                        max_value=200000,
                        value=50000,
                        step=5000,
                        key="viz_edge_stats_rows",
                    )
                    max_feats = st.number_input(
                        "Maximo de features a mostrar (aristas)",
                        min_value=min_feats,
                        max_value=max(1, min(200, dim_e)),
                        value=min(40, dim_e),
                        step=1,
                        key="viz_edge_stats_feats",
                    )
                    if st.button("Calcular stats (aristas)", key="viz_edge_stats_btn"):
                        names = [f"edge_f{i}" for i in range(dim_e)]
                        df_stats = _tensor_feature_stats(
                            edge_attr,
                            names,
                            max_rows=int(max_rows),
                            max_features=int(max_feats),
                        )
                        if df_stats.empty:
                            st.info("Sin stats disponibles.")
                        else:
                            st.dataframe(df_stats, width="stretch")

    st.markdown("---")
    visual_mode = st.radio(
        "Modo Visual Graph",
        ["Subgraph", "XAI"],
        horizontal=True,
        key="viz_graph_mode",
        help="Subgraph inspecciona conectividad local del grafo; XAI carga atención y predicciones guardadas tras Evaluación Modelo.",
    )
    if visual_mode == "XAI":
        _render_xai_visual_graph_mode(loaded_obj)
        return

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

    if pm_index is not None:
        info = _lookup_pm_index(pm_index, int(selected_idx))
        if info:
            portico, ts_min = info
            dt = datetime.fromtimestamp(ts_min * 60)
            st.caption(f"PM info: portico={portico}, ts={dt:%Y-%m-%d %H:%M}")

    pm_store = data["pm"] if "pm" in data.node_types else None
    if pm_store is not None and hasattr(pm_store, "x"):
        if st.checkbox("Mostrar features del nodo seleccionado", value=False, key="viz_node_detail_toggle"):
            x = pm_store.x
            if x is not None and int(selected_idx) < x.shape[0]:
                names = _feature_names_for("pm", int(x.shape[1]), feature_cols)
                x_row = x[int(selected_idx)].detach().cpu()
                df_node = pd.DataFrame(
                    {
                        "feature": names,
                        "value": x_row.tolist(),
                    }
                )
                st.dataframe(df_node, width="stretch")
            else:
                st.info("No hay features disponibles para este nodo.")

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
