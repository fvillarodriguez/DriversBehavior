from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml4roadsafety_validation.config import (  # noqa: E402
        DATA_DIR,
        DEFAULT_MAX_SEGMENTS,
        DEFAULT_MONTHS,
        DEFAULT_SEED,
        DEFAULT_STATE,
        NODE_FEATURE_COLUMNS,
        STATIC_EDGE_FEATURE_KEYS,
        MonthSpec,
        parse_months,
    )
    from ml4roadsafety_validation.download import validate_state_layout  # noqa: E402
    from ml4roadsafety_validation.metrics import split_prevalence  # noqa: E402
else:
    from .config import (
        DATA_DIR,
        DEFAULT_MAX_SEGMENTS,
        DEFAULT_MONTHS,
        DEFAULT_SEED,
        DEFAULT_STATE,
        NODE_FEATURE_COLUMNS,
        STATIC_EDGE_FEATURE_KEYS,
        MonthSpec,
        parse_months,
    )
    from .download import validate_state_layout
    from .metrics import split_prevalence


EDGE_TYPES = (("pm", "spatial", "pm"), ("pm", "temporal", "pm"))


def _torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_aligned_vector(values: torch.Tensor, length: int, *, fill: float = 0.0) -> torch.Tensor:
    values = values.detach().cpu().float().reshape(-1)
    values = torch.nan_to_num(values, nan=float(fill), posinf=float(fill), neginf=float(fill))
    if int(values.numel()) == length:
        return values
    out = torch.full((length,), float(fill), dtype=torch.float32)
    n = min(length, int(values.numel()))
    if n > 0:
        out[:n] = values[:n]
    return out


def _sparse_feature_values(feature_dict: Mapping[str, object], key: str, length: int) -> torch.Tensor:
    raw = feature_dict.get(key)
    if raw is None:
        return torch.zeros(length, dtype=torch.float32)
    tensor = raw if torch.is_tensor(raw) else torch.as_tensor(raw)
    if getattr(tensor, "is_sparse", False):
        tensor = tensor.coalesce().values()
    else:
        tensor = tensor.reshape(-1)
    finite = tensor.detach().cpu().float()
    finite_values = finite[torch.isfinite(finite)]
    fill = float(finite_values.mean().item()) if finite_values.numel() else 0.0
    return _as_aligned_vector(finite, length, fill=fill)


def load_static_network(state_path: Path) -> torch.Tensor:
    adj = _torch_load(state_path / "adj_matrix.pt")
    if not torch.is_tensor(adj):
        raise TypeError("adj_matrix.pt debe contener un tensor sparse de PyTorch.")
    edge_index = adj.coalesce().indices().long()
    if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("adj_matrix.pt no contiene edge_index con shape [2, E].")
    return edge_index


def load_static_edge_features(state_path: Path, edge_count: int) -> torch.Tensor:
    feature_path = state_path / "Edges" / "edge_features.pt"
    if not feature_path.exists():
        return torch.zeros((edge_count, 1), dtype=torch.float32)
    feature_dict = _torch_load(feature_path)
    if not isinstance(feature_dict, Mapping):
        return torch.zeros((edge_count, 1), dtype=torch.float32)
    columns = [
        _sparse_feature_values(feature_dict, key, edge_count)
        for key in STATIC_EDGE_FEATURE_KEYS
    ]
    return torch.stack(columns, dim=1).float()


def load_yearly_edge_traffic(state_path: Path, year: int, edge_count: int) -> torch.Tensor:
    feature_path = state_path / "Edges" / f"edge_features_traffic_{year}.pt"
    if not feature_path.exists():
        return torch.zeros((edge_count, 1), dtype=torch.float32)
    feature_dict = _torch_load(feature_path)
    if not isinstance(feature_dict, Mapping):
        return torch.zeros((edge_count, 1), dtype=torch.float32)
    return _sparse_feature_values(feature_dict, "AADT", edge_count).view(-1, 1)


def load_monthly_node_features(
    state_path: Path,
    month: MonthSpec,
    *,
    num_road_nodes: int,
) -> torch.Tensor:
    path = state_path / "Nodes" / f"node_features_{month.year}_{month.month}.csv"
    if not path.exists():
        return torch.zeros((num_road_nodes, len(NODE_FEATURE_COLUMNS)), dtype=torch.float32)
    df = pd.read_csv(path)
    for column in NODE_FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
    df = df[list(NODE_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.mean(numeric_only=True)).fillna(0.0)
    values = torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)
    if int(values.shape[0]) == num_road_nodes:
        return values
    out = torch.zeros((num_road_nodes, len(NODE_FEATURE_COLUMNS)), dtype=torch.float32)
    n = min(num_road_nodes, int(values.shape[0]))
    if n > 0:
        out[:n] = values[:n]
    return out


def load_monthly_accidents(state_path: Path, month: MonthSpec) -> tuple[torch.Tensor, torch.Tensor]:
    path = state_path / "accidents_monthly.csv"
    if not path.exists():
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    df = pd.read_csv(path)
    required = {"year", "month", "node_1_idx", "node_2_idx", "acc_count"}
    if not required.issubset(df.columns):
        raise ValueError(f"accidents_monthly.csv no contiene columnas requeridas: {sorted(required)}")
    df = df[(df["year"] == month.year) & (df["month"] == month.month)]
    if df.empty:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    pairs = torch.tensor(df[["node_1_idx", "node_2_idx"]].to_numpy(), dtype=torch.long)
    counts = torch.tensor(df["acc_count"].to_numpy(dtype=np.float32), dtype=torch.float32)
    return pairs, counts


def _build_pair_to_segment(edge_index: torch.Tensor) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for idx in range(int(edge_index.shape[1])):
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        mapping.setdefault((src, dst), idx)
    for idx in range(int(edge_index.shape[1])):
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        mapping.setdefault((dst, src), idx)
    return mapping


def _segment_accident_counts(
    pairs: torch.Tensor,
    counts: torch.Tensor,
    pair_to_segment: Mapping[tuple[int, int], int],
) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for idx in range(int(pairs.shape[0])):
        pair = (int(pairs[idx, 0].item()), int(pairs[idx, 1].item()))
        segment_idx = pair_to_segment.get(pair)
        if segment_idx is None:
            continue
        out[int(segment_idx)] += float(counts[idx].item())
    return dict(out)


def _select_segments(
    *,
    edge_count: int,
    positive_segments: Iterable[int],
    max_segments: int | None,
    seed: int,
) -> torch.Tensor:
    positives = sorted({int(v) for v in positive_segments if 0 <= int(v) < edge_count})
    if max_segments is None or int(max_segments) <= 0 or edge_count <= int(max_segments):
        return torch.arange(edge_count, dtype=torch.long)
    max_segments = int(max_segments)
    if len(positives) >= max_segments:
        return torch.tensor(positives, dtype=torch.long)
    rng = np.random.default_rng(seed)
    positive_set = set(positives)
    candidates = np.asarray([idx for idx in range(edge_count) if idx not in positive_set], dtype=np.int64)
    n_extra = min(max_segments - len(positives), int(candidates.size))
    sampled = rng.choice(candidates, size=n_extra, replace=False).tolist() if n_extra else []
    return torch.tensor(sorted(positives + [int(v) for v in sampled]), dtype=torch.long)


def _segment_month_features(
    *,
    edge_index: torch.Tensor,
    selected_segments: torch.Tensor,
    state_path: Path,
    months: Sequence[MonthSpec],
    static_edge_features: torch.Tensor,
    edge_traffic_by_year: Mapping[int, torch.Tensor],
    num_road_nodes: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    endpoints = edge_index[:, selected_segments]
    src = endpoints[0].clamp(0, max(num_road_nodes - 1, 0))
    dst = endpoints[1].clamp(0, max(num_road_nodes - 1, 0))
    static_selected = static_edge_features.index_select(0, selected_segments)
    for month in months:
        node_features = load_monthly_node_features(
            state_path,
            month,
            num_road_nodes=num_road_nodes,
        )
        endpoint_features = (node_features.index_select(0, src) + node_features.index_select(0, dst)) / 2.0
        traffic = edge_traffic_by_year[month.year].index_select(0, selected_segments)
        angle = 2.0 * np.pi * float(month.month) / 12.0
        month_features = torch.tensor(
            [np.sin(angle), np.cos(angle)],
            dtype=torch.float32,
        ).view(1, 2).repeat(int(selected_segments.numel()), 1)
        rows.append(torch.cat([static_selected, traffic, endpoint_features, month_features], dim=1))
    return torch.cat(rows, dim=0).float()


def _normalise_from_train(x: torch.Tensor, train_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if int(train_mask.sum().item()) <= 0:
        raise ValueError("train_mask no contiene nodos.")
    train_x = x[train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (x - mean) / std, mean, std


def _build_masks(n_segments: int, n_months: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    month_index = torch.arange(n_months).repeat_interleave(n_segments)
    train_mask = month_index < max(n_months - 2, 1)
    val_mask = month_index == (n_months - 2)
    test_mask = month_index == (n_months - 1)
    return train_mask, val_mask, test_mask


def _build_labels(
    *,
    selected_segments: torch.Tensor,
    months: Sequence[MonthSpec],
    month_counts: Mapping[str, Mapping[int, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    segment_to_local = {int(seg.item()): idx for idx, seg in enumerate(selected_segments)}
    labels: list[torch.Tensor] = []
    counts_rows: list[torch.Tensor] = []
    for month in months:
        y = torch.zeros(int(selected_segments.numel()), dtype=torch.long)
        counts = torch.zeros(int(selected_segments.numel()), dtype=torch.float32)
        for segment_idx, count in month_counts.get(month.label, {}).items():
            local_idx = segment_to_local.get(int(segment_idx))
            if local_idx is None:
                continue
            counts[local_idx] = float(count)
            y[local_idx] = int(float(count) > 0.0)
        labels.append(y)
        counts_rows.append(counts)
    return torch.cat(labels, dim=0), torch.cat(counts_rows, dim=0)


def _line_graph_base_edges(edge_index: torch.Tensor, selected_segments: torch.Tensor) -> torch.Tensor:
    incident: dict[int, list[int]] = defaultdict(list)
    for local_idx, segment_idx in enumerate(selected_segments.tolist()):
        src = int(edge_index[0, segment_idx].item())
        dst = int(edge_index[1, segment_idx].item())
        incident[src].append(local_idx)
        incident[dst].append(local_idx)

    pairs: set[tuple[int, int]] = set()
    for segment_ids in incident.values():
        unique = sorted(set(segment_ids))
        for src in unique:
            for dst in unique:
                if src != dst:
                    pairs.add((src, dst))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    ordered = sorted(pairs)
    return torch.tensor(ordered, dtype=torch.long).t().contiguous()


def _repeat_month_edges(base_edges: torch.Tensor, n_segments: int, n_months: int) -> torch.Tensor:
    if int(base_edges.numel()) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    chunks = []
    for month_idx in range(n_months):
        chunks.append(base_edges + month_idx * n_segments)
    return torch.cat(chunks, dim=1)


def _temporal_edges(n_segments: int, n_months: int) -> torch.Tensor:
    if n_months < 2:
        return torch.empty((2, 0), dtype=torch.long)
    src_chunks = []
    dst_chunks = []
    local = torch.arange(n_segments, dtype=torch.long)
    for month_idx in range(n_months - 1):
        src_chunks.append(local + month_idx * n_segments)
        dst_chunks.append(local + (month_idx + 1) * n_segments)
    return torch.stack([torch.cat(src_chunks), torch.cat(dst_chunks)], dim=0)


def _edge_attrs_from_x(x: torch.Tensor, edge_index: torch.Tensor, *, kind: str) -> torch.Tensor:
    if int(edge_index.numel()) == 0:
        dim = int(x.shape[1]) * 3 if kind == "spatial" else int(x.shape[1])
        return torch.empty((0, dim), dtype=torch.float32)
    src = edge_index[0]
    dst = edge_index[1]
    if kind == "spatial":
        src_x = x.index_select(0, src)
        dst_x = x.index_select(0, dst)
        return torch.cat([src_x, dst_x, torch.abs(dst_x - src_x)], dim=1).float()
    if kind == "temporal":
        return (x.index_select(0, dst) - x.index_select(0, src)).float()
    raise ValueError(f"Tipo de edge_attr desconocido: {kind}")


def _assert_graph_contract(data: HeteroData) -> None:
    pm = data["pm"]
    n_nodes = int(pm.x.shape[0])
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(pm, mask_name)
        if int(mask.numel()) != n_nodes:
            raise ValueError(f"{mask_name} no coincide con el numero de nodos.")
    combined = pm.train_mask.long() + pm.val_mask.long() + pm.test_mask.long()
    if bool((combined > 1).any().item()):
        raise ValueError("Las mascaras train/val/test se solapan.")
    if int(pm.train_mask.sum().item()) == 0 or int(pm.val_mask.sum().item()) == 0 or int(pm.test_mask.sum().item()) == 0:
        raise ValueError("Cada split debe contener al menos un nodo.")
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr
        if int(edge_index.shape[1]) != int(edge_attr.shape[0]):
            raise ValueError(f"edge_attr desalineado para {edge_type}.")


def build_ml4roadsafety_graph(
    *,
    data_dir: Path = DATA_DIR,
    state: str = DEFAULT_STATE,
    months: Sequence[MonthSpec] | Sequence[str] = DEFAULT_MONTHS,
    max_segments: int | None = DEFAULT_MAX_SEGMENTS,
    seed: int = DEFAULT_SEED,
) -> HeteroData:
    parsed_months = parse_months(tuple(m.label if isinstance(m, MonthSpec) else str(m) for m in months))
    state_path = validate_state_layout(Path(data_dir), state)
    edge_index = load_static_network(state_path)
    edge_count = int(edge_index.shape[1])
    num_road_nodes = int(edge_index.max().item()) + 1 if edge_count else 0
    pair_to_segment = _build_pair_to_segment(edge_index)

    month_counts: dict[str, dict[int, float]] = {}
    positive_segments: set[int] = set()
    for month in parsed_months:
        pairs, counts = load_monthly_accidents(state_path, month)
        segment_counts = _segment_accident_counts(pairs, counts, pair_to_segment)
        month_counts[month.label] = segment_counts
        positive_segments.update(segment_counts.keys())

    selected_segments = _select_segments(
        edge_count=edge_count,
        positive_segments=positive_segments,
        max_segments=max_segments,
        seed=seed,
    )
    n_segments = int(selected_segments.numel())
    if n_segments <= 0:
        raise ValueError("No se seleccionaron segmentos para el grafo.")

    static_edge_features = load_static_edge_features(state_path, edge_count)
    years = sorted({m.year for m in parsed_months})
    edge_traffic_by_year = {
        year: load_yearly_edge_traffic(state_path, year, edge_count)
        for year in years
    }
    raw_x = _segment_month_features(
        edge_index=edge_index,
        selected_segments=selected_segments,
        state_path=state_path,
        months=parsed_months,
        static_edge_features=static_edge_features,
        edge_traffic_by_year=edge_traffic_by_year,
        num_road_nodes=num_road_nodes,
    )
    train_mask, val_mask, test_mask = _build_masks(n_segments, len(parsed_months))
    x, mean, std = _normalise_from_train(raw_x, train_mask)
    y, accident_count = _build_labels(
        selected_segments=selected_segments,
        months=parsed_months,
        month_counts=month_counts,
    )

    base_spatial = _line_graph_base_edges(edge_index, selected_segments)
    spatial_edge_index = _repeat_month_edges(base_spatial, n_segments, len(parsed_months))
    temporal_edge_index = _temporal_edges(n_segments, len(parsed_months))

    data = HeteroData()
    data["pm"].x = x.float()
    data["pm"].y = y.long()
    data["pm"].accident_count = accident_count.float()
    data["pm"].segment_id = selected_segments.repeat(len(parsed_months)).long()
    data["pm"].month_index = torch.arange(len(parsed_months)).repeat_interleave(n_segments).long()
    data["pm"].train_mask = train_mask
    data["pm"].val_mask = val_mask
    data["pm"].test_mask = test_mask
    data[("pm", "spatial", "pm")].edge_index = spatial_edge_index
    data[("pm", "spatial", "pm")].edge_attr = _edge_attrs_from_x(x, spatial_edge_index, kind="spatial")
    data[("pm", "temporal", "pm")].edge_index = temporal_edge_index
    data[("pm", "temporal", "pm")].edge_attr = _edge_attrs_from_x(x, temporal_edge_index, kind="temporal")

    data.ml4rs_metadata = {
        "state": state.upper(),
        "months": [m.label for m in parsed_months],
        "source_edge_count": edge_count,
        "selected_segments": n_segments,
        "max_segments": int(max_segments or 0),
        "seed": int(seed),
        "normalization": {
            "node_feature_dim": int(x.shape[1]),
            "fit_split": "train",
            "fit_month": parsed_months[0].label,
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
        "label": "segment-month accident occurrence",
        "split_contract": {
            "train": [m.label for m in parsed_months[: max(len(parsed_months) - 2, 1)]],
            "val": parsed_months[-2].label,
            "test": parsed_months[-1].label,
        },
    }
    data.edge_attr_dims = {
        str(edge_type): int(data[edge_type].edge_attr.shape[1])
        for edge_type in data.edge_types
    }
    _assert_graph_contract(data)
    return data


def graph_diagnostics(data: HeteroData) -> dict[str, object]:
    return {
        "num_nodes": int(data["pm"].num_nodes),
        "num_features": int(data["pm"].x.shape[1]),
        "num_edges": {
            str(edge_type): int(data[edge_type].edge_index.shape[1])
            for edge_type in data.edge_types
        },
        "edge_attr_dims": {
            str(edge_type): int(data[edge_type].edge_attr.shape[1])
            for edge_type in data.edge_types
        },
        "splits": split_prevalence(data),
        "metadata": getattr(data, "ml4rs_metadata", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el grafo HeteroData ML4RoadSafety.")
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--months", nargs="+", default=list(DEFAULT_MONTHS))
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SEGMENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = build_ml4roadsafety_graph(
        data_dir=args.data_dir,
        state=args.state,
        months=args.months,
        max_segments=args.max_segments,
        seed=args.seed,
    )
    diagnostics = graph_diagnostics(data)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": data, "diagnostics": diagnostics}, args.output)
        print(f"Grafo guardado en {args.output}")


if __name__ == "__main__":
    main()
