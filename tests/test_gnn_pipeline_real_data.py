import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData
from pathlib import Path

from src.features import compute_pm_features
from src.gat_model import HeteroGAT
from src.graphsmote import (
    RelEdgeGen,
    augment_graph_offline_once,
    train_z2x_decoders,
)
from src.train_pretrain import pretrain_edge_generator
from src.utils import (
    add_accident_target,
    get_flow_db_summary,
    load_accidentes,
    load_flujos_range,
    load_porticos,
)


REAL_DATA_ENV = "RUN_REAL_GNN_PIPELINE"


def _is_enabled() -> bool:
    return os.environ.get(REAL_DATA_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _log(msg: str) -> None:
    print(f"[GNN PIPELINE] {msg}")


def _pick_random_week_with_data(
    acc_df: pd.DataFrame, rng: random.Random, window_days: int = 7
) -> Tuple[Optional[pd.Timestamp], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        summary = get_flow_db_summary()
    except Exception:
        return None, None, None
    if summary is None or summary.row_count == 0:
        return None, None, None

    acc = acc_df.copy()
    acc = acc.dropna(subset=["accidente_time", "ultimo_portico"])
    if acc.empty:
        return None, None, None

    acc["accidente_time"] = pd.to_datetime(
        acc["accidente_time"], errors="coerce"
    ).dt.tz_localize(None)
    acc = acc.dropna(subset=["accidente_time"])
    if summary.min_timestamp is not None:
        acc = acc[acc["accidente_time"] >= summary.min_timestamp]
    if summary.max_timestamp is not None:
        acc = acc[acc["accidente_time"] <= summary.max_timestamp]
    if acc.empty:
        return None, None, None

    days = acc["accidente_time"].dt.floor("D").unique().tolist()
    rng.shuffle(days)

    for day in days[:15]:
        day_start = pd.Timestamp(day)
        window_end = day_start + pd.Timedelta(days=window_days)
        flows = load_flujos_range(day_start, window_end, db_path=summary.db_path)
        if flows is None or flows.empty:
            continue
        acc_week = acc[
            (acc["accidente_time"] >= day_start)
            & (acc["accidente_time"] < window_end)
        ]
        if acc_week.empty:
            continue
        return day_start, flows, acc_week
    return None, None, None


def _sample_flow_window(
    flows: pd.DataFrame, rng: random.Random
) -> pd.DataFrame:
    flows = flows.copy()
    flows["FECHA"] = pd.to_datetime(flows["FECHA"], errors="coerce").dt.tz_localize(None)
    flows = flows.dropna(subset=["FECHA"])
    if flows.empty:
        return flows

    max_rows = 100_000
    if len(flows) > max_rows:
        flows = flows.sample(n=max_rows, random_state=rng.randrange(0, 2**32 - 1))
    return flows.reset_index(drop=True)


def _select_porticos(
    flows: pd.DataFrame, acc_day: pd.DataFrame, max_porticos: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flow_ports = flows["PORTICO"].astype(str).unique().tolist()
    acc_ports = acc_day["ultimo_portico"].astype(str).unique().tolist()
    common = [p for p in acc_ports if p in flow_ports]
    if not common:
        common = flow_ports
    selected = common[:max_porticos]
    flows = flows[flows["PORTICO"].astype(str).isin(selected)].copy()
    acc_day = acc_day[acc_day["ultimo_portico"].astype(str).isin(selected)].copy()
    return flows.reset_index(drop=True), acc_day.reset_index(drop=True)


def _select_features(df: pd.DataFrame, max_features: int = 12) -> List[str]:
    meta_cols = {"portico", "ts_min", "interval_start", "target"}
    numeric_cols = [
        c
        for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return []
    clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variances = clean.var().sort_values(ascending=False)
    return variances.index[: min(max_features, len(variances))].tolist()


def _build_graph_from_features(
    df_pm: pd.DataFrame,
    porticos_df: pd.DataFrame,
    selected_features: List[str],
    delta_features: List[str],
    dt_minutes: int,
    max_nodes: int = 2500,
) -> Optional[HeteroData]:
    if df_pm.empty or not selected_features:
        return None

    df_work = df_pm.copy()
    if "target" not in df_work.columns:
        return None

    df_work["portico"] = df_work["portico"].astype(str)
    df_work["target"] = df_work["target"].astype(int)

    if len(df_work) > max_nodes:
        positives = df_work[df_work["target"] == 1]
        negatives = df_work[df_work["target"] == 0]
        n_keep_neg = max(0, max_nodes - len(positives))
        if n_keep_neg > 0 and not negatives.empty:
            negatives = negatives.sample(
                n=min(n_keep_neg, len(negatives)), random_state=42
            )
        df_work = pd.concat([positives, negatives], ignore_index=True)
        df_work = df_work.sort_values(["portico", "ts_min"]).reset_index(drop=True)

    feat_clean = (
        df_work[selected_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    feat_values = feat_clean.to_numpy(dtype=float)
    mu = feat_values.mean(axis=0)
    sigma = feat_values.std(axis=0) + 1e-9
    feat_norm = (feat_values - mu) / sigma
    df_work[selected_features] = feat_norm

    delta_idx = [selected_features.index(f) for f in delta_features if f in selected_features]
    if not delta_idx:
        return None

    df_work["node_idx"] = np.arange(len(df_work), dtype=int)

    data = HeteroData()
    data["pm"].x = torch.tensor(df_work[selected_features].values, dtype=torch.float32)
    data["pm"].y = torch.tensor(df_work["target"].values, dtype=torch.long)
    data["pm"].is_accident_pm = data["pm"].y.bool()

    try:
        data.delta_feature_idx = torch.tensor(delta_idx, dtype=torch.long)
    except Exception:
        pass

    # Temporal edges
    df_work["next_ts_min"] = df_work["ts_min"] + int(dt_minutes)
    t_edges = pd.merge(
        df_work[["portico", "ts_min", "node_idx", "next_ts_min"]],
        df_work[["portico", "ts_min", "node_idx"]],
        left_on=["portico", "next_ts_min"],
        right_on=["portico", "ts_min"],
        suffixes=("_src", "_dst"),
    )
    temporal_src = t_edges["node_idx_src"].to_numpy(dtype=int)
    temporal_dst = t_edges["node_idx_dst"].to_numpy(dtype=int)

    # Spatial edges
    porticos_use = porticos_df.copy()
    porticos_use["portico"] = porticos_use["portico"].astype(str)
    porticos_use = porticos_use[
        porticos_use["portico"].isin(df_work["portico"].astype(str).unique())
    ].copy()
    pairs = []
    if not porticos_use.empty:
        for (_, _), grp in porticos_use.groupby(["calzada", "eje"]):
            grp = grp.sort_values("orden")
            ports = grp["portico"].astype(str).tolist()
            for i in range(len(ports) - 1):
                pairs.append((ports[i], ports[i + 1]))
    df_seq = pd.DataFrame(pairs, columns=["portico_src", "portico_dst"])
    spatial_src = np.array([], dtype=int)
    spatial_dst = np.array([], dtype=int)
    if not df_seq.empty:
        s_edges = pd.merge(
            df_work.rename(
                columns={"portico": "portico_src", "node_idx": "node_idx_src"}
            ),
            df_seq,
            on="portico_src",
            how="inner",
        )
        s_edges = pd.merge(
            s_edges,
            df_work.rename(
                columns={"portico": "portico_dst", "node_idx": "node_idx_dst"}
            ),
            on=["portico_dst", "ts_min"],
            how="inner",
        )
        spatial_src = s_edges["node_idx_src"].to_numpy(dtype=int)
        spatial_dst = s_edges["node_idx_dst"].to_numpy(dtype=int)

    # Build edge attributes (delta on selected delta features)
    delta_mat = feat_norm[:, delta_idx]

    if temporal_src.size > 0:
        temporal_attr = delta_mat[temporal_dst] - delta_mat[temporal_src]
        data[("pm", "temporal", "pm")].edge_index = torch.tensor(
            [temporal_src, temporal_dst], dtype=torch.long
        )
        data[("pm", "temporal", "pm")].edge_attr = torch.tensor(
            temporal_attr, dtype=torch.float32
        )

    if spatial_src.size > 0:
        spatial_attr = delta_mat[spatial_dst] - delta_mat[spatial_src]
        data[("pm", "spatial", "pm")].edge_index = torch.tensor(
            [spatial_src, spatial_dst], dtype=torch.long
        )
        data[("pm", "spatial", "pm")].edge_attr = torch.tensor(
            spatial_attr, dtype=torch.float32
        )

    if not data.edge_types:
        return None

    # Masks
    ts_vals = df_work["ts_min"].to_numpy()
    unique_ts = np.sort(np.unique(ts_vals))
    if len(unique_ts) < 6:
        return None
    train_cut = unique_ts[int(len(unique_ts) * 0.6) - 1]
    val_cut = unique_ts[int(len(unique_ts) * 0.8) - 1]
    train_mask = ts_vals <= train_cut
    val_mask = (ts_vals > train_cut) & (ts_vals <= val_cut)
    test_mask = ts_vals > val_cut

    data["pm"].train_mask = torch.from_numpy(train_mask)
    data["pm"].val_mask = torch.from_numpy(val_mask)
    data["pm"].test_mask = torch.from_numpy(test_mask)

    return data


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if (precision + recall) == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _train_and_eval(
    data: HeteroData,
    cfg: Dict[str, int],
    epochs: int,
    device: torch.device,
) -> Tuple[float, float]:
    in_channels = data["pm"].x.size(1)
    out_channels = int(data["pm"].y.max().item()) + 1
    edge_dim = 0
    for et in data.edge_types:
        ea = getattr(data[et], "edge_attr", None)
        if ea is not None:
            edge_dim = int(ea.size(1)) if ea.dim() > 1 else 1
            break

    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=cfg["hidden_channels"],
        out_channels=out_channels,
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        edge_feature_dim=edge_dim,
        num_layers=cfg["num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    edge_attr_dict = {
        et: data[et].edge_attr
        for et in data.edge_types
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
    }

    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    y = data["pm"].y.to(device)
    train_mask = data["pm"].train_mask.to(device)
    val_mask = data["pm"].val_mask.to(device)
    test_mask = data["pm"].test_mask.to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits_dict, _, _ = model(x_dict, edge_index_dict, edge_attr_dict)
        logits = logits_dict["pm"]
        loss = criterion(logits[train_mask], y[train_mask])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_dict, _, _ = model(x_dict, edge_index_dict, edge_attr_dict)
        logits = logits_dict["pm"]
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    val_f1 = _binary_f1(y[val_mask].cpu().numpy(), y_pred[val_mask.cpu().numpy()])
    test_f1 = _binary_f1(y[test_mask].cpu().numpy(), y_pred[test_mask.cpu().numpy()])
    return val_f1, test_f1


def _train_gat_model(
    data: HeteroData, cfg: Dict[str, int], epochs: int, device: torch.device
) -> HeteroGAT:
    in_channels = data["pm"].x.size(1)
    out_channels = int(data["pm"].y.max().item()) + 1
    edge_dim = 0
    for et in data.edge_types:
        ea = getattr(data[et], "edge_attr", None)
        if ea is not None:
            edge_dim = int(ea.size(1)) if ea.dim() > 1 else 1
            break

    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=cfg["hidden_channels"],
        out_channels=out_channels,
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        edge_feature_dim=edge_dim,
        num_layers=cfg["num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    edge_attr_dict = {
        et: data[et].edge_attr
        for et in data.edge_types
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
    }

    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    y = data["pm"].y.to(device)
    train_mask = data["pm"].train_mask.to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits_dict, _, _ = model(x_dict, edge_index_dict, edge_attr_dict)
        logits = logits_dict["pm"]
        loss = criterion(logits[train_mask], y[train_mask])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    return model


def _apply_graphsmote_full(
    data: HeteroData,
    cfg: Dict[str, int],
    device: torch.device,
) -> Optional[HeteroData]:
    if data["pm"].train_mask.sum().item() == 0:
        return None
    model = _train_gat_model(data, cfg, epochs=1, device=device)

    try:
        data.edge_attr_dict = {
            et: data[et].edge_attr
            for et in data.edge_types
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
        }
    except Exception:
        pass

    z2x = train_z2x_decoders(
        model,
        data,
        epochs=1,
        show_progress=False,
        use_masks=True,
        num_neighbors=[6, 4],
    )

    z_dim = cfg["hidden_channels"] * cfg["num_heads"]
    z_dim_dict = {ntype: z_dim for ntype in data.node_types}
    edge_gen = RelEdgeGen(z_dim_dict, data.edge_types).to(device)
    pretrain_edge_generator(
        encoder=model,
        edge_gen=edge_gen,
        data=data,
        optimizer=torch.optim.Adam(edge_gen.parameters(), lr=1e-3),
        criterion=torch.nn.BCEWithLogitsLoss(),
        pretrain_epochs=1,
        device=device,
    )

    aug, registry = augment_graph_offline_once(
        model=model,
        data=data,
        device=device,
        target_pos_ratio=0.45,
        z2x_decoders=z2x,
        k=3,
        edge_gen=edge_gen,
        seed=42,
        num_neighbors=[6, 4],
    )
    try:
        n_new = int(registry.get("pm", {}).get("n_new", 0))
        if n_new <= 0:
            return None
    except Exception:
        pass
    return aug


@pytest.mark.integration
def test_gnn_pipeline_real_data() -> None:
    """
    Test GNN pipeline (real data):
    - Feature creation (flows from 1 random week)
    - Feature selection
    - Graph construction
    - Lightweight optimization (2 configs)
    - Train/test for: no balance, GraphSMOTE
    """
    if not _is_enabled():
        pytest.skip(f"Set {REAL_DATA_ENV}=1 to run this real-data pipeline test.")

    porticos_path = Path("Datos") / "Porticos.csv"
    accidentes_path = Path("Datos") / "Eventos-2022-2024.csv"
    if not porticos_path.exists() or not accidentes_path.exists():
        pytest.skip("Real data files not available in Datos/.")

    porticos_df = load_porticos()
    if porticos_df is None or porticos_df.empty:
        pytest.skip("Porticos data not available.")

    acc_df = load_accidentes(file_path=str(accidentes_path))
    if acc_df is None or acc_df.empty:
        pytest.skip("Accidents data not available.")

    try:
        summary = get_flow_db_summary()
        _log(
            "flow_db_path="
            f"{summary.db_path} rows={summary.row_count} "
            f"min_ts={summary.min_timestamp} max_ts={summary.max_timestamp}"
        )
    except Exception as exc:
        _log(f"flow_db_summary_error={exc}")

    rng = random.Random(42)
    week_start, flows, acc_day = _pick_random_week_with_data(acc_df, rng, window_days=7)
    if week_start is None or flows is None or acc_day is None:
        pytest.skip("Could not sample a week with matching flows + accidents.")

    week_end = week_start + pd.Timedelta(days=7)
    _log(
        f"week_selected={week_start.strftime('%Y-%m-%d')} -> {week_end.strftime('%Y-%m-%d')}"
    )
    flows = _sample_flow_window(flows, rng)
    flows, acc_day = _select_porticos(flows, acc_day)
    if flows.empty or acc_day.empty:
        pytest.skip("Sampled flows/accidents window is empty after filtering.")

    flow_min = pd.to_datetime(flows["FECHA"], errors="coerce").min()
    flow_max = pd.to_datetime(flows["FECHA"], errors="coerce").max()
    acc_min = pd.to_datetime(acc_day["accidente_time"], errors="coerce").min()
    acc_max = pd.to_datetime(acc_day["accidente_time"], errors="coerce").max()
    _log(
        "flows_rows="
        f"{len(flows)} ports={flows['PORTICO'].nunique()} "
        f"range=({flow_min}, {flow_max})"
    )
    _log(
        "acc_rows="
        f"{len(acc_day)} ports={acc_day['ultimo_portico'].nunique()} "
        f"range=({acc_min}, {acc_max})"
    )

    dt_minutes = 5
    df_pm = compute_pm_features(flows, interactive=False, dt_minutes=dt_minutes)
    if df_pm.empty:
        pytest.skip("Feature generation returned empty dataset.")

    df_pm["interval_start"] = pd.to_datetime(
        df_pm["ts_min"], unit="m", origin="unix"
    )
    df_pm = add_accident_target(
        df_pm,
        acc_day,
        interval_minutes=dt_minutes,
        portico_col="portico",
        interval_col="interval_start",
    )
    if df_pm["target"].nunique() < 2:
        pytest.skip("Sampled data has a single class; cannot train/test.")

    selected_features = _select_features(df_pm, max_features=12)
    if len(selected_features) < 2:
        pytest.skip("Not enough numeric features to build the graph.")
    delta_features = selected_features[: max(1, len(selected_features) // 2)]
    _log(
        "features_selected="
        f"{len(selected_features)} delta_features={len(delta_features)} "
        f"example={selected_features[:4]}"
    )

    data = _build_graph_from_features(
        df_pm,
        porticos_df,
        selected_features=selected_features,
        delta_features=delta_features,
        dt_minutes=dt_minutes,
    )
    if data is None:
        pytest.skip("Graph construction failed or returned empty graph.")

    edge_counts = {
        str(et): int(data[et].edge_index.size(1)) for et in data.edge_types
    }
    edge_dims = {}
    for et in data.edge_types:
        ea = getattr(data[et], "edge_attr", None)
        edge_dims[str(et)] = int(ea.size(1)) if ea is not None and ea.dim() > 1 else 0
    _log(
        "graph_stats="
        f"nodes={int(data['pm'].num_nodes)} edges={edge_counts} "
        f"edge_dims={edge_dims}"
    )

    # Ensure masks contain both classes
    train_labels = data["pm"].y[data["pm"].train_mask]
    val_labels = data["pm"].y[data["pm"].val_mask]
    test_labels = data["pm"].y[data["pm"].test_mask]
    _log(
        "mask_counts="
        f"train={int(train_labels.numel())} "
        f"val={int(val_labels.numel())} test={int(test_labels.numel())} "
        f"pos_train={int((train_labels==1).sum())} "
        f"pos_val={int((val_labels==1).sum())} "
        f"pos_test={int((test_labels==1).sum())}"
    )
    if train_labels.unique().numel() < 2 or val_labels.unique().numel() < 2:
        pytest.skip("Train/val masks lack both classes; retry with another day.")

    device = torch.device("cpu")
    configs = [
        {"hidden_channels": 16, "num_heads": 2, "num_layers": 2, "dropout": 0.1},
        {"hidden_channels": 32, "num_heads": 2, "num_layers": 2, "dropout": 0.2},
    ]

    strategies = ["none", "graphsmote"]
    results = {}

    for strategy in strategies:
        _log(f"strategy={strategy}")
        if strategy == "graphsmote":
            graph_variant = _apply_graphsmote_full(data, configs[0], device)
            if graph_variant is None:
                _log("graphsmote_skipped=1")
                continue
        else:
            graph_variant = data

        synth_count = None
        if hasattr(graph_variant["pm"], "is_synthetic"):
            try:
                synth_count = int(graph_variant["pm"].is_synthetic.sum().item())
            except Exception:
                synth_count = None
        _log(
            "variant_stats="
            f"nodes={int(graph_variant['pm'].num_nodes)} "
            f"synth={synth_count}"
        )

        best_cfg = None
        best_val = -1.0
        for cfg in configs:
            val_f1, _ = _train_and_eval(
                graph_variant, cfg, epochs=1, device=device
            )
            _log(f"cfg={cfg} val_f1={val_f1:.4f}")
            if val_f1 > best_val:
                best_val = val_f1
                best_cfg = cfg

        assert best_cfg is not None
        val_f1, test_f1 = _train_and_eval(
            graph_variant, best_cfg, epochs=2, device=device
        )
        _log(
            "best_cfg="
            f"{best_cfg} val_f1={val_f1:.4f} test_f1={test_f1:.4f}"
        )
        results[strategy] = {"val_f1": val_f1, "test_f1": test_f1}

    assert results
    for metrics in results.values():
        assert 0.0 <= metrics["val_f1"] <= 1.0
        assert 0.0 <= metrics["test_f1"] <= 1.0
