from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml4roadsafety_validation.build_graph import (  # noqa: E402
        build_ml4roadsafety_graph,
        graph_diagnostics,
    )
    from ml4roadsafety_validation.config import (  # noqa: E402
        DATA_DIR,
        DEFAULT_MAX_SEGMENTS,
        DEFAULT_MONTHS,
        DEFAULT_SEED,
        DEFAULT_STATE,
        RESULTS_DIR,
    )
    from ml4roadsafety_validation.download import download_state  # noqa: E402
    from ml4roadsafety_validation.metrics import (  # noqa: E402
        classification_metrics,
        flatten_metric_rows,
        safe_auprc,
        select_threshold_by_top_k,
        select_threshold_by_fbeta,
    )
else:
    from .build_graph import build_ml4roadsafety_graph, graph_diagnostics
    from .config import (
        DATA_DIR,
        DEFAULT_MAX_SEGMENTS,
        DEFAULT_MONTHS,
        DEFAULT_SEED,
        DEFAULT_STATE,
        RESULTS_DIR,
    )
    from .download import download_state
    from .metrics import (
        classification_metrics,
        flatten_metric_rows,
        safe_auprc,
        select_threshold_by_top_k,
        select_threshold_by_fbeta,
    )

from src.gat_model import HeteroGAT


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    name = str(value or "cpu").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y[mask].long(), minlength=2).float()
    weights = counts.sum().clamp_min(1.0) / (2.0 * counts.clamp_min(1.0))
    weights = torch.where(counts > 0, weights, torch.ones_like(weights))
    return weights


def _probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 2 or int(logits.shape[1]) < 2:
        raise ValueError("Se esperaban logits binarios con shape [N, 2].")
    return F.softmax(logits, dim=-1)[:, 1]


def _threshold_candidates(
    y_val: torch.Tensor,
    prob_val: torch.Tensor,
    *,
    primary_beta: float,
) -> dict[str, float]:
    val_positive_count = int(y_val.long().sum().item())
    n_val = int(y_val.numel())
    policies = {
        "val_f0.5": select_threshold_by_fbeta(y_val, prob_val, beta=0.5),
        "val_f1": select_threshold_by_fbeta(y_val, prob_val, beta=1.0),
        "val_f2": select_threshold_by_fbeta(y_val, prob_val, beta=2.0),
        "primary_fbeta": select_threshold_by_fbeta(y_val, prob_val, beta=primary_beta),
        "fixed_0.5": 0.5,
    }
    if val_positive_count > 0:
        policies["val_top_positive_count"] = select_threshold_by_top_k(
            prob_val,
            k=val_positive_count,
        )
    for k in (100, 500, 1000):
        if n_val >= k:
            policies[f"val_top_{k}"] = select_threshold_by_top_k(prob_val, k=k)
    return {name: float(value) for name, value in policies.items()}


def _split_metric_bundle(
    data,
    probabilities: torch.Tensor,
    *,
    beta: float = 2.0,
) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, Mapping[str, object]]]]:
    y = data["pm"].y.detach().cpu()
    probs = probabilities.detach().cpu()
    val_mask = data["pm"].val_mask.detach().cpu().bool()
    threshold_by_policy = _threshold_candidates(
        y[val_mask],
        probs[val_mask],
        primary_beta=float(beta),
    )
    primary_policy = "primary_fbeta"
    primary_threshold = threshold_by_policy[primary_policy]

    out: dict[str, Mapping[str, object]] = {}
    diagnostics: dict[str, Mapping[str, Mapping[str, object]]] = {}
    split_masks = {
        split: getattr(data["pm"], f"{split}_mask").detach().cpu().bool()
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        mask = split_masks[split]
        metrics = classification_metrics(y[mask], probs[mask], threshold=primary_threshold)
        metrics["threshold_policy"] = primary_policy
        metrics["threshold_beta"] = float(beta)
        out[split] = metrics
    for policy, threshold in threshold_by_policy.items():
        diagnostics[policy] = {}
        for split, mask in split_masks.items():
            metrics = classification_metrics(y[mask], probs[mask], threshold=threshold)
            metrics["threshold_policy"] = policy
            metrics["threshold_beta"] = float(beta) if policy == "primary_fbeta" else None
            diagnostics[policy][split] = metrics
    return out, diagnostics


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    data,
    *,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    threshold_beta: float,
) -> dict[str, object]:
    data_dev = data.clone().to(device)
    x = data_dev["pm"].x
    y = data_dev["pm"].y.long()
    train_mask = data_dev["pm"].train_mask.bool()
    val_mask = data_dev["pm"].val_mask.bool()
    model = _MLP(int(x.shape[1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(y, train_mask).to(device))

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = _probabilities_from_logits(model(x))
            val_auprc = safe_auprc(y[val_mask], probs[val_mask])
        monitor = float(val_auprc if val_auprc is not None else -1.0)
        if monitor > best_val + 1e-12:
            best_val = monitor
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_probs = _probabilities_from_logits(model(x)).detach().cpu()
    metrics, threshold_diagnostics = _split_metric_bundle(
        data,
        final_probs,
        beta=threshold_beta,
    )
    return {
        "model": "mlp",
        "best_epoch": best_epoch,
        "best_val_auprc": best_val,
        "primary_threshold_policy": "primary_fbeta",
        "threshold_beta": float(threshold_beta),
        "metrics": metrics,
        "threshold_diagnostics": threshold_diagnostics,
    }


def _edge_feature_dims(data) -> dict[tuple[str, str, str], int]:
    dims: dict[tuple[str, str, str], int] = {}
    for edge_type in data.edge_types:
        edge_attr = getattr(data[edge_type], "edge_attr", None)
        if edge_attr is None:
            dims[edge_type] = 0
        elif edge_attr.dim() == 1:
            dims[edge_type] = 1
        else:
            dims[edge_type] = int(edge_attr.shape[1])
    return dims


def train_heterogat(
    data,
    *,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    threshold_beta: float,
) -> dict[str, object]:
    data_dev = data.clone().to(device)
    y = data_dev["pm"].y.long()
    train_mask = data_dev["pm"].train_mask.bool()
    val_mask = data_dev["pm"].val_mask.bool()
    edge_dims = _edge_feature_dims(data_dev)
    model = HeteroGAT(
        in_channels=int(data_dev["pm"].x.shape[1]),
        hidden_channels=32,
        out_channels=2,
        num_heads=2,
        dropout=0.1,
        edge_feature_dim=max(edge_dims.values()) if edge_dims else 0,
        edge_feature_dims=edge_dims,
        edge_types=tuple(data_dev.edge_types),
        num_layers=2,
        use_residual=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(y, train_mask).to(device))
    edge_attr_dict = {
        edge_type: data_dev[edge_type].edge_attr
        for edge_type in data_dev.edge_types
        if getattr(data_dev[edge_type], "edge_attr", None) is not None
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits_dict, _, _ = model(data_dev.x_dict, data_dev.edge_index_dict, edge_attr_dict)
        logits = logits_dict["pm"]
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_dict, _, _ = model(data_dev.x_dict, data_dev.edge_index_dict, edge_attr_dict)
            probs = _probabilities_from_logits(logits_dict["pm"])
            val_auprc = safe_auprc(y[val_mask], probs[val_mask])
        monitor = float(val_auprc if val_auprc is not None else -1.0)
        if monitor > best_val + 1e-12:
            best_val = monitor
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_dict, _, _ = model(data_dev.x_dict, data_dev.edge_index_dict, edge_attr_dict)
        final_probs = _probabilities_from_logits(logits_dict["pm"]).detach().cpu()
    metrics, threshold_diagnostics = _split_metric_bundle(
        data,
        final_probs,
        beta=threshold_beta,
    )
    return {
        "model": "heterogat",
        "best_epoch": best_epoch,
        "best_val_auprc": best_val,
        "primary_threshold_policy": "primary_fbeta",
        "threshold_beta": float(threshold_beta),
        "metrics": metrics,
        "threshold_diagnostics": threshold_diagnostics,
    }


def run_pilot(
    *,
    state: str,
    months: list[str],
    data_dir: Path,
    results_dir: Path,
    max_segments: int,
    max_epochs: int,
    patience: int,
    lr: float,
    threshold_beta: float,
    device_name: str,
    seed: int,
    skip_download: bool,
) -> dict[str, object]:
    set_seed(seed)
    download_state(data_dir=data_dir, state=state, skip_download=skip_download)
    data = build_ml4roadsafety_graph(
        data_dir=data_dir,
        state=state,
        months=months,
        max_segments=max_segments,
        seed=seed,
    )
    device = resolve_device(device_name)
    diagnostics = graph_diagnostics(data)
    mlp = train_mlp(
        data,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        threshold_beta=threshold_beta,
    )
    heterogat = train_heterogat(
        data,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        threshold_beta=threshold_beta,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "state": state.upper(),
        "months": list(months),
        "device": str(device),
        "max_segments": int(max_segments),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "lr": float(lr),
        "threshold_beta": float(threshold_beta),
        "seed": int(seed),
        "diagnostics": diagnostics,
        "models": {"mlp": mlp, "heterogat": heterogat},
    }
    summary_path = results_dir / f"pilot_summary_{run_id}.json"
    summary_path.write_text(json.dumps(_clean_json(summary), indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    rows.extend(flatten_metric_rows("mlp", mlp["metrics"]))
    rows.extend(flatten_metric_rows("heterogat", heterogat["metrics"]))
    diagnostic_rows = []
    for model_name, result in (("mlp", mlp), ("heterogat", heterogat)):
        for policy, split_metrics in result["threshold_diagnostics"].items():
            policy_rows = flatten_metric_rows(model_name, split_metrics)
            for row in policy_rows:
                row["threshold_policy"] = policy
            diagnostic_rows.extend(policy_rows)
    metrics_path = results_dir / f"pilot_metrics_{run_id}.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    diagnostics_path = results_dir / f"pilot_threshold_diagnostics_{run_id}.csv"
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_path, index=False)
    print(f"Resumen: {summary_path}")
    print(f"Metricas: {metrics_path}")
    print(f"Diagnostico umbrales: {diagnostics_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Piloto HeteroGAT sobre ML4RoadSafety.")
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--months", nargs="+", default=list(DEFAULT_MONTHS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SEGMENTS)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=2.0,
        help="Beta del umbral principal F-beta; 2.0 prioriza recall para clase rara.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps o auto.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run_pilot(
        state=args.state,
        months=args.months,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        max_segments=args.max_segments,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        threshold_beta=args.threshold_beta,
        device_name=args.device,
        seed=args.seed,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
