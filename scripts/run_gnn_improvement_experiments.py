#!/usr/bin/env python3
"""Run reproducible GNN improvement experiments.

The default mode is a fast pilot: it keeps the full graph topology and full
validation/test masks, but downsamples train negatives. This screens hypotheses
without spending a full training run on every variant.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import gnn_main  # noqa: E402


DEFAULT_GRAPH = ROOT / "Resultados" / "highway_graph_stream_build_15052026_1521.pt"
DEFAULT_HPARAMS = (
    ROOT
    / "Resultados"
    / "optuna_hyperparams_network_20260515_172309_533b8548afefd789_Base.csv"
)
DEFAULT_OUT_DIR = ROOT / "Resultados" / "gnn_improvement_experiments"


@dataclass(frozen=True)
class Experiment:
    name: str
    train_sampler_mode: str = "neighbor"
    positive_fraction: float | None = None
    hard_window: int | None = None
    hard_per_positive: int | None = None
    loss_type: str | None = None
    focal_alpha: float | None = None
    focal_gamma: float | None = None
    loss_weight_mode: str | None = None
    checkpoint_metric: str | None = None
    ranking_loss_mode: str | None = None
    ranking_loss_weight: float | None = None
    ranking_loss_margin: float | None = None
    ranking_loss_max_pairs: int | None = None
    objective_metric: str | None = None
    threshold_beta: float | None = None
    num_neighbors: Any | None = None
    horizon_minutes: int | None = None


def _sampler_experiments() -> list[Experiment]:
    spatial_only = {
        "('pm', 'temporal', 'pm')": [0, 0],
        "('pm', 'spatial', 'pm')": [25, 25],
    }
    temporal_only = {
        "('pm', 'temporal', 'pm')": [25, 25],
        "('pm', 'spatial', 'pm')": [0, 0],
    }
    return [
        Experiment("control_neighbor_focal_current"),
        Experiment(
            "posaware_005_hard0_focal_current",
            train_sampler_mode="positive_aware",
            positive_fraction=0.005,
            hard_window=60,
            hard_per_positive=0,
        ),
        Experiment(
            "posaware_010_hard0_focal_current",
            train_sampler_mode="positive_aware",
            positive_fraction=0.010,
            hard_window=60,
            hard_per_positive=0,
        ),
        Experiment(
            "posaware_020_hard0_focal_current",
            train_sampler_mode="positive_aware",
            positive_fraction=0.020,
            hard_window=60,
            hard_per_positive=0,
        ),
        Experiment(
            "posaware_010_hard1_w30_focal_current",
            train_sampler_mode="positive_aware",
            positive_fraction=0.010,
            hard_window=30,
            hard_per_positive=1,
        ),
        Experiment(
            "neighbor_focal_soft_alpha050_gamma100",
            focal_alpha=0.50,
            focal_gamma=1.00,
        ),
        Experiment(
            "posaware_010_hard0_focal_soft",
            train_sampler_mode="positive_aware",
            positive_fraction=0.010,
            hard_window=60,
            hard_per_positive=0,
            focal_alpha=0.50,
            focal_gamma=1.00,
        ),
        Experiment("neighbor_spatial_only", num_neighbors=spatial_only),
        Experiment("neighbor_temporal_only", num_neighbors=temporal_only),
        Experiment(
            "posaware_010_spatial_only",
            train_sampler_mode="positive_aware",
            positive_fraction=0.010,
            hard_window=60,
            hard_per_positive=0,
            num_neighbors=spatial_only,
        ),
        Experiment("horizon15_neighbor_current", horizon_minutes=15),
        Experiment("horizon30_neighbor_current", horizon_minutes=30),
        Experiment(
            "horizon30_posaware010_hard0_soft",
            train_sampler_mode="positive_aware",
            positive_fraction=0.010,
            hard_window=60,
            hard_per_positive=0,
            focal_alpha=0.50,
            focal_gamma=1.00,
            horizon_minutes=30,
        ),
        Experiment("horizon60_neighbor_current", horizon_minutes=60),
    ]


def _objective_experiments() -> list[Experiment]:
    return [
        Experiment("objective_control_neighbor_focal_current"),
        Experiment(
            "objective_focal_alpha085_gamma100",
            loss_type="FocalLoss",
            focal_alpha=0.85,
            focal_gamma=1.00,
        ),
        Experiment(
            "objective_focal_alpha090_gamma100",
            loss_type="FocalLoss",
            focal_alpha=0.90,
            focal_gamma=1.00,
        ),
        Experiment(
            "objective_focal_alpha095_gamma100",
            loss_type="FocalLoss",
            focal_alpha=0.95,
            focal_gamma=1.00,
        ),
        Experiment(
            "objective_focal_alpha090_gamma050",
            loss_type="FocalLoss",
            focal_alpha=0.90,
            focal_gamma=0.50,
        ),
        Experiment(
            "objective_focal_alpha090_gamma200",
            loss_type="FocalLoss",
            focal_alpha=0.90,
            focal_gamma=2.00,
        ),
        Experiment(
            "objective_weighted_cross_entropy",
            loss_type="CrossEntropy",
        ),
        Experiment(
            "objective_focal_distance_weights",
            loss_type="FocalLoss",
            focal_alpha=0.75,
            focal_gamma=1.50,
            loss_weight_mode="distance",
        ),
        Experiment(
            "objective_monitor_f05_current_loss",
            checkpoint_metric="val_f05",
        ),
        Experiment(
            "objective_monitor_mcc_current_loss",
            checkpoint_metric="val_mcc",
        ),
    ]


def _ranking_experiments() -> list[Experiment]:
    return [
        Experiment("ranking_control_neighbor_focal_current"),
        Experiment(
            "ranking_pairwise_w010_auprc",
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.10,
            ranking_loss_max_pairs=4096,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "ranking_pairwise_w025_auprc",
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.25,
            ranking_loss_max_pairs=4096,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "ranking_pairwise_w050_auprc",
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.50,
            ranking_loss_max_pairs=4096,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "ranking_topk_hardneg_w025_auprc",
            ranking_loss_mode="topk_pairwise",
            ranking_loss_weight=0.25,
            ranking_loss_max_pairs=4096,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "ranking_pairwise_w025_monitor_recall100",
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.25,
            ranking_loss_max_pairs=4096,
            checkpoint_metric="val_recall_at_100",
        ),
        Experiment(
            "ranking_pairwise_w025_horizon15",
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.25,
            ranking_loss_max_pairs=4096,
            checkpoint_metric="val_auprc",
            horizon_minutes=15,
        ),
        Experiment(
            "ranking_pairwise_w025_horizon30",
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.25,
            ranking_loss_max_pairs=4096,
            checkpoint_metric="val_auprc",
            horizon_minutes=30,
        ),
    ]


def _experiments(suite: str) -> list[Experiment]:
    key = str(suite or "sampler").strip().lower()
    if key == "sampler":
        return _sampler_experiments()
    if key == "objective":
        return _objective_experiments()
    if key == "ranking":
        return _ranking_experiments()
    if key == "all":
        return _sampler_experiments() + _objective_experiments() + _ranking_experiments()
    raise ValueError(f"Suite desconocida: {suite!r}")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _mask_stats(data: Any) -> dict[str, int]:
    y = data["pm"].y.detach().cpu().long()
    out: dict[str, int] = {}
    for split in ("train", "val", "test"):
        mask = getattr(data["pm"], f"{split}_mask").detach().cpu().bool()
        out[f"{split}_n"] = int(mask.sum().item())
        out[f"{split}_pos"] = int((y[mask] == 1).sum().item())
        out[f"{split}_neg"] = int((y[mask] == 0).sum().item())
    return out


def _make_pilot_train_mask(data: Any, neg_fraction: float, seed: int) -> None:
    train_mask = data["pm"].train_mask.detach().cpu().bool()
    y = data["pm"].y.detach().cpu().long()
    pos_idx = torch.where(train_mask & (y == 1))[0]
    neg_idx = torch.where(train_mask & (y == 0))[0]
    keep_neg = int(max(1, round(float(neg_fraction) * int(neg_idx.numel()))))
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(int(neg_idx.numel()), generator=gen)[:keep_neg]
    new_train = torch.zeros_like(train_mask)
    new_train[pos_idx] = True
    new_train[neg_idx[perm]] = True
    data["pm"].train_mask = new_train.to(data["pm"].train_mask.device)


def _pm_index_rev(pm_index: Any) -> Any:
    if isinstance(pm_index, dict):
        return pm_index.get("_rev")
    return getattr(pm_index, "_rev", None)


def _build_horizon_labels(data: Any, pm_index: Any, horizon_minutes: int) -> torch.Tensor:
    if not hasattr(data["pm"], "is_accident_pm"):
        raise ValueError("El grafo no contiene data['pm'].is_accident_pm para crear horizontes.")
    rev = _pm_index_rev(pm_index)
    if rev is None:
        raise ValueError("pm_index no contiene '_rev'; no se puede mapear node_idx a portico/ts_min.")

    n = int(data["pm"].num_nodes)
    porticos = np.empty(n, dtype=object)
    ts_min = np.empty(n, dtype=np.int64)
    for idx in range(n):
        p, ts = rev[idx]
        porticos[idx] = str(p)
        ts_min[idx] = int(ts)

    direct = data["pm"].is_accident_pm.detach().cpu().bool().numpy()
    labels = np.zeros(n, dtype=bool)
    for portico in np.unique(porticos):
        port_mask = porticos == portico
        event_times = np.sort(ts_min[port_mask & direct])
        if event_times.size == 0:
            continue
        node_times = ts_min[port_mask]
        left = np.searchsorted(event_times, node_times, side="left")
        has_event = left < event_times.size
        dt = np.full(node_times.shape, fill_value=np.iinfo(np.int64).max, dtype=np.int64)
        dt[has_event] = event_times[left[has_event]] - node_times[has_event]
        labels[np.where(port_mask)[0]] = (dt >= 0) & (dt <= int(horizon_minutes))
    return torch.from_numpy(labels.astype(np.int64))


def _prepare_loaded_obj(base_obj: dict[str, Any], exp: Experiment, args: argparse.Namespace) -> dict[str, Any]:
    obj = copy.deepcopy(base_obj)
    data = obj["data"]
    if exp.horizon_minutes is not None:
        data["pm"].y = _build_horizon_labels(data, obj.get("pm_index") or {}, exp.horizon_minutes)
        if hasattr(data["pm"], "loss_weight"):
            data["pm"].loss_weight = torch.ones_like(data["pm"].y, dtype=torch.float)
    if args.mode == "pilot":
        _make_pilot_train_mask(data, args.pilot_neg_fraction, args.seed)
    return obj


def _write_hparams(base_hparams: pd.DataFrame, exp: Experiment, path: Path, args: argparse.Namespace) -> None:
    row = base_hparams.iloc[0].copy()
    row["value"] = 0.0
    row["hparams_source"] = f"gnn_improvement_{args.suite}_{args.mode}"
    row["train_sampler_mode"] = exp.train_sampler_mode
    row["disable_hard_undersampling"] = True
    row["deterministic_sampling"] = True
    row["sampling_seed"] = int(args.seed)
    row["checkpoint_metric"] = exp.checkpoint_metric or "val_auprc"
    if exp.objective_metric is not None:
        row["objective_metric"] = str(exp.objective_metric)
    if exp.threshold_beta is not None:
        row["threshold_beta"] = float(exp.threshold_beta)
    if exp.loss_type is not None:
        row["loss_type"] = str(exp.loss_type)
    if exp.positive_fraction is not None:
        row["positive_sampler_target_fraction"] = float(exp.positive_fraction)
    if exp.hard_window is not None:
        row["positive_sampler_hard_window_minutes"] = int(exp.hard_window)
    if exp.hard_per_positive is not None:
        row["positive_sampler_hard_negatives_per_positive"] = int(exp.hard_per_positive)
    if exp.focal_alpha is not None:
        row["focal_alpha"] = float(exp.focal_alpha)
    if exp.focal_gamma is not None:
        row["focal_gamma"] = float(exp.focal_gamma)
    if exp.loss_weight_mode is not None:
        row["loss_weight_mode"] = str(exp.loss_weight_mode)
    if exp.ranking_loss_mode is not None:
        row["ranking_loss_mode"] = str(exp.ranking_loss_mode)
    if exp.ranking_loss_weight is not None:
        row["ranking_loss_weight"] = float(exp.ranking_loss_weight)
    if exp.ranking_loss_margin is not None:
        row["ranking_loss_margin"] = float(exp.ranking_loss_margin)
    if exp.ranking_loss_max_pairs is not None:
        row["ranking_loss_max_pairs"] = int(exp.ranking_loss_max_pairs)
    if exp.num_neighbors is not None:
        row["num_neighbors"] = json.dumps(exp.num_neighbors)
    pd.DataFrame([row]).to_csv(path, index=False)


def _find_new_model(start_time: float) -> Path | None:
    candidates = []
    for path in (ROOT / "Resultados").glob("gat_model_BEST_GNN_*.pt"):
        try:
            if path.stat().st_mtime >= start_time - 2.0:
                candidates.append(path)
        except OSError:
            pass
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_sidecar(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    sidecar = model_path.with_name(model_path.stem + "_hparams.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return {}


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_result_rows(path)
    rows.append({k: _json_for_csv(v) for k, v in row.items()})
    _write_result_rows(path, rows)


def _expanded_result_header() -> list[str]:
    return sorted(
        {
            "best_epoch",
            "best_val_auc",
            "best_val_auprc",
            "best_val_f05",
            "best_val_f1",
            "best_val_far",
            "best_val_tau",
            "checkpoint_metric",
            "elapsed_seconds",
            "focal_alpha",
            "focal_gamma",
            "hard_per_positive",
            "hard_window",
            "horizon_minutes",
            "hparams_path",
            "loss_type",
            "loss_type_used",
            "loss_weight_mode",
            "loss_weight_mode_used",
            "metrics_history_path",
            "mode",
            "model_hparams_path",
            "model_path",
            "monitor_metric",
            "name",
            "num_neighbors",
            "objective_metric",
            "positive_fraction",
            "positive_sampler_stats",
            "ranking_loss_margin",
            "ranking_loss_max_pairs",
            "ranking_loss_mode",
            "ranking_loss_mode_used",
            "ranking_loss_weight",
            "ranking_loss_weight_used",
            "started_at",
            "status",
            "suite",
            "threshold_beta",
            "best_val_precision_at_k",
            "best_val_recall_at_k",
            "test_n",
            "test_neg",
            "test_pos",
            "train_n",
            "train_neg",
            "train_pos",
            "train_sampler_impl",
            "train_sampler_mode",
            "val_n",
            "val_neg",
            "val_pos",
        }
    )


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return []
        expanded_header = _expanded_result_header()
        rows: list[dict[str, Any]] = []
        for raw in reader:
            if not raw:
                continue
            if len(raw) == len(header):
                mapped = dict(zip(header, raw))
            elif len(raw) == len(expanded_header):
                mapped = dict(zip(expanded_header, raw))
            else:
                mapped = dict(zip(header, raw[: len(header)]))
                if len(raw) > len(header):
                    mapped["_extra"] = json.dumps(raw[len(header) :], ensure_ascii=True)
            rows.append(mapped)
    return rows


def _write_result_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for item in rows for key in item.keys()})
    if not fields:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _repair_results_file(path: Path) -> None:
    if path.exists():
        _write_result_rows(path, _read_result_rows(path))


def _json_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def _completed_names(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    try:
        df = pd.read_csv(results_path)
    except Exception:
        return set()
    if "status" not in df.columns or "name" not in df.columns:
        return set()
    return set(df.loc[df["status"].astype(str) == "ok", "name"].astype(str).tolist())


def _metric(sidecar: dict[str, Any], key: str) -> Any:
    value = sidecar.get(key)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir).resolve()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"{args.mode}_results.csv"
    hparams_dir = out_dir / "hparams"
    history_dir = out_dir / "histories"
    hparams_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    _repair_results_file(results_path)

    base_obj = _torch_load(Path(args.graph))
    if not isinstance(base_obj, dict) or "data" not in base_obj:
        raise ValueError(f"Grafo invalido: {args.graph}")
    base_hparams = pd.read_csv(args.hparams)
    completed = _completed_names(results_path) if args.resume else set()
    experiments = _experiments(args.suite)
    if args.only:
        wanted = set(args.only.split(","))
        experiments = [exp for exp in experiments if exp.name in wanted]

    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "suite": args.suite,
        "graph": str(Path(args.graph).resolve()),
        "hparams": str(Path(args.hparams).resolve()),
        "max_epochs": int(args.max_epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "pilot_neg_fraction": float(args.pilot_neg_fraction),
        "experiments": [asdict(exp) for exp in experiments],
    }
    (out_dir / f"{args.mode}_manifest_{run_id}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True)
    )

    for idx, exp in enumerate(experiments, start=1):
        if exp.name in completed:
            print(f"[{idx}/{len(experiments)}] skip completed {exp.name}", flush=True)
            continue
        print(f"[{idx}/{len(experiments)}] running {exp.name}", flush=True)
        hp_path = hparams_dir / f"{args.mode}_{exp.name}.csv"
        _write_hparams(base_hparams, exp, hp_path, args)
        loaded_obj = _prepare_loaded_obj(base_obj, exp, args)
        stats = _mask_stats(loaded_obj["data"])
        history_path = history_dir / f"{args.mode}_{exp.name}_metrics_history.jsonl"
        start = time.time()
        row: dict[str, Any] = {
            "name": exp.name,
            "mode": args.mode,
            "suite": args.suite,
            "status": "ok",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hparams_path": str(hp_path),
            "metrics_history_path": str(history_path),
            **asdict(exp),
            **stats,
        }
        try:
            gnn_main.run_gat_training(
                loaded_obj,
                force_use_graphsmote=False,
                purpose=f"gnn_improvement_{args.suite}_{args.mode}:{exp.name}",
                early_stop=True,
                early_stop_patience=int(args.early_stop_patience),
                early_stop_min_delta=float(args.early_stop_min_delta),
                max_epochs=int(args.max_epochs),
                train_sampler_mode=exp.train_sampler_mode,
                deterministic_sampling=True,
                sampling_seed=int(args.seed),
                disable_hard_undersampling=True,
                positive_sampler_target_fraction=exp.positive_fraction,
                positive_sampler_hard_window_minutes=exp.hard_window,
                positive_sampler_hard_negatives_per_positive=exp.hard_per_positive,
                eval_neighbors_mode="same",
                checkpoint_metric=exp.checkpoint_metric or "val_auprc",
                metrics_history_path=str(history_path),
                test_eval_interval_epochs=0,
                hparams_path=str(hp_path),
                hparams_index=None,
                reuse_hparams=True,
                allow_hpo_search=False,
            )
            model_path = _find_new_model(start)
            sidecar = _read_sidecar(model_path)
            row.update(
                {
                    "elapsed_seconds": round(time.time() - start, 3),
                    "model_path": str(model_path) if model_path else "",
                    "model_hparams_path": str(model_path.with_name(model_path.stem + "_hparams.json"))
                    if model_path
                    else "",
                    "best_epoch": _metric(sidecar, "best_epoch"),
                    "best_val_auprc": _metric(sidecar, "best_val_auprc"),
                    "best_val_auc": _metric(sidecar, "best_val_auc"),
                    "best_val_f1": _metric(sidecar, "best_val_f1"),
                    "best_val_f05": _metric(sidecar, "best_val_f05"),
                    "best_val_far": _metric(sidecar, "best_val_far"),
                    "best_val_tau": _metric(sidecar, "best_val_tau"),
                    "monitor_metric": _metric(sidecar, "monitor_metric"),
                    "loss_type_used": _metric(sidecar, "loss_type"),
                    "loss_weight_mode_used": _metric(sidecar, "loss_weight_mode"),
                    "ranking_loss_mode_used": _metric(sidecar, "ranking_loss_mode"),
                    "ranking_loss_weight_used": _metric(sidecar, "ranking_loss_weight"),
                    "best_val_recall_at_k": _metric(sidecar, "best_val_recall_at_k"),
                    "best_val_precision_at_k": _metric(sidecar, "best_val_precision_at_k"),
                    "train_sampler_impl": _metric(sidecar, "sampler_impl"),
                    "positive_sampler_stats": _metric(sidecar, "positive_sampler_stats"),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "elapsed_seconds": round(time.time() - start, 3),
                    "error": repr(exc),
                }
            )
            print(f"[{idx}/{len(experiments)}] ERROR {exp.name}: {exc!r}", flush=True)
        _append_row(results_path, row)
    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--suite", choices=["sampler", "objective", "ranking", "all"], default="sampler")
    parser.add_argument("--pilot-neg-fraction", type=float, default=0.15)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=19091985)
    parser.add_argument("--only", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(f"results_path={output}", flush=True)
