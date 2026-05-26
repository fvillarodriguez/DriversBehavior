#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT_FILE_RULES: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
    (("highway_graph_ImGAGN*.pt", "graph_imgagn_relational_*.pt"), ("gnn", "graphs", "balanced", "imgagn")),
    (("highway_graph_SMOTE*.pt", "graph_aug*.pt", "graph_smote_*.pt"), ("gnn", "graphs", "balanced", "graphsmote")),
    (("highway_graph_*.pt", "graph_*.pt"), ("gnn", "graphs", "base")),
    (("gat_model_BEST*.pt", "gat_model_BEST*_hparams.json"), ("gnn", "models", "gat")),
    (("GraphSMOTE_embeddings_model*.pt", "GraphSMOTE_embeddings_model*_hparams.json"), ("gnn", "models", "graphsmote")),
    (("optuna_hyperparams_*.csv", "optuna_full_study_*.csv", "optuna_studies.db"), ("gnn", "hpo", "optuna")),
    (("imgagn_hyperparams_*.csv", "imgagn_full_study_*.csv"), ("gnn", "hpo", "imgagn")),
    (("gnn_experiments_results_*.csv", "experiment_live_*.sqlite", "gnn_architecture_pilot_*_metadata.json"), ("gnn", "experiments", "results")),
    (("gnn_self_loop_ablation_*.json",), ("gnn", "experiments", "results")),
    (("feature_selection_features_*_gnn*.json", "feature_selection_features_*_gnn*.csv"), ("gnn", "features")),
    (("gnn_history.jsonl",), ("gnn",)),
    (("network_config*.json",), ("gnn", "network", "configs")),
    (("relevant_edges_*.csv",), ("gnn", "xai")),
    (("preds_gat_report*.csv", "results_gat_report*.csv", "preds_gnn_anomaly*.csv", "results_gnn_anomaly*.csv"), ("gnn", "evaluation")),
]

ROOT_DIR_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("gnn_eval_live", ("gnn", "evaluation", "live_runs")),
    ("gnn_optuna_live", ("gnn", "hpo", "optuna", "live_runs")),
    ("ray_tune", ("gnn", "hpo", "ray_tune")),
    ("gnn_train_checkpoints", ("gnn", "training", "checkpoints")),
    ("runs_attention", ("gnn", "training", "tensorboard")),
    ("gnn_xai", ("gnn", "xai")),
    ("gnn_mlp_baselines", ("gnn", "baselines", "mlp")),
    ("gnn_network_architectures", ("gnn", "network", "architectures")),
    ("imgagn_relational_validations", ("gnn", "balance", "imgagn_validations")),
    ("graphsmote_generators", ("gnn", "models", "graphsmote", "graphsmote_generators")),
    ("edge_attr_decoders", ("gnn", "models", "graphsmote", "edge_attr_decoders")),
    ("z2x_decoders", ("gnn", "models", "graphsmote", "z2x_decoders")),
    ("gnn_improvement_experiments", ("gnn", "experiments", "legacy_workspaces", "gnn_improvement_experiments")),
    ("gnn_improvement_experiments_smoke", ("gnn", "experiments", "legacy_workspaces", "gnn_improvement_experiments_smoke")),
    ("gnn_improvement_experiments_smoke_objective", ("gnn", "experiments", "legacy_workspaces", "gnn_improvement_experiments_smoke_objective")),
    ("gnn_improvement_experiments_smoke_ranking", ("gnn", "experiments", "legacy_workspaces", "gnn_improvement_experiments_smoke_ranking")),
    ("gnn_improvement_experiments_smoke_ranking_posaware", ("gnn", "experiments", "legacy_workspaces", "gnn_improvement_experiments_smoke_ranking_posaware")),
]


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_migrated{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _iter_root_files(resultados: Path) -> Iterable[Tuple[Path, Path]]:
    seen: set[Path] = set()
    for patterns, dest_parts in ROOT_FILE_RULES:
        dest_dir = resultados.joinpath(*dest_parts)
        for pattern in patterns:
            for src in resultados.glob(pattern):
                if not src.is_file() or src in seen:
                    continue
                seen.add(src)
                yield src, _unique_destination(dest_dir / src.name)


def _iter_root_dirs(resultados: Path) -> Iterable[Tuple[Path, Path]]:
    for dirname, dest_parts in ROOT_DIR_RULES:
        src = resultados / dirname
        if src.exists() and src.is_dir():
            yield src, resultados.joinpath(*dest_parts)


def _move(src: Path, dest: Path, *, execute: bool) -> dict:
    record = {
        "source": str(src),
        "destination": str(dest),
        "kind": "directory" if src.is_dir() else "file",
        "executed": bool(execute),
    }
    if execute:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir() and dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                shutil.move(str(child), str(_unique_destination(dest / child.name)))
            src.rmdir()
        else:
            shutil.move(str(src), str(_unique_destination(dest)))
    return record


def migrate(resultados: Path, *, execute: bool) -> dict:
    resultados = resultados.resolve()
    records: list[dict] = []
    for src, dest in list(_iter_root_files(resultados)) + list(_iter_root_dirs(resultados)):
        if src.resolve().is_relative_to((resultados / "gnn").resolve()):
            continue
        records.append(_move(src, dest, execute=execute))

    manifest = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "resultados_dir": str(resultados),
        "mode": "execute" if execute else "dry-run",
        "count": len(records),
        "moves": records,
    }
    if execute:
        manifest_dir = resultados / "gnn"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = manifest_dir / f"migration_manifest_{stamp}.json"
        manifest["manifest_path"] = str(manifest_path)
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate GNN artifacts under Resultados/gnn.")
    parser.add_argument("--resultados-dir", default="Resultados")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = migrate(Path(args.resultados_dir), execute=bool(args.execute))
    print(json.dumps({"mode": manifest["mode"], "count": manifest["count"], "manifest_path": manifest.get("manifest_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
