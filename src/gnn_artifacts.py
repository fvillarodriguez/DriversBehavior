from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List, Optional


GNN_ARTIFACT_DIRS = {
    "root": (),
    "graphs_base": ("graphs", "base"),
    "graphs_graphsmote": ("graphs", "balanced", "graphsmote"),
    "graphs_imgagn": ("graphs", "balanced", "imgagn"),
    "models_gat": ("models", "gat"),
    "models_graphsmote": ("models", "graphsmote"),
    "hpo_optuna": ("hpo", "optuna"),
    "hpo_imgagn": ("hpo", "imgagn"),
    "hpo_ray_tune": ("hpo", "ray_tune"),
    "training_checkpoints": ("training", "checkpoints"),
    "training_tensorboard": ("training", "tensorboard"),
    "evaluation": ("evaluation",),
    "xai": ("xai",),
    "features": ("features",),
    "imgagn_validations": ("balance", "imgagn_validations"),
    "baselines_mlp": ("baselines", "mlp"),
    "network_configs": ("network", "configs"),
    "network_architectures": ("network", "architectures"),
    "experiments_results": ("experiments", "results"),
    "experiments_legacy_workspaces": ("experiments", "legacy_workspaces"),
    "legacy": ("legacy",),
}


def gnn_root(resultados_dir: str | os.PathLike[str]) -> Path:
    return Path(resultados_dir) / "gnn"


def gnn_dir(
    key: str,
    resultados_dir: str | os.PathLike[str],
    *,
    create: bool = False,
) -> Path:
    if key not in GNN_ARTIFACT_DIRS:
        raise KeyError(f"Unknown GNN artifact directory: {key}")
    path = gnn_root(resultados_dir).joinpath(*GNN_ARTIFACT_DIRS[key])
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def gnn_path(
    key: str,
    *parts: str | os.PathLike[str],
    resultados_dir: str | os.PathLike[str],
    create_parent: bool = False,
) -> Path:
    path = gnn_dir(key, resultados_dir).joinpath(*map(Path, parts))
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def gnn_glob(
    key: str,
    pattern: str,
    resultados_dir: str | os.PathLike[str],
    *,
    sort_mtime: bool = False,
    reverse: bool = False,
) -> List[str]:
    base = gnn_dir(key, resultados_dir)
    matches = glob.glob(str(base / pattern))
    if sort_mtime:
        matches = sorted(matches, key=os.path.getmtime, reverse=reverse)
    else:
        matches = sorted(matches, reverse=reverse)
    return matches


def ensure_gnn_dirs(
    resultados_dir: str | os.PathLike[str],
    keys: Optional[Iterable[str]] = None,
) -> None:
    for key in keys or GNN_ARTIFACT_DIRS:
        gnn_dir(key, resultados_dir, create=True)


def resolve_gnn_artifact(
    filename: str | os.PathLike[str],
    resultados_dir: str | os.PathLike[str],
    keys: Iterable[str],
) -> Optional[Path]:
    raw = Path(filename)
    if raw.exists():
        return raw
    name = raw.name
    for key in keys:
        candidate = gnn_dir(key, resultados_dir) / name
        if candidate.exists():
            return candidate
    return None
