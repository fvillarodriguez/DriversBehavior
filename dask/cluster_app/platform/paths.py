from __future__ import annotations

import os
from pathlib import Path


def default_state_dir() -> Path:
    if os.name == "nt":
        root = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return Path(root) / "DaskClusterApp"
    return Path("~/.dask-cluster-app").expanduser()


def default_workspace_dir() -> Path:
    if os.name == "nt":
        return Path(r"C:\DaskCluster\workspace")
    return Path("~/DaskCluster/workspace").expanduser()


def normalize_user_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()

