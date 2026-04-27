from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any


def checkpoint_dir() -> Path:
    root = os.environ.get("CLUSTER_APP_CHECKPOINT_DIR")
    if not root:
        root = str(Path.cwd() / "checkpoints")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(name: str, value: Any) -> Path:
    path = checkpoint_dir() / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    else:
        tmp.write_bytes(pickle.dumps(value))
    tmp.replace(path)
    return path


def load_checkpoint(name: str, default: Any = None) -> Any:
    path = checkpoint_dir() / name
    if not path.exists():
        return default
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pickle.loads(path.read_bytes())


def has_checkpoint(name: str) -> bool:
    return (checkpoint_dir() / name).exists()

