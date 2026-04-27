from __future__ import annotations

import os
from pathlib import Path


def sqlite_storage_url(name: str = "optuna.db") -> str:
    workspace = Path(os.environ.get("CLUSTER_APP_WORKSPACE", Path.cwd()))
    db_path = workspace / "output" / name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"

