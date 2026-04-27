from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(logs_dir: str | Path, level: int = logging.INFO) -> None:
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs / "cluster-app.log", encoding="utf-8"),
        ],
    )

