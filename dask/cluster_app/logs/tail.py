from __future__ import annotations

from pathlib import Path


def tail(path: str | Path, lines: int = 200) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    data = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return data[-lines:]

