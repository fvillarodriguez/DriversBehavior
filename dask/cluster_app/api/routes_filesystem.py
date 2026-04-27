from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/filesystem", tags=["filesystem"])


@dataclass(frozen=True, slots=True)
class FileEntry:
    name: str
    path: str
    kind: str
    is_python: bool = False


@router.get("/list")
async def list_path(path: str | None = None):
    current = _resolve_path(path)
    if not current.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {current}")
    if not current.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a folder: {current}")

    entries: list[FileEntry] = []
    try:
        children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError as exc:
        raise HTTPException(status_code=403, detail=f"Cannot read folder: {current}") from exc

    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                entries.append(FileEntry(child.name, str(child), "directory"))
            elif child.is_file() and child.suffix == ".py":
                entries.append(FileEntry(child.name, str(child), "file", is_python=True))
        except OSError:
            continue

    return {
        "path": str(current),
        "parent": str(current.parent) if current.parent != current else None,
        "home": str(Path.home()),
        "entries": [asdict(entry) for entry in entries],
    }


@router.get("/python-files")
async def python_files(path: str):
    root = _resolve_path(path)
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {root}")
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a folder: {root}")

    files: list[dict[str, str]] = []
    try:
        for script in sorted(root.rglob("*.py"), key=lambda item: str(item.relative_to(root)).lower()):
            if _skip(script, root):
                continue
            files.append({"path": str(script), "relative_path": str(script.relative_to(root))})
            if len(files) >= 200:
                break
    except OSError as exc:
        raise HTTPException(status_code=403, detail=f"Cannot scan folder: {root}") from exc

    return {"path": str(root), "files": files}


def _resolve_path(value: str | None) -> Path:
    if not value:
        return Path.home().resolve()
    return Path(value).expanduser().resolve()


def _skip(path: Path, root: Path) -> bool:
    ignored = {".git", ".hg", ".svn", ".venv", "venv", "__pycache__"}
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in ignored or part.startswith(".") for part in relative_parts)

