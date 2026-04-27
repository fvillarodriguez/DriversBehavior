from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class JobPackage:
    source_dir: Path
    entrypoint: Path
    requirements: Path | None


def inspect_job_folder(source_dir: str | Path, entrypoint: str | None = None) -> JobPackage:
    source = Path(source_dir).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Job source folder does not exist: {source}")

    if entrypoint:
        entry = (source / entrypoint).resolve()
        if not entry.is_file() or source not in entry.parents:
            raise FileNotFoundError(f"Entrypoint must be a Python file inside the source folder: {entry}")
    else:
        scripts = sorted(path for path in source.rglob("*.py") if _is_user_script(path))
        if len(scripts) == 0:
            raise ValueError(f"No Python script found in job folder: {source}")
        if len(scripts) > 1:
            names = ", ".join(str(path.relative_to(source)) for path in scripts[:10])
            raise ValueError(f"Multiple Python scripts found; choose an entrypoint: {names}")
        entry = scripts[0]

    requirements = source / "requirements.txt"
    return JobPackage(source, entry.relative_to(source), requirements if requirements.exists() else None)


def _is_user_script(path: Path) -> bool:
    parts = set(path.parts)
    return "__pycache__" not in parts and ".venv" not in parts and "venv" not in parts

