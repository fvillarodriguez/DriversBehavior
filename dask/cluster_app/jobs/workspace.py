from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluster_app.jobs.packager import JobPackage, inspect_job_folder


@dataclass(frozen=True, slots=True)
class JobWorkspace:
    root: Path
    code_dir: Path
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    metadata_path: Path
    entrypoint: Path
    requirements: Path | None


def prepare_workspace(
    workspace_root: str | Path,
    job_id: str,
    source_dir: str | Path,
    entrypoint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> JobWorkspace:
    package = inspect_job_folder(source_dir, entrypoint)
    root = Path(workspace_root).expanduser().resolve() / job_id
    if root.exists():
        shutil.rmtree(root)
    code_dir = root / "code"
    input_dir = root / "input"
    output_dir = root / "output"
    logs_dir = root / "logs"
    checkpoints_dir = root / "checkpoints"
    for path in (code_dir, input_dir, output_dir, logs_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)

    _copy_code(package, code_dir)
    entry = code_dir / package.entrypoint
    requirements = code_dir / "requirements.txt" if (code_dir / "requirements.txt").exists() else None
    metadata_path = root / "metadata.json"
    metadata_payload = {
        "job_id": job_id,
        "source_dir": str(package.source_dir),
        "entrypoint": str(package.entrypoint),
        "workspace": str(root),
        **(metadata or {}),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return JobWorkspace(root, code_dir, input_dir, output_dir, logs_dir, checkpoints_dir, metadata_path, entry, requirements)


def workspace_from_existing(root: str | Path, entrypoint: str) -> JobWorkspace:
    root = Path(root).expanduser().resolve()
    code_dir = root / "code"
    input_dir = root / "input"
    output_dir = root / "output"
    logs_dir = root / "logs"
    checkpoints_dir = root / "checkpoints"
    metadata_path = root / "metadata.json"
    entry = code_dir / entrypoint
    if not entry.exists():
        raise FileNotFoundError(f"Stored job entrypoint is missing: {entry}")
    requirements = code_dir / "requirements.txt" if (code_dir / "requirements.txt").exists() else None
    for path in (input_dir, output_dir, logs_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)
    return JobWorkspace(root, code_dir, input_dir, output_dir, logs_dir, checkpoints_dir, metadata_path, entry, requirements)


def _copy_code(package: JobPackage, code_dir: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "*.pyc",
        ".DS_Store",
    )
    for child in package.source_dir.iterdir():
        target = code_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, ignore=ignore)
        else:
            shutil.copy2(child, target)
