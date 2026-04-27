from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    key: str
    path: Path
    python: Path
    requirements_hash: str


class DependencyInstaller:
    def __init__(self, envs_dir: str | Path, wheel_cache: str | Path | None = None):
        self.envs_dir = Path(envs_dir).expanduser().resolve()
        self.wheel_cache = Path(wheel_cache).expanduser().resolve() if wheel_cache else None
        self.envs_dir.mkdir(parents=True, exist_ok=True)

    def environment_key(self, requirements: str | Path | None, gpu_backend: str | None = None) -> str:
        requirements_hash = hash_requirements(requirements)
        payload = "|".join(
            [
                requirements_hash,
                f"python={sys.version_info.major}.{sys.version_info.minor}",
                f"system={platform.system()}",
                f"machine={platform.machine()}",
                f"gpu={gpu_backend or 'cpu'}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def ensure(self, requirements: str | Path | None, gpu_backend: str | None = None) -> EnvironmentInfo:
        key = self.environment_key(requirements, gpu_backend)
        env_path = self.envs_dir / key
        python = _python_path(env_path)
        if not python.exists():
            venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt").create(env_path)
        if requirements:
            marker = env_path / ".requirements.sha256"
            req_hash = hash_requirements(requirements)
            if not marker.exists() or marker.read_text(encoding="utf-8") != req_hash:
                self._install(python, Path(requirements))
                marker.write_text(req_hash, encoding="utf-8")
        else:
            req_hash = "empty"
        return EnvironmentInfo(key, env_path, python, req_hash)

    def _install(self, python: Path, requirements: Path) -> None:
        cmd = [str(python), "-m", "pip", "install", "-r", str(requirements)]
        if self.wheel_cache and self.wheel_cache.exists():
            cmd.extend(["--find-links", str(self.wheel_cache)])
        subprocess.run(cmd, check=True)


def hash_requirements(requirements: str | Path | None) -> str:
    if requirements is None:
        return "empty"
    path = Path(requirements)
    if not path.exists():
        return "missing"
    normalized = "\n".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _python_path(env_path: Path) -> Path:
    if os.name == "nt":
        return env_path / "Scripts" / "python.exe"
    return env_path / "bin" / "python"

