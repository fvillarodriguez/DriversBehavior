#!/usr/bin/env python3
"""
Create or update the local virtualenv and install dependencies.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"


def _run(cmd: list[str], *, check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT_DIR)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def _find_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_python = os.environ.get("VENV_PYTHON")
    if env_python:
        return env_python
    for candidate in ("python3.12", "python3.11", "python3.10", "python3"):
        path = shutil.which(candidate)
        if path:
            return path
    return sys.executable


def _resolve_python_exec(python_exec: str) -> Path | None:
    if os.path.sep in python_exec or python_exec.startswith("."):
        candidate = Path(python_exec)
        return candidate if candidate.exists() else None
    resolved = shutil.which(python_exec)
    return Path(resolved) if resolved else None


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _python_version(python_exec: Path) -> str:
    cmd = [
        str(python_exec),
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
    ]
    result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()

def _parse_version(version: str) -> tuple[int, int] | None:
    try:
        major_str, minor_str = version.split(".", 1)
        return int(major_str), int(minor_str)
    except ValueError:
        return None

def _ensure_supported_python(python_exec: Path) -> None:
    version = _python_version(python_exec)
    parsed = _parse_version(version)
    if not parsed:
        return
    major, minor = parsed
    if major == 3 and minor >= 13:
        print(
            "Error: Python 3.13+ no es compatible con pyg-lib/torch-sparse.\n"
            "Use Python 3.12 para NeighborLoader."
        )
        raise SystemExit(1)

def _install_pyg_stack(python_exec: Path) -> None:
    _run([str(python_exec), "-m", "pip", "install", "torch==2.4.1"])
    _run(
        [
            str(python_exec),
            "-m",
            "pip",
            "install",
            "pyg-lib",
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
            "-f",
            "https://data.pyg.org/whl/torch-2.4.0+cpu.html",
        ]
    )


def _prompt_choice() -> str:
    print("El entorno virtual ya existe en .venv.")
    print("Opciones: [U] actualizar dependencias, [R] borrar y reinstalar, [S] salir")
    choice = input("Seleccione una opcion [U/R/S] (por defecto U): ").strip().lower()
    return choice or "u"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crear o actualizar el entorno virtual local."
    )
    parser.add_argument(
        "--python",
        help="Ruta o nombre del ejecutable de Python para crear el venv.",
    )
    args = parser.parse_args()

    if VENV_DIR.exists():
        choice = _prompt_choice()
        if choice.startswith("s"):
            print("Sin cambios.")
            return
        if choice.startswith("r"):
            print("Eliminando entorno virtual existente...")
            shutil.rmtree(VENV_DIR)
        elif not choice.startswith("u"):
            print("Opcion invalida. Cancelando.")
            return

    if not VENV_DIR.exists():
        python_exec = _find_python(args.python)
        resolved = _resolve_python_exec(python_exec)
        if resolved is None:
            print(
                "No se encontro el ejecutable de Python solicitado.\n"
                "Sugerencia (macOS): `brew install python@3.12` o instala desde python.org."
            )
            raise SystemExit(1)
        print(f"Creando venv con: {resolved}")
        _ensure_supported_python(resolved)
        _run([str(resolved), "-m", "venv", str(VENV_DIR)])

    venv_python = _venv_python(VENV_DIR)
    version = _python_version(venv_python)
    if version != "unknown":
        print(f"Python del venv: {version}")
        parsed = _parse_version(version)
        if parsed and parsed[0] == 3 and parsed[1] >= 13:
            print(
                "Error: Python 3.13+ no es compatible con pyg-lib/torch-sparse.\n"
                "Borre el venv y use Python 3.12."
            )
            raise SystemExit(1)

    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _install_pyg_stack(venv_python)
    _run([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"])

    print("Entorno virtual listo.")
    if os.name == "nt":
        print("Activar: .venv\\Scripts\\activate")
    else:
        print("Activar: source .venv/bin/activate")


if __name__ == "__main__":
    main()
