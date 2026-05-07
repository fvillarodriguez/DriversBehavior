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
PYTORCH_VERSION = "2.4.1"
PYG_TORCH_WHEEL_VERSION = "2.4.0"
PYG_PACKAGES = [
    "pyg-lib",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torch-spline-conv",
]
TORCH_BACKENDS = {
    "cpu": {
        "label": "CPU",
        "torch_index_url": None,
        "pyg_find_links": f"https://data.pyg.org/whl/torch-{PYG_TORCH_WHEEL_VERSION}+cpu.html",
    },
    "cu121": {
        "label": "CUDA 12.1",
        "torch_index_url": "https://download.pytorch.org/whl/cu121",
        "pyg_find_links": f"https://data.pyg.org/whl/torch-{PYG_TORCH_WHEEL_VERSION}+cu121.html",
    },
}


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


def _nvidia_smi_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0 and "GPU" in (result.stdout or result.stderr)


def _resolve_torch_backend(requested: str | None) -> str:
    requested_norm = (
        requested or os.environ.get("TORCH_BACKEND") or "auto"
    ).strip().lower()
    if requested_norm == "auto":
        if os.name == "nt" and _nvidia_smi_available():
            return "cu121"
        return "cpu"
    if requested_norm in TORCH_BACKENDS:
        return requested_norm
    allowed = ", ".join(["auto", *TORCH_BACKENDS.keys()])
    print(
        f"Error: backend de PyTorch no soportado: {requested!r}. "
        f"Use uno de: {allowed}."
    )
    raise SystemExit(1)


def _install_pyg_stack(python_exec: Path, *, torch_backend: str = "auto") -> str:
    backend = _resolve_torch_backend(torch_backend)
    cfg = TORCH_BACKENDS[backend]
    print(f"Instalando stack PyTorch/PyG para backend: {cfg['label']}")
    torch_cmd = [
        str(python_exec),
        "-m",
        "pip",
        "install",
        f"torch=={PYTORCH_VERSION}",
    ]
    if cfg["torch_index_url"]:
        torch_cmd += ["--index-url", str(cfg["torch_index_url"])]
    _run(torch_cmd)
    _run(
        [
            str(python_exec),
            "-m",
            "pip",
            "install",
            *PYG_PACKAGES,
            "-f",
            str(cfg["pyg_find_links"]),
        ]
    )
    return backend


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
    parser.add_argument(
        "--torch-backend",
        default=os.environ.get("TORCH_BACKEND", "auto"),
        choices=["auto", *TORCH_BACKENDS.keys()],
        help=(
            "Backend de PyTorch/PyG a instalar. "
            "`auto` usa CUDA 12.1 en Windows si nvidia-smi detecta una GPU NVIDIA; "
            "en otros casos usa CPU."
        ),
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

    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    backend = _install_pyg_stack(venv_python, torch_backend=args.torch_backend)
    _run([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"])

    print("Entorno virtual listo.")
    if backend == "cu121":
        print("Verificar CUDA:")
        print(
            "  python -c \"import torch; print(torch.__version__, torch.version.cuda, "
            "torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin CUDA')\""
        )
    if os.name == "nt":
        print("Activar: .venv\\Scripts\\activate")
    else:
        print("Activar: source .venv/bin/activate")


if __name__ == "__main__":
    main()
