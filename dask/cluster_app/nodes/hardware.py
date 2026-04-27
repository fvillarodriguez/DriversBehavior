from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    hostname: str
    system: str
    machine: str
    cpu_count: int
    total_ram_bytes: int
    gpu_backends: list[str]

    def dask_resources(self) -> dict[str, float]:
        resources: dict[str, float] = {"CPU": float(self.cpu_count)}
        if self.gpu_backends:
            resources["GPU"] = 1.0
        for backend in self.gpu_backends:
            resources[f"GPU_{backend.upper()}"] = 1.0
        return resources


def detect_hardware() -> HardwareProfile:
    return HardwareProfile(
        hostname=platform.node() or "unknown-node",
        system=platform.system(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        total_ram_bytes=_total_ram(),
        gpu_backends=_gpu_backends(),
    )


def _total_ram() -> int:
    try:
        import psutil
    except ModuleNotFoundError:
        if platform.system() == "Darwin":
            try:
                return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            except (OSError, subprocess.CalledProcessError, ValueError):
                return 0
        return 0
    return int(psutil.virtual_memory().total)


def _gpu_backends() -> list[str]:
    backends: list[str] = []
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        if importlib.util.find_spec("mlx") is not None:
            backends.append("mlx")
        if _torch_mps_available():
            backends.append("mps")
    if system == "Windows" and shutil.which("nvidia-smi"):
        backends.append("cuda")
    return backends


def _torch_mps_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

