from pathlib import Path

import pytest

import venv_start


def test_resolve_torch_backend_auto_uses_cuda_on_windows_with_nvidia(monkeypatch):
    monkeypatch.setattr(venv_start.os, "name", "nt", raising=False)
    monkeypatch.setattr(venv_start, "_nvidia_smi_available", lambda: True)

    assert venv_start._resolve_torch_backend("auto") == "cu121"


def test_resolve_torch_backend_auto_uses_cpu_without_nvidia(monkeypatch):
    monkeypatch.setattr(venv_start.os, "name", "nt", raising=False)
    monkeypatch.setattr(venv_start, "_nvidia_smi_available", lambda: False)

    assert venv_start._resolve_torch_backend("auto") == "cpu"


def test_install_pyg_stack_cuda_uses_cuda_torch_and_pyg_indices(monkeypatch):
    calls = []

    def fake_run(cmd, *, check=True):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(venv_start, "_run", fake_run)

    backend = venv_start._install_pyg_stack(
        Path("python.exe"),
        torch_backend="cu121",
    )

    assert backend == "cu121"
    assert calls[0][-2:] == ["--index-url", "https://download.pytorch.org/whl/cu121"]
    assert "torch==2.4.1" in calls[0]
    assert calls[1][-2:] == [
        "-f",
        "https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    ]


def test_install_pyg_stack_cpu_uses_cpu_pyg_index_without_torch_index(monkeypatch):
    calls = []

    def fake_run(cmd, *, check=True):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(venv_start, "_run", fake_run)

    backend = venv_start._install_pyg_stack(
        Path("python"),
        torch_backend="cpu",
    )

    assert backend == "cpu"
    assert "--index-url" not in calls[0]
    assert calls[1][-2:] == [
        "-f",
        "https://data.pyg.org/whl/torch-2.4.0+cpu.html",
    ]


def test_resolve_torch_backend_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        venv_start._resolve_torch_backend("cuda")
