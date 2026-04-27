from __future__ import annotations

import socket
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cluster_app.discovery.manual_ip import local_ip, probe_tcp


@dataclass(frozen=True, slots=True)
class DiagnosticStep:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class DiagnosticReport:
    steps: list[DiagnosticStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.steps.append(DiagnosticStep(name, ok, detail))


def run_network_diagnostics(host: str, tcp_port: int, tls_port: int | None = None) -> DiagnosticReport:
    report = DiagnosticReport()
    ip = local_ip()
    report.add("local_ip", ip != "127.0.0.1", ip)
    tcp = probe_tcp(host, tcp_port)
    report.add("tcp_connectivity", tcp.reachable, tcp.error or f"{host}:{tcp_port} reachable")
    if tls_port is not None:
        report.add("tls_connectivity", _probe_tls(host, tls_port), f"{host}:{tls_port}")
    report.add("file_transfer_probe", _probe_small_file(), "local small-file write/read")
    return report


def choose_free_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        bind_host = host if host != "0.0.0.0" else ""
        if preferred <= 0:
            sock.bind((bind_host, 0))
            return int(sock.getsockname()[1])
        try:
            sock.bind((bind_host, preferred))
            return preferred
        except OSError:
            sock.bind((bind_host, 0))
            return int(sock.getsockname()[1])


def _probe_tls(host: str, port: int) -> bool:
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=2.0) as raw:
            with context.wrap_socket(raw, server_hostname=host):
                return True
    except OSError:
        return False


def _probe_small_file() -> bool:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.bin"
            path.write_bytes(b"dask-cluster-app-probe")
            return path.read_bytes() == b"dask-cluster-app-probe"
    except OSError:
        return False
