from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TcpProbeResult:
    host: str
    port: int
    reachable: bool
    error: str | None = None


def probe_tcp(host: str, port: int, timeout: float = 2.0) -> TcpProbeResult:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return TcpProbeResult(host, port, True)
    except OSError as exc:
        return TcpProbeResult(host, port, False, str(exc))


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()

