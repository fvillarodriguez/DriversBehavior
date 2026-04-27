from __future__ import annotations

from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.dask_runtime.resources import memory_limit as compute_memory_limit
from cluster_app.dask_runtime.worker import WorkerProcess
from cluster_app.discovery.manual_ip import local_ip
from cluster_app.nodes.identity import load_node_identity
from cluster_app.nodes.hardware import detect_hardware
from cluster_app.security.ca import CertificateBundle
from cluster_app.security.certs import ensure_node_cert


class WorkerRuntime:
    def __init__(self, config: AppConfig, identity_path: str | Path | None = None):
        self.config = config
        self.identity_path = Path(identity_path or config.paths.state_dir / "node-id.json")
        self.process: WorkerProcess | None = None

    def start(
        self,
        scheduler_address: str | None = None,
        cert_bundle: CertificateBundle | None = None,
    ) -> dict[str, object]:
        if self.process and self.process.process and self.process.process.poll() is None:
            return self.status()
        identity = load_node_identity(self.identity_path)
        host = local_ip()
        bundle = cert_bundle or ensure_node_cert(
            self.config.certs_dir,
            identity.uuid,
            [host, identity.name, "localhost", "127.0.0.1"],
            self.config.security.cert_valid_days,
        )
        hardware = detect_hardware()
        resources = hardware.dask_resources()
        mem_limit = compute_memory_limit(hardware.total_ram_bytes, self.config.dask.worker_memory_fraction)
        addr = scheduler_address or f"tls://{host}:{self.config.network.dask_scheduler_port}"
        self.process = WorkerProcess(self.config, bundle, addr, resources, mem_limit)
        self.process.start()
        return self.status()

    def stop(self) -> dict[str, object]:
        if self.process:
            self.process.stop()
        return self.status()

    def status(self) -> dict[str, object]:
        running = bool(
            self.process and self.process.process and self.process.process.poll() is None
        )
        return {
            "running": running,
            "pid": self.process.process.pid if running and self.process else None,
            "scheduler_address": self.process.scheduler_address if running and self.process else None,
        }
