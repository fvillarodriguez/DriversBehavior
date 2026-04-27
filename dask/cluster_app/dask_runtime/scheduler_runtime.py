from __future__ import annotations

import time
from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.dask_runtime.dashboard_proxy import dashboard_link
from cluster_app.dask_runtime.scheduler import SchedulerProcess
from cluster_app.discovery.manual_ip import local_ip, probe_tcp
from cluster_app.nodes.identity import load_node_identity
from cluster_app.security.certs import ensure_node_cert


class SchedulerRuntime:
    def __init__(self, config: AppConfig, identity_path: str | Path | None = None):
        self.config = config
        self.identity_path = Path(identity_path or config.paths.state_dir / "node-id.json")
        self.process: SchedulerProcess | None = None

    def start(self) -> dict[str, object]:
        if self.process and self.process.process and self.process.process.poll() is None:
            return self.status()
        identity = load_node_identity(self.identity_path)
        host = local_ip()
        scheduler_host = (
            host if self.config.network.host in {"0.0.0.0", "::"} else self.config.network.host
        )
        bundle = ensure_node_cert(
            self.config.certs_dir,
            identity.uuid,
            [host, scheduler_host, identity.name, "localhost", "127.0.0.1"],
            self.config.security.cert_valid_days,
        )
        self.process = SchedulerProcess(self.config, bundle)
        self.process.start()
        self._wait_until_reachable(scheduler_host)
        return self.status()

    def stop(self) -> dict[str, object]:
        if self.process:
            self.process.stop()
        return self.status()

    def status(self) -> dict[str, object]:
        process_running = bool(
            self.process and self.process.process and self.process.process.poll() is None
        )
        dashboard_host = (
            "127.0.0.1"
            if self.config.network.host in {"0.0.0.0", "::"}
            else self.config.network.host
        )
        dashboard_reachable = probe_tcp(
            dashboard_host,
            self.config.network.dask_dashboard_port,
            timeout=0.2,
        ).reachable
        scheduler_host = (
            local_ip()
            if self.config.network.host in {"0.0.0.0", "::"}
            else self.config.network.host
        )
        scheduler_reachable = probe_tcp(
            scheduler_host,
            self.config.network.dask_scheduler_port,
            timeout=0.2,
        ).reachable
        return {
            "running": scheduler_reachable,
            "managed": process_running,
            "pid": self.process.process.pid if process_running and self.process else None,
            "address": f"tls://{scheduler_host}:{self.config.network.dask_scheduler_port}",
            "dashboard": dashboard_link(
                self.config.network.host,
                self.config.network.dask_dashboard_port,
            ),
            "dashboard_reachable": dashboard_reachable,
            "scheduler_reachable": scheduler_reachable,
        }

    def _wait_until_reachable(self, host: str) -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if probe_tcp(host, self.config.network.dask_scheduler_port, timeout=0.2).reachable:
                return
            if self.process and self.process.process and self.process.process.poll() is not None:
                return
            time.sleep(0.2)
