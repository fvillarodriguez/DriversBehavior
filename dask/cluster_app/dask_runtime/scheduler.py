from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.security.ca import CertificateBundle
from cluster_app.security.tls_config import dask_tls_env


@dataclass(slots=True)
class SchedulerProcess:
    config: AppConfig
    bundle: CertificateBundle
    process: subprocess.Popen[str] | None = None

    @property
    def address(self) -> str:
        return f"tls://{self.config.network.host}:{self.config.network.dask_scheduler_port}"

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.config.network.host}:{self.config.network.dask_dashboard_port}/status"

    def start(self) -> subprocess.Popen[str]:
        if self.process and self.process.poll() is None:
            return self.process
        env = os.environ.copy()
        env.update(dask_tls_env(self.bundle))
        log_path = Path(self.config.paths.logs_dir) / "dask-scheduler.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "distributed.cli.dask_scheduler",
            "--protocol",
            "tls",
            "--host",
            self.config.network.host,
            "--port",
            str(self.config.network.dask_scheduler_port),
            "--dashboard-address",
            f":{self.config.network.dask_dashboard_port}",
            "--tls-ca-file",
            str(self.bundle.ca_cert),
            "--tls-cert",
            str(self.bundle.cert),
            "--tls-key",
            str(self.bundle.key),
        ]
        self.process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return self.process

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
