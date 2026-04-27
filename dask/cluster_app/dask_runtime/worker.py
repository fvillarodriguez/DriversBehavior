from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.dask_runtime.resources import dask_resource_flags
from cluster_app.discovery.manual_ip import local_ip
from cluster_app.security.ca import CertificateBundle
from cluster_app.security.tls_config import dask_tls_env
from cluster_app.shared_paths import (
    PATH_MAPPINGS_FILE_ENV,
    PATH_MAPPINGS_JSON_ENV,
    load_path_mappings,
    mapping_file,
)


@dataclass(slots=True)
class WorkerProcess:
    config: AppConfig
    bundle: CertificateBundle
    scheduler_address: str
    resources: dict[str, float]
    memory_limit_bytes: int
    process: subprocess.Popen[str] | None = None

    def start(self) -> subprocess.Popen[str]:
        if self.process and self.process.poll() is None:
            return self.process
        env = os.environ.copy()
        env.update(dask_tls_env(self.bundle))
        env["DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE"] = str(self.config.dask.pause_threshold)
        env["DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE"] = str(
            self.config.dask.terminate_threshold
        )
        env["CLUSTER_APP_STATE_DIR"] = str(self.config.paths.state_dir)
        env[PATH_MAPPINGS_FILE_ENV] = str(mapping_file(self.config))
        env[PATH_MAPPINGS_JSON_ENV] = json.dumps(asdict(load_path_mappings(self.config)))
        log_path = Path(self.config.paths.logs_dir) / "dask-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        worker_host = (
            local_ip()
            if self.config.network.host in {"0.0.0.0", "::"}
            else self.config.network.host
        )
        cmd = [
            sys.executable,
            "-m",
            "distributed.cli.dask_worker",
            self.scheduler_address,
            "--host",
            worker_host,
            "--tls-ca-file",
            str(self.bundle.ca_cert),
            "--tls-cert",
            str(self.bundle.cert),
            "--tls-key",
            str(self.bundle.key),
            "--nworkers",
            "1",
            "--nthreads",
            str(os.cpu_count() or 1),
            "--memory-limit",
            str(self.memory_limit_bytes),
            *dask_resource_flags(self.resources),
        ]
        if not self.config.dask.nanny:
            cmd.append("--no-nanny")
        self.process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return self.process

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
