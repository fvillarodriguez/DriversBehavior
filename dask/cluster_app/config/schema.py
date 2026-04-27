from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _expand(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


@dataclass(slots=True)
class ClusterConfig:
    name: str = "local-dask-cluster"
    scheduler_wait_seconds: int = 15
    preferred_scheduler: bool = False
    service_type: str = "auto"
    node_presence: bool = True
    presence_scan_seconds: int = 5
    presence_stale_seconds: int = 30
    presence_discovery_timeout_seconds: float = 5.0


@dataclass(slots=True)
class PathConfig:
    state_dir: Path = field(default_factory=lambda: _expand("~/.dask-cluster-app"))
    workspace_dir: Path = field(default_factory=lambda: _expand("~/DaskCluster/workspace"))
    envs_dir: Path = field(default_factory=lambda: _expand("~/DaskCluster/envs"))
    logs_dir: Path = field(default_factory=lambda: _expand("~/DaskCluster/logs"))


@dataclass(slots=True)
class NetworkConfig:
    host: str = "0.0.0.0"
    web_port: int = 8080
    dask_scheduler_port: int = 8786
    dask_dashboard_port: int = 8787
    worker_port_start: int = 9000
    worker_port_end: int = 9999
    auto_ports: bool = True
    discovery_service: str = "_dask-cluster._tcp.local."


@dataclass(slots=True)
class SecurityConfig:
    tls_required: bool = True
    cert_valid_days: int = 180
    token_autoapprove: bool = True
    ca_backup_replicas: bool = True


@dataclass(slots=True)
class JobConfig:
    single_active_job: bool = True
    warn_after_hours: int = 24
    retention_days: int = 30
    retry_count: int = 1
    copy_inputs: bool = True
    interrupt_on_time_limit: bool = False


@dataclass(slots=True)
class DaskConfig:
    worker_memory_fraction: float = 0.9
    pause_threshold: float = 0.8
    terminate_threshold: float = 0.95
    nanny: bool = True


@dataclass(slots=True)
class GPUConfig:
    cpu_fallback: bool = True
    strict_default: bool = False


@dataclass(slots=True)
class PathMappingConfig:
    enabled: bool = True
    auto_cwd: bool = True
    auto_home: bool = True
    mappings: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    jobs: JobConfig = field(default_factory=JobConfig)
    dask: DaskConfig = field(default_factory=DaskConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    path_mappings: PathMappingConfig = field(default_factory=PathMappingConfig)

    @property
    def db_path(self) -> Path:
        return self.paths.state_dir / "cluster.db"

    @property
    def certs_dir(self) -> Path:
        return self.paths.state_dir / "certs"

    @property
    def ca_backup_path(self) -> Path:
        return self.paths.state_dir / "ca-backup.enc"

    def ensure_directories(self) -> None:
        for path in (
            self.paths.state_dir,
            self.paths.workspace_dir,
            self.paths.envs_dir,
            self.paths.logs_dir,
            self.certs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def config_from_dict(raw: dict[str, Any] | None) -> AppConfig:
    raw = raw or {}
    cfg = AppConfig()
    for section_name, section_values in raw.items():
        if not hasattr(cfg, section_name) or not isinstance(section_values, dict):
            continue
        section = getattr(cfg, section_name)
        for key, value in section_values.items():
            if not hasattr(section, key):
                continue
            if section_name == "paths" and value is not None:
                value = _expand(value)
            if section_name == "path_mappings" and key == "mappings" and isinstance(value, dict):
                value = {str(alias): str(path) for alias, path in value.items()}
            setattr(section, key, value)
    return cfg


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["paths"] = {key: str(value) for key, value in data["paths"].items()}
    return data
