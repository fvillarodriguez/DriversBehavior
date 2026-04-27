from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cluster_app.config.loader import load_config
from cluster_app.config.schema import AppConfig
from cluster_app.control_plane import EmbeddedControlPlane
from cluster_app.dask_runtime.client import DaskClientFactory
from cluster_app.discovery.manual_ip import local_ip
from cluster_app.nodes.health import choose_free_port
from cluster_app.nodes.identity import load_node_identity
from cluster_app.security.ca import CertificateBundle
from cluster_app.security.certs import ensure_node_cert
from cluster_app.services import create_services
from cluster_app.shared_paths import (
    describe_path,
    load_path_mappings,
    mapping_payload,
    path_specs,
    resolve_path_spec,
    resolved_paths_exist,
    save_path_mappings,
)


class DaskCluster:
    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        auto_start: bool = True,
        start_local_worker: bool = True,
        control_plane: bool = True,
    ):
        self.config_path = config_path
        self.auto_start = auto_start
        self.start_local_worker = start_local_worker
        self.control_plane_enabled = control_plane
        self.config = load_config(config_path)
        self._configure_auto_ports(self.config)
        self.services = create_services(self.config)
        self._control_plane: EmbeddedControlPlane | None = None
        self._client: Any | None = None
        self._started = False

    def start(self) -> dict[str, Any]:
        if self._started:
            return self.status()
        self.start_scheduler()
        if self.start_local_worker:
            self.start_worker()
        self._started = True
        return self.status()

    def start_scheduler(self) -> dict[str, Any]:
        self._ensure_control_plane()
        self._scan_presence_once()
        scheduler_status = self.services.cluster_scheduler.status()
        if scheduler_status.get("running"):
            self._started = True
            return scheduler_status
        status = self.services.scheduler.start()
        self._started = True
        return status

    def stop_scheduler(self) -> dict[str, Any]:
        if self._client is not None:
            self._client.close()
            self._client = None
        status = self.services.scheduler.stop()
        self._started = bool(self.services.worker.status().get("running"))
        return status

    def start_worker(self) -> dict[str, Any]:
        self._ensure_control_plane()
        self._scan_presence_once()
        active_scheduler = self.services.cluster_scheduler.status()
        if not active_scheduler.get("running"):
            raise RuntimeError("Start a scheduler before starting a worker.")
        cert_bundle = self.services.cluster_scheduler.remote_worker_certificate(active_scheduler)
        status = self.services.worker.start(str(active_scheduler["address"]), cert_bundle=cert_bundle)
        self._started = True
        return status

    def stop_worker(self) -> dict[str, Any]:
        status = self.services.worker.stop()
        self._started = bool(self.services.cluster_scheduler.status().get("running"))
        return status

    def client(self):
        if self._client is not None:
            return self._client
        if not self._started:
            if not self.auto_start:
                raise RuntimeError("DaskCluster is not started. Call start() first.")
            self.start()
        scheduler_status = self.services.cluster_scheduler.status()
        if not scheduler_status.get("running"):
            raise RuntimeError("No running Dask scheduler is available.")
        bundle = self._client_certificate_bundle(scheduler_status)
        self._client = DaskClientFactory(str(scheduler_status["address"]), bundle).connect()
        return self._client

    def submit(self, fn, *args, **kwargs):
        return self.client().submit(fn, *args, **kwargs)

    def map(self, fn, iterable, *iterables, **kwargs):
        return self.client().map(fn, iterable, *iterables, **kwargs)

    def gather(self, futures):
        return self.client().gather(futures)

    def status(self) -> dict[str, Any]:
        scheduler = self.services.cluster_scheduler.status()
        worker = self.services.worker.status()
        return {
            "cluster": self.config.cluster.name,
            "started": self._started,
            "control_plane": self._control_plane.url if self._control_plane else None,
            "scheduler": scheduler,
            "worker": worker,
            "dashboard": scheduler.get("dashboard"),
            "client_connected": self._client is not None,
            "nodes": [asdict(node) for node in self.services.nodes.list()],
        }

    @property
    def dashboard_url(self) -> str | None:
        dashboard = self.status().get("dashboard")
        return str(dashboard) if dashboard else None

    def nodes(self) -> list[dict[str, Any]]:
        return [asdict(node) for node in self.services.nodes.list()]

    def path_mappings(self) -> dict[str, Any]:
        return mapping_payload(self.config)

    def configure_path_mappings(
        self,
        mappings: dict[str, str],
        *,
        enabled: bool | None = None,
        auto_cwd: bool | None = None,
        auto_home: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"mappings": mappings}
        if enabled is not None:
            payload["enabled"] = enabled
        if auto_cwd is not None:
            payload["auto_cwd"] = auto_cwd
        if auto_home is not None:
            payload["auto_home"] = auto_home
        save_path_mappings(self.config, payload)
        return self.path_mappings()

    def resolve_shared_path(self, path: str | Path) -> Path:
        mappings = load_path_mappings(self.config)
        spec = describe_path(path, mappings)
        return resolve_path_spec(spec, mappings)

    def validate_shared_paths(self, paths: list[str | Path]) -> dict[str, Any]:
        mappings = load_path_mappings(self.config)
        specs = path_specs(paths, mappings)
        local = resolved_paths_exist(specs, mappings)
        workers: dict[str, dict[str, dict[str, Any]]] = {}
        errors: dict[str, str] = {}
        if self._started or self.auto_start:
            try:
                workers = self.client().run(resolved_paths_exist, specs)
            except Exception as exc:
                errors["workers"] = str(exc)
        ok = all(item["exists"] for item in local.values()) and all(
            all(item["exists"] for item in result.values()) for result in workers.values()
        )
        if errors:
            ok = False
        return {"ok": ok, "paths": specs, "local": local, "workers": workers, "errors": errors}

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self.services.worker.stop()
        self.services.scheduler.stop()
        if self._control_plane is not None:
            self._control_plane.stop()
            self._control_plane = None
        elif self.services.presence:
            self._stop_presence()
        self._started = False

    def _client_certificate_bundle(self, scheduler_status: dict[str, Any]) -> CertificateBundle:
        if not scheduler_status.get("local", True):
            bundle = self.services.cluster_scheduler.remote_worker_certificate(scheduler_status)
            if bundle is None:
                raise RuntimeError("Remote scheduler did not provide a client certificate.")
            return bundle
        identity = load_node_identity(self.config.paths.state_dir / "node-id.json")
        host = local_ip()
        scheduler_host = (
            host if self.config.network.host in {"0.0.0.0", "::"} else self.config.network.host
        )
        return ensure_node_cert(
            self.config.certs_dir,
            identity.uuid,
            [host, scheduler_host, identity.name, "localhost", "127.0.0.1"],
            self.config.security.cert_valid_days,
        )

    def _ensure_control_plane(self) -> None:
        if not self.control_plane_enabled or self._control_plane is not None:
            return
        self._control_plane = EmbeddedControlPlane(self.services)
        self._control_plane.start()

    def _scan_presence_once(self) -> None:
        if self.services.presence is None:
            return
        _run_async_blocking(self.services.presence.scan_once())

    def _stop_presence(self) -> None:
        if self.services.presence is None:
            return
        _run_async_blocking(self.services.presence.stop())

    @staticmethod
    def _configure_auto_ports(config: AppConfig) -> None:
        if not config.network.auto_ports:
            return
        bind_host = config.network.host
        config.network.web_port = choose_free_port(bind_host, config.network.web_port)
        config.network.dask_scheduler_port = choose_free_port(
            bind_host,
            config.network.dask_scheduler_port,
        )
        config.network.dask_dashboard_port = choose_free_port(
            bind_host,
            config.network.dask_dashboard_port,
        )


def _run_async_blocking(coro) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return

    result: list[BaseException | None] = [None]

    def runner() -> None:
        try:
            asyncio.run(coro)
        except BaseException as exc:
            result[0] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if result[0] is not None:
        raise result[0]
