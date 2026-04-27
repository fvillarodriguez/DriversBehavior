from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field

from cluster_app.config.schema import AppConfig
from cluster_app.security.ca import CertificateAuthority
from cluster_app.services import AppServices, create_services
from cluster_app.shared_paths import (
    load_path_mappings,
    mapping_payload,
    path_specs,
    resolved_paths_exist,
    save_path_mappings,
    validate_mapping_payload,
)


class WorkerCertificateRequest(BaseModel):
    node_uuid: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    node_name: str | None = Field(default=None, max_length=255)
    hosts: list[str] = Field(default_factory=list, max_length=12)


class PathMappingsRequest(BaseModel):
    enabled: bool | None = None
    auto_cwd: bool | None = None
    auto_home: bool | None = None
    mappings: dict[str, str] | None = None


class ResolvePathsRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, max_length=200)


def create_control_plane_app(
    config: AppConfig | None = None,
    services: AppServices | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException
    except ModuleNotFoundError as exc:
        raise RuntimeError("fastapi is required to start the embedded control plane.") from exc

    app_services = services or create_services(config)
    app = FastAPI(title="Dask Cluster Control Plane", version="0.1.0")
    app.state.services = app_services

    @app.get("/api/nodes/self")
    async def self_node():
        if app_services.presence:
            node = app_services.nodes.get(app_services.presence.identity.uuid)
            if node:
                return asdict(node)
        nodes = app_services.nodes.list()
        if nodes:
            return asdict(nodes[0])
        raise HTTPException(status_code=404, detail="This node has not announced itself yet")

    @app.get("/api/admin/scheduler/status")
    async def scheduler_status():
        return app_services.scheduler.status()

    @app.post("/api/admin/worker/certificate")
    async def worker_certificate(payload: WorkerCertificateRequest):
        if not app_services.config.security.token_autoapprove:
            raise HTTPException(
                status_code=403,
                detail="Headless control plane requires token_autoapprove for worker certificates.",
            )
        if not app_services.scheduler.status().get("running"):
            raise HTTPException(status_code=409, detail="This node is not running a scheduler.")
        hosts = [host for host in payload.hosts if host]
        if payload.node_name:
            hosts.append(payload.node_name)
        ca = CertificateAuthority(app_services.config.certs_dir)
        bundle = ca.issue_node_certificate(
            payload.node_uuid,
            hosts,
            app_services.config.security.cert_valid_days,
        )
        return {
            "ca_cert": bundle.ca_cert.read_text(encoding="utf-8"),
            "cert": bundle.cert.read_text(encoding="utf-8"),
            "key": bundle.key.read_text(encoding="utf-8"),
            "fingerprint": bundle.fingerprint,
        }

    @app.get("/api/admin/path-mappings")
    async def get_path_mappings():
        return mapping_payload(app_services.config)

    @app.put("/api/admin/path-mappings")
    async def put_path_mappings(payload: PathMappingsRequest):
        raw = payload.model_dump(exclude_none=True)
        try:
            validate_mapping_payload(raw)
            save_path_mappings(app_services.config, raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return mapping_payload(app_services.config)

    @app.post("/api/admin/path-mappings/resolve")
    async def resolve_path_mappings(payload: ResolvePathsRequest):
        mappings = load_path_mappings(app_services.config)
        specs = path_specs(payload.paths, mappings)
        return {"paths": specs, "local": resolved_paths_exist(specs, mappings)}

    @app.on_event("startup")
    async def startup() -> None:
        if app_services.presence:
            await app_services.presence.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if app_services.presence:
            await app_services.presence.stop()

    return app


class EmbeddedControlPlane:
    def __init__(
        self,
        services: AppServices,
        host: str | None = None,
        port: int | None = None,
    ):
        self.services = services
        self.host = host or services.config.network.host
        self.port = port or services.config.network.web_port
        self.server: Any | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        try:
            import uvicorn
        except ModuleNotFoundError as exc:
            raise RuntimeError("uvicorn is required to start the embedded control plane.") from exc
        app = create_control_plane_app(services=self.services)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="dask-control-plane", daemon=True)
        self.thread.start()
        self._wait_until_started()

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.thread = None
        self.server = None

    def _wait_until_started(self) -> None:
        if self.server is None or self.thread is None:
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if getattr(self.server, "started", False):
                return
            if not self.thread.is_alive():
                return
            time.sleep(0.05)
