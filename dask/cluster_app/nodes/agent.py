from __future__ import annotations

import asyncio
from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.discovery.election import Candidate, should_self_promote
from cluster_app.discovery.manual_ip import local_ip
from cluster_app.discovery.mdns import MdnsPublisher, discover
from cluster_app.dask_runtime.resources import memory_limit
from cluster_app.dask_runtime.scheduler import SchedulerProcess
from cluster_app.dask_runtime.worker import WorkerProcess
from cluster_app.nodes.hardware import detect_hardware
from cluster_app.nodes.identity import load_node_identity, service_instance_name
from cluster_app.security.certs import ensure_node_cert
from cluster_app.storage.repositories import NodeRepository


class NodeAgent:
    def __init__(self, config: AppConfig, nodes: NodeRepository, identity_path: str | Path | None = None):
        self.config = config
        self.nodes = nodes
        self.identity_path = Path(identity_path or config.paths.state_dir / "node-id.json")
        self.identity = load_node_identity(self.identity_path)
        self.hardware = detect_hardware()

    async def start(self) -> None:
        host = local_ip()
        bundle = ensure_node_cert(
            self.config.certs_dir,
            self.identity.uuid,
            [host, self.identity.name, "localhost"],
            self.config.security.cert_valid_days,
        )
        self.nodes.upsert(
            uuid=self.identity.uuid,
            name=self.identity.name,
            host=host,
            port=self.config.network.web_port,
            resources=self.hardware.dask_resources(),
            cert_fingerprint=bundle.fingerprint,
            preferred=self.config.cluster.preferred_scheduler,
        )
        scheduler = await self.elect_scheduler(host)
        scheduler_process: SchedulerProcess | None = None
        worker_process: WorkerProcess | None = None
        scheduler_address = self._scheduler_address(host, scheduler)
        if scheduler:
            scheduler_process = SchedulerProcess(self.config, bundle)
            scheduler_process.start()
        worker_process = WorkerProcess(
            self.config,
            bundle,
            scheduler_address,
            self.hardware.dask_resources(),
            memory_limit(self.hardware.total_ram_bytes, self.config.dask.worker_memory_fraction),
        )
        worker_process.start()
        publisher = self._publisher(host, scheduler)
        publisher.start()
        try:
            while True:
                await asyncio.sleep(5)
                self.nodes.upsert(
                    self.identity.uuid,
                    self.identity.name,
                    host,
                    self.config.network.web_port,
                    self.hardware.dask_resources(),
                    bundle.fingerprint,
                    self.config.cluster.preferred_scheduler,
                )
        finally:
            publisher.close()
            if worker_process:
                worker_process.stop()
            if scheduler_process:
                scheduler_process.stop()

    async def elect_scheduler(self, host: str) -> bool:
        await asyncio.sleep(self.config.cluster.scheduler_wait_seconds)
        seen = []
        try:
            services = discover(
                self.config.network.discovery_service,
                timeout=self.config.cluster.presence_discovery_timeout_seconds,
            )
        except RuntimeError:
            services = []
        for service in services:
            node_uuid = service.properties.get("uuid")
            if node_uuid:
                seen.append(
                    Candidate(
                        uuid=node_uuid,
                        name=service.name,
                        host=service.host,
                        port=service.port,
                        preferred=service.properties.get("preferred") == "true",
                    )
                )
        self_candidate = Candidate(
            uuid=self.identity.uuid,
            name=self.identity.name,
            host=host,
            port=self.config.network.web_port,
            preferred=self.config.cluster.preferred_scheduler,
        )
        return should_self_promote(self_candidate, seen)

    def _publisher(self, host: str, scheduler: bool) -> MdnsPublisher:
        return MdnsPublisher(
            self.config.network.discovery_service,
            service_instance_name(self.identity),
            host,
            self.config.network.web_port,
            {
                "uuid": self.identity.uuid,
                "name": self.identity.name,
                "role": "agent",
                "scheduler": str(scheduler).lower(),
                "preferred": str(self.config.cluster.preferred_scheduler).lower(),
            },
        )

    def _scheduler_address(self, host: str, scheduler: bool) -> str:
        if scheduler:
            return f"tls://{host}:{self.config.network.dask_scheduler_port}"
        try:
            services = discover(
                self.config.network.discovery_service,
                timeout=self.config.cluster.presence_discovery_timeout_seconds,
            )
        except RuntimeError:
            services = []
        for service in services:
            if service.properties.get("scheduler") == "true":
                return f"tls://{service.host}:{self.config.network.dask_scheduler_port}"
        return f"tls://{host}:{self.config.network.dask_scheduler_port}"
