from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cluster_app.config.schema import AppConfig
from cluster_app.discovery.manual_ip import local_ip
from cluster_app.discovery.mdns import MdnsPublisher, discover
from cluster_app.nodes.hardware import detect_hardware
from cluster_app.nodes.identity import load_node_identity, service_instance_name
from cluster_app.security.certs import ensure_node_cert
from cluster_app.storage.repositories import NodeRepository

LOGGER = logging.getLogger(__name__)


class NodePresenceMonitor:
    def __init__(
        self,
        config: AppConfig,
        nodes: NodeRepository,
        identity_path: str | Path | None = None,
    ):
        self.config = config
        self.nodes = nodes
        self.identity_path = Path(identity_path or config.paths.state_dir / "node-id.json")
        self.identity = load_node_identity(self.identity_path)
        self.hardware = detect_hardware()
        self.publisher: MdnsPublisher | None = None
        self.task: asyncio.Task[None] | None = None
        self.running = False

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        await asyncio.to_thread(self._publish_local_node)
        self.task = asyncio.create_task(self._loop(), name="node-presence-monitor")

    async def stop(self) -> None:
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        if self.publisher:
            publisher = self.publisher
            self.publisher = None
            await asyncio.to_thread(publisher.close)

    async def scan_once(self) -> None:
        self._upsert_local_node()
        try:
            services = await asyncio.to_thread(
                discover,
                self.config.network.discovery_service,
                self.config.cluster.presence_discovery_timeout_seconds,
            )
        except RuntimeError as exc:
            LOGGER.debug("mDNS discovery unavailable: %s", exc)
            services = []
        for service in services:
            node_uuid = service.properties.get("uuid")
            if not node_uuid:
                continue
            resources = _loads(service.properties.get("resources_json"), {})
            self.nodes.upsert(
                uuid=node_uuid,
                name=service.properties.get("name") or service.name.removesuffix(
                    f".{self.config.network.discovery_service}"
                ),
                host=service.host,
                port=service.port,
                resources=resources,
                cert_fingerprint=service.properties.get("cert_fingerprint"),
                preferred=service.properties.get("preferred") == "true",
            )
        cutoff = datetime.now(UTC) - timedelta(seconds=self.config.cluster.presence_stale_seconds)
        self.nodes.mark_stale_offline(cutoff.isoformat(timespec="seconds"))

    async def _loop(self) -> None:
        while self.running:
            await self.scan_once()
            await asyncio.sleep(max(1, self.config.cluster.presence_scan_seconds))

    def _publish_local_node(self) -> None:
        host = local_ip()
        bundle = ensure_node_cert(
            self.config.certs_dir,
            self.identity.uuid,
            [host, self.identity.name, "localhost"],
            self.config.security.cert_valid_days,
        )
        resources = self.hardware.dask_resources()
        self.nodes.upsert(
            uuid=self.identity.uuid,
            name=self.identity.name,
            host=host,
            port=self.config.network.web_port,
            resources=resources,
            cert_fingerprint=bundle.fingerprint,
            preferred=self.config.cluster.preferred_scheduler,
        )
        properties = {
            "uuid": self.identity.uuid,
            "name": self.identity.name,
            "role": "web",
            "scheduler": "false",
            "preferred": str(self.config.cluster.preferred_scheduler).lower(),
            "cert_fingerprint": bundle.fingerprint,
            "resources_json": json.dumps(resources, separators=(",", ":")),
        }
        try:
            self.publisher = MdnsPublisher(
                self.config.network.discovery_service,
                service_instance_name(self.identity),
                host,
                self.config.network.web_port,
                properties,
            )
            self.publisher.start()
        except RuntimeError as exc:
            LOGGER.warning("mDNS publisher unavailable: %s", exc)
            self.publisher = None
        except OSError as exc:
            LOGGER.warning("Could not publish mDNS presence: %s", exc)
            self.publisher = None

    def _upsert_local_node(self) -> None:
        host = local_ip()
        self.nodes.upsert(
            uuid=self.identity.uuid,
            name=self.identity.name,
            host=host,
            port=self.config.network.web_port,
            resources=self.hardware.dask_resources(),
            preferred=self.config.cluster.preferred_scheduler,
        )


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
