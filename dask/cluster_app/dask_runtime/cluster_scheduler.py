from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from cluster_app.config.schema import AppConfig
from cluster_app.dask_runtime.scheduler_runtime import SchedulerRuntime
from cluster_app.discovery.manual_ip import local_ip, probe_tcp
from cluster_app.nodes.identity import load_node_identity
from cluster_app.security.ca import CertificateBundle
from cluster_app.storage.models import Node, NodeStatus
from cluster_app.storage.repositories import NodeRepository


class ClusterSchedulerResolver:
    def __init__(
        self,
        config: AppConfig,
        nodes: NodeRepository,
        local_scheduler: SchedulerRuntime,
    ):
        self.config = config
        self.nodes = nodes
        self.local_scheduler = local_scheduler

    def status(self) -> dict[str, Any]:
        local = self._local_status()
        if local.get("running"):
            return local
        remote = self._remote_status()
        return remote or local

    def remote_worker_certificate(self, scheduler_status: dict[str, Any]) -> CertificateBundle | None:
        if scheduler_status.get("local", True):
            return None
        node_host = scheduler_status.get("node_host")
        node_port = int(scheduler_status.get("node_port") or 0)
        if not node_host or not node_port:
            raise RuntimeError("Remote scheduler did not include its web endpoint.")

        identity = load_node_identity(self.config.paths.state_dir / "node-id.json")
        host = local_ip()
        payload = {
            "node_uuid": identity.uuid,
            "node_name": identity.name,
            "hosts": [host, identity.name, "localhost", "127.0.0.1"],
        }
        response = _json_request(
            f"http://{node_host}:{node_port}/api/admin/worker/certificate",
            method="POST",
            payload=payload,
            timeout=5.0,
        )
        scheduler_id = str(scheduler_status.get("node_uuid") or node_host)
        cert_dir = self.config.certs_dir / "remote-schedulers" / _safe_name(scheduler_id)
        cert_dir.mkdir(parents=True, exist_ok=True)
        ca_path = cert_dir / "ca.pem"
        cert_path = cert_dir / f"{identity.uuid}.pem"
        key_path = cert_dir / f"{identity.uuid}-key.pem"
        ca_path.write_text(_required_text(response, "ca_cert"), encoding="utf-8")
        cert_path.write_text(_required_text(response, "cert"), encoding="utf-8")
        key_path.write_text(_required_text(response, "key"), encoding="utf-8")
        return CertificateBundle(
            ca_cert=ca_path,
            cert=cert_path,
            key=key_path,
            fingerprint=str(response.get("fingerprint") or ""),
        )

    def _local_status(self) -> dict[str, Any]:
        status = dict(self.local_scheduler.status())
        scheduler_reachable = _service_reachable(str(status.get("address") or ""))
        status["scheduler_reachable"] = scheduler_reachable
        status["running"] = scheduler_reachable
        status.update(
            {
                "local": True,
                "node_uuid": None,
                "node_name": "This node",
                "node_host": local_ip(),
                "node_port": self.config.network.web_port,
            }
        )
        return status

    def _remote_status(self) -> dict[str, Any] | None:
        candidates = [
            node
            for node in self.nodes.list()
            if node.status == NodeStatus.ONLINE and not self._is_local_node(node)
        ]
        candidates.sort(key=lambda node: (not node.is_preferred_scheduler, node.name, node.uuid))
        for node in candidates:
            status = self._fetch_remote_status(node)
            if status and status.get("running"):
                return status
        return None

    def _fetch_remote_status(self, node: Node) -> dict[str, Any] | None:
        port = int(node.port or self.config.network.web_port)
        try:
            payload = _json_request(
                f"http://{node.host}:{port}/api/admin/scheduler/status",
                timeout=1.5,
            )
        except RuntimeError:
            return None
        if not isinstance(payload, dict):
            return None
        status = dict(payload)
        status["address"] = _public_service_url(str(status.get("address") or ""), node.host)
        scheduler_reachable = _service_reachable(str(status["address"]))
        dashboard = _public_service_url(str(status.get("dashboard") or ""), node.host)
        status["dashboard"] = dashboard
        status["scheduler_reachable"] = scheduler_reachable
        status["running"] = bool(status.get("running") and scheduler_reachable)
        status["dashboard_reachable"] = _dashboard_reachable(dashboard)
        status.update(
            {
                "local": False,
                "node_uuid": node.uuid,
                "node_name": node.name,
                "node_host": node.host,
                "node_port": port,
            }
        )
        return status

    def _is_local_node(self, node: Node) -> bool:
        local_hosts = {"127.0.0.1", "localhost", local_ip()}
        return node.host in local_hosts and int(node.port or 0) == self.config.network.web_port


def _json_request(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = UrlRequest(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Could not reach {url}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{url} returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{url} returned invalid payload")
    return parsed


def _public_service_url(value: str, host: str) -> str:
    if "://" not in value:
        return value
    parsed = urlparse(value)
    if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::"}:
        return value
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _dashboard_reachable(url: str) -> bool:
    return _service_reachable(url)


def _service_reachable(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        return False
    return probe_tcp(parsed.hostname, parsed.port, timeout=0.3).reachable


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Remote scheduler did not return {key}.")
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "scheduler"
