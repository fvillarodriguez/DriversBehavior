from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any
from urllib.error import URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from cluster_app.api.app import services_from_request
from cluster_app.api.routes_auth import current_user_id
from cluster_app.nodes.health import probe_tcp

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


class ManualNodeRequest(BaseModel):
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=18080, ge=1, le=65535)


@router.get("")
async def list_nodes(request: Request):
    services = services_from_request(request)
    return [asdict(node) for node in services.nodes.list()]


@router.get("/self")
async def self_node(request: Request):
    services = services_from_request(request)
    if services.presence:
        node = services.nodes.get(services.presence.identity.uuid)
        if node:
            return asdict(node)
    nodes = services.nodes.list()
    if nodes:
        return asdict(nodes[0])
    raise HTTPException(status_code=404, detail="This node has not announced itself yet")


@router.post("/manual")
async def add_manual_node(payload: ManualNodeRequest, request: Request):
    services = services_from_request(request)
    remote = await asyncio.to_thread(_fetch_remote_node, payload.host, payload.port)
    node_host = remote.get("host") or payload.host
    if node_host in {"127.0.0.1", "0.0.0.0", "localhost"}:
        node_host = payload.host
    node = services.nodes.upsert(
        uuid=_required(remote, "uuid"),
        name=str(remote.get("name") or payload.host),
        host=str(node_host),
        port=int(remote.get("port") or payload.port),
        resources=remote.get("resources") if isinstance(remote.get("resources"), dict) else {},
        cert_fingerprint=remote.get("cert_fingerprint"),
        preferred=bool(remote.get("is_preferred_scheduler")),
    )
    return asdict(node)


@router.post("/cleanup-old")
async def cleanup_old_nodes(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    removed = services.nodes.delete_inactive()
    return {"removed": removed}


@router.post("/{node_uuid}/revoke")
async def revoke_node(node_uuid: str, request: Request):
    current_user_id(request)
    services = services_from_request(request)
    services.nodes.revoke(node_uuid)
    return {"ok": True}


@router.get("/health")
async def health_check(request: Request):
    services = services_from_request(request)
    results = []
    for node in services.nodes.list():
        is_self = _is_self_node(services, node.uuid)
        web_port = services.config.network.web_port if is_self else node.port
        web_port = web_port or services.config.network.web_port
        probe_host = "127.0.0.1" if is_self else node.host
        tcp = probe_tcp(probe_host, web_port)
        results.append({
            "uuid": node.uuid,
            "name": node.name,
            "host": node.host,
            "port": web_port,
            "checked_host": probe_host,
            "status": node.status.value,
            "reachable": tcp.reachable,
            "error": tcp.error,
            "resources": node.resources or {},
        })
    return {"nodes": results}


def _fetch_remote_node(host: str, port: int) -> dict[str, Any]:
    if "://" in host or "/" in host:
        raise HTTPException(status_code=400, detail="Use only the host or IP, without http://")
    url = f"http://{host}:{port}/api/nodes/self"
    request = UrlRequest(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=3) as response:
            body = response.read().decode("utf-8")
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach node at {host}:{port}") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Remote node returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Remote node returned invalid payload")
    return payload


def _required(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not value:
        raise HTTPException(status_code=502, detail=f"Remote node did not include {key}")
    return str(value)


def _is_self_node(services: Any, node_uuid: str) -> bool:
    return bool(services.presence and services.presence.identity.uuid == node_uuid)
