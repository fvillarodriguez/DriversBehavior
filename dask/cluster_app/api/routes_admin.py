from __future__ import annotations

import sys
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from cluster_app.api.app import services_from_request
from cluster_app.api.routes_auth import current_user_id, forget_user_sessions, online_user_activity
from cluster_app.security.ca import CertificateAuthority
from cluster_app.nodes.firewall import plan_firewall
from cluster_app.nodes.service import plan_service_install
from cluster_app.storage.repositories import UserHasActiveJobsError

router = APIRouter(prefix="/api/admin", tags=["admin"])


class WorkerCertificateRequest(BaseModel):
    node_uuid: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    node_name: str | None = Field(default=None, max_length=255)
    hosts: list[str] = Field(default_factory=list, max_length=12)


@router.get("/firewall-plan")
async def firewall_plan(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    ports = [
        services.config.network.web_port,
        services.config.network.dask_scheduler_port,
        services.config.network.dask_dashboard_port,
    ]
    return asdict(plan_firewall(sys.executable, ports))


@router.get("/service-plan")
async def service_plan(request: Request):
    current_user_id(request)
    return asdict(plan_service_install())


@router.get("/users")
async def users(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    rows = []
    for user in services.users.list():
        row = asdict(user)
        row.pop("password_hash", None)
        rows.append(row)
    return {"users": rows}


@router.get("/users/online")
async def online_users(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    activity = online_user_activity(request)
    rows = []
    for user in services.users.list():
        if user.id not in activity:
            continue
        row = asdict(user)
        row.pop("password_hash", None)
        row["last_seen_at"] = activity[int(user.id or 0)]
        rows.append(row)
    return {"users": rows}


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, request: Request):
    requester_id = current_user_id(request)
    services = services_from_request(request)
    if services.users.get(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        deleted = services.users.delete(user_id)
    except UserHasActiveJobsError as exc:
        raise HTTPException(
            status_code=409,
            detail="Stop or clear this user's queued/running jobs before deleting the user.",
        ) from exc
    forget_user_sessions(request, user_id)
    return {"ok": deleted, "deleted": user_id, "self_deleted": requester_id == user_id}


@router.get("/scheduler/status")
async def scheduler_status(request: Request):
    services = services_from_request(request)
    return services.scheduler.status()


@router.post("/scheduler/start")
async def scheduler_start(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    active = services.cluster_scheduler.status()
    if active.get("running") and not active.get("local", True):
        return active
    try:
        return services.scheduler.start()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not start scheduler: {exc}") from exc


@router.post("/scheduler/stop")
async def scheduler_stop(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    return services.scheduler.stop()


@router.get("/worker/status")
async def worker_status(request: Request):
    services = services_from_request(request)
    return services.worker.status()


@router.post("/worker/start")
async def worker_start(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    scheduler_status = services.cluster_scheduler.status()
    if not scheduler_status.get("running"):
        raise HTTPException(status_code=409, detail="Start a scheduler before starting a worker.")
    scheduler_address = str(scheduler_status["address"])
    try:
        cert_bundle = services.cluster_scheduler.remote_worker_certificate(scheduler_status)
        return services.worker.start(scheduler_address, cert_bundle=cert_bundle)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not start worker: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/worker/stop")
async def worker_stop(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    return services.worker.stop()


@router.post("/worker/certificate")
async def worker_certificate(payload: WorkerCertificateRequest, request: Request):
    services = services_from_request(request)
    if not services.config.security.token_autoapprove:
        current_user_id(request)
    if not services.scheduler.status().get("running"):
        raise HTTPException(status_code=409, detail="This node is not running a scheduler.")
    hosts = [host for host in payload.hosts if host]
    if payload.node_name:
        hosts.append(payload.node_name)
    ca = CertificateAuthority(services.config.certs_dir)
    bundle = ca.issue_node_certificate(
        payload.node_uuid,
        hosts,
        services.config.security.cert_valid_days,
    )
    return {
        "ca_cert": bundle.ca_cert.read_text(encoding="utf-8"),
        "cert": bundle.cert.read_text(encoding="utf-8"),
        "key": bundle.key.read_text(encoding="utf-8"),
        "fingerprint": bundle.fingerprint,
    }
