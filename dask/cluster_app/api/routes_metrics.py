from __future__ import annotations

from fastapi import APIRouter, Request

from cluster_app.api.app import services_from_request

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/status")
async def status(request: Request):
    services = services_from_request(request)
    active = services.jobs.active()
    scheduler = services.cluster_scheduler.status()
    return {
        "cluster": services.config.cluster.name,
        "queue_depth": services.jobs.queue_depth(),
        "active_job": active.id if active else None,
        "nodes": len(services.nodes.list()),
        "scheduler_running": scheduler["running"],
        "scheduler_address": scheduler["address"],
        "dask_dashboard_reachable": scheduler["dashboard_reachable"],
        "dask_dashboard": scheduler["dashboard"],
    }


@router.get("/node-details")
async def node_details(request: Request):
    services = services_from_request(request)
    rows = []
    for node in services.nodes.list():
        res = node.resources or {}
        cpu = res.get("CPU", 0)
        gpu_backends = sorted(
            k.replace("GPU_", "") for k in res if k.startswith("GPU_")
        )
        if res.get("GPU"):
            gpu_backends.append("generic")
        rows.append(
            {
                "uuid": node.uuid,
                "name": node.name,
                "host": node.host,
                "port": node.port,
                "status": node.status.value,
                "cpu": cpu,
                "ram_gb": round(res.get("RAM", 0) / (1024**3), 1) if res.get("RAM") else None,
                "gpu_backends": gpu_backends,
                "is_preferred_scheduler": node.is_preferred_scheduler,
                "last_seen_at": node.last_seen_at,
                "cert_fingerprint": node.cert_fingerprint,
            }
        )
    return {"nodes": rows}
