from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cluster_app.api.app import services_from_request
from cluster_app.api.routes_auth import current_user_id
from cluster_app.storage.models import JobStatus

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class SubmitJobRequest(BaseModel):
    source_dir: str
    entrypoint: str | None = None
    args: list[str] = []
    name: str | None = None


@router.get("")
async def list_jobs(request: Request):
    services = services_from_request(request)
    return [asdict(job) for job in services.jobs.list()]


@router.delete("/records")
async def clear_job_records(request: Request):
    current_user_id(request)
    services = services_from_request(request)
    removed = services.jobs.clear_finished_records()
    return {"removed": removed}


@router.post("")
async def submit_job(payload: SubmitJobRequest, request: Request):
    user_id = current_user_id(request)
    services = services_from_request(request)
    try:
        job = services.queue.submit(
            user_id=user_id,
            source_dir=payload.source_dir,
            entrypoint=payload.entrypoint,
            args=payload.args,
            name=payload.name,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return asdict(job)


@router.get("/{job_id}")
async def get_job(job_id: str, request: Request):
    services = services_from_request(request)
    job = services.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str, request: Request, after_id: int = 0):
    services = services_from_request(request)
    return [asdict(item) for item in services.jobs.logs(job_id, after_id=after_id)]


@router.get("/{job_id}/detail")
async def job_detail(job_id: str, request: Request):
    services = services_from_request(request)
    job = services.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    logs = [asdict(item) for item in services.jobs.logs(job_id, limit=200)]
    meta = job.metadata or {}
    stderr_lines = [line["message"] for line in logs if line["stream"] == "stderr"]
    failure = None
    if job.status == JobStatus.FAILED:
        failure = {
            "return_code": meta.get("return_code"),
            "error_message": stderr_lines[-1] if stderr_lines else None,
            "traceback": "\n".join(stderr_lines[-20:]) if stderr_lines else None,
        }
    duration = None
    if job.started_at and job.finished_at:
        from datetime import datetime
        try:
            duration = (datetime.fromisoformat(job.finished_at) - datetime.fromisoformat(job.started_at)).total_seconds()
        except Exception:
            pass
    queue_wait = None
    if job.created_at and job.started_at:
        from datetime import datetime
        try:
            queue_wait = (datetime.fromisoformat(job.started_at) - datetime.fromisoformat(job.created_at)).total_seconds()
        except Exception:
            pass
    return {
        "job": asdict(job),
        "logs": logs[-100:],
        "failure": failure,
        "duration_seconds": duration,
        "queue_wait_seconds": queue_wait,
    }


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request):
    current_user_id(request)
    services = services_from_request(request)
    job = services.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.RUNNING:
        services.queue.cancel_job(job_id)
    services.jobs.set_status(job_id, JobStatus.CANCELED)
    return {"ok": True}
