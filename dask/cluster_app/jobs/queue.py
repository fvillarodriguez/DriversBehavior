from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.jobs.runner import JobRunner
from cluster_app.jobs.workspace import prepare_workspace, workspace_from_existing
from cluster_app.storage.models import Job, JobStatus
from cluster_app.storage.repositories import JobRepository

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SchedulerConnection:
    address: str | None
    certs_dir: Path | None


class JobQueueManager:
    def __init__(self, config: AppConfig, jobs: JobRepository, scheduler_runtime=None):
        self.config = config
        self.jobs = jobs
        self.runner = JobRunner(config, jobs)
        self.scheduler_runtime = scheduler_runtime
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    def _scheduler_connection(self) -> SchedulerConnection:
        if self.scheduler_runtime is None:
            return SchedulerConnection(None, self.config.certs_dir)
        status = self.scheduler_runtime.status()
        if not status.get("running"):
            return SchedulerConnection(None, self.config.certs_dir)
        certs_dir = self.config.certs_dir
        remote_certificate = getattr(self.scheduler_runtime, "remote_worker_certificate", None)
        if callable(remote_certificate):
            bundle = remote_certificate(status)
            if bundle:
                certs_dir = bundle.ca_cert.parent
        return SchedulerConnection(str(status["address"]), certs_dir)

    def submit(
        self,
        user_id: int,
        source_dir: str | Path,
        entrypoint: str | None,
        args: list[str] | None = None,
        name: str | None = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        package_name = name or Path(source_dir).resolve().name
        package = prepare_workspace(
            self.config.paths.workspace_dir,
            job_id,
            source_dir,
            entrypoint,
            metadata={"submitted_args": args or []},
        )
        return self.jobs.create(
            job_id=job_id,
            user_id=user_id,
            name=package_name,
            source_path=str(Path(source_dir).expanduser().resolve()),
            entrypoint=str(package.entrypoint.relative_to(package.code_dir)),
            args=args or [],
            retries_left=self.config.jobs.retry_count,
            metadata={"workspace": str(package.root), "original_source": str(Path(source_dir).resolve())},
            workspace_path=str(package.root),
        )

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            await self._task

    async def run_once(self) -> bool:
        if self.jobs.active() is not None:
            self._warn_if_needed()
            return False
        job = self.jobs.next_queued()
        if job is None:
            return False
        workspace = workspace_from_existing(job.workspace_path or job.metadata["workspace"], job.entrypoint)
        connection = self._scheduler_connection()
        result = await self.runner.run(
            job.id,
            workspace,
            job.args,
            connection.address,
            dask_certs_dir=connection.certs_dir,
        )
        if result.status == JobStatus.FAILED and job.retries_left > 0:
            self.jobs.decrement_retry(job.id)
            self.jobs.requeue(job.id, {"return_code": result.return_code})
        return True

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception:
                LOGGER.exception("Job queue iteration failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=2.0)
            except TimeoutError:
                pass

    def cancel_job(self, job_id: str) -> None:
        self.runner.cancel(job_id)

    def _warn_if_needed(self) -> None:
        active = self.jobs.active()
        if not active or not active.started_at:
            return
        if self.jobs.queue_depth() == 0 or active.last_warning_at is not None:
            return
        started = datetime.fromisoformat(active.started_at)
        elapsed_hours = (datetime.now(UTC) - started).total_seconds() / 3600
        if elapsed_hours >= self.config.jobs.warn_after_hours:
            self.jobs.add_log(
                active.id,
                "system",
                f"Job exceeded {self.config.jobs.warn_after_hours}h while queue is waiting; no interruption configured.",
            )
            self.jobs.mark_time_warning(active.id)
