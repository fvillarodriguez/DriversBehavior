from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.jobs.dependency_installer import DependencyInstaller
from cluster_app.jobs.workspace import JobWorkspace
from cluster_app.storage.models import JobStatus
from cluster_app.storage.repositories import JobRepository


LogCallback = Callable[[str, str], None]


@dataclass(frozen=True, slots=True)
class JobRunResult:
    status: JobStatus
    return_code: int


class JobRunner:
    def __init__(self, config: AppConfig, jobs: JobRepository):
        self.config = config
        self.jobs = jobs
        self.installer = DependencyInstaller(config.paths.envs_dir)
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel_requested = False

    async def run(
        self,
        job_id: str,
        workspace: JobWorkspace,
        args: list[str],
        dask_scheduler_url: str | None = None,
        gpu_backend: str | None = None,
        dask_certs_dir: str | Path | None = None,
    ) -> JobRunResult:
        self._cancel_requested = False
        env_info = self.installer.ensure(workspace.requirements, gpu_backend)
        log_file = workspace.logs_dir / "combined.log"
        self.jobs.set_running(job_id, str(workspace.root), dask_scheduler_url)
        env = self._env(workspace, dask_scheduler_url, gpu_backend, dask_certs_dir)
        cmd = [str(env_info.python), str(workspace.entrypoint), *args]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(workspace.code_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        await asyncio.gather(
            self._pipe(job_id, "stdout", self._proc.stdout, log_file),
            self._pipe(job_id, "stderr", self._proc.stderr, log_file),
        )
        return_code = await self._proc.wait()
        self._proc = None
        if self._cancel_requested:
            status = JobStatus.CANCELED
            self.jobs.add_log(job_id, "system", "Job process terminated by user.")
        else:
            status = JobStatus.SUCCEEDED if return_code == 0 else JobStatus.FAILED
        self.jobs.set_status(job_id, status, {"return_code": return_code})
        return JobRunResult(status, return_code)

    def cancel(self, job_id: str) -> None:
        self._cancel_requested = True
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()

    async def _pipe(
        self,
        job_id: str,
        stream: str,
        reader: asyncio.StreamReader,
        log_file: Path,
    ) -> None:
        with log_file.open("a", encoding="utf-8") as fh:
            while line := await reader.readline():
                message = line.decode("utf-8", errors="replace").rstrip()
                fh.write(f"[{stream}] {message}\n")
                fh.flush()
                self.jobs.add_log(job_id, stream, message)

    def _env(
        self,
        workspace: JobWorkspace,
        dask_scheduler_url: str | None,
        gpu_backend: str | None,
        dask_certs_dir: str | Path | None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "CLUSTER_APP_WORKSPACE": str(workspace.root),
                "CLUSTER_APP_OUTPUT_DIR": str(workspace.output_dir),
                "CLUSTER_APP_CHECKPOINT_DIR": str(workspace.checkpoints_dir),
                "CLUSTER_APP_GPU_BACKEND": gpu_backend or "cpu",
                "PYTHONPATH": _with_project_path(env.get("PYTHONPATH")),
            }
        )
        if dask_scheduler_url:
            env["DASK_SCHEDULER_ADDRESS"] = dask_scheduler_url
        if dask_certs_dir:
            env["CLUSTER_APP_CERTS_DIR"] = str(Path(dask_certs_dir))
        return env


def _with_project_path(existing: str | None) -> str:
    root = str(Path(__file__).resolve().parents[2])
    return root if not existing else os.pathsep.join([root, existing])
