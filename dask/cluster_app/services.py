from __future__ import annotations

from dataclasses import dataclass

from cluster_app.config.loader import load_config
from cluster_app.config.schema import AppConfig
from cluster_app.dask_runtime.cluster_scheduler import ClusterSchedulerResolver
from cluster_app.dask_runtime.scheduler_runtime import SchedulerRuntime
from cluster_app.dask_runtime.worker_runtime import WorkerRuntime
from cluster_app.jobs.queue import JobQueueManager
from cluster_app.nodes.presence import NodePresenceMonitor
from cluster_app.storage.db import Database, initialize_database
from cluster_app.storage.repositories import JobRepository, NodeRepository, UserRepository


@dataclass(slots=True)
class AppServices:
    config: AppConfig
    db: Database
    users: UserRepository
    nodes: NodeRepository
    jobs: JobRepository
    queue: JobQueueManager
    presence: NodePresenceMonitor | None
    scheduler: SchedulerRuntime
    worker: WorkerRuntime
    cluster_scheduler: ClusterSchedulerResolver


def create_services(config: AppConfig | None = None) -> AppServices:
    cfg = config or load_config()
    db = Database(cfg.db_path)
    initialize_database(db)
    jobs = JobRepository(db)
    nodes = NodeRepository(db)
    scheduler = SchedulerRuntime(cfg)
    worker = WorkerRuntime(cfg)
    cluster_scheduler = ClusterSchedulerResolver(cfg, nodes, scheduler)
    return AppServices(
        config=cfg,
        db=db,
        users=UserRepository(db),
        nodes=nodes,
        jobs=jobs,
        scheduler=scheduler,
        worker=worker,
        cluster_scheduler=cluster_scheduler,
        queue=JobQueueManager(cfg, jobs, scheduler_runtime=cluster_scheduler),
        presence=NodePresenceMonitor(cfg, nodes) if cfg.cluster.node_presence else None,
    )


def services_from_request(request):
    return request.app.state.services
