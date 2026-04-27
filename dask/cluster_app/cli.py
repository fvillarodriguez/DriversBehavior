from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict

from cluster_app.config.loader import load_config, write_default_config
from cluster_app.nodes.health import choose_free_port, run_network_diagnostics
from cluster_app.nodes.service import plan_service_install
from cluster_app.storage.db import Database, initialize_database
from cluster_app.storage.repositories import JobRepository, UserRepository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cluster-app")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    start = sub.add_parser("start")
    start.add_argument("--host", default=None)
    start.add_argument("--port", type=int, default=None)

    sub.add_parser("agent")
    sub.add_parser("status")
    sub.add_parser("diagnose")

    submit = sub.add_parser("submit")
    submit.add_argument("source_dir")
    submit.add_argument("--entrypoint")
    submit.add_argument("--name")
    submit.add_argument("--arg", action="append", default=[])

    service = sub.add_parser("service")
    service.add_argument("action", choices=["plan-install"])

    args = parser.parse_args(argv)

    if args.command == "init":
        path = write_default_config(args.config)
        print(f"Wrote {path}")
        return 0
    if args.command == "start":
        return _start(args)
    if args.command == "agent":
        return asyncio.run(_agent(args))
    if args.command == "status":
        return _status(args)
    if args.command == "diagnose":
        return _diagnose(args)
    if args.command == "submit":
        return _submit(args)
    if args.command == "service":
        return _service(args)
    return 2


def _config(args) -> object:
    return load_config(args.config)


def _start(args) -> int:
    cfg = load_config(args.config)
    explicit_web_port = args.port is not None
    if args.host:
        cfg.network.host = args.host
    if args.port:
        cfg.network.web_port = args.port
    if cfg.network.auto_ports:
        bind_host = cfg.network.host
        if not explicit_web_port:
            cfg.network.web_port = choose_free_port(bind_host, cfg.network.web_port)
        cfg.network.dask_scheduler_port = choose_free_port(bind_host, cfg.network.dask_scheduler_port)
        cfg.network.dask_dashboard_port = choose_free_port(bind_host, cfg.network.dask_dashboard_port)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError("uvicorn is required to start the web server. Install the project venv.") from exc
    from cluster_app.api.app import create_app

    app = create_app(cfg)
    print(f"Web UI: http://127.0.0.1:{cfg.network.web_port}")
    uvicorn.run(app, host=cfg.network.host, port=cfg.network.web_port)
    return 0


async def _agent(args) -> int:
    cfg = load_config(args.config)
    from cluster_app.nodes.agent import NodeAgent
    from cluster_app.storage.repositories import NodeRepository

    db = Database(cfg.db_path)
    initialize_database(db)
    await NodeAgent(cfg, NodeRepository(db)).start()
    return 0


def _status(args) -> int:
    cfg = load_config(args.config)
    db = Database(cfg.db_path)
    initialize_database(db)
    jobs = JobRepository(db)
    payload = {
        "db": str(cfg.db_path),
        "active_job": asdict(jobs.active()) if jobs.active() else None,
        "queue_depth": jobs.queue_depth(),
        "recent_jobs": [asdict(job) for job in jobs.list(limit=10)],
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _diagnose(args) -> int:
    cfg = load_config(args.config)
    report = run_network_diagnostics(
        "127.0.0.1",
        cfg.network.web_port,
        cfg.network.dask_scheduler_port if cfg.security.tls_required else None,
    )
    print(json.dumps({"ok": report.ok, "steps": [asdict(step) for step in report.steps]}, indent=2))
    return 0 if report.ok else 1


def _submit(args) -> int:
    cfg = load_config(args.config)
    db = Database(cfg.db_path)
    initialize_database(db)
    users = UserRepository(db)
    user = users.get_by_email("cli@local")
    if user is None:
        user = users.create("CLI User", "cli@local", "local-only")
    from cluster_app.jobs.queue import JobQueueManager

    queue = JobQueueManager(cfg, JobRepository(db))
    job = queue.submit(
        user_id=int(user.id or 0),
        source_dir=args.source_dir,
        entrypoint=args.entrypoint,
        args=args.arg,
        name=args.name,
    )
    print(json.dumps(asdict(job), indent=2, default=str))
    return 0


def _service(args) -> int:
    if args.action == "plan-install":
        plan = plan_service_install(args.config)
        print(json.dumps(asdict(plan), indent=2))
        return 0 if plan.supported else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
