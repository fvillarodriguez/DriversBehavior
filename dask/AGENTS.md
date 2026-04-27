# Dask Cluster App — Agent Guide

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

| Command | Purpose |
|---|---|
| `cluster-app init` | Write default `config.yaml` |
| `cluster-app start --host 127.0.0.1` | Start FastAPI + uvicorn web server |
| `cluster-app diagnose` | Network diagnostics |
| `cluster-app submit <dir> --entrypoint main.py` | Submit a job folder |
| `cluster-app status` | JSON dump of queue + active job |

Entrypoint: `cluster_app/cli.py:main` → `cluster_app.api.app:create_app`

## Testing

All tests use stdlib `unittest` (no pytest fixtures). Run any of:

```bash
python -m pytest tests/          # all tests
python -m pytest tests/test_api.py -k test_register  # single test
python tests/test_storage.py     # direct python invocation also works
```

Test pattern: every test creates an `AppConfig` with `ensure_directories()` into `tempfile.TemporaryDirectory`. No global state, no integration services needed.

## Architecture

- **`cluster_app/cli.py`** — CLI dispatch (`init`/`start`/`agent`/`status`/`diagnose`/`submit`/`service`)
- **`cluster_app/api/app.py`** — FastAPI factory `create_app()`, assembles routers + services
- **`cluster_app/config/schema.py`** — `AppConfig` dataclass (slots), loaded from `config.yaml` (or `.json`) via `config_from_dict()`
- **`cluster_app/storage/`** — SQLite via `Database` context manager, WAL mode, FK enabled. Schema: `User`, `Node`, `Job`, `JobLog`
- **`cluster_app/nodes/`** — `NodeAgent` (full agent), `NodePresenceMonitor` (lighter, web-mode only), hardware detection, mDNS
- **`cluster_app/dask_runtime/`** — `SchedulerProcess`/`WorkerProcess` subprocess managers, `DaskClientFactory` with TLS, dashboard proxy
- **`cluster_app/jobs/`** — `JobQueueManager` (queue loop, single active job), `JobRunner` (subprocess), `JobWorkspace` (dir layout), dependency installer with hash-based venv caching
- **`cluster_app/security/`** — Internal CA, TLS certs per node, HMAC pairing tokens, Fernet CA backup
- **`cluster_app/discovery/`** — mDNS (Zeroconf) publish/scan, scheduler election, TCP probing
- **`cluster_app/ui/`** — Jinja2 templates + vanilla JS (`app.js`, `dashboard.js`), served by FastAPI

Key invariants:
- TLS is required by default (`security.tls_required: true`) — Dask subprocesses get env vars via `dask_tls_env()`
- Only one active job at a time (`jobs.single_active_job: true`)
- Config supports both YAML and JSON (auto-detected by extension)

## Config

`config.yaml` at repo root (or `config.json`). Properties from `AppConfig` dataclass mirror the YAML keys. Config paths (`state_dir`, `workspace_dir`, `envs_dir`, `logs_dir`) are expanded via `Path.expanduser()`.

## Ruff

Line length: 100, target-version py312. Run:
```bash
ruff check .
```

## Gotchas

- Tests expect to be run from the repo root (via `pythonpath = ["."]` in pyproject.toml)
- `cluster-app start` uses `uvicorn.run()` — imports are lazy inside the function
- Scheduler/worker subprocesses use `sys.executable -m distributed.cli.dask_scheduler` (current Python, not a fixed binary)
- mDNS discovery service: `_dask-cluster._tcp.local.`
- `config.yaml` is gitignored; tests create fresh configs in temp dirs
- Skills are installed locally (see `skills-lock.json`), not relevant for development