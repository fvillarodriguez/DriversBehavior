# Dask Cluster App

Aplicación local distribuida sobre Dask Distributed para administrar un clúster permanente
en Windows 11 y macOS Apple Silicon.

## Estado de esta implementación

Esta es una primera implementación funcional del plan acordado:

- API local con FastAPI y WebSocket.
- Node Agent con diagnóstico, descubrimiento mDNS y fallback manual.
- SQLite persistente como fuente de verdad.
- TLS obligatorio para Dask, con CA interna y certificados por nodo.
- Cola multiusuario con un solo job global activo.
- Ejecución de carpetas de job con scripts Python libres.
- Workspaces por job, logs, checkpoints opcionales y venv reutilizable por hash.
- Wrappers para Dask scheduler, Dask worker con Nanny y dashboard.
- CLI `cluster-app`.

No es un sandbox de seguridad fuerte para scripts maliciosos. Aísla operacionalmente por
workspace, subprocess, venv y logs separados.

## Instalación

Usa Python 3.12 en un `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

En Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Uso rápido

macOS:

```bash
./star_app.command
```

Windows:

```powershell
.\start_app.bat
```

Los launchers verifican Python 3.12/3.13, crean `.venv`, instalan dependencias,
crean `config.yaml` si falta, arrancan la app y abren `http://127.0.0.1:18080`.
Por defecto escuchan en `0.0.0.0:18080`, para que otros nodos de la LAN puedan
alcanzar `/api/nodes/self`; la URL que abren localmente sigue siendo loopback.

```bash
cluster-app init
cluster-app start --host 127.0.0.1
cluster-app diagnose
cluster-app submit /ruta/a/carpeta-job --entrypoint main.py --arg --seed --arg 42
cluster-app status
```

La UI queda disponible en el puerto elegido automáticamente o en `config.yaml`.

## Uso como librería dentro de una app ML

Además del flujo legacy de **Submit Job**, la app expone un SDK para que un proyecto
existente use el clúster sin reescribir su arquitectura ni copiar carpetas de job.
El SDK arranca el scheduler, un worker local y un control plane embebido usando el
Python activo del environment donde se importa.

```python
from cluster_app.sdk import DaskCluster

cluster = DaskCluster()
futures = cluster.map(predict_one, rows)
predictions = cluster.gather(futures)
```

En Streamlit puedes mostrar un panel mínimo de administración:

```python
from cluster_app.sdk import DaskCluster
from cluster_app.integrations.streamlit import render_cluster_panel

cluster = render_cluster_panel()
results = cluster.gather(cluster.map(predict_one, rows))
```

Para LAN, todos los nodos deben tener instalado el mismo proyecto ML y acceso lógico
a los mismos datos/modelos, pero las rutas absolutas pueden variar entre Windows y
macOS. Por defecto el SDK intenta resolver rutas de forma portable en este orden:
alias manual, ruta relativa al directorio actual del proyecto, ruta relativa al home,
y ruta absoluta como fallback.

```python
cluster.validate_shared_paths(["/ruta/dataset", "/ruta/modelos"])
```

Si un dataset vive en `/Volumes/ml-data` en macOS y en `D:\ml-data` en Windows,
configura el mismo alias localmente en cada nodo:

```python
cluster.configure_path_mappings({"data": "/Volumes/ml-data"})
```

El control plane embebido también expone:

```text
GET /api/admin/path-mappings
PUT /api/admin/path-mappings
POST /api/admin/path-mappings/resolve
```

## Nodos en red local

Cada Web UI publica su presencia por mDNS y escanea `_dask-cluster._tcp.local.`
para registrar nodos remotos en SQLite. Si la red bloquea multicast, usa el
fallback manual en la tarjeta **Nodes**:

```text
Node IP: IP LAN del otro computador
Port: 18080, salvo que hayas cambiado CLUSTER_APP_PORT
```

El otro equipo debe tener la app arrancada con los launchers actualizados o con:

```bash
cluster-app start --host 0.0.0.0 --port 18080
```

## Formato de job

La entrada es una carpeta local. La app la copia a:

```text
workspace/<job_id>/
  code/
  input/
  output/
  logs/
  checkpoints/
  metadata.json
```

Si existe `requirements.txt`, se crea/reutiliza un venv por hash de dependencias,
plataforma, versión de Python y backend GPU solicitado.

Para checkpoint opcional:

```python
from cluster_app.jobs.checkpoint import load_checkpoint, save_checkpoint

state = load_checkpoint("state.pkl", default={"i": 0})
state["i"] += 1
save_checkpoint("state.pkl", state)
```

Los scripts libres sin esta API se reinician desde cero después de una interrupción.

ALL FILES AND THEIR PURPOSES

Source (cluster_app/)

File	Purpose
__init__.py	Package metadata, __version__ = "0.1.0"
main.py	Entry point, delegates to cli:main()
cli.py	CLI argument parsing and command dispatch (init/start/agent/status/diagnose/submit/service)
api/:

File	Purpose
__init__.py	Exports create_app
app.py	FastAPI app factory, service initialization, startup/shutdown hooks
routes_auth.py	POST /api/auth/register, /api/auth/login; token-based session management in memory + HMAC-signed tokens persisted on disk
routes_jobs.py	GET/POST /api/jobs, GET /api/jobs/{id}/logs, POST /api/jobs/{id}/cancel
routes_nodes.py	GET /api/nodes, /api/nodes/self, POST /api/nodes/manual, POST /api/nodes/{uuid}/revoke
routes_admin.py	GET /api/admin/firewall-plan, /api/admin/service-plan, /api/admin/scheduler/status, POST /api/admin/scheduler/start, /api/admin/scheduler/stop
routes_filesystem.py	GET /api/filesystem/list (browse dirs), /api/filesystem/python-files (find .py files)
routes_metrics.py	GET /api/metrics/status (cluster health summary)
websocket.py	WebSocket /ws/events (real-time status push every 2s)
config/:

File	Purpose
__init__.py	Exports AppConfig, load_config, write_default_config
schema.py	Dataclass definitions: AppConfig, ClusterConfig, PathConfig, NetworkConfig, SecurityConfig, JobConfig, DaskConfig, GPUConfig; config_from_dict/ config_to_dict
loader.py	YAML/JSON config loading with load_config(), write_default_config()
defaults.yaml	Default configuration values
nodes/:

File	Purpose
__init__.py	Exports NodeAgent, HardwareProfile, detect_hardware
identity.py	NodeIdentity dataclass, persistent identity loading from JSON
hardware.py	HardwareProfile dataclass and detection (CPU, RAM, GPU backends)
agent.py	NodeAgent - standalone agent that manages scheduler/worker subprocesses + mDNS
presence.py	NodePresenceMonitor - lighter web-mode node discovery via mDNS scans
health.py	Network diagnostics (run_network_diagnostics, choose_free_port)
firewall.py	plan_firewall() - generates firewall rule commands for macOS/Windows
service.py	plan_service_install() - generates LaunchDaemon/Windows service config
dask_runtime/:

File	Purpose
__init__.py	Exports DaskClientFactory, SchedulerProcess, WorkerProcess
scheduler.py	SchedulerProcess - subprocess management for Dask scheduler
scheduler_runtime.py	SchedulerRuntime - higher-level scheduler lifecycle (start/stop/status)
worker.py	WorkerProcess - subprocess management for Dask worker with Nanny
client.py	DaskClientFactory - creates Dask Client with TLS security
dashboard_proxy.py	dashboard_link() - builds dashboard URL
resources.py	WorkerResources dataclass, memory_limit(), dask_resource_flags()
jobs/:

File	Purpose
__init__.py	Exports JobQueueManager, JobRunner, JobWorkspace, prepare_workspace
queue.py	JobQueueManager - queue loop, submission, retry logic
runner.py	JobRunner - subprocess execution, log piping, environment setup
workspace.py	JobWorkspace dataclass, prepare_workspace(), workspace_from_existing()
packager.py	JobPackage dataclass, inspect_job_folder() - validates job folder structure
dependency_installer.py	DependencyInstaller - hash-based venv caching and pip install
checkpoint.py	save_checkpoint() / load_checkpoint() - job checkpoint API (pickle/JSON)
optuna_adapter.py	sqlite_storage_url() - helper for Optuna hyperparameter tuning within jobs
states.py	Re-exports JobStatus from models
discovery/:

File	Purpose
__init__.py	Exports Candidate, choose_scheduler
mdns.py	MdnsPublisher, discover() - mDNS/Zeroconf service registration and discovery
election.py	Candidate, choose_scheduler(), should_self_promote() for scheduler election
manual_ip.py	local_ip(), probe_tcp(), TcpProbeResult - IP detection and TCP probing
heartbeat.py	HeartbeatLoop - generic asyncio heartbeat with configurable interval
security/:

File	Purpose
__init__.py	Exports PairingTokenService
ca.py	CertificateAuthority - internal CA key/cert generation using cryptography, issue_node_certificate()
certs.py	ensure_node_cert() - idempotent node cert creation
tls_config.py	dask_security() (distributed.Security object), dask_tls_env() (env vars for Dask subprocesses)
tokens.py	PairingTokenService - HMAC-based node pairing tokens
backup.py	backup_ca() / restore_ca() - Fernet-encrypted CA backup
storage/:

File	Purpose
__init__.py	Exports Database, initialize_database, JobRepository, NodeRepository, UserRepository
db.py	Database - SQLite connection context manager with WAL mode and foreign keys
models.py	Dataclasses: User, Node, Job, JobLog; Enums: UserRole, NodeStatus, JobStatus
repositories.py	UserRepository (create, authenticate, get), NodeRepository (upsert, revoke, mark_stale_offline, list), JobRepository (create, next_queued, active, set_running, set_status, requeue, add_log, logs, cleanup)
platform/:

File	Purpose
__init__.py	Exports default_state_dir, default_workspace_dir
paths.py	Platform-aware default paths (Windows vs Unix)
macos.py	Firewall rules, LaunchDaemon plist generation
windows.py	Firewall rules, Windows service command generation
ui/:

File	Purpose
templates/index.html	Main single-page app HTML (login, job submission, queue, nodes)
templates/dashboard.html	Dashboard view (metrics, recent jobs, nodes, Dask dashboard link)
static/styles.css	Styling with light/dark color-scheme support
static/app.js	Main UI logic (auth, job submission, file browser, WebSocket updates, scheduler start)
static/dashboard.js	Dashboard logic (metrics refresh, scheduler start)
notebooks/:

File	Purpose
__init__.py	Exports connect
client_helper.py	connect() - helper for Jupyter notebooks to connect to managed Dask scheduler
Tests (tests/)

File	Purpose
test_storage.py	Tests for UserRepository, NodeRepository, JobRepository
test_config.py	Tests for config loading, default writing, dict conversion
test_jobs.py	Tests for packager (entrypoint detection), workspace (copy/reload), requirements hashing
test_nodes_dask.py	Tests for memory_limit, resource_flags, dashboard_link, choose_free_port, SchedulerProcess, WorkerProcess, NodePresenceMonitor
test_api.py	Tests for root/dashboard HTML serving, auth (register/login/token persistence), filesystem listing, node manual add
test_discovery_security.py	Tests for scheduler election, pairing token verification
Launchers

File	Purpose
start_app.command	macOS bash launcher (venv creation, install, start, open browser)
start_app.bat	Windows batch launcher (same functionality)
