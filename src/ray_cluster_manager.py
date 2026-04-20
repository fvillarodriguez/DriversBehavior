#!/usr/bin/env python3
"""
Backend helpers for managing a small Ray cluster from the Streamlit app.
"""
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "Resultados" / "ray_cluster"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_HEAD_IP = "10.10.10.1"
DEFAULT_WORKER_IP = "10.10.10.2"
DEFAULT_NETMASK = "255.255.255.252"
DEFAULT_RAY_VERSION = "2.53.0"
DEFAULT_WORKER_PORT_MIN = 10002
DEFAULT_WORKER_PORT_MAX = 10100
COMMON_SSH_PRIVATE_KEY_NAMES = (
    "id_ed25519",
    "id_rsa",
    "id_ecdsa",
    "id_ed25519_sk",
    "id_ecdsa_sk",
)
SSH_PUBLIC_KEY_PREFIXES = (
    "ssh-ed25519",
    "ssh-rsa",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "sk-ecdsa-sha2-nistp256@openssh.com",
    "sk-ssh-ed25519@openssh.com",
)
SSH_PUBLIC_KEY_BODY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


@dataclass(frozen=True)
class RayClusterConfig:
    head_ip: str = DEFAULT_HEAD_IP
    worker_ip: str = DEFAULT_WORKER_IP
    netmask: str = DEFAULT_NETMASK
    ssh_user: str = os.environ.get("USER", "")
    ssh_key_path: str = "~/.ssh/id_ed25519"
    remote_repo_path: str = str(ROOT_DIR)
    head_cpus: int = 8
    worker_reserved_cpus: int = 2
    ray_version: str = DEFAULT_RAY_VERSION
    head_port: int = 6379
    dashboard_port: int = 8265
    ray_client_port: int = 10001
    object_manager_port: int = 8076
    node_manager_port: int = 8077
    worker_port_min: int = DEFAULT_WORKER_PORT_MIN
    worker_port_max: int = DEFAULT_WORKER_PORT_MAX
    command_timeout_s: int = 30

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.head_ip}:{self.dashboard_port}"

    @property
    def ray_address(self) -> str:
        return f"{self.head_ip}:{self.head_port}"


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        chunks = [self.stdout.strip(), self.stderr.strip()]
        return "\n".join(chunk for chunk in chunks if chunk)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    command: str = ""


class CommandRunner:
    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path = ROOT_DIR,
        timeout: int = 30,
        env: Optional[dict[str, str]] = None,
    ) -> CommandResult:
        display = " ".join(shlex.quote(str(part)) for part in args)
        try:
            completed = subprocess.run(
                [str(part) for part in args],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                ok=False,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or f"Timeout despues de {timeout}s.",
                command=display,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return CommandResult(
                ok=False,
                returncode=127,
                stderr=str(exc),
                command=display,
            )
        return CommandResult(
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            command=display,
        )


def default_config() -> RayClusterConfig:
    return RayClusterConfig()


def config_to_json_dict(config: RayClusterConfig) -> dict[str, Any]:
    return asdict(config)


def load_config(path: Path = CONFIG_FILE) -> RayClusterConfig:
    if not path.exists():
        return default_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_config()
    allowed = set(RayClusterConfig.__dataclass_fields__.keys())
    clean = {key: value for key, value in payload.items() if key in allowed}
    try:
        return RayClusterConfig(**clean)
    except TypeError:
        return default_config()


def save_config(config: RayClusterConfig, path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config_to_json_dict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def ray_bin(root_dir: Path = ROOT_DIR) -> Path:
    return root_dir / ".venv" / "bin" / "ray"


def python_bin(root_dir: Path = ROOT_DIR) -> Path:
    return root_dir / ".venv" / "bin" / "python"


def ssh_private_key_path(ssh_key_path: str) -> Path:
    raw_value = ssh_key_path.strip()
    if not raw_value:
        raise ValueError("Ingrese la ruta de la llave SSH privada.")
    key_path = Path(raw_value).expanduser()
    if not key_path.name:
        raise ValueError("La ruta de la llave SSH privada no es valida.")
    if key_path.name.endswith(".pub"):
        raise ValueError("La ruta configurada debe apuntar a la llave SSH privada, no al archivo .pub.")
    return key_path


def ssh_public_key_path(ssh_key_path: str) -> Path:
    private_key = ssh_private_key_path(ssh_key_path)
    return private_key.with_name(f"{private_key.name}.pub")


def authorized_keys_path() -> Path:
    return Path.home() / ".ssh" / "authorized_keys"


def detect_private_keys(ssh_dir: Optional[Path] = None) -> list[Path]:
    root = (ssh_dir or (Path.home() / ".ssh")).expanduser()
    matches: list[Path] = []
    for name in COMMON_SSH_PRIVATE_KEY_NAMES:
        candidate = root / name
        if candidate.exists() and candidate.is_file():
            matches.append(candidate)
    return matches


def normalize_public_key(public_key: str) -> str:
    normalized = " ".join(segment.strip() for segment in public_key.splitlines() if segment.strip())
    if not normalized:
        raise ValueError("Ingrese una llave publica SSH.")
    parts = normalized.split(None, 2)
    if len(parts) < 2:
        raise ValueError("Formato de llave publica SSH no reconocido.")
    key_type, key_body = parts[0], parts[1]
    if key_type not in SSH_PUBLIC_KEY_PREFIXES:
        raise ValueError("Tipo de llave publica SSH no soportado.")
    if any(char not in SSH_PUBLIC_KEY_BODY_CHARS for char in key_body):
        raise ValueError("La llave publica SSH contiene caracteres invalidos.")
    return normalized


def read_public_key(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> str:
    public_key_file = ssh_public_key_path(config.ssh_key_path)
    if public_key_file.exists():
        payload = public_key_file.read_text(encoding="utf-8").strip()
        if payload:
            return normalize_public_key(payload)

    private_key_file = ssh_private_key_path(config.ssh_key_path)
    if not private_key_file.exists():
        raise FileNotFoundError(f"No existe la llave SSH configurada: {private_key_file}")

    active_runner = runner or CommandRunner()
    result = active_runner.run(
        ["ssh-keygen", "-y", "-f", str(private_key_file)],
        timeout=min(10, config.command_timeout_s),
    )
    if not result.ok or not result.stdout.strip():
        raise RuntimeError(result.combined_output or "No se pudo exportar la llave publica SSH.")
    return normalize_public_key(result.stdout)


def import_public_key(
    public_key: str,
    *,
    target_path: Optional[Path] = None,
) -> str:
    normalized = normalize_public_key(public_key)
    authorized_keys = (target_path or authorized_keys_path()).expanduser()
    ssh_dir = authorized_keys.parent
    ssh_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        ssh_dir.chmod(0o700)
    except OSError:
        pass

    existing_text = authorized_keys.read_text(encoding="utf-8") if authorized_keys.exists() else ""
    existing_keys: set[str] = set()
    for line in existing_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            existing_keys.add(normalize_public_key(stripped))
        except ValueError:
            continue
    if normalized in existing_keys:
        try:
            authorized_keys.chmod(0o600)
        except OSError:
            pass
        return f"La llave publica ya estaba presente en {authorized_keys}."

    separator = "" if not existing_text or existing_text.endswith("\n") else "\n"
    authorized_keys.write_text(f"{existing_text}{separator}{normalized}\n", encoding="utf-8")
    try:
        authorized_keys.chmod(0o600)
    except OSError:
        pass
    return f"Llave publica importada en {authorized_keys}."


def bridge_manual_command(ip: str, netmask: str = DEFAULT_NETMASK) -> str:
    return f"sudo networksetup -setmanual 'Thunderbolt Bridge' {shlex.quote(ip)} {shlex.quote(netmask)} 0.0.0.0"


def build_head_start_args(config: RayClusterConfig, *, root_dir: Path = ROOT_DIR) -> list[str]:
    return [
        str(ray_bin(root_dir)),
        "start",
        "--head",
        f"--node-ip-address={config.head_ip}",
        f"--port={config.head_port}",
        "--dashboard-host=0.0.0.0",
        f"--dashboard-port={config.dashboard_port}",
        f"--ray-client-server-port={config.ray_client_port}",
        f"--object-manager-port={config.object_manager_port}",
        f"--node-manager-port={config.node_manager_port}",
        f"--min-worker-port={config.worker_port_min}",
        f"--max-worker-port={config.worker_port_max}",
        f"--num-cpus={max(1, int(config.head_cpus))}",
        "--disable-usage-stats",
    ]


def build_worker_start_script(config: RayClusterConfig) -> str:
    repo = shlex.quote(config.remote_repo_path)
    reserved = max(0, int(config.worker_reserved_cpus))
    return "\n".join(
        [
            f"cd {repo}",
            "CPUS=$(($(sysctl -n hw.ncpu)-%d))" % reserved,
            '[ "$CPUS" -lt 1 ] && CPUS=1',
            ".venv/bin/ray stop",
            "exec .venv/bin/ray start "
            f"--address={shlex.quote(config.ray_address)} "
            f"--node-ip-address={shlex.quote(config.worker_ip)} "
            f"--object-manager-port={config.object_manager_port} "
            f"--node-manager-port={config.node_manager_port} "
            f"--min-worker-port={config.worker_port_min} "
            f"--max-worker-port={config.worker_port_max} "
            '--num-cpus="$CPUS" '
            "--disable-usage-stats",
        ]
    )


def ssh_base_args(config: RayClusterConfig) -> list[str]:
    private_key = ssh_private_key_path(config.ssh_key_path)
    return [
        "ssh",
        "-i",
        str(private_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{config.ssh_user}@{config.worker_ip}",
    ]


def run_remote_script(
    config: RayClusterConfig,
    script: str,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: Optional[int] = None,
) -> CommandResult:
    active_runner = runner or CommandRunner()
    try:
        ssh_args = ssh_base_args(config)
    except ValueError as exc:
        return CommandResult(
            ok=False,
            returncode=2,
            stderr=str(exc),
            command="ssh",
        )
    remote_command = f"bash -lc {shlex.quote(script)}"
    return active_runner.run(
        [*ssh_args, remote_command],
        timeout=timeout or config.command_timeout_s,
    )


def stop_head(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    active_runner = runner or CommandRunner()
    return active_runner.run(
        [str(ray_bin()), "stop"],
        cwd=ROOT_DIR,
        timeout=config.command_timeout_s,
    )


def start_head(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    active_runner = runner or CommandRunner()
    stop_head(config, runner=active_runner)
    return active_runner.run(
        build_head_start_args(config),
        cwd=ROOT_DIR,
        timeout=config.command_timeout_s,
    )


def stop_worker(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    return run_remote_script(
        config,
        f"cd {shlex.quote(config.remote_repo_path)} && .venv/bin/ray stop",
        runner=runner,
        timeout=config.command_timeout_s,
    )


def start_worker(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    return run_remote_script(
        config,
        build_worker_start_script(config),
        runner=runner,
        timeout=config.command_timeout_s,
    )


def stop_cluster(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> list[CommandResult]:
    active_runner = runner or CommandRunner()
    return [
        stop_worker(config, runner=active_runner),
        stop_head(config, runner=active_runner),
    ]


def start_cluster(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> list[CommandResult]:
    active_runner = runner or CommandRunner()
    head = start_head(config, runner=active_runner)
    worker = start_worker(config, runner=active_runner) if head.ok else CommandResult(
        ok=False,
        returncode=1,
        stderr="Head no inicio; worker omitido.",
        command="start worker",
    )
    return [head, worker]


def restart_cluster(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> list[CommandResult]:
    active_runner = runner or CommandRunner()
    return [*stop_cluster(config, runner=active_runner), *start_cluster(config, runner=active_runner)]


def ray_status(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None, timeout: int = 10) -> CommandResult:
    active_runner = runner or CommandRunner()
    return active_runner.run(
        [str(ray_bin()), "status", f"--address={config.ray_address}"],
        cwd=ROOT_DIR,
        timeout=timeout,
    )


def fixed_ray_ports(config: RayClusterConfig) -> list[int]:
    return [
        config.head_port,
        config.dashboard_port,
        config.ray_client_port,
        config.object_manager_port,
        config.node_manager_port,
    ]


def _port_check_script(config: RayClusterConfig) -> str:
    ports = " ".join(str(port) for port in fixed_ray_ports(config))
    return (
        "busy=''; "
        f"for port in {ports}; do "
        "if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then busy=\"$busy $port\"; fi; "
        "done; "
        "if [ -z \"$busy\" ]; then echo 'Puertos libres'; else echo \"Puertos ocupados:$busy\"; exit 1; fi"
    )


def check_ports_available(
    config: RayClusterConfig,
    *,
    remote: bool = False,
    runner: Optional[CommandRunner] = None,
) -> CheckResult:
    active_runner = runner or CommandRunner()
    if remote:
        result = run_remote_script(config, _port_check_script(config), runner=active_runner, timeout=8)
        name = "Puertos worker"
    else:
        result = active_runner.run(["bash", "-lc", _port_check_script(config)], cwd=ROOT_DIR, timeout=8)
        name = "Puertos head"
    return CheckResult(
        name,
        result.ok,
        result.combined_output or ("Puertos libres" if result.ok else "No se pudo verificar puertos."),
        result.command,
    )


def _version_ok(output: str, expected_prefix: str) -> bool:
    return expected_prefix in output.strip()


def _bridge_has_ip(output: str, ip: str) -> bool:
    return f"inet {ip} " in output or f"inet {ip}\n" in output


def _bridge_active(output: str) -> bool:
    return "status: active" in output.lower()


def _result_detail(result: CommandResult, ok_message: str) -> str:
    if result.ok:
        return ok_message
    return result.combined_output or f"Comando fallo con codigo {result.returncode}."


def run_preflight(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> list[CheckResult]:
    active_runner = runner or CommandRunner()
    checks: list[CheckResult] = []

    local_bridge = active_runner.run(["ifconfig", "bridge0"], timeout=5)
    checks.append(
        CheckResult(
            "Thunderbolt local",
            local_bridge.ok and _bridge_has_ip(local_bridge.stdout, config.head_ip) and _bridge_active(local_bridge.stdout),
            (
                f"bridge0 activo con {config.head_ip}."
                if local_bridge.ok and _bridge_has_ip(local_bridge.stdout, config.head_ip) and _bridge_active(local_bridge.stdout)
                else f"Configure: {bridge_manual_command(config.head_ip, config.netmask)}"
            ),
            local_bridge.command,
        )
    )

    ssh_check = active_runner.run([*ssh_base_args(config), "true"], timeout=8)
    checks.append(
        CheckResult(
            "SSH worker",
            ssh_check.ok,
            _result_detail(ssh_check, f"SSH OK hacia {config.ssh_user}@{config.worker_ip}."),
            ssh_check.command,
        )
    )

    ping_worker = active_runner.run(["ping", "-c", "2", "-W", "1000", config.worker_ip], timeout=6)
    checks.append(
        CheckResult(
            "Ping worker",
            ping_worker.ok,
            _result_detail(ping_worker, f"{config.worker_ip} responde por Thunderbolt."),
            ping_worker.command,
        )
    )

    local_python = active_runner.run([str(python_bin()), "--version"], timeout=5)
    checks.append(
        CheckResult(
            "Python local",
            local_python.ok and "Python 3.12" in local_python.combined_output,
            _result_detail(local_python, local_python.combined_output.strip()),
            local_python.command,
        )
    )

    local_ray = active_runner.run([str(ray_bin()), "--version"], timeout=5)
    checks.append(
        CheckResult(
            "Ray local",
            local_ray.ok and _version_ok(local_ray.combined_output, config.ray_version),
            _result_detail(local_ray, local_ray.combined_output.strip()),
            local_ray.command,
        )
    )

    if ssh_check.ok:
        remote_bridge = run_remote_script(config, "ifconfig bridge0", runner=active_runner, timeout=8)
        checks.append(
            CheckResult(
                "Thunderbolt worker",
                remote_bridge.ok and _bridge_has_ip(remote_bridge.stdout, config.worker_ip) and _bridge_active(remote_bridge.stdout),
                (
                    f"bridge0 activo con {config.worker_ip}."
                    if remote_bridge.ok and _bridge_has_ip(remote_bridge.stdout, config.worker_ip) and _bridge_active(remote_bridge.stdout)
                    else f"Configure en worker: {bridge_manual_command(config.worker_ip, config.netmask)}"
                ),
                remote_bridge.command,
            )
        )

        remote_python = run_remote_script(
            config,
            f"cd {shlex.quote(config.remote_repo_path)} && .venv/bin/python --version",
            runner=active_runner,
            timeout=8,
        )
        checks.append(
            CheckResult(
                "Python worker",
                remote_python.ok and "Python 3.12" in remote_python.combined_output,
                _result_detail(remote_python, remote_python.combined_output.strip()),
                remote_python.command,
            )
        )

        remote_ray = run_remote_script(
            config,
            f"cd {shlex.quote(config.remote_repo_path)} && .venv/bin/ray --version",
            runner=active_runner,
            timeout=8,
        )
        checks.append(
            CheckResult(
                "Ray worker",
                remote_ray.ok and _version_ok(remote_ray.combined_output, config.ray_version),
                _result_detail(remote_ray, remote_ray.combined_output.strip()),
                remote_ray.command,
            )
        )
    else:
        checks.extend(
            [
                CheckResult("Thunderbolt worker", False, "Omitido porque SSH no conecta."),
                CheckResult("Python worker", False, "Omitido porque SSH no conecta."),
                CheckResult("Ray worker", False, "Omitido porque SSH no conecta."),
            ]
        )

    stop_probe = stop_head(config, runner=active_runner)
    checks.append(
        CheckResult(
            "ray stop local",
            stop_probe.returncode in (0, 1),
            "ray stop local ejecutado; el head queda limpio para iniciar.",
            stop_probe.command,
        )
    )
    checks.append(check_ports_available(config, runner=active_runner))
    if ssh_check.ok:
        remote_stop = stop_worker(config, runner=active_runner)
        checks.append(
            CheckResult(
                "ray stop worker",
                remote_stop.returncode in (0, 1),
                "ray stop worker ejecutado; el worker queda limpio para iniciar.",
                remote_stop.command,
            )
        )
        checks.append(check_ports_available(config, remote=True, runner=active_runner))
    else:
        checks.append(CheckResult("ray stop worker", False, "Omitido porque SSH no conecta."))
        checks.append(CheckResult("Puertos worker", False, "Omitido porque SSH no conecta."))

    return checks


def tail_logs(config: RayClusterConfig, *, remote: bool = False, lines: int = 80, runner: Optional[CommandRunner] = None) -> CommandResult:
    safe_lines = max(10, min(int(lines), 300))
    script = (
        "for file in /tmp/ray/session_latest/logs/raylet.err "
        "/tmp/ray/session_latest/logs/raylet.out "
        "/tmp/ray/session_latest/logs/gcs_server.err "
        "/tmp/ray/session_latest/logs/dashboard.err; do "
        "[ -f \"$file\" ] && echo \"===== $file =====\" && tail -n "
        f"{safe_lines} \"$file\"; "
        "done"
    )
    active_runner = runner or CommandRunner()
    if remote:
        return run_remote_script(config, script, runner=active_runner, timeout=10)
    return active_runner.run(["bash", "-lc", script], cwd=ROOT_DIR, timeout=10)


def parse_json_from_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise ValueError("No se encontro JSON en la salida.")


def parse_ray_status_summary(output: str) -> dict[str, Any]:
    active_nodes = 0
    usage: dict[str, str] = {}
    in_active = False
    in_usage = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == "Active:":
            in_active = True
            in_usage = False
            continue
        if line in {"Pending:", "Recent failures:", "Resources", "Demands:"}:
            in_active = False
        if line == "Usage:":
            in_usage = True
            continue
        if line == "Demands:":
            in_usage = False
            continue
        if in_active and line and not line.startswith("("):
            active_nodes += 1
        if in_usage and "/" in line and not line.startswith("("):
            parts = line.split()
            if parts:
                resource = parts[-1]
                usage[resource] = " ".join(parts[:-1])
    return {"active_nodes": active_nodes, "usage": usage}


def build_benchmark_script(config: RayClusterConfig, tasks: int = 80) -> str:
    safe_tasks = max(1, min(int(tasks), 10000))
    return f"""
import json
import os
import socket
from collections import Counter

import ray

ray.init(address={config.ray_address!r}, ignore_reinit_error=True)

@ray.remote
def task(index):
    return {{"host": socket.gethostname(), "pid": os.getpid(), "index": index}}

results = ray.get([task.remote(i) for i in range({safe_tasks})])
hosts = Counter(item["host"] for item in results)
nodes = [
    {{
        "node_ip": node.get("NodeManagerAddress"),
        "alive": node.get("Alive"),
        "resources": node.get("Resources", {{}}),
    }}
    for node in ray.nodes()
]
print(json.dumps({{
    "tasks": {safe_tasks},
    "tasks_by_host": dict(hosts),
    "cluster_resources": ray.cluster_resources(),
    "nodes": nodes,
}}, sort_keys=True))
ray.shutdown()
"""


def run_distributed_benchmark(
    config: RayClusterConfig,
    *,
    tasks: int = 80,
    runner: Optional[CommandRunner] = None,
    timeout: int = 60,
) -> tuple[CommandResult, Optional[dict[str, Any]]]:
    active_runner = runner or CommandRunner()
    result = active_runner.run(
        [str(python_bin()), "-c", build_benchmark_script(config, tasks=tasks)],
        cwd=ROOT_DIR,
        timeout=timeout,
    )
    if not result.ok:
        return result, None
    try:
        return result, parse_json_from_output(result.stdout)
    except Exception as exc:
        return (
            CommandResult(
                ok=False,
                returncode=1,
                stdout=result.stdout,
                stderr=f"No se pudo parsear benchmark: {exc}",
                command=result.command,
            ),
            None,
        )


def local_hostname() -> str:
    return socket.gethostname()


def check_config_warnings(config: RayClusterConfig) -> list[str]:
    warnings: list[str] = []
    if not config.ssh_user.strip():
        warnings.append("Ingrese el usuario SSH del worker.")
    if not config.remote_repo_path.strip():
        warnings.append("Ingrese la ruta del repo en el worker.")
    try:
        private_key = ssh_private_key_path(config.ssh_key_path)
    except ValueError as exc:
        warnings.append(str(exc))
    else:
        if not private_key.exists():
            warnings.append(f"No existe la llave SSH privada configurada: {private_key}")
    if config.head_ip == config.worker_ip:
        warnings.append("Head y worker no pueden usar la misma IP.")
    return warnings


def command_outputs_to_text(results: Iterable[CommandResult]) -> str:
    chunks: list[str] = []
    for result in results:
        status = "OK" if result.ok else "ERROR"
        chunks.append(f"$ {result.command}\n[{status}] rc={result.returncode}")
        if result.combined_output:
            chunks.append(result.combined_output)
    return "\n\n".join(chunks)
