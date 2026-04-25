#!/usr/bin/env python3
"""
Backend helpers for managing a small Ray cluster from the Streamlit app.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "Resultados" / "ray_cluster"
CONFIG_FILE = CONFIG_DIR / "config.json"
THUNDERBOLT_BRIDGE_NAME = "Thunderbolt Bridge"
DEFAULT_SSH_PRIVATE_KEY_FILENAME = "id_ed25519"
SSH_BOOTSTRAP_PASSWORD_ENV = "SUMO_RAY_SSH_PASSWORD"

DEFAULT_HEAD_IP = "10.10.10.1"
DEFAULT_WORKER_IP = "10.10.10.2"
DEFAULT_NETMASK = "255.255.255.252"
DEFAULT_RAY_VERSION = "2.53.0"
DEFAULT_WORKER_PORT_MIN = 10002
DEFAULT_WORKER_PORT_MAX = 10100
DISK_WARNING_USED_RATIO = 0.90
DISK_WARNING_FREE_BYTES = 30 * 1024**3
DISK_BLOCK_USED_RATIO = 0.95
DISK_BLOCK_FREE_BYTES = 20 * 1024**3
RAY_TMP_ROOT = Path("/tmp/ray")
REQUIRED_RAY_PYTHON_MODULES = (
    "ray",
    "aiohttp_cors",
    "opencensus",
    "opentelemetry",
    "prometheus_client",
)
RAY_MACOS_CLUSTER_ENV = {
    "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER": "1",
    "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL": "minor",
}
INVALID_SSH_KEY_FILENAMES = frozenset({"authorized_keys", "known_hosts", "config"})
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


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "stdout", _coerce_text(self.stdout))
        object.__setattr__(self, "stderr", _coerce_text(self.stderr))
        object.__setattr__(self, "command", _coerce_text(self.command))

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
    blocking: bool = False


@dataclass(frozen=True)
class RayHealthSnapshot:
    local_environment_checks: tuple[CheckResult, ...]
    local_environment_summary: CheckResult
    disk_check: CheckResult
    gcs_check: CheckResult
    dashboard_check: CheckResult
    remote_repo_check: CheckResult
    head_start_checks: tuple[CheckResult, ...]
    worker_start_checks: tuple[CheckResult, ...]
    benchmark_checks: tuple[CheckResult, ...]

    @property
    def summary_checks(self) -> tuple[CheckResult, ...]:
        return (
            self.local_environment_summary,
            self.disk_check,
            self.gcs_check,
            self.dashboard_check,
            self.remote_repo_check,
        )


@dataclass(frozen=True)
class TmpRayCleanupResult:
    attempted: bool
    bytes_freed: int = 0
    removed_paths: tuple[str, ...] = ()


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


def requirements_file(root_dir: Path = ROOT_DIR) -> Path:
    return root_dir / "requirements.txt"


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.1f} GiB"


def _disk_headroom_stats(path: Path) -> tuple[int, int, float, float]:
    usage = shutil.disk_usage(path)
    total = max(1, int(usage.total))
    free = int(usage.free)
    used_ratio = 1.0 - (free / total)
    used_pct = used_ratio * 100.0
    return total, free, used_ratio, used_pct


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        return 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    total = 0
    for root, _, files in os.walk(path, topdown=True):
        for filename in files:
            candidate = Path(root) / filename
            try:
                if candidate.is_symlink():
                    continue
                total += int(candidate.stat().st_size)
            except OSError:
                continue
    return total


def local_ray_processes_running(
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 5,
) -> bool:
    active_runner = runner or CommandRunner()
    result = active_runner.run(
        [
            "bash",
            "-lc",
            "pgrep -f 'raylet|gcs_server|dashboard.py|log_monitor.py|runtime_env_agent|ray::' >/dev/null",
        ],
        cwd=ROOT_DIR,
        timeout=timeout,
    )
    return result.returncode == 0


def cleanup_local_ray_tmp(
    ray_tmp_root: Path = RAY_TMP_ROOT,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 5,
) -> TmpRayCleanupResult:
    if not ray_tmp_root.exists():
        return TmpRayCleanupResult(attempted=False)

    active_runner = runner or CommandRunner()
    ray_active = local_ray_processes_running(runner=active_runner, timeout=timeout)
    latest_link = ray_tmp_root / "session_latest"
    latest_target: Optional[Path] = None
    if latest_link.exists() or latest_link.is_symlink():
        try:
            latest_target = latest_link.resolve(strict=False)
        except OSError:
            latest_target = None

    attempted = False
    freed_bytes = 0
    removed_paths: list[str] = []
    for entry in sorted(ray_tmp_root.iterdir()):
        if entry.name == "session_latest" or not entry.name.startswith("session_"):
            continue
        attempted = True
        try:
            resolved_entry = entry.resolve(strict=False)
        except OSError:
            resolved_entry = entry
        if ray_active and latest_target is not None and resolved_entry == latest_target:
            continue
        freed_bytes += _path_size_bytes(entry)
        try:
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                shutil.rmtree(entry)
            removed_paths.append(str(entry))
        except FileNotFoundError:
            continue
        except OSError:
            continue

    if latest_link.is_symlink():
        remove_latest_link = False
        if not ray_active:
            remove_latest_link = True
        elif latest_target is not None and not latest_target.exists():
            remove_latest_link = True
        if remove_latest_link:
            attempted = True
            try:
                latest_link.unlink()
            except OSError:
                pass

    return TmpRayCleanupResult(
        attempted=attempted,
        bytes_freed=freed_bytes,
        removed_paths=tuple(removed_paths),
    )


def blocking_checks(checks: Iterable[CheckResult]) -> list[CheckResult]:
    return [check for check in checks if not check.ok and check.blocking]


def warning_checks(checks: Iterable[CheckResult]) -> list[CheckResult]:
    return [check for check in checks if not check.ok and not check.blocking]


def checks_to_text(checks: Iterable[CheckResult]) -> str:
    lines = []
    for check in checks:
        level = "BLOQUEANTE" if check.blocking else "ADVERTENCIA"
        lines.append(f"- [{level}] {check.name}: {check.detail}")
    return "\n".join(lines)


def _summary_detail(checks: Iterable[CheckResult], success_detail: str) -> str:
    failed = [check.detail for check in checks if not check.ok and check.detail.strip()]
    if failed:
        return " | ".join(failed[:3])
    return success_detail


def summarize_checks(name: str, checks: Iterable[CheckResult], *, success_detail: str) -> CheckResult:
    materialized = list(checks)
    failed = [check for check in materialized if not check.ok]
    return CheckResult(
        name=name,
        ok=not failed,
        detail=_summary_detail(materialized, success_detail),
        blocking=any(check.blocking for check in failed),
    )


def ray_process_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(RAY_MACOS_CLUSTER_ENV)
    return env


def ray_env_export_script() -> str:
    return "\n".join(
        f"export {key}={shlex.quote(value)}"
        for key, value in RAY_MACOS_CLUSTER_ENV.items()
    )


def local_worker_cpus(config: RayClusterConfig) -> int:
    cpus = os.cpu_count() or 1
    return max(1, cpus - max(0, int(config.worker_reserved_cpus)))


def build_worker_start_args(
    config: RayClusterConfig,
    *,
    root_dir: Path = ROOT_DIR,
    block: bool = False,
) -> list[str]:
    args = [
        str(ray_bin(root_dir)),
        "start",
        f"--address={config.ray_address}",
        f"--node-ip-address={config.worker_ip}",
        f"--object-manager-port={config.object_manager_port}",
        f"--node-manager-port={config.node_manager_port}",
        f"--min-worker-port={config.worker_port_min}",
        f"--max-worker-port={config.worker_port_max}",
        f"--num-cpus={local_worker_cpus(config)}",
        "--disable-usage-stats",
    ]
    if block:
        args.append("--block")
    return args


def _tail_file(path: Path, *, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    payload = path.read_text(encoding="utf-8", errors="replace")
    return payload[-max_chars:]


def run_background_command(
    args: Sequence[str],
    *,
    cwd: Path = ROOT_DIR,
    env: Optional[dict[str, str]] = None,
    stdout_path: Path,
    stderr_path: Path,
    startup_wait_s: float = 2.0,
) -> CommandResult:
    display = " ".join(shlex.quote(str(part)) for part in args)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_path.open("ab") as stdout_file, stderr_path.open("ab") as stderr_file:
            process = subprocess.Popen(
                [str(part) for part in args],
                cwd=cwd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
    except FileNotFoundError as exc:
        return CommandResult(ok=False, returncode=127, stderr=str(exc), command=display)
    except OSError as exc:
        return CommandResult(ok=False, returncode=1, stderr=str(exc), command=display)

    time.sleep(max(0.0, startup_wait_s))
    returncode = process.poll()
    if returncode is None:
        return CommandResult(
            ok=True,
            returncode=0,
            stdout=(
                f"Worker local iniciado en segundo plano. PID launcher: {process.pid}\n"
                f"stdout: {stdout_path}\nstderr: {stderr_path}"
            ),
            command=display,
        )

    return CommandResult(
        ok=False,
        returncode=returncode,
        stdout=_tail_file(stdout_path),
        stderr=_tail_file(stderr_path) or f"El proceso termino al iniciar con codigo {returncode}.",
        command=display,
    )


def default_ssh_private_key_path() -> Path:
    return Path.home() / ".ssh" / DEFAULT_SSH_PRIVATE_KEY_FILENAME


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


def _resolved_ssh_private_key_target(config: RayClusterConfig) -> Path:
    try:
        return ssh_private_key_path(config.ssh_key_path)
    except ValueError:
        return default_ssh_private_key_path()


def _is_reserved_ssh_filename(path: Path) -> bool:
    return path.name in INVALID_SSH_KEY_FILENAMES


def _looks_like_public_key_text(payload: str) -> bool:
    stripped = payload.strip()
    return any(stripped.startswith(prefix) for prefix in SSH_PUBLIC_KEY_PREFIXES)


def _looks_like_private_key_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.suffix == ".pub" or _is_reserved_ssh_filename(path):
        return False
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    except OSError:
        return False
    if "PRIVATE KEY" in sample:
        return True
    if _looks_like_public_key_text(sample):
        return False
    return False


def _preferred_generation_key_path(config: RayClusterConfig) -> Path:
    configured = _resolved_ssh_private_key_target(config)
    if configured.suffix != ".pub" and not _is_reserved_ssh_filename(configured):
        return configured
    return default_ssh_private_key_path()


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
    private_key_file = resolved_ssh_private_key_path(config, runner=runner)
    public_key_file = private_key_file.with_name(f"{private_key_file.name}.pub")
    if public_key_file.exists():
        payload = public_key_file.read_text(encoding="utf-8").strip()
        if payload:
            return normalize_public_key(payload)

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
    return (
        f"sudo networksetup -setmanual {shlex.quote(THUNDERBOLT_BRIDGE_NAME)} "
        f"{shlex.quote(ip)} {shlex.quote(netmask)} 0.0.0.0"
    )


def automatic_bridge_config(config: RayClusterConfig) -> RayClusterConfig:
    return replace(
        config,
        head_ip=DEFAULT_HEAD_IP,
        worker_ip=DEFAULT_WORKER_IP,
        netmask=DEFAULT_NETMASK,
    )


def _bridge_set_command(ip: str, netmask: str = DEFAULT_NETMASK) -> list[str]:
    return [
        "networksetup",
        "-setmanual",
        THUNDERBOLT_BRIDGE_NAME,
        ip,
        netmask,
        "0.0.0.0",
    ]


def _macos_admin_command(args: Sequence[str]) -> list[str]:
    shell_command = " ".join(shlex.quote(str(part)) for part in args)
    applescript = f"do shell script {json.dumps(shell_command)} with administrator privileges"
    return ["osascript", "-e", applescript]


def configure_local_bridge(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> CommandResult:
    active_runner = runner or CommandRunner()
    effective = automatic_bridge_config(config)
    return active_runner.run(
        _macos_admin_command(_bridge_set_command(effective.head_ip, effective.netmask)),
        timeout=max(20, effective.command_timeout_s),
    )


def _remote_bridge_apply_script(ip: str, netmask: str) -> str:
    bridge_cmd = " ".join(shlex.quote(part) for part in _bridge_set_command(ip, netmask))
    applescript = f"do shell script {json.dumps(bridge_cmd)} with administrator privileges"
    return " || ".join(
        [
            f"osascript -e {shlex.quote(applescript)}",
            f"sudo -n {bridge_cmd}",
            bridge_cmd,
        ]
    )


def configure_remote_bridge(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> CommandResult:
    effective = automatic_bridge_config(config)
    return run_remote_script(
        effective,
        _remote_bridge_apply_script(effective.worker_ip, effective.netmask),
        runner=runner,
        timeout=max(20, effective.command_timeout_s),
    )


def apply_automatic_bridge(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> list[CommandResult]:
    active_runner = runner or CommandRunner()
    effective = automatic_bridge_config(config)
    return [
        configure_local_bridge(effective, runner=active_runner),
        configure_remote_bridge(effective, runner=active_runner),
    ]


def ensure_local_ssh_identity(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> tuple[CommandResult, Path]:
    key_path = _resolved_ssh_private_key_target(config)
    if _looks_like_private_key_file(key_path):
        return (
            CommandResult(
                ok=True,
                returncode=0,
                stdout=f"Llave SSH local lista: {key_path}",
                command="ssh key auto-detect",
            ),
            key_path,
        )

    for candidate in detect_private_keys():
        if _looks_like_private_key_file(candidate):
            return (
                CommandResult(
                    ok=True,
                    returncode=0,
                    stdout=f"Llave SSH local lista: {candidate}",
                    command="ssh key auto-detect",
                ),
                candidate,
            )

    key_path = _preferred_generation_key_path(config)
    if _looks_like_private_key_file(key_path):
        return (
            CommandResult(
                ok=True,
                returncode=0,
                stdout=f"Llave SSH local lista: {key_path}",
                command="ssh key auto-detect",
            ),
            key_path,
        )

    ssh_dir = key_path.parent
    ssh_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        ssh_dir.chmod(0o700)
    except OSError:
        pass

    active_runner = runner or CommandRunner()
    comment = f"{os.environ.get('USER', 'sumo')}@ray-cluster"
    result = active_runner.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-f",
            str(key_path),
            "-N",
            "",
            "-C",
            comment,
        ],
        timeout=min(20, max(10, int(config.command_timeout_s))),
    )
    if result.ok and key_path.exists():
        return (
            CommandResult(
                ok=True,
                returncode=0,
                stdout=f"Llave SSH local generada: {key_path}",
                command=result.command,
            ),
            key_path,
        )
    return (
        CommandResult(
            ok=False,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr or f"No se pudo generar la llave SSH local en {key_path}.",
            command=result.command,
            timed_out=result.timed_out,
        ),
        key_path,
    )


def resolved_ssh_private_key_path(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> Path:
    result, key_path = ensure_local_ssh_identity(config, runner=runner)
    if result.ok and key_path.exists():
        return key_path
    raise RuntimeError(result.combined_output or f"No se pudo preparar la llave SSH local en {key_path}.")


def ssh_target(config: RayClusterConfig) -> str:
    return f"{config.ssh_user}@{config.worker_ip}"


def ssh_check(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 8,
) -> CommandResult:
    active_runner = runner or CommandRunner()
    try:
        args = [*ssh_base_args(config, runner=active_runner), "true"]
    except (ValueError, RuntimeError) as exc:
        return CommandResult(
            ok=False,
            returncode=2,
            stderr=str(exc),
            command="ssh true",
        )
    return active_runner.run(args, timeout=timeout)


def _tcl_quote(value: str) -> str:
    return "{" + value.replace("\\", "\\\\").replace("}", "\\}") + "}"


def _ssh_copy_id_expect_script(copy_id_args: Sequence[str]) -> str:
    command_list = " ".join(_tcl_quote(str(part)) for part in copy_id_args)
    return """
set timeout 30
set password $env(SUMO_RAY_SSH_PASSWORD)
set cmd [list {command_list}]
spawn {{*}}$cmd
expect {{
    -re "(?i)continue connecting.*" {{
        send -- "yes\r"
        exp_continue
    }}
    -re "(?i)(password|contrasena|contraseña).*:" {{
        send -- "$password\r"
        exp_continue
    }}
    eof {{
        catch wait result
        set rc [lindex $result 3]
        exit $rc
    }}
}}
""".format(command_list=command_list)


def bootstrap_ssh_access(
    config: RayClusterConfig,
    password: str,
    *,
    runner: Optional[CommandRunner] = None,
) -> CommandResult:
    effective = automatic_bridge_config(config)
    active_runner = runner or CommandRunner()
    if not password.strip():
        return CommandResult(
            ok=False,
            returncode=2,
            stderr="Ingrese la password del worker para autorizar la llave SSH automaticamente.",
            command="ssh-copy-id",
        )
    key_result, private_key = ensure_local_ssh_identity(effective, runner=active_runner)
    if not key_result.ok:
        return key_result
    public_key = private_key.with_name(f"{private_key.name}.pub")
    if not public_key.exists():
        try:
            public_payload = read_public_key(effective, runner=active_runner)
        except Exception as exc:
            return CommandResult(
                ok=False,
                returncode=2,
                stderr=str(exc),
                command="ssh-copy-id",
            )
        public_key.write_text(f"{public_payload}\n", encoding="utf-8")
    env = dict(os.environ)
    env[SSH_BOOTSTRAP_PASSWORD_ENV] = password
    copy_id_args = [
        "ssh-copy-id",
        "-i",
        str(public_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        ssh_target(effective),
    ]
    return active_runner.run(
        [
            "expect",
            "-c",
            _ssh_copy_id_expect_script(copy_id_args),
        ],
        timeout=min(45, max(15, int(effective.command_timeout_s) * 2)),
        env=env,
    )


def prepare_ssh_access(
    config: RayClusterConfig,
    password: str = "",
    *,
    runner: Optional[CommandRunner] = None,
) -> list[CommandResult]:
    effective = automatic_bridge_config(config)
    active_runner = runner or CommandRunner()
    key_result, _ = ensure_local_ssh_identity(effective, runner=active_runner)
    if not key_result.ok:
        return [key_result]

    precheck = ssh_check(effective, runner=active_runner, timeout=8)
    if precheck.ok:
        return [key_result, precheck]

    if not password.strip():
        return [
            key_result,
            CommandResult(
                ok=False,
                returncode=2,
                stderr=(
                    "SSH aun no conecta. Ingrese la password del worker una sola vez para "
                    "autorizar la llave automaticamente."
                ),
                command="ssh bootstrap required",
            ),
        ]

    bootstrap = bootstrap_ssh_access(effective, password, runner=active_runner)
    postcheck = ssh_check(effective, runner=active_runner, timeout=8)
    return [key_result, bootstrap, postcheck]


def build_head_start_args(
    config: RayClusterConfig,
    *,
    root_dir: Path = ROOT_DIR,
    block: bool = False,
) -> list[str]:
    args = [
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
    if block:
        args.append("--block")
    return args


def build_worker_start_script(config: RayClusterConfig) -> str:
    repo = shlex.quote(config.remote_repo_path)
    reserved = max(0, int(config.worker_reserved_cpus))
    return "\n".join(
        [
            ray_env_export_script(),
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


def ssh_base_args(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> list[str]:
    private_key = resolved_ssh_private_key_path(config, runner=runner)
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
        ssh_target(config),
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
        ssh_args = ssh_base_args(config, runner=active_runner)
    except (ValueError, RuntimeError) as exc:
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


def stop_local_worker(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    active_runner = runner or CommandRunner()
    return active_runner.run(
        [str(ray_bin()), "stop"],
        cwd=ROOT_DIR,
        timeout=config.command_timeout_s,
        env=ray_process_env(),
    )


def start_local_worker(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    active_runner = runner or CommandRunner()
    blockers = blocking_checks(head_start_health_checks(config, runner=active_runner))
    if blockers:
        return health_failure_result(
            blockers,
            command="ray start worker local",
            prefix="No se puede iniciar el worker local mientras existan checks bloqueantes.",
        )
    stop_local_worker(config, runner=active_runner)
    return run_background_command(
        build_worker_start_args(config, block=True),
        cwd=ROOT_DIR,
        env=ray_process_env(),
        stdout_path=CONFIG_DIR / "worker_local.out.log",
        stderr_path=CONFIG_DIR / "worker_local.err.log",
        startup_wait_s=3.0,
    )


def start_head(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    active_runner = runner or CommandRunner()
    blockers = blocking_checks(head_start_health_checks(config, runner=active_runner))
    if blockers:
        return health_failure_result(
            blockers,
            command="ray start --head",
            prefix="No se puede iniciar el head mientras existan checks bloqueantes.",
        )
    stop_head(config, runner=active_runner)
    return run_background_command(
        build_head_start_args(config, block=True),
        cwd=ROOT_DIR,
        env=ray_process_env(),
        stdout_path=CONFIG_DIR / "head.out.log",
        stderr_path=CONFIG_DIR / "head.err.log",
        startup_wait_s=3.0,
    )


def stop_worker(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    return run_remote_script(
        config,
        f"cd {shlex.quote(config.remote_repo_path)} && .venv/bin/ray stop",
        runner=runner,
        timeout=config.command_timeout_s,
    )


def start_worker(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> CommandResult:
    blockers = blocking_checks(worker_start_health_checks(config, runner=runner))
    if blockers:
        return health_failure_result(
            blockers,
            command="ray start worker remoto",
            prefix="No se puede iniciar el worker remoto mientras existan checks bloqueantes.",
        )
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
    blockers = blocking_checks(worker_start_health_checks(config, runner=active_runner))
    if blockers:
        return [
            health_failure_result(
                blockers,
                command="ray start cluster",
                prefix="No se puede iniciar el cluster mientras existan checks bloqueantes.",
            )
        ]
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
        env=ray_process_env(),
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


def _worker_port_check_script(config: RayClusterConfig) -> str:
    ports = " ".join(str(port) for port in [config.object_manager_port, config.node_manager_port])
    return (
        "busy=''; "
        f"for port in {ports}; do "
        "if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then busy=\"$busy $port\"; fi; "
        "done; "
        "if [ -z \"$busy\" ]; then echo 'Puertos worker libres'; "
        "else echo \"Puertos worker ocupados:$busy\"; exit 1; fi"
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
        blocking=not result.ok,
    )


def check_worker_ports_available(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> CheckResult:
    active_runner = runner or CommandRunner()
    result = active_runner.run(["bash", "-lc", _worker_port_check_script(config)], cwd=ROOT_DIR, timeout=8)
    return CheckResult(
        "Puertos worker local",
        result.ok,
        result.combined_output or ("Puertos worker libres" if result.ok else "No se pudo verificar puertos worker."),
        result.command,
        blocking=not result.ok,
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


def _bridge_expected_detail(ip: str, netmask: str) -> str:
    return f"{THUNDERBOLT_BRIDGE_NAME} debe quedar en {ip}/{netmask}."


def _is_executable_file(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _venv_missing_entries(venv_path: Path) -> list[str]:
    expected = (
        venv_path / "bin" / "python",
        venv_path / "bin" / "ray",
    )
    return [str(path.relative_to(venv_path)) for path in expected if not path.exists()]


def sync_local_venv(
    *,
    root_dir: Path = ROOT_DIR,
    venv_path: Optional[Path] = None,
    requirements_path: Optional[Path] = None,
    runner: Optional[CommandRunner] = None,
    timeout: int = 600,
) -> CommandResult:
    effective_root = root_dir.expanduser()
    effective_venv = (venv_path or (effective_root / ".venv")).expanduser()
    effective_requirements = (requirements_path or requirements_file(effective_root)).expanduser()
    python_path = effective_venv / "bin" / "python"

    if not effective_requirements.exists():
        return CommandResult(
            ok=False,
            returncode=2,
            stderr=f"No existe {effective_requirements}.",
            command=f"{python_path} -m pip install -r {effective_requirements}",
        )
    if not _is_executable_file(python_path):
        return CommandResult(
            ok=False,
            returncode=2,
            stderr=f"No se puede resincronizar automaticamente porque falta {python_path} o no es ejecutable.",
            command=f"{python_path} -m pip install -r {effective_requirements}",
        )

    active_runner = runner or CommandRunner()
    result = active_runner.run(
        [str(python_path), "-m", "pip", "install", "-r", str(effective_requirements)],
        cwd=effective_root,
        timeout=timeout,
    )
    if result.ok:
        try:
            os.utime(effective_venv, None)
        except OSError:
            pass
    return result


def _tcp_check(
    host: str,
    port: int,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 5,
) -> CommandResult:
    active_runner = runner or CommandRunner()
    script = (
        "import socket, sys\n"
        "host = sys.argv[1]\n"
        "port = int(sys.argv[2])\n"
        "sock = socket.create_connection((host, port), timeout=3)\n"
        "sock.close()\n"
        "print(f'{host}:{port} abierto')\n"
    )
    return active_runner.run(["python3", "-c", script, host, str(port)], timeout=timeout)


def check_local_venv_sync(
    *,
    root_dir: Path = ROOT_DIR,
    venv_path: Optional[Path] = None,
    requirements_path: Optional[Path] = None,
    runner: Optional[CommandRunner] = None,
    timeout: int = 600,
) -> CheckResult:
    effective_venv = (venv_path or (root_dir / ".venv")).expanduser()
    effective_requirements = (requirements_path or requirements_file(root_dir)).expanduser()
    if not effective_requirements.exists():
        return CheckResult(
            "Venv local",
            False,
            f"No existe {effective_requirements}.",
            blocking=True,
        )
    if not effective_venv.exists():
        return CheckResult(
            "Venv local",
            False,
            f"No existe {effective_venv}.",
            blocking=True,
        )
    missing = _venv_missing_entries(effective_venv)
    if missing:
        sync_result: CommandResult | None = None
        if _is_executable_file(effective_venv / "bin" / "python"):
            sync_result = sync_local_venv(
                root_dir=root_dir,
                venv_path=effective_venv,
                requirements_path=effective_requirements,
                runner=runner,
                timeout=timeout,
            )
            if sync_result.ok:
                missing = _venv_missing_entries(effective_venv)
                if not missing and effective_venv.stat().st_mtime >= effective_requirements.stat().st_mtime:
                    return CheckResult(
                        "Venv local",
                        True,
                        f"Venv local resincronizada automaticamente con {effective_requirements.name}.",
                        sync_result.command,
                    )
        return CheckResult(
            "Venv local",
            False,
            (
                f"La venv local no esta completa; faltan {', '.join(missing)}."
                if sync_result is None
                else (
                    f"La venv local no esta completa; faltan {', '.join(missing)}. "
                    f"La resincronizacion automatica no alcanzo a corregirlo: {_result_detail(sync_result, '')}"
                )
            ),
            sync_result.command if sync_result else "",
            blocking=True,
        )
    if effective_venv.stat().st_mtime < effective_requirements.stat().st_mtime:
        sync_result = sync_local_venv(
            root_dir=root_dir,
            venv_path=effective_venv,
            requirements_path=effective_requirements,
            runner=runner,
            timeout=timeout,
        )
        if sync_result.ok and effective_venv.stat().st_mtime >= effective_requirements.stat().st_mtime:
            return CheckResult(
                "Venv local",
                True,
                f"Venv local resincronizada automaticamente con {effective_requirements.name}.",
                sync_result.command,
            )
        return CheckResult(
            "Venv local",
            False,
            (
                f"La venv local ({effective_venv}) es mas antigua que {effective_requirements.name}. "
                "La resincronizacion automatica no pudo completarse: "
                f"{_result_detail(sync_result, '')}"
            ),
            sync_result.command,
            blocking=True,
        )
    return CheckResult(
        "Venv local",
        True,
        f"Venv local sincronizada con {effective_requirements.name}.",
    )


def check_local_python_version(
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 5,
) -> CheckResult:
    active_runner = runner or CommandRunner()
    result = active_runner.run([str(python_bin()), "--version"], timeout=timeout)
    return CheckResult(
        "Python local",
        result.ok and "Python 3.12" in result.combined_output,
        _result_detail(result, result.combined_output.strip() or "Python 3.12 detectado."),
        result.command,
        blocking=not (result.ok and "Python 3.12" in result.combined_output),
    )


def check_local_ray_version(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 5,
) -> CheckResult:
    active_runner = runner or CommandRunner()
    result = active_runner.run([str(ray_bin()), "--version"], timeout=timeout)
    ok = result.ok and _version_ok(result.combined_output, config.ray_version)
    return CheckResult(
        "Ray local",
        ok,
        _result_detail(result, result.combined_output.strip() or f"Ray {config.ray_version} detectado."),
        result.command,
        blocking=not ok,
    )


def check_local_ray_dependencies(
    *,
    root_dir: Path = ROOT_DIR,
    runner: Optional[CommandRunner] = None,
    timeout: int = 10,
    sync_timeout: int = 600,
) -> CheckResult:
    def run_dependency_probe() -> CommandResult:
        script = (
            "import importlib, json, sys\n"
            "missing = []\n"
            "for name in sys.argv[1:]:\n"
            "    try:\n"
            "        importlib.import_module(name)\n"
            "    except Exception as exc:\n"
            "        missing.append({'module': name, 'error': f'{type(exc).__name__}: {exc}'})\n"
            "if missing:\n"
            "    print(json.dumps({'missing': missing}, ensure_ascii=False))\n"
            "    raise SystemExit(1)\n"
            "print('OK')\n"
        )
        return active_runner.run(
            [str(python_bin(root_dir)), "-c", script, *REQUIRED_RAY_PYTHON_MODULES],
            timeout=timeout,
        )

    def parse_missing_modules(result: CommandResult) -> list[str]:
        missing_modules: list[str] = []
        payload = result.stdout.strip()
        if payload.startswith("{"):
            try:
                data = json.loads(payload)
                missing_modules = [
                    entry.get("module", "")
                    for entry in data.get("missing", [])
                    if entry.get("module")
                ]
            except Exception:
                missing_modules = []
        return missing_modules

    active_runner = runner or CommandRunner()
    result = run_dependency_probe()
    if result.ok:
        return CheckResult(
            "Extras Ray local",
            True,
            "Extras de dashboard y metrics presentes en la venv local.",
            result.command,
        )

    missing_modules = parse_missing_modules(result)
    if missing_modules:
        sync_result = sync_local_venv(
            root_dir=root_dir,
            runner=active_runner,
            timeout=sync_timeout,
        )
        if sync_result.ok:
            retry = run_dependency_probe()
            if retry.ok:
                return CheckResult(
                    "Extras Ray local",
                    True,
                    "Extras de dashboard y metrics restaurados automaticamente en la venv local.",
                    sync_result.command,
                )
            missing_modules = parse_missing_modules(retry)
            if missing_modules:
                detail = (
                    "Faltan modulos de Ray en la venv local incluso despues de la resincronizacion automatica: "
                    + ", ".join(missing_modules)
                    + "."
                )
            else:
                detail = _result_detail(
                    retry,
                    "No se pudo validar los extras de Ray en la venv local despues de la resincronizacion automatica.",
                )
            return CheckResult(
                "Extras Ray local",
                False,
                detail,
                retry.command,
                blocking=True,
            )

        detail = (
            "Faltan modulos de Ray en la venv local: "
            + ", ".join(missing_modules)
            + ". La resincronizacion automatica fallo: "
            + _result_detail(sync_result, "")
        )
        return CheckResult(
            "Extras Ray local",
            False,
            detail,
            sync_result.command,
            blocking=True,
        )

    detail = _result_detail(
        result,
        "No se pudo validar los extras de Ray en la venv local.",
    )
    return CheckResult(
        "Extras Ray local",
        False,
        detail,
        result.command,
        blocking=True,
    )


def check_tmp_disk_headroom(
    path: Path = Path("/tmp"),
    *,
    attempt_cleanup: bool = False,
    runner: Optional[CommandRunner] = None,
    ray_tmp_root: Optional[Path] = None,
) -> CheckResult:
    _, free, used_ratio, used_pct = _disk_headroom_stats(path)
    cleanup_result: Optional[TmpRayCleanupResult] = None
    needs_attention = used_ratio >= DISK_WARNING_USED_RATIO or free < DISK_WARNING_FREE_BYTES
    if attempt_cleanup and needs_attention:
        cleanup_result = cleanup_local_ray_tmp(
            ray_tmp_root=(ray_tmp_root or (path / "ray")),
            runner=runner,
        )
        if cleanup_result.attempted and cleanup_result.bytes_freed > 0:
            _, free, used_ratio, used_pct = _disk_headroom_stats(path)

    detail_parts = [f"{path} usa {used_pct:.1f}% del volumen y deja {format_gib(free)} libres."]
    if cleanup_result and cleanup_result.attempted:
        if cleanup_result.bytes_freed > 0:
            detail_parts.append(
                "Limpieza automatica de Ray libero "
                f"{format_gib(cleanup_result.bytes_freed)} en {len(cleanup_result.removed_paths)} sesiones."
            )
        else:
            detail_parts.append(
                "La limpieza automatica de Ray no encontro temporales ociosos suficientes para liberar espacio."
            )
    detail = " ".join(detail_parts)
    if used_ratio >= DISK_BLOCK_USED_RATIO or free < DISK_BLOCK_FREE_BYTES:
        suffix = (
            " La limpieza automatica no alcanzo el margen minimo requerido por Ray."
            if cleanup_result and cleanup_result.attempted
            else " Libere espacio antes de iniciar Ray."
        )
        return CheckResult(
            "Disco /tmp",
            False,
            detail + suffix,
            blocking=True,
        )
    if used_ratio >= DISK_WARNING_USED_RATIO or free < DISK_WARNING_FREE_BYTES:
        return CheckResult(
            "Disco /tmp",
            False,
            detail + " Hay poco margen para spill y logs de Ray.",
            blocking=False,
        )
    return CheckResult(
        "Disco /tmp",
        True,
        detail + " Hay margen suficiente para Ray.",
    )


def check_gcs_health(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 10,
) -> CheckResult:
    result = ray_status(config, runner=runner, timeout=timeout)
    return CheckResult(
        "GCS head",
        result.ok,
        _result_detail(result, f"Ray responde en {config.ray_address}."),
        result.command,
        blocking=not result.ok,
    )


def check_dashboard_available(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 5,
) -> CheckResult:
    active_runner = runner or CommandRunner()
    result = active_runner.run(
        ["bash", "-lc", f"lsof -nP -iTCP:{config.dashboard_port} -sTCP:LISTEN"],
        cwd=ROOT_DIR,
        timeout=timeout,
    )
    if result.ok:
        return CheckResult(
            "Dashboard Ray",
            True,
            f"Dashboard disponible en {config.dashboard_url}.",
            result.command,
        )
    return CheckResult(
        "Dashboard Ray",
        False,
        (
            f"Dashboard no disponible en {config.dashboard_url}. "
            "Si el head esta iniciado, revise la instalacion de ray[default] y dashboard.log."
        ),
        result.command,
        blocking=False,
    )


def check_remote_repo_path(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int = 10,
) -> CheckResult:
    repo_path = config.remote_repo_path.strip()
    if not config.ssh_user.strip():
        return CheckResult(
            "Repo worker",
            False,
            "Falta el usuario SSH del worker.",
            blocking=True,
        )
    if not repo_path:
        return CheckResult(
            "Repo worker",
            False,
            "Falta la ruta del repo en el worker.",
            blocking=True,
        )
    import_probe = (
        "import sys; "
        f"sys.path.insert(0, {repo_path!r}); "
        "import src"
    )
    script = (
        f"repo={shlex.quote(repo_path)}; "
        "[ -d \"$repo\" ] || { echo \"Ruta remota inexistente: $repo\"; exit 1; }; "
        "missing=''; "
        "for rel in .venv/bin/python .venv/bin/ray requirements.txt src; do "
        "[ -e \"$repo/$rel\" ] || missing=\"$missing $rel\"; "
        "done; "
        "if [ -n \"$missing\" ]; then "
        "echo \"Ruta remota invalida: faltan$missing en $repo\"; exit 1; "
        "fi; "
        f"\"$repo/.venv/bin/python\" -c {shlex.quote(import_probe)} >/dev/null 2>&1 "
        "|| { echo \"Ruta remota invalida: no se pudo importar src desde $repo\"; exit 1; }; "
        "echo \"Ruta remota valida: $repo\""
    )
    result = run_remote_script(config, script, runner=runner, timeout=timeout)
    return CheckResult(
        "Repo worker",
        result.ok,
        _result_detail(result, f"Ruta remota valida: {repo_path}."),
        result.command,
        blocking=not result.ok,
    )


def local_ray_environment_checks(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> list[CheckResult]:
    return [
        check_local_venv_sync(runner=runner),
        check_local_python_version(runner=runner),
        check_local_ray_version(config, runner=runner),
        check_local_ray_dependencies(runner=runner),
    ]


def head_start_health_checks(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> list[CheckResult]:
    return [
        *local_ray_environment_checks(config, runner=runner),
        check_tmp_disk_headroom(attempt_cleanup=True, runner=runner),
    ]


def worker_start_health_checks(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> list[CheckResult]:
    return [
        *head_start_health_checks(config, runner=runner),
        check_remote_repo_path(config, runner=runner),
    ]


def runtime_connection_health_checks(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> list[CheckResult]:
    return [
        *head_start_health_checks(config, runner=runner),
        check_gcs_health(config, runner=runner),
    ]


def collect_health_snapshot(
    config: RayClusterConfig,
    *,
    runner: Optional[CommandRunner] = None,
) -> RayHealthSnapshot:
    active_runner = runner or CommandRunner()
    local_env_checks = tuple(local_ray_environment_checks(config, runner=active_runner))
    disk_check = check_tmp_disk_headroom(attempt_cleanup=True, runner=active_runner)
    gcs_check = check_gcs_health(config, runner=active_runner)
    dashboard_check = check_dashboard_available(config, runner=active_runner)
    remote_repo_check = check_remote_repo_path(config, runner=active_runner)
    head_checks = tuple([*local_env_checks, disk_check])
    worker_checks = tuple([*head_checks, remote_repo_check])
    benchmark_checks = tuple([*head_checks, gcs_check])
    return RayHealthSnapshot(
        local_environment_checks=local_env_checks,
        local_environment_summary=summarize_checks(
            "Entorno local Ray",
            local_env_checks,
            success_detail="Entorno local Ray listo para head, worker y dashboard.",
        ),
        disk_check=disk_check,
        gcs_check=gcs_check,
        dashboard_check=dashboard_check,
        remote_repo_check=remote_repo_check,
        head_start_checks=head_checks,
        worker_start_checks=worker_checks,
        benchmark_checks=benchmark_checks,
    )


def health_failure_result(checks: Iterable[CheckResult], *, command: str, prefix: str) -> CommandResult:
    return CommandResult(
        ok=False,
        returncode=2,
        stderr=f"{prefix}\n{checks_to_text(checks)}",
        command=command,
    )


def run_preflight(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> list[CheckResult]:
    active_runner = runner or CommandRunner()
    effective = automatic_bridge_config(config)
    checks: list[CheckResult] = []

    checks.extend(head_start_health_checks(effective, runner=active_runner))

    local_bridge = active_runner.run(["ifconfig", "bridge0"], timeout=5)
    checks.append(
        CheckResult(
            "Thunderbolt local",
            local_bridge.ok and _bridge_has_ip(local_bridge.stdout, effective.head_ip) and _bridge_active(local_bridge.stdout),
            (
                f"bridge0 activo con {effective.head_ip}."
                if local_bridge.ok and _bridge_has_ip(local_bridge.stdout, effective.head_ip) and _bridge_active(local_bridge.stdout)
                else _bridge_expected_detail(effective.head_ip, effective.netmask)
            ),
            local_bridge.command,
            blocking=True,
        )
    )

    ssh_result = ssh_check(effective, runner=active_runner, timeout=8)
    checks.append(
        CheckResult(
            "SSH worker",
            ssh_result.ok,
            _result_detail(ssh_result, f"SSH OK hacia {effective.ssh_user}@{effective.worker_ip}."),
            ssh_result.command,
            blocking=True,
        )
    )

    ping_worker = active_runner.run(["ping", "-c", "2", "-W", "1000", effective.worker_ip], timeout=6)
    checks.append(
        CheckResult(
            "Ping worker",
            ping_worker.ok,
            _result_detail(ping_worker, f"{effective.worker_ip} responde por Thunderbolt."),
            ping_worker.command,
            blocking=False,
        )
    )

    if ssh_result.ok:
        remote_bridge = run_remote_script(effective, "ifconfig bridge0", runner=active_runner, timeout=8)
        checks.append(
            CheckResult(
                "Thunderbolt worker",
                remote_bridge.ok and _bridge_has_ip(remote_bridge.stdout, effective.worker_ip) and _bridge_active(remote_bridge.stdout),
                (
                    f"bridge0 activo con {effective.worker_ip}."
                    if remote_bridge.ok and _bridge_has_ip(remote_bridge.stdout, effective.worker_ip) and _bridge_active(remote_bridge.stdout)
                    else _bridge_expected_detail(effective.worker_ip, effective.netmask)
                ),
                remote_bridge.command,
                blocking=True,
            )
        )

        repo_worker = check_remote_repo_path(effective, runner=active_runner, timeout=8)
        checks.append(repo_worker)
        if repo_worker.ok:
            remote_python = run_remote_script(
                effective,
                f"cd {shlex.quote(effective.remote_repo_path)} && .venv/bin/python --version",
                runner=active_runner,
                timeout=8,
            )
            checks.append(
                CheckResult(
                    "Python worker",
                    remote_python.ok and "Python 3.12" in remote_python.combined_output,
                    _result_detail(remote_python, remote_python.combined_output.strip()),
                    remote_python.command,
                    blocking=not (remote_python.ok and "Python 3.12" in remote_python.combined_output),
                )
            )

            remote_ray = run_remote_script(
                effective,
                f"cd {shlex.quote(effective.remote_repo_path)} && .venv/bin/ray --version",
                runner=active_runner,
                timeout=8,
            )
            checks.append(
                CheckResult(
                    "Ray worker",
                    remote_ray.ok and _version_ok(remote_ray.combined_output, effective.ray_version),
                    _result_detail(remote_ray, remote_ray.combined_output.strip()),
                    remote_ray.command,
                    blocking=not (remote_ray.ok and _version_ok(remote_ray.combined_output, effective.ray_version)),
                )
            )
        else:
            checks.extend(
                [
                    CheckResult("Python worker", False, "Omitido porque la ruta remota del repo es invalida."),
                    CheckResult("Ray worker", False, "Omitido porque la ruta remota del repo es invalida."),
                ]
            )
    else:
        checks.append(
            CheckResult("Thunderbolt worker", False, "Omitido porque SSH no conecta.")
        )
        checks.append(CheckResult("Repo worker", False, "Omitido porque SSH no conecta.", blocking=True))
        checks.append(CheckResult("Python worker", False, "Omitido porque SSH no conecta."))
        checks.append(CheckResult("Ray worker", False, "Omitido porque SSH no conecta."))

    stop_probe = stop_head(effective, runner=active_runner)
    checks.append(
        CheckResult(
            "ray stop local",
            stop_probe.returncode in (0, 1),
            "ray stop local ejecutado; el head queda limpio para iniciar.",
            stop_probe.command,
        )
    )
    checks.append(check_ports_available(effective, runner=active_runner))
    if ssh_result.ok:
        remote_stop = stop_worker(effective, runner=active_runner)
        checks.append(
            CheckResult(
                "ray stop worker",
                remote_stop.returncode in (0, 1),
                "ray stop worker ejecutado; el worker queda limpio para iniciar.",
                remote_stop.command,
            )
        )
        checks.append(check_ports_available(effective, remote=True, runner=active_runner))
    else:
        checks.append(CheckResult("ray stop worker", False, "Omitido porque SSH no conecta."))
        checks.append(CheckResult("Puertos worker", False, "Omitido porque SSH no conecta."))

    return checks


def run_worker_preflight(config: RayClusterConfig, *, runner: Optional[CommandRunner] = None) -> list[CheckResult]:
    active_runner = runner or CommandRunner()
    effective = automatic_bridge_config(config)
    checks: list[CheckResult] = head_start_health_checks(effective, runner=active_runner)

    local_bridge = active_runner.run(["ifconfig", "bridge0"], timeout=5)
    bridge_ok = (
        local_bridge.ok
        and _bridge_has_ip(local_bridge.stdout, effective.worker_ip)
        and _bridge_active(local_bridge.stdout)
    )
    checks.append(
        CheckResult(
            "Thunderbolt worker local",
            bridge_ok,
            (
                f"bridge0 activo con {effective.worker_ip}."
                if bridge_ok
                else _bridge_expected_detail(effective.worker_ip, effective.netmask)
            ),
            local_bridge.command,
            blocking=True,
        )
    )

    ping_head = active_runner.run(["ping", "-c", "2", "-W", "1000", effective.head_ip], timeout=6)
    checks.append(
        CheckResult(
            "Ping head",
            ping_head.ok,
            _result_detail(ping_head, f"{effective.head_ip} responde por Thunderbolt."),
            ping_head.command,
            blocking=False,
        )
    )

    head_gcs = _tcp_check(effective.head_ip, effective.head_port, runner=active_runner, timeout=5)
    checks.append(
        CheckResult(
            "Puerto head Ray",
            head_gcs.ok,
            _result_detail(head_gcs, f"Head Ray escucha en {effective.ray_address}."),
            head_gcs.command,
            blocking=True,
        )
    )

    status = ray_status(effective, runner=active_runner, timeout=10)
    checks.append(
        CheckResult(
            "ray status head",
            status.ok,
            _result_detail(status, f"Ray responde desde {effective.ray_address}."),
            status.command,
            blocking=not status.ok,
        )
    )

    stop_probe = stop_local_worker(effective, runner=active_runner)
    checks.append(
        CheckResult(
            "ray stop worker local",
            stop_probe.returncode in (0, 1),
            "ray stop local ejecutado; el worker queda limpio para iniciar.",
            stop_probe.command,
        )
    )
    checks.append(check_worker_ports_available(effective, runner=active_runner))

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
        if line in {"Usage:", "Total Usage:"}:
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
    blockers = blocking_checks(runtime_connection_health_checks(config, runner=active_runner))
    if blockers:
        return (
            health_failure_result(
                blockers,
                command="ray distributed benchmark",
                prefix="No se puede ejecutar el benchmark mientras existan checks bloqueantes.",
            ),
            None,
        )
    result = active_runner.run(
        [str(python_bin()), "-c", build_benchmark_script(config, tasks=tasks)],
        cwd=ROOT_DIR,
        timeout=timeout,
        env=ray_process_env(),
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
