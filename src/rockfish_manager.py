#!/usr/bin/env python3
"""Backend helpers for managing the JHU Rockfish HPC cluster from Streamlit.

The design mirrors ``src/ray_cluster_manager.py``: pure-python dataclasses for
configuration plus a small command runner that shells out to ``ssh``/``rsync``
on the user's local machine. All cluster operations go through a multiplexed
SSH ControlMaster socket that the user authenticates once per day.
"""
from __future__ import annotations

import csv
import io
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "Resultados" / "rockfish_cluster"
CONFIG_FILE = CONFIG_DIR / "config.json"
SBATCH_TEMPLATE = ROOT_DIR / "scripts" / "slurm" / "gnn_experiments.sbatch.tmpl"
SETUP_SCRIPT = ROOT_DIR / "scripts" / "slurm" / "setup_rockfish_env.sh"

DEFAULT_SSH_HOST = "login.rockfish.jhu.edu"
DEFAULT_DTN_HOST = "rfdtn1.rockfish.jhu.edu"
DEFAULT_QOS = "qos_gpu"
DEFAULT_PARTITION = "ica100"
DEFAULT_TIME = "24:00:00"
DEFAULT_GPUS = 1
DEFAULT_CPUS = 8
DEFAULT_MEM_GB = 64

GPU_PARTITIONS: dict[str, dict[str, Any]] = {
    "a100": {"gpu": "A100 40GB", "max_gpus": 4, "max_walltime": "3-00:00:00"},
    "ica100": {"gpu": "A100 80GB", "max_gpus": 4, "max_walltime": "3-00:00:00"},
    "l40s": {"gpu": "L40s 48GB", "max_gpus": 8, "max_walltime": "1-00:00:00"},
    "mig_class": {"gpu": "A100 MIG 20GB", "max_gpus": 12, "max_walltime": "1-00:00:00"},
}

RSYNC_DEFAULT_EXCLUDES: tuple[str, ...] = (
    "Datos/",
    "Resultados/",
    "cache/",
    "venv*/",
    ".venv*/",
    "__pycache__/",
    "*.pyc",
    ".git/",
    "dask/",
    ".mypy_cache/",
    ".pytest_cache/",
    "docs/",
    "papers/",
    "simulación/",
    "NLP/",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RockfishConfig:
    """Persisted configuration for the Rockfish cluster integration."""

    jhed: str = ""
    pi_account: str = ""
    group: str = ""
    ssh_host: str = DEFAULT_SSH_HOST
    dtn_host: str = DEFAULT_DTN_HOST
    ssh_key_path: str = "~/.ssh/id_ed25519"
    control_path: str = "~/.ssh/cm-%r@%h:%p"
    remote_repo_dir: str = ""
    remote_data_dir: str = ""
    remote_scratch_dir: str = ""
    remote_venv_path: str = ""
    python_module: str = "anaconda"
    cuda_module: str = "cuda/12.1"
    default_partition: str = DEFAULT_PARTITION
    default_qos: str = DEFAULT_QOS
    default_time: str = DEFAULT_TIME
    default_gpus: int = DEFAULT_GPUS
    default_cpus: int = DEFAULT_CPUS
    default_mem_gb: int = DEFAULT_MEM_GB
    notification_email: str = ""
    notify_via_app: bool = True
    command_timeout_s: int = 60

    # ---- derived ---------------------------------------------------------

    @property
    def ssh_target(self) -> str:
        return f"{self.jhed}@{self.ssh_host}" if self.jhed else self.ssh_host

    @property
    def dtn_target(self) -> str:
        return f"{self.jhed}@{self.dtn_host}" if self.jhed else self.dtn_host

    def autodetect_paths(self) -> "RockfishConfig":
        """Fill empty remote paths with sensible defaults based on jhed/group."""
        if not self.jhed:
            return self
        remote_repo = self.remote_repo_dir or f"/home/{self.jhed}/Tesis"
        remote_data = self.remote_data_dir
        if not remote_data and self.group:
            remote_data = f"/data/{self.group}/{self.jhed}/Tesis"
        remote_scratch = self.remote_scratch_dir
        if not remote_scratch and self.group:
            remote_scratch = f"/scratch16/{self.group}/{self.jhed}/Tesis"
        venv = self.remote_venv_path or f"{remote_repo}/venv_gpu"
        return replace(
            self,
            remote_repo_dir=remote_repo,
            remote_data_dir=remote_data,
            remote_scratch_dir=remote_scratch,
            remote_venv_path=venv,
        )


@dataclass(frozen=True)
class JobSpec:
    """User-facing description of a single sbatch submission."""

    job_name: str = "gnn_exp"
    partition: str = DEFAULT_PARTITION
    qos: str = DEFAULT_QOS
    gpus: int = DEFAULT_GPUS
    cpus_per_task: int = DEFAULT_CPUS
    mem_gb: int = DEFAULT_MEM_GB
    time_limit: str = DEFAULT_TIME
    graph_path: str = ""
    hparams_path: str = ""
    hparams_index: int = 0
    purpose: str = "rockfish_run"
    max_epochs: int = 50
    early_stop: bool = True
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-6
    accumulation_steps: int = 1
    train_sampler_mode: str = "neighbor"
    force_use_graphsmote: Optional[bool] = None
    seed: int = 19091985
    extra_cli_args: str = ""
    mail_type: str = "END,FAIL"

    def to_python_cli(self) -> list[str]:
        """Build the ``python -m src.gnn_main`` argv (without the prefix)."""
        argv: list[str] = ["-m", "src.gnn_main"]
        if self.graph_path:
            argv += ["--graph", self.graph_path]
        if self.hparams_path:
            argv += ["--hparams", self.hparams_path, "--hparams-index", str(self.hparams_index)]
        argv += [
            "--purpose", self.purpose,
            "--max-epochs", str(self.max_epochs),
            "--early-stop-patience", str(self.early_stop_patience),
            "--early-stop-min-delta", str(self.early_stop_min_delta),
            "--accumulation-steps", str(self.accumulation_steps),
            "--train-sampler-mode", self.train_sampler_mode,
            "--seed", str(self.seed),
        ]
        if not self.early_stop:
            argv.append("--no-early-stop")
        if self.force_use_graphsmote is True:
            argv.append("--force-use-graphsmote")
        elif self.force_use_graphsmote is False:
            argv.append("--no-graphsmote")
        if self.extra_cli_args.strip():
            argv += shlex.split(self.extra_cli_args)
        return argv


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def combined_output(self) -> str:
        return (self.stdout or "") + (("\n" + self.stderr) if self.stderr else "")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    blocking: bool = False


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    name: str
    state: str
    partition: str
    time_used: str
    nodes: str
    gres: str
    reason: str = ""

    @classmethod
    def from_squeue_row(cls, row: dict[str, str]) -> "JobRecord":
        return cls(
            job_id=row.get("JOBID", "").strip(),
            name=row.get("NAME", "").strip(),
            state=row.get("STATE", "").strip(),
            partition=row.get("PARTITION", "").strip(),
            time_used=row.get("TIME", "").strip(),
            nodes=row.get("NODELIST(REASON)", "").strip() or row.get("NODELIST", "").strip(),
            gres=row.get("TRES_PER_NODE", "").strip() or row.get("GRES", "").strip(),
            reason=row.get("REASON", "").strip(),
        )


@dataclass(frozen=True)
class RockfishHealthSnapshot:
    timestamp: float
    ssh: CheckResult
    modules: list[CheckResult]
    gpu_access: CheckResult
    venv: CheckResult
    quotas: list[CheckResult]
    summary: list[CheckResult]


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def default_config() -> RockfishConfig:
    return RockfishConfig()


def config_to_json_dict(config: RockfishConfig) -> dict[str, Any]:
    return asdict(config)


def load_config(path: Path = CONFIG_FILE) -> RockfishConfig:
    if not path.exists():
        return default_config()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default_config()
    cfg = default_config()
    known = {k for k in asdict(cfg).keys()}
    safe = {k: v for k, v in data.items() if k in known}
    try:
        return replace(cfg, **safe)
    except TypeError:
        return cfg


def save_config(config: RockfishConfig, path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_json_dict(config), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------


def _run(command: list[str], *, timeout: int, input_text: Optional[str] = None) -> CommandResult:
    rendered = " ".join(shlex.quote(part) for part in command)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(command=rendered, returncode=127, stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=rendered,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\n[timeout after {timeout}s]",
        )
    return CommandResult(
        command=rendered,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _ssh_base(config: RockfishConfig, host: Optional[str] = None) -> list[str]:
    target = host or config.ssh_target
    args = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", f"ControlMaster=auto",
        "-o", f"ControlPath={config.control_path}",
        "-o", "ControlPersist=4h",
    ]
    return args + [target]


def ssh_run(config: RockfishConfig, remote_cmd: str, *, timeout: Optional[int] = None) -> CommandResult:
    """Execute ``remote_cmd`` on the login node via SSH."""
    cmd = _ssh_base(config) + ["--", "bash", "-lc", remote_cmd]
    return _run(cmd, timeout=timeout or config.command_timeout_s)


def ssh_check_master(config: RockfishConfig) -> CheckResult:
    """Verify the ControlMaster socket is live (cheap, no password prompt)."""
    if not config.jhed:
        return CheckResult("SSH ControlMaster", False, "JHED no configurado", blocking=True)
    cmd = ["ssh", "-O", "check",
           "-o", f"ControlPath={config.control_path}",
           config.ssh_target]
    res = _run(cmd, timeout=5)
    if res.ok:
        return CheckResult("SSH ControlMaster", True, "Socket activo")
    return CheckResult(
        "SSH ControlMaster",
        False,
        "Sin socket activo. Ejecuta en una terminal: "
        f"ssh -fNM -o ControlPath={config.control_path} {config.ssh_target}",
        blocking=True,
    )


def ssh_probe(config: RockfishConfig) -> CheckResult:
    """Run ``hostname`` over SSH as a connectivity smoke test."""
    res = ssh_run(config, "hostname && whoami && id -gn", timeout=15)
    if res.ok:
        return CheckResult("SSH login", True, res.stdout.strip())
    detail = (res.stderr or res.stdout or "").strip()
    return CheckResult("SSH login", False, detail or "comando fallo", blocking=True)


# ---------------------------------------------------------------------------
# rsync helpers (use the dedicated DTN host)
# ---------------------------------------------------------------------------


def _rsync_ssh_opt(config: RockfishConfig) -> str:
    return (
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        f"-o ControlMaster=auto -o ControlPath={config.control_path} "
        "-o ControlPersist=4h"
    )


def rsync_push(
    config: RockfishConfig,
    local_dir: Path,
    remote_dir: str,
    *,
    excludes: Sequence[str] = RSYNC_DEFAULT_EXCLUDES,
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    timeout: int = 900,
) -> CommandResult:
    target_host = config.dtn_target
    args = ["rsync", "-avz", "--partial", "--human-readable", "-e", _rsync_ssh_opt(config)]
    for pattern in excludes:
        args += ["--exclude", pattern]
    if dry_run:
        args.append("--dry-run")
    args += list(extra_args)
    args.append(str(local_dir).rstrip("/") + "/")
    args.append(f"{target_host}:{remote_dir.rstrip('/')}/")
    return _run(args, timeout=timeout)


def rsync_pull(
    config: RockfishConfig,
    remote_dir: str,
    local_dir: Path,
    *,
    excludes: Sequence[str] = (),
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    timeout: int = 900,
) -> CommandResult:
    target_host = config.dtn_target
    args = ["rsync", "-avz", "--partial", "--human-readable", "-e", _rsync_ssh_opt(config)]
    for pattern in excludes:
        args += ["--exclude", pattern]
    if dry_run:
        args.append("--dry-run")
    args += list(extra_args)
    args.append(f"{target_host}:{remote_dir.rstrip('/')}/")
    local_dir.mkdir(parents=True, exist_ok=True)
    args.append(str(local_dir).rstrip("/") + "/")
    return _run(args, timeout=timeout)


# ---------------------------------------------------------------------------
# Module / quota / GPU access checks
# ---------------------------------------------------------------------------


def check_module_available(config: RockfishConfig, module_name: str) -> CheckResult:
    res = ssh_run(config, f"module spider {shlex.quote(module_name)} 2>&1 | head -40", timeout=20)
    if not res.ok:
        return CheckResult(f"module: {module_name}", False, res.stderr.strip() or "fallo")
    text = res.stdout.strip()
    if not text or "couldn't find" in text.lower() or "not found" in text.lower():
        return CheckResult(f"module: {module_name}", False, "no encontrado en Lmod", blocking=False)
    return CheckResult(f"module: {module_name}", True, text.splitlines()[0][:120])


def check_quotas(config: RockfishConfig) -> list[CheckResult]:
    cmd = (
        "echo '--HOME--'; du -sh $HOME 2>/dev/null | head -1; "
        "echo '--DATA--'; df -h /data 2>/dev/null | tail -1; "
        "echo '--SCRATCH16--'; df -h /scratch16 2>/dev/null | tail -1"
    )
    res = ssh_run(config, cmd, timeout=20)
    if not res.ok:
        return [CheckResult("Filesystems", False, res.stderr.strip() or "fallo")]
    out = res.stdout or ""
    blocks = {}
    current = None
    for line in out.splitlines():
        if line.startswith("--") and line.endswith("--"):
            current = line.strip("- ")
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line.strip())
    results: list[CheckResult] = []
    for label in ("HOME", "DATA", "SCRATCH16"):
        body = " | ".join(filter(None, blocks.get(label, [])))
        results.append(CheckResult(f"FS: {label.lower()}", bool(body), body or "sin datos"))
    return results


def check_gpu_account(config: RockfishConfig) -> CheckResult:
    if not config.pi_account:
        return CheckResult("Account GPU", False, "pi_account no configurado", blocking=True)
    cmd = f"sacctmgr -nP show assoc user={shlex.quote(config.jhed)} format=account,qos | grep -i {shlex.quote(config.pi_account)}"
    res = ssh_run(config, cmd, timeout=20)
    if not res.ok or not res.stdout.strip():
        return CheckResult(
            "Account GPU",
            False,
            f"No se ve la asociacion '{config.pi_account}'. Pide al PI que confirme con help@arch.jhu.edu.",
            blocking=True,
        )
    return CheckResult("Account GPU", True, res.stdout.strip().splitlines()[0])


def check_venv(config: RockfishConfig) -> CheckResult:
    if not config.remote_venv_path:
        return CheckResult("venv CUDA", False, "remote_venv_path vacio", blocking=False)
    cmd = (
        f"test -x {shlex.quote(config.remote_venv_path)}/bin/python && "
        f"{shlex.quote(config.remote_venv_path)}/bin/python -c "
        "'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>&1"
    )
    res = ssh_run(config, cmd, timeout=30)
    if not res.ok:
        return CheckResult(
            "venv CUDA",
            False,
            "no inicializado. Ejecuta scripts/slurm/setup_rockfish_env.sh dentro de un job interactivo.",
            blocking=False,
        )
    return CheckResult("venv CUDA", True, res.stdout.strip())


def collect_health_snapshot(config: RockfishConfig) -> RockfishHealthSnapshot:
    master = ssh_check_master(config)
    if not master.ok:
        return RockfishHealthSnapshot(
            timestamp=time.time(),
            ssh=master,
            modules=[],
            gpu_access=CheckResult("Account GPU", False, "pendiente (sin SSH)"),
            venv=CheckResult("venv CUDA", False, "pendiente (sin SSH)"),
            quotas=[],
            summary=[master],
        )
    ssh = ssh_probe(config)
    modules: list[CheckResult] = []
    if ssh.ok:
        modules.append(check_module_available(config, config.python_module))
        modules.append(check_module_available(config, config.cuda_module))
    gpu_access = check_gpu_account(config) if ssh.ok else CheckResult("Account GPU", False, "pendiente (sin SSH)")
    venv = check_venv(config) if ssh.ok else CheckResult("venv CUDA", False, "pendiente (sin SSH)")
    quotas = check_quotas(config) if ssh.ok else []
    summary = [master, ssh, gpu_access, venv]
    return RockfishHealthSnapshot(
        timestamp=time.time(),
        ssh=ssh,
        modules=modules,
        gpu_access=gpu_access,
        venv=venv,
        quotas=quotas,
        summary=summary,
    )


def blocking_checks(checks: Iterable[CheckResult]) -> list[CheckResult]:
    return [c for c in checks if c.blocking and not c.ok]


def checks_to_text(checks: Iterable[CheckResult]) -> str:
    return "\n".join(f"  - {c.name}: {c.detail}" for c in checks)


# ---------------------------------------------------------------------------
# sbatch rendering / submission
# ---------------------------------------------------------------------------


def render_sbatch(spec: JobSpec, config: RockfishConfig) -> str:
    """Materialise the sbatch script from the template."""
    template_path = SBATCH_TEMPLATE
    if not template_path.exists():
        raise FileNotFoundError(f"Plantilla sbatch no encontrada: {template_path}")
    template = template_path.read_text()
    cli_args = " ".join(shlex.quote(part) for part in spec.to_python_cli())
    mail_user = config.notification_email or f"{config.jhed}@jh.edu"
    substitutions = {
        "{{JOB_NAME}}": spec.job_name,
        "{{PARTITION}}": spec.partition,
        "{{ACCOUNT}}": config.pi_account,
        "{{QOS}}": spec.qos,
        "{{GPUS}}": str(spec.gpus),
        "{{CPUS}}": str(spec.cpus_per_task),
        "{{MEM_GB}}": str(spec.mem_gb),
        "{{TIME}}": spec.time_limit,
        "{{MAIL_TYPE}}": spec.mail_type,
        "{{MAIL_USER}}": mail_user,
        "{{LOG_DIR}}": f"{config.remote_scratch_dir.rstrip('/')}/logs",
        "{{REMOTE_REPO}}": config.remote_repo_dir,
        "{{REMOTE_DATA}}": config.remote_data_dir,
        "{{REMOTE_RESULTS}}": f"{config.remote_data_dir.rstrip('/')}/Resultados",
        "{{REMOTE_VENV}}": config.remote_venv_path,
        "{{PYTHON_MODULE}}": config.python_module,
        "{{CUDA_MODULE}}": config.cuda_module,
        "{{CLI_ARGS}}": cli_args,
    }
    rendered = template
    for key, value in substitutions.items():
        rendered = rendered.replace(key, value)
    return rendered


def upload_sbatch(config: RockfishConfig, content: str, remote_relpath: str = "scripts/slurm/_last_submit.sbatch") -> CommandResult:
    """Write the rendered script under the remote repo via stdin."""
    remote_full = f"{config.remote_repo_dir.rstrip('/')}/{remote_relpath}"
    remote_dir = remote_full.rsplit("/", 1)[0]
    cmd = f"mkdir -p {shlex.quote(remote_dir)} && cat > {shlex.quote(remote_full)}"
    ssh_cmd = _ssh_base(config) + ["--", "bash", "-lc", cmd]
    res = _run(ssh_cmd, timeout=config.command_timeout_s, input_text=content)
    if not res.ok:
        return res
    return CommandResult(command=res.command + f"  # wrote {remote_full}", returncode=0,
                         stdout=remote_full, stderr=res.stderr)


def submit_sbatch(config: RockfishConfig, remote_sbatch_path: str) -> CommandResult:
    cmd = f"sbatch {shlex.quote(remote_sbatch_path)}"
    return ssh_run(config, cmd, timeout=30)


def parse_submit_output(output: str) -> Optional[str]:
    for line in output.splitlines():
        line = line.strip()
        if line.lower().startswith("submitted batch job"):
            return line.split()[-1]
    return None


# ---------------------------------------------------------------------------
# squeue / scancel / jobstats
# ---------------------------------------------------------------------------


_SQUEUE_FORMAT = "JobID|Name|State|Partition|TimeUsed|NodeList|tres-per-node|Reason"


def list_jobs(config: RockfishConfig) -> tuple[CommandResult, pd.DataFrame]:
    cmd = f"squeue -u {shlex.quote(config.jhed)} -h -O '{_SQUEUE_FORMAT}'"
    res = ssh_run(config, cmd, timeout=20)
    if not res.ok or not res.stdout.strip():
        return res, pd.DataFrame(columns=["JobID", "Name", "State", "Partition", "TimeUsed", "NodeList", "TRES", "Reason"])
    rows = []
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue
        rows.append({
            "JobID": parts[0],
            "Name": parts[1],
            "State": parts[2],
            "Partition": parts[3],
            "TimeUsed": parts[4],
            "NodeList": parts[5],
            "TRES": parts[6],
            "Reason": parts[7],
        })
    return res, pd.DataFrame(rows)


def cancel_job(config: RockfishConfig, job_id: str) -> CommandResult:
    return ssh_run(config, f"scancel {shlex.quote(job_id)}", timeout=15)


def job_stats(config: RockfishConfig, job_id: str) -> CommandResult:
    return ssh_run(config, f"jobstats {shlex.quote(job_id)}", timeout=30)


def tail_log(config: RockfishConfig, job_id: str, *, kind: str = "out", lines: int = 500) -> CommandResult:
    """Tail stdout/stderr from the standard log location used by the template."""
    suffix = "err" if kind == "err" else "out"
    log_dir = f"{config.remote_scratch_dir.rstrip('/')}/logs"
    pattern = f"{log_dir}/*_{job_id}.{suffix}"
    cmd = f"ls -1t {pattern} 2>/dev/null | head -1 | xargs -r tail -n {int(lines)}"
    return ssh_run(config, cmd, timeout=20)


# ---------------------------------------------------------------------------
# Setup helper (run once)
# ---------------------------------------------------------------------------


def push_setup_script(config: RockfishConfig) -> CommandResult:
    """Upload setup_rockfish_env.sh next to the remote repo."""
    if not SETUP_SCRIPT.exists():
        return CommandResult(command="cp setup_rockfish_env.sh", returncode=1,
                             stderr=f"missing local file: {SETUP_SCRIPT}")
    content = SETUP_SCRIPT.read_text()
    remote_full = f"{config.remote_repo_dir.rstrip('/')}/scripts/slurm/setup_rockfish_env.sh"
    remote_dir = remote_full.rsplit("/", 1)[0]
    cmd = (
        f"mkdir -p {shlex.quote(remote_dir)} && "
        f"cat > {shlex.quote(remote_full)} && chmod +x {shlex.quote(remote_full)}"
    )
    ssh_cmd = _ssh_base(config) + ["--", "bash", "-lc", cmd]
    res = _run(ssh_cmd, timeout=config.command_timeout_s, input_text=content)
    if not res.ok:
        return res
    return CommandResult(command=res.command + f"  # wrote {remote_full}", returncode=0,
                         stdout=remote_full)
