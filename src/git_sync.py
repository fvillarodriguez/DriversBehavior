#!/usr/bin/env python3
"""
git_sync.py
===========
Script para sincronizar automáticamente el código local con el repositorio remoto de GitHub.
Soporta streaming de logs para integración en UI.
"""

import subprocess
import datetime
import sys
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Generator

GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024
DEFAULT_GITHUB_REPO_URL = "https://github.com/fvillarodriguez/DriversBehavior.git"
DEFAULT_GITHUB_BRANCH = "main"
LOCAL_UPDATE_CLEAN_EXCLUDES = (
    "Datos/",
    "Resultados/",
    "simulación/",
    "simulación/",
    "docs/",
    "NLP/",
    "DRIFT/",
    ".venv/",
    "venv/",
)

# Códigos de escape ANSI para colores
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run_command_stream(command: List[str], description: str) -> Generator[str, None, bool]:
    """
    Ejecuta un comando de shell y yielda la salida paso a paso.
    Retorna True si exit code es 0, False si no.
    Al ser un generador, el retorno se captura con `yield from` o iterando hasta StopIteration.
    Para simplificar, yieldaremos mensajes y al final un booleano especial o controlaremos flujo fuera.
    
    Mejor enfoque para UI: Yield strings. Si falla, yeild string de error.
    El llamador debe deducir éxito/error o podemos yieldar una tupla, pero streamtexto es más simple.
    """
    log_msg = f"==> {description}..."
    print(f"{Colors.OKCYAN}{log_msg}{Colors.ENDC}")
    yield log_msg
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd()
        )
        
        for line in process.stdout:
            line = line.rstrip()
            print(f"{Colors.OKBLUE}{line}{Colors.ENDC}")
            yield line
            
        process.wait()
        
        if process.returncode == 0:
            success_msg = "✔ Éxito."
            print(f"{Colors.OKGREEN}{success_msg}{Colors.ENDC}")
            yield success_msg
            return True
        else:
            fail_msg = f"✘ Error ejecutando: {' '.join(command)} (Exit code: {process.returncode})"
            print(f"{Colors.FAIL}{fail_msg}{Colors.ENDC}")
            yield fail_msg
            return False
            
    except Exception as e:
        fail_msg = f"✘ Excepción ejecutando: {e}"
        print(f"{Colors.FAIL}{fail_msg}{Colors.ENDC}")
        yield fail_msg
        return False

# Mantenemos la versión sincrónica para compatibilidad si alguien la usa
def run_command(command: List[str], description: str) -> Tuple[bool, str]:
    logs = []
    success = False
    gen = run_command_stream(command, description)
    try:
        while True:
            msg = next(gen)
            logs.append(msg)
    except StopIteration as e:
        success = e.value
        
    return success, "\n".join(logs)


def _run_quiet_git(args: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )


def _format_git_error(action: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = "\n".join(
        part.strip()
        for part in (result.stderr, result.stdout)
        if part and part.strip()
    )
    if len(detail) > 600:
        detail = f"{detail[:600]}..."
    if detail:
        return f"{action}: {detail}"
    return f"{action}: Git terminó con código {result.returncode}."


def _is_git_work_tree(repo_dir: Path) -> bool:
    result = _run_quiet_git(["rev-parse", "--show-toplevel"], repo_dir)
    if result.returncode != 0:
        return False

    try:
        return Path(result.stdout.strip()).resolve() == repo_dir.resolve()
    except OSError:
        return False


def _clean_untracked_nonignored(repo_dir: Path) -> Tuple[bool, str]:
    command = ["clean", "-ffd"]
    for pattern in LOCAL_UPDATE_CLEAN_EXCLUDES:
        command.extend(["-e", pattern])
    command.extend(["--", "."])

    result = _run_quiet_git(command, repo_dir)
    if result.returncode != 0:
        return False, _format_git_error("No se pudo limpiar archivos locales no versionados", result)
    return True, ""


def _set_origin_url(repo_dir: Path, remote_url: str) -> Tuple[bool, str]:
    current = _run_quiet_git(["remote", "get-url", "origin"], repo_dir)
    if current.returncode == 0:
        result = _run_quiet_git(["remote", "set-url", "origin", remote_url], repo_dir)
        action = "No se pudo actualizar el remoto origin"
    else:
        result = _run_quiet_git(["remote", "add", "origin", remote_url], repo_dir)
        action = "No se pudo configurar el remoto origin"

    if result.returncode != 0:
        return False, _format_git_error(action, result)
    return True, ""


def update_local_repo_from_github(
    repo_dir: Optional[Path] = None,
    remote_url: str = DEFAULT_GITHUB_REPO_URL,
    branch: str = DEFAULT_GITHUB_BRANCH,
) -> Tuple[bool, str]:
    """
    Sobrescribe la copia local con la rama pública de GitHub usando HTTPS.

    La descarga no depende de origin ni de claves SSH. Los archivos no
    versionados que no estén ignorados se eliminan; las raíces de datos/salidas
    se preservan explícitamente además de las reglas de .gitignore vigentes.
    """
    target_dir = Path(repo_dir).resolve() if repo_dir is not None else Path.cwd().resolve()
    selected_branch = branch.strip()

    if not target_dir.exists() or not target_dir.is_dir():
        return False, f"La carpeta del proyecto no existe: {target_dir}"
    if not remote_url.strip():
        return False, "La URL remota está vacía."
    if not selected_branch:
        return False, "La rama remota está vacía."

    try:
        if not _is_git_work_tree(target_dir):
            result = _run_quiet_git(["init"], target_dir)
            if result.returncode != 0:
                return False, _format_git_error("No se pudo inicializar Git en la carpeta local", result)

        refspec = f"+refs/heads/{selected_branch}:refs/remotes/origin/{selected_branch}"
        fetch = _run_quiet_git(["fetch", "--prune", remote_url, refspec], target_dir)
        if fetch.returncode != 0:
            return False, _format_git_error("No se pudo descargar la base de código desde GitHub", fetch)

        success, message = _set_origin_url(target_dir, remote_url)
        if not success:
            return False, message

        success, message = _clean_untracked_nonignored(target_dir)
        if not success:
            return False, message

        remote_ref = f"refs/remotes/origin/{selected_branch}"
        reset = _run_quiet_git(["reset", "--hard", remote_ref], target_dir)
        if reset.returncode != 0:
            return False, _format_git_error("No se pudo sobrescribir la copia local", reset)

        checkout = _run_quiet_git(["checkout", "-B", selected_branch, remote_ref], target_dir)
        if checkout.returncode != 0:
            return False, _format_git_error("No se pudo activar la rama local actualizada", checkout)

        success, message = _clean_untracked_nonignored(target_dir)
        if not success:
            return False, message

        return True, "Repositorio local actualizado desde GitHub."
    except Exception as exc:
        return False, f"Error inesperado al actualizar el repositorio local: {exc}"

def check_git_status() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, text=True, capture_output=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def has_staged_changes() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            check=True,
            text=True,
            capture_output=True,
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

def is_git_repo() -> bool:
    return _is_git_work_tree(Path.cwd())


def get_tracked_ignored_paths() -> List[str]:
    result = subprocess.run(
        ["git", "ls-files", "-ci", "--exclude-standard", "-z"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        return []

    return [path for path in result.stdout.split("\0") if path]


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]

def initialize_repo_stream(remote_url: str) -> Generator[str, None, bool]:
    yield f"Iniciando configuración de repositorio... {remote_url}"

    # 1. Git Init
    if not is_git_repo():
        success = yield from run_command_stream(["git", "init"], "Inicializando git")
        if not success: return False
    else:
        yield "Ya es un repositorio git."

    # 2. Add Remote
    success_check, _ = run_command(["git", "remote", "get-url", "origin"], "Verificando remote")
    if not success_check:
        success = yield from run_command_stream(["git", "remote", "add", "origin", remote_url], f"Agregando remote {remote_url}")
        if not success: return False
    else:
        success = yield from run_command_stream(["git", "remote", "set-url", "origin", remote_url], f"Actualizando remote a {remote_url}")

    # 3. Add & Commit inicial
    if check_git_status() or True:
        yield from run_command_stream(["git", "add", "."], "Agregando archivos locales")
        if check_git_status():
             yield from run_command_stream(["git", "commit", "-m", "Initial commit from SUMO App"], "Haciendo commit inicial")
    
    # 4. Pull
    success = yield from run_command_stream(["git", "pull", "origin", "main", "--allow-unrelated-histories", "--rebase"], "Trayendo historia remota")
    # No retornamos False aquí para intentar push igual si es repo vacío
    
    # 5. Branch
    yield from run_command_stream(["git", "branch", "-M", "main"], "Renombrando rama a main")

    # 6. Push
    success = yield from run_command_stream(["git", "push", "-u", "origin", "main"], "Enviando a remoto")
    if not success:
        yield "⚠️ El push falló. Verifica si tienes permisos de escritura."
        return False

    yield "🎉 Repositorio inicializado y sincronizado."
    return True

# Wrapper para compatibilidad
def initialize_repo(remote_url: str) -> Tuple[bool, List[str]]:
    logs = []
    success = False
    gen = initialize_repo_stream(remote_url)
    try:
        while True:
            msg = next(gen)
            logs.append(msg)
    except StopIteration as e:
        success = e.value
    return success, logs

def get_ssh_public_key() -> Optional[str]:
    ssh_dir = Path.home() / ".ssh"
    pub_keys = ["id_ed25519.pub", "id_rsa.pub"]
    for key_name in pub_keys:
        key_path = ssh_dir / key_name
        if key_path.exists():
            try: return key_path.read_text().strip()
            except: continue
    return None

def generate_ssh_key(email: str = "", overwrite: bool = False) -> Tuple[bool, str]:
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    key_path = ssh_dir / "id_ed25519"
    
    if key_path.exists() and not overwrite:
        return False, f"La clave ya existe en {key_path}. No se sobrescribirá."
    
    comment = email if email else "sumo-app-generated"
    cmd = ["ssh-keygen", "-t", "ed25519", "-C", comment, "-f", str(key_path), "-N", ""]
    
    if key_path.exists() and overwrite:
        key_path.unlink()
        if key_path.with_suffix(".pub").exists(): key_path.with_suffix(".pub").unlink()
                
    success, _ = run_command(cmd, "Generando clave SSH")
    if success: return True, f"Clave generada en {key_path}"
    else: return False, "Error generando clave"

def get_git_user() -> Tuple[str, str]:
    """Retorna (name, email) configurados globalmente o localmente."""
    try:
        name = subprocess.check_output(["git", "config", "user.name"], text=True).strip()
    except:
        name = ""
    try:
        email = subprocess.check_output(["git", "config", "user.email"], text=True).strip()
    except:
        email = ""
    return name, email

def configure_git_user(name: str, email: str) -> Tuple[bool, str]:
    """Configura user.name y user.email localmente para este repositorio."""
    try:
        run_command(["git", "config", "user.name", name], "Configurando user.name")
        run_command(["git", "config", "user.email", email], "Configurando user.email")
        return True, "Usuario Git configurado exitosamente."
    except Exception as e:
        return False, f"Error configurando git: {e}"


def get_current_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            text=True,
            capture_output=True,
        )
        branch = result.stdout.strip()
        return branch if branch else "main"
    except subprocess.CalledProcessError:
        return "main"


def remote_branch_exists(branch: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/remotes/origin/{branch}"],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def get_branch_divergence(branch: str) -> Tuple[int, int]:
    if not remote_branch_exists(branch):
        return 0, 0

    result = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", f"{branch}...origin/{branch}"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return 0, 0

    counts = result.stdout.strip().split()
    if len(counts) != 2:
        return 0, 0

    ahead, behind = counts
    return int(ahead), int(behind)


def backup_remote_branch_stream(branch: str) -> Generator[str, None, bool]:
    backup_suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_branch = f"backup/{branch.replace('/', '-')}-{backup_suffix}"
    yield f"Respaldando origin/{branch} en origin/{backup_branch}..."

    success = yield from run_command_stream(
        [
            "git",
            "push",
            "origin",
            f"refs/remotes/origin/{branch}:refs/heads/{backup_branch}",
        ],
        "Creando respaldo remoto",
    )
    if not success:
        yield "No se pudo crear el respaldo remoto."
        return False

    yield f"Respaldo remoto creado en origin/{backup_branch}."
    return True


def get_tracked_files_over_limit(limit_bytes: int = GITHUB_FILE_LIMIT_BYTES) -> List[Tuple[str, int]]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-l", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []

    oversized_files: List[Tuple[str, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) != 5:
            continue

        size_text = parts[3]
        path = parts[4]
        if not size_text.isdigit():
            continue

        size = int(size_text)
        if size > limit_bytes:
            oversized_files.append((path, size))

    return oversized_files


def sync_with_github_stream() -> Generator[str, None, bool]:
    yield "Iniciando Sincronización con GitHub..."
    current_branch = get_current_branch()
    has_remote_branch = False

    success = yield from run_command_stream(["git", "fetch", "origin"], "Trayendo cambios remotos")
    if not success:
        yield "Error al hacer fetch del remoto."
        return False

    has_remote_branch = remote_branch_exists(current_branch)

    if check_git_status():
        success = yield from run_command_stream(["git", "add", "."], "Agregando archivos")
        if not success:
            return False

        if check_git_status():
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            success = yield from run_command_stream(
                ["git", "commit", "-m", f"Auto-sync: {ts}"],
                "Haciendo commit",
            )
            if not success:
                return False
        else:
            yield "ℹ️ No hubo cambios indexables después de aplicar .gitignore."
    else:
        yield "✨ No hay cambios locales nuevos para crear commit."

    ahead, behind = get_branch_divergence(current_branch)
    yield f"Estado de la rama '{current_branch}': ahead {ahead}, behind {behind}."

    oversized_files = get_tracked_files_over_limit()
    if oversized_files:
        yield "❌ GitHub rechazará el push porque hay archivos rastreados mayores a 100 MB."
        for path, size in oversized_files:
            size_mb = size / (1024 * 1024)
            yield f" - {path} ({size_mb:.2f} MB)"
        yield "Quite esos archivos del historial de Git o muévalos a Git LFS antes de reintentar."
        return False

    if not has_remote_branch:
        success = yield from run_command_stream(
            ["git", "push", "-u", "origin", f"{current_branch}:{current_branch}"],
            "Publicando rama local por primera vez",
        )
        if not success:
            yield "Error al publicar la rama en el remoto."
            return False

        yield "🎉 Sincronización completada exitosamente."
        return True

    if behind > 0 and ahead == 0:
        success = yield from run_command_stream(
            ["git", "merge", "--ff-only", f"origin/{current_branch}"],
            "Aplicando fast-forward desde remoto",
        )
        if not success:
            yield "No fue posible adelantar la rama local con fast-forward."
            return False

        yield "🎉 Sincronización completada exitosamente."
        return True

    if behind > 0:
        success = yield from backup_remote_branch_stream(current_branch)
        if not success:
            yield "Se abortó la sincronización para no sobrescribir el remoto sin respaldo."
            return False

        success = yield from run_command_stream(
            ["git", "push", "--force-with-lease", "-u", "origin", f"{current_branch}:{current_branch}"],
            "Enviando cambios locales y reemplazando el remoto",
        )
        if not success:
            yield "Error al hacer push forzado."
            return False

        yield "🎉 Sincronización completada exitosamente. La rama local prevaleció sobre el remoto."
        return True

    if ahead > 0:
        success = yield from run_command_stream(
            ["git", "push", "-u", "origin", f"{current_branch}:{current_branch}"],
            "Enviando cambios (push)",
        )
        if not success:
            yield "Error al hacer push."
            return False

        yield "🎉 Sincronización completada exitosamente."
        return True

    yield "✅ Repositorio ya sincronizado. No hubo nada que enviar."
    return True


def remove_ignored_tracked_files_from_remote_stream(push: bool = True) -> Generator[str, None, bool]:
    yield "Buscando archivos rastreados que hoy están cubiertos por .gitignore..."

    tracked_ignored_paths = get_tracked_ignored_paths()
    if not tracked_ignored_paths:
        yield "✅ No hay archivos trackeados que coincidan con .gitignore."
        return True

    yield f"Se encontraron {len(tracked_ignored_paths)} rutas trackeadas e ignoradas."

    if has_staged_changes():
        yield (
            "⚠️ Hay cambios ya indexados en Git. La purga se abortó para no mezclar "
            "este commit de emergencia con otros cambios staged."
        )
        return False

    current_branch = get_current_branch()
    has_remote_branch = False

    if push:
        success = yield from run_command_stream(["git", "fetch", "origin"], "Trayendo cambios remotos")
        if not success:
            yield "No fue posible actualizar referencias remotas antes de la purga."
            return False

        has_remote_branch = remote_branch_exists(current_branch)
        ahead, behind = get_branch_divergence(current_branch)
        yield f"Estado de la rama '{current_branch}' antes de purgar: ahead {ahead}, behind {behind}."

        if behind > 0:
            yield (
                "⚠️ La rama local está detrás del remoto. Sincroniza primero y vuelve a "
                "intentar para evitar sobrescribir cambios remotos."
            )
            return False

    for path_batch in chunked(tracked_ignored_paths, 100):
        success = yield from run_command_stream(
            ["git", "rm", "--cached", "--ignore-unmatch", "--", *path_batch],
            f"Quitando del índice {len(path_batch)} archivos ignorados",
        )
        if not success:
            return False

    if not has_staged_changes():
        yield "ℹ️ No se generaron cambios indexados después de limpiar el índice."
        return True

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    success = yield from run_command_stream(
        ["git", "commit", "-m", f"Emergency purge ignored files: {ts}"],
        "Registrando commit de purga",
    )
    if not success:
        return False

    if not push:
        yield "✅ La purga quedó committeada localmente. Ejecuta Sync Now para publicarla en GitHub."
        return True

    if not has_remote_branch:
        success = yield from run_command_stream(
            ["git", "push", "-u", "origin", f"{current_branch}:{current_branch}"],
            "Publicando rama local por primera vez",
        )
    else:
        success = yield from run_command_stream(
            ["git", "push", "-u", "origin", f"{current_branch}:{current_branch}"],
            "Enviando borrados al remoto",
        )

    if not success:
        yield "❌ No fue posible publicar los borrados en GitHub."
        return False

    yield (
        "🎉 Los archivos ignorados quedaron eliminados del repositorio remoto. "
        "Tus archivos locales siguen intactos."
    )
    return True

# Wrapper para compatibilidad
def sync_with_github() -> Tuple[bool, List[str]]:
    logs = []
    success = False
    gen = sync_with_github_stream()
    try:
        while True:
            msg = next(gen)
            logs.append(msg)
    except StopIteration as e:
        success = e.value
    return success, logs

def force_reset_stream() -> Generator[str, None, bool]:
    """
    Descarga todo de origin y hace un reset --hard para igualar la copia local al remoto.
    CUIDADO: Esto borra cambios locales no commiteados.
    """
    yield "🔵 Starting Force Reset (Fetch + Reset --hard)..."

    # 1. Fetch
    yield "🔵 Fetching from origin..."
    try:
        # yield from returns the return value of the subgenerator
        success = yield from run_command_stream(["git", "fetch", "origin"], "Fetch")
        if not success:
            yield "❌ Fetch failed."
            return False
    except Exception as e:
        yield f"❌ Error executing fetch: {e}"
        return False

    # 2. Determine Branch
    current_branch = "main"
    try:
        res = subprocess.run(
            ["git", "branch", "--show-current"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            current_branch = res.stdout.strip()
    except Exception:
        yield "⚠️ Could not auto-detect branch. Assuming 'main'."

    target = f"origin/{current_branch}"
    yield f"🔵 Hard resetting local branch '{current_branch}' to '{target}'..."
    
    # 3. Reset --hard
    try:
        success = yield from run_command_stream(["git", "reset", "--hard", target], "Reset Hard")
        if not success:
            yield "❌ Reset failed."
            return False
    except Exception as e:
        yield f"❌ Error executing reset: {e}"
        return False

    yield "✅ Repository successfully reset to match remote."
    return True

def force_reset() -> Tuple[bool, List[str]]:
    logs = []
    success = False
    gen = force_reset_stream()
    try:
        while True:
            msg = next(gen)
            logs.append(msg)
    except StopIteration as e:
        success = e.value
    return success, logs

if __name__ == "__main__":
    s, l = sync_with_github()
    for line in l: print(line)
    if not s: sys.exit(1)
