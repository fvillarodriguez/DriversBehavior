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
from typing import List, Tuple, Optional, Generator

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

def check_git_status() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, text=True, capture_output=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

def is_git_repo() -> bool:
    return Path(".git").is_dir()

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

def sync_with_github_stream() -> Generator[str, None, bool]:
    yield "Iniciando Sincronización con GitHub..."
    
    # 1. Pull
    success = yield from run_command_stream(["git", "pull", "origin", "main"], "Trayendo cambios remotos")
    if not success:
        yield "Error al hacer pull. Conflictos posibles."
        return False

    if not check_git_status():
        yield "✨ No hay cambios locales *nuevos* para crear commit."
        # No retornamos aquí, seguimos para hacer push de commits previos si los hay.
    else:
        # 2. Add
        success = yield from run_command_stream(["git", "add", "."], "Agregando archivos")
        if not success: return False
    
        # 3. Commit
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success = yield from run_command_stream(["git", "commit", "-m", f"Auto-sync: {ts}"], "Haciendo commit")
        if not success: return False

    # 4. Push
    success = yield from run_command_stream(["git", "push", "-u", "origin", "main"], "Enviando cambios (push)")
    if not success:
        yield "Error al hacer push."
        return False

    yield "🎉 Sincronización completada exitosamente."
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

if __name__ == "__main__":
    s, l = sync_with_github()
    for line in l: print(line)
    if not s: sys.exit(1)
