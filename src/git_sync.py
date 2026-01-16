#!/usr/bin/env python3
"""
git_sync.py
===========
Script para sincronizar automáticamente el código local con el repositorio remoto de GitHub.
Realiza las siguientes operaciones:
1. git pull (para traer cambios remotos)
2. git add . (para agregar todos los cambios locales)
3. git commit (con mensaje automático)
4. git push (para enviar cambios al remoto)
"""

import subprocess
import datetime
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional

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

def run_command(command: List[str], description: str) -> Tuple[bool, str]:
    """
    Ejecuta un comando de shell y maneja errores.
    Retorna (éxito: bool, mensaje_log: str).
    """
    log = []
    log_msg = f"==> {description}..."
    print(f"{Colors.OKCYAN}{log_msg}{Colors.ENDC}")
    log.append(log_msg)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            cwd=os.getcwd()
        )
        success_msg = "✔ Éxito."
        print(f"{Colors.OKGREEN}{success_msg}{Colors.ENDC}")
        log.append(success_msg)
        
        if result.stdout:
            out_msg = result.stdout.strip()
            print(f"{Colors.OKBLUE}{out_msg}{Colors.ENDC}")
            log.append(out_msg)
            
        return True, "\n".join(log)
        
    except subprocess.CalledProcessError as e:
        fail_msg = f"✘ Error ejecutando: {' '.join(command)}"
        print(f"{Colors.FAIL}{fail_msg}{Colors.ENDC}")
        log.append(fail_msg)
        
        print(f"{Colors.FAIL}Salida de error:{Colors.ENDC}")
        log.append("Salida de error:")
        
        err_out = e.stderr.strip() if e.stderr else "Sin salida de error."
        print(err_out)
        log.append(err_out)
        
        return False, "\n".join(log)

def check_git_status() -> bool:
    """Verifica si hay cambios para commitear."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            text=True,
            capture_output=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

def is_git_repo() -> bool:
    """Verifica si el directorio actual es un repositorio git."""
    return Path(".git").is_dir()

def initialize_repo(remote_url: str) -> Tuple[bool, List[str]]:
    """
    Inicializa un repositorio git, agrega el remoto y trae cambios.
    """
    logs = []
    
    start_msg = "Iniciando configuración de repositorio..."
    print(f"{Colors.HEADER}{Colors.BOLD}{start_msg}{Colors.ENDC}\n")
    logs.append(start_msg)

    # 1. Git Init
    if not is_git_repo():
        success, log = run_command(["git", "init"], "Inicializando git (git init)")
        logs.append(log)
        if not success:
            return False, logs
    else:
        logs.append("Ya es un repositorio git.")

    # 2. Add Remote
    # Verificar si ya existe remote origin
    if not run_command(["git", "remote", "get-url", "origin"], "Verificando remote")[0]:
        success, log = run_command(["git", "remote", "add", "origin", remote_url], f"Agregando remote {remote_url}")
        logs.append(log)
        if not success:
            return False, logs
    else:
        success, log = run_command(["git", "remote", "set-url", "origin", remote_url], f"Actualizando remote a {remote_url}")
        logs.append(log)

    # 3. Add & Commit inicial (Commit before pull to avoid untracked files error)
    if check_git_status() or True: # Force check for untracked
        success, log = run_command(["git", "add", "."], "Agregando archivos locales")
        logs.append(log)
        # Check if there is anything to commit
        if check_git_status():
             success, log = run_command(["git", "commit", "-m", "Initial commit from SUMO App"], "Haciendo commit inicial")
             logs.append(log)
    
    # 4. Pull initial (allow unrelated histories)
    # Use --rebase to apply local commits on top of remote if remote exists
    # If remote is empty, pull might fail harmlessly or say "no branch"
    success, log = run_command(["git", "pull", "origin", "main", "--allow-unrelated-histories", "--rebase"], "Trayendo historia remota (git pull --rebase)")
    logs.append(log)
    
    # 5. Git Branch -M main
    run_command(["git", "branch", "-M", "main"], "Renombrando rama a main")

    # 6. Push
    success, log = run_command(["git", "push", "-u", "origin", "main"], "Enviando a remoto (git push -u origin main)")
    logs.append(log)
    if not success:
        logs.append("⚠️ El push falló. Verifica si tienes permisos de escritura o si hay conflictos.")
        return False, logs

    success_msg = "🎉 Repositorio inicializado y sincronizado."
    logs.append(success_msg)
    return True, logs

def update_remote_url(token: str, username: str = None) -> Tuple[bool, str]:
    """
    Actualiza la URL del remoto 'origin' para incluir el token de acceso.
    Si no se da username, se asume que solo se inyecta el token en la URL HTTPS estándar.
    Formato esperado: https://<token>@github.com/<user>/<repo>.git
    """
    try:
        # Obtener la URL actual
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            text=True,
            capture_output=True
        )
        current_url = result.stdout.strip()
        
        # Parsear URL básica para reconstruirla
        # Asumiendo formato https://github.com/... o https://user:pass@github.com/...
        if "@" in current_url:
             # Ya tiene credenciales, intentar reemplazarlas o limpiar
             base_url = current_url.split("@")[-1]
        else:
            # Eliminar https:// o http:// inicial
            base_url = current_url.replace("https://", "").replace("http://", "")
            
        new_url = f"https://{token}@{base_url}"
        
        # Actualizar remote
        run_command(["git", "remote", "set-url", "origin", new_url], "Actualizando URL remota con token")
        return True, "URL remota actualizada correctamente."
        
    except subprocess.CalledProcessError as e:
        return False, f"Error actualizando remote: {e}"


def get_ssh_public_key() -> Optional[str]:
    """
    Intenta leer la clave pública SSH por defecto (id_ed25519 o id_rsa).
    Retorna el contenido de la clave o None si no existe.
    """
    ssh_dir = Path.home() / ".ssh"
    # Prioridad: Ed25519 > RSA
    pub_keys = ["id_ed25519.pub", "id_rsa.pub"]
    
    for key_name in pub_keys:
        key_path = ssh_dir / key_name
        if key_path.exists():
            try:
                return key_path.read_text().strip()
            except Exception:
                continue
    return None

def generate_ssh_key(email: str = "", overwrite: bool = False) -> Tuple[bool, str]:
    """
    Genera una clave SSH ed25519 nueva.
    Retorna (éxito, mensaje).
    """
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    
    key_path = ssh_dir / "id_ed25519"
    
    if key_path.exists() and not overwrite:
        return False, f"La clave ya existe en {key_path}. No se sobrescribirá para evitar pérdida de acceso."
    
    # Comentario para la clave
    comment = email if email else "sumo-app-generated"
    
    cmd = [
        "ssh-keygen",
        "-t", "ed25519",
        "-C", comment,
        "-f", str(key_path),
        "-N", ""  # Passphrase vacía para automatización
    ]
    
    try:
        # Si existe y overwrite es True, ssh-keygen preguntará, pero con -f forza o falla.
        # Mejor borrar antes si overwrite es True (aunque ssh-keygen puede manejarlo, python subprocess interaction con prompts es tedioso).
        if key_path.exists() and overwrite:
            key_path.unlink()
            if key_path.with_suffix(".pub").exists():
                key_path.with_suffix(".pub").unlink()
                
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        return True, f"Clave SSH generada exitosamente en {key_path}"
    except subprocess.CalledProcessError as e:
        return False, f"Error generando clave: {e.stderr}"

def sync_with_github() -> Tuple[bool, List[str]]:
    """
    Ejecuta el flujo de sincronización.
    Retorna (éxito_global: bool, lista_de_logs: List[str]).
    """
    logs = []
    
    start_msg = "Iniciando Sincronización con GitHub..."
    print(f"{Colors.HEADER}{Colors.BOLD}{start_msg}{Colors.ENDC}\n")
    logs.append(start_msg)

    # 1. Git Pull
    # Intentamos pull específico de origin main para evitar errores de tracking
    success, log = run_command(["git", "pull", "origin", "main"], "Trayendo cambios remotos (git pull origin main)")
    logs.append(log)
    if not success:
        err_msg = "Error al hacer pull. Puede haber conflictos. Por favor revisa manualmente."
        print(f"{Colors.FAIL}{err_msg}{Colors.ENDC}")
        logs.append(err_msg)
        return False, logs

    # Verificar si hay cambios locales
    if not check_git_status():
        no_changes_msg = "✨ No hay cambios locales para enviar. El repositorio está actualizado."
        print(f"\n{Colors.OKGREEN}{no_changes_msg}{Colors.ENDC}")
        logs.append(no_changes_msg)
        return True, logs

    # 2. Git Add
    success, log = run_command(["git", "add", "."], "Agregando archivos (git add .)")
    logs.append(log)
    if not success:
        return False, logs

    # 3. Git Commit
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Auto-sync: {timestamp}"
    success, log = run_command(["git", "commit", "-m", commit_message], f"Haciendo commit ('{commit_message}')")
    logs.append(log)
    if not success:
        return False, logs

    # 4. Git Push
    # Usamos -u para asegurar tracking por si acaso se perdió
    success, log = run_command(["git", "push", "-u", "origin", "main"], "Enviando cambios (git push -u origin main)")
    logs.append(log)
    if not success:
        err_msg = "Error al hacer push. Verifica tu conexión o credenciales."
        print(f"{Colors.FAIL}{err_msg}{Colors.ENDC}")
        logs.append(err_msg)
        return False, logs

    final_msg = "🎉 Sincronización completada exitosamente."
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}{final_msg}{Colors.ENDC}")
    logs.append(final_msg)
    
    return True, logs

if __name__ == "__main__":
    success, _ = sync_with_github()
    if not success:
        sys.exit(1)
