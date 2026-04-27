from __future__ import annotations

import base64
from pathlib import Path


def _fernet(secret: str):
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ModuleNotFoundError as exc:
        raise RuntimeError("cryptography is required for encrypted CA backups.") from exc
    salt = b"dask-cluster-app-ca-backup-v1"
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390_000).derive(
        secret.encode("utf-8")
    )
    return Fernet(base64.urlsafe_b64encode(key))


def backup_ca(certs_dir: str | Path, target: str | Path, recovery_secret: str) -> Path:
    certs_dir = Path(certs_dir)
    payload = b""
    for name in ("ca.pem", "ca-key.pem"):
        data = (certs_dir / name).read_bytes()
        payload += f"---FILE:{name}:{len(data)}---\n".encode("utf-8") + data + b"\n"
    encrypted = _fernet(recovery_secret).encrypt(payload)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encrypted)
    return target


def restore_ca(source: str | Path, certs_dir: str | Path, recovery_secret: str) -> None:
    payload = _fernet(recovery_secret).decrypt(Path(source).read_bytes())
    certs_dir = Path(certs_dir)
    certs_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    while cursor < len(payload):
        if payload[cursor : cursor + 8] != b"---FILE:":
            break
        header_end = payload.index(b"---\n", cursor)
        header = payload[cursor + 8 : header_end].decode("utf-8")
        name, size_text = header.rsplit(":", 1)
        size = int(size_text)
        data_start = header_end + 4
        data_end = data_start + size
        (certs_dir / name).write_bytes(payload[data_start:data_end])
        cursor = data_end + 1

