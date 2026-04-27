from __future__ import annotations

import hmac
import secrets
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cluster_app.api.app import services_from_request

router = APIRouter(prefix="/api/auth", tags=["auth"])
_sessions: dict[str, int] = {}
_session_scopes: dict[str, str] = {}
_session_seen: dict[str, datetime] = {}
_ONLINE_WINDOW = timedelta(minutes=2)
_SECRET_FILE = "auth-secret.key"


class RegisterRequest(BaseModel):
    name: str
    email: str


class LoginRequest(BaseModel):
    name: str | None = None
    email: str


@router.post("/register")
async def register(payload: RegisterRequest, request: Request):
    services = services_from_request(request)
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if not payload.email.strip():
        raise HTTPException(status_code=400, detail="Email is required")
    user = services.users.get_or_create(payload.name.strip(), payload.email.strip())
    token = issue_token(request, int(user.id or 0))
    data = asdict(user)
    data.pop("password_hash", None)
    return {"token": token, "user": data}


@router.post("/login")
async def login(payload: LoginRequest, request: Request):
    services = services_from_request(request)
    if not payload.email.strip():
        raise HTTPException(status_code=400, detail="Email is required")
    name = payload.name.strip() if payload.name else payload.email.split("@", 1)[0]
    user = services.users.get_or_create(name, payload.email.strip())
    token = issue_token(request, int(user.id or 0))
    data = asdict(user)
    data.pop("password_hash", None)
    return {"token": token, "user": data}


def issue_token(request: Request, user_id: int) -> str:
    nonce = secrets.token_urlsafe(16)
    body = f"{user_id}.{nonce}"
    signature = hmac.new(_auth_secret(request), body.encode("utf-8"), sha256).hexdigest()
    token = f"v1.{body}.{signature}"
    _remember_session(request, token, user_id)
    return token


def current_user_id(request: Request) -> int:
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip()
    user_id = _sessions.get(token)
    if user_id and _session_scopes.get(token) == _auth_scope(request):
        services = services_from_request(request)
        if services.users.get(user_id) is not None:
            _touch_session(request, token, user_id)
            return user_id
    if token:
        _forget_session(token)
    user_id = _verify_signed_token(request, token)
    if user_id:
        _remember_session(request, token, user_id)
        return user_id
    raise HTTPException(status_code=401, detail="Session expired. Please login again.")


def online_user_activity(request: Request) -> dict[int, str]:
    scope = _auth_scope(request)
    _prune_sessions()
    activity: dict[int, datetime] = {}
    for token, user_id in _sessions.items():
        if _session_scopes.get(token) != scope:
            continue
        seen_at = _session_seen.get(token)
        if seen_at is None:
            continue
        previous = activity.get(user_id)
        if previous is None or seen_at > previous:
            activity[user_id] = seen_at
    return {
        user_id: seen_at.isoformat(timespec="seconds")
        for user_id, seen_at in activity.items()
    }


def forget_user_sessions(request: Request, user_id: int) -> None:
    scope = _auth_scope(request)
    for token, session_user_id in list(_sessions.items()):
        if session_user_id == user_id and _session_scopes.get(token) == scope:
            _forget_session(token)


def _auth_secret(request: Request) -> bytes:
    services = services_from_request(request)
    path = services.config.paths.state_dir / _SECRET_FILE
    if path.exists():
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    secret = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret.hex(), encoding="utf-8")
    return secret


def _verify_signed_token(request: Request, token: str) -> int | None:
    parts = token.split(".")
    if len(parts) != 4 or parts[0] != "v1":
        return None
    _, user_id_text, nonce, signature = parts
    if not user_id_text.isdigit() or not nonce:
        return None
    body = f"{user_id_text}.{nonce}"
    expected = hmac.new(_auth_secret(request), body.encode("utf-8"), sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    services = services_from_request(request)
    user_id = int(user_id_text)
    if services.users.get(user_id) is None:
        return None
    return user_id


def _remember_session(request: Request, token: str, user_id: int) -> None:
    _sessions[token] = user_id
    _session_scopes[token] = _auth_scope(request)
    _touch_session(request, token, user_id)


def _touch_session(request: Request, token: str, user_id: int) -> None:
    _sessions[token] = user_id
    _session_scopes[token] = _auth_scope(request)
    _session_seen[token] = datetime.now(UTC)
    _prune_sessions()


def _prune_sessions() -> None:
    cutoff = datetime.now(UTC) - _ONLINE_WINDOW
    for token, seen_at in list(_session_seen.items()):
        if seen_at < cutoff:
            _forget_session(token)


def _forget_session(token: str) -> None:
    _sessions.pop(token, None)
    _session_scopes.pop(token, None)
    _session_seen.pop(token, None)


def _auth_scope(request: Request) -> str:
    services = services_from_request(request)
    return str(services.config.paths.state_dir.resolve())
