from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"


class NodeStatus(StrEnum):
    PENDING = "pending"
    ONLINE = "online"
    OFFLINE = "offline"
    REVOKED = "revoked"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    INTERRUPTED = "interrupted"
    PAUSED = "paused"


@dataclass(slots=True)
class User:
    id: int | None
    name: str
    email: str
    password_hash: str
    role: UserRole
    created_at: str


@dataclass(slots=True)
class Node:
    id: int | None
    uuid: str
    name: str
    host: str
    port: int | None
    status: NodeStatus
    is_preferred_scheduler: bool
    resources: dict[str, Any]
    cert_fingerprint: str | None
    last_seen_at: str | None
    created_at: str


@dataclass(slots=True)
class Job:
    id: str
    user_id: int
    name: str
    source_path: str
    entrypoint: str
    args: list[str]
    status: JobStatus
    priority: int
    retries_left: int
    workspace_path: str | None
    dask_scheduler_url: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    last_warning_at: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class JobLog:
    id: int | None
    job_id: str
    stream: str
    message: str
    created_at: str

