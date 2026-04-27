from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from hashlib import pbkdf2_hmac
from typing import Any

from cluster_app.storage.db import Database
from cluster_app.storage.models import Job, JobLog, JobStatus, Node, NodeStatus, User, UserRole, utcnow


def _json_loads(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        _, salt, expected = encoded.split("$", 2)
    except ValueError:
        return False
    return secrets.compare_digest(hash_password(password, salt).split("$", 2)[2], expected)


class UserHasActiveJobsError(RuntimeError):
    pass


class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    def count(self) -> int:
        with self.db.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"])

    def create(self, name: str, email: str, password: str | None = None) -> User:
        role = UserRole.ADMIN if self.count() == 0 else UserRole.USER
        password = password or secrets.token_urlsafe(32)
        with self.db.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO users(name, email, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, email.lower(), hash_password(password), role.value, utcnow()),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
            return self._from_row(row)

    def get_or_create(self, name: str, email: str) -> User:
        existing = self.get_by_email(email)
        if existing:
            if name.strip() and existing.name != name.strip():
                with self.db.connect() as conn:
                    conn.execute(
                        "UPDATE users SET name = ? WHERE id = ?",
                        (name.strip(), existing.id),
                    )
                return self.get(int(existing.id or 0)) or existing
            return existing
        return self.create(name.strip(), email.strip())

    def authenticate(self, email: str, password: str) -> User | None:
        user = self.get_by_email(email)
        if not user or not verify_password(password, user.password_hash):
            return None
        return user

    def get_by_email(self, email: str) -> User | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
        return self._from_row(row) if row else None

    def get(self, user_id: int) -> User | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._from_row(row) if row else None

    def list(self) -> list[User]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC, id DESC").fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, user_id: int) -> bool:
        active_statuses = (JobStatus.QUEUED.value, JobStatus.RUNNING.value, JobStatus.PAUSED.value)
        placeholders = ",".join("?" for _ in active_statuses)
        with self.db.connect() as conn:
            active = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM jobs
                WHERE user_id = ? AND status IN ({placeholders})
                """,
                (user_id, *active_statuses),
            ).fetchone()
            if int(active["count"]):
                raise UserHasActiveJobsError
            conn.execute("DELETE FROM jobs WHERE user_id = ?", (user_id,))
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return bool(cur.rowcount)

    @staticmethod
    def _from_row(row: Any) -> User:
        return User(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            password_hash=row["password_hash"],
            role=UserRole(row["role"]),
            created_at=row["created_at"],
        )


class NodeRepository:
    def __init__(self, db: Database):
        self.db = db

    def upsert(
        self,
        uuid: str,
        name: str,
        host: str,
        port: int | None,
        resources: dict[str, Any] | None = None,
        cert_fingerprint: str | None = None,
        preferred: bool = False,
    ) -> Node:
        now = utcnow()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO nodes(
                    uuid, name, host, port, status, is_preferred_scheduler,
                    resources_json, cert_fingerprint, last_seen_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uuid) DO UPDATE SET
                    name = excluded.name,
                    host = excluded.host,
                    port = excluded.port,
                    status = excluded.status,
                    is_preferred_scheduler = excluded.is_preferred_scheduler,
                    resources_json = excluded.resources_json,
                    cert_fingerprint = COALESCE(excluded.cert_fingerprint, nodes.cert_fingerprint),
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    uuid,
                    name,
                    host,
                    port,
                    NodeStatus.ONLINE.value,
                    int(preferred),
                    json.dumps(resources or {}),
                    cert_fingerprint,
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM nodes WHERE uuid = ?", (uuid,)).fetchone()
            return self._from_row(row)

    def mark_offline(self, uuid: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE nodes SET status = ?, last_seen_at = ? WHERE uuid = ?",
                (NodeStatus.OFFLINE.value, utcnow(), uuid),
            )

    def mark_stale_offline(self, cutoff_iso: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE nodes
                SET status = ?, last_seen_at = ?
                WHERE status = ? AND last_seen_at < ?
                """,
                (NodeStatus.OFFLINE.value, utcnow(), NodeStatus.ONLINE.value, cutoff_iso),
            )

    def revoke(self, uuid: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE nodes SET status = ?, last_seen_at = ? WHERE uuid = ?",
                (NodeStatus.REVOKED.value, utcnow(), uuid),
            )

    def delete_inactive(self) -> int:
        with self.db.connect() as conn:
            cur = conn.execute(
                "DELETE FROM nodes WHERE status != ?",
                (NodeStatus.ONLINE.value,),
            )
            return int(cur.rowcount)

    def list(self) -> list[Node]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM nodes ORDER BY name").fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, uuid: str) -> Node | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE uuid = ?", (uuid,)).fetchone()
        return self._from_row(row) if row else None

    @staticmethod
    def _from_row(row: Any) -> Node:
        return Node(
            id=row["id"],
            uuid=row["uuid"],
            name=row["name"],
            host=row["host"],
            port=row["port"],
            status=NodeStatus(row["status"]),
            is_preferred_scheduler=bool(row["is_preferred_scheduler"]),
            resources=_json_loads(row["resources_json"], {}),
            cert_fingerprint=row["cert_fingerprint"],
            last_seen_at=row["last_seen_at"],
            created_at=row["created_at"],
        )


class JobRepository:
    def __init__(self, db: Database):
        self.db = db

    def create(
        self,
        job_id: str,
        user_id: int,
        name: str,
        source_path: str,
        entrypoint: str,
        args: list[str] | None = None,
        retries_left: int = 1,
        metadata: dict[str, Any] | None = None,
        workspace_path: str | None = None,
    ) -> Job:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    id, user_id, name, source_path, entrypoint, args_json, status,
                    priority, retries_left, workspace_path, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    user_id,
                    name,
                    source_path,
                    entrypoint,
                    json.dumps(args or []),
                    JobStatus.QUEUED.value,
                    retries_left,
                    workspace_path,
                    utcnow(),
                    json.dumps(metadata or {}),
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._from_row(row)

    def get(self, job_id: str) -> Job | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._from_row(row) if row else None

    def list(self, limit: int = 100) -> list[Job]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def next_queued(self) -> Job | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (JobStatus.QUEUED.value,),
            ).fetchone()
        return self._from_row(row) if row else None

    def active(self) -> Job | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY started_at ASC LIMIT 1",
                (JobStatus.RUNNING.value,),
            ).fetchone()
        return self._from_row(row) if row else None

    def set_running(
        self, job_id: str, workspace_path: str, dask_scheduler_url: str | None = None
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = COALESCE(started_at, ?),
                    workspace_path = ?, dask_scheduler_url = ?
                WHERE id = ?
                """,
                (JobStatus.RUNNING.value, utcnow(), workspace_path, dask_scheduler_url, job_id),
            )

    def set_status(self, job_id: str, status: JobStatus, metadata: dict[str, Any] | None = None) -> None:
        with self.db.connect() as conn:
            if metadata is None:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ? WHERE id = ?",
                    (status.value, utcnow(), job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, finished_at = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (status.value, utcnow(), json.dumps(metadata), job_id),
                )

    def requeue(self, job_id: str, metadata: dict[str, Any] | None = None) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = NULL, metadata_json = COALESCE(?, metadata_json)
                WHERE id = ?
                """,
                (JobStatus.QUEUED.value, json.dumps(metadata) if metadata is not None else None, job_id),
            )

    def mark_time_warning(self, job_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("UPDATE jobs SET last_warning_at = ? WHERE id = ?", (utcnow(), job_id))

    def decrement_retry(self, job_id: str) -> int:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE jobs SET retries_left = MAX(retries_left - 1, 0) WHERE id = ?",
                (job_id,),
            )
            row = conn.execute("SELECT retries_left FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return int(row["retries_left"])

    def add_log(self, job_id: str, stream: str, message: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO job_logs(job_id, stream, message, created_at) VALUES (?, ?, ?, ?)",
                (job_id, stream, message, utcnow()),
            )

    def logs(self, job_id: str, after_id: int = 0, limit: int = 500) -> list[JobLog]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_logs
                WHERE job_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (job_id, after_id, limit),
            ).fetchall()
        return [JobLog(row["id"], row["job_id"], row["stream"], row["message"], row["created_at"]) for row in rows]

    def queue_depth(self) -> int:
        with self.db.connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM jobs WHERE status = ?", (JobStatus.QUEUED.value,)
                ).fetchone()["count"]
            )

    def cleanup_finished_before(self, cutoff_iso: str) -> int:
        finished = (
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELED.value,
            JobStatus.INTERRUPTED.value,
        )
        with self.db.connect() as conn:
            cur = conn.execute(
                f"""
                DELETE FROM jobs
                WHERE status IN ({",".join("?" for _ in finished)})
                AND finished_at IS NOT NULL
                AND finished_at < ?
                """,
                (*finished, cutoff_iso),
            )
            return int(cur.rowcount)

    def clear_finished_records(self) -> int:
        finished = (
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELED.value,
            JobStatus.INTERRUPTED.value,
            JobStatus.PAUSED.value,
        )
        with self.db.connect() as conn:
            cur = conn.execute(
                f"""
                DELETE FROM jobs
                WHERE status IN ({",".join("?" for _ in finished)})
                """,
                finished,
            )
            return int(cur.rowcount)

    @staticmethod
    def should_warn_time_limit(job: Job, warn_after_hours: int, queue_depth: int) -> bool:
        if queue_depth == 0 or job.started_at is None or job.last_warning_at is not None:
            return False
        started = datetime.fromisoformat(job.started_at)
        return datetime.now(UTC) - started >= timedelta_hours(warn_after_hours)

    @staticmethod
    def _from_row(row: Any) -> Job:
        return Job(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            source_path=row["source_path"],
            entrypoint=row["entrypoint"],
            args=_json_loads(row["args_json"], []),
            status=JobStatus(row["status"]),
            priority=row["priority"],
            retries_left=row["retries_left"],
            workspace_path=row["workspace_path"],
            dask_scheduler_url=row["dask_scheduler_url"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            last_warning_at=row["last_warning_at"],
            metadata=_json_loads(row["metadata_json"], {}),
        )


def timedelta_hours(hours: int):
    from datetime import timedelta

    return timedelta(hours=hours)
