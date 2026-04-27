from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cluster_app.storage.db import Database, initialize_database
from cluster_app.storage.models import JobStatus, UserRole
from cluster_app.storage.repositories import JobRepository, NodeRepository, UserRepository


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "cluster.db")
        initialize_database(self.db)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_first_user_is_admin_then_user(self) -> None:
        users = UserRepository(self.db)
        first = users.create("Admin", "admin@example.com", "secret")
        second = users.create("User", "user@example.com", "secret")
        self.assertEqual(first.role, UserRole.ADMIN)
        self.assertEqual(second.role, UserRole.USER)
        self.assertIsNotNone(users.authenticate("admin@example.com", "secret"))

    def test_node_upsert_and_revoke(self) -> None:
        nodes = NodeRepository(self.db)
        node = nodes.upsert("uuid-1", "node", "127.0.0.1", 8080, {"CPU": 4})
        self.assertEqual(node.resources["CPU"], 4)
        nodes.revoke("uuid-1")
        self.assertEqual(nodes.get("uuid-1").status.value, "revoked")

    def test_job_queue_order(self) -> None:
        users = UserRepository(self.db)
        user = users.create("Admin", "admin@example.com", "secret")
        jobs = JobRepository(self.db)
        jobs.create("job-1", int(user.id), "one", "/tmp/one", "main.py")
        jobs.create("job-2", int(user.id), "two", "/tmp/two", "main.py")
        self.assertEqual(jobs.next_queued().id, "job-1")
        jobs.set_running("job-1", "/tmp/ws")
        self.assertEqual(jobs.active().status, JobStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()

