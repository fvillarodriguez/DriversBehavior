from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cluster_app.api.app import create_app
from cluster_app.config.schema import AppConfig
from cluster_app.discovery.manual_ip import TcpProbeResult
from cluster_app.storage.models import JobStatus


class ApiTests(unittest.TestCase):
    def test_root_serves_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.ensure_directories()
            client = TestClient(create_app(cfg))
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])
            self.assertIn("Dask Cluster App", response.text)
            self.assertIn("Start Scheduler", response.text)
            self.assertIn("Start Worker", response.text)
            self.assertIn("Online Users", response.text)
            self.assertNotIn("Registered Users", response.text)
            self.assertNotIn('type="password"', response.text)

    def test_dashboard_serves_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.ensure_directories()
            client = TestClient(create_app(cfg))
            response = client.get("/dashboard")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Dashboard", response.text)
            self.assertIn("Funciones de la app", response.text)
            self.assertIn("Jobs y cola", response.text)
            self.assertIn("CLI cluster-app", response.text)
            self.assertIn("Registered Users", response.text)
            self.assertNotIn("Dask Native Dashboard", response.text)
            self.assertNotIn("Start Worker", response.text)

    def test_scheduler_status_and_start_requires_login(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.cluster.node_presence = False
            cfg.ensure_directories()
            client = TestClient(create_app(cfg))

            status = client.get("/api/admin/scheduler/status")
            self.assertEqual(status.status_code, 200)
            self.assertIn("running", status.json())
            start = client.post("/api/admin/scheduler/start")
            self.assertEqual(start.status_code, 401)

    def test_startup_does_not_autostart_dask_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.cluster.node_presence = False
            cfg.ensure_directories()
            with (
                patch("cluster_app.dask_runtime.scheduler_runtime.SchedulerRuntime.start") as scheduler,
                patch("cluster_app.dask_runtime.worker_runtime.WorkerRuntime.start") as worker,
            ):
                with TestClient(create_app(cfg)) as client:
                    response = client.get("/api/metrics/status")
                    self.assertEqual(response.status_code, 200)

            scheduler.assert_not_called()
            worker.assert_not_called()

    def test_passwordless_register_is_idempotent_and_users_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.ensure_directories()
            client = TestClient(create_app(cfg))
            payload = {"name": "Test", "email": "test@example.com"}
            created = client.post("/api/auth/register", json=payload)
            self.assertEqual(created.status_code, 200)
            token = created.json()["token"]
            duplicate = client.post(
                "/api/auth/register",
                json={"name": "Test Renamed", "email": "test@example.com"},
            )
            self.assertEqual(duplicate.status_code, 200)
            self.assertEqual(duplicate.json()["user"]["name"], "Test Renamed")
            login = client.post(
                "/api/auth/login",
                json={"email": "test@example.com"},
            )
            self.assertEqual(login.status_code, 200)
            users = client.get(
                "/api/admin/users",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(users.status_code, 200)
            self.assertEqual(users.json()["users"][0]["email"], "test@example.com")
            self.assertNotIn("password_hash", users.json()["users"][0])
            online = client.get(
                "/api/admin/users/online",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(online.status_code, 200)
            self.assertIn(
                "test@example.com",
                [user["email"] for user in online.json()["users"]],
            )
            second = client.post(
                "/api/auth/register",
                json={"name": "Second", "email": "second@example.com"},
            )
            self.assertEqual(second.status_code, 200)
            second_id = second.json()["user"]["id"]
            deleted = client.delete(
                f"/api/admin/users/{second_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(deleted.status_code, 200)
            users_after_delete = client.get(
                "/api/admin/users",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(
                [user["email"] for user in users_after_delete.json()["users"]],
                ["test@example.com"],
            )

    def test_auth_token_survives_app_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.cluster.node_presence = False
            cfg.ensure_directories()
            first_client = TestClient(create_app(cfg))
            registered = first_client.post(
                "/api/auth/register",
                json={"name": "Admin", "email": "admin@example.com"},
            )
            self.assertEqual(registered.status_code, 200)
            token = registered.json()["token"]

            second_client = TestClient(create_app(cfg))
            protected = second_client.get(
                "/api/admin/firewall-plan",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(protected.status_code, 200)

    def test_filesystem_lists_folders_and_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "job").mkdir()
            (root / "job" / "main.py").write_text("print('ok')", encoding="utf-8")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            cfg = AppConfig()
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.ensure_directories()
            client = TestClient(create_app(cfg))

            listing = client.get("/api/filesystem/list", params={"path": str(root)})
            self.assertEqual(listing.status_code, 200)
            entries = listing.json()["entries"]
            job_entry = next(entry for entry in entries if entry["name"] == "job")
            self.assertEqual(job_entry["kind"], "directory")

            scripts = client.get("/api/filesystem/python-files", params={"path": str(root / "job")})
            self.assertEqual(scripts.status_code, 200)
            self.assertEqual(
                scripts.json()["files"],
                [{"path": str((root / "job" / "main.py").resolve()), "relative_path": "main.py"}],
            )

    def test_node_self_and_manual_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.cluster.node_presence = False
            cfg.ensure_directories()
            client = TestClient(create_app(cfg))

            remote = {
                "uuid": "remote-uuid",
                "name": "Remote Node",
                "host": "127.0.0.1",
                "port": 18080,
                "resources": {"CPU": 4},
            }
            with patch("cluster_app.api.routes_nodes._fetch_remote_node", return_value=remote):
                response = client.post(
                    "/api/nodes/manual",
                    json={"host": "192.168.1.50", "port": 18080},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["host"], "192.168.1.50")
            listing = client.get("/api/nodes")
            self.assertEqual(listing.status_code, 200)
            self.assertEqual(listing.json()[0]["uuid"], "remote-uuid")

    def test_cleanup_job_records_and_old_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.cluster.node_presence = False
            cfg.ensure_directories()
            app = create_app(cfg)
            client = TestClient(app)
            registered = client.post(
                "/api/auth/register",
                json={"name": "Admin", "email": "admin@example.com"},
            )
            token = registered.json()["token"]
            services = app.state.services
            user_id = int(registered.json()["user"]["id"])
            services.jobs.create("done-job", user_id, "done", "/tmp/done", "main.py")
            services.jobs.create("queued-job", user_id, "queued", "/tmp/queued", "main.py")
            services.jobs.set_status("done-job", JobStatus.SUCCEEDED)
            services.nodes.upsert("online", "Online", "127.0.0.1", 18080)
            services.nodes.upsert("old", "Old", "192.168.1.20", 18080)
            services.nodes.revoke("old")

            cleared = client.delete(
                "/api/jobs/records",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(cleared.status_code, 200)
            self.assertEqual(cleared.json()["removed"], 1)
            self.assertIsNone(services.jobs.get("done-job"))
            self.assertIsNotNone(services.jobs.get("queued-job"))

            removed = client.post(
                "/api/nodes/cleanup-old",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(removed.status_code, 200)
            self.assertEqual(removed.json()["removed"], 1)
            self.assertIsNotNone(services.nodes.get("online"))
            self.assertIsNone(services.nodes.get("old"))

    def test_node_health_checks_self_on_loopback_configured_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            root = Path(tmp)
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.network.web_port = 18180
            cfg.ensure_directories()
            app = create_app(cfg)
            client = TestClient(app)
            services = app.state.services
            self.assertIsNotNone(services.presence)
            node_uuid = services.presence.identity.uuid
            services.nodes.upsert(node_uuid, "Self", "192.168.3.121", 18080)

            with patch(
                "cluster_app.api.routes_nodes.probe_tcp",
                return_value=TcpProbeResult("127.0.0.1", 18180, True),
            ) as probe:
                response = client.get("/api/nodes/health")

            self.assertEqual(response.status_code, 200)
            probe.assert_called_once_with("127.0.0.1", 18180)
            node = response.json()["nodes"][0]
            self.assertEqual(node["host"], "192.168.3.121")
            self.assertEqual(node["checked_host"], "127.0.0.1")
            self.assertEqual(node["port"], 18180)
            self.assertTrue(node["reachable"])


if __name__ == "__main__":
    unittest.main()
