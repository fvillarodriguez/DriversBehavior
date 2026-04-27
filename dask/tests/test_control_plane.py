from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from cluster_app.config.schema import AppConfig
from cluster_app.control_plane import create_control_plane_app
from cluster_app.services import create_services


class ControlPlaneTests(unittest.TestCase):
    def test_headless_control_plane_exposes_minimal_lan_endpoints_without_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(Path(tmp))
            services = create_services(cfg)
            services.queue.start = Mock()
            services.scheduler.status = Mock(
                return_value={
                    "running": True,
                    "managed": True,
                    "pid": 123,
                    "address": "tls://127.0.0.1:8786",
                    "dashboard": "http://127.0.0.1:8787/status",
                    "dashboard_reachable": True,
                    "scheduler_reachable": True,
                }
            )
            services.nodes.upsert("node-1", "Node 1", "127.0.0.1", 18080, {"CPU": 2})
            app = create_control_plane_app(services=services)

            with TestClient(app) as client:
                node = client.get("/api/nodes/self")
                scheduler = client.get("/api/admin/scheduler/status")
                cert = client.post(
                    "/api/admin/worker/certificate",
                    json={
                        "node_uuid": "worker-1",
                        "node_name": "Worker 1",
                        "hosts": ["127.0.0.1"],
                    },
                )
                mappings = client.put(
                    "/api/admin/path-mappings",
                    json={"mappings": {"data": str(Path(tmp) / "data")}},
                )
                resolved = client.post(
                    "/api/admin/path-mappings/resolve",
                    json={"paths": [str(Path(tmp) / "data" / "model.bin")]},
                )

            self.assertEqual(node.status_code, 200)
            self.assertEqual(node.json()["uuid"], "node-1")
            self.assertEqual(scheduler.status_code, 200)
            self.assertTrue(scheduler.json()["running"])
            self.assertEqual(cert.status_code, 200)
            self.assertIn("ca_cert", cert.json())
            self.assertIn("cert", cert.json())
            self.assertIn("key", cert.json())
            self.assertEqual(mappings.status_code, 200)
            self.assertEqual(mappings.json()["mappings"]["data"], str(Path(tmp) / "data"))
            self.assertEqual(resolved.status_code, 200)
            self.assertEqual(resolved.json()["paths"][0]["strategy"], "mapping")
            self.assertEqual(resolved.json()["paths"][0]["alias"], "data")
            services.queue.start.assert_not_called()


def _config(root: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.paths.state_dir = root / "state"
    cfg.paths.workspace_dir = root / "workspace"
    cfg.paths.envs_dir = root / "envs"
    cfg.paths.logs_dir = root / "logs"
    cfg.cluster.node_presence = False
    cfg.ensure_directories()
    return cfg


if __name__ == "__main__":
    unittest.main()
