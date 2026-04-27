from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cluster_app.config.schema import AppConfig
from cluster_app.discovery.mdns import DiscoveredService
from cluster_app.dask_runtime.cluster_scheduler import ClusterSchedulerResolver
from cluster_app.dask_runtime.dashboard_proxy import dashboard_link
from cluster_app.dask_runtime.resources import dask_resource_flags, memory_limit
from cluster_app.dask_runtime.scheduler import SchedulerProcess
from cluster_app.dask_runtime.worker import WorkerProcess
from cluster_app.nodes.health import choose_free_port
from cluster_app.nodes.presence import NodePresenceMonitor
from cluster_app.security.ca import CertificateBundle
from cluster_app.storage.db import Database, initialize_database
from cluster_app.storage.repositories import NodeRepository


class NodesDaskTests(unittest.TestCase):
    def test_memory_limit_fraction(self) -> None:
        self.assertEqual(memory_limit(1000, 0.9), 900)

    def test_resource_flags_are_deterministic(self) -> None:
        self.assertEqual(dask_resource_flags({"GPU": 1, "CPU": 4}), ["--resources", "CPU=4,GPU=1"])

    def test_dashboard_link_uses_loopback_for_any_host(self) -> None:
        self.assertEqual(dashboard_link("0.0.0.0", 8787), "http://127.0.0.1:8787/status")

    def test_choose_free_port_returns_int(self) -> None:
        port = choose_free_port("127.0.0.1", 0)
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)

    def test_scheduler_process_uses_current_python_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, bundle = _runtime_config(Path(tmp))
            with patch("cluster_app.dask_runtime.scheduler.subprocess.Popen") as popen:
                popen.return_value = Mock()
                SchedulerProcess(cfg, bundle).start()

            cmd = popen.call_args.args[0]
            self.assertEqual(cmd[:3], [sys.executable, "-m", "distributed.cli.dask_scheduler"])
            self.assertIn("--tls-ca-file", cmd)

    def test_worker_process_uses_current_python_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, bundle = _runtime_config(Path(tmp))
            with patch("cluster_app.dask_runtime.worker.subprocess.Popen") as popen:
                popen.return_value = Mock()
                WorkerProcess(cfg, bundle, "tls://127.0.0.1:8786", {"CPU": 2}, 1024).start()

            cmd = popen.call_args.args[0]
            self.assertEqual(cmd[:3], [sys.executable, "-m", "distributed.cli.dask_worker"])
            self.assertIn("--host", cmd)
            self.assertIn("--tls-ca-file", cmd)

    def test_presence_scan_registers_discovered_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = AppConfig()
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.cluster.presence_stale_seconds = 300
            cfg.ensure_directories()
            db = Database(cfg.db_path)
            initialize_database(db)
            nodes = NodeRepository(db)
            monitor = NodePresenceMonitor(cfg, nodes)
            fake = DiscoveredService(
                name="remote._dask-cluster._tcp.local.",
                host="192.168.1.55",
                port=18080,
                properties={
                    "uuid": "remote-uuid",
                    "name": "remote-node",
                    "preferred": "true",
                    "resources_json": '{"CPU":4}',
                },
            )

            with patch("cluster_app.nodes.presence.discover", return_value=[fake]):
                asyncio.run(monitor.scan_once())

            remote = nodes.get("remote-uuid")
            self.assertIsNotNone(remote)
            assert remote is not None
            self.assertEqual(remote.name, "remote-node")
            self.assertEqual(remote.host, "192.168.1.55")
            self.assertEqual(remote.resources, {"CPU": 4})

    def test_cluster_scheduler_resolver_uses_remote_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = AppConfig()
            cfg.paths.state_dir = root / "state"
            cfg.paths.workspace_dir = root / "workspace"
            cfg.paths.envs_dir = root / "envs"
            cfg.paths.logs_dir = root / "logs"
            cfg.ensure_directories()
            db = Database(cfg.db_path)
            initialize_database(db)
            nodes = NodeRepository(db)
            nodes.upsert(
                uuid="remote-uuid",
                name="Remote Node",
                host="192.168.1.50",
                port=18080,
                resources={"CPU": 4},
                preferred=True,
            )
            local_scheduler = Mock()
            local_scheduler.status.return_value = {
                "running": False,
                "managed": False,
                "pid": None,
                "address": "tls://127.0.0.1:8786",
                "dashboard": "http://127.0.0.1:8787/status",
                "dashboard_reachable": False,
            }
            resolver = ClusterSchedulerResolver(cfg, nodes, local_scheduler)

            with (
                patch(
                    "cluster_app.dask_runtime.cluster_scheduler._json_request",
                    return_value={
                        "running": True,
                        "managed": True,
                        "pid": 123,
                        "address": "tls://127.0.0.1:8786",
                        "dashboard": "http://127.0.0.1:8787/status",
                        "dashboard_reachable": True,
                    },
                ),
                patch(
                    "cluster_app.dask_runtime.cluster_scheduler._service_reachable",
                    side_effect=lambda url: "192.168.1.50" in url,
                ),
            ):
                status = resolver.status()

            self.assertFalse(status["local"])
            self.assertEqual(status["node_uuid"], "remote-uuid")
            self.assertEqual(status["address"], "tls://192.168.1.50:8786")
            self.assertEqual(status["dashboard"], "http://192.168.1.50:8787/status")


def _runtime_config(root: Path) -> tuple[AppConfig, CertificateBundle]:
    cfg = AppConfig()
    cfg.paths.state_dir = root / "state"
    cfg.paths.workspace_dir = root / "workspace"
    cfg.paths.envs_dir = root / "envs"
    cfg.paths.logs_dir = root / "logs"
    cfg.ensure_directories()
    certs = root / "certs"
    certs.mkdir()
    bundle = CertificateBundle(
        ca_cert=certs / "ca.pem",
        cert=certs / "node.pem",
        key=certs / "node-key.pem",
        fingerprint="abc",
    )
    return cfg, bundle


if __name__ == "__main__":
    unittest.main()
