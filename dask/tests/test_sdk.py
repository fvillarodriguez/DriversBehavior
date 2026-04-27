from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cluster_app.config.schema import AppConfig, config_to_dict
from cluster_app.sdk import DaskCluster
from cluster_app.security.ca import CertificateBundle


class SdkTests(unittest.TestCase):
    def test_client_auto_starts_scheduler_worker_and_connects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cluster = DaskCluster(config_path=_config_path(root), control_plane=False)
            stopped = _scheduler_status(False)
            running = _scheduler_status(True)
            pending_statuses = [stopped]

            def scheduler_status():
                return pending_statuses.pop(0) if pending_statuses else running

            cluster.services.cluster_scheduler.status = Mock(side_effect=scheduler_status)
            cluster.services.cluster_scheduler.remote_worker_certificate = Mock(return_value=None)
            cluster.services.scheduler.start = Mock(return_value=running)
            cluster.services.worker.start = Mock(return_value={"running": True})
            cluster.services.worker.status = Mock(return_value={"running": True, "pid": 123})
            fake_client = Mock()

            with (
                patch.object(cluster, "_client_certificate_bundle", return_value=_bundle(root)),
                patch("cluster_app.sdk.DaskClientFactory") as factory,
            ):
                factory.return_value.connect.return_value = fake_client
                self.assertIs(cluster.client(), fake_client)

            cluster.services.scheduler.start.assert_called_once()
            cluster.services.worker.start.assert_called_once_with("tls://127.0.0.1:8786", cert_bundle=None)
            factory.assert_called_once()

    def test_submit_map_and_gather_delegate_to_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cluster = DaskCluster(
                config_path=_config_path(Path(tmp)),
                auto_start=False,
                control_plane=False,
            )
            fake_client = Mock()
            fake_client.submit.return_value = "future"
            fake_client.map.return_value = ["future-1"]
            fake_client.gather.return_value = [1]
            cluster._client = fake_client

            self.assertEqual(cluster.submit(str, 1), "future")
            self.assertEqual(cluster.map(str, [1]), ["future-1"])
            self.assertEqual(cluster.gather(["future-1"]), [1])
            fake_client.submit.assert_called_once_with(str, 1)
            fake_client.map.assert_called_once_with(str, [1])
            fake_client.gather.assert_called_once_with(["future-1"])

    def test_validate_shared_paths_checks_local_paths_without_autostart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cluster = DaskCluster(
                config_path=_config_path(root),
                auto_start=False,
                control_plane=False,
            )
            existing = root / "data"
            existing.mkdir()

            result = cluster.validate_shared_paths([existing, root / "missing"])

            self.assertFalse(result["ok"])
            self.assertEqual(result["local"][str(existing)]["exists"], True)
            self.assertEqual(result["local"][str(root / "missing")]["exists"], False)
            self.assertEqual(result["workers"], {})

    def test_configure_path_mappings_persists_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cluster = DaskCluster(
                config_path=_config_path(root),
                auto_start=False,
                control_plane=False,
            )

            payload = cluster.configure_path_mappings({"data": str(root / "data")})

            self.assertEqual(payload["mappings"]["data"], str(root / "data"))
            self.assertEqual(cluster.path_mappings()["mappings"]["data"], str(root / "data"))

    def test_stop_closes_client_and_owned_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cluster = DaskCluster(config_path=_config_path(Path(tmp)), control_plane=False)
            client = Mock()
            control_plane = Mock()
            cluster._client = client
            cluster._control_plane = control_plane
            cluster.services.worker.stop = Mock()
            cluster.services.scheduler.stop = Mock()

            cluster.stop()

            client.close.assert_called_once()
            control_plane.stop.assert_called_once()
            cluster.services.worker.stop.assert_called_once()
            cluster.services.scheduler.stop.assert_called_once()


def _config_path(root: Path) -> Path:
    cfg = AppConfig()
    cfg.paths.state_dir = root / "state"
    cfg.paths.workspace_dir = root / "workspace"
    cfg.paths.envs_dir = root / "envs"
    cfg.paths.logs_dir = root / "logs"
    cfg.cluster.node_presence = False
    cfg.network.auto_ports = False
    path = root / "config.json"
    path.write_text(json.dumps(config_to_dict(cfg)), encoding="utf-8")
    return path


def _scheduler_status(running: bool) -> dict[str, object]:
    return {
        "running": running,
        "local": True,
        "managed": running,
        "pid": 123 if running else None,
        "address": "tls://127.0.0.1:8786",
        "dashboard": "http://127.0.0.1:8787/status",
        "dashboard_reachable": running,
        "scheduler_reachable": running,
    }


def _bundle(root: Path) -> CertificateBundle:
    certs = root / "certs"
    certs.mkdir(exist_ok=True)
    return CertificateBundle(
        ca_cert=certs / "ca.pem",
        cert=certs / "node.pem",
        key=certs / "node-key.pem",
        fingerprint="abc",
    )


if __name__ == "__main__":
    unittest.main()
