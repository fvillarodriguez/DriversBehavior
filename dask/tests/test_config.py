from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cluster_app.config.loader import load_config, write_default_config
from cluster_app.config.schema import AppConfig, config_from_dict, config_to_dict


class ConfigTests(unittest.TestCase):
    def test_config_from_dict_overrides_nested_values(self) -> None:
        cfg = config_from_dict({
            "network": {"web_port": 9090},
            "cluster": {"preferred_scheduler": True},
            "path_mappings": {"mappings": {"data": "~/datasets"}},
        })
        self.assertEqual(cfg.network.web_port, 9090)
        self.assertTrue(cfg.cluster.preferred_scheduler)
        self.assertEqual(cfg.path_mappings.mappings, {"data": "~/datasets"})

    def test_config_to_dict_handles_slotted_dataclasses(self) -> None:
        data = config_to_dict(AppConfig())
        self.assertIn("network", data)
        self.assertIn("paths", data)

    def test_write_default_config_is_loadable_without_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            write_default_config(path)
            cfg = load_config(path)
            self.assertEqual(cfg.cluster.scheduler_wait_seconds, 15)


if __name__ == "__main__":
    unittest.main()
