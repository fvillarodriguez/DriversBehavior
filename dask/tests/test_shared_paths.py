from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cluster_app.config.schema import PathMappingConfig
from cluster_app.shared_paths import describe_path, resolved_paths_exist


class SharedPathTests(unittest.TestCase):
    def test_manual_alias_makes_path_portable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            file_path = root / "images" / "one.png"
            file_path.parent.mkdir(parents=True)
            file_path.write_text("ok", encoding="utf-8")
            mappings = PathMappingConfig(mappings={"data": str(root)})

            spec = describe_path(file_path, mappings)
            result = resolved_paths_exist([spec], mappings)

            self.assertEqual(spec["strategy"], "mapping")
            self.assertEqual(spec["alias"], "data")
            self.assertEqual(spec["relative"], "images/one.png")
            self.assertTrue(result[str(file_path)]["exists"])

    def test_cwd_relative_paths_are_smart_default(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                os.chdir(root)
                file_path = root / "models" / "model.pkl"
                file_path.parent.mkdir()
                file_path.write_text("ok", encoding="utf-8")

                spec = describe_path(file_path, PathMappingConfig())

                self.assertEqual(spec["strategy"], "cwd")
                self.assertEqual(spec["relative"], "models/model.pkl")
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
