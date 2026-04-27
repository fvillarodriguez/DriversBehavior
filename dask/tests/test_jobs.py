from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cluster_app.jobs.dependency_installer import hash_requirements
from cluster_app.jobs.packager import inspect_job_folder
from cluster_app.jobs.workspace import prepare_workspace, workspace_from_existing


class JobTests(unittest.TestCase):
    def test_packager_requires_entrypoint_when_multiple_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "a.py").write_text("print('a')", encoding="utf-8")
            (source / "b.py").write_text("print('b')", encoding="utf-8")
            with self.assertRaises(ValueError):
                inspect_job_folder(source)
            package = inspect_job_folder(source, "a.py")
            self.assertEqual(str(package.entrypoint), "a.py")

    def test_workspace_copies_code_and_can_be_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            workspace_root = Path(tmp) / "workspace"
            source.mkdir()
            (source / "main.py").write_text("print('ok')", encoding="utf-8")
            workspace = prepare_workspace(workspace_root, "job-1", source)
            self.assertTrue((workspace.code_dir / "main.py").exists())
            loaded = workspace_from_existing(workspace.root, "main.py")
            self.assertEqual(loaded.entrypoint, workspace.entrypoint)

    def test_requirements_hash_normalizes_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("\n# comment\nfastapi==1\n", encoding="utf-8")
            self.assertEqual(hash_requirements(req), hash_requirements(req))
            self.assertNotEqual(hash_requirements(req), "empty")


if __name__ == "__main__":
    unittest.main()

