from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch

from cluster_app.integrations.streamlit import render_cluster_panel


class StreamlitIntegrationTests(unittest.TestCase):
    def test_streamlit_dependency_is_lazy_and_reports_clear_error(self) -> None:
        with patch.dict(sys.modules, {"streamlit": None}):
            with self.assertRaisesRegex(RuntimeError, "streamlit is required"):
                render_cluster_panel(Mock())

    def test_render_panel_does_not_start_cluster_without_button_press(self) -> None:
        fake_streamlit = _FakeStreamlit()
        cluster = Mock()
        cluster.status.return_value = {
            "started": False,
            "scheduler": {
                "running": False,
                "address": "tls://127.0.0.1:8786",
            },
            "worker": {"running": False},
            "nodes": [],
            "dashboard": "http://127.0.0.1:8787/status",
        }

        with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            returned = render_cluster_panel(cluster)

        self.assertIs(returned, cluster)
        cluster.start.assert_not_called()
        cluster.stop.assert_not_called()


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state = {}

    def subheader(self, value: str) -> None:
        self.subheader_value = value

    def columns(self, count: int):
        return [_FakeColumn() for _ in range(count)]

    def link_button(self, label: str, url: str) -> None:
        self.link = (label, url)

    def caption(self, value: str) -> None:
        self.caption_value = value

    def rerun(self) -> None:
        self.reran = True


class _FakeColumn:
    def metric(self, label: str, value: str) -> None:
        self.metric_value = (label, value)

    def button(self, *args, **kwargs) -> bool:
        return False


if __name__ == "__main__":
    unittest.main()
