from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _import_clustering_tabs_app(monkeypatch):
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda *args, **kwargs: args[0] if args else []
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    sys.modules.pop("clustering_tabs_app", None)
    return importlib.import_module("clustering_tabs_app")


def test_char_normalize_profile_uses_reference_for_single_cluster(monkeypatch):
    clustering_tabs_app = _import_clustering_tabs_app(monkeypatch)

    reference = pd.DataFrame(
        {
            "avg_speed_kmh": [60.0, 80.0, 100.0],
            "avg_relative_speed": [0.2, 0.5, 0.8],
            "avg_headway_s": [1.0, 2.0, 3.0],
        },
        index=[0, 1, 2],
    )
    selected = reference.loc[[1]]

    normalized = clustering_tabs_app._char_normalize_profile(
        selected,
        "Min-max",
        reference=reference,
    )

    assert normalized.loc[1, "avg_speed_kmh"] == pytest.approx(0.5)
    assert normalized.loc[1, "avg_relative_speed"] == pytest.approx(0.5)
    assert normalized.loc[1, "avg_headway_s"] == pytest.approx(0.5)


def test_char_normalize_profile_returns_zero_for_constant_reference(
    monkeypatch,
):
    clustering_tabs_app = _import_clustering_tabs_app(monkeypatch)

    reference = pd.DataFrame(
        {
            "avg_speed_kmh": [80.0, 80.0, 80.0],
            "avg_relative_speed": [0.1, 0.3, 0.5],
            "avg_headway_s": [2.0, 2.0, 2.0],
        },
        index=[0, 1, 2],
    )
    selected = reference.loc[[1]]

    normalized = clustering_tabs_app._char_normalize_profile(
        selected,
        "Min-max",
        reference=reference,
    )

    assert normalized.loc[1, "avg_speed_kmh"] == 0.0
    assert normalized.loc[1, "avg_relative_speed"] == pytest.approx(0.5)
    assert normalized.loc[1, "avg_headway_s"] == 0.0


def test_cluster_visualization_normalize_profile_uses_reference():
    import cluster_visualization_app

    reference = pd.DataFrame(
        {
            "avg_speed_kmh": [60.0, 80.0, 100.0],
            "avg_relative_speed": [0.2, 0.5, 0.8],
            "avg_headway_s": [1.0, 2.0, 3.0],
        },
        index=[0, 1, 2],
    )
    selected = reference.loc[[1]]

    normalized = cluster_visualization_app.normalize_profile(
        selected,
        "Min-max",
        reference=reference,
    )

    assert normalized.loc[1, "avg_speed_kmh"] == pytest.approx(0.5)
    assert normalized.loc[1, "avg_relative_speed"] == pytest.approx(0.5)
    assert normalized.loc[1, "avg_headway_s"] == pytest.approx(0.5)
