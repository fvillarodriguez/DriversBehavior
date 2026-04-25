from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clustering  # noqa: E402


class _DummyProgress:
    def update(self, *_args, **_kwargs) -> None:
        return None

    def set_description(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def _make_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plate": ["A", "B", "C", "D"],
            "feature_1": [0.0, 1.0, 2.0, 3.0],
            "feature_2": [1.0, 1.5, 2.0, 2.5],
        }
    )


def _patch_common_flow(monkeypatch, tmp_path: Path, method: str) -> dict:
    features_df = _make_features_df()
    captured: dict = {}

    monkeypatch.setattr(clustering, "list_cluster_feature_db_paths", lambda: [])
    monkeypatch.setattr(
        clustering,
        "Clusterization",
        lambda *args, **kwargs: features_df.copy(),
    )
    monkeypatch.setattr(clustering, "_prompt_cluster_feature_db_suffix", lambda: "test")
    monkeypatch.setattr(
        clustering,
        "_build_cluster_feature_db_path",
        lambda suffix: tmp_path / f"{suffix}.duckdb",
    )
    monkeypatch.setattr(
        clustering,
        "save_cluster_features_duckdb",
        lambda df, db_path=None, **kwargs: db_path or tmp_path / "cluster_features.duckdb",
    )
    monkeypatch.setattr(
        clustering,
        "_prompt_feature_selection",
        lambda df: ["feature_1", "feature_2"],
    )
    monkeypatch.setattr(
        clustering,
        "_prepare_cluster_features",
        lambda df, cols: df[["plate", *cols]].copy(),
    )
    monkeypatch.setattr(clustering, "_prompt_cluster_method", lambda: method)
    monkeypatch.setattr(clustering, "_maybe_export_cluster_inputs", lambda *args: None)
    monkeypatch.setattr(clustering, "_prompt_int_value", lambda *args, **kwargs: 2)
    monkeypatch.setattr(clustering, "tqdm", lambda *args, **kwargs: _DummyProgress())

    def _save_cluster_labels(df, method_name, k=None):
        captured["clustered"] = df.copy()
        captured["method"] = method_name
        captured["k"] = k
        return tmp_path / f"{method_name}_labels.csv"

    monkeypatch.setattr(clustering, "save_cluster_labels", _save_cluster_labels)
    monkeypatch.setattr(
        clustering,
        "save_cluster_summary",
        lambda df, method_name, k=None: tmp_path / f"{method_name}_summary.csv",
    )
    monkeypatch.setattr(
        clustering,
        "save_cluster_descriptive",
        lambda df, method_name, k=None: tmp_path / f"{method_name}_descriptive.csv",
    )
    return captured


def _patch_inputs(monkeypatch, answers: list[str]) -> None:
    answers_iter = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers_iter))


def _patch_fake_sklearn_cluster(monkeypatch, expected_scaled: np.ndarray, labels: np.ndarray) -> None:
    fake_cluster = types.ModuleType("sklearn.cluster")

    class FakeKMeans:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def fit_predict(self, x):
            np.testing.assert_array_equal(x, expected_scaled)
            return labels

    fake_cluster.KMeans = FakeKMeans
    fake_sklearn = types.ModuleType("sklearn")
    fake_sklearn.__path__ = []
    fake_sklearn.cluster = fake_cluster
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_cluster)


def _patch_fake_sklearn_mixture(monkeypatch, expected_scaled: np.ndarray, labels: np.ndarray) -> None:
    fake_mixture = types.ModuleType("sklearn.mixture")

    class FakeGaussianMixture:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def fit_predict(self, x):
            np.testing.assert_array_equal(x, expected_scaled)
            return labels

    fake_mixture.GaussianMixture = FakeGaussianMixture
    fake_sklearn = types.ModuleType("sklearn")
    fake_sklearn.__path__ = []
    fake_sklearn.mixture = fake_mixture
    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.mixture", fake_mixture)


def test_handle_clusterization_kmeans_without_metrics_unpacks_scaled_features(
    monkeypatch, tmp_path
):
    captured = _patch_common_flow(monkeypatch, tmp_path, method="kmeans")
    expected_scaled = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        dtype=float,
    )
    monkeypatch.setattr(
        clustering,
        "_scale_cluster_features",
        lambda df, cols: (expected_scaled, object()),
    )
    _patch_fake_sklearn_cluster(
        monkeypatch,
        expected_scaled=expected_scaled,
        labels=np.array([0, 0, 1, 1]),
    )
    _patch_inputs(monkeypatch, ["n", "n", "s"])

    session = types.SimpleNamespace(flujos_df=pd.DataFrame({"dummy": [1]}))
    clustering.handle_clusterization(session)

    assert captured["method"] == "kmeans"
    assert captured["k"] == 2
    assert captured["clustered"]["cluster_label"].tolist() == [0, 0, 1, 1]


def test_handle_clusterization_gmm_without_metrics_unpacks_scaled_features(
    monkeypatch, tmp_path
):
    captured = _patch_common_flow(monkeypatch, tmp_path, method="gmm")
    expected_scaled = np.array(
        [[0.5, 0.0], [1.5, 1.0], [2.5, 2.0], [3.5, 3.0]],
        dtype=float,
    )
    monkeypatch.setattr(
        clustering,
        "_scale_cluster_features",
        lambda df, cols: (expected_scaled, object()),
    )
    _patch_fake_sklearn_mixture(
        monkeypatch,
        expected_scaled=expected_scaled,
        labels=np.array([1, 1, 0, 0]),
    )
    _patch_inputs(monkeypatch, ["n", "n", "s"])

    session = types.SimpleNamespace(flujos_df=pd.DataFrame({"dummy": [1]}))
    clustering.handle_clusterization(session)

    assert captured["method"] == "gmm"
    assert captured["k"] == 2
    assert captured["clustered"]["cluster_label"].tolist() == [1, 1, 0, 0]


def test_handle_clusterization_hdbscan_unpacks_scaled_features(monkeypatch, tmp_path):
    captured = _patch_common_flow(monkeypatch, tmp_path, method="hdbscan")
    expected_scaled = np.array(
        [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]],
        dtype=float,
    )
    monkeypatch.setattr(
        clustering,
        "_scale_cluster_features",
        lambda df, cols: (expected_scaled, object()),
    )

    fake_hdbscan = types.ModuleType("hdbscan")

    class FakeHDBSCAN:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def fit_predict(self, x):
            np.testing.assert_array_equal(x, expected_scaled)
            return np.array([0, -1, 1, 1])

    fake_hdbscan.HDBSCAN = FakeHDBSCAN
    monkeypatch.setitem(sys.modules, "hdbscan", fake_hdbscan)
    _patch_inputs(monkeypatch, ["n", "n"])

    session = types.SimpleNamespace(flujos_df=pd.DataFrame({"dummy": [1]}))
    clustering.handle_clusterization(session)

    assert captured["method"] == "hdbscan"
    assert captured["k"] is None
    assert captured["clustered"]["cluster_label"].tolist() == [0, -1, 1, 1]
