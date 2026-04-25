from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("optuna")

from src.cluster_accident_app import _load_cluster_labels


def test_load_cluster_labels_preserves_gmm_membership_probabilities(tmp_path):
    path = tmp_path / "cluster_gmm_k2.csv"
    pd.DataFrame(
        {
            "plate": ["A1", "A2"],
            "cluster_label": [0, -1],
            "cluster_prob_1": [0.25, 0.55],
            "cluster_prob_0": [0.75, 0.45],
            "avg_speed_kmh": [80.0, 100.0],
        }
    ).to_csv(path, index=False)

    loaded = _load_cluster_labels(path)

    assert list(loaded.columns) == [
        "plate",
        "cluster_label",
        "cluster_prob_0",
        "cluster_prob_1",
    ]
    assert loaded.loc[1, "cluster_label"] == -1
    assert loaded.loc[1, ["cluster_prob_0", "cluster_prob_1"]].sum() == 1.0


def test_load_cluster_labels_accepts_soft_file_without_hard_label(tmp_path):
    path = tmp_path / "cluster_gmm_k2.csv"
    pd.DataFrame(
        {
            "plate": ["A1"],
            "cluster_prob_0": [0.35],
            "cluster_prob_1": [0.65],
        }
    ).to_csv(path, index=False)

    loaded = _load_cluster_labels(path)

    assert list(loaded.columns) == ["plate", "cluster_prob_0", "cluster_prob_1"]
