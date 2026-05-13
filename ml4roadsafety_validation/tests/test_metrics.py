from __future__ import annotations

import numpy as np

from ml4roadsafety_validation.metrics import (
    classification_metrics,
    safe_auprc,
    safe_auroc,
    select_threshold_by_fbeta,
)


def test_metrics_handle_single_class_split():
    y_true = np.zeros(5, dtype=int)
    y_prob = np.linspace(0.1, 0.5, 5)

    assert safe_auroc(y_true, y_prob) is None
    assert safe_auprc(y_true, y_prob) == 0.0
    threshold = select_threshold_by_fbeta(y_true, y_prob)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)

    assert metrics["n"] == 5
    assert metrics["positives"] == 0
    assert metrics["auroc"] is None
    assert metrics["tp"] == 0


def test_threshold_prefers_high_fbeta_on_validation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    threshold = select_threshold_by_fbeta(y_true, y_prob, beta=0.5)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)

    assert 0.2 <= threshold <= 0.9
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0

