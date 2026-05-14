from __future__ import annotations

import numpy as np

from ml4roadsafety_validation.metrics import (
    classification_metrics,
    safe_auprc,
    safe_auroc,
    select_threshold_by_top_k,
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
    assert metrics["predicted_positives"] == 1
    assert metrics["auroc"] is None
    assert metrics["tp"] == 0
    assert metrics["score_max"] == 0.5


def test_threshold_prefers_high_fbeta_on_validation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    threshold = select_threshold_by_fbeta(y_true, y_prob, beta=0.5)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)

    assert 0.2 <= threshold <= 0.9
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_top_k_threshold_and_score_diagnostics_expose_zero_alert_risk():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.95, 0.7, 0.4, 0.3, 0.1])

    threshold = select_threshold_by_top_k(y_prob, k=2)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)

    assert threshold == 0.7
    assert metrics["predicted_positives"] == 2
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["positive_score_max"] == 0.7
