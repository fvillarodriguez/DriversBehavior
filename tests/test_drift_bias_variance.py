import pytest

from src.drift_bias_variance import (
    compute_bias_variance_noise_from_roc_items,
    compute_brier_score_from_roc_items,
    enrich_drift_rows_with_bias_variance,
)


def test_bias_variance_noise_decomposition_is_additive_for_repeated_predictions():
    roc_items = [
        {
            "strategy": "static",
            "model": "XGBoost",
            "balance_mode": "none",
            "segment": "2019",
            "y_true": [0, 1],
            "calibrated_scores": [0.2, 0.8],
        },
        {
            "strategy": "static",
            "model": "XGBoost",
            "balance_mode": "none",
            "segment": "2019",
            "y_true": [0, 1],
            "calibrated_scores": [0.4, 0.6],
        },
    ]

    decomposition = compute_bias_variance_noise_from_roc_items(roc_items)
    brier_score = compute_brier_score_from_roc_items(roc_items)

    assert decomposition["bias2"] == pytest.approx(0.09)
    assert decomposition["variance"] == pytest.approx(0.01)
    assert decomposition["noise"] == pytest.approx(0.0)
    assert brier_score == pytest.approx(
        decomposition["bias2"] + decomposition["variance"] + decomposition["noise"]
    )


def test_bias_variance_noise_decomposition_has_zero_variance_for_single_prediction_run():
    roc_items = [
        {
            "strategy": "static",
            "model": "XGBoost",
            "balance_mode": "none",
            "segment": "2019",
            "y_true": [0, 1],
            "calibrated_scores": [0.2, 0.8],
        }
    ]

    decomposition = compute_bias_variance_noise_from_roc_items(roc_items)
    brier_score = compute_brier_score_from_roc_items(roc_items)

    assert decomposition["variance"] == pytest.approx(0.0)
    assert brier_score == pytest.approx(
        decomposition["bias2"] + decomposition["variance"] + decomposition["noise"]
    )


def test_enrich_drift_rows_can_overwrite_stale_decomposition_values():
    rows = [
        {
            "strategy": "static",
            "model": "XGBoost",
            "balance_mode": "none",
            "prediction_year": 2019,
            "brier_score": 999.0,
            "bias2": 999.0,
            "variance": 999.0,
            "noise": 999.0,
        }
    ]
    roc_payload = [
        {
            "strategy": "static",
            "model": "XGBoost",
            "balance_mode": "none",
            "segment": "2019",
            "y_true": [0, 1],
            "calibrated_scores": [0.2, 0.8],
        }
    ]

    enriched = enrich_drift_rows_with_bias_variance(
        rows,
        roc_payload,
        yearly=True,
        overwrite_existing=True,
    )

    assert enriched[0]["brier_score"] == pytest.approx(0.04)
    assert enriched[0]["bias2"] == pytest.approx(0.04)
    assert enriched[0]["variance"] == pytest.approx(0.0)
    assert enriched[0]["noise"] == pytest.approx(0.0)
