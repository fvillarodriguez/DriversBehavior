from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

BIAS_VARIANCE_NOISE_COLUMNS = ("bias2", "variance", "noise")
BALANCE_MODE_NOT_APPLICABLE = "not_applicable"


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numeric):
        return float("nan")
    return numeric


def is_missing_numeric(value: object) -> bool:
    return math.isnan(_safe_float(value))


def sequence_to_numeric(values: object) -> list[float]:
    if values is None or isinstance(values, (str, bytes)):
        return []
    try:
        raw_values = list(values)
    except TypeError:
        return []
    return [_safe_float(value) for value in raw_values]


def probability_scores_for_brier(roc_item: Mapping[str, Any], *, expected_length: int) -> list[float]:
    calibrated_scores = sequence_to_numeric(roc_item.get("calibrated_scores"))
    if len(calibrated_scores) == expected_length and any(math.isfinite(v) for v in calibrated_scores):
        return calibrated_scores
    return sequence_to_numeric(roc_item.get("scores"))


def _normalize_segment_token(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return ""
        return text
    numeric = _safe_float(value)
    if math.isfinite(numeric):
        if abs(numeric - round(numeric)) < 1e-9:
            return str(int(round(numeric)))
        return str(numeric)
    return ""


def drift_roc_group_key(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("strategy") or ""),
        str(item.get("model") or ""),
        str(item.get("balance_mode") or BALANCE_MODE_NOT_APPLICABLE),
        _normalize_segment_token(item.get("segment")),
    )


def drift_row_group_key(row: Mapping[str, Any], *, yearly: bool) -> tuple[str, str, str, str]:
    strategy = str(row.get("strategy") or "")
    model = str(row.get("model") or "")
    balance_mode = str(row.get("balance_mode") or BALANCE_MODE_NOT_APPLICABLE)
    if yearly:
        segment = _normalize_segment_token(row.get("prediction_year"))
    else:
        if strategy == "adaptive_arf":
            segment = _normalize_segment_token(row.get("prediction_year"))
        else:
            drift_token = _normalize_segment_token(row.get("drift"))
            if drift_token and not drift_token.startswith("drift_"):
                segment = f"drift_{drift_token}"
            else:
                segment = drift_token or _normalize_segment_token(row.get("segment"))
    return (strategy, model, balance_mode, segment)


def _rounded_binary(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(max(0, min(1, int(round(value)))))


def compute_bias_variance_noise_from_roc_items(
    roc_items: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    reference_y: list[float] | None = None
    probability_runs: list[list[float]] = []

    for item in roc_items:
        y_true = [_rounded_binary(v) for v in sequence_to_numeric(item.get("y_true"))]
        if not y_true:
            continue
        probability_scores = probability_scores_for_brier(item, expected_length=len(y_true))
        if len(probability_scores) != len(y_true):
            continue
        probability_scores = [
            float("nan") if not math.isfinite(v) else min(max(float(v), 0.0), 1.0)
            for v in probability_scores
        ]
        if reference_y is None:
            reference_y = y_true
        else:
            if len(reference_y) != len(y_true):
                continue
            mismatch = any(
                math.isfinite(ref_val)
                and math.isfinite(curr_val)
                and abs(ref_val - curr_val) > 1e-9
                for ref_val, curr_val in zip(reference_y, y_true)
            )
            if mismatch:
                continue
        probability_runs.append(probability_scores)

    if reference_y is None or not probability_runs:
        return {col: float("nan") for col in BIAS_VARIANCE_NOISE_COLUMNS}

    bias2_total = 0.0
    variance_total = 0.0
    noise_total = 0.0
    n_obs = 0

    for idx, y_val in enumerate(reference_y):
        if not math.isfinite(y_val):
            continue
        obs_scores = [run[idx] for run in probability_runs if idx < len(run) and math.isfinite(run[idx])]
        if not obs_scores:
            continue
        mean_prob = min(max(sum(obs_scores) / float(len(obs_scores)), 0.0), 1.0)
        bias2_total += (mean_prob - y_val) ** 2
        variance_total += mean_prob * (1.0 - mean_prob)
        noise_total += y_val * (1.0 - y_val)
        n_obs += 1

    if n_obs <= 0:
        return {col: float("nan") for col in BIAS_VARIANCE_NOISE_COLUMNS}

    return {
        "bias2": float(bias2_total / float(n_obs)),
        "variance": float(variance_total / float(n_obs)),
        "noise": float(noise_total / float(n_obs)),
    }


def build_bias_variance_noise_lookup(
    roc_payload: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    grouped_items: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in roc_payload:
        if isinstance(item, Mapping):
            grouped_items[drift_roc_group_key(item)].append(item)
    return {
        key: compute_bias_variance_noise_from_roc_items(items)
        for key, items in grouped_items.items()
    }


def enrich_drift_rows_with_bias_variance(
    rows: Sequence[Mapping[str, Any]],
    roc_payload: Sequence[Mapping[str, Any]],
    *,
    yearly: bool,
) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not out or not roc_payload:
        return out

    decomposition_lookup = build_bias_variance_noise_lookup(roc_payload)
    for row in out:
        metrics = decomposition_lookup.get(drift_row_group_key(row, yearly=yearly))
        if not isinstance(metrics, dict):
            continue
        for col in BIAS_VARIANCE_NOISE_COLUMNS:
            if is_missing_numeric(row.get(col)):
                row[col] = metrics.get(col)
    return out
