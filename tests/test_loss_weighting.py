"""
Tests para el weighting de pérdida basado en la distancia accidente→pórtico
aguas arriba. Cubren:

- _compute_pm_loss_weight: pesos correctos para positivos y negativos, clip,
  floor, NaN→1.0, max-aggregation cuando un nodo tiene múltiples accidentes.
- train_pretrain._apply_distance_weighting: cuando el batch tiene loss_weight,
  produce una loss distinta a la criterion estándar; cuando no, cae al
  criterion estándar.
- Backward compatibility: con loss_weight_mode="uniform", la loss equivale
  exactamente al criterion estándar.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import (
    _compute_pm_loss_weight,
    _LOSS_WEIGHT_D_MAX_KM,
    _LOSS_WEIGHT_FLOOR,
)
from src.train_pretrain import _apply_distance_weighting


# ---------------------------------------------------------------------------
# _compute_pm_loss_weight
# ---------------------------------------------------------------------------


def test_no_affected_returns_all_ones():
    w = _compute_pm_loss_weight(10, pd.DataFrame())
    assert w.shape == (10,)
    assert torch.all(w == 1.0)


def test_positives_get_distance_weight():
    affected = pd.DataFrame(
        [
            {"node_idx": 0, "dist_to_post_km": 0.0},   # → peso 1.0
            {"node_idx": 1, "dist_to_post_km": 2.5},   # → 1 - 2.5/5 = 0.5
            {"node_idx": 2, "dist_to_post_km": 4.0},   # → 1 - 4/5   = 0.2 (floor)
            {"node_idx": 3, "dist_to_post_km": 10.0},  # → clip a floor 0.2
        ]
    )
    w = _compute_pm_loss_weight(5, affected)
    assert w[0].item() == pytest.approx(1.0)
    assert w[1].item() == pytest.approx(0.5)
    assert w[2].item() == pytest.approx(_LOSS_WEIGHT_FLOOR)
    assert w[3].item() == pytest.approx(_LOSS_WEIGHT_FLOOR)
    # Negativo (no afectado): peso 1.0.
    assert w[4].item() == pytest.approx(1.0)


def test_nan_distance_does_not_penalise_node():
    affected = pd.DataFrame([{"node_idx": 0, "dist_to_post_km": np.nan}])
    w = _compute_pm_loss_weight(2, affected)
    # Sin distancia conocida: no castigamos al accidente → peso 1.0.
    assert w[0].item() == pytest.approx(1.0)
    assert w[1].item() == pytest.approx(1.0)


def test_multiple_accidents_same_node_take_max_weight():
    """Si varios accidentes mapean al mismo nodo, el más cercano (peso más alto) gana."""
    affected = pd.DataFrame(
        [
            {"node_idx": 0, "dist_to_post_km": 4.0},   # peso 0.2 (con floor)
            {"node_idx": 0, "dist_to_post_km": 0.5},   # peso 0.9
            {"node_idx": 0, "dist_to_post_km": 3.0},   # peso 0.4
        ]
    )
    w = _compute_pm_loss_weight(1, affected)
    assert w[0].item() == pytest.approx(0.9)


def test_missing_dist_column_returns_all_ones():
    """Sin la columna dist_to_post_km (entrada vieja sin PR 3): no-op."""
    affected = pd.DataFrame([{"node_idx": 0}])
    w = _compute_pm_loss_weight(2, affected)
    assert torch.all(w == 1.0)


def test_d_max_constant_value():
    """Sanity: el constante D_MAX es 5 km (consistente con el outlier threshold)."""
    assert _LOSS_WEIGHT_D_MAX_KM == 5.0
    assert _LOSS_WEIGHT_FLOOR == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _apply_distance_weighting
# ---------------------------------------------------------------------------


class _FakeStore:
    """Simula batch[node_type] con atributo loss_weight."""

    def __init__(self, loss_weight=None):
        if loss_weight is not None:
            self.loss_weight = loss_weight


def test_distance_weighting_falls_back_when_no_attr():
    crit = nn.CrossEntropyLoss()
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.7]])
    y = torch.tensor([0, 1])
    store = _FakeStore(loss_weight=None)  # sin atributo
    out = _apply_distance_weighting(crit, logits, y, store, batch_size=2)
    expected = crit(logits, y)
    assert torch.allclose(out, expected)


def test_distance_weighting_matches_manual_calculation():
    """Con pesos conocidos, la salida coincide con la fórmula manual."""
    crit = nn.CrossEntropyLoss()
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.7], [0.0, 0.0]])
    y = torch.tensor([0, 1, 1])
    w = torch.tensor([1.0, 0.5, 1.0])
    store = _FakeStore(loss_weight=w)
    out = _apply_distance_weighting(crit, logits, y, store, batch_size=3)

    per_sample = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    expected = (per_sample * w).sum() / w.sum()
    assert torch.allclose(out, expected)


def test_distance_weighting_constant_scale_equals_mean():
    """Pesos uniformes ≠ 1.0 producen la misma loss que el promedio (homogénea)."""
    crit = nn.CrossEntropyLoss()
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.7], [0.4, -0.3]])
    y = torch.tensor([0, 1, 0])
    w_half = torch.full((3,), 0.5)
    store = _FakeStore(loss_weight=w_half)
    out = _apply_distance_weighting(crit, logits, y, store, batch_size=3)
    expected = crit(logits, y)
    assert torch.allclose(out, expected, atol=1e-6)


def test_uniform_weights_match_criterion_exactly():
    """Backward compat: pesos todos en 1.0 → loss idéntica al criterion."""
    crit = nn.CrossEntropyLoss()
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.7], [0.4, -0.3]])
    y = torch.tensor([0, 1, 0])
    w_uniform = torch.ones(3)
    store = _FakeStore(loss_weight=w_uniform)
    out = _apply_distance_weighting(crit, logits, y, store, batch_size=3)
    expected = crit(logits, y)
    assert torch.allclose(out, expected, atol=1e-6)
