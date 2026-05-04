import pandas as pd
import pytest

from src.graph_builder_app import _resolve_portico_interval_by_endpoints


def _porticos_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portico": ["P1", "P2", "P3", "P4", "Q1", "Q2"],
            "autopista": ["C", "C", "C", "C", "C", "C"],
            "calzada": ["Norte", "Norte", "Norte", "Norte", "Sur", "Sur"],
            "eje": ["E1", "E1", "E1", "E1", "E2", "E2"],
            "orden": [1, 2, 3, 4, 1, 2],
            "km": [10.0, 11.0, 12.0, 13.0, 20.0, 21.0],
        }
    )


def test_resolve_portico_interval_includes_all_between_endpoints() -> None:
    selected = _resolve_portico_interval_by_endpoints(
        _porticos_df(), "P2", "P4"
    )

    assert selected == ["P2", "P3", "P4"]


def test_resolve_portico_interval_accepts_reverse_order() -> None:
    selected = _resolve_portico_interval_by_endpoints(
        _porticos_df(), "p4", "p2"
    )

    assert selected == ["P2", "P3", "P4"]


def test_resolve_portico_interval_normalizes_numeric_inputs() -> None:
    df = pd.DataFrame(
        {
            "PORTICO": [1, 2, 3, 4],
            "CALZADA": ["A", "A", "A", "A"],
            "EJE": ["E", "E", "E", "E"],
            "ORDEN": [1, 2, 3, 4],
        }
    )

    selected = _resolve_portico_interval_by_endpoints(df, "2.0", "4")

    assert selected == ["2", "3", "4"]


def test_resolve_portico_interval_rejects_different_sequences() -> None:
    with pytest.raises(ValueError, match="no comparten una secuencia"):
        _resolve_portico_interval_by_endpoints(_porticos_df(), "P2", "Q2")
