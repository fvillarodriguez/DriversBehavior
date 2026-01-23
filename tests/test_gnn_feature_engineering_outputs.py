import pandas as pd
import polars as pl

from src.features import compute_pm_features
from src.snapshot_pipeline import (
    SnapshotConfig,
    SnapshotFeatureBuilder,
    flatten_snapshot_features,
)
from src.utils import compute_cluster_features


def _make_flow_df() -> pd.DataFrame:
    times = pd.date_range("2024-01-01 08:00:00", periods=4, freq="5min")
    rows = []
    for ts in times:
        rows.extend(
            [
                {
                    "FECHA": ts,
                    "VELOCIDAD": 80,
                    "PORTICO": 1,
                    "CATEGORIA": 1,
                    "CARRIL": 1,
                    "MATRICULA": "A1",
                },
                {
                    "FECHA": ts,
                    "VELOCIDAD": 70,
                    "PORTICO": 1,
                    "CATEGORIA": 2,
                    "CARRIL": 2,
                    "MATRICULA": "A2",
                },
                {
                    "FECHA": ts,
                    "VELOCIDAD": 60,
                    "PORTICO": 2,
                    "CATEGORIA": 1,
                    "CARRIL": 1,
                    "MATRICULA": "B1",
                },
                {
                    "FECHA": ts,
                    "VELOCIDAD": 50,
                    "PORTICO": 2,
                    "CATEGORIA": 2,
                    "CARRIL": 2,
                    "MATRICULA": "B2",
                },
            ]
        )
    return pd.DataFrame(rows)


def _make_porticos_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PORTICO": [1, 2],
            "KM": [0.0, 1.0],
            "ORDEN": [1, 2],
            "EJE": ["N", "N"],
            "CALZADA": ["Oriente", "Oriente"],
        }
    )


def test_feature_engineering_outputs_have_positive_values() -> None:
    flows = _make_flow_df()

    flow_features = compute_pm_features(
        flows, interactive=False, dt_minutes=5
    )
    assert not flow_features.empty
    assert "flujo_total" in flow_features.columns
    assert flow_features["flujo_total"].gt(0).any()
    assert "flujo_1" in flow_features.columns
    assert "flujo_2" in flow_features.columns
    assert "velocidad_media_1" in flow_features.columns
    assert flow_features["velocidad_media_1"].gt(0).any()

    snap_config = SnapshotConfig(dt_minutes=5, window_minutes=10)
    snap_builder = SnapshotFeatureBuilder(snap_config)
    # Convert to Polars for the new pipeline
    flows_pl = pl.from_pandas(flows)
    snapshot_df, _ = snap_builder.build(flows_pl, _make_porticos_df())
    flat_snapshot = flatten_snapshot_features(snapshot_df)
    for col in ["speed_mean", "flow_total", "tod_sin", "tod_cos"]:
        assert col in flat_snapshot.columns
    assert flat_snapshot["speed_mean"].gt(0).any()
    assert flat_snapshot["flow_total"].gt(0).any()

    cluster_labels = pd.DataFrame(
        {
            "plate": ["A1", "A2", "B1", "B2"],
            "cluster_label": [0, 0, 1, 1],
        }
    )
    cluster_features = compute_cluster_features(
        flows,
        cluster_labels,
        interval_minutes=5,
        include_counts=True,
        include_speed=True,
        include_density=True,
    )
    assert not cluster_features.empty
    cluster_cols = [
        col
        for col in cluster_features.columns
        if col.startswith(
            (
                "cluster_share_",
                "cluster_flow_",
                "cluster_speed_",
                "cluster_density_",
            )
        )
    ]
    assert cluster_cols
    assert (cluster_features[cluster_cols].to_numpy() > 0).any()
