from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.clustering import Clusterization, DEFAULT_CLUSTER_FEATURES
from src.utils import (
    FlowColumns,
    add_accident_target,
    compute_cluster_features,
    compute_flow_features,
    process_accidentes_df,
    read_csv_with_progress,
)


def _make_flow_df() -> pd.DataFrame:
    np.random.seed(0)
    start = pd.Timestamp("2024-01-01 08:00:00")
    times = [start + pd.Timedelta(minutes=5 * i) for i in range(12)]
    plates_a = ["AAAA1", "AAAA2", "AAAA3"]
    plates_b = ["BBBB1", "BBBB2", "BBBB3"]
    rows = []
    for idx, ts in enumerate(times):
        for plate in plates_a:
            rows.append(
                {
                    "FECHA": ts,
                    "VELOCIDAD": 80 + np.random.normal(0, 3),
                    "PORTICO": "1",
                    "CARRIL": 1 if idx % 2 == 0 else 2,
                    "MATRICULA": plate,
                    "CATEGORIA": 1,
                }
            )
        for plate in plates_b:
            rows.append(
                {
                    "FECHA": ts,
                    "VELOCIDAD": 50 + np.random.normal(0, 3),
                    "PORTICO": "1",
                    "CARRIL": 2 if idx % 2 == 0 else 3,
                    "MATRICULA": plate,
                    "CATEGORIA": 2,
                }
            )
    return pd.DataFrame(rows)


def _make_porticos_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portico": ["1", "2"],
            "km": [0.0, 1.0],
            "calzada": ["Oriente", "Oriente"],
            "orden": [1, 2],
            "eje": ["N", "N"],
        }
    )


def _make_accidents_df() -> pd.DataFrame:
    times = ["08:16:00", "08:36:00", "08:46:00", "08:56:00"]
    rows = []
    for t in times:
        rows.append(
            {
                "Tipo": "Accidente",
                "Via": "expresa",
                "Calzada": "Oriente",
                "SubTipo": "AC-01 - Test",
                "Fechas Inicio": "01/01/2024",
                "Hora Inicio": t,
                "Fecha Fin": "01/01/2024",
                "Hora Fin": "08:59:00",
                "Km.": "0.2",
                "Eje": "N",
            }
        )
    return pd.DataFrame(rows)


def _get_cluster_cols(df: pd.DataFrame) -> List[str]:
    cluster_prefixes = (
        "cluster_share_",
        "cluster_flow_",
        "cluster_count_",
        "cluster_speed_",
        "cluster_density_",
        "cluster_delta_speed_",
        "cluster_delta_density_",
        "cluster_entropy",
    )
    return [col for col in df.columns if col.startswith(cluster_prefixes)]


def build_synthetic_base_df(
    tmp_path: Path, *, interval_minutes: int = 5
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    flow_df = _make_flow_df()
    flow_path = tmp_path / "flows.csv"
    flow_df.to_csv(flow_path, index=False)
    flow_loaded = read_csv_with_progress(str(flow_path))

    porticos_df = _make_porticos_df()
    accidents_df = _make_accidents_df()
    accidents_path = tmp_path / "accidents.csv"
    accidents_df.to_csv(accidents_path, index=False)
    accidents_loaded = read_csv_with_progress(str(accidents_path))
    accidents_processed, _excluded = process_accidentes_df(
        accidents_loaded, porticos_df, return_excluded=True
    )

    flow_cols = FlowColumns(
        timestamp="FECHA",
        speed_kmh="VELOCIDAD",
        portico="PORTICO",
        lane="CARRIL",
        plate_id="MATRICULA",
    )
    cluster_features = Clusterization(
        flow_loaded,
        flow_cols,
        include_counts=True,
        max_headway_s=600.0,
        outlier_action="none",
    )

    feature_cols = [
        col for col in DEFAULT_CLUSTER_FEATURES if col in cluster_features.columns
    ]
    if len(feature_cols) < 2:
        raise AssertionError("Not enough clustering features for KMeans.")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = cluster_features[feature_cols].fillna(0).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_scaled)
    cluster_labels = cluster_features[["plate"]].copy()
    cluster_labels["cluster_label"] = labels

    flow_features = compute_flow_features(
        flow_loaded,
        interval_minutes=interval_minutes,
        lanes=2,
        metrics=["flow", "speed", "density"],
        categories=["Light", "Heavy"],
    )
    cluster_features_agg = compute_cluster_features(
        flow_loaded,
        cluster_labels,
        interval_minutes=interval_minutes,
        include_counts=True,
        include_speed=True,
    )
    merged = flow_features.merge(
        cluster_features_agg, on=["portico", "interval_start"], how="left"
    )
    for col in cluster_features_agg.columns:
        if col in {"portico", "interval_start"}:
            continue
        merged[col] = merged[col].fillna(0)

    base_df = add_accident_target(
        merged, accidents_processed, interval_minutes=interval_minutes
    )
    if base_df.empty or base_df["target"].nunique() < 2:
        raise AssertionError("Base dataset is empty or has a single class.")

    numeric_cols = [
        col
        for col in base_df.columns
        if col != "target" and pd.api.types.is_numeric_dtype(base_df[col])
    ]
    cluster_cols = _get_cluster_cols(base_df)
    base_cols = [col for col in numeric_cols if col not in cluster_cols]
    return base_df, numeric_cols, base_cols, cluster_cols
