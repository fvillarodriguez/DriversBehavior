import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SnapshotConfig:
    """Configuration for snapshot feature generation."""

    dt_minutes: int = 5
    window_minutes: int = 30
    slow_speed_kmh: float = 30.0
    speed_entropy_bins: Sequence[float] = tuple(np.linspace(0, 140, 15))
    add_reverse_edges: bool = True
    add_self_loops: bool = True

    @property
    def window_steps(self) -> int:
        steps = int(round(self.window_minutes / float(self.dt_minutes)))
        return max(1, steps)


@dataclass
class GraphTopology:
    """Static graph topology wrapper."""

    node_order: List[int]
    node_to_idx: Dict[int, int]
    upstream: Dict[int, Optional[int]]
    downstream: Dict[int, Optional[int]]
    edge_index: np.ndarray


def _resolve_column(
    df: pd.DataFrame,
    expected: str,
    optional: bool = False,
) -> Optional[str]:
    """Best-effort column resolver that avoids interactive prompts."""
    if expected in df.columns:
        return expected
    normalized = {
        col.strip().lower().replace(" ", ""): col for col in df.columns
    }
    key = expected.strip().lower().replace(" ", "")
    if key in normalized:
        return normalized[key]
    for col in df.columns:
        if col.lower() == expected.lower():
            return col
    if optional:
        return None
    raise KeyError(f"No se encontró la columna esperada '{expected}'.")


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if not np.issubdtype(series.dtype, np.datetime64):
        series = pd.to_datetime(series, errors="coerce", utc=True)
        try:
            series = series.dt.tz_convert(None)  # type: ignore[attr-defined]
        except Exception:
            series = series.dt.tz_localize(None)  # type: ignore[attr-defined]
    return series


def discretize_vehicle_flows(
    flujos_df: pd.DataFrame,
    config: SnapshotConfig,
) -> pd.DataFrame:
    """
    Aggregate raw per-vehicle data into Δ-minute blocks per pórtico.

    Returns:
        DataFrame indexed by (portico, snapshot_time) with block-level metrics.
    """
    if flujos_df.empty:
        raise ValueError("El DataFrame de flujos está vacío.")

    fecha_col = _resolve_column(flujos_df, "FECHA")
    velocidad_col = _resolve_column(flujos_df, "VELOCIDAD")
    portico_col = _resolve_column(flujos_df, "PORTICO")
    categoria_col = _resolve_column(flujos_df, "CATEGORIA", optional=True)
    carril_col = _resolve_column(flujos_df, "CARRIL", optional=True)

    work = flujos_df[[fecha_col, velocidad_col, portico_col]].copy()
    if categoria_col:
        work[categoria_col] = flujos_df[categoria_col]
    if carril_col:
        work[carril_col] = flujos_df[carril_col]

    work = work.dropna(subset=[fecha_col, velocidad_col, portico_col])
    if work.empty:
        raise ValueError("No quedan registros válidos después de limpiar nulos.")

    work[fecha_col] = _ensure_datetime(work[fecha_col])
    work = work.dropna(subset=[fecha_col])

    block_label = "snapshot_time"
    work[block_label] = work[fecha_col].dt.floor(f"{config.dt_minutes}min")

    group_keys = [portico_col, block_label]
    speed = work.groupby(group_keys)[velocidad_col]

    base = speed.agg(
        speed_mean="mean",
        speed_std="std",
        speed_var="var",
        speed_median="median",
        speed_min="min",
        speed_max="max",
    )
    base["speed_q25"] = speed.quantile(0.25)
    base["speed_q75"] = speed.quantile(0.75)
    base["speed_q10"] = speed.quantile(0.10)
    base["speed_q90"] = speed.quantile(0.90)
    base["speed_iqr"] = base["speed_q75"] - base["speed_q25"]

    def _harmonic(x: pd.Series) -> float:
        arr = x.to_numpy(dtype=float)
        if arr.size == 0:
            return 0.0
        denom = np.sum(1.0 / np.clip(arr, 1e-3, None))
        return float(arr.size / denom) if denom > 0 else 0.0

    base["speed_harmonic"] = speed.apply(_harmonic)

    def _entropy(series: pd.Series) -> float:
        arr = series.to_numpy(dtype=float)
        if arr.size == 0:
            return 0.0
        counts, _ = np.histogram(arr, bins=config.speed_entropy_bins)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts.astype(float) / float(total)
        mask = probs > 0
        return float(-(probs[mask] * np.log(probs[mask])).sum())

    base["speed_entropy"] = speed.apply(_entropy)

    counts = work.groupby(group_keys).size().rename("flow_total").astype(float)

    work["is_slow"] = work[velocidad_col] <= config.slow_speed_kmh
    slow_pct = work.groupby(group_keys)["is_slow"].mean().rename("slow_pct")

    cat_features = []
    mix_features = []
    if categoria_col:
        cat_counts = (
            work.groupby(group_keys + [categoria_col]).size().unstack(fill_value=0)
        )
        for cat in cat_counts.columns:
            col_name = f"flow_cat_{cat}"
            cat_features.append(col_name)
            base[col_name] = cat_counts[cat].astype(float)
        total = cat_counts.sum(axis=1).replace(0, np.nan)
        for cat in cat_counts.columns:
            mix = cat_counts[cat] / total
            mix_name = f"mix_cat_{cat}"
            mix_features.append(mix_name)
            base[mix_name] = mix.fillna(0.0).astype(float)
    else:
        base["flow_cat_unknown"] = 0.0
        base["mix_cat_unknown"] = 0.0

    if carril_col:
        lane_speed = work.groupby(group_keys + [carril_col])[velocidad_col].mean()
        lane_flow = work.groupby(group_keys + [carril_col])[velocidad_col].count()
        lane_speed_std = (
            lane_speed.groupby(level=group_keys).std().rename("lane_speed_mean_std")
        )
        lane_flow_std = (
            lane_flow.groupby(level=group_keys).std().rename("lane_flow_std")
        )
        lane_count = (
            work.groupby(group_keys)[carril_col].nunique().rename("lane_count")
        )
    else:
        lane_speed_std = pd.Series(
            0.0, index=base.index, name="lane_speed_mean_std"
        )
        lane_flow_std = pd.Series(0.0, index=base.index, name="lane_flow_std")
        lane_count = pd.Series(0.0, index=base.index, name="lane_count")

    features = base.join(counts).join(slow_pct)
    features = features.join(lane_speed_std, how="left")
    features = features.join(lane_flow_std, how="left")
    features = features.join(lane_count, how="left")

    features["slow_pct"] = features["slow_pct"].fillna(0.0)
    for col in [
        "speed_std",
        "speed_var",
        "lane_speed_mean_std",
        "lane_flow_std",
    ]:
        features[col] = features[col].fillna(0.0)

    features["flow_total"] = features["flow_total"].fillna(0.0)
    for col in cat_features:
        features[col] = features[col].fillna(0.0)
    for col in mix_features:
        features[col] = features[col].fillna(0.0)

    features.index = features.index.set_names(["portico", "snapshot_time"])
    features.sort_index(inplace=True)
    return features


def _groupby_rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    func: str,
) -> pd.Series:
    grouped = frame.groupby(level="portico")[column]
    rolled = grouped.transform(lambda s: getattr(s.rolling(window, min_periods=1), func)())
    return rolled.astype(float)


def _groupby_shift(
    frame: pd.DataFrame,
    column: str,
    periods: int,
) -> pd.Series:
    grouped = frame.groupby(level="portico")[column]
    shifted = grouped.shift(periods)
    return shifted.astype(float)


def _rolling_slope(arr: np.ndarray) -> float:
    n = arr.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = arr.astype(float)
    xm = x.mean()
    ym = y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom <= 0:
        return 0.0
    num = np.sum((x - xm) * (y - ym))
    return float(num / denom)


def add_window_features(
    block_features: pd.DataFrame,
    config: SnapshotConfig,
    tracked_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Apply rolling window statistics, lags, and slopes over the block-level metrics.
    """
    if block_features.empty:
        raise ValueError("No hay features para procesar.")

    work = block_features.copy()
    window = config.window_steps

    if tracked_columns is None:
        tracked_columns = [
            "speed_mean",
            "speed_std",
            "speed_var",
            "speed_median",
            "speed_harmonic",
            "flow_total",
            "slow_pct",
        ]

    for col in tracked_columns:
        if col not in work.columns:
            continue
        work[f"{col}_w_mean"] = _groupby_rolling(work, col, window, "mean")
        work[f"{col}_w_std"] = _groupby_rolling(work, col, window, "std").fillna(0.0)
        work[f"{col}_lag1"] = _groupby_shift(work, col, 1).fillna(0.0)
        work[f"{col}_lag2"] = _groupby_shift(work, col, 2).fillna(0.0)
        slope = (
            work.groupby(level="portico")[col]
            .transform(
                lambda s: s.rolling(window, min_periods=2).apply(
                    _rolling_slope, raw=True
                )
            )
            .fillna(0.0)
        )
        work[f"{col}_slope"] = slope

    return work


def add_temporal_encodings(features: pd.DataFrame) -> pd.DataFrame:
    """Append sin/cos encodings for time-of-day and day-of-week."""
    work = features.copy()
    times = work.index.get_level_values("snapshot_time")
    minutes = times.hour * 60 + times.minute
    radians_day = 2 * math.pi * (minutes.astype(float) / (24 * 60))
    work["tod_sin"] = np.sin(radians_day)
    work["tod_cos"] = np.cos(radians_day)

    weekday = times.weekday.astype(float)
    radians_week = 2 * math.pi * (weekday / 7.0)
    work["dow_sin"] = np.sin(radians_week)
    work["dow_cos"] = np.cos(radians_week)
    return work


def build_static_topology(
    porticos_df: pd.DataFrame,
    config: SnapshotConfig,
) -> GraphTopology:
    """Create a fixed directed topology from portico metadata."""
    if porticos_df.empty:
        raise ValueError("El DataFrame de pórticos está vacío.")

    portico_col = _resolve_column(porticos_df, "PORTICO")
    orden_col = _resolve_column(porticos_df, "ORDEN", optional=True)
    eje_col = _resolve_column(porticos_df, "EJE", optional=True)
    calzada_col = _resolve_column(porticos_df, "CALZADA", optional=True)
    km_col = _resolve_column(porticos_df, "KM", optional=True)

    work = porticos_df.copy()
    work[portico_col] = work[portico_col].astype(int)
    if orden_col:
        work[orden_col] = pd.to_numeric(work[orden_col], errors="coerce")
    if km_col:
        work[km_col] = pd.to_numeric(work[km_col], errors="coerce")

    group_cols: List[str] = []
    if eje_col:
        group_cols.append(eje_col)
    if calzada_col:
        group_cols.append(calzada_col)

    sort_keys: List[str] = []
    sort_keys.extend(group_cols)
    if orden_col:
        sort_keys.append(orden_col)
    if km_col and km_col not in sort_keys:
        sort_keys.append(km_col)

    work = work.sort_values(sort_keys, kind="mergesort")

    node_order = work[portico_col].tolist()
    node_to_idx = {int(p): idx for idx, p in enumerate(node_order)}

    upstream: Dict[int, Optional[int]] = {int(p): None for p in node_order}
    downstream: Dict[int, Optional[int]] = {int(p): None for p in node_order}
    edges: List[Tuple[int, int]] = []

    grouped = (
        work.groupby(group_cols, sort=False)
        if group_cols
        else [(None, work)]
    )
    for _, group in grouped:
        ids = group[portico_col].tolist()
        for idx in range(len(ids) - 1):
            src = int(ids[idx])
            dst = int(ids[idx + 1])
            edges.append((src, dst))
            downstream[src] = dst
            upstream[dst] = src

    if config.add_reverse_edges:
        rev_edges = [(dst, src) for (src, dst) in edges]
        edges.extend(rev_edges)

    if config.add_self_loops:
        edges.extend([(node, node) for node in node_order])

    edge_pairs_idx = [
        (node_to_idx[src], node_to_idx[dst])
        for src, dst in edges
        if src in node_to_idx and dst in node_to_idx
    ]
    if not edge_pairs_idx:
        raise ValueError("No se pudieron construir aristas para la topología fija.")

    edge_index = np.asarray(edge_pairs_idx, dtype=np.int64).T
    return GraphTopology(
        node_order=node_order,
        node_to_idx=node_to_idx,
        upstream=upstream,
        downstream=downstream,
        edge_index=edge_index,
    )


def add_spatial_gradients(
    features: pd.DataFrame,
    topology: GraphTopology,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Add upstream/downstream gradients for selected feature columns."""
    if columns is None:
        columns = ("speed_mean", "speed_mean_w_mean")

    work = features.copy()
    swap = work.swaplevel().sort_index()

    for col in columns:
        if col not in swap.columns:
            continue
        wide = swap[col].unstack(fill_value=np.nan)
        grad_up = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
        grad_down = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)

        for portico in wide.columns:
            up = topology.upstream.get(portico)
            down = topology.downstream.get(portico)

            if up is None or up not in wide.columns:
                grad_up[portico] = 0.0
            else:
                grad_up[portico] = wide[portico] - wide[up]

            if down is None or down not in wide.columns:
                grad_down[portico] = 0.0
            else:
                grad_down[portico] = wide[portico] - wide[down]

        swap[f"{col}_grad_upstream"] = grad_up.fillna(0.0).stack(
            future_stack=True
        )
        swap[f"{col}_grad_downstream"] = grad_down.fillna(0.0).stack(
            future_stack=True
        )

    swap.sort_index(inplace=True)
    return swap.swaplevel().sort_index()


class SnapshotFeatureBuilder:
    """High-level helper to prepare snapshot-ready node features."""

    def __init__(self, config: Optional[SnapshotConfig] = None):
        self.config = config or SnapshotConfig()

    def build(
        self,
        flujos_df: pd.DataFrame,
        porticos_df: pd.DataFrame,
        tracked_columns: Optional[Iterable[str]] = None,
        *,
        gradient_columns: Optional[Sequence[str]] = None,
        include_window_features: bool = True,
        include_spatial_gradients: bool = True,
        include_temporal_encodings: bool = True,
    ) -> Tuple[pd.DataFrame, GraphTopology]:
        """Generate snapshot features and static topology."""
        block_features = discretize_vehicle_flows(flujos_df, self.config)
        if include_window_features:
            enriched = add_window_features(
                block_features,
                self.config,
                tracked_columns=tracked_columns,
            )
        else:
            enriched = block_features
        topology = build_static_topology(porticos_df, self.config)
        if include_spatial_gradients:
            with_gradients = add_spatial_gradients(
                enriched,
                topology,
                columns=gradient_columns,
            )
        else:
            with_gradients = enriched
        if include_temporal_encodings:
            temporalized = add_temporal_encodings(with_gradients)
        else:
            temporalized = with_gradients
        return temporalized, topology


def flatten_snapshot_features(
    features: pd.DataFrame,
    include_timestamp: bool = False,
) -> pd.DataFrame:
    """
    Convert snapshot-indexed features into a flat table with columns.

    Args:
        features: DataFrame indexed by (portico, snapshot_time)
        include_timestamp: If True, keep the timestamp column besides ts_min.
    """
    if not isinstance(features.index, pd.MultiIndex):
        raise ValueError("Se esperaba un índice MultiIndex (portico, snapshot_time).")

    flat = features.reset_index()
    flat = flat.rename(columns={"snapshot_time": "timestamp"})

    ts = pd.to_datetime(flat["timestamp"], errors="coerce").astype("datetime64[ns]")
    ts_ns = ts.astype("int64")
    ts_min = pd.Series(ts_ns // 60_000_000_000, index=flat.index)
    ts_min = ts_min.mask(ts.isna(), pd.NA).astype("Int64")
    flat["ts_min"] = ts_min

    if not include_timestamp:
        flat = flat.drop(columns=["timestamp"])

    col_order = ["portico", "ts_min"] + [
        c for c in flat.columns if c not in {"portico", "ts_min"}
    ]
    return flat[col_order].copy()
