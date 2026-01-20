from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.snapshot_pipeline import GraphTopology


@dataclass
class SequenceConfig:
    """Configuration for temporal snapshot sequences."""

    sequence_length: int = 6
    guard_band_minutes: int = 10
    horizon_minutes: int = 20
    include_downstream: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length debe ser >= 1.")
        if self.guard_band_minutes < 0:
            raise ValueError("guard_band_minutes no puede ser negativo.")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes debe ser > 0.")


@dataclass
class SequenceIndex:
    """Indexed representation of snapshot sequences aligned with node rows."""

    sequence_rows: np.ndarray  # shape = [num_sequences, sequence_length]
    target_rows: np.ndarray    # shape = [num_sequences]
    labels: np.ndarray         # shape = [num_sequences]
    porticos: np.ndarray       # shape = [num_sequences]
    target_ts_min: np.ndarray  # shape = [num_sequences]
    config: SequenceConfig


def _collect_accident_times(
    accidents_df: pd.DataFrame,
    portico_col: str = "ultimo_portico",
    ts_col: str = "ts_min",
) -> Dict[int | str, np.ndarray]:
    grouped: Dict[int | str, np.ndarray] = {}
    if accidents_df is None or accidents_df.empty:
        return grouped

    for portico, group in accidents_df.groupby(portico_col):
        values = group[ts_col].dropna().astype(int).to_numpy()
        if values.size:
            grouped[portico] = np.sort(values)
    return grouped


def _combine_accidents(
    base: np.ndarray,
    extra: Optional[np.ndarray],
) -> np.ndarray:
    if extra is None or extra.size == 0:
        return base
    if base.size == 0:
        return extra
    return np.sort(np.concatenate([base, extra]))


def _labels_for_group(
    ts_array: np.ndarray,
    accident_times: np.ndarray,
    guard_band: int,
    horizon: int,
) -> np.ndarray:
    if accident_times.size == 0 or ts_array.size == 0:
        return np.zeros(ts_array.shape, dtype=np.int8)

    guard_start = ts_array + guard_band
    guard_end = ts_array + guard_band + horizon
    left = np.searchsorted(accident_times, guard_start, side="left")
    right = np.searchsorted(accident_times, guard_end, side="left")
    return (right > left).astype(np.int8)


def build_sequence_index(
    df_pm: pd.DataFrame,
    df_accidents: pd.DataFrame,
    topology: Optional[GraphTopology],
    config: SequenceConfig,
    portico_col: str = "portico",
    ts_col: str = "ts_min",
    step_minutes: Optional[int] = None,
) -> Tuple[SequenceIndex, pd.DataFrame]:
    """
    Build temporal sequences (length L) and future-risk labels for each snapshot.

    Args:
        df_pm: DataFrame with at least `portico` and `ts_min`, ordered as node rows.
        df_accidents: DataFrame with `ultimo_portico` and `ts_min` of accidents.
        topology: GraphTopology to access downstream relationships (optional).
        config: SequenceConfig parameters (L, γ, H, downstream usage).
        portico_col: Column name in df_pm representing the portico identifier.
        ts_col: Column name in df_pm with timestamps in minutes since epoch.

    Returns:
        sequence_index: SequenceIndex referencing row positions in df_pm.
        labels_df: DataFrame with columns [row_id, portico, ts_min, label].
    """
    if portico_col not in df_pm.columns or ts_col not in df_pm.columns:
        raise KeyError(f"df_pm debe contener columnas '{portico_col}' y '{ts_col}'.")

    df_work = df_pm[[portico_col, ts_col]].copy()
    df_work = df_work.reset_index(drop=True)
    df_work["row_id"] = np.arange(len(df_work), dtype=np.int64)
    df_work[ts_col] = df_work[ts_col].astype(int)

    accidents_by_portico = _collect_accident_times(df_accidents)
    downstream_map: Dict[int | str, Optional[int | str]] = {}
    if config.include_downstream and topology is not None:
        downstream_map = topology.downstream

    sequence_rows: List[np.ndarray] = []
    target_rows: List[int] = []
    porticos_list: List[int | str] = []
    target_ts_list: List[int] = []

    labels = np.zeros(len(df_work), dtype=np.int8)

    def _iter_contiguous_segments(
        row_ids: np.ndarray,
        ts_values: np.ndarray,
        step: Optional[int],
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if row_ids.size == 0:
            return []
        if step is None:
            return [(row_ids, ts_values)]
        diffs = np.diff(ts_values)
        breaks = np.where(diffs != int(step))[0]
        segments = []
        start = 0
        for brk in breaks:
            end = brk + 1
            segments.append((row_ids[start:end], ts_values[start:end]))
            start = end
        segments.append((row_ids[start:], ts_values[start:]))
        return segments

    for portico, group in df_work.groupby(portico_col, sort=False):
        row_ids = group["row_id"].to_numpy()
        ts_values = group[ts_col].to_numpy()
        base_accidents = accidents_by_portico.get(portico, np.empty(0, dtype=int))
        if config.include_downstream and downstream_map.get(portico) is not None:
            downstream_port = downstream_map[portico]
            downstream_accidents = accidents_by_portico.get(downstream_port, np.empty(0, dtype=int))
            accident_times = _combine_accidents(base_accidents, downstream_accidents)
        else:
            accident_times = base_accidents

        group_labels = _labels_for_group(
            ts_values,
            accident_times,
            config.guard_band_minutes,
            config.horizon_minutes,
        )
        labels[row_ids] = group_labels

        segments = _iter_contiguous_segments(row_ids, ts_values, step_minutes)
        for seg_rows, seg_ts in segments:
            if seg_rows.size < config.sequence_length:
                continue
            for idx in range(config.sequence_length - 1, seg_rows.size):
                seq = seg_rows[idx - config.sequence_length + 1 : idx + 1]
                sequence_rows.append(seq)
                target_rows.append(seg_rows[idx])
                porticos_list.append(portico)
                target_ts_list.append(int(seg_ts[idx]))

    if sequence_rows:
        sequence_array = np.vstack(sequence_rows).astype(np.int64)
        target_array = np.asarray(target_rows, dtype=np.int64)
        labels_array = labels[target_array].astype(np.int8)
        porticos_array = np.asarray(porticos_list)
        target_ts_array = np.asarray(target_ts_list, dtype=np.int64)
    else:
        sequence_array = np.empty((0, config.sequence_length), dtype=np.int64)
        target_array = np.empty((0,), dtype=np.int64)
        labels_array = np.empty((0,), dtype=np.int8)
        porticos_array = np.empty((0,), dtype=object)
        target_ts_array = np.empty((0,), dtype=np.int64)

    labels_df = pd.DataFrame({
        "row_id": df_work["row_id"],
        portico_col: df_work[portico_col],
        ts_col: df_work[ts_col],
        "label": labels.astype(np.int8),
    })

    seq_index = SequenceIndex(
        sequence_rows=sequence_array,
        target_rows=target_array,
        labels=labels_array,
        porticos=porticos_array,
        target_ts_min=target_ts_array,
        config=config,
    )
    return seq_index, labels_df
