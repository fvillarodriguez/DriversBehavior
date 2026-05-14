"""
Static per-pórtico geometry features derived from the canonical porticos_df.

The kilometer marker resets at eje transitions on the same highway (e.g. the
pórtico-15 → pórtico-14 transition where Ruta 5 Norte ends and Ruta 5 Sur
begins). All neighbor and normalization computations therefore stay strictly
within each (eje, calzada) group; cross-eje neighbors are treated as absent
and flagged via is_eje_first / is_eje_last.

Output columns (one row per pórtico):
    portico
    km_norm_eje          ∈ [0,1] within (eje, calzada)
    orden_norm_eje       ∈ [0,1] within (eje, calzada)
    dist_to_upstream_km  km to previous pórtico in the same eje (0 at eje start)
    dist_to_downstream_km km to next pórtico in the same eje (0 at eje end)
    is_eje_first         1 if this is the first pórtico of its (eje, calzada)
    is_eje_last          1 if this is the last
    eje_<value>          one-hot
    calzada_<value>      one-hot
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars is a hard runtime dep
    pl = None  # type: ignore[assignment]

_KM_FILL = 0.0
_BASE_COLS: List[str] = [
    "portico",
    "km_norm_eje",
    "orden_norm_eje",
    "dist_to_upstream_km",
    "dist_to_downstream_km",
    "is_eje_first",
    "is_eje_last",
]


def compute_portico_geometry(
    porticos_df: pd.DataFrame,
    *,
    eje_categories: Optional[Sequence[str]] = None,
    calzada_categories: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if porticos_df is None or porticos_df.empty:
        return pd.DataFrame(columns=_BASE_COLS)

    required = {"portico", "km", "orden", "eje", "calzada"}
    missing = required - set(porticos_df.columns)
    if missing:
        raise ValueError(
            f"porticos_df no tiene las columnas requeridas: {sorted(missing)}"
        )

    work = porticos_df.copy()
    work["km"] = pd.to_numeric(work["km"], errors="coerce")
    work["orden"] = pd.to_numeric(work["orden"], errors="coerce")
    work["eje"] = work["eje"].astype(str).str.strip()
    work["calzada"] = work["calzada"].astype(str).str.strip()

    pieces = []
    for _, grp in work.groupby(["eje", "calzada"], sort=False):
        pieces.append(_geometry_within_group(grp))
    out = pd.concat(pieces, ignore_index=True) if pieces else work.iloc[0:0].copy()

    out["dist_to_upstream_km"] = out["dist_to_upstream_km"].fillna(_KM_FILL)
    out["dist_to_downstream_km"] = out["dist_to_downstream_km"].fillna(_KM_FILL)
    out["is_eje_first"] = out["is_eje_first"].astype("int8")
    out["is_eje_last"] = out["is_eje_last"].astype("int8")

    eje_dummies = _stable_one_hot(out["eje"], "eje", eje_categories)
    calzada_dummies = _stable_one_hot(out["calzada"], "calzada", calzada_categories)

    result = pd.concat([out[_BASE_COLS], eje_dummies, calzada_dummies], axis=1)
    return result.reset_index(drop=True)


def _geometry_within_group(grp: pd.DataFrame) -> pd.DataFrame:
    g = grp.sort_values("orden", kind="mergesort").copy()

    upstream_km = g["km"].shift(1)
    downstream_km = g["km"].shift(-1)
    g["dist_to_upstream_km"] = (g["km"] - upstream_km).abs()
    g["dist_to_downstream_km"] = (g["km"] - downstream_km).abs()
    g["is_eje_first"] = upstream_km.isna()
    g["is_eje_last"] = downstream_km.isna()

    km_min = g["km"].min()
    km_max = g["km"].max()
    span = km_max - km_min
    if not np.isfinite(span) or span <= 0:
        g["km_norm_eje"] = 0.0
    else:
        g["km_norm_eje"] = (g["km"] - km_min) / span

    orden_max = g["orden"].max()
    if not np.isfinite(orden_max) or orden_max <= 0:
        g["orden_norm_eje"] = 0.0
    else:
        g["orden_norm_eje"] = g["orden"] / orden_max

    return g


def _has_canonical_schema(porticos_df: pd.DataFrame) -> bool:
    required = {"portico", "km", "orden", "eje", "calzada"}
    return required.issubset(set(porticos_df.columns))


def _stable_one_hot(
    series: pd.Series,
    prefix: str,
    categories: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    s = series.astype(str).str.strip()
    if categories is None:
        cats = sorted(c for c in s.dropna().unique().tolist() if c)
    else:
        cats = list(categories)
    cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
    s_cat = s.astype(cat_type)
    return pd.get_dummies(s_cat, prefix=prefix, dummy_na=False, dtype="int8")


def attach_portico_geometry(
    df_pm: pd.DataFrame,
    porticos_df: pd.DataFrame,
    *,
    portico_col: str = "portico",
    eje_categories: Optional[Sequence[str]] = None,
    calzada_categories: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Idempotent merge of static pórtico geometry into a flat features frame.

    Columns already present in df_pm are not overwritten; only missing columns
    are added. The join is by stringified pórtico ID to be robust to dtype
    drift between feature pipelines and the canonical porticos_df. If
    porticos_df does not carry the canonical schema (portico/km/orden/eje/
    calzada), the call is a no-op so callers using non-canonical fixtures
    aren't penalised.
    """
    if df_pm is None or df_pm.empty:
        return df_pm
    if porticos_df is None or porticos_df.empty:
        return df_pm
    if not _has_canonical_schema(porticos_df):
        return df_pm

    geom = compute_portico_geometry(
        porticos_df,
        eje_categories=eje_categories,
        calzada_categories=calzada_categories,
    )
    new_cols = [c for c in geom.columns if c != "portico" and c not in df_pm.columns]
    if not new_cols:
        return df_pm

    # df_pm está indexado por portico (un mismo ID puede aparecer en distintos
    # (eje, calzada) en porticos_df; quedamos con la primera ocurrencia para
    # mantener una geometría única por ID, consistente con build_static_topology
    # que también colapsa duplicados).
    geom_to_merge = (
        geom[["portico"] + new_cols]
        .assign(portico=lambda d: d["portico"].astype(str).str.strip())
        .drop_duplicates(subset="portico", keep="first")
    )

    out = df_pm.copy()
    out["__geom_join_key__"] = out[portico_col].astype(str).str.strip()
    merged = out.merge(
        geom_to_merge,
        left_on="__geom_join_key__",
        right_on="portico",
        how="left",
        suffixes=("", "_geom"),
    )
    merged = merged.drop(columns=["__geom_join_key__"])
    if portico_col != "portico" and "portico" in merged.columns:
        merged = merged.drop(columns=["portico"])
    return merged


def attach_portico_geometry_polars(
    features: "pl.DataFrame",
    porticos_df: pd.DataFrame,
    *,
    portico_col: str = "portico",
    eje_categories: Optional[Sequence[str]] = None,
    calzada_categories: Optional[Sequence[str]] = None,
) -> "pl.DataFrame":
    """
    Polars-side counterpart of attach_portico_geometry. Idempotent.
    """
    if pl is None:
        raise RuntimeError("polars no está instalado")
    if features is None or features.is_empty():
        return features
    if porticos_df is None or porticos_df.empty:
        return features
    if not _has_canonical_schema(porticos_df):
        return features

    geom = compute_portico_geometry(
        porticos_df,
        eje_categories=eje_categories,
        calzada_categories=calzada_categories,
    )
    new_cols = [c for c in geom.columns if c != "portico" and c not in features.columns]
    if not new_cols:
        return features

    geom_to_merge = (
        geom[["portico"] + new_cols]
        .assign(portico=lambda d: d["portico"].astype(str).str.strip())
        .drop_duplicates(subset="portico", keep="first")
    )
    geom_pl = pl.from_pandas(geom_to_merge).with_columns(
        pl.col("portico").cast(pl.Utf8)
    )
    out = features.with_columns(
        pl.col(portico_col).cast(pl.Utf8).str.strip_chars().alias("__geom_join_key__")
    )
    out = out.join(
        geom_pl,
        left_on="__geom_join_key__",
        right_on="portico",
        how="left",
    ).drop("__geom_join_key__")
    return out
