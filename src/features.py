import re
import sys
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl

# Barra de progreso: usar tqdm si está disponible, con fallback liviano
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable, total=None, desc=None):  # simple fallback
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except Exception:
                total = None
        count = 0
        print((desc or "Progreso") + (f" (total={total})" if total else ""))
        for item in iterable:
            count += 1
            if total:
                # Actualiza ~10 veces
                step = max(1, total // 10)
                if count == 1 or count == total or (count % step == 0):
                    print(f"  {desc or 'Progreso'}: {count}/{total}")
                    sys.stdout.flush()
            yield item

from src.utils import select_flujos_csv_path
from src.config import (
    DT_MINUTES,
    DEBUG,
    HARMONIC_V_MIN,
)


def _log_debug(message: str) -> None:
    if DEBUG:
        print(message)


def _clean_pivot_columns(
    columns: list[str],
    *,
    index_cols: set[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in columns:
        if name in index_cols:
            continue
        cleaned = name
        cleaned = cleaned.replace('"', "").replace("'", "")
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = cleaned.replace("(", "").replace(")", "")
        cleaned = cleaned.replace(" ", "")
        cleaned = cleaned.replace(",", "_").replace(":", "_")
        cleaned = re.sub(r"_([0-9]+)\\.0$", r"_\\1", cleaned)
        if cleaned != name:
            mapping[name] = cleaned
    return mapping


def compute_pm_features(
    flujos_df: Union[pd.DataFrame, pl.DataFrame],
    interactive: bool = True,
    velocidad_bins: Optional[np.ndarray] = None,
    dt_minutes: int = int(DT_MINUTES),
) -> pd.DataFrame:
    """
    Polars implementation of Feature Engineering.
    Agrega los datos de flujo por ventanas de DT_MINUTES y pivotea categorías en ancho.
    """
    _log_debug(f"[compute_pm_features] start dt={dt_minutes}")

    if isinstance(flujos_df, pl.LazyFrame):
        lf = flujos_df.collect()
    elif isinstance(flujos_df, pd.DataFrame):
        lf = pl.from_pandas(flujos_df)
    else:
        lf = flujos_df

    if lf is None or lf.is_empty():
        return pd.DataFrame()

    # 2. Standardize Columns
    # Rename to lower stripping spaces
    lf = lf.rename({c: c.strip().lower() for c in lf.columns})
    
    # Identify key columns
    cols = lf.columns
    fecha_col = next((c for c in cols if "fecha" in c), "fecha")
    portico_col = next((c for c in cols if "portico" in c), "portico")
    vel_col = next((c for c in cols if "velocidad" in c), "velocidad")
    cat_col = next((c for c in cols if "categoria" in c), "categoria")
    required = [fecha_col, portico_col, vel_col, cat_col]
    missing = [c for c in required if c not in lf.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # 3. Preprocessing (Cast Types & Time)
    lf = lf.with_columns([
        pl.col(vel_col).cast(pl.Float64, strict=False),
        pl.col(cat_col).cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int32)
    ])

    # Parse Date
    # Try multiple formats or use reliable strptime if format known, else ISO
    if lf.select(pl.col(fecha_col)).dtypes[0] == pl.Utf8:
        lf = lf.with_columns(
            pl.col(fecha_col).str.to_datetime(time_unit="ns", strict=False) # Generic parser
        )

    # Filter Valid Dates
    lf = lf.filter(pl.col(fecha_col).is_not_null())
    if lf.is_empty():
        return pd.DataFrame()

    # Filter Outliers (Vel)
    lower_bound = 2.0
    upper_bound = 180.0
    
    # Outlier Rate calculation (Lazy)
    lf = lf.with_columns(
        (
            pl.col(fecha_col).dt.timestamp("us")
            // (dt_minutes * 60 * 1_000_000)
        )
        .cast(pl.Int64)
        .alias("_ts_idx")
    )
    
    # Outlier Mask
    is_outlier = (pl.col(vel_col) < lower_bound) | (pl.col(vel_col) > upper_bound)
    
    # Calc Outlier Rate per window (grouping pre-filter)
    outlier_rate_df = (
        lf.group_by([portico_col, "_ts_idx"])
        .agg([
            (pl.col(vel_col).filter(is_outlier).count() / pl.count()).alias("outlier_rate")
        ])
    )

    # Apply Filter
    lf = lf.filter(~is_outlier)

    # 4. Feature Engineering Expressions
    # Time Bins
    lf = lf.with_columns(
        (pl.col("_ts_idx") * dt_minutes).alias("ts_min")
    )

    # Expressions for basic metrics
    # Velocidad Harmonica: n / sum(1/v)
    v_safe = pl.max_horizontal(pl.col(vel_col), pl.lit(HARMONIC_V_MIN))
    expr_harm = pl.count() / (1.0 / v_safe).sum()

    # MAD Consistent (Approx via sub-expression)
    expr_mad = (pl.col(vel_col) - pl.col(vel_col).median()).abs().median() * 1.4826

    agg_exprs = [
        pl.count(vel_col).alias("flujo"),
        pl.mean(vel_col).alias("velocidad_media"),
        pl.std(vel_col).alias("velocidad_std"),
        expr_harm.alias("velocidad_harmonica"),
        pl.median(vel_col).alias("velocidad_mediana"),
        pl.col(vel_col).quantile(0.25).alias("velocidad_q25"),
        pl.col(vel_col).quantile(0.75).alias("velocidad_q75"),
        (pl.col(vel_col).quantile(0.75) - pl.col(vel_col).quantile(0.25)).alias("velocidad_iqr"),
        expr_mad.alias("velocidad_mad"),
        (pl.std(vel_col) / (pl.mean(vel_col) + 1e-12)).alias("velocidad_cv"),
        pl.col(vel_col).skew().alias("velocidad_skew"),
        pl.col(vel_col).kurtosis().alias("velocidad_kurt_exceso"),
    ]
    
    _log_debug(f"[compute_pm_features] polars {pl.__version__}")
    q_long = (
        lf.group_by([portico_col, "ts_min", cat_col])
        .agg(agg_exprs)
    )
    if q_long.is_empty():
        return pd.DataFrame()
    _log_debug("[compute_pm_features] pivot")
    q_wide = (
        q_long
        .pivot(
            index=[portico_col, "ts_min"],
            on=cat_col,
            values=[
                 "flujo", "velocidad_media", "velocidad_std",
                 "velocidad_harmonica", "velocidad_mediana",
                 "velocidad_q25", "velocidad_q75", "velocidad_iqr",
                 "velocidad_mad", "velocidad_cv", "velocidad_skew",
                 "velocidad_kurt_exceso"
            ],
            aggregate_function="first"
        )
    )
    if q_wide.is_empty():
        return pd.DataFrame()

    # Join Outlier Rate
    outlier_rate_df = outlier_rate_df.with_columns(
        (pl.col("_ts_idx") * dt_minutes).alias("ts_min")
    ).drop("_ts_idx")
    q_wide = (
        q_wide.join(outlier_rate_df, on=[portico_col, "ts_min"], how="left")
        .fill_null(strategy="forward")
        .fill_null(0)
    )

    index_cols = {portico_col, "ts_min"}
    rename_map = _clean_pivot_columns(q_wide.columns, index_cols=index_cols)
    if rename_map:
        q_wide = q_wide.rename(rename_map)

    # 5. Global & Derivative Features
    cats = [1, 2, 3]
    dens_exprs = []
    for c in cats:
        flujo_col = f"flujo_{c}"
        vel_col_name = f"velocidad_media_{c}"
        if flujo_col in q_wide.columns and vel_col_name in q_wide.columns:
            denom = (
                pl.when(pl.col(vel_col_name) == 0)
                .then(1.0)
                .otherwise(pl.col(vel_col_name))
            )
            dens_exprs.append(
                (pl.col(flujo_col) / denom).alias(f"densidad_{c}")
            )
    if dens_exprs:
        q_wide = q_wide.with_columns(dens_exprs)

    flujo_cols = [f"flujo_{c}" for c in cats if f"flujo_{c}" in q_wide.columns]
    if flujo_cols:
        q_wide = q_wide.with_columns(
            pl.sum_horizontal([pl.col(c) for c in flujo_cols]).alias(
                "flujo_total"
            )
        )
    else:
        q_wide = q_wide.with_columns(pl.lit(0.0).alias("flujo_total"))

    if "flujo_2" in q_wide.columns:
        denom = (
            pl.when(pl.col("flujo_total") == 0)
            .then(1.0)
            .otherwise(pl.col("flujo_total"))
        )
        q_wide = q_wide.with_columns(
            (pl.col("flujo_2") / denom).alias("prop_pesados")
        )

    dens_cols = [
        f"densidad_{c}" for c in cats if f"densidad_{c}" in q_wide.columns
    ]
    if dens_cols:
        q_wide = q_wide.with_columns(
            pl.sum_horizontal([pl.col(c) for c in dens_cols]).alias(
                "densidad_total"
            )
        )
    else:
        q_wide = q_wide.with_columns(pl.lit(0.0).alias("densidad_total"))

    q_wide = q_wide.with_columns(
        (
            pl.col("flujo_total")
            / pl.when(pl.col("densidad_total") == 0)
            .then(1e-9)
            .otherwise(pl.col("densidad_total"))
        ).alias("v_armonica")
    )

    v_cols = [
        f"velocidad_media_{c}"
        for c in cats
        if f"velocidad_media_{c}" in q_wide.columns
    ]
    if v_cols:
        q_wide = q_wide.with_columns(
            (pl.col("flujo_total") / (pl.sum_horizontal([pl.col(c) for c in v_cols]) + 1e-9)).alias(
                "indice_congestion"
            )
        )

    q95 = float(q_wide.select(pl.col("flujo_total").quantile(0.95)).item())
    q_wide = q_wide.with_columns(pl.lit(q95).alias("q_max_p95"))
    q_wide = q_wide.with_columns(
        (pl.col("flujo_total") / (q95 + 1e-9)).alias("vc_ratio")
    )

    if portico_col in q_wide.columns:
        q_wide = q_wide.sort([portico_col, "ts_min"])

    return q_wide.to_pandas()

def compute_pm_features_streaming(file_path=None, chunksize=500_000, sep=None, encoding_prefer='utf-8'):
    """ Legacy Streaming wrapper - Refactored to use Polars for file reading if possible? """
    # For now, keep legacy logic using Polars Scan
    
    import polars as pl
    if file_path is None: 
        file_path = select_flujos_csv_path()
    
    if not file_path: return None
    
    print("🚀 Streaming via Polars Scan...")
    q = pl.scan_csv(file_path, separator=sep if sep else ',', ignore_errors=True)
    
    try:
        df_pl = q.collect()
        return compute_pm_features(df_pl)
    except Exception as e:
        print(f"Error Polars Scan: {e}")
        return None
