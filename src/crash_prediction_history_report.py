from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
MODEL_HISTORY_DIR = RESULTS_DIR / "model_history"
THESIS_DESIGN_ROOT = Path.home() / ".codex" / "skills" / "tesis-doctoral-design"

DEFAULT_RUN_IDS = [
    "20260420_234633_d8758eb8",
    "20260420_232323_c11e03e0",
    "20260420_184300_dfbe05d2",
    "20260420_145545_b4956a4c",
    "20260420_145353_7baab223",
    "20260419_194714_bfadadf5",
    "20260419_194113_4f3b17ac",
]

OUTPUT_STEM = "crash_prediction_history_2026-04-19_2026-04-20"

STRATEGY_LABEL_TO_MODEL_VARIANT = {
    "base": "Base",
    "cluster": "Cluster",
    "base_cluster": "Base+Cluster",
}

CSV_COLUMN_ORDER = [
    "run_id",
    "saved_at",
    "model_variant",
    "model_name",
    "features_path",
    "features_path_full",
    "label",
    "threshold_protocol",
    "threshold_objective",
    "calibration_method",
    "balance_strategy",
    "k",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "f1",
    "f1_global",
    "far",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "mcc",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "test_rows",
    "positive_support",
    "context_key",
    "is_duplicate_context",
    "duplicate_group_size",
    "is_latest_for_context",
    "duplicate_rank_latest",
]

NUMERIC_COLUMNS = [
    "k",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "f1",
    "f1_global",
    "far",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "mcc",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "test_rows",
    "positive_support",
]

BOOLEAN_COLUMNS = [
    "is_duplicate_context",
    "is_latest_for_context",
]

MODEL_COLOR_MAP = {
    "Base": "#1b3a6b",
    "Cluster": "#d4541a",
    "Base+Cluster": "#3b7a57",
}


@dataclass
class ReportArtifacts:
    csv_path: Path
    html_path: Path
    full_history_df: pd.DataFrame
    latest_context_df: pd.DataFrame
    champion_row: pd.Series


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_basename(path_str: str) -> str:
    if not path_str:
        return ""
    return Path(path_str).name


def _normalize_features_alias(path_str: str) -> str:
    basename = _safe_basename(path_str)
    if not basename:
        return "-"
    stem = basename.removesuffix(".duckdb")
    stem = stem.replace("accident_flow_features_cluster_gmm_k5_", "")
    stem = stem.replace("(2022_01-2024_09)", "2022_01-2024_09")
    stem = stem.replace("2022_01-2024_07_TTC_fijo", "2022_01-2024_07 | TTC fijo")
    stem = stem.replace("_", " ")
    stem = stem.replace("  ", " ")
    return stem.strip()


def _format_decimal_es(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    formatted = f"{float(value):,.{digits}f}"
    integer, decimal = formatted.split(".")
    integer = integer.replace(",", "\u202f")
    decimal = decimal.rstrip("0")
    if not decimal:
        return integer
    return f"{integer},{decimal}"


def _format_int_es(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}".replace(",", "\u202f")


def _format_bool_es(value: object) -> str:
    return "Sí" if bool(value) else "No"


def _html_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return html.escape(str(value))


def _extract_manifest_row(manifest_path: Path) -> dict[str, object]:
    manifest = _load_json(manifest_path)
    metrics = dict(manifest.get("metrics") or {})
    dataset = dict(manifest.get("dataset") or {})
    tramo = dict(dataset.get("tramo") or {})
    selected_features = list(manifest.get("selected_features") or [])
    strategy_label = str(manifest.get("strategy_label") or "").strip()
    features_path_full = str(manifest.get("features_path") or "")

    return {
        "run_id": str(manifest.get("run_id") or manifest_path.parents[1].name),
        "saved_at": str(manifest.get("saved_at") or ""),
        "model_variant": STRATEGY_LABEL_TO_MODEL_VARIANT.get(strategy_label, strategy_label or "Unknown"),
        "model_name": str(manifest.get("model_name") or ""),
        "features_path": _safe_basename(features_path_full),
        "features_path_full": features_path_full,
        "label": str(tramo.get("label") or ""),
        "threshold_protocol": str(metrics.get("threshold_protocol") or ""),
        "threshold_objective": str(metrics.get("threshold_objective") or ""),
        "calibration_method": str(metrics.get("calibration_method") or ""),
        "balance_strategy": str(metrics.get("balance_strategy") or ""),
        "k": int(len(selected_features)),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "sensitivity": metrics.get("sensitivity"),
        "f1": metrics.get("f1"),
        "f1_global": metrics.get("f1_global"),
        "far": metrics.get("far"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "brier_score": metrics.get("brier_score"),
        "mcc": metrics.get("mcc"),
        "true_positive": metrics.get("true_positives"),
        "false_positive": metrics.get("false_positives"),
        "true_negative": metrics.get("true_negatives"),
        "false_negative": metrics.get("false_negatives"),
        "test_rows": (manifest.get("split_info") or {}).get("test_rows"),
        "positive_support": metrics.get("positive_support"),
        "manifest_path": str(manifest_path),
    }


def build_history_dataframe(
    *,
    model_history_dir: Path,
    run_ids: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id in run_ids:
        run_dir = model_history_dir / run_id
        for strategy_label in ("base", "cluster", "base_cluster"):
            manifest_path = run_dir / strategy_label / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"No existe el manifest esperado: {manifest_path}")
            rows.append(_extract_manifest_row(manifest_path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No se encontraron manifests para construir el historial.")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["saved_at_ts"] = pd.to_datetime(df["saved_at"], errors="coerce")
    df["context_key"] = (
        df["features_path"].astype(str)
        + " | "
        + df["model_variant"].astype(str)
        + " | "
        + df["label"].astype(str)
        + " | "
        + df["threshold_protocol"].astype(str)
        + " | "
        + df["threshold_objective"].astype(str)
        + " | "
        + df["calibration_method"].astype(str)
        + " | "
        + df["balance_strategy"].astype(str)
    )

    ranked = df.sort_values(
        ["context_key", "saved_at_ts", "run_id"],
        ascending=[True, False, False],
        kind="stable",
    ).copy()
    ranked["duplicate_rank_latest"] = ranked.groupby("context_key", dropna=False).cumcount() + 1

    df = df.merge(
        ranked[["context_key", "run_id", "model_variant", "duplicate_rank_latest"]],
        on=["context_key", "run_id", "model_variant"],
        how="left",
        validate="one_to_one",
    )
    df["duplicate_group_size"] = df.groupby("context_key", dropna=False)["context_key"].transform("size")
    df["is_duplicate_context"] = df["duplicate_group_size"].gt(1)
    df["is_latest_for_context"] = df["duplicate_rank_latest"].eq(1)

    ordered = df.sort_values(
        [
            "features_path",
            "label",
            "threshold_protocol",
            "threshold_objective",
            "calibration_method",
            "balance_strategy",
            "model_variant",
            "run_id",
        ],
        ascending=[True, True, True, True, True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)

    for column in BOOLEAN_COLUMNS:
        ordered[column] = ordered[column].fillna(False).astype(bool)
    ordered["duplicate_group_size"] = pd.to_numeric(ordered["duplicate_group_size"], errors="coerce").astype("Int64")
    ordered["duplicate_rank_latest"] = pd.to_numeric(ordered["duplicate_rank_latest"], errors="coerce").astype("Int64")

    return ordered


def _pareto_frontier_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    tp = pd.to_numeric(df["true_positive"], errors="coerce").fillna(float("-inf"))
    fp = pd.to_numeric(df["false_positive"], errors="coerce").fillna(float("inf"))
    mask = pd.Series(True, index=df.index, dtype=bool)
    for idx in df.index:
        dominates = ((tp >= tp.loc[idx]) & (fp <= fp.loc[idx])) & ((tp > tp.loc[idx]) | (fp < fp.loc[idx]))
        dominates.loc[idx] = False
        if dominates.any():
            mask.loc[idx] = False
    return mask


def select_champion(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("No es posible seleccionar campeón sobre un DataFrame vacío.")

    work = df.copy()
    for column in ["true_positive", "false_positive", "mcc", "pr_auc", "f1_global", "far"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["saved_at_ts"] = pd.to_datetime(work["saved_at"], errors="coerce")
    work["is_pareto_frontier"] = _pareto_frontier_mask(work)
    frontier = work.loc[work["is_pareto_frontier"]].copy()

    champion = frontier.sort_values(
        ["true_positive", "false_positive", "mcc", "pr_auc", "f1_global", "far", "saved_at_ts", "run_id"],
        ascending=[False, True, False, False, False, True, False, False],
        kind="stable",
    ).iloc[0]
    return champion


def _select_best_per_group(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, group in df.groupby(list(group_cols), dropna=False, sort=False):
        champion = select_champion(group)
        row = champion.to_dict()
        row["is_pareto_frontier"] = True
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(list(group_cols), kind="stable").reset_index(drop=True)


def _build_duplicate_deltas(full_history_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    duplicate_df = full_history_df.loc[full_history_df["is_duplicate_context"]].copy()
    if duplicate_df.empty:
        return pd.DataFrame()

    duplicate_df["saved_at_ts"] = pd.to_datetime(duplicate_df["saved_at"], errors="coerce")
    ordered = duplicate_df.sort_values(
        ["context_key", "saved_at_ts", "run_id"],
        ascending=[True, False, False],
        kind="stable",
    )

    for context_key, group in ordered.groupby("context_key", dropna=False, sort=False):
        if len(group) < 2:
            continue
        latest = group.iloc[0]
        previous = group.iloc[1]
        rows.append(
            {
                "context_key": context_key,
                "features_path": latest["features_path"],
                "features_alias": _normalize_features_alias(str(latest["features_path"])),
                "model_variant": latest["model_variant"],
                "threshold_objective": latest["threshold_objective"],
                "calibration_method": latest["calibration_method"],
                "balance_strategy": latest["balance_strategy"],
                "latest_run_id": latest["run_id"],
                "previous_run_id": previous["run_id"],
                "latest_saved_at": latest["saved_at"],
                "previous_saved_at": previous["saved_at"],
                "delta_true_positive": pd.to_numeric(latest["true_positive"], errors="coerce")
                - pd.to_numeric(previous["true_positive"], errors="coerce"),
                "delta_false_positive": pd.to_numeric(latest["false_positive"], errors="coerce")
                - pd.to_numeric(previous["false_positive"], errors="coerce"),
                "delta_mcc": pd.to_numeric(latest["mcc"], errors="coerce") - pd.to_numeric(previous["mcc"], errors="coerce"),
                "delta_pr_auc": pd.to_numeric(latest["pr_auc"], errors="coerce")
                - pd.to_numeric(previous["pr_auc"], errors="coerce"),
                "delta_f1_global": pd.to_numeric(latest["f1_global"], errors="coerce")
                - pd.to_numeric(previous["f1_global"], errors="coerce"),
                "delta_far": pd.to_numeric(latest["far"], errors="coerce") - pd.to_numeric(previous["far"], errors="coerce"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["features_path", "model_variant", "balance_strategy", "latest_run_id"],
        kind="stable",
    ).reset_index(drop=True)


def _table_html(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    title_map: dict[str, str] | None = None,
    numeric_columns: Iterable[str] = (),
    integer_columns: Iterable[str] = (),
    bool_columns: Iterable[str] = (),
) -> str:
    if df.empty:
        return '<div class="m-callout ochre"><div class="label">Tabla vacía</div><p>No hay filas para esta sección.</p></div>'

    title_map = title_map or {}
    numeric_columns = set(numeric_columns)
    integer_columns = set(integer_columns)
    bool_columns = set(bool_columns)

    header_html = "".join(
        f"<th>{_html_escape(title_map.get(column, column))}</th>"
        for column in columns
    )

    body_rows: list[str] = []
    for _, row in df.loc[:, list(columns)].iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if column in bool_columns:
                rendered = _format_bool_es(value)
            elif column in integer_columns:
                rendered = _format_int_es(value)
            elif column in numeric_columns:
                rendered = _format_decimal_es(value)
            else:
                rendered = _html_escape(value)
            cells.append(f"<td>{rendered}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="table-scroll"><table class="report-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _load_design_css(design_root: Path) -> str:
    colors_path = design_root / "colors_and_type.css"
    manuscript_path = design_root / "ui_kits" / "manuscript" / "manuscript.css"

    fallback_css = """
    :root {
      --paper: #fdfcfa;
      --paper-2: #f6f3ee;
      --rule: #d8d5cd;
      --ink: #1a1a1a;
      --ink-2: #3d3d3d;
      --ink-3: #6b6b6b;
      --accent: #1b3a6b;
      --accent-2: #2a5396;
      --ok: #3b7a57;
      --warn: #b08a2e;
      --err: #a8493d;
      --font-serif: Georgia, 'Times New Roman', serif;
      --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --font-mono: Menlo, Consolas, monospace;
      --measure: 680px;
      --measure-wide: 900px;
      --ease-out: cubic-bezier(0.2, 0.7, 0.2, 1);
    }
    body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--font-serif); }
    """

    if not colors_path.exists() or not manuscript_path.exists():
        return fallback_css

    colors_css = colors_path.read_text(encoding="utf-8")
    manuscript_css = manuscript_path.read_text(encoding="utf-8")
    manuscript_lines = [
        line for line in manuscript_css.splitlines()
        if "@import" not in line
    ]
    return colors_css + "\n" + "\n".join(manuscript_lines)


def _build_scatter_chart(latest_df: pd.DataFrame, champion_row: pd.Series) -> str:
    work = latest_df.copy()
    work["features_alias"] = work["features_path"].map(_normalize_features_alias)
    work["config_label"] = (
        work["model_variant"].astype(str)
        + " | "
        + work["threshold_objective"].astype(str)
        + " | "
        + work["calibration_method"].astype(str)
        + " | "
        + work["balance_strategy"].astype(str)
    )
    work["is_pareto_frontier"] = _pareto_frontier_mask(work)

    fig = px.scatter(
        work,
        x="false_positive",
        y="true_positive",
        color="model_variant",
        symbol="features_alias",
        color_discrete_map=MODEL_COLOR_MAP,
        hover_name="config_label",
        hover_data={
            "run_id": True,
            "features_alias": True,
            "mcc": ":.4f",
            "pr_auc": ":.4f",
            "f1_global": ":.4f",
            "far": ":.4f",
            "false_positive": True,
            "true_positive": True,
        },
        labels={
            "false_positive": "False positives",
            "true_positive": "True positives",
            "model_variant": "Modelo",
            "features_alias": "Features",
        },
        template="plotly_white",
        height=560,
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=0.8, color="#fdfcfa")))

    frontier = work.loc[work["is_pareto_frontier"]].copy()
    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["false_positive"],
                y=frontier["true_positive"],
                mode="markers",
                name="Pareto",
                marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="#1a1a1a", width=2)),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    champion_features_alias = _normalize_features_alias(str(champion_row["features_path"]))
    champion_legend_label = (
        f"Campeón ({champion_row['model_variant']}, {champion_features_alias})"
    )

    fig.add_trace(
        go.Scatter(
            x=[champion_row["false_positive"]],
            y=[champion_row["true_positive"]],
            mode="markers+text",
            name=champion_legend_label,
            marker=dict(size=20, color="#1b3a6b", line=dict(color="#d4541a", width=3), symbol="star"),
            text=["Campeón"],
            textposition="top center",
            hovertext=[
                (
                    f"{champion_row['model_variant']} | {_normalize_features_alias(str(champion_row['features_path']))}"
                    f"<br>TP={int(champion_row['true_positive'])} | FP={int(champion_row['false_positive'])}"
                    f"<br>MCC={champion_row['mcc']:.4f} | PR-AUC={champion_row['pr_auc']:.4f}"
                )
            ],
            hoverinfo="text",
            showlegend=True,
        )
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#fdfcfa",
        plot_bgcolor="#fdfcfa",
    )
    fig.update_xaxes(gridcolor="#e6e3dc", zeroline=False)
    fig.update_yaxes(gridcolor="#e6e3dc", zeroline=False)
    return fig.to_html(full_html=False, include_plotlyjs="inline", config={"responsive": True, "displayModeBar": False})


def _build_bar_chart(latest_df: pd.DataFrame) -> str:
    work = latest_df.copy().sort_values(
        ["true_positive", "false_positive", "mcc"],
        ascending=[False, True, False],
        kind="stable",
    )
    work["config_label"] = (
        work["model_variant"].astype(str)
        + "<br>"
        + work["threshold_objective"].astype(str)
        + " · "
        + work["calibration_method"].astype(str)
        + " · "
        + work["balance_strategy"].astype(str)
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("True positives", "False positives"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=work["config_label"],
            y=work["true_positive"],
            marker_color=[MODEL_COLOR_MAP.get(value, "#1b3a6b") for value in work["model_variant"]],
            name="TP",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=work["config_label"],
            y=work["false_positive"],
            marker_color=[MODEL_COLOR_MAP.get(value, "#1b3a6b") for value in work["model_variant"]],
            name="FP",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=70, b=120),
        paper_bgcolor="#fdfcfa",
        plot_bgcolor="#fdfcfa",
    )
    fig.update_xaxes(tickangle=-35, gridcolor="#e6e3dc")
    fig.update_yaxes(gridcolor="#e6e3dc")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False})


def _build_heatmap_chart(latest_df: pd.DataFrame) -> str:
    work = latest_df.copy()
    work["features_alias"] = work["features_path"].map(_normalize_features_alias)
    work["config_label"] = (
        work["model_variant"].astype(str)
        + " | "
        + work["features_alias"].astype(str)
        + " | "
        + work["threshold_objective"].astype(str)
        + " | "
        + work["calibration_method"].astype(str)
        + " | "
        + work["balance_strategy"].astype(str)
    )
    metrics = ["mcc", "pr_auc", "recall", "far", "f1_global", "brier_score"]
    maximize_metrics = {"mcc", "pr_auc", "recall", "f1_global"}
    matrix_rows: list[dict[str, object]] = []
    for metric in metrics:
        numeric = pd.to_numeric(work[metric], errors="coerce")
        if numeric.nunique(dropna=True) <= 1:
            normalized = pd.Series(1.0, index=work.index)
        else:
            rank = numeric.rank(method="dense", ascending=metric not in maximize_metrics)
            normalized = 1.0 - (rank - 1) / max(rank.max() - 1, 1)
        for idx, row in work.iterrows():
            matrix_rows.append(
                {
                    "config_label": row["config_label"],
                    "metric": metric,
                    "actual": pd.to_numeric(row[metric], errors="coerce"),
                    "score": float(normalized.loc[idx]),
                }
            )

    matrix_df = pd.DataFrame(matrix_rows)
    pivot_score = matrix_df.pivot(index="config_label", columns="metric", values="score")
    pivot_actual = matrix_df.pivot(index="config_label", columns="metric", values="actual")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_score.values,
            x=[col.upper() for col in pivot_score.columns],
            y=pivot_score.index.tolist(),
            colorscale=[
                [0.0, "#ecebe6"],
                [0.4, "#c6d1e4"],
                [0.7, "#5977a7"],
                [1.0, "#1b3a6b"],
            ],
            colorbar=dict(title="Ranking rel."),
            text=[[f"{pivot_actual.iloc[r, c]:.4f}" for c in range(pivot_actual.shape[1])] for r in range(pivot_actual.shape[0])],
            hovertemplate="Config: %{y}<br>Métrica: %{x}<br>Valor: %{text}<br>Ranking relativo: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(420, 32 * len(pivot_score)),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#fdfcfa",
        plot_bgcolor="#fdfcfa",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False})


def _build_duplicate_delta_chart(duplicate_deltas_df: pd.DataFrame) -> str:
    if duplicate_deltas_df.empty:
        return '<div class="m-callout"><div class="label">Sin repetidos</div><p>No hubo contextos con más de una corrida para comparar.</p></div>'

    work = duplicate_deltas_df.copy()
    work["config_label"] = (
        work["model_variant"].astype(str)
        + "<br>"
        + work["features_alias"].astype(str)
        + "<br>"
        + work["balance_strategy"].astype(str)
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Δ true positives (última - previa)", "Δ false positives (última - previa)"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=work["config_label"],
            y=work["delta_true_positive"],
            marker_color="#3b7a57",
            name="Δ TP",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=work["config_label"],
            y=work["delta_false_positive"],
            marker_color="#a8493d",
            name="Δ FP",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_width=1, line_color="#6b6b6b", row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#6b6b6b", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=20, r=20, t=70, b=120),
        paper_bgcolor="#fdfcfa",
        plot_bgcolor="#fdfcfa",
    )
    fig.update_xaxes(tickangle=-35, gridcolor="#e6e3dc")
    fig.update_yaxes(gridcolor="#e6e3dc")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False})


def _build_report_html(
    *,
    full_history_df: pd.DataFrame,
    latest_context_df: pd.DataFrame,
    champion_row: pd.Series,
    group_best_df: pd.DataFrame,
    duplicate_deltas_df: pd.DataFrame,
    design_root: Path,
    model_history_dir: Path,
    csv_path: Path,
) -> str:
    css = _load_design_css(design_root)
    scatter_html = _build_scatter_chart(latest_context_df, champion_row)
    bar_html = _build_bar_chart(latest_context_df)
    heatmap_html = _build_heatmap_chart(latest_context_df)
    duplicate_chart_html = _build_duplicate_delta_chart(duplicate_deltas_df)

    global_frontier_count = int(_pareto_frontier_mask(latest_context_df).sum())
    features_count = int(full_history_df["features_path"].nunique())

    best_by_features = _select_best_per_group(latest_context_df, ["features_path"])
    best_by_features = best_by_features.assign(features_alias=lambda df: df["features_path"].map(_normalize_features_alias))

    summary_table = _table_html(
        group_best_df.assign(features_alias=lambda df: df["features_path"].map(_normalize_features_alias)),
        [
            "features_alias",
            "model_variant",
            "run_id",
            "threshold_objective",
            "calibration_method",
            "balance_strategy",
            "true_positive",
            "false_positive",
            "mcc",
            "pr_auc",
            "recall",
            "far",
        ],
        title_map={
            "features_alias": "Features",
            "model_variant": "Modelo",
            "run_id": "Run ID",
            "threshold_objective": "Objetivo threshold",
            "calibration_method": "Calibración",
            "balance_strategy": "Balance",
            "true_positive": "TP",
            "false_positive": "FP",
            "mcc": "MCC",
            "pr_auc": "PR-AUC",
            "recall": "Recall",
            "far": "FAR",
        },
        numeric_columns={"mcc", "pr_auc", "recall", "far"},
        integer_columns={"true_positive", "false_positive"},
    )

    duplicate_table = _table_html(
        duplicate_deltas_df,
        [
            "features_alias",
            "model_variant",
            "balance_strategy",
            "previous_run_id",
            "latest_run_id",
            "delta_true_positive",
            "delta_false_positive",
            "delta_mcc",
            "delta_pr_auc",
        ],
        title_map={
            "features_alias": "Features",
            "model_variant": "Modelo",
            "balance_strategy": "Balance",
            "previous_run_id": "Run previo",
            "latest_run_id": "Run último",
            "delta_true_positive": "Δ TP",
            "delta_false_positive": "Δ FP",
            "delta_mcc": "Δ MCC",
            "delta_pr_auc": "Δ PR-AUC",
        },
        numeric_columns={"delta_mcc", "delta_pr_auc"},
        integer_columns={"delta_true_positive", "delta_false_positive"},
    )

    best_features_items = "".join(
        (
            "<li><strong>"
            + _html_escape(row["features_alias"])
            + ":</strong> "
            + _html_escape(row["model_variant"])
            + " con TP="
            + _format_int_es(row["true_positive"])
            + ", FP="
            + _format_int_es(row["false_positive"])
            + ", MCC="
            + _format_decimal_es(row["mcc"])
            + " y PR-AUC="
            + _format_decimal_es(row["pr_auc"])
            + ".</li>"
        )
        for _, row in best_by_features.iterrows()
    )

    duplicate_commentary = (
        "Los contextos repetidos se conservaron íntegramente en el CSV y el análisis principal filtró solo la corrida más reciente por contexto."
        if not duplicate_deltas_df.empty
        else "No hubo contextos repetidos en esta ventana de análisis."
    )

    champion_description = (
        f"{champion_row['model_variant']} sobre {_normalize_features_alias(str(champion_row['features_path']))}, "
        f"con threshold {champion_row['threshold_objective']}, calibración {champion_row['calibration_method']} "
        f"y balance {champion_row['balance_strategy']}. Alcanzó TP={_format_int_es(champion_row['true_positive'])}, "
        f"FP={_format_int_es(champion_row['false_positive'])}, MCC={_format_decimal_es(champion_row['mcc'])}, "
        f"PR-AUC={_format_decimal_es(champion_row['pr_auc'])}, F1 global={_format_decimal_es(champion_row['f1_global'])} "
        f"y FAR={_format_decimal_es(champion_row['far'])}."
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crash Prediction · History 2026-04-19 a 2026-04-20</title>
  <style>
  {css}
  .report-page {{
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr) 220px;
    gap: 48px;
    max-width: 1280px;
    margin: 0 auto;
    padding: 48px 32px 128px;
  }}
  .report-main {{
    max-width: var(--measure);
    min-width: 0;
  }}
  .report-meta-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px 24px;
    margin: 24px 0 40px;
    padding: 18px 0;
    border-top: 0.5px solid var(--rule);
    border-bottom: 0.5px solid var(--rule);
  }}
  .report-meta-item {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .report-meta-item .k {{
    font-family: var(--font-sans);
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-3);
  }}
  .report-meta-item .v {{
    font-family: var(--font-serif);
    font-size: 16px;
    color: var(--ink);
  }}
  .report-table {{
    width: 100%;
    border-collapse: collapse;
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--ink);
    font-family: var(--font-sans);
    font-size: 12.5px;
    line-height: 1.45;
    margin: 16px 0 32px;
  }}
  .report-table th,
  .report-table td {{
    padding: 10px 10px;
    border-bottom: 0.5px solid var(--rule);
    vertical-align: top;
    text-align: left;
  }}
  .report-table thead th {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-3);
  }}
  .table-scroll {{
    overflow-x: auto;
  }}
  .chart-card {{
    margin: 24px 0 40px;
  }}
  .chart-caption {{
    font-family: var(--font-sans);
    font-size: 12.5px;
    line-height: 1.55;
    color: var(--ink-3);
    margin-top: 10px;
  }}
  .report-list {{
    margin: 0 0 24px 22px;
    padding: 0;
  }}
  .report-list li {{
    margin: 0 0 10px;
    font-family: var(--font-serif);
    font-size: 17px;
    line-height: 1.7;
  }}
  .report-code {{
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--paper-2);
    border: 1px solid var(--rule);
    padding: 2px 6px;
    border-radius: 2px;
  }}
  @media (max-width: 1100px) {{
    .report-page {{
      grid-template-columns: 1fr;
      gap: 24px;
      padding: 24px 18px 72px;
    }}
    .m-toc, .m-margin {{
      position: static;
      max-height: none;
    }}
  }}
  </style>
</head>
<body>
  <header class="m-topbar">
    <span class="mono">Crash Prediction</span>
    <span class="title">History consolidation and model selection</span>
    <span class="spacer"></span>
    <span class="mono">{_html_escape(csv_path.name)}</span>
  </header>

  <div class="report-page">
    <aside class="m-toc" aria-label="Contenido">
      <div class="toc-label">Contenido</div>
      <a href="#resumen">Resumen</a>
      <a href="#tabla-modelos">Tabla por features y modelo</a>
      <a href="#grafico-pareto">TP vs FP</a>
      <a href="#grafico-barras">Comparativo TP/FP</a>
      <a href="#grafico-metricas">Mapa de métricas</a>
      <a href="#repetidos">Repetidos</a>
      <a href="#hallazgos">Hallazgos</a>
    </aside>

    <main class="report-main">
      <div class="m-eyebrow">Crash prediction · History tab</div>
      <h1 class="m-title">Resultados consolidados de entrenamientos entre 2026-04-19 y 2026-04-20</h1>
      <p class="m-subtitle">Consolidación de 7 corridas del historial para dos conjuntos de <span class="report-code">features_path</span>, preservando repetidos y seleccionando el campeón operativo desde la vista deduplicada más reciente.</p>

      <div class="report-meta-grid">
        <div class="report-meta-item"><span class="k">Fuente</span><span class="v">{_html_escape(str(model_history_dir))}</span></div>
        <div class="report-meta-item"><span class="k">CSV consolidado</span><span class="v">{_html_escape(str(csv_path))}</span></div>
        <div class="report-meta-item"><span class="k">Runs analizados</span><span class="v">{_format_int_es(full_history_df['run_id'].nunique())}</span></div>
        <div class="report-meta-item"><span class="k">Filas históricas</span><span class="v">{_format_int_es(len(full_history_df))}</span></div>
        <div class="report-meta-item"><span class="k">Vista latest_context</span><span class="v">{_format_int_es(len(latest_context_df))}</span></div>
        <div class="report-meta-item"><span class="k">Features_path distintos</span><span class="v">{_format_int_es(features_count)}</span></div>
        <div class="report-meta-item"><span class="k">Contextos repetidos</span><span class="v">{_format_int_es(duplicate_deltas_df.shape[0])}</span></div>
        <div class="report-meta-item"><span class="k">Frontera de Pareto</span><span class="v">{_format_int_es(global_frontier_count)} configuraciones</span></div>
      </div>

      <section class="m-section" id="resumen">
        <div class="section-num">§ 1</div>
        <h2>Resumen operativo</h2>
        <p>El reporte consolida 21 resultados de test, una fila por <span class="report-code">run_id × modelo</span>, con los tres variantes <span class="report-code">Base</span>, <span class="report-code">Cluster</span> y <span class="report-code">Base+Cluster</span>. La lógica de trazabilidad conserva todas las corridas y marca los contextos repetidos mediante <span class="report-code">context_key</span>, tamaño de grupo y ranking temporal.</p>
        <div class="m-callout sage">
          <div class="label">Campeón global</div>
          <p>{_html_escape(champion_description)}</p>
        </div>
        <p>La selección del campeón se hizo sobre la vista <span class="report-code">latest_context_view</span>, filtrando solo la versión más reciente de cada contexto repetido. Primero se construyó el frente de Pareto con <strong>true positives altos</strong> y <strong>false positives bajos</strong>; dentro de ese frente se desempató por <span class="report-code">mcc</span>, luego <span class="report-code">pr_auc</span>, luego <span class="report-code">f1_global</span> y finalmente <span class="report-code">far</span>.</p>
        <p>{_html_escape(duplicate_commentary)}</p>
      </section>

      <section class="m-section" id="tabla-modelos">
        <div class="section-num">§ 2</div>
        <h2>Mejor configuración por features_path y modelo</h2>
        <p>La tabla resume, para cada combinación de <span class="report-code">features_path</span> y variante de modelo, la mejor configuración disponible en la vista deduplicada.</p>
        {summary_table}
      </section>

      <section class="m-section" id="grafico-pareto">
        <div class="section-num">§ 3</div>
        <h2>TP vs FP en la vista deduplicada</h2>
        <div class="chart-card">
          {scatter_html}
          <div class="chart-caption"><strong>Figura 3.1.</strong> Dispersión de <em>false positives</em> vs <em>true positives</em>. El anillo negro marca el frente de Pareto y la estrella identifica el campeón global.</div>
        </div>
      </section>

      <section class="m-section" id="grafico-barras">
        <div class="section-num">§ 4</div>
        <h2>Comparativo directo de alertas correctas y falsas alarmas</h2>
        <div class="chart-card">
          {bar_html}
          <div class="chart-caption"><strong>Figura 4.1.</strong> Comparación horizontal de cada configuración vigente. La lectura combinada permite evaluar el costo operativo del recall obtenido.</div>
        </div>
      </section>

      <section class="m-section" id="grafico-metricas">
        <div class="section-num">§ 5</div>
        <h2>Matriz comparativa de métricas de soporte</h2>
        <div class="chart-card">
          {heatmap_html}
          <div class="chart-caption"><strong>Figura 5.1.</strong> Heatmap de ranking relativo por métrica. El color resume posición relativa por columna y el <em>tooltip</em> muestra el valor real de test.</div>
        </div>
      </section>

      <section class="m-section" id="repetidos">
        <div class="section-num">§ 6</div>
        <h2>Comparación de corridas repetidas</h2>
        <p>Se detectaron contextos duplicados en el <span class="report-code">features_path</span> 2022-01 a 2024-09, tanto para <span class="report-code">balance_strategy = none</span> como para <span class="report-code">class_weight</span>. La comparación siguiente muestra el cambio entre la corrida previa y la más reciente por contexto.</p>
        <div class="chart-card">
          {duplicate_chart_html}
          <div class="chart-caption"><strong>Figura 6.1.</strong> Delta entre última corrida y corrida previa para contextos repetidos. Δ TP positivo es mejora; Δ FP negativo también es mejora.</div>
        </div>
        {duplicate_table}
      </section>

      <section class="m-section" id="hallazgos">
        <div class="section-num">§ 7</div>
        <h2>Hallazgos y lectura final</h2>
        <h3>Ganadores por features_path</h3>
        <ul class="report-list">
          {best_features_items}
        </ul>
        <h3>Conclusión</h3>
        <p>El ganador global prioriza desempeño operacional sobre pureza de una métrica aislada: captura más eventos positivos útiles manteniendo las falsas alarmas en un rango competitivo frente al resto de configuraciones vigentes. Las métricas de soporte confirman la decisión, especialmente <span class="report-code">MCC</span> y <span class="report-code">PR-AUC</span>, que permanecen consistentes con el criterio TP/FP.</p>
        <p>La existencia de repetidos en el set 2024-09 muestra que el comportamiento no es completamente estable entre corridas. Por eso el CSV consolidado conserva el historial completo y el HTML diferencia explícitamente entre <span class="report-code">full_history</span> y <span class="report-code">latest_context_view</span>.</p>
      </section>
    </main>

    <aside class="m-margin">
      <div class="margin-label">Metodología</div>
      <p>Ventana temporal fija: 2026-04-19 a 2026-04-20.</p>
      <p>Los conteos absolutos TP/FP son comparables porque todos los candidatos comparten <span class="report-code">test_rows = 49\u202f793</span> y <span class="report-code">positive_support = 56</span>.</p>
      <p>Diseño visual inspirado en el sistema local <span class="report-code">tesis-doctoral-design</span> y su kit de manuscrito.</p>
    </aside>
  </div>
</body>
</html>
"""


def generate_report(
    *,
    model_history_dir: Path = MODEL_HISTORY_DIR,
    results_dir: Path = RESULTS_DIR,
    run_ids: Sequence[str] = DEFAULT_RUN_IDS,
    design_root: Path = THESIS_DESIGN_ROOT,
    output_stem: str = OUTPUT_STEM,
) -> ReportArtifacts:
    full_history_df = build_history_dataframe(model_history_dir=model_history_dir, run_ids=run_ids)
    latest_context_df = full_history_df.loc[full_history_df["is_latest_for_context"]].copy().reset_index(drop=True)
    latest_context_df["is_pareto_frontier"] = _pareto_frontier_mask(latest_context_df)
    champion_row = select_champion(latest_context_df)
    group_best_df = _select_best_per_group(latest_context_df, ["features_path", "model_variant"])
    duplicate_deltas_df = _build_duplicate_deltas(full_history_df)

    csv_path = results_dir / f"{output_stem}.csv"
    html_path = results_dir / f"{output_stem}.html"

    csv_df = full_history_df.loc[:, [column for column in CSV_COLUMN_ORDER if column in full_history_df.columns]].copy()
    csv_df.to_csv(csv_path, index=False)

    html_content = _build_report_html(
        full_history_df=full_history_df,
        latest_context_df=latest_context_df,
        champion_row=champion_row,
        group_best_df=group_best_df,
        duplicate_deltas_df=duplicate_deltas_df,
        design_root=design_root,
        model_history_dir=model_history_dir,
        csv_path=csv_path,
    )
    html_path.write_text(html_content, encoding="utf-8")

    return ReportArtifacts(
        csv_path=csv_path,
        html_path=html_path,
        full_history_df=full_history_df,
        latest_context_df=latest_context_df,
        champion_row=champion_row,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolida resultados de Crash Prediction y genera un HTML de análisis.")
    parser.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        default=None,
        help="Run ID a incluir. Puede repetirse; si no se pasa, usa la lista por defecto.",
    )
    parser.add_argument(
        "--model-history-dir",
        type=Path,
        default=MODEL_HISTORY_DIR,
        help="Directorio base con los bundles de model_history.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directorio donde se escribirán CSV y HTML.",
    )
    parser.add_argument(
        "--design-root",
        type=Path,
        default=THESIS_DESIGN_ROOT,
        help="Raíz del sistema visual tesis-doctoral-design.",
    )
    parser.add_argument(
        "--output-stem",
        default=OUTPUT_STEM,
        help="Prefijo del CSV y HTML de salida, sin extensión.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_ids = args.run_ids or DEFAULT_RUN_IDS
    artifacts = generate_report(
        model_history_dir=args.model_history_dir,
        results_dir=args.results_dir,
        run_ids=run_ids,
        design_root=args.design_root,
        output_stem=args.output_stem,
    )
    print(f"CSV: {artifacts.csv_path}")
    print(f"HTML: {artifacts.html_path}")
    print(f"Filas históricas: {len(artifacts.full_history_df)}")
    print(f"Vista latest_context: {len(artifacts.latest_context_df)}")
    print(
        "Campeón: "
        f"{artifacts.champion_row['model_variant']} | {artifacts.champion_row['run_id']} | "
        f"TP={int(artifacts.champion_row['true_positive'])} | "
        f"FP={int(artifacts.champion_row['false_positive'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
