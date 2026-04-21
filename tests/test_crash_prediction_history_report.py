import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import src.crash_prediction_history_report as report


def _write_manifest(
    *,
    base_dir: Path,
    run_id: str,
    strategy_label: str,
    saved_at: str,
    features_path: str,
    threshold_objective: str,
    calibration_method: str,
    balance_strategy: str,
    tp: int,
    fp: int,
    mcc: float,
    pr_auc: float,
    recall: float,
    far: float,
    f1_global: float = 0.4,
    k: int = 5,
) -> None:
    manifest_path = base_dir / run_id / strategy_label / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "strategy_label": strategy_label,
        "model_name": "XGBoost",
        "saved_at": saved_at,
        "features_path": features_path,
        "selected_features": [f"f{i}" for i in range(k)],
        "split_info": {"test_rows": 49793},
        "dataset": {
            "tramo": {
                "label": "RUTA 5 NORTE | Oriente | 16 -> 17",
            }
        },
        "metrics": {
            "accuracy": 0.9,
            "precision": 0.1,
            "recall": recall,
            "sensitivity": recall,
            "f1": 0.2,
            "f1_global": f1_global,
            "far": far,
            "roc_auc": 0.7,
            "pr_auc": pr_auc,
            "brier_score": 0.01,
            "mcc": mcc,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": 100,
            "false_negatives": 20,
            "positive_support": 56,
            "threshold_protocol": "robust",
            "threshold_objective": threshold_objective,
            "calibration_method": calibration_method,
            "balance_strategy": balance_strategy,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_design_root(design_root: Path) -> None:
    (design_root / "ui_kits" / "manuscript").mkdir(parents=True, exist_ok=True)
    (design_root / "colors_and_type.css").write_text(
        ":root { --paper: #fff; --paper-2: #f6f3ee; --rule: #d8d5cd; --ink: #111; --ink-2: #333; --ink-3: #666; --accent: #1b3a6b; --ok: #3b7a57; --warn: #b08a2e; --err: #a8493d; --font-serif: Georgia, serif; --font-sans: Arial, sans-serif; --font-mono: Menlo, monospace; --measure: 680px; --measure-wide: 900px; --ease-out: ease-out; }",
        encoding="utf-8",
    )
    (design_root / "ui_kits" / "manuscript" / "manuscript.css").write_text(
        ".m-topbar{display:flex}.m-toc,.m-margin{font-size:12px}.m-title{font-size:40px}.m-section{margin-top:32px}",
        encoding="utf-8",
    )


def test_build_history_dataframe_marks_duplicates_and_latest(tmp_path):
    model_history_dir = tmp_path / "model_history"
    features = "/tmp/accident_flow_features_cluster_gmm_k5_(2022_01-2024_09).duckdb"

    for strategy_label, tp_latest, tp_prev in [
        ("base", 32, 30),
        ("cluster", 20, 18),
        ("base_cluster", 28, 27),
    ]:
        _write_manifest(
            base_dir=model_history_dir,
            run_id="20260420_latest",
            strategy_label=strategy_label,
            saved_at="2026-04-20T14:55:45",
            features_path=features,
            threshold_objective="far",
            calibration_method="isotonic",
            balance_strategy="class_weight",
            tp=tp_latest,
            fp=8000,
            mcc=0.03,
            pr_auc=0.004,
            recall=0.5,
            far=0.16,
        )
        _write_manifest(
            base_dir=model_history_dir,
            run_id="20260419_previous",
            strategy_label=strategy_label,
            saved_at="2026-04-19T19:41:13",
            features_path=features,
            threshold_objective="far",
            calibration_method="isotonic",
            balance_strategy="class_weight",
            tp=tp_prev,
            fp=9000,
            mcc=0.02,
            pr_auc=0.003,
            recall=0.4,
            far=0.18,
        )

    df = report.build_history_dataframe(
        model_history_dir=model_history_dir,
        run_ids=["20260420_latest", "20260419_previous"],
    )

    assert len(df) == 6
    assert set(df["model_variant"]) == {"Base", "Cluster", "Base+Cluster"}
    assert df["is_duplicate_context"].all()
    assert set(df["duplicate_group_size"]) == {2}

    latest_base = df.loc[df["model_variant"].eq("Base") & df["run_id"].eq("20260420_latest")].iloc[0]
    previous_base = df.loc[df["model_variant"].eq("Base") & df["run_id"].eq("20260419_previous")].iloc[0]
    assert latest_base["duplicate_rank_latest"] == 1
    assert bool(latest_base["is_latest_for_context"]) is True
    assert previous_base["duplicate_rank_latest"] == 2
    assert bool(previous_base["is_latest_for_context"]) is False
    assert latest_base["k"] == 5


def test_select_champion_prioritizes_tp_fp_then_secondary_metrics():
    df = pd.DataFrame(
        [
            {
                "run_id": "A",
                "saved_at": "2026-04-20T10:00:00",
                "true_positive": 28,
                "false_positive": 5,
                "mcc": 0.3,
                "pr_auc": 0.2,
                "f1_global": 0.4,
                "far": 0.1,
            },
            {
                "run_id": "B",
                "saved_at": "2026-04-20T10:05:00",
                "true_positive": 30,
                "false_positive": 10,
                "mcc": 0.2,
                "pr_auc": 0.19,
                "f1_global": 0.39,
                "far": 0.2,
            },
            {
                "run_id": "C",
                "saved_at": "2026-04-20T10:10:00",
                "true_positive": 30,
                "false_positive": 10,
                "mcc": 0.5,
                "pr_auc": 0.21,
                "f1_global": 0.41,
                "far": 0.19,
            },
            {
                "run_id": "D",
                "saved_at": "2026-04-20T10:15:00",
                "true_positive": 25,
                "false_positive": 4,
                "mcc": 0.9,
                "pr_auc": 0.8,
                "f1_global": 0.8,
                "far": 0.05,
            },
        ]
    )

    champion = report.select_champion(df)

    assert champion["run_id"] == "C"


def test_generate_report_writes_csv_and_html(tmp_path):
    model_history_dir = tmp_path / "model_history"
    results_dir = tmp_path / "Resultados"
    design_root = tmp_path / "tesis-doctoral-design"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_design_root(design_root)

    specs = [
        ("20260420_latest", "2026-04-20T23:00:00", "/tmp/a.duckdb", "far", "sigmoid", "class_weight", 34, 110, 0.04, 0.02, 0.6, 0.01),
        ("20260419_previous", "2026-04-19T23:00:00", "/tmp/a.duckdb", "far", "sigmoid", "class_weight", 30, 140, 0.03, 0.01, 0.5, 0.02),
    ]
    for run_id, saved_at, features_path, threshold_objective, calibration_method, balance_strategy, tp, fp, mcc, pr_auc, recall, far in specs:
        for strategy_label, delta_tp in [("base", 0), ("cluster", -4), ("base_cluster", 2)]:
            _write_manifest(
                base_dir=model_history_dir,
                run_id=run_id,
                strategy_label=strategy_label,
                saved_at=saved_at,
                features_path=features_path,
                threshold_objective=threshold_objective,
                calibration_method=calibration_method,
                balance_strategy=balance_strategy,
                tp=tp + delta_tp,
                fp=fp + abs(delta_tp) * 10,
                mcc=mcc,
                pr_auc=pr_auc,
                recall=recall,
                far=far,
            )

    artifacts = report.generate_report(
        model_history_dir=model_history_dir,
        results_dir=results_dir,
        run_ids=["20260420_latest", "20260419_previous"],
        design_root=design_root,
        output_stem="demo_report",
    )

    assert artifacts.csv_path.exists()
    assert artifacts.html_path.exists()

    csv_df = pd.read_csv(artifacts.csv_path)
    assert len(csv_df) == 6
    assert csv_df["is_duplicate_context"].all()
    assert csv_df["is_latest_for_context"].sum() == 3

    html_text = artifacts.html_path.read_text(encoding="utf-8")
    assert "Resultados consolidados de entrenamientos" in html_text
    assert "Campeón global" in html_text
    assert "demo_report.csv" in html_text
