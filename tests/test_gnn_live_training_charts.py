import pytest
import json
import zipfile

from src import graph_builder_app as app


def test_saved_balanced_graph_listing_skips_truncated_torch_archives(tmp_path, monkeypatch):
    valid_path = tmp_path / "graph_imgagn_relational_valid.pt"
    corrupt_path = tmp_path / "graph_imgagn_relational_corrupt.pt"
    app.torch.save({"data": app.HeteroData()}, valid_path)
    with zipfile.ZipFile(corrupt_path, "w") as zf:
        zf.writestr("graph_imgagn_relational_corrupt/version", "3\n")
        zf.writestr("graph_imgagn_relational_corrupt/byteorder", "little")

    monkeypatch.setattr(app, "RESULTADOS_DIR", str(tmp_path))

    valid, invalid = app._list_saved_balanced_graph_files()

    assert valid == [str(valid_path)]
    assert invalid == [str(corrupt_path)]


def test_atomic_torch_save_does_not_replace_destination_with_incomplete_archive(tmp_path, monkeypatch):
    target = tmp_path / "graph_imgagn_relational_target.pt"
    app.torch.save({"data": app.HeteroData()}, target)
    original_size = target.stat().st_size

    def fake_save(_obj, path):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("bad/version", "3\n")
            zf.writestr("bad/byteorder", "little")

    monkeypatch.setattr(app.torch, "save", fake_save)

    with pytest.raises(RuntimeError, match="data.pkl"):
        app._torch_save_atomic({"data": app.HeteroData()}, target)

    assert target.stat().st_size == original_size
    assert not list(tmp_path.glob(".*.tmp"))


def test_gnn_live_training_chart_frames_keep_requested_series_only():
    metrics = [
        {
            "epoch": 2,
            "train_loss": 0.42,
            "val_loss": 0.55,
            "train_cls_loss": 0.40,
            "train_edge_loss": 0.01,
            "train_l2_att_loss": 0.02,
            "train_ranking_loss": 0.03,
            "val_accuracy": 0.88,
            "val_recall_pos": 0.61,
            "val_precision_pos": 0.73,
            "val_f1_macro": 0.69,
            "val_auc": 0.91,
            "val_auprc": 0.37,
            "val_mcc": 0.24,
            "val_far": 0.08,
            "val_objective_score": 0.31,
            "monitor_value": 0.37,
            "best_monitor_value": 0.39,
            "val_tau": 0.52,
        },
        {
            "epoch": 1,
            "train_loss": 0.50,
            "val_loss": 0.60,
            "train_cls_loss": 0.48,
            "train_ranking_loss": 0.04,
            "val_accuracy": 0.84,
            "val_recall_pos": 0.57,
            "val_precision_pos": 0.70,
            "val_f1_macro": 0.65,
            "val_auprc": 0.32,
            "val_mcc": 0.20,
            "val_far": 0.10,
        },
    ]

    frames = app._build_gnn_live_training_chart_frames(metrics)

    assert list(frames["loss"].columns) == ["Train Loss", "Rank Loss", "Val Loss"]
    assert "train_cls_loss" not in frames["loss"].columns
    assert "train_edge_loss" not in frames["loss"].columns
    assert "train_l2_att_loss" not in frames["loss"].columns
    assert frames["loss"].loc[2, "Rank Loss"] == pytest.approx(0.03)

    assert list(frames["classification"].columns) == [
        "Accuracy",
        "Recall",
        "Precision",
        "F1_macro",
    ]
    assert list(frames["ranking"].columns) == ["AUPRC", "MCC"]
    assert "val_auc" not in frames["ranking"].columns
    assert "val_objective_score" not in frames["ranking"].columns
    assert "monitor_value" not in frames["ranking"].columns

    assert list(frames["operational"].columns) == ["FAR", "TPR"]
    assert frames["operational"].loc[2, "TPR"] == pytest.approx(0.61)
    assert frames["operational"].loc[2, "FAR"] == pytest.approx(0.08)


def test_gnn_training_history_loader_ignores_bad_lines_and_preserves_curves(tmp_path):
    history_path = tmp_path / "metrics_history.jsonl"
    rows = [
        {
            "scope": "gnn_training",
            "event": "train_start",
            "run_id": "run-a",
        },
        {
            "scope": "gnn_training",
            "event": "epoch",
            "run_id": "run-a",
            "epoch": 2,
            "train_loss": 0.4,
            "train_ranking_loss": 0.04,
            "val_loss": 0.5,
            "val_auprc": 0.2,
        },
        "{bad json",
        {
            "scope": "gnn_training",
            "event": "epoch",
            "run_id": "run-a",
            "epoch": 1,
            "train_loss": 0.6,
            "train_ranking_loss": 0.06,
            "val_loss": 0.7,
            "val_auprc": 0.1,
        },
        {
            "scope": "gnn_training",
            "event": "epoch",
            "run_id": "run-a",
            "epoch": 2,
            "train_loss": 0.3,
            "train_ranking_loss": 0.03,
            "val_loss": 0.45,
            "val_auprc": 0.25,
        },
        {
            "scope": "gnn_training",
            "event": "test_result",
            "run_id": "run-a",
            "epoch": 2,
            "eval_target": "current_epoch",
            "automatic": True,
            "auprc": 0.22,
        },
    ]
    history_path.write_text(
        "\n".join(
            row if isinstance(row, str) else json.dumps(row)
            for row in rows
        ),
        encoding="utf-8",
    )

    metrics, test_results, run_id = app._load_gnn_training_history(history_path)
    frames = app._build_gnn_live_training_chart_frames(
        metrics
        + [
            {
                "epoch": 3,
                "train_loss": 0.2,
                "train_ranking_loss": 0.02,
                "val_loss": 0.4,
                "val_auprc": 0.3,
            }
        ]
    )

    assert run_id == "run-a"
    assert [row["epoch"] for row in metrics] == [1, 2]
    assert metrics[1]["train_loss"] == pytest.approx(0.3)
    assert len(test_results) == 1
    assert test_results[0]["eval_target"] == "current_epoch"
    assert list(frames["loss"].index) == [1, 2, 3]
    assert frames["loss"].loc[2, "Train Loss"] == pytest.approx(0.3)
    assert frames["loss"].loc[2, "Rank Loss"] == pytest.approx(0.03)
    assert frames["ranking"].loc[3, "AUPRC"] == pytest.approx(0.3)
