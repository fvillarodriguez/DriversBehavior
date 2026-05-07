import pytest

from src import graph_builder_app as app


def test_gnn_live_training_chart_frames_keep_requested_series_only():
    metrics = [
        {
            "epoch": 2,
            "train_loss": 0.42,
            "val_loss": 0.55,
            "train_cls_loss": 0.40,
            "train_edge_loss": 0.01,
            "train_l2_att_loss": 0.02,
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

    assert list(frames["loss"].columns) == ["Train Loss", "Val Loss"]
    assert "train_cls_loss" not in frames["loss"].columns
    assert "train_edge_loss" not in frames["loss"].columns
    assert "train_l2_att_loss" not in frames["loss"].columns

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
