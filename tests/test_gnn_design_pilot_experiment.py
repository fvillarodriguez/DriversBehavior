import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import graph_builder_app as app
from src.graph import PMIndex
from src.snapshot_sequences import SequenceConfig, SequenceIndex


def _make_pm_index(num_nodes: int) -> PMIndex:
    pm_map = {(f"P{idx % 4}", idx): idx for idx in range(num_nodes)}
    pm_rev = {idx: (f"P{idx % 4}", idx) for idx in range(num_nodes)}
    return PMIndex(pm_map, pm_rev)


def _make_design_graph(num_nodes: int = 60):
    HeteroData = app.HeteroData
    data = HeteroData()
    data["pm"].x = torch.arange(num_nodes * 3, dtype=torch.float32).view(num_nodes, 3)
    y = torch.zeros(num_nodes, dtype=torch.long)
    positive_idx = [idx for idx in [4, 8, 12, 18, 32, 38, 52] if idx < num_nodes]
    if positive_idx:
        y[positive_idx] = 1
    data["pm"].y = y
    data["pm"].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data["pm"].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data["pm"].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data["pm"].train_mask[:30] = True
    data["pm"].val_mask[30:45] = True
    data["pm"].test_mask[45:] = True
    data["pm"].sequence_mask = torch.ones(num_nodes, dtype=torch.bool)
    edge_src = torch.arange(0, num_nodes - 1, dtype=torch.long)
    edge_dst = edge_src + 1
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_attr = torch.stack(
        [edge_src.to(torch.float32), edge_dst.to(torch.float32)],
        dim=1,
    )
    data["pm", "spatial", "pm"].edge_index = edge_index
    data["pm", "spatial", "pm"].edge_attr = edge_attr
    data["pm", "temporal", "pm"].edge_index = edge_index.flip(0)
    data["pm", "temporal", "pm"].edge_attr = edge_attr + 100.0
    data["pm"].num_nodes = num_nodes
    return data


def test_select_stratified_pilot_nodes_excludes_test_and_preserves_positives():
    data = _make_design_graph()
    pm_index = _make_pm_index(data["pm"].num_nodes)

    selection = app._select_stratified_pilot_pm_nodes(
        data,
        pm_index,
        pilot_fraction=0.10,
        seed=7,
        time_bins=4,
        min_train_pos=3,
        min_val_pos=2,
    )

    selected = selection["selected_nodes"]
    assert not data["pm"].test_mask[selected].any()
    counts = selection["metadata"]["selected_counts"]
    assert counts["train"]["positive"] >= 3
    assert counts["val"]["positive"] >= 2


def test_build_pm_induced_heterodata_remaps_edges_attrs_and_pm_index():
    data = _make_design_graph(num_nodes=8)
    pm_nodes = torch.tensor([0, 1, 2, 5])

    induced, meta = app._build_pm_induced_heterodata(
        data,
        pm_nodes,
        supervised_nodes=torch.tensor([0, 2]),
    )

    assert induced["pm"].num_nodes == 4
    assert induced["pm"].n_id.tolist() == [0, 1, 2, 5]
    assert induced["pm"].test_mask.sum().item() == 0
    assert induced["pm"].train_mask.tolist() == [True, False, True, False]
    assert induced["pm", "spatial", "pm"].edge_index.tolist() == [[0, 1], [1, 2]]
    assert induced["pm", "spatial", "pm"].edge_attr.tolist() == [[0.0, 1.0], [1.0, 2.0]]
    assert meta["edge_counts"]["('pm', 'spatial', 'pm')"] == 2

    local_pm_index = app._remap_pm_index_for_pilot(_make_pm_index(8), meta["local_to_global"])
    assert local_pm_index is not None
    assert local_pm_index._rev[2] == ("P2", 2)
    assert local_pm_index._map[("P1", 1)] == 1


def test_remap_sequence_index_for_pilot_filters_invalid_sequences():
    seq = SequenceIndex(
        sequence_rows=np.asarray([[0, 1, 2], [2, 3, 4], [5, 6, 7]], dtype=np.int64),
        target_rows=np.asarray([2, 4, 7], dtype=np.int64),
        labels=np.asarray([0, 1, 1], dtype=np.int8),
        porticos=np.asarray(["A", "B", "C"], dtype=object),
        target_ts_min=np.asarray([2, 4, 7], dtype=np.int64),
        config=SequenceConfig(sequence_length=3),
    )
    remapped = app._remap_sequence_index_for_pilot(
        seq,
        {0: 0, 1: 1, 2: 2, 5: 3, 6: 4, 7: 5},
    )

    assert remapped is not None
    assert remapped.sequence_rows.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert remapped.target_rows.tolist() == [2, 5]
    assert remapped.labels.tolist() == [0, 1]


def test_architecture_pilot_runs_optuna_on_pilot_and_full_train_on_original(
    tmp_path,
    monkeypatch,
):
    data = _make_design_graph(num_nodes=60)
    graph_obj = {
        "data": data,
        "filename": "demo_graph.pt",
        "pm_index": _make_pm_index(60),
        "sequence_index": None,
        "metadata": {"label_lookahead_minutes": 0},
    }
    monkeypatch.setattr(app, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(app, "HISTORY_PATH", Path(tmp_path) / "gnn_history.jsonl")
    monkeypatch.setattr(app, "_append_gnn_history", lambda entry: None)

    optuna_calls = []
    train_calls = []

    def fake_run_optuna_search(**kwargs):
        pilot_data = kwargs["graph_obj"]["data"]
        optuna_calls.append(int(pilot_data["pm"].num_nodes))
        assert int(pilot_data["pm"].num_nodes) < int(data["pm"].num_nodes)
        assert int(pilot_data["pm"].test_mask.sum().item()) == 0
        return {
            "best_params": {
                "value": 0.5,
                "gnn_variant": kwargs["objective_settings"]["gnn_variant"],
                "batch_size": 8,
                "num_neighbors": "[5, 3]",
                "train_sampler_mode": "neighbor",
                "checkpoint_metric": "val_auprc",
                "val_f1": 0.12,
                "val_precision": 0.2,
                "val_recall": 0.3,
                "val_far": 0.05,
                "val_auprc": 0.4,
                "val_auc": 0.7,
                "val_mcc": 0.1,
                "val_loss": 0.9,
                "best_val_tau": 0.6,
                "best_epoch": 2,
            },
            "best_path": str(tmp_path / "pilot_hparams.csv"),
            "full_path": str(tmp_path / "pilot_full.csv"),
            "range_analysis": {},
        }

    def fake_train_gnn_with_best_params(graph_obj, best_params_path, **kwargs):
        train_calls.append(int(graph_obj["data"]["pm"].num_nodes))
        return str(tmp_path / f"model_{len(train_calls)}.pt")

    def fake_eval(**kwargs):
        return {
            "threshold": 0.7,
            "calibration": {"far": 0.05, "sens": 0.4},
            "report": {
                "Accidente (1)": {
                    "f1-score": 0.2,
                    "precision": 0.25,
                    "recall": 0.3,
                },
                "accuracy": 0.9,
            },
            "metrics": {
                "far": 0.04,
                "false_alarm_ratio": 0.04,
                "specificity": 0.96,
                "brier_score": 0.1,
            },
            "auprc": 0.22,
            "auc": 0.74,
            "mcc": 0.18,
            "confusion_matrix": [[10, 1], [2, 1]],
        }

    monkeypatch.setattr(app, "_run_optuna_search", fake_run_optuna_search)
    monkeypatch.setattr(app, "_train_gnn_with_best_params", fake_train_gnn_with_best_params)
    monkeypatch.setattr(app, "_evaluate_gnn_model_far_target", fake_eval)

    result = app._run_gnn_architecture_pilot_experiment(
        graph_obj=graph_obj,
        graph_data=data,
        pm_index_ref=graph_obj["pm_index"],
        graph_source_label="Original",
        train_ratio=50,
        val_ratio=25,
        test_ratio=25,
        balancing_strategies=["Sin balancear"],
        objective_metrics=["AUPRC"],
        selected_variants=["gat_snapshot"],
        search_space={},
        optuna_settings={"n_trials": 1},
        threshold_beta=1.0,
        far_target=0.2,
        max_epochs_train=2,
        early_stop_train=True,
        early_patience_train=1,
        early_min_delta_train=0.0,
        pilot_fraction=0.2,
        pilot_seed=11,
        time_bins=4,
        min_train_pos=2,
        min_val_pos=1,
        top_k_finalists=1,
        full_repeats=2,
        render_ui=False,
    )

    assert optuna_calls and optuna_calls[0] < 60
    assert train_calls == [60, 60]
    assert len(result["pilot_rows"]) == 1
    assert len(result["full_rows"]) == 2
    assert Path(result["metadata_path"]).exists()
    assert Path(result["results_path"]).exists()
    rows = pd.read_csv(result["results_path"])
    assert set(rows["role"]) == {"pilot_candidate", "full_finalist"}
    meta = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert meta["selection"]["selected_counts"]["test"]["total"] == 0
