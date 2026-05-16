import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import graph_builder_app as app


def test_network_hparams_are_discoverable_for_training_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RESULTADOS_DIR", str(tmp_path))
    graph_hash = "a" * 64
    graph_obj = {"graph_hash": graph_hash, "filename": "demo_graph.pt"}

    path = app._save_network_hparams(
        {"gnn_variant": "gat_gru", "hidden_channels": 64},
        use_graphsmote=False,
        graph_obj=graph_obj,
    )

    assert path is not None
    assert f"_{graph_hash[:16]}" in os.path.basename(path)

    row = pd.read_csv(path).iloc[0].to_dict()
    assert row["graph_hash"] == graph_hash
    assert row["hparams_source"] == "Network"
    assert row["optimizer"] == "AdamW"
    assert row["lr_scheduler"] == "one_cycle"
    assert row["lr"] == 3e-4
    assert row["weight_decay"] == 1e-4
    assert row["checkpoint_metric"] == "val_auprc"
    assert bool(row["use_residual"]) is True
    assert bool(row["use_relation_self_loops"]) is False

    graph_filtered = app._list_hpo_files_for_training(
        use_graphsmote=False,
        graph_obj=graph_obj,
    )
    prompt_files = app._list_hpo_files_for_gnn_training_prompt(
        use_graphsmote=False,
        graph_obj=graph_obj,
    )

    assert path in graph_filtered
    assert path in prompt_files


def test_network_exports_sequence_length_from_loaded_graph_sequence_index():
    sequence_index = SimpleNamespace(
        sequence_rows=np.asarray(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
            ],
            dtype=np.int64,
        ),
        target_rows=np.asarray([3, 4, 5], dtype=np.int64),
    )

    seq_stats = app._graph_sequence_stats({"sequence_index": sequence_index})
    params = app._network_config_to_hparams(
        {"seq_length": seq_stats["sequence_length"]},
        use_graphsmote=False,
    )

    assert seq_stats["sequence_length"] == 4
    assert seq_stats["sequence_count"] == 3
    assert seq_stats["source"] == "SequenceIndex"
    assert params["seq_length"] == 4


def test_graph_node_feature_preview_accepts_dataframe_columns_index():
    df_pm_cache = pd.DataFrame(
        {
            "flow": [1.0, 2.0],
            "speed": [50.0, 55.0],
            "target": [0, 1],
            "Portico": ["A", "B"],
        }
    )

    available, source_label, error = app._available_graph_node_feature_columns(
        None, df_pm_cache
    )

    assert error is None
    assert source_label == "features en memoria"
    assert available == ["flow", "speed"]
