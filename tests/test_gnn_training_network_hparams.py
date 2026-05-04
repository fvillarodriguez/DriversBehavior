import os

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
