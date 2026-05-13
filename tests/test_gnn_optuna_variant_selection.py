from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from src import graph_builder_app as app


class _FakeStreamlit:
    def __init__(self, session_state=None):
        self.session_state = dict(session_state or {})

    def caption(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _make_tiny_graph() -> HeteroData:
    data = HeteroData()
    data["pm"].x = torch.zeros((8, 3), dtype=torch.float32)
    data["pm"].y = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.long)
    data["pm"].train_mask = torch.tensor(
        [True, True, True, False, False, False, False, False]
    )
    data["pm"].val_mask = torch.tensor(
        [False, False, False, True, True, False, False, False]
    )
    data["pm"].test_mask = torch.tensor(
        [False, False, False, False, False, True, True, True]
    )
    data["pm", "spatial", "pm"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    data["pm", "spatial", "pm"].edge_attr = torch.ones((6, 2), dtype=torch.float32)
    return data


def _patch_collect_settings_dependencies(monkeypatch, session_state):
    fake_st = _FakeStreamlit(session_state)
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(
        app,
        "_apply_temporal_split_to_graph",
        lambda *args, **kwargs: {
            "train_count": 3,
            "val_count": 2,
            "test_count": 3,
        },
    )
    monkeypatch.setattr(app, "_warn_legacy_temporal_split", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "get_auto_device", lambda: torch.device("cpu"))
    return fake_st


def test_collect_optuna_ray_settings_includes_selected_gnn_variant(monkeypatch):
    graph_data = _make_tiny_graph()
    graph_obj = {
        "data": graph_data,
        "filename": "tiny_graph.pt",
        "pm_index": object(),
        "sequence_index": None,
    }
    fake_st = _patch_collect_settings_dependencies(
        monkeypatch,
        {"gnn_optuna_gnn_variant": "gat_snapshot"},
    )

    settings, errors = app._collect_optuna_ray_settings(
        graph_obj=graph_obj,
        graph_data=graph_data,
    )

    assert errors == []
    assert settings["objective_settings"]["gnn_variant"] == "gat_snapshot"
    assert fake_st.session_state["gnn_optuna_gnn_variant"] == "gat_snapshot"


def test_collect_optuna_ray_settings_blocks_temporal_variant_without_sequence(
    monkeypatch,
):
    graph_data = _make_tiny_graph()
    graph_obj = {
        "data": graph_data,
        "filename": "tiny_graph.pt",
        "pm_index": object(),
        "sequence_index": None,
    }
    _patch_collect_settings_dependencies(
        monkeypatch,
        {"gnn_optuna_gnn_variant": "gat_gru"},
    )

    settings, errors = app._collect_optuna_ray_settings(
        graph_obj=graph_obj,
        graph_data=graph_data,
    )

    assert settings["objective_settings"]["gnn_variant"] == "gat_gru"
    assert any("SequenceIndex" in error for error in errors)


def test_collect_optuna_ray_settings_adds_edge_encoder_space_for_edge_mlp_gru(
    monkeypatch,
):
    graph_data = _make_tiny_graph()
    graph_obj = {
        "data": graph_data,
        "filename": "tiny_graph.pt",
        "pm_index": object(),
        "sequence_index": type(
            "SequenceIndexStub",
            (),
            {"sequence_rows": [[0, 1, 2]], "target_rows": [2]},
        )(),
    }
    _patch_collect_settings_dependencies(
        monkeypatch,
        {
            "gnn_optuna_gnn_variant": "gat_edge_mlp_gru",
            "gnn_optuna_edge_encoder_mode": "Por tipo de arista",
            "gnn_optuna_edge_kind_choices_pm__spatial__pm": [
                "mlp_residual",
                "layernorm_mlp",
            ],
            "gnn_optuna_use_residual": [False, True],
            "gnn_optuna_relation_self_loops": [False, True],
        },
    )

    settings, errors = app._collect_optuna_ray_settings(
        graph_obj=graph_obj,
        graph_data=graph_data,
    )

    assert errors == []
    search_space = settings["search_space"]
    assert search_space["use_residual"] == [False, True]
    assert search_space["use_relation_self_loops"] == [False, True]
    edge_space = search_space["edge_encoder"]
    assert edge_space["mode"] == "per_type"
    spatial_space = edge_space["per_type"]["pm__spatial__pm"]
    assert spatial_space["in_dim"] == 2
    assert spatial_space["kind_choices"] == ["mlp_residual", "layernorm_mlp"]


def test_suggest_edge_encoder_params_returns_training_serializable_metadata():
    class FakeTrial:
        def suggest_categorical(self, _name, choices):
            return list(choices)[0]

        def suggest_int(self, _name, low, _high, *, step=1):
            return int(low)

        def suggest_float(self, _name, low, _high, *, step=None):
            return float(low)

    edge_type = ("pm", "spatial", "pm")
    edge_space = {
        "mode": "per_type",
        "per_type": {
            "pm__spatial__pm": {
                "kind_choices": ["mlp_residual"],
                "hidden_dim": {"min": 4, "max": 8, "step": 4},
                "encoded_dim": {"min": 2, "max": 4, "step": 1},
                "dropout": {"min": 0.1, "max": 0.2, "step": 0.1},
            }
        },
    }

    hidden, encoded, dropouts, kinds, metadata = (
        app._suggest_edge_encoder_params_for_trial(
            FakeTrial(),
            edge_encoder_space=edge_space,
            edge_feature_dims_per_type={edge_type: 2},
        )
    )

    assert hidden[edge_type] == 4
    assert encoded[edge_type] == 2
    assert dropouts[edge_type] == 0.1
    assert kinds[edge_type] == "mlp_residual"
    assert metadata["pm__spatial__pm"]["encoded_dim"] == 2


def test_compute_optuna_study_name_separates_gnn_variants(monkeypatch):
    graph_obj = {"filename": "tiny_graph.pt", "data": _make_tiny_graph()}
    monkeypatch.setattr(
        app,
        "_resolve_graph_hash_for_loaded_graph",
        lambda _graph_obj: "abcdef1234567890fedcba",
    )

    snapshot_name, snapshot_storage = app._compute_optuna_study_name(
        graph_obj=graph_obj,
        balancing_strategy="Sin balancear",
        objective_metric="AUPRC",
        gnn_variant="gat_snapshot",
        space_signature="space1",
    )
    gru_name, gru_storage = app._compute_optuna_study_name(
        graph_obj=graph_obj,
        balancing_strategy="Sin balancear",
        objective_metric="AUPRC",
        gnn_variant="gat_gru",
        space_signature="space1",
    )

    assert snapshot_name != gru_name
    assert "GNN_gat_snapshot" in snapshot_name
    assert "GNN_gat_gru" in gru_name
    assert snapshot_storage == gru_storage


def test_run_optuna_search_objective_bundle_exposes_fixed_gnn_variant(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("optuna")
    monkeypatch.setattr(app, "RESULTADOS_DIR", str(tmp_path))

    bundle = app._run_optuna_search(
        graph_obj={
            "data": _make_tiny_graph(),
            "filename": "tiny_graph.pt",
            "sequence_index": None,
        },
        balancing_strategy="Sin balancear",
        search_space={"aggr1": ["sum"], "aggr2": ["sum"]},
        optuna_settings={
            "epochs": 1,
            "eval_every": 1,
            "seed": 7,
            "multivariate": True,
            "group": True,
            "pruner": "Ninguno",
            "min_resource": 1,
            "max_resource": 1,
            "reduction_factor": 2,
            "n_warmup_steps": 0,
            "debug": False,
        },
        objective_settings={
            "metric": "AUPRC",
            "threshold_beta": 1.0,
            "device": torch.device("cpu"),
            "gnn_variant": "gat_snapshot",
        },
        return_objective=True,
        storage_path=str(Path(tmp_path) / "optuna_variant.sqlite"),
    )

    assert bundle["gnn_variant"] == "gat_snapshot"
    assert "GNN_gat_snapshot" in bundle["variant_tag"]
