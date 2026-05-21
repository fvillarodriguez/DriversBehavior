from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from src.imgagn_relational import (
    RelationalImGAGNConfig,
    _train_linear_auc_np,
    build_relational_imgagn_graph,
    to_cpu_graph_object,
)
from src.graph import PMIndex
from src.snapshot_sequences import SequenceConfig, SequenceIndex


def _tiny_relational_graph() -> dict:
    torch.manual_seed(7)
    data = HeteroData()
    x = torch.randn(8, 66)
    y = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.long)
    data["pm"].x = x
    data["pm"].y = y
    data["pm"].train_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    data["pm"].val_mask = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool)
    data["pm"].test_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool)
    data["pm"].num_nodes = 8

    temporal_ei = torch.tensor(
        [
            [0, 1, 3, 4, 6],
            [1, 2, 4, 5, 7],
        ],
        dtype=torch.long,
    )
    data[("pm", "temporal", "pm")].edge_index = temporal_ei
    data[("pm", "temporal", "pm")].edge_attr = torch.randn(temporal_ei.shape[1], 66)

    spatial_ei = torch.tensor(
        [
            [2, 0, 5, 3, 1, 4],
            [0, 2, 3, 5, 2, 5],
        ],
        dtype=torch.long,
    )
    data[("pm", "spatial", "pm")].edge_index = spatial_ei
    data[("pm", "spatial", "pm")].edge_attr = torch.randn(spatial_ei.shape[1], 73)

    sequence_index = SequenceIndex(
        sequence_rows=np.asarray(
            [
                [0, 1, 2],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
            ],
            dtype=np.int64,
        ),
        target_rows=np.asarray([2, 5, 6, 7], dtype=np.int64),
        labels=np.asarray([1, 1, 0, 0], dtype=np.int8),
        porticos=np.asarray(["P1", "P2", "P2", "P3"], dtype=object),
        target_ts_min=np.asarray([10, 20, 30, 40], dtype=np.int64),
        config=SequenceConfig(sequence_length=3),
    )
    return {
        "data": data,
        "sequence_index": sequence_index,
        "filename": "tiny_graph.pt",
    }


def _cfg() -> RelationalImGAGNConfig:
    return RelationalImGAGNConfig(
        target_pos_ratio=0.5,
        dz=8,
        hidden_g=16,
        hidden_d=16,
        epochs=1,
        d_steps=1,
        batch_size=32,
        spatial_copy_k=2,
        seed=19092086,
    )


def test_relational_imgagn_adds_train_only_synthetics_and_preserves_relations() -> None:
    graph_obj = _tiny_relational_graph()
    result = build_relational_imgagn_graph(graph_obj, _cfg(), device="cpu")
    out = result.graph_obj
    data = out["data"]
    synth = data["pm"].is_synthetic.detach().cpu().bool()
    synth_idx = torch.where(synth)[0].cpu().numpy()

    assert int(synth.sum().item()) == 1
    assert int((synth & data["pm"].train_mask.cpu()).sum().item()) == 1
    assert int((synth & data["pm"].val_mask.cpu()).sum().item()) == 0
    assert int((synth & data["pm"].test_mask.cpu()).sum().item()) == 0

    seq = out["sequence_index"]
    assert set(synth_idx.tolist()).issubset(set(np.asarray(seq.target_rows).tolist()))
    assert data[("pm", "temporal", "pm")].edge_attr.shape[1] == 66
    assert data[("pm", "spatial", "pm")].edge_attr.shape[1] == 73
    assert ("pm", "imgagn", "pm") not in data.edge_types
    assert result.validation["ok"] is True


def test_relational_imgagn_records_val_loss_and_best_checkpoint(tmp_path) -> None:
    graph_obj = _tiny_relational_graph()
    cfg = RelationalImGAGNConfig(
        target_pos_ratio=0.5,
        dz=8,
        hidden_g=16,
        hidden_d=16,
        epochs=2,
        d_steps=1,
        batch_size=32,
        spatial_copy_k=2,
        seed=19092086,
        early_stopping_patience=5,
    )
    checkpoint_path = tmp_path / "imgagn_best.pt"

    result = build_relational_imgagn_graph(
        graph_obj,
        cfg,
        device="cpu",
        checkpoint_artifact_path=checkpoint_path,
    )

    history = result.build_meta["imgagn_history"]
    assert history
    assert all("val_loss" in row for row in history)
    assert all("tabsyndex_score" in row for row in history)
    assert all("adversarial_val_loss" in row for row in history)
    assert result.build_meta["imgagn_feature_quality"]["best_epoch"] >= 1
    assert result.build_meta["imgagn_feature_quality"]["checkpoint_path"] == str(checkpoint_path)
    assert result.build_meta["imgagn_feature_quality"]["checkpoint_metric"] == "tabsyndex_loss"
    assert result.build_meta["imgagn_feature_quality"]["best_tabsyndex_loss"] is not None
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_metric"] == "tabsyndex_loss"
    assert checkpoint["checkpoint_source"] == "internal_train_holdout_distribution"
    assert checkpoint["best_tabsyndex_loss"] is not None
    assert checkpoint["epoch"] == result.build_meta["imgagn_feature_quality"]["best_epoch"]


def test_internal_classifier_metric_keeps_gradients_under_no_grad() -> None:
    rng = np.random.default_rng(11)
    train_x = np.vstack(
        [
            rng.normal(loc=-0.5, scale=0.3, size=(24, 6)),
            rng.normal(loc=0.5, scale=0.3, size=(24, 6)),
        ]
    )
    train_y = np.concatenate([np.zeros(24, dtype=np.int64), np.ones(24, dtype=np.int64)])
    eval_x = np.vstack(
        [
            rng.normal(loc=-0.5, scale=0.3, size=(8, 6)),
            rng.normal(loc=0.5, scale=0.3, size=(8, 6)),
        ]
    )
    eval_y = np.concatenate([np.zeros(8, dtype=np.int64), np.ones(8, dtype=np.int64)])

    with torch.no_grad():
        metrics = _train_linear_auc_np(
            train_x,
            train_y,
            eval_x,
            eval_y,
            steps=5,
            lr=0.05,
            seed=19092086,
        )

    assert metrics["auc"] is not None
    assert metrics["accuracy"] is not None


def test_relational_imgagn_accepts_structural_sequence_index_object() -> None:
    graph_obj = _tiny_relational_graph()
    seq = graph_obj["sequence_index"]
    graph_obj["sequence_index"] = SimpleNamespace(
        sequence_rows=seq.sequence_rows,
        target_rows=seq.target_rows,
        labels=seq.labels,
        porticos=seq.porticos,
        target_ts_min=seq.target_ts_min,
        config={
            "sequence_length": seq.config.sequence_length,
            "guard_band_minutes": seq.config.guard_band_minutes,
            "horizon_minutes": seq.config.horizon_minutes,
            "include_downstream": seq.config.include_downstream,
        },
    )

    result = build_relational_imgagn_graph(graph_obj, _cfg(), device="cpu")

    assert isinstance(result.graph_obj["sequence_index"], SequenceIndex)
    assert result.validation["ok"] is True


def test_to_cpu_graph_object_rebuilds_stale_pm_index_for_pickle(tmp_path) -> None:
    class StalePMIndex:
        def __init__(self):
            self._map = {("P1", 10): 0, ("P2", 10): 1}
            self._rev = {0: ("P1", 10), 1: ("P2", 10)}

    graph_obj = _tiny_relational_graph()
    graph_obj["pm_index"] = StalePMIndex()

    out = to_cpu_graph_object(graph_obj)
    save_path = tmp_path / "graph.pt"
    torch.save(out, save_path)
    loaded = torch.load(save_path, map_location="cpu", weights_only=False)

    assert isinstance(out["pm_index"], PMIndex)
    assert isinstance(loaded["pm_index"], PMIndex)
    assert loaded["pm_index"]._rev[1] == ("P2", 10)


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("sequence_index", "sequence_index"),
        ("temporal", "temporal"),
        ("spatial", "spatial"),
    ],
)
def test_relational_imgagn_requires_sequence_and_core_relations(
    missing_key: str,
    message: str,
) -> None:
    graph_obj = _tiny_relational_graph()
    if missing_key == "sequence_index":
        graph_obj.pop("sequence_index")
    else:
        data = copy.deepcopy(graph_obj["data"])
        if missing_key == "temporal":
            del data[("pm", "temporal", "pm")]
        else:
            del data[("pm", "spatial", "pm")]
        graph_obj["data"] = data

    with pytest.raises(ValueError, match=message):
        build_relational_imgagn_graph(graph_obj, _cfg(), device="cpu")
