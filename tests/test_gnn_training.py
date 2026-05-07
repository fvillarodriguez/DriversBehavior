
import json
import math
import hashlib
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import pandas as pd
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gat_model import HeteroGAT
from src.graph_builder_app import _infer_edge_feature_dim
from src import gnn_main
from src.train_pretrain import train_minibatch

def test_gat_model_training_step(dummy_graph_data):
    """
    Test Goal: Verify that the GNN model can perform a complete training step:
    - Forward pass
    - Loss calculation
    - Backward pass (Gradient computation)
    - Optimizer step
    """
    data = dummy_graph_data
    in_channels = data['pm'].x.shape[1]
    out_channels = 2
    edge_dim = _infer_edge_feature_dim(data)
    
    # 1. Initialize Model
    model = HeteroGAT(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=out_channels,
        num_heads=2,
        dropout=0.1,
        edge_feature_dim=edge_dim,
        num_layers=2
    )
    
    # 2. Setup Optimizer and Criteria
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    optimizer.zero_grad()
    
    # 3. Forward Pass
    edge_attr_dict = {
        ('pm', 'spatial', 'pm'): data['pm', 'spatial', 'pm'].edge_attr,
        ('pm', 'temporal', 'pm'): data['pm', 'temporal', 'pm'].edge_attr
    }
    
    # Simulate a batch (using full graph here for simplicity)
    x_dict, _, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    
    # 4. Compute Loss
    logits = x_dict['pm']
    target = data['pm'].y
    print(f"Logits shape: {logits.shape}, Target shape: {target.shape}")
    loss = criterion(logits, target)
    
    # Check loss is valid
    assert torch.isfinite(loss)
    assert loss.item() > 0
    
    # 5. Backward Pass
    loss.backward()
    
    # 6. Verify Gradients
    # Check that at least one parameter has gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            assert torch.isfinite(param.grad).all()
            # break # check all?
    
    assert has_grads, "Model parameters should have gradients after backward pass"
    
    # 7. Optimizer Step
    optimizer.step()
    
    # 8. Check that weights changed (optional but good)
    # Ideally checking before/after, but basic gradient check is sufficient for 'can train' verification.


def test_gat_residual_without_relation_self_loops_strict_roundtrip(dummy_graph_data):
    data = dummy_graph_data
    model = HeteroGAT(
        in_channels=data["pm"].x.shape[1],
        hidden_channels=8,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=_infer_edge_feature_dim(data),
        num_layers=2,
        use_residual=True,
        use_relation_self_loops=False,
    )

    edge_attr_dict = {
        edge_type: data[edge_type].edge_attr
        for edge_type in data.edge_types
        if "edge_attr" in data[edge_type]
    }
    out, z, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    assert out["pm"].shape == (data["pm"].num_nodes, 2)
    assert z["pm"].shape[0] == data["pm"].num_nodes
    assert len(model.residual_lins) == 2

    rebuilt = HeteroGAT(
        in_channels=data["pm"].x.shape[1],
        hidden_channels=8,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=_infer_edge_feature_dim(data),
        num_layers=2,
        use_residual=True,
        use_relation_self_loops=False,
    )
    missing, unexpected = rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert missing == []
    assert unexpected == []


def test_gat_checkpointing_backpropagates_into_conv_layers(dummy_graph_data):
    data = dummy_graph_data
    assert data["pm"].x.requires_grad is False

    model = HeteroGAT(
        in_channels=data["pm"].x.shape[1],
        hidden_channels=8,
        out_channels=2,
        num_heads=2,
        dropout=0.0,
        edge_feature_dim=_infer_edge_feature_dim(data),
        num_layers=1,
        use_checkpointing=True,
    )
    model.train()

    edge_attr_dict = {
        edge_type: data[edge_type].edge_attr
        for edge_type in data.edge_types
        if "edge_attr" in data[edge_type]
    }
    out, _, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    loss = torch.nn.CrossEntropyLoss()(out["pm"], data["pm"].y)
    loss.backward()

    conv_grad_norm = sum(
        param.grad.detach().abs().sum().item()
        for name, param in model.named_parameters()
        if name.startswith("convs.") and param.grad is not None
    )
    assert conv_grad_norm > 0.0


def test_val_loss_monitor_resets_patience_only_on_min_delta_improvement():
    improved, best_loss, patience = gnn_main._update_val_loss_monitor(
        val_loss=0.80,
        best_val_loss=float("inf"),
        patience_counter=3,
        min_delta=0.01,
    )
    assert improved is True
    assert best_loss == pytest.approx(0.80)
    assert patience == 0

    improved, best_loss, patience = gnn_main._update_val_loss_monitor(
        val_loss=0.795,
        best_val_loss=best_loss,
        patience_counter=patience,
        min_delta=0.01,
    )
    assert improved is False
    assert best_loss == pytest.approx(0.80)
    assert patience == 1

    improved, best_loss, patience = gnn_main._update_val_loss_monitor(
        val_loss=0.78,
        best_val_loss=best_loss,
        patience_counter=patience,
        min_delta=0.01,
    )
    assert improved is True
    assert best_loss == pytest.approx(0.78)
    assert patience == 0

    for current_loss in (0.781, 0.782):
        improved, best_loss, patience = gnn_main._update_val_loss_monitor(
            val_loss=current_loss,
            best_val_loss=best_loss,
            patience_counter=patience,
            min_delta=0.01,
        )
        assert improved is False

    assert patience >= 2
    improved, best_loss, patience_after_nan = gnn_main._update_val_loss_monitor(
        val_loss=float("nan"),
        best_val_loss=best_loss,
        patience_counter=patience,
        min_delta=0.0,
    )
    assert improved is False
    assert best_loss == pytest.approx(0.78)
    assert patience_after_nan == patience + 1


def test_binary_eval_extras_include_false_alarm_ratio_and_brier_score():
    y_true = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    y_pred = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    y_prob = torch.tensor(
        [
            [0.90, 0.10],
            [0.40, 0.60],
            [0.20, 0.80],
            [0.70, 0.30],
        ],
        dtype=torch.float32,
    )

    metrics = gnn_main._compute_binary_eval_extras(y_true, y_pred, y_prob)

    assert metrics["false_alarm_ratio"] == pytest.approx(0.5)
    assert metrics["far"] == pytest.approx(0.5)
    assert metrics["brier_score"] == pytest.approx(0.225)
    assert metrics["brier"] == pytest.approx(0.225)


def test_platt_calibrated_probabilities_preserve_source_tensor_dtype_and_device():
    class Float64PlattModel:
        def predict_proba(self, x):
            assert x.dtype == np.float32
            return np.asarray(
                [
                    [0.75, 0.25],
                    [0.20, 0.80],
                ],
                dtype=np.float64,
            )

    prob1 = torch.tensor([0.1, 0.9], dtype=torch.float32)

    calibrated = gnn_main._calibrated_probability_tensor(prob1, Float64PlattModel())

    assert calibrated.dtype == prob1.dtype
    assert calibrated.device == prob1.device
    assert calibrated.tolist() == pytest.approx([0.25, 0.80])


def test_train_minibatch_reports_unscaled_loss_with_accumulation():
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    data = HeteroData()
    data["pm"].x = torch.zeros(4, 2)
    data["pm"].y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    data["pm"].batch_size = 4
    data["pm"].n_id = torch.arange(4)
    data["pm", "spatial", "pm"].edge_index = torch.empty((2, 0), dtype=torch.long)

    class ConstantLogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x_dict, edge_index_dict, edge_attr_dict):
            del edge_index_dict, edge_attr_dict
            logits = self.bias.expand(x_dict["pm"].shape[0], -1)
            return {"pm": logits}, {"pm": logits}, {}

    model = ConstantLogitModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss, cls_loss, edge_loss, l2_loss = train_minibatch(
        model,
        [data, data],
        optimizer,
        torch.nn.CrossEntropyLoss(),
        accumulation_steps=2,
    )

    assert loss == pytest.approx(math.log(2), rel=1e-5)
    assert cls_loss == pytest.approx(math.log(2), rel=1e-5)
    assert edge_loss == pytest.approx(0.0)
    assert l2_loss == pytest.approx(0.0)


def _make_small_training_graph():
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    data = HeteroData()
    data["pm"].x = torch.zeros(4, 3)
    data["pm"].y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    data["pm"].train_mask = torch.tensor([True, True, False, False])
    data["pm"].val_mask = torch.tensor([False, False, True, True])
    return data


def _write_fast_hparams(tmp_path, *, checkpoint_metric=None):
    row = {
        "value": 0.5,
        "hidden_channels": 8,
        "num_heads": 1,
        "dropout": 0.0,
        "num_layers": 1,
        "lr": 0.01,
        "weight_decay": 0.0,
        "batch_size": 4,
        "num_neighbors": "[1]",
        "optimizer": "Adam",
        "loss_type": "CrossEntropy",
        "objective_metric": "F1",
    }
    if checkpoint_metric is not None:
        row["checkpoint_metric"] = checkpoint_metric

    hp_path = tmp_path / "optuna_hyperparams_Base.csv"
    pd.DataFrame([row]).to_csv(hp_path, index=False)
    return hp_path


def test_safe_mcc_returns_zero_for_single_class_predictions():
    y_true = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    y_pred = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    assert gnn_main._safe_matthews_corrcoef(y_true, y_pred) == pytest.approx(0.0)


def test_metric_monitor_treats_nonfinite_value_as_no_improvement():
    is_best, best_value, patience_counter = gnn_main._update_metric_monitor(
        monitor_value=float("nan"),
        best_monitor_value=0.25,
        patience_counter=2,
        min_delta=0.0,
        monitor_mode="max",
    )

    assert is_best is False
    assert best_value == pytest.approx(0.25)
    assert patience_counter == 3


def test_lr_scheduler_choice_normalization_and_step_scope():
    assert gnn_main._normalize_lr_scheduler_choice("OneCycleLR") == "one_cycle"
    assert (
        gnn_main._normalize_lr_scheduler_choice("cosine-warm-restarts")
        == "cosine_warm_restarts"
    )
    assert (
        gnn_main._normalize_lr_scheduler_choice("reduce_lr_on_plateau")
        == "plateau_restart"
    )
    assert gnn_main._lr_scheduler_steps_per_batch("plateau_restart") is False
    assert gnn_main._lr_scheduler_steps_per_batch("one_cycle") is True


def _install_fast_training_mocks(monkeypatch, tmp_path, val_losses, prob_sequences, *, saved_paths=None):
    class FakeLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter([object()])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.use_checkpointing = False

    test_criteria = []
    if saved_paths is None:
        saved_paths = []

    def fake_test(*args, criterion=None, **kwargs):
        del args, kwargs
        test_criteria.append(criterion)
        idx = len(test_criteria) - 1
        y_true = torch.tensor([0, 1], dtype=torch.long)
        prob1 = torch.tensor(prob_sequences[idx], dtype=torch.float32)
        probs = torch.stack([1.0 - prob1, prob1], dim=1)
        f1_report = 0.95 if idx >= 2 else 0.10
        return {
            "val_mask": {
                "true": y_true,
                "probs": probs,
                "report": {
                    "Accidente (1)": {
                        "f1-score": f1_report,
                        "precision": f1_report,
                        "recall": f1_report,
                    },
                    "macro avg": {"f1-score": f1_report},
                    "accuracy": f1_report,
                },
                "cm": [[1, 0], [0, 1]],
                "auc": f1_report,
                "auprc": f1_report,
                "mcc": f1_report,
                "loss": val_losses[idx],
            }
        }

    monkeypatch.setattr(gnn_main, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(gnn_main, "SummaryWriter", None)
    monkeypatch.setattr(gnn_main, "SAVE_GAT_ALIASES", 0)
    monkeypatch.setattr(gnn_main, "NeighborLoader", lambda *args, **kwargs: FakeLoader())
    monkeypatch.setattr(gnn_main, "_build_gnn_model", lambda **kwargs: FakeModel())
    monkeypatch.setattr(gnn_main, "_prime_temporal_cache_if_needed", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        gnn_main,
        "train_minibatch",
        lambda *args, **kwargs: (0.5, 0.5, 0.0, 0.0),
    )
    monkeypatch.setattr(gnn_main, "test", fake_test)
    monkeypatch.setattr(gnn_main, "_get_repo_version", lambda: "test")
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    return test_criteria, saved_paths


def test_run_gat_training_defaults_to_objective_for_best_checkpoint(tmp_path, monkeypatch):
    loaded_obj = {"data": _make_small_training_graph(), "filename": "demo_graph.pt"}
    _write_fast_hparams(tmp_path)
    saved_paths = []
    test_criteria, _ = _install_fast_training_mocks(
        monkeypatch,
        tmp_path,
        val_losses=[0.90, 0.80, 0.81, 0.82, 0.83],
        prob_sequences=[
            [0.9, 0.2],
            [0.8, 0.3],
            [0.1, 0.95],
            [0.95, 0.1],
            [0.85, 0.15],
        ],
        saved_paths=saved_paths,
    )
    monkeypatch.setattr(gnn_main.torch, "save", lambda obj, path: saved_paths.append(str(path)))

    progress_events = []
    gnn_main.run_gat_training(
        loaded_obj,
        force_use_graphsmote=False,
        early_stop=True,
        early_stop_patience=2,
        early_stop_min_delta=0.0,
        max_epochs=5,
        progress_callback=lambda **payload: progress_events.append(dict(payload)),
    )

    epoch_events = [event for event in progress_events if event.get("epoch")]
    assert len(test_criteria) == 5
    assert all(criterion is not None for criterion in test_criteria)
    assert len(saved_paths) == 2
    assert epoch_events[-1]["epoch"] == 5
    assert epoch_events[-1]["monitor_metric"] == "val_objective_score"
    assert epoch_events[-1]["monitor_mode"] == "max"
    assert epoch_events[-1]["best_val_loss"] == pytest.approx(0.81)
    assert epoch_events[-1]["best_monitor_value"] == pytest.approx(1.0)
    assert epoch_events[-1]["patience_counter"] == 2

    hparams_files = sorted(tmp_path.glob("gat_model_BEST*_hparams.json"))
    assert hparams_files
    meta = json.loads(hparams_files[-1].read_text(encoding="utf-8"))
    assert meta["monitor_metric"] == "val_objective_score"
    assert meta["monitor_mode"] == "max"
    assert meta["best_monitor_value"] == pytest.approx(1.0)
    assert meta["best_val_loss"] == pytest.approx(0.81)
    assert meta["best_epoch"] == 3


def test_run_gat_training_can_fallback_to_val_loss_monitor(tmp_path, monkeypatch):
    loaded_obj = {"data": _make_small_training_graph(), "filename": "demo_graph.pt"}
    _write_fast_hparams(tmp_path, checkpoint_metric="val_loss")
    saved_paths = []
    test_criteria, _ = _install_fast_training_mocks(
        monkeypatch,
        tmp_path,
        val_losses=[0.90, 0.80, 0.81, 0.82, 0.70],
        prob_sequences=[
            [0.9, 0.2],
            [0.8, 0.3],
            [0.1, 0.95],
            [0.05, 0.98],
            [0.1, 0.9],
        ],
        saved_paths=saved_paths,
    )
    monkeypatch.setattr(gnn_main.torch, "save", lambda obj, path: saved_paths.append(str(path)))

    progress_events = []
    gnn_main.run_gat_training(
        loaded_obj,
        force_use_graphsmote=False,
        early_stop=True,
        early_stop_patience=2,
        early_stop_min_delta=0.0,
        max_epochs=5,
        progress_callback=lambda **payload: progress_events.append(dict(payload)),
    )

    epoch_events = [event for event in progress_events if event.get("epoch")]
    assert len(test_criteria) == 4
    assert len(saved_paths) == 2
    assert epoch_events[-1]["epoch"] == 4
    assert epoch_events[-1]["monitor_metric"] == "val_loss"
    assert epoch_events[-1]["monitor_mode"] == "min"
    assert epoch_events[-1]["best_val_loss"] == pytest.approx(0.80)
    assert epoch_events[-1]["best_monitor_value"] == pytest.approx(0.80)

    hparams_files = sorted(tmp_path.glob("gat_model_BEST*_hparams.json"))
    assert hparams_files
    meta = json.loads(hparams_files[-1].read_text(encoding="utf-8"))
    assert meta["monitor_metric"] == "val_loss"
    assert meta["monitor_mode"] == "min"
    assert meta["best_val_loss"] == pytest.approx(0.80)
    assert meta["best_epoch"] == 2


def test_run_gat_training_honors_manual_stop_after_checkpoint(tmp_path, monkeypatch):
    loaded_obj = {"data": _make_small_training_graph(), "filename": "demo_graph.pt"}
    _write_fast_hparams(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    test_criteria, _ = _install_fast_training_mocks(
        monkeypatch,
        tmp_path,
        val_losses=[0.90, 0.80, 0.70, 0.60, 0.50],
        prob_sequences=[
            [0.9, 0.2],
            [0.8, 0.3],
            [0.1, 0.95],
            [0.05, 0.98],
            [0.1, 0.9],
        ],
    )

    gnn_main.run_gat_training(
        loaded_obj,
        force_use_graphsmote=False,
        early_stop=False,
        max_epochs=5,
        save_state_path=str(checkpoint_path),
        should_stop=lambda: len(test_criteria) >= 2,
    )

    assert len(test_criteria) == 2
    saved_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved_ckpt["epoch"] == 2
    assert saved_ckpt["last_val_loss"] == pytest.approx(0.80)


def test_run_gat_training_runs_requested_test_and_continues(tmp_path, monkeypatch):
    loaded_obj = {"data": _make_small_training_graph(), "filename": "demo_graph.pt"}
    _write_fast_hparams(tmp_path)
    _install_fast_training_mocks(
        monkeypatch,
        tmp_path,
        val_losses=[0.90],
        prob_sequences=[[0.9, 0.2]],
    )

    val_calls = 0
    test_calls = 0
    test_requested = False

    def fake_test(*args, masks=None, threshold=None, **kwargs):
        del args, kwargs
        nonlocal val_calls, test_calls
        y_true = torch.tensor([0, 1], dtype=torch.long)
        if masks == ["test_mask"]:
            test_calls += 1
            assert threshold is None or isinstance(float(threshold), float)
            return {
                "test_mask": {
                    "true": y_true,
                    "probs": torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
                    "report": {
                        "Accidente (1)": {
                            "f1-score": 0.80,
                            "precision": 0.75,
                            "recall": 0.86,
                        },
                        "macro avg": {"f1-score": 0.82},
                        "accuracy": 0.85,
                    },
                    "cm": [[1, 0], [0, 1]],
                    "auc": 0.90,
                    "auprc": 0.88,
                    "mcc": 0.70,
                    "far": 0.05,
                }
            }

        val_calls += 1
        f1_report = 0.20 + 0.10 * val_calls
        return {
            "val_mask": {
                "true": y_true,
                "probs": torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
                "report": {
                    "Accidente (1)": {
                        "f1-score": f1_report,
                        "precision": f1_report,
                        "recall": f1_report,
                    },
                    "macro avg": {"f1-score": f1_report},
                    "accuracy": f1_report,
                },
                "cm": [[1, 0], [0, 1]],
                "auc": f1_report,
                "auprc": f1_report,
                "mcc": f1_report,
                "loss": 1.0 - 0.1 * val_calls,
            }
        }

    def should_test():
        nonlocal test_requested
        if val_calls >= 2 and not test_requested:
            test_requested = True
            return True
        return False

    monkeypatch.setattr(gnn_main, "test", fake_test)

    gnn_main.run_gat_training(
        loaded_obj,
        force_use_graphsmote=False,
        early_stop=False,
        max_epochs=4,
        should_test=should_test,
    )

    assert val_calls == 4
    assert test_calls == 1


def test_run_gat_training_resumes_old_val_loss_checkpoint_with_new_monitor(tmp_path, monkeypatch):
    data = _make_small_training_graph()
    loaded_obj = {"data": data, "filename": "demo_graph.pt"}
    _write_fast_hparams(tmp_path)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.use_checkpointing = False

    old_ckpt = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 1,
            "max_epochs": 3,
            "model_state": FakeModel().state_dict(),
            "best_val_loss": 0.10,
            "best_val_f1": 0.10,
            "best_val_objective_score": 0.10,
            "best_val_auprc": 0.10,
            "best_epoch": 1,
            "monitor_metric": "val_loss",
            "monitor_mode": "min",
            "best_monitor_value": 0.10,
            "patience_counter": 2,
        },
        old_ckpt,
    )

    test_criteria, _ = _install_fast_training_mocks(
        monkeypatch,
        tmp_path,
        val_losses=[0.91, 0.92],
        prob_sequences=[
            [0.1, 0.95],
            [0.95, 0.1],
        ],
    )
    monkeypatch.setattr(gnn_main, "_build_gnn_model", lambda **kwargs: FakeModel())

    progress_events = []
    gnn_main.run_gat_training(
        loaded_obj,
        force_use_graphsmote=False,
        early_stop=True,
        early_stop_patience=2,
        early_stop_min_delta=0.0,
        max_epochs=3,
        resume_state_path=str(old_ckpt),
        save_state_path=str(old_ckpt),
        progress_callback=lambda **payload: progress_events.append(dict(payload)),
    )

    epoch_events = [event for event in progress_events if event.get("epoch")]
    assert len(test_criteria) == 2
    assert epoch_events[0]["epoch"] == 2
    assert epoch_events[0]["monitor_metric"] == "val_objective_score"
    assert epoch_events[0]["best_monitor_value"] == pytest.approx(1.0)

    saved_ckpt = torch.load(old_ckpt, map_location="cpu", weights_only=False)
    assert saved_ckpt["monitor_metric"] == "val_objective_score"
    assert saved_ckpt["monitor_mode"] == "max"
    assert saved_ckpt["best_epoch"] == 2
    assert saved_ckpt["best_val_loss"] == pytest.approx(0.91)


def test_graph_identity_uses_semantic_hash_and_records_file_hash(tmp_path):
    graph_path = tmp_path / "graph.pt"
    graph_path.write_bytes(b"graph payload")
    expected_file_hash = hashlib.sha256(b"graph payload").hexdigest()
    semantic_hash = "c" * 64
    data = _make_small_training_graph()
    data.graph_metadata = {"graph_hash": "a" * 64}

    identity = gnn_main._resolve_graph_identity(
        {
            "data": data,
            "metadata": {"graph_hash": "b" * 64},
            "graph_hash": semantic_hash,
            "graph_path": str(graph_path),
        }
    )

    assert identity["graph_hash"] == semantic_hash
    assert identity["graph_file_hash"] == expected_file_hash
    assert identity["graph_hash_source"] == "semantic_metadata"
