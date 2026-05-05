
import json
import math
import pytest
import torch
import torch.nn.functional as F
import sys
import os
import pandas as pd

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
    with pytest.raises(ValueError):
        gnn_main._update_val_loss_monitor(
            val_loss=float("nan"),
            best_val_loss=best_loss,
            patience_counter=patience,
            min_delta=0.0,
        )


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


def test_run_gat_training_uses_val_loss_for_best_checkpoint_and_stop(tmp_path, monkeypatch):
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    data = HeteroData()
    data["pm"].x = torch.zeros(4, 3)
    data["pm"].y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    data["pm"].train_mask = torch.tensor([True, True, False, False])
    data["pm"].val_mask = torch.tensor([False, False, True, True])
    loaded_obj = {"data": data, "filename": "demo_graph.pt"}

    hp_path = tmp_path / "optuna_hyperparams_Base.csv"
    pd.DataFrame(
        [
            {
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
        ]
    ).to_csv(hp_path, index=False)

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

    saved_paths = []
    test_criteria = []
    val_losses = [0.90, 0.80, 0.81, 0.82, 0.70]
    prob_sequences = [
        [0.9, 0.2],
        [0.8, 0.3],
        [0.1, 0.95],
        [0.05, 0.98],
        [0.1, 0.9],
    ]

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
    monkeypatch.setattr(gnn_main.torch, "save", lambda obj, path: saved_paths.append(str(path)))
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

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
    assert all(criterion is not None for criterion in test_criteria)
    assert len(saved_paths) == 2
    assert epoch_events[-1]["epoch"] == 4
    assert epoch_events[-1]["best_val_loss"] == pytest.approx(0.80)
    assert epoch_events[-1]["patience_counter"] == 2

    hparams_files = sorted(tmp_path.glob("gat_model_BEST*_hparams.json"))
    assert hparams_files
    meta = json.loads(hparams_files[-1].read_text(encoding="utf-8"))
    assert meta["monitor_metric"] == "val_loss"
    assert meta["monitor_mode"] == "min"
    assert meta["best_val_loss"] == pytest.approx(0.80)
    assert meta["best_epoch"] == 2
