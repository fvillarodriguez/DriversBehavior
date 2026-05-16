import os
from types import SimpleNamespace

import pytest


def _maybe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


torch = _maybe_import("torch")
optuna = _maybe_import("optuna")
psutil = _maybe_import("psutil")


pytestmark = pytest.mark.skipif(
    torch is None or optuna is None,
    reason="Requires torch and optuna",
)


def _rss_mb():
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


class _DummyProgress:
    def progress(self, *_args, **_kwargs):
        return None


class _DummyText:
    def markdown(self, *_args, **_kwargs):
        return None

    def code(self, *_args, **_kwargs):
        return None


def _run_optuna_memory_check(tmp_path, monkeypatch, *, use_checkpointing: bool) -> float:
    tg = _maybe_import("torch_geometric")
    if tg is None:
        pytest.skip("torch_geometric not installed")
    if psutil is None:
        pytest.skip("psutil not installed")

    from torch_geometric.data import HeteroData

    import src.graph_builder_app as gba
    import src.gnn_main as gnn_main

    # Redirect outputs to a temp directory to avoid polluting Resultados.
    monkeypatch.setattr(gba, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(gnn_main, "RESULTADOS_DIR", str(tmp_path))

    # Build a tiny hetero graph for Optuna.
    num_nodes = 2000
    num_edges = 4000
    x = torch.randn(num_nodes, 7)
    y = torch.randint(0, 2, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    def _edge():
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        eidx = torch.stack([src, dst], dim=0)
        eattr = torch.randn(num_edges, 1)
        return eidx, eattr

    data = HeteroData()
    data["pm"].x = x
    data["pm"].y = y
    data["pm"].train_mask = train_mask
    data["pm"].val_mask = val_mask
    data["pm"].test_mask = test_mask

    for rel in ["spatial", "temporal", "st_fwd"]:
        eidx, eattr = _edge()
        data[("pm", rel, "pm")].edge_index = eidx
        data[("pm", rel, "pm")].edge_attr = eattr

    graph_obj = {"data": data, "sequence_index": None, "filename": None}

    search_space = {
        "hidden_channels": {"min": 16, "max": 16, "step": 16},
        "num_heads": {"min": 2, "max": 2, "step": 1},
        "num_layers": {"min": 2, "max": 2, "step": 1},
        "dropout": {"min": 0.1, "max": 0.1, "step": 0.1},
        "aggr1": ["sum"],
        "aggr2": ["sum"],
        "use_checkpointing": [use_checkpointing],
        "neighbor_choices": ["compact"],
        "neighbor_profiles": {"compact": [5, 5]},
        "lr": {"min": 1e-3, "max": 1e-3},
        "optimizers": ["Adam"],
        "weight_decay": {"min": 1e-6, "max": 1e-6},
        "lambda_l2_att": {"min": 1e-4, "max": 1e-4},
        "lambda_H_mode": ["fixed"],
        "lambda_H_fixed": {"min": 1e-4, "max": 1e-4},
        "initial_lambda_H": {"min": 1e-4, "max": 1e-4},
        "final_lambda_H": {"min": 1e-4, "max": 1e-4},
        "loss_type": ["CrossEntropy"],
        "focal_gamma": {"min": 2.0, "max": 2.0, "step": 0.1},
        "focal_alpha": {"min": 0.5, "max": 0.5, "step": 0.05},
        "batch_size": {"min": 64, "max": 64, "step": 1},
    }

    optuna_settings = {
        "n_trials": 3,
        "epochs": 2,
        "eval_every": 1,
        "seed": 7,
        "pruner": "Ninguno",
        "min_resource": 1,
        "max_resource": 2,
        "reduction_factor": 3,
        "n_warmup_steps": 0,
        "multivariate": False,
        "group": False,
        "sample_train_nodes": 500,
        "sample_each_epoch": True,
        "sample_val_nodes": 500,
        "debug": False,
    }

    objective_settings = {
        "metric": "F1",
        "threshold_beta": 1.0,
        "device": "cpu",
        "gnn_variant": "gat_snapshot",
    }

    # Track RSS after each trial via Optuna callback.
    rss_points = []
    orig_optimize = optuna.study.Study.optimize

    def _wrapped_optimize(self, func, n_trials=None, callbacks=None, **kwargs):
        def _cb(_study, _trial):
            rss = _rss_mb()
            if rss is not None:
                rss_points.append(rss)

        cb_list = list(callbacks or []) + [_cb]
        return orig_optimize(self, func, n_trials=n_trials, callbacks=cb_list, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", _wrapped_optimize)

    result = gba._run_optuna_search(
        graph_obj=graph_obj,
        balancing_strategy="Optuna sin balanceo",
        search_space=search_space,
        optuna_settings=optuna_settings,
        objective_settings=objective_settings,
        log_container=_DummyText(),
        progress_bar=_DummyProgress(),
        status_text=_DummyText(),
    )

    assert "best_params" in result
    assert rss_points, "No RSS samples captured"

    # For a tiny graph, RSS should not grow unbounded across trials.
    rss_delta = max(rss_points) - min(rss_points)
    return rss_delta


@pytest.mark.optuna
@pytest.mark.parametrize("use_checkpointing", [False, True])
def test_optuna_memory_no_growth(tmp_path, monkeypatch, use_checkpointing):
    rss_delta = _run_optuna_memory_check(
        tmp_path, monkeypatch, use_checkpointing=use_checkpointing
    )
    assert rss_delta < 200.0, f"RSS grew too much across trials: {rss_delta:.1f} MB"
