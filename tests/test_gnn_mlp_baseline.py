from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")
HeteroData = pytest.importorskip("torch_geometric.data").HeteroData

from src.gnn_mlp_baseline import run_gnn_mlp_baselines


METRIC_COLS = [
    "auprc",
    "auc",
    "f1_at_tau_val",
    "f05_at_tau_val",
    "precision",
    "recall",
    "far",
    "mcc",
    "accuracy",
    "brier_score",
    "tau",
]


class _SequenceIndex:
    def __init__(self, sequence_rows, target_rows):
        self.sequence_rows = np.asarray(sequence_rows, dtype=np.int64)
        self.target_rows = np.asarray(target_rows, dtype=np.int64)


def _make_graph(*, imbalanced: bool = False) -> HeteroData:
    data = HeteroData()
    x = torch.tensor(
        [
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 1.0],
            [3.0, 3.1, 0.0],
            [3.2, 2.9, 0.0],
            [0.2, 0.2, 1.0],
            [3.3, 3.1, 0.0],
            [0.3, 0.1, 1.0],
            [3.4, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    if imbalanced:
        y = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.long)
    else:
        y = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1], dtype=torch.long)
    data["pm"].x = x
    data["pm"].y = y
    data["pm"].train_mask = torch.tensor([True, True, True, True, False, False, False, False])
    data["pm"].val_mask = torch.tensor([False, False, False, False, True, True, False, False])
    data["pm"].test_mask = torch.tensor([False, False, False, False, False, False, True, True])
    return data


def _make_sequence_index() -> _SequenceIndex:
    return _SequenceIndex(
        sequence_rows=[
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
        ],
        target_rows=[1, 2, 3, 4, 5, 6, 7],
    )


def _run_fast(loaded_obj, baselines, **kwargs):
    return run_gnn_mlp_baselines(
        loaded_obj,
        baselines=baselines,
        epochs=2,
        patience=1,
        batch_size=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        device="cpu",
        save_dir=None,
        seed=123,
        **kwargs,
    )


def _assert_completed_metrics(df: pd.DataFrame, baselines: set[str]) -> pd.DataFrame:
    completed = df[df["status"] == "completed"]
    assert set(completed["baseline"]) == baselines
    for baseline in baselines:
        assert set(completed.loc[completed["baseline"] == baseline, "split"]) == {"val", "test"}
    values = completed[METRIC_COLS].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    for col in ("tp", "tn", "fp", "fn"):
        assert col in completed.columns
        assert np.isfinite(completed[col].to_numpy(dtype=float)).all()
    return completed


def test_mlp_current_baseline_runs_without_edges():
    data = _make_graph()

    df = _run_fast({"data": data, "graph_hash": "abc123"}, ("current",))

    assert set(df["baseline"]) == {"current"}
    assert set(df["split"]) == {"val", "test"}
    assert (df["status"] == "completed").all()
    assert df["train_samples"].iloc[0] == 4
    assert df["val_samples"].iloc[0] == 2
    assert df["test_samples"].iloc[0] == 2


def test_temporal_baseline_filters_by_target_masks():
    data = _make_graph()
    seq = _make_sequence_index()

    df = _run_fast({"data": data, "sequence_index": seq}, ("temporal",))

    completed = df[df["status"] == "completed"]
    assert set(completed["baseline"]) == {"temporal"}
    assert completed["train_samples"].iloc[0] == 3
    assert completed["val_samples"].iloc[0] == 2
    assert completed["test_samples"].iloc[0] == 2


def test_temporal_baseline_skips_without_sequence_index():
    data = _make_graph()

    df = _run_fast({"data": data}, ("current", "temporal"))

    current = df[df["baseline"] == "current"]
    temporal = df[df["baseline"] == "temporal"]
    assert (current["status"] == "completed").all()
    assert (temporal["status"] == "skipped").all()
    assert temporal["reason"].str.contains("sequence_index", case=False).all()


def test_imbalanced_baseline_metrics_are_finite():
    data = _make_graph(imbalanced=True)

    df = _run_fast({"data": data}, ("current",))

    completed = df[df["status"] == "completed"]
    values = completed[METRIC_COLS].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    for col in ("tp", "tn", "fp", "fn"):
        assert col in completed.columns
        assert np.isfinite(completed[col].to_numpy(dtype=float)).all()


def test_pr_threshold_selectors_use_unshifted_precision_recall_alignment():
    from src import graph_builder_app as app
    from src import gnn_main
    import src.gnn_mlp_baseline as baseline_module
    import src.mlp_tabular as mlp_tabular
    import src.xgboost as xgb_module

    y_true = np.array([0, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.7, 0.6, 0.2], dtype=float)
    expected_tau = 0.6

    assert baseline_module._pick_tau_fbeta(y_true, y_prob, beta=1.0) == pytest.approx(expected_tau)
    assert mlp_tabular._pick_threshold_from_val(y_true, y_prob, beta=1.0)[0] == pytest.approx(expected_tau)
    assert xgb_module._pick_threshold_from_val(y_true, y_prob, beta=1.0)[0] == pytest.approx(expected_tau)
    assert gnn_main.pick_threshold_from_val(y_true, y_prob, beta=1.0)[0] == pytest.approx(expected_tau)
    assert app._comparison_tau_from_val_result(
        {
            "true": torch.tensor(y_true, dtype=torch.long),
            "probs": torch.tensor(np.column_stack([1.0 - y_prob, y_prob]), dtype=torch.float32),
        }
    ) == pytest.approx(expected_tau)


def test_baseline_comparison_defaults_to_far_threshold_metadata():
    pytest.importorskip("xgboost")
    data = _make_graph()

    df = _run_fast(
        {"data": data},
        ("current", "xgboost_current"),
        threshold_strategy="far",
        far_target=0.2,
    )

    completed = _assert_completed_metrics(df, {"current", "xgboost_current"})
    assert set(completed["threshold_strategy"]) == {"far"}
    assert completed["far_target"].astype(float).eq(0.2).all()
    assert completed["tau_source"].astype(str).str.contains("far_target").all()
    assert set(completed["threshold_mask_source"]) == {"val_mask"}


def test_xgboost_baselines_run_on_current_and_temporal_features():
    pytest.importorskip("xgboost")
    data = _make_graph()
    seq = _make_sequence_index()

    df = _run_fast(
        {"data": data, "sequence_index": seq},
        ("xgboost_current", "xgboost_temporal"),
    )

    completed = _assert_completed_metrics(df, {"xgboost_current", "xgboost_temporal"})
    assert set(completed["model_family"]) == {"xgboost"}
    assert set(completed["feature_view"]) == {"current", "temporal"}
    assert set(completed["model"]) == {"XGBoost actual", "XGBoost temporal"}


def test_svm_baselines_run_on_current_and_temporal_features():
    data = _make_graph()
    seq = _make_sequence_index()

    df = _run_fast(
        {"data": data, "sequence_index": seq},
        ("svm_current", "svm_temporal"),
    )

    completed = _assert_completed_metrics(df, {"svm_current", "svm_temporal"})
    assert set(completed["model_family"]) == {"svm"}
    assert set(completed["feature_view"]) == {"current", "temporal"}
    assert set(completed["model"]) == {"SVM actual", "SVM temporal"}
    assert completed["backend"].astype(str).str.len().gt(0).all()


def test_tabular_temporal_baselines_skip_without_sequence_index():
    data = _make_graph()

    df = _run_fast({"data": data}, ("xgboost_temporal", "svm_temporal"))

    assert set(df["baseline"]) == {"xgboost_temporal", "svm_temporal"}
    assert (df["status"] == "skipped").all()
    assert df["reason"].str.contains("sequence_index", case=False).all()


def test_training_panel_runs_runner_and_exposes_saved_results(tmp_path: Path, monkeypatch):
    from src import graph_builder_app as app
    import src.gnn_mlp_baseline as baseline_module

    class _FakeProgress:
        def __init__(self):
            self.values = []

        def progress(self, value):
            self.values.append(value)

    class _FakeSlot:
        def caption(self, *args, **kwargs):
            return None

    class _FakeSpinner:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeColumn:
        def __init__(self, st):
            self._st = st

        def checkbox(self, *args, **kwargs):
            return self._st.checkbox(*args, **kwargs)

        def number_input(self, *args, **kwargs):
            return self._st.number_input(*args, **kwargs)

        def text_input(self, *args, **kwargs):
            return self._st.text_input(*args, **kwargs)

    class _FakeSt:
        def __init__(self):
            self.session_state = {}
            self.dataframes = []
            self.success_messages = []

        def markdown(self, *args, **kwargs):
            return None

        def caption(self, *args, **kwargs):
            return None

        def columns(self, n):
            return [_FakeColumn(self) for _ in range(int(n))]

        def checkbox(self, label, value=False, key=None, **kwargs):
            if key is not None:
                self.session_state.setdefault(key, value)
                return self.session_state[key]
            return value

        def number_input(self, label, value=0, key=None, **kwargs):
            if key is not None:
                self.session_state.setdefault(key, value)
                return self.session_state[key]
            return value

        def selectbox(self, label, options, index=0, key=None, **kwargs):
            selected = options[index]
            if key is not None:
                self.session_state.setdefault(key, selected)
                return self.session_state[key]
            return selected

        def text_input(self, label, value="", key=None, **kwargs):
            if key is not None:
                self.session_state.setdefault(key, value)
                return self.session_state[key]
            return value

        def progress(self, value):
            progress = _FakeProgress()
            progress.progress(value)
            return progress

        def empty(self):
            return _FakeSlot()

        def button(self, label, key=None, **kwargs):
            return key == "gnn_mlp_baseline_run"

        def spinner(self, *args, **kwargs):
            return _FakeSpinner()

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            raise AssertionError(args[0] if args else "streamlit error")

        def success(self, message, *args, **kwargs):
            self.success_messages.append(message)

        def dataframe(self, df, *args, **kwargs):
            self.dataframes.append(df.copy())

        def multiselect(self, label, options, default=None, key=None, **kwargs):
            selected = list(default or [])
            if key is not None:
                self.session_state.setdefault(key, selected)
                return self.session_state[key]
            return selected

        def plotly_chart(self, *args, **kwargs):
            return None

        def bar_chart(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    fake_st = _FakeSt()
    calls = {}

    def fake_runner(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        artifact_path = tmp_path / "gnn_mlp_baselines" / "fake.csv"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "model": "MLP actual",
                    "baseline": "current",
                    "split": "val",
                    "status": "completed",
                    "auprc": 0.7,
                    "auc": 0.8,
                    "f1_at_tau_val": 0.5,
                    "f05_at_tau_val": 0.6,
                    "precision": 0.5,
                    "recall": 0.5,
                    "far": 0.1,
                    "mcc": 0.2,
                    "accuracy": 0.75,
                    "tp": 1,
                    "tn": 1,
                    "fp": 0,
                    "fn": 0,
                    "brier_score": 0.2,
                    "tau": 0.4,
                },
                {
                    "model": "MLP actual",
                    "baseline": "current",
                    "split": "test",
                    "status": "completed",
                    "auprc": 0.8,
                    "auc": 0.9,
                    "f1_at_tau_val": 0.6,
                    "f05_at_tau_val": 0.7,
                    "precision": 0.6,
                    "recall": 0.6,
                    "far": 0.05,
                    "mcc": 0.3,
                    "accuracy": 0.8,
                    "tp": 1,
                    "tn": 1,
                    "fp": 0,
                    "fn": 0,
                    "brier_score": 0.1,
                    "tau": 0.4,
                }
            ]
        )
        df.attrs["artifact_path"] = str(artifact_path)
        return df

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "RESULTADOS_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_list_gnn_model_files_for_baseline", lambda: [])
    monkeypatch.setattr(app, "_resolve_graph_hash_for_loaded_graph", lambda obj: "abc123")
    monkeypatch.setattr(baseline_module, "run_gnn_mlp_baselines", fake_runner)

    data = _make_graph()
    app._render_gnn_mlp_baseline_panel({"data": data}, data)

    assert calls["kwargs"]["baselines"] == (
        "current",
        "temporal",
        "xgboost_current",
        "xgboost_temporal",
        "svm_current",
        "svm_temporal",
    )
    assert calls["kwargs"]["graph_hash"] == "abc123"
    assert calls["kwargs"]["threshold_strategy"] == "far"
    assert calls["kwargs"]["far_target"] == pytest.approx(0.20)
    artifact_path = Path(fake_st.session_state["gnn_mlp_baseline_artifact_path"])
    assert artifact_path.exists()
    assert fake_st.dataframes
    assert set(fake_st.dataframes[-1]["split"]) == {"test"}
    assert "gnn_mlp_baseline_results" in fake_st.session_state


def test_gnn_mlp_history_filters_results_by_loaded_graph_hash(tmp_path: Path, monkeypatch):
    from src import graph_builder_app as app

    graph_hash = "a" * 64
    other_hash = "b" * 64
    history_dir = tmp_path / "gnn_mlp_baselines"
    history_dir.mkdir()

    pd.DataFrame(
        [
            {
                "model": "MLP actual",
                "split": "test",
                "status": "completed",
                "graph_hash": graph_hash,
                "auprc": 0.7,
            }
        ]
    ).to_csv(history_dir / f"mlp_baseline_20260101_000000_{graph_hash[:16]}.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "MLP actual",
                "split": "test",
                "status": "completed",
                "graph_hash": other_hash,
                "auprc": 0.2,
            }
        ]
    ).to_csv(history_dir / f"mlp_baseline_20260101_000001_{other_hash[:16]}.csv", index=False)

    monkeypatch.setattr(app, "RESULTADOS_DIR", str(tmp_path))

    entries = app._list_gnn_mlp_baseline_history(graph_hash)

    assert len(entries) == 1
    assert entries[0]["path"].name.endswith(f"{graph_hash[:16]}.csv")
    assert entries[0]["df"]["graph_hash"].iloc[0] == graph_hash


def test_gnn_mlp_history_selectbox_uses_scalar_options(tmp_path: Path, monkeypatch):
    from src import graph_builder_app as app

    graph_hash = "a" * 64
    result_path = tmp_path / "mlp_baseline.csv"
    history_df = pd.DataFrame(
        [
            {
                "model": "MLP actual",
                "split": "test",
                "status": "completed",
                "graph_hash": graph_hash,
                "auprc": 0.7,
                "auc": 0.8,
            },
            {
                "model": "XGBoost actual",
                "split": "test",
                "status": "completed",
                "graph_hash": graph_hash,
                "auprc": 0.8,
                "auc": 0.85,
                "model_family": "xgboost",
                "feature_view": "current",
            },
            {
                "model": "SVM temporal",
                "split": "test",
                "status": "completed",
                "graph_hash": graph_hash,
                "auprc": 0.6,
                "auc": 0.65,
                "model_family": "svm",
                "feature_view": "temporal",
            }
        ]
    )
    selected_options = {}

    class _FakeSt:
        def markdown(self, *args, **kwargs):
            return None

        def caption(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            raise AssertionError(args[0] if args else "unexpected warning")

        def info(self, *args, **kwargs):
            return None

        def dataframe(self, *args, **kwargs):
            return None

        def selectbox(self, label, options, index=0, key=None, format_func=None, **kwargs):
            selected_options["options"] = list(options)
            assert all(isinstance(option, str) for option in options)
            return options[index]

        def multiselect(self, label, options, default=None, key=None, **kwargs):
            return list(default or [])

        def plotly_chart(self, *args, **kwargs):
            return None

        def bar_chart(self, *args, **kwargs):
            return None

    monkeypatch.setattr(app, "st", _FakeSt())
    monkeypatch.setattr(
        app,
        "_resolve_graph_identity_for_loaded_graph",
        lambda graph_obj: {
            "graph_hash": graph_hash,
            "graph_file_hash": graph_hash,
            "graph_hash_source": "semantic_metadata",
        },
    )
    monkeypatch.setattr(
        app,
        "_list_gnn_mlp_baseline_history",
        lambda hash_value: [
            {
                "path": result_path,
                "df": history_df,
                "mtime": 1.0,
            }
        ],
    )

    app._render_gnn_mlp_baseline_history_panel({"graph_hash": graph_hash})

    assert selected_options["options"] == [str(result_path.resolve())]


def test_render_comparison_tab_uses_current_and_history_subtabs(monkeypatch):
    from src import graph_builder_app as app

    class _Tab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeSt:
        def __init__(self):
            self.session_state = {
                "loaded_graph": {
                    "data": _make_graph(),
                    "filename": "graph.pt",
                    "graph_hash": "a" * 64,
                }
            }
            self.tab_labels = []

        def subheader(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            raise AssertionError(args[0] if args else "unexpected warning")

        def checkbox(self, *args, **kwargs):
            return False

        def markdown(self, *args, **kwargs):
            return None

        def tabs(self, labels):
            self.tab_labels.append(list(labels))
            return [_Tab() for _ in labels]

    fake_st = _FakeSt()
    called = []
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(
        app,
        "_render_gnn_mlp_baseline_panel",
        lambda graph_obj, graph_data: called.append(("current", graph_obj["_comparison_source_label"])),
    )
    monkeypatch.setattr(
        app,
        "_render_gnn_mlp_baseline_history_panel",
        lambda graph_obj: called.append(("history", graph_obj["graph_hash"])),
    )

    app._render_comparison_tab()

    assert ["Estado actual", "Resultados anteriores"] in fake_st.tab_labels
    assert called == [("current", "Original"), ("history", "a" * 64)]


def test_gnn_checkpoint_comparison_calibrates_on_val_and_reports_test_only(tmp_path: Path, monkeypatch):
    from src import graph_builder_app as app
    from src import gnn_main

    data = _make_graph()
    model_path = tmp_path / "gat_model_BEST_fake.pt"
    model_path.write_bytes(b"fake")
    meta = {
        "hidden_channels": 4,
        "num_heads": 1,
        "num_layers": 1,
        "dropout": 0.0,
        "out_channels": 2,
        "best_epoch": 3,
        "best_val_auprc": 0.42,
    }
    calls = []

    class _FakeModel:
        def load_state_dict(self, state_dict, strict=True):
            self.loaded = (state_dict, strict)
            return [], []

        def eval(self):
            return self

    def fake_result(y_true):
        return {
            "report": {
                "Accidente (1)": {
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1-score": 2.0 / 3.0,
                },
                "accuracy": 0.75,
            },
            "true": torch.tensor(y_true, dtype=torch.long),
            "probs": torch.tensor(
                [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]],
                dtype=torch.float32,
            ),
            "auprc": 0.7,
            "auc": 0.8,
            "mcc": 0.4,
            "far": 0.1,
            "brier_score": 0.2,
            "cm": np.array([[1, 1], [0, 2]], dtype=int),
        }

    def fake_test(model, graph_data, **kwargs):
        calls.append(kwargs)
        mask_name = kwargs.get("mask_name")
        if mask_name == "val_calib":
            return {"val_calib": fake_result([0, 1, 1, 0])}
        if mask_name == "val_threshold":
            return {"val_threshold": fake_result([0, 1, 1, 0])}
        assert kwargs.get("masks") == ["test_mask"]
        return {"test_mask": fake_result([0, 1, 0, 1])}

    platt_model = object()

    def fake_platt_scale(y_true, y_prob):
        return np.asarray(y_prob, dtype=float), platt_model

    threshold_calls = []

    def fake_select_threshold(y_true, y_prob, *, far_target, mode):
        threshold_calls.append(
            {
                "y_true": np.asarray(y_true),
                "y_prob": np.asarray(y_prob),
                "far_target": far_target,
                "mode": mode,
            }
        )
        return 0.42, {"far": 0.1, "sens": 0.5}

    monkeypatch.setattr(app, "_load_hparams_for_model", lambda path: dict(meta))
    monkeypatch.setattr(app.torch, "load", lambda *args, **kwargs: {"weight": torch.ones(1)})
    monkeypatch.setattr(app, "_check_model_graph_compat", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(
        app,
        "_infer_arch_from_state_dict",
        lambda state_dict: {"hidden_channels": 4, "num_heads": 1, "num_layers": 1},
    )
    monkeypatch.setattr(app, "_resolve_gnn_variant_for_checkpoint", lambda *args: "gat_snapshot")
    monkeypatch.setattr(app, "_checkpoint_temporal_kind", lambda state_dict: None)
    monkeypatch.setattr(app, "_infer_edge_feature_dim", lambda graph_data: 0)
    monkeypatch.setattr(
        app,
        "_split_val_mask_for_calibration_threshold",
        lambda *args, **kwargs: {
            "calib_idx": torch.tensor([4, 5], dtype=torch.long),
            "threshold_idx": torch.tensor([4, 5], dtype=torch.long),
            "calibration_mask_source": "val_calib",
            "threshold_mask_source": "val_threshold",
            "calib_count": 2,
            "threshold_count": 2,
        },
    )
    monkeypatch.setattr(gnn_main, "_build_gnn_model", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(gnn_main, "test", fake_test)
    monkeypatch.setattr(gnn_main, "AUTOCALIBRATE_PROBS", True)
    monkeypatch.setattr(gnn_main, "_platt_scale_probabilities", fake_platt_scale)
    monkeypatch.setattr(app, "_select_threshold_for_far_target", fake_select_threshold)

    rows = app._evaluate_gnn_checkpoint_for_comparison(
        model_path=str(model_path),
        graph_obj={"data": data},
        graph_data=data,
        device="CPU",
        batch_size=4,
    )

    df = pd.DataFrame(rows)
    assert set(df["split"]) == {"test"}
    assert (df["status"] == "completed").all()
    assert df.loc[df["split"] == "test", "auprc"].iloc[0] == pytest.approx(0.7)
    assert df.loc[df["split"] == "test", "tp"].iloc[0] == 2
    assert df.loc[df["split"] == "test", "tn"].iloc[0] == 1
    assert df.loc[df["split"] == "test", "fp"].iloc[0] == 1
    assert df.loc[df["split"] == "test", "fn"].iloc[0] == 0
    assert df.loc[df["split"] == "test", "brier_score"].iloc[0] == pytest.approx(0.2)
    assert set(df["calibration_method"]) == {"platt_scaling"}
    assert df.loc[df["split"] == "test", "tau"].iloc[0] == pytest.approx(0.42)
    assert df.loc[df["split"] == "test", "tau_source"].iloc[0] == "val_threshold_far_target"
    assert df.loc[df["split"] == "test", "threshold_strategy"].iloc[0] == "far"
    assert df.loc[df["split"] == "test", "far_target"].iloc[0] == pytest.approx(0.20)
    assert df.loc[df["split"] == "test", "calibration_mask_source"].iloc[0] == "val_calib"
    assert threshold_calls and threshold_calls[0]["far_target"] == pytest.approx(0.20)
    assert [call.get("mask_name") for call in calls[:2]] == ["val_calib", "val_threshold"]
    eval_calls = [call for call in calls if call.get("masks") == ["test_mask"]]
    assert eval_calls and "test_mask" in eval_calls[-1]["masks"]
    assert eval_calls[-1]["threshold"] == pytest.approx(0.42)
    assert eval_calls[-1]["calibration_model"] is platt_model


def test_perform_model_evaluation_final_call_uses_only_test_mask(
    tmp_path: Path,
    monkeypatch,
):
    from src import graph_builder_app as app
    from src import gnn_main

    data = _make_graph()
    model_path = tmp_path / "gat_model_BEST_fake.pt"
    model_path.write_bytes(b"fake")
    calls = []

    class _FakeProgress:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def fail(self, *args, **kwargs):
            pass

        def complete(self, *args, **kwargs):
            pass

    class _FakeSt:
        session_state = {}

        def caption(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    class _FakeModel:
        def load_state_dict(self, state_dict, strict=True):
            self.loaded = (state_dict, strict)
            return [], []

        def eval(self):
            return self

    def fake_result(name):
        return {
            name: {
                "true": torch.tensor([0, 1], dtype=torch.long),
                "probs": torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32),
                "report": {"accuracy": 1.0},
                "cm": np.array([[1, 0], [0, 1]], dtype=int),
            }
        }

    def fake_test(model, graph_data, **kwargs):
        calls.append(kwargs)
        mask_name = kwargs.get("mask_name")
        if mask_name in {"val_calib", "val_threshold"}:
            return fake_result(mask_name)
        assert kwargs.get("masks") == ["test_mask"]
        return {}

    monkeypatch.setattr(app, "st", _FakeSt())
    monkeypatch.setattr(app, "_GNNGraphEvaluationProgress", _FakeProgress)
    monkeypatch.setattr(app, "_load_hparams_for_model", lambda path: {})
    monkeypatch.setattr(app.torch, "load", lambda *args, **kwargs: {"weight": torch.ones(1)})
    monkeypatch.setattr(app, "_check_model_graph_compat", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(
        app,
        "_split_val_mask_for_calibration_threshold",
        lambda *args, **kwargs: {
            "calib_idx": torch.tensor([4, 5], dtype=torch.long),
            "threshold_idx": torch.tensor([4, 5], dtype=torch.long),
        },
    )
    monkeypatch.setattr(app, "_select_threshold_for_far_target", lambda *args, **kwargs: (0.5, {}))
    monkeypatch.setattr(gnn_main, "_build_gnn_model", lambda **kwargs: _FakeModel())
    monkeypatch.setattr(gnn_main, "test", fake_test)
    monkeypatch.setattr(gnn_main, "AUTOCALIBRATE_PROBS", False)

    app._perform_model_evaluation(
        model_path=str(model_path),
        graph_data=data,
        device="cpu",
        threshold_strategy="far",
        masks=["val_mask", "test_mask"],
        batch_size=4,
    )

    assert [call.get("mask_name") for call in calls[:2]] == ["val_calib", "val_threshold"]
    final_calls = [call for call in calls if call.get("mask_name") is None]
    assert final_calls
    assert final_calls[-1]["masks"] == ["test_mask"]


def test_render_graph_builder_includes_comparison_tab(monkeypatch):
    from src import graph_builder_app as app

    class _Tab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeStTabs:
        def __init__(self):
            self.labels = []

        def tabs(self, labels):
            self.labels = list(labels)
            return [_Tab() for _ in labels]

        def radio(self, *args, **kwargs):
            return "En memoria"

    fake_st = _FakeStTabs()
    called = []
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "_render_history_tab", lambda: called.append("history"))
    monkeypatch.setattr(app, "_render_events_tab", lambda: called.append("events"))
    monkeypatch.setattr(app, "_render_feature_engineering", lambda: called.append("features"))
    monkeypatch.setattr(app, "render_feature_explorer", lambda: called.append("feature_explorer"))
    monkeypatch.setattr(app, "_render_feature_selection_tab", lambda: called.append("selection"))
    monkeypatch.setattr(app, "_render_in_memory_graph", lambda: called.append("graph"))
    monkeypatch.setattr(app, "_render_network_tab", lambda: called.append("network"))
    monkeypatch.setattr(app, "_render_optimization_tab", lambda: called.append("optimization"))
    monkeypatch.setattr(app, "_render_balance_tab", lambda: called.append("balance"))
    monkeypatch.setattr(app, "_render_training_tab", lambda: called.append("training"))
    monkeypatch.setattr(app, "_render_comparison_tab", lambda: called.append("comparison"))
    monkeypatch.setattr(app, "_render_evaluation_tab", lambda: called.append("evaluation"))
    monkeypatch.setattr(app, "_render_gnn_experiments_tab", lambda: called.append("experiments"))

    app.render_graph_builder()

    assert "Comparación" in fake_st.labels
    assert fake_st.labels.index("Comparación") > fake_st.labels.index("Training")
    assert "Network Builder" not in fake_st.labels
    assert "comparison" in called
    assert "network" in called
