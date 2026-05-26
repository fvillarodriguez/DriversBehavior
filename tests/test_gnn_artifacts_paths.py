from __future__ import annotations

import importlib.util
from pathlib import Path

from src.gnn_artifacts import gnn_dir, gnn_glob, gnn_path


def _load_migrator():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_gnn_resultados.py"
    spec = importlib.util.spec_from_file_location("migrate_gnn_resultados", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gnn_paths_live_under_gnn_root(tmp_path: Path) -> None:
    root = tmp_path / "Resultados"
    assert gnn_dir("models_gat", root) == root / "gnn" / "models" / "gat"
    assert (
        gnn_path("graphs_graphsmote", "graph_aug.pt", resultados_dir=root)
        == root / "gnn" / "graphs" / "balanced" / "graphsmote" / "graph_aug.pt"
    )


def test_gnn_glob_is_strict_to_new_tree(tmp_path: Path) -> None:
    root = tmp_path / "Resultados"
    legacy = root / "gat_model_BEST_legacy.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    current = gnn_path("models_gat", "gat_model_BEST_current.pt", resultados_dir=root, create_parent=True)
    current.write_text("current", encoding="utf-8")

    matches = gnn_glob("models_gat", "gat_model_BEST*.pt", root)

    assert matches == [str(current)]


def test_migrator_moves_gnn_root_files_and_writes_manifest(tmp_path: Path) -> None:
    migrator = _load_migrator()
    root = tmp_path / "Resultados"
    root.mkdir()
    model = root / "gat_model_BEST_demo.pt"
    hparams = root / "gat_model_BEST_demo_hparams.json"
    graph = root / "highway_graph_stream_build_demo.pt"
    history = root / "gnn_history.jsonl"
    feature_selection = root / "feature_selection_features_snapshot_gnn_demo.json"
    live_dir = root / "gnn_eval_live"
    model.write_text("model", encoding="utf-8")
    hparams.write_text("{}", encoding="utf-8")
    graph.write_text("graph", encoding="utf-8")
    history.write_text("{}\n", encoding="utf-8")
    feature_selection.write_text("{}", encoding="utf-8")
    live_dir.mkdir()
    (live_dir / "events.jsonl").write_text("{}", encoding="utf-8")

    dry = migrator.migrate(root, execute=False)
    assert dry["count"] == 6
    assert model.exists()

    result = migrator.migrate(root, execute=True)

    assert result["count"] == 6
    assert (root / "gnn" / "models" / "gat" / model.name).exists()
    assert (root / "gnn" / "models" / "gat" / hparams.name).exists()
    assert (root / "gnn" / "graphs" / "base" / graph.name).exists()
    assert (root / "gnn" / history.name).exists()
    assert (root / "gnn" / "features" / feature_selection.name).exists()
    assert (root / "gnn" / "evaluation" / "live_runs" / "events.jsonl").exists()
    assert list((root / "gnn").glob("migration_manifest_*.json"))
