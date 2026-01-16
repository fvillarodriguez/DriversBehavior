import pytest

from src.experiments_logic import ExperimentsRunner
from src.model_training import temporal_train_test_split, train_model
from tests.pipeline_helpers import build_synthetic_base_df


def test_full_pipeline_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)

    runner = ExperimentsRunner(random_state=42)
    importance_df = runner.calculate_feature_importance(
        base_df, feature_cols, n_estimators=20
    )
    assert not importance_df.empty

    model_params = {"n_estimators": 20, "max_depth": 3}
    result = train_model(
        base_df,
        feature_cols,
        "Random Forest",
        model_params,
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        random_state=42,
    )
    metrics = result.get("metrics", {})
    assert "f1" in metrics and "far" in metrics

    train_val_df, test_df = temporal_train_test_split(base_df, test_size=0.2)
    train_df, val_df = temporal_train_test_split(train_val_df, test_size=0.2)
    search_space = {
        "smote": {
            "k_neighbors": {"min": 1, "max": 1},
            "sampling_strategy": {"min": 1.0, "max": 1.0},
        },
        "model": {
            "n_estimators": {"min": 10, "max": 10},
            "max_depth": {"min": 0, "max": 0},
        },
    }
    optuna_result = runner.run_optimization_loop(
        train_df,
        val_df,
        test_df,
        feature_cols,
        "Random Forest",
        n_trials=2,
        timeout=30,
        far_target=0.2,
        search_space_config=search_space,
    )
    assert optuna_result["best_f1"] >= 0.0
