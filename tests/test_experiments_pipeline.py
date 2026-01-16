import pytest

from src.experiments_logic import ExperimentsRunner
from tests.pipeline_helpers import build_synthetic_base_df


def test_experiments_loop(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    pytest.importorskip("imblearn")

    base_df, feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)

    runner = ExperimentsRunner(random_state=42)
    base_importance = runner.calculate_feature_importance(
        base_df, base_cols, n_estimators=10
    )
    base_ordered = base_importance["variable"].tolist()
    combined_importance = runner.calculate_feature_importance(
        base_df, feature_cols, n_estimators=10
    )
    combined_ordered = combined_importance["variable"].tolist()

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
    results = runner.run_iterative_experiment(
        base_df=base_df,
        base_features_ordered=base_ordered,
        cluster_features=combined_ordered,
        model_choice="Random Forest",
        n_trials=1,
        timeout=30,
        far_target=0.2,
        search_space_config=search_space,
        step_size=5,
        test_size=0.2,
        val_size=0.2,
    )
    assert results
    types = {row["type"] for row in results}
    assert "Base" in types
    assert "Base+Cluster" in types
