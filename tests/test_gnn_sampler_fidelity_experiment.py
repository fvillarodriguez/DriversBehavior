import math

from src.graph_builder_app import (
    _build_sampler_fidelity_grid,
    _compute_sampler_fidelity_score,
    _rank_sampler_fidelity_results,
)


def test_build_sampler_fidelity_grid_cartesian_product():
    configs = _build_sampler_fidelity_grid(
        sampler_modes=["neighbor", "cluster_gcn", "graphsaint"],
        neighbor_profiles=[[15, 10], [25, 15, 10]],
        cluster_num_parts=[32],
        cluster_parts_per_epoch=[0, 4],
        graphsaint_modes=["node", "random_walk"],
        graphsaint_batch_sizes=[1024],
        graphsaint_num_steps=[4],
        graphsaint_walk_lengths=[2, 4],
        deterministic_sampling=True,
        sampling_seed=11,
        disable_hard_undersampling=True,
        eval_neighbors_mode="exhaustive",
        eval_num_neighbors=None,
        repeats=2,
    )

    # Per profile:
    # neighbor=1, cluster_gcn=2, graphsaint(node=1, random_walk=2) => 6
    # profiles=2 => 12, repeats=2 => 24.
    assert len(configs) == 24
    assert configs[0]["config_idx"] == 1
    assert configs[-1]["config_idx"] == 24

    seeds_by_repeat = {}
    for cfg in configs:
        if (
            cfg["train_sampler_mode"] == "neighbor"
            and cfg["num_neighbors"] == [15, 10]
        ):
            seeds_by_repeat[int(cfg["repeat"])] = int(cfg["sampling_seed"])
    assert seeds_by_repeat[1] == 11
    assert seeds_by_repeat[2] == 12

    node_walks = {
        int(cfg["graphsaint_walk_length"])
        for cfg in configs
        if cfg["train_sampler_mode"] == "graphsaint"
        and cfg["graphsaint_mode"] == "node"
    }
    assert node_walks == {1}

    rw_walks = {
        int(cfg["graphsaint_walk_length"])
        for cfg in configs
        if cfg["train_sampler_mode"] == "graphsaint"
        and cfg["graphsaint_mode"] == "random_walk"
    }
    assert rw_walks == {2, 4}


def test_compute_sampler_fidelity_score_handles_missing_metrics():
    score, deltas = _compute_sampler_fidelity_score(
        candidate_metrics={"test_f1": None},
        reference_metrics={"test_f1": 0.8},
        metric_weights={"test_f1": 3.0},
    )
    assert math.isinf(score)
    assert deltas == {}


def test_rank_sampler_fidelity_results_prefers_closest_reference():
    reference = {
        "test_f1": 0.80,
        "test_recall": 0.76,
        "test_far": 0.11,
    }
    rows = [
        {
            "config_name": "cfg_close",
            "status": "ok",
            "test_f1": 0.79,
            "test_recall": 0.75,
            "test_far": 0.12,
        },
        {
            "config_name": "cfg_far",
            "status": "ok",
            "test_f1": 0.70,
            "test_recall": 0.60,
            "test_far": 0.30,
        },
        {
            "config_name": "cfg_error",
            "status": "train_error",
            "test_f1": 0.95,
            "test_recall": 0.95,
            "test_far": 0.01,
        },
    ]

    ranked, best = _rank_sampler_fidelity_results(
        rows,
        reference_metrics=reference,
        metric_weights={
            "test_f1": 3.0,
            "test_recall": 2.0,
            "test_far": 2.0,
        },
    )

    assert best is not None
    assert best["config_name"] == "cfg_close"
    assert ranked[0]["config_name"] == "cfg_close"
    assert "delta_test_f1" in ranked[0]

    err_row = next(row for row in ranked if row["config_name"] == "cfg_error")
    assert math.isinf(float(err_row["fidelity_score"]))
