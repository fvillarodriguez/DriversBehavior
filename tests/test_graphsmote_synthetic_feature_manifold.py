import torch
import pytest


def test_smote_returns_parent_metadata_for_feature_interpolation():
    from src import graphsmote

    z = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    rng = graphsmote.np.random.RandomState(7)

    syn_z, syn_y, parents = graphsmote._smote_in_z_space(
        z,
        y,
        minority_class=1,
        k=1,
        n_samples=3,
        rng=rng,
        return_parent_info=True,
    )

    assert syn_z.shape == (3, 2)
    assert syn_y.tolist() == [1, 1, 1]
    assert set(parents["base_idx"].tolist()).issubset({0, 2})
    assert set(parents["neighbor_idx"].tolist()).issubset({0, 2})
    assert torch.all((parents["alpha"] >= 0.0) & (parents["alpha"] <= 1.0))


def test_feature_interp_synthetics_are_convex_train_positive_features(monkeypatch):
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    from src import graphsmote

    data = HeteroData()
    data["pm"].x = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 2.0, 2.0],
            [10.0, 10.0, 3.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    data["pm"].y = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.long)
    data["pm"].train_mask = torch.tensor([True, True, True, True, True, True])
    data["pm"].val_mask = torch.zeros(6, dtype=torch.bool)
    data["pm"].test_mask = torch.zeros(6, dtype=torch.bool)

    z_train = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-0.5, -0.5],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(
        graphsmote,
        "compute_train_embeddings",
        lambda *args, **kwargs: {"pm": z_train},
    )
    monkeypatch.setattr(
        graphsmote,
        "_generate_edges_for_synthetics",
        lambda *args, **kwargs: None,
    )

    model = torch.nn.Linear(3, 2)
    aug, registry = graphsmote.augment_graph_offline_once(
        model=model,
        data=data,
        device=torch.device("cpu"),
        target_pos_ratio=0.6,
        z2x_decoders=None,
        k=1,
        edge_gen=None,
        seed=11,
        synthetic_feature_mode="feature_interp",
    )

    synth = aug["pm"].is_synthetic.bool()
    assert int(synth.sum()) == 2
    assert bool(aug["pm"].train_mask.bool()[synth].all())
    assert int((synth & aug["pm"].val_mask.bool()).sum()) == 0
    assert int((synth & aug["pm"].test_mask.bool()).sum()) == 0

    pm_registry = registry["pm"]
    x_train = data["pm"].x[data["pm"].train_mask].float()
    base = pm_registry["smote_parent_base_idx"].long()
    neighbor = pm_registry["smote_parent_neighbor_idx"].long()
    alpha = pm_registry["smote_alpha"].float().view(-1, 1)
    expected = (1.0 - alpha) * x_train.index_select(0, base) + alpha * x_train.index_select(0, neighbor)

    torch.testing.assert_close(aug["pm"].x[synth].float(), expected)
    assert pm_registry["synthetic_feature_mode"] == "feature_interp"
    assert pm_registry["synthetic_feature_quality"]["feature_outside_train_positive_minmax_frac"] == 0.0
