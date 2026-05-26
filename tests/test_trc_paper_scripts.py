"""
Unit tests for the pure functions inside src/trc_paper/*.py scripts.

These tests do not invoke the CLI — they call the helper functions directly
with synthetic inputs to validate the math:

  • markov_matrix.compute_p_matrix      → row-stochastic, correct estimator
  • markov_matrix.bootstrap_p           → returns (B, K, K) tensor
  • markov_matrix.long_format           → bijection with state index
  • stationary_asymmetry.long_to_matrix
  • stationary_asymmetry.stationary_distribution → π is left eigenvector
  • stationary_asymmetry.mixing_time             → ∞ for non-mixing
  • stationary_asymmetry.kolmogorov_cycle_test   → detects asymmetry
  • stationary_asymmetry.asymmetry_pairs         → top-k correctness
  • homogeneity_test.long_to_matrices and matrix → tests
  • homogeneity_test.total_variation
  • homogeneity_test.robust_chi2_homogeneity     → high stat for distinct P's
  • covid_decomposition.stationary
  • event_matching.cohens_d                      → sign and known cases
  • integration_h_bound: end-to-end H(π) ≤ Ē[H] + R
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_script(name: str):
    """Load a numbered-script-style module from src/trc_paper/."""
    spec = importlib.util.spec_from_file_location(
        f"trc_paper_test_{name}",
        REPO_ROOT / "src" / "trc_paper" / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def markov_mod():
    return _load_script("markov_matrix")


@pytest.fixture(scope="module")
def stationary_mod():
    return _load_script("stationary_asymmetry")


@pytest.fixture(scope="module")
def homogeneity_mod():
    return _load_script("homogeneity_test")


@pytest.fixture(scope="module")
def covid_mod():
    return _load_script("covid_decomposition")


@pytest.fixture(scope="module")
def events_mod():
    return _load_script("event_matching")


@pytest.fixture(scope="module")
def integration_mod():
    return _load_script("integration_h_bound")


# ---------------------------------------------------------------------------
# markov_matrix
# ---------------------------------------------------------------------------


def _synthetic_pairs_hard(n_plates: int = 50, n_steps: int = 10, K: int = 3, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic (t, t+1) pairs frame with HARD labels only."""
    rng = np.random.default_rng(seed)
    rows = []
    # True transition matrix used to simulate
    P_true = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ])
    for m in range(n_plates):
        state = rng.integers(0, K)
        for _ in range(n_steps):
            next_state = rng.choice(K, p=P_true[state])
            rows.append({
                "plate": f"P{m:04d}",
                "state": int(state),
                "next_state": int(next_state),
            })
            state = next_state
    return pd.DataFrame(rows)


def _synthetic_pairs_soft(n_plates: int = 30, n_steps: int = 6, K: int = 3, seed: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """Build a synthetic pairs frame WITH soft probabilities."""
    rng = np.random.default_rng(seed)
    prob_cols = [f"cluster_prob_{k}" for k in range(K)]
    next_prob_cols = [f"next_cluster_prob_{k}" for k in range(K)]
    rows = []
    for m in range(n_plates):
        for _ in range(n_steps):
            rt = rng.dirichlet(np.ones(K))
            rt1 = rng.dirichlet(np.ones(K))
            row = {"plate": f"P{m:04d}",
                   "state": int(np.argmax(rt)),
                   "next_state": int(np.argmax(rt1))}
            for c, v in zip(prob_cols, rt):
                row[c] = float(v)
            for c, v in zip(next_prob_cols, rt1):
                row[c] = float(v)
            rows.append(row)
    return pd.DataFrame(rows), prob_cols


class TestComputePMatrix:
    def test_hard_estimator_recovers_true_P(self, markov_mod) -> None:
        df = _synthetic_pairs_hard(n_plates=400, n_steps=40, K=3, seed=42)
        state_index = {0: 0, 1: 1, 2: 2}
        P = markov_mod.compute_p_matrix(df, prob_cols=[], state_index=state_index)
        P_true = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ])
        assert P.shape == (3, 3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(P, P_true, atol=0.05)

    def test_soft_estimator_is_row_stochastic(self, markov_mod) -> None:
        df, prob_cols = _synthetic_pairs_soft(n_plates=100, n_steps=20, K=4, seed=7)
        state_index = {0: 0, 1: 1, 2: 2, 3: 3}
        P = markov_mod.compute_p_matrix(df, prob_cols=prob_cols, state_index=state_index)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
        assert P.shape == (4, 4)
        assert (P >= 0).all()

    def test_empty_rows_yield_zero(self, markov_mod) -> None:
        df = pd.DataFrame(columns=["plate", "state", "next_state"])
        P = markov_mod.compute_p_matrix(df, prob_cols=[], state_index={0: 0, 1: 1})
        assert P.shape == (2, 2)
        assert (P == 0).all()


class TestBootstrapP:
    def test_returns_correct_shape(self, markov_mod) -> None:
        df = _synthetic_pairs_hard(n_plates=20, n_steps=5, K=3)
        state_index = {0: 0, 1: 1, 2: 2}
        cube = markov_mod.bootstrap_p(df, prob_cols=[], state_index=state_index,
                                      n_replicas=5, random_state=123)
        assert cube.shape == (5, 3, 3)
        # Each replica is row-stochastic
        for b in range(5):
            np.testing.assert_allclose(cube[b].sum(axis=1), 1.0, atol=1e-9)

    def test_zero_replicas_returns_empty_cube(self, markov_mod) -> None:
        df = _synthetic_pairs_hard(n_plates=5, n_steps=3, K=3)
        state_index = {0: 0, 1: 1, 2: 2}
        cube = markov_mod.bootstrap_p(df, prob_cols=[], state_index=state_index,
                                      n_replicas=0, random_state=0)
        assert cube.shape == (0, 3, 3)


class TestLongFormat:
    def test_round_trip(self, markov_mod) -> None:
        state_index = {0: 0, 1: 1, 5: 2}
        P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.0, 0.4, 0.6]])
        long = markov_mod.long_format(P, state_index)
        assert set(long.columns) == {"from_state", "to_state", "P_ij"}
        assert len(long) == 9
        # Reconstruct matrix
        P_back = np.zeros((3, 3))
        for _, row in long.iterrows():
            i = state_index[int(row["from_state"])]
            j = state_index[int(row["to_state"])]
            P_back[i, j] = row["P_ij"]
        np.testing.assert_allclose(P, P_back, atol=1e-12)


# ---------------------------------------------------------------------------
# stationary_asymmetry
# ---------------------------------------------------------------------------


class TestStationary:
    def test_long_to_matrix_normalizes_rows(self, stationary_mod) -> None:
        df = pd.DataFrame([
            {"from_state": 0, "to_state": 0, "P_ij": 1.0},
            {"from_state": 0, "to_state": 1, "P_ij": 3.0},
            {"from_state": 1, "to_state": 0, "P_ij": 0.0},
            {"from_state": 1, "to_state": 1, "P_ij": 0.0},  # row sums to 0 → kept zero
        ])
        P, state_idx = stationary_mod.long_to_matrix(df)
        # First row normalized
        np.testing.assert_allclose(P[0], [0.25, 0.75], atol=1e-12)
        # Second row stayed zero (no normalization because denom==0)
        np.testing.assert_allclose(P[1], [0.0, 0.0], atol=1e-12)

    def test_stationary_distribution_sums_to_one(self, stationary_mod) -> None:
        P = np.array([
            [0.5, 0.5],
            [0.2, 0.8],
        ])
        pi = stationary_mod.stationary_distribution(P)
        assert abs(pi.sum() - 1.0) < 1e-9
        # Stationary: πP == π
        np.testing.assert_allclose(pi @ P, pi, atol=1e-8)

    def test_stationary_uniform_for_doubly_stochastic(self, stationary_mod) -> None:
        # Doubly stochastic ⇒ uniform stationary
        P = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        pi = stationary_mod.stationary_distribution(P)
        np.testing.assert_allclose(pi, [0.5, 0.5], atol=1e-8)

    def test_mixing_time_finite_for_aperiodic(self, stationary_mod) -> None:
        P = np.array([
            [0.5, 0.5],
            [0.2, 0.8],
        ])
        tau = stationary_mod.mixing_time(P)
        assert tau > 0 and np.isfinite(tau)

    def test_mixing_time_infinite_for_identity(self, stationary_mod) -> None:
        P = np.eye(3)
        tau = stationary_mod.mixing_time(P)
        assert np.isinf(tau)

    def test_kolmogorov_detects_asymmetry(self, stationary_mod) -> None:
        # Strongly non-reversible cycle 0 → 1 → 2 → 0 with high probability
        P = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        result = stationary_mod.kolmogorov_cycle_test(P)
        assert result["max_log_ratio"] > 0.0

    def test_asymmetry_pairs_top_k(self, stationary_mod) -> None:
        P = np.array([
            [0.5, 0.4, 0.1],
            [0.1, 0.5, 0.4],
            [0.4, 0.1, 0.5],
        ])
        pi = np.array([1/3, 1/3, 1/3])
        state_inv = {0: 0, 1: 1, 2: 2}
        df = stationary_mod.asymmetry_pairs(P, pi, state_inv, top_k=2)
        assert len(df) == 2
        assert "abs_A" in df.columns
        # Must be sorted descending by abs_A
        assert df["abs_A"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# homogeneity_test
# ---------------------------------------------------------------------------


class TestHomogeneity:
    def test_total_variation_distinct_matrices(self, homogeneity_mod) -> None:
        P1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        P2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        tv = homogeneity_mod.total_variation(P1, P2)
        assert abs(tv - 1.0) < 1e-12

    def test_total_variation_identity(self, homogeneity_mod) -> None:
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        assert homogeneity_mod.total_variation(P, P) == 0.0

    def test_robust_chi2_zero_when_matrices_match_pooled(self, homogeneity_mod) -> None:
        P_pooled = np.array([[0.6, 0.4], [0.3, 0.7]])
        denom = np.array([100.0, 100.0])
        P_per_split = {"a": P_pooled.copy(), "b": P_pooled.copy()}
        denom_per_split = {"a": denom, "b": denom}
        out = homogeneity_mod.robust_chi2_homogeneity(P_per_split, denom_per_split, P_pooled)
        assert out["statistic"] == pytest.approx(0.0, abs=1e-9)
        assert out["p_value"] == pytest.approx(1.0, abs=1e-6)

    def test_robust_chi2_positive_when_matrices_differ(self, homogeneity_mod) -> None:
        P_pooled = np.array([[0.5, 0.5], [0.5, 0.5]])
        denom = np.array([200.0, 200.0])
        P_per_split = {
            "a": np.array([[0.9, 0.1], [0.1, 0.9]]),
            "b": np.array([[0.1, 0.9], [0.9, 0.1]]),
        }
        denom_per_split = {"a": denom, "b": denom}
        out = homogeneity_mod.robust_chi2_homogeneity(P_per_split, denom_per_split, P_pooled)
        assert out["statistic"] > 50.0  # very large for these synthetic matrices
        assert out["p_value"] < 1e-6


# ---------------------------------------------------------------------------
# covid_decomposition
# ---------------------------------------------------------------------------


class TestCovid:
    def test_stationary_matches_eigvec(self, covid_mod) -> None:
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = covid_mod.stationary(P)
        assert abs(pi.sum() - 1.0) < 1e-9
        np.testing.assert_allclose(pi @ P, pi, atol=1e-8)


# ---------------------------------------------------------------------------
# event_matching
# ---------------------------------------------------------------------------


class TestEventMatchingHelpers:
    def test_cohens_d_zero_for_identical_means(self, events_mod) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert events_mod.cohens_d(a, b) == pytest.approx(0.0)

    def test_cohens_d_sign(self, events_mod) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        d = events_mod.cohens_d(a, b)
        assert d < 0  # a < b

    def test_cohens_d_nan_for_tiny_samples(self, events_mod) -> None:
        a = np.array([1.0])
        b = np.array([2.0, 3.0, 4.0])
        d = events_mod.cohens_d(a, b)
        assert np.isnan(d)


# ---------------------------------------------------------------------------
# integration_h_bound — end-to-end mini test
# ---------------------------------------------------------------------------


class TestIntegrationHBound:
    def test_bound_holds_on_synthetic_inputs(
        self, tmp_path: Path, integration_mod
    ) -> None:
        """H(π) ≤ Ē[H] + R must hold for any valid distributions."""
        # Build a 3-state stationary JSON
        pi = {"0": 0.5, "1": 0.3, "2": 0.2}
        H_pi = float(-sum(v * np.log(v) for v in pi.values()))
        stat_payload = {
            "stationary_pi": pi,
            "entropy_pi": H_pi,
            "state_index": {"0": 0, "1": 1, "2": 2},
        }
        stat_path = tmp_path / "stat.json"
        stat_path.write_text(__import__("json").dumps(stat_payload))

        # Build an H_{p,τ} parquet with shares per row + H per row
        rows = []
        for portico in ["P1", "P2", "P3"]:
            for tau_minute in range(30):
                shares = np.random.dirichlet([2.0, 2.0, 2.0])
                row = {
                    "portico": portico,
                    "tau": pd.Timestamp("2022-01-01") + pd.Timedelta(minutes=15 * tau_minute),
                    "share_0": float(shares[0]),
                    "share_1": float(shares[1]),
                    "share_2": float(shares[2]),
                    "H": float(-np.sum(shares * np.log(np.clip(shares, 1e-12, 1.0)))),
                }
                rows.append(row)
        h_path = tmp_path / "H.parquet"
        pd.DataFrame(rows).to_parquet(h_path, index=False)

        # Empty homogeneity placeholder (not used in the bound numerics)
        homog_path = tmp_path / "homog.json"
        homog_path.write_text("{}")

        out_result = tmp_path / "result.json"
        out_crosstab = tmp_path / "crosstab.parquet"

        # Build argv and run main()
        argv = [
            "integration_h_bound.py",
            "--stationary", str(stat_path),
            "--h-15min", str(h_path),
            "--homogeneity", str(homog_path),
            "--output-result", str(out_result),
            "--output-crosstab", str(out_crosstab),
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            exit_code = integration_mod.main()
        finally:
            sys.argv = old_argv

        assert exit_code == 0
        payload = __import__("json").loads(out_result.read_text())
        assert payload["bound_holds"] is True
        assert payload["gap_upper_minus_Hpi"] >= -1e-6
        # bar_H is bounded above by log(K) = log(3) ≈ 1.0986
        assert payload["bar_H"] <= np.log(3) + 1e-9
        assert payload["R_mean_KL"] >= -1e-9
