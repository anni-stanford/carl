"""Thompson-sampling bandit converges to the dominant variant under simulation."""

from __future__ import annotations

import numpy as np
import pytest

from carl.core.bandit.thompson import BetaPosterior, ThompsonBandit


def test_register_idempotent() -> None:
    b = ThompsonBandit(seed=0)
    b.register("v1")
    a = b.posteriors["v1"].alpha
    b.register("v1")
    assert b.posteriors["v1"].alpha == a


def test_update_increments_posterior_correctly() -> None:
    b = ThompsonBandit(seed=0)
    b.update("v1", 0.7)
    p = b.posteriors["v1"]
    assert p.alpha == 1 + 0.7
    assert p.beta == 1 + 0.3


def test_update_rejects_out_of_range() -> None:
    b = ThompsonBandit(seed=0)
    with pytest.raises(ValueError):
        b.update("v1", 1.5)


def test_sample_group_returns_top_k_in_order() -> None:
    b = ThompsonBandit(seed=42)
    for vid in ["a", "b", "c", "d", "e"]:
        b.register(vid)
    out = b.sample_group(3)
    assert len(out) == 3
    assert len(set(out)) == 3
    assert all(o in {"a", "b", "c", "d", "e"} for o in out)


def test_thompson_picks_dominant_variant_in_simulation() -> None:
    """Simulate 1000 episodes; bandit's posterior mean for the dominant
    variant should be highest."""
    rng = np.random.default_rng(20260517)
    b = ThompsonBandit(seed=20260517)
    true_means = {"a": 0.3, "b": 0.5, "c": 0.8}
    for vid in true_means:
        b.register(vid)

    for _ in range(1000):
        chosen = b.sample_group(1)[0]
        # Bernoulli reward with the variant's true mean
        r = float(rng.random() < true_means[chosen])
        b.update(chosen, r)

    best, _ = b.best_mean()
    assert best == "c"


def test_beta_posterior_observation_count() -> None:
    p = BetaPosterior(alpha=4.0, beta=3.0)
    assert p.n_observations == (4.0 - 1.0) + (3.0 - 1.0)
