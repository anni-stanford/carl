"""Paired-bootstrap promotion gate behavior on synthetic deltas.

These tests don't validate scipy's bootstrap implementation; they validate
the *gate logic* — promotion criteria, n-tasks floor, p-value direction,
seed-based reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from carl.core.promotion.gate import evaluate_gate
from carl.settings import PromotionGateConfig


@pytest.fixture
def cfg() -> PromotionGateConfig:
    return PromotionGateConfig(
        n_resamples=2000,  # smaller in tests for speed
        confidence=0.95,
        min_probe_tasks=30,
        require_ci_lower_bound_above=0.0,
    )


def test_unequal_lengths_raise(cfg: PromotionGateConfig) -> None:
    with pytest.raises(ValueError, match="paired sequences"):
        evaluate_gate([0.5, 0.6], [0.4], cfg)


def test_below_min_probes_rejects(cfg: PromotionGateConfig) -> None:
    cand = [0.9] * 10  # huge lift but only 10 paired tasks
    base = [0.1] * 10
    out = evaluate_gate(cand, base, cfg)
    assert not out.promote
    assert "insufficient probe tasks" in out.reason
    assert out.n_tasks == 10


def test_strong_real_lift_promotes(cfg: PromotionGateConfig) -> None:
    rng = np.random.default_rng(42)
    base = rng.uniform(0.40, 0.55, size=40)
    cand = base + rng.uniform(0.05, 0.15, size=40)  # always positive lift
    out = evaluate_gate(list(cand), list(base), cfg, rng_seed=42)
    assert out.promote
    assert out.mean_lift > 0
    assert out.ci_low > 0
    assert out.p_value < 0.05
    assert out.n_tasks == 40


def test_no_lift_does_not_promote(cfg: PromotionGateConfig) -> None:
    rng = np.random.default_rng(7)
    base = rng.uniform(0.40, 0.55, size=40)
    cand = base + rng.normal(0, 0.05, size=40)  # zero-mean noise
    out = evaluate_gate(list(cand), list(base), cfg, rng_seed=7)
    # CI lower bound should not clear zero with this much variance
    assert not out.promote
    assert out.p_value >= 0.05 or out.ci_low <= 0


def test_negative_lift_rejected(cfg: PromotionGateConfig) -> None:
    rng = np.random.default_rng(13)
    base = rng.uniform(0.50, 0.65, size=40)
    cand = base - rng.uniform(0.05, 0.15, size=40)
    out = evaluate_gate(list(cand), list(base), cfg, rng_seed=13)
    assert not out.promote
    assert out.mean_lift < 0


def test_seeded_reproducibility(cfg: PromotionGateConfig) -> None:
    cand = [0.6 + 0.01 * i for i in range(40)]
    base = [0.55 + 0.01 * i for i in range(40)]
    a = evaluate_gate(cand, base, cfg, rng_seed=20260517)
    b = evaluate_gate(cand, base, cfg, rng_seed=20260517)
    assert a == b


def test_single_task_does_not_crash(cfg: PromotionGateConfig) -> None:
    """n=1 must return a graceful REJECT, never raise (bootstrap needs >= 2)."""
    relaxed = PromotionGateConfig(
        n_resamples=2000, confidence=0.95, min_probe_tasks=1,
        require_ci_lower_bound_above=0.0,
    )
    out = evaluate_gate([0.7], [0.5], relaxed, rng_seed=1)
    assert out.promote is False
    assert out.n_tasks == 1
    assert "insufficient probe tasks" in out.reason


def test_two_tasks_runs_bootstrap(cfg: PromotionGateConfig) -> None:
    """n=2 is the minimum the bootstrap can handle; must not raise."""
    relaxed = PromotionGateConfig(
        n_resamples=2000, confidence=0.95, min_probe_tasks=2,
        require_ci_lower_bound_above=0.0,
    )
    out = evaluate_gate([0.7, 0.8], [0.5, 0.6], relaxed, rng_seed=1)
    assert out.n_tasks == 2
