"""Paired-bootstrap promotion gate.

A candidate policy is promoted iff the 95 % CI lower bound of its mean
reward lift over the baseline (computed via paired bootstrap with 10 000
resamples) is strictly greater than zero on a held-out probe set of n ≥ 30
tasks. This is the statistical test that turns CARL from "we changed some
prompts and the number went up" into a research artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap

from carl.settings import PromotionGateConfig

MIN_PROBE_TASKS = 30  # Reviewers expect n ≥ 30 for paired-bootstrap CIs to be defensible.


@dataclass(frozen=True)
class GateResult:
    """Output of the promotion gate.

    Always reported in the paper — promote/reject is the binary, but the
    decomposition (mean lift, CI bounds, p-value, n) is what reviewers audit.
    """

    promote: bool
    mean_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_tasks: int
    n_resamples: int
    confidence: float
    reason: str  # human-readable why the decision was made


def evaluate_gate(
    candidate_rewards: Sequence[float],
    baseline_rewards: Sequence[float],
    config: PromotionGateConfig,
    rng_seed: int | None = None,
) -> GateResult:
    """Run the paired-bootstrap promotion gate.

    Both ``candidate_rewards`` and ``baseline_rewards`` must be **paired**:
    the i-th element of each comes from the **same task** under the candidate
    policy and the baseline policy respectively. This pairing dramatically
    reduces variance versus an independent-samples test (Efron & Tibshirani,
    1993, "An Introduction to the Bootstrap", §15).
    """
    if len(candidate_rewards) != len(baseline_rewards):
        raise ValueError(
            f"paired sequences must be equal length: "
            f"{len(candidate_rewards)} vs {len(baseline_rewards)}"
        )
    n = len(candidate_rewards)
    if n < config.min_probe_tasks:
        return GateResult(
            promote=False,
            mean_lift=0.0,
            ci_low=0.0,
            ci_high=0.0,
            p_value=1.0,
            n_tasks=n,
            n_resamples=config.n_resamples,
            confidence=config.confidence,
            reason=(
                f"insufficient probe tasks: n={n} < min_probe_tasks="
                f"{config.min_probe_tasks}"
            ),
        )

    deltas = np.asarray(candidate_rewards, dtype=float) - np.asarray(
        baseline_rewards, dtype=float
    )
    mean_lift = float(deltas.mean())

    rng = np.random.default_rng(rng_seed)
    res = bootstrap(
        (deltas,),
        np.mean,
        n_resamples=config.n_resamples,
        confidence_level=config.confidence,
        random_state=rng,
        method="BCa",  # bias-corrected and accelerated; tighter CIs than percentile
    )
    ci_low = float(res.confidence_interval.low)
    ci_high = float(res.confidence_interval.high)

    p_value = _one_sided_bootstrap_p(deltas, n_resamples=config.n_resamples, rng=rng)

    threshold = config.require_ci_lower_bound_above
    promote = mean_lift > 0.0 and ci_low > threshold

    if promote:
        reason = (
            f"PROMOTE: mean lift {mean_lift:+.4f}, "
            f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}], p={p_value:.4f}, n={n}"
        )
    else:
        reason = (
            f"REJECT: mean lift {mean_lift:+.4f}, "
            f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}] does not clear threshold "
            f"{threshold:+.4f}; p={p_value:.4f}, n={n}"
        )

    return GateResult(
        promote=promote,
        mean_lift=mean_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_tasks=n,
        n_resamples=config.n_resamples,
        confidence=config.confidence,
        reason=reason,
    )


def _one_sided_bootstrap_p(
    deltas: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> float:
    """One-sided bootstrap p-value: P(resampled mean ≤ 0 | observed deltas).

    Computes the fraction of bootstrap-resampled means that are ≤ 0 after
    centering, which is the standard non-parametric one-sided p-value for
    the null hypothesis "candidate is no better than baseline".
    """
    if deltas.size == 0:
        return 1.0
    centered = deltas - deltas.mean()  # null: mean lift is zero
    n = deltas.size
    indices = rng.integers(low=0, high=n, size=(n_resamples, n))
    resampled_means = centered[indices].mean(axis=1)
    observed_mean = deltas.mean()
    # P(bootstrap mean >= observed) under the null
    p = float(np.mean(resampled_means >= observed_mean))
    # Clamp to (0, 1) to avoid p=0 when the observed mean is outside every resample
    return max(p, 1.0 / (n_resamples + 1))
