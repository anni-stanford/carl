"""Thompson sampling over policy variants — Bernoulli–Beta with reward in [0, 1].

Each variant maintains a ``Beta(α, β)`` posterior over its expected reward.
When ``sample_group(K)`` is called, we draw one sample from each posterior
and return the K variants with the highest sampled values. ``update(variant,
reward)`` treats the bounded reward as a fractional success: ``α += r``,
``β += 1 − r``. This is the standard Bernoulli-Beta extension to bounded
continuous rewards (Agrawal & Goyal, 2012, "Analysis of Thompson Sampling
for the Multi-armed Bandit Problem").

Automatic exploration: a fresh variant starts ``α = β = 1`` (uniform prior),
so its samples have high variance and frequently win the top-K draw — that
is the exploration. As more episodes land, the posterior tightens and
exploitation takes over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BetaPosterior:
    """``Beta(α, β)`` posterior over the expected reward of one variant."""

    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n_observations(self) -> float:
        return (self.alpha - 1.0) + (self.beta - 1.0)


@dataclass
class ThompsonBandit:
    """Thompson-sampling bandit over an evolving set of named policy variants."""

    posteriors: dict[str, BetaPosterior] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def register(self, variant_id: str) -> None:
        if variant_id not in self.posteriors:
            self.posteriors[variant_id] = BetaPosterior()

    def update(self, variant_id: str, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward must be in [0, 1], got {reward}")
        self.register(variant_id)
        post = self.posteriors[variant_id]
        post.alpha += reward
        post.beta += 1.0 - reward

    def sample_group(self, k: int, candidate_ids: list[str] | None = None) -> list[str]:
        """Draw one sample from each posterior; return the top-K variants by sampled value.

        ``candidate_ids`` restricts the draw to a subset (used to gate the
        bandit to only the *active* variants in the candidate pool). When
        ``None``, draws over every registered variant.
        """
        ids = candidate_ids if candidate_ids is not None else list(self.posteriors)
        if not ids:
            return []
        for vid in ids:
            self.register(vid)
        samples = np.array(
            [self._rng.beta(self.posteriors[vid].alpha, self.posteriors[vid].beta) for vid in ids]
        )
        order = np.argsort(samples)[::-1]
        k = min(k, len(ids))
        return [ids[i] for i in order[:k]]

    def best_mean(self) -> tuple[str, float]:
        """Return the variant with the highest posterior mean and that mean."""
        if not self.posteriors:
            raise ValueError("no variants registered")
        vid = max(self.posteriors, key=lambda v: self.posteriors[v].mean)
        return vid, self.posteriors[vid].mean
