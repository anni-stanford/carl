"""Composite reward stack.

- :mod:`carl.core.reward.verifier` — RLVR backbone (deterministic CI signal).
- :mod:`carl.core.reward.composite` — weighted aggregation w/ judge gating.
- :mod:`carl.core.reward.types` — :class:`RewardComponents` decomposition.
- :mod:`carl.core.reward.judge` — RLAIF with bias controls (Day 5).
- :mod:`carl.core.reward.hack_probe` — adversarial reward-hacking probes (Day 5).
"""

from carl.core.reward.composite import compose_reward
from carl.core.reward.types import (
    HackComponents,
    JudgeComponents,
    RewardComponents,
    VerifierComponents,
)
from carl.core.reward.verifier import CIArtifacts, compute_verifier

__all__ = [
    "CIArtifacts",
    "HackComponents",
    "JudgeComponents",
    "RewardComponents",
    "VerifierComponents",
    "compose_reward",
    "compute_verifier",
]
