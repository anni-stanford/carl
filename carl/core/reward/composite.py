"""Composite reward = w_v · r_verifier + w_j · r_judge − w_h · r_hack.

The composition is intentionally trivial; all the interesting code is in the
sub-modules (verifier / judge / hack_probe). Keeping this thin makes the
ablation studies clean: each ablation toggles one of the three terms.
"""

from __future__ import annotations

from carl.core.reward.types import (
    HackComponents,
    JudgeComponents,
    RewardComponents,
    VerifierComponents,
)
from carl.settings import RewardWeights


def compose_reward(
    verifier: VerifierComponents,
    judge: JudgeComponents | None,
    hack: HackComponents | None,
    weights: RewardWeights,
    judge_gate_threshold: float = 0.5,
) -> RewardComponents:
    """Combine the three reward components into a final scalar in ``[0, 1]``.

    The ``judge_gate_threshold`` enforces an important honesty property: the
    LLM judge cannot rescue a trajectory that fails CI. If the verifier score
    is below threshold, the judge contribution is forced to zero. Without
    this, a model could learn to write convincing-looking code that doesn't
    actually pass tests.
    """
    r_verifier = verifier.composite
    r_judge = judge.score if (judge is not None and r_verifier >= judge_gate_threshold) else 0.0
    r_hack = hack.penalty if hack is not None else 0.0

    r_total = (
        weights.verifier * r_verifier
        + weights.judge * r_judge
        - weights.hack_penalty * r_hack
    )
    r_total = max(0.0, min(1.0, r_total))

    return RewardComponents(
        r_total=r_total,
        r_verifier=r_verifier,
        r_judge=r_judge,
        r_hack=r_hack,
        verifier_breakdown=verifier,
        judge_breakdown=judge,
        hack_breakdown=hack,
    )
