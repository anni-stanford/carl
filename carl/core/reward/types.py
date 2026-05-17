"""Shared dataclasses for the reward stack.

Every reward computation returns a :class:`RewardComponents` rather than a bare
scalar. The decomposition is what makes the paper's ablation tables and
"what drove the lift?" qualitative analysis possible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerifierComponents:
    """Sub-scores of the deterministic CI-based verifier reward.

    All fields are normalized to ``[0.0, 1.0]``. A ``None`` indicates that the
    signal was not measured for this episode (e.g. the repo has no mypy
    config); ``composite`` already accounts for missing-signal renormalization.
    """

    tests_passed: float
    coverage_delta: float
    lint_clean: float
    typecheck_clean: float | None
    security_clean: float | None
    composite: float  # weighted sum after renormalization for missing signals
    raw_test_count: int
    raw_failed_count: int


@dataclass(frozen=True)
class JudgeComponents:
    """Sub-scores from the bias-controlled LLM-judge.

    ``score`` is averaged across the rotation set after position-flip pruning
    of inconsistent calls. ``inter_judge_agreement`` is reported in the paper
    so reviewers can audit the judge's reliability.
    """

    score: float
    inter_judge_agreement: float
    n_judges_consistent: int
    n_judges_total: int


@dataclass(frozen=True)
class HackComponents:
    """Adversarial reward-hacking probe penalty.

    Higher ``penalty`` means more exploit patterns detected. Subtracted from
    the composite reward.
    """

    penalty: float
    detected_patterns: tuple[str, ...]


@dataclass(frozen=True)
class RewardComponents:
    """Full reward decomposition for a single trajectory.

    ``r_total`` is what the loop, the bandit, and the gate see.
    The other fields are kept for the dashboard, ablations, and the paper.
    """

    r_total: float
    r_verifier: float
    r_judge: float
    r_hack: float
    verifier_breakdown: VerifierComponents
    judge_breakdown: JudgeComponents | None
    hack_breakdown: HackComponents | None
