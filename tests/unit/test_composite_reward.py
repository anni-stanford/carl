"""Composite reward respects weights, judge gating, and stays in [0, 1]."""

from __future__ import annotations

from carl.core.reward.composite import compose_reward
from carl.core.reward.types import (
    HackComponents,
    JudgeComponents,
    VerifierComponents,
)
from carl.settings import RewardWeights


def _vc(score: float) -> VerifierComponents:
    return VerifierComponents(
        tests_passed=score,
        coverage_delta=score,
        lint_clean=score,
        typecheck_clean=score,
        security_clean=score,
        composite=score,
        raw_test_count=10,
        raw_failed_count=0,
    )


def test_failing_verifier_zeroes_judge_contribution() -> None:
    """A judge cannot rescue a trajectory whose CI failed (anti-hacking property)."""
    out = compose_reward(
        verifier=_vc(0.1),  # below the 0.5 gate threshold
        judge=JudgeComponents(score=0.95, inter_judge_agreement=1.0, n_judges_consistent=4, n_judges_total=4),
        hack=None,
        weights=RewardWeights(),
        judge_gate_threshold=0.5,
    )
    assert out.r_judge == 0.0
    # r_total should be (0.65 * 0.1) = 0.065
    assert abs(out.r_total - 0.65 * 0.1) < 1e-9


def test_hack_penalty_subtracts() -> None:
    out = compose_reward(
        verifier=_vc(0.9),
        judge=None,
        hack=HackComponents(penalty=0.5, detected_patterns=("test_deletion",)),
        weights=RewardWeights(),
    )
    expected = 0.65 * 0.9 - 0.10 * 0.5  # judge term zero (no judge)
    assert abs(out.r_total - expected) < 1e-9


def test_total_clamped_to_unit_interval() -> None:
    # Construct a pathological case to verify clamping
    out = compose_reward(
        verifier=_vc(0.0),
        judge=None,
        hack=HackComponents(penalty=10.0, detected_patterns=("foo",)),
        weights=RewardWeights(),
    )
    assert out.r_total == 0.0


def test_decomposition_passed_through() -> None:
    out = compose_reward(
        verifier=_vc(0.7),
        judge=JudgeComponents(score=0.8, inter_judge_agreement=0.9, n_judges_consistent=3, n_judges_total=4),
        hack=HackComponents(penalty=0.0, detected_patterns=()),
        weights=RewardWeights(),
    )
    assert out.r_verifier == 0.7
    assert out.r_judge == 0.8
    assert out.r_hack == 0.0
    assert out.verifier_breakdown.tests_passed == 0.7
    assert out.judge_breakdown is not None
    assert out.judge_breakdown.inter_judge_agreement == 0.9
