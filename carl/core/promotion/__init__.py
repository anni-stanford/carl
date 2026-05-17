"""Paired-bootstrap promotion gate.

The gate is the statistical test that turns CARL into a research artifact.
A candidate policy is promoted iff the 95 % CI lower bound (BCa, 10 000
resamples) of its paired reward lift over the baseline on a held-out probe
set of n ≥ 30 tasks is strictly greater than zero.
"""

from carl.core.promotion.gate import MIN_PROBE_TASKS, GateResult, evaluate_gate

__all__ = ["GateResult", "MIN_PROBE_TASKS", "evaluate_gate"]
