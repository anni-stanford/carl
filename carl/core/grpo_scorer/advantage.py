"""GRPO-style group-relative advantage at trajectory level.

For each task, run K trajectories under K policy variants, then compute
per-trajectory advantage:

    advantage_i = (r_i - mean(r_1..K)) / (std(r_1..K) + eps)

The transposition from weight-space GRPO (Shao et al., DeepSeekMath, 2024):
weight-space GRPO computes group-relative advantages over K **rollouts of
the same policy** to estimate a gradient. CARL computes group-relative
advantages over K **policy variants on the same task** to decide which
variant to **promote**. Same math, different update target.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def group_advantages(rewards: list[float] | np.ndarray) -> list[float]:
    """Return one advantage per element of ``rewards``, group-normalized.

    A degenerate group (all rewards equal) returns all-zero advantages. This
    is the correct behavior because no variant can dominate in this case.
    """
    arr = np.asarray(rewards, dtype=float)
    if arr.size == 0:
        return []
    mean = float(arr.mean())
    std = float(arr.std())
    if std < EPS:
        return [0.0] * arr.size
    return [(float(r) - mean) / (std + EPS) for r in arr]


def best_advantage_index(rewards: list[float] | np.ndarray) -> int:
    """Index of the single best trajectory in the group (max advantage)."""
    arr = np.asarray(rewards, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot pick best from empty group")
    return int(arr.argmax())


def positive_advantage_mask(rewards: list[float] | np.ndarray) -> list[bool]:
    """Boolean mask of variants with strictly positive group-relative advantage."""
    return [a > 0.0 for a in group_advantages(rewards)]
