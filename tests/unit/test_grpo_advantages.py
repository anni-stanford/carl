"""Group-relative advantage math is correct, including degenerate cases."""

from __future__ import annotations

import numpy as np
import pytest

from carl.core.grpo_scorer.advantage import (
    best_advantage_index,
    group_advantages,
    positive_advantage_mask,
)


def test_uniform_group_returns_zero_advantages() -> None:
    out = group_advantages([0.5, 0.5, 0.5, 0.5])
    assert out == [0.0, 0.0, 0.0, 0.0]


def test_advantages_zero_mean() -> None:
    out = group_advantages([0.1, 0.5, 0.9, 0.4])
    assert abs(sum(out)) < 1e-6


def test_advantages_unit_std() -> None:
    arr = np.asarray(group_advantages([0.1, 0.4, 0.9, 0.5]))
    assert abs(arr.std() - 1.0) < 1e-3


def test_best_index_is_max() -> None:
    assert best_advantage_index([0.2, 0.7, 0.1, 0.5]) == 1


def test_best_index_empty_raises() -> None:
    with pytest.raises(ValueError):
        best_advantage_index([])


def test_positive_mask_picks_above_average() -> None:
    mask = positive_advantage_mask([0.2, 0.5, 0.9])
    assert mask == [False, False, True]


def test_empty_input_returns_empty() -> None:
    assert group_advantages([]) == []
    assert positive_advantage_mask([]) == []
