"""Judge bias controls: position-flip, family-rotation, rubric-shuffle."""

from __future__ import annotations

import pytest

from carl.core.llm_client import FakeLLMClient
from carl.core.reward.judge import (
    JudgePromptParts,
    absolute_score,
    pairwise_winner,
    stable_seed_for_task,
)
from carl.settings import JudgeConfig

pytestmark = pytest.mark.asyncio


@pytest.fixture
def parts() -> JudgePromptParts:
    return JudgePromptParts(
        task_description="Implement add(a, b)",
        candidate_output="def add(a, b):\n    return a + b\n",
        rubric_items=["Correctness", "Style", "Tests"],
    )


@pytest.fixture
def cfg() -> JudgeConfig:
    return JudgeConfig(
        primary_model="claude-opus-4-7",
        rotation_models=["gpt-5.5", "claude-sonnet-4-6"],
        position_flip=True,
        consistent_only=True,
        rubric_shuffle=True,
    )


async def test_absolute_score_invokes_every_family(parts: JudgePromptParts, cfg: JudgeConfig) -> None:
    client = FakeLLMClient(default='{"score": 0.7, "reasoning": "ok"}')
    out = await absolute_score(parts, client=client, config=cfg)
    # primary + 2 rotation = 3 models; with rubric_shuffle each model gets 2 calls = 6 total
    assert client.call_count == 6
    families_called = {client.family(model) for model, _ in client.calls}
    assert families_called == {"anthropic", "openai"}
    assert 0.6 < out.score < 0.8
    assert 0 <= out.inter_judge_agreement <= 1


async def test_pairwise_only_counts_consistent_winners(parts: JudgePromptParts, cfg: JudgeConfig) -> None:
    """When (A, B) says A wins and (B, A) says A wins (i.e. positional bias
    flipped, A wins both orderings), the result must be A. When orderings
    disagree, result is None."""
    consistent = FakeLLMClient(default='{"winner": "A", "reasoning": "..."}')
    out = await pairwise_winner("task", "candidate A", "candidate B", client=consistent, config=cfg)
    # Both calls return A → A wins forward, B wins reverse (because in reverse, A is in B-slot)
    # → inconsistent → None
    assert out is None

    # Now a client whose winner depends on position; never consistent
    inconsistent = FakeLLMClient(
        script={"## Candidate A\ncandidate A\n\n## Candidate B\ncandidate B": '{"winner": "A"}'},
        default='{"winner": "A"}',
    )
    out = await pairwise_winner("task", "x", "y", client=inconsistent, config=cfg)
    assert out is None


async def test_stable_seed_deterministic() -> None:
    a = stable_seed_for_task("task_42")
    b = stable_seed_for_task("task_42")
    c = stable_seed_for_task("task_43")
    assert a == b
    assert a != c


async def test_judge_score_extraction_robust() -> None:
    """The judge tolerates a noisy model that wraps JSON in prose."""
    client = FakeLLMClient(default='Here is the score:\n{"score": 0.42, "reasoning": "..."}\nThanks!')
    cfg = JudgeConfig(
        primary_model="claude-opus-4-7",
        rotation_models=[],
        position_flip=True,
        consistent_only=True,
        rubric_shuffle=False,
    )
    parts = JudgePromptParts(
        task_description="t", candidate_output="o", rubric_items=["r1"]
    )
    out = await absolute_score(parts, client=client, config=cfg)
    assert abs(out.score - 0.42) < 1e-6
