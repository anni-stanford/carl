"""LLM judge with bias controls — RLAIF with three defenses against judge bias.

Defenses:

1. **Position-flip** for pairwise comparisons. Run ``(A, B)`` and ``(B, A)``;
   only count consistent wins.
2. **Family rotation** across the call set: at most one judgement per
   ``(provider family)``. The rotation list is configured in
   :class:`JudgeConfig`. We compute and report inter-judge agreement.
3. **Rubric-shuffle** for absolute scoring. Same rubric, two random orderings;
   the score is the mean.

The judge is invoked through :class:`carl.core.llm_client.LLMClient`, which
makes the entire judge unit-testable with :class:`FakeLLMClient` — no
network calls in tests.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass

from carl.core.llm_client import LLMClient
from carl.core.reward.types import JudgeComponents
from carl.settings import JudgeConfig

# Regex to pull a 0-1 score out of a model's structured response.
_SCORE_RE = re.compile(r'"?score"?\s*:\s*(-?[\d.]+)')


@dataclass(frozen=True)
class JudgePromptParts:
    """Pieces of the judge prompt that get composed by :func:`absolute_score`."""

    task_description: str
    candidate_output: str
    rubric_items: list[str]


async def absolute_score(
    parts: JudgePromptParts,
    *,
    client: LLMClient,
    config: JudgeConfig,
    rng_seed: int = 20260517,
) -> JudgeComponents:
    """Score a single trajectory's output against a rubric in ``[0, 1]``.

    Two defenses are applied:

    - The ``primary_model`` is called once with the rubric in canonical order
      and once with a deterministically shuffled order; the mean is taken.
    - The judge is also called on each rotation model exactly once. The
      reported ``score`` is the median across the rotation set, which is
      robust to any single judge being an outlier.
    """
    rng = random.Random(rng_seed)
    canonical_rubric = list(parts.rubric_items)
    shuffled_rubric = list(canonical_rubric)
    rng.shuffle(shuffled_rubric)

    models = [config.primary_model, *config.rotation_models]

    raw_scores: list[float] = []
    for model in models:
        s_can = await _ask_for_score(parts, canonical_rubric, model=model, client=client)
        if config.rubric_shuffle:
            s_shuf = await _ask_for_score(parts, shuffled_rubric, model=model, client=client)
            score = 0.5 * (s_can + s_shuf)
        else:
            score = s_can
        raw_scores.append(_clip(score))

    if not raw_scores:
        return JudgeComponents(
            score=0.0, inter_judge_agreement=0.0, n_judges_consistent=0, n_judges_total=0
        )

    median = sorted(raw_scores)[len(raw_scores) // 2]
    spread = max(raw_scores) - min(raw_scores)
    agreement = max(0.0, 1.0 - spread)  # high spread => low agreement

    return JudgeComponents(
        score=float(median),
        inter_judge_agreement=float(agreement),
        n_judges_consistent=sum(1 for s in raw_scores if abs(s - median) <= 0.10),
        n_judges_total=len(raw_scores),
    )


async def pairwise_winner(
    task_description: str,
    output_a: str,
    output_b: str,
    *,
    client: LLMClient,
    config: JudgeConfig,
) -> str | None:
    """Decide whether A or B is better, with **position-flip**.

    Returns ``"A"`` if A wins both ``(A, B)`` and ``(B, A)`` orderings,
    ``"B"`` if B wins both, or ``None`` if the orderings disagree
    (inconsistent → not counted, per Hu et al. 2024 on judge bias).
    """
    if not config.position_flip:
        # Without position-flip we'd just trust one ordering; rejected by docs/reward_design.md.
        raise RuntimeError("position_flip must be enabled for pairwise judgement")

    forward = await _pairwise_call(task_description, output_a, output_b, client=client, model=config.primary_model)
    reverse = await _pairwise_call(task_description, output_b, output_a, client=client, model=config.primary_model)
    if forward == "A" and reverse == "B":
        return "A"
    if forward == "B" and reverse == "A":
        return "B"
    return None


# ----- internals -------------------------------------------------------------


async def _ask_for_score(
    parts: JudgePromptParts,
    rubric: list[str],
    *,
    model: str,
    client: LLMClient,
) -> float:
    rubric_block = "\n".join(f"- {item}" for item in rubric)
    prompt = (
        "You are an evaluator. Score the candidate output against the rubric.\n"
        "Respond with strict JSON: {\"score\": <float in [0,1]>, \"reasoning\": <string>}\n\n"
        f"## Task\n{parts.task_description}\n\n"
        f"## Candidate Output\n{parts.candidate_output}\n\n"
        f"## Rubric\n{rubric_block}\n"
    )
    resp = await client.complete(prompt, model=model, response_format="json", temperature=0.0)
    return _extract_score(resp.text)


async def _pairwise_call(
    task: str, a: str, b: str, *, client: LLMClient, model: str
) -> str | None:
    prompt = (
        "Compare two candidate outputs A and B for the task. Choose the better one.\n"
        'Respond with strict JSON: {"winner": "A"|"B", "reasoning": <string>}\n\n'
        f"## Task\n{task}\n\n## Candidate A\n{a}\n\n## Candidate B\n{b}\n"
    )
    resp = await client.complete(prompt, model=model, response_format="json", temperature=0.0)
    try:
        data = json.loads(resp.text)
        winner = data.get("winner")
        if winner == "A":
            return "A"
        if winner == "B":
            return "B"
    except json.JSONDecodeError:
        pass
    return None


def _extract_score(text: str) -> float:
    """Robust 0-1 score extraction from a (possibly noisy) judge response."""
    m = _SCORE_RE.search(text)
    if m is None:
        return 0.0
    try:
        return _clip(float(m.group(1)))
    except ValueError:
        return 0.0


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def stable_seed_for_task(task_id: str) -> int:
    """Deterministic seed derived from the task id — same task → same shuffle."""
    h = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")
