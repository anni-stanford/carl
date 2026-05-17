"""DPO-style preference ranker (v1: structured-output prompt; v2: classifier).

The v1 ranker asks an Opus-class model to score every candidate diff
against the failure context; v2 (planned for Day 10) trains a real
classifier on the accumulated ``preference_pairs`` table once we have
≥ 1000 pairs. The interface is the same so the loop doesn't change.

Cuts gate cost ≈ 3× by sending only the top-K candidates to the
expensive paired-bootstrap promotion gate.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from carl.core.diagnosis.types import FailureAttribution
from carl.core.llm_client import LLMClient
from carl.core.policy.artifacts import PolicyDiff

_PROMPT = """\
You are a CARL DPO ranker. Given a failure context and a list of candidate
mutation diffs, rank them by expected ability to fix the failure.

Respond with JSON: {{"ranking": [<index>, <index>, ...]}}
where indices refer to the input candidates (0-based). Highest-quality first.

## Failure attribution
{attribution_json}

## Candidates
{candidates_json}

Respond with JSON only.
"""


class _Ranking(BaseModel):
    ranking: list[int] = Field(..., min_length=1)


async def rank_candidates(
    attribution: FailureAttribution,
    candidates: list[PolicyDiff],
    *,
    client: LLMClient,
    model: str,
) -> list[PolicyDiff]:
    """Return ``candidates`` re-ordered by the LLM ranker.

    Falls back to the input order when the LLM output is malformed (so the
    promotion gate still gets *some* candidates to evaluate).
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates)

    candidates_json = json.dumps(
        [
            {
                "index": i,
                "operation": c.operation,
                "artifact_name": c.artifact_name,
                "artifact_type": c.artifact_type.value,
                "line_range": list(c.line_range) if c.line_range else None,
                "rationale": c.rationale,
                "expected_lift": c.expected_lift,
                "confidence": c.confidence,
            }
            for i, c in enumerate(candidates)
        ],
        indent=2,
    )
    prompt = _PROMPT.format(
        attribution_json=attribution.model_dump_json(indent=2),
        candidates_json=candidates_json,
    )
    resp = await client.complete(prompt, model=model, response_format="json", temperature=0.0)
    try:
        ranking = _Ranking.model_validate_json(_strip_markdown_fences(resp.text))
    except Exception:  # noqa: BLE001 — any parser failure: fall back deterministically
        return list(candidates)

    seen: set[int] = set()
    out: list[PolicyDiff] = []
    for idx in ranking.ranking:
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            out.append(candidates[idx])
    # Append any candidates the ranker forgot, preserving input order
    for i, c in enumerate(candidates):
        if i not in seen:
            out.append(c)
    return out


def _strip_markdown_fences(text: str) -> str:
    import re

    s = text.strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", s, re.DOTALL)
    if fenced is not None:
        return fenced.group(1).strip()
    return s
