"""Mutation proposer — turns a :class:`FailureAttribution` into ``N`` ranked candidate
:class:`PolicyDiff` objects subject to the locality budget.

The proposer is intentionally conservative: every diff must be ≤ ``max_diff_lines``,
new skill creation requires a failure-cluster size ≥ ``min_failure_cluster``,
and ``settings.json`` mutations are restricted to the tool allowlist /
disallowlist (no schema changes). Hooks are checked against ``shellcheck``
elsewhere (gate); here we just produce candidates.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from carl.core.diagnosis.types import FailureAttribution
from carl.core.llm_client import LLMClient
from carl.core.policy.artifacts import ArtifactType, Policy, PolicyDiff

_PROMPT_TEMPLATE = """\
You are a CARL mutation proposer. Given a failure attribution and the current policy,
propose {n_candidates} candidate diffs to repair the agent's behavior.

Constraints:
- Every diff edits at most {max_diff_lines} contiguous lines (or, for new skills,
  produces a single SKILL.md with rationale).
- settings.json edits are restricted to "tools.allowed" and "tools.disallowed"
  lists; never change the schema or non-tool keys.
- Hook scripts must end in a newline and start with `#!/usr/bin/env bash`.
- Each diff must reference at least one piece of trace evidence from the
  attribution's evidence_from_trace.

Respond with a single JSON object:
{{
  "candidates": [
    {{
      "operation": "add_line"|"edit_line"|"remove_line"|"create_skill"|"edit_skill"|"tighten_setting"|"modify_hook",
      "artifact_name": <string>,
      "artifact_type": "rules"|"skill"|"agent"|"hook"|"mcp_config"|"command",
      "line_range": [start, end] or null,
      "old_content": <string or null>,
      "new_content": <string or null>,
      "rationale": <string>,
      "expected_lift": <float in [0, 1]>,
      "confidence": <float in [0, 1]>
    }},
    ...
  ]
}}

## Failure attribution
{attribution_json}

## Current policy artifact summary
{artifact_summary}

Respond with JSON only.
"""


class _CandidateRecord(BaseModel):
    operation: str
    artifact_name: str
    artifact_type: ArtifactType
    line_range: tuple[int, int] | None = None
    old_content: str | None = None
    new_content: str | None = None
    rationale: str = Field(..., min_length=10)
    expected_lift: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class _CandidateBatch(BaseModel):
    candidates: list[_CandidateRecord]


async def propose_candidates(
    attribution: FailureAttribution,
    policy: Policy,
    *,
    client: LLMClient,
    model: str,
    n_candidates: int = 5,
    max_diff_lines: int = 5,
) -> list[PolicyDiff]:
    """Call the mutator LLM, validate output, and enforce the locality budget."""
    prompt = _PROMPT_TEMPLATE.format(
        n_candidates=n_candidates,
        max_diff_lines=max_diff_lines,
        attribution_json=attribution.model_dump_json(indent=2),
        artifact_summary="\n".join(
            f"- {a.type.value}/{a.name} ({len(a.content.splitlines())} lines)" for a in policy.artifacts
        ),
    )
    resp = await client.complete(prompt, model=model, response_format="json", temperature=0.2)
    text = _strip_markdown_fences(resp.text)
    try:
        batch = _CandidateBatch.model_validate_json(text)
    except (json.JSONDecodeError, ValidationError):
        return []

    diffs: list[PolicyDiff] = []
    for c in batch.candidates:
        diff = PolicyDiff(
            artifact_name=c.artifact_name,
            artifact_type=c.artifact_type,
            operation=c.operation,
            line_range=c.line_range,
            old_content=c.old_content,
            new_content=c.new_content,
            rationale=c.rationale,
            expected_lift=c.expected_lift,
            confidence=c.confidence,
        )
        if diff.is_within_locality_budget(max_diff_lines):
            diffs.append(diff)
    return diffs


def _strip_markdown_fences(text: str) -> str:
    import re

    s = text.strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", s, re.DOTALL)
    if fenced is not None:
        return fenced.group(1).strip()
    return s
