"""LLM-driven failure attribution.

Given a failed :class:`Trajectory`, ask Claude Opus (or whichever
:class:`LLMClient` is configured) to produce a :class:`FailureAttribution`
in strict JSON. We validate the response with Pydantic; any malformed
attribution is rejected, the agent retries up to ``max_retries`` times,
and after that we surface ``None`` so the calling loop can skip mutation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import ValidationError

from carl.adapters.base import TraceEvent, Trajectory
from carl.core.diagnosis.types import FailureAttribution
from carl.core.llm_client import LLMClient

_PROMPT_TEMPLATE = """\
You are a CARL diagnosis agent. Read the failed trajectory below and
attribute the failure to a specific artifact in the agent's policy.

Output a single JSON object that matches this schema (strict; no extra keys):
{{
  "failed_artifact_type": one of "rules"|"skill"|"agent"|"hook"|"mcp_config"|"command",
  "failed_artifact_name": <string>,
  "failed_line_or_section": <string or null>,
  "evidence_from_trace": [
    {{"event_kind": <string>, "snippet": <string>, "file": <string or null>, "line": <int or null>}},
    ...  // MUST contain at least 2 entries
  ],
  "root_cause": <string of >= 20 characters>,
  "proposed_intervention": one of "add_line"|"edit_line"|"remove_line"|"create_skill"|"edit_skill"|"tighten_setting"|"modify_hook",
  "expected_lift": <float in [0, 1]>,
  "confidence": <float in [0, 1]>
}}

## Task description
{task_prompt}

## Active policy artifacts
{artifact_index}

## Trajectory events (truncated)
{events}

## CI output (last {ci_tail_len} chars)
{ci_tail}

Respond with the JSON object only. No prose, no Markdown fences.
"""


async def attribute_failure(
    trajectory: Trajectory,
    *,
    client: LLMClient,
    model: str,
    max_retries: int = 2,
) -> FailureAttribution | None:
    """Call the diagnosis LLM and return a validated :class:`FailureAttribution`."""
    prompt = _build_prompt(trajectory)
    last_error: str | None = None
    for _ in range(max_retries + 1):
        resp = await client.complete(
            prompt + (f"\n\n## Previous error\n{last_error}\nFix and retry." if last_error else ""),
            model=model,
            response_format="json",
            temperature=0.0,
            max_tokens=2048,
        )
        try:
            payload = json.loads(_strip_markdown_fences(resp.text))
            return FailureAttribution.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)[:500]
    return None


def _build_prompt(traj: Trajectory) -> str:
    artifact_index = "\n".join(
        f"- ({a.type.value}) {a.name}" for a in traj.policy.artifacts
    )
    events_block = _format_events(traj.events, max_events=40)
    ci_tail = (traj.raw_ci_output or "")[-2000:]
    return _PROMPT_TEMPLATE.format(
        task_prompt=traj.task.prompt,
        artifact_index=artifact_index,
        events=events_block,
        ci_tail_len=len(ci_tail),
        ci_tail=ci_tail,
    )


def _format_events(events: Sequence[TraceEvent], *, max_events: int = 40) -> str:
    if len(events) > max_events:
        events = list(events[:max_events // 2]) + list(events[-(max_events // 2):])
    return "\n".join(
        f"[{e.timestamp:.2f}] {e.kind}: {json.dumps(e.payload)[:300]}" for e in events
    )


def _strip_markdown_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite being asked not to."""
    import re

    s = text.strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", s, re.DOTALL)
    if fenced is not None:
        return fenced.group(1).strip()
    return s
