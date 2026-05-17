"""Mutation proposer enforces locality budget; rejects out-of-budget candidates."""

from __future__ import annotations

import json

import pytest

from carl.core.diagnosis.types import FailureAttribution, TraceEvidence
from carl.core.llm_client import FakeLLMClient
from carl.core.mutation.proposer import propose_candidates
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy

pytestmark = pytest.mark.asyncio


def _attribution() -> FailureAttribution:
    return FailureAttribution(
        failed_artifact_type=ArtifactType.RULES,
        failed_artifact_name="CLAUDE.md",
        failed_line_or_section="line 12",
        evidence_from_trace=[
            TraceEvidence(event_kind="tool_call", snippet="edit a.py"),
            TraceEvidence(event_kind="ci_output", snippet="FAILED test_x"),
        ],
        root_cause="rules do not enforce running pytest before submitting",
        proposed_intervention="add_line",
        expected_lift=0.1,
        confidence=0.7,
    )


def _policy() -> Policy:
    return Policy(
        artifacts=[Artifact(name="CLAUDE.md", type=ArtifactType.RULES, content="# rules\n")],
        version="v0",
    )


async def test_three_in_budget_two_out_of_budget_only_three_returned() -> None:
    response = {
        "candidates": [
            {
                "operation": "add_line",
                "artifact_name": "CLAUDE.md",
                "artifact_type": "rules",
                "line_range": [10, 13],  # 4 lines, in budget
                "old_content": None,
                "new_content": "Always run pytest before submitting",
                "rationale": "Prevents the failure mode observed in the trace evidence",
                "expected_lift": 0.1,
                "confidence": 0.8,
            },
            {
                "operation": "edit_line",
                "artifact_name": "CLAUDE.md",
                "artifact_type": "rules",
                "line_range": [10, 18],  # 9 lines, OUT of budget
                "old_content": "old",
                "new_content": "new",
                "rationale": "Sweeping change that the proposer should not be doing",
                "expected_lift": 0.1,
                "confidence": 0.5,
            },
            {
                "operation": "create_skill",
                "artifact_name": "pre-test-runner",
                "artifact_type": "skill",
                "line_range": None,  # skill creation: bypasses line budget
                "old_content": None,
                "new_content": "# Skill\nRun pytest before any commit.\n",
                "rationale": "New skill addresses the failure cluster cleanly",
                "expected_lift": 0.15,
                "confidence": 0.7,
            },
            {
                "operation": "edit_line",
                "artifact_name": "CLAUDE.md",
                "artifact_type": "rules",
                "line_range": [10, 100],  # 91 lines, OUT
                "old_content": "old",
                "new_content": "new",
                "rationale": "Way out of locality budget; should be filtered",
                "expected_lift": 0.05,
                "confidence": 0.4,
            },
            {
                "operation": "tighten_setting",
                "artifact_name": "settings.json",
                "artifact_type": "mcp_config",
                "line_range": [5, 7],  # 3 lines, in budget
                "old_content": "old",
                "new_content": "new",
                "rationale": "Tighten the tool allowlist per the trace evidence",
                "expected_lift": 0.08,
                "confidence": 0.6,
            },
        ]
    }
    client = FakeLLMClient(default=json.dumps(response))
    out = await propose_candidates(
        _attribution(), _policy(), client=client, model="claude-opus-4-7"
    )
    # 3 in-budget candidates expected (add_line, create_skill, tighten_setting)
    assert len(out) == 3
    operations = {c.operation for c in out}
    assert operations == {"add_line", "create_skill", "tighten_setting"}


async def test_malformed_response_returns_empty_list() -> None:
    client = FakeLLMClient(default="not valid json at all")
    out = await propose_candidates(
        _attribution(), _policy(), client=client, model="claude-opus-4-7"
    )
    assert out == []
