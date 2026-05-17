"""Diagnosis agent: structured-output validation rejects malformed responses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carl.adapters.base import Task, TraceEvent, Trajectory
from carl.core.diagnosis.attributor import attribute_failure
from carl.core.diagnosis.types import FailureAttribution
from carl.core.llm_client import FakeLLMClient
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy

pytestmark = pytest.mark.asyncio


def _trajectory() -> Trajectory:
    policy = Policy(
        artifacts=[
            Artifact(name="CLAUDE.md", type=ArtifactType.RULES, content="# rules"),
            Artifact(name="testing-policy", type=ArtifactType.SKILL, content="# pytest"),
        ],
        version="v0",
    )
    task = Task(
        task_id="t",
        repo_path=Path("/x"),
        prompt="Fix the failing test in tests/test_a.py",
        adapter_name="claude_code",
    )
    return Trajectory(
        task=task,
        policy=policy,
        events=[
            TraceEvent(timestamp=0.1, kind="tool_call", payload={"tool": "edit", "file": "a.py"}),
            TraceEvent(timestamp=0.5, kind="ci_output", payload={"text": "FAILED tests/test_a.py::test_x"}),
        ],
        files_changed=["a.py"],
        exit_code=1,
        duration_s=12.5,
        raw_ci_output="==== FAILURES ====\ntest_x AssertionError\n",
    )


async def test_valid_response_parses() -> None:
    valid = json.dumps(
        {
            "failed_artifact_type": "rules",
            "failed_artifact_name": "CLAUDE.md",
            "failed_line_or_section": "tests/test_a.py",
            "evidence_from_trace": [
                {"event_kind": "ci_output", "snippet": "FAILED test_x"},
                {"event_kind": "tool_call", "snippet": "edit a.py"},
            ],
            "root_cause": "Project rule does not require running tests before commit",
            "proposed_intervention": "add_line",
            "expected_lift": 0.15,
            "confidence": 0.8,
        }
    )
    client = FakeLLMClient(default=valid)
    out = await attribute_failure(_trajectory(), client=client, model="claude-opus-4-7")
    assert isinstance(out, FailureAttribution)
    assert out.failed_artifact_type == ArtifactType.RULES
    assert len(out.evidence_from_trace) == 2


async def test_too_few_evidence_entries_retries_then_fails() -> None:
    too_few = json.dumps(
        {
            "failed_artifact_type": "rules",
            "failed_artifact_name": "CLAUDE.md",
            "evidence_from_trace": [{"event_kind": "x", "snippet": "y"}],
            "root_cause": "this root cause is at least twenty characters long",
            "proposed_intervention": "add_line",
            "expected_lift": 0.1,
            "confidence": 0.7,
        }
    )
    client = FakeLLMClient(default=too_few)
    out = await attribute_failure(
        _trajectory(), client=client, model="claude-opus-4-7", max_retries=1
    )
    assert out is None
    # Verifies the retry semantics (initial + 1 retry = 2 calls)
    assert client.call_count == 2


async def test_malformed_json_retries() -> None:
    client = FakeLLMClient(default="not even json{")
    out = await attribute_failure(_trajectory(), client=client, model="x", max_retries=2)
    assert out is None
    assert client.call_count == 3


async def test_markdown_fenced_json_is_stripped() -> None:
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "failed_artifact_type": "skill",
                "failed_artifact_name": "testing-policy",
                "evidence_from_trace": [
                    {"event_kind": "ci_output", "snippet": "x"},
                    {"event_kind": "tool_call", "snippet": "y"},
                ],
                "root_cause": "skill omits the strict-markers flag for pytest",
                "proposed_intervention": "edit_skill",
                "expected_lift": 0.05,
                "confidence": 0.9,
            }
        )
        + "\n```"
    )
    client = FakeLLMClient(default=fenced)
    out = await attribute_failure(_trajectory(), client=client, model="x")
    assert out is not None
    assert out.failed_artifact_type == ArtifactType.SKILL
