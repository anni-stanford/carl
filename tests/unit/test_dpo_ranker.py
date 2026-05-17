"""DPO ranker re-orders candidates per LLM ranking; falls back gracefully."""

from __future__ import annotations

import json

import pytest

from carl.core.diagnosis.types import FailureAttribution, TraceEvidence
from carl.core.dpo_ranker.ranker import rank_candidates
from carl.core.llm_client import FakeLLMClient
from carl.core.policy.artifacts import ArtifactType, PolicyDiff

pytestmark = pytest.mark.asyncio


def _attr() -> FailureAttribution:
    return FailureAttribution(
        failed_artifact_type=ArtifactType.RULES,
        failed_artifact_name="CLAUDE.md",
        evidence_from_trace=[
            TraceEvidence(event_kind="x", snippet="a"),
            TraceEvidence(event_kind="y", snippet="b"),
        ],
        root_cause="this root cause is at least twenty characters long",
        proposed_intervention="add_line",
        expected_lift=0.1,
        confidence=0.7,
    )


def _diff(name: str) -> PolicyDiff:
    return PolicyDiff(
        artifact_name=name,
        artifact_type=ArtifactType.RULES,
        operation="add_line",
        line_range=(10, 11),
        old_content=None,
        new_content="x",
        rationale=f"diff for {name}",
        expected_lift=0.1,
        confidence=0.7,
    )


async def test_ranker_reorders_per_llm_response() -> None:
    cands = [_diff("a"), _diff("b"), _diff("c")]
    client = FakeLLMClient(default=json.dumps({"ranking": [2, 0, 1]}))
    out = await rank_candidates(_attr(), cands, client=client, model="x")
    assert [c.artifact_name for c in out] == ["c", "a", "b"]


async def test_partial_ranking_fills_missing_in_input_order() -> None:
    cands = [_diff("a"), _diff("b"), _diff("c"), _diff("d")]
    client = FakeLLMClient(default=json.dumps({"ranking": [3, 1]}))
    out = await rank_candidates(_attr(), cands, client=client, model="x")
    # 3 ("d"), 1 ("b"), then a, c in input order
    assert [c.artifact_name for c in out] == ["d", "b", "a", "c"]


async def test_malformed_response_falls_back_to_input_order() -> None:
    cands = [_diff("a"), _diff("b")]
    client = FakeLLMClient(default="garbage")
    out = await rank_candidates(_attr(), cands, client=client, model="x")
    assert [c.artifact_name for c in out] == ["a", "b"]


async def test_single_candidate_short_circuits() -> None:
    cands = [_diff("solo")]
    client = FakeLLMClient(default="never used")
    out = await rank_candidates(_attr(), cands, client=client, model="x")
    assert out == cands
    assert client.call_count == 0
