"""Structured-output schema for the failure-attribution agent.

The diagnosis agent receives a failed :class:`Trajectory` and returns a
:class:`FailureAttribution` instance. Pydantic validates the LLM's response;
any field outside the enumerated literals raises a ``ValidationError`` and
the diagnosis is rejected (rather than silently corrupting the mutation
proposer's input).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from carl.core.policy.artifacts import ArtifactType


class TraceEvidence(BaseModel):
    """One concrete piece of evidence linking a failure to an artifact line.

    The diagnosis prompt requires at least two of these per attribution; this
    is what makes the post-hoc audit possible (every promoted diff cites the
    trace events that motivated it).
    """

    event_kind: str = Field(..., description='e.g. "tool_call", "stdout", "ci_output"')
    snippet: str = Field(..., min_length=1)
    file: str | None = None
    line: int | None = None


class FailureAttribution(BaseModel):
    """The diagnosis agent's structured verdict on one failed trajectory."""

    failed_artifact_type: ArtifactType
    failed_artifact_name: str = Field(..., min_length=1)
    failed_line_or_section: str | None = None
    evidence_from_trace: list[TraceEvidence] = Field(..., min_length=2)
    root_cause: str = Field(..., min_length=20)
    proposed_intervention: Literal[
        "add_line",
        "edit_line",
        "remove_line",
        "create_skill",
        "edit_skill",
        "tighten_setting",
        "modify_hook",
    ]
    expected_lift: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("evidence_from_trace")
    @classmethod
    def at_least_two_evidence(cls, v: list[TraceEvidence]) -> list[TraceEvidence]:
        if len(v) < 2:
            raise ValueError("attribution must cite at least 2 pieces of trace evidence")
        return v
