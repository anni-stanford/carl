"""``PolicyDiff.is_within_locality_budget`` enforces the ≤ N-line mutation rule."""

from __future__ import annotations

from carl.core.policy.artifacts import ArtifactType, PolicyDiff


def test_short_edit_is_within_budget() -> None:
    diff = PolicyDiff(
        artifact_name="CLAUDE.md",
        artifact_type=ArtifactType.RULES,
        operation="edit_line",
        line_range=(10, 12),
        old_content="old",
        new_content="new",
        rationale="r",
        expected_lift=0.05,
        confidence=0.7,
    )
    assert diff.is_within_locality_budget(5)


def test_long_edit_is_rejected() -> None:
    diff = PolicyDiff(
        artifact_name="CLAUDE.md",
        artifact_type=ArtifactType.RULES,
        operation="edit_line",
        line_range=(10, 30),
        old_content="x",
        new_content="y",
        rationale="r",
        expected_lift=0.05,
        confidence=0.7,
    )
    assert not diff.is_within_locality_budget(5)


def test_skill_creation_bypasses_line_budget() -> None:
    diff = PolicyDiff(
        artifact_name="new-skill",
        artifact_type=ArtifactType.SKILL,
        operation="create_skill",
        line_range=None,
        old_content=None,
        new_content="# Skill body",
        rationale="r",
        expected_lift=0.1,
        confidence=0.8,
    )
    assert diff.is_within_locality_budget(5)
