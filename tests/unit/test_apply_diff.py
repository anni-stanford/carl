"""``apply_diff`` correctly mutates the policy for every operation type."""

from __future__ import annotations

import pytest

from carl.core.policy.apply_diff import apply_diff
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy, PolicyDiff


def _seed_policy() -> Policy:
    return Policy(
        artifacts=[
            Artifact(
                name="CLAUDE.md",
                type=ArtifactType.RULES,
                content="# Project rules\nLine 1\nLine 2\nLine 3\n",
            ),
            Artifact(name="testing-policy", type=ArtifactType.SKILL, content="# old skill\n"),
        ],
        version="v0",
    )


def test_add_line_inserts_at_position() -> None:
    diff = PolicyDiff(
        artifact_name="CLAUDE.md",
        artifact_type=ArtifactType.RULES,
        operation="add_line",
        line_range=(2, 2),
        old_content=None,
        new_content="INSERTED",
        rationale="r",
        expected_lift=0.05,
        confidence=0.7,
    )
    new_policy = apply_diff(_seed_policy(), diff)
    artifact = new_policy.find("CLAUDE.md")
    assert artifact is not None
    assert artifact.content.split("\n") == [
        "# Project rules",
        "INSERTED",
        "Line 1",
        "Line 2",
        "Line 3",
        "",
    ]


def test_edit_line_replaces_range() -> None:
    diff = PolicyDiff(
        artifact_name="CLAUDE.md",
        artifact_type=ArtifactType.RULES,
        operation="edit_line",
        line_range=(2, 3),
        old_content="Line 1\nLine 2",
        new_content="REPLACED",
        rationale="r",
        expected_lift=0.05,
        confidence=0.7,
    )
    new_policy = apply_diff(_seed_policy(), diff)
    artifact = new_policy.find("CLAUDE.md")
    assert artifact is not None
    assert "REPLACED" in artifact.content
    assert "Line 1" not in artifact.content
    assert "Line 2" not in artifact.content


def test_remove_line_drops_range() -> None:
    diff = PolicyDiff(
        artifact_name="CLAUDE.md",
        artifact_type=ArtifactType.RULES,
        operation="remove_line",
        line_range=(2, 2),
        old_content="Line 1",
        new_content=None,
        rationale="r",
        expected_lift=0.0,
        confidence=0.5,
    )
    new_policy = apply_diff(_seed_policy(), diff)
    artifact = new_policy.find("CLAUDE.md")
    assert artifact is not None
    assert "Line 1" not in artifact.content


def test_create_skill_appends_new_artifact() -> None:
    diff = PolicyDiff(
        artifact_name="brand-new-skill",
        artifact_type=ArtifactType.SKILL,
        operation="create_skill",
        line_range=None,
        old_content=None,
        new_content="# Brand new skill body",
        rationale="r",
        expected_lift=0.1,
        confidence=0.8,
    )
    new_policy = apply_diff(_seed_policy(), diff)
    skill = new_policy.find("brand-new-skill")
    assert skill is not None
    assert skill.type == ArtifactType.SKILL
    assert skill.content == "# Brand new skill body"


def test_edit_skill_overwrites_existing_skill() -> None:
    diff = PolicyDiff(
        artifact_name="testing-policy",
        artifact_type=ArtifactType.SKILL,
        operation="edit_skill",
        line_range=None,
        old_content="# old skill\n",
        new_content="# brand-new content\n",
        rationale="r",
        expected_lift=0.05,
        confidence=0.7,
    )
    new_policy = apply_diff(_seed_policy(), diff)
    skill = new_policy.find("testing-policy")
    assert skill is not None
    assert skill.content == "# brand-new content\n"


def test_modify_hook_creates_or_replaces() -> None:
    seed = _seed_policy()
    diff = PolicyDiff(
        artifact_name="pre-commit",
        artifact_type=ArtifactType.HOOK,
        operation="modify_hook",
        line_range=None,
        old_content=None,
        new_content="#!/usr/bin/env bash\nruff check .\n",
        rationale="r",
        expected_lift=0.04,
        confidence=0.6,
    )
    new_policy = apply_diff(seed, diff)
    hook = new_policy.find("pre-commit")
    assert hook is not None
    assert hook.type == ArtifactType.HOOK


def test_invalid_op_raises() -> None:
    diff = PolicyDiff(
        artifact_name="CLAUDE.md",
        artifact_type=ArtifactType.RULES,
        operation="banana",
        line_range=None,
        old_content=None,
        new_content="x",
        rationale="r",
        expected_lift=0.0,
        confidence=0.0,
    )
    with pytest.raises(ValueError, match="unknown PolicyDiff.operation"):
        apply_diff(_seed_policy(), diff)


def test_version_derived_deterministically() -> None:
    seed = _seed_policy()
    diff = PolicyDiff(
        artifact_name="CLAUDE.md",
        artifact_type=ArtifactType.RULES,
        operation="add_line",
        line_range=(1, 1),
        old_content=None,
        new_content="ABC",
        rationale="r",
        expected_lift=0.05,
        confidence=0.7,
    )
    p1 = apply_diff(seed, diff)
    p2 = apply_diff(seed, diff)
    assert p1.version == p2.version
    assert p1.parent_version == seed.version
