"""Round-trip ``read_policy`` → ``write_policy`` → ``read_policy`` parity test.

Asserts that any policy written by ``ClaudeCodeAdapter.write_policy`` and
re-read by ``ClaudeCodeAdapter.read_policy`` produces a policy with an
identical ``policy_hash``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carl.adapters.claude_code import ClaudeCodeAdapter
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_policy() -> Policy:
    artifacts = [
        Artifact(
            name="CLAUDE.md",
            type=ArtifactType.RULES,
            content="# Project rules\nAlways write tests before fixes.\n",
        ),
        Artifact(
            name="testing-guide",
            type=ArtifactType.SKILL,
            content="# Skill: Testing Guide\nPrefer pytest; mark async tests.\n",
        ),
        Artifact(
            name="reviewer",
            type=ArtifactType.AGENT,
            content="# Sub-agent: reviewer\nReturn structured review JSON.\n",
        ),
        Artifact(
            name="pre-commit",
            type=ArtifactType.HOOK,
            content="#!/usr/bin/env bash\nset -euo pipefail\nruff check .\n",
        ),
        Artifact(
            name="settings.json",
            type=ArtifactType.MCP_CONFIG,
            content=json.dumps({"tools": {"allowed": ["bash", "edit"]}}, indent=2),
        ),
        Artifact(
            name="explain",
            type=ArtifactType.COMMAND,
            content="# /explain\nGive a concise explanation of the selected file.\n",
        ),
    ]
    return Policy(artifacts=artifacts, version="initial-test")


async def test_round_trip_preserves_policy_hash(tmp_path: Path, sample_policy: Policy) -> None:
    adapter = ClaudeCodeAdapter()

    await adapter.write_policy(tmp_path, sample_policy)

    # Verify expected files exist
    assert (tmp_path / "CLAUDE.md").is_file()
    assert (tmp_path / ".claude" / "skills" / "testing-guide" / "SKILL.md").is_file()
    assert (tmp_path / ".claude" / "agents" / "reviewer.md").is_file()
    assert (tmp_path / ".claude" / "hooks" / "pre-commit.sh").is_file()
    assert (tmp_path / ".claude" / "settings.json").is_file()
    assert (tmp_path / ".claude" / "commands" / "explain.md").is_file()

    # Hook must be executable
    hook = tmp_path / ".claude" / "hooks" / "pre-commit.sh"
    assert hook.stat().st_mode & 0o111, "hook should be executable"

    round_tripped = await adapter.read_policy(tmp_path)
    assert round_tripped.policy_hash == sample_policy.policy_hash


async def test_invalid_settings_json_is_rejected(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter()
    bad_policy = Policy(
        artifacts=[
            Artifact(
                name="settings.json",
                type=ArtifactType.MCP_CONFIG,
                content="{not: valid json",
            )
        ],
        version="bad",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        await adapter.write_policy(tmp_path, bad_policy)
