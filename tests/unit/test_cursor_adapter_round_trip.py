"""Round-trip parity for the Cursor adapter (read_policy / write_policy only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carl.adapters.cursor import CursorAdapter
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy

pytestmark = pytest.mark.asyncio


async def test_cursor_round_trip(tmp_path: Path) -> None:
    adapter = CursorAdapter()
    policy = Policy(
        artifacts=[
            Artifact(name="rules", type=ArtifactType.RULES, content="# Rules\nAlways prefer typed code.\n"),
            Artifact(name="testing", type=ArtifactType.SKILL, content="# Skill\nUse pytest.\n"),
            Artifact(name="reviewer", type=ArtifactType.AGENT, content="# Agent\nReview diffs.\n"),
            Artifact(name="hooks.json", type=ArtifactType.HOOK, content=json.dumps({"pre_commit": []})),
            Artifact(name="mcp.json", type=ArtifactType.MCP_CONFIG, content=json.dumps({"servers": {}})),
        ],
        version="cursor-test",
    )

    await adapter.write_policy(tmp_path, policy)

    assert (tmp_path / ".cursor" / "rules").is_file()
    assert (tmp_path / ".cursor" / "skills" / "testing" / "SKILL.md").is_file()
    assert (tmp_path / ".cursor" / "agents" / "reviewer.md").is_file()
    assert (tmp_path / ".cursor" / "hooks.json").is_file()
    assert (tmp_path / ".cursor" / "mcp.json").is_file()

    round_tripped = await adapter.read_policy(tmp_path)
    assert round_tripped.policy_hash == policy.policy_hash
