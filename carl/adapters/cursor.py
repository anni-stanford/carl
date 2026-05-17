"""Cursor adapter — Day-1 stub. Read/write of the Cursor artifact layout.

``run_episode`` will spawn a Node.js subprocess that uses ``@cursor/sdk`` over
JSON-RPC on stdin/stdout. The bridge lives in ``js/cursor-bridge/`` and is a
separate npm package: ``@carl-loop/cursor``.
"""

from __future__ import annotations

import json
from pathlib import Path

from carl.adapters.base import PolicyAdapter, Task, Trajectory
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy

_CURSOR_DIR = ".cursor"


class CursorAdapter(PolicyAdapter):
    """Read/write Cursor's ``.cursor/`` layout. Episode execution via Node bridge."""

    def name(self) -> str:
        return "cursor"

    def list_artifact_types(self) -> list[ArtifactType]:
        return [
            ArtifactType.RULES,
            ArtifactType.SKILL,
            ArtifactType.AGENT,
            ArtifactType.HOOK,
            ArtifactType.MCP_CONFIG,
        ]

    async def read_policy(self, repo_path: Path) -> Policy:
        repo_path = Path(repo_path)
        artifacts: list[Artifact] = []
        cursor_dir = repo_path / _CURSOR_DIR

        if cursor_dir.is_dir():
            rules = cursor_dir / "rules"
            if rules.is_file():
                artifacts.append(
                    Artifact(
                        name="rules",
                        type=ArtifactType.RULES,
                        content=rules.read_text(encoding="utf-8"),
                    )
                )
            for skill_md in (cursor_dir / "skills").glob("*/SKILL.md"):
                artifacts.append(
                    Artifact(
                        name=skill_md.parent.name,
                        type=ArtifactType.SKILL,
                        content=skill_md.read_text(encoding="utf-8"),
                    )
                )
            for agent_md in (cursor_dir / "agents").glob("*.md"):
                artifacts.append(
                    Artifact(
                        name=agent_md.stem,
                        type=ArtifactType.AGENT,
                        content=agent_md.read_text(encoding="utf-8"),
                    )
                )
            hooks = cursor_dir / "hooks.json"
            if hooks.is_file():
                artifacts.append(
                    Artifact(
                        name="hooks.json",
                        type=ArtifactType.HOOK,
                        content=hooks.read_text(encoding="utf-8"),
                    )
                )
            mcp = cursor_dir / "mcp.json"
            if mcp.is_file():
                artifacts.append(
                    Artifact(
                        name="mcp.json",
                        type=ArtifactType.MCP_CONFIG,
                        content=mcp.read_text(encoding="utf-8"),
                    )
                )

        from carl.adapters.claude_code import _stable_version  # local import OK

        return Policy(artifacts=artifacts, version=_stable_version(artifacts))

    async def write_policy(self, repo_path: Path, policy: Policy) -> None:
        repo_path = Path(repo_path)
        cursor_dir = repo_path / _CURSOR_DIR
        for sub in ("skills", "agents"):
            (cursor_dir / sub).mkdir(parents=True, exist_ok=True)

        for art in policy.artifacts:
            if art.type == ArtifactType.RULES:
                (cursor_dir / "rules").write_text(art.content, encoding="utf-8")
            elif art.type == ArtifactType.SKILL:
                d = cursor_dir / "skills" / art.name
                d.mkdir(parents=True, exist_ok=True)
                (d / "SKILL.md").write_text(art.content, encoding="utf-8")
            elif art.type == ArtifactType.AGENT:
                (cursor_dir / "agents" / f"{art.name}.md").write_text(
                    art.content, encoding="utf-8"
                )
            elif art.type == ArtifactType.HOOK:
                (cursor_dir / "hooks.json").write_text(art.content, encoding="utf-8")
            elif art.type == ArtifactType.MCP_CONFIG:
                try:
                    json.loads(art.content)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"mcp.json invalid JSON: {exc}") from exc
                (cursor_dir / "mcp.json").write_text(art.content, encoding="utf-8")

    async def run_episode(
        self,
        repo_path: Path,
        task: Task,
        policy: Policy,
        timeout_s: int,
    ) -> Trajectory:
        # Spawned Node.js bridge will live in js/cursor-bridge. JSON-RPC protocol
        # is documented in docs/multi_ide.md.
        raise NotImplementedError(
            "CursorAdapter.run_episode pending @cursor/sdk JSON-RPC bridge"
        )
