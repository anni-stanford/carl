"""Claude Code adapter — concrete :class:`PolicyAdapter` implementation.

This first revision covers ``read_policy`` and ``write_policy`` only. The
``run_episode`` method is a typed stub raising :class:`NotImplementedError`
until the Docker sandbox + Claude Agent SDK runner lands (Day 3 of the build
sequence).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carl.adapters.base import PolicyAdapter, Task, Trajectory
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy

_CLAUDE_DIR = ".claude"
_RULES_FILE = "CLAUDE.md"


class ClaudeCodeAdapter(PolicyAdapter):
    """Round-trips between :class:`Policy` and the standard Claude Code layout."""

    def name(self) -> str:
        return "claude_code"

    def list_artifact_types(self) -> list[ArtifactType]:
        return [
            ArtifactType.RULES,
            ArtifactType.SKILL,
            ArtifactType.AGENT,
            ArtifactType.HOOK,
            ArtifactType.MCP_CONFIG,
            ArtifactType.COMMAND,
        ]

    async def read_policy(self, repo_path: Path) -> Policy:
        repo_path = Path(repo_path)
        artifacts: list[Artifact] = []

        rules_path = repo_path / _RULES_FILE
        if rules_path.is_file():
            artifacts.append(
                Artifact(
                    name=_RULES_FILE,
                    type=ArtifactType.RULES,
                    content=rules_path.read_text(encoding="utf-8"),
                    metadata={"path": str(rules_path.relative_to(repo_path))},
                )
            )

        claude_dir = repo_path / _CLAUDE_DIR
        if claude_dir.is_dir():
            for skill_md in (claude_dir / "skills").glob("*/SKILL.md"):
                artifacts.append(
                    Artifact(
                        name=skill_md.parent.name,
                        type=ArtifactType.SKILL,
                        content=skill_md.read_text(encoding="utf-8"),
                        metadata={"path": str(skill_md.relative_to(repo_path))},
                    )
                )
            for agent_md in (claude_dir / "agents").glob("*.md"):
                artifacts.append(
                    Artifact(
                        name=agent_md.stem,
                        type=ArtifactType.AGENT,
                        content=agent_md.read_text(encoding="utf-8"),
                        metadata={"path": str(agent_md.relative_to(repo_path))},
                    )
                )
            for hook in (claude_dir / "hooks").glob("*.sh"):
                artifacts.append(
                    Artifact(
                        name=hook.stem,
                        type=ArtifactType.HOOK,
                        content=hook.read_text(encoding="utf-8"),
                        metadata={"path": str(hook.relative_to(repo_path))},
                    )
                )
            for cmd in (claude_dir / "commands").glob("*.md"):
                artifacts.append(
                    Artifact(
                        name=cmd.stem,
                        type=ArtifactType.COMMAND,
                        content=cmd.read_text(encoding="utf-8"),
                        metadata={"path": str(cmd.relative_to(repo_path))},
                    )
                )
            settings = claude_dir / "settings.json"
            if settings.is_file():
                artifacts.append(
                    Artifact(
                        name="settings.json",
                        type=ArtifactType.MCP_CONFIG,
                        content=settings.read_text(encoding="utf-8"),
                        metadata={"path": str(settings.relative_to(repo_path))},
                    )
                )

        version = _stable_version(artifacts)
        return Policy(artifacts=artifacts, version=version)

    async def write_policy(self, repo_path: Path, policy: Policy) -> None:
        repo_path = Path(repo_path)
        repo_path.mkdir(parents=True, exist_ok=True)
        claude_dir = repo_path / _CLAUDE_DIR
        for sub in ("skills", "agents", "hooks", "commands"):
            (claude_dir / sub).mkdir(parents=True, exist_ok=True)

        for art in policy.artifacts:
            if art.type == ArtifactType.RULES:
                (repo_path / _RULES_FILE).write_text(art.content, encoding="utf-8")
            elif art.type == ArtifactType.SKILL:
                skill_dir = claude_dir / "skills" / art.name
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(art.content, encoding="utf-8")
            elif art.type == ArtifactType.AGENT:
                (claude_dir / "agents" / f"{art.name}.md").write_text(
                    art.content, encoding="utf-8"
                )
            elif art.type == ArtifactType.HOOK:
                hook = claude_dir / "hooks" / f"{art.name}.sh"
                hook.write_text(art.content, encoding="utf-8")
                hook.chmod(0o755)
            elif art.type == ArtifactType.COMMAND:
                (claude_dir / "commands" / f"{art.name}.md").write_text(
                    art.content, encoding="utf-8"
                )
            elif art.type == ArtifactType.MCP_CONFIG:
                # Validate JSON before writing — never commit broken settings.
                try:
                    json.loads(art.content)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"settings.json artifact has invalid JSON: {exc}"
                    ) from exc
                (claude_dir / "settings.json").write_text(art.content, encoding="utf-8")

    async def run_episode(
        self,
        repo_path: Path,
        task: Task,
        policy: Policy,
        timeout_s: int,
    ) -> Trajectory:
        # Implemented Day 3 of the build sequence: Docker sandbox + Claude Agent SDK.
        raise NotImplementedError(
            "ClaudeCodeAdapter.run_episode pending Docker sandbox integration"
        )


def _stable_version(artifacts: list[Artifact]) -> str:
    """Stable short version derived from artifact contents."""
    import hashlib

    h = hashlib.sha256()
    for a in sorted(artifacts, key=lambda x: (x.type.value, x.name)):
        h.update(a.type.value.encode("utf-8"))
        h.update(a.name.encode("utf-8"))
        h.update(a.content_hash.encode("utf-8"))
    return h.hexdigest()[:12]


def _serialize_artifact(art: Artifact) -> dict[str, Any]:
    return {
        "name": art.name,
        "type": art.type.value,
        "content_hash": art.content_hash,
        "metadata": art.metadata,
    }
