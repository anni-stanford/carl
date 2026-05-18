"""Agent-agnostic representation of the coding agent's text-space policy.

The Claude Code adapter writes a ``Policy`` to ``CLAUDE.md`` plus
``.claude/`` subdirectories. The data model is deliberately kept
agent-agnostic so future adapters (Codex, Aider, …) can reuse it.
Adapters own the disk serialization; the core RL loop never touches
``open(path, ...)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ArtifactType(StrEnum):
    """Semantic role of a policy artifact, independent of disk layout."""

    RULES = "rules"
    SKILL = "skill"
    AGENT = "agent"
    HOOK = "hook"
    MCP_CONFIG = "mcp_config"
    COMMAND = "command"


@dataclass
class Artifact:
    """A single editable text artifact in the agent's policy."""

    name: str
    type: ArtifactType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Stable SHA-256 of ``content`` — used for diff identity."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Artifact.name must be non-empty")
        if not isinstance(self.type, ArtifactType):
            raise TypeError("Artifact.type must be ArtifactType")


@dataclass
class Policy:
    """Versioned snapshot of a full agent policy across all artifact types."""

    artifacts: list[Artifact]
    version: str
    parent_version: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_type(self, artifact_type: ArtifactType) -> list[Artifact]:
        return [a for a in self.artifacts if a.type == artifact_type]

    def find(self, name: str) -> Artifact | None:
        for a in self.artifacts:
            if a.name == name:
                return a
        return None

    def upsert(self, artifact: Artifact) -> None:
        for i, existing in enumerate(self.artifacts):
            if existing.name == artifact.name and existing.type == artifact.type:
                self.artifacts[i] = artifact
                return
        self.artifacts.append(artifact)

    @property
    def policy_hash(self) -> str:
        """Deterministic identity of the full policy snapshot."""
        h = hashlib.sha256()
        for a in sorted(self.artifacts, key=lambda x: (x.type.value, x.name)):
            h.update(a.type.value.encode("utf-8"))
            h.update(a.name.encode("utf-8"))
            h.update(a.content_hash.encode("utf-8"))
        return h.hexdigest()


@dataclass
class PolicyDiff:
    """Structured representation of a proposed mutation, ≤ ``max_diff_lines``."""

    artifact_name: str
    artifact_type: ArtifactType
    operation: str  # "add_line" | "edit_line" | "remove_line" | "create_skill" | ...
    line_range: tuple[int, int] | None
    old_content: str | None
    new_content: str | None
    rationale: str
    expected_lift: float
    confidence: float

    def is_within_locality_budget(self, max_lines: int) -> bool:
        if self.line_range is None:
            return self.operation in {"create_skill", "edit_skill"}
        start, end = self.line_range
        return (end - start + 1) <= max_lines
