"""``PolicyAdapter`` — the boundary between agent-agnostic core and agent-specific I/O.

Today the only shipped adapter is
:class:`carl.adapters.claude_code.ClaudeCodeAdapter`. The ABC is preserved
so future agents (Codex, Aider, …) implement the same five methods and
plug into the existing RL machinery without changing the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carl.core.policy.artifacts import ArtifactType, Policy


@dataclass
class Task:
    """A unit of work assigned to the coding agent."""

    task_id: str
    repo_path: Path
    prompt: str
    adapter_name: str  # "claude_code" today; "codex"/"aider"/… in the future
    expected_diff: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    """One observable event during episode execution."""

    timestamp: float
    kind: str  # "tool_call" | "file_edit" | "subagent_spawn" | "retry" | "stdout" | "exit"
    payload: dict[str, Any]


@dataclass
class Trajectory:
    """Full record of an episode: events, files touched, exit info, raw CI output."""

    task: Task
    policy: Policy
    events: list[TraceEvent]
    files_changed: list[str]
    exit_code: int
    duration_s: float
    raw_ci_output: str | None = None
    raw_test_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyAdapter(ABC):
    """Translate between :class:`Policy` and an IDE's on-disk artifact layout."""

    @abstractmethod
    async def read_policy(self, repo_path: Path) -> Policy:
        """Scan ``repo_path`` and build the IDE-agnostic policy snapshot."""

    @abstractmethod
    async def write_policy(self, repo_path: Path, policy: Policy) -> None:
        """Materialize ``policy`` to disk in this IDE's expected layout."""

    @abstractmethod
    async def run_episode(
        self,
        repo_path: Path,
        task: Task,
        policy: Policy,
        timeout_s: int,
    ) -> Trajectory:
        """Execute ``task`` against the IDE under ``policy``; return a :class:`Trajectory`."""

    @abstractmethod
    def list_artifact_types(self) -> list[ArtifactType]:
        """Artifact types supported by this IDE."""

    @abstractmethod
    def name(self) -> str:
        """Stable adapter name (e.g. ``"claude_code"``)."""
