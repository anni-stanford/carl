"""Concrete :class:`PolicyAdapter` implementations.

Adapters declared in ``pyproject.toml`` under ``[project.entry-points."carl.adapters"]``
are auto-discovered by :func:`carl.adapters._registry.load_adapters`.
The Claude Code adapter is the only adapter shipped today; the
:class:`PolicyAdapter` ABC is preserved so future agents (Codex, Aider, …)
can plug into the same RL machinery.
"""

from carl.adapters.base import PolicyAdapter, Task, TraceEvent, Trajectory
from carl.adapters.claude_code import ClaudeCodeAdapter

__all__ = [
    "ClaudeCodeAdapter",
    "PolicyAdapter",
    "Task",
    "TraceEvent",
    "Trajectory",
]
