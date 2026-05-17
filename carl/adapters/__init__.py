"""IDE-specific :class:`PolicyAdapter` implementations.

Adapters declared in ``pyproject.toml`` under ``[project.entry-points."carl.adapters"]``
are auto-discovered by :func:`carl.adapters._registry.load_adapters`.
"""

from carl.adapters.base import PolicyAdapter, Task, TraceEvent, Trajectory
from carl.adapters.claude_code import ClaudeCodeAdapter
from carl.adapters.cursor import CursorAdapter

__all__ = [
    "ClaudeCodeAdapter",
    "CursorAdapter",
    "PolicyAdapter",
    "Task",
    "TraceEvent",
    "Trajectory",
]
