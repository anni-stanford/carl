"""SQLite replay buffer for trajectories, preference pairs, gate decisions.

v1 uses stdlib :mod:`sqlite3` so there is no external dependency. The schema
is forward-compatible with the v2 Postgres + pgvector backend planned in
``docs/architecture.md``.
"""

from carl.core.buffer.storage import ReplayBuffer, TrajectoryRow

__all__ = ["ReplayBuffer", "TrajectoryRow"]
