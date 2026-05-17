"""Task discovery and synthetic task generation for ``carl auto``.

A "task" is a unit of work for the coding agent: a prompt + the repo it
runs against. CARL ``auto`` discovers a small task list from the repo
itself (failing tests, lint warnings, untyped functions) so the user
does not have to author one by hand.
"""

from carl.tasks.discovery import default_synthetic_tasks, discover_tasks

__all__ = ["discover_tasks", "default_synthetic_tasks"]
