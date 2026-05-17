"""Sandboxed execution environments for episode runs.

- :mod:`carl.env.docker_sandbox` — local Docker isolation (Day 3).
- :mod:`carl.env.cursor_cloud` — Cursor cloud VM (Day 4; planned).
- :mod:`carl.env.repo_loader` — clone, snapshot, restore (Day 3; planned).
- :mod:`carl.env.task_queue` — task scheduling (Day 9; planned).
"""

from carl.env.docker_sandbox import SandboxResult, docker_sandbox, run_in_sandbox

__all__ = ["SandboxResult", "docker_sandbox", "run_in_sandbox"]
