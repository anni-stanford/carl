"""Sandboxed execution environments for episode runs.

- :mod:`carl.env.docker_sandbox` — local Docker isolation.
- :mod:`carl.env.repo_loader` — clone, snapshot, restore (planned).
- :mod:`carl.env.task_queue` — task scheduling (planned).
"""

from carl.env.docker_sandbox import SandboxResult, docker_sandbox, run_in_sandbox

__all__ = ["SandboxResult", "docker_sandbox", "run_in_sandbox"]
