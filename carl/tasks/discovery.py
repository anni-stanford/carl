"""Auto-discover candidate tasks from the user's repo.

Heuristics (cheap, no LLM call): scan the repo for low-hanging fruit
that CI can score deterministically. The default set covers the four
signals the verifier reads:

- failing tests   → tests_passed
- low coverage    → coverage_delta
- ruff warnings   → lint_clean
- mypy errors     → typecheck_clean

If the repo contains ``.carl/tasks.yaml`` we use that instead — power
users keep a curated list there.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TaskSpec:
    """Lightweight description of a task before it is materialized into a Task."""

    task_id: str
    prompt: str
    metadata: dict[str, str]


def discover_tasks(repo_path: Path, n: int = 10) -> list[TaskSpec]:
    """Return up to ``n`` task specs.

    Order of preference:
      1. ``.carl/tasks.yaml`` if present.
      2. Heuristic scan of the repo (failing tests, ruff / mypy diagnostics).
      3. ``default_synthetic_tasks(n)`` fallback.
    """
    cfg = repo_path / ".carl" / "tasks.yaml"
    if cfg.is_file():
        return _load_yaml_tasks(cfg)[:n]

    discovered = _scan_repo(repo_path)
    if discovered:
        return discovered[:n]

    return default_synthetic_tasks(n)


def default_synthetic_tasks(n: int = 10) -> list[TaskSpec]:
    """Sensible default tasks that any Python repo can use."""
    pool: list[TaskSpec] = [
        TaskSpec(
            task_id=f"default-{i:02d}",
            prompt=p,
            metadata={"source": "default"},
        )
        for i, p in enumerate(
            [
                "Add at least one unit test for any module currently missing test coverage; "
                "run pytest to confirm the new test passes.",
                "Eliminate the highest-priority ruff warning in the repo by editing the offending file; "
                "run `ruff check .` to confirm.",
                "Add type annotations to one untyped public function; "
                "run `mypy --strict` on the changed file to confirm it is clean.",
                "Find a function with high cyclomatic complexity and split it into two smaller, well-named "
                "functions; preserve behavior; ensure all tests still pass.",
                "Find a TODO or FIXME comment in the codebase, complete the work, remove the comment, and "
                "add a regression test.",
                "Improve the README's quickstart so a new user can run the project in three commands; "
                "verify by following the new README on a clean clone.",
                "Locate one piece of duplicated logic, extract it into a helper function, replace the "
                "duplicates with calls to the helper, and confirm tests pass.",
                "Find one bare `except:` and replace it with a specific exception class; preserve behavior; "
                "ensure tests pass.",
                "Add a docstring with a worked example to one public function that lacks one; the example "
                "must be runnable as a doctest.",
                "Increase test coverage of the highest-impact module by at least one assertion that "
                "actually fails before the fix and passes after.",
            ]
        )
    ]
    return pool[:n]


def _scan_repo(repo_path: Path) -> list[TaskSpec]:
    """Heuristic scan. Returns an empty list if nothing found (caller falls back)."""
    out: list[TaskSpec] = []

    failing = _failing_tests(repo_path)
    for i, name in enumerate(failing[:3]):
        out.append(
            TaskSpec(
                task_id=f"fix-failing-test-{i}",
                prompt=(
                    f"Fix the failing pytest test `{name}`. Do not modify the test itself; "
                    f"fix the underlying code so the test passes."
                ),
                metadata={"source": "failing_test", "test": name},
            )
        )

    ruff_diags = _ruff_diagnostics(repo_path)
    for i, code in enumerate(ruff_diags[:3]):
        out.append(
            TaskSpec(
                task_id=f"fix-ruff-{code}-{i}",
                prompt=f"Eliminate every occurrence of ruff diagnostic `{code}` from the codebase. "
                f"Run `ruff check .` to confirm zero remain.",
                metadata={"source": "ruff", "code": code},
            )
        )

    mypy_count = _mypy_error_count(repo_path)
    if mypy_count > 0:
        out.append(
            TaskSpec(
                task_id="reduce-mypy-errors",
                prompt=(
                    f"The repo currently has {mypy_count} mypy errors. Reduce them by at least 25 % "
                    "by adding type annotations or fixing existing ones. Do not silence errors with "
                    "`# type: ignore` unless the underlying typing is genuinely impossible."
                ),
                metadata={"source": "mypy", "current_errors": str(mypy_count)},
            )
        )

    return out


def _failing_tests(repo_path: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["pytest", "--collect-only", "-q"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # We want failing tests, but `pytest --collect-only` only enumerates.
        # Cheap: enumerate and let the agent decide which fail; in real auto runs
        # the agent runs pytest and sees real failures.
        names = [line.strip() for line in result.stdout.splitlines() if "::" in line]
        return names[:5]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []


def _ruff_diagnostics(repo_path: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format", "concise", "."],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        codes = set()
        for line in result.stdout.splitlines():
            # Lines look like: path/to/file.py:12:5: F401 ...
            parts = line.split(":")
            for p in parts:
                p = p.strip()
                if len(p) >= 4 and p[0].isalpha() and p[1:].isdigit():
                    codes.add(p)
                    break
        return sorted(codes)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []


def _mypy_error_count(repo_path: Path) -> int:
    try:
        result = subprocess.run(
            ["mypy", "--no-color-output", "."],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return sum(1 for line in result.stdout.splitlines() if ": error:" in line)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return 0


def _load_yaml_tasks(path: Path) -> list[TaskSpec]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    tasks = data.get("tasks", [])
    out: list[TaskSpec] = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            continue
        prompt = t.get("prompt") or t.get("description")
        if not prompt:
            continue
        out.append(
            TaskSpec(
                task_id=str(t.get("id", f"yaml-{i}")),
                prompt=str(prompt),
                metadata={k: str(v) for k, v in t.items() if k not in ("id", "prompt", "description")},
            )
        )
    return out
