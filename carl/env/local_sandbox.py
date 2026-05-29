"""No-Docker local episode runner.

This is the path that lets a Claude Code user run CARL with a single command
and no Docker. Instead of spinning up a container, it:

  1. Copies the target repo into a throwaway temp directory (so the user's
     working tree is never mutated).
  2. Writes the policy artifacts into that copy.
  3. Invokes the Claude Code CLI in headless ``-p`` (print) mode against the
     copy, with the task prompt.
  4. Runs the project's CI locally (pytest / coverage / ruff / mypy) inside
     the copy and writes the artifact files the verifier reads.

Isolation is weaker than Docker — the agent and the tests run on the host —
so this path is best used on a throwaway or version-controlled repo. The
trade-off buys a zero-dependency, single-command experience.

The Claude Code binary is auto-detected (``claude`` then ``claude-code``).
``CARL_CLAUDE_BIN`` overrides the binary path, and ``CARL_CLAUDE_ARGS``
overrides the argv template; both exist so the runner can be unit-tested
with a fake agent and so a user whose CLI differs can adapt without code
changes.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

#: An async callable that writes a policy into a working directory.
PolicyWriter = Callable[[Path], Awaitable[None]]


@dataclass(frozen=True)
class LocalResult:
    """Outcome of a local (no-Docker) episode."""

    exit_code: int
    stdout: str
    stderr: str
    artifact_dir: Path
    duration_s: float
    agent_ran: bool


def claude_cli_available() -> str | None:
    """Return the path to the Claude Code CLI binary, or ``None`` if absent.

    Honors ``CARL_CLAUDE_BIN`` first (used by tests and by users whose binary
    is not on ``PATH`` under a standard name).
    """
    override = os.environ.get("CARL_CLAUDE_BIN")
    if override:
        return override
    for name in ("claude", "claude-code"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _agent_argv(claude_bin: str, prompt: str) -> list[str]:
    """Build the argv for one headless Claude Code run.

    Default is ``<claude_bin> -p <prompt>`` (Claude Code's print / headless
    mode). Override the whole template with ``CARL_CLAUDE_ARGS`` using
    ``{prompt}`` as a placeholder, e.g.::

        CARL_CLAUDE_ARGS="-p {prompt} --permission-mode acceptEdits"
    """
    template = os.environ.get("CARL_CLAUDE_ARGS")
    if template:
        parts = shlex.split(template)
        return [claude_bin, *[p.replace("{prompt}", prompt) for p in parts]]
    return [claude_bin, "-p", prompt]


async def run_local_episode(
    repo_path: Path,
    write_policy_into: PolicyWriter,
    task_prompt: str,
    *,
    claude_bin: str,
    timeout_s: int,
) -> LocalResult:
    """Run one episode locally and return CI artifacts.

    ``write_policy_into`` is an async callable ``(work_dir: Path) -> None``
    that materializes the policy into the temp working copy; the adapter
    supplies it so this module stays agent-agnostic.
    """
    work = Path(tempfile.mkdtemp(prefix="carl_local_work_"))
    artifact_dir = Path(tempfile.mkdtemp(prefix="carl_local_artifacts_"))
    start = time.monotonic()
    agent_ran = False
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    exit_code = 0

    try:
        _copy_repo(repo_path, work)
        await write_policy_into(work)

        # 1) Agent pass (headless Claude Code). Tolerate failure; CI still runs.
        agent_argv = _agent_argv(claude_bin, task_prompt)
        try:
            a_out, a_err, a_code = await _run(agent_argv, cwd=work, timeout_s=timeout_s)
            stdout_parts.append(f"$ {' '.join(agent_argv[:2])} ...\n{a_out}")
            stderr_parts.append(a_err)
            agent_ran = a_code == 0
            exit_code = a_code
        except FileNotFoundError:
            stderr_parts.append(f"agent binary not found: {claude_bin}")
            exit_code = 127
        except TimeoutError:
            stderr_parts.append(f"agent timed out after {timeout_s}s")
            exit_code = 124

        # 2) CI pass — always run so we get a reward signal even if the agent
        #    did nothing. Each command is best-effort; missing tools are skipped.
        await _run_ci(work, artifact_dir, stdout_parts, stderr_parts)

    finally:
        shutil.rmtree(work, ignore_errors=True)

    return LocalResult(
        exit_code=exit_code,
        stdout="\n".join(stdout_parts),
        stderr="\n".join(stderr_parts),
        artifact_dir=artifact_dir,
        duration_s=time.monotonic() - start,
        agent_ran=agent_ran,
    )


# ---- internals --------------------------------------------------------------


def _copy_repo(src: Path, dst: Path) -> None:
    """Copy ``src`` into ``dst`` (which already exists), excluding heavy dirs."""
    ignore = shutil.ignore_patterns(
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        ".mypy_cache", ".ruff_cache", ".pytest_cache", "carl_run",
        "*.egg-info", ".tox",
    )
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


async def _run(
    argv: list[str], *, cwd: Path, timeout_s: int
) -> tuple[str, str, int]:
    """Run ``argv`` in ``cwd`` with a timeout; return (stdout, stderr, exit_code)."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return (
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
        int(proc.returncode or 0),
    )


async def _run_ci(
    work: Path,
    artifact_dir: Path,
    stdout_parts: list[str],
    stderr_parts: list[str],
) -> None:
    """Run pytest / coverage / ruff / mypy locally; write artifact files.

    Every command is best-effort. A missing tool or a non-zero exit never
    raises — the verifier interprets whatever artifacts exist (and falls back
    to the pytest exit code when no JSON report was produced).
    """
    import sys

    py = sys.executable

    # pytest with optional json + coverage. If pytest-json-report / coverage
    # are not installed in the user's env, these flags fail and we retry plain.
    pytest_json = artifact_dir / "pytest.json"
    rich = await _try(
        [py, "-m", "pytest", "-q",
         "--json-report", f"--json-report-file={pytest_json}",
         "--cov", "--cov-report", f"xml:{artifact_dir / 'coverage.xml'}"],
        cwd=work, stdout_parts=stdout_parts, stderr_parts=stderr_parts,
    )
    if rich is None or not pytest_json.exists():
        # Plain pytest fallback — verifier will use the exit code.
        await _try(
            [py, "-m", "pytest", "-q"],
            cwd=work, stdout_parts=stdout_parts, stderr_parts=stderr_parts,
        )

    # ruff JSON
    ruff_json = artifact_dir / "ruff.json"
    code = await _try(
        ["ruff", "check", "--output-format", "json", "."],
        cwd=work, stdout_parts=stdout_parts, stderr_parts=stderr_parts,
        capture_to=ruff_json,
    )
    if code is None:
        await _try(
            [py, "-m", "ruff", "check", "--output-format", "json", "."],
            cwd=work, stdout_parts=stdout_parts, stderr_parts=stderr_parts,
            capture_to=ruff_json,
        )

    # mypy text
    mypy_txt = artifact_dir / "mypy.txt"
    code = await _try(
        ["mypy", "."], cwd=work, stdout_parts=stdout_parts,
        stderr_parts=stderr_parts, capture_to=mypy_txt,
    )
    if code is None:
        await _try(
            [py, "-m", "mypy", "."], cwd=work, stdout_parts=stdout_parts,
            stderr_parts=stderr_parts, capture_to=mypy_txt,
        )


async def _try(  # noqa: PLR0913
    argv: list[str],
    *,
    cwd: Path,
    stdout_parts: list[str],
    stderr_parts: list[str],
    capture_to: Path | None = None,
) -> int | None:
    """Run a CI command; return exit code, or ``None`` if the tool is missing.

    When ``capture_to`` is set, stdout is written there (for ruff JSON / mypy
    text artifacts that the verifier reads from a file).
    """
    try:
        out, err, code = await _run(argv, cwd=cwd, timeout_s=600)
    except FileNotFoundError:
        return None
    except TimeoutError:
        stderr_parts.append(f"CI command timed out: {' '.join(argv[:3])}")
        return 124
    if capture_to is not None:
        capture_to.write_text(out, encoding="utf-8")  # noqa: ASYNC240 (tiny artifact write)
    stdout_parts.append(f"$ {' '.join(argv[:3])} ... (exit {code})")
    if err.strip():
        stderr_parts.append(err[-1000:])
    return code
