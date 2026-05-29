"""Claude Code adapter — concrete :class:`PolicyAdapter` implementation.

Implements ``read_policy``, ``write_policy``, and ``run_episode`` end-to-end.
Episode execution uses :mod:`carl.env.docker_sandbox` to run the Claude Agent
SDK against a clone of the target repo, then runs the repo's CI inside the
container so :class:`carl.core.reward.verifier.compute_verifier` can score
the resulting trajectory.

Real execution requires a running Docker daemon plus an ``ANTHROPIC_API_KEY``
in the environment. Unit tests cover the read/write round-trip and the
trajectory-construction path; they do not require Docker.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from carl.adapters.base import PolicyAdapter, Task, TraceEvent, Trajectory
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy
from carl.env.docker_sandbox import docker_sandbox, run_in_sandbox

_CLAUDE_DIR = ".claude"
_RULES_FILE = "CLAUDE.md"


class ClaudeCodeAdapter(PolicyAdapter):
    """Round-trips between :class:`Policy` and the standard Claude Code layout."""

    def name(self) -> str:
        return "claude_code"

    def list_artifact_types(self) -> list[ArtifactType]:
        return [
            ArtifactType.RULES,
            ArtifactType.SKILL,
            ArtifactType.AGENT,
            ArtifactType.HOOK,
            ArtifactType.MCP_CONFIG,
            ArtifactType.COMMAND,
        ]

    async def read_policy(self, repo_path: Path) -> Policy:
        repo_path = Path(repo_path)
        artifacts: list[Artifact] = []

        rules_path = repo_path / _RULES_FILE
        if rules_path.is_file():
            artifacts.append(
                Artifact(
                    name=_RULES_FILE,
                    type=ArtifactType.RULES,
                    content=rules_path.read_text(encoding="utf-8"),
                    metadata={"path": str(rules_path.relative_to(repo_path))},
                )
            )

        claude_dir = repo_path / _CLAUDE_DIR
        if claude_dir.is_dir():
            for skill_md in (claude_dir / "skills").glob("*/SKILL.md"):
                artifacts.append(
                    Artifact(
                        name=skill_md.parent.name,
                        type=ArtifactType.SKILL,
                        content=skill_md.read_text(encoding="utf-8"),
                        metadata={"path": str(skill_md.relative_to(repo_path))},
                    )
                )
            for agent_md in (claude_dir / "agents").glob("*.md"):
                artifacts.append(
                    Artifact(
                        name=agent_md.stem,
                        type=ArtifactType.AGENT,
                        content=agent_md.read_text(encoding="utf-8"),
                        metadata={"path": str(agent_md.relative_to(repo_path))},
                    )
                )
            for hook in (claude_dir / "hooks").glob("*.sh"):
                artifacts.append(
                    Artifact(
                        name=hook.stem,
                        type=ArtifactType.HOOK,
                        content=hook.read_text(encoding="utf-8"),
                        metadata={"path": str(hook.relative_to(repo_path))},
                    )
                )
            for cmd in (claude_dir / "commands").glob("*.md"):
                artifacts.append(
                    Artifact(
                        name=cmd.stem,
                        type=ArtifactType.COMMAND,
                        content=cmd.read_text(encoding="utf-8"),
                        metadata={"path": str(cmd.relative_to(repo_path))},
                    )
                )
            settings = claude_dir / "settings.json"
            if settings.is_file():
                artifacts.append(
                    Artifact(
                        name="settings.json",
                        type=ArtifactType.MCP_CONFIG,
                        content=settings.read_text(encoding="utf-8"),
                        metadata={"path": str(settings.relative_to(repo_path))},
                    )
                )

        version = _stable_version(artifacts)
        return Policy(artifacts=artifacts, version=version)

    async def write_policy(self, repo_path: Path, policy: Policy) -> None:
        repo_path = Path(repo_path)
        repo_path.mkdir(parents=True, exist_ok=True)
        claude_dir = repo_path / _CLAUDE_DIR
        for sub in ("skills", "agents", "hooks", "commands"):
            (claude_dir / sub).mkdir(parents=True, exist_ok=True)

        for art in policy.artifacts:
            if art.type == ArtifactType.RULES:
                (repo_path / _RULES_FILE).write_text(art.content, encoding="utf-8")
            elif art.type == ArtifactType.SKILL:
                skill_dir = claude_dir / "skills" / art.name
                skill_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(art.content, encoding="utf-8")
            elif art.type == ArtifactType.AGENT:
                (claude_dir / "agents" / f"{art.name}.md").write_text(
                    art.content, encoding="utf-8"
                )
            elif art.type == ArtifactType.HOOK:
                hook = claude_dir / "hooks" / f"{art.name}.sh"
                hook.write_text(art.content, encoding="utf-8")
                hook.chmod(0o755)
            elif art.type == ArtifactType.COMMAND:
                (claude_dir / "commands" / f"{art.name}.md").write_text(
                    art.content, encoding="utf-8"
                )
            elif art.type == ArtifactType.MCP_CONFIG:
                # Validate JSON before writing — never commit broken settings.
                try:
                    json.loads(art.content)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"settings.json artifact has invalid JSON: {exc}"
                    ) from exc
                (claude_dir / "settings.json").write_text(art.content, encoding="utf-8")

    async def run_episode_local(
        self,
        repo_path: Path,
        task: Task,
        policy: Policy,
        timeout_s: int,
        *,
        claude_bin: str,
    ) -> Trajectory:
        """Run a single Claude-Code episode locally (no Docker).

        Copies the repo to a temp dir, writes ``policy`` into it, runs the
        Claude Code CLI in headless ``-p`` mode, then runs CI locally and
        returns a :class:`Trajectory` whose metadata points at the artifact
        files the verifier reads. The user's working tree is never mutated.
        """
        from carl.env.local_sandbox import run_local_episode

        start = time.monotonic()

        async def _writer(work_dir: Path) -> None:
            await self.write_policy(work_dir, policy)

        result = await run_local_episode(
            Path(repo_path),
            _writer,
            task.prompt,
            claude_bin=claude_bin,
            timeout_s=timeout_s,
        )
        events = [
            TraceEvent(
                timestamp=time.monotonic() - start,
                kind="exit",
                payload={
                    "exit_code": result.exit_code,
                    "agent_ran": result.agent_ran,
                    "duration_s": result.duration_s,
                },
            )
        ]
        return Trajectory(
            task=task,
            policy=policy,
            events=events,
            files_changed=[],
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            raw_ci_output=result.stdout + "\n" + result.stderr,
            raw_test_output=result.stdout,
            metadata={
                "artifact_dir": str(result.artifact_dir),
                "pytest_report_json": str(result.artifact_dir / "pytest.json"),
                "coverage_xml": str(result.artifact_dir / "coverage.xml"),
                "ruff_json": str(result.artifact_dir / "ruff.json"),
                "mypy_output": str(result.artifact_dir / "mypy.txt"),
                "mode": "local",
            },
        )

    async def run_episode(
        self,
        repo_path: Path,
        task: Task,
        policy: Policy,
        timeout_s: int,
        *,
        image: str = "carl/episode-claude:latest",
        anthropic_api_key: str | None = None,
    ) -> Trajectory:
        """Run a single Claude-Code episode in a Docker sandbox.

        Steps:

        1. Materialize ``policy`` to disk inside ``repo_path``.
        2. Create a fresh sandbox tmp dir for CI artifacts.
        3. Run a wrapper script inside the container that:
           a. Calls Claude Code with ``task.prompt``.
           b. Runs the repo's CI (``pytest``, ``ruff``, ``mypy``, ``coverage``).
           c. Writes artifact files (pytest.json, coverage.xml, ruff.json,
              mypy.txt) under ``/artifacts``.
        4. Build a :class:`Trajectory` from the container's stdout/stderr and
           the path-stamped artifact files.

        The trajectory's ``raw_ci_output`` is the concatenation of the
        wrapper's stdout + stderr; the verifier reads its inputs from the
        artifact files (paths attached as :class:`Trajectory` metadata).
        """
        repo_path = Path(repo_path)
        await self.write_policy(repo_path, policy)
        start = time.monotonic()
        events: list[TraceEvent] = []

        async with docker_sandbox(image, repo_path) as artifact_dir:
            wrapper_cmd = [
                "bash",
                "-lc",
                _EPISODE_WRAPPER.format(prompt=_shell_escape(task.prompt)),
            ]
            extra_env: dict[str, str] = {}
            if anthropic_api_key:
                extra_env["ANTHROPIC_API_KEY"] = anthropic_api_key

            result = await run_in_sandbox(
                image,
                repo_path,
                artifact_dir,
                wrapper_cmd,
                timeout_s=timeout_s,
                extra_env=extra_env,
            )

            files_changed = _collect_diff_files(repo_path)
            events.append(
                TraceEvent(
                    timestamp=time.monotonic() - start,
                    kind="exit",
                    payload={"exit_code": result.exit_code, "duration_s": result.duration_s},
                )
            )

            return Trajectory(
                task=task,
                policy=policy,
                events=events,
                files_changed=files_changed,
                exit_code=result.exit_code,
                duration_s=result.duration_s,
                raw_ci_output=result.stdout + "\n" + result.stderr,
                raw_test_output=result.stdout,
                metadata={
                    "artifact_dir": str(artifact_dir),
                    "pytest_report_json": str(artifact_dir / "pytest.json"),
                    "coverage_xml": str(artifact_dir / "coverage.xml"),
                    "ruff_json": str(artifact_dir / "ruff.json"),
                    "mypy_output": str(artifact_dir / "mypy.txt"),
                },
            )


def _stable_version(artifacts: list[Artifact]) -> str:
    """Stable short version derived from artifact contents."""
    import hashlib

    h = hashlib.sha256()
    for a in sorted(artifacts, key=lambda x: (x.type.value, x.name)):
        h.update(a.type.value.encode("utf-8"))
        h.update(a.name.encode("utf-8"))
        h.update(a.content_hash.encode("utf-8"))
    return h.hexdigest()[:12]


def _serialize_artifact(art: Artifact) -> dict[str, Any]:
    return {
        "name": art.name,
        "type": art.type.value,
        "content_hash": art.content_hash,
        "metadata": art.metadata,
    }


# In-container wrapper that runs Claude Code, then CI, and emits artifact files
# at the standard locations the verifier expects under ``/artifacts``.
_EPISODE_WRAPPER = r"""
set -uo pipefail
mkdir -p /artifacts
PROMPT={prompt}
echo "[carl] running claude code"
claude-code --print "$PROMPT" > /artifacts/agent_output.txt 2>&1 || true

echo "[carl] running pytest"
pytest --json-report --json-report-file=/artifacts/pytest.json -q || true

echo "[carl] running coverage"
coverage xml -o /artifacts/coverage.xml || true

echo "[carl] running ruff"
ruff check --output-format=json . > /artifacts/ruff.json 2>/dev/null || true

echo "[carl] running mypy"
mypy . > /artifacts/mypy.txt 2>&1 || true
"""


def _shell_escape(text: str) -> str:
    """Escape ``text`` for embedding inside a single-quoted shell argument."""
    return "'" + text.replace("'", "'\\''") + "'"


def _collect_diff_files(repo_path: Path) -> list[str]:
    """Best-effort: list the files git considers dirty after the episode.

    Falls back to an empty list if the repo isn't a git repo (which should
    never happen in production but happens in tests with synthetic dirs).
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only"], cwd=str(repo_path), text=True, stderr=subprocess.DEVNULL
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _which_or_none(binary: str) -> str | None:
    return shutil.which(binary)
