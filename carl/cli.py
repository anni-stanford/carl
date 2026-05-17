"""CARL command-line interface — ``carl init|run|status|gate``.

Implements the four commands the README documents. ``init`` scaffolds a
baseline policy (``CLAUDE.md`` or ``.cursor/rules``) into the target repo.
``run`` calls :func:`carl.loop.carl_loop` over a single repo + a small
manifest of tasks; without an ``ANTHROPIC_API_KEY`` it errors clearly.
``status`` reads the SQLite replay buffer and prints summary statistics.
``gate`` runs the paired-bootstrap promotion gate on already-collected
rewards (this is the same logic as ``experiments.ab_compare``, exposed
under the ``carl`` entry point for convenience).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

import click

_DEFAULT_BUFFER = Path("carl_run/buffer.sqlite")


@click.group()
@click.version_option(package_name="carl-loop")
def main() -> None:
    """CARL — Continuous Agent Reinforcement Loop."""


# ---- carl init ---------------------------------------------------------------


@main.command()
@click.option(
    "--adapter",
    type=click.Choice(["claude_code", "cursor"]),
    required=True,
    help="Which IDE adapter to scaffold.",
)
@click.option(
    "--repo",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
)
def init(adapter: str, repo: Path) -> None:
    """Write a baseline policy (CLAUDE.md or .cursor/rules) into ``--repo``.

    The seed policy is a minimal but real artifact set: project rules, one
    skill, and (for Claude Code) a sample sub-agent. CARL will refine these
    as episodes accumulate.
    """
    from carl.adapters.base import PolicyAdapter
    from carl.adapters.claude_code import ClaudeCodeAdapter
    from carl.adapters.cursor import CursorAdapter
    from carl.core.policy.artifacts import Artifact, ArtifactType, Policy

    repo = Path(repo).resolve()
    repo.mkdir(parents=True, exist_ok=True)

    rules_seed = (
        "# Project rules (CARL-managed)\n\n"
        "1. Always run the test suite before claiming a fix.\n"
        "2. Type-annotate all public functions; mypy --strict must pass.\n"
        "3. Prefer pytest with `--strict-markers`.\n"
        "4. Edit files locally, do not modify the reward stack from inside an episode.\n"
    )
    skill_seed = (
        "# Skill: testing-policy\n\n"
        "When writing or modifying tests, prefer pytest. Mark slow tests with\n"
        "`@pytest.mark.slow`. Assert behavior, not implementation details.\n"
    )

    ad: PolicyAdapter
    if adapter == "claude_code":
        ad = ClaudeCodeAdapter()
        policy = Policy(
            artifacts=[
                Artifact(name="CLAUDE.md", type=ArtifactType.RULES, content=rules_seed),
                Artifact(name="testing-policy", type=ArtifactType.SKILL, content=skill_seed),
            ],
            version="seed",
        )
    else:
        ad = CursorAdapter()
        policy = Policy(
            artifacts=[
                Artifact(name="rules", type=ArtifactType.RULES, content=rules_seed),
                Artifact(name="testing-policy", type=ArtifactType.SKILL, content=skill_seed),
            ],
            version="seed",
        )

    asyncio.run(ad.write_policy(repo, policy))
    click.echo(f"[carl init] wrote {len(policy.artifacts)} seed artifact(s) for adapter={adapter}")
    click.echo(f"[carl init] inspect with: ls -la {repo}/{'.claude' if adapter == 'claude_code' else '.cursor'}/")


# ---- carl run ----------------------------------------------------------------


@main.command()
@click.option("--repo", type=click.Path(file_okay=False, path_type=Path), default=Path("."))
@click.option("--adapter", type=click.Choice(["claude_code", "cursor"]), default="claude_code")
@click.option(
    "--task",
    multiple=True,
    help="Task prompt to queue; repeat to add multiple tasks.",
)
@click.option(
    "--buffer",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_BUFFER,
    show_default=True,
    help="SQLite replay buffer path.",
)
@click.option(
    "--require-anthropic/--no-require-anthropic",
    default=True,
    help="Hard-fail if ANTHROPIC_API_KEY is unset (default). Pass --no-require-anthropic to dry-run.",
)
def run(repo: Path, adapter: str, task: tuple[str, ...], buffer: Path, require_anthropic: bool) -> None:
    """Run the CARL loop over ``--repo`` with ``--adapter``.

    A real run requires ``ANTHROPIC_API_KEY`` (and a Docker daemon for the
    Claude Code adapter). Without those, the loop will refuse to start
    rather than producing meaningless results — pass
    ``--no-require-anthropic`` to validate the wiring without executing.
    """
    if require_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo(
            "error: ANTHROPIC_API_KEY is not set. Real episode execution requires the\n"
            "Anthropic API for the LLM judge / mutator / diagnosis agent.\n"
            "Set the env var, or pass --no-require-anthropic to dry-run.",
            err=True,
        )
        sys.exit(2)

    if not task:
        click.echo("error: at least one --task is required", err=True)
        sys.exit(2)

    from carl.adapters.base import PolicyAdapter
    from carl.adapters.base import Task as CarlTask
    from carl.adapters.claude_code import ClaudeCodeAdapter
    from carl.adapters.cursor import CursorAdapter
    from carl.core.buffer.storage import ReplayBuffer
    from carl.loop import carl_loop
    from carl.settings import CARLConfig

    buffer.parent.mkdir(parents=True, exist_ok=True)
    buf = ReplayBuffer(buffer)
    cfg = CARLConfig()

    adapter_obj: PolicyAdapter = (
        ClaudeCodeAdapter() if adapter == "claude_code" else CursorAdapter()
    )
    initial_policy = asyncio.run(adapter_obj.read_policy(repo))
    initial_policies = {"seed": initial_policy}

    tasks_in = [
        CarlTask(task_id=f"cli-{i}", repo_path=repo, prompt=t, adapter_name=adapter)
        for i, t in enumerate(task)
    ]

    if not require_anthropic:
        click.echo(
            f"[carl run] dry-run: would queue {len(tasks_in)} task(s) on {adapter} "
            f"against {repo} with policy {initial_policy.version}"
        )
        return

    asyncio.run(
        carl_loop(
            adapters=[adapter_obj],
            config=cfg,
            initial_policies=initial_policies,
            tasks=tasks_in,
            buffer=buf,
        )
    )
    click.echo(f"[carl run] {buf.trajectory_count()} trajectories in {buffer}")


# ---- carl status -------------------------------------------------------------


@main.command()
@click.option(
    "--buffer",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_BUFFER,
    show_default=True,
)
def status(buffer: Path) -> None:
    """Summarize the current replay buffer."""
    if not buffer.is_file():
        click.echo(f"[carl status] no buffer at {buffer}", err=True)
        sys.exit(1)
    with sqlite3.connect(str(buffer)) as conn:
        n_traj = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        versions = [
            r[0] for r in conn.execute("SELECT DISTINCT policy_version FROM trajectories").fetchall()
        ]
        gates = conn.execute(
            "SELECT COUNT(*), SUM(promote) FROM gate_decisions"
        ).fetchone()
        last_reward = conn.execute(
            "SELECT created_at, policy_version, r_total FROM trajectories "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    click.echo(f"buffer:               {buffer}")
    click.echo(f"trajectories:         {n_traj:>6}")
    click.echo(f"policy versions seen: {len(versions)}  ({', '.join(versions[:5])}{'...' if len(versions) > 5 else ''})")
    click.echo(f"gate decisions:       {gates[0]} ({gates[1] or 0} promoted)")
    if last_reward:
        click.echo(f"last reward:          {last_reward[2]:.4f} on {last_reward[1]} at {last_reward[0]}")


# ---- carl gate ---------------------------------------------------------------


@main.command()
@click.option(
    "--buffer",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_BUFFER,
    show_default=True,
)
@click.option("--candidate", required=True)
@click.option("--baseline", required=True)
@click.option("--seed", type=int, default=20260517)
def gate(buffer: Path, candidate: str, baseline: str, seed: int) -> None:
    """Run the paired-bootstrap promotion gate on buffered rewards."""
    from experiments.ab_compare import main as ab_main

    rc = ab_main(
        [
            "--buffer", str(buffer),
            "--candidate", candidate,
            "--baseline", baseline,
            "--seed", str(seed),
        ]
    )
    sys.exit(rc)


if __name__ == "__main__":  # pragma: no cover
    main()
