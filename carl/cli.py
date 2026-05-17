"""CARL command-line interface — ``carl init|run|eval|status``."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="carl-loop")
def main() -> None:
    """CARL — Continuous Agent Reinforcement Loop."""


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
    """Scaffold ``CLAUDE.md`` (or ``.cursor/rules``) and a baseline policy."""
    click.echo(f"[carl init] adapter={adapter} repo={repo}")
    click.echo("Day-1 stub. Full scaffolding lands Day 2 of the build sequence.")


@main.command()
@click.option(
    "--repo", type=click.Path(file_okay=False, path_type=Path), default=Path(".")
)
@click.option(
    "--adapter", type=click.Choice(["claude_code", "cursor"]), default="claude_code"
)
def run(repo: Path, adapter: str) -> None:
    """Start the CARL loop on ``--repo`` using ``--adapter``."""
    click.echo(f"[carl run] repo={repo} adapter={adapter}")
    click.echo(
        "Day-1 stub. End-to-end loop lands Day 12; episode execution Days 3–4."
    )


@main.command()
def status() -> None:
    """Show current policy version, recent reward, and queue depth."""
    click.echo("[carl status] not yet implemented")


if __name__ == "__main__":  # pragma: no cover
    main()
