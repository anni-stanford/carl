"""CLI commands ``init``, ``run --no-require-anthropic``, ``status`` actually work.

These are the commands the README documents and that external reviewers
test against. They previously printed "Day-1 stub"; this test pins the
real behavior so they can't silently regress.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from carl.cli import main


def test_init_writes_seed_policy_for_claude_code(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--repo", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "CLAUDE.md").is_file()
    assert (tmp_path / ".claude" / "skills" / "testing-policy" / "SKILL.md").is_file()
    assert "wrote" in result.output


def test_run_dry_run_without_api_key(tmp_path: Path, monkeypatch) -> None:
    """`carl run --no-require-anthropic` should succeed even with no ANTHROPIC_API_KEY."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Seed a policy first so read_policy doesn't return empty
    runner = CliRunner()
    init_res = runner.invoke(main, ["init", "--repo", str(tmp_path)])
    assert init_res.exit_code == 0
    res = runner.invoke(
        main,
        [
            "run",
            "--repo", str(tmp_path),
            "--task", "Improve test coverage by 5 pp",
            "--no-require-anthropic",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "dry-run" in res.output


def test_run_without_api_key_errors_with_clear_message(tmp_path: Path, monkeypatch) -> None:
    """Default `carl run` (no --no-require-anthropic) must fail loudly without API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "run", "--repo", str(tmp_path),
            "--task", "anything",
        ],
    )
    assert res.exit_code == 2
    assert "ANTHROPIC_API_KEY" in res.output


def test_status_on_missing_buffer(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["status", "--buffer", str(tmp_path / "nope.sqlite")])
    assert res.exit_code == 1
    assert "no buffer" in res.output


def test_status_on_real_buffer(tmp_path: Path) -> None:
    """End-to-end: synthetic-demo writes a buffer, `carl status` summarizes it."""
    from experiments.run_synthetic_demo import main as demo_main

    buf_path = tmp_path / "buf.sqlite"
    demo_main(["--out", str(buf_path), "--n-tasks", "30", "--seed", "1"])

    runner = CliRunner()
    res = runner.invoke(main, ["status", "--buffer", str(buf_path)])
    assert res.exit_code == 0, res.output
    assert "trajectories:" in res.output
    assert "60" in res.output  # 30 paired tasks * 2 versions = 60 rows
