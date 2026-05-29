"""``carl auto --dry-run`` runs end-to-end and produces a real CARL_REPORT.md.

This is the user-facing supershort-command experience verified end-to-end
without Docker, without an API key, without network. The dry-run uses
deterministic synthetic rewards so the produced report is fully reproducible.
"""

from __future__ import annotations

import re
from pathlib import Path

from click.testing import CliRunner

from carl.cli import main


def test_auto_dry_run_writes_report_with_real_numbers(tmp_path: Path) -> None:
    runner = CliRunner()
    report = tmp_path / "CARL_REPORT.md"
    buffer = tmp_path / "buffer.sqlite"
    repo = tmp_path / "fakerepo"
    repo.mkdir()

    result = runner.invoke(
        main,
        [
            "auto",
            "--repo", str(repo),
            "--probe-n", "12",
            "--episodes", "8",
            "--report", str(report),
            "--buffer", str(buffer),
            "--dry-run",
            "--seed", "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert report.is_file(), "auto did not produce CARL_REPORT.md"
    body = report.read_text(encoding="utf-8")

    # Real headline numbers (not placeholders) — the Mean reward row must contain
    # actual decimals from the synthetic pipeline.
    assert "Headline result" in body
    assert "{{LIFT}}" not in body  # no placeholders left over
    # Decision is one of the explicit verbs
    assert ("**PROMOTE**" in body) or ("**REJECT**" in body)
    # CI bound row appears
    assert re.search(r"95\s*%\s*CI", body)
    # Reward decomposition section present
    assert "Reward decomposition" in body
    # Reproducibility instruction present
    assert "carl auto" in body
    # The CLI prints the headline summary at the end
    assert "mean lift" in result.output


def test_auto_dry_run_is_deterministic_in_headline(tmp_path: Path) -> None:
    """Same seed → same headline numbers (not byte-equal because the report contains a wall-clock timestamp)."""
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()

    headlines: list[str] = []
    for trial in (1, 2):
        report = tmp_path / f"r{trial}.md"
        buffer = tmp_path / f"b{trial}.sqlite"
        result = runner.invoke(
            main,
            [
                "auto",
                "--repo", str(repo),
                "--probe-n", "10",
                "--episodes", "6",
                "--report", str(report),
                "--buffer", str(buffer),
                "--dry-run",
                "--seed", "11",
            ],
        )
        assert result.exit_code == 0, result.output
        body = report.read_text(encoding="utf-8")
        headline_block = body.split("## Reward decomposition")[0]
        # Strip wall-clock + repo-path lines (paths to tmp dirs differ)
        headline_block = re.sub(r"\*\*Generated:\*\* .*\n", "", headline_block)
        headline_block = re.sub(r"\*\*Repository:\*\* .*\n", "", headline_block)
        headlines.append(headline_block)
    assert headlines[0] == headlines[1], (
        "auto headline is not deterministic at fixed seed:\n"
        f"--- trial 1 ---\n{headlines[0]}\n--- trial 2 ---\n{headlines[1]}"
    )


def test_auto_docker_mode_fails_loudly_without_env(tmp_path: Path, monkeypatch) -> None:
    """--mode docker without ANTHROPIC_API_KEY => clear error, not a corrupt run."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    result = runner.invoke(
        main,
        [
            "auto",
            "--repo", str(repo),
            "--mode", "docker",
            "--probe-n", "10",
            "--episodes", "4",
            "--report", str(tmp_path / "r.md"),
            "--buffer", str(tmp_path / "b.sqlite"),
        ],
    )
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_auto_mode_falls_back_to_dry_run_gracefully(tmp_path: Path, monkeypatch) -> None:
    """Default --mode auto with no Claude CLI, no Docker, no key => clean dry-run.

    This is the key behavior change that makes `carl auto` a true single
    command: it never hard-crashes for missing prerequisites; it falls back
    to a clearly-labeled synthetic dry-run and still writes a report.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CARL_CLAUDE_BIN", raising=False)
    # Force claude detection + docker detection to fail.
    monkeypatch.setattr("carl.env.local_sandbox.claude_cli_available", lambda: None)
    monkeypatch.setattr("carl.auto._docker_available", lambda: False)

    runner = CliRunner()
    repo = tmp_path / "repo"
    repo.mkdir()
    report = tmp_path / "r.md"
    result = runner.invoke(
        main,
        [
            "auto",
            "--repo", str(repo),
            "--probe-n", "6",
            "--episodes", "4",
            "--report", str(report),
            "--buffer", str(tmp_path / "b.sqlite"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower()
    assert report.is_file()
