"""``python -m carl`` invokes the same CLI as the ``carl`` entry-point script.

This test pins the entry point so ``python -m carl auto --dry-run`` keeps
working even when a user's ``PATH`` does not include the bin directory
where ``pip install --user`` placed the ``carl`` script (a common
situation on macOS where ``~/Library/Python/3.x/bin`` is not on the
default ``PATH``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_python_m_carl_help() -> None:
    """``python -m carl --help`` exits 0 and lists the subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "carl", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for sub in ("auto", "init", "run", "status", "gate"):
        assert sub in out, f"subcommand {sub!r} missing from `python -m carl --help` output"


def test_python_m_carl_auto_dry_run(tmp_path: Path) -> None:
    """``python -m carl auto --dry-run`` runs the same pipeline as ``carl auto --dry-run``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    report = tmp_path / "REPORT.md"
    buffer = tmp_path / "buf.sqlite"
    result = subprocess.run(
        [
            sys.executable, "-m", "carl", "auto",
            "--repo", str(repo),
            "--probe-n", "8",
            "--episodes", "4",
            "--report", str(report),
            "--buffer", str(buffer),
            "--dry-run",
            "--seed", "20260517",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert report.is_file()
    body = report.read_text(encoding="utf-8")
    assert "Headline result" in body
    assert "{{LIFT}}" not in body
