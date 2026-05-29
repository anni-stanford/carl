"""End-to-end test of the no-Docker local runner with a fake Claude CLI.

This proves the local pipeline (copy repo -> write policy -> run agent ->
run real CI -> parse artifacts -> reward -> gate -> report) works with only
the *agent* mocked. The real Claude Code CLI is replaced by a tiny script
(via CARL_CLAUDE_BIN); everything else — the actual pytest run, the verifier
parsing, the buffer, the bootstrap gate, the report — is real.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from carl.cli import main


def _make_fake_claude(tmp_path: Path) -> Path:
    """A fake Claude Code CLI that accepts `-p <prompt>` and does nothing useful."""
    script = tmp_path / "fake_claude.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'echo "fake-claude received: $*"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def _make_target_repo(tmp_path: Path) -> Path:
    """A minimal but real Python repo with a passing pytest suite."""
    repo = tmp_path / "target_repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_math.py").write_text(
        "from pkg.math_utils import add, mul\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_mul():\n    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )
    return repo


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="fake bash CLI is POSIX-only"
)
def test_carl_auto_local_produces_real_report(tmp_path: Path, monkeypatch) -> None:
    fake_claude = _make_fake_claude(tmp_path)
    repo = _make_target_repo(tmp_path)
    report = tmp_path / "CARL_REPORT.md"
    buffer = tmp_path / "buf.sqlite"

    monkeypatch.setenv("CARL_CLAUDE_BIN", str(fake_claude))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "auto",
            "--repo", str(repo),
            "--mode", "local",
            "--probe-n", "4",
            "--episodes", "4",
            "--report", str(report),
            "--buffer", str(buffer),
            "--seed", "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "using local runner" in result.output or "local" in result.output.lower()
    assert report.is_file(), f"no report produced. output:\n{result.output}"

    body = report.read_text(encoding="utf-8")
    assert "Headline result" in body
    assert "{{LIFT}}" not in body
    # The fake agent ran (the CLI was invoked); CI ran for real against the repo.
    # We don't assert a positive lift (the fake agent changes nothing), only
    # that the pipeline produced a real, well-formed report end to end.
    assert ("PROMOTE" in body) or ("REJECT" in body)


def test_claude_cli_detection_prefers_override(tmp_path: Path, monkeypatch) -> None:
    from carl.env.local_sandbox import claude_cli_available

    fake = tmp_path / "x"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("CARL_CLAUDE_BIN", str(fake))
    assert claude_cli_available() == str(fake)

    monkeypatch.delenv("CARL_CLAUDE_BIN", raising=False)
    # Without override and without a real claude on PATH in CI, this is None
    # (or a real path on a dev machine that has Claude Code installed).
    result = claude_cli_available()
    assert result is None or isinstance(result, str)
