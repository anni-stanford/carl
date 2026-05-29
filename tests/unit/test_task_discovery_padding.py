"""Task discovery pads up to the requested count so small repos aren't degenerate."""

from __future__ import annotations

from pathlib import Path

from carl.tasks.discovery import discover_tasks


def test_empty_repo_pads_to_requested_n(tmp_path: Path) -> None:
    """An empty repo has no discoverable tasks; discovery still returns n."""
    out = discover_tasks(tmp_path, n=6)
    assert len(out) == 6
    assert len({t.task_id for t in out}) == 6  # unique ids


def test_tiny_repo_pads_to_requested_n(tmp_path: Path) -> None:
    """A repo with one trivial test still yields n probe tasks."""
    (tmp_path / "m.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_m.py").write_text(
        "from m import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    out = discover_tasks(tmp_path, n=5)
    assert len(out) == 5
    assert len({t.task_id for t in out}) == 5


def test_does_not_exceed_requested_n(tmp_path: Path) -> None:
    out = discover_tasks(tmp_path, n=3)
    assert len(out) == 3
