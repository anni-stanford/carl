"""SQLite replay buffer round-trips trajectories and joins paired rewards by task."""

from __future__ import annotations

from pathlib import Path

from carl.core.buffer.storage import ReplayBuffer, TrajectoryRow
from carl.core.reward.types import (
    RewardComponents,
    VerifierComponents,
)


def _components(r: float) -> RewardComponents:
    vc = VerifierComponents(
        tests_passed=r,
        coverage_delta=r,
        lint_clean=r,
        typecheck_clean=r,
        security_clean=r,
        composite=r,
        raw_test_count=10,
        raw_failed_count=0,
    )
    return RewardComponents(
        r_total=r,
        r_verifier=r,
        r_judge=0.0,
        r_hack=0.0,
        verifier_breakdown=vc,
        judge_breakdown=None,
        hack_breakdown=None,
    )


def test_paired_rewards_join_by_task_id(tmp_path: Path) -> None:
    buf = ReplayBuffer(tmp_path / "buf.sqlite")
    for i, r_base in enumerate([0.40, 0.55, 0.60, 0.50, 0.45]):
        buf.append_trajectory(
            TrajectoryRow(
                id=f"base-{i}",
                adapter_name="claude_code",
                repo_path="/x",
                task_id=f"t{i}",
                policy_version="v0.0.0+stock",
                components=_components(r_base),
                exit_code=0,
                duration_s=10.0,
            )
        )
    for i, r_cand in enumerate([0.55, 0.65, 0.70, 0.55, 0.60]):
        buf.append_trajectory(
            TrajectoryRow(
                id=f"cand-{i}",
                adapter_name="claude_code",
                repo_path="/x",
                task_id=f"t{i}",
                policy_version="v0.1.0+carl",
                components=_components(r_cand),
                exit_code=0,
                duration_s=10.0,
            )
        )

    cand, base, ids = buf.paired_rewards("v0.1.0+carl", "v0.0.0+stock")
    assert ids == ["t0", "t1", "t2", "t3", "t4"]
    assert cand == [0.55, 0.65, 0.70, 0.55, 0.60]
    assert base == [0.40, 0.55, 0.60, 0.50, 0.45]
    assert buf.trajectory_count() == 10


def test_unpaired_tasks_dropped(tmp_path: Path) -> None:
    buf = ReplayBuffer(tmp_path / "buf.sqlite")
    buf.append_trajectory(
        TrajectoryRow(
            id="b1",
            adapter_name="claude_code",
            repo_path="/x",
            task_id="task_a",
            policy_version="v0",
            components=_components(0.5),
            exit_code=0,
            duration_s=1.0,
        )
    )
    # Candidate exists for a *different* task — should not pair
    buf.append_trajectory(
        TrajectoryRow(
            id="c1",
            adapter_name="claude_code",
            repo_path="/x",
            task_id="task_b",
            policy_version="v1",
            components=_components(0.7),
            exit_code=0,
            duration_s=1.0,
        )
    )
    cand, base, ids = buf.paired_rewards("v1", "v0")
    assert cand == [] and base == [] and ids == []


def test_gate_decision_persisted(tmp_path: Path) -> None:
    buf = ReplayBuffer(tmp_path / "buf.sqlite")
    buf.append_gate_decision(
        candidate_version="v1",
        baseline_version="v0",
        promote=True,
        mean_lift=0.07,
        ci_low=0.02,
        ci_high=0.12,
        p_value=0.003,
        n_tasks=40,
        n_resamples=10000,
        reason="PROMOTE: lift 0.07, CI [0.02, 0.12]",
    )
    # Just verify the row exists by counting via a raw query
    import sqlite3

    with sqlite3.connect(buf.db_path) as c:
        (n,) = c.execute("SELECT COUNT(*) FROM gate_decisions").fetchone()
    assert n == 1
