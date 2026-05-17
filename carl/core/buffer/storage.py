"""Minimal SQLite trajectory + preference-pair store for v1.

Postgres + pgvector (per the project specification) is the v2 backend. For
the May 29 deliverable we use stdlib ``sqlite3`` because:

  * No external dependency.
  * No daemon to install on every developer's machine.
  * Same SQL schema, drop-in replaceable when v2 lands.

The schema mirrors what an RL framework actually needs for ablations and
the paper's tables: per-trajectory reward decomposition, per-promotion gate
result, paired pre/post deltas for the bootstrap test.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from carl.core.reward.types import RewardComponents

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    adapter_name    TEXT NOT NULL,
    repo_path       TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    r_total         REAL NOT NULL,
    r_verifier      REAL NOT NULL,
    r_judge         REAL NOT NULL,
    r_hack          REAL NOT NULL,
    components_json TEXT NOT NULL,
    exit_code       INTEGER NOT NULL,
    duration_s      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traj_task ON trajectories(task_id);
CREATE INDEX IF NOT EXISTS idx_traj_policy ON trajectories(policy_version);
CREATE INDEX IF NOT EXISTS idx_traj_adapter ON trajectories(adapter_name);

CREATE TABLE IF NOT EXISTS preference_pairs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    preferred    TEXT NOT NULL,        -- serialized PolicyDiff
    rejected     TEXT,                 -- serialized PolicyDiff (nullable)
    context_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gate_decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    candidate_version TEXT NOT NULL,
    baseline_version  TEXT NOT NULL,
    promote           INTEGER NOT NULL,
    mean_lift         REAL NOT NULL,
    ci_low            REAL NOT NULL,
    ci_high           REAL NOT NULL,
    p_value           REAL NOT NULL,
    n_tasks           INTEGER NOT NULL,
    n_resamples       INTEGER NOT NULL,
    reason            TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class TrajectoryRow:
    """A row in the ``trajectories`` table — paired-bootstrap input lives here."""

    id: str
    adapter_name: str
    repo_path: str
    task_id: str
    policy_version: str
    components: RewardComponents
    exit_code: int
    duration_s: float


class ReplayBuffer:
    """SQLite-backed replay buffer for trajectories, preferences, and gate results."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # --------- writes ---------

    def append_trajectory(self, row: TrajectoryRow) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO trajectories "
                "(id, created_at, adapter_name, repo_path, task_id, policy_version, "
                "r_total, r_verifier, r_judge, r_hack, components_json, exit_code, duration_s) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row.id,
                    datetime.now(tz=UTC).isoformat(),
                    row.adapter_name,
                    row.repo_path,
                    row.task_id,
                    row.policy_version,
                    row.components.r_total,
                    row.components.r_verifier,
                    row.components.r_judge,
                    row.components.r_hack,
                    json.dumps(_components_to_dict(row.components)),
                    row.exit_code,
                    row.duration_s,
                ),
            )
            conn.commit()

    def append_gate_decision(
        self,
        candidate_version: str,
        baseline_version: str,
        promote: bool,
        mean_lift: float,
        ci_low: float,
        ci_high: float,
        p_value: float,
        n_tasks: int,
        n_resamples: int,
        reason: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO gate_decisions "
                "(created_at, candidate_version, baseline_version, promote, "
                "mean_lift, ci_low, ci_high, p_value, n_tasks, n_resamples, reason) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(tz=UTC).isoformat(),
                    candidate_version,
                    baseline_version,
                    int(promote),
                    mean_lift,
                    ci_low,
                    ci_high,
                    p_value,
                    n_tasks,
                    n_resamples,
                    reason,
                ),
            )
            conn.commit()

    # --------- reads ---------

    def paired_rewards(
        self,
        candidate_version: str,
        baseline_version: str,
    ) -> tuple[list[float], list[float], list[str]]:
        """Return (candidate_r_totals, baseline_r_totals, task_ids) paired by task.

        Tasks for which either side is missing are dropped — paired bootstrap
        requires equal-length, same-task sequences.
        """
        sql = (
            "SELECT t1.task_id, t1.r_total, t2.r_total "
            "FROM trajectories t1 "
            "JOIN trajectories t2 ON t1.task_id = t2.task_id "
            "WHERE t1.policy_version = ? AND t2.policy_version = ? "
            "ORDER BY t1.task_id"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, (candidate_version, baseline_version)).fetchall()
        cand = [r[1] for r in rows]
        base = [r[2] for r in rows]
        ids = [r[0] for r in rows]
        return cand, base, ids

    def trajectory_count(self) -> int:
        with self._conn() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()
        return int(n)


def _components_to_dict(c: RewardComponents) -> dict[str, object]:
    """Best-effort JSON serialization of the nested dataclass tree."""
    out: dict[str, object] = {
        "r_total": c.r_total,
        "r_verifier": c.r_verifier,
        "r_judge": c.r_judge,
        "r_hack": c.r_hack,
        "verifier_breakdown": asdict(c.verifier_breakdown),
    }
    if c.judge_breakdown is not None:
        out["judge_breakdown"] = asdict(c.judge_breakdown)
    if c.hack_breakdown is not None:
        out["hack_breakdown"] = asdict(c.hack_breakdown)
    return out
