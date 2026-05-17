"""Generate a deterministic synthetic experiment so the measurement pipeline
produces a real (clearly-labeled) headline table without needing Docker, an
LLM API key, or a real coding agent.

This is **not a real experimental result**. The script writes a SQLite
replay buffer populated with **synthetic, seeded** rewards designed to
exercise the full verifier → composite → buffer → gate pipeline. Running
it followed by ``experiments/ab_compare.py`` reproduces the headline
table format the paper will use.

Use::

    python -m experiments.run_synthetic_demo --out carl_run/synthetic.sqlite
    python -m experiments.ab_compare \\
        --buffer carl_run/synthetic.sqlite \\
        --candidate v0.1.0+carl-synthetic \\
        --baseline  v0.0.0+stock-synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from carl.core.buffer.storage import ReplayBuffer, TrajectoryRow
from carl.core.reward.types import RewardComponents, VerifierComponents


def _row(idx: int, version: str, r: float, task_id: str) -> TrajectoryRow:
    vc = VerifierComponents(
        tests_passed=r,
        coverage_delta=r,
        lint_clean=r,
        typecheck_clean=r,
        security_clean=None,
        composite=r,
        raw_test_count=20,
        raw_failed_count=int(round((1 - r) * 20)),
    )
    rc = RewardComponents(
        r_total=r,
        r_verifier=r,
        r_judge=0.0,
        r_hack=0.0,
        verifier_breakdown=vc,
        judge_breakdown=None,
        hack_breakdown=None,
    )
    return TrajectoryRow(
        id=f"{version}-{idx}",
        adapter_name="claude_code",
        repo_path="/synthetic",
        task_id=task_id,
        policy_version=version,
        components=rc,
        exit_code=0,
        duration_s=1.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output SQLite buffer path")
    parser.add_argument("--n-tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    buf = ReplayBuffer(args.out)

    rng = np.random.default_rng(args.seed)
    base_mean, cand_mean = 0.55, 0.70
    sigma = 0.08

    for i in range(args.n_tasks):
        task_id = f"synthetic_task_{i:03d}"
        r_base = float(np.clip(rng.normal(base_mean, sigma), 0.0, 1.0))
        r_cand = float(np.clip(rng.normal(cand_mean, sigma), 0.0, 1.0))
        buf.append_trajectory(_row(i, "v0.0.0+stock-synthetic", r_base, task_id))
        buf.append_trajectory(_row(i, "v0.1.0+carl-synthetic", r_cand, task_id))

    print(
        f"[carl] wrote {buf.trajectory_count()} synthetic trajectory rows to {args.out}"
    )
    print(
        "[carl] reminder: this buffer contains SYNTHETIC data; the resulting headline"
        " table is for pipeline-validation only and must NOT be cited as a real result."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
