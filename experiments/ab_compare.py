"""``carl.experiments.ab_compare`` — driver for E1's headline table.

Reads paired (candidate, baseline) reward rows from a CARL replay buffer,
runs the paired-bootstrap promotion gate, and prints the headline table that
goes straight into ``paper/draft.md``. No LLM calls; pure statistics on
already-collected rewards.

Example:
    python -m experiments.ab_compare \\
        --buffer carl_run/buffer.sqlite \\
        --candidate v0.1.0+carl \\
        --baseline  v0.0.0+stock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from carl.core.buffer.storage import ReplayBuffer
from carl.core.promotion.gate import evaluate_gate
from carl.settings import PromotionGateConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buffer", type=Path, required=True, help="path to SQLite replay buffer")
    parser.add_argument("--candidate", required=True, help="candidate policy_version")
    parser.add_argument("--baseline", required=True, help="baseline policy_version")
    parser.add_argument("--seed", type=int, default=20260517, help="RNG seed for reproducibility")
    parser.add_argument(
        "--n-resamples", type=int, default=10_000, help="bootstrap resample count"
    )
    parser.add_argument(
        "--min-probe-tasks", type=int, default=30, help="reject if n < this"
    )
    args = parser.parse_args(argv)

    if not args.buffer.is_file():
        print(f"error: buffer file not found: {args.buffer}", file=sys.stderr)
        return 2

    buf = ReplayBuffer(args.buffer)
    cand, base, task_ids = buf.paired_rewards(args.candidate, args.baseline)

    if len(cand) == 0:
        print(
            f"error: no paired rewards found for candidate={args.candidate!r} "
            f"baseline={args.baseline!r}",
            file=sys.stderr,
        )
        return 2

    cfg = PromotionGateConfig(
        n_resamples=args.n_resamples,
        confidence=0.95,
        min_probe_tasks=args.min_probe_tasks,
        require_ci_lower_bound_above=0.0,
    )
    result = evaluate_gate(cand, base, cfg, rng_seed=args.seed)

    print()
    print("=" * 78)
    print(f"  CARL — A/B comparison: {args.candidate}  vs  {args.baseline}")
    print("=" * 78)
    print(f"  paired tasks (n)           : {result.n_tasks}")
    print(f"  candidate mean reward      : {sum(cand) / len(cand):.4f}")
    print(f"  baseline  mean reward      : {sum(base) / len(base):.4f}")
    print(f"  mean lift                  : {result.mean_lift:+.4f}")
    print(f"  95% CI (BCa, n_resamples={result.n_resamples})")
    print(f"                             : [{result.ci_low:+.4f}, {result.ci_high:+.4f}]")
    print(f"  one-sided p-value          : {result.p_value:.4f}")
    print(f"  decision                   : {'PROMOTE' if result.promote else 'REJECT '}")
    print(f"  reason                     : {result.reason}")
    print("=" * 78)
    print()

    # Persist the gate decision so the dashboard can show history
    buf.append_gate_decision(
        candidate_version=args.candidate,
        baseline_version=args.baseline,
        promote=result.promote,
        mean_lift=result.mean_lift,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        p_value=result.p_value,
        n_tasks=result.n_tasks,
        n_resamples=result.n_resamples,
        reason=result.reason,
    )

    return 0 if result.promote else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
