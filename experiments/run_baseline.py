"""Stub for the stock-agent baseline runner (Day 12 of the build sequence).

Will iterate over the held-out probe set, run the agent without any policy
updates, score with :mod:`carl.core.reward`, and store the resulting
:class:`TrajectoryRow` rows under ``policy_version="v0.0.0+stock"``.

The CARL run (``run_carl.py``) writes under a different ``policy_version``
and ``ab_compare.py`` joins them on ``task_id``.
"""

from __future__ import annotations


def main() -> int:  # pragma: no cover
    raise NotImplementedError(
        "experiments.run_baseline runs Day 12 of the build sequence "
        "(after run_episode is wired)"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
