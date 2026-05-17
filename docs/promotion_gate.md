# CARL — Promotion gate

A proposed `PolicyDiff` is merged into the active policy iff:

1. It is **within the locality budget** (`≤ max_diff_lines`, default 5; new skills allowed only when failure-cluster size ≥ 5).
2. Its **DPO ranker** score is in the top-K (default 2 of 5 candidates).
3. The **paired bootstrap** on per-task reward deltas across a held-out probe set returns a 95 % CI **lower bound > 0**.

## Procedure

```python
# experiments/run_carl.py uses scipy.stats.bootstrap(paired=True)
deltas = [r_candidate(t) - r_baseline(t) for t in held_out_probes]
res = bootstrap((deltas,), np.mean, paired=True,
                n_resamples=10_000, confidence_level=0.95)
if res.confidence_interval.low > 0:
    promote(candidate, mean_lift=np.mean(deltas),
            ci=res.confidence_interval, p_value=one_sided_p(deltas))
```

## What is reported per promotion

- Mean reward lift over baseline.
- 95 % CI (low, high).
- One-sided p-value.
- Number of probe tasks (`n`).
- Adapter name, target artifact, line-range, rationale citation back to the failure trace evidence.

Each promotion is a **git tag** (`v0.1.0+r0.034`) so anyone can reproduce the policy at any point in history.
