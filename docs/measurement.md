# CARL — How outcomes are measured

This document is the contract between CARL's code and its paper. Every claim in `paper/draft.md` traces back to a function defined here.

## 1. The unit: per-episode reward `r_total ∈ [0, 1]`

Computed by `carl.core.reward.composite.compose_reward`:

```
r_total = w_v · r_verifier + w_j · r_judge − w_h · r_hack
        = 0.65            + 0.25         − 0.10              # default weights
```

Each episode logs the **decomposition**, not just the scalar (see `RewardComponents` in `carl/core/reward/types.py`). This is what makes ablations and the paper's qualitative analysis possible.

| Component | Source | Implementation |
|---|---|---|
| `r_verifier` | CI pipeline (deterministic) | `carl/core/reward/verifier.py` |
| `r_judge` | bias-controlled LLM reviewer | `carl/core/reward/judge.py` (Day 5) |
| `r_hack` | adversarial reward-hacking probes | `carl/core/reward/hack_probe.py` (Day 5) |

The verifier is the **backbone**: it is deterministic, hack-resistant, and the only one of the three that's already implemented. The other two are designed; their absence in v1 is acknowledged in the paper's limitations.

## 2. Judge gating (anti-hacking property)

`compose_reward` enforces: if `r_verifier < 0.5`, the judge contribution is forced to zero. Without this, a model could learn to write convincing-looking code that doesn't pass tests. This is **not** a CARL invention — it follows the RLAIF safety pattern (Bai et al., 2022) — but it must be in the composition function.

## 3. The promotion gate — paired bootstrap with `n ≥ 30`

`carl.core.promotion.gate.evaluate_gate` runs a **paired** bootstrap on per-task reward deltas using `scipy.stats.bootstrap(method="BCa")`. A candidate policy is promoted iff:

1. `n_tasks ≥ 30` (conventional minimum for credible bootstrap CIs).
2. `mean_lift > 0`.
3. `95 % CI lower bound > 0` (the strict criterion).

The function returns a `GateResult` carrying mean lift, CI bounds, one-sided _p_-value, sample size, resample count, and a human-readable reason — which is what gets cited in commit messages, the dashboard, and the paper.

The pairing dramatically reduces variance vs an independent-samples test. It works because the candidate and baseline are evaluated on the **same task set with the same seeds**. Reference: Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, §15.

## 4. Required sample sizes (defended by the rubric)

| Use case | Required n | Why |
|---|---|---|
| **Promotion-gate probe set** | ≥ 30 | Bootstrap CIs are credible past this threshold; reviewers will reject smaller. |
| **CARL vs baseline (E1, headline)** | ≥ 30 paired tasks | Same gate, same statistical test. |
| **Per-technique ablation (E2)** | ≥ 30 paired tasks per ablated condition | Each ablation is a separate paired test against full-CARL. |
| **Diagnosis-attribution validation (E4)** | ≥ 30 manually-labeled failed trajectories | Top-1 / top-3 accuracy is reported with binomial CIs (Wilson score, since proportions). |
| **Reward-hacking probes (E8)** | ≥ 30 probes with known ground truth | Hack-rate reported with Wilson CI. |

These sizes are floors. More is better. Less than 30 is not defensible to a reviewer and does not appear in the paper without an explicit `n < 30` caveat.

## 5. What goes in the paper's headline table (E1)

Generated end-to-end by `python -m experiments.ab_compare --buffer carl_run/buffer.sqlite --candidate v0.1.0+carl --baseline v0.0.0+stock`:

```
================================================================
  CARL — A/B comparison: v0.1.0+carl  vs  v0.0.0+stock
================================================================
  paired tasks (n)           : 40
  candidate mean reward      : 0.7142
  baseline  mean reward      : 0.5688
  mean lift                  : +0.1454
  95 % CI (BCa, n_resamples=10000)
                             : [+0.0801, +0.2117]
  one-sided p-value          : 0.0008
  decision                   : PROMOTE
================================================================
```

(Numbers above are illustrative; actual run rewrites them.)

## 6. Anti-patterns we will not fall into

- **Single-run "lift"**: never reported without `n ≥ 30` and a CI.
- **Mean-lift > 0 used as the promotion criterion**: rejected; we require CI lower bound > 0. This is what the test `test_no_lift_does_not_promote` enforces.
- **Test-set leakage**: train tasks and probe tasks come from disjoint commits or disjoint repos; the split's hash is committed to the repo.
- **Same model family as mutator and judge**: family rotation defends against this; if rotation degenerates we report it.
- **Cherry-picked seeds**: every result is the mean over ≥ 3 seeds; the paper reports mean ± 95 % CI across seeds.

## 7. Reproducibility

Every gate decision is persisted to the `gate_decisions` table of the SQLite replay buffer (`carl/core/buffer/storage.py`). Every promoted policy is git-tagged with its reward delta. `bash scripts/run_all.sh` reproduces the full table from a clean clone.
