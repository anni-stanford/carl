# CARL — Reward design

```
r_total = w_v · r_verifier + w_j · r_judge - w_h · r_hack
```

## `r_verifier` — RLVR backbone

Deterministic, hack-resistant. Composite of:

| Signal | Source | Default weight |
|---|---|---|
| `tests_passed` | pytest exit + per-test report | 0.55 |
| `coverage_delta` | coverage.xml diff | 0.20 |
| `lint_clean` | ruff exit | 0.10 |
| `typecheck_clean` | mypy exit | 0.10 |
| `security_clean` | bandit / safety | 0.05 |

Each normalized to `[0, 1]`; weighted sum → `r_verifier ∈ [0, 1]`.

## `r_judge` — RLAIF with bias controls

LLM reviewer score with **all** of:

- **Position-flip**: `(A, B)` and `(B, A)`; only consistent wins counted.
- **Family rotation**: `claude-opus-4-7` ↔ `gpt-5.5` ↔ `composer-2` ↔ `claude-sonnet-4-6`.
- **Rubric-shuffle**: shuffled order of rubric items between calls.
- **CI gating**: `r_judge` is only used when `r_verifier ≥ τ`; otherwise force `r_judge = 0`.
- **Inter-judge agreement** is reported; high variance flags get audited.

## `r_hack` — adversarial probes

Detect:

- `try / except` wrapped around test invocations.
- Ignored exit codes.
- Coverage inflation via test deletions or trivial assertions.
- Hook scripts that touch the CI runner.
- Skills that explicitly tell the agent to bypass type-checking.

`r_hack ∈ [0, 1]` is **subtracted** from the composite reward. See [`reward_hacking.md`](reward_hacking.md).
