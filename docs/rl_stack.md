# CARL — The four-technique RL stack

CARL transposes the canonical 2026 post-training RL stack to **text-space policy parameters**. We do **not** fine-tune model weights; we update artifacts that condition the agent (CLAUDE.md, `.claude/skills`, sub-agents, hooks, MCP config, `.cursor/rules`, `.cursor/skills`, etc.).

## 1. RLVR — verifiable reward backbone

The CI pipeline is the verifier. `r_verifier` is a deterministic, hack-resistant composite of pytest exit code, per-test results, coverage delta, ruff cleanliness, mypy cleanliness, and security findings. Implemented in `carl/core/reward/verifier.py`.

## 2. GRPO-style group-relative scoring

For each task, run **K** trajectories under K policy variants. Compute trajectory-level advantage:

$$\text{advantage}_i = \frac{r_i - \mathrm{mean}(r_{1..K})}{\mathrm{std}(r_{1..K}) + \varepsilon}$$

Promote variants with **positive advantage AND CI lower bound > 0**. This is GRPO transposed: the same group-normalization math, but the update target is "promote variant" rather than "gradient step on weights." `K = 4` by default.

## 3. DPO over policy diffs

Accumulate `(preferred_diff, rejected_diff, context)` triples from past promotion decisions. Train a lightweight preference model that **ranks candidate mutations before** running them through the expensive promotion gate. v1: Opus structured prompt as the ranker; v2: classifier head once we have ≥ 1000 preference pairs. Cuts gate cost by ~3×.

## 4. Contextual bandits (Thompson sampling)

Each active policy variant maintains a `Beta(α, β)` posterior on its expected reward. `sample_group(K)` draws K variants with the highest sampled reward. Automatic exploration; no tuning required.

## Composite reward

```
r = w_v · r_verifier + w_j · r_judge - w_h · r_hack
```

Defaults: `w_v = 0.65, w_j = 0.25, w_h = 0.10` (`carl/settings.py`). Ablated in experiments.

## Why this is honest RL

- The **policy** is mathematically well-defined (the artifact set).
- The **action** is the agent's full trajectory under that policy.
- The **reward** is verifiable and CI-grounded.
- The **update** is a paired-bootstrap-gated promotion, not a gradient step — but it has the same role as a gradient step: changes that demonstrably improve return get adopted.

The contribution is the **transposition**, not a new algorithm. The transposition is non-trivial because (a) the search space is structured (artifact types), (b) the update is discrete and locality-bounded, and (c) reward hacking pressure is high since the agent can in principle modify hooks that touch the verifier.
