# CARL — Reward hacking, honestly

CARL's reward signal is grounded in CI, but the agent can in principle modify artifacts that influence CI behavior (hooks, settings, skills that suggest bypasses). We explicitly assume reward-hacking pressure is high.

## Defenses (implemented or specified)

1. **Adversarial probe set.** A held-out set of tasks where ground truth is known. We measure the agent's hack-rate (e.g., does it delete failing tests under pressure?) on these probes after every promotion.
2. **Hook shellcheck gate.** Any hook diff that fails `shellcheck` is rejected before promotion.
3. **Settings allowlist.** `settings.json` mutations are restricted to the tool allowlist/disallow-list; schema changes are rejected.
4. **Coverage anti-gaming.** Coverage delta is computed against a frozen test set when measuring `r_verifier` for the promotion gate.
5. **CI gating of judge.** `r_judge` is zeroed when `r_verifier < τ`, so the agent can't talk its way past failing tests.
6. **Hack-penalty term.** `r_total -= w_h · r_hack`. `r_hack` is computed by a separate Opus call with structured output enumerating detected exploit patterns.

## Defenses we don't claim

- We do not eliminate reward hacking; we measure and penalize it.
- We do not guarantee that adversarial probes cover all exploit modes; new ones will be discovered and added.
- We do not protect against Sybil-style judges (i.e., the same model family being used as both mutator and judge); family rotation only mitigates this.

## Evaluation

Experiment **E8** in `experiments/` runs CARL with and without `r_hack` term and measures the hack-rate gap on the probe set. The headline number for the paper is the **delta** in hack-rate, reported with confidence intervals.
