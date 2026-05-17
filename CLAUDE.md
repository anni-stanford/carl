# CARL — agent rules (dogfooded)

This file is CARL's **own** policy artifact. CARL trains itself on itself; the
final committed version of this file is the result of running CARL on the CARL
repo. Currently this is the seed.

## Hard rules

1. Never edit `carl/core/reward/*.py` from inside an episode unless the failure
   trace explicitly attributes the failure to a reward-side bug. Reward code is
   a high-pressure target for hacking; locality and reviewer scrutiny are extra
   strict here.
2. Every promoted diff must reference at least two pieces of trace evidence
   (`TraceEvidence`) in its rationale.
3. Every `pytest` invocation runs with `-q --strict-markers --cov=carl`.
4. Every shell hook must pass `shellcheck` before promotion.
5. `settings.json` mutations are limited to the tool allow/disallow lists.

## Style

- Type-annotate all public functions; `mypy --strict` must pass.
- Async by default for I/O.
- Loguru for logging; OpenTelemetry for spans.
- Keep functions ≤ 50 LOC; refactor when crossing.

## Skills the agent should consult

- `testing-policy` — when writing or modifying tests.
- `adapter-author` — when adding a new `PolicyAdapter`.
- `reward-design` — when touching `carl/core/reward/`.

## Sub-agents available

- `reviewer` — structured review of a candidate diff; output schema in
  `carl/core/diagnosis/prompts.py`.
- `attributor` — failure-trace → artifact attribution.
