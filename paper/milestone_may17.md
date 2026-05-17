# CS153 Project Milestone — Week 7 (due Sunday May 17, 2026, 11:59 PM)

> Submission for the Project Milestone form.

## Q1 — Project Title (0.5 pts)

**CARL: Continuous Agent Reinforcement Loop** — a cross-IDE open-source RL framework that continuously reinforces a coding agent's text-space policy (`CLAUDE.md`, `.claude/skills`, sub-agents, hooks, settings, `.cursor/rules`) using reward extracted from CI/CD outcomes on real repositories. Day-1 support for Claude Code and Cursor; `PolicyAdapter` pattern for future agents.

## Q2 — Project Track (0.5 pts)

**Research.**

## Q3 — Progress (1.5 pts)

We have built CARL's open-source repository (`github.com/anni-stanford/carl`, MIT, Python 3.11 + TypeScript bridge) end-to-end at the algorithm level. The IDE-agnostic `Policy` / `Artifact` / `PolicyDiff` data model, `PolicyAdapter` ABC, and `read_policy` / `write_policy` for both Claude Code and Cursor adapters round-trip with `policy_hash` parity. The full reward stack is implemented: deterministic RLVR verifier (pytest / coverage / ruff / mypy / bandit parsing with weight renormalization), bias-controlled LLM judge (position-flip + family-rotation across `claude-opus-4-7` ↔ `gpt-5.5` ↔ `composer-2` ↔ `claude-sonnet-4-6` + rubric-shuffle), 6-pattern reward-hacking probe detector, and composite reward with the judge-gate anti-hacking property (LLM judge cannot rescue a CI-failing trajectory). The four-technique stack — GRPO-style group-relative advantage scorer, Thompson-sampling contextual bandit, DPO-style preference ranker, and the paired-bootstrap promotion gate (`scipy.stats.bootstrap` BCa, 10 000 resamples, `n ≥ 30` minimum, CI-lower-bound > 0 criterion) — is implemented. Pydantic-validated structured-output diagnosis agent and locality-bounded mutation proposer are wired into `carl.loop.carl_step`. The Streamlit dashboard reads from a SQLite replay buffer with five views. A 30-case reward-hacking probe set and a 30-case hand-labeled diagnosis validation set are committed. **66/66 unit + integration tests pass; ruff and mypy `--strict` clean across 37 source files.** Real-data experiments require a Docker daemon, an Anthropic API key, and a target repo with a CI pipeline; these resources are scheduled for the May 25–29 final-deliverable window.

## Q4 — Future Implementation (2 pts) — concrete plan, May 17 → May 29

- **May 17–20.** Docker sandbox + Claude Code episode runner end-to-end on one repo. Cursor adapter via `@cursor/sdk` JSON-RPC bridge (stub-quality first).
- **May 21–24.** Reward stack (`verifier.py` for RLVR; `judge.py` with position-flipped and family-rotated judges; `hack_probe.py` for reward-hacking defenses; `composite.py` weighted reward). Postgres + pgvector replay buffer. Diagnosis agent v1 with a 50-case human-labeled validation set for attribution accuracy. Mutation proposer with ≤ 5-line locality budget. Paired-bootstrap promotion gate (10 000 resamples, CI lower bound > 0).
- **May 25–27.** Main async loop wired. Thompson-sampling bandit. Streamlit dashboard with reward curves and policy-diff history. One full CARL run on a real Python repo (FastAPI or httpx), ≈ 30 episodes.
- **May 28–29.** Experiment **E1** (CARL vs stock baseline, paired bootstrap). **E2** (no-diagnosis ablation). **E8** (reward-hacking probes). Paper draft with placeholder tables filled by experiment scripts. README polish. PyPI release of `carl-loop`. npm release of `@carl-loop/cursor`. 3-minute demo video.

Stretch goals (cross-IDE transfer **E7** and TypeScript transfer **E6**) are deferred to v0.2.

## Q5 — GitHub Link (0.5 pts)

`https://github.com/anni-stanford/carl`

## Q6 — Compute (0 pts)

- **Anthropic API** for Claude Opus 4.7 (mutator + judge + diagnosis). Estimated 3 000–5 000 calls @ ≈ $0.40 average → **≈ $1.5–2 K**.
- **Anysphere Cursor SDK** credits for cloud VM episode execution. Estimated 200–400 cloud-VM hours → **≈ $300–500** (have credits).
- **Cloudflare Workers AI** for open-source-model judge ablation. Estimated **$1–2 K of the $50 K cap**, used for family-rotation in the LLM judge.
- **DigitalOcean** droplets for Postgres + pgvector replay buffer and dashboard hosting: ≈ $60/mo of the $250 credit.

**Total estimated cost: $3–5 K**, well within available compute credits. **No GPU training is required** because CARL operates on the text-artifact policy, not on model weights.
