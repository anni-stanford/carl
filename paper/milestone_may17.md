# CS153 Project Milestone — Week 7 (due Sunday May 17, 2026, 11:59 PM)

> Submission for the Project Milestone form.

## Q1 — Project Title (0.5 pts)

**CARL: Continuous Agent Reinforcement Loop** — a cross-IDE open-source RL framework that continuously reinforces a coding agent's text-space policy (`CLAUDE.md`, `.claude/skills`, sub-agents, hooks, settings, `.cursor/rules`) using reward extracted from CI/CD outcomes on real repositories. Day-1 support for Claude Code and Cursor; `PolicyAdapter` pattern for future agents.

## Q2 — Project Track (0.5 pts)

**Research.**

## Q3 — Progress (1.5 pts)

We have scaffolded the open-source repository (`github.com/anni-stanford/carl`, MIT, Python 3.11 + TypeScript bridge), defined the IDE-agnostic `Policy` / `Artifact` / `PolicyDiff` data model, and implemented `PolicyAdapter` ABC plus working `read_policy` / `write_policy` for both the Claude Code adapter (`CLAUDE.md`, `.claude/skills`, `.claude/agents`, `.claude/hooks`, `.claude/settings.json`) and the Cursor adapter (`.cursor/rules`, `.cursor/skills`, `.cursor/agents`, `.cursor/hooks.json`, `.cursor/mcp.json`) with round-trip unit tests. The deterministic RLVR verifier (`carl/core/reward/verifier.py` — pytest, coverage, ruff, mypy parsing with weight renormalization for missing signals), the composite reward function with judge-gate anti-hacking property, the SQLite replay buffer (`carl/core/buffer/storage.py`), and the paired-bootstrap promotion gate (`carl/core/promotion/gate.py` — `scipy.stats.bootstrap` BCa, 10 000 resamples, `n ≥ 30` minimum, CI-lower-bound > 0 criterion) are implemented and unit-tested end-to-end on synthetic CI artifacts. `pytest` 12/12 pass, `ruff` clean, `mypy --strict` clean. End-to-end episode execution, the LLM-judge with bias controls, the diagnosis-mutation loop, and the contextual bandit are the next milestones.

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
