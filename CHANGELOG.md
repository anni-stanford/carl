# Changelog

All notable changes to CARL will be documented in this file. Conventional Commits style.

## [Unreleased]

### Added

- Repo skeleton: `pyproject.toml`, CI workflow, Docker base, `.gitignore`, `.env.example`.
- `carl/` Python package: `core/` (reward, grpo_scorer, dpo_ranker, bandit, diagnosis, mutation, promotion, buffer, greso, async_runner), `adapters/`, `env/`, `tasks/`, `report.py`, `auto.py`.
- Agent-agnostic `Policy` / `Artifact` / `PolicyDiff` data classes (`carl/core/policy/artifacts.py`) and `PolicyAdapter` ABC (`carl/adapters/base.py`).
- Claude Code adapter with `read_policy`, `write_policy`, and Docker-sandboxed `run_episode` end-to-end (`carl/adapters/claude_code.py`).
- Deterministic RLVR verifier; bias-controlled LLM judge (position-flip + family rotation + rubric-shuffle); composite reward with judge-gate anti-hacking; six-pattern reward-hacking probe detector.
- GRPO group-relative advantage scorer; Thompson-sampling contextual bandit; Pydantic-validated structured-output diagnosis agent; locality-bounded mutation proposer; DPO-style preference ranker; paired-bootstrap promotion gate (BCa, 10 000 resamples, n ≥ 30, CI lower bound > 0).
- `carl auto` orchestrator: pre-flight + benchmark BEFORE + training (with `apply_diff` actually mutating `CLAUDE.md`) + benchmark AFTER + paired-bootstrap gate + `CARL_REPORT.md` generation.
- 30-case hand-labeled diagnosis-attribution validation set (`tasks/diagnosis_labels.yaml`) and 30-case reward-hacking probe set (`tasks/probes/reward_hacking.yaml`).
- Streamlit dashboard with four views: overview, reward curve, gate-decision history, episode replay.
- `scripts/quickstart.sh` — single curl-pipe-able installer.
- `python -m carl` invocation so the CLI works regardless of `PATH`.
- Documentation: `docs/architecture.md`, `docs/rl_stack.md`, `docs/reward_design.md`, `docs/promotion_gate.md`, `docs/reward_hacking.md`, `docs/measurement.md`, `docs/reproducing_results.md`.
- Self-dogfooded `CLAUDE.md` and `.claude/` (skills, agent, hook, command, settings).

### Removed

- Cursor adapter (`carl/adapters/cursor.py`) and the JS bridge (`js/cursor-bridge/`). CARL is now focused exclusively on Claude Code; the `PolicyAdapter` ABC remains agent-agnostic so future agents (Codex, Aider) can plug in without core changes.
