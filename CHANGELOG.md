# Changelog

All notable changes to CARL will be documented in this file. Conventional Commits style.

## [Unreleased]

### Added

- Repo skeleton: `pyproject.toml`, CI workflow, Docker base, `.gitignore`, `.env.example`.
- `carl/` Python package skeleton with `core/` (reward, grpo_scorer, dpo_ranker, bandit, diagnosis, mutation, promotion, buffer, greso, async_runner), `adapters/`, `env/`.
- `Policy` and `Artifact` IDE-agnostic data classes (`carl/core/policy/artifacts.py`).
- `PolicyAdapter` ABC (`carl/adapters/base.py`).
- Claude Code adapter with `read_policy` + `write_policy` round-trip and unit tests (`carl/adapters/claude_code.py`).
- Cursor adapter stub + `js/cursor-bridge/` TypeScript package skeleton.
- `docs/architecture.md`, `docs/rl_stack.md`, `docs/multi_ide.md`, `docs/reward_design.md`, `docs/promotion_gate.md`, `docs/reward_hacking.md`.
- `paper/draft.md` (placeholders), `paper/demo_script.md`, `paper/milestone_may17.md`.
- Self-dogfooded `CLAUDE.md` and `.cursor/rules`.
