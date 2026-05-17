# CARL: Cross-IDE Reinforcement of Coding Agent Configuration via CI/CD-Signaled Updates over Text-Space Policy Artifacts

**Anni Zimina** — Stanford CS153 Spring 2026

> **Numbers in this draft are placeholders.** Experiment scripts in `experiments/` overwrite tables and figures. No fabricated results.

## Abstract (≈ 150 words, placeholder)

Coding agents like Claude Code and Cursor are conditioned by a growing surface of editable text artifacts: project rules (`CLAUDE.md`, `.cursor/rules`), skills, sub-agents, hooks, and MCP configuration. Practitioners hand-tune these artifacts; we propose **CARL**, an open-source RL framework that does so automatically. CARL transposes the canonical 2026 post-training RL stack — RLVR, GRPO group-relative scoring, DPO over pairwise preferences, contextual bandits — to **text-space policy parameters**, gated by paired-bootstrap CI lower bounds on a held-out probe set. We instantiate CARL on Claude Code and Cursor via a `PolicyAdapter` pattern, run it on real Python and TypeScript repositories, and report `{{HEADLINE_LIFT}}` mean reward lift over stock-agent baselines on a SWE-Bench Verified subset (`{{N}}` tasks; CI `{{CI}}`; _p_ < `{{P}}`).

## 1 Introduction

(See `docs/rl_stack.md` and `docs/architecture.md` for the formal framing; this section condenses to ≤ 1.5 pages.)

## 2 Related work

- SWE-Bench / SWE-Bench Verified.
- Reflexion (Shinn et al., 2023) — intra-episode self-edit, no policy persistence.
- STaR (Zelikman et al., 2022).
- Robeyns, Szummer, Aitchison (2025) "A self-improving coding agent" — orthogonal to CARL: intra-system self-editing vs. cross-IDE configuration RL.
- DSPy (Khattab et al.).
- GRPO (Shao et al., DeepSeekMath).
- DPO (Rafailov et al.).
- RLVR / RLAIF.
- Contextual bandits for LLM routing.
- 2026 LLM-judge bias literature (FairJudge et al.).
- Cursor SDK (Anysphere, April 2026).
- Claude Agent SDK (Anthropic).

## 3 Method

### 3.1 RL formulation

(See `docs/rl_stack.md`.)

### 3.2 Adapter pattern

(See `docs/architecture.md` and `docs/multi_ide.md`.)

### 3.3 Promotion gate

(See `docs/promotion_gate.md`.)

### 3.4 Reward hacking defenses

(See `docs/reward_hacking.md`.)

## 4 Experiments

| ID | Name | Status |
|---|---|---|
| E1 | Baseline vs CARL — main result | scripts ready, runs Day 13–14 |
| E2 | Per-technique ablation | scripts ready |
| E3 | Per-artifact ablation | scripts ready |
| E4 | Diagnosis quality (50-case human-labeled set) | labels in progress |
| E5 | Promotion-gate strictness (CI-LB > 0 vs naive) | scripts ready |
| E6 | Cross-language transfer (Python → TS) | deferred to v0.2 |
| E7 | Cross-IDE transfer (Claude Code → Cursor) | deferred to v0.2 |
| E8 | Reward-hacking probes | scripts ready |
| E9 | SWE-Bench Verified | runs Day 14 |

## 5 Results

| Adapter | Stock | CARL | Lift | 95 % CI | _p_ |
|---|---|---|---|---|---|
| Claude Code | `{{CC_BASE}}` | `{{CC_CARL}}` | `{{CC_LIFT}}` | `{{CC_CI}}` | `{{CC_P}}` |
| Cursor | `{{CU_BASE}}` | `{{CU_CARL}}` | `{{CU_LIFT}}` | `{{CU_CI}}` | `{{CU_P}}` |

Cross-IDE transfer (E7, deferred): `{{TRANSFER}}`.

## 6 Limitations

- Closed-weight model dependency (mutator + judge).
- Reward-hacking residual (mitigated, not eliminated).
- Episode cost.
- Generalization beyond Python / TS preliminary.
- LLM-judge bias residual after position-flip + family-rotation.

## 7 Conclusion and future work

Adapters for Codex and Aider; hierarchical artifact optimization (sub-agent specs that compose); transfer to non-IDE agents.

## 8 References

(BibTeX in `bibliography.bib`.)
