"""CARL — Continuous Agent Reinforcement Loop.

RL framework that improves Claude Code's text-space policy artifacts
(CLAUDE.md, .claude/skills, .claude/agents, .claude/hooks, .claude/settings.json)
using reward signal extracted from CI/CD outcomes on real repositories.
RLVR + GRPO group-relative scoring + DPO over policy diffs + contextual bandits.

CARL does NOT fine-tune model weights. The contribution is the *transposition*
of the 2026 post-training RL stack to text artifacts that condition Claude Code.
"""

__version__ = "0.1.0"
__author__ = "Anni Zimina"
__license__ = "MIT"
