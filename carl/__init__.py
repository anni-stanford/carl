"""CARL — Continuous Agent Reinforcement Loop.

Cross-IDE RL framework for coding agent configuration. RLVR + GRPO group-relative
scoring + DPO over policy diffs + contextual bandits, applied to **text-space**
policy artifacts (CLAUDE.md, .claude/skills, sub-agents, hooks, MCP config,
.cursor/rules, .cursor/skills, .cursor/agents, .cursor/hooks.json).

CARL does NOT fine-tune model weights. The contribution is the *transposition*
of the 2026 post-training RL stack to text artifacts that condition coding agents.
"""

__version__ = "0.1.0"
__author__ = "Anni Zimina"
__license__ = "MIT"
