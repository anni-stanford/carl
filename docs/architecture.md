# CARL — Architecture

CARL is split into an agent-agnostic core and a Claude-Code-specific adapter.
The ABC pattern is preserved so future agents (Codex, Aider, …) can plug into
the same RL machinery without changing the loop.

```mermaid
flowchart TB
    subgraph CORE[CARL CORE]
      direction TB
      R1[reward<br/>RLVR · LLM judge · hack probes]
      R2[grpo_scorer<br/>group-relative advantage]
      R3[dpo_ranker<br/>preference model over diffs]
      R4[bandit<br/>Thompson sampling]
      R5[diagnosis · mutation · promotion · buffer]
    end
    CORE --> CC_AD[PolicyAdapter — Claude Code]
    CC_AD --> CC[CLAUDE.md / .claude/skills / .claude/agents / .claude/hooks / settings.json / .claude/commands]
```

## Artifact mapping

| Semantic role | Disk path |
|---|---|
| Project rules / context | `CLAUDE.md` |
| Skills | `.claude/skills/*/SKILL.md` |
| Sub-agents | `.claude/agents/*.md` |
| Hooks | `.claude/hooks/*.sh` |
| MCP / settings config | `.claude/settings.json` |
| Slash commands | `.claude/commands/*.md` |

## Data classes

- `Artifact(name, type, content, metadata)` — single editable text artifact.
- `Policy(artifacts, version, parent_version, …)` — versioned snapshot with `policy_hash`.
- `PolicyDiff(operation, line_range, old_content, new_content, …)` — proposed mutation, ≤ `max_diff_lines`.
- `Task(task_id, repo_path, prompt, adapter_name, …)` — unit of work.
- `Trajectory(task, policy, events, files_changed, exit_code, …)` — full episode record.

## PolicyAdapter contract

```python
class PolicyAdapter(ABC):
    async def read_policy(self, repo_path: Path) -> Policy: ...
    async def write_policy(self, repo_path: Path, policy: Policy) -> None: ...
    async def run_episode(self, repo_path: Path, task: Task,
                          policy: Policy, timeout_s: int) -> Trajectory: ...
    def list_artifact_types(self) -> list[ArtifactType]: ...
    def name(self) -> str: ...
```

The Claude Code adapter round-trips `read_policy(write_policy(P)) == P` (`policy_hash` parity, see `tests/unit/test_claude_code_adapter_round_trip.py`).
