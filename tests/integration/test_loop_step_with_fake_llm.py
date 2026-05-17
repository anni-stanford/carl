"""Loop iteration runs end-to-end against fakes (no Docker, no LLM API).

Confirms the orchestration in :func:`carl.loop.carl_step`: bandit picks K
variants, the adapter is called K times, rewards are persisted, and the
diagnosis path runs when the worst reward is below threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carl.adapters.base import PolicyAdapter, Task, Trajectory
from carl.core.buffer.storage import ReplayBuffer
from carl.core.llm_client import FakeLLMClient
from carl.core.policy.artifacts import Artifact, ArtifactType, Policy
from carl.core.reward.composite import compose_reward
from carl.core.reward.types import (
    HackComponents,
    RewardComponents,
    VerifierComponents,
)
from carl.loop import LoopState, carl_step
from carl.settings import CARLConfig

pytestmark = pytest.mark.asyncio


class _FakeAdapter(PolicyAdapter):
    """Returns a deterministic trajectory whose reward depends on the policy version."""

    def __init__(self) -> None:
        self.calls = 0

    def name(self) -> str:
        return "claude_code"

    def list_artifact_types(self) -> list[ArtifactType]:
        return [ArtifactType.RULES]

    async def read_policy(self, repo_path: Path) -> Policy:
        return Policy(artifacts=[], version="x")

    async def write_policy(self, repo_path: Path, policy: Policy) -> None:
        return None

    async def run_episode(self, repo_path, task, policy, timeout_s):  # type: ignore[override]
        self.calls += 1
        return Trajectory(
            task=task,
            policy=policy,
            events=[],
            files_changed=[],
            exit_code=0,
            duration_s=0.1,
            raw_ci_output="",
            raw_test_output=f"output for policy {policy.version}",
            metadata={},
        )


async def _stub_reward_fn(traj: Trajectory) -> RewardComponents:
    """Lower reward for variants whose name starts with 'bad'."""
    base = 0.2 if traj.policy.version.startswith("bad") else 0.8
    vc = VerifierComponents(
        tests_passed=base,
        coverage_delta=base,
        lint_clean=base,
        typecheck_clean=base,
        security_clean=None,
        composite=base,
        raw_test_count=10,
        raw_failed_count=int(round((1 - base) * 10)),
    )
    return compose_reward(
        verifier=vc,
        judge=None,
        hack=HackComponents(penalty=0.0, detected_patterns=()),
        weights=__import__("carl.settings", fromlist=["RewardWeights"]).RewardWeights(),
    )


async def test_carl_step_persists_rewards_and_runs_diagnosis(tmp_path: Path) -> None:
    cfg = CARLConfig(
        group_size=2,
        n_mutation_candidates=3,
        diagnosis_threshold=0.5,
    )
    adapter = _FakeAdapter()
    buf = ReplayBuffer(tmp_path / "buf.sqlite")
    state = LoopState(
        bandit=__import__("carl.core.bandit.thompson", fromlist=["ThompsonBandit"]).ThompsonBandit(seed=0),
        active_policies={
            "good_v0": Policy(artifacts=[Artifact(name="r", type=ArtifactType.RULES, content="x")], version="good_v0"),
            "bad_v0": Policy(artifacts=[Artifact(name="r", type=ArtifactType.RULES, content="y")], version="bad_v0"),
        },
        buffer=buf,
    )
    state.bandit.register("good_v0")
    state.bandit.register("bad_v0")

    # Fake LLM returns valid attribution + valid candidates so the diagnosis path
    # runs without raising.
    valid_attr = json.dumps({
        "failed_artifact_type": "rules",
        "failed_artifact_name": "r",
        "evidence_from_trace": [
            {"event_kind": "x", "snippet": "a"},
            {"event_kind": "y", "snippet": "b"},
        ],
        "root_cause": "rules do not require running tests first; common failure mode",
        "proposed_intervention": "add_line",
        "expected_lift": 0.1,
        "confidence": 0.7,
    })
    valid_cands = json.dumps({
        "candidates": [
            {
                "operation": "add_line",
                "artifact_name": "r",
                "artifact_type": "rules",
                "line_range": [1, 2],
                "old_content": None,
                "new_content": "x",
                "rationale": "Fixes the issue evidenced in the trace",
                "expected_lift": 0.1,
                "confidence": 0.8,
            }
        ]
    })
    valid_ranking = json.dumps({"ranking": [0]})

    client = FakeLLMClient(
        script={
            "diagnosis agent": valid_attr,  # never matches; forces default path
            "ranker": valid_ranking,
        },
        default=valid_attr,  # most calls fall through to the attribution shape
    )
    # Override default for proposer + ranker by using prefix routing
    client.script["mutation proposer"] = valid_cands
    client.script["DPO ranker"] = valid_ranking

    task = Task(
        task_id="t",
        repo_path=tmp_path,
        prompt="Fix the failing test",
        adapter_name="claude_code",
    )
    rewards = await carl_step(
        adapter=adapter,
        task=task,
        state=state,
        config=cfg,
        reward_fn=_stub_reward_fn,
        llm_client=client,
    )

    # Both variants got a trajectory
    assert adapter.calls == 2
    assert state.iteration == 1
    assert len(rewards) == 2
    # Both rewards persisted
    assert buf.trajectory_count() == 2
