"""``carl.loop`` — the main async RL loop.

Wires the four-technique stack end-to-end:

  1. **Bandit** (Thompson sampling) selects K active variants for the next task.
  2. The **adapter** runs K trajectories in parallel (group rollout).
  3. **Verifier + judge + hack-probe** compose the per-trajectory reward.
  4. **GRPO scorer** computes group-relative advantages.
  5. **Diagnosis agent** attributes the worst trajectory's failure to an artifact.
  6. **Mutation proposer** generates N candidate diffs (locality-bounded).
  7. **DPO ranker** orders candidates; top-K go to the **promotion gate**.
  8. The **paired-bootstrap gate** decides which (if any) gets promoted.

Each step is replaceable: the loop only depends on the public interfaces in
the corresponding sub-modules.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from carl.adapters.base import PolicyAdapter, Task, Trajectory
from carl.core.bandit.thompson import ThompsonBandit
from carl.core.buffer.storage import ReplayBuffer, TrajectoryRow
from carl.core.diagnosis.attributor import attribute_failure
from carl.core.dpo_ranker.ranker import rank_candidates
from carl.core.grpo_scorer.advantage import group_advantages
from carl.core.llm_client import LLMClient
from carl.core.mutation.proposer import propose_candidates
from carl.core.policy.artifacts import Policy, PolicyDiff
from carl.core.promotion.gate import evaluate_gate
from carl.core.reward.composite import compose_reward
from carl.core.reward.hack_probe import detect_in_paths
from carl.core.reward.judge import JudgePromptParts, absolute_score
from carl.core.reward.types import RewardComponents
from carl.core.reward.verifier import CIArtifacts, compute_verifier
from carl.settings import CARLConfig

#: Type of the function that scores one trajectory; lets us swap the
#: composition in tests without touching the loop.
RewardFn = Callable[[Trajectory], Awaitable[RewardComponents]]


@dataclass
class LoopState:
    """Mutable state carried across iterations of the loop."""

    bandit: ThompsonBandit
    active_policies: dict[str, Policy]  # variant_id → policy
    buffer: ReplayBuffer
    iteration: int = 0


def make_default_reward_fn(
    config: CARLConfig,
    llm_client: LLMClient | None,
) -> RewardFn:
    """Build a ``RewardFn`` that wires verifier + judge (optional) + hack probe."""

    async def reward_fn(traj: Trajectory) -> RewardComponents:
        meta = traj.metadata
        artifacts = CIArtifacts(
            pytest_exit_code=traj.exit_code,
            pytest_report_json=Path(meta["pytest_report_json"]) if meta.get("pytest_report_json") else None,
            coverage_xml=Path(meta["coverage_xml"]) if meta.get("coverage_xml") else None,
            coverage_xml_baseline=Path(meta["coverage_xml_baseline"]) if meta.get("coverage_xml_baseline") else None,
            ruff_json=Path(meta["ruff_json"]) if meta.get("ruff_json") else None,
            mypy_output=Path(meta["mypy_output"]) if meta.get("mypy_output") else None,
        )
        verifier = compute_verifier(artifacts, config.verifier_weights)

        judge = None
        if llm_client is not None and config.judge.primary_model:
            judge = await absolute_score(
                JudgePromptParts(
                    task_description=traj.task.prompt,
                    candidate_output=(traj.raw_test_output or "")[-4000:],
                    rubric_items=[
                        "Does the output address the task?",
                        "Is the code correct and idiomatic?",
                        "Are the tests comprehensive?",
                    ],
                ),
                client=llm_client,
                config=config.judge,
            )

        hack = detect_in_paths(
            Path(traj.task.repo_path),
            [Path(traj.task.repo_path) / f for f in traj.files_changed],
        )

        return compose_reward(verifier, judge, hack, config.reward_weights)

    return reward_fn


async def carl_step(
    *,
    adapter: PolicyAdapter,
    task: Task,
    state: LoopState,
    config: CARLConfig,
    reward_fn: RewardFn,
    llm_client: LLMClient | None,
) -> dict[str, RewardComponents]:
    """One iteration: rollout K trajectories, score, maybe propose & promote."""
    if not state.active_policies:
        raise ValueError("loop has no active policies; register at least one")

    # 1) Bandit picks K variants.
    chosen_ids = state.bandit.sample_group(
        config.group_size, list(state.active_policies)
    )
    chosen_policies = [state.active_policies[v] for v in chosen_ids]

    # 2) Run K trajectories in parallel (group rollout).
    trajectories: list[Trajectory] = await asyncio.gather(*[
        adapter.run_episode(Path(task.repo_path), task, p, config.episode_timeout_s)
        for p in chosen_policies
    ])

    # 3) Score every trajectory; persist to the buffer.
    rewards = await asyncio.gather(*[reward_fn(t) for t in trajectories])
    for vid, traj, r in zip(chosen_ids, trajectories, rewards, strict=True):
        state.bandit.update(vid, r.r_total)
        state.buffer.append_trajectory(
            TrajectoryRow(
                id=f"{state.iteration}-{vid}-{datetime.now(tz=UTC).isoformat()}",
                adapter_name=adapter.name(),
                repo_path=str(task.repo_path),
                task_id=task.task_id,
                policy_version=vid,
                components=r,
                exit_code=traj.exit_code,
                duration_s=traj.duration_s,
            )
        )

    # 4) GRPO advantages — informational; bandit already updated.
    _ = group_advantages([r.r_total for r in rewards])

    # 5) If the worst trajectory is below threshold and we have an LLM, propose & rank.
    worst_idx = int(min(range(len(rewards)), key=lambda i: rewards[i].r_total))
    worst_reward = rewards[worst_idx].r_total
    if (
        llm_client is not None
        and worst_reward < config.diagnosis_threshold
    ):
        attribution = await attribute_failure(
            trajectories[worst_idx],
            client=llm_client,
            model=config.diagnosis_model,
        )
        if attribution is not None:
            candidates: list[PolicyDiff] = await propose_candidates(
                attribution,
                chosen_policies[worst_idx],
                client=llm_client,
                model=config.mutator_model,
                n_candidates=config.n_mutation_candidates,
                max_diff_lines=config.max_diff_lines,
            )
            ranked = await rank_candidates(
                attribution, candidates, client=llm_client, model=config.mutator_model
            )
            # Promotion gate runs over a held-out probe set; here we only persist
            # the candidates as proposals — the actual gate runs once per epoch
            # via :func:`run_promotion_gate_for_pool`.
            for c in ranked[: config.top_k_to_gate]:
                state.buffer.append_gate_decision(
                    candidate_version=f"proposal:{c.artifact_name}:{c.operation}",
                    baseline_version=chosen_ids[worst_idx],
                    promote=False,
                    mean_lift=0.0,
                    ci_low=0.0,
                    ci_high=0.0,
                    p_value=1.0,
                    n_tasks=0,
                    n_resamples=0,
                    reason="proposed; awaiting probe-set evaluation",
                )

    state.iteration += 1
    return {vid: r for vid, r in zip(chosen_ids, rewards, strict=True)}


def run_promotion_gate_for_pool(
    *,
    candidate_version: str,
    baseline_version: str,
    state: LoopState,
    config: CARLConfig,
    rng_seed: int | None = None,
) -> bool:
    """Run the paired-bootstrap promotion gate on data already in the buffer.

    Returns ``True`` iff the gate promotes. The gate's full decision is
    persisted to the ``gate_decisions`` table.
    """
    cand, base, _task_ids = state.buffer.paired_rewards(
        candidate_version, baseline_version
    )
    result = evaluate_gate(cand, base, config.promotion_gate, rng_seed=rng_seed)
    state.buffer.append_gate_decision(
        candidate_version=candidate_version,
        baseline_version=baseline_version,
        promote=result.promote,
        mean_lift=result.mean_lift,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        p_value=result.p_value,
        n_tasks=result.n_tasks,
        n_resamples=result.n_resamples,
        reason=result.reason,
    )
    return result.promote


# ----- Public top-level entry point ------------------------------------------


async def carl_loop(
    adapters: Sequence[PolicyAdapter],
    config: CARLConfig,
    *,
    initial_policies: dict[str, Policy],
    tasks: Sequence[Task],
    buffer: ReplayBuffer,
    llm_client: LLMClient | None = None,
    reward_fn: RewardFn | None = None,
) -> LoopState:
    """Drive the full CARL loop over ``tasks`` and return final :class:`LoopState`."""
    state = LoopState(
        bandit=ThompsonBandit(),
        active_policies=dict(initial_policies),
        buffer=buffer,
    )
    for vid in initial_policies:
        state.bandit.register(vid)

    rfn = reward_fn or make_default_reward_fn(config, llm_client)
    adapter_by_name = {a.name(): a for a in adapters}

    for task in tasks:
        adapter = adapter_by_name.get(task.adapter_name)
        if adapter is None:
            continue
        await carl_step(
            adapter=adapter,
            task=task,
            state=state,
            config=config,
            reward_fn=rfn,
            llm_client=llm_client,
        )
    return state
