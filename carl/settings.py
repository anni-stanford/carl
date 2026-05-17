"""CARLConfig — single pydantic settings model controlling every CARL knob.

Defaults come from the project specification. Override via environment variables
(prefixed ``CARL_``) or a YAML file passed to ``carl run --config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class VerifierWeights(BaseModel):
    """Weights for components of the deterministic CI-based verifier reward."""

    tests_passed: float = Field(default=0.55, ge=0.0)
    coverage_delta: float = Field(default=0.20, ge=0.0)
    lint_clean: float = Field(default=0.10, ge=0.0)
    typecheck_clean: float = Field(default=0.10, ge=0.0)
    security_clean: float = Field(default=0.05, ge=0.0)


class RewardWeights(BaseModel):
    """Top-level composite reward weights: r = w_v r_verifier + w_j r_judge - w_h r_hack."""

    verifier: float = Field(default=0.65, ge=0.0)
    judge: float = Field(default=0.25, ge=0.0)
    hack_penalty: float = Field(default=0.10, ge=0.0)


class JudgeConfig(BaseModel):
    """LLM-judge bias controls."""

    primary_model: str = "claude-opus-4-7"
    rotation_models: list[str] = Field(
        default_factory=lambda: ["gpt-5.5", "composer-2", "claude-sonnet-4-6"]
    )
    position_flip: bool = True
    consistent_only: bool = True
    rubric_shuffle: bool = True


class PromotionGateConfig(BaseModel):
    """Paired-bootstrap promotion gate parameters."""

    n_resamples: int = 10_000
    confidence: float = 0.95
    min_probe_tasks: int = 30  # n ≥ 30 — conventional threshold for credible bootstrap CIs
    require_ci_lower_bound_above: float = 0.0


class CARLConfig(BaseModel):
    """Top-level CARL configuration."""

    parallelism: int = 4
    group_size: int = Field(default=4, description="K in GRPO-style group rollouts")
    max_diff_lines: int = 5
    diagnosis_threshold: float = 0.5
    n_mutation_candidates: int = 5
    top_k_to_gate: int = 2
    dpo_retrain_interval: int = 1000

    mutator_model: str = "claude-opus-4-7"
    diagnosis_model: str = "claude-opus-4-7"
    cursor_default_model: str = "composer-2"

    use_cloud_episodes: bool = False
    episode_timeout_s: int = 1800

    verifier_weights: VerifierWeights = Field(default_factory=VerifierWeights)
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    promotion_gate: PromotionGateConfig = Field(default_factory=PromotionGateConfig)

    held_out_probes: Path = Path("tasks/probes/held_out.yaml")
    repo_registry: Path = Path("repos/")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def stop_signal(self) -> bool:
        """Override in tests / CLI to terminate the loop cleanly."""
        return False
