"""End-to-end: synthetic CI artifacts → verifier → composite → buffer → gate.

This is the "is the measurement pipeline really wired?" smoke test. It runs
without any LLM call, without Docker, without an actual coding agent. It
proves that if real episodes were dropping artifact files into a directory,
CARL would correctly compute rewards and then run the paired-bootstrap gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carl.core.buffer.storage import ReplayBuffer, TrajectoryRow
from carl.core.promotion.gate import evaluate_gate
from carl.core.reward.composite import compose_reward
from carl.core.reward.verifier import CIArtifacts, compute_verifier
from carl.settings import PromotionGateConfig, RewardWeights, VerifierWeights


def _fake_pytest(path: Path, total: int, passed: int) -> Path:
    path.write_text(
        json.dumps({"summary": {"total": total, "passed": passed, "failed": total - passed}}),
        encoding="utf-8",
    )
    return path


def _fake_coverage(path: Path, line_rate: float) -> Path:
    path.write_text(f'<coverage line-rate="{line_rate:.3f}" />', encoding="utf-8")
    return path


def _fake_ruff(path: Path, n_diags: int) -> Path:
    path.write_text(json.dumps([{"code": "E501"}] * n_diags), encoding="utf-8")
    return path


def _episode(tmp: Path, *, total: int, passed: int, cov: float, ruff_n: int) -> CIArtifacts:
    tmp.mkdir(parents=True, exist_ok=True)
    return CIArtifacts(
        pytest_exit_code=0 if passed == total else 1,
        pytest_report_json=_fake_pytest(tmp / "pytest.json", total, passed),
        coverage_xml=_fake_coverage(tmp / "cov.xml", cov),
        coverage_xml_baseline=_fake_coverage(tmp / "cov_base.xml", cov - 0.02),
        ruff_json=_fake_ruff(tmp / "ruff.json", ruff_n),
        mypy_output=None,
    )


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_pipeline_promotes_when_candidate_dominates(tmp_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(tmp_path / "buf.sqlite")

    n = 40  # ≥ 30, satisfies min_probe_tasks
    for i in range(n):
        # Baseline: ~70 % of tests pass, no coverage gain, lots of lint warnings
        base_total, base_passed = 20, int(rng.integers(12, 16))
        base_art = _episode(
            tmp_path / f"b{i}",
            total=base_total,
            passed=base_passed,
            cov=0.55,
            ruff_n=int(rng.integers(8, 12)),
        )
        base_v = compute_verifier(base_art, VerifierWeights())
        base_r = compose_reward(base_v, None, None, RewardWeights())

        # Candidate: ~95 % of tests pass, +2 pp coverage, almost lint-clean
        cand_total, cand_passed = 20, int(rng.integers(18, 21))
        cand_art = _episode(
            tmp_path / f"c{i}",
            total=cand_total,
            passed=cand_passed,
            cov=0.57,
            ruff_n=int(rng.integers(0, 2)),
        )
        cand_v = compute_verifier(cand_art, VerifierWeights())
        cand_r = compose_reward(cand_v, None, None, RewardWeights())

        buf.append_trajectory(
            TrajectoryRow(
                id=f"b-{i}",
                adapter_name="claude_code",
                repo_path="/x",
                task_id=f"task_{i}",
                policy_version="v0.0.0+stock",
                components=base_r,
                exit_code=0,
                duration_s=1.0,
            )
        )
        buf.append_trajectory(
            TrajectoryRow(
                id=f"c-{i}",
                adapter_name="claude_code",
                repo_path="/x",
                task_id=f"task_{i}",
                policy_version="v0.1.0+carl",
                components=cand_r,
                exit_code=0,
                duration_s=1.0,
            )
        )

    cand_seq, base_seq, _ids = buf.paired_rewards("v0.1.0+carl", "v0.0.0+stock")
    assert len(cand_seq) == n

    cfg = PromotionGateConfig(
        n_resamples=2000, confidence=0.95, min_probe_tasks=30, require_ci_lower_bound_above=0.0
    )
    out = evaluate_gate(cand_seq, base_seq, cfg, rng_seed=seed)
    # The synthetic gap is large and consistent — the gate should clear.
    assert out.promote, f"gate did not promote: {out}"
    assert out.mean_lift > 0.05
    assert out.ci_low > 0.0
    assert out.p_value < 0.05
    assert out.n_tasks == n
