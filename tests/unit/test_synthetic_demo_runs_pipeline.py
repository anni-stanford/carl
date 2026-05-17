"""End-to-end: ``run_synthetic_demo`` produces a buffer the gate can consume.

This is the full pipeline shape — write synthetic rewards → load buffer →
run paired bootstrap → assert promotion — exercising every measurement
component without any LLM call.
"""

from __future__ import annotations

from pathlib import Path

from carl.core.buffer.storage import ReplayBuffer
from carl.core.promotion.gate import evaluate_gate
from carl.settings import PromotionGateConfig
from experiments.run_synthetic_demo import main as demo_main


def test_synthetic_demo_then_gate_promotes(tmp_path: Path) -> None:
    out_path = tmp_path / "syn.sqlite"
    rc = demo_main(["--out", str(out_path), "--n-tasks", "40", "--seed", "20260517"])
    assert rc == 0
    assert out_path.is_file()

    buf = ReplayBuffer(out_path)
    cand, base, ids = buf.paired_rewards("v0.1.0+carl-synthetic", "v0.0.0+stock-synthetic")
    assert len(cand) == 40

    out = evaluate_gate(
        cand,
        base,
        PromotionGateConfig(
            n_resamples=2000,
            confidence=0.95,
            min_probe_tasks=30,
            require_ci_lower_bound_above=0.0,
        ),
        rng_seed=20260517,
    )
    # Synthetic data has ~+0.15 mean lift; gate should clear comfortably.
    assert out.promote
    assert out.mean_lift > 0.10
    assert out.ci_low > 0
    assert out.p_value < 0.01
