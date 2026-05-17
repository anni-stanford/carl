"""Verifier reward parses real CI artifacts and renormalizes when signals are missing."""

from __future__ import annotations

import json
from pathlib import Path

from carl.core.reward.verifier import CIArtifacts, compute_verifier
from carl.settings import VerifierWeights


def _write(p: Path, s: str) -> Path:
    p.write_text(s, encoding="utf-8")
    return p


def test_all_signals_present_perfect_run(tmp_path: Path) -> None:
    pytest_report = _write(
        tmp_path / "pytest.json",
        json.dumps({"summary": {"total": 50, "passed": 50, "failed": 0, "error": 0}}),
    )
    coverage = _write(tmp_path / "coverage.xml", '<coverage line-rate="0.92" />')
    coverage_base = _write(tmp_path / "coverage_base.xml", '<coverage line-rate="0.90" />')
    ruff = _write(tmp_path / "ruff.json", "[]")
    mypy = _write(tmp_path / "mypy.txt", "Success: no issues found in 23 source files\n")

    art = CIArtifacts(
        pytest_exit_code=0,
        pytest_report_json=pytest_report,
        coverage_xml=coverage,
        coverage_xml_baseline=coverage_base,
        ruff_json=ruff,
        mypy_output=mypy,
    )
    out = compute_verifier(art, VerifierWeights())
    assert out.tests_passed == 1.0
    assert out.coverage_delta > 0.5  # +2 pp coverage maps via [-10,+10]→[0,1]
    assert out.lint_clean == 1.0
    assert out.typecheck_clean == 1.0
    # Composite is the weighted mean over tests/coverage/lint/typecheck (security
    # absent → renormalized). Coverage's 0.60 score pulls the composite down.
    assert 0.85 <= out.composite <= 1.0
    assert out.raw_test_count == 50
    assert out.raw_failed_count == 0


def test_failing_tests_dominates(tmp_path: Path) -> None:
    pytest_report = _write(
        tmp_path / "pytest.json",
        json.dumps({"summary": {"total": 50, "passed": 10, "failed": 40, "error": 0}}),
    )
    art = CIArtifacts(
        pytest_exit_code=1,
        pytest_report_json=pytest_report,
        coverage_xml=None,
        coverage_xml_baseline=None,
        ruff_json=None,
        mypy_output=None,
    )
    out = compute_verifier(art, VerifierWeights())
    assert out.tests_passed == 0.2  # 10/50
    assert out.composite == out.tests_passed  # only signal present, weight renormalizes


def test_missing_signals_renormalize_weights(tmp_path: Path) -> None:
    """If only tests + lint are reported, those weights renormalize to 100 %."""
    pytest_report = _write(
        tmp_path / "pytest.json",
        json.dumps({"summary": {"total": 10, "passed": 8, "failed": 2}}),
    )
    ruff = _write(tmp_path / "ruff.json", "[]")
    art = CIArtifacts(
        pytest_exit_code=1,
        pytest_report_json=pytest_report,
        coverage_xml=None,
        coverage_xml_baseline=None,
        ruff_json=ruff,
        mypy_output=None,
    )
    w = VerifierWeights()
    out = compute_verifier(art, w)
    expected = (w.tests_passed * 0.8 + w.lint_clean * 1.0) / (w.tests_passed + w.lint_clean)
    assert abs(out.composite - expected) < 1e-9


def test_pytest_crash_uses_exit_code_when_no_report(tmp_path: Path) -> None:
    art = CIArtifacts(
        pytest_exit_code=2,
        pytest_report_json=None,
        coverage_xml=None,
        coverage_xml_baseline=None,
        ruff_json=None,
        mypy_output=None,
    )
    out = compute_verifier(art, VerifierWeights())
    assert out.tests_passed == 0.0
    assert out.composite == 0.0


def test_lint_clean_decays_with_diagnostic_count(tmp_path: Path) -> None:
    ruff_zero = _write(tmp_path / "r0.json", "[]")
    ruff_three = _write(
        tmp_path / "r3.json",
        json.dumps([{"code": "F401"}, {"code": "F401"}, {"code": "E501"}]),
    )
    base = CIArtifacts(
        pytest_exit_code=0,
        pytest_report_json=None,
        coverage_xml=None,
        coverage_xml_baseline=None,
        ruff_json=None,
        mypy_output=None,
    )
    out_zero = compute_verifier(
        CIArtifacts(**{**base.__dict__, "ruff_json": ruff_zero}), VerifierWeights()
    )
    out_three = compute_verifier(
        CIArtifacts(**{**base.__dict__, "ruff_json": ruff_three}), VerifierWeights()
    )
    assert out_zero.lint_clean == 1.0
    assert out_three.lint_clean == 0.25  # 1 / (1 + 3)


def test_coverage_delta_clamps_to_unit_interval(tmp_path: Path) -> None:
    # 50 pp jump: way above the +/- 10 pp clip range
    new = _write(tmp_path / "n.xml", '<coverage line-rate="0.95" />')
    old = _write(tmp_path / "o.xml", '<coverage line-rate="0.45" />')
    art = CIArtifacts(
        pytest_exit_code=0,
        pytest_report_json=None,
        coverage_xml=new,
        coverage_xml_baseline=old,
        ruff_json=None,
        mypy_output=None,
    )
    out = compute_verifier(art, VerifierWeights())
    assert out.coverage_delta == 1.0  # clipped
