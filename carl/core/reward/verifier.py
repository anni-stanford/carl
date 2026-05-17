"""Deterministic CI-based verifier — the RLVR backbone.

Parses the natural-language artifacts CI emits (pytest exit code, coverage XML,
ruff JSON, mypy text, optional bandit JSON) into a normalized
:class:`VerifierComponents`. No LLM calls, no randomness; the same inputs
always produce the same score. This is the property that makes RLVR
hack-resistant.

Inputs are *paths* to artifact files so the same code works for local Docker
runs and Cursor cloud-VM runs (we just need the artifact files copied back).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from carl.core.reward.types import VerifierComponents
from carl.settings import VerifierWeights


@dataclass(frozen=True)
class CIArtifacts:
    """Paths to the artifact files emitted by an episode's CI run.

    Any field may be ``None`` to indicate the signal was not produced. Missing
    signals are excluded from the composite via weight renormalization, which
    is essential when running CARL on repos that don't have e.g. mypy.
    """

    pytest_exit_code: int
    pytest_report_json: Path | None  # `pytest --json-report --json-report-file=...`
    coverage_xml: Path | None
    coverage_xml_baseline: Path | None  # the *previous* coverage, for delta
    ruff_json: Path | None  # `ruff check --output-format=json`
    mypy_output: Path | None  # `mypy ... 2>&1` text output
    bandit_json: Path | None = None  # optional security scan


def compute_verifier(
    artifacts: CIArtifacts,
    weights: VerifierWeights,
) -> VerifierComponents:
    """Compute the verifier reward and full decomposition from CI artifacts."""

    raw_passed = _parse_tests_passed(artifacts.pytest_exit_code, artifacts.pytest_report_json)
    coverage_delta = _parse_coverage_delta(
        artifacts.coverage_xml, artifacts.coverage_xml_baseline
    )
    lint_clean = _parse_ruff_clean(artifacts.ruff_json)
    typecheck_clean = _parse_mypy_clean(artifacts.mypy_output)
    security_clean = _parse_bandit_clean(artifacts.bandit_json)

    # Renormalize weights when a signal is absent so the composite stays in [0, 1]
    components: list[tuple[float, float | None]] = [
        (weights.tests_passed, raw_passed.score),
        (weights.coverage_delta, coverage_delta),
        (weights.lint_clean, lint_clean),
        (weights.typecheck_clean, typecheck_clean),
        (weights.security_clean, security_clean),
    ]
    active = [(w, s) for w, s in components if s is not None]
    if not active:
        composite = 0.0
    else:
        total_w = sum(w for w, _ in active)
        composite = sum(w * s for w, s in active) / total_w if total_w > 0 else 0.0
        composite = max(0.0, min(1.0, composite))

    return VerifierComponents(
        tests_passed=raw_passed.score,
        coverage_delta=coverage_delta if coverage_delta is not None else 0.0,
        lint_clean=lint_clean if lint_clean is not None else 0.0,
        typecheck_clean=typecheck_clean,
        security_clean=security_clean,
        composite=composite,
        raw_test_count=raw_passed.total,
        raw_failed_count=raw_passed.failed,
    )


# --------- internals ----------------------------------------------------------


@dataclass(frozen=True)
class _TestsResult:
    score: float
    total: int
    failed: int


def _parse_tests_passed(exit_code: int, report_path: Path | None) -> _TestsResult:
    """Score = passed / total. If pytest crashed (no report), use exit code."""
    if report_path is not None and report_path.is_file():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        total = int(summary.get("total", 0))
        passed = int(summary.get("passed", 0))
        failed = int(summary.get("failed", 0)) + int(summary.get("error", 0))
        if total == 0:
            return _TestsResult(score=0.0, total=0, failed=0)
        return _TestsResult(score=passed / total, total=total, failed=failed)

    # No report: fall back to exit code (0 = pass, anything else = fail)
    if exit_code == 0:
        return _TestsResult(score=1.0, total=1, failed=0)
    return _TestsResult(score=0.0, total=1, failed=1)


def _parse_coverage_delta(
    new_xml: Path | None, baseline_xml: Path | None
) -> float | None:
    """Coverage delta normalized into [0, 1] via clipping at ±10 pp."""
    if new_xml is None or not new_xml.is_file():
        return None
    new = _coverage_line_rate(new_xml)
    if new is None:
        return None
    if baseline_xml is None or not baseline_xml.is_file():
        # First run: just report absolute coverage
        return float(new)
    baseline = _coverage_line_rate(baseline_xml)
    if baseline is None:
        return float(new)
    delta_pp = (new - baseline) * 100.0  # percentage points
    # Map delta from [-10, +10] pp into [0, 1]; clip outside that range
    return max(0.0, min(1.0, 0.5 + delta_pp / 20.0))


def _coverage_line_rate(xml_path: Path) -> float | None:
    """coverage.xml's root element has a ``line-rate`` attribute in [0, 1]."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        rate = root.attrib.get("line-rate")
        if rate is None:
            return None
        return float(rate)
    except (ET.ParseError, ValueError, OSError):
        return None


def _parse_ruff_clean(ruff_json: Path | None) -> float | None:
    """1.0 if zero diagnostics; else 1 / (1 + n_diagnostics)."""
    if ruff_json is None or not ruff_json.is_file():
        return None
    try:
        data = json.loads(ruff_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    n = len(data)
    return 1.0 / (1.0 + n)


_MYPY_ERROR_RE = re.compile(r"^.*: error: ", re.MULTILINE)


def _parse_mypy_clean(mypy_out: Path | None) -> float | None:
    if mypy_out is None or not mypy_out.is_file():
        return None
    text = mypy_out.read_text(encoding="utf-8")
    n = len(_MYPY_ERROR_RE.findall(text))
    return 1.0 / (1.0 + n)


def _parse_bandit_clean(bandit_json: Path | None) -> float | None:
    if bandit_json is None or not bandit_json.is_file():
        return None
    try:
        data = json.loads(bandit_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    results = data.get("results", []) if isinstance(data, dict) else []
    high = sum(1 for r in results if str(r.get("issue_severity", "")).upper() == "HIGH")
    medium = sum(1 for r in results if str(r.get("issue_severity", "")).upper() == "MEDIUM")
    weighted = high * 1.0 + medium * 0.3
    return 1.0 / (1.0 + weighted)
