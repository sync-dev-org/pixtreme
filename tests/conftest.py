from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

PerformanceResult = tuple[str, str, float, float, float, float, float, float]

_PERFORMANCE_RESULTS: list[PerformanceResult] = []


@pytest.fixture(scope="session")
def vocabulary_markdown() -> str:
    """REQ-TEST-008: load the repo-only token reference or skip in a docs-free distribution."""
    vocabulary_path = Path(__file__).resolve().parents[1] / "docs_site" / "tokens.md"
    if not vocabulary_path.is_file():
        pytest.skip("repo-only documentation contract: docs_site/tokens.md is absent from this distribution")
    return vocabulary_path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def performance_results() -> list[PerformanceResult]:
    return _PERFORMANCE_RESULTS


def pytest_sessionstart(session: pytest.Session) -> None:
    _PERFORMANCE_RESULTS.clear()


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    if not _PERFORMANCE_RESULTS:
        return

    terminalreporter.section("v1-performance FHD measurement report")
    terminalreporter.write_line(
        "| target | representative parameters | mean ms | median ms | fps | p5 ms | p95 ms | effective GB/s | > 1 ms |"
    )
    terminalreporter.write_line("|---|---|---:|---:|---:|---:|---:|---:|:---:|")
    for target, parameters, mean_ms, median_ms, fps, p5_ms, p95_ms, effective_gbps in _PERFORMANCE_RESULTS:
        marker = "yes" if median_ms > 1.0 else ""
        safe_target = target.replace("|", r"\|")
        safe_parameters = parameters.replace("|", r"\|")
        terminalreporter.write_line(
            f"| {safe_target} | {safe_parameters} | {mean_ms:.3f} | {median_ms:.3f} | {fps:.1f} | "
            f"{p5_ms:.3f} | {p95_ms:.3f} | {effective_gbps:.1f} | {marker} |"
        )

    exceeded = [
        f"{target} ({parameters}, {median_ms:.3f} ms)"
        for target, parameters, _mean_ms, median_ms, _fps, _p5_ms, _p95_ms, _effective_gbps in _PERFORMANCE_RESULTS
        if median_ms > 1.0
    ]
    terminalreporter.write_line("")
    terminalreporter.write_line(f"1 ms exceedances: {len(exceeded)}")
    for item in exceeded:
        terminalreporter.write_line(f"- {item}")
