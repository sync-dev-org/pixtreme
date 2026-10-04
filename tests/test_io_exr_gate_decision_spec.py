"""Specification tests for EXR backend adoption and repeat-run decisions."""

from __future__ import annotations

import math

import pytest
from exr_gate_decision import (
    GateDecision,
    GateRun,
    assert_gate_decisions_match_source_selection,
    synthesize_gate_decision,
)
from exr_gate_measurement import _measure, _warmup

import pixtreme._io.formats.exr.selection as io


class _FakeTimer:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _fake_exr_gate_decisions() -> dict[tuple[str, str], GateDecision]:
    decisions: dict[tuple[str, str], GateDecision] = {}
    read_selection = {
        "zip": "custom_cpu",
        "zips": "custom_cpu",
        "pxr24": "custom_cpu",
        "rle": "gpu",
        "b44": "gpu",
        "b44a": "gpu",
        "piz": "gpu",
        "dwaa": "gpu",
        "dwab": "gpu",
    }
    for compression, selected in read_selection.items():
        read_candidates = {"custom_cpu": 90.0, "gpu": 90.0}
        read_candidates[selected] = 79.0
        decisions[(compression, "read")] = synthesize_gate_decision(
            GateRun(openexr_ms=100.0, candidates_ms=read_candidates)
        )
    for compression in ("none", *read_selection):
        decisions[(compression, "write")] = synthesize_gate_decision(
            GateRun(openexr_ms=100.0, candidates_ms={"gpu": 79.0})
        )
    return decisions


def test_exr_gate_adopts_clear_initial_winner_without_repeat() -> None:
    """The EXR performance gate selects a clear fastest backend without a repeat run."""
    decision = synthesize_gate_decision(GateRun(openexr_ms=100.0, candidates_ms={"gpu": 79.0}))

    assert decision.initial_candidate == "gpu"
    assert decision.initial_ratio == pytest.approx(0.79)
    assert not decision.repeat_required
    assert decision.selected == "gpu"


def test_exr_gate_requires_both_runs_to_meet_the_threshold() -> None:
    """The EXR gate keeps OpenEXR selected if either comparison run misses its threshold."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 94.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 96.0}),
    )

    assert decision.repeat_required
    assert decision.initial_ratio == pytest.approx(0.94)
    assert decision.repeat_ratios == {"gpu": pytest.approx(0.96)}
    assert decision.selected == "cpu"


def test_exr_gate_initial_residual_cannot_be_reversed_by_repeat() -> None:
    """The EXR gate keeps OpenEXR when the initial ratio exceeds 0.95 despite a faster repeat."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 96.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 90.0}),
    )

    assert decision.repeat_required
    assert decision.selected == "cpu"


def test_exr_gate_includes_exact_095_in_provisional_adoption() -> None:
    """The EXR gate includes a ratio of exactly 0.95 in provisional adoption for both runs."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=200.0, candidates_ms={"gpu": 190.0}),
        GateRun(openexr_ms=80.0, candidates_ms={"gpu": 76.0}),
    )

    assert decision.initial_ratio == pytest.approx(0.95)
    assert decision.repeat_ratios == {"gpu": pytest.approx(0.95)}
    assert decision.selected == "gpu"


@pytest.mark.parametrize(
    ("initial_ratio", "repeat_required"),
    (
        pytest.param(0.8, True, id="lower-bound-inclusive"),
        pytest.param(1.0, True, id="upper-bound-inclusive"),
        pytest.param(math.nextafter(0.8, 0.0), False, id="below-lower-bound"),
        pytest.param(math.nextafter(1.0, math.inf), False, id="above-upper-bound"),
    ),
)
def test_exr_gate_repeat_band_has_inclusive_endpoints(
    initial_ratio: float,
    repeat_required: bool,
) -> None:
    """The EXR gate repeats comparisons for ratios from 0.8 through 1.0, including both ends."""
    decision = synthesize_gate_decision(GateRun(openexr_ms=1.0, candidates_ms={"gpu": initial_ratio}))

    assert decision.repeat_required is repeat_required


def test_exr_read_gate_uses_two_run_worst_ratio_before_repeat_median() -> None:
    """The EXR read gate ranks candidates by their worst ratio across two runs before comparing repeat medians."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 90.0, "custom_cpu": 94.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 94.0, "custom_cpu": 93.0}),
    )

    assert decision.initial_candidate == "gpu"
    assert decision.selected == "custom_cpu"
    assert decision.worst_ratios == {"gpu": pytest.approx(0.94), "custom_cpu": pytest.approx(0.94)}


def test_exr_read_gate_rejects_candidate_that_misses_either_run() -> None:
    """The EXR read gate excludes a backend that misses the threshold in either run."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 90.0, "custom_cpu": 94.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 96.0, "custom_cpu": 93.0}),
    )

    assert decision.selected == "custom_cpu"
    assert decision.worst_ratios == {"custom_cpu": pytest.approx(0.94)}


def test_exr_gate_measurements_match_source_fixed_selection() -> None:
    """EXR gate results from controlled measurements match every gate-eligible built-in route."""
    assert_gate_decisions_match_source_selection(_fake_exr_gate_decisions(), io._EXR_ROUTING)


def test_exr_gate_selection_oracle_rejects_a_source_fixed_mismatch() -> None:
    """The EXR gate reports a mismatch between synthesized backend selection and built-in routing."""
    mismatched_selection = dict(io._EXR_ROUTING)
    mismatched_selection[("rle", "write")] = "cpu"

    with pytest.raises(AssertionError, match="source-fixed EXR selection"):
        assert_gate_decisions_match_source_selection(_fake_exr_gate_decisions(), mismatched_selection)


def test_exr_gate_selection_oracle_requires_repeat_synthesis_before_comparison() -> None:
    """The EXR gate compares built-in routing with a repeat-band result only after the repeat is synthesized."""
    decisions = _fake_exr_gate_decisions()
    initial = GateRun(openexr_ms=100.0, candidates_ms={"custom_cpu": 94.0, "gpu": 90.0})
    decisions[("rle", "read")] = synthesize_gate_decision(initial)

    with pytest.raises(AssertionError, match="isolated repeat results are required"):
        assert_gate_decisions_match_source_selection(decisions, io._EXR_ROUTING)

    decisions[("rle", "read")] = synthesize_gate_decision(
        initial,
        GateRun(openexr_ms=100.0, candidates_ms={"custom_cpu": 96.0, "gpu": 94.0}),
    )
    assert_gate_decisions_match_source_selection(decisions, io._EXR_ROUTING)


def test_exr_gate_measurement_warmup_synchronizes_each_iteration_for_half_a_second() -> None:
    """EXR performance measurements synchronize each warmup iteration and warm the device for at least half a second."""
    timer = _FakeTimer()
    events: list[str] = []

    def operation() -> object:
        events.append("operation")
        timer.advance(0.125)
        return object()

    def synchronize() -> None:
        events.append("synchronize")

    _warmup(operation, synchronize, timer=timer)

    assert timer.now == pytest.approx(0.5)
    assert events == ["synchronize", "operation", "synchronize"] * 4


@pytest.mark.parametrize(
    ("seconds_per_iteration", "expected_iterations"),
    (
        pytest.param(0.2, 20, id="iteration-floor-outlasts-time-floor"),
        pytest.param(0.125, 24, id="time-floor-outlasts-iteration-floor"),
    ),
)
def test_exr_gate_measurement_synchronizes_and_requires_both_floors(
    seconds_per_iteration: float,
    expected_iterations: int,
) -> None:
    """EXR performance measurements synchronize around each operation and meet 20 iterations and three seconds."""
    timer = _FakeTimer()
    events: list[str] = []

    def operation() -> object:
        events.append("operation")
        timer.advance(seconds_per_iteration)
        return object()

    def synchronize() -> None:
        events.append("synchronize")

    durations_ms = _measure(operation, synchronize, timer=timer)

    assert durations_ms == pytest.approx([seconds_per_iteration * 1000.0] * expected_iterations)
    assert events == ["synchronize"] + ["synchronize", "operation", "synchronize"] * expected_iterations
