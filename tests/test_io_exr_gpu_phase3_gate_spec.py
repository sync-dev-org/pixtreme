"""Specification tests for the Phase 3 OpenEXR performance-gate synthesis."""

from __future__ import annotations

import math

import pytest
from exr_phase3_gate import (
    GateDecision,
    GateRun,
    assert_gate_decisions_match_source_selection,
    synthesize_gate_decision,
)
from exr_phase3_performance import PHASE3_COMPRESSIONS, _measure, _warmup

import pixtreme._io.formats.exr.selection as io


class _FakeTimer:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _fake_phase3_gate_decisions() -> dict[tuple[str, str], GateDecision]:
    decisions: dict[tuple[str, str], GateDecision] = {}
    for compression in PHASE3_COMPRESSIONS:
        read_candidates = (
            {"custom_cpu": 79.0, "gpu": 90.0} if compression == "pxr24" else {"custom_cpu": 90.0, "gpu": 79.0}
        )
        decisions[(compression, "read")] = synthesize_gate_decision(
            GateRun(openexr_ms=100.0, candidates_ms=read_candidates)
        )
        decisions[(compression, "write")] = synthesize_gate_decision(
            GateRun(openexr_ms=100.0, candidates_ms={"gpu": 79.0})
        )
    return decisions


def test_phase3_gate_adopts_clear_initial_winner_without_repeat() -> None:
    """v1-exr-gpu-phase3 acceptance 38-39: a ratio below 0.8 adopts the fastest initial candidate directly."""
    decision = synthesize_gate_decision(GateRun(openexr_ms=100.0, candidates_ms={"gpu": 79.0}))

    assert decision.initial_candidate == "gpu"
    assert decision.initial_ratio == pytest.approx(0.79)
    assert not decision.repeat_required
    assert decision.selected == "gpu"


def test_phase3_gate_requires_both_runs_to_meet_the_threshold() -> None:
    """v1-exr-gpu-phase3 acceptance 39: one repeat miss keeps the OpenEXR backend selected."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 94.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 96.0}),
    )

    assert decision.repeat_required
    assert decision.initial_ratio == pytest.approx(0.94)
    assert decision.repeat_ratios == {"gpu": pytest.approx(0.96)}
    assert decision.selected == "cpu"


def test_phase3_gate_initial_residual_cannot_be_reversed_by_repeat() -> None:
    """v1-exr-gpu-phase3 acceptance 39: an initial ratio above 0.95 remains OpenEXR despite a faster repeat."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 96.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 90.0}),
    )

    assert decision.repeat_required
    assert decision.selected == "cpu"


def test_phase3_gate_includes_exact_095_in_provisional_adoption() -> None:
    """v1-exr-gpu-phase3 acceptance 38-39: the 0.95 boundary is inclusive in both runs."""
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
def test_phase3_gate_repeat_band_has_inclusive_endpoints(
    initial_ratio: float,
    repeat_required: bool,
) -> None:
    """v1-exr-gpu-phase3 acceptance 39: the 0.8..1.0 repeat band is a closed interval."""
    decision = synthesize_gate_decision(GateRun(openexr_ms=1.0, candidates_ms={"gpu": initial_ratio}))

    assert decision.repeat_required is repeat_required


def test_phase3_read_gate_uses_two_run_worst_ratio_before_repeat_median() -> None:
    """v1-exr-gpu-phase3 acceptance 39: read candidates rank by worst ratio, then isolated median."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 90.0, "custom_cpu": 94.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 94.0, "custom_cpu": 93.0}),
    )

    assert decision.initial_candidate == "gpu"
    assert decision.selected == "custom_cpu"
    assert decision.worst_ratios == {"gpu": pytest.approx(0.94), "custom_cpu": pytest.approx(0.94)}


def test_phase3_read_gate_rejects_candidate_that_misses_either_run() -> None:
    """v1-exr-gpu-phase3 acceptance 39: read tie-break considers only candidates passing both runs."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 90.0, "custom_cpu": 94.0}),
        GateRun(openexr_ms=100.0, candidates_ms={"gpu": 96.0, "custom_cpu": 93.0}),
    )

    assert decision.selected == "custom_cpu"
    assert decision.worst_ratios == {"custom_cpu": pytest.approx(0.94)}


def test_phase3_fake_gate_measurements_match_source_fixed_selection() -> None:
    """v1-exr-runtime-independence acceptance 36: fake medians match the final self-owned routes."""
    assert_gate_decisions_match_source_selection(_fake_phase3_gate_decisions(), io._EXR_ROUTING)


def test_phase3_gate_selection_oracle_rejects_a_source_fixed_mismatch() -> None:
    """v1-exr-gpu-phase3 acceptance 38-40: a synthesized/source-fixed mismatch fails the gate oracle."""
    mismatched_selection = dict(io._EXR_ROUTING)
    mismatched_selection[("rle", "write")] = "cpu"

    with pytest.raises(AssertionError, match="source-fixed Phase 3 selection"):
        assert_gate_decisions_match_source_selection(_fake_phase3_gate_decisions(), mismatched_selection)


def test_phase3_gate_selection_oracle_requires_repeat_synthesis_before_comparison() -> None:
    """v1-exr-gpu-phase3 acceptance 39-40: repeat-band decisions compare only after isolated repeat synthesis."""
    decisions = _fake_phase3_gate_decisions()
    initial = GateRun(openexr_ms=100.0, candidates_ms={"custom_cpu": 94.0, "gpu": 90.0})
    decisions[("rle", "read")] = synthesize_gate_decision(initial)

    with pytest.raises(AssertionError, match="isolated repeat results are required"):
        assert_gate_decisions_match_source_selection(decisions, io._EXR_ROUTING)

    decisions[("rle", "read")] = synthesize_gate_decision(
        initial,
        GateRun(openexr_ms=100.0, candidates_ms={"custom_cpu": 96.0, "gpu": 94.0}),
    )
    assert_gate_decisions_match_source_selection(decisions, io._EXR_ROUTING)


def test_phase3_performance_warmup_synchronizes_each_iteration_for_half_a_second() -> None:
    """v1-exr-gpu-phase3 acceptance 37: warmup surrounds every iteration and reaches 0.5 seconds."""
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
def test_phase3_performance_measurement_synchronizes_and_requires_both_floors(
    seconds_per_iteration: float,
    expected_iterations: int,
) -> None:
    """v1-exr-gpu-phase3 acceptance 37: measurement uses pre/post sync and the 20-iteration AND 3-second floor."""
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


def test_phase3_routing_is_source_fixed_with_final_self_owned_lanes() -> None:
    """v1-exr-runtime-independence acceptance 36-37: Phase 3 and prior codecs use final self-owned lanes."""
    phase3_selection = {
        key: backend for key, backend in io._EXR_ROUTING.items() if key[0] in {"rle", "pxr24", "b44", "b44a"}
    }
    prior_selection = {
        key: backend for key, backend in io._EXR_ROUTING.items() if key[0] in {"none", "zip", "zips", "dwaa", "dwab"}
    }

    assert phase3_selection == {
        ("rle", "read"): "gpu",
        ("rle", "write"): "gpu",
        ("pxr24", "read"): "custom_cpu",
        ("pxr24", "write"): "gpu",
        ("b44", "read"): "gpu",
        ("b44", "write"): "gpu",
        ("b44a", "read"): "gpu",
        ("b44a", "write"): "gpu",
    }
    assert prior_selection == {
        ("none", "read"): "native",
        ("none", "write"): "gpu",
        ("zip", "read"): "custom_cpu",
        ("zip", "write"): "gpu",
        ("zips", "read"): "custom_cpu",
        ("zips", "write"): "gpu",
        ("dwaa", "read"): "gpu",
        ("dwaa", "write"): "gpu",
        ("dwab", "read"): "gpu",
        ("dwab", "write"): "gpu",
    }
