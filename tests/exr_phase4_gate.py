"""Pure synthesis logic shared by the Phase 4 PIZ performance gate and its tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_ADOPTION_RATIO = 0.95
_REPEAT_RATIO_MINIMUM = 0.8
_REPEAT_RATIO_MAXIMUM = 1.0


@dataclass(frozen=True)
class GateRun:
    """One same-condition run of OpenEXR and every non-OpenEXR candidate."""

    openexr_ms: float
    candidates_ms: Mapping[str, float]


@dataclass(frozen=True)
class GateDecision:
    """Mechanically synthesized provisional backend selection."""

    initial_candidate: str
    initial_ratio: float
    repeat_required: bool
    selected: str
    initial_ratios: Mapping[str, float]
    repeat_ratios: Mapping[str, float]
    worst_ratios: Mapping[str, float]


def _validated_ratios(run: GateRun) -> dict[str, float]:
    if not math.isfinite(run.openexr_ms) or run.openexr_ms <= 0.0:
        raise ValueError("OpenEXR median must be finite and positive")
    if not run.candidates_ms:
        raise ValueError("at least one non-OpenEXR candidate median is required")
    ratios: dict[str, float] = {}
    for backend, candidate_ms in run.candidates_ms.items():
        if not backend or backend == "cpu":
            raise ValueError("candidate backend names must be non-empty and exclude the OpenEXR 'cpu' backend")
        if not math.isfinite(candidate_ms) or candidate_ms <= 0.0:
            raise ValueError(f"candidate median for {backend!r} must be finite and positive")
        ratios[backend] = candidate_ms / run.openexr_ms
    return ratios


def synthesize_gate_decision(initial: GateRun, repeat: GateRun | None = None) -> GateDecision:
    """Apply the Phase 4 0.95 threshold, repeat band, and read-candidate tie-break."""
    initial_ratios = _validated_ratios(initial)
    initial_candidate = min(initial.candidates_ms, key=lambda backend: (initial.candidates_ms[backend], backend))
    initial_ratio = initial_ratios[initial_candidate]
    repeat_required = _REPEAT_RATIO_MINIMUM <= initial_ratio <= _REPEAT_RATIO_MAXIMUM
    repeat_ratios: dict[str, float] = {}
    worst_ratios: dict[str, float] = {}

    if not repeat_required:
        selected = initial_candidate if initial_ratio <= _ADOPTION_RATIO else "cpu"
    elif repeat is None:
        selected = "cpu"
    else:
        if set(repeat.candidates_ms) != set(initial.candidates_ms):
            raise ValueError("isolated repeat must measure the same non-OpenEXR candidates as the initial run")
        repeat_ratios = _validated_ratios(repeat)
        worst_ratios = {
            backend: max(initial_ratios[backend], repeat_ratios[backend])
            for backend in initial.candidates_ms
            if initial_ratios[backend] <= _ADOPTION_RATIO and repeat_ratios[backend] <= _ADOPTION_RATIO
        }
        if worst_ratios:
            selected = min(
                worst_ratios,
                key=lambda backend: (worst_ratios[backend], repeat.candidates_ms[backend], backend),
            )
        else:
            selected = "cpu"

    return GateDecision(
        initial_candidate=initial_candidate,
        initial_ratio=initial_ratio,
        repeat_required=repeat_required,
        selected=selected,
        initial_ratios=MappingProxyType(initial_ratios),
        repeat_ratios=MappingProxyType(repeat_ratios),
        worst_ratios=MappingProxyType(worst_ratios),
    )


def assert_gate_decisions_match_source_selection(
    decisions: Mapping[tuple[str, str], GateDecision],
    source_selection: Mapping[tuple[str, str], str],
) -> None:
    """Fail unless every completed gate decision matches its source-fixed backend entry."""
    unresolved_repeats = tuple(
        sorted(key for key, decision in decisions.items() if decision.repeat_required and not decision.repeat_ratios)
    )
    if unresolved_repeats:
        raise AssertionError(f"isolated repeat results are required before source comparison: {unresolved_repeats!r}")
    synthesized_selection = {key: decision.selected for key, decision in decisions.items()}
    expected_selection = {key: source_selection[key] for key in decisions}
    if synthesized_selection != expected_selection:
        raise AssertionError(
            "source-fixed Phase 4 selection does not match synthesized gate decisions: "
            f"synthesized={synthesized_selection!r} source_fixed={expected_selection!r}"
        )
