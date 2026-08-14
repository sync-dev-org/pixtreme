"""Specification tests for the repeatable DWAA/DWAB fp16/fp32 performance gate."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import exr_dwa_performance as dwa_gate
import pytest
import run_exr_dwa_gate as dwa_runner
from exr_dwa_performance import (
    DWA_COMPRESSIONS,
    DWA_DTYPES,
    GateMeasurement,
    GateRun,
    _measure,
    _warmup,
    build_dwa_performance_inputs,
    inspect_dwa_fixture,
    measure_dwa_gate_case,
    synthesize_gate_decision,
)
from openexr_dev_oracle import write_frame as write_openexr_frame

import pixtreme as px
import pixtreme._io.formats.exr.selection as io


class _FakeTimer:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(scope="module")
def dwa_inputs(tmp_path_factory: pytest.TempPathFactory) -> dwa_gate.DwaPerformanceInputs:
    """v1-exr-gpu-phase2 acceptance 27-29: tests share one real FHD fixture matrix."""
    return build_dwa_performance_inputs(tmp_path_factory.mktemp("dwa-gate"))


def test_dwa_performance_warmup_synchronizes_each_iteration_for_half_a_second() -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: warmup surrounds every iteration and reaches 0.5 seconds."""
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
def test_dwa_performance_measurement_requires_both_protocol_floors(
    seconds_per_iteration: float,
    expected_iterations: int,
) -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: timing uses pre/post sync and the 20-iteration AND 3-second floor."""
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


def test_dwa_gate_adopts_clear_winner_without_repeat() -> None:
    """v1-exr-gpu-phase2 acceptance 29: a ratio below the repeat band adopts the fastest candidate directly."""
    decision = synthesize_gate_decision(GateRun(openexr_ms=100.0, candidates_ms={"custom_cpu": 79.0, "gpu": 70.0}))

    assert decision.initial_ratios == {
        "custom_cpu": pytest.approx(0.79),
        "gpu": pytest.approx(0.70),
    }
    assert decision.repeat_required is False
    assert decision.selected == "gpu"


def test_dwa_gate_validates_both_candidates_in_both_runs_and_selects_the_fastest_passing_median() -> None:
    """v1-exr-gpu-phase2 acceptance 29: read selection compares both candidates after two-run validation."""
    decision = synthesize_gate_decision(
        GateRun(openexr_ms=100.0, candidates_ms={"custom_cpu": 93.0, "gpu": 94.0}),
        GateRun(openexr_ms=80.0, candidates_ms={"custom_cpu": 75.2, "gpu": 76.8}),
    )

    assert decision.initial_ratios == {
        "custom_cpu": pytest.approx(0.93),
        "gpu": pytest.approx(0.94),
    }
    assert decision.repeat_ratios == {
        "custom_cpu": pytest.approx(0.94),
        "gpu": pytest.approx(0.96),
    }
    assert decision.selected == "custom_cpu"


def test_dwa_gate_repeat_requires_the_same_candidate_mapping() -> None:
    """v1-exr-gpu-phase2 acceptance 29: repeat synthesis cannot silently omit custom CPU read."""
    initial = GateRun(openexr_ms=100.0, candidates_ms={"custom_cpu": 93.0, "gpu": 94.0})
    repeat = GateRun(openexr_ms=100.0, candidates_ms={"gpu": 94.0})

    with pytest.raises(ValueError, match="same non-OpenEXR candidates"):
        synthesize_gate_decision(initial, repeat)


@pytest.mark.parametrize(
    ("initial_ratio", "repeat_required"),
    (
        pytest.param(0.8, True, id="lower-bound-inclusive"),
        pytest.param(1.0, True, id="upper-bound-inclusive"),
        pytest.param(math.nextafter(0.8, 0.0), False, id="below-lower-bound"),
        pytest.param(math.nextafter(1.0, math.inf), False, id="above-upper-bound"),
    ),
)
def test_dwa_gate_repeat_band_has_inclusive_endpoints(initial_ratio: float, repeat_required: bool) -> None:
    """v1-exr-gpu-phase2 acceptance 29: the isolated-repeat band is the closed interval 0.8 through 1.0."""
    decision = synthesize_gate_decision(GateRun(openexr_ms=1.0, candidates_ms={"gpu": initial_ratio}))

    assert decision.repeat_required is repeat_required


@pytest.mark.parametrize("compression", DWA_COMPRESSIONS)
@pytest.mark.parametrize("dtype", DWA_DTYPES)
def test_dwa_fixture_inspection_reports_every_gate_axis_from_the_real_file(
    dwa_inputs: dwa_gate.DwaPerformanceInputs,
    compression: str,
    dtype: str,
) -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: parsed fixtures prove FHD RGB dtype, level, and compressed DWA."""
    inspection = inspect_dwa_fixture(dwa_inputs.read_path(compression, dtype), compression, dtype)

    assert (inspection.width, inspection.height) == (1920, 1080)
    assert inspection.channels == ("B", "G", "R")
    assert inspection.dtype == dtype
    assert inspection.dwa_level == pytest.approx(45.0)
    assert inspection.compressed_chunks > 0


def test_dwa_fixture_inspection_rejects_requested_dtype_mismatch(
    dwa_inputs: dwa_gate.DwaPerformanceInputs,
) -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: the requested dtype must match every parsed channel."""
    with pytest.raises(AssertionError, match="dtype"):
        inspect_dwa_fixture(dwa_inputs.read_path("dwaa", "fp16"), "dwaa", "fp32")


def test_dwa_fixture_inspection_rejects_dimension_mismatch(tmp_path: Path) -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: dimensions are verified from the parsed EXR dataWindow."""
    path = tmp_path / "wrong-dimensions.exr"
    frame = px.io.from_array(
        cp.zeros((16, 32, 3), dtype=cp.float16),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    write_openexr_frame(path, frame, compression="dwaa", dwa_level=45.0)

    with pytest.raises(AssertionError, match="dimensions"):
        inspect_dwa_fixture(path, "dwaa", "fp16")


def test_dwa_fixture_inspection_rejects_channel_mismatch(tmp_path: Path) -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: the parsed fixture must contain exactly RGB channels."""
    path = tmp_path / "wrong-channels.exr"
    frame = px.io.from_array(
        cp.zeros((1080, 1920, 1), dtype=cp.float16),
        colorspace="ACEScg",
        gamma="linear",
        channels=("Y",),
    )
    write_openexr_frame(path, frame, compression="dwaa", dwa_level=45.0)

    with pytest.raises(AssertionError, match="channels"):
        inspect_dwa_fixture(path, "dwaa", "fp16")


def test_dwa_fixture_inspection_rejects_dwa_level_mismatch(
    tmp_path: Path,
    dwa_inputs: dwa_gate.DwaPerformanceInputs,
) -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: the parsed DWA level must equal the requested gate level."""
    path = tmp_path / "wrong-level.exr"
    write_openexr_frame(path, dwa_inputs.fp16_frame, compression="dwaa", dwa_level=44.0)

    with pytest.raises(AssertionError, match="DWA level"):
        inspect_dwa_fixture(path, "dwaa", "fp16")


def test_dwa_measurement_isolates_dev_backends_and_restores_source_fixed_routing(
    monkeypatch: pytest.MonkeyPatch,
    dwa_inputs: dwa_gate.DwaPerformanceInputs,
) -> None:
    """v1-exr-runtime-independence acceptance 36 and 47: dev measurements cannot persist routing changes."""
    trace: list[tuple[str, str, str, str]] = []
    backend_queue: list[str] = []
    active_case = ("", "")

    def fake_measure(
        operation: Callable[[], object],
        synchronize: Callable[[], None],
        *,
        timer: Callable[[], float] = dwa_gate.perf_counter,
    ) -> list[float]:
        del synchronize, timer
        backend = backend_queue.pop(0)
        output = operation()
        del output
        compression, direction = active_case
        route = io._EXR_ROUTING[(compression, direction)]
        trace.append((compression, direction, backend, route))
        return {
            "cpu": [3.0, 2.0, 1.0],
            "custom_cpu": [2.0, 1.5, 1.0],
            "gpu": [1.0, 0.75, 0.5],
        }[backend]

    monkeypatch.setattr(
        dwa_gate,
        "inspect_dwa_fixture",
        lambda path, compression, dtype: _FakeFixtureInspection(compression=compression, dtype=dtype),
    )
    monkeypatch.setattr(dwa_gate, "_warmup", lambda operation, synchronize: None)
    monkeypatch.setattr(dwa_gate, "_measure", fake_measure)
    original_selection = dict(io._EXR_ROUTING)

    for compression in DWA_COMPRESSIONS:
        for dtype in DWA_DTYPES:
            for direction, backend_order in (
                ("read", ("gpu", "custom_cpu", "cpu")),
                ("write", ("gpu", "cpu")),
            ):
                active_case = (compression, direction)
                backend_queue.extend(backend_order)
                before = len(trace)
                measurement = measure_dwa_gate_case(
                    dwa_inputs,
                    compression,
                    dtype,
                    direction,
                    backend_order=backend_order,
                )
                assert [item[2] for item in trace[before:]] == list(backend_order)
                assert {item[3] for item in trace[before:]} <= {"native", "custom_cpu", "gpu"}
                assert measurement.medians_ms == {
                    backend: {"cpu": 2.0, "custom_cpu": 1.5, "gpu": 0.75}[backend] for backend in backend_order
                }
                assert measurement.iterations == {backend: 3 for backend in backend_order}
                if direction == "read":
                    assert measurement.output_bytes == {}
                else:
                    assert set(measurement.output_bytes) == {"cpu", "gpu"}
                    assert all(size > 0 for size in measurement.output_bytes.values())

    assert io._EXR_ROUTING == original_selection


def test_dwa_public_boundary_rejects_an_unsupported_routing_label(
    monkeypatch: pytest.MonkeyPatch,
    dwa_inputs: dwa_gate.DwaPerformanceInputs,
) -> None:
    """v1-exr-runtime-independence acceptance 36 and 48: the real public boundary rejects unsupported routes."""
    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "read"), "cpu")
    operation = dwa_gate._boundary_operation(dwa_inputs, "dwaa", "fp16", "read", "gpu")

    with pytest.raises(RuntimeError, match="unknown internal lane"):
        operation()


@dataclass(frozen=True)
class _FakeFixtureInspection:
    compression: str
    dtype: str
    total_chunks: int = 1
    compressed_chunks: int = 1
    file_bytes: int = 100


@dataclass(frozen=True)
class _FakeDeviceIdentity:
    name: str = "test-device"
    driver_version: int = 1
    runtime_version: int = 2


class _FakeInputs:
    def read_path(self, compression: str, dtype: str) -> Path:
        return Path(f"{compression}-{dtype}.exr")


def test_dwa_runner_payload_fixes_directional_order_and_custom_cpu_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """v1-exr-gpu-phase2 acceptance 27-29: runner JSON records three read lanes and two write lanes."""
    calls: list[tuple[str, str, str, tuple[str, ...]]] = []

    def fake_measurement(
        inputs: object,
        compression: str,
        dtype: str,
        direction: str,
        *,
        backend_order: tuple[str, ...],
    ) -> GateMeasurement:
        del inputs
        calls.append((compression, dtype, direction, backend_order))
        return GateMeasurement(
            compression=compression,
            dtype=dtype,
            direction=direction,
            medians_ms={backend: float(index + 1) for index, backend in enumerate(backend_order)},
            iterations={backend: 20 for backend in backend_order},
            output_bytes={backend: 100 + index for index, backend in enumerate(backend_order)}
            if direction == "write"
            else {},
        )

    monkeypatch.setattr(sys, "argv", ["run_exr_dwa_gate.py", "--backend-order", "gpu-first"])
    monkeypatch.setattr(dwa_runner, "build_dwa_performance_inputs", lambda directory: _FakeInputs())
    monkeypatch.setattr(
        dwa_runner,
        "inspect_dwa_fixture",
        lambda path, compression, dtype: _FakeFixtureInspection(compression=compression, dtype=dtype),
    )
    monkeypatch.setattr(dwa_runner, "measure_dwa_gate_case", fake_measurement)
    monkeypatch.setattr(dwa_runner, "device_identity", _FakeDeviceIdentity)

    dwa_runner.main()
    payload = json.loads(capsys.readouterr().out)

    expected_read_order = ("gpu", "custom_cpu", "cpu")
    expected_write_order = ("gpu", "cpu")
    assert payload["backend_orders"] == {
        "read": list(expected_read_order),
        "write": list(expected_write_order),
    }
    assert calls == [
        (compression, dtype, direction, expected_read_order if direction == "read" else expected_write_order)
        for compression in DWA_COMPRESSIONS
        for dtype in DWA_DTYPES
        for direction in ("read", "write")
    ]
    for measurement in payload["measurements"]:
        expected = {"cpu", "custom_cpu", "gpu"} if measurement["direction"] == "read" else {"cpu", "gpu"}
        assert set(measurement["medians_ms"]) == expected
        assert set(measurement["iterations"]) == expected
        assert set(measurement["output_bytes"]) == (expected if measurement["direction"] == "write" else set())
