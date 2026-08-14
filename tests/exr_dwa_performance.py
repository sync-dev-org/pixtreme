"""Reusable FHD fp16/fp32 harness for repeatable DWAA/DWAB performance gates."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import cupy as cp
import numpy as np
from openexr_dev_oracle import read_frame as read_openexr_frame
from openexr_dev_oracle import write_frame as write_openexr_frame

import pixtreme as px
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

WIDTH = 1920
HEIGHT = 1080
WARMUP_MINIMUM_SECONDS = 0.5
MEASURED_MINIMUM_ITERATIONS = 20
MEASURED_MINIMUM_SECONDS = 3.0
DWA_COMPRESSIONS = ("dwaa", "dwab")
DWA_DTYPES = ("fp16", "fp32")
_DWA_READ_BACKENDS = ("cpu", "custom_cpu", "gpu")
_DWA_WRITE_BACKENDS = ("cpu", "gpu")
_ADOPTION_RATIO = 0.95
_REPEAT_RATIO_MINIMUM = 0.8
_REPEAT_RATIO_MAXIMUM = 1.0
_DWA_LEVEL = 45.0
_SEED = 20260809 + 21


@dataclass(frozen=True)
class DwaPerformanceInputs:
    """Deterministic source frames and backend-specific file destinations."""

    fp16_frame: px.core.Frame
    fp32_frame: px.core.Frame
    directory: Path

    def frame(self, dtype: str) -> px.core.Frame:
        if dtype == "fp16":
            return self.fp16_frame
        if dtype == "fp32":
            return self.fp32_frame
        raise ValueError(f"unsupported DWA dtype: {dtype!r}")

    def read_path(self, compression: str, dtype: str) -> Path:
        _validate_axes(compression, dtype)
        return self.directory / f"read-{compression}-{dtype}.exr"

    def write_path(self, compression: str, dtype: str, backend: str) -> Path:
        _validate_axes(compression, dtype)
        if backend not in _DWA_WRITE_BACKENDS:
            raise ValueError(f"unsupported DWA backend: {backend!r}")
        return self.directory / f"write-{compression}-{dtype}-{backend}.exr"


@dataclass(frozen=True)
class FixtureInspection:
    """Compressed-structure evidence for one DWA file."""

    compression: str
    dtype: str
    width: int
    height: int
    channels: tuple[str, ...]
    dwa_level: float
    total_chunks: int
    compressed_chunks: int
    file_bytes: int


@dataclass(frozen=True)
class GateMeasurement:
    """Same-run OpenEXR/GPU medians and optional write output sizes."""

    compression: str
    dtype: str
    direction: str
    medians_ms: dict[str, float]
    iterations: dict[str, int]
    output_bytes: dict[str, int]


@dataclass(frozen=True)
class DeviceIdentity:
    """CUDA device identity recorded with one gate process."""

    name: str
    driver_version: int
    runtime_version: int


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


def _validate_axes(compression: str, dtype: str) -> None:
    if compression not in DWA_COMPRESSIONS:
        raise ValueError(f"unsupported DWA compression: {compression!r}")
    if dtype not in DWA_DTYPES:
        raise ValueError(f"unsupported DWA dtype: {dtype!r}")


def _backends_for_direction(direction: str) -> tuple[str, ...]:
    if direction == "read":
        return _DWA_READ_BACKENDS
    if direction == "write":
        return _DWA_WRITE_BACKENDS
    raise ValueError(f"unsupported DWA direction: {direction!r}")


def _dwa_source_frames() -> tuple[px.core.Frame, px.core.Frame]:
    generator = cp.random.default_rng(_SEED)
    x = cp.arange(WIDTH, dtype=cp.float32)[None, :] / np.float32(WIDTH - 1)
    y = cp.arange(HEIGHT, dtype=cp.float32)[:, None] / np.float32(HEIGHT - 1)
    detail = generator.integers(0, 16, size=(HEIGHT, WIDTH), dtype=cp.uint8).astype(cp.float32)
    detail *= np.float32(1.0 / 4096.0)
    data = cp.stack(
        (
            cp.broadcast_to(x, (HEIGHT, WIDTH)) + detail,
            cp.broadcast_to(y, (HEIGHT, WIDTH)) - detail,
            (x + y) * np.float32(0.5) + detail,
        ),
        axis=2,
    )
    fp32_frame = px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")
    fp16_frame = px.io.from_array(data.astype(cp.float16), colorspace="ACEScg", gamma="linear", channels="RGB")
    return fp16_frame, fp32_frame


def build_dwa_performance_inputs(directory: Path) -> DwaPerformanceInputs:
    """Create deterministic frames and OpenEXR-reference read fixtures for all four input axes."""
    directory.mkdir(parents=True, exist_ok=True)
    fp16_frame, fp32_frame = _dwa_source_frames()
    inputs = DwaPerformanceInputs(fp16_frame=fp16_frame, fp32_frame=fp32_frame, directory=directory)
    for compression in DWA_COMPRESSIONS:
        for dtype in DWA_DTYPES:
            write_openexr_frame(
                inputs.read_path(compression, dtype),
                inputs.frame(dtype),
                compression=compression,
                dwa_level=_DWA_LEVEL,
            )
    return inputs


def inspect_dwa_fixture(path: Path, compression: str, dtype: str) -> FixtureInspection:
    """Fail unless parsed file metadata and compressed structure match every requested gate axis."""
    _validate_axes(compression, dtype)
    container = io_header._parse_exr(path)
    part = container.parts[0]
    x_min, y_min, x_max, y_max = container.data_window
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if (width, height) != (WIDTH, HEIGHT):
        raise AssertionError(
            f"{path} dimensions do not match the DWA gate: parsed={(width, height)!r} expected={(WIDTH, HEIGHT)!r}"
        )
    channels = tuple(sorted(channel.name for channel in part.channels))
    if channels != ("B", "G", "R"):
        raise AssertionError(f"{path} channels do not match the DWA gate RGB set: parsed={channels!r}")
    expected_storage_dtype = {"fp16": "float16", "fp32": "float32"}[dtype]
    storage_dtypes = tuple(sorted({channel.dtype for channel in part.channels}))
    if storage_dtypes != (expected_storage_dtype,):
        raise AssertionError(
            f"{path} dtype does not match the requested DWA gate axis: "
            f"parsed={storage_dtypes!r} requested={expected_storage_dtype!r}"
        )
    level_attribute = part.attributes.get("dwaCompressionLevel")
    if level_attribute is None or level_attribute.attribute_type != "float" or len(level_attribute.payload) != 4:
        raise AssertionError(f"{path} has no scalar float DWA level metadata")
    dwa_level = float(struct.unpack("<f", level_attribute.payload)[0])
    if not math.isclose(dwa_level, _DWA_LEVEL, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(f"{path} DWA level does not match the gate: parsed={dwa_level!r} expected={_DWA_LEVEL!r}")
    compressed_chunks = sum(not chunk.raw_stored for chunk in container.chunks)
    if container.compression != compression or not container.dwa_eligible or not compressed_chunks:
        raise AssertionError(f"{path} is not an eligible {compression.upper()} file with a compressed DWA v2 chunk")
    return FixtureInspection(
        compression=compression,
        dtype=dtype,
        width=width,
        height=height,
        channels=channels,
        dwa_level=dwa_level,
        total_chunks=len(container.chunks),
        compressed_chunks=compressed_chunks,
        file_bytes=path.stat().st_size,
    )


def _warmup(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    timer: Callable[[], float] = perf_counter,
) -> None:
    started_at = timer()
    while timer() - started_at < WARMUP_MINIMUM_SECONDS:
        synchronize()
        output = operation()
        synchronize()
        del output


def _measure(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    timer: Callable[[], float] = perf_counter,
) -> list[float]:
    durations_ms: list[float] = []
    synchronize()
    measurement_started_at = timer()
    elapsed_seconds = 0.0
    while len(durations_ms) < MEASURED_MINIMUM_ITERATIONS or elapsed_seconds < MEASURED_MINIMUM_SECONDS:
        synchronize()
        iteration_started_at = timer()
        output = operation()
        synchronize()
        iteration_finished_at = timer()
        durations_ms.append((iteration_finished_at - iteration_started_at) * 1000.0)
        del output
        elapsed_seconds = iteration_finished_at - measurement_started_at
    return durations_ms


def _boundary_operation(
    inputs: DwaPerformanceInputs,
    compression: str,
    dtype: str,
    direction: str,
    backend: str,
) -> Callable[[], object]:
    if direction == "read":
        path = inputs.read_path(compression, dtype)
        if backend == "cpu":
            return lambda: read_openexr_frame(path)
        return lambda: px.io.read_image(path, unchanged=True)
    if direction == "write":
        path = inputs.write_path(compression, dtype, backend)
        frame = inputs.frame(dtype)
        if backend == "cpu":
            return lambda: write_openexr_frame(path, frame, compression=compression, dwa_level=_DWA_LEVEL)
        output_dtype = {"fp16": "float16", "fp32": "float32"}[dtype]
        return lambda: px.io.write_image(
            path,
            frame,
            compression=compression,
            dwa_level=_DWA_LEVEL,
            dtype=output_dtype,
        )
    raise ValueError(f"unsupported DWA direction: {direction!r}")


def measure_dwa_gate_case(
    inputs: DwaPerformanceInputs,
    compression: str,
    dtype: str,
    direction: str,
    *,
    backend_order: Sequence[str] | None = None,
) -> GateMeasurement:
    """Measure every direction-specific backend in one process with an explicit drift-resistant order."""
    _validate_axes(compression, dtype)
    expected_backends = _backends_for_direction(direction)
    resolved_backend_order = expected_backends if backend_order is None else tuple(backend_order)
    if len(resolved_backend_order) != len(expected_backends) or set(resolved_backend_order) != set(expected_backends):
        raise ValueError(f"backend_order for {direction} must contain exactly {expected_backends!r}")
    key = (compression, direction)
    sentinel = object()
    original_backend: object = io._EXR_ROUTING.get(key, sentinel)
    medians_ms: dict[str, float] = {}
    iterations: dict[str, int] = {}
    output_bytes: dict[str, int] = {}
    synchronize = cp.cuda.Device().synchronize
    try:
        for backend in resolved_backend_order:
            if backend != "cpu":
                io._EXR_ROUTING[key] = backend
            operation = _boundary_operation(inputs, compression, dtype, direction, backend)
            _warmup(operation, synchronize)
            durations_ms = _measure(operation, synchronize)
            medians_ms[backend] = median(durations_ms)
            iterations[backend] = len(durations_ms)
            if direction == "write":
                path = inputs.write_path(compression, dtype, backend)
                output_bytes[backend] = inspect_dwa_fixture(path, compression, dtype).file_bytes
    finally:
        if original_backend is sentinel:
            io._EXR_ROUTING.pop(key, None)
        else:
            io._EXR_ROUTING[key] = str(original_backend)
    return GateMeasurement(
        compression=compression,
        dtype=dtype,
        direction=direction,
        medians_ms=medians_ms,
        iterations=iterations,
        output_bytes=output_bytes,
    )


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
    """Apply the 0.95 threshold, repeat band, and fastest-passing-candidate rule."""
    initial_ratios = _validated_ratios(initial)
    initial_candidate = min(initial.candidates_ms, key=lambda backend: (initial.candidates_ms[backend], backend))
    initial_ratio = initial_ratios[initial_candidate]
    repeat_required = _REPEAT_RATIO_MINIMUM <= initial_ratio <= _REPEAT_RATIO_MAXIMUM
    repeat_ratios: dict[str, float] = {}
    if repeat is not None:
        if set(repeat.candidates_ms) != set(initial.candidates_ms):
            raise ValueError("isolated repeat must measure the same non-OpenEXR candidates as the initial run")
        repeat_ratios = _validated_ratios(repeat)

    if repeat_required and repeat is None:
        selected = "cpu"
    else:
        passing_candidates = tuple(
            backend
            for backend in initial.candidates_ms
            if initial_ratios[backend] <= _ADOPTION_RATIO
            and (repeat is None or repeat_ratios[backend] <= _ADOPTION_RATIO)
        )
        selected = (
            min(passing_candidates, key=lambda backend: (initial.candidates_ms[backend], backend))
            if passing_candidates
            else "cpu"
        )
    return GateDecision(
        initial_candidate=initial_candidate,
        initial_ratio=initial_ratio,
        repeat_required=repeat_required,
        selected=selected,
        initial_ratios=MappingProxyType(initial_ratios),
        repeat_ratios=MappingProxyType(repeat_ratios),
    )


def device_identity() -> DeviceIdentity:
    """Return the exact CUDA device identity recorded with a gate process."""
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return DeviceIdentity(
        name=str(name),
        driver_version=cp.cuda.runtime.driverGetVersion(),
        runtime_version=cp.cuda.runtime.runtimeGetVersion(),
    )
