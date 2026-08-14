"""Reusable FHD fixtures and boundary harness for the Phase 4 PIZ performance gate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np
from openexr_dev_oracle import read_frame as read_openexr_frame
from openexr_dev_oracle import write_frame as write_openexr_frame

import pixtreme as px
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

WIDTH = 1920
HEIGHT = 1080
WARMUP_MINIMUM_SECONDS = 0.5
MEASURED_MINIMUM_ITERATIONS = 20
MEASURED_MINIMUM_SECONDS = 3.0
PHASE4_DTYPES = ("fp16", "fp32")
_SEED = 20260809


@dataclass(frozen=True)
class Phase4PerformanceInputs:
    """Shared source frames, reference files, and write destinations for the PIZ gate."""

    fp16_frame: px.core.Frame
    fp32_frame: px.core.Frame
    directory: Path

    def frame(self, dtype: str) -> px.core.Frame:
        if dtype == "fp16":
            return self.fp16_frame
        if dtype == "fp32":
            return self.fp32_frame
        raise ValueError(f"unsupported Phase 4 dtype: {dtype!r}")

    def read_path(self, dtype: str) -> Path:
        if dtype not in PHASE4_DTYPES:
            raise ValueError(f"unsupported Phase 4 dtype: {dtype!r}")
        return self.directory / f"phase4-read-piz-{dtype}.exr"

    def write_path(self, dtype: str) -> Path:
        if dtype not in PHASE4_DTYPES:
            raise ValueError(f"unsupported Phase 4 dtype: {dtype!r}")
        return self.directory / f"phase4-write-piz-{dtype}.exr"


@dataclass(frozen=True)
class FixtureInspection:
    """PIZ compressed-structure evidence gathered before timing begins."""

    dtype: str
    total_chunks: int
    compressed_chunks: int
    nonempty_bitmaps: int
    maximum_wavelet_levels: int
    huffman_tables: int
    huffman_table_bytes: int
    huffman_data_bytes: int


@dataclass(frozen=True)
class GateMeasurement:
    """Medians and iteration counts for one dtype/direction forced-backend comparison."""

    dtype: str
    direction: str
    medians_ms: dict[str, float]
    iterations: dict[str, int]


@dataclass(frozen=True)
class DeviceIdentity:
    """CUDA device identity recorded with one gate run."""

    name: str
    driver_version: int
    runtime_version: int


def _phase4_source_frames() -> tuple[px.core.Frame, px.core.Frame]:
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
    data[64:96, :, :] = cp.asarray((0.125, 0.5, 1.25), dtype=cp.float32)
    fp32_frame = px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")
    fp16_frame = px.io.from_array(data.astype(cp.float16), colorspace="ACEScg", gamma="linear", channels="RGB")
    return fp16_frame, fp32_frame


def build_phase4_performance_inputs(directory: Path) -> Phase4PerformanceInputs:
    """Create deterministic fp16/fp32 frames and OpenEXR-reference PIZ read fixtures."""
    directory.mkdir(parents=True, exist_ok=True)
    fp16_frame, fp32_frame = _phase4_source_frames()
    inputs = Phase4PerformanceInputs(fp16_frame=fp16_frame, fp32_frame=fp32_frame, directory=directory)
    for dtype in PHASE4_DTYPES:
        write_openexr_frame(inputs.read_path(dtype), inputs.frame(dtype), compression="piz", dwa_level=None)
    return inputs


def _wavelet_level_count(width: int, row_count: int) -> int:
    levels = 0
    level_extent = 2
    while level_extent <= min(width, row_count):
        levels += 1
        level_extent *= 2
    return levels


def inspect_phase4_gate_fixture(path: Path, dtype: str) -> FixtureInspection:
    """Fail unless a reference fixture reaches every compressed PIZ structure required by the gate."""
    if dtype not in PHASE4_DTYPES:
        raise ValueError(f"unsupported Phase 4 dtype: {dtype!r}")
    container = io_header._parse_exr(path)
    if container.compression != "piz" or not container.piz_eligible:
        raise AssertionError(f"{path} is not an eligible PIZ Phase 4 fixture")
    descriptors = tuple(chunk.piz for chunk in container.chunks if chunk.piz is not None)
    compressed = tuple(descriptor for descriptor in descriptors if not descriptor.raw_stored)
    if not compressed:
        raise AssertionError(f"{path} contains no PIZ-compressed chunk")

    width = container.data_window[2] - container.data_window[0] + 1
    nonempty_bitmaps = 0
    maximum_wavelet_levels = 0
    huffman_tables = 0
    huffman_table_bytes = 0
    huffman_data_bytes = 0
    for descriptor in compressed:
        bitmap = container.data[descriptor.bitmap_span.start : descriptor.bitmap_span.end]
        nonempty_bitmaps += int(any(bitmap))
        maximum_wavelet_levels = max(maximum_wavelet_levels, _wavelet_level_count(width, descriptor.row_count))
        stream = container.data[descriptor.huffman_span.start : descriptor.huffman_span.end]
        table = exr_container._parse_piz_huffman_table(stream)
        huffman_tables += 1
        huffman_table_bytes += table.table_span.size
        huffman_data_bytes += table.data_span.size

    if nonempty_bitmaps == 0:
        raise AssertionError(f"{path} contains no nonempty PIZ bitmap")
    if maximum_wavelet_levels < 2:
        raise AssertionError(f"{path} does not exercise multiple PIZ wavelet levels")
    if huffman_tables == 0 or huffman_table_bytes == 0 or huffman_data_bytes == 0:
        raise AssertionError(f"{path} contains no complete PIZ Huffman table and data")
    return FixtureInspection(
        dtype=dtype,
        total_chunks=len(descriptors),
        compressed_chunks=len(compressed),
        nonempty_bitmaps=nonempty_bitmaps,
        maximum_wavelet_levels=maximum_wavelet_levels,
        huffman_tables=huffman_tables,
        huffman_table_bytes=huffman_table_bytes,
        huffman_data_bytes=huffman_data_bytes,
    )


def phase4_boundary_operation(
    inputs: Phase4PerformanceInputs,
    direction: str,
    backend: str,
    *,
    dtype: str,
) -> Callable[[], object]:
    """Bind an OpenEXR baseline or accepted self-hosted PIZ route to the shared fixture."""
    if dtype not in PHASE4_DTYPES:
        raise ValueError(f"unsupported Phase 4 dtype: {dtype!r}")
    if direction == "read":
        path = inputs.read_path(dtype)
        if backend == "cpu":
            return lambda: read_openexr_frame(path)
        return lambda: px.io.read_image(path, unchanged=True)
    if direction == "write":
        path = inputs.write_path(dtype)
        frame = inputs.frame(dtype)
        if backend == "cpu":
            return lambda: write_openexr_frame(path, frame, compression="piz", dwa_level=None)
        return lambda: px.io.write_image(path, frame, compression="piz")
    raise ValueError(f"unsupported gate direction: {direction!r}")


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


def measure_phase4_gate_case(
    inputs: Phase4PerformanceInputs,
    direction: str,
    *,
    dtype: str,
) -> GateMeasurement:
    """Measure every forced backend for one PIZ dtype/direction pair with source-fixed restoration."""
    backends = ("cpu", "custom_cpu", "gpu") if direction == "read" else ("cpu", "gpu")
    key = ("piz", direction)
    sentinel = object()
    original_backend: object = io._EXR_ROUTING.get(key, sentinel)
    medians_ms: dict[str, float] = {}
    iterations: dict[str, int] = {}
    synchronize = cp.cuda.Device().synchronize
    try:
        for backend in backends:
            if backend != "cpu":
                io._EXR_ROUTING[key] = backend
            operation = phase4_boundary_operation(inputs, direction, backend, dtype=dtype)
            _warmup(operation, synchronize)
            durations_ms = _measure(operation, synchronize)
            medians_ms[backend] = median(durations_ms)
            iterations[backend] = len(durations_ms)
    finally:
        if original_backend is sentinel:
            io._EXR_ROUTING.pop(key, None)
        else:
            io._EXR_ROUTING[key] = str(original_backend)
    return GateMeasurement(dtype=dtype, direction=direction, medians_ms=medians_ms, iterations=iterations)


def device_identity() -> DeviceIdentity:
    """Return the exact CUDA device identity recorded with a gate run."""
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
