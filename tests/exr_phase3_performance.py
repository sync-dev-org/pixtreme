"""Reusable FHD fixture and boundary harness for the Phase 3 EXR performance gate."""

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
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

WIDTH = 1920
HEIGHT = 1080
WARMUP_MINIMUM_SECONDS = 0.5
MEASURED_MINIMUM_ITERATIONS = 20
MEASURED_MINIMUM_SECONDS = 3.0
PHASE3_COMPRESSIONS = ("rle", "pxr24", "b44", "b44a")
_SEED = 20260809


@dataclass(frozen=True)
class Phase3PerformanceInputs:
    """Shared source frames, reference files, and write destinations for all eight gate pairs."""

    fp32_frame: px.core.Frame
    fp16_frame: px.core.Frame
    directory: Path

    def read_path(self, compression: str) -> Path:
        return self.directory / f"phase3-read-{compression}.exr"

    def write_path(self, compression: str) -> Path:
        return self.directory / f"phase3-write-{compression}.exr"

    def frame(self, compression: str) -> px.core.Frame:
        return self.fp16_frame if compression in {"b44", "b44a"} else self.fp32_frame


@dataclass(frozen=True)
class FixtureInspection:
    compression: str
    total_chunks: int
    compressed_chunks: int
    rle_packets: int
    pxr24_planes: int
    dense_blocks: int
    flat_blocks: int


@dataclass(frozen=True)
class GateMeasurement:
    compression: str
    direction: str
    medians_ms: dict[str, float]
    iterations: dict[str, int]


@dataclass(frozen=True)
class DeviceIdentity:
    name: str
    driver_version: int
    runtime_version: int


def _phase3_source_frames() -> tuple[px.core.Frame, px.core.Frame]:
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
    # A constant stripe guarantees representative RLE packets without removing the gradient/detail corpus.
    data[64:96, :, :] = cp.asarray((0.125, 0.5, 1.25), dtype=cp.float32)
    fp32_frame = px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")

    half_data = data.astype(cp.float16)
    # Aligned constant 4x4 blocks coexist with dense gradient blocks so B44A exercises both wire forms.
    half_data[128:384, 256:768, :] = cp.asarray((0.25, 0.5, 1.5), dtype=cp.float16)
    fp16_frame = px.io.from_array(half_data, colorspace="ACEScg", gamma="linear", channels="RGB")
    return fp32_frame, fp16_frame


def build_phase3_performance_inputs(directory: Path) -> Phase3PerformanceInputs:
    """Create the deterministic source frames and OpenEXR-reference read fixtures."""
    directory.mkdir(parents=True, exist_ok=True)
    fp32_frame, fp16_frame = _phase3_source_frames()
    inputs = Phase3PerformanceInputs(fp32_frame=fp32_frame, fp16_frame=fp16_frame, directory=directory)
    for compression in PHASE3_COMPRESSIONS:
        write_openexr_frame(
            inputs.read_path(compression),
            inputs.frame(compression),
            compression=compression,
            dwa_level=None,
        )
    return inputs


def inspect_phase3_gate_fixture(path: Path, compression: str) -> FixtureInspection:
    """Fail unless a reference fixture reaches the codec-specific compressed structure required by the gate."""
    container = io_header._parse_exr(path)
    if container.compression != compression or not container.phase3_eligible:
        raise AssertionError(f"{path} is not an eligible {compression.upper()} Phase 3 fixture")
    descriptors = tuple(chunk.phase3 for chunk in container.chunks if chunk.phase3 is not None)
    compressed = tuple(descriptor for descriptor in descriptors if not descriptor.raw_stored)
    if not compressed:
        raise AssertionError(f"{path} contains no codec-compressed {compression.upper()} chunk")
    packet_count = sum(len(descriptor.packets) for descriptor in compressed)
    plane_count = sum(len(descriptor.planes) for descriptor in compressed)
    dense_blocks = sum(block.stored_size == 14 for descriptor in compressed for block in descriptor.blocks)
    flat_blocks = sum(block.stored_size == 3 for descriptor in compressed for block in descriptor.blocks)
    if compression == "rle" and packet_count == 0:
        raise AssertionError(f"{path} contains no compressed RLE packet")
    if compression == "pxr24" and plane_count == 0:
        raise AssertionError(f"{path} contains no compressed PXR24 plane")
    if compression == "b44" and dense_blocks == 0:
        raise AssertionError(f"{path} contains no dense B44 block")
    if compression == "b44a" and (dense_blocks == 0 or flat_blocks == 0):
        raise AssertionError(f"{path} must contain both dense and flat B44A blocks")
    return FixtureInspection(
        compression=compression,
        total_chunks=len(descriptors),
        compressed_chunks=len(compressed),
        rle_packets=packet_count,
        pxr24_planes=plane_count,
        dense_blocks=dense_blocks,
        flat_blocks=flat_blocks,
    )


def phase3_boundary_operation(
    inputs: Phase3PerformanceInputs,
    compression: str,
    direction: str,
    backend: str,
) -> Callable[[], object]:
    """Bind an OpenEXR baseline or accepted self-hosted route to the shared fixture."""
    if compression not in PHASE3_COMPRESSIONS:
        raise ValueError(f"unsupported Phase 3 compression: {compression!r}")
    if direction == "read":
        path = inputs.read_path(compression)
        if backend == "cpu":
            return lambda: read_openexr_frame(path)
        return lambda: px.io.read_image(path, unchanged=True)
    if direction == "write":
        path = inputs.write_path(compression)
        frame = inputs.frame(compression)
        if backend == "cpu":
            return lambda: write_openexr_frame(path, frame, compression=compression, dwa_level=None)
        return lambda: px.io.write_image(path, frame, compression=compression)
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


def measure_phase3_gate_case(
    inputs: Phase3PerformanceInputs,
    compression: str,
    direction: str,
) -> GateMeasurement:
    """Measure every forced backend for one codec/direction pair with source-fixed restoration."""
    backends = ("cpu", "custom_cpu", "gpu") if direction == "read" else ("cpu", "gpu")
    key = (compression, direction)
    sentinel = object()
    original_backend: object = io._EXR_ROUTING.get(key, sentinel)
    medians_ms: dict[str, float] = {}
    iterations: dict[str, int] = {}
    synchronize = cp.cuda.Device().synchronize
    try:
        for backend in backends:
            if backend != "cpu":
                io._EXR_ROUTING[key] = backend
            operation = phase3_boundary_operation(inputs, compression, direction, backend)
            _warmup(operation, synchronize)
            durations_ms = _measure(operation, synchronize)
            medians_ms[backend] = median(durations_ms)
            iterations[backend] = len(durations_ms)
    finally:
        if original_backend is sentinel:
            io._EXR_ROUTING.pop(key, None)
        else:
            io._EXR_ROUTING[key] = str(original_backend)
    return GateMeasurement(
        compression=compression,
        direction=direction,
        medians_ms=medians_ms,
        iterations=iterations,
    )


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
