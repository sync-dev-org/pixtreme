"""Shared EXR gate image corpus and unchanged timing of one backend boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np

import pixtreme as px
import pixtreme._io.formats.exr.selection as io

WIDTH = 1920
HEIGHT = 1080
WARMUP_MINIMUM_SECONDS = 0.5
MEASURED_MINIMUM_ITERATIONS = 20
MEASURED_MINIMUM_SECONDS = 3.0
_SEED = 20260809


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


def _gate_source_frames() -> tuple[px.core.Frame, px.core.Frame]:
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


def measure_exr_gate_case(
    compression: str,
    direction: str,
    operation_for_backend: Callable[[str], Callable[[], object]],
) -> GateMeasurement:
    """Measure one compression and direction with the same backend restoration and timing floors."""
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
            operation = operation_for_backend(backend)
            _warmup(operation, synchronize)
            durations_ms = _measure(operation, synchronize)
            medians_ms[backend] = median(durations_ms)
            iterations[backend] = len(durations_ms)
    finally:
        if original_backend is sentinel:
            io._EXR_ROUTING.pop(key, None)
        else:
            io._EXR_ROUTING[key] = str(original_backend)
    return GateMeasurement(compression=compression, direction=direction, medians_ms=medians_ms, iterations=iterations)


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
