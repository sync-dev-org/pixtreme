"""B44 and B44A EXR gate inputs, fixture inspection, and measurement entry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from exr_b44_gate_fixture import inspect_b44_chunks
from exr_gate_measurement import GateMeasurement, measure_exr_gate_case
from openexr_dev_oracle import read_frame as read_openexr_frame
from openexr_dev_oracle import write_frame as write_openexr_frame

import pixtreme as px
import pixtreme._io.header as io_header
from pixtreme._io.formats.exr.container import _container_gpu_eligible


@dataclass(frozen=True)
class B44PerformanceInputs:
    source_frame: px.core.Frame
    directory: Path

    def read_path(self, compression: str) -> Path:
        if compression not in ("b44", "b44a"):
            raise ValueError(f"B44 gate received compression={compression!r}")
        return self.directory / f"gate-read-{compression}.exr"

    def write_path(self, compression: str) -> Path:
        if compression not in ("b44", "b44a"):
            raise ValueError(f"B44 gate received compression={compression!r}")
        return self.directory / f"gate-write-{compression}.exr"

    def frame(self, compression: str) -> px.core.Frame:
        if compression not in ("b44", "b44a"):
            raise ValueError(f"B44 gate received compression={compression!r}")
        return self.source_frame


@dataclass(frozen=True)
class B44FixtureInspection:
    compression: str
    total_chunks: int
    compressed_chunks: int
    dense_blocks: int
    flat_blocks: int


def build_b44_performance_inputs(directory: Path, source_frame: px.core.Frame) -> B44PerformanceInputs:
    directory.mkdir(parents=True, exist_ok=True)
    inputs = B44PerformanceInputs(source_frame, directory)
    for compression in ("b44", "b44a"):
        write_openexr_frame(inputs.read_path(compression), source_frame, compression=compression, dwa_level=None)
    return inputs


def inspect_b44_gate_fixture(path: Path, compression: str) -> B44FixtureInspection:
    container = io_header._parse_exr(path)
    if container.compression != compression or not _container_gpu_eligible(container):
        raise AssertionError(f"{path} is not an eligible {compression.upper()} gate fixture")
    total_chunks, compressed_chunks, dense_blocks, flat_blocks = inspect_b44_chunks(container, compression)
    return B44FixtureInspection(compression, total_chunks, compressed_chunks, dense_blocks, flat_blocks)


def b44_boundary_operation(
    inputs: B44PerformanceInputs, compression: str, direction: str, backend: str
) -> Callable[[], object]:
    if direction == "read":
        path = inputs.read_path(compression)
        if backend == "cpu":
            return lambda: read_openexr_frame(path)
        return lambda: px.io.read_image(path, unchanged=True)
    if direction == "write":
        path = inputs.write_path(compression)
        if backend == "cpu":
            return lambda: write_openexr_frame(path, inputs.source_frame, compression=compression, dwa_level=None)
        return lambda: px.io.write_image(path, inputs.source_frame, compression=compression)
    raise ValueError(f"unsupported B44 gate direction: {direction!r}")


def measure_b44_gate_case(inputs: B44PerformanceInputs, compression: str, direction: str) -> GateMeasurement:
    if compression not in ("b44", "b44a"):
        raise ValueError(f"B44 gate received compression={compression!r}")
    return measure_exr_gate_case(
        compression, direction, lambda backend: b44_boundary_operation(inputs, compression, direction, backend)
    )
