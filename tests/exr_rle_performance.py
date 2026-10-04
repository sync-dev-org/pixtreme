"""RLE EXR gate inputs, fixture inspection, and measurement entry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from exr_gate_measurement import GateMeasurement, measure_exr_gate_case
from exr_rle_gate_fixture import inspect_rle_chunks
from openexr_dev_oracle import read_frame as read_openexr_frame
from openexr_dev_oracle import write_frame as write_openexr_frame

import pixtreme as px
import pixtreme._io.header as io_header
from pixtreme._io.formats.exr.container import _container_gpu_eligible


@dataclass(frozen=True)
class RlePerformanceInputs:
    source_frame: px.core.Frame
    directory: Path

    def read_path(self, compression: str = "rle") -> Path:
        if compression != "rle":
            raise ValueError(f"RLE gate received compression={compression!r}")
        return self.directory / "gate-read-rle.exr"

    def write_path(self, compression: str = "rle") -> Path:
        if compression != "rle":
            raise ValueError(f"RLE gate received compression={compression!r}")
        return self.directory / "gate-write-rle.exr"

    def frame(self, compression: str = "rle") -> px.core.Frame:
        if compression != "rle":
            raise ValueError(f"RLE gate received compression={compression!r}")
        return self.source_frame


@dataclass(frozen=True)
class RleFixtureInspection:
    compression: str
    total_chunks: int
    compressed_chunks: int
    rle_packets: int


def build_rle_performance_inputs(directory: Path, source_frame: px.core.Frame) -> RlePerformanceInputs:
    directory.mkdir(parents=True, exist_ok=True)
    inputs = RlePerformanceInputs(source_frame, directory)
    write_openexr_frame(inputs.read_path(), source_frame, compression="rle", dwa_level=None)
    return inputs


def inspect_rle_gate_fixture(path: Path, compression: str) -> RleFixtureInspection:
    if compression != "rle":
        raise ValueError(f"RLE gate received compression={compression!r}")
    container = io_header._parse_exr(path)
    if container.compression != "rle" or not _container_gpu_eligible(container):
        raise AssertionError(f"{path} is not an eligible RLE gate fixture")
    total_chunks, compressed_chunks, packet_count = inspect_rle_chunks(container)
    return RleFixtureInspection("rle", total_chunks, compressed_chunks, packet_count)


def rle_boundary_operation(inputs: RlePerformanceInputs, direction: str, backend: str) -> Callable[[], object]:
    if direction == "read":
        path = inputs.read_path()
        if backend == "cpu":
            return lambda: read_openexr_frame(path)
        return lambda: px.io.read_image(path, unchanged=True)
    if direction == "write":
        path = inputs.write_path()
        if backend == "cpu":
            return lambda: write_openexr_frame(path, inputs.source_frame, compression="rle", dwa_level=None)
        return lambda: px.io.write_image(path, inputs.source_frame, compression="rle")
    raise ValueError(f"unsupported RLE gate direction: {direction!r}")


def measure_rle_gate_case(inputs: RlePerformanceInputs, compression: str, direction: str) -> GateMeasurement:
    if compression != "rle":
        raise ValueError(f"RLE gate received compression={compression!r}")
    return measure_exr_gate_case("rle", direction, lambda backend: rle_boundary_operation(inputs, direction, backend))
