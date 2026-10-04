"""PXR24 EXR gate inputs, fixture inspection, and measurement entry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from exr_gate_measurement import GateMeasurement, measure_exr_gate_case
from exr_pxr24_gate_fixture import inspect_pxr24_chunks
from openexr_dev_oracle import read_frame as read_openexr_frame
from openexr_dev_oracle import write_frame as write_openexr_frame

import pixtreme as px
import pixtreme._io.header as io_header
from pixtreme._io.formats.exr.container import _container_gpu_eligible


@dataclass(frozen=True)
class Pxr24PerformanceInputs:
    source_frame: px.core.Frame
    directory: Path

    def read_path(self, compression: str = "pxr24") -> Path:
        if compression != "pxr24":
            raise ValueError(f"PXR24 gate received compression={compression!r}")
        return self.directory / "gate-read-pxr24.exr"

    def write_path(self, compression: str = "pxr24") -> Path:
        if compression != "pxr24":
            raise ValueError(f"PXR24 gate received compression={compression!r}")
        return self.directory / "gate-write-pxr24.exr"

    def frame(self, compression: str = "pxr24") -> px.core.Frame:
        if compression != "pxr24":
            raise ValueError(f"PXR24 gate received compression={compression!r}")
        return self.source_frame


@dataclass(frozen=True)
class Pxr24FixtureInspection:
    compression: str
    total_chunks: int
    compressed_chunks: int
    pxr24_planes: int


def build_pxr24_performance_inputs(directory: Path, source_frame: px.core.Frame) -> Pxr24PerformanceInputs:
    directory.mkdir(parents=True, exist_ok=True)
    inputs = Pxr24PerformanceInputs(source_frame, directory)
    write_openexr_frame(inputs.read_path(), source_frame, compression="pxr24", dwa_level=None)
    return inputs


def inspect_pxr24_gate_fixture(path: Path, compression: str) -> Pxr24FixtureInspection:
    if compression != "pxr24":
        raise ValueError(f"PXR24 gate received compression={compression!r}")
    container = io_header._parse_exr(path)
    if container.compression != "pxr24" or not _container_gpu_eligible(container):
        raise AssertionError(f"{path} is not an eligible PXR24 gate fixture")
    total_chunks, compressed_chunks, plane_count = inspect_pxr24_chunks(container)
    return Pxr24FixtureInspection("pxr24", total_chunks, compressed_chunks, plane_count)


def pxr24_boundary_operation(inputs: Pxr24PerformanceInputs, direction: str, backend: str) -> Callable[[], object]:
    if direction == "read":
        path = inputs.read_path()
        if backend == "cpu":
            return lambda: read_openexr_frame(path)
        return lambda: px.io.read_image(path, unchanged=True)
    if direction == "write":
        path = inputs.write_path()
        if backend == "cpu":
            return lambda: write_openexr_frame(path, inputs.source_frame, compression="pxr24", dwa_level=None)
        return lambda: px.io.write_image(path, inputs.source_frame, compression="pxr24")
    raise ValueError(f"unsupported PXR24 gate direction: {direction!r}")


def measure_pxr24_gate_case(inputs: Pxr24PerformanceInputs, compression: str, direction: str) -> GateMeasurement:
    if compression != "pxr24":
        raise ValueError(f"PXR24 gate received compression={compression!r}")
    return measure_exr_gate_case(
        "pxr24", direction, lambda backend: pxr24_boundary_operation(inputs, direction, backend)
    )
