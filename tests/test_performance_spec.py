from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import cupy as cp
import exr_phase4_gate as phase4_gate
import numpy as np
import pytest
from exr_phase3_gate import (
    GateDecision,
    GateRun,
    synthesize_gate_decision,
)
from exr_phase3_performance import (
    PHASE3_COMPRESSIONS,
    Phase3PerformanceInputs,
    build_phase3_performance_inputs,
    device_identity,
    inspect_phase3_gate_fixture,
    measure_phase3_gate_case,
)
from exr_phase4_performance import (
    GateMeasurement as Phase4GateMeasurement,
)
from exr_phase4_performance import (
    Phase4PerformanceInputs,
    build_phase4_performance_inputs,
    inspect_phase4_gate_fixture,
    measure_phase4_gate_case,
)
from openexr_dev_oracle import read_frame as read_openexr_frame
from openexr_dev_oracle import write_frame as write_openexr_frame

import pixtreme as px
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

_WIDTH = 1920
_HEIGHT = 1080
_CHANNELS = 3
_WARMUP_MINIMUM_SECONDS = 0.5
_MEASURED_MINIMUM_FRAMES = 1000
_MEASURED_MINIMUM_SECONDS = 3.0
_BOUNDARY_MEASURED_MINIMUM_FRAMES = 20
_LUT_SIZE = 65
_SEED = 20260717
_FHD_FP32_Y_BYTES = _WIDTH * _HEIGHT * np.dtype(np.float32).itemsize
_FHD_FP16_RGB_BYTES = _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.float16).itemsize
_FHD_FP32_RGB_BYTES = _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.float32).itemsize
_FHD_FP32_RGBA_BYTES = _WIDTH * _HEIGHT * 4 * np.dtype(np.float32).itemsize
_FHD_FP16_RGB_READ_WRITE_BYTES = _FHD_FP16_RGB_BYTES * 2
_FHD_FP32_RGB_READ_WRITE_BYTES = _FHD_FP32_RGB_BYTES * 2
_FHD_FP32_TO_FP16_RGB_BYTES = _FHD_FP32_RGB_BYTES + _FHD_FP16_RGB_BYTES
_LUT_65_FP32_RGB_BYTES = _LUT_SIZE**3 * _CHANNELS * np.dtype(np.float32).itemsize
_LUT_65_FP32_RGBA_BYTES = _LUT_SIZE**3 * 4 * np.dtype(np.float32).itemsize
_LUT_1D_FP32_RGB_BYTES = _LUT_SIZE * _CHANNELS * np.dtype(np.float32).itemsize
_LUT_17_FP32_RGB_BYTES = 17**3 * _CHANNELS * np.dtype(np.float32).itemsize
_LUT_17_FP32_RGBA_BYTES = 17**3 * 4 * np.dtype(np.float32).itemsize

_COPY_KERNEL_SOURCE = r"""
extern "C" __global__ void pixtreme_performance_copy(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const long long element_count
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index < element_count) {
        destination[index] = source[index];
    }
}
"""

_PUBLIC_GPU_PIXEL_FUNCTIONS = frozenset(
    {
        "from_array",
        "to_array",
        "to_uyvy422",
        "to_v210",
        "to_nv12",
        "to_p010",
        "to_yuv420p",
        "to_yuv422p",
        "to_yuv444p",
        "to_yuva444p",
        "shuffle",
        "gamma_to_linear",
        "hsv_to_rgb",
        "linear_to_gamma",
        "rgb_to_grayscale",
        "rgb_to_hsv",
        "rgb_to_rgb",
        "rgb_to_ycbcr",
        "chromatic_adaptation",
        "white_balance",
        "white_point_simulation",
        "ycbcr_to_rgb",
        "ycbcr_to_ycbcr",
        "apply_lut",
        "resize",
        "warp_affine",
        "stack",
        "merge",
        "gaussian_blur",
        "unsharp_mask",
        "box_blur",
        "median_blur",
        "bilateral_blur",
        "convolve_box",
        "directional_blur",
        "zoom_blur",
        "spin_blur",
        "vector_blur",
        "lens_blur",
        "erosion",
        "dilation",
        "opening",
        "closing",
        "morphological_gradient",
        "white_tophat",
        "black_tophat",
        "sobel",
        "laplacian",
        "canny",
        "sharpen",
        "difference_of_gaussians",
        "equalize_histogram",
        "clahe",
        "line",
        "polyline",
        "rectangle",
        "circle",
        "ellipse",
        "polygon",
        "text",
        "ramp",
        "grid",
        "checkerboard",
        "color_bars",
        "fractal_noise",
        "turbulent_noise",
        "grain",
        "quantize",
        "dequantize",
        "legal_to_full",
        "full_to_legal",
        "cast_dtype",
        "recode_dtype",
        "from_uyvy422",
        "from_v210",
        "from_nv12",
        "from_p010",
        "from_yuv420p",
        "from_yuv422p",
        "from_yuv444p",
        "from_yuva444p",
        "corner_harris",
        "match_template",
        "psnr",
        "ssim",
        "ssim_map",
    }
)
_PERFORMANCE_BOUNDARY_FUNCTIONS = frozenset(
    {"read_image", "write_image", "read_header", "read_lut", "decode_lut", "write_lut", "decode_image", "encode_image"}
)
# ``channels`` only normalizes named channel tokens; it touches neither pixels nor a file/bytes boundary.
_NON_PIXEL_PUBLIC_FUNCTIONS = frozenset({"channels"})
_PERFORMANCE_FRAME_METHOD_EXCLUSIONS = frozenset()
_PUBLIC_OPERATION_MODULES = (
    px.core,
    px.io,
    px.color,
    px.filter,
    px.transform,
    px.draw,
    px.generate,
    px.morphology,
    px.metrics,
    px.feature,
    px.values,
    px.channel,
    px.composite,
)


@dataclass(frozen=True)
class _Inputs:
    frame: px.core.Frame
    exr_phase1_frame: px.core.Frame
    exr_phase3: Phase3PerformanceInputs
    exr_phase4: Phase4PerformanceInputs
    analysis_template: px.core.Frame
    lut: px.core.Lut
    lut1d: px.core.Lut1D
    stack_frames: tuple[px.core.Frame, ...]
    shuffle_reorder_outputs: dict[str, tuple[px.core.Frame, str] | float]
    shuffle_multi_outputs: dict[str, tuple[px.core.Frame, str] | float]
    shuffle_adapt_outputs: dict[str, tuple[px.core.Frame, str] | float]
    composite_foreground: px.core.Frame
    hsv_frame: px.core.Frame
    ycbcr_frame: px.core.Frame
    ycbcra_frame: px.core.Frame
    copy_source: cp.ndarray
    copy_destination: cp.ndarray
    chw_data: cp.ndarray
    chw_code10: cp.ndarray
    code8_frame: px.core.Frame
    encoded_png: bytes
    encoded_jpeg: bytes
    encoded_tiff: bytes
    encoded_jpeg2000: bytes
    encoded_webp: bytes
    encoded_bmp: bytes
    encoded_pnm: bytes
    decode_lut_cube1d: bytes
    decode_lut_cube3d: bytes
    decode_lut_3dl: bytes
    decode_lut_spi1d: bytes
    decode_lut_spi3d: bytes
    read_png_path: Path
    read_jpeg_path: Path
    read_tiff_path: Path
    read_exr_path: Path
    read_exr_none_path: Path
    read_exr_zip_path: Path
    read_exr_zips_path: Path
    read_exr_dwaa_path: Path
    read_exr_dwab_path: Path
    read_jpeg2000_path: Path
    read_webp_path: Path
    read_bmp_path: Path
    read_pnm_path: Path
    read_tga_path: Path
    read_hdr_path: Path
    read_dpx_path: Path
    read_lut_path: Path
    read_lut_cube1d_path: Path
    read_lut_3dl_path: Path
    read_lut_spi1d_path: Path
    read_lut_spi3d_path: Path
    write_lut1d_path: Path
    write_lut3d_path: Path
    write_png_path: Path
    write_jpeg_path: Path
    write_tiff_path: Path
    write_exr_path: Path
    write_exr_none_path: Path
    write_exr_zip_path: Path
    write_exr_zips_path: Path
    write_exr_dwaa_path: Path
    write_exr_dwab_path: Path
    write_jpeg2000_path: Path
    write_webp_path: Path
    write_bmp_path: Path
    write_pnm_path: Path
    write_tga_path: Path
    write_hdr_path: Path
    write_dpx_path: Path
    uyvy422: cp.ndarray
    v210: cp.ndarray
    nv12: cp.ndarray
    p010: cp.ndarray
    yuv420p: cp.ndarray
    yuv422p10: cp.ndarray
    yuv444p: cp.ndarray
    yuva444p: cp.ndarray
    vector_8: px.core.Frame
    vector_32: px.core.Frame
    vector_128: px.core.Frame
    vector_rotation_32: px.core.Frame


@dataclass(frozen=True)
class _PerformanceCase:
    case_id: str
    target: str
    parameters: str
    operation: Callable[..., object]
    input_attribute: str | None = "frame"
    kwargs: tuple[tuple[str, object], ...] = ()
    fixture_kwargs: tuple[tuple[str, str], ...] = ()
    kwargs_attribute: str | None = None
    transferred_bytes: int = _FHD_FP32_RGB_READ_WRITE_BYTES
    minimum_frames: int = _MEASURED_MINIMUM_FRAMES
    minimum_seconds: float = _MEASURED_MINIMUM_SECONDS

    def bind(self, inputs: _Inputs) -> Callable[[], object]:
        resolved_kwargs = dict(self.kwargs)
        resolved_kwargs.update(
            (parameter, getattr(inputs, input_attribute)) for parameter, input_attribute in self.fixture_kwargs
        )
        if self.kwargs_attribute is not None:
            resolved_kwargs.update(getattr(inputs, self.kwargs_attribute))
        if self.input_attribute is None:
            return lambda: self.operation(**resolved_kwargs)
        source = getattr(inputs, self.input_attribute)
        return lambda: self.operation(source, **resolved_kwargs)


@dataclass(frozen=True)
class _PerformanceMetrics:
    mean_ms: float
    median_ms: float
    fps: float
    p5_ms: float
    p95_ms: float


def _case(
    case_id: str,
    target: str,
    parameters: str,
    operation: Callable[..., object],
    *,
    input_attribute: str | None = "frame",
    kwargs: dict[str, object] | None = None,
    fixture_kwargs: dict[str, str] | None = None,
    kwargs_attribute: str | None = None,
    transferred_bytes: int = _FHD_FP32_RGB_READ_WRITE_BYTES,
    minimum_frames: int = _MEASURED_MINIMUM_FRAMES,
    minimum_seconds: float = _MEASURED_MINIMUM_SECONDS,
) -> _PerformanceCase:
    return _PerformanceCase(
        case_id=case_id,
        target=target,
        parameters=parameters,
        operation=operation,
        input_attribute=input_attribute,
        kwargs=tuple((kwargs or {}).items()),
        fixture_kwargs=tuple((fixture_kwargs or {}).items()),
        kwargs_attribute=kwargs_attribute,
        transferred_bytes=transferred_bytes,
        minimum_frames=minimum_frames,
        minimum_seconds=minimum_seconds,
    )


def _boundary_case(
    case_id: str,
    target: str,
    parameters: str,
    operation: Callable[..., object],
    *,
    input_attribute: str | None = "frame",
    kwargs: dict[str, object] | None = None,
    fixture_kwargs: dict[str, str] | None = None,
    transferred_bytes: int = _FHD_FP32_RGB_READ_WRITE_BYTES,
) -> _PerformanceCase:
    return _case(
        case_id,
        target,
        parameters,
        operation,
        input_attribute=input_attribute,
        kwargs=kwargs,
        fixture_kwargs=fixture_kwargs,
        transferred_bytes=transferred_bytes,
        minimum_frames=_BOUNDARY_MEASURED_MINIMUM_FRAMES,
    )


def _read_phase3_exr(inputs: Phase3PerformanceInputs, *, compression: str) -> px.core.Frame:
    return px.io.read_image(inputs.read_path(compression), unchanged=True)


def _write_phase3_exr(inputs: Phase3PerformanceInputs, *, compression: str) -> None:
    px.io.write_image(inputs.write_path(compression), inputs.frame(compression), compression=compression)


def _read_phase4_exr(inputs: Phase4PerformanceInputs) -> px.core.Frame:
    return px.io.read_image(inputs.read_path("fp16"), unchanged=True)


def _write_phase4_exr(inputs: Phase4PerformanceInputs) -> None:
    px.io.write_image(inputs.write_path("fp16"), inputs.frame("fp16"), compression="piz")


def _run_phase4_isolated_repeat(direction: str) -> Phase4GateMeasurement:
    """Run the acceptance-43 repeat in a fresh process with a bounded timeout and parse its final JSON line."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_exr_phase4_gate_repeat.py")), direction],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"Phase 4 isolated repeat failed with exit {completed.returncode}: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    json_lines = tuple(line for line in completed.stdout.splitlines() if line.startswith("{"))
    if not json_lines:
        raise AssertionError(f"Phase 4 isolated repeat produced no JSON payload: {completed.stdout!r}")
    payload = json.loads(json_lines[-1])
    measurement = payload["measurement"]
    return Phase4GateMeasurement(
        dtype=str(measurement["dtype"]),
        direction=str(measurement["direction"]),
        medians_ms={str(key): float(value) for key, value in measurement["medians_ms"].items()},
        iterations={str(key): int(value) for key, value in measurement["iterations"].items()},
    )


@lru_cache(maxsize=1)
def _copy_kernel() -> cp.RawKernel:
    return cp.RawKernel(_COPY_KERNEL_SOURCE, "pixtreme_performance_copy")


def _copy_f32_hwc(source: cp.ndarray, *, destination: cp.ndarray) -> cp.ndarray:
    element_count = source.size
    threads = 256
    blocks = (element_count + threads - 1) // threads
    _copy_kernel()((blocks,), (threads,), (source, destination, np.int64(element_count)))
    return destination


def _run_warmup(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    minimum_seconds: float,
    timer: Callable[[], float] = perf_counter,
) -> None:
    started_at = timer()
    elapsed_seconds = 0.0
    while elapsed_seconds < minimum_seconds:
        output = operation()
        synchronize()
        del output
        elapsed_seconds = timer() - started_at


def _measure_durations_ms(
    operation: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    minimum_frames: int,
    minimum_seconds: float,
    timer: Callable[[], float] = perf_counter,
) -> list[float]:
    synchronize()
    measurement_started_at = timer()
    elapsed_seconds = 0.0
    durations_ms: list[float] = []
    while len(durations_ms) < minimum_frames or elapsed_seconds < minimum_seconds:
        synchronize()
        frame_started_at = timer()
        output = operation()
        synchronize()
        frame_finished_at = timer()
        durations_ms.append((frame_finished_at - frame_started_at) * 1000.0)
        del output
        elapsed_seconds = frame_finished_at - measurement_started_at
    return durations_ms


def _assert_compressed_dwa_v2_output(path: Path) -> None:
    container = io_header._parse_exr(path)
    assert container.dwa_eligible, f"{path} is not eligible for the DWA v2 gate"
    assert any(
        not chunk.raw_stored
        and chunk.dwa is not None
        and chunk.dwa.leader is not None
        and chunk.dwa.leader.version == 2
        for chunk in container.chunks
    ), f"{path} contains no compressed DWA v2 chunk"


def _performance_metrics(durations_ms: list[float]) -> _PerformanceMetrics:
    median_ms = median(durations_ms)
    p5_ms, p95_ms = np.percentile(durations_ms, (5.0, 95.0), method="linear")
    return _PerformanceMetrics(
        mean_ms=fmean(durations_ms),
        median_ms=median_ms,
        fps=1000.0 / median_ms,
        p5_ms=float(p5_ms),
        p95_ms=float(p95_ms),
    )


_COPY_CASES = (
    _case(
        "copy-fhd-fp32-rgb",
        "copy",
        "FHD fp32 RGB read+write",
        _copy_f32_hwc,
        input_attribute="copy_source",
        fixture_kwargs={"destination": "copy_destination"},
    ),
)


_RESIZE_CASES = tuple(
    _case(
        f"resize-down-{interpolation}",
        "resize",
        f"1920x1080 -> 960x540, interpolation={interpolation}",
        px.transform.resize,
        kwargs={"width": 960, "height": 540, "interpolation": interpolation},
        transferred_bytes=_FHD_FP32_RGB_BYTES + 960 * 540 * _CHANNELS * np.dtype(np.float32).itemsize,
    )
    for interpolation in (
        "nearest",
        "bilinear",
        "bicubic",
        "b-spline",
        "mitchell",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "area",
    )
) + tuple(
    _case(
        f"resize-up-{interpolation}",
        "resize",
        f"1920x1080 -> 3840x2160, interpolation={interpolation}",
        px.transform.resize,
        kwargs={"width": 3840, "height": 2160, "interpolation": interpolation},
        transferred_bytes=_FHD_FP32_RGB_BYTES + 3840 * 2160 * _CHANNELS * np.dtype(np.float32).itemsize,
    )
    for interpolation in (
        "nearest",
        "bilinear",
        "bicubic",
        "b-spline",
        "mitchell",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "area",
    )
)

_WARP_ANGLE = math.radians(5.0)
_WARP_SCALE = 1.01
_WARP_CENTER_X = (_WIDTH - 1) / 2.0
_WARP_CENTER_Y = (_HEIGHT - 1) / 2.0
_WARP_LINEAR = np.asarray(
    [
        [_WARP_SCALE * math.cos(_WARP_ANGLE), -_WARP_SCALE * math.sin(_WARP_ANGLE)],
        [_WARP_SCALE * math.sin(_WARP_ANGLE), _WARP_SCALE * math.cos(_WARP_ANGLE)],
    ],
    dtype=np.float32,
)
_WARP_CENTER = np.asarray((_WARP_CENTER_X, _WARP_CENTER_Y), dtype=np.float32)
_WARP_TRANSLATION = _WARP_CENTER - _WARP_LINEAR @ _WARP_CENTER
_WARP_MATRIX = np.column_stack((_WARP_LINEAR, _WARP_TRANSLATION)).astype(np.float32)
_WARP_AFFINE_CASES = (
    _case(
        "warp-affine-fhd-auto-lanczos4",
        "warp_affine",
        "FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0",
        px.transform.warp_affine,
        kwargs={"matrix": _WARP_MATRIX},
    ),
)

_STACK_CASES = (
    _case(
        "stack-vertical-two-fhd",
        "stack",
        "2x FHD fp32 RGB, direction=vertical, adapt=False",
        px.transform.stack,
        input_attribute="stack_frames",
        kwargs={"direction": "vertical", "adapt": False},
        transferred_bytes=_FHD_FP32_RGB_BYTES * 4,
    ),
)

_SHUFFLE_CASES = (
    _case(
        "shuffle-reorder-fhd",
        "shuffle",
        "single FHD fp32 Frame BGR reorder, adapt=False",
        lambda **outputs: px.channel.shuffle(**outputs),
        input_attribute=None,
        kwargs_attribute="shuffle_reorder_outputs",
        transferred_bytes=_FHD_FP32_RGB_READ_WRITE_BYTES,
    ),
    _case(
        "shuffle-multi-fill-fhd",
        "shuffle",
        "FHD fp32 RGBA from 2 Frames + constant, adapt=False",
        lambda **outputs: px.channel.shuffle(**outputs),
        input_attribute=None,
        kwargs_attribute="shuffle_multi_outputs",
        transferred_bytes=_FHD_FP32_RGB_BYTES + _FHD_FP32_RGBA_BYTES,
    ),
    _case(
        "shuffle-adapt-fhd",
        "shuffle",
        "2 FHD fp32 RGB Frames, sRGB/sRGB source adapted to ACEScg/linear",
        lambda **outputs: px.channel.shuffle(**outputs),
        input_attribute=None,
        kwargs={"adapt": True},
        kwargs_attribute="shuffle_adapt_outputs",
        transferred_bytes=_FHD_FP32_RGB_BYTES * 4,
    ),
)

_COMPOSITE_CASES = (
    _case(
        "composite-transform-fhd",
        "merge",
        "FHD background + transformed 960x540 foreground, bilinear, normal",
        px.composite.merge,
        fixture_kwargs={"foreground": "composite_foreground"},
        kwargs={"scale": (1.25, 0.9), "rotation": 17.0},
        transferred_bytes=(
            _FHD_FP32_RGB_BYTES + 960 * 540 * _CHANNELS * np.dtype(np.float32).itemsize + _FHD_FP32_RGB_BYTES
        ),
    ),
)

_BLUR_CASES = (
    _case("gaussian-1", "gaussian_blur", "sigma=1", px.filter.gaussian_blur, kwargs={"sigma": 1.0}),
    _case("gaussian-2", "gaussian_blur", "sigma=2", px.filter.gaussian_blur, kwargs={"sigma": 2.0}),
    _case("gaussian-4", "gaussian_blur", "sigma=4", px.filter.gaussian_blur, kwargs={"sigma": 4.0}),
    _case(
        "unsharp-2-1",
        "unsharp_mask",
        "sigma=2, amount=1",
        px.filter.unsharp_mask,
        kwargs={"sigma": 2.0, "amount": 1.0},
    ),
    _case("box-3", "box_blur", "size=3", px.filter.box_blur, kwargs={"size": 3}),
    _case("box-9", "box_blur", "size=9", px.filter.box_blur, kwargs={"size": 9}),
    _case("median-3", "median_blur", "size=3", px.filter.median_blur, kwargs={"size": 3}),
    _case("median-5", "median_blur", "size=5", px.filter.median_blur, kwargs={"size": 5}),
    _case("median-7", "median_blur", "size=7", px.filter.median_blur, kwargs={"size": 7}),
    _case(
        "bilateral-1",
        "bilateral_blur",
        "sigma_space=1, sigma_value=0.1",
        px.filter.bilateral_blur,
        kwargs={"sigma_space": 1.0, "sigma_value": 0.1},
    ),
    _case(
        "bilateral-2",
        "bilateral_blur",
        "sigma_space=2, sigma_value=0.1",
        px.filter.bilateral_blur,
        kwargs={"sigma_space": 2.0, "sigma_value": 0.1},
    ),
    _case(
        "convolve-box-1x31",
        "convolve_box",
        "size=(1,31), normalize=True",
        px.filter.convolve_box,
        kwargs={"size": (1, 31), "normalize": True},
    ),
)

_MORPHOLOGY_CASES = tuple(
    _case(
        f"morphology-{name.removeprefix('morphology_')}-5",
        name,
        "radius=5, shape=disk",
        getattr(px.morphology, name),
        kwargs={"radius": 5, "shape": "disk"},
    )
    for name in (
        "erosion",
        "dilation",
        "opening",
        "closing",
        "morphological_gradient",
        "white_tophat",
        "black_tophat",
    )
)

_DERIVATIVE_CASES = (
    *(
        _case(
            f"sobel-{direction}",
            "sobel",
            f"direction={direction}",
            px.filter.sobel,
            kwargs={"direction": direction},
        )
        for direction in ("x", "y", "magnitude")
    ),
    _case("laplacian-3x3", "laplacian", "kernel=3x3", px.filter.laplacian),
    _case(
        "canny-0.5-1-mirror",
        "canny",
        "threshold_low=0.5, threshold_high=1.0, border=mirror",
        px.filter.canny,
        kwargs={"threshold_low": 0.5, "threshold_high": 1.0, "border": "mirror"},
    ),
    _case(
        "sharpen-1-mirror",
        "sharpen",
        "amount=1, border=mirror",
        px.filter.sharpen,
        kwargs={"amount": 1.0, "border": "mirror"},
    ),
    _case(
        "dog-1-2",
        "difference_of_gaussians",
        "sigma1=1, sigma2=2",
        px.filter.difference_of_gaussians,
        kwargs={"sigma1": 1.0, "sigma2": 2.0},
    ),
)

_ANALYSIS_CASES = (
    _case(
        "corner-harris-3-004-mirror",
        "corner_harris",
        "FHD fp32 RGB, block_size=3, k=0.04, border=mirror",
        px.feature.corner_harris,
        kwargs={"block_size": 3, "k": 0.04, "border": "mirror"},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _FHD_FP32_Y_BYTES,
    ),
    _case(
        "match-template-64-ccoeff-normed",
        "match_template",
        "FHD fp32 RGB + 64x64 fp32 RGB, method=ccoeff_normed",
        px.feature.match_template,
        fixture_kwargs={"template": "analysis_template"},
        kwargs={"method": "ccoeff_normed"},
        transferred_bytes=(
            _FHD_FP32_RGB_BYTES
            + 64 * 64 * _CHANNELS * np.dtype(np.float32).itemsize
            + (1920 - 64 + 1) * (1080 - 64 + 1) * np.dtype(np.float32).itemsize
        ),
    ),
    _case(
        "quality-psnr-default",
        "psnr",
        "FHD fp32 RGB reference/candidate, data_range=1.0 default",
        px.metrics.psnr,
        fixture_kwargs={"candidate": "frame"},
        transferred_bytes=2 * _FHD_FP32_RGB_BYTES + np.dtype(np.float32).itemsize,
    ),
    _case(
        "quality-ssim-default",
        "ssim",
        "FHD fp32 RGB reference/candidate, data_range=1.0 default",
        px.metrics.ssim,
        fixture_kwargs={"candidate": "frame"},
        transferred_bytes=2 * _FHD_FP32_RGB_BYTES + np.dtype(np.float32).itemsize,
    ),
    _case(
        "quality-ssim-map-default",
        "ssim_map",
        "FHD fp32 RGB reference/candidate, data_range=1.0 default",
        px.metrics.ssim_map,
        fixture_kwargs={"candidate": "frame"},
        transferred_bytes=(2 * _FHD_FP32_RGB_BYTES + (_WIDTH - 10) * (_HEIGHT - 10) * np.dtype(np.float32).itemsize),
    ),
)

_HISTOGRAM_CASES = (
    _case(
        "equalize-histogram-1024",
        "equalize_histogram",
        "domain=(0,1), bins=1024",
        px.color.equalize_histogram,
        kwargs={"domain": (0.0, 1.0), "bins": 1024},
    ),
    _case(
        "clahe-2-8x8-1024",
        "clahe",
        "clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024",
        px.color.clahe,
        kwargs={"clip_limit": 2.0, "tiles_y": 8, "tiles_x": 8, "domain": (0.0, 1.0), "bins": 1024},
    ),
)

_PATH_BLUR_CASES = (
    _case(
        "directional-8",
        "directional_blur",
        "angle=30, length=8",
        px.filter.directional_blur,
        kwargs={"angle": 30.0, "length": 8.0},
    ),
    _case(
        "directional-32",
        "directional_blur",
        "angle=30, length=32",
        px.filter.directional_blur,
        kwargs={"angle": 30.0, "length": 32.0},
    ),
    _case(
        "directional-128",
        "directional_blur",
        "angle=30, length=128",
        px.filter.directional_blur,
        kwargs={"angle": 30.0, "length": 128.0},
    ),
    _case("zoom-005", "zoom_blur", "amount=0.05", px.filter.zoom_blur, kwargs={"amount": 0.05}),
    _case("zoom-02", "zoom_blur", "amount=0.2", px.filter.zoom_blur, kwargs={"amount": 0.2}),
    _case("spin-2", "spin_blur", "angle=2", px.filter.spin_blur, kwargs={"angle": 2.0}),
    _case("spin-10", "spin_blur", "angle=10", px.filter.spin_blur, kwargs={"angle": 10.0}),
)

_VECTOR_BLUR_CASES = (
    _case(
        "vector-uniform-8",
        "vector_blur",
        "uniform |v|=8, shutter=centered",
        px.filter.vector_blur,
        fixture_kwargs={"vector": "vector_8"},
        transferred_bytes=_FHD_FP32_RGB_BYTES
        + _WIDTH * _HEIGHT * 2 * np.dtype(np.float32).itemsize
        + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "vector-uniform-32",
        "vector_blur",
        "uniform |v|=32, shutter=centered",
        px.filter.vector_blur,
        fixture_kwargs={"vector": "vector_32"},
        transferred_bytes=_FHD_FP32_RGB_BYTES
        + _WIDTH * _HEIGHT * 2 * np.dtype(np.float32).itemsize
        + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "vector-uniform-128",
        "vector_blur",
        "uniform |v|=128, shutter=centered",
        px.filter.vector_blur,
        fixture_kwargs={"vector": "vector_128"},
        transferred_bytes=_FHD_FP32_RGB_BYTES
        + _WIDTH * _HEIGHT * 2 * np.dtype(np.float32).itemsize
        + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "vector-rotation-32",
        "vector_blur",
        "rotation field, corner |v|=32, shutter=centered",
        px.filter.vector_blur,
        fixture_kwargs={"vector": "vector_rotation_32"},
        transferred_bytes=_FHD_FP32_RGB_BYTES
        + _WIDTH * _HEIGHT * 2 * np.dtype(np.float32).itemsize
        + _FHD_FP32_RGB_BYTES,
    ),
)

_LENS_BLUR_CASES = (
    _case("lens-circle-4", "lens_blur", "circle radius=4", px.filter.lens_blur, kwargs={"radius": 4.0}),
    _case("lens-circle-8", "lens_blur", "circle radius=8", px.filter.lens_blur, kwargs={"radius": 8.0}),
    _case("lens-circle-16", "lens_blur", "circle radius=16", px.filter.lens_blur, kwargs={"radius": 16.0}),
    _case("lens-circle-32", "lens_blur", "circle radius=32", px.filter.lens_blur, kwargs={"radius": 32.0}),
    _case(
        "lens-hexagon-16",
        "lens_blur",
        "blades=6, radius=16",
        px.filter.lens_blur,
        kwargs={"radius": 16.0, "blades": 6},
    ),
    _case(
        "lens-hexagon-32",
        "lens_blur",
        "blades=6, radius=32",
        px.filter.lens_blur,
        kwargs={"radius": 32.0, "blades": 6},
    ),
)

_DRAW_CASES = (
    _case(
        "draw-line-diagonal",
        "line",
        "diagonal thickness=4, aa=distance",
        px.draw.line,
        kwargs={
            "start": (64.0, 64.0),
            "end": (1856.0, 1016.0),
            "color": (1.0, 0.25, -0.1),
            "thickness": 4.0,
        },
    ),
    _case(
        "draw-polyline-closed",
        "polyline",
        "5 points, closed, thickness=6, aa=distance",
        px.draw.polyline,
        kwargs={
            "points": ((240.0, 180.0), (960.0, 80.0), (1680.0, 180.0), (1440.0, 900.0), (480.0, 900.0)),
            "color": (0.2, 1.2, 0.4),
            "thickness": 6.0,
            "closed": True,
        },
    ),
    _case(
        "draw-rectangle-fill",
        "rectangle",
        "1280x720 fill, corner_radius=48, aa=distance",
        px.draw.rectangle,
        kwargs={
            "top_left": (320.0, 180.0),
            "bottom_right": (1600.0, 900.0),
            "color": (0.1, 0.6, 1.4),
            "fill": True,
            "corner_radius": 48.0,
        },
    ),
    _case(
        "draw-circle-supersample",
        "circle",
        "fill radius=320, aa=supersample",
        px.draw.circle,
        kwargs={
            "center": (960.0, 540.0),
            "radius": 320.0,
            "color": (1.0, 0.5, 0.0),
            "fill": True,
            "aa": "supersample",
        },
    ),
    _case(
        "draw-ellipse-outline",
        "ellipse",
        "radii=(520,260), rotation=25, thickness=8",
        px.draw.ellipse,
        kwargs={
            "center": (960.0, 540.0),
            "radii": (520.0, 260.0),
            "rotation": 25.0,
            "color": (-0.1, 0.8, 1.5),
            "thickness": 8.0,
        },
    ),
    _case(
        "draw-polygon-even-odd",
        "polygon",
        "8-point concave fill, aa=distance",
        px.draw.polygon,
        kwargs={
            "points": (
                (960.0, 80.0),
                (1120.0, 380.0),
                (1680.0, 320.0),
                (1280.0, 580.0),
                (1480.0, 980.0),
                (960.0, 740.0),
                (440.0, 980.0),
                (640.0, 580.0),
            ),
            "color": (0.9, 0.2, 1.1),
        },
    ),
    _case(
        "draw-text-cjk-outline",
        "text",
        "single-line CJK, size=64, one outline, supersample=False",
        px.draw.text,
        kwargs={
            "text": "pixtreme 文字描画",
            "position": (960.0, 540.0),
            "size": 64.0,
            "color": (1.2, 0.5, -0.1),
            "anchor": "center-center",
            "outlines": (((0.05, 0.1, 0.2), 2.0),),
            "supersample": False,
        },
    ),
    _case(
        "draw-text-cjk-outline-supersample",
        "text",
        "single-line CJK, size=64, one outline, supersample=True",
        px.draw.text,
        kwargs={
            "text": "pixtreme 文字描画",
            "position": (960.0, 540.0),
            "size": 64.0,
            "color": (1.2, 0.5, -0.1),
            "anchor": "center-center",
            "outlines": (((0.05, 0.1, 0.2), 2.0),),
            "supersample": True,
        },
    ),
)

_GENERATOR_CASES = (
    _case(
        "generate-ramp-linear",
        "ramp",
        "FHD linear RGB",
        px.generate.ramp,
        input_attribute=None,
        kwargs={
            "width": _WIDTH,
            "height": _HEIGHT,
            "start": (0.0, 0.0),
            "end": (float(_WIDTH), float(_HEIGHT)),
            "start_color": (-0.1, 0.0, 0.25),
            "end_color": (1.0, 1.25, 2.0),
            "colorspace": "ACEScg",
        },
        transferred_bytes=_FHD_FP32_RGB_BYTES,
    ),
    _case(
        "generate-grid-distance",
        "grid",
        "FHD cell=(64,64), line_width=2, aa=distance",
        px.generate.grid,
        input_attribute=None,
        kwargs={
            "width": _WIDTH,
            "height": _HEIGHT,
            "cell": (64.0, 64.0),
            "line_width": 2.0,
            "color": (1.0, 0.25, -0.1),
            "background": (0.0, 0.0, 0.0),
            "colorspace": "ACEScg",
        },
        transferred_bytes=_FHD_FP32_RGB_BYTES,
    ),
    _case(
        "generate-checkerboard-distance",
        "checkerboard",
        "FHD cell=(64,64), aa=distance",
        px.generate.checkerboard,
        input_attribute=None,
        kwargs={
            "width": _WIDTH,
            "height": _HEIGHT,
            "cell": (64.0, 64.0),
            "colors": ((1.0, 0.25, -0.1), (0.0, 0.0, 0.0)),
            "colorspace": "ACEScg",
        },
        transferred_bytes=_FHD_FP32_RGB_BYTES,
    ),
    _case(
        "generate-color-bars-arib",
        "color_bars",
        "FHD ARIB STD-B28 normalized",
        px.generate.color_bars,
        input_attribute=None,
        kwargs={
            "width": _WIDTH,
            "height": _HEIGHT,
            "standard": "ARIB-STD-B28",
        },
        transferred_bytes=_FHD_FP32_RGB_BYTES,
    ),
)

_NOISE_CASES = (
    _case(
        "generate-fractal-noise",
        "fractal_noise",
        "FHD scale=64, octaves=4",
        px.generate.fractal_noise,
        input_attribute=None,
        kwargs={
            "width": _WIDTH,
            "height": _HEIGHT,
            "scale": 64.0,
            "colorspace": "ACEScg",
        },
        transferred_bytes=_FHD_FP32_Y_BYTES,
    ),
    _case(
        "generate-turbulent-noise",
        "turbulent_noise",
        "FHD scale=64, octaves=4",
        px.generate.turbulent_noise,
        input_attribute=None,
        kwargs={
            "width": _WIDTH,
            "height": _HEIGHT,
            "scale": 64.0,
            "colorspace": "ACEScg",
        },
        transferred_bytes=_FHD_FP32_Y_BYTES,
    ),
    _case(
        "generate-color-grain",
        "grain",
        "FHD intensity=0.1, size=1, RGB",
        px.generate.grain,
        input_attribute=None,
        kwargs={
            "width": _WIDTH,
            "height": _HEIGHT,
            "monochromatic": False,
            "colorspace": "ACEScg",
        },
        transferred_bytes=_FHD_FP32_RGB_BYTES,
    ),
)

_TRANSFORM_BOUNDARY_CASES = (
    _case(
        "from-array-fused",
        "from_array",
        "CHW + affine scale=255 -> float32 HWC",
        px.io.from_array,
        input_attribute="chw_data",
        kwargs={
            "colorspace": "ACEScg",
            "gamma": "linear",
            "channels": "RGB",
            "layout": "CHW",
            "dtype": "float32",
            "scale": 255.0,
        },
    ),
    _case(
        "from-array-bit-depth-10",
        "from_array",
        "CHW uint16, bit_depth=10 -> float32 HWC",
        px.io.from_array,
        input_attribute="chw_code10",
        kwargs={
            "colorspace": "ACEScg",
            "gamma": "linear",
            "channels": "RGB",
            "layout": "CHW",
            "bit_depth": 10,
        },
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint16).itemsize + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "to-array-fused",
        "to_array",
        "BGR + NCHW + float16 + affine",
        px.io.to_array,
        kwargs={
            "channels": "BGR",
            "layout": "NCHW",
            "dtype": "float16",
            "scale": (255.0, 255.0, 255.0),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.float16).itemsize,
    ),
    _case(
        "to-array-bit-depth-10",
        "to_array",
        "bit_depth=10 -> uint16 HWC",
        px.io.to_array,
        kwargs={"bit_depth": 10},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint16).itemsize,
    ),
    _case(
        "color-acescg-srgb",
        "rgb_to_rgb",
        "ACEScg linear -> sRGB sRGB",
        px.color.rgb_to_rgb,
        kwargs={"output_colorspace": "sRGB", "output_gamma": "sRGB"},
    ),
    _case(
        "color-aces13-analytic-srgb",
        "rgb_to_rgb",
        "ACES 1.3 analytic -> sRGB sRGB",
        px.color.rgb_to_rgb,
        kwargs={"output_colorspace": "sRGB", "output_gamma": "sRGB", "tonemap": "ACES-1.3"},
    ),
    _case(
        "color-aces20-analytic-srgb",
        "rgb_to_rgb",
        "ACES 2.0 analytic -> sRGB sRGB",
        px.color.rgb_to_rgb,
        kwargs={"output_colorspace": "sRGB", "output_gamma": "sRGB", "tonemap": "ACES-2.0"},
    ),
    _case(
        "color-bt2408-rec2020-pq",
        "rgb_to_rgb",
        "BT.2408 direct mapping -> Rec.2020 pq",
        px.color.rgb_to_rgb,
        kwargs={"output_colorspace": "Rec.2020", "output_gamma": "PQ", "tonemap": "BT.2408"},
    ),
    _case(
        "color-chromatic-adaptation",
        "chromatic_adaptation",
        "FHD fp32 RGB, D50 input -> D60 output, CAT02",
        px.color.chromatic_adaptation,
        kwargs={"input_white": (0.34567, 0.35850), "output_white": (0.32168, 0.33767)},
    ),
    _case(
        "color-white-balance",
        "white_balance",
        "FHD fp32 RGB, Temperature=5000 K, Tint=0 Duv, CAT02",
        px.color.white_balance,
        kwargs={"temperature": 5000.0},
    ),
    _case(
        "color-white-point-simulation",
        "white_point_simulation",
        "FHD fp32 RGB, D65 input display -> D93 output display",
        px.color.white_point_simulation,
        kwargs={"input_white": "D65", "output_white": "D93"},
    ),
    _case(
        "color-rgb-ycbcr",
        "rgb_to_ycbcr",
        "RGB -> YCbCr, matrix=native",
        px.color.rgb_to_ycbcr,
        kwargs={"matrix": "native"},
    ),
    _case(
        "color-rgb-hsv",
        "rgb_to_hsv",
        "RGB -> HSV, label-driven scene values",
        px.color.rgb_to_hsv,
    ),
    _case(
        "color-hsv-rgb",
        "hsv_to_rgb",
        "HSV six sectors, S=[0,1], V=[0,2] -> RGB",
        px.color.hsv_to_rgb,
        input_attribute="hsv_frame",
    ),
    _case(
        "color-ycbcr-rgb",
        "ycbcr_to_rgb",
        "YCbCr -> RGB, matrix=bt709",
        px.color.ycbcr_to_rgb,
        input_attribute="ycbcr_frame",
        kwargs={"matrix": "BT.709"},
    ),
    _case(
        "color-rgb-grayscale",
        "rgb_to_grayscale",
        "RGB -> Y, matrix=native",
        px.color.rgb_to_grayscale,
        kwargs={"matrix": "native"},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _FHD_FP32_Y_BYTES,
    ),
    _case(
        "color-gamma-linear",
        "gamma_to_linear",
        "gamma=Gamma-2.6 claim -> linear",
        px.color.gamma_to_linear,
        kwargs={"gamma": "Gamma-2.6"},
    ),
    _case(
        "color-linear-gamma",
        "linear_to_gamma",
        "linear -> gamma=Gamma-2.6",
        px.color.linear_to_gamma,
        kwargs={"gamma": "Gamma-2.6"},
    ),
    _case(
        "color-ycbcr-ycbcr",
        "ycbcr_to_ycbcr",
        "YCbCr bt709 -> native rematrix",
        px.color.ycbcr_to_ycbcr,
        input_attribute="ycbcr_frame",
        kwargs={"input_matrix": "BT.709", "output_matrix": "native"},
    ),
    _case(
        "full-to-legal-10",
        "full_to_legal",
        "full -> legal, bit_depth=10",
        px.values.full_to_legal,
        kwargs={"bit_depth": 10},
    ),
    _case(
        "legal-to-full-10",
        "legal_to_full",
        "legal -> full, bit_depth=10",
        px.values.legal_to_full,
        kwargs={"bit_depth": 10},
    ),
    _case(
        "quantize-values-8",
        "quantize",
        "float32 -> uint8, bit_depth=8",
        px.values.quantize,
        kwargs={"bit_depth": 8},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize,
    ),
    _case(
        "dequantize-values-8",
        "dequantize",
        "uint8 -> float32, bit_depth=8",
        px.values.dequantize,
        input_attribute="code8_frame",
        kwargs={"bit_depth": 8},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "cast-f32-f16",
        "cast_dtype",
        "float32 -> float16",
        px.values.cast_dtype,
        kwargs={"dtype": "float16"},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.float16).itemsize,
    ),
    _case(
        "recode-u8-f32",
        "recode_dtype",
        "uint8 -> float32",
        px.values.recode_dtype,
        input_attribute="code8_frame",
        kwargs={"dtype": "float32"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "recode-f32-u8",
        "recode_dtype",
        "float32 -> uint8",
        px.values.recode_dtype,
        kwargs={"dtype": "uint8"},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize,
    ),
    _case(
        "from-uyvy422",
        "from_uyvy422",
        "legal range",
        px.io.from_uyvy422,
        input_attribute="uyvy422",
        kwargs={"width": _WIDTH, "height": _HEIGHT},
        transferred_bytes=_WIDTH * _HEIGHT * 2 * np.dtype(np.uint8).itemsize + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "from-v210",
        "from_v210",
        "legal range",
        px.io.from_v210,
        input_attribute="v210",
        kwargs={"width": _WIDTH, "height": _HEIGHT},
        transferred_bytes=((_WIDTH + 47) // 48) * 32 * _HEIGHT * np.dtype(np.uint32).itemsize + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "from-nv12",
        "from_nv12",
        "legal range, siting=left, interpolation=bilinear",
        px.io.from_nv12,
        input_attribute="nv12",
        kwargs={"width": _WIDTH, "height": _HEIGHT},
        transferred_bytes=(_WIDTH * _HEIGHT + _WIDTH * _HEIGHT // 2) * np.dtype(np.uint8).itemsize
        + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "from-p010",
        "from_p010",
        "legal range, siting=left, interpolation=bilinear",
        px.io.from_p010,
        input_attribute="p010",
        kwargs={"width": _WIDTH, "height": _HEIGHT},
        transferred_bytes=(_WIDTH * _HEIGHT + _WIDTH * _HEIGHT // 2) * np.dtype(np.uint16).itemsize
        + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "from-yuv420p",
        "from_yuv420p",
        "legal range, interpolation=bilinear",
        px.io.from_yuv420p,
        input_attribute="yuv420p",
        kwargs={"width": _WIDTH, "height": _HEIGHT, "interpolation": "bilinear"},
        transferred_bytes=(_WIDTH * _HEIGHT + _WIDTH * _HEIGHT // 2) * np.dtype(np.uint8).itemsize
        + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "from-yuv422p-10",
        "from_yuv422p",
        "legal range",
        px.io.from_yuv422p,
        input_attribute="yuv422p10",
        kwargs={"width": _WIDTH, "height": _HEIGHT, "bit_depth": 10},
        transferred_bytes=_WIDTH * _HEIGHT * 2 * np.dtype(np.uint16).itemsize + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "from-yuv444p",
        "from_yuv444p",
        "10-bit legal range",
        px.io.from_yuv444p,
        input_attribute="yuv444p",
        kwargs={"width": _WIDTH, "height": _HEIGHT},
        transferred_bytes=_WIDTH * _HEIGHT * 3 * np.dtype(np.uint16).itemsize + _FHD_FP32_RGB_BYTES,
    ),
    _case(
        "from-yuva444p",
        "from_yuva444p",
        "12-bit legal range",
        px.io.from_yuva444p,
        input_attribute="yuva444p",
        kwargs={"width": _WIDTH, "height": _HEIGHT},
        transferred_bytes=_WIDTH * _HEIGHT * 4 * np.dtype(np.uint16).itemsize + _FHD_FP32_RGBA_BYTES,
    ),
)

_LUT_CASES = (
    _case(
        "lut-transform-trilinear",
        "apply_lut",
        "FHD fp32 RGB, 65^3 LUT, interpolation=trilinear",
        px.color.apply_lut,
        kwargs={"interpolation": "trilinear"},
        fixture_kwargs={"lut": "lut"},
    ),
    _case(
        "lut-transform-tetrahedral",
        "apply_lut",
        "FHD fp32 RGB, 65^3 LUT, interpolation=tetrahedral",
        px.color.apply_lut,
        kwargs={"interpolation": "tetrahedral"},
        fixture_kwargs={"lut": "lut"},
    ),
    _case(
        "lut-transform-linear-1d",
        "apply_lut",
        "FHD fp32 RGB, 65-sample 1D LUT, interpolation=linear",
        px.color.apply_lut,
        kwargs={"interpolation": "linear"},
        fixture_kwargs={"lut": "lut1d"},
    ),
)

_TO_FORMAT_CASES = (
    _case(
        "to-uyvy422",
        "to_uyvy422",
        "FHD area, legal",
        px.io.to_uyvy422,
        input_attribute="ycbcr_frame",
        kwargs={},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 2,
    ),
    _case(
        "to-v210",
        "to_v210",
        "FHD area, legal, 128-byte rows",
        px.io.to_v210,
        input_attribute="ycbcr_frame",
        kwargs={},
        transferred_bytes=_FHD_FP32_RGB_BYTES + ((_WIDTH + 47) // 48) * 32 * _HEIGHT * np.dtype(np.uint32).itemsize,
    ),
    _case(
        "to-nv12",
        "to_nv12",
        "FHD area, legal, siting=left",
        px.io.to_nv12,
        input_attribute="ycbcr_frame",
        kwargs={},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 3 // 2,
    ),
    _case(
        "to-p010",
        "to_p010",
        "FHD area, legal, siting=left",
        px.io.to_p010,
        input_attribute="ycbcr_frame",
        kwargs={},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 3 * np.dtype(np.uint16).itemsize // 2,
    ),
    _case(
        "to-yuv420p",
        "to_yuv420p",
        "8-bit area, legal, siting=left",
        px.io.to_yuv420p,
        input_attribute="ycbcr_frame",
        kwargs={},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 3 // 2,
    ),
    _case(
        "to-yuv422p",
        "to_yuv422p",
        "10-bit area, legal",
        px.io.to_yuv422p,
        input_attribute="ycbcr_frame",
        kwargs={"bit_depth": 10},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 2 * np.dtype(np.uint16).itemsize,
    ),
    _case(
        "to-yuv444p",
        "to_yuv444p",
        "10-bit legal",
        px.io.to_yuv444p,
        input_attribute="ycbcr_frame",
        kwargs={},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 3 * np.dtype(np.uint16).itemsize,
    ),
    _case(
        "to-yuva444p",
        "to_yuva444p",
        "12-bit legal, alpha full",
        px.io.to_yuva444p,
        input_attribute="ycbcra_frame",
        kwargs={},
        transferred_bytes=_FHD_FP32_RGBA_BYTES + _WIDTH * _HEIGHT * 4 * np.dtype(np.uint16).itemsize,
    ),
)

# Boundary measurements are end-to-end wall-clock observations, separate from GPU pixel-operation throughput.
# File cases include temporary-file I/O (without clearing OS caches); bytes cases include host bytes exchange.
# Their 20-sample floor limits repeated disk writes and slow CPU codec work while retaining the 3-second time floor.
_FILE_BOUNDARY_CASES = (
    _boundary_case(
        "file-read-lut-cube-65",
        "read_lut",
        "65^3 RGB .cube file, parse, float4 packing, and host-to-device transfer included",
        px.io.read_lut,
        input_attribute="read_lut_path",
        transferred_bytes=_LUT_65_FP32_RGB_BYTES + _LUT_65_FP32_RGBA_BYTES,
    ),
    _boundary_case(
        "file-read-lut-cube-1d",
        "read_lut",
        "65-sample RGB Cube 1D file, parse and host-to-device transfer included",
        px.io.read_lut,
        input_attribute="read_lut_cube1d_path",
        transferred_bytes=_LUT_1D_FP32_RGB_BYTES * 2,
    ),
    _boundary_case(
        "file-read-lut-3dl",
        "read_lut",
        "17^3 RGB headerless 3DL file, parse, packing, and host-to-device transfer included",
        px.io.read_lut,
        input_attribute="read_lut_3dl_path",
        transferred_bytes=_LUT_17_FP32_RGB_BYTES + _LUT_17_FP32_RGBA_BYTES,
    ),
    _boundary_case(
        "file-read-lut-spi1d",
        "read_lut",
        "65-sample RGB SPI1D file, parse and host-to-device transfer included",
        px.io.read_lut,
        input_attribute="read_lut_spi1d_path",
        transferred_bytes=_LUT_1D_FP32_RGB_BYTES * 2,
    ),
    _boundary_case(
        "file-read-lut-spi3d",
        "read_lut",
        "17^3 RGB SPI3D file, explicit-index parse, packing, and host-to-device transfer included",
        px.io.read_lut,
        input_attribute="read_lut_spi3d_path",
        transferred_bytes=_LUT_17_FP32_RGB_BYTES + _LUT_17_FP32_RGBA_BYTES,
    ),
    _boundary_case(
        "file-write-lut-1d",
        "write_lut",
        "65-sample RGB Lut1D, device-to-host transfer and Cube file write included",
        px.io.write_lut,
        input_attribute="write_lut1d_path",
        fixture_kwargs={"lut": "lut1d"},
        transferred_bytes=_LUT_1D_FP32_RGB_BYTES * 2,
    ),
    _boundary_case(
        "file-write-lut-3d",
        "write_lut",
        "65^3 RGB Lut, device-to-host transfer and Cube file write included",
        px.io.write_lut,
        input_attribute="write_lut3d_path",
        fixture_kwargs={"lut": "lut"},
        transferred_bytes=_LUT_65_FP32_RGB_BYTES * 2,
    ),
    _boundary_case(
        "file-read-png",
        "read_image",
        "FHD uint8 RGB PNG file, unchanged, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_png_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-jpeg",
        "read_image",
        "FHD uint8 RGB JPEG file, unchanged, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_jpeg_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-tiff",
        "read_image",
        "FHD uint8 RGB TIFF file, unchanged, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_tiff_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-exr",
        "read_image",
        "FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_exr_path",
        kwargs={"unchanged": True},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase1-read-none",
        "read_image",
        "FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_exr_none_path",
        kwargs={"unchanged": True},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase1-read-zip",
        "read_image",
        "FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_exr_zip_path",
        kwargs={"unchanged": True},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase1-read-zips",
        "read_image",
        "FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_exr_zips_path",
        kwargs={"unchanged": True},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase2-read-dwaa",
        "read_image",
        "FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_exr_dwaa_path",
        kwargs={"unchanged": True},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase2-read-dwab",
        "read_image",
        "FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_exr_dwab_path",
        kwargs={"unchanged": True},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-read-rle",
        "read_image",
        "FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included",
        _read_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "rle"},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-read-pxr24",
        "read_image",
        "FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
        _read_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "pxr24"},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-read-b44",
        "read_image",
        "FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included",
        _read_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "b44"},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-read-b44a",
        "read_image",
        "FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included",
        _read_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "b44a"},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase4-read-piz",
        "read_image",
        "FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included",
        _read_phase4_exr,
        input_attribute="exr_phase4",
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-read-jpeg2000",
        "read_image",
        "FHD uint8 RGB JPEG 2000 file, unchanged, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_jpeg2000_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-webp",
        "read_image",
        "FHD uint8 RGB WebP file, unchanged, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_webp_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-bmp",
        "read_image",
        "FHD uint8 RGB BMP file, unchanged, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_bmp_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-pnm",
        "read_image",
        "FHD uint8 RGB PNM file, unchanged, temporary-file I/O included",
        px.io.read_image,
        input_attribute="read_pnm_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-tga",
        "read_image",
        "FHD uint8 RGB TGA file, unchanged, temporary-file I/O and CPU RLE included",
        px.io.read_image,
        input_attribute="read_tga_path",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-read-hdr",
        "read_image",
        "FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included",
        px.io.read_image,
        input_attribute="read_hdr_path",
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 4 * np.dtype(np.uint8).itemsize,
    ),
    _boundary_case(
        "file-read-dpx",
        "read_image",
        "FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU unpack included",
        px.io.read_image,
        input_attribute="read_dpx_path",
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 4 * np.dtype(np.uint8).itemsize,
    ),
    _boundary_case(
        "file-write-png",
        "write_image",
        "FHD uint8 RGB PNG file, compression_level=4, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_png_path",
        fixture_kwargs={"frame": "code8_frame"},
        kwargs={"compression_level": 4},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-jpeg",
        "write_image",
        "FHD uint8 RGB JPEG file, quality=95, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_jpeg_path",
        fixture_kwargs={"frame": "code8_frame"},
        kwargs={"quality": 95},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-tiff",
        "write_image",
        "FHD uint8 RGB TIFF file, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_tiff_path",
        fixture_kwargs={"frame": "code8_frame"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-exr",
        "write_image",
        "FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_exr_path",
        fixture_kwargs={"frame": "frame"},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase1-write-none",
        "write_image",
        "FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_exr_none_path",
        fixture_kwargs={"frame": "exr_phase1_frame"},
        kwargs={"compression": "none"},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase1-write-zip",
        "write_image",
        "FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_exr_zip_path",
        fixture_kwargs={"frame": "exr_phase1_frame"},
        kwargs={"compression": "zip"},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase1-write-zips",
        "write_image",
        "FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_exr_zips_path",
        fixture_kwargs={"frame": "exr_phase1_frame"},
        kwargs={"compression": "zips"},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase2-write-dwaa",
        "write_image",
        "FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_exr_dwaa_path",
        fixture_kwargs={"frame": "exr_phase1_frame"},
        kwargs={"compression": "dwaa", "dwa_level": 45.0},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase2-write-dwab",
        "write_image",
        "FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_exr_dwab_path",
        fixture_kwargs={"frame": "exr_phase1_frame"},
        kwargs={"compression": "dwab", "dwa_level": 45.0},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-write-rle",
        "write_image",
        "FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        _write_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "rle"},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-write-pxr24",
        "write_image",
        "FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        _write_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "pxr24"},
        transferred_bytes=_FHD_FP32_TO_FP16_RGB_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-write-b44",
        "write_image",
        "FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        _write_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "b44"},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase3-write-b44a",
        "write_image",
        "FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        _write_phase3_exr,
        input_attribute="exr_phase3",
        kwargs={"compression": "b44a"},
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-exr-phase4-write-piz",
        "write_image",
        "FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
        _write_phase4_exr,
        input_attribute="exr_phase4",
        transferred_bytes=_FHD_FP16_RGB_READ_WRITE_BYTES,
    ),
    _boundary_case(
        "file-write-jpeg2000",
        "write_image",
        "FHD uint8 RGB JPEG 2000 file, lossless, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_jpeg2000_path",
        fixture_kwargs={"frame": "code8_frame"},
        kwargs={"lossless": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-webp",
        "write_image",
        "FHD uint8 RGB WebP file, lossless, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_webp_path",
        fixture_kwargs={"frame": "code8_frame"},
        kwargs={"lossless": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-bmp",
        "write_image",
        "FHD uint8 RGB BMP file, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_bmp_path",
        fixture_kwargs={"frame": "code8_frame"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-pnm",
        "write_image",
        "FHD uint8 RGB PNM file, temporary-file I/O included",
        px.io.write_image,
        input_attribute="write_pnm_path",
        fixture_kwargs={"frame": "code8_frame"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-tga",
        "write_image",
        "FHD uint8 RGB TGA file, temporary-file I/O and CPU RLE included",
        px.io.write_image,
        input_attribute="write_tga_path",
        fixture_kwargs={"frame": "code8_frame"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "file-write-hdr",
        "write_image",
        "FHD fp32 RGB HDR file, temporary-file I/O and CPU RLE included",
        px.io.write_image,
        input_attribute="write_hdr_path",
        fixture_kwargs={"frame": "frame"},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 4 * np.dtype(np.uint8).itemsize,
    ),
    _boundary_case(
        "file-write-dpx",
        "write_image",
        "FHD fp32 RGB 10-bit DPX file, temporary-file I/O and GPU packing included",
        px.io.write_image,
        input_attribute="write_dpx_path",
        fixture_kwargs={"frame": "frame"},
        kwargs={"bit_depth": 10},
        transferred_bytes=_FHD_FP32_RGB_BYTES + _WIDTH * _HEIGHT * 4 * np.dtype(np.uint8).itemsize,
    ),
    _boundary_case(
        "file-read-header-png",
        "read_header",
        "FHD uint8 RGB PNG header, temporary-file I/O included",
        px.io.read_header,
        input_attribute="read_png_path",
        transferred_bytes=0,
    ),
)

_BYTES_BOUNDARY_CASES = (
    _boundary_case(
        "bytes-decode-lut-cube-1d",
        "decode_lut",
        "65-sample RGB Cube 1D UTF-8 bytes, sniff, parse, and host-to-device transfer included",
        px.io.decode_lut,
        input_attribute="decode_lut_cube1d",
        transferred_bytes=_LUT_1D_FP32_RGB_BYTES * 2,
    ),
    _boundary_case(
        "bytes-decode-lut-cube-3d",
        "decode_lut",
        "65^3 RGB Cube 3D UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included",
        px.io.decode_lut,
        input_attribute="decode_lut_cube3d",
        transferred_bytes=_LUT_65_FP32_RGB_BYTES + _LUT_65_FP32_RGBA_BYTES,
    ),
    _boundary_case(
        "bytes-decode-lut-3dl",
        "decode_lut",
        "17^3 RGB headerless 3DL UTF-8 bytes, sniff, parse, packing, and host-to-device transfer included",
        px.io.decode_lut,
        input_attribute="decode_lut_3dl",
        transferred_bytes=_LUT_17_FP32_RGB_BYTES + _LUT_17_FP32_RGBA_BYTES,
    ),
    _boundary_case(
        "bytes-decode-lut-spi1d",
        "decode_lut",
        "65-sample RGB SPI1D UTF-8 bytes, sniff, parse, and host-to-device transfer included",
        px.io.decode_lut,
        input_attribute="decode_lut_spi1d",
        transferred_bytes=_LUT_1D_FP32_RGB_BYTES * 2,
    ),
    _boundary_case(
        "bytes-decode-lut-spi3d",
        "decode_lut",
        "17^3 RGB SPI3D UTF-8 bytes, sniff, explicit-index parse, and host-to-device transfer included",
        px.io.decode_lut,
        input_attribute="decode_lut_spi3d",
        transferred_bytes=_LUT_17_FP32_RGB_BYTES + _LUT_17_FP32_RGBA_BYTES,
    ),
    _boundary_case(
        "bytes-decode-png",
        "decode_image",
        "FHD uint8 RGB PNG, unchanged, host bytes exchange included",
        px.io.decode_image,
        input_attribute="encoded_png",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-decode-jpeg",
        "decode_image",
        "FHD uint8 RGB JPEG, unchanged, host bytes exchange included",
        px.io.decode_image,
        input_attribute="encoded_jpeg",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-decode-tiff",
        "decode_image",
        "FHD uint8 RGB TIFF, unchanged, host bytes exchange included",
        px.io.decode_image,
        input_attribute="encoded_tiff",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-decode-jpeg2000",
        "decode_image",
        "FHD uint8 RGB JPEG 2000, unchanged, host bytes exchange included",
        px.io.decode_image,
        input_attribute="encoded_jpeg2000",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-decode-webp",
        "decode_image",
        "FHD uint8 RGB WebP, unchanged, host bytes exchange included",
        px.io.decode_image,
        input_attribute="encoded_webp",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-decode-bmp",
        "decode_image",
        "FHD uint8 RGB BMP, unchanged, host bytes exchange included",
        px.io.decode_image,
        input_attribute="encoded_bmp",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-decode-pnm",
        "decode_image",
        "FHD uint8 RGB PNM, unchanged, host bytes exchange included",
        px.io.decode_image,
        input_attribute="encoded_pnm",
        kwargs={"unchanged": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-encode-png",
        "encode_image",
        "FHD uint8 RGB PNG, compression_level=4, host bytes exchange included",
        px.io.encode_image,
        input_attribute="code8_frame",
        kwargs={"format": "png", "compression_level": 4},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-encode-jpeg",
        "encode_image",
        "FHD uint8 RGB JPEG, quality=95, host bytes exchange included",
        px.io.encode_image,
        input_attribute="code8_frame",
        kwargs={"format": "jpeg", "quality": 95},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-encode-tiff",
        "encode_image",
        "FHD uint8 RGB TIFF, host bytes exchange included",
        px.io.encode_image,
        input_attribute="code8_frame",
        kwargs={"format": "tiff"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-encode-jpeg2000",
        "encode_image",
        "FHD uint8 RGB JPEG 2000, lossless, host bytes exchange included",
        px.io.encode_image,
        input_attribute="code8_frame",
        kwargs={"format": "jpeg2000", "lossless": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-encode-webp",
        "encode_image",
        "FHD uint8 RGB WebP, lossless, host bytes exchange included",
        px.io.encode_image,
        input_attribute="code8_frame",
        kwargs={"format": "webp", "lossless": True},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-encode-bmp",
        "encode_image",
        "FHD uint8 RGB BMP, host bytes exchange included",
        px.io.encode_image,
        input_attribute="code8_frame",
        kwargs={"format": "bmp"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
    _boundary_case(
        "bytes-encode-pnm",
        "encode_image",
        "FHD uint8 RGB PNM, host bytes exchange included",
        px.io.encode_image,
        input_attribute="code8_frame",
        kwargs={"format": "pnm"},
        transferred_bytes=_WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.uint8).itemsize * 2,
    ),
)

_PERFORMANCE_CASES = (
    *_COPY_CASES,
    *_RESIZE_CASES,
    *_WARP_AFFINE_CASES,
    *_STACK_CASES,
    *_SHUFFLE_CASES,
    *_COMPOSITE_CASES,
    *_BLUR_CASES,
    *_MORPHOLOGY_CASES,
    *_DERIVATIVE_CASES,
    *_ANALYSIS_CASES,
    *_HISTOGRAM_CASES,
    *_PATH_BLUR_CASES,
    *_VECTOR_BLUR_CASES,
    *_LENS_BLUR_CASES,
    *_DRAW_CASES,
    *_GENERATOR_CASES,
    *_NOISE_CASES,
    *_TRANSFORM_BOUNDARY_CASES,
    *_LUT_CASES,
    *_TO_FORMAT_CASES,
    *_FILE_BOUNDARY_CASES,
    *_BYTES_BOUNDARY_CASES,
)


def _uniform_vector(length: float) -> px.core.Frame:
    data = cp.empty((_HEIGHT, _WIDTH, 2), dtype=cp.float32)
    data[..., 0] = np.float32(length)
    data[..., 1] = np.float32(0.0)
    return px.io.from_array(data, colorspace="sRGB", gamma="linear", channels=("X", "Y"))


def _rotation_vector() -> px.core.Frame:
    data = cp.empty((_HEIGHT, _WIDTH, 2), dtype=cp.float32)
    center_x = (_WIDTH - 1) / 2.0
    center_y = (_HEIGHT - 1) / 2.0
    corner_radius = math.hypot(center_x, center_y)
    scale = np.float32(32.0 / corner_radius)
    data[..., 0] = -(cp.arange(_HEIGHT, dtype=cp.float32)[:, None] - np.float32(center_y)) * scale
    data[..., 1] = (cp.arange(_WIDTH, dtype=cp.float32)[None, :] - np.float32(center_x)) * scale
    return px.io.from_array(data, colorspace="sRGB", gamma="linear", channels=("X", "Y"))


@pytest.fixture(scope="session")
def performance_inputs(tmp_path_factory: pytest.TempPathFactory) -> _Inputs:
    generator = cp.random.default_rng(_SEED)
    data = generator.random((_HEIGHT, _WIDTH, _CHANNELS), dtype=cp.float32)
    frame = px.io.from_array(data, colorspace="ACEScg", gamma="linear", channels="RGB")
    exr_generator = cp.random.default_rng(_SEED + 21)
    exr_x = cp.arange(_WIDTH, dtype=cp.float32)[None, :] / np.float32(_WIDTH - 1)
    exr_y = cp.arange(_HEIGHT, dtype=cp.float32)[:, None] / np.float32(_HEIGHT - 1)
    exr_detail = exr_generator.integers(0, 16, size=(_HEIGHT, _WIDTH), dtype=cp.uint8).astype(cp.float32)
    exr_detail *= np.float32(1.0 / 4096.0)
    exr_data = cp.stack(
        (
            cp.broadcast_to(exr_x, (_HEIGHT, _WIDTH)) + exr_detail,
            cp.broadcast_to(exr_y, (_HEIGHT, _WIDTH)) - exr_detail,
            (exr_x + exr_y) * np.float32(0.5) + exr_detail,
        ),
        axis=2,
    )
    exr_phase1_frame = px.io.from_array(exr_data, colorspace="ACEScg", gamma="linear", channels="RGB")
    hsv_data = cp.empty_like(data)
    hsv_columns = cp.arange(_WIDTH, dtype=cp.float32)[None, :]
    hsv_rows = cp.arange(_HEIGHT, dtype=cp.float32)[:, None]
    hsv_data[..., 0] = (cp.mod(hsv_columns, np.float32(6.0)) + np.float32(0.5)) / np.float32(6.0)
    hsv_data[..., 1] = hsv_rows / np.float32(_HEIGHT - 1)
    hsv_data[..., 2] = np.float32(2.0) * hsv_columns / np.float32(_WIDTH - 1)
    hsv_frame = px.io.from_array(data=hsv_data, colorspace="ACEScg", gamma="linear", channels="HSV")
    analysis_template = px.io.from_array(
        generator.random((64, 64, _CHANNELS), dtype=cp.float32),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    assemble_source = px.io.from_array(
        generator.random((_HEIGHT, _WIDTH, _CHANNELS), dtype=cp.float32),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    shuffle_reorder_outputs: dict[str, tuple[px.core.Frame, str] | float] = {
        "B": (frame, "B"),
        "G": (frame, "G"),
        "R": (frame, "R"),
    }
    shuffle_multi_outputs: dict[str, tuple[px.core.Frame, str] | float] = {
        "R": (frame, "R"),
        "G": (frame, "G"),
        "B": (assemble_source, "B"),
        "A": 1.0,
    }
    shuffle_adapt_source = px.io.from_array(
        generator.random((_HEIGHT, _WIDTH, _CHANNELS), dtype=cp.float32),
        colorspace="sRGB",
        gamma="sRGB",
        channels="RGB",
    )
    shuffle_adapt_outputs: dict[str, tuple[px.core.Frame, str] | float] = {
        "R": (frame, "R"),
        "G": (shuffle_adapt_source, "G"),
        "B": (shuffle_adapt_source, "B"),
    }
    composite_foreground = px.io.from_array(
        generator.random((540, 960, _CHANNELS), dtype=cp.float32),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    ycbcr_frame = px.io.from_array(data, colorspace="Rec.709", gamma="Rec.709", channels="YCbCr")
    data4 = generator.random((_HEIGHT, _WIDTH, 4), dtype=cp.float32)
    ycbcra_frame = px.io.from_array(data4, colorspace="Rec.709", gamma="Rec.709", channels=("Y", "Cb", "Cr", "A"))
    chw_data = cp.ascontiguousarray(data.transpose(2, 0, 1))
    chw_code10 = generator.integers(0, 1024, size=(_CHANNELS, _HEIGHT, _WIDTH), dtype=cp.uint16)
    code8_data = generator.integers(0, 256, size=(_HEIGHT, _WIDTH, _CHANNELS), dtype=cp.uint8)
    code8_frame = px.io.from_array(code8_data, colorspace="ACEScg", gamma="linear", channels="RGB")
    io_directory = tmp_path_factory.mktemp("performance-io")
    exr_phase3 = build_phase3_performance_inputs(io_directory / "exr-phase3")
    for compression in ("rle", "pxr24"):
        px.io.write_image(
            exr_phase3.read_path(compression),
            exr_phase3.fp32_frame,
            compression=compression,
            dtype="float16",
        )
    exr_phase4 = build_phase4_performance_inputs(io_directory / "exr-phase4")
    read_png_path = io_directory / "read.png"
    read_jpeg_path = io_directory / "read.jpg"
    read_tiff_path = io_directory / "read.tiff"
    read_exr_path = io_directory / "read.exr"
    read_exr_none_path = io_directory / "read-none.exr"
    read_exr_zip_path = io_directory / "read-zip.exr"
    read_exr_zips_path = io_directory / "read-zips.exr"
    read_exr_dwaa_path = io_directory / "read-dwaa.exr"
    read_exr_dwab_path = io_directory / "read-dwab.exr"
    read_jpeg2000_path = io_directory / "read.jp2"
    read_webp_path = io_directory / "read.webp"
    read_bmp_path = io_directory / "read.bmp"
    read_pnm_path = io_directory / "read.pnm"
    read_tga_path = io_directory / "read.tga"
    read_hdr_path = io_directory / "read.hdr"
    read_dpx_path = io_directory / "read.dpx"
    read_lut_path = io_directory / "read.cube"
    read_lut_cube1d_path = io_directory / "read-1d.cube"
    read_lut_3dl_path = io_directory / "read.3dl"
    read_lut_spi1d_path = io_directory / "read.spi1d"
    read_lut_spi3d_path = io_directory / "read.spi3d"
    write_lut1d_path = io_directory / "write-1d.cube"
    write_lut3d_path = io_directory / "write-3d.cube"
    px.io.write_image(read_png_path, code8_frame, compression_level=4)
    px.io.write_image(read_jpeg_path, code8_frame, quality=95)
    px.io.write_image(read_tiff_path, code8_frame)
    px.io.write_image(read_exr_path, frame)
    px.io.write_image(read_exr_none_path, exr_phase1_frame, compression="none")
    px.io.write_image(read_exr_zip_path, exr_phase1_frame, compression="zip")
    px.io.write_image(read_exr_zips_path, exr_phase1_frame, compression="zips")
    assert any(not chunk.raw_stored for chunk in io_header._parse_exr(read_exr_zip_path).chunks)
    assert any(not chunk.raw_stored for chunk in io_header._parse_exr(read_exr_zips_path).chunks)
    px.io.write_image(read_exr_dwaa_path, exr_phase1_frame, compression="dwaa", dwa_level=45.0, dtype="float16")
    px.io.write_image(read_exr_dwab_path, exr_phase1_frame, compression="dwab", dwa_level=45.0, dtype="float16")
    for dwa_path in (read_exr_dwaa_path, read_exr_dwab_path):
        _assert_compressed_dwa_v2_output(dwa_path)
    px.io.write_image(read_jpeg2000_path, code8_frame, lossless=True)
    px.io.write_image(read_webp_path, code8_frame, lossless=True)
    px.io.write_image(read_bmp_path, code8_frame)
    px.io.write_image(read_pnm_path, code8_frame)
    px.io.write_image(read_tga_path, code8_frame)
    px.io.write_image(read_hdr_path, frame)
    px.io.write_image(read_dpx_path, frame, bit_depth=10)
    blue, green, red = np.indices((_LUT_SIZE, _LUT_SIZE, _LUT_SIZE), dtype=np.float32)
    cube_rows = np.stack((red, green, blue), axis=-1).reshape(-1, 3) / np.float32(_LUT_SIZE - 1)
    with read_lut_path.open("w", encoding="utf-8") as stream:
        stream.write(f"LUT_3D_SIZE {_LUT_SIZE}\n")
        np.savetxt(stream, cube_rows, fmt="%.8g")
    lut_axis = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32)
    lut1d_rows = np.stack((lut_axis, lut_axis**2, 1.0 - lut_axis), axis=-1)
    with read_lut_cube1d_path.open("w", encoding="utf-8") as stream:
        stream.write(f"LUT_1D_SIZE {_LUT_SIZE}\n")
        np.savetxt(stream, lut1d_rows, fmt="%.8g")

    three_dl_edge = 17
    three_dl_spacing = np.rint(np.linspace(0.0, 4095.0, three_dl_edge)).astype(np.int64)
    three_dl_red, three_dl_green, three_dl_blue = np.indices(
        (three_dl_edge, three_dl_edge, three_dl_edge), dtype=np.float64
    )
    three_dl_rows = np.rint(
        np.stack((three_dl_red, three_dl_green, three_dl_blue), axis=-1).reshape(-1, 3) * (4095.0 / (three_dl_edge - 1))
    ).astype(np.int64)
    with read_lut_3dl_path.open("w", encoding="utf-8") as stream:
        np.savetxt(stream, three_dl_spacing[None, :], fmt="%d")
        np.savetxt(stream, three_dl_rows, fmt="%d")

    with read_lut_spi1d_path.open("w", encoding="utf-8") as stream:
        stream.write(f"Version 1\nFrom 0 1\nLength {_LUT_SIZE}\nComponents 3\n{{\n")
        np.savetxt(stream, lut1d_rows, fmt="%.8g")
        stream.write("}\n")

    spi3d_indices = np.stack((three_dl_red, three_dl_green, three_dl_blue), axis=-1).reshape(-1, 3)
    spi3d_outputs = spi3d_indices / (three_dl_edge - 1)
    spi3d_rows = np.concatenate((spi3d_indices, spi3d_outputs), axis=1)
    with read_lut_spi3d_path.open("w", encoding="utf-8") as stream:
        stream.write(f"SPILUT 1.0\n3 3\n{three_dl_edge} {three_dl_edge} {three_dl_edge}\n")
        np.savetxt(stream, spi3d_rows, fmt=("%d", "%d", "%d", "%.8g", "%.8g", "%.8g"))
    lut = px.io.read_lut(read_lut_path)
    assert isinstance(lut, px.core.Lut)
    lut1d = px.io.read_lut(read_lut_cube1d_path)
    assert isinstance(lut1d, px.core.Lut1D)
    decode_lut_cube1d = read_lut_cube1d_path.read_bytes()
    decode_lut_cube3d = read_lut_path.read_bytes()
    decode_lut_3dl = read_lut_3dl_path.read_bytes()
    decode_lut_spi1d = read_lut_spi1d_path.read_bytes()
    decode_lut_spi3d = read_lut_spi3d_path.read_bytes()
    encoded_png = px.io.encode_image(code8_frame, format="png", compression_level=4)
    encoded_jpeg = px.io.encode_image(code8_frame, format="jpeg", quality=95)
    encoded_tiff = px.io.encode_image(code8_frame, format="tiff")
    encoded_jpeg2000 = px.io.encode_image(code8_frame, format="jpeg2000", lossless=True)
    encoded_webp = px.io.encode_image(code8_frame, format="webp", lossless=True)
    encoded_bmp = px.io.encode_image(code8_frame, format="bmp")
    encoded_pnm = px.io.encode_image(code8_frame, format="pnm")
    pixel_count = _WIDTH * _HEIGHT
    v210_word_count = ((_WIDTH + 47) // 48) * 32 * _HEIGHT
    p010 = generator.integers(0, 1024, size=pixel_count + pixel_count // 2, dtype=cp.uint16)
    p010 <<= np.uint16(6)
    return _Inputs(
        frame=frame,
        exr_phase1_frame=exr_phase1_frame,
        exr_phase3=exr_phase3,
        exr_phase4=exr_phase4,
        analysis_template=analysis_template,
        lut=lut,
        lut1d=lut1d,
        stack_frames=(frame, assemble_source),
        shuffle_reorder_outputs=shuffle_reorder_outputs,
        shuffle_multi_outputs=shuffle_multi_outputs,
        shuffle_adapt_outputs=shuffle_adapt_outputs,
        composite_foreground=composite_foreground,
        hsv_frame=hsv_frame,
        ycbcr_frame=ycbcr_frame,
        ycbcra_frame=ycbcra_frame,
        copy_source=data,
        copy_destination=cp.empty_like(data),
        chw_data=chw_data,
        chw_code10=chw_code10,
        code8_frame=code8_frame,
        encoded_png=encoded_png,
        encoded_jpeg=encoded_jpeg,
        encoded_tiff=encoded_tiff,
        encoded_jpeg2000=encoded_jpeg2000,
        encoded_webp=encoded_webp,
        encoded_bmp=encoded_bmp,
        encoded_pnm=encoded_pnm,
        decode_lut_cube1d=decode_lut_cube1d,
        decode_lut_cube3d=decode_lut_cube3d,
        decode_lut_3dl=decode_lut_3dl,
        decode_lut_spi1d=decode_lut_spi1d,
        decode_lut_spi3d=decode_lut_spi3d,
        read_png_path=read_png_path,
        read_jpeg_path=read_jpeg_path,
        read_tiff_path=read_tiff_path,
        read_exr_path=read_exr_path,
        read_exr_none_path=read_exr_none_path,
        read_exr_zip_path=read_exr_zip_path,
        read_exr_zips_path=read_exr_zips_path,
        read_exr_dwaa_path=read_exr_dwaa_path,
        read_exr_dwab_path=read_exr_dwab_path,
        read_jpeg2000_path=read_jpeg2000_path,
        read_webp_path=read_webp_path,
        read_bmp_path=read_bmp_path,
        read_pnm_path=read_pnm_path,
        read_tga_path=read_tga_path,
        read_hdr_path=read_hdr_path,
        read_dpx_path=read_dpx_path,
        read_lut_path=read_lut_path,
        read_lut_cube1d_path=read_lut_cube1d_path,
        read_lut_3dl_path=read_lut_3dl_path,
        read_lut_spi1d_path=read_lut_spi1d_path,
        read_lut_spi3d_path=read_lut_spi3d_path,
        write_lut1d_path=write_lut1d_path,
        write_lut3d_path=write_lut3d_path,
        write_png_path=io_directory / "write.png",
        write_jpeg_path=io_directory / "write.jpg",
        write_tiff_path=io_directory / "write.tiff",
        write_exr_path=io_directory / "write.exr",
        write_exr_none_path=io_directory / "write-none.exr",
        write_exr_zip_path=io_directory / "write-zip.exr",
        write_exr_zips_path=io_directory / "write-zips.exr",
        write_exr_dwaa_path=io_directory / "write-dwaa.exr",
        write_exr_dwab_path=io_directory / "write-dwab.exr",
        write_jpeg2000_path=io_directory / "write.jp2",
        write_webp_path=io_directory / "write.webp",
        write_bmp_path=io_directory / "write.bmp",
        write_pnm_path=io_directory / "write.pnm",
        write_tga_path=io_directory / "write.tga",
        write_hdr_path=io_directory / "write.hdr",
        write_dpx_path=io_directory / "write.dpx",
        uyvy422=generator.integers(0, 256, size=pixel_count * 2, dtype=cp.uint8),
        v210=generator.integers(0, 1 << 30, size=v210_word_count, dtype=cp.uint32),
        nv12=generator.integers(0, 256, size=pixel_count + pixel_count // 2, dtype=cp.uint8),
        p010=p010,
        yuv420p=generator.integers(0, 256, size=pixel_count + pixel_count // 2, dtype=cp.uint8),
        yuv422p10=generator.integers(0, 1024, size=pixel_count * 2, dtype=cp.uint16),
        yuv444p=generator.integers(0, 1024, size=pixel_count * 3, dtype=cp.uint16),
        yuva444p=generator.integers(0, 4096, size=pixel_count * 4, dtype=cp.uint16),
        vector_8=_uniform_vector(8.0),
        vector_32=_uniform_vector(32.0),
        vector_128=_uniform_vector(128.0),
        vector_rotation_32=_rotation_vector(),
    )


@pytest.mark.performance
def test_performance_registry_covers_every_public_gpu_pixel_operation() -> None:
    """REQ-TEST-010; v1-color-semantics acceptance 37; v1-white-balance acceptance 14;
    v1-white-point-simulation acceptance 14:
    registry classifies every public GPU pixel and boundary operation.
    """
    exported_functions = {
        name
        for module in _PUBLIC_OPERATION_MODULES
        for name in module.__all__
        if inspect.isfunction(getattr(module, name))
    } | {name for name in px.__all__ if inspect.isfunction(getattr(px, name))}
    public_frame_targets = {
        f"Frame.{name}"
        for name, member in inspect.getmembers(px.core.Frame, inspect.isfunction)
        if not name.startswith("_") and member.__qualname__.startswith(f"{px.core.Frame.__qualname__}.")
    }
    registry_targets = {case.target for case in _PERFORMANCE_CASES}
    registry_frame_targets = {target for target in registry_targets if target.startswith("Frame.")}

    assert _PERFORMANCE_BOUNDARY_FUNCTIONS == {
        "read_image",
        "write_image",
        "read_header",
        "read_lut",
        "decode_lut",
        "write_lut",
        "decode_image",
        "encode_image",
    }
    assert _NON_PIXEL_PUBLIC_FUNCTIONS == {"channels"}
    assert exported_functions - _NON_PIXEL_PUBLIC_FUNCTIONS == (
        _PUBLIC_GPU_PIXEL_FUNCTIONS | _PERFORMANCE_BOUNDARY_FUNCTIONS
    )
    assert registry_frame_targets == public_frame_targets - _PERFORMANCE_FRAME_METHOD_EXCLUSIONS
    assert registry_targets - registry_frame_targets == (
        _PUBLIC_GPU_PIXEL_FUNCTIONS | _PERFORMANCE_BOUNDARY_FUNCTIONS | {"copy"}
    )


@pytest.mark.performance
def test_performance_registry_covers_each_color_semantics_path() -> None:
    """v1-view-transform-lut-removal acceptance 8; v1-color-semantics acceptance 37: registry includes
    technical, both ACES analytics, and BT.2408.
    """
    color_cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.startswith("color-"))
    assert {case.target for case in color_cases} == {
        "chromatic_adaptation",
        "gamma_to_linear",
        "hsv_to_rgb",
        "linear_to_gamma",
        "rgb_to_grayscale",
        "rgb_to_hsv",
        "rgb_to_rgb",
        "rgb_to_ycbcr",
        "ycbcr_to_rgb",
        "ycbcr_to_ycbcr",
        "white_balance",
        "white_point_simulation",
    }
    rgb_to_rgb_cases = tuple(case for case in color_cases if case.target == "rgb_to_rgb")
    assert tuple((case.case_id, dict(case.kwargs)) for case in rgb_to_rgb_cases) == (
        ("color-acescg-srgb", {"output_colorspace": "sRGB", "output_gamma": "sRGB"}),
        (
            "color-aces13-analytic-srgb",
            {"output_colorspace": "sRGB", "output_gamma": "sRGB", "tonemap": "ACES-1.3"},
        ),
        (
            "color-aces20-analytic-srgb",
            {"output_colorspace": "sRGB", "output_gamma": "sRGB", "tonemap": "ACES-2.0"},
        ),
        (
            "color-bt2408-rec2020-pq",
            {"output_colorspace": "Rec.2020", "output_gamma": "PQ", "tonemap": "BT.2408"},
        ),
    )


@pytest.mark.performance
def test_performance_registry_includes_both_white_balance_public_calls() -> None:
    """v1-white-balance acceptance 14: registry has one FHD low-level and one FHD convenience call."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target in {"chromatic_adaptation", "white_balance"})
    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in cases) == (
        (
            "color-chromatic-adaptation",
            "chromatic_adaptation",
            {"input_white": (0.34567, 0.35850), "output_white": (0.32168, 0.33767)},
        ),
        ("color-white-balance", "white_balance", {"temperature": 5000.0}),
    )


@pytest.mark.performance
def test_performance_registry_includes_one_white_point_simulation_case() -> None:
    """v1-white-point-simulation acceptance 14: registry adds one FHD RGB float32 public-call case."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target == "white_point_simulation")
    assert tuple((case.case_id, case.input_attribute, dict(case.kwargs)) for case in cases) == (
        (
            "color-white-point-simulation",
            "frame",
            {"input_white": "D65", "output_white": "D93"},
        ),
    )


@pytest.mark.performance
def test_performance_registry_hsv_cases_use_the_required_deterministic_fixture(
    performance_inputs: _Inputs,
) -> None:
    """v1-hsv acceptance 16: the two FHD cases cover six hue sectors, nominal S, and scene-scale V."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target in {"rgb_to_hsv", "hsv_to_rgb"})
    assert tuple((case.case_id, case.target, case.input_attribute) for case in cases) == (
        ("color-rgb-hsv", "rgb_to_hsv", "frame"),
        ("color-hsv-rgb", "hsv_to_rgb", "hsv_frame"),
    )
    hsv = performance_inputs.hsv_frame.data
    sectors = cp.unique(cp.floor(cp.mod(hsv[..., 0], np.float32(1.0)) * np.float32(6.0))).get()
    np.testing.assert_array_equal(sectors, np.arange(6, dtype=np.float32))
    assert float(cp.min(hsv[..., 1]).get()) == 0.0
    assert float(cp.max(hsv[..., 1]).get()) == 1.0
    assert float(cp.min(hsv[..., 2]).get()) == 0.0
    assert float(cp.max(hsv[..., 2]).get()) == 2.0


@pytest.mark.performance
def test_performance_registry_includes_representative_unsharp_mask_case() -> None:
    """v1-unsharp-mask acceptance 10: registry includes one FHD Gaussian sharpening case."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target == "unsharp_mask")

    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in cases) == (
        ("unsharp-2-1", "unsharp_mask", {"sigma": 2.0, "amount": 1.0}),
    )


@pytest.mark.performance
def test_performance_registry_includes_representative_sharpen_case() -> None:
    """v1-sharpen acceptance 12: registry includes FHD fp32 RGB amount=1 with mirror border."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target == "sharpen")

    assert tuple((case.case_id, case.parameters, dict(case.kwargs)) for case in cases) == (
        ("sharpen-1-mirror", "amount=1, border=mirror", {"amount": 1.0, "border": "mirror"}),
    )


@pytest.mark.performance
def test_performance_registry_includes_seven_radius_five_morphology_cases() -> None:
    """v1-morphology acceptance 12: registry includes one FHD disk radius-5 case for every public operation."""
    morphology_names = frozenset(px.morphology.__all__)
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target in morphology_names)

    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in cases) == tuple(
        (f"morphology-{name}-5", name, {"radius": 5, "shape": "disk"})
        for name in (
            "erosion",
            "dilation",
            "opening",
            "closing",
            "morphological_gradient",
            "white_tophat",
            "black_tophat",
        )
    )


@pytest.mark.performance
def test_performance_registry_includes_all_derivative_filter_paths() -> None:
    """v1-derivative-filters acceptance 18: registry covers three Sobel paths, Laplacian, and representative DoG."""
    derivative_cases = tuple(
        case for case in _DERIVATIVE_CASES if case.target in {"sobel", "laplacian", "difference_of_gaussians"}
    )
    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in derivative_cases) == (
        ("sobel-x", "sobel", {"direction": "x"}),
        ("sobel-y", "sobel", {"direction": "y"}),
        ("sobel-magnitude", "sobel", {"direction": "magnitude"}),
        ("laplacian-3x3", "laplacian", {}),
        ("dog-1-2", "difference_of_gaussians", {"sigma1": 1.0, "sigma2": 2.0}),
    )


@pytest.mark.performance
def test_performance_registry_includes_representative_canny_case() -> None:
    """v1-canny acceptance 17: registry includes the specified FHD fp32 RGB Canny case."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target == "canny")

    assert tuple((case.case_id, case.parameters, dict(case.kwargs)) for case in cases) == (
        (
            "canny-0.5-1-mirror",
            "threshold_low=0.5, threshold_high=1.0, border=mirror",
            {"threshold_low": 0.5, "threshold_high": 1.0, "border": "mirror"},
        ),
    )


@pytest.mark.performance
def test_performance_registry_includes_both_analysis_cases() -> None:
    """v1-analysis-pair acceptance 26: registry has the exact FHD Harris and 64x64 template cases."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target in {"corner_harris", "match_template"})
    assert tuple(
        (
            case.case_id,
            case.target,
            dict(case.kwargs),
            dict(case.fixture_kwargs),
            case.input_attribute,
            case.transferred_bytes,
        )
        for case in cases
    ) == (
        (
            "corner-harris-3-004-mirror",
            "corner_harris",
            {"block_size": 3, "k": 0.04, "border": "mirror"},
            {},
            "frame",
            _FHD_FP32_RGB_BYTES + _FHD_FP32_Y_BYTES,
        ),
        (
            "match-template-64-ccoeff-normed",
            "match_template",
            {"method": "ccoeff_normed"},
            {"template": "analysis_template"},
            "frame",
            _FHD_FP32_RGB_BYTES
            + 64 * 64 * _CHANNELS * np.dtype(np.float32).itemsize
            + (1920 - 64 + 1) * (1080 - 64 + 1) * np.dtype(np.float32).itemsize,
        ),
    )


@pytest.mark.performance
def test_performance_registry_includes_three_quality_metric_cases() -> None:
    """v1-quality-metrics acceptance 24: registry has three default-range FHD fp32 RGB comparisons."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target in {"psnr", "ssim", "ssim_map"})
    assert tuple(
        (case.case_id, case.target, case.parameters, dict(case.kwargs), dict(case.fixture_kwargs)) for case in cases
    ) == (
        (
            "quality-psnr-default",
            "psnr",
            "FHD fp32 RGB reference/candidate, data_range=1.0 default",
            {},
            {"candidate": "frame"},
        ),
        (
            "quality-ssim-default",
            "ssim",
            "FHD fp32 RGB reference/candidate, data_range=1.0 default",
            {},
            {"candidate": "frame"},
        ),
        (
            "quality-ssim-map-default",
            "ssim_map",
            "FHD fp32 RGB reference/candidate, data_range=1.0 default",
            {},
            {"candidate": "frame"},
        ),
    )


@pytest.mark.performance
def test_performance_registry_includes_both_histogram_equalization_cases() -> None:
    """v1-histogram acceptance 22: registry has the two exact FHD fp32 RGB representative cases."""
    assert tuple((case.case_id, case.target, case.parameters, dict(case.kwargs)) for case in _HISTOGRAM_CASES) == (
        (
            "equalize-histogram-1024",
            "equalize_histogram",
            "domain=(0,1), bins=1024",
            {"domain": (0.0, 1.0), "bins": 1024},
        ),
        (
            "clahe-2-8x8-1024",
            "clahe",
            "clip_limit=2, tiles_y=8, tiles_x=8, domain=(0,1), bins=1024",
            {"clip_limit": 2.0, "tiles_y": 8, "tiles_x": 8, "domain": (0.0, 1.0), "bins": 1024},
        ),
    )


@pytest.mark.performance
def test_performance_registry_includes_representative_to_format_cases() -> None:
    """v1-public-namespace acceptance 15: registry covers every wire-format export function."""
    assert {case.target for case in _TO_FORMAT_CASES} == {
        f"to_{name}" for name in ("uyvy422", "v210", "nv12", "p010", "yuv420p", "yuv422p", "yuv444p", "yuva444p")
    }
    assert all(case.transferred_bytes is not None for case in _TO_FORMAT_CASES)


@pytest.mark.performance
def test_performance_registry_includes_representative_bytes_boundary_cases() -> None:
    """v1-io-formats acceptance 22; v1-bytes-boundary acceptance 15: bytes cases cover every encoded raster format."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target in {"decode_image", "encode_image"})

    assert {case.case_id for case in cases} == {
        "bytes-decode-png",
        "bytes-decode-jpeg",
        "bytes-decode-tiff",
        "bytes-decode-jpeg2000",
        "bytes-decode-webp",
        "bytes-decode-bmp",
        "bytes-decode-pnm",
        "bytes-encode-png",
        "bytes-encode-jpeg",
        "bytes-encode-tiff",
        "bytes-encode-jpeg2000",
        "bytes-encode-webp",
        "bytes-encode-bmp",
        "bytes-encode-pnm",
    }


@pytest.mark.performance
def test_performance_registry_includes_representative_file_boundary_cases() -> None:
    """v1-io-formats acceptance 22; v1-bytes-boundary acceptance 15: file cases cover every raster format
    beside EXR and LUT boundaries.
    """
    cases = tuple(
        case for case in _PERFORMANCE_CASES if case.target in {"read_image", "write_image", "read_header", "read_lut"}
    )

    assert {case.case_id for case in cases} == {
        "file-read-png",
        "file-read-jpeg",
        "file-read-tiff",
        "file-read-exr",
        "file-exr-phase1-read-none",
        "file-exr-phase1-read-zip",
        "file-exr-phase1-read-zips",
        "file-exr-phase2-read-dwaa",
        "file-exr-phase2-read-dwab",
        "file-exr-phase3-read-rle",
        "file-exr-phase3-read-pxr24",
        "file-exr-phase3-read-b44",
        "file-exr-phase3-read-b44a",
        "file-exr-phase4-read-piz",
        "file-read-jpeg2000",
        "file-read-webp",
        "file-read-bmp",
        "file-read-pnm",
        "file-read-tga",
        "file-read-hdr",
        "file-read-dpx",
        "file-write-png",
        "file-write-jpeg",
        "file-write-tiff",
        "file-write-exr",
        "file-exr-phase1-write-none",
        "file-exr-phase1-write-zip",
        "file-exr-phase1-write-zips",
        "file-exr-phase2-write-dwaa",
        "file-exr-phase2-write-dwab",
        "file-exr-phase3-write-rle",
        "file-exr-phase3-write-pxr24",
        "file-exr-phase3-write-b44",
        "file-exr-phase3-write-b44a",
        "file-exr-phase4-write-piz",
        "file-write-jpeg2000",
        "file-write-webp",
        "file-write-bmp",
        "file-write-pnm",
        "file-write-tga",
        "file-write-hdr",
        "file-write-dpx",
        "file-read-header-png",
        "file-read-lut-cube-65",
        "file-read-lut-cube-1d",
        "file-read-lut-3dl",
        "file-read-lut-spi1d",
        "file-read-lut-spi3d",
    }


@pytest.mark.performance
def test_performance_registry_includes_phase1_exr_compression_cases() -> None:
    """v1-exr-gpu-phase1 acceptance 24: NONE, ZIP, and ZIPS each contribute public read/write FHD cases."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.startswith("file-exr-phase1-"))

    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in cases) == (
        ("file-exr-phase1-read-none", "read_image", {"unchanged": True}),
        ("file-exr-phase1-read-zip", "read_image", {"unchanged": True}),
        ("file-exr-phase1-read-zips", "read_image", {"unchanged": True}),
        ("file-exr-phase1-write-none", "write_image", {"compression": "none"}),
        ("file-exr-phase1-write-zip", "write_image", {"compression": "zip"}),
        ("file-exr-phase1-write-zips", "write_image", {"compression": "zips"}),
    )
    assert all(
        case.minimum_frames == _BOUNDARY_MEASURED_MINIMUM_FRAMES and case.minimum_seconds == _MEASURED_MINIMUM_SECONDS
        for case in cases
    )


@pytest.mark.performance
def test_performance_registry_includes_phase2_dwa_cases() -> None:
    """v1-exr-gpu-phase2 acceptance 27-30: DWAA and DWAB add public read/write FHD file cases."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.startswith("file-exr-phase2-"))

    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in cases) == (
        ("file-exr-phase2-read-dwaa", "read_image", {"unchanged": True}),
        ("file-exr-phase2-read-dwab", "read_image", {"unchanged": True}),
        ("file-exr-phase2-write-dwaa", "write_image", {"compression": "dwaa", "dwa_level": 45.0}),
        ("file-exr-phase2-write-dwab", "write_image", {"compression": "dwab", "dwa_level": 45.0}),
    )
    assert all(
        case.minimum_frames == _BOUNDARY_MEASURED_MINIMUM_FRAMES and case.minimum_seconds == _MEASURED_MINIMUM_SECONDS
        for case in cases
    )


@pytest.mark.performance
def test_performance_registry_includes_phase3_exr_cases() -> None:
    """v1-exr-gpu-phase3 acceptance 40: all four codecs keep source-fixed public read/write FHD cases."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.startswith("file-exr-phase3-"))

    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in cases) == (
        ("file-exr-phase3-read-rle", "read_image", {"compression": "rle"}),
        ("file-exr-phase3-read-pxr24", "read_image", {"compression": "pxr24"}),
        ("file-exr-phase3-read-b44", "read_image", {"compression": "b44"}),
        ("file-exr-phase3-read-b44a", "read_image", {"compression": "b44a"}),
        ("file-exr-phase3-write-rle", "write_image", {"compression": "rle"}),
        ("file-exr-phase3-write-pxr24", "write_image", {"compression": "pxr24"}),
        ("file-exr-phase3-write-b44", "write_image", {"compression": "b44"}),
        ("file-exr-phase3-write-b44a", "write_image", {"compression": "b44a"}),
    )
    assert all(
        case.input_attribute == "exr_phase3"
        and case.minimum_frames == _BOUNDARY_MEASURED_MINIMUM_FRAMES
        and case.minimum_seconds == _MEASURED_MINIMUM_SECONDS
        for case in cases
    )


@pytest.mark.performance
def test_performance_registry_includes_phase4_piz_cases() -> None:
    """v1-exr-gpu-phase4 acceptance 45: PIZ keeps source-fixed HALF read/write FHD file cases."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.startswith("file-exr-phase4-"))

    assert tuple((case.case_id, case.target, dict(case.kwargs)) for case in cases) == (
        ("file-exr-phase4-read-piz", "read_image", {}),
        ("file-exr-phase4-write-piz", "write_image", {}),
    )
    assert all(
        case.input_attribute == "exr_phase4"
        and case.minimum_frames == _BOUNDARY_MEASURED_MINIMUM_FRAMES
        and case.minimum_seconds == _MEASURED_MINIMUM_SECONDS
        and case.transferred_bytes == _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.float16).itemsize * 2
        for case in cases
    )


@pytest.mark.performance
def test_performance_registry_tracks_source_fixed_exr_routes_and_storage_dtypes(
    performance_inputs: _Inputs,
) -> None:
    """v1-exr-runtime-independence acceptance 50: keep 22 EXR cases aligned with final routing and dtypes."""
    fp16_read_write_bytes = _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.float16).itemsize * 2
    fp32_to_fp16_bytes = _WIDTH * _HEIGHT * _CHANNELS * (np.dtype(np.float32).itemsize + np.dtype(np.float16).itemsize)
    expected = {
        "file-read-exr": (
            "FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase1-read-none": (
            "FHD HALF RGB EXR NONE file, unchanged, source-fixed native lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase1-read-zip": (
            "FHD HALF RGB EXR ZIP file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase1-read-zips": (
            "FHD HALF RGB EXR ZIPS file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase2-read-dwaa": (
            "FHD HALF RGB EXR DWAA file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase2-read-dwab": (
            "FHD HALF RGB EXR DWAB file, unchanged, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase3-read-rle": (
            "FHD HALF RGB EXR RLE file, unchanged, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase3-read-pxr24": (
            "FHD HALF RGB EXR PXR24 file, unchanged, source-fixed custom CPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase3-read-b44": (
            "FHD HALF RGB EXR B44 file, unchanged, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase3-read-b44a": (
            "FHD HALF RGB EXR B44A file, unchanged, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase4-read-piz": (
            "FHD HALF RGB EXR PIZ file, unchanged, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-write-exr": (
            "FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase1-write-none": (
            "FHD fp32 RGB to EXR NONE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase1-write-zip": (
            "FHD fp32 RGB to EXR ZIP/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase1-write-zips": (
            "FHD fp32 RGB to EXR ZIPS/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase2-write-dwaa": (
            "FHD fp32 RGB to EXR DWAA/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase2-write-dwab": (
            "FHD fp32 RGB to EXR DWAB/HALF, dtype omitted, dwa_level=45.0, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase3-write-rle": (
            "FHD fp32 RGB to EXR RLE/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase3-write-pxr24": (
            "FHD fp32 RGB to EXR PXR24/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp32_to_fp16_bytes,
        ),
        "file-exr-phase3-write-b44": (
            "FHD fp16 RGB to EXR B44/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase3-write-b44a": (
            "FHD fp16 RGB to EXR B44A/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
        "file-exr-phase4-write-piz": (
            "FHD fp16 RGB to EXR PIZ/HALF, dtype omitted, source-fixed GPU lane, temporary-file I/O included",
            fp16_read_write_bytes,
        ),
    }
    cases = {
        case.case_id: (case.parameters, case.transferred_bytes)
        for case in _PERFORMANCE_CASES
        if case.case_id in {"file-read-exr", "file-write-exr"} or case.case_id.startswith("file-exr-phase")
    }

    assert cases == expected

    fixture_paths = {
        "file-exr-phase2-read-dwaa": performance_inputs.read_exr_dwaa_path,
        "file-exr-phase2-read-dwab": performance_inputs.read_exr_dwab_path,
        "file-exr-phase3-read-rle": performance_inputs.exr_phase3.read_path("rle"),
        "file-exr-phase3-read-pxr24": performance_inputs.exr_phase3.read_path("pxr24"),
    }
    fixture_storage_dtypes = {
        case_id: {channel.dtype for part in io_header._parse_exr(path).parts for channel in part.channels}
        for case_id, path in fixture_paths.items()
    }
    assert fixture_storage_dtypes == {case_id: {"float16"} for case_id in fixture_paths}


@pytest.mark.performance
def test_exr_phase1_backend_performance_report(
    performance_inputs: _Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-runtime-independence acceptance 36 and 47: measure and report accepted lanes against the dev baseline;
    assertions cover measurement completeness and sanity, while fixed-fixture contracts own gate decisions."""
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    print(
        "EXR_REPORT_DEVICE "
        f"name={device_name} driver={cp.cuda.runtime.driverGetVersion()} runtime={cp.cuda.runtime.runtimeGetVersion()}"
    )
    medians: dict[tuple[str, str, str], float] = {}
    synchronize = device.synchronize
    for direction in ("read", "write"):
        for compression in ("none", "zip", "zips"):
            accepted_backends = (
                (("native", "custom_cpu", "gpu") if compression == "none" else ("custom_cpu", "gpu"))
                if direction == "read"
                else ("gpu",)
            )
            backends = ("cpu", *accepted_backends)
            for backend in backends:
                if direction == "read":
                    path = getattr(performance_inputs, f"read_exr_{compression}_path")
                    operation = (
                        (lambda path=path: read_openexr_frame(path))
                        if backend == "cpu"
                        else (lambda path=path: px.io.read_image(path, unchanged=True))
                    )
                else:
                    path = getattr(performance_inputs, f"write_exr_{compression}_path")
                    operation = (
                        (
                            lambda path=path, compression=compression: write_openexr_frame(
                                path,
                                performance_inputs.exr_phase1_frame,
                                compression=compression,
                                dwa_level=None,
                            )
                        )
                        if backend == "cpu"
                        else (
                            lambda path=path, compression=compression: px.io.write_image(
                                path,
                                performance_inputs.exr_phase1_frame,
                                compression=compression,
                            )
                        )
                    )
                if backend != "cpu":
                    monkeypatch.setitem(io._EXR_ROUTING, (compression, direction), backend)
                _run_warmup(operation, synchronize, minimum_seconds=_WARMUP_MINIMUM_SECONDS)
                durations_ms = _measure_durations_ms(
                    operation,
                    synchronize,
                    minimum_frames=_BOUNDARY_MEASURED_MINIMUM_FRAMES,
                    minimum_seconds=_MEASURED_MINIMUM_SECONDS,
                )
                medians[(compression, direction, backend)] = median(durations_ms)
            cpu_ms = medians[(compression, direction, "cpu")]
            candidate_medians = {backend: medians[(compression, direction, backend)] for backend in accepted_backends}
            ratios = {backend: value / cpu_ms for backend, value in candidate_medians.items()}
            eligible = tuple(
                (backend, value) for backend, value in candidate_medians.items() if ratios[backend] <= 0.95
            )
            selected = min(eligible, key=lambda item: item[1])[0] if eligible else "cpu"
            print(
                "EXR_REPORT_RESULT "
                f"compression={compression} direction={direction} openexr_median_ms={cpu_ms:.6f} "
                f"candidate_medians_ms={candidate_medians!r} ratios={ratios!r} selected={selected}"
            )

    assert len(medians) == 16
    assert all(math.isfinite(value) and value > 0.0 for value in medians.values())


@pytest.mark.performance
def test_exr_phase2_performance_gate_rejects_raw_stored_only_output(tmp_path: Path) -> None:
    """v1-exr-gpu-phase2 acceptance 28: a raw-stored-only DWA file cannot contribute a gate median."""
    path = tmp_path / "raw-stored-only-dwaa.exr"
    frame = px.io.from_array(
        cp.zeros((1, 1, 3), dtype=cp.float32),
        colorspace="ACEScg",
        gamma="linear",
        channels="RGB",
    )
    write_openexr_frame(path, frame, compression="dwaa", dwa_level=45.0)
    container = io_header._parse_exr(path)
    assert container.dwa_eligible
    assert all(chunk.raw_stored for chunk in container.chunks)

    with pytest.raises(AssertionError, match="no compressed DWA v2 chunk"):
        _assert_compressed_dwa_v2_output(path)


@pytest.mark.performance
def test_exr_phase2_backend_performance_report(
    performance_inputs: _Inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-runtime-independence acceptance 36 and 47: measure and report accepted DWA lanes against the dev
    baseline; assertions cover measurement completeness and sanity, while fixed-fixture contracts own gate decisions."""
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    print(
        "EXR_DWA_REPORT_DEVICE "
        f"name={device_name} driver={cp.cuda.runtime.driverGetVersion()} runtime={cp.cuda.runtime.runtimeGetVersion()}"
    )
    source_selection = dict(io._EXR_ROUTING)
    medians: dict[tuple[str, str, str], float] = {}
    synchronize = device.synchronize
    for compression in ("dwaa", "dwab"):
        read_path = getattr(performance_inputs, f"read_exr_{compression}_path")
        _assert_compressed_dwa_v2_output(read_path)
        for direction in ("read", "write"):
            backends = ("cpu", "custom_cpu", "gpu") if direction == "read" else ("cpu", "gpu")
            for backend in backends:
                if direction == "read":
                    operation = (
                        (lambda path=read_path: read_openexr_frame(path))
                        if backend == "cpu"
                        else (lambda path=read_path: px.io.read_image(path, unchanged=True))
                    )
                else:
                    write_path = getattr(performance_inputs, f"write_exr_{compression}_path")
                    operation = (
                        (
                            lambda path=write_path, compression=compression: write_openexr_frame(
                                path,
                                performance_inputs.exr_phase1_frame,
                                compression=compression,
                                dwa_level=45.0,
                            )
                        )
                        if backend == "cpu"
                        else (
                            lambda path=write_path, compression=compression: px.io.write_image(
                                path,
                                performance_inputs.exr_phase1_frame,
                                compression=compression,
                                dwa_level=45.0,
                            )
                        )
                    )
                if backend != "cpu":
                    monkeypatch.setitem(io._EXR_ROUTING, (compression, direction), backend)
                _run_warmup(operation, synchronize, minimum_seconds=_WARMUP_MINIMUM_SECONDS)
                durations_ms = _measure_durations_ms(
                    operation,
                    synchronize,
                    minimum_frames=_BOUNDARY_MEASURED_MINIMUM_FRAMES,
                    minimum_seconds=_MEASURED_MINIMUM_SECONDS,
                )
                if direction == "write":
                    _assert_compressed_dwa_v2_output(write_path)
                medians[(compression, direction, backend)] = median(durations_ms)

            cpu_ms = medians[(compression, direction, "cpu")]
            candidate_backends = ("custom_cpu", "gpu") if direction == "read" else ("gpu",)
            candidate_medians = {backend: medians[(compression, direction, backend)] for backend in candidate_backends}
            ratios = {backend: value / cpu_ms for backend, value in candidate_medians.items()}
            eligible = tuple(
                (backend, value) for backend, value in candidate_medians.items() if ratios[backend] <= 0.95
            )
            selected = min(eligible, key=lambda item: item[1])[0] if eligible else "cpu"
            print(
                "EXR_DWA_REPORT_RESULT "
                f"compression={compression} direction={direction} openexr_median_ms={cpu_ms:.6f} "
                f"candidate_medians_ms={candidate_medians!r} ratios={ratios!r} selected={selected} "
                f"source_fixed={source_selection[(compression, direction)]}"
            )

    assert len(medians) == 10
    assert all(math.isfinite(value) and value > 0.0 for value in medians.values())


@pytest.mark.performance
def test_exr_phase3_initial_backend_performance_report(performance_inputs: _Inputs) -> None:
    """v1-exr-runtime-independence acceptance 36 and 47: measure and report Phase 3 lanes against the dev baseline;
    assertions cover measurement completeness and sanity, while fixed-fixture contracts own gate decisions."""
    identity = device_identity()
    print(
        "EXR_PHASE3_REPORT_DEVICE "
        f"name={identity.name} driver={identity.driver_version} runtime={identity.runtime_version}"
    )
    measurements = []
    initial_runs: dict[tuple[str, str], GateRun] = {}
    decisions: dict[tuple[str, str], GateDecision] = {}
    for compression in PHASE3_COMPRESSIONS:
        inspection = inspect_phase3_gate_fixture(
            performance_inputs.exr_phase3.read_path(compression),
            compression,
        )
        print(
            "EXR_PHASE3_REPORT_FIXTURE "
            f"compression={compression} total_chunks={inspection.total_chunks} "
            f"compressed_chunks={inspection.compressed_chunks} rle_packets={inspection.rle_packets} "
            f"pxr24_planes={inspection.pxr24_planes} dense_blocks={inspection.dense_blocks} "
            f"flat_blocks={inspection.flat_blocks}"
        )
        for direction in ("read", "write"):
            measurement = measure_phase3_gate_case(performance_inputs.exr_phase3, compression, direction)
            measurements.append(measurement)
            candidates_ms = {backend: value for backend, value in measurement.medians_ms.items() if backend != "cpu"}
            key = (compression, direction)
            initial_run = GateRun(openexr_ms=measurement.medians_ms["cpu"], candidates_ms=candidates_ms)
            initial_runs[key] = initial_run
            decision = synthesize_gate_decision(initial_run)
            decisions[key] = decision
            print(
                "EXR_PHASE3_REPORT_RESULT "
                f"compression={compression} direction={direction} medians_ms={measurement.medians_ms!r} "
                f"iterations={measurement.iterations!r} ratios={dict(decision.initial_ratios)!r} "
                f"initial_candidate={decision.initial_candidate} initial_ratio={decision.initial_ratio!r} "
                f"repeat_required={decision.repeat_required} initial_selection={decision.selected}"
            )

    repeat_measurements = []
    for (compression, direction), decision in tuple(decisions.items()):
        if not decision.repeat_required:
            continue
        measurement = measure_phase3_gate_case(performance_inputs.exr_phase3, compression, direction)
        repeat_measurements.append(measurement)
        candidates_ms = {backend: value for backend, value in measurement.medians_ms.items() if backend != "cpu"}
        decision = synthesize_gate_decision(
            initial_runs[(compression, direction)],
            GateRun(openexr_ms=measurement.medians_ms["cpu"], candidates_ms=candidates_ms),
        )
        decisions[(compression, direction)] = decision
        print(
            "EXR_PHASE3_REPORT_REPEAT_RESULT "
            f"compression={compression} direction={direction} medians_ms={measurement.medians_ms!r} "
            f"iterations={measurement.iterations!r} ratios={dict(decision.repeat_ratios)!r} "
            f"selected={decision.selected}"
        )

    synthesized_selection = {key: decision.selected for key, decision in decisions.items()}
    source_selection = {key: io._EXR_ROUTING[key] for key in decisions}
    print(
        f"EXR_PHASE3_REPORT_SOURCE_COMPARISON synthesized={synthesized_selection!r} source_fixed={source_selection!r}"
    )
    assert len(measurements) == 8
    assert all(
        set(measurement.medians_ms)
        == ({"cpu", "custom_cpu", "gpu"} if measurement.direction == "read" else {"cpu", "gpu"})
        and all(math.isfinite(value) and value > 0.0 for value in measurement.medians_ms.values())
        and all(count >= _BOUNDARY_MEASURED_MINIMUM_FRAMES for count in measurement.iterations.values())
        for measurement in (*measurements, *repeat_measurements)
    )


@pytest.mark.performance
def test_exr_phase3_public_backend_performance_report(performance_inputs: _Inputs) -> None:
    """v1-exr-gpu-phase3 acceptance 40: report all eight fixed public boundary medians without forced routing."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.startswith("file-exr-phase3-"))
    synchronize = cp.cuda.Device().synchronize
    medians_ms: dict[str, float] = {}
    iterations: dict[str, int] = {}
    for case in cases:
        operation = case.bind(performance_inputs)
        _run_warmup(operation, synchronize, minimum_seconds=_WARMUP_MINIMUM_SECONDS)
        durations_ms = _measure_durations_ms(
            operation,
            synchronize,
            minimum_frames=case.minimum_frames,
            minimum_seconds=case.minimum_seconds,
        )
        value = median(durations_ms)
        medians_ms[case.case_id] = value
        iterations[case.case_id] = len(durations_ms)
        compression = case.case_id.rsplit("-", maxsplit=1)[1]
        direction = "read" if "-read-" in case.case_id else "write"
        backend = io._EXR_ROUTING[(compression, direction)]
        print(
            "EXR_PHASE3_PUBLIC_RESULT "
            f"case_id={case.case_id} backend={backend} median_ms={value!r} iterations={len(durations_ms)}"
        )

    assert len(medians_ms) == 8
    assert all(math.isfinite(value) and value > 0.0 for value in medians_ms.values())
    assert all(count >= _BOUNDARY_MEASURED_MINIMUM_FRAMES for count in iterations.values())


@pytest.mark.performance
def test_exr_phase4_initial_backend_performance_report(performance_inputs: _Inputs) -> None:
    """v1-exr-runtime-independence acceptance 36 and 47: measure and report PIZ lanes against the dev baseline;
    assertions cover measurement completeness and sanity, while fixed-fixture contracts own gate decisions."""
    identity = device_identity()
    print(
        "EXR_PHASE4_REPORT_DEVICE "
        f"name={identity.name} driver={identity.driver_version} runtime={identity.runtime_version}"
    )
    for dtype in ("fp16", "fp32"):
        inspection = inspect_phase4_gate_fixture(performance_inputs.exr_phase4.read_path(dtype), dtype)
        print(
            "EXR_PHASE4_REPORT_FIXTURE "
            f"dtype={dtype} total_chunks={inspection.total_chunks} "
            f"compressed_chunks={inspection.compressed_chunks} nonempty_bitmaps={inspection.nonempty_bitmaps} "
            f"maximum_wavelet_levels={inspection.maximum_wavelet_levels} huffman_tables={inspection.huffman_tables} "
            f"huffman_table_bytes={inspection.huffman_table_bytes} huffman_data_bytes={inspection.huffman_data_bytes}"
        )

    initial_measurements: list[Phase4GateMeasurement] = []
    repeat_measurements: list[Phase4GateMeasurement] = []
    decisions: dict[tuple[str, str], phase4_gate.GateDecision] = {}
    for direction in ("read", "write"):
        measurement = measure_phase4_gate_case(performance_inputs.exr_phase4, direction, dtype="fp16")
        initial_measurements.append(measurement)
        candidates_ms = {backend: value for backend, value in measurement.medians_ms.items() if backend != "cpu"}
        initial = phase4_gate.GateRun(openexr_ms=measurement.medians_ms["cpu"], candidates_ms=candidates_ms)
        decision = phase4_gate.synthesize_gate_decision(initial)
        print(
            "EXR_PHASE4_REPORT_RESULT "
            f"dtype=fp16 direction={direction} medians_ms={measurement.medians_ms!r} "
            f"iterations={measurement.iterations!r} ratios={dict(decision.initial_ratios)!r} "
            f"initial_candidate={decision.initial_candidate} initial_ratio={decision.initial_ratio!r} "
            f"repeat_required={decision.repeat_required} initial_selection={decision.selected}"
        )
        if decision.repeat_required:
            repeat = _run_phase4_isolated_repeat(direction)
            repeat_measurements.append(repeat)
            repeat_candidates = {backend: value for backend, value in repeat.medians_ms.items() if backend != "cpu"}
            decision = phase4_gate.synthesize_gate_decision(
                initial,
                phase4_gate.GateRun(openexr_ms=repeat.medians_ms["cpu"], candidates_ms=repeat_candidates),
            )
            print(
                "EXR_PHASE4_REPORT_REPEAT_RESULT "
                f"dtype=fp16 direction={direction} medians_ms={repeat.medians_ms!r} "
                f"iterations={repeat.iterations!r} ratios={dict(decision.repeat_ratios)!r} "
                f"worst_ratios={dict(decision.worst_ratios)!r} selected={decision.selected}"
            )
        decisions[("piz", direction)] = decision

    synthesized_selection = {key: decision.selected for key, decision in decisions.items()}
    source_selection = {key: io._EXR_ROUTING[key] for key in decisions}
    print(
        f"EXR_PHASE4_REPORT_SOURCE_COMPARISON synthesized={synthesized_selection!r} source_fixed={source_selection!r}"
    )

    reference_measurements: list[Phase4GateMeasurement] = []
    for direction in ("read", "write"):
        measurement = measure_phase4_gate_case(performance_inputs.exr_phase4, direction, dtype="fp32")
        reference_measurements.append(measurement)
        openexr_ms = measurement.medians_ms["cpu"]
        ratios = {backend: value / openexr_ms for backend, value in measurement.medians_ms.items() if backend != "cpu"}
        print(
            "EXR_PHASE4_REPORT_REFERENCE_RESULT "
            f"dtype=fp32 direction={direction} medians_ms={measurement.medians_ms!r} "
            f"iterations={measurement.iterations!r} ratios={ratios!r}"
        )

    assert all(
        set(measurement.medians_ms)
        == ({"cpu", "custom_cpu", "gpu"} if measurement.direction == "read" else {"cpu", "gpu"})
        and all(math.isfinite(value) and value > 0.0 for value in measurement.medians_ms.values())
        and all(count >= _BOUNDARY_MEASURED_MINIMUM_FRAMES for count in measurement.iterations.values())
        for measurement in (*initial_measurements, *repeat_measurements, *reference_measurements)
    )


@pytest.mark.performance
def test_exr_phase4_public_backend_performance_report(performance_inputs: _Inputs) -> None:
    """v1-exr-gpu-phase4 acceptance 45: report both fixed public fp16 PIZ boundary medians."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.startswith("file-exr-phase4-"))
    synchronize = cp.cuda.Device().synchronize
    medians_ms: dict[str, float] = {}
    iterations: dict[str, int] = {}
    for case in cases:
        operation = case.bind(performance_inputs)
        _run_warmup(operation, synchronize, minimum_seconds=_WARMUP_MINIMUM_SECONDS)
        durations_ms = _measure_durations_ms(
            operation,
            synchronize,
            minimum_frames=case.minimum_frames,
            minimum_seconds=case.minimum_seconds,
        )
        value = median(durations_ms)
        medians_ms[case.case_id] = value
        iterations[case.case_id] = len(durations_ms)
        direction = "read" if "-read-" in case.case_id else "write"
        backend = io._EXR_ROUTING[("piz", direction)]
        print(
            "EXR_PHASE4_PUBLIC_RESULT "
            f"case_id={case.case_id} backend={backend} median_ms={value!r} iterations={len(durations_ms)}"
        )

    assert len(medians_ms) == 2
    assert all(math.isfinite(value) and value > 0.0 for value in medians_ms.values())
    assert all(count >= _BOUNDARY_MEASURED_MINIMUM_FRAMES for count in iterations.values())


@pytest.mark.performance
def test_performance_registry_has_sixteen_new_format_boundary_cases() -> None:
    """v1-io-formats acceptance 22: four formats each contribute read/write/decode/encode FHD cases."""
    expected = {
        f"{boundary}-{operation}-{format_token}"
        for format_token in ("jpeg2000", "webp", "bmp", "pnm")
        for boundary, operation in (
            ("file", "read"),
            ("file", "write"),
            ("bytes", "decode"),
            ("bytes", "encode"),
        )
    }
    assert {case.case_id for case in _PERFORMANCE_CASES} >= expected


@pytest.mark.performance
def test_performance_registry_includes_tga_file_boundary_cases() -> None:
    """v1-tga acceptance 12: TGA contributes one FHD read and one FHD write file-boundary case."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.endswith("-tga"))
    assert tuple((case.case_id, case.target, case.input_attribute) for case in cases) == (
        ("file-read-tga", "read_image", "read_tga_path"),
        ("file-write-tga", "write_image", "write_tga_path"),
    )


@pytest.mark.performance
def test_performance_registry_includes_hdr_file_boundary_cases() -> None:
    """v1-hdr acceptance 10: HDR contributes one FHD read and one FHD write file-boundary case."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.endswith("-hdr"))
    assert tuple((case.case_id, case.target, case.input_attribute) for case in cases) == (
        ("file-read-hdr", "read_image", "read_hdr_path"),
        ("file-write-hdr", "write_image", "write_hdr_path"),
    )


@pytest.mark.performance
def test_performance_registry_includes_dpx_file_boundary_cases() -> None:
    """v1-dpx acceptance 12: DPX contributes FHD fp32 RGB 10-bit read and write file-boundary cases."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.case_id.endswith("-dpx"))
    assert tuple((case.case_id, case.target, case.input_attribute, dict(case.kwargs)) for case in cases) == (
        ("file-read-dpx", "read_image", "read_dpx_path", {}),
        ("file-write-dpx", "write_image", "write_dpx_path", {"bit_depth": 10}),
    )


@pytest.mark.performance
def test_performance_registry_includes_both_lut_interpolation_tokens() -> None:
    """v1-lut acceptance 18; v1-lut-extensions acceptance 29: registry covers 3D and 1D LUT application."""
    assert [(case.case_id, case.target, dict(case.kwargs), dict(case.fixture_kwargs)) for case in _LUT_CASES] == [
        ("lut-transform-trilinear", "apply_lut", {"interpolation": "trilinear"}, {"lut": "lut"}),
        ("lut-transform-tetrahedral", "apply_lut", {"interpolation": "tetrahedral"}, {"lut": "lut"}),
        ("lut-transform-linear-1d", "apply_lut", {"interpolation": "linear"}, {"lut": "lut1d"}),
    ]


@pytest.mark.performance
def test_performance_registry_includes_every_lut_boundary_format() -> None:
    """v1-lut-extensions acceptance 29: registry covers every required LUT file, byte, and Cube write path."""
    lut_boundaries = tuple(
        case for case in _PERFORMANCE_CASES if case.target in {"read_lut", "decode_lut", "write_lut"}
    )

    assert {
        case.target: {item.case_id for item in lut_boundaries if item.target == case.target} for case in lut_boundaries
    } == {
        "read_lut": {
            "file-read-lut-cube-65",
            "file-read-lut-cube-1d",
            "file-read-lut-3dl",
            "file-read-lut-spi1d",
            "file-read-lut-spi3d",
        },
        "decode_lut": {
            "bytes-decode-lut-cube-1d",
            "bytes-decode-lut-cube-3d",
            "bytes-decode-lut-3dl",
            "bytes-decode-lut-spi1d",
            "bytes-decode-lut-spi3d",
        },
        "write_lut": {"file-write-lut-1d", "file-write-lut-3d"},
    }


@pytest.mark.performance
def test_performance_registry_boundary_cases_use_the_end_to_end_sampling_contract() -> None:
    """REQ-TEST-010 acceptance 8: boundary cases retain the time floor with a bounded repetition floor."""
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target in _PERFORMANCE_BOUNDARY_FUNCTIONS)

    assert {case.minimum_frames for case in cases} == {_BOUNDARY_MEASURED_MINIMUM_FRAMES}
    assert {case.minimum_seconds for case in cases} == {_MEASURED_MINIMUM_SECONDS}
    assert _BOUNDARY_MEASURED_MINIMUM_FRAMES == 20


@pytest.mark.performance
def test_performance_registry_includes_representative_stack_case() -> None:
    """v1-stack acceptance 8 / REQ-TEST-010: registry covers vertical concatenation of multiple FHD Frames."""
    assert [
        (
            case.case_id,
            case.target,
            dict(case.kwargs),
            case.input_attribute,
            case.transferred_bytes,
        )
        for case in _STACK_CASES
    ] == [
        (
            "stack-vertical-two-fhd",
            "stack",
            {"direction": "vertical", "adapt": False},
            "stack_frames",
            _FHD_FP32_RGB_BYTES * 4,
        )
    ]


@pytest.mark.performance
def test_performance_registry_includes_representative_warp_affine_case() -> None:
    """v1-warp-affine acceptance 21: registry fixes the centered FHD auto-lanczos4 representative case."""
    assert len(_WARP_AFFINE_CASES) == 1
    case = _WARP_AFFINE_CASES[0]
    assert (case.case_id, case.target, case.parameters, case.input_attribute, case.transferred_bytes) == (
        "warp-affine-fhd-auto-lanczos4",
        "warp_affine",
        "FHD fp32 RGB, centered 1.01x scale + 5deg rotation, auto lanczos4, constant 0",
        "frame",
        _FHD_FP32_RGB_READ_WRITE_BYTES,
    )
    kwargs = dict(case.kwargs)
    assert set(kwargs) == {"matrix"}
    np.testing.assert_array_equal(kwargs["matrix"], _WARP_MATRIX)
    np.testing.assert_allclose(
        np.hypot(_WARP_MATRIX[0, :2], _WARP_MATRIX[1, :2]),
        np.asarray((1.01, 1.01), dtype=np.float32),
        rtol=0.0,
        atol=2e-7,
    )


@pytest.mark.performance
def test_performance_registry_includes_representative_shuffle_cases() -> None:
    """v1-channel-shuffle acceptance 23: registry covers reorder, multi-Frame fill, and adaptation."""
    assert [
        (
            case.case_id,
            case.target,
            case.parameters,
            case.input_attribute,
            case.kwargs_attribute,
            dict(case.kwargs),
            case.transferred_bytes,
        )
        for case in _SHUFFLE_CASES
    ] == [
        (
            "shuffle-reorder-fhd",
            "shuffle",
            "single FHD fp32 Frame BGR reorder, adapt=False",
            None,
            "shuffle_reorder_outputs",
            {},
            _FHD_FP32_RGB_READ_WRITE_BYTES,
        ),
        (
            "shuffle-multi-fill-fhd",
            "shuffle",
            "FHD fp32 RGBA from 2 Frames + constant, adapt=False",
            None,
            "shuffle_multi_outputs",
            {},
            _FHD_FP32_RGB_BYTES + _FHD_FP32_RGBA_BYTES,
        ),
        (
            "shuffle-adapt-fhd",
            "shuffle",
            "2 FHD fp32 RGB Frames, sRGB/sRGB source adapted to ACEScg/linear",
            None,
            "shuffle_adapt_outputs",
            {"adapt": True},
            _FHD_FP32_RGB_BYTES * 4,
        ),
    ]


@pytest.mark.performance
def test_performance_registry_includes_representative_composite_case() -> None:
    """v1-composite acceptance 20 / REQ-TEST-010: registry covers FHD transform composition."""
    assert [
        (case.case_id, case.target, case.parameters, case.input_attribute, case.fixture_kwargs, case.transferred_bytes)
        for case in _COMPOSITE_CASES
    ] == [
        (
            "composite-transform-fhd",
            "merge",
            "FHD background + transformed 960x540 foreground, bilinear, normal",
            "frame",
            (("foreground", "composite_foreground"),),
            _FHD_FP32_RGB_BYTES + 960 * 540 * _CHANNELS * np.dtype(np.float32).itemsize + _FHD_FP32_RGB_BYTES,
        )
    ]


@pytest.mark.performance
def test_performance_registry_covers_the_value_quantization_successors() -> None:
    """v1-quantize-values acceptance 19: registry replaces legacy cases and adds both boundary directions."""
    expected = {
        ("full-to-legal-10", "full_to_legal"),
        ("legal-to-full-10", "legal_to_full"),
        ("quantize-values-8", "quantize"),
        ("dequantize-values-8", "dequantize"),
        ("to-array-bit-depth-10", "to_array"),
        ("from-array-bit-depth-10", "from_array"),
    }

    assert expected <= {(case.case_id, case.target) for case in _TRANSFORM_BOUNDARY_CASES}


@pytest.mark.performance
def test_performance_registry_includes_representative_recode_dtype_directions() -> None:
    """v1-recode-dtype acceptance 10 / REQ-TEST-010: registry covers both everyday FHD dtype directions."""
    assert hasattr(px.values, "recode_dtype")
    expected = {
        ("recode-u8-f32", "recode_dtype", "uint8 -> float32", "code8_frame", (("dtype", "float32"),)),
        ("recode-f32-u8", "recode_dtype", "float32 -> uint8", "frame", (("dtype", "uint8"),)),
    }
    actual = {
        (case.case_id, case.target, case.parameters, case.input_attribute, case.kwargs)
        for case in _TRANSFORM_BOUNDARY_CASES
        if case.target == "recode_dtype"
    }
    assert actual == expected


@pytest.mark.performance
def test_performance_registry_includes_raw_copy_bandwidth_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-performance acceptance 1-4 and provisional 8: execute the registered RawKernel copy entry."""
    expected_bytes = _WIDTH * _HEIGHT * _CHANNELS * np.dtype(np.float32).itemsize * 2
    cases = tuple(case for case in _PERFORMANCE_CASES if case.target == "copy")
    assert [
        (case.case_id, case.input_attribute, dict(case.fixture_kwargs), case.transferred_bytes) for case in cases
    ] == [("copy-fhd-fp32-rgb", "copy_source", {"destination": "copy_destination"}, expected_bytes)]

    source = cp.arange(12, dtype=cp.float32).reshape(2, 2, 3)
    destination = cp.full_like(source, np.float32(-1.0))
    copy_kernel_factory = _copy_kernel
    factory_calls: list[None] = []

    def observed_copy_kernel() -> cp.RawKernel:
        factory_calls.append(None)
        return copy_kernel_factory()

    monkeypatch.setattr(sys.modules[__name__], "_copy_kernel", observed_copy_kernel)
    result = cases[0].operation(source, destination=destination)

    assert factory_calls == [None]
    assert result is destination
    cp.testing.assert_array_equal(destination, source)


@pytest.mark.performance
def test_performance_registry_includes_hexagonal_lens_radius_32() -> None:
    """v1-performance acceptance 5-6: the lens registry includes the representative radius-32 hexagon case."""
    assert [
        (case.case_id, case.parameters, dict(case.kwargs))
        for case in _LENS_BLUR_CASES
        if case.case_id == "lens-hexagon-32"
    ] == [("lens-hexagon-32", "blades=6, radius=32", {"radius": 32.0, "blades": 6})]


@pytest.mark.performance
def test_performance_registry_includes_representative_draw_text_case() -> None:
    """v1-draw-text-unification acceptance 15; v1-draw-text-supersample acceptance 11: retain False and add True."""
    assert [
        (case.case_id, case.target, case.parameters, dict(case.kwargs)) for case in _DRAW_CASES if case.target == "text"
    ] == [
        (
            "draw-text-cjk-outline",
            "text",
            "single-line CJK, size=64, one outline, supersample=False",
            {
                "text": "pixtreme 文字描画",
                "position": (960.0, 540.0),
                "size": 64.0,
                "color": (1.2, 0.5, -0.1),
                "anchor": "center-center",
                "outlines": (((0.05, 0.1, 0.2), 2.0),),
                "supersample": False,
            },
        ),
        (
            "draw-text-cjk-outline-supersample",
            "text",
            "single-line CJK, size=64, one outline, supersample=True",
            {
                "text": "pixtreme 文字描画",
                "position": (960.0, 540.0),
                "size": 64.0,
                "color": (1.2, 0.5, -0.1),
                "anchor": "center-center",
                "outlines": (((0.05, 0.1, 0.2), 2.0),),
                "supersample": True,
            },
        ),
    ]


@pytest.mark.performance
def test_draw_text_supersample_warm_performance_gate(performance_inputs: _Inputs) -> None:
    """v1-draw-text-supersample acceptance 12: cache-hit True stays within the same-run warm ratio and latency gates."""
    text_cases = {case.case_id: case for case in _DRAW_CASES if case.target == "text"}
    synchronize = cp.cuda.Device(0).synchronize
    medians: dict[str, float] = {}
    for case_id in ("draw-text-cjk-outline", "draw-text-cjk-outline-supersample"):
        case = text_cases[case_id]
        operation = case.bind(performance_inputs)
        _run_warmup(operation, synchronize, minimum_seconds=_WARMUP_MINIMUM_SECONDS)
        durations = _measure_durations_ms(
            operation,
            synchronize,
            minimum_frames=_MEASURED_MINIMUM_FRAMES,
            minimum_seconds=_MEASURED_MINIMUM_SECONDS,
        )
        medians[case_id] = median(durations)

    normal_ms = medians["draw-text-cjk-outline"]
    sampled_ms = medians["draw-text-cjk-outline-supersample"]
    print(
        f"draw_text warm median: False={normal_ms:.6f} ms True={sampled_ms:.6f} ms ratio={sampled_ms / normal_ms:.6f}"
    )
    assert sampled_ms <= normal_ms * 1.25
    assert sampled_ms <= 1.0


@pytest.mark.performance
def test_draw_text_supersample_cold_performance_gate(performance_inputs: _Inputs) -> None:
    """v1-draw-text-supersample acceptance 13: fully cold text caches meet the same-run ratio and latency gates."""
    import pixtreme._draw.text as draw_text_module

    text_cases = {case.case_id: case for case in _DRAW_CASES if case.target == "text"}
    operations = {
        False: text_cases["draw-text-cjk-outline"].bind(performance_inputs),
        True: text_cases["draw-text-cjk-outline-supersample"].bind(performance_inputs),
    }
    synchronize = cp.cuda.Device(0).synchronize
    cache_names = (
        "_font_bytes",
        "_shape_text",
        "_freetype_face",
        "_font_metrics_26_6",
        "_font_line_advance_26_6",
        "_glyph_bitmap",
        "_block_layout",
        "_build_block_atlas",
    )

    # Time-based GPU warmup: ramp the GPU from idle to boost clocks before the cold-cache loop so
    # that early samples are not inflated by the clock ramp (I-58). Cache clearing below still makes
    # every measured iteration cold on the CPU side.
    warmup_started_at = perf_counter()
    while perf_counter() - warmup_started_at < _WARMUP_MINIMUM_SECONDS:
        for operation in operations.values():
            output = operation()
            synchronize()
            del output

    durations: dict[bool, list[float]] = {False: [], True: []}
    for _sample in range(20):
        for supersample in (False, True):
            for name in cache_names:
                getattr(draw_text_module, name).cache_clear()
            synchronize()
            started_at = perf_counter()
            output = operations[supersample]()
            synchronize()
            durations[supersample].append((perf_counter() - started_at) * 1000.0)
            del output

    normal_ms = median(durations[False])
    sampled_ms = median(durations[True])
    print(
        f"draw_text cold median: False={normal_ms:.6f} ms True={sampled_ms:.6f} ms ratio={sampled_ms / normal_ms:.6f}"
    )
    assert sampled_ms <= normal_ms * 12.0
    assert sampled_ms <= 120.0


@pytest.mark.performance
def test_performance_registry_includes_representative_generator_cases() -> None:
    """v1-generator acceptance 15 and 19: performance covers one FHD case for each deterministic generator."""
    assert {case.target for case in _GENERATOR_CASES} == {
        "ramp",
        "grid",
        "checkerboard",
        "color_bars",
    }
    assert all(case.transferred_bytes == _FHD_FP32_RGB_BYTES for case in _GENERATOR_CASES)


@pytest.mark.performance
def test_performance_registry_includes_representative_noise_cases() -> None:
    """v1-noise acceptance 13: performance registry covers each GPU per-pixel noise generator."""
    assert {case.target for case in _NOISE_CASES} == {
        "fractal_noise",
        "turbulent_noise",
        "grain",
    }
    assert [case.transferred_bytes for case in _NOISE_CASES] == [
        _FHD_FP32_Y_BYTES,
        _FHD_FP32_Y_BYTES,
        _FHD_FP32_RGB_BYTES,
    ]


@pytest.mark.performance
def test_performance_warmup_runs_until_minimum_elapsed_time() -> None:
    """v1-performance acceptance 2: warmup continues through synchronized iterations for at least 0.5 seconds."""
    timer_values = iter((0.0, 0.2, 0.4, 0.6))
    operation_calls: list[None] = []
    synchronize_calls: list[None] = []

    _run_warmup(
        lambda: operation_calls.append(None),
        lambda: synchronize_calls.append(None),
        minimum_seconds=_WARMUP_MINIMUM_SECONDS,
        timer=lambda: next(timer_values),
    )

    assert _WARMUP_MINIMUM_SECONDS == 0.5
    assert len(operation_calls) == 3
    assert len(synchronize_calls) == 3


@pytest.mark.performance
def test_performance_measurement_runs_until_minimum_frame_count() -> None:
    """v1-performance acceptance 3: measurement does not stop before the minimum frame count."""
    timer_values = iter((0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08))

    durations_ms = _measure_durations_ms(
        lambda: None,
        lambda: None,
        minimum_frames=4,
        minimum_seconds=0.0,
        timer=lambda: next(timer_values),
    )

    assert len(durations_ms) == 4


@pytest.mark.performance
def test_performance_measurement_runs_until_minimum_elapsed_time() -> None:
    """v1-performance acceptance 3: measurement continues after 1000 frames until the minimum duration is met."""
    timer_values = iter((0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06))

    durations_ms = _measure_durations_ms(
        lambda: None,
        lambda: None,
        minimum_frames=2,
        minimum_seconds=0.05,
        timer=lambda: next(timer_values),
    )

    assert len(durations_ms) == 3


@pytest.mark.performance
def test_performance_metrics_include_linear_percentiles_and_median_fps() -> None:
    """v1-performance acceptance 3-4: metrics report linear p5/p95 and median-derived fps."""
    metrics = _performance_metrics([1.0, 2.0, 3.0, 4.0, 5.0])

    assert metrics.mean_ms == pytest.approx(3.0)
    assert metrics.median_ms == pytest.approx(3.0)
    assert metrics.fps == pytest.approx(1000.0 / 3.0)
    assert metrics.p5_ms == pytest.approx(1.2)
    assert metrics.p95_ms == pytest.approx(4.8)


@pytest.mark.performance
@pytest.mark.parametrize("case", _PERFORMANCE_CASES, ids=lambda case: case.case_id)
def test_fhd_steady_state_performance_report(
    case: _PerformanceCase,
    performance_inputs: _Inputs,
    performance_results: list[tuple[str, str, float, float, float, float, float, float]],
) -> None:
    """REQ-TEST-010: report FHD timing and dispersion after excluded time-based warmup."""
    operation = case.bind(performance_inputs)
    synchronize = cp.cuda.Device().synchronize

    _run_warmup(operation, synchronize, minimum_seconds=_WARMUP_MINIMUM_SECONDS)
    durations_ms = _measure_durations_ms(
        operation,
        synchronize,
        minimum_frames=case.minimum_frames,
        minimum_seconds=case.minimum_seconds,
    )

    metrics = _performance_metrics(durations_ms)
    effective_gbps = case.transferred_bytes / (metrics.median_ms * 1_000_000.0)
    performance_results.append(
        (
            case.target,
            case.parameters,
            metrics.mean_ms,
            metrics.median_ms,
            metrics.fps,
            metrics.p5_ms,
            metrics.p95_ms,
            effective_gbps,
        )
    )
