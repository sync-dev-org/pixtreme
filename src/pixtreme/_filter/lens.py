"""Flat-aperture lens blur with coverage-weighted direct convolution."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Integral, Real
from typing import cast

import cupy as cp
import numpy as np
from cupyx.scipy.fft import next_fast_len

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._core.vocabulary import Border
from pixtreme._filter.common import _shape_arguments

_SUBSAMPLES = 16
_SUBPIXEL_CENTERS = (np.arange(_SUBSAMPLES, dtype=np.float64) + 0.5) / _SUBSAMPLES - 0.5
_LOCAL_X, _LOCAL_Y = np.meshgrid(_SUBPIXEL_CENTERS, _SUBPIXEL_CENTERS)
_COVERAGE_BATCH_SIZE = 4096
_COVERAGE_RASTER_MAX_SAMPLES = 4 * 1024 * 1024
_LENS_LUT_CACHE_SIZE = 32
_LENS_FFT_CACHE_SIZE = 16
# The FHD crossover is about 180 taps; tap count adapts the cutoff to aperture area.
_LENS_FFT_MIN_TAPS = 180
_RAW_KERNEL_BLOCK = (32, 8)
_RAW_KERNEL_PAD_BLOCK = 256
_RAW_KERNEL_SHARED_LIMIT = 48 * 1024
_RAW_KERNEL_GRID_Z_LIMIT = 65535

_LENS_BLUR_KERNEL_SOURCE = (
    _BORDER_PREAMBLE
    + r"""
extern "C" __global__ void pixtreme_blur_lens_shared(
    const float* __restrict__ source,
    const int2* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long tap_count,
    const long long bound,
    const int border,
    const float border_value
) {
    extern __shared__ float tile[];
    const long long tile_width = blockDim.x + 2 * bound;
    const long long tile_height = blockDim.y + 2 * bound;
    const long long tile_element_count = tile_width * tile_height;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;
    const long long channel = blockIdx.z;

    for (
        long long tile_index = thread_index;
        tile_index < tile_element_count;
        tile_index += thread_count
    ) {
        const long long local_x = tile_index % tile_width;
        const long long local_y = tile_index / tile_width;
        tile[tile_index] = pixtreme_border_sample(
            source,
            block_x + local_x - bound,
            block_y + local_y - bound,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }
    __syncthreads();

    const long long output_x = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_x >= width || output_y >= height) {
        return;
    }
    float weighted_sum = 0.0f;
    for (long long tap = 0; tap < tap_count; ++tap) {
        const int2 offset = offsets[tap];
        const long long local_x = threadIdx.x + bound - (long long)offset.x;
        const long long local_y = threadIdx.y + bound - (long long)offset.y;
        weighted_sum += tile[local_y * tile_width + local_x] * weights[tap];
    }
    output[(output_y * width + output_x) * channel_count + channel] = weighted_sum;
}

extern "C" __global__ void pixtreme_blur_lens_global(
    const float* __restrict__ source,
    const int2* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long tap_count,
    const long long bound,
    const int border,
    const float border_value
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= width || output_y >= height) {
        return;
    }
    for (long long channel = 0; channel < channel_count; ++channel) {
        float weighted_sum = 0.0f;
        for (long long tap = 0; tap < tap_count; ++tap) {
            const int2 offset = offsets[tap];
            weighted_sum += pixtreme_border_sample(
                source,
                output_x - (long long)offset.x,
                output_y - (long long)offset.y,
                width,
                height,
                channel_count,
                channel,
                border,
                border_value
            ) * weights[tap];
        }
        output[(output_y * width + output_x) * channel_count + channel] = weighted_sum;
    }
}

extern "C" __global__ void pixtreme_blur_lens_pad(
    const float* __restrict__ source,
    float* __restrict__ padded,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long bound,
    const int border,
    const float border_value
) {
    const long long padded_width = width + 2 * bound;
    const long long padded_height = height + 2 * bound;
    const long long padded_pixel_count = padded_width * padded_height;
    const long long element_count = padded_pixel_count * channel_count;
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const long long channel = index / padded_pixel_count;
    const long long pixel_index = index - channel * padded_pixel_count;
    const long long padded_x = pixel_index % padded_width;
    const long long padded_y = pixel_index / padded_width;
    padded[index] = pixtreme_border_sample(
        source,
        padded_x - bound,
        padded_y - bound,
        width,
        height,
        channel_count,
        channel,
        border,
        border_value
    );
}
"""
)


@lru_cache(maxsize=1)
def _lens_blur_shared_kernel() -> cp.RawKernel:
    return cp.RawKernel(_LENS_BLUR_KERNEL_SOURCE, "pixtreme_blur_lens_shared")


@lru_cache(maxsize=1)
def _lens_blur_global_kernel() -> cp.RawKernel:
    return cp.RawKernel(_LENS_BLUR_KERNEL_SOURCE, "pixtreme_blur_lens_global")


@lru_cache(maxsize=1)
def _lens_blur_pad_kernel() -> cp.RawKernel:
    return cp.RawKernel(_LENS_BLUR_KERNEL_SOURCE, "pixtreme_blur_lens_pad")


def _validate_radius(radius: object) -> float:
    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise ValueError(
            _actionable_error(
                why="lens_blur radius must be a nonnegative finite real number",
                what=f"received radius={radius!r}",
                how="pass a finite int or float radius of at least 0",
            )
        )
    resolved = float(radius)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(
            _actionable_error(
                why="lens_blur radius must be finite and nonnegative",
                what=f"received radius={radius!r}",
                how="pass a finite real radius of at least 0; there is no fixed upper bound",
            )
        )
    return resolved


def _validate_blades(blades: object) -> int | None:
    if blades is None:
        return None
    if isinstance(blades, bool) or not isinstance(blades, Integral) or blades < 3:
        raise ValueError(
            _actionable_error(
                why="lens_blur blades must be None or a non-bool integer of at least 3",
                what=f"received blades={blades!r}",
                how="pass blades=None for a circle or pass an integer blade count of at least 3",
            )
        )
    return int(blades)


def _resolve_rotation(rotation: object, *, blades: int | None) -> float:
    if blades is None:
        if rotation is not None:
            raise ValueError(
                _actionable_error(
                    why="lens_blur rotation applies only to polygon apertures",
                    what=f"received blades=None with rotation={rotation!r}",
                    how="omit rotation for a circular aperture, or pass blades of at least 3",
                )
            )
        return 0.0
    if rotation is None:
        return 0.0
    if isinstance(rotation, bool) or not isinstance(rotation, Real):
        raise ValueError(
            _actionable_error(
                why="lens_blur rotation must be a finite real number of degrees",
                what=f"received rotation={rotation!r}",
                how="pass a finite int or float rotation, or omit it for 0 degrees",
            )
        )
    resolved = float(rotation)
    if not math.isfinite(resolved):
        raise ValueError(
            _actionable_error(
                why="lens_blur rotation must be finite",
                what=f"received rotation={rotation!r}",
                how="pass a finite real rotation in degrees",
            )
        )
    return resolved


def _polygon_vertices(*, radius: float, blades: int, rotation: float) -> np.ndarray:
    # Period reduction preserves the specified geometry while keeping trig input bounded.
    angles = math.radians(math.fmod(rotation, 360.0)) + np.arange(blades, dtype=np.float64) * (2.0 * np.pi / blades)
    return np.stack((radius * np.cos(angles), -radius * np.sin(angles)), axis=1)


def _aperture_coverage_counts(
    *,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    radius: float,
    vertices: np.ndarray | None,
) -> np.ndarray:
    samples_x = offsets_x[:, np.newaxis, np.newaxis] + _LOCAL_X
    samples_y = offsets_y[:, np.newaxis, np.newaxis] + _LOCAL_Y
    if vertices is None:
        inside = samples_x * samples_x + samples_y * samples_y <= radius * radius
    else:
        inside = np.ones(samples_x.shape, dtype=np.bool_)
        for vertex_index in range(len(vertices)):
            start = vertices[vertex_index]
            end = vertices[(vertex_index + 1) % len(vertices)]
            edge_x, edge_y = end - start
            cross = edge_x * (samples_y - start[1]) - edge_y * (samples_x - start[0])
            inside &= cross <= 0.0
    return cast(np.ndarray, np.count_nonzero(inside, axis=(1, 2)).astype(np.uint16, copy=False))


def _aperture_coverage_raster(
    *,
    axis_offsets: np.ndarray,
    radius: float,
    vertices: np.ndarray | None,
) -> np.ndarray:
    sample_axis = (axis_offsets[:, np.newaxis] + _SUBPIXEL_CENTERS).reshape(-1)
    samples_x = sample_axis[np.newaxis, :]
    samples_y = sample_axis[:, np.newaxis]
    if vertices is None:
        inside = samples_x * samples_x + samples_y * samples_y <= radius * radius
    else:
        inside = np.ones((sample_axis.size, sample_axis.size), dtype=np.bool_)
        for vertex_index in range(len(vertices)):
            start = vertices[vertex_index]
            end = vertices[(vertex_index + 1) % len(vertices)]
            edge_x, edge_y = end - start
            cross = edge_x * (samples_y - start[1]) - edge_y * (samples_x - start[0])
            inside &= cross <= 0.0
    axis_size = axis_offsets.size
    coverage = inside.reshape(axis_size, _SUBSAMPLES, axis_size, _SUBSAMPLES)
    return cast(np.ndarray, coverage.sum(axis=(1, 3), dtype=np.uint16).reshape(-1))


def _generate_aperture_kernel(
    *, radius: float, blades: int | None, rotation: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bound = math.ceil(radius + 0.5)
    vertices = None if blades is None else _polygon_vertices(radius=radius, blades=blades, rotation=rotation)
    axis_offsets = np.arange(-bound, bound + 1, dtype=np.int32)
    candidate_offsets_x = np.tile(axis_offsets, axis_offsets.size)
    candidate_offsets_y = np.repeat(axis_offsets, axis_offsets.size)
    raster_axis_size = axis_offsets.size * _SUBSAMPLES
    if raster_axis_size * raster_axis_size <= _COVERAGE_RASTER_MAX_SAMPLES:
        coverage_counts = _aperture_coverage_raster(
            axis_offsets=axis_offsets,
            radius=radius,
            vertices=vertices,
        )
    else:
        coverage_counts = np.empty(candidate_offsets_x.size, dtype=np.uint16)
        for start in range(0, candidate_offsets_x.size, _COVERAGE_BATCH_SIZE):
            stop = min(start + _COVERAGE_BATCH_SIZE, candidate_offsets_x.size)
            coverage_counts[start:stop] = _aperture_coverage_counts(
                offsets_x=candidate_offsets_x[start:stop],
                offsets_y=candidate_offsets_y[start:stop],
                radius=radius,
                vertices=vertices,
            )

    covered = coverage_counts > 0
    offsets_x = np.ascontiguousarray(candidate_offsets_x[covered])
    offsets_y = np.ascontiguousarray(candidate_offsets_y[covered])
    weights = coverage_counts[covered].astype(np.float32)
    if weights.size:
        weights /= np.float32(_SUBSAMPLES * _SUBSAMPLES)
        weights /= np.sum(weights, dtype=np.float32)
    return offsets_x, offsets_y, weights


@lru_cache(maxsize=_LENS_LUT_CACHE_SIZE)
def _aperture_lut(
    device_id: int,
    radius: float,
    blades: int | None,
    rotation: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    host_offset_x, host_offset_y, host_weights = _generate_aperture_kernel(
        radius=radius,
        blades=blades,
        rotation=rotation,
    )
    host_offsets = np.empty((host_weights.size, 2), dtype=np.int32)
    host_offsets[:, 0] = host_offset_x
    host_offsets[:, 1] = host_offset_y
    with cp.cuda.Device(device_id):
        return cp.asarray(host_offsets), cp.asarray(host_weights)


@lru_cache(maxsize=_LENS_FFT_CACHE_SIZE)
def _aperture_fft_lut(
    device_id: int,
    height: int,
    width: int,
    radius: float,
    blades: int | None,
    rotation: float,
) -> tuple[cp.ndarray, tuple[int, int], int]:
    host_offset_x, host_offset_y, host_weights = _generate_aperture_kernel(
        radius=radius,
        blades=blades,
        rotation=rotation,
    )
    bound = math.ceil(radius + 0.5)
    host_kernel = np.zeros((2 * bound + 1, 2 * bound + 1), dtype=np.float32)
    host_kernel[host_offset_y + bound, host_offset_x + bound] = host_weights
    fft_shape = (
        int(next_fast_len(height + 2 * bound)),
        int(next_fast_len(width + 2 * bound)),
    )
    with cp.cuda.Device(device_id):
        spectrum = cp.fft.rfftn(cp.asarray(host_kernel), s=fft_shape, axes=(0, 1))
    return spectrum, fft_shape, bound


def _launch_lens_blur(
    source: cp.ndarray,
    offsets: cp.ndarray,
    weights: cp.ndarray,
    output: cp.ndarray,
    *,
    frame: Frame,
    bound: int,
    border: Border,
    border_value: float,
) -> None:
    block_x, block_y = _RAW_KERNEL_BLOCK
    grid_xy = ((frame.width + block_x - 1) // block_x, (frame.height + block_y - 1) // block_y)
    tile_width = block_x + 2 * bound
    tile_height = block_y + 2 * bound
    shared_bytes = tile_width * tile_height * np.dtype(np.float32).itemsize
    channel_count = len(frame.channels)
    use_shared = shared_bytes <= _RAW_KERNEL_SHARED_LIMIT and channel_count <= _RAW_KERNEL_GRID_Z_LIMIT
    kernel = _lens_blur_shared_kernel() if use_shared else _lens_blur_global_kernel()
    grid = (*grid_xy, channel_count) if use_shared else grid_xy
    kernel(
        grid,
        _RAW_KERNEL_BLOCK,
        (
            source,
            offsets,
            weights,
            output,
            *_shape_arguments(frame),
            np.int64(weights.size),
            np.int64(bound),
            _border_argument(border),
            np.float32(border_value),
        ),
        shared_mem=shared_bytes if use_shared else 0,
    )


def _launch_lens_blur_fft(
    source: cp.ndarray,
    kernel_spectrum: cp.ndarray,
    *,
    frame: Frame,
    fft_shape: tuple[int, int],
    bound: int,
    border: Border,
    border_value: float,
) -> cp.ndarray:
    padded_shape = (len(frame.channels), frame.height + 2 * bound, frame.width + 2 * bound)
    padded = cp.empty(padded_shape, dtype=cp.float32)
    element_count = math.prod(padded_shape)
    _lens_blur_pad_kernel()(
        ((element_count + _RAW_KERNEL_PAD_BLOCK - 1) // _RAW_KERNEL_PAD_BLOCK,),
        (_RAW_KERNEL_PAD_BLOCK,),
        (
            source,
            padded,
            *_shape_arguments(frame),
            np.int64(bound),
            _border_argument(border),
            np.float32(border_value),
        ),
    )
    transformed = cp.fft.rfftn(padded, s=fft_shape, axes=(1, 2))
    transformed *= kernel_spectrum[np.newaxis, :, :]
    convolved = cp.fft.irfftn(transformed, s=fft_shape, axes=(1, 2))
    crop_start = 2 * bound
    return cp.ascontiguousarray(
        convolved[
            :,
            crop_start : crop_start + frame.height,
            crop_start : crop_start + frame.width,
        ].transpose(1, 2, 0)
    )


def lens_blur(
    frame: Frame,
    *,
    radius: float,
    blades: int | None = None,
    rotation: float | None = None,
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Convolve with a flat uniform aperture for spatially invariant bokeh.

    This is the optical flat-aperture counterpart of lens blur: with
    scene-linear input, highlights above 1.0 open into bokeh shaped like the
    aperture. ``radius`` is the circle radius or a regular polygon's
    circumradius. At the same radius, changing ``blades`` changes aperture area.
    ``blades=None`` selects a circle; an integer of at least 3 selects a regular
    polygon.

    ``rotation`` is available only with ``blades``. A vertex is at +x at
    0 degrees, and positive rotation is visually counterclockwise. Every kernel
    pixel uses partial coverage from a fixed 16 x 16 center-subsample grid, then
    all nonzero coverage weights are normalized for convolution.

    Border defaults to ``mirror``; ``replicate`` clamps to the edge and ``wrap``
    uses periodic indexing. ``constant`` uses ``border_value`` outside the image.
    ``border_value`` is required with ``constant`` and forbidden for every other
    border. Calculation is fp32 independently per channel and does not clamp
    negative scene values or values above 1. The result has new storage and
    unchanged Frame metadata.

    ``radius = 0`` is an exact identity in new storage. A positive radius whose
    fixed grid has zero aperture coverage follows the same exact identity rule.
    Small apertures use direct convolution, whose cost grows in proportion to
    radius squared; larger apertures use equivalent FFT convolution.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.lens_blur")
    checked_radius = _validate_radius(radius)
    checked_blades = _validate_blades(blades)
    checked_rotation = _resolve_rotation(rotation, blades=checked_blades)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    if checked_radius == 0.0:
        return _new_frame(checked_frame, checked_frame.data.copy())

    device_offsets, device_weights = _aperture_lut(
        cp.cuda.runtime.getDevice(),
        checked_radius,
        checked_blades,
        checked_rotation,
    )
    if not device_weights.size:
        return _new_frame(checked_frame, checked_frame.data.copy())

    if device_weights.size >= _LENS_FFT_MIN_TAPS:
        kernel_spectrum, fft_shape, bound = _aperture_fft_lut(
            cp.cuda.runtime.getDevice(),
            checked_frame.height,
            checked_frame.width,
            checked_radius,
            checked_blades,
            checked_rotation,
        )
        output = _launch_lens_blur_fft(
            checked_frame.data,
            kernel_spectrum,
            frame=checked_frame,
            fft_shape=fft_shape,
            bound=bound,
            border=checked_border,
            border_value=checked_border_value,
        )
    else:
        output = cp.empty(checked_frame.shape, dtype=cp.float32)
        _launch_lens_blur(
            checked_frame.data,
            device_offsets,
            device_weights,
            output,
            frame=checked_frame,
            bound=math.ceil(checked_radius + 0.5),
            border=checked_border,
            border_value=checked_border_value,
        )
    return _new_frame(checked_frame, output)
