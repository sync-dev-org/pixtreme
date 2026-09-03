"""GPU geometry resize with explicit interpolation contracts."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _validate_float32_frame
from pixtreme._core.interpolation import _POINT_INTERPOLATION_DEVICE_SOURCE, _POINT_INTERPOLATION_TOKENS
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.vocabulary import Interpolation

_INTERPOLATION_TOKENS = (*_POINT_INTERPOLATION_TOKENS, "area")

_RAW_KERNEL_BLOCK = (32, 8)
_AXIS_PLAN_BLOCK = 256
_AXIS_PLAN_CACHE_SIZE = 32
_AXIS_PLAN_CACHE_MAX_BYTES = 2 * 1024 * 1024

_RESIZE_KERNEL_SOURCE = (
    _POINT_INTERPOLATION_DEVICE_SOURCE
    + r"""
__device__ long long pixtreme_clamp_index(const long long index, const long long extent) {
    return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
}

extern "C" __global__ void pixtreme_resize_nearest(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long input_width,
    const long long input_height,
    const long long output_width,
    const long long output_height,
    const long long channel_count
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= output_width || output_y >= output_height) {
        return;
    }
    const float source_x = ((float)output_x + 0.5f) * (float)input_width / (float)output_width - 0.5f;
    const float source_y = ((float)output_y + 0.5f) * (float)input_height / (float)output_height - 0.5f;
    const long long nearest_x = pixtreme_clamp_index((long long)floorf(source_x + 0.5f), input_width);
    const long long nearest_y = pixtreme_clamp_index((long long)floorf(source_y + 0.5f), input_height);
    const long long source_offset = (nearest_y * input_width + nearest_x) * channel_count;
    const long long output_offset = (output_y * output_width + output_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        output[output_offset + channel] = source[source_offset + channel];
    }
}

extern "C" __global__ void pixtreme_resize_build_point_axis(
    long long* __restrict__ indices,
    float* __restrict__ weights,
    const long long input_extent,
    const long long output_extent,
    const int interpolation,
    const int sample_count
) {
    const long long output_coordinate = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (output_coordinate >= output_extent) {
        return;
    }
    const float source_coordinate =
        ((float)output_coordinate + 0.5f) * (float)input_extent / (float)output_extent - 0.5f;
    const long long base = (long long)floorf(source_coordinate);
    const int lobes = interpolation >= 5 ? interpolation - 3 : 0;
    const long long start = interpolation == 1 ? base : (base - (lobes > 0 ? lobes - 1 : 1));
    const long long plan_offset = output_coordinate * sample_count;
    float weight_sum = 0.0f;

    #pragma unroll
    for (int offset = 0; offset < sample_count; ++offset) {
        const long long sample = start + offset;
        const float weight = pixtreme_point_weight(interpolation, source_coordinate - (float)sample);
        indices[plan_offset + offset] = pixtreme_clamp_index(sample, input_extent);
        weights[plan_offset + offset] = weight;
        weight_sum += weight;
    }
    const float inverse_weight_sum = weight_sum != 0.0f ? 1.0f / weight_sum : 0.0f;
    #pragma unroll
    for (int offset = 0; offset < sample_count; ++offset) {
        weights[plan_offset + offset] *= inverse_weight_sum;
    }
}

extern "C" __global__ void pixtreme_resize_bilinear(
    const float* __restrict__ source,
    const long long* __restrict__ horizontal_indices,
    const float* __restrict__ horizontal_weights,
    const long long* __restrict__ vertical_indices,
    const float* __restrict__ vertical_weights,
    float* __restrict__ output,
    const long long input_width,
    const long long output_width,
    const long long output_height,
    const long long channel_count
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= output_width || output_y >= output_height) {
        return;
    }
    const long long horizontal_offset = output_x * 2;
    const long long vertical_offset = output_y * 2;
    const long long source_x0 = horizontal_indices[horizontal_offset];
    const long long source_x1 = horizontal_indices[horizontal_offset + 1];
    const long long source_y0 = vertical_indices[vertical_offset];
    const long long source_y1 = vertical_indices[vertical_offset + 1];
    const float weight_x0 = horizontal_weights[horizontal_offset];
    const float weight_x1 = horizontal_weights[horizontal_offset + 1];
    const float weight_y0 = vertical_weights[vertical_offset];
    const float weight_y1 = vertical_weights[vertical_offset + 1];
    const long long output_offset = (output_y * output_width + output_x) * channel_count;

    for (long long channel = 0; channel < channel_count; ++channel) {
        const float top =
            source[(source_y0 * input_width + source_x0) * channel_count + channel] * weight_x0 +
            source[(source_y0 * input_width + source_x1) * channel_count + channel] * weight_x1;
        const float bottom =
            source[(source_y1 * input_width + source_x0) * channel_count + channel] * weight_x0 +
            source[(source_y1 * input_width + source_x1) * channel_count + channel] * weight_x1;
        output[output_offset + channel] = top * weight_y0 + bottom * weight_y1;
    }
}

extern "C" __global__ void pixtreme_resize_point_horizontal(
    const float* __restrict__ source,
    const long long* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long input_width,
    const long long input_height,
    const long long output_width,
    const long long channel_count,
    const int sample_count
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= output_width || output_y >= input_height) {
        return;
    }
    const long long plan_offset = output_x * sample_count;
    const long long output_offset = (output_y * output_width + output_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        float value = 0.0f;
        #pragma unroll
        for (int offset = 0; offset < sample_count; ++offset) {
            const long long sample_x = indices[plan_offset + offset];
            const long long source_index = (output_y * input_width + sample_x) * channel_count + channel;
            value += source[source_index] * weights[plan_offset + offset];
        }
        output[output_offset + channel] = value;
    }
}

extern "C" __global__ void pixtreme_resize_point_vertical(
    const float* __restrict__ source,
    const long long* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long input_height,
    const long long output_height,
    const long long channel_count,
    const int sample_count
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= width || output_y >= output_height) {
        return;
    }
    const long long plan_offset = output_y * sample_count;
    const long long output_offset = (output_y * width + output_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        float value = 0.0f;
        #pragma unroll
        for (int offset = 0; offset < sample_count; ++offset) {
            const long long sample_y = indices[plan_offset + offset];
            const long long source_index = (sample_y * width + output_x) * channel_count + channel;
            value += source[source_index] * weights[plan_offset + offset];
        }
        output[output_offset + channel] = value;
    }
}

extern "C" __global__ void pixtreme_resize_area_horizontal(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long input_width,
    const long long input_height,
    const long long output_width,
    const long long channel_count
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= output_width || output_y >= input_height) {
        return;
    }
    const float left = (float)output_x * (float)input_width / (float)output_width;
    const float right = (float)(output_x + 1) * (float)input_width / (float)output_width;
    const long long start_x = (long long)floorf(left);
    const long long stop_x = (long long)ceilf(right);
    float weight_sum = 0.0f;
    for (long long source_x = start_x; source_x < stop_x; ++source_x) {
        weight_sum += fmaxf(
            0.0f,
            fminf(right, (float)(source_x + 1)) - fmaxf(left, (float)source_x)
        );
    }

    const long long output_offset = (output_y * output_width + output_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        float value = 0.0f;
        for (long long source_x = start_x; source_x < stop_x; ++source_x) {
            const float weight_x = fmaxf(
                0.0f,
                fminf(right, (float)(source_x + 1)) - fmaxf(left, (float)source_x)
            );
            const long long clamped_x = pixtreme_clamp_index(source_x, input_width);
            const long long source_index = (output_y * input_width + clamped_x) * channel_count + channel;
            value += source[source_index] * weight_x;
        }
        output[output_offset + channel] = weight_sum != 0.0f ? value / weight_sum : 0.0f;
    }
}

extern "C" __global__ void pixtreme_resize_area_vertical(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long input_height,
    const long long output_height,
    const long long channel_count
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= width || output_y >= output_height) {
        return;
    }
    const float top = (float)output_y * (float)input_height / (float)output_height;
    const float bottom = (float)(output_y + 1) * (float)input_height / (float)output_height;
    const long long start_y = (long long)floorf(top);
    const long long stop_y = (long long)ceilf(bottom);
    float weight_sum = 0.0f;
    for (long long source_y = start_y; source_y < stop_y; ++source_y) {
        weight_sum += fmaxf(
            0.0f,
            fminf(bottom, (float)(source_y + 1)) - fmaxf(top, (float)source_y)
        );
    }

    const long long output_offset = (output_y * width + output_x) * channel_count;
    for (long long channel = 0; channel < channel_count; ++channel) {
        float value = 0.0f;
        for (long long source_y = start_y; source_y < stop_y; ++source_y) {
            const float weight_y = fmaxf(
                0.0f,
                fminf(bottom, (float)(source_y + 1)) - fmaxf(top, (float)source_y)
            );
            const long long clamped_y = pixtreme_clamp_index(source_y, input_height);
            const long long source_index = (clamped_y * width + output_x) * channel_count + channel;
            value += source[source_index] * weight_y;
        }
        output[output_offset + channel] = weight_sum != 0.0f ? value / weight_sum : 0.0f;
    }
}
"""
)


@lru_cache(maxsize=1)
def _nearest_kernel() -> cp.RawKernel:
    return cp.RawKernel(_RESIZE_KERNEL_SOURCE, "pixtreme_resize_nearest")


@lru_cache(maxsize=1)
def _point_axis_plan_kernel() -> cp.RawKernel:
    return cp.RawKernel(_RESIZE_KERNEL_SOURCE, "pixtreme_resize_build_point_axis")


@lru_cache(maxsize=1)
def _bilinear_kernel() -> cp.RawKernel:
    return cp.RawKernel(_RESIZE_KERNEL_SOURCE, "pixtreme_resize_bilinear")


@lru_cache(maxsize=1)
def _point_horizontal_kernel() -> cp.RawKernel:
    return cp.RawKernel(_RESIZE_KERNEL_SOURCE, "pixtreme_resize_point_horizontal")


@lru_cache(maxsize=1)
def _point_vertical_kernel() -> cp.RawKernel:
    return cp.RawKernel(_RESIZE_KERNEL_SOURCE, "pixtreme_resize_point_vertical")


@lru_cache(maxsize=1)
def _area_horizontal_kernel() -> cp.RawKernel:
    return cp.RawKernel(_RESIZE_KERNEL_SOURCE, "pixtreme_resize_area_horizontal")


@lru_cache(maxsize=1)
def _area_vertical_kernel() -> cp.RawKernel:
    return cp.RawKernel(_RESIZE_KERNEL_SOURCE, "pixtreme_resize_area_vertical")


def _point_sample_count(interpolation_index: int) -> int:
    if interpolation_index == 1:
        return 2
    if interpolation_index >= 5:
        return 2 * (interpolation_index - 3)
    return 4


def _build_point_axis_plan(
    device_id: int,
    input_extent: int,
    output_extent: int,
    interpolation_index: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    sample_count = _point_sample_count(interpolation_index)
    with cp.cuda.Device(device_id):
        indices = cp.empty(output_extent * sample_count, dtype=cp.int64)
        weights = cp.empty(output_extent * sample_count, dtype=cp.float32)
        blocks = (output_extent + _AXIS_PLAN_BLOCK - 1) // _AXIS_PLAN_BLOCK
        _point_axis_plan_kernel()(
            (blocks,),
            (_AXIS_PLAN_BLOCK,),
            (
                indices,
                weights,
                np.int64(input_extent),
                np.int64(output_extent),
                np.int32(interpolation_index),
                np.int32(sample_count),
            ),
        )
    return indices, weights


@lru_cache(maxsize=_AXIS_PLAN_CACHE_SIZE)
def _cached_point_axis_plan(
    device_id: int,
    input_extent: int,
    output_extent: int,
    interpolation_index: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    return _build_point_axis_plan(device_id, input_extent, output_extent, interpolation_index)


def _point_axis_plan(
    input_extent: int,
    output_extent: int,
    interpolation_index: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    sample_count = _point_sample_count(interpolation_index)
    plan_bytes = output_extent * sample_count * (np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize)
    device_id = cp.cuda.runtime.getDevice()
    if plan_bytes <= _AXIS_PLAN_CACHE_MAX_BYTES:
        return _cached_point_axis_plan(device_id, input_extent, output_extent, interpolation_index)
    return _build_point_axis_plan(device_id, input_extent, output_extent, interpolation_index)


def _launch_raw_kernel(
    kernel: cp.RawKernel,
    output_width: int,
    output_height: int,
    arguments: tuple[object, ...],
) -> None:
    block_x, block_y = _RAW_KERNEL_BLOCK
    grid = ((output_width + block_x - 1) // block_x, (output_height + block_y - 1) // block_y)
    kernel(grid, _RAW_KERNEL_BLOCK, arguments)


def _resolve_output_size(
    frame: Frame,
    *,
    width: int | None,
    height: int | None,
    factor: float | None,
) -> tuple[int, int]:
    explicit_size = width is not None and height is not None and factor is None
    factor_size = width is None and height is None and factor is not None
    if not (explicit_size or factor_size):
        raise ValueError(
            _actionable_error(
                why="resize has exactly two mutually exclusive size modes",
                what=f"received width={width!r}, height={height!r}, factor={factor!r}",
                how="pass width and height together, or pass factor alone",
            )
        )

    if explicit_size:
        if type(width) is not int or type(height) is not int or width < 1 or height < 1:
            raise ValueError(
                _actionable_error(
                    why="explicit resize dimensions must be positive integers",
                    what=f"received width={width!r}, height={height!r}",
                    how="pass int width and height values of at least 1",
                )
            )
        return width, height

    if isinstance(factor, bool) or not isinstance(factor, Real):
        raise ValueError(
            _actionable_error(
                why="resize factor must be a positive real number",
                what=f"received factor={factor!r}",
                how="pass a finite int or float greater than 0",
            )
        )
    try:
        resolved_factor = float(factor)
    except (OverflowError, TypeError, ValueError) as conversion_error:
        raise ValueError(
            _actionable_error(
                why="resize factor must be finite and greater than 0",
                what=f"received factor={factor!r}",
                how="pass a finite positive real number",
            )
        ) from conversion_error
    if not math.isfinite(resolved_factor) or resolved_factor <= 0.0:
        raise ValueError(
            _actionable_error(
                why="resize factor must be finite and greater than 0",
                what=f"received factor={factor!r}",
                how="pass a finite positive real number",
            )
        )
    scaled_width = frame.width * resolved_factor
    scaled_height = frame.height * resolved_factor
    maximum_dimension = np.iinfo(np.intp).max
    if (
        not math.isfinite(scaled_width)
        or not math.isfinite(scaled_height)
        or scaled_width > maximum_dimension
        or scaled_height > maximum_dimension
    ):
        raise ValueError(
            _actionable_error(
                why="factor-derived resize dimensions must be finite and representable",
                what=(f"factor={factor!r} produces scaled width={scaled_width!r}, scaled height={scaled_height!r}"),
                how="pass a smaller factor or explicit positive width and height",
            )
        )
    output_width = math.floor(scaled_width + 0.5)
    output_height = math.floor(scaled_height + 0.5)
    if output_width < 1 or output_height < 1:
        raise ValueError(
            _actionable_error(
                why="factor-derived resize dimensions must both be at least 1",
                what=f"factor={factor!r} produces width={output_width}, height={output_height}",
                how="increase factor or pass explicit positive width and height",
            )
        )
    return output_width, output_height


def _resolve_interpolation(frame: Frame, *, width: int, height: int, interpolation: str | None) -> str:
    if interpolation is None:
        return "area" if width < frame.width or height < frame.height else "lanczos4"
    return _normalized_closed_token(interpolation, axis="interpolation", accepted=_INTERPOLATION_TOKENS)


def resize(
    frame: Frame,
    *,
    width: int | None = None,
    height: int | None = None,
    factor: float | None = None,
    interpolation: Interpolation | None = None,
) -> Frame:
    """Resize a Frame geometrically without changing its metadata or colorimetry.

    Pass width and height together, or pass factor alone; the modes are mutually
    exclusive. Factor dimensions use half-up rounding as
    ``floor(dim * factor + 0.5)``. When interpolation is omitted, any shrinking
    axis selects ``area`` and an all-nonshrinking resize selects ``lanczos4``.

    Every point-sampled kernel uses pixel-center coordinates
    ``src = (dst + 0.5) * (input / output) - 0.5`` and replicate edge handling.
    Input Frame data must be float32; use ``px.values.cast_dtype`` or another
    explicit public value conversion before resizing other storage dtypes. Resize
    calculates float32 output independently per channel and does not clamp scene
    values or cubic/Lanczos overshoot. The result is always a new Frame and a new
    data allocation, including same-size calls.
    """
    frame = _validate_float32_frame(frame, operation="transform.resize")

    output_width, output_height = _resolve_output_size(
        frame,
        width=width,
        height=height,
        factor=factor,
    )
    resolved_interpolation = _resolve_interpolation(
        frame,
        width=output_width,
        height=output_height,
        interpolation=interpolation,
    )
    channel_count = len(frame.channels)
    source = frame.data
    output = cp.empty((output_height, output_width, channel_count), dtype=cp.float32)
    interpolation_index = _INTERPOLATION_TOKENS.index(resolved_interpolation)

    if resolved_interpolation == "nearest":
        _launch_raw_kernel(
            _nearest_kernel(),
            output_width,
            output_height,
            (
                source,
                output,
                np.int64(frame.width),
                np.int64(frame.height),
                np.int64(output_width),
                np.int64(output_height),
                np.int64(channel_count),
            ),
        )
    elif resolved_interpolation == "bilinear":
        horizontal_indices, horizontal_weights = _point_axis_plan(
            frame.width,
            output_width,
            interpolation_index,
        )
        vertical_indices, vertical_weights = _point_axis_plan(
            frame.height,
            output_height,
            interpolation_index,
        )
        _launch_raw_kernel(
            _bilinear_kernel(),
            output_width,
            output_height,
            (
                source,
                horizontal_indices,
                horizontal_weights,
                vertical_indices,
                vertical_weights,
                output,
                np.int64(frame.width),
                np.int64(output_width),
                np.int64(output_height),
                np.int64(channel_count),
            ),
        )
    else:
        intermediate = cp.empty((frame.height, output_width, channel_count), dtype=cp.float32)
        if resolved_interpolation == "area":
            _launch_raw_kernel(
                _area_horizontal_kernel(),
                output_width,
                frame.height,
                (
                    source,
                    intermediate,
                    np.int64(frame.width),
                    np.int64(frame.height),
                    np.int64(output_width),
                    np.int64(channel_count),
                ),
            )
            _launch_raw_kernel(
                _area_vertical_kernel(),
                output_width,
                output_height,
                (
                    intermediate,
                    output,
                    np.int64(output_width),
                    np.int64(frame.height),
                    np.int64(output_height),
                    np.int64(channel_count),
                ),
            )
        else:
            sample_count = _point_sample_count(interpolation_index)
            horizontal_indices, horizontal_weights = _point_axis_plan(
                frame.width,
                output_width,
                interpolation_index,
            )
            vertical_indices, vertical_weights = _point_axis_plan(
                frame.height,
                output_height,
                interpolation_index,
            )
            _launch_raw_kernel(
                _point_horizontal_kernel(),
                output_width,
                frame.height,
                (
                    source,
                    horizontal_indices,
                    horizontal_weights,
                    intermediate,
                    np.int64(frame.width),
                    np.int64(frame.height),
                    np.int64(output_width),
                    np.int64(channel_count),
                    np.int32(sample_count),
                ),
            )
            _launch_raw_kernel(
                _point_vertical_kernel(),
                output_width,
                output_height,
                (
                    intermediate,
                    vertical_indices,
                    vertical_weights,
                    output,
                    np.int64(output_width),
                    np.int64(frame.height),
                    np.int64(output_height),
                    np.int64(channel_count),
                    np.int32(sample_count),
                ),
            )
    return Frame(
        data=output,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=frame.matrix,
    )
