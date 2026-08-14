"""Shared GPU blur validation, border, and separable-kernel infrastructure."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.vocabulary import _BORDER_TOKENS as _BORDER_TOKENS

_RAW_KERNEL_BLOCK = (32, 8)


_RAW_KERNEL_SHARED_LIMIT = 48 * 1024

type _SeparableAxisArgumentBuilder = Callable[
    [tuple[np.int64, np.int64, np.int64], np.int64, np.int32, np.float32], tuple[object, ...]
]


_SEPARABLE_KERNEL_SOURCE = (
    _BORDER_PREAMBLE
    + r"""
extern "C" __global__ void pixtreme_box_horizontal(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const float scale,
    const int border,
    const float border_value
) {
    extern __shared__ float tile[];
    const long long row_elements = width * channel_count;
    const long long halo = radius * channel_count;
    const long long tile_width = blockDim.x + 2 * halo;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;
    const long long tile_count = tile_width * blockDim.y;
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;

    for (long long tile_index = thread_index; tile_index < tile_count; tile_index += thread_count) {
        const long long local_y = tile_index / tile_width;
        const long long local_xc = tile_index - local_y * tile_width;
        const long long source_xc = block_x + local_xc - halo;
        const long long channel = pixtreme_positive_modulo(source_xc, channel_count);
        const long long source_x = (source_xc - channel) / channel_count;
        tile[tile_index] = pixtreme_border_sample(
            source,
            source_x,
            block_y + local_y,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }
    __syncthreads();

    const long long output_xc = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    float sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        sum += tile[threadIdx.y * tile_width + threadIdx.x + halo + offset * channel_count];
    }
    output[output_y * row_elements + output_xc] = sum * scale;
}

extern "C" __global__ void pixtreme_box_vertical(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const float scale,
    const int border,
    const float border_value
) {
    extern __shared__ float tile[];
    const long long row_elements = width * channel_count;
    const long long tile_height = blockDim.y + 2 * radius;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;
    const long long tile_count = blockDim.x * tile_height;
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;

    for (long long tile_index = thread_index; tile_index < tile_count; tile_index += thread_count) {
        const long long local_y = tile_index / blockDim.x;
        const long long local_xc = tile_index - local_y * blockDim.x;
        const long long source_xc = block_x + local_xc;
        const long long channel = source_xc % channel_count;
        const long long source_x = source_xc / channel_count;
        tile[tile_index] = pixtreme_border_sample(
            source,
            source_x,
            block_y + local_y - radius,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }
    __syncthreads();

    const long long output_xc = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    float sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        sum += tile[(threadIdx.y + radius + offset) * blockDim.x + threadIdx.x];
    }
    output[output_y * row_elements + output_xc] = sum * scale;
}

extern "C" __global__ void pixtreme_gaussian_horizontal(
    const float* __restrict__ source,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const int border,
    const float border_value
) {
    extern __shared__ float tile[];
    const long long row_elements = width * channel_count;
    const long long halo = radius * channel_count;
    const long long tile_width = blockDim.x + 2 * halo;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;
    const long long tile_count = tile_width * blockDim.y;
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;

    for (long long tile_index = thread_index; tile_index < tile_count; tile_index += thread_count) {
        const long long local_y = tile_index / tile_width;
        const long long local_xc = tile_index - local_y * tile_width;
        const long long source_xc = block_x + local_xc - halo;
        const long long channel = pixtreme_positive_modulo(source_xc, channel_count);
        const long long source_x = (source_xc - channel) / channel_count;
        tile[tile_index] = pixtreme_border_sample(
            source,
            source_x,
            block_y + local_y,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }
    __syncthreads();

    const long long output_xc = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    float weighted_sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        weighted_sum +=
            tile[threadIdx.y * tile_width + threadIdx.x + halo + offset * channel_count] *
            weights[offset + radius];
    }
    output[output_y * row_elements + output_xc] = weighted_sum;
}

extern "C" __global__ void pixtreme_gaussian_vertical(
    const float* __restrict__ source,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const int border,
    const float border_value
) {
    extern __shared__ float tile[];
    const long long row_elements = width * channel_count;
    const long long tile_height = blockDim.y + 2 * radius;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;
    const long long tile_count = blockDim.x * tile_height;
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;

    for (long long tile_index = thread_index; tile_index < tile_count; tile_index += thread_count) {
        const long long local_y = tile_index / blockDim.x;
        const long long local_xc = tile_index - local_y * blockDim.x;
        const long long source_xc = block_x + local_xc;
        const long long channel = source_xc % channel_count;
        const long long source_x = source_xc / channel_count;
        tile[tile_index] = pixtreme_border_sample(
            source,
            source_x,
            block_y + local_y - radius,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }
    __syncthreads();

    const long long output_xc = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    float weighted_sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        weighted_sum += tile[(threadIdx.y + radius + offset) * blockDim.x + threadIdx.x] *
                        weights[offset + radius];
    }
    output[output_y * row_elements + output_xc] = weighted_sum;
}

extern "C" __global__ void pixtreme_box_horizontal_global(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const float scale,
    const int border,
    const float border_value
) {
    const long long output_xc = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    const long long row_elements = width * channel_count;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    const long long channel = output_xc % channel_count;
    const long long output_x = output_xc / channel_count;
    float sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        sum += pixtreme_border_sample(
            source,
            output_x + offset,
            output_y,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }
    output[output_y * row_elements + output_xc] = sum * scale;
}

extern "C" __global__ void pixtreme_box_vertical_global(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const float scale,
    const int border,
    const float border_value
) {
    const long long output_xc = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    const long long row_elements = width * channel_count;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    const long long channel = output_xc % channel_count;
    const long long output_x = output_xc / channel_count;
    float sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        sum += pixtreme_border_sample(
            source,
            output_x,
            output_y + offset,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }
    output[output_y * row_elements + output_xc] = sum * scale;
}

extern "C" __global__ void pixtreme_gaussian_horizontal_global(
    const float* __restrict__ source,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const int border,
    const float border_value
) {
    const long long output_xc = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    const long long row_elements = width * channel_count;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    const long long channel = output_xc % channel_count;
    const long long output_x = output_xc / channel_count;
    float weighted_sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        weighted_sum += pixtreme_border_sample(
            source,
            output_x + offset,
            output_y,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        ) * weights[offset + radius];
    }
    output[output_y * row_elements + output_xc] = weighted_sum;
}

extern "C" __global__ void pixtreme_gaussian_vertical_global(
    const float* __restrict__ source,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const int border,
    const float border_value
) {
    const long long output_xc = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    const long long row_elements = width * channel_count;
    if (output_xc >= row_elements || output_y >= height) {
        return;
    }
    const long long channel = output_xc % channel_count;
    const long long output_x = output_xc / channel_count;
    float weighted_sum = 0.0f;
    for (long long offset = -radius; offset <= radius; ++offset) {
        weighted_sum += pixtreme_border_sample(
            source,
            output_x,
            output_y + offset,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        ) * weights[offset + radius];
    }
    output[output_y * row_elements + output_xc] = weighted_sum;
}
"""
)


def _validate_sigma(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be a positive real number",
                what=f"received {name}={value!r}",
                how=f"pass a finite int or float {name} greater than 0",
            )
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(
            _actionable_error(
                why=f"{name} must be finite and greater than 0",
                what=f"received {name}={value!r}",
                how=f"pass a finite positive real number for {name}",
            )
        )
    return resolved


def _validate_amount(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why="amount must be a finite real number",
                what=f"received amount={value!r}",
                how="pass a finite int or float amount; negative values are allowed",
            )
        )
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(
            _actionable_error(
                why="amount must be finite",
                what=f"received amount={value!r}",
                how="pass a finite int or float amount; negative values are allowed",
            )
        )
    return resolved


def _validate_odd_size(value: object, *, operation: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or value % 2 == 0:
        raise ValueError(
            _actionable_error(
                why=f"{operation} size must be a positive odd built-in int",
                what=f"received size={value!r}",
                how="pass an odd int of at least 1",
            )
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            _actionable_error(
                why=f"{operation} uses a bounded per-size GPU median kernel",
                what=f"received size={value}, maximum supported size={maximum}",
                how=f"pass an odd int from 1 through {maximum}",
            )
        )
    return value


def _shape_arguments(frame: Frame) -> tuple[np.int64, np.int64, np.int64]:
    return np.int64(frame.width), np.int64(frame.height), np.int64(len(frame.channels))


def _launch_separable_axis(
    kernel: cp.RawKernel,
    global_kernel: cp.RawKernel,
    *,
    frame: Frame,
    radius: int,
    border: str,
    border_value: float,
    horizontal: bool,
    argument_builder: _SeparableAxisArgumentBuilder,
) -> None:
    block_x, block_y = _RAW_KERNEL_BLOCK
    row_elements = frame.width * len(frame.channels)
    grid = ((row_elements + block_x - 1) // block_x, (frame.height + block_y - 1) // block_y)
    shared_elements = (
        (block_x + 2 * radius * len(frame.channels)) * block_y if horizontal else block_x * (block_y + 2 * radius)
    )
    shared_bytes = shared_elements * np.dtype(np.float32).itemsize
    selected_kernel = kernel if shared_bytes <= _RAW_KERNEL_SHARED_LIMIT else global_kernel
    kernel_arguments = argument_builder(
        _shape_arguments(frame),
        np.int64(radius),
        _border_argument(border),
        np.float32(border_value),
    )
    selected_kernel(
        grid,
        _RAW_KERNEL_BLOCK,
        kernel_arguments,
        shared_mem=shared_bytes if selected_kernel is kernel else 0,
    )


def _launch_box_axis(
    kernel: cp.RawKernel,
    global_kernel: cp.RawKernel,
    source: cp.ndarray,
    output: cp.ndarray,
    *,
    frame: Frame,
    radius: int,
    scale: float,
    border: str,
    border_value: float,
    horizontal: bool,
) -> None:
    def build_arguments(
        shape_arguments: tuple[np.int64, np.int64, np.int64],
        radius_argument: np.int64,
        border_argument: np.int32,
        border_value_argument: np.float32,
    ) -> tuple[object, ...]:
        return (
            source,
            output,
            *shape_arguments,
            radius_argument,
            np.float32(scale),
            border_argument,
            border_value_argument,
        )

    _launch_separable_axis(
        kernel,
        global_kernel,
        frame=frame,
        radius=radius,
        border=border,
        border_value=border_value,
        horizontal=horizontal,
        argument_builder=build_arguments,
    )


def _launch_gaussian_axis(
    kernel: cp.RawKernel,
    global_kernel: cp.RawKernel,
    source: cp.ndarray,
    weights: cp.ndarray,
    output: cp.ndarray,
    *,
    frame: Frame,
    radius: int,
    border: str,
    border_value: float,
    horizontal: bool,
) -> None:
    def build_arguments(
        shape_arguments: tuple[np.int64, np.int64, np.int64],
        radius_argument: np.int64,
        border_argument: np.int32,
        border_value_argument: np.float32,
    ) -> tuple[object, ...]:
        return (
            source,
            weights,
            output,
            *shape_arguments,
            radius_argument,
            border_argument,
            border_value_argument,
        )

    _launch_separable_axis(
        kernel,
        global_kernel,
        frame=frame,
        radius=radius,
        border=border,
        border_value=border_value,
        horizontal=horizontal,
        argument_builder=build_arguments,
    )
