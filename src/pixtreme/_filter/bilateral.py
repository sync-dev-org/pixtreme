"""GPU bilateral blur."""

from __future__ import annotations

import math
from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._filter.common import (
    _RAW_KERNEL_SHARED_LIMIT,
    _shape_arguments,
    _validate_sigma,
)

_BILATERAL_KERNEL_BLOCK = (16, 16)


_BILATERAL_WEIGHT_CACHE_SIZE = 32


_BILATERAL_FUSED_MAX_CHANNELS = 4


_BILATERAL_FALLBACK_OPERATION = r"""
const long long channel = i % channel_count;
const long long pixel = i / channel_count;
const long long x = pixel % width;
const long long y = pixel / width;
const long long center_index = (y * width + x) * channel_count;
float weighted_sum = 0.0f;
float weight_sum = 0.0f;
int sample = 0;

for (long long offset_y = -radius; offset_y <= radius; ++offset_y) {
    for (long long offset_x = -radius; offset_x <= radius; ++offset_x) {
        float value_distance_squared = 0.0f;
        for (long long distance_channel = 0; distance_channel < channel_count; ++distance_channel) {
            const float neighbor = pixtreme_border_sample(
                source,
                x + offset_x,
                y + offset_y,
                width,
                height,
                channel_count,
                distance_channel,
                border,
                border_value
            );
            const float difference = neighbor - (float)source[center_index + distance_channel];
            value_distance_squared += difference * difference;
        }
        const float weight = spatial_weights[sample] * expf(value_distance_squared * value_coefficient);
        weighted_sum += pixtreme_border_sample(
            source, x + offset_x, y + offset_y, width, height, channel_count, channel, border, border_value
        ) * weight;
        weight_sum += weight;
        ++sample;
    }
}

filtered = weighted_sum / weight_sum;
"""


@lru_cache(maxsize=1)
def _bilateral_fallback_kernel() -> cp.ElementwiseKernel:
    return cp.ElementwiseKernel(
        "raw T source, raw float32 spatial_weights, int64 width, int64 height, int64 channel_count, int64 radius, "
        "float32 value_coefficient, int32 border, float32 border_value",
        "float32 filtered",
        _BILATERAL_FALLBACK_OPERATION,
        "pixtreme_blur_bilateral_fallback",
        preamble=_BORDER_PREAMBLE,
    )


@lru_cache(maxsize=_BILATERAL_WEIGHT_CACHE_SIZE)
def _bilateral_spatial_weights(device_id: int, sigma_space: float) -> cp.ndarray:
    radius = math.ceil(3.0 * sigma_space)
    with cp.cuda.Device(device_id):
        coordinates = cp.arange(-radius, radius + 1, dtype=cp.float32)
        horizontal, vertical = cp.meshgrid(coordinates, coordinates)
        distances_squared = horizontal * horizontal + vertical * vertical
        weights = cp.exp(distances_squared * np.float32(-0.5 / (sigma_space * sigma_space)))
    return weights.ravel()


def _bilateral_kernel_source(channel_count: int) -> str:
    center_names = [f"center_{channel}" for channel in range(channel_count)]
    weighted_names = [f"weighted_{channel}" for channel in range(channel_count)]
    lines = [
        _BORDER_PREAMBLE,
        f"""
extern "C" __global__ void pixtreme_blur_bilateral_{channel_count}(
    const float* __restrict__ source,
    const float* __restrict__ spatial_weights,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long radius,
    const float value_coefficient,
    const int border,
    const float border_value
) {{
    extern __shared__ float tile[];
    const long long channel_count = {channel_count};
    const long long tile_width = blockDim.x + 2 * radius;
    const long long tile_height = blockDim.y + 2 * radius;
    const long long tile_pixel_count = tile_width * tile_height;
    const long long tile_count = tile_pixel_count * channel_count;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;

    for (long long tile_index = thread_index; tile_index < tile_count; tile_index += thread_count) {{
        const long long channel = tile_index % channel_count;
        const long long tile_pixel = tile_index / channel_count;
        const long long local_x = tile_pixel % tile_width;
        const long long local_y = tile_pixel / tile_width;
        tile[tile_index] = pixtreme_border_sample(
            source,
            block_x + local_x - radius,
            block_y + local_y - radius,
            width,
            height,
            channel_count,
            channel,
            border,
            border_value
        );
    }}
    __syncthreads();

    const long long output_x = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_x >= width || output_y >= height) {{
        return;
    }}
    const long long center_index =
        ((threadIdx.y + radius) * tile_width + threadIdx.x + radius) * channel_count;
""",
    ]
    for channel, name in enumerate(center_names):
        lines.append(f"    const float {name} = tile[center_index + {channel}];\n")
    for name in weighted_names:
        lines.append(f"    float {name} = 0.0f;\n")
    lines.extend(
        (
            "    float weight_sum = 0.0f;\n",
            "    long long sample = 0;\n",
            "    for (long long offset_y = 0; offset_y < 2 * radius + 1; ++offset_y) {\n",
            "        for (long long offset_x = 0; offset_x < 2 * radius + 1; ++offset_x) {\n",
            "            const long long neighbor_index =\n",
            "                ((threadIdx.y + offset_y) * tile_width + threadIdx.x + offset_x) * channel_count;\n",
            "            float value_distance_squared = 0.0f;\n",
        )
    )
    for channel, center_name in enumerate(center_names):
        lines.append(
            f"            const float difference_{channel} = tile[neighbor_index + {channel}] - {center_name};\n"
        )
        lines.append(f"            value_distance_squared += difference_{channel} * difference_{channel};\n")
    lines.append(
        "            const float weight = spatial_weights[sample] * __expf(value_distance_squared * value_coefficient);\n"
    )
    for channel, weighted_name in enumerate(weighted_names):
        lines.append(f"            {weighted_name} += tile[neighbor_index + {channel}] * weight;\n")
    lines.extend(
        (
            "            weight_sum += weight;\n",
            "            ++sample;\n",
            "        }\n",
            "    }\n",
            "    const long long output_index = (output_y * width + output_x) * channel_count;\n",
        )
    )
    for channel, weighted_name in enumerate(weighted_names):
        lines.append(f"    output[output_index + {channel}] = {weighted_name} / weight_sum;\n")
    lines.append("}\n")
    return "".join(lines)


@lru_cache(maxsize=_BILATERAL_FUSED_MAX_CHANNELS)
def _bilateral_kernel(channel_count: int) -> cp.RawKernel:
    return cp.RawKernel(
        _bilateral_kernel_source(channel_count),
        f"pixtreme_blur_bilateral_{channel_count}",
    )


def bilateral_blur(
    frame: Frame,
    *,
    sigma_space: float,
    sigma_value: float,
    border: str = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Apply a bilateral blur using one all-channel value distance per neighbor.

    Kernel radius is fixed as ``radius = ceil(3 * sigma_space)``. Border defaults
    to ``mirror`` (edge-excluding reflection); ``replicate`` clamps to the edge
    and ``wrap`` uses periodic indices. ``sigma_value`` is measured in the
    Frame's working-value scale: changing scene-linear exposure changes its
    meaning, and this function does not normalize values automatically.
    ``constant`` uses ``border_value`` for every virtual pixel outside the image;
    ``border_value`` is required with ``constant`` and forbidden for every other
    border. The fp32 calculation does not clamp negative values or values above
    1 and returns new storage with unchanged metadata.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.bilateral_blur")
    checked_sigma_space = _validate_sigma(sigma_space, name="sigma_space")
    checked_sigma_value = _validate_sigma(sigma_value, name="sigma_value")
    checked_border, checked_border_value = _resolve_border(border, border_value)
    radius = math.ceil(3.0 * checked_sigma_space)
    spatial_weights = _bilateral_spatial_weights(cp.cuda.runtime.getDevice(), checked_sigma_space)
    value_coefficient = np.float32(-0.5 / (checked_sigma_value * checked_sigma_value))
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    channel_count = len(checked_frame.channels)
    block_x, block_y = _BILATERAL_KERNEL_BLOCK
    shared_elements = (block_x + 2 * radius) * (block_y + 2 * radius) * channel_count
    shared_bytes = shared_elements * np.dtype(np.float32).itemsize
    if channel_count <= _BILATERAL_FUSED_MAX_CHANNELS and shared_bytes <= _RAW_KERNEL_SHARED_LIMIT:
        grid = ((checked_frame.width + block_x - 1) // block_x, (checked_frame.height + block_y - 1) // block_y)
        _bilateral_kernel(channel_count)(
            grid,
            _BILATERAL_KERNEL_BLOCK,
            (
                checked_frame.data,
                spatial_weights,
                output,
                np.int64(checked_frame.width),
                np.int64(checked_frame.height),
                np.int64(radius),
                value_coefficient,
                _border_argument(checked_border),
                np.float32(checked_border_value),
            ),
            shared_mem=shared_bytes,
        )
    else:
        _bilateral_fallback_kernel()(
            checked_frame.data,
            spatial_weights,
            *_shape_arguments(checked_frame),
            np.int64(radius),
            value_coefficient,
            _border_argument(checked_border),
            np.float32(checked_border_value),
            output,
        )
    return _new_frame(checked_frame, output)
