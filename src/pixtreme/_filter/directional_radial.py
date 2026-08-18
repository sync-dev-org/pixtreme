"""GPU path-integral blurs with bicubic sampling and explicit border contracts."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame

_PATH_DIRECTIONAL = 0
_PATH_ZOOM = 1
_PATH_SPIN = 2
_RAW_KERNEL_BLOCK = (32, 8)
_SPIN_KERNEL_BLOCK = (16, 16)
_RAW_KERNEL_SHARED_LIMIT = 48 * 1024

_PATH_BLUR_PREAMBLE = (
    _BORDER_PREAMBLE
    + r"""
__device__ __forceinline__ void pixtreme_path_keys_weights(
    const float coordinate,
    long long* base,
    float weights[4]
) {
    const float floored = floorf(coordinate);
    const float fraction = coordinate - floored;
    const float square = fraction * fraction;
    const float cube = square * fraction;
    *base = (long long)floored;
    weights[0] = -0.5f * fraction + square - 0.5f * cube;
    weights[1] = 1.0f - 2.5f * square + 1.5f * cube;
    weights[2] = 0.5f * fraction + 2.0f * square - 1.5f * cube;
    weights[3] = -0.5f * square + 0.5f * cube;
}

__device__ __forceinline__ float pixtreme_path_sample_channel(
    const float* __restrict__ source,
    const long long x,
    const long long y,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long channel,
    const int border,
    const float border_value
) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return source[(y * width + x) * channel_count + channel];
    }
    if (border == 3) {
        return border_value;
    }
    const long long source_x = pixtreme_border_index(x, width, border);
    const long long source_y = pixtreme_border_index(y, height, border);
    return source[(source_y * width + source_x) * channel_count + channel];
}

__device__ __forceinline__ float3 pixtreme_path_sample_rgb(
    const float* __restrict__ source,
    const long long x,
    const long long y,
    const long long width,
    const long long height,
    const int border,
    const float border_value
) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        const long long index = (y * width + x) * 3;
        return make_float3(source[index], source[index + 1], source[index + 2]);
    }
    if (border == 3) {
        return make_float3(border_value, border_value, border_value);
    }
    const long long source_x = pixtreme_border_index(x, width, border);
    const long long source_y = pixtreme_border_index(y, height, border);
    const long long index = (source_y * width + source_x) * 3;
    return make_float3(source[index], source[index + 1], source[index + 2]);
}

struct pixtreme_path_rgb_tile_geometry {
    long long origin_x;
    long long origin_y;
    long long width;
    long long height;
};

__device__ __forceinline__ pixtreme_path_rgb_tile_geometry pixtreme_path_load_rgb_tile(
    const float* __restrict__ source,
    float* __restrict__ tile,
    const long long width,
    const long long height,
    const long long block_x,
    const long long block_y,
    const long long halo_x,
    const long long halo_y,
    const int border,
    const float border_value
) {
    pixtreme_path_rgb_tile_geometry geometry;
    geometry.origin_x = block_x - halo_x;
    geometry.origin_y = block_y - halo_y;
    geometry.width = blockDim.x + 2 * halo_x;
    geometry.height = blockDim.y + 2 * halo_y;
    const long long tile_elements = geometry.width * geometry.height * 3;
    const long long thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const long long thread_count = blockDim.x * blockDim.y;

    for (long long index = thread_index; index < tile_elements; index += thread_count) {
        const long long channel = index % 3;
        const long long tile_pixel = index / 3;
        const long long local_x = tile_pixel % geometry.width;
        const long long local_y = tile_pixel / geometry.width;
        tile[index] = pixtreme_path_sample_channel(
            source,
            geometry.origin_x + local_x,
            geometry.origin_y + local_y,
            width,
            height,
            3,
            channel,
            border,
            border_value
        );
    }
    return geometry;
}

__device__ __forceinline__ float pixtreme_path_bicubic_channel(
    const float* __restrict__ source,
    const float sample_x,
    const float sample_y,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long channel,
    const int border,
    const float border_value
) {
    long long base_x;
    long long base_y;
    float weight_x[4];
    float weight_y[4];
    pixtreme_path_keys_weights(sample_x, &base_x, weight_x);
    pixtreme_path_keys_weights(sample_y, &base_y, weight_y);
    float interpolated = 0.0f;

    #pragma unroll
    for (int y_offset = 0; y_offset < 4; ++y_offset) {
        const long long tap_y = base_y + y_offset - 1;
        float row = 0.0f;
        #pragma unroll
        for (int x_offset = 0; x_offset < 4; ++x_offset) {
            row += pixtreme_path_sample_channel(
                source,
                base_x + x_offset - 1,
                tap_y,
                width,
                height,
                channel_count,
                channel,
                border,
                border_value
            ) * weight_x[x_offset];
        }
        interpolated += row * weight_y[y_offset];
    }
    return interpolated;
}

__device__ __forceinline__ bool pixtreme_path_rgb_footprint_in_bounds(
    const long long base_x,
    const long long base_y,
    const long long width,
    const long long height
) {
    return base_x >= 1 && base_x + 2 < width && base_y >= 1 && base_y + 2 < height;
}

__device__ __forceinline__ float3 pixtreme_path_bicubic_rgb_precomputed(
    const float* __restrict__ source,
    const long long base_x,
    const long long base_y,
    const float weight_x[4],
    const float weight_y[4],
    const long long width,
    const long long height,
    const int border,
    const float border_value
) {
    float3 interpolated = make_float3(0.0f, 0.0f, 0.0f);

    if (pixtreme_path_rgb_footprint_in_bounds(base_x, base_y, width, height)) {
        #pragma unroll
        for (int y_offset = 0; y_offset < 4; ++y_offset) {
            const long long row_index = ((base_y + y_offset - 1) * width + base_x - 1) * 3;
            float3 row = make_float3(0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int x_offset = 0; x_offset < 4; ++x_offset) {
                const long long index = row_index + x_offset * 3;
                const float weight = weight_x[x_offset];
                row.x += source[index] * weight;
                row.y += source[index + 1] * weight;
                row.z += source[index + 2] * weight;
            }
            const float weight = weight_y[y_offset];
            interpolated.x += row.x * weight;
            interpolated.y += row.y * weight;
            interpolated.z += row.z * weight;
        }
        return interpolated;
    }

    #pragma unroll
    for (int y_offset = 0; y_offset < 4; ++y_offset) {
        const long long tap_y = base_y + y_offset - 1;
        float3 row = make_float3(0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int x_offset = 0; x_offset < 4; ++x_offset) {
            const float3 tap = pixtreme_path_sample_rgb(
                source,
                base_x + x_offset - 1,
                tap_y,
                width,
                height,
                border,
                border_value
            );
            const float weight = weight_x[x_offset];
            row.x += tap.x * weight;
            row.y += tap.y * weight;
            row.z += tap.z * weight;
        }
        const float weight = weight_y[y_offset];
        interpolated.x += row.x * weight;
        interpolated.y += row.y * weight;
        interpolated.z += row.z * weight;
    }
    return interpolated;
}

__device__ __forceinline__ float3 pixtreme_path_bicubic_rgb_border(
    const float* __restrict__ source,
    const float sample_x,
    const float sample_y,
    const long long width,
    const long long height,
    const int border,
    const float border_value
);

__device__ __forceinline__ float3 pixtreme_path_bicubic_rgb(
    const float* __restrict__ source,
    const float sample_x,
    const float sample_y,
    const long long width,
    const long long height,
    const int border,
    const float border_value
) {
    #ifdef PIXTREME_PATH_SHARED
    return pixtreme_path_bicubic_rgb_border(
        source, sample_x, sample_y, width, height, border, border_value
    );
    #else
    long long base_x;
    long long base_y;
    float weight_x[4];
    float weight_y[4];
    pixtreme_path_keys_weights(sample_x, &base_x, weight_x);
    pixtreme_path_keys_weights(sample_y, &base_y, weight_y);
    return pixtreme_path_bicubic_rgb_precomputed(
        source,
        base_x,
        base_y,
        weight_x,
        weight_y,
        width,
        height,
        border,
        border_value
    );
    #endif
}

__device__ __forceinline__ float3 pixtreme_path_bicubic_rgb_border(
    const float* __restrict__ source,
    const float sample_x,
    const float sample_y,
    const long long width,
    const long long height,
    const int border,
    const float border_value
) {
    long long base_x;
    long long base_y;
    float weight_x[4];
    float weight_y[4];
    pixtreme_path_keys_weights(sample_x, &base_x, weight_x);
    pixtreme_path_keys_weights(sample_y, &base_y, weight_y);
    float3 interpolated = make_float3(0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int y_offset = 0; y_offset < 4; ++y_offset) {
        const long long tap_y = base_y + y_offset - 1;
        float3 row = make_float3(0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int x_offset = 0; x_offset < 4; ++x_offset) {
            const float3 tap = pixtreme_path_sample_rgb(
                source,
                base_x + x_offset - 1,
                tap_y,
                width,
                height,
                border,
                border_value
            );
            const float weight = weight_x[x_offset];
            row.x += tap.x * weight;
            row.y += tap.y * weight;
            row.z += tap.z * weight;
        }
        const float weight = weight_y[y_offset];
        interpolated.x += row.x * weight;
        interpolated.y += row.y * weight;
        interpolated.z += row.z * weight;
    }
    return interpolated;
}

__device__ __forceinline__ float3 pixtreme_path_bicubic_rgb_tile(
    const float* __restrict__ source,
    const float* __restrict__ tile,
    const float sample_x,
    const float sample_y,
    const long long width,
    const long long height,
    const long long tile_origin_x,
    const long long tile_origin_y,
    const long long tile_width,
    const long long tile_height,
    const int border,
    const float border_value
) {
    long long base_x;
    long long base_y;
    float weight_x[4];
    float weight_y[4];
    pixtreme_path_keys_weights(sample_x, &base_x, weight_x);
    pixtreme_path_keys_weights(sample_y, &base_y, weight_y);
    const long long local_x = base_x - 1 - tile_origin_x;
    const long long local_y = base_y - 1 - tile_origin_y;
    if (local_x < 0 || local_x + 3 >= tile_width || local_y < 0 || local_y + 3 >= tile_height) {
        return pixtreme_path_bicubic_rgb_border(source, sample_x, sample_y, width, height, border, border_value);
    }
    float3 interpolated = make_float3(0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int y_offset = 0; y_offset < 4; ++y_offset) {
        float3 row = make_float3(0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int x_offset = 0; x_offset < 4; ++x_offset) {
            const long long index =
                ((local_y + y_offset) * tile_width + local_x + x_offset) * 3;
            const float weight = weight_x[x_offset];
            row.x += tile[index] * weight;
            row.y += tile[index + 1] * weight;
            row.z += tile[index + 2] * weight;
        }
        const float weight = weight_y[y_offset];
        interpolated.x += row.x * weight;
        interpolated.y += row.y * weight;
        interpolated.z += row.z * weight;
    }
    return interpolated;
}

"""
)

_PATH_BLUR_KERNEL_SOURCE = (
    _PATH_BLUR_PREAMBLE
    + r"""
__device__ __forceinline__ long long pixtreme_path_sample_count(
    const int path_kind,
    const float extent,
    const float offset_x,
    const float offset_y
) {
    const float path_length = path_kind == 0 ? extent : hypotf(offset_x, offset_y) * extent;
    const long long count = (long long)ceilf(path_length) + 1;
    return count < 2 ? 2 : count;
}

__device__ __forceinline__ void pixtreme_path_linear_geometry(
    const int path_kind,
    const float extent,
    const float direction_x,
    const float direction_y,
    const float pixel_x,
    const float pixel_y,
    const float offset_x,
    const float offset_y,
    const long long sample_count,
    float* sample_x,
    float* sample_y,
    float* step_x,
    float* step_y
) {
    const float inverse_intervals = 1.0f / (float)(sample_count - 1);
    if (path_kind == 0) {
        *sample_x = pixel_x - 0.5f * extent * direction_x;
        *sample_y = pixel_y - 0.5f * extent * direction_y;
        *step_x = extent * direction_x * inverse_intervals;
        *step_y = extent * direction_y * inverse_intervals;
        return;
    }
    *sample_x = pixel_x - 0.5f * extent * offset_x;
    *sample_y = pixel_y - 0.5f * extent * offset_y;
    *step_x = extent * offset_x * inverse_intervals;
    *step_y = extent * offset_y * inverse_intervals;
}

template <bool use_tile>
__device__ __forceinline__ void pixtreme_path_blur_rgb_pixel(
    const float* __restrict__ source,
    const float* __restrict__ tile,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long output_x,
    const long long output_y,
    const long long tile_origin_x,
    const long long tile_origin_y,
    const long long tile_width,
    const long long tile_height,
    const int path_kind,
    const float extent,
    const float direction_x,
    const float direction_y,
    const float center_x,
    const float center_y,
    const int border,
    const float border_value
) {
    const float pixel_x = (float)output_x;
    const float pixel_y = (float)output_y;
    const float offset_x = pixel_x - center_x;
    const float offset_y = pixel_y - center_y;
    const long long sample_count = pixtreme_path_sample_count(path_kind, extent, offset_x, offset_y);
    float3 total = make_float3(0.0f, 0.0f, 0.0f);

    if (path_kind != 2) {
        float sample_x;
        float sample_y;
        float step_x;
        float step_y;
        pixtreme_path_linear_geometry(
            path_kind,
            extent,
            direction_x,
            direction_y,
            pixel_x,
            pixel_y,
            offset_x,
            offset_y,
            sample_count,
            &sample_x,
            &sample_y,
            &step_x,
            &step_y
        );
        for (long long sample = 0; sample < sample_count; ++sample) {
            float3 interpolated;
            if constexpr (use_tile) {
                interpolated = pixtreme_path_bicubic_rgb_tile(
                    source,
                    tile,
                    sample_x,
                    sample_y,
                    width,
                    height,
                    tile_origin_x,
                    tile_origin_y,
                    tile_width,
                    tile_height,
                    border,
                    border_value
                );
            } else {
                interpolated = pixtreme_path_bicubic_rgb(
                    source, sample_x, sample_y, width, height, border, border_value
                );
            }
            total.x += interpolated.x;
            total.y += interpolated.y;
            total.z += interpolated.z;
            sample_x += step_x;
            sample_y += step_y;
        }
    } else {
        const float angle_step = extent / (float)(sample_count - 1);
        float step_sine;
        float step_cosine;
        if (fabsf(angle_step) <= 0.25f) {
            const float square = angle_step * angle_step;
            const float fourth = square * square;
            step_sine = angle_step * (1.0f - square / 6.0f + fourth / 120.0f);
            step_cosine = 1.0f - square / 2.0f + fourth / 24.0f - fourth * square / 720.0f;
        } else {
            sincosf(angle_step, &step_sine, &step_cosine);
        }
        float sine = direction_y;
        float cosine = direction_x;
        for (long long sample = 0; sample < sample_count; ++sample) {
            const float sample_x = center_x + offset_x * cosine + offset_y * sine;
            const float sample_y = center_y - offset_x * sine + offset_y * cosine;
            float3 interpolated;
            if constexpr (use_tile) {
                interpolated = pixtreme_path_bicubic_rgb_tile(
                    source,
                    tile,
                    sample_x,
                    sample_y,
                    width,
                    height,
                    tile_origin_x,
                    tile_origin_y,
                    tile_width,
                    tile_height,
                    border,
                    border_value
                );
            } else {
                interpolated = pixtreme_path_bicubic_rgb(
                    source, sample_x, sample_y, width, height, border, border_value
                );
            }
            total.x += interpolated.x;
            total.y += interpolated.y;
            total.z += interpolated.z;
            const long long next_sample = sample + 1;
            const float next_cosine = cosine * step_cosine - sine * step_sine;
            sine = sine * step_cosine + cosine * step_sine;
            cosine = next_cosine;
            if ((next_sample & 31) == 0 && next_sample < sample_count) {
                const float inverse_length = rsqrtf(cosine * cosine + sine * sine);
                cosine *= inverse_length;
                sine *= inverse_length;
            }
        }
    }
    const float scale = 1.0f / (float)sample_count;
    const long long output_index = (output_y * width + output_x) * 3;
    output[output_index] = total.x * scale;
    output[output_index + 1] = total.y * scale;
    output[output_index + 2] = total.z * scale;
}

extern "C" __global__ void pixtreme_path_blur_rgb(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const int path_kind,
    const float extent,
    const float direction_x,
    const float direction_y,
    const float center_x,
    const float center_y,
    const int border,
    const float border_value
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= width || output_y >= height) {
        return;
    }
    pixtreme_path_blur_rgb_pixel<false>(
        source,
        nullptr,
        output,
        width,
        height,
        output_x,
        output_y,
        0,
        0,
        0,
        0,
        path_kind,
        extent,
        direction_x,
        direction_y,
        center_x,
        center_y,
        border,
        border_value
    );
}

extern "C" __global__ void pixtreme_path_blur_rgb_shared(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long halo_x,
    const long long halo_y,
    const int path_kind,
    const float extent,
    const float direction_x,
    const float direction_y,
    const float center_x,
    const float center_y,
    const int border,
    const float border_value
) {
    extern __shared__ float tile[];
    const long long block_x = (long long)blockIdx.x * blockDim.x;
    const long long block_y = (long long)blockIdx.y * blockDim.y;
    const pixtreme_path_rgb_tile_geometry tile_geometry = pixtreme_path_load_rgb_tile(
        source,
        tile,
        width,
        height,
        block_x,
        block_y,
        halo_x,
        halo_y,
        border,
        border_value
    );
    __syncthreads();

    const long long output_x = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_x >= width || output_y >= height) {
        return;
    }
    pixtreme_path_blur_rgb_pixel<true>(
        source,
        tile,
        output,
        width,
        height,
        output_x,
        output_y,
        tile_geometry.origin_x,
        tile_geometry.origin_y,
        tile_geometry.width,
        tile_geometry.height,
        path_kind,
        extent,
        direction_x,
        direction_y,
        center_x,
        center_y,
        border,
        border_value
    );
}

extern "C" __global__ void pixtreme_path_blur_generic(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const int path_kind,
    const float extent,
    const float direction_x,
    const float direction_y,
    const float center_x,
    const float center_y,
    const int border,
    const float border_value
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= width || output_y >= height) {
        return;
    }
    const float pixel_x = (float)output_x;
    const float pixel_y = (float)output_y;
    const float offset_x = pixel_x - center_x;
    const float offset_y = pixel_y - center_y;
    const long long sample_count = pixtreme_path_sample_count(path_kind, extent, offset_x, offset_y);
    const float scale = 1.0f / (float)sample_count;

    for (long long channel = 0; channel < channel_count; ++channel) {
        float total = 0.0f;
        if (path_kind != 2) {
            float sample_x;
            float sample_y;
            float step_x;
            float step_y;
            pixtreme_path_linear_geometry(
                path_kind,
                extent,
                direction_x,
                direction_y,
                pixel_x,
                pixel_y,
                offset_x,
                offset_y,
                sample_count,
                &sample_x,
                &sample_y,
                &step_x,
                &step_y
            );
            for (long long sample = 0; sample < sample_count; ++sample) {
                total += pixtreme_path_bicubic_channel(
                    source,
                    sample_x,
                    sample_y,
                    width,
                    height,
                    channel_count,
                    channel,
                    border,
                    border_value
                );
                sample_x += step_x;
                sample_y += step_y;
            }
        } else {
            const float angle_step = extent / (float)(sample_count - 1);
            float step_sine;
            float step_cosine;
            if (fabsf(angle_step) <= 0.25f) {
                const float square = angle_step * angle_step;
                const float fourth = square * square;
                step_sine = angle_step * (1.0f - square / 6.0f + fourth / 120.0f);
                step_cosine = 1.0f - square / 2.0f + fourth / 24.0f - fourth * square / 720.0f;
            } else {
                sincosf(angle_step, &step_sine, &step_cosine);
            }
            float sine = direction_y;
            float cosine = direction_x;
            for (long long sample = 0; sample < sample_count; ++sample) {
                const float sample_x = center_x + offset_x * cosine + offset_y * sine;
                const float sample_y = center_y - offset_x * sine + offset_y * cosine;
                total += pixtreme_path_bicubic_channel(
                    source,
                    sample_x,
                    sample_y,
                    width,
                    height,
                    channel_count,
                    channel,
                    border,
                    border_value
                );
                const long long next_sample = sample + 1;
                const float next_cosine = cosine * step_cosine - sine * step_sine;
                sine = sine * step_cosine + cosine * step_sine;
                cosine = next_cosine;
                if ((next_sample & 31) == 0 && next_sample < sample_count) {
                    const float inverse_length = rsqrtf(cosine * cosine + sine * sine);
                    cosine *= inverse_length;
                    sine *= inverse_length;
                }
            }
        }
        output[(output_y * width + output_x) * channel_count + channel] = total * scale;
    }
}
"""
)


@lru_cache(maxsize=1)
def _path_blur_rgb_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PATH_BLUR_KERNEL_SOURCE, "pixtreme_path_blur_rgb")


@lru_cache(maxsize=1)
def _path_blur_rgb_shared_kernel() -> cp.RawKernel:
    return cp.RawKernel(
        _PATH_BLUR_KERNEL_SOURCE,
        "pixtreme_path_blur_rgb_shared",
        options=("-DPIXTREME_PATH_SHARED",),
    )


@lru_cache(maxsize=1)
def _path_blur_generic_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PATH_BLUR_KERNEL_SOURCE, "pixtreme_path_blur_generic")


def _validate_real(value: object, *, name: str, positive: bool) -> float:
    requirement = "a positive real number" if positive else "a real number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be {requirement}",
                what=f"received {name}={value!r}",
                how=f"pass a finite int or float for {name}" + (" greater than 0" if positive else ""),
            )
        )
    resolved = float(value)
    if not math.isfinite(resolved) or (positive and resolved <= 0.0):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be finite" + (" and greater than 0" if positive else ""),
                what=f"received {name}={value!r}",
                how=f"pass a finite real number for {name}" + (" greater than 0" if positive else ""),
            )
        )
    return resolved


def _resolve_center(frame: Frame, center: object, *, operation: str) -> tuple[float, float]:
    if center is None:
        return (frame.width - 1) / 2.0, (frame.height - 1) / 2.0
    if type(center) is not tuple or len(center) != 2:
        raise ValueError(
            _actionable_error(
                why=f"{operation} center must be None or a two-element real tuple",
                what=f"received center={center!r}",
                how="pass center=None or pass a finite (x, y) tuple; coordinates may lie outside the image",
            )
        )
    try:
        center_x = _validate_real(center[0], name="center x", positive=False)
        center_y = _validate_real(center[1], name="center y", positive=False)
    except ValueError as error:
        raise ValueError(
            _actionable_error(
                why=f"{operation} center coordinates must both be finite real numbers",
                what=f"received center={center!r}",
                how="pass center=None or pass a finite (x, y) tuple; coordinates may lie outside the image",
            )
        ) from error
    return center_x, center_y


def _shared_path_halo(
    frame: Frame,
    *,
    path_kind: int,
    extent: float,
    direction_x: float,
    direction_y: float,
    center: tuple[float, float],
) -> tuple[int, int] | None:
    if path_kind == _PATH_DIRECTIONAL:
        displacement_x = 0.5 * extent * abs(direction_x)
        displacement_y = 0.5 * extent * abs(direction_y)
    else:
        maximum_x = max(abs(center[0]), abs((frame.width - 1) - center[0]))
        maximum_y = max(abs(center[1]), abs((frame.height - 1) - center[1]))
        if path_kind == _PATH_ZOOM:
            displacement_x = 0.5 * extent * maximum_x
            displacement_y = 0.5 * extent * maximum_y
        else:
            half_extent = 0.5 * extent
            sine_bound = 1.0 if half_extent >= math.pi / 2.0 else math.sin(half_extent)
            cosine_delta_bound = 2.0 if half_extent >= math.pi else 1.0 - math.cos(half_extent)
            displacement_x = maximum_x * cosine_delta_bound + maximum_y * sine_bound
            displacement_y = maximum_x * sine_bound + maximum_y * cosine_delta_bound
    if not math.isfinite(displacement_x) or not math.isfinite(displacement_y):
        return None
    return math.ceil(displacement_x) + 2, math.ceil(displacement_y) + 2


def _path_blur(
    frame: Frame,
    *,
    path_kind: int,
    extent: float,
    direction_radians: float,
    center: tuple[float, float],
    border: str,
    border_value: float,
) -> Frame:
    output = cp.empty(frame.shape, dtype=cp.float32)
    kernel_block = _SPIN_KERNEL_BLOCK if path_kind == _PATH_SPIN else _RAW_KERNEL_BLOCK
    block_x, block_y = kernel_block
    grid = ((frame.width + block_x - 1) // block_x, (frame.height + block_y - 1) // block_y)
    if path_kind == _PATH_DIRECTIONAL:
        direction_x = math.cos(direction_radians)
        direction_y = -math.sin(direction_radians)
    elif path_kind == _PATH_SPIN:
        direction_x = math.cos(-0.5 * extent)
        direction_y = math.sin(-0.5 * extent)
    else:
        direction_x = 0.0
        direction_y = 0.0
    common_arguments = (
        frame.data,
        output,
        np.int64(frame.width),
        np.int64(frame.height),
    )
    path_arguments = (
        np.int32(path_kind),
        np.float32(extent),
        np.float32(direction_x),
        np.float32(direction_y),
        np.float32(center[0]),
        np.float32(center[1]),
        _border_argument(border),
        np.float32(border_value),
    )
    if len(frame.channels) == 3:
        halo_x = 0
        halo_y = 0
        halo = _shared_path_halo(
            frame,
            path_kind=path_kind,
            extent=extent,
            direction_x=direction_x,
            direction_y=direction_y,
            center=center,
        )
        if halo is not None:
            halo_x, halo_y = halo
            shared_elements = (block_x + 2 * halo_x) * (block_y + 2 * halo_y) * 3
            shared_bytes = shared_elements * np.dtype(np.float32).itemsize
        else:
            shared_bytes = _RAW_KERNEL_SHARED_LIMIT + 1
        if shared_bytes <= _RAW_KERNEL_SHARED_LIMIT:
            _path_blur_rgb_shared_kernel()(
                grid,
                kernel_block,
                common_arguments + (np.int64(halo_x), np.int64(halo_y)) + path_arguments,
                shared_mem=shared_bytes,
            )
        else:
            _path_blur_rgb_kernel()(grid, kernel_block, common_arguments + path_arguments)
    else:
        _path_blur_generic_kernel()(
            grid,
            kernel_block,
            common_arguments + (np.int64(len(frame.channels)),) + path_arguments,
        )
    return _new_frame(frame, output)


def directional_blur(
    frame: Frame,
    *,
    angle: float,
    length: float,
    border: str = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Average a symmetric straight path through each pixel.

    The path is ``p + t * (cos(angle), -sin(angle))`` for t in
    ``[-length / 2, +length / 2]``. Angles use degrees: 0 degrees is +x and
    positive is visually counterclockwise. Sampling uses
    ``max(2, ceil(path length) + 1)`` uniformly weighted points and fixed
    bicubic interpolation with Keys a = -0.5.

    Border defaults to ``mirror``; ``replicate`` and ``wrap`` are also
    accepted and apply independently to every bicubic tap. ``constant`` uses
    ``border_value`` for taps outside the image; ``border_value`` is required
    with ``constant`` and forbidden for every other border. Calculation is fp32
    per channel, does not clamp scene values, and returns new storage.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.directional_blur")
    checked_angle = _validate_real(angle, name="angle", positive=False)
    checked_length = _validate_real(length, name="length", positive=True)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    return _path_blur(
        checked_frame,
        path_kind=_PATH_DIRECTIONAL,
        extent=checked_length,
        direction_radians=math.radians(math.fmod(checked_angle, 180.0)),
        center=(0.0, 0.0),
        border=checked_border,
        border_value=checked_border_value,
    )


def zoom_blur(
    frame: Frame,
    *,
    amount: float,
    center: tuple[float, float] | None = None,
    border: str = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Average a symmetric radial scale path through each pixel.

    The path is ``center + (p - center) * s`` for s from ``1 - amount / 2``
    through ``1 + amount / 2``. Center defaults to the geometric center and
    may lie outside the image. Sampling uses
    ``max(2, ceil(path length) + 1)`` uniformly weighted points with fixed
    bicubic interpolation using Keys a = -0.5. Path length is center distance
    times amount, so cost grows with center distance.

    Border defaults to ``mirror``; ``replicate`` and ``wrap`` are also
    accepted and apply independently to every bicubic tap. ``constant`` uses
    ``border_value`` for taps outside the image; ``border_value`` is required
    with ``constant`` and forbidden for every other border. Calculation is fp32
    per channel, does not clamp scene values, and returns new storage.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.zoom_blur")
    checked_amount = _validate_real(amount, name="amount", positive=True)
    checked_center = _resolve_center(checked_frame, center, operation="filter.zoom_blur")
    checked_border, checked_border_value = _resolve_border(border, border_value)
    return _path_blur(
        checked_frame,
        path_kind=_PATH_ZOOM,
        extent=checked_amount,
        direction_radians=0.0,
        center=checked_center,
        border=checked_border,
        border_value=checked_border_value,
    )


def spin_blur(
    frame: Frame,
    *,
    angle: float,
    center: tuple[float, float] | None = None,
    border: str = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Average a symmetric circular arc around center through each pixel.

    The circular arc rotates p from ``-angle / 2`` through ``+angle / 2``.
    Angles use degrees: 0 degrees is +x and positive is visually
    counterclockwise. Center defaults to the geometric center and may lie
    outside the image. Sampling uses ``max(2, ceil(path length) + 1)``
    uniformly weighted points with fixed bicubic interpolation using
    Keys a = -0.5. Path length is center distance times angle in radians, so
    cost grows with center distance.

    Border defaults to ``mirror``; ``replicate`` and ``wrap`` are also
    accepted and apply independently to every bicubic tap. ``constant`` uses
    ``border_value`` for taps outside the image; ``border_value`` is required
    with ``constant`` and forbidden for every other border. Calculation is fp32
    per channel, does not clamp scene values, and returns new storage.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.spin_blur")
    checked_angle = _validate_real(angle, name="angle", positive=True)
    checked_center = _resolve_center(checked_frame, center, operation="filter.spin_blur")
    checked_border, checked_border_value = _resolve_border(border, border_value)
    return _path_blur(
        checked_frame,
        path_kind=_PATH_SPIN,
        extent=math.radians(checked_angle),
        direction_radians=0.0,
        center=checked_center,
        border=checked_border,
        border_value=checked_border_value,
    )
