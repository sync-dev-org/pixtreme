"""Per-pixel vector-field blur with straight bicubic gather paths."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _border_argument, _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.vocabulary import _VECTOR_BLUR_SHUTTER_TOKENS, Border, VectorBlurShutter
from pixtreme._filter.directional_radial import _PATH_BLUR_PREAMBLE, _RAW_KERNEL_BLOCK

_SHUTTER_TOKENS = _VECTOR_BLUR_SHUTTER_TOKENS
_SHUTTER_STARTS = (-0.5, 0.0, -1.0)
_VECTOR_SHARED_HALO = 16

_VECTOR_BLUR_KERNEL_SOURCE = (
    _PATH_BLUR_PREAMBLE
    + r"""
__device__ __forceinline__ long long pixtreme_vector_sample_count(const float vector_x, const float vector_y) {
    const long long count = (long long)ceilf(hypotf(vector_x, vector_y)) + 1;
    return count < 2 ? 2 : count;
}

__device__ __noinline__ float3 pixtreme_vector_blur_rgb_global_path(
    const float* __restrict__ source,
    const long long width,
    const long long height,
    const long long sample_count,
    float sample_x,
    float sample_y,
    const float step_x,
    const float step_y,
    const int border,
    const float border_value
) {
    float3 total = make_float3(0.0f, 0.0f, 0.0f);
    for (long long sample = 0; sample < sample_count; ++sample) {
        long long base_x;
        long long base_y;
        float weight_x[4];
        float weight_y[4];
        pixtreme_path_keys_weights(sample_x, &base_x, weight_x);
        pixtreme_path_keys_weights(sample_y, &base_y, weight_y);
        const float3 interpolated = pixtreme_path_bicubic_rgb_precomputed(
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
        total.x += interpolated.x;
        total.y += interpolated.y;
        total.z += interpolated.z;
        sample_x += step_x;
        sample_y += step_y;
    }
    return total;
}

extern "C" __global__ void pixtreme_vector_blur_rgb_shared(
    const float* __restrict__ source,
    const float* __restrict__ vector_field,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long halo,
    const float shutter_start,
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
        halo,
        halo,
        border,
        border_value
    );
    __syncthreads();

    const long long output_x = block_x + threadIdx.x;
    const long long output_y = block_y + threadIdx.y;
    if (output_x >= width || output_y >= height) {
        return;
    }
    const long long pixel = output_y * width + output_x;
    const float vector_x = vector_field[pixel * 2];
    const float vector_y = vector_field[pixel * 2 + 1];
    const long long sample_count = pixtreme_vector_sample_count(vector_x, vector_y);
    const float inverse_intervals = 1.0f / (float)(sample_count - 1);
    float sample_x = (float)output_x + shutter_start * vector_x;
    float sample_y = (float)output_y + shutter_start * vector_y;
    const float step_x = vector_x * inverse_intervals;
    const float step_y = vector_y * inverse_intervals;
    float3 total = make_float3(0.0f, 0.0f, 0.0f);

    if (sample_count >= 65) {
        total = pixtreme_vector_blur_rgb_global_path(
            source,
            width,
            height,
            sample_count,
            sample_x,
            sample_y,
            step_x,
            step_y,
            border,
            border_value
        );
    } else {
        for (long long sample = 0; sample < sample_count; ++sample) {
            const float3 interpolated = pixtreme_path_bicubic_rgb_tile(
                source,
                tile,
                sample_x,
                sample_y,
                width,
                height,
                tile_geometry.origin_x,
                tile_geometry.origin_y,
                tile_geometry.width,
                tile_geometry.height,
                border,
                border_value
            );
            total.x += interpolated.x;
            total.y += interpolated.y;
            total.z += interpolated.z;
            sample_x += step_x;
            sample_y += step_y;
        }
    }
    const float scale = 1.0f / (float)sample_count;
    const long long output_index = pixel * 3;
    output[output_index] = total.x * scale;
    output[output_index + 1] = total.y * scale;
    output[output_index + 2] = total.z * scale;
}

extern "C" __global__ void pixtreme_vector_blur_generic(
    const float* __restrict__ source,
    const float* __restrict__ vector_field,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const float shutter_start,
    const int border,
    const float border_value
) {
    const long long output_x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_y = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (output_x >= width || output_y >= height) {
        return;
    }
    const long long pixel = output_y * width + output_x;
    const float vector_x = vector_field[pixel * 2];
    const float vector_y = vector_field[pixel * 2 + 1];
    const long long sample_count = pixtreme_vector_sample_count(vector_x, vector_y);
    const float inverse_intervals = 1.0f / (float)(sample_count - 1);
    const float start_x = (float)output_x + shutter_start * vector_x;
    const float start_y = (float)output_y + shutter_start * vector_y;
    const float step_x = vector_x * inverse_intervals;
    const float step_y = vector_y * inverse_intervals;
    const float scale = 1.0f / (float)sample_count;

    for (long long channel = 0; channel < channel_count; ++channel) {
        float sample_x = start_x;
        float sample_y = start_y;
        float total = 0.0f;
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
        output[pixel * channel_count + channel] = total * scale;
    }
}
"""
)


@lru_cache(maxsize=1)
def _vector_blur_rgb_shared_kernel() -> cp.RawKernel:
    return cp.RawKernel(
        _VECTOR_BLUR_KERNEL_SOURCE,
        "pixtreme_vector_blur_rgb_shared",
        options=("-DPIXTREME_PATH_SHARED",),
    )


@lru_cache(maxsize=1)
def _vector_blur_generic_kernel() -> cp.RawKernel:
    return cp.RawKernel(_VECTOR_BLUR_KERNEL_SOURCE, "pixtreme_vector_blur_generic")


def _validate_vector(vector: object, *, frame: Frame) -> Frame:
    if not isinstance(vector, Frame):
        raise ValueError(
            _actionable_error(
                why="vector_blur requires vector to be a metadata-bearing Frame",
                what=f"received {type(vector).__module__}.{type(vector).__qualname__}",
                how="construct a two-channel vector Frame with px.io.from_array",
            )
        )
    checked_vector = _validate_float32_frame(vector, operation="filter.vector_blur")
    if len(checked_vector.channels) != 2:
        raise ValueError(
            _actionable_error(
                why="vector_blur requires exactly two vector channels in x-y position order",
                what=f"received vector shape {checked_vector.shape!r} with channels={checked_vector.channels!r}",
                how="provide a vector Frame whose data has exactly two channels: channel 0 = x, channel 1 = y",
            )
        )
    if checked_vector.width != frame.width or checked_vector.height != frame.height:
        raise ValueError(
            _actionable_error(
                why="vector_blur requires vector and frame to have matching width and height",
                what=f"received frame shape {frame.shape!r} and vector shape {checked_vector.shape!r}",
                how="provide one two-channel vector value for every frame pixel",
            )
        )
    return checked_vector


def _validate_shutter(shutter: object) -> str:
    return _normalized_closed_token(shutter, axis="shutter", accepted=_SHUTTER_TOKENS)


def vector_blur(
    frame: Frame,
    *,
    vector: Frame,
    shutter: VectorBlurShutter = "centered",
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Average a per-pixel straight line selected by one gather vector.

    For output pixel p, the path is ``p + t * v(p)``: v(p) is read once at p,
    and the vector field is not followed again along the path. This gather
    contract cannot create the scatter-style contribution of a moving object
    across a motion boundary; that visual limitation is intentional.

    Vector channel 0 = x and channel 1 = y by position, independent of labels.
    Coordinates are measured in pixels without y inversion: +x is right; +y is down.
    ``centered`` integrates t in ``[-1/2, +1/2]``, ``forward`` uses
    ``[0, 1]``, and ``backward`` uses ``[-1, 0]``. Every interval has length 1.

    The uniformly weighted endpoint-inclusive sample count is
    ``max(2, ceil(|v(p)|) + 1)``. Sampling uses fixed bicubic interpolation with
    Keys a = -0.5. ``mirror`` is the default border; ``replicate`` and ``wrap``
    use edge clamp and periodic indexing. ``constant`` uses ``border_value`` for
    every out-of-image bicubic tap. ``border_value`` is required with
    ``constant`` and forbidden for every other border.

    Geometry, samples, and accumulation use fp32 independently per channel and
    the result does not clamp negative values or values above 1. Vector values
    are assumed finite and are not checked; output for non-finite vectors is
    undefined. The result owns new storage with frame metadata unchanged, and
    computational cost grows in proportion to |v|.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.vector_blur")
    checked_vector = _validate_vector(vector, frame=checked_frame)
    checked_shutter = _validate_shutter(shutter)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    block_x, block_y = _RAW_KERNEL_BLOCK
    grid = ((checked_frame.width + block_x - 1) // block_x, (checked_frame.height + block_y - 1) // block_y)
    common_arguments = (
        checked_frame.data,
        checked_vector.data,
        output,
        np.int64(checked_frame.width),
        np.int64(checked_frame.height),
    )
    vector_arguments = (
        np.float32(_SHUTTER_STARTS[_SHUTTER_TOKENS.index(checked_shutter)]),
        _border_argument(checked_border),
        np.float32(checked_border_value),
    )
    if len(checked_frame.channels) == 3:
        halo = _VECTOR_SHARED_HALO
        shared_elements = (block_x + 2 * halo) * (block_y + 2 * halo) * 3
        _vector_blur_rgb_shared_kernel()(
            grid,
            _RAW_KERNEL_BLOCK,
            common_arguments + (np.int64(halo),) + vector_arguments,
            shared_mem=shared_elements * np.dtype(np.float32).itemsize,
        )
    else:
        _vector_blur_generic_kernel()(
            grid,
            _RAW_KERNEL_BLOCK,
            common_arguments + (np.int64(len(checked_frame.channels)),) + vector_arguments,
        )
    return _new_frame(checked_frame, output)
