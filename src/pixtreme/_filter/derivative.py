"""GPU derivative filters with explicit direction, scale, and border contracts."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.vocabulary import _SOBEL_DIRECTION_TOKENS, Border, SobelDirection
from pixtreme._filter.common import _validate_sigma
from pixtreme._filter.gaussian import gaussian_blur

_DIRECTION_TOKENS = _SOBEL_DIRECTION_TOKENS
_THREADS_PER_BLOCK = 256

_DERIVATIVE_KERNEL_SOURCE = (
    _BORDER_PREAMBLE
    + r"""
extern "C" __global__ void pixtreme_sobel(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const int direction,
    const int border,
    const float border_value
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long element_count = width * height * channel_count;
    if (index >= element_count) {
        return;
    }

    const long long channel = index % channel_count;
    const long long pixel_index = index / channel_count;
    const long long x = pixel_index % width;
    const long long y = pixel_index / width;

    const float top_left = pixtreme_border_sample(
        source, x - 1, y - 1, width, height, channel_count, channel, border, border_value
    );
    const float top = pixtreme_border_sample(
        source, x, y - 1, width, height, channel_count, channel, border, border_value
    );
    const float top_right = pixtreme_border_sample(
        source, x + 1, y - 1, width, height, channel_count, channel, border, border_value
    );
    const float left = pixtreme_border_sample(
        source, x - 1, y, width, height, channel_count, channel, border, border_value
    );
    const float right = pixtreme_border_sample(
        source, x + 1, y, width, height, channel_count, channel, border, border_value
    );
    const float bottom_left = pixtreme_border_sample(
        source, x - 1, y + 1, width, height, channel_count, channel, border, border_value
    );
    const float bottom = pixtreme_border_sample(
        source, x, y + 1, width, height, channel_count, channel, border, border_value
    );
    const float bottom_right = pixtreme_border_sample(
        source, x + 1, y + 1, width, height, channel_count, channel, border, border_value
    );

    const float derivative_x =
        -top_left + top_right - 2.0f * left + 2.0f * right - bottom_left + bottom_right;
    const float derivative_y =
        -top_left - 2.0f * top - top_right + bottom_left + 2.0f * bottom + bottom_right;
    output[index] = direction == 0 ? derivative_x : (direction == 1 ? derivative_y : hypotf(derivative_x, derivative_y));
}

extern "C" __global__ void pixtreme_laplacian(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const int border,
    const float border_value
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long element_count = width * height * channel_count;
    if (index >= element_count) {
        return;
    }

    const long long channel = index % channel_count;
    const long long pixel_index = index / channel_count;
    const long long x = pixel_index % width;
    const long long y = pixel_index / width;
    const float center = source[index];
    const float left = pixtreme_border_sample(
        source, x - 1, y, width, height, channel_count, channel, border, border_value
    );
    const float right = pixtreme_border_sample(
        source, x + 1, y, width, height, channel_count, channel, border, border_value
    );
    const float top = pixtreme_border_sample(
        source, x, y - 1, width, height, channel_count, channel, border, border_value
    );
    const float bottom = pixtreme_border_sample(
        source, x, y + 1, width, height, channel_count, channel, border, border_value
    );
    output[index] = left + right + top + bottom - 4.0f * center;
}
"""
)


@lru_cache(maxsize=1)
def _sobel_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DERIVATIVE_KERNEL_SOURCE, "pixtreme_sobel")


@lru_cache(maxsize=1)
def _laplacian_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DERIVATIVE_KERNEL_SOURCE, "pixtreme_laplacian")


def _validate_direction(direction: object) -> str:
    return _normalized_closed_token(direction, axis="direction", accepted=_DIRECTION_TOKENS)


def _block_count(frame: Frame) -> int:
    element_count = int(frame.data.size)
    return (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK


def sobel(
    frame: Frame,
    *,
    direction: SobelDirection = "magnitude",
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Apply the standard non-normalized 3x3 Sobel derivative to a float32 Frame.

    ``x`` uses derivative ``[-1, 0, 1]`` horizontally and smoothing
    ``[1, 2, 1]`` vertically; it responds to vertical edges. ``y`` is the
    transpose and responds to horizontal edges. ``magnitude`` is the default
    and equals ``sqrt(x**2 + y**2)`` per channel. The scale is not normalized:
    a unit horizontal ramp has an interior x response of 8.

    Border defaults to ``mirror``; ``replicate`` clamps to the edge, ``wrap``
    is periodic, and ``constant`` requires a finite ``border_value``. The
    calculation applies independently and uniformly to all channels, preserves
    Frame metadata and input storage, and does not clamp negative values or
    values above 1. Convert non-float32 storage with ``px.values.cast_dtype``,
    ``px.values.recode_dtype``, or ``px.values.dequantize`` according to its meaning.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.sobel")
    checked_direction = _validate_direction(direction)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    blocks = _block_count(checked_frame)
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    _sobel_kernel()(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (
            checked_frame.data,
            output,
            np.int64(checked_frame.width),
            np.int64(checked_frame.height),
            np.int64(len(checked_frame.channels)),
            np.int32(_DIRECTION_TOKENS.index(checked_direction)),
            _border_argument(checked_border),
            np.float32(checked_border_value),
        ),
    )
    return _new_frame(checked_frame, output)


def laplacian(
    frame: Frame,
    *,
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Apply the fixed non-normalized 3x3 Laplacian to a float32 Frame.

    The kernel is ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]``. No Gaussian
    smoothing or LoG behavior is built in. Border defaults to ``mirror``;
    ``replicate`` clamps to the edge, ``wrap`` is periodic, and ``constant``
    requires a finite ``border_value``. A uniform image is zero at every pixel
    when a constant border uses the same uniform value.

    The calculation applies independently and uniformly to all channels,
    preserves Frame metadata and input storage, and does not clamp negative
    values or values above 1. Convert non-float32 storage with
    ``px.values.cast_dtype``, ``px.values.recode_dtype``, or ``px.values.dequantize``.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.laplacian")
    checked_border, checked_border_value = _resolve_border(border, border_value)
    blocks = _block_count(checked_frame)
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    _laplacian_kernel()(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (
            checked_frame.data,
            output,
            np.int64(checked_frame.width),
            np.int64(checked_frame.height),
            np.int64(len(checked_frame.channels)),
            _border_argument(checked_border),
            np.float32(checked_border_value),
        ),
    )
    return _new_frame(checked_frame, output)


def difference_of_gaussians(
    frame: Frame,
    *,
    sigma1: float,
    sigma2: float,
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Subtract ``gaussian_blur(sigma2)`` from ``gaussian_blur(sigma1)``.

    Each Gaussian uses the public ``gaussian_blur`` contract, including
    ``radius = ceil(3 * sigma)``. ``sigma1`` and ``sigma2`` must be positive
    finite real values. Their order is unrestricted and determines the sign;
    equal sigmas produce a zero Frame without error.

    Border defaults to ``mirror``; ``replicate`` clamps to the edge, ``wrap``
    is periodic, and ``constant`` requires a finite ``border_value``. The
    float32 calculation applies independently and uniformly to all channels,
    preserves Frame metadata and input storage, and does not clamp negative
    values or values above 1. Convert non-float32 storage with
    ``px.values.cast_dtype``, ``px.values.recode_dtype``, or ``px.values.dequantize``.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.difference_of_gaussians")
    checked_sigma1 = _validate_sigma(sigma1, name="sigma1")
    checked_sigma2 = _validate_sigma(sigma2, name="sigma2")
    checked_border, checked_border_value = _resolve_border(border, border_value)
    gaussian_border_value = checked_border_value if checked_border == "constant" else None
    first = gaussian_blur(
        checked_frame,
        sigma=checked_sigma1,
        border=checked_border,
        border_value=gaussian_border_value,
    )
    second = gaussian_blur(
        checked_frame,
        sigma=checked_sigma2,
        border=checked_border,
        border_value=gaussian_border_value,
    )
    return _new_frame(checked_frame, first.data - second.data)
