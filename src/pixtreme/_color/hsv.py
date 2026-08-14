"""Label-driven RGB / HSV coordinate conversion."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame

_RGB_CHANNELS = ("R", "G", "B")
_HSV_CHANNELS = ("H", "S", "V")

_HSV_KERNEL_SOURCE = r"""
extern "C" __global__
void rgb_to_hsv_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int red_index,
    const int green_index,
    const int blue_index
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }

    const long long input_base = pixel * 3;
    const float red = input[input_base + red_index];
    const float green = input[input_base + green_index];
    const float blue = input[input_base + blue_index];
    const float maximum = fmaxf(red, fmaxf(green, blue));
    const float minimum = fminf(red, fminf(green, blue));
    const float delta = maximum - minimum;

    float hue = 0.0f;
    const float saturation = maximum == 0.0f ? 0.0f : delta / maximum;
    if (delta > 0.0f) {
        if (red == maximum) {
            hue = ((green - blue) / delta) / 6.0f;
        } else if (green == maximum) {
            hue = (2.0f + (blue - red) / delta) / 6.0f;
        } else {
            hue = (4.0f + (red - green) / delta) / 6.0f;
        }
        hue -= floorf(hue);
    }

    const long long output_base = pixel * 3;
    output[output_base] = hue;
    output[output_base + 1] = saturation;
    output[output_base + 2] = maximum;
}

extern "C" __global__
void hsv_to_rgb_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const long long pixel_count,
    const int hue_index,
    const int saturation_index,
    const int value_index
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }

    const long long input_base = pixel * 3;
    const float hue = input[input_base + hue_index];
    const float saturation = input[input_base + saturation_index];
    const float value = input[input_base + value_index];
    const long long output_base = pixel * 3;
    if (saturation == 0.0f) {
        output[output_base] = value;
        output[output_base + 1] = value;
        output[output_base + 2] = value;
        return;
    }

    const float wrapped_hue = hue - floorf(hue);
    const float h6 = 6.0f * wrapped_hue;
    const int sector = (int)floorf(h6);
    const float chroma = value * saturation;
    const float x = chroma * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
    const float minimum = value - chroma;
    float red_prime;
    float green_prime;
    float blue_prime;
    if (sector == 0) {
        red_prime = chroma;
        green_prime = x;
        blue_prime = 0.0f;
    } else if (sector == 1) {
        red_prime = x;
        green_prime = chroma;
        blue_prime = 0.0f;
    } else if (sector == 2) {
        red_prime = 0.0f;
        green_prime = chroma;
        blue_prime = x;
    } else if (sector == 3) {
        red_prime = 0.0f;
        green_prime = x;
        blue_prime = chroma;
    } else if (sector == 4) {
        red_prime = x;
        green_prime = 0.0f;
        blue_prime = chroma;
    } else {
        red_prime = chroma;
        green_prime = 0.0f;
        blue_prime = x;
    }

    output[output_base] = red_prime + minimum;
    output[output_base + 1] = green_prime + minimum;
    output[output_base + 2] = blue_prime + minimum;
}
"""


@lru_cache(maxsize=2)
def _hsv_kernel(name: str) -> cp.RawKernel:
    return cp.RawKernel(_HSV_KERNEL_SOURCE, name)


def _error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _validate_frame(frame: Frame, *, operation: str, expected_channels: tuple[str, str, str]) -> Frame:
    if not isinstance(frame, Frame):
        raise _error(
            why=f"{operation} is a Frame-to-Frame color operation",
            what=f"received {type(frame).__module__}.{type(frame).__qualname__}",
            how="construct a float32 px.core.Frame with the documented color triplet",
        )
    if np.dtype(frame.data.dtype) != np.dtype(np.float32):
        raise _error(
            why=f"{operation} evaluates HSV formulae in float32",
            what=f"received dtype={frame.data.dtype!s}",
            how=(
                "use px.values.cast_dtype for literal float values, px.values.recode_dtype for normalized integer "
                "containers, or px.values.dequantize for an explicit integer bit grid"
            ),
        )
    if len(frame.channels) != 3 or any(frame.channels.count(label) != 1 for label in expected_channels):
        raise _error(
            why=f"{operation} requires exactly one each of {expected_channels!r} and no other channels",
            what=f"received channels={frame.channels!r}",
            how=f"use px.channel.shuffle to produce the exact {expected_channels!r} triplet first",
        )
    return frame


def _run(frame: Frame, *, kernel_name: str, source_channels: tuple[str, str, str]) -> cp.ndarray:
    output = cp.empty(frame.shape, dtype=cp.float32)
    pixel_count = frame.height * frame.width
    indices = tuple(frame.channels.index(label) for label in source_channels)
    block_size = 256
    block_count = (pixel_count + block_size - 1) // block_size
    _hsv_kernel(kernel_name)(
        (block_count,),
        (block_size,),
        (frame.data, output, np.int64(pixel_count), *(np.int32(index) for index in indices)),
    )
    return output


def rgb_to_hsv(frame: Frame) -> Frame:
    """Convert an exact label-driven R/G/B float32 Frame to canonical H/S/V.

    ``rgb_to_hsv(frame) -> Frame`` accepts no domain, range, clip, or metadata
    override. The input must contain exactly one each of the R, G, and B labels
    in any order. With ``delta = maximum - minimum``, V is ``maximum``, S is
    ``delta / maximum`` except that maximum zero produces S zero, and H uses
    the maximum R/G/B sector equation divided by six and reduced modulo 1. V
    retains unbounded scene scale. Nonnegative finite RGB round-trips through
    :func:`hsv_to_rgb`; negative and nonfinite
    values are accepted but carry no nominal-range or round-trip guarantee.

    The result is a new C-contiguous float32 Frame with H/S/V labels in that
    order. It preserves colorspace and gamma, sets matrix to None, and leaves
    the input Frame and storage unchanged. Convert other dtypes according to
    value meaning with px.values.cast_dtype, px.values.recode_dtype, or
    px.values.dequantize before calling this function.
    """
    validated = _validate_frame(frame, operation="rgb_to_hsv", expected_channels=_RGB_CHANNELS)
    data = _run(validated, kernel_name="rgb_to_hsv_kernel", source_channels=_RGB_CHANNELS)
    return Frame(
        data=data,
        colorspace=validated.colorspace,
        gamma=validated.gamma,
        channels=_HSV_CHANNELS,
        matrix=None,
    )


def hsv_to_rgb(frame: Frame) -> Frame:
    """Convert an exact label-driven H/S/V float32 Frame to canonical R/G/B.

    ``hsv_to_rgb(frame) -> Frame`` accepts no domain, range, clip, or metadata
    override. The input must contain exactly one each of the H, S, and V labels
    in any order. H is wrapped modulo 1 before selecting one of six sectors;
    ``C = V * S``, ``X = C * (1 - abs((6 * H modulo 2) - 1))``, and
    ``m = V - C`` form the sector result. S and V enter these equations
    unchanged, V retains unbounded scene scale, and S zero returns ``(V, V, V)``.
    Nonnegative finite RGB produced by :func:`rgb_to_hsv` round-trips; negative
    and nonfinite values are accepted but carry no round-trip guarantee.

    The result is a new C-contiguous float32 Frame with R/G/B labels in that
    order. It preserves colorspace and gamma, sets matrix to None, and leaves
    the input Frame and storage unchanged. Convert other dtypes according to
    value meaning with px.values.cast_dtype, px.values.recode_dtype, or
    px.values.dequantize before calling this function.
    """
    validated = _validate_frame(frame, operation="hsv_to_rgb", expected_channels=_HSV_CHANNELS)
    data = _run(validated, kernel_name="hsv_to_rgb_kernel", source_channels=_HSV_CHANNELS)
    return Frame(
        data=data,
        colorspace=validated.colorspace,
        gamma=validated.gamma,
        channels=_RGB_CHANNELS,
        matrix=None,
    )
