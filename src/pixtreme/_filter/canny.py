"""Deterministic per-channel Canny edge detection on the GPU."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._filter.derivative import _block_count, sobel

_THREADS_PER_BLOCK = 256

_CANNY_KERNEL_SOURCE = (
    _BORDER_PREAMBLE
    + r"""
extern "C" __global__ void pixtreme_canny_nms_threshold(
    const float* __restrict__ derivative_x,
    const float* __restrict__ derivative_y,
    const float* __restrict__ magnitude,
    float* __restrict__ output,
    unsigned char* __restrict__ weak,
    const long long width,
    const long long height,
    const long long channel_count,
    const float threshold_low,
    const float threshold_high,
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
    float theta = atan2f(derivative_y[index], derivative_x[index]) * 57.29577951308232f;
    if (theta < 0.0f) {
        theta += 180.0f;
    }
    if (theta >= 180.0f) {
        theta -= 180.0f;
    }

    long long offset_x;
    long long offset_y;
    if (theta < 22.5f || theta >= 157.5f) {
        offset_x = 1;
        offset_y = 0;
    } else if (theta < 67.5f) {
        offset_x = 1;
        offset_y = 1;
    } else if (theta < 112.5f) {
        offset_x = 0;
        offset_y = 1;
    } else {
        offset_x = -1;
        offset_y = 1;
    }

    const float current = magnitude[index];
    const float negative = pixtreme_border_sample(
        magnitude,
        x - offset_x,
        y - offset_y,
        width,
        height,
        channel_count,
        channel,
        border,
        border_value
    );
    const float positive = pixtreme_border_sample(
        magnitude,
        x + offset_x,
        y + offset_y,
        width,
        height,
        channel_count,
        channel,
        border,
        border_value
    );
    const float suppressed = current > negative && current >= positive ? current : 0.0f;
    const bool is_strong = suppressed >= threshold_high;
    output[index] = is_strong ? 1.0f : 0.0f;
    weak[index] = !is_strong && suppressed >= threshold_low ? 1 : 0;
}

extern "C" __global__ void pixtreme_canny_propagate(
    const unsigned char* __restrict__ weak,
    float* __restrict__ output,
    int* __restrict__ changed,
    const long long width,
    const long long height,
    const long long channel_count
) {
    const long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long element_count = width * height * channel_count;
    if (index >= element_count || weak[index] == 0 || output[index] != 0.0f) {
        return;
    }

    const long long channel = index % channel_count;
    const long long pixel_index = index / channel_count;
    const long long x = pixel_index % width;
    const long long y = pixel_index / width;
    for (long long offset_y = -1; offset_y <= 1; ++offset_y) {
        const long long neighbor_y = y + offset_y;
        if (neighbor_y < 0 || neighbor_y >= height) {
            continue;
        }
        for (long long offset_x = -1; offset_x <= 1; ++offset_x) {
            if (offset_x == 0 && offset_y == 0) {
                continue;
            }
            const long long neighbor_x = x + offset_x;
            if (neighbor_x < 0 || neighbor_x >= width) {
                continue;
            }
            const long long neighbor_index = (neighbor_y * width + neighbor_x) * channel_count + channel;
            if (output[neighbor_index] != 0.0f) {
                output[index] = 1.0f;
                atomicExch(changed, 1);
                return;
            }
        }
    }
}
"""
)


@lru_cache(maxsize=1)
def _nms_threshold_kernel() -> cp.RawKernel:
    return cp.RawKernel(_CANNY_KERNEL_SOURCE, "pixtreme_canny_nms_threshold")


@lru_cache(maxsize=1)
def _propagate_kernel() -> cp.RawKernel:
    return cp.RawKernel(_CANNY_KERNEL_SOURCE, "pixtreme_canny_propagate")


def _validate_threshold(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be a nonnegative finite real number",
                what=f"received {name}={value!r}",
                how=f"pass a finite int or float {name} greater than or equal to 0",
            )
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(
            _actionable_error(
                why=f"{name} must be finite and greater than or equal to 0",
                what=f"received {name}={value!r}",
                how=f"pass a finite nonnegative int or float for {name}",
            )
        )
    return resolved


def _validate_threshold_order(threshold_low: float, threshold_high: float) -> None:
    if threshold_low > threshold_high:
        raise ValueError(
            _actionable_error(
                why="threshold_low must not exceed threshold_high",
                what=f"received threshold_low={threshold_low!r}, threshold_high={threshold_high!r}",
                how="pass ordered thresholds; equal values select the supported single-threshold mode",
            )
        )


def canny(
    frame: Frame,
    *,
    threshold_low: float,
    threshold_high: float,
    border: str = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Detect binary edges in a float32 Frame with a deterministic Canny pipeline.

    ``threshold_low`` and ``threshold_high`` are required nonnegative absolute
    scene-gradient strengths on the non-normalized 3x3 Sobel scale. The
    pipeline applies Sobel L2 magnitude, four-sector NMS, double thresholding,
    and 8-connected hysteresis with complete convergence and no iteration cap.
    NMS retains a sample exactly when ``current > magnitude(-v)`` and
    ``current >= magnitude(+v)``; equal thresholds select a single-threshold
    mode with no weak edge set. No Gaussian smoothing, normalization, or
    grayscale conversion is implicit.

    Border defaults to ``mirror``; ``replicate`` clamps to the edge, ``wrap``
    is periodic, and ``constant`` requires a finite ``border_value``. The same
    border rule applies to Sobel source samples and NMS magnitude samples, but
    hysteresis never connects through virtual pixels. Processing is per channel
    without using channel labels. The result is a private C-contiguous float32
    Frame containing only 0.0 and 1.0, with metadata preserved and the input
    storage unchanged.

    Convert non-float32 storage according to its value meaning with
    ``px.values.cast_dtype``, ``px.values.recode_dtype``, or
    ``px.values.dequantize`` before calling this operation.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.canny")
    checked_threshold_low = _validate_threshold(threshold_low, name="threshold_low")
    checked_threshold_high = _validate_threshold(threshold_high, name="threshold_high")
    _validate_threshold_order(checked_threshold_low, checked_threshold_high)
    checked_border, checked_border_value = _resolve_border(border, border_value)

    sobel_border_value = checked_border_value if checked_border == "constant" else None
    derivative_x = sobel(
        checked_frame,
        direction="x",
        border=checked_border,
        border_value=sobel_border_value,
    )
    derivative_y = sobel(
        checked_frame,
        direction="y",
        border=checked_border,
        border_value=sobel_border_value,
    )
    magnitude = sobel(
        checked_frame,
        direction="magnitude",
        border=checked_border,
        border_value=sobel_border_value,
    )

    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    weak = cp.empty(checked_frame.shape, dtype=cp.uint8)
    blocks = _block_count(checked_frame)
    shape_arguments = (
        np.int64(checked_frame.width),
        np.int64(checked_frame.height),
        np.int64(len(checked_frame.channels)),
    )
    _nms_threshold_kernel()(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (
            derivative_x.data,
            derivative_y.data,
            magnitude.data,
            output,
            weak,
            *shape_arguments,
            np.float32(checked_threshold_low),
            np.float32(checked_threshold_high),
            _border_argument(checked_border),
            np.float32(checked_border_value),
        ),
    )

    changed = cp.empty((1,), dtype=cp.int32)
    while True:
        changed.fill(0)
        _propagate_kernel()(
            (blocks,),
            (_THREADS_PER_BLOCK,),
            (weak, output, changed, *shape_arguments),
        )
        if int(changed.get()[0]) == 0:
            break

    return _new_frame(checked_frame, output)
