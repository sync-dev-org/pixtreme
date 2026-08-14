"""GPU box blur and box convolution."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp

from pixtreme._core.border import _resolve_border
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._filter.common import (
    _SEPARABLE_KERNEL_SOURCE,
    _launch_box_axis,
    _validate_odd_size,
)


@lru_cache(maxsize=1)
def _box_horizontal_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_box_horizontal")


@lru_cache(maxsize=1)
def _box_vertical_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_box_vertical")


@lru_cache(maxsize=1)
def _box_horizontal_global_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_box_horizontal_global")


@lru_cache(maxsize=1)
def _box_vertical_global_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_box_vertical_global")


def _convolve_box(
    frame: Frame,
    *,
    height: int,
    width: int,
    normalize: bool,
    border: str,
    border_value: float,
) -> Frame:
    output = cp.empty(frame.shape, dtype=cp.float32)
    horizontal_scale = 1.0 / width if normalize else 1.0
    vertical_scale = 1.0 / height if normalize else 1.0

    if width > 1 or height == 1:
        horizontal_output = output if height == 1 else cp.empty(frame.shape, dtype=cp.float32)
        _launch_box_axis(
            _box_horizontal_kernel(),
            _box_horizontal_global_kernel(),
            frame.data,
            horizontal_output,
            frame=frame,
            radius=width // 2,
            scale=horizontal_scale,
            border=border,
            border_value=border_value,
            horizontal=True,
        )
        if height == 1:
            return _new_frame(frame, output)
        vertical_source = horizontal_output
        vertical_border_value = border_value * width * horizontal_scale
    else:
        vertical_source = frame.data
        vertical_border_value = border_value

    _launch_box_axis(
        _box_vertical_kernel(),
        _box_vertical_global_kernel(),
        vertical_source,
        output,
        frame=frame,
        radius=height // 2,
        scale=vertical_scale,
        border=border,
        border_value=vertical_border_value,
        horizontal=False,
    )
    return _new_frame(frame, output)


def box_blur(
    frame: Frame,
    *,
    size: int,
    border: str = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Replace each channel value with its square-window mean.

    ``size`` is a positive odd integer. Border defaults to ``mirror``
    (edge-excluding reflection); ``replicate`` clamps to the edge and ``wrap``
    uses periodic indices. ``constant`` uses ``border_value`` for every virtual
    pixel outside the image; ``border_value`` is required with ``constant`` and
    forbidden for every other border. Calculation is fp32 per channel.
    It does not clamp scene values; negative values and values above 1 pass through.
    Size 1 is an identity in new storage.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.box_blur")
    checked_size = _validate_odd_size(size, operation="filter.box_blur")
    checked_border, checked_border_value = _resolve_border(border, border_value)
    return _convolve_box(
        checked_frame,
        height=checked_size,
        width=checked_size,
        normalize=True,
        border=checked_border,
        border_value=checked_border_value,
    )
