"""GPU Gaussian blur and unsharp masking."""

from __future__ import annotations

import math
from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _resolve_border
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._core.vocabulary import Border
from pixtreme._filter.common import (
    _SEPARABLE_KERNEL_SOURCE,
    _launch_gaussian_axis,
    _validate_sigma,
)

_GAUSSIAN_WEIGHT_CACHE_SIZE = 32


@lru_cache(maxsize=1)
def _gaussian_horizontal_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_gaussian_horizontal")


@lru_cache(maxsize=1)
def _gaussian_vertical_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_gaussian_vertical")


@lru_cache(maxsize=1)
def _gaussian_horizontal_global_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_gaussian_horizontal_global")


@lru_cache(maxsize=1)
def _gaussian_vertical_global_kernel() -> cp.RawKernel:
    return cp.RawKernel(_SEPARABLE_KERNEL_SOURCE, "pixtreme_gaussian_vertical_global")


@lru_cache(maxsize=_GAUSSIAN_WEIGHT_CACHE_SIZE)
def _gaussian_weights(device_id: int, sigma: float) -> cp.ndarray:
    radius = math.ceil(3.0 * sigma)
    with cp.cuda.Device(device_id):
        coordinates = cp.arange(-radius, radius + 1, dtype=cp.float32)
        weights = cp.exp(coordinates * coordinates * np.float32(-0.5 / (sigma * sigma)))
        weights /= cp.sum(weights, dtype=cp.float32)
    return weights


def gaussian_blur(
    frame: Frame,
    *,
    sigma: float,
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Apply an isotropic Gaussian blur without changing Frame metadata.

    Kernel radius is fixed as ``radius = ceil(3 * sigma)``; discrete 2D
    Gaussian weights are normalized by their sum. Border defaults to ``mirror``
    (edge-excluding reflection); ``replicate`` clamps to the edge and ``wrap``
    uses periodic indices. ``constant`` uses ``border_value`` for every virtual
    pixel outside the image; ``border_value`` is required with ``constant`` and
    forbidden for every other border. Calculation is fp32 per channel.
    It does not clamp scene values; negative values and values above 1 pass through.
    The result always owns a new allocation.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.gaussian_blur")
    checked_sigma = _validate_sigma(sigma, name="sigma")
    checked_border, checked_border_value = _resolve_border(border, border_value)
    radius = math.ceil(3.0 * checked_sigma)
    weights = _gaussian_weights(cp.cuda.runtime.getDevice(), checked_sigma)
    intermediate = cp.empty(checked_frame.shape, dtype=cp.float32)
    output = cp.empty(checked_frame.shape, dtype=cp.float32)
    _launch_gaussian_axis(
        _gaussian_horizontal_kernel(),
        _gaussian_horizontal_global_kernel(),
        checked_frame.data,
        weights,
        intermediate,
        frame=checked_frame,
        radius=radius,
        border=checked_border,
        border_value=checked_border_value,
        horizontal=True,
    )
    _launch_gaussian_axis(
        _gaussian_vertical_kernel(),
        _gaussian_vertical_global_kernel(),
        intermediate,
        weights,
        output,
        frame=checked_frame,
        radius=radius,
        border=checked_border,
        border_value=checked_border_value,
        horizontal=False,
    )
    return _new_frame(checked_frame, output)
