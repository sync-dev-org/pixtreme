"""GPU morphology filters with explicit structuring-element and border contracts."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _BORDER_PREAMBLE, _border_argument, _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_frame
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import _MORPHOLOGY_SHAPE_TOKENS

_SHAPE_TOKENS = _MORPHOLOGY_SHAPE_TOKENS
_THREADS_PER_BLOCK = 256

_MORPHOLOGY_KERNEL_SOURCE = (
    _BORDER_PREAMBLE
    + r"""
extern "C" __global__ void pixtreme_morphology(
    const float* __restrict__ source,
    float* __restrict__ output,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long radius,
    const int shape,
    const int border,
    const float border_value,
    const int dilate
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
    float value = source[index];
    const double radius_squared = (double)radius * (double)radius;

    for (long long offset_y = -radius; offset_y <= radius; ++offset_y) {
        for (long long offset_x = -radius; offset_x <= radius; ++offset_x) {
            if (
                shape == 0
                && (double)offset_x * (double)offset_x + (double)offset_y * (double)offset_y > radius_squared
            ) {
                continue;
            }
            const float sample = pixtreme_border_sample(
                source,
                x + offset_x,
                y + offset_y,
                width,
                height,
                channel_count,
                channel,
                border,
                border_value
            );
            value = dilate ? fmaxf(value, sample) : fminf(value, sample);
        }
    }
    output[index] = value;
}
"""
)


@lru_cache(maxsize=1)
def _morphology_kernel() -> cp.RawKernel:
    return cp.RawKernel(_MORPHOLOGY_KERNEL_SOURCE, "pixtreme_morphology")


def _validate_radius(radius: object, *, operation: str) -> int:
    if type(radius) is not int or radius < 1:
        raise ValueError(
            _actionable_error(
                why=f"{operation} radius must be a built-in int of at least 1",
                what=f"received radius={radius!r}",
                how="pass radius as an int greater than or equal to 1",
            )
        )
    return radius


def _validate_shape(shape: object) -> str:
    if shape not in _SHAPE_TOKENS:
        raise ValueError(
            _actionable_error(
                why="shape is a closed, case-sensitive morphology token axis",
                what=f"received shape={shape!r}",
                how=f"pass one of {_SHAPE_TOKENS!r}",
            )
        )
    return str(shape)


def _validate_arguments(
    frame: object,
    *,
    operation: str,
    radius: object,
    shape: object,
    border: object,
    border_value: object,
) -> tuple[Frame, int, str, str, float]:
    checked_frame = _validate_frame(frame, operation=operation)
    dtype = np.dtype(checked_frame.dtype)
    if dtype != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why=f"{operation} requires float32 Frame data",
                what=f"received Frame data dtype {dtype.name}",
                how=_float32_conversion_guidance(dtype),
            )
        )
    checked_radius = _validate_radius(radius, operation=operation)
    checked_shape = _validate_shape(shape)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    return checked_frame, checked_radius, checked_shape, checked_border, checked_border_value


def _primitive(
    frame: Frame,
    *,
    radius: int,
    shape: str,
    border: str,
    border_value: float,
    dilate: bool,
) -> Frame:
    output = cp.empty(frame.shape, dtype=cp.float32)
    element_count = int(frame.data.size)
    blocks = (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _morphology_kernel()(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (
            frame.data,
            output,
            np.int64(frame.width),
            np.int64(frame.height),
            np.int64(len(frame.channels)),
            np.int64(radius),
            np.int32(_SHAPE_TOKENS.index(shape)),
            _border_argument(border),
            np.float32(border_value),
            np.int32(dilate),
        ),
    )
    return _new_frame(frame, output)


def _validated_primitive(
    frame: Frame,
    *,
    operation: str,
    radius: int,
    shape: str,
    border: str,
    border_value: float | None,
    dilate: bool,
) -> Frame:
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation=operation,
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    return _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=dilate,
    )


def erosion(
    frame: Frame,
    *,
    radius: int,
    shape: str = "disk",
    border: str = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Replace every fp32 channel value with its structuring-element minimum.

    ``radius`` is an int of at least 1. ``disk`` includes integer offsets whose
    squared distance is at most ``radius ** 2``; ``square`` includes the full
    ``(2 * radius + 1)`` window. Border defaults to neutral ``replicate``;
    ``mirror`` reflects without repeating the edge, ``wrap`` is periodic, and
    ``constant`` requires a finite ``border_value``. Processing is independent
    per channel, preserves Frame metadata, and does not clamp scene values.
    """
    return _validated_primitive(
        frame,
        operation="morphology.erosion",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        dilate=False,
    )


def dilation(
    frame: Frame,
    *,
    radius: int,
    shape: str = "disk",
    border: str = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Replace every fp32 channel value with its structuring-element maximum.

    ``radius`` is an int of at least 1. ``disk`` includes integer offsets whose
    squared distance is at most ``radius ** 2``; ``square`` includes the full
    ``(2 * radius + 1)`` window. Border defaults to neutral ``replicate``;
    ``mirror`` reflects without repeating the edge, ``wrap`` is periodic, and
    ``constant`` requires a finite ``border_value``. Processing is independent
    per channel, preserves Frame metadata, and does not clamp scene values.
    """
    return _validated_primitive(
        frame,
        operation="morphology.dilation",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
        dilate=True,
    )


def opening(
    frame: Frame,
    *,
    radius: int,
    shape: str = "disk",
    border: str = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Erode then dilate an fp32 Frame with one shared structuring element.

    ``radius`` is an int of at least 1; ``shape`` is ``disk`` or ``square``.
    Both stages use the same border, defaulting to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The per-channel operation
    preserves metadata and does not clamp negative values or values above 1.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.opening",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    eroded = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )
    return _primitive(
        eroded,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )


def closing(
    frame: Frame,
    *,
    radius: int,
    shape: str = "disk",
    border: str = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Dilate then erode an fp32 Frame with one shared structuring element.

    ``radius`` is an int of at least 1; ``shape`` is ``disk`` or ``square``.
    Both stages use the same border, defaulting to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The per-channel operation
    preserves metadata and does not clamp negative values or values above 1.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.closing",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    dilated = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )
    return _primitive(
        dilated,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )


def morphological_gradient(
    frame: Frame,
    *,
    radius: int,
    shape: str = "disk",
    border: str = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Return dilation minus erosion for an fp32 Frame.

    Both extrema use the same ``radius``, ``disk`` or ``square`` shape, and
    border. Border defaults to neutral ``replicate``; ``constant`` requires a
    finite ``border_value``. The subtraction is per channel, preserves Frame
    metadata, and does not clamp negative values or values above 1.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.morphological_gradient",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    eroded = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )
    dilated = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )
    return _new_frame(checked_frame, dilated.data - eroded.data)


def white_tophat(
    frame: Frame,
    *,
    radius: int,
    shape: str = "disk",
    border: str = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Return input minus opening to extract small bright detail.

    Opening uses one ``radius``, ``disk`` or ``square`` shape, and border for
    its erosion and dilation. Border defaults to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The fp32 subtraction is per
    channel, preserves Frame metadata, and does not clamp scene values.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.white_tophat",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    eroded = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )
    opened = _primitive(
        eroded,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )
    return _new_frame(checked_frame, checked_frame.data - opened.data)


def black_tophat(
    frame: Frame,
    *,
    radius: int,
    shape: str = "disk",
    border: str = "replicate",
    border_value: float | None = None,
) -> Frame:
    """Return closing minus input to extract small dark detail.

    Closing uses one ``radius``, ``disk`` or ``square`` shape, and border for
    its dilation and erosion. Border defaults to neutral ``replicate``;
    ``constant`` requires finite ``border_value``. The fp32 subtraction is per
    channel, preserves Frame metadata, and does not clamp scene values.
    """
    checked_frame, checked_radius, checked_shape, checked_border, checked_border_value = _validate_arguments(
        frame,
        operation="morphology.black_tophat",
        radius=radius,
        shape=shape,
        border=border,
        border_value=border_value,
    )
    dilated = _primitive(
        checked_frame,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=True,
    )
    closed = _primitive(
        dilated,
        radius=checked_radius,
        shape=checked_shape,
        border=checked_border,
        border_value=checked_border_value,
        dilate=False,
    )
    return _new_frame(checked_frame, closed.data - checked_frame.data)
