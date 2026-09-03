"""Fixed-Laplacian sharpening with shared filter validation."""

from __future__ import annotations

import numpy as np

from pixtreme._core.border import _resolve_border
from pixtreme._core.frame import Frame, _new_frame, _validate_float32_frame
from pixtreme._core.vocabulary import Border
from pixtreme._filter.common import _validate_amount
from pixtreme._filter.derivative import laplacian


def sharpen(
    frame: Frame,
    *,
    amount: float,
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Sharpen a float32 Frame with ``input - amount * laplacian(input)``.

    The basis is the fixed non-normalized 3x3 Laplacian
    ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]`` with no Gaussian smoothing.
    ``amount`` is required, accepts any finite real except bool, and may be
    negative. Zero returns a private bit-exact copy. Border defaults to
    ``mirror``; ``replicate`` clamps to the edge, ``wrap`` is periodic, and
    ``constant`` requires a finite ``border_value``. Other borders forbid
    ``border_value``.

    The float32 calculation applies independently and uniformly to all channels
    and does not clamp negative values, values above 1, or sharpening halos.
    Frame metadata passes through and the input remains unchanged. Convert
    non-float32 storage with ``px.values.cast_dtype``, ``px.values.recode_dtype``,
    or ``px.values.dequantize`` according to its value meaning.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.sharpen")
    checked_amount = _validate_amount(amount)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    if checked_amount == 0.0:
        return _new_frame(checked_frame, checked_frame.data.copy())

    derivative = laplacian(
        checked_frame,
        border=checked_border,
        border_value=checked_border_value if checked_border == "constant" else None,
    )
    output = checked_frame.data - np.float32(checked_amount) * derivative.data
    return _new_frame(checked_frame, output)
