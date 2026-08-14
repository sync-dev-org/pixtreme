"""GPU unsharp masking filter."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.border import _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _new_frame, _validate_frame
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._filter.common import _validate_amount, _validate_sigma
from pixtreme._filter.gaussian import gaussian_blur


@lru_cache(maxsize=1)
def _unsharp_mask_kernel() -> cp.ElementwiseKernel:
    return cp.ElementwiseKernel(
        "float32 source, float32 blurred, float32 amount",
        "float32 sharpened",
        "sharpened = source + amount * (source - blurred)",
        "pixtreme_unsharp_mask",
    )


def unsharp_mask(
    frame: Frame,
    *,
    sigma: float,
    amount: float,
    border: str = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Sharpen a float32 Frame with ``input + amount * (input - G(input))``.

    ``G`` is the same isotropic Gaussian as :func:`gaussian_blur`: its kernel
    radius is fixed as ``radius = ceil(3 * sigma)`` and its discrete weights are
    normalized by their sum. ``amount`` may be negative; zero returns a private
    bit-exact copy. Border defaults to ``mirror`` (edge-excluding reflection),
    ``replicate`` clamps to the edge, and ``wrap`` uses periodic indices.
    ``constant`` uses ``border_value`` outside the image; ``border_value`` is
    required with ``constant`` and forbidden for every other border.

    The fp32 calculation applies independently and uniformly to all channels,
    including alpha, and does not clamp negative values, values above 1, or
    sharpening halos. Frame metadata passes through and the input remains
    unchanged. Convert float16 literal values first with
    ``px.values.cast_dtype(frame, dtype="float32")``; use ``px.values.recode_dtype`` or
    ``px.values.dequantize`` for integer storage according to its value meaning.
    """
    checked_frame = _validate_frame(frame, operation="filter.unsharp_mask")
    dtype = np.dtype(checked_frame.dtype)
    if dtype != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why="unsharp_mask requires float32 Frame data",
                what=f"received Frame data dtype {dtype.name}",
                how=_float32_conversion_guidance(dtype),
            )
        )
    checked_sigma = _validate_sigma(sigma, name="sigma")
    checked_amount = _validate_amount(amount)
    checked_border, checked_border_value = _resolve_border(border, border_value)
    if checked_amount == 0.0:
        return _new_frame(checked_frame, checked_frame.data.copy())

    blurred = gaussian_blur(
        checked_frame,
        sigma=checked_sigma,
        border=checked_border,
        border_value=checked_border_value if checked_border == "constant" else None,
    )
    output = _unsharp_mask_kernel()(checked_frame.data, blurred.data, np.float32(checked_amount))
    return _new_frame(checked_frame, output)
