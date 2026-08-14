"""Border validation and GPU sampling substrate."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.vocabulary import _BORDER_TOKENS

_BORDER_PREAMBLE = r"""
__device__ long long pixtreme_positive_modulo(const long long value, const long long modulus) {
    const long long remainder = value % modulus;
    return remainder < 0 ? remainder + modulus : remainder;
}

__device__ long long pixtreme_border_index(
    const long long index,
    const long long extent,
    const int border
) {
    if (extent <= 1) {
        return 0;
    }
    if (border == 1) {
        return index < 0 ? 0 : (index >= extent ? extent - 1 : index);
    }
    if (border == 2) {
        return pixtreme_positive_modulo(index, extent);
    }
    const long long period = 2 * extent - 2;
    const long long reflected = pixtreme_positive_modulo(index, period);
    return reflected < extent ? reflected : period - reflected;
}

template <typename T>
__device__ float pixtreme_border_sample(
    const T& source,
    const long long x,
    const long long y,
    const long long width,
    const long long height,
    const long long channel_count,
    const long long channel,
    const int border,
    const float border_value
) {
    if (border == 3 && (x < 0 || x >= width || y < 0 || y >= height)) {
        return border_value;
    }
    const long long source_x = pixtreme_border_index(x, width, border);
    const long long source_y = pixtreme_border_index(y, height, border);
    const long long source_index = (source_y * width + source_x) * channel_count + channel;
    return (float)source[source_index];
}
"""


def _validate_border(border: object) -> str:
    if border not in _BORDER_TOKENS:
        raise ValueError(
            _actionable_error(
                why="border is a closed, case-sensitive token axis",
                what=f"received border={border!r}",
                how=f"pass one of {_BORDER_TOKENS!r}",
            )
        )
    return str(border)


def _resolve_border(border: object, border_value: object) -> tuple[str, float]:
    checked_border = _validate_border(border)
    if checked_border == "constant":
        if isinstance(border_value, bool) or not isinstance(border_value, Real):
            raise ValueError(
                _actionable_error(
                    why="constant border requires border_value as a finite real number",
                    what=f"received border_value={border_value!r}",
                    how="pass a finite int or float border_value; negative values and values above 1 are allowed",
                )
            )
        resolved = float(border_value)
        if not math.isfinite(resolved):
            raise ValueError(
                _actionable_error(
                    why="constant border requires a finite border_value",
                    what=f"received border_value={border_value!r}",
                    how="pass a finite int or float border_value; negative values and values above 1 are allowed",
                )
            )
        return checked_border, resolved
    if border_value is not None:
        raise ValueError(
            _actionable_error(
                why="border_value applies only when border='constant'",
                what=f"received border={checked_border!r} with border_value={border_value!r}",
                how="omit border_value, or pass border='constant' with a finite real border_value",
            )
        )
    return checked_border, 0.0


def _border_argument(border: str) -> np.int32:
    return np.int32(_BORDER_TOKENS.index(border))
