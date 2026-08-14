"""Shared validation for value-grid operations."""

from __future__ import annotations

import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.vocabulary import _RANGE_TOKENS as _RANGE_TOKENS

_BIT_DEPTHS = (8, 10, 12, 14, 16)


def _validate_bit_depth(value: object) -> int:
    if type(value) is not int or value not in _BIT_DEPTHS:
        raise ValueError(
            _actionable_error(
                why="bit_depth is a closed integer code-grid domain",
                what=f"received bit_depth={value!r}",
                how=f"pass a supported integer; expected one of {_BIT_DEPTHS!r}",
            )
        )
    return value


def _bit_depth_maximum(bit_depth: int) -> np.float32:
    return np.float32((1 << bit_depth) - 1)


def _bit_depth_scale(bit_depth: int) -> np.float32:
    return np.float32(1.0 / ((1 << bit_depth) - 1))


def _container_dtype(bit_depth: int) -> np.dtype[np.generic]:
    return np.dtype(np.uint8 if bit_depth == 8 else np.uint16)


def _legal_parameters(bit_depth: int) -> tuple[np.float32, np.float32, np.float32]:
    code_scale = 1 << (bit_depth - 8)
    container_maximum = np.float32((1 << bit_depth) - 1)
    lower_code = np.float32(16 * code_scale)
    luma_upper_code = np.float32(235 * code_scale)
    chroma_upper_code = np.float32(240 * code_scale)
    lower = np.float32(lower_code / container_maximum)
    luma_upper = np.float32(luma_upper_code / container_maximum)
    chroma_upper = np.float32(chroma_upper_code / container_maximum)
    return lower, np.float32(luma_upper - lower), np.float32(chroma_upper - lower)


def _float32_conversion_guidance(dtype: np.dtype[np.generic]) -> str:
    if dtype == np.dtype(np.float16):
        return 'preserve literal values with px.values.cast_dtype(frame, dtype="float32")'
    if dtype == np.dtype(np.uint8):
        return (
            'normalize container values with px.values.recode_dtype(frame, dtype="float32"), '
            "or use px.values.dequantize(frame, bit_depth=8) for an explicit bit grid"
        )
    if dtype == np.dtype(np.uint32):
        return (
            'convert numerically with px.values.recode_dtype(frame, dtype="float32"); '
            "integer identity above 2^24 is documented lossy in float32"
        )
    return (
        'normalize container values with px.values.recode_dtype(frame, dtype="float32"), '
        "or use px.values.dequantize(frame, bit_depth=<effective code bits>) for an explicit bit grid"
    )
