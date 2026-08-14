"""Storage dtype cast and recode implementations."""

from __future__ import annotations

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.vocabulary import _DTYPE_TOKENS
from pixtreme._values.common import _new_frame, _validate_frame
from pixtreme._values.quantize import quantize

_CONTAINER_MAXIMA = {"uint8": 255, "uint16": 65535, "uint32": 4294967295}


def _validate_dtype_token(value: object) -> str:
    if not isinstance(value, str) or value not in _DTYPE_TOKENS:
        raise ValueError(
            _actionable_error(
                why="dtype is a closed, case-sensitive storage token",
                what=f"received dtype={value!r}",
                how=f"pass one of {_DTYPE_TOKENS!r}",
            )
        )
    return value


def recode_dtype(frame: Frame, *, dtype: str) -> Frame:
    """Recode storage while preserving normalized image meaning.

    Unsigned integer containers map their complete code range to ``[0, 1]``.
    Floating-point values map to unsigned integers by clipping to ``[0, 1]``,
    scaling to the target container maximum, and rounding half away from zero.
    Float-to-float conversion is a literal cast. Metadata is unchanged and
    every call returns a new allocation, including same-dtype conversion.

    Use :func:`cast_dtype` for literal numeric preservation. Use
    :func:`quantize` and :func:`dequantize` for the explicit
    bit-depth grid lane when effective code bits, rather than the storage
    container, define the scale.
    """
    frame = _validate_frame(frame, operation="values.recode_dtype")
    dtype = _validate_dtype_token(dtype)
    source_dtype = frame.dtype.name

    if source_dtype == dtype or (source_dtype.startswith("float") and dtype.startswith("float")):
        return cast_dtype(frame, dtype=dtype)

    if source_dtype.startswith("uint") and dtype.startswith("uint"):
        source_maximum = _CONTAINER_MAXIMA[source_dtype]
        target_maximum = _CONTAINER_MAXIMA[dtype]
        source = frame.data.astype(cp.uint64)
        scaled = (source * np.uint64(target_maximum) + np.uint64(source_maximum // 2)) // np.uint64(source_maximum)
        return _new_frame(frame, scaled.astype(dtype))

    if source_dtype.startswith("uint"):
        source_maximum = _CONTAINER_MAXIMA[source_dtype]
        normalized = _new_frame(frame, frame.data.astype(cp.float32) * np.float32(1.0 / source_maximum))
        if dtype == "float32":
            return normalized
        if dtype == "float16":
            return cast_dtype(normalized, dtype=dtype)
        raise AssertionError(f"unreachable uint target dtype after integer branch: {dtype!r}")

    normalized = frame if source_dtype == "float32" else cast_dtype(frame, dtype="float32")
    if dtype == "uint32":
        maximum = np.float64(_CONTAINER_MAXIMA[dtype])
        output = cp.floor(cp.clip(normalized.data.astype(cp.float64), 0.0, 1.0) * maximum + np.float64(0.5)).astype(
            cp.uint32
        )
        return _new_frame(frame, output)
    return quantize(normalized, bit_depth=8 if dtype == "uint8" else 16)


def cast_dtype(frame: Frame, *, dtype: str) -> Frame:
    """Cast storage while preserving literal numeric values.

    This is a direct CuPy ``astype`` operation: it adds no scaling, clipping,
    or explicit rounding. Metadata is unchanged and every call returns a new
    allocation. Use :func:`recode_dtype` when normalized image meaning must be
    preserved across storage dtypes. Use :func:`quantize` or
    :func:`dequantize` when an explicit bit-depth grid defines the scale.
    """
    frame = _validate_frame(frame, operation="values.cast_dtype")
    dtype = _validate_dtype_token(dtype)
    return _new_frame(frame, frame.data.astype(dtype))
