"""Pixel-value quantization and dequantization implementations."""

from __future__ import annotations

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import (
    _BIT_DEPTHS as _SUPPORTED_BIT_DEPTHS,
)
from pixtreme._core.value_domain import (
    _bit_depth_maximum,
    _bit_depth_scale,
    _container_dtype,
    _validate_bit_depth,
)
from pixtreme._core.value_kernel import _dequantize_expression, _linear_value_kernel, _quantize_expression
from pixtreme._values.common import _new_frame, _validate_float32_frame, _validate_frame

_BIT_DEPTHS = _SUPPORTED_BIT_DEPTHS
_THREADS_PER_BLOCK = 512


def quantize(frame: Frame, *, bit_depth: int) -> Frame:
    """Quantize normalized fp32 values onto a uniform unsigned full-scale grid.

    ``bit_depth`` accepts 8, 10, 12, 14, or 16 and defines maximum code
    ``2^bit_depth - 1``. Values are clipped to ``[0, 1]``, scaled, and rounded
    half away from zero. The result uses uint8 for 8-bit codes and uint16 for
    10-16-bit codes.

    This is per-call pixel-value quantization, not palette quantization or ML
    affine quantization. Frame stores no bit-depth state. Use :func:`cast_dtype`
    when literal numeric values, rather than their normalized meaning, must be
    preserved.
    """
    frame = _validate_frame(frame, operation="values.quantize")
    bit_depth = _validate_bit_depth(bit_depth)
    _validate_float32_frame(frame, operation="values.quantize")
    destination_dtype = _container_dtype(bit_depth)
    output = cp.empty(frame.shape, dtype=destination_dtype)
    element_count = int(frame.data.size)
    block_count = (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _linear_value_kernel(
        "pixtreme_quantize_values",
        "float32",
        destination_dtype.name,
        _quantize_expression,
        "value_maximum",
    )(
        (block_count,),
        (_THREADS_PER_BLOCK,),
        (frame.data, output, np.int64(element_count), _bit_depth_maximum(bit_depth)),
    )
    return _new_frame(frame, output)


def dequantize(frame: Frame, *, bit_depth: int) -> Frame:
    """Map unsigned integer codes to fp32 by dividing by ``2^bit_depth - 1``.

    ``bit_depth`` accepts 8, 10, 12, 14, or 16. Input storage must be uint8 for
    8-bit codes and uint16 for 10-16-bit codes. Codes above the declared
    maximum are not clipped and therefore remain above 1.0. Frame stores no
    bit-depth state. Use :func:`cast_dtype` for literal storage conversion.
    """
    frame = _validate_frame(frame, operation="values.dequantize")
    bit_depth = _validate_bit_depth(bit_depth)
    expected = _container_dtype(bit_depth)
    if np.dtype(frame.data.dtype) != expected:
        raise ValueError(
            _actionable_error(
                why=f"dequantize bit_depth={bit_depth} requires {expected.name} Frame data",
                what=f"received bit_depth={bit_depth!r}, dtype={np.dtype(frame.data.dtype)!r}",
                how=f"pass a {expected.name} Frame for bit_depth={bit_depth}",
            )
        )
    output = cp.empty(frame.shape, dtype=cp.float32)
    element_count = int(frame.data.size)
    block_count = (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _linear_value_kernel(
        "pixtreme_dequantize_values",
        expected.name,
        "float32",
        _dequantize_expression,
        "value_scale",
    )(
        (block_count,),
        (_THREADS_PER_BLOCK,),
        (frame.data, output, np.int64(element_count), _bit_depth_scale(bit_depth)),
    )
    return _new_frame(frame, output)
