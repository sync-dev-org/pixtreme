"""Storage dtype cast and recode implementations."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.frame import Frame
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.vocabulary import _DTYPE_TOKENS, Dtype
from pixtreme._values.common import _new_frame, _validate_frame

_CONTAINER_MAXIMA = {"uint8": 255, "uint16": 65535, "uint32": 4294967295}
_CUDA_STORAGE_TYPES = {
    "float32": "float",
    "float16": "unsigned short",
    "uint8": "unsigned char",
    "uint16": "unsigned short",
    "uint32": "unsigned int",
}
_THREADS_PER_BLOCK = 512


def _cuda_load_expression(dtype: str, value: str) -> str:
    if dtype == "float32":
        return value
    if dtype == "float16":
        return f"__half2float(__ushort_as_half({value}))"
    return f"(float)({value})"


def _cuda_store_expression(dtype: str, value: str) -> str:
    if dtype == "float32":
        return value
    if dtype == "float16":
        return f"__half_as_ushort(__float2half_rn({value}))"
    return f"({_CUDA_STORAGE_TYPES[dtype]})({value})"


def _recode_dtype_expression(
    source_dtype: str,
    target_dtype: str,
    source: str = "source[element]",
) -> str:
    if source_dtype.startswith("uint") and target_dtype.startswith("uint"):
        source_maximum = _CONTAINER_MAXIMA[source_dtype]
        target_maximum = _CONTAINER_MAXIMA[target_dtype]
        return (
            f"(unsigned long long)((((unsigned long long){source}) * {target_maximum}ULL "
            f"+ {source_maximum // 2}ULL) / {source_maximum}ULL)"
        )

    loaded = _cuda_load_expression(source_dtype, source)
    if source_dtype.startswith("uint"):
        return _cuda_store_expression(target_dtype, f"({loaded} * parameter)")

    if target_dtype == "uint32":
        clipped = f"fmin(fmax((double)({loaded}), 0.0), 1.0)"
        rounded = f"floor(__dadd_rn(__dmul_rn({clipped}, 4294967295.0), 0.5))"
        return _cuda_store_expression(target_dtype, f"(isnan((double)({loaded})) ? 2147483648.0 : {rounded})")

    rounded = f"floorf(fminf(fmaxf({loaded}, 0.0f), 1.0f) * parameter + 0.5f)"
    return _cuda_store_expression(target_dtype, rounded)


@lru_cache(maxsize=18)
def _recode_dtype_kernel(source_dtype: str, target_dtype: str) -> cp.RawKernel:
    expression = _recode_dtype_expression(source_dtype, target_dtype)
    source = f"""
#include <cuda_fp16.h>

extern "C" __global__ void pixtreme_recode_dtype(
    const {_CUDA_STORAGE_TYPES[source_dtype]}* __restrict__ source,
    {_CUDA_STORAGE_TYPES[target_dtype]}* __restrict__ destination,
    const long long element_count,
    const float parameter
) {{
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {{
        return;
    }}
    destination[element] = {expression};
}}
"""
    return cp.RawKernel(source, "pixtreme_recode_dtype")


def _recode_dtype_parameter(source_dtype: str, target_dtype: str) -> np.float32:
    if source_dtype.startswith("uint") and target_dtype.startswith("float"):
        return np.float32(1.0 / _CONTAINER_MAXIMA[source_dtype])
    if source_dtype.startswith("float") and target_dtype in {"uint8", "uint16"}:
        return np.float32(_CONTAINER_MAXIMA[target_dtype])
    return np.float32(0.0)


def _recode_dtype_rawkernel(frame: Frame, *, dtype: Dtype) -> Frame:
    frame = _validate_frame(frame, operation="values.recode_dtype")
    dtype = _validate_dtype_token(dtype)
    source_dtype = frame.dtype.name

    if source_dtype == dtype or (source_dtype.startswith("float") and dtype.startswith("float")):
        return cast_dtype(frame, dtype=dtype)

    output = cp.empty(frame.shape, dtype=dtype)
    element_count = int(frame.data.size)
    block_count = (element_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _recode_dtype_kernel(source_dtype, dtype)(
        (block_count,),
        (_THREADS_PER_BLOCK,),
        (frame.data, output, np.int64(element_count), _recode_dtype_parameter(source_dtype, dtype)),
    )
    return _new_frame(frame, output)


def _validate_dtype_token(value: object) -> Dtype:
    return _normalized_closed_token(value, axis="dtype", accepted=_DTYPE_TOKENS)


def recode_dtype(frame: Frame, *, dtype: Dtype) -> Frame:
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
    return _recode_dtype_rawkernel(frame, dtype=dtype)


def cast_dtype(frame: Frame, *, dtype: Dtype) -> Frame:
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
