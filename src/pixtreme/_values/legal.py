"""Legal- and full-range conversion implementations."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import (
    _RANGE_TOKENS as _RANGE_TOKENS,
)
from pixtreme._core.value_domain import _legal_parameters, _validate_bit_depth
from pixtreme._values.common import _new_frame, _validate_float32_frame, _validate_frame

_RANGE_CHANNELS = frozenset(("R", "G", "B", "Y", "Cb", "Cr"))
_CHROMA_CHANNELS = frozenset(("Cb", "Cr"))
_DTYPE_ITEM_SIZES = {"float32": 4}
_VECTOR_WIDTH = 4
_THREADS_PER_BLOCK = 256


def _range_expression(value: str, *, extent: str, direction: str) -> str:
    if direction == "legal_to_full":
        return f"__fdiv_rn(__fsub_rn({value}, lower), {extent})"
    return f"__fadd_rn(__fmul_rn({value}, {extent}), lower)"


@lru_cache(maxsize=16)
def _range_kernel(chroma_flags: tuple[bool, ...], direction: str) -> cp.RawKernel:
    uniform_extent = None
    if all(chroma_flags):
        uniform_extent = "chroma_extent"
    elif not any(chroma_flags):
        uniform_extent = "luma_extent"

    if uniform_extent is not None:
        vector_assignments = "\n".join(
            f"            output_value.{component} = "
            f"{_range_expression(f'input_value.{component}', extent=uniform_extent, direction=direction)};"
            for component in "xyzw"
        )
        scalar_expression = _range_expression("source[element]", extent=uniform_extent, direction=direction)
        kernel_body = f"""
    const long long vector_count = element_count / {_VECTOR_WIDTH};
    if (vectorized != 0) {{
        if (thread < vector_count) {{
            const float4 input_value = reinterpret_cast<const float4*>(source)[thread];
            float4 output_value;
{vector_assignments}
            reinterpret_cast<float4*>(destination)[thread] = output_value;
            return;
        }}
        const long long element = vector_count * {_VECTOR_WIDTH} + thread - vector_count;
        if (element < element_count) {{
            destination[element] = {scalar_expression};
        }}
        return;
    }}
    if (thread < element_count) {{
        const long long element = thread;
        destination[element] = {scalar_expression};
    }}
"""
    else:
        channel_assignments = "\n".join(
            (
                f"    destination[base + {channel}] = "
                f"{_range_expression(f'source[base + {channel}]', extent='chroma_extent' if is_chroma else 'luma_extent', direction=direction)};"
            )
            for channel, is_chroma in enumerate(chroma_flags)
        )
        kernel_body = f"""
    if (thread >= pixel_count) {{
        return;
    }}
    const long long base = thread * {len(chroma_flags)};
{channel_assignments}
"""

    source = f"""
extern "C" __global__ void pixtreme_convert_value_range(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const long long element_count,
    const long long pixel_count,
    const float lower,
    const float luma_extent,
    const float chroma_extent,
    const int vectorized
) {{
    const long long thread = (long long)blockDim.x * blockIdx.x + threadIdx.x;
{kernel_body}
}}
"""
    return cp.RawKernel(source, "pixtreme_convert_value_range")


def _run_range_kernel(
    data: cp.ndarray,
    channels: tuple[str, ...],
    *,
    direction: str,
    bit_depth: int,
) -> cp.ndarray:
    output = cp.empty_like(data)
    element_count = int(data.size)

    chroma_flags = tuple(label in _CHROMA_CHANNELS for label in channels)
    uniform_channels = all(chroma_flags) or not any(chroma_flags)
    vectorized = (
        uniform_channels
        and data.data.ptr % (_DTYPE_ITEM_SIZES["float32"] * _VECTOR_WIDTH) == 0
        and output.data.ptr % (_DTYPE_ITEM_SIZES["float32"] * _VECTOR_WIDTH) == 0
    )
    work_items = (
        element_count // _VECTOR_WIDTH + element_count % _VECTOR_WIDTH
        if vectorized
        else element_count
        if uniform_channels
        else int(data.shape[0] * data.shape[1])
    )
    block_count = (work_items + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    lower, luma_extent, chroma_extent = _legal_parameters(bit_depth)
    _range_kernel(chroma_flags, direction)(
        (block_count,),
        (_THREADS_PER_BLOCK,),
        (
            data,
            output,
            np.int64(element_count),
            np.int64(data.shape[0] * data.shape[1]),
            lower,
            luma_extent,
            chroma_extent,
            np.int32(vectorized),
        ),
    )
    return output


def _convert_range(frame: Frame, *, bit_depth: int, direction: str, operation: str) -> Frame:
    public_operation = f"values.{operation}"
    frame = _validate_frame(frame, operation=public_operation)
    bit_depth = _validate_bit_depth(bit_depth)
    _validate_float32_frame(frame, operation=public_operation)
    unsupported = tuple(label for label in frame.channels if label not in _RANGE_CHANNELS)
    if unsupported:
        raise ValueError(
            _actionable_error(
                why=f"{operation} has no range semantics for unsupported channel labels",
                what=f"received channels={frame.channels!r}, unsupported={unsupported!r}",
                how=f"pass a Frame whose channels are drawn from {tuple(sorted(_RANGE_CHANNELS))!r}",
            )
        )
    return _new_frame(
        frame,
        _run_range_kernel(frame.data, frame.channels, direction=direction, bit_depth=bit_depth),
    )


def legal_to_full(frame: Frame, *, bit_depth: int = 8) -> Frame:
    """Expand H.273 legal code positions to full-range fp32 without clipping.

    R, G, B, and Y use the luma interval; Cb and Cr use the chroma interval.
    ``bit_depth`` accepts 8, 10, 12, 14, or 16. Metadata and input storage are
    unchanged.

    To repair RGB produced by applying a matrix before legal-range expansion,
    reverse that composition in the same matrix domain::

        restored_ycbcr = px.color.rgb_to_ycbcr(frame, matrix="bt709")
        full_ycbcr = px.values.legal_to_full(restored_ycbcr, bit_depth=8)
        corrected_rgb = px.color.ycbcr_to_rgb(full_ycbcr, matrix="bt709")
    """
    return _convert_range(frame, bit_depth=bit_depth, direction="legal_to_full", operation="legal_to_full")


def full_to_legal(frame: Frame, *, bit_depth: int = 8) -> Frame:
    """Compress full-range fp32 values to H.273 legal positions without clipping.

    R, G, B, and Y use the luma interval; Cb and Cr use the chroma interval.
    ``bit_depth`` accepts 8, 10, 12, 14, or 16. Metadata and input storage are
    unchanged.
    """
    return _convert_range(frame, bit_depth=bit_depth, direction="full_to_legal", operation="full_to_legal")
