"""p010 in-memory wire import and export."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import _RANGE_TOKENS
from pixtreme._core.vocabulary import ChromaSiting, Colorspace, Gamma, Interpolation, Matrix, Range
from pixtreme._io.wire.sampling import (
    _INTERPOLATION_TOKENS,
    _SITING_TOKENS,
    _TO_INTERPOLATION_TOKENS,
    _dimensions,
    _from_subsampled,
    _matrix,
    _metadata,
    _subsampled_kernel_source,
    _to_subsampled,
    _to_subsampled_kernel_source,
    _token,
    _validate_buffer,
    _validate_frame,
)


@lru_cache(maxsize=None)
def _from_kernel(bit_depth: int, interpolation: str, siting: str) -> cp.RawKernel:
    source = _subsampled_kernel_source("p010", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_from_subsampled")


@lru_cache(maxsize=None)
def _to_kernel(bit_depth: int, interpolation: str, siting: str) -> cp.RawKernel:
    source = _to_subsampled_kernel_source("p010", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_to_subsampled")


def from_p010(
    buf: cp.ndarray,
    *,
    width: int,
    height: int,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    matrix: Matrix | None = None,
    range: Range = "legal",
    siting: ChromaSiting = "left",
    interpolation: Interpolation = "bilinear",
) -> Frame:
    """Construct a full-range fp32 YCbCr444 Frame from uint16 P010.

    The C-contiguous 1D layout is one Y plane followed by interleaved ``Cb Cr``.
    Each 10-bit code is MSB aligned in uint16; the lower 6 bits are padding and
    are ignored. ``siting`` selects the H.273 4:2:0 phase, ``interpolation``
    selects one of eight point filters, and ``range`` expands legal positions
    without clipping or maps the full 10-bit domain.
    ``colorspace`` / ``gamma`` override placeholders and ``matrix`` stamps basis provenance only.
    The conversion kernel is enqueued on the current CuPy stream and the call
    does not perform host synchronization. Order the decoder surface onto that
    stream before calling; a different consumer stream must wait on a CUDA event.
    """
    colorspace, gamma = _metadata(colorspace, gamma)
    matrix = _matrix(matrix)
    width, height = _dimensions(width, height, even_width=True, even_height=True)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    siting = _token(siting, axis="siting", accepted=_SITING_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_INTERPOLATION_TOKENS)
    pixel_count = width * height
    element_count = pixel_count + pixel_count // 2
    input_data = _validate_buffer(
        buf,
        operation="from_p010",
        dtype=np.dtype(np.uint16),
        element_count=element_count,
        shapes=((element_count,),),
    )
    return _from_subsampled(
        input_data,
        kernel=_from_kernel(10, interpolation, siting),
        layout="p010",
        width=width,
        height=height,
        bit_depth=10,
        range=range,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_p010(
    frame: Frame,
    *,
    range: Range = "legal",
    siting: ChromaSiting = "left",
    interpolation: Interpolation = "area",
) -> cp.ndarray:
    """Pack ``frame`` as P010 with 10-bit codes in each uint16 word's MSB.

    The lower 6 bits are zero. ``range`` selects code-value mapping, ``siting``
    selects the 4:2:0 chroma phase, and ``interpolation`` selects its filter.
    Packing is enqueued on the current CuPy stream without host synchronization;
    consume on that stream or pass its handle/event to the encoder.
    """
    _validate_frame(frame, operation="to_p010")
    _dimensions(frame.width, frame.height, even_width=True, even_height=True)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    siting = _token(siting, axis="siting", accepted=_SITING_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_TO_INTERPOLATION_TOKENS)
    return _to_subsampled(
        frame,
        kernel=_to_kernel(10, interpolation, siting),
        layout="p010",
        bit_depth=10,
        range=range,
    )
