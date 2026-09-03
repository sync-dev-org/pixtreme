"""nv12 in-memory wire import and export."""

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
    source = _subsampled_kernel_source("nv12", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_from_subsampled")


@lru_cache(maxsize=None)
def _to_kernel(bit_depth: int, interpolation: str, siting: str) -> cp.RawKernel:
    source = _to_subsampled_kernel_source("nv12", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_to_subsampled")


def from_nv12(
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
    """Construct a full-range fp32 YCbCr444 Frame from uint8 NV12.

    The C-contiguous 1D layout is one Y plane followed by an interleaved
    ``Cb Cr`` plane. ``siting`` places 4:2:0 chroma at H.273 ``left``,
    ``center``, or ``topleft`` offsets; ``interpolation`` evaluates one of the
    eight resize-family point kernels at that phase with replicate edges.
    ``range`` expands legal code positions without clipping or maps full uint8.
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
        operation="from_nv12",
        dtype=np.dtype(np.uint8),
        element_count=element_count,
        shapes=((element_count,),),
    )
    return _from_subsampled(
        input_data,
        kernel=_from_kernel(8, interpolation, siting),
        layout="nv12",
        width=width,
        height=height,
        bit_depth=8,
        range=range,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_nv12(
    frame: Frame,
    *,
    range: Range = "legal",
    siting: ChromaSiting = "left",
    interpolation: Interpolation = "area",
) -> cp.ndarray:
    """Pack ``frame`` as an 8-bit Y plane followed by interleaved Cb/Cr.

    ``range`` selects code-value mapping, ``siting`` selects the 4:2:0 chroma
    phase, and ``interpolation`` selects the chroma downsampling filter.
    Packing is enqueued on the current CuPy stream without host synchronization;
    consume on that stream or pass its handle/event to the encoder.
    """
    _validate_frame(frame, operation="to_nv12")
    _dimensions(frame.width, frame.height, even_width=True, even_height=True)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    siting = _token(siting, axis="siting", accepted=_SITING_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_TO_INTERPOLATION_TOKENS)
    return _to_subsampled(
        frame,
        kernel=_to_kernel(8, interpolation, siting),
        layout="nv12",
        bit_depth=8,
        range=range,
    )
