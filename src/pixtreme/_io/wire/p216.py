"""P216 in-memory wire import and export."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import _RANGE_TOKENS
from pixtreme._core.vocabulary import Colorspace, Gamma, Interpolation, Matrix, Range
from pixtreme._io.wire.sampling import (
    _INTERPOLATION_TOKENS,
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
def _from_kernel(interpolation: str) -> cp.RawKernel:
    source = _subsampled_kernel_source("p216", 16, interpolation, "topleft")
    return cp.RawKernel(source, "pixtreme_from_subsampled")


@lru_cache(maxsize=None)
def _to_kernel(interpolation: str) -> cp.RawKernel:
    source = _to_subsampled_kernel_source("p216", 16, interpolation, "topleft")
    return cp.RawKernel(source, "pixtreme_to_subsampled")


def from_p216(
    buf: cp.ndarray,
    *,
    width: int,
    height: int,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    matrix: Matrix | None = None,
    range: Range = "legal",
    interpolation: Interpolation = "bilinear",
) -> Frame:
    """Construct a C-contiguous float32 YCbCr444 Frame from P216.

    The input is a C-contiguous uint16 array with shape ``(2 * width * height,)``:
    one raster-order Y plane followed by a same-height plane that interleaves
    ``Cb, Cr`` pairs. Every uint16 word has 16 effective bits. Width must be even;
    chroma is horizontally co-sited and has vertical full resolution (the same row,
    with no vertical filtering).

    ``range="legal"`` decodes luma with extent 56064 and offset 4096 and chroma
    with extent 57344 and center 32768. ``range="full"`` uses extent 65535.
    The inverse export rounds half away from zero and clips only to the uint16
    container 0..65535, so legal headroom is retained. ``interpolation`` accepts
    ``nearest``, ``bilinear``, ``bicubic``, ``b-spline``, ``mitchell``,
    ``lanczos2``, ``lanczos3``, and ``lanczos4``; the default is ``bilinear``.

    Constant chroma round-trips bit-exactly across every supported filter pair,
    and a P216-origin Frame round-trips bit-exactly with ``nearest``. General
    4:4:4 subsampling is lossy and has no round-trip identity guarantee.
    ``colorspace``, ``gamma``, and ``matrix`` only stamp metadata.

    The kernel is enqueued asynchronously on the current CuPy stream with no host
    synchronization. The producer must order writes before the call, and a
    consumer on another stream must wait for completion. Keep the input buffer
    alive until the kernel completes. The returned Frame owns a new independent
    allocation.
    """
    colorspace, gamma = _metadata(colorspace, gamma)
    matrix = _matrix(matrix)
    width, height = _dimensions(width, height, even_width=True, even_height=False)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_INTERPOLATION_TOKENS)
    element_count = 2 * width * height
    input_data = _validate_buffer(
        buf,
        operation="from_p216",
        dtype=np.dtype(np.uint16),
        element_count=element_count,
        shapes=((element_count,),),
    )
    return _from_subsampled(
        input_data,
        kernel=_from_kernel(interpolation),
        layout="p216",
        width=width,
        height=height,
        bit_depth=16,
        range=range,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_p216(
    frame: Frame,
    *,
    range: Range = "legal",
    interpolation: Interpolation = "area",
) -> cp.ndarray:
    """Pack a float32 YCbCr444 Frame into C-contiguous uint16 P216.

    The returned shape is ``(2 * width * height,)``: one raster-order Y plane
    followed by a same-height plane that interleaves ``Cb, Cr`` pairs. Every
    uint16 word has 16 effective bits. Width must be even; chroma is horizontally
    co-sited and has vertical full resolution (the same row, with no vertical
    filtering).

    ``range="legal"`` maps luma with extent 56064 and offset 4096 and chroma
    with extent 57344 and center 32768. ``range="full"`` uses extent 65535.
    Codes are rounded half away from zero and clip only to the uint16 container
    0..65535, retaining legal headroom. ``interpolation`` accepts ``nearest``,
    ``bilinear``, ``bicubic``, and ``area``; the default is ``area``.

    Constant chroma round-trips bit-exactly across every supported filter pair,
    and a P216-origin Frame round-trips bit-exactly with ``nearest``. General
    4:4:4 subsampling is lossy and has no round-trip identity guarantee.
    The Frame's ``colorspace``, ``gamma``, and ``matrix`` metadata are unchanged
    and are not encoded into P216.

    The kernel is enqueued asynchronously on the current CuPy stream with no host
    synchronization. The producer must order Frame writes before the call, and a
    consumer on another stream must wait for completion. Keep the input Frame
    alive until the kernel completes. The returned array owns a new independent
    allocation on every call.
    """
    _validate_frame(frame, operation="to_p216")
    _dimensions(frame.width, frame.height, even_width=True, even_height=False)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_TO_INTERPOLATION_TOKENS)
    return _to_subsampled(
        frame,
        kernel=_to_kernel(interpolation),
        layout="p216",
        bit_depth=16,
        range=range,
    )
