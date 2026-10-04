"""P210 in-memory wire import and export."""

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
    source = _subsampled_kernel_source("p210", 10, interpolation, "topleft")
    return cp.RawKernel(source, "pixtreme_from_subsampled")


@lru_cache(maxsize=None)
def _to_kernel(interpolation: str) -> cp.RawKernel:
    source = _to_subsampled_kernel_source("p210", 10, interpolation, "topleft")
    return cp.RawKernel(source, "pixtreme_to_subsampled")


def from_p210(
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
    """Construct a C-contiguous float32 YCbCr444 Frame from P210.

    The input is a C-contiguous uint16 array with shape ``(2 * width * height,)``:
    one raster-order Y plane followed by a same-height plane that interleaves
    ``Cb, Cr`` pairs. Every uint16 word has 10 effective code bits MSB-aligned
    in the upper 10 bits; the lower 6 padding bits are ignored. Width must be even;
    chroma is horizontally co-sited at (2k, y) and has vertical full resolution
    (the same row, with no vertical filtering).

    ``range="legal"`` decodes luma with extent 876 and offset 64 and chroma
    with extent 896 and center 512. ``range="full"`` uses extent 1023.
    The inverse export rounds half away from zero and clips only to code
    [0, 1023], so legal headroom is retained. ``interpolation`` accepts
    ``nearest``, ``bilinear``, ``bicubic``, ``b-spline``, ``mitchell``,
    ``lanczos2``, ``lanczos3``, and ``lanczos4``; the default is ``bilinear``.

    Constant chroma round-trips active codes bit-exactly across every supported
    filter pair, and a P210-origin Frame round-trips active codes bit-exactly
    with ``nearest``. Re-export sets padding to zero. General 4:4:4 subsampling
    is lossy and has no round-trip identity guarantee. ``colorspace``, ``gamma``,
    and ``matrix`` only stamp canonical metadata; omitted or None values stamp
    Rec.709, Rec.709, and None respectively.

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
        operation="from_p210",
        dtype=np.dtype(np.uint16),
        element_count=element_count,
        shapes=((element_count,),),
    )
    return _from_subsampled(
        input_data,
        kernel=_from_kernel(interpolation),
        layout="p210",
        width=width,
        height=height,
        bit_depth=10,
        range=range,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_p210(
    frame: Frame,
    *,
    range: Range = "legal",
    interpolation: Interpolation = "area",
) -> cp.ndarray:
    """Pack a float32 YCbCr444 Frame into C-contiguous uint16 P210.

    The returned shape is ``(2 * width * height,)``: one raster-order Y plane
    followed by a same-height plane that interleaves ``Cb, Cr`` pairs. Every
    uint16 word carries 10 effective code bits MSB-aligned in the upper 10 bits;
    the lower 6 padding bits are zero. Width must be even; chroma is horizontally
    co-sited at (2k, y) and has vertical full resolution (the same row, with no
    vertical filtering).

    ``range="legal"`` maps luma with extent 876 and offset 64 and chroma
    with extent 896 and center 512. ``range="full"`` uses extent 1023.
    Codes are rounded half away from zero and clip only to code [0, 1023],
    retaining legal headroom. ``interpolation`` accepts ``nearest``,
    ``bilinear``, ``bicubic``, and ``area``; the default is ``area``.

    Constant chroma round-trips active codes bit-exactly across every supported
    filter pair, and a P210-origin Frame round-trips active codes bit-exactly
    with ``nearest``. General 4:4:4 subsampling is lossy and has no round-trip
    identity guarantee.
    The Frame's ``colorspace``, ``gamma``, and ``matrix`` metadata are unchanged
    and are not encoded into P210.

    The kernel is enqueued asynchronously on the current CuPy stream with no host
    synchronization. The producer must order Frame writes before the call, and a
    consumer on another stream must wait for completion. Keep the input Frame
    alive until the kernel completes. The returned array owns a new independent
    allocation on every call.
    """
    _validate_frame(frame, operation="to_p210")
    _dimensions(frame.width, frame.height, even_width=True, even_height=False)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_TO_INTERPOLATION_TOKENS)
    return _to_subsampled(
        frame,
        kernel=_to_kernel(interpolation),
        layout="p210",
        bit_depth=10,
        range=range,
    )
