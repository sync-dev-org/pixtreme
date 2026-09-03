"""yuv444p in-memory wire import and export."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import _RANGE_TOKENS
from pixtreme._core.vocabulary import Colorspace, Gamma, Matrix, Range
from pixtreme._io.wire.sampling import (
    _bit_depth,
    _dimensions,
    _from_planar_444,
    _matrix,
    _metadata,
    _planar_444_kernel_source,
    _to_planar_444,
    _to_planar_444_kernel_source,
    _token,
    _validate_buffer,
    _validate_frame,
)


@lru_cache(maxsize=None)
def _from_kernel(bit_depth: int) -> cp.RawKernel:
    source = _planar_444_kernel_source(bit_depth, alpha=False)
    return cp.RawKernel(source, "pixtreme_from_planar_444")


@lru_cache(maxsize=None)
def _to_kernel() -> cp.RawKernel:
    source = _to_planar_444_kernel_source(alpha=False)
    return cp.RawKernel(source, "pixtreme_to_planar_444")


def from_yuv444p(
    buf: cp.ndarray,
    *,
    width: int,
    height: int,
    bit_depth: int = 10,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    matrix: Matrix | None = None,
    range: Range = "legal",
) -> Frame:
    """Construct a full-range fp32 YCbCr444 Frame from planar YUV444.

    The C-contiguous 1D plane order is ``Y, Cb, Cr``. ``bit_depth`` accepts 10
    or 12, stored lower-aligned in uint16. No chroma resampling is performed.
    ``range`` expands H.273 legal positions without clipping or maps the full
    code domain.
    ``colorspace`` / ``gamma`` override placeholders and ``matrix`` stamps basis provenance only.
    """
    colorspace, gamma = _metadata(colorspace, gamma)
    matrix = _matrix(matrix)
    width, height = _dimensions(width, height, even_width=False, even_height=False)
    bit_depth = _bit_depth(bit_depth, operation="from_yuv444p")
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    pixel_count = width * height
    element_count = pixel_count * 3
    input_data = _validate_buffer(
        buf,
        operation="from_yuv444p",
        dtype=np.dtype(np.uint16),
        element_count=element_count,
        shapes=((element_count,),),
    )
    return _from_planar_444(
        input_data,
        kernel=_from_kernel(bit_depth),
        width=width,
        height=height,
        bit_depth=bit_depth,
        range=range,
        alpha=False,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_yuv444p(
    frame: Frame,
    *,
    bit_depth: int = 10,
    range: Range = "legal",
) -> cp.ndarray:
    """Pack ``frame`` as planar Y, Cb, Cr 4:4:4 samples.

    Parameters
    ----------
    frame:
        A float32 Frame with exact ``("Y", "Cb", "Cr")`` channels and positive
        geometry. No chroma resampling is performed.
    bit_depth:
        Stored code depth, either 10 (default) or 12. Codes are lower-aligned in
        a uint16 container.
    range:
        ``"full"`` or ``"legal"`` YCbCr code mapping. Values are rounded half
        away from zero and clipped only to the physical container domain, not
        the legal interval.

    Returns
    -------
    cupy.ndarray
        A new privately owned C-contiguous 1D uint16 array with shape
        ``(H * W * 3,)`` and plane order Y, Cb, Cr.

    Raises
    ------
    ValueError
        If ``frame`` has the wrong channels or dtype, geometry is invalid, or
        ``bit_depth`` or ``range`` is outside its closed domain.
    """
    _validate_frame(frame, operation="to_yuv444p")
    _dimensions(frame.width, frame.height, even_width=False, even_height=False)
    bit_depth = _bit_depth(bit_depth, operation="to_yuv444p")
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    return _to_planar_444(frame, kernel=_to_kernel(), bit_depth=bit_depth, range=range, alpha=False)
