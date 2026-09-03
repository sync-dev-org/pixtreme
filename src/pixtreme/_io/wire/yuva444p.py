"""yuva444p in-memory wire import and export."""

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
    source = _planar_444_kernel_source(bit_depth, alpha=True)
    return cp.RawKernel(source, "pixtreme_from_planar_444")


@lru_cache(maxsize=None)
def _to_kernel() -> cp.RawKernel:
    source = _to_planar_444_kernel_source(alpha=True)
    return cp.RawKernel(source, "pixtreme_to_planar_444")


def from_yuva444p(
    buf: cp.ndarray,
    *,
    width: int,
    height: int,
    bit_depth: int = 12,
    colorspace: Colorspace | None = None,
    gamma: Gamma | None = None,
    matrix: Matrix | None = None,
    range: Range = "legal",
) -> Frame:
    """Construct a full-range fp32 YCbCrA4444 Frame from planar YUVA444.

    The C-contiguous 1D plane order is ``Y, Cb, Cr, A``. ``bit_depth`` is 12,
    stored lower-aligned in uint16. YCbCr follows the selected ``range`` without
    clipping, while A is always decoded against full scale ``0...4095``.
    No chroma resampling is performed.
    ``colorspace`` / ``gamma`` override placeholders and ``matrix`` stamps basis provenance only.
    """
    colorspace, gamma = _metadata(colorspace, gamma)
    matrix = _matrix(matrix)
    width, height = _dimensions(width, height, even_width=False, even_height=False)
    bit_depth = _bit_depth(bit_depth, operation="from_yuva444p")
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    pixel_count = width * height
    element_count = pixel_count * 4
    input_data = _validate_buffer(
        buf,
        operation="from_yuva444p",
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
        alpha=True,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_yuva444p(
    frame: Frame,
    *,
    bit_depth: int = 12,
    range: Range = "legal",
) -> cp.ndarray:
    """Pack ``frame`` as planar Y, Cb, Cr, A 4:4:4:4 samples.

    Parameters
    ----------
    frame:
        A float32 Frame with exact ``("Y", "Cb", "Cr", "A")`` channels and
        positive geometry. No chroma resampling is performed.
    bit_depth:
        Stored code depth, fixed at 12. Codes are lower-aligned in a uint16
        container.
    range:
        ``"full"`` or ``"legal"`` YCbCr code mapping. Alpha always maps over
        full scale. Values are rounded half away from zero and clipped only to
        the physical container domain, not the legal interval.

    Returns
    -------
    cupy.ndarray
        A new privately owned C-contiguous 1D uint16 array with shape
        ``(H * W * 4,)`` and plane order Y, Cb, Cr, A.

    Raises
    ------
    ValueError
        If ``frame`` has the wrong channels or dtype, geometry is invalid, or
        ``bit_depth`` or ``range`` is outside its closed domain.
    """
    _validate_frame(frame, operation="to_yuva444p", alpha=True)
    _dimensions(frame.width, frame.height, even_width=False, even_height=False)
    bit_depth = _bit_depth(bit_depth, operation="to_yuva444p")
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    return _to_planar_444(frame, kernel=_to_kernel(), bit_depth=bit_depth, range=range, alpha=True)
