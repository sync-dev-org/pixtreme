"""yuv422p in-memory wire import and export."""

from __future__ import annotations

from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._core.frame import Frame
from pixtreme._core.value_domain import _RANGE_TOKENS
from pixtreme._io.wire.sampling import (
    _INTERPOLATION_TOKENS,
    _TO_INTERPOLATION_TOKENS,
    _bit_depth,
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
    source = _subsampled_kernel_source("yuv422p", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_from_subsampled")


@lru_cache(maxsize=None)
def _to_kernel(bit_depth: int, interpolation: str, siting: str) -> cp.RawKernel:
    source = _to_subsampled_kernel_source("yuv422p", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_to_subsampled")


def from_yuv422p(
    buf: cp.ndarray,
    *,
    width: int,
    height: int,
    bit_depth: int = 8,
    colorspace: str | None = None,
    gamma: str | None = None,
    matrix: str | None = None,
    range: str = "legal",
    interpolation: str = "bilinear",
) -> Frame:
    """Construct a full-range fp32 YCbCr444 Frame from planar YUV422.

    The C-contiguous 1D plane order is ``Y, Cb, Cr``. ``bit_depth`` accepts 8
    (uint8), 10, or 12 (lower-aligned uint16). Chroma is horizontally co-sited
    and vertically full; ``interpolation`` applies only on the horizontal axis.
    ``range`` expands H.273 legal positions without clipping or maps the full
    code domain.
    ``colorspace`` / ``gamma`` override placeholders and ``matrix`` stamps basis provenance only.
    """
    colorspace, gamma = _metadata(colorspace, gamma)
    matrix = _matrix(matrix)
    width, height = _dimensions(width, height, even_width=True, even_height=False)
    bit_depth = _bit_depth(bit_depth, operation="from_yuv422p")
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_INTERPOLATION_TOKENS)
    pixel_count = width * height
    element_count = pixel_count * 2
    dtype = np.dtype(np.uint8 if bit_depth == 8 else np.uint16)
    input_data = _validate_buffer(
        buf,
        operation="from_yuv422p",
        dtype=dtype,
        element_count=element_count,
        shapes=((element_count,),),
    )
    return _from_subsampled(
        input_data,
        kernel=_from_kernel(bit_depth, interpolation, "topleft"),
        layout="yuv422p",
        width=width,
        height=height,
        bit_depth=bit_depth,
        range=range,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_yuv422p(
    frame: Frame,
    *,
    bit_depth: int = 8,
    range: str = "legal",
    interpolation: str = "area",
) -> cp.ndarray:
    """Pack ``frame`` as contiguous planar Y, Cb, Cr 4:2:2 samples.

    ``bit_depth`` chooses the code container, ``range`` chooses code-value
    mapping, and ``interpolation`` selects horizontal chroma downsampling.
    """
    _validate_frame(frame, operation="to_yuv422p")
    _dimensions(frame.width, frame.height, even_width=True, even_height=False)
    bit_depth = _bit_depth(bit_depth, operation="to_yuv422p")
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_TO_INTERPOLATION_TOKENS)
    return _to_subsampled(
        frame,
        kernel=_to_kernel(bit_depth, interpolation, "topleft"),
        layout="yuv422p",
        bit_depth=bit_depth,
        range=range,
    )
