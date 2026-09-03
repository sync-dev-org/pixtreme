"""v210 in-memory wire import and export."""

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
def _from_kernel(bit_depth: int, interpolation: str, siting: str) -> cp.RawKernel:
    source = _subsampled_kernel_source("v210", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_from_subsampled")


@lru_cache(maxsize=None)
def _to_kernel(bit_depth: int, interpolation: str, siting: str) -> cp.RawKernel:
    source = _to_subsampled_kernel_source("v210", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_to_subsampled")


def from_v210(
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
    """Construct a full-range fp32 YCbCr444 Frame from v210 uint32 words.

    Each four-word group stores six horizontally co-sited 10-bit pixels. Every
    row occupies a 128-byte aligned word span (48-pixel storage units), including
    the zero-filled padding convention. The input is one C-contiguous 1D uint32
    buffer. ``interpolation`` upsamples horizontal chroma, and ``range`` expands
    H.273 legal code positions without clipping or maps the full 10-bit domain.
    ``colorspace`` / ``gamma`` override placeholders and ``matrix`` stamps basis provenance only.
    """
    colorspace, gamma = _metadata(colorspace, gamma)
    matrix = _matrix(matrix)
    width, height = _dimensions(width, height, even_width=False, even_height=False)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_INTERPOLATION_TOKENS)
    row_words = ((width + 47) // 48) * 32
    input_data = _validate_buffer(
        buf,
        operation="from_v210",
        dtype=np.dtype(np.uint32),
        element_count=row_words * height,
        shapes=((row_words * height,),),
    )
    return _from_subsampled(
        input_data,
        kernel=_from_kernel(10, interpolation, "topleft"),
        layout="v210",
        width=width,
        height=height,
        bit_depth=10,
        range=range,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
        row_words=row_words,
    )


def to_v210(
    frame: Frame,
    *,
    range: Range = "legal",
    interpolation: Interpolation = "area",
) -> cp.ndarray:
    """Pack ``frame`` into little-endian v210 uint32 words.

    Each scanline is aligned to a 128-byte boundary and all padding words are
    zero. ``range`` controls code-value mapping and ``interpolation`` controls
    horizontal chroma downsampling.
    """
    _validate_frame(frame, operation="to_v210")
    _dimensions(frame.width, frame.height, even_width=False, even_height=False)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_TO_INTERPOLATION_TOKENS)
    return _to_subsampled(
        frame,
        kernel=_to_kernel(10, interpolation, "topleft"),
        layout="v210",
        bit_depth=10,
        range=range,
    )
