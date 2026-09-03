"""uyvy422 in-memory wire import and export."""

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
    source = _subsampled_kernel_source("uyvy422", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_from_subsampled")


@lru_cache(maxsize=None)
def _to_kernel(bit_depth: int, interpolation: str, siting: str) -> cp.RawKernel:
    source = _to_subsampled_kernel_source("uyvy422", bit_depth, interpolation, siting)
    return cp.RawKernel(source, "pixtreme_to_subsampled")


def from_uyvy422(
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
    """Construct a full-range fp32 YCbCr444 Frame from uint8 UYVY422.

    The packed order is ``U0 Y0 V0 Y1``. A C-contiguous 1D buffer or NDI-style
    ``(H, W, 2)`` view is accepted; reshape the buffer without a copy when
    moving between those forms. Horizontal chroma is co-sited at even luma
    samples and ``interpolation`` selects one of the eight point filters.
    ``range`` expands H.273 legal code positions without clipping, or maps the
    full uint8 container when set to ``"full"``. ``colorspace`` / ``gamma``
    override placeholders and ``matrix`` stamps basis provenance without
    changing pixel values.
    """
    colorspace, gamma = _metadata(colorspace, gamma)
    matrix = _matrix(matrix)
    width, height = _dimensions(width, height, even_width=True, even_height=False)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_INTERPOLATION_TOKENS)
    pixel_count = width * height
    input_data = _validate_buffer(
        buf,
        operation="from_uyvy422",
        dtype=np.dtype(np.uint8),
        element_count=pixel_count * 2,
        shapes=((pixel_count * 2,), (height, width, 2)),
    )
    return _from_subsampled(
        input_data,
        kernel=_from_kernel(8, interpolation, "topleft"),
        layout="uyvy422",
        width=width,
        height=height,
        bit_depth=8,
        range=range,
        colorspace=colorspace,
        gamma=gamma,
        matrix=matrix,
    )


def to_uyvy422(
    frame: Frame,
    *,
    range: Range = "legal",
    interpolation: Interpolation = "area",
) -> cp.ndarray:
    """Pack ``frame`` as U0 Y0 V0 Y1 bytes in a private 1D array.

    The output is the flattened form of an ``(H, W, 2)`` uint8 image. ``range``
    maps normalized YCbCr to full- or legal-range codes, while ``interpolation``
    selects the horizontal chroma downsampling filter.
    """
    _validate_frame(frame, operation="to_uyvy422")
    _dimensions(frame.width, frame.height, even_width=True, even_height=False)
    range = _token(range, axis="range", accepted=_RANGE_TOKENS)
    interpolation = _token(interpolation, axis="interpolation", accepted=_TO_INTERPOLATION_TOKENS)
    return _to_subsampled(
        frame,
        kernel=_to_kernel(8, interpolation, "topleft"),
        layout="uyvy422",
        bit_depth=8,
        range=range,
    )
