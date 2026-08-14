"""Image header probing across supported formats."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    _COLORSPACE_TOKENS,
)
from pixtreme._io.common import (
    _colorspace_chromaticities,
    _empty_color,
    _path_and_format,
)
from pixtreme._io.formats.bmp import _parse_bmp
from pixtreme._io.formats.dpx import (
    _parse_dpx,
)
from pixtreme._io.formats.exr.container import (
    _ExrContainer,
    _parse_exr_container,
)
from pixtreme._io.formats.hdr import (
    _parse_hdr,
)
from pixtreme._io.formats.jpeg import _parse_jpeg
from pixtreme._io.formats.jpeg2000 import _parse_jpeg2000
from pixtreme._io.formats.png import _parse_png
from pixtreme._io.formats.pnm import _parse_pnm
from pixtreme._io.formats.tga import (
    _parse_tga,
)
from pixtreme._io.formats.tiff import _parse_tiff
from pixtreme._io.formats.webp import _parse_webp
from pixtreme._io.models import (
    ImageHeader,
    _ImageColorInfo,
    _ImagePart,
)


def _map_exr_color(raw: dict[str, object]) -> _ImageColorInfo:
    if raw.get("acesImageContainerFlag"):
        return _ImageColorInfo(raw=raw, colorspace="ACES2065-1", gamma="linear", mappable=True)
    chromaticities = raw.get("chromaticities")
    if chromaticities is None:
        return _empty_color()
    values = np.asarray(chromaticities, dtype=np.float64)
    for token in _COLORSPACE_TOKENS:
        if np.allclose(values, _colorspace_chromaticities(token), rtol=0.0, atol=5e-5):
            mapped = "Rec.709" if token == "sRGB" else token
            return _ImageColorInfo(raw=raw, colorspace=mapped, gamma=None, mappable=True)
    return _ImageColorInfo(raw=raw, colorspace=None, gamma=None, mappable=False)


def _parse_exr(path: Path) -> _ExrContainer:
    return _parse_exr_container(path)


def _exr_header(container: _ExrContainer) -> ImageHeader:
    raw_color: dict[str, object] = {}
    first_attributes = container.parts[0].attributes
    chromaticities = first_attributes.get("chromaticities")
    if chromaticities is not None and len(chromaticities.payload) == 32:
        raw_color["chromaticities"] = struct.unpack("<8f", chromaticities.payload)
    aces_flag = first_attributes.get("acesImageContainerFlag")
    if aces_flag is not None and len(aces_flag.payload) == 4:
        raw_color["acesImageContainerFlag"] = bool(struct.unpack("<i", aces_flag.payload)[0])
    x_min, y_min, x_max, y_max = container.data_window
    return ImageHeader(
        format="EXR",
        width=x_max - x_min + 1,
        height=y_max - y_min + 1,
        parts=tuple(
            _ImagePart(
                name=part.name,
                channels={channel.name: channel.dtype for channel in part.channels},
                deep=part.deep,
            )
            for part in container.parts
        ),
        color=_map_exr_color(raw_color),
    )


def read_header(path: str | os.PathLike[str]) -> ImageHeader:
    """Inspect image structure and color attributes without decoding pixels.

    ``path`` is selected case-insensitively by its supported raster, EXR, HDR, or DPX
    extension. The returned :class:`ImageHeader` reports format, dimensions,
    part names, channel storage dtypes, per-part deep state, and raw plus vocabulary-mapped color
    metadata. EXR dimensions come from the first part's data window.
    HDR ``EXPOSURE``, ``PRIMARIES``, and ``COLORCORR`` assignments are exposed
    as raw attributes only; pixel reads do not apply their values.
    DPX reports its native bit depth, byte order, packing, and mapped transfer.

    Header probing imports no pixel codec and creates no CUDA context or GPU
    allocation. A missing file raises :class:`FileNotFoundError`, an unsupported
    extension raises :class:`ValueError`, and malformed or unreadable header data
    raises :class:`RuntimeError`.
    """
    file_path, format_name = _path_and_format(path, require_exists=True)
    if format_name == "EXR":
        try:
            return _exr_header(_parse_exr(file_path))
        except (OSError, ValueError, struct.error) as error:
            raise RuntimeError(
                _actionable_error(
                    why=f"the {format_name} header could not be parsed: {error}",
                    what=str(file_path),
                    how="verify that the file is a valid, supported image",
                )
            ) from error
    parser = {
        "PNG": _parse_png,
        "JPEG": _parse_jpeg,
        "TIFF": _parse_tiff,
        "JPEG2000": _parse_jpeg2000,
        "WEBP": _parse_webp,
        "BMP": _parse_bmp,
        "PNM": _parse_pnm,
        "TGA": _parse_tga,
        "HDR": _parse_hdr,
        "DPX": _parse_dpx,
    }[format_name]
    try:
        return parser(file_path)
    except (OSError, ValueError, struct.error) as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the {format_name} header could not be parsed: {error}",
                what=str(file_path),
                how="verify that the file is a valid, supported image",
            )
        ) from error


def _sniff_raster_format(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise ValueError(
            _actionable_error(
                why="encoded image data must be bytes",
                what=type(data).__name__,
                how="pass a supported encoded raster payload as bytes",
            )
        )
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "TIFF"
    if data.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n") or data.startswith(b"\xff\x4f\xff\x51"):
        return "JPEG2000"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "WEBP"
    if data.startswith(b"BM"):
        return "BMP"
    if len(data) >= 2 and data[:2] in (b"P2", b"P3", b"P5", b"P6"):
        return "PNM"
    raise ValueError(
        _actionable_error(
            why="the encoded image format is not supported",
            what="unrecognized encoded data",
            how="pass encoded JPEG, PNG, TIFF, JPEG 2000, WebP, BMP, or PNM bytes",
        )
    )


def _read_raster_header(data: bytes, format_name: str) -> ImageHeader:
    parser = {
        "PNG": _parse_png,
        "JPEG": _parse_jpeg,
        "TIFF": _parse_tiff,
        "JPEG2000": _parse_jpeg2000,
        "WEBP": _parse_webp,
        "BMP": _parse_bmp,
        "PNM": _parse_pnm,
    }[format_name]
    try:
        return parser(data)
    except (OSError, ValueError, struct.error) as error:
        raise ValueError(
            _actionable_error(
                why=f"the {format_name} header could not be parsed: {error}",
                what=f"{len(data)} encoded bytes",
                how="pass a complete, valid encoded image payload",
            )
        ) from error
