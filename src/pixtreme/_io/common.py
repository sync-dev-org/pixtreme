"""Shared image I/O validation and metadata helpers."""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np

from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    _COLORSPACE_TOKENS,
    _GAMMA_TOKENS,
    ChannelInput,
    _normalize_channels,
    _validate_token,
)
from pixtreme._core.vocabulary import _EXR_COMPRESSION_TOKENS as _EXR_COMPRESSION_TOKENS
from pixtreme._core.vocabulary import _IMAGE_FORMAT_TOKENS
from pixtreme._core.vocabulary import _TIFF_COMPRESSION_TOKENS as _TIFF_COMPRESSION_TOKENS
from pixtreme._io.models import (
    ImageHeader,
    _ImageColorInfo,
)

_SUPPORTED_EXTENSIONS: Mapping[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".exr": "EXR",
    ".jp2": "JPEG2000",
    ".j2k": "JPEG2000",
    ".j2c": "JPEG2000",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".pnm": "PNM",
    ".ppm": "PNM",
    ".pgm": "PNM",
    ".tga": "TGA",
    ".hdr": "HDR",
    ".dpx": "DPX",
}
_RASTER_DEFAULTS = ("sRGB", "srgb")
_EXR_DEFAULTS = ("ACES2065-1", "linear")
_HDR_DEFAULTS = ("Rec.709", "linear")
_DPX_DEFAULTS = ("Rec.709", "linear")
_DTYPE_MAXIMA = {"uint8": np.float32(255.0), "uint16": np.float32(65535.0)}
_ENCODE_FORMAT_TOKENS = _IMAGE_FORMAT_TOKENS
_EXR_DWA_COMPRESSION_TOKENS = _EXR_COMPRESSION_TOKENS[-2:]
_ENCODE_FORMAT_NAMES = {
    "jpeg": "JPEG",
    "png": "PNG",
    "tiff": "TIFF",
    "jpeg2000": "JPEG2000",
    "webp": "WEBP",
    "bmp": "BMP",
    "pnm": "PNM",
}
_ENCODE_CODEC_EXTENSIONS = {
    "JPEG": ".jpeg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "JPEG2000": ".jp2",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "PNM": ".pnm",
}


def _path_and_format(path: str | os.PathLike[str], *, require_exists: bool) -> tuple[Path, str]:
    file_path = Path(path)
    if require_exists and not file_path.is_file():
        raise FileNotFoundError(
            _actionable_error(
                why="the image file does not exist",
                what=str(file_path),
                how="provide an existing PNG, JPEG, TIFF, EXR, JPEG 2000, WebP, BMP, PNM, TGA, HDR, or DPX path",
            )
        )
    format_name = _SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())
    if format_name is None:
        raise ValueError(
            _actionable_error(
                why="the path extension is not a supported image format",
                what=file_path.suffix or "<no extension>",
                how="use a supported PNG, JPEG, TIFF, EXR, JPEG 2000, WebP, BMP, PNM, TGA, HDR, or DPX extension",
            )
        )
    return file_path, format_name


def _empty_color() -> _ImageColorInfo:
    return _ImageColorInfo(raw={}, colorspace=None, gamma=None, mappable=None)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(
            _actionable_error(
                why="the image payload ended before the requested field was complete",
                what=f"requested={size} bytes, received={len(value)} bytes",
                how="pass a complete, valid encoded image file or payload",
            )
        )
    return value


@contextmanager
def _binary_stream(source: Path | bytes) -> Iterator[BinaryIO]:
    if isinstance(source, Path):
        with source.open("rb") as stream:
            yield stream
    else:
        with BytesIO(source) as stream:
            yield stream


def _colorspace_chromaticities(colorspace: str) -> tuple[float, ...]:
    primaries, white = _COLORSPACE_DEFINITIONS[colorspace]
    return tuple(value for point in (*primaries, white) for value in point)


def _resolve_metadata(
    header: ImageHeader,
    *,
    colorspace: str | None,
    gamma: str | None,
) -> tuple[str, str]:
    if colorspace is not None:
        colorspace = _validate_token(colorspace, axis="colorspace", accepted=_COLORSPACE_TOKENS)
    if gamma is not None:
        gamma = _validate_token(gamma, axis="gamma", accepted=_GAMMA_TOKENS)
    if header.format == "EXR":
        defaults = _EXR_DEFAULTS
    elif header.format == "HDR":
        defaults = _HDR_DEFAULTS
    elif header.format == "DPX":
        defaults = _DPX_DEFAULTS
    else:
        defaults = _RASTER_DEFAULTS
    if header.color.mappable is False:
        warnings.warn(
            "file color metadata cannot be mapped to pixtreme vocabulary; using specification defaults",
            UserWarning,
            stacklevel=3,
        )
    return colorspace or header.color.colorspace or defaults[0], gamma or header.color.gamma or defaults[1]


def _resolve_channel_locations(
    header: ImageHeader,
    requested: ChannelInput | None,
) -> list[tuple[int, str, str]]:
    naked: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    flat_naked: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    qualified: dict[str, tuple[int, str]] = {}
    for part_index, part in enumerate(header.parts):
        for channel_name in part.channels:
            naked[channel_name].append((part_index, channel_name))
            if not part.deep:
                flat_naked[channel_name].append((part_index, channel_name))
            if part.name:
                qualified[f"{part.name}.{channel_name}"] = (part_index, channel_name)

    def resolve(
        label: str,
        candidates: defaultdict[str, list[tuple[int, str]]],
    ) -> tuple[int, str, str]:
        matches = candidates.get(label, [])
        if len(matches) == 1:
            part_index, channel_name = matches[0]
            return part_index, channel_name, label
        if len(matches) > 1:
            examples = tuple(f"{header.parts[part_index].name}.{channel_name}" for part_index, channel_name in matches)
            raise ValueError(
                _actionable_error(
                    why=f"channel {label!r} is ambiguous across EXR parts; use a qualified name such as {examples[0]!r}",
                    what=label,
                    how=f"choose one of {examples!r}",
                )
            )
        location = qualified.get(label)
        if location is None:
            raise ValueError(
                _actionable_error(
                    why="the requested channel is absent from the image",
                    what=label,
                    how="inspect available channels with px.io.read_header(path)",
                )
            )
        return location[0], location[1], label

    if requested is not None:
        return [resolve(label, naked) for label in _normalize_channels(requested)]
    if header.format == "EXR":
        locations = [resolve(label, flat_naked) for label in ("R", "G", "B")]
        if "A" in flat_naked:
            locations.append(resolve("A", flat_naked))
        return locations
    labels = tuple(header.parts[0].channels)
    return [(0, label, label) for label in labels]
