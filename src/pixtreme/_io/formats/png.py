"""PNG header parsing."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import cast

from pixtreme._core.errors import _actionable_error
from pixtreme._io.common import _binary_stream, _empty_color, _read_exact
from pixtreme._io.models import ImageHeader, _ImageColorInfo, _ImagePart

_CICP_COLORSPACES = {1: "Rec.709", 9: "Rec.2020"}
_CICP_GAMMAS = {1: "rec709", 6: "rec709", 13: "srgb", 16: "pq", 18: "hlg"}


def _png_color_info(raw: dict[str, object]) -> _ImageColorInfo:
    if "cICP" in raw:
        primary, transfer, matrix, full_range = cast(tuple[int, int, int, int], raw["cICP"])
        colorspace = _CICP_COLORSPACES.get(primary)
        mapped_gamma = _CICP_GAMMAS.get(transfer)
        mappable = colorspace is not None and mapped_gamma is not None and matrix == 0 and full_range in (0, 1)
        return _ImageColorInfo(raw=raw, colorspace=colorspace, gamma=mapped_gamma, mappable=mappable)
    if "sRGB" in raw:
        return _ImageColorInfo(raw=raw, colorspace="sRGB", gamma="srgb", mappable=True)
    if "gAMA" in raw:
        gamma_value = cast(int, raw["gAMA"])
        gamma: str | None
        if abs(gamma_value - 100000) <= 1:
            gamma = "linear"
        elif abs(gamma_value - 45455) <= 1:
            gamma = "2.2"
        elif abs(gamma_value - 41667) <= 1:
            gamma = "2.4"
        else:
            gamma = None
        return _ImageColorInfo(raw=raw, colorspace=None, gamma=gamma, mappable=gamma is not None)
    return _empty_color()


def _parse_png(source: Path | bytes) -> ImageHeader:
    raw_color: dict[str, object] = {}
    transparent = False
    with _binary_stream(source) as stream:
        signature = _read_exact(stream, 8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(
                _actionable_error(
                    why="the image does not have a valid PNG signature",
                    what=f"signature={signature!r}",
                    how="pass a file beginning with the PNG signature b'\\x89PNG\\r\\n\\x1a\\n'",
                )
            )
        width = height = bit_depth = color_type = 0
        while True:
            length = struct.unpack(">I", _read_exact(stream, 4))[0]
            chunk_type = _read_exact(stream, 4)
            payload = _read_exact(stream, length)
            _read_exact(stream, 4)
            if chunk_type == b"IHDR":
                if len(payload) != 13:
                    raise ValueError(
                        _actionable_error(
                            why="the PNG IHDR chunk does not have the required 13-byte payload",
                            what=f"payload_length={len(payload)}",
                            how="pass a PNG whose IHDR chunk contains exactly 13 bytes",
                        )
                    )
                width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
            elif chunk_type == b"cICP" and len(payload) == 4:
                raw_color["cICP"] = tuple(payload)
            elif chunk_type == b"sRGB" and len(payload) == 1:
                raw_color["sRGB"] = int(payload[0])
            elif chunk_type == b"gAMA" and len(payload) == 4:
                raw_color["gAMA"] = struct.unpack(">I", payload)[0]
            elif chunk_type == b"tRNS":
                transparent = True
            if chunk_type == b"IEND":
                break

    channel_map = {
        0: ("Y",),
        2: ("R", "G", "B"),
        3: ("R", "G", "B"),
        4: ("Y", "A"),
        6: ("R", "G", "B", "A"),
    }
    labels = channel_map.get(color_type)
    if labels is None or width <= 0 or height <= 0:
        raise ValueError(
            _actionable_error(
                why="the PNG header contains an unsupported color type or non-positive dimensions",
                what=f"color_type={color_type}, width={width}, height={height}",
                how="pass a PNG with a supported color type and positive width and height",
            )
        )
    if transparent and color_type in (0, 2, 3):
        labels = (*labels, "A")
    dtype = "uint16" if bit_depth == 16 else "uint8"
    return ImageHeader(
        format="PNG",
        width=width,
        height=height,
        parts=(_ImagePart(name="", channels=dict.fromkeys(labels, dtype)),),
        color=_png_color_info(raw_color),
    )
