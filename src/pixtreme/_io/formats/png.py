"""PNG header parsing."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, cast

from pixtreme._core.errors import _actionable_error
from pixtreme._io.common import _binary_stream, _empty_color, _read_exact
from pixtreme._io.icc import _MAX_PNG_COMPRESSED_SIZE, _icc_carrier_color, _IccCarrier, _png_icc_carrier
from pixtreme._io.models import ImageHeader, _ImageColorInfo, _ImagePart
from pixtreme._io.orientation import _oriented_dimensions, _parse_exif_orientation

_CICP_COLORSPACES = {1: "Rec.709", 9: "Rec.2020"}
_CICP_GAMMAS = {1: "Rec.709", 6: "Rec.709", 13: "sRGB", 16: "PQ", 18: "HLG"}


def _png_color_info(
    raw: dict[str, object],
    carrier: _IccCarrier,
    *,
    compatible: bool,
) -> _ImageColorInfo:
    if "cICP" in raw:
        primary, transfer, matrix, full_range = cast(tuple[int, int, int, int], raw["cICP"])
        colorspace = _CICP_COLORSPACES.get(primary)
        mapped_gamma = _CICP_GAMMAS.get(transfer)
        mappable = colorspace is not None and mapped_gamma is not None and matrix == 0 and full_range in (0, 1)
        return _ImageColorInfo(raw=raw, colorspace=colorspace, gamma=mapped_gamma, mappable=mappable)
    if carrier.present:
        return _icc_carrier_color(carrier, compatible=compatible, raw=raw)
    if "sRGB" in raw:
        return _ImageColorInfo(raw=raw, colorspace="sRGB", gamma="sRGB", mappable=True)
    if "gAMA" in raw:
        gamma_value = cast(int, raw["gAMA"])
        gamma: str | None
        if abs(gamma_value - 100000) <= 1:
            gamma = "linear"
        elif abs(gamma_value - 45455) <= 1:
            gamma = "Gamma-2.2"
        elif abs(gamma_value - 41667) <= 1:
            gamma = "Gamma-2.4"
        else:
            gamma = None
        return _ImageColorInfo(raw=raw, colorspace=None, gamma=gamma, mappable=gamma is not None)
    return _empty_color()


def _read_png_iccp_payload(stream: BinaryIO, *, offset: int, length: int) -> bytes | None:
    """Materialize one iCCP only after its declared compressed size is known to be bounded."""
    if length < 3:
        return None
    current = stream.tell()
    try:
        stream.seek(offset)
        prefix = _read_exact(stream, min(length, 81))
        separator = prefix.find(b"\x00")
        if separator < 1 or separator > 79 or separator + 2 > length:
            return None
        name = prefix[:separator]
        if name[0] == 0x20 or name[-1] == 0x20 or b"  " in name:
            return None
        if any(not (32 <= value <= 126 or 161 <= value <= 255) for value in name):
            return None
        if prefix[separator + 1] != 0:
            return None
        compressed_size = length - separator - 2
        if compressed_size > _MAX_PNG_COMPRESSED_SIZE:
            return None
        stream.seek(offset)
        return _read_exact(stream, length)
    finally:
        stream.seek(current)


def _parse_png(source: Path | bytes) -> ImageHeader:
    raw_color: dict[str, object] = {}
    iccp_location: tuple[int, int] | None = None
    duplicate_iccp = False
    transparent = False
    orientation = 1
    saw_exif = False
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
            payload_offset = stream.tell()
            if chunk_type == b"iCCP":
                if iccp_location is None and not duplicate_iccp:
                    iccp_location = (payload_offset, length)
                else:
                    iccp_location = None
                    duplicate_iccp = True
                stream.seek(length, 1)
            elif chunk_type in (b"IHDR", b"cICP", b"sRGB", b"gAMA", b"tRNS", b"eXIf"):
                payload = _read_exact(stream, length)
            else:
                payload = b""
                stream.seek(length, 1)
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
            elif chunk_type == b"eXIf" and not saw_exif:
                orientation = _parse_exif_orientation(payload, description="PNG eXIf chunk")
                saw_exif = True
            if chunk_type == b"IEND":
                break

        if "cICP" in raw_color:
            carrier = _IccCarrier(present=False, profile=None)
        elif duplicate_iccp:
            carrier = _IccCarrier(present=True, profile=None)
        elif iccp_location is None:
            carrier = _IccCarrier(present=False, profile=None)
        else:
            offset, length = iccp_location
            iccp_payload = _read_png_iccp_payload(stream, offset=offset, length=length)
            carrier = (
                _IccCarrier(present=True, profile=None) if iccp_payload is None else _png_icc_carrier([iccp_payload])
            )

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
    width, height = _oriented_dimensions(width, height, orientation)
    return ImageHeader(
        format="PNG",
        width=width,
        height=height,
        parts=(_ImagePart(name="", channels=dict.fromkeys(labels, dtype)),),
        color=_png_color_info(raw_color, carrier, compatible=labels in (("R", "G", "B"), ("R", "G", "B", "A"))),
        orientation=orientation,
    )
