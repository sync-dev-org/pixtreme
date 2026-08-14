"""TIFF header parsing."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, cast

from pixtreme._core.errors import _actionable_error
from pixtreme._io.common import _binary_stream, _empty_color, _read_exact
from pixtreme._io.models import ImageHeader, _ImagePart

_TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


def _tiff_values(stream: BinaryIO, endian: str, field_type: int, count: int, field: bytes) -> tuple[int, ...]:
    item_size = _TIFF_TYPE_SIZES.get(field_type)
    if item_size is None:
        return ()
    total_size = item_size * count
    if total_size <= 4:
        payload = field[:total_size]
    else:
        offset = struct.unpack(f"{endian}I", field)[0]
        current = stream.tell()
        stream.seek(offset)
        payload = _read_exact(stream, total_size)
        stream.seek(current)
    if field_type == 3:
        return cast(tuple[int, ...], struct.unpack(f"{endian}{count}H", payload))
    if field_type in (4, 13):
        return cast(tuple[int, ...], struct.unpack(f"{endian}{count}I", payload))
    if field_type == 1:
        return tuple(payload)
    return ()


def _parse_tiff(source: Path | bytes) -> ImageHeader:
    with _binary_stream(source) as stream:
        byte_order = _read_exact(stream, 2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError(
                _actionable_error(
                    why="the TIFF byte-order marker is neither little-endian II nor big-endian MM",
                    what=f"byte_order={byte_order!r}",
                    how="pass a TIFF beginning with b'II' or b'MM'",
                )
            )
        magic, ifd_offset = struct.unpack(f"{endian}HI", _read_exact(stream, 6))
        if magic != 42:
            raise ValueError(
                _actionable_error(
                    why="the TIFF header does not contain the required magic value 42",
                    what=f"magic={magic}",
                    how="pass a classic TIFF file with a valid header",
                )
            )
        stream.seek(ifd_offset)
        entry_count = struct.unpack(f"{endian}H", _read_exact(stream, 2))[0]
        tags: dict[int, tuple[int, ...]] = {}
        for _ in range(entry_count):
            entry = _read_exact(stream, 12)
            tag, field_type, count = struct.unpack(f"{endian}HHI", entry[:8])
            tags[tag] = _tiff_values(stream, endian, field_type, count, entry[8:12])

    width = tags.get(256, (0,))[0]
    height = tags.get(257, (0,))[0]
    samples = tags.get(277, (1,))[0]
    bits = tags.get(258, (8,))
    sample_formats = tags.get(339, (1,))
    photometric = tags.get(262, (1,))[0]
    bit_depth = max(bits)
    if all(value == 3 for value in sample_formats) and bit_depth == 32:
        dtype = "float32"
    elif bit_depth == 8:
        dtype = "uint8"
    elif bit_depth == 16:
        dtype = "uint16"
    else:
        dtype = f"uint{bit_depth}"
    labels: tuple[str, ...]
    if photometric in (0, 1) and samples == 1:
        labels = ("Y",)
    elif photometric in (0, 1) and samples == 2:
        labels = ("Y", "A")
    elif samples == 4:
        labels = ("R", "G", "B", "A")
    elif samples == 3 or photometric in (2, 3, 6):
        labels = ("R", "G", "B")
    else:
        labels = tuple(f"channel-{index}" for index in range(samples))
    if width <= 0 or height <= 0:
        raise ValueError(
            _actionable_error(
                why="the TIFF header contains non-positive image dimensions",
                what=f"width={width}, height={height}",
                how="pass a TIFF whose ImageWidth and ImageLength tags are positive",
            )
        )
    return ImageHeader(
        format="TIFF",
        width=width,
        height=height,
        parts=(_ImagePart(name="", channels=dict.fromkeys(labels, dtype)),),
        color=_empty_color(),
    )
