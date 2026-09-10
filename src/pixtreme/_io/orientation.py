"""EXIF orientation parsing and public-boundary validation."""

from __future__ import annotations

import struct
import warnings
from io import BytesIO
from typing import BinaryIO

from pixtreme._core.errors import _actionable_error

_EXIF_PREFIX = b"Exif\x00\x00"
_ORIENTATION_TAG = 274


class _InvalidExifOrientation(ValueError):
    """Internal marker for malformed optional orientation metadata."""


def _read_exact_exif(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise _InvalidExifOrientation(f"truncated field: expected {size} bytes, found {len(value)}")
    return value


def _read_exif_orientation(stream: BinaryIO, *, description: str) -> int:
    base = stream.tell()
    try:
        byte_order = _read_exact_exif(stream, 2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise _InvalidExifOrientation(f"byte order is {byte_order!r}, not b'II' or b'MM'")
        magic, ifd_offset = struct.unpack(f"{endian}HI", _read_exact_exif(stream, 6))
        if magic != 42:
            raise _InvalidExifOrientation(f"TIFF magic is {magic}, not 42")
        stream.seek(base + ifd_offset)
        entry_count = struct.unpack(f"{endian}H", _read_exact_exif(stream, 2))[0]
        entries = [_read_exact_exif(stream, 12) for _ in range(entry_count)]
        _read_exact_exif(stream, 4)
        orientation_entries = [
            entry for entry in entries if struct.unpack(f"{endian}H", entry[:2])[0] == _ORIENTATION_TAG
        ]
        if not orientation_entries:
            return 1
        if len(orientation_entries) != 1:
            raise _InvalidExifOrientation(
                f"primary IFD contains {len(orientation_entries)} Orientation entries instead of one"
            )
        entry = orientation_entries[0]
        _tag, field_type, count = struct.unpack(f"{endian}HHI", entry[:8])
        if field_type != 3 or count != 1:
            raise _InvalidExifOrientation(f"Orientation type/count is {field_type}/{count}, expected SHORT/1")
        orientation = int(struct.unpack(f"{endian}H", entry[8:10])[0])
        if not 1 <= orientation <= 8:
            raise _InvalidExifOrientation(f"Orientation value is {orientation}, expected 1 through 8")
        return orientation
    except (OSError, OverflowError, struct.error, _InvalidExifOrientation) as error:
        warnings.warn(
            f"invalid EXIF orientation metadata in {description}; using orientation 1: {error}",
            UserWarning,
            stacklevel=4,
        )
        return 1


def _parse_exif_orientation(payload: bytes, *, description: str) -> int:
    data = payload[len(_EXIF_PREFIX) :] if payload.startswith(_EXIF_PREFIX) else payload
    return _read_exif_orientation(BytesIO(data), description=description)


def _oriented_dimensions(width: int, height: int, orientation: int) -> tuple[int, int]:
    return (height, width) if orientation >= 5 else (width, height)


def _strip_webp_exif(payload: bytes) -> bytes:
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WEBP":
        return payload
    riff_end = struct.unpack_from("<I", payload, 4)[0] + 8
    if riff_end > len(payload):
        return payload

    offset = 12
    chunks: list[tuple[bytes, bytes, bytes]] = []
    removed = False
    while offset < riff_end:
        if riff_end - offset < 8:
            return payload
        chunk_type = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        next_offset = chunk_end + (chunk_size & 1)
        if chunk_end > riff_end or next_offset > len(payload):
            return payload
        if chunk_type == b"EXIF":
            removed = True
        else:
            chunk = payload[chunk_start:chunk_end]
            if chunk_type == b"VP8X" and len(chunk) == 10 and chunk[0] & 0x08:
                chunk = bytes((chunk[0] & ~0x08,)) + chunk[1:]
            chunks.append((chunk_type, chunk, payload[chunk_end:next_offset]))
        offset = next_offset
    if not removed:
        return payload

    body = bytearray(b"WEBP")
    for chunk_type, chunk, padding in chunks:
        body.extend(chunk_type)
        body.extend(struct.pack("<I", len(chunk)))
        body.extend(chunk)
        body.extend(padding)
    return b"RIFF" + struct.pack("<I", len(body)) + bytes(body)


def _validate_apply_exif_orientation(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(
            _actionable_error(
                why="apply_exif_orientation must be an exact bool",
                what=repr(value),
                how="pass apply_exif_orientation=True or apply_exif_orientation=False",
            )
        )
    return value
