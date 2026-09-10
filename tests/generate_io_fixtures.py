"""Reproducible fixtures for the v1 image-I/O specification tests."""

from __future__ import annotations

import io
import struct
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def orientation_pattern(*, height: int = 18, width: int = 24) -> NDArray[np.uint8]:
    """Return a non-square RGB image whose pixels and edges are distinguishable."""
    rows, columns = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            (columns * 17 + rows * 3 + 11) % 256,
            (columns * 5 + rows * 29 + 37) % 256,
            (columns * 31 + rows * 7 + 73) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def encode_oriented_raster(format_name: str, orientation: int) -> bytes:
    """Encode the deterministic orientation pattern with a valid EXIF tag."""
    exif = Image.Exif()
    exif[274] = orientation
    options: dict[str, object] = {"exif": exif.tobytes()}
    if format_name == "JPEG":
        options.update(quality=100, subsampling=0)
    elif format_name == "TIFF":
        options["compression"] = "raw"
    elif format_name == "WEBP":
        options.update(quality=100, method=6)
    output = io.BytesIO()
    Image.fromarray(orientation_pattern(), mode="RGB").save(output, format=format_name, **options)
    return output.getvalue()


def encode_plain_raster(format_name: str) -> bytes:
    """Encode the deterministic orientation pattern without EXIF metadata."""
    options: dict[str, object] = {}
    if format_name == "JPEG":
        options.update(quality=100, subsampling=0)
    elif format_name == "TIFF":
        options["compression"] = "raw"
    elif format_name == "WEBP":
        options.update(quality=100, method=6)
    output = io.BytesIO()
    Image.fromarray(orientation_pattern(), mode="RGB").save(output, format=format_name, **options)
    return output.getvalue()


def encode_lossless_oriented_webp(orientation: int) -> bytes:
    """Encode a VP8L WebP whose extended container carries EXIF orientation."""
    exif = Image.Exif()
    exif[274] = orientation
    output = io.BytesIO()
    Image.fromarray(orientation_pattern(), mode="RGB").save(
        output,
        format="WEBP",
        lossless=True,
        exif=exif.tobytes(),
    )
    return output.getvalue()


def exif_payload(
    entries: Sequence[tuple[int, int, int, int]],
    *,
    byte_order: bytes = b"II",
    secondary_entries: Sequence[tuple[int, int, int, int]] = (),
) -> bytes:
    """Build a classic-TIFF EXIF payload with explicit first/secondary IFD entries."""
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return byte_order + b"\x00" * 14

    def make_ifd(values: Sequence[tuple[int, int, int, int]], next_offset: int) -> bytes:
        encoded = bytearray(struct.pack(f"{endian}H", len(values)))
        for tag, field_type, count, value in values:
            if field_type == 3 and count == 1:
                field = struct.pack(f"{endian}H", value) + b"\x00\x00"
            else:
                field = struct.pack(f"{endian}I", value)
            encoded.extend(struct.pack(f"{endian}HHI", tag, field_type, count))
            encoded.extend(field)
        encoded.extend(struct.pack(f"{endian}I", next_offset))
        return bytes(encoded)

    first_size = 2 + 12 * len(entries) + 4
    next_offset = 8 + first_size if secondary_entries else 0
    first = make_ifd(entries, next_offset)
    secondary = make_ifd(secondary_entries, 0) if secondary_entries else b""
    return byte_order + struct.pack(f"{endian}HI", 42, 8) + first + secondary


def png_with_exif(payload: bytes) -> bytes:
    """Insert one eXIf chunk after IHDR in a deterministic PNG."""
    output = io.BytesIO()
    Image.fromarray(orientation_pattern(), mode="RGB").save(output, format="PNG")
    plain = output.getvalue()
    ihdr_end = 8 + 12 + struct.unpack_from(">I", plain, 8)[0]
    chunk_type = b"eXIf"
    chunk = (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )
    return plain[:ihdr_end] + chunk + plain[ihdr_end:]


def write_exr(
    path: Path,
    channels: Mapping[str, NDArray[np.generic]],
    *,
    header: Mapping[str, object] | None = None,
) -> None:
    """Write one-part EXR, including Blender-style dotted channel names."""
    from openexr_dev_oracle import OpenEXR

    OpenEXR.File(dict(header or {}), dict(channels)).write(str(path))


def write_multipart_exr(
    path: Path,
    parts: Sequence[tuple[str, Mapping[str, NDArray[np.generic]], Mapping[str, object]]],
) -> None:
    """Write a true multi-part EXR with explicit part names."""
    from openexr_dev_oracle import OpenEXR

    dimensions = [next(iter(channels.values())).shape for _, channels, _ in parts]
    display_height = max(shape[0] for shape in dimensions)
    display_width = max(shape[1] for shape in dimensions)
    display_window = (
        np.array((0, 0), dtype=np.int32),
        np.array((display_width - 1, display_height - 1), dtype=np.int32),
    )
    exr_parts = []
    for name, channels, header in parts:
        part_header = {**header, "displayWindow": display_window}
        exr_parts.append(OpenEXR.Part(part_header, dict(channels), name))
    OpenEXR.File(exr_parts).write(str(path))
