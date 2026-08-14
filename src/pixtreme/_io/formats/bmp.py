"""BMP header parsing."""

from __future__ import annotations

import struct
from pathlib import Path

from pixtreme._core.errors import _actionable_error
from pixtreme._io.formats.nvimgcodec import _new_raster_header
from pixtreme._io.models import ImageHeader


def _parse_bmp(source: Path | bytes) -> ImageHeader:
    data = source.read_bytes() if isinstance(source, Path) else source
    if len(data) < 18 or data[:2] != b"BM":
        raise ValueError(
            _actionable_error(
                why="the BMP file header is truncated or lacks the BM signature",
                what=f"signature={data[:2]!r}, length={len(data)}",
                how="pass a BMP file with its 14-byte file header and DIB header",
            )
        )
    file_size, pixel_offset = struct.unpack_from("<I4xI", data, 2)
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40 or len(data) < 14 + dib_size:
        raise ValueError(
            _actionable_error(
                why="the BMP DIB header is unsupported or truncated",
                what=f"dib_size={dib_size}, available={max(0, len(data) - 14)}",
                how="pass a BMP with a complete BITMAPINFOHEADER or later DIB header",
            )
        )
    width, stored_height, planes, bits_per_pixel, compression = struct.unpack_from("<iiHHI", data, 18)
    height = abs(stored_height)
    if not 14 + dib_size <= pixel_offset <= file_size <= len(data) or planes != 1 or compression != 0:
        raise ValueError(
            _actionable_error(
                why="the BMP file geometry or encoding fields are structurally invalid",
                what=(
                    f"file_size={file_size}, available={len(data)}, pixel_offset={pixel_offset}, "
                    f"planes={planes}, compression={compression}"
                ),
                how="use an uncompressed single-plane BMP with valid file and pixel offsets",
            )
        )
    component_count = {8: 1, 24: 3}.get(bits_per_pixel)
    if component_count is None:
        raise ValueError(
            _actionable_error(
                why="the BMP bit depth does not map to supported Y or RGB storage",
                what=f"bits_per_pixel={bits_per_pixel}",
                how="encode an 8-bit grayscale or 24-bit RGB BMP",
            )
        )
    return _new_raster_header("BMP", width=width, height=height, component_count=component_count, dtype="uint8")
