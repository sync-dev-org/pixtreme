"""WebP header parsing."""

from __future__ import annotations

import struct
from pathlib import Path

from pixtreme._core.errors import _actionable_error
from pixtreme._io.formats.nvimgcodec import _new_raster_header
from pixtreme._io.models import ImageHeader


def _parse_webp(source: Path | bytes) -> ImageHeader:
    data = source.read_bytes() if isinstance(source, Path) else source
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError(
            _actionable_error(
                why="the WebP payload lacks its RIFF/WEBP signature",
                what=f"signature={data[:12]!r}",
                how="pass a RIFF container whose form type is WEBP",
            )
        )
    riff_size = struct.unpack_from("<I", data, 4)[0]
    riff_end = riff_size + 8
    if riff_end > len(data) or riff_size < 4:
        raise ValueError(
            _actionable_error(
                why="the WebP RIFF size exceeds the available payload",
                what=f"declared={riff_end} bytes, available={len(data)} bytes",
                how="pass a complete WebP RIFF container",
            )
        )
    offset = 12
    while offset < riff_end:
        if riff_end - offset < 8:
            raise ValueError(
                _actionable_error(
                    why="the WebP chunk header is truncated",
                    what=f"offset={offset}, remaining={riff_end - offset}",
                    how="pass a WebP container with complete chunk headers",
                )
            )
        chunk_type = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > riff_end:
            raise ValueError(
                _actionable_error(
                    why="the WebP chunk size exceeds the RIFF container",
                    what=f"chunk={chunk_type!r}, size={chunk_size}, remaining={riff_end - chunk_start}",
                    how="pass a complete WebP image chunk",
                )
            )
        chunk = data[chunk_start:chunk_end]
        if chunk_type == b"VP8 ":
            if len(chunk) < 10 or chunk[3:6] != b"\x9d\x01\x2a":
                raise ValueError(
                    _actionable_error(
                        why="the lossy WebP VP8 frame header is truncated or invalid",
                        what=f"frame_header={chunk[:10]!r}",
                        how="pass a VP8 chunk with the 9d 01 2a start code and dimensions",
                    )
                )
            width = struct.unpack_from("<H", chunk, 6)[0] & 0x3FFF
            height = struct.unpack_from("<H", chunk, 8)[0] & 0x3FFF
            return _new_raster_header("WEBP", width=width, height=height, component_count=3, dtype="uint8")
        if chunk_type == b"VP8L":
            if len(chunk) < 5 or chunk[0] != 0x2F:
                raise ValueError(
                    _actionable_error(
                        why="the lossless WebP VP8L frame header is truncated or invalid",
                        what=f"frame_header={chunk[:5]!r}",
                        how="pass a VP8L chunk beginning with signature byte 0x2f",
                    )
                )
            bits = int.from_bytes(chunk[1:5], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            alpha = (bits >> 28) & 1
            version = bits >> 29
            if alpha or version:
                raise ValueError(
                    _actionable_error(
                        why="the WebP VP8L stream uses alpha or a nonzero version outside the supported contract",
                        what=f"alpha={alpha}, version={version}",
                        how="encode a version-zero RGB WebP without alpha",
                    )
                )
            return _new_raster_header("WEBP", width=width, height=height, component_count=3, dtype="uint8")
        if chunk_type == b"VP8X":
            if len(chunk) != 10:
                raise ValueError(
                    _actionable_error(
                        why="the extended WebP VP8X header is not 10 bytes",
                        what=f"length={len(chunk)}",
                        how="pass a complete 10-byte VP8X image header",
                    )
                )
            flags = chunk[0]
            if flags & 0xC1 or chunk[1:4] != b"\x00\x00\x00":
                raise ValueError(
                    _actionable_error(
                        why="the extended WebP VP8X header has nonzero reserved fields",
                        what=f"vp8x_flags=0x{flags:02x}, reserved={chunk[1:4]!r}",
                        how="clear the VP8X reserved flag bits and reserved bytes",
                    )
                )
            if flags & 0x12:
                raise ValueError(
                    _actionable_error(
                        why="animated or alpha WebP is outside the supported RGB contract",
                        what=f"vp8x_flags=0x{flags:02x}",
                        how="encode a still RGB WebP without alpha",
                    )
                )
            width = int.from_bytes(chunk[4:7], "little") + 1
            height = int.from_bytes(chunk[7:10], "little") + 1
            return _new_raster_header("WEBP", width=width, height=height, component_count=3, dtype="uint8")
        offset = chunk_end + (chunk_size & 1)
    raise ValueError(
        _actionable_error(
            why="the WebP container has no VP8, VP8L, or VP8X image header",
            what=f"riff_size={riff_size}",
            how="pass a still WebP image with a supported image chunk",
        )
    )
