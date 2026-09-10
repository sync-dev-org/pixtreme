"""WebP header parsing."""

from __future__ import annotations

import os
import struct
from pathlib import Path

from pixtreme._core.errors import _actionable_error
from pixtreme._io.common import _binary_stream, _read_exact
from pixtreme._io.formats.nvimgcodec import _new_raster_header
from pixtreme._io.icc import _MAX_PROFILE_SIZE, _icc_carrier_color, _IccCarrier
from pixtreme._io.models import ImageHeader
from pixtreme._io.orientation import _oriented_dimensions, _parse_exif_orientation

_IMAGE_CHUNKS = frozenset((b"ANIM", b"ANMF", b"ALPH", b"VP8 ", b"VP8L"))


def _parse_webp(source: Path | bytes) -> ImageHeader:
    with _binary_stream(source) as stream:
        stream.seek(0, os.SEEK_END)
        available = stream.tell()
        stream.seek(0)
        signature = _read_exact(stream, 12)
        if signature[:4] != b"RIFF" or signature[8:12] != b"WEBP":
            raise ValueError(
                _actionable_error(
                    why="the WebP payload lacks its RIFF/WEBP signature",
                    what=f"signature={signature!r}",
                    how="pass a RIFF container whose form type is WEBP",
                )
            )
        riff_size = struct.unpack_from("<I", signature, 4)[0]
        riff_end = riff_size + 8
        if riff_end > available or riff_size < 4:
            raise ValueError(
                _actionable_error(
                    why="the WebP RIFF size exceeds the available payload",
                    what=f"declared={riff_end} bytes, available={available} bytes",
                    how="pass a complete WebP RIFF container",
                )
            )

        offset = 12
        header: ImageHeader | None = None
        orientation = 1
        saw_exif = False
        leading_vp8x = False
        icc_flag = False
        icc_count = 0
        icc_offset: int | None = None
        icc_profile: bytes | None = None
        icc_invalid = False
        first_image_offset: int | None = None
        while offset < riff_end:
            stream.seek(offset)
            if riff_end - offset < 8:
                raise ValueError(
                    _actionable_error(
                        why="the WebP chunk header is truncated",
                        what=f"offset={offset}, remaining={riff_end - offset}",
                        how="pass a WebP container with complete chunk headers",
                    )
                )
            chunk_type = _read_exact(stream, 4)
            chunk_size = struct.unpack("<I", _read_exact(stream, 4))[0]
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
            if chunk_type in _IMAGE_CHUNKS and first_image_offset is None:
                first_image_offset = offset

            if chunk_type == b"ICCP":
                icc_count += 1
                if icc_count == 1:
                    icc_offset = offset
                valid_position = leading_vp8x and icc_flag and first_image_offset is None and offset > 12
                if icc_invalid or icc_count != 1 or not valid_position or chunk_size > _MAX_PROFILE_SIZE:
                    icc_invalid = True
                    icc_profile = None
                else:
                    icc_profile = _read_exact(stream, chunk_size)
            elif chunk_type == b"EXIF":
                chunk = _read_exact(stream, chunk_size)
                if not saw_exif:
                    orientation = _parse_exif_orientation(chunk, description="WebP EXIF chunk")
                    saw_exif = True
            elif chunk_type == b"VP8 ":
                chunk = _read_exact(stream, min(chunk_size, 10))
                if len(chunk) < 10 or chunk[3:6] != b"\x9d\x01\x2a":
                    raise ValueError(
                        _actionable_error(
                            why="the lossy WebP VP8 frame header is truncated or invalid",
                            what=f"frame_header={chunk!r}",
                            how="pass a VP8 chunk with the 9d 01 2a start code and dimensions",
                        )
                    )
                width = struct.unpack_from("<H", chunk, 6)[0] & 0x3FFF
                height = struct.unpack_from("<H", chunk, 8)[0] & 0x3FFF
                if header is None:
                    header = _new_raster_header("WEBP", width=width, height=height, component_count=3, dtype="uint8")
            elif chunk_type == b"VP8L":
                chunk = _read_exact(stream, min(chunk_size, 5))
                if len(chunk) < 5 or chunk[0] != 0x2F:
                    raise ValueError(
                        _actionable_error(
                            why="the lossless WebP VP8L frame header is truncated or invalid",
                            what=f"frame_header={chunk!r}",
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
                if header is None:
                    header = _new_raster_header("WEBP", width=width, height=height, component_count=3, dtype="uint8")
            elif chunk_type == b"VP8X":
                chunk = _read_exact(stream, chunk_size)
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
                if offset == 12:
                    leading_vp8x = True
                    icc_flag = bool(flags & 0x20)
                width = int.from_bytes(chunk[4:7], "little") + 1
                height = int.from_bytes(chunk[7:10], "little") + 1
                if header is None:
                    header = _new_raster_header("WEBP", width=width, height=height, component_count=3, dtype="uint8")
            offset = chunk_end + (chunk_size & 1)

    if header is None:
        raise ValueError(
            _actionable_error(
                why="the WebP container has no VP8, VP8L, or VP8X image header",
                what=f"riff_size={riff_size}",
                how="pass a still WebP image with a supported image chunk",
            )
        )
    if not icc_flag and icc_count == 0:
        carrier = _IccCarrier(present=False, profile=None)
    else:
        valid_icc = (
            not icc_invalid
            and leading_vp8x
            and icc_flag
            and icc_count == 1
            and icc_profile is not None
            and icc_offset is not None
            and icc_offset > 12
            and first_image_offset is not None
            and icc_offset < first_image_offset
        )
        carrier = _IccCarrier(present=True, profile=icc_profile if valid_icc else None)
    color = _icc_carrier_color(carrier, compatible=True)
    width, height = _oriented_dimensions(header.width, header.height, orientation)
    return header.model_copy(update={"width": width, "height": height, "orientation": orientation, "color": color})
