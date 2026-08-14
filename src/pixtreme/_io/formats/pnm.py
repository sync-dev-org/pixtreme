"""PNM header parsing."""

from __future__ import annotations

from pathlib import Path

from pixtreme._core.errors import _actionable_error
from pixtreme._io.formats.nvimgcodec import _new_raster_header
from pixtreme._io.models import ImageHeader

_PNM_WHITESPACE = b" \t\r\n\v\f"


def _pnm_token(data: bytes, offset: int) -> tuple[bytes, int]:
    while offset < len(data):
        if data[offset] in _PNM_WHITESPACE:
            offset += 1
            continue
        if data[offset] == ord("#"):
            newline = data.find(b"\n", offset + 1)
            if newline < 0:
                raise ValueError(
                    _actionable_error(
                        why="the PNM comment reaches the end of the payload without a line ending",
                        what=f"comment_offset={offset}",
                        how="terminate every PNM comment with a newline",
                    )
                )
            offset = newline + 1
            continue
        break
    start = offset
    while offset < len(data) and data[offset] not in _PNM_WHITESPACE and data[offset] != ord("#"):
        offset += 1
    if start == offset:
        raise ValueError(
            _actionable_error(
                why="the PNM header or ASCII raster ended before the next token",
                what=f"offset={offset}, payload_length={len(data)}",
                how="pass complete PNM width, height, maxval, and sample tokens",
            )
        )
    return data[start:offset], offset


def _parse_pnm(source: Path | bytes) -> ImageHeader:
    data = source.read_bytes() if isinstance(source, Path) else source
    magic_bytes, offset = _pnm_token(data, 0)
    if magic_bytes not in (b"P2", b"P3", b"P5", b"P6"):
        raise ValueError(
            _actionable_error(
                why="the PNM magic is not a supported grayscale or RGB variant",
                what=f"magic={magic_bytes!r}",
                how="use P2/P3 ASCII or P5/P6 binary PNM",
            )
        )
    try:
        width_bytes, offset = _pnm_token(data, offset)
        height_bytes, offset = _pnm_token(data, offset)
        maxval_bytes, offset = _pnm_token(data, offset)
        width, height, maxval = int(width_bytes), int(height_bytes), int(maxval_bytes)
    except ValueError as error:
        if "why=" in str(error):
            raise
        raise ValueError(
            _actionable_error(
                why="the PNM dimensions or maxval are not decimal integers",
                what=f"width={width_bytes!r}, height={height_bytes!r}, maxval={maxval_bytes!r}",
                how="use positive decimal dimensions and maxval 255 or 65535",
            )
        ) from error
    if width <= 0 or height <= 0 or maxval not in (255, 65535):
        raise ValueError(
            _actionable_error(
                why="the PNM dimensions or maxval are outside the supported contract",
                what=f"width={width}, height={height}, maxval={maxval}",
                how="use positive dimensions and maxval 255 or 65535",
            )
        )
    component_count = 1 if magic_bytes in (b"P2", b"P5") else 3
    sample_count = width * height * component_count
    if magic_bytes in (b"P5", b"P6"):
        if offset >= len(data) or data[offset] not in _PNM_WHITESPACE:
            raise ValueError(
                _actionable_error(
                    why="the binary PNM maxval is not followed by a raster delimiter",
                    what=f"offset={offset}, next={data[offset : offset + 1]!r}",
                    how="place one ASCII whitespace delimiter before the binary raster",
                )
            )
        offset += 1
        if data[offset - 1 : offset] == b"\r" and data[offset : offset + 1] == b"\n":
            offset += 1
        expected_bytes = sample_count * (2 if maxval == 65535 else 1)
        if len(data) - offset < expected_bytes:
            raise ValueError(
                _actionable_error(
                    why="the binary PNM raster is shorter than its dimensions and maxval require",
                    what=f"expected={expected_bytes} bytes, available={len(data) - offset} bytes",
                    how="pass every big-endian binary sample declared by the PNM header",
                )
            )
    return _new_raster_header(
        "PNM",
        width=width,
        height=height,
        component_count=component_count,
        dtype="uint8" if maxval == 255 else "uint16",
    )
