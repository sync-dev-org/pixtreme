"""JPEG 2000 header parsing."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

from pixtreme._core.errors import _actionable_error
from pixtreme._io.formats.nvimgcodec import _new_raster_header
from pixtreme._io.models import ImageHeader


def _jp2_boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    offset = start
    while offset < end:
        if end - offset < 8:
            raise ValueError(
                _actionable_error(
                    why="the JPEG 2000 box header is truncated",
                    what=f"offset={offset}, remaining={end - offset} bytes",
                    how="pass a JP2 file with complete length and type fields",
                )
            )
        length, box_type = struct.unpack_from(">I4s", data, offset)
        header_size = 8
        if length == 1:
            if end - offset < 16:
                raise ValueError(
                    _actionable_error(
                        why="the JPEG 2000 extended box length is truncated",
                        what=f"box={box_type!r}, offset={offset}",
                        how="pass a JP2 file with a complete 64-bit extended length",
                    )
                )
            length = struct.unpack_from(">Q", data, offset + 8)[0]
            header_size = 16
        box_end = end if length == 0 else offset + length
        if (length != 0 and length < header_size) or box_end > end:
            raise ValueError(
                _actionable_error(
                    why="the JPEG 2000 box length exceeds its containing payload",
                    what=f"box={box_type!r}, length={length}, remaining={end - offset}",
                    how="pass a JP2 file whose box lengths stay within their parent boxes",
                )
            )
        yield box_type, offset + header_size, box_end
        offset = box_end


def _parse_jp2(data: bytes) -> ImageHeader:
    signature = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
    if not data.startswith(signature):
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 file does not have the JP2 signature box",
                what=f"signature={data[:12]!r}",
                how="pass a JP2 container beginning with its 12-byte signature box",
            )
        )
    ihdr: tuple[int, int, int, int] | None = None
    component_depths: tuple[int, ...] | None = None
    for box_type, payload_start, box_end in _jp2_boxes(data, 0, len(data)):
        if box_type != b"jp2h":
            continue
        for child_type, child_start, child_end in _jp2_boxes(data, payload_start, box_end):
            child = data[child_start:child_end]
            if child_type == b"ihdr":
                if len(child) != 14:
                    raise ValueError(
                        _actionable_error(
                            why="the JPEG 2000 image header box is not 14 bytes",
                            what=f"length={len(child)}",
                            how="pass a JP2 container with a complete ihdr box",
                        )
                    )
                height, width, components, bpc, compression, unknown_color, intellectual_property = struct.unpack(
                    ">IIHBBBB", child
                )
                if compression != 7 or unknown_color not in (0, 1) or intellectual_property not in (0, 1):
                    raise ValueError(
                        _actionable_error(
                            why="the JPEG 2000 image header contains invalid fixed fields",
                            what=(
                                f"compression={compression}, unknown_color={unknown_color}, "
                                f"intellectual_property={intellectual_property}"
                            ),
                            how="use the JP2 baseline compression and boolean header flags",
                        )
                    )
                ihdr = height, width, components, bpc
            elif child_type == b"bpcc":
                component_depths = tuple(child)
        break
    if ihdr is None:
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 container has no image header box",
                what="missing jp2h/ihdr",
                how="pass a JP2 container with an ihdr box inside jp2h",
            )
        )
    height, width, components, bpc = ihdr
    if bpc == 255:
        if component_depths is None:
            raise ValueError(
                _actionable_error(
                    why="the JPEG 2000 image header delegates component depths but has no bpcc box",
                    what="ihdr bpc=255, bpcc missing",
                    how="provide one bpcc entry for every component",
                )
            )
        depths = component_depths
    else:
        depths = (bpc,) * components
    if len(depths) != components:
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 component depth table does not match the component count",
                what=f"components={components}, depth_entries={len(depths)}",
                how="provide one bpcc entry for every component",
            )
        )
    if any(value & 0x80 for value in depths):
        raise ValueError(
            _actionable_error(
                why="signed JPEG 2000 components are outside the supported image contract",
                what=f"component_depths={depths!r}",
                how="encode unsigned 8-bit or 16-bit components",
            )
        )
    precisions = tuple((value & 0x7F) + 1 for value in depths)
    if len(set(precisions)) != 1 or precisions[0] not in (8, 16):
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 component precisions are unsupported or mixed",
                what=f"precisions={precisions!r}",
                how="encode every component as unsigned 8-bit or every component as unsigned 16-bit",
            )
        )
    return _new_raster_header(
        "JPEG2000",
        width=width,
        height=height,
        component_count=components,
        dtype=f"uint{precisions[0]}",
    )


def _parse_j2k(data: bytes) -> ImageHeader:
    if not data.startswith(b"\xff\x4f\xff\x51"):
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 codestream lacks the SOC and SIZ markers",
                what=f"signature={data[:4]!r}",
                how="pass a raw J2K codestream beginning with ff 4f ff 51",
            )
        )
    if len(data) < 6:
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 SIZ marker length is truncated",
                what=f"payload_length={len(data)}",
                how="pass a complete SIZ marker segment",
            )
        )
    length = struct.unpack_from(">H", data, 4)[0]
    if length < 41 or len(data) < 4 + length:
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 SIZ marker segment is truncated or too short",
                what=f"declared_length={length}, available={max(0, len(data) - 4)}",
                how="pass a complete SIZ segment with at least one component entry",
            )
        )
    fields = struct.unpack_from(">HIIIIIIIIH", data, 6)
    _, x_size, y_size, x_origin, y_origin, x_tile, y_tile, x_tile_origin, y_tile_origin, components = fields
    expected_length = 38 + 3 * components
    if length != expected_length:
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 SIZ length does not match its component count",
                what=f"length={length}, components={components}, expected={expected_length}",
                how="provide exactly three SIZ bytes for every component",
            )
        )
    if x_tile <= 0 or y_tile <= 0 or x_tile_origin > x_origin or y_tile_origin > y_origin:
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 SIZ tile geometry is invalid",
                what=(
                    f"tile={x_tile}x{y_tile}, tile_origin=({x_tile_origin},{y_tile_origin}), "
                    f"image_origin=({x_origin},{y_origin})"
                ),
                how="use positive tile dimensions whose origin does not exceed the image origin",
            )
        )
    entries = data[42 : 42 + 3 * components]
    precisions: list[int] = []
    for index in range(components):
        sample, x_subsampling, y_subsampling = entries[index * 3 : index * 3 + 3]
        if sample & 0x80 or x_subsampling != 1 or y_subsampling != 1:
            raise ValueError(
                _actionable_error(
                    why="the JPEG 2000 component is signed or subsampled outside the supported contract",
                    what=(f"component={index}, sample={sample}, subsampling=({x_subsampling},{y_subsampling})"),
                    how="encode unsigned components with 1x1 sampling",
                )
            )
        precisions.append((sample & 0x7F) + 1)
    if len(set(precisions)) != 1 or precisions[0] not in (8, 16):
        raise ValueError(
            _actionable_error(
                why="the JPEG 2000 component precisions are unsupported or mixed",
                what=f"precisions={tuple(precisions)!r}",
                how="encode every component as unsigned 8-bit or every component as unsigned 16-bit",
            )
        )
    return _new_raster_header(
        "JPEG2000",
        width=x_size - x_origin,
        height=y_size - y_origin,
        component_count=components,
        dtype=f"uint{precisions[0]}",
    )


def _parse_jpeg2000(source: Path | bytes) -> ImageHeader:
    data = source.read_bytes() if isinstance(source, Path) else source
    if data.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
        return _parse_jp2(data)
    return _parse_j2k(data)
