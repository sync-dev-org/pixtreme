"""OpenEXR headers, offset tables, and chunk ownership."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from pixtreme._core.errors import _actionable_error

if TYPE_CHECKING:
    from pixtreme._io.formats.exr.codec_b44 import _B44ChunkDescriptor
    from pixtreme._io.formats.exr.codec_dwa import _DwaChunkDescriptor
    from pixtreme._io.formats.exr.codec_piz import _PizChunkDescriptor
    from pixtreme._io.formats.exr.codec_pxr24 import _Pxr24ChunkDescriptor
    from pixtreme._io.formats.exr.codec_rle import _RleChunkDescriptor

_EXR_MAGIC = 20000630
_EXR_VERSION = 2
_EXR_TILED_FLAG = 0x200
_EXR_LONG_NAMES_FLAG = 0x400
_EXR_NON_IMAGE_FLAG = 0x800
_EXR_MULTIPART_FLAG = 0x1000
_EXR_SUPPORTED_VERSION_FLAGS = _EXR_TILED_FLAG | _EXR_LONG_NAMES_FLAG | _EXR_NON_IMAGE_FLAG | _EXR_MULTIPART_FLAG
_EXR_PART_TYPES = frozenset(("scanlineimage", "tiledimage", "deepscanline", "deeptile"))
_EXR_TILED_PART_TYPES = frozenset(("tiledimage", "deeptile"))
_EXR_DEEP_PART_TYPES = frozenset(("deepscanline", "deeptile"))
_EXR_MAX_INTEGER = (1 << 63) - 1
_EXR_THREADS_PER_BLOCK = 256
_EXR_MAX_GRID_Y = (1 << 16) - 1
_EXR_MAX_GRID_X = (1 << 16) - 1
_EXR_RESTORE_TILE_BYTES = 4096
_EXR_HOST_RESTORE_BATCH_BYTES = 2 * 1024 * 1024

_EXR_COMPRESSION_NAMES = {
    0: "none",
    1: "rle",
    2: "zips",
    3: "zip",
    4: "piz",
    5: "pxr24",
    6: "b44",
    7: "b44a",
    8: "dwaa",
    9: "dwab",
    10: "htj2k256",
    11: "htj2k32",
}
_EXR_COMPRESSION_CODES = {name: code for code, name in _EXR_COMPRESSION_NAMES.items()}
_EXR_LINES_PER_CHUNK = {
    "none": 1,
    "rle": 1,
    "zips": 1,
    "zip": 16,
    "piz": 32,
    "pxr24": 16,
    "b44": 32,
    "b44a": 32,
    "dwaa": 32,
    "dwab": 256,
    "htj2k256": 256,
    "htj2k32": 32,
}
_EXR_DTYPE_INFO = {
    0: ("uint32", 4),
    1: ("float16", 2),
    2: ("float32", 4),
}


@dataclass(frozen=True)
class _ExrAttribute:
    name: str
    attribute_type: str
    payload: bytes = field(repr=False)
    payload_start: int
    payload_end: int


@dataclass(frozen=True)
class _ExrSamplingGeometry:
    data_window: tuple[int, int, int, int]
    x_sampling: int
    y_sampling: int
    x_start: int
    y_start: int
    width: int
    height: int

    @property
    def x_coordinates(self) -> range:
        return range(self.x_start, self.x_start + self.width * self.x_sampling, self.x_sampling)

    @property
    def y_coordinates(self) -> range:
        return range(self.y_start, self.y_start + self.height * self.y_sampling, self.y_sampling)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class _ExrChannel:
    name: str
    pixel_type: int
    dtype: str
    bytes_per_sample: int
    perceptually_linear: bool
    x_sampling: int
    y_sampling: int
    sampling: _ExrSamplingGeometry | None = None


@dataclass(frozen=True)
class _ExrTileDescription:
    x_size: int
    y_size: int
    level_mode: int
    rounding_mode: int


@dataclass(frozen=True)
class _ExrTileLevel:
    level_x: int
    level_y: int
    width: int
    height: int
    tile_columns: int
    tile_rows: int
    table_start: int
    table_count: int
    offsets: tuple[int, ...] = ()
    chunks: tuple[_ExrChunk, ...] = ()


@dataclass(frozen=True)
class _ExrPart:
    name: str
    image_type: str
    attributes: Mapping[str, _ExrAttribute]
    channels: tuple[_ExrChannel, ...]
    compression: str
    line_order: int
    data_window: tuple[int, int, int, int]
    display_window: tuple[int, int, int, int]
    index: int = 0
    deep: bool = False
    tile_description: _ExrTileDescription | None = None
    levels: tuple[_ExrTileLevel, ...] = ()
    expected_chunk_count: int = 0
    offset_table: tuple[int, ...] = ()
    chunks: tuple[_ExrChunk, ...] = ()


@dataclass(frozen=True)
class _ExrChunk:
    y: int
    row_start: int
    row_count: int
    packed_size: int
    payload_start: int
    payload_end: int
    expected_size: int
    raw_stored: bool
    dwa: _DwaChunkDescriptor | None = None
    rle: _RleChunkDescriptor | None = None
    pxr24: _Pxr24ChunkDescriptor | None = None
    b44: _B44ChunkDescriptor | None = None
    piz: _PizChunkDescriptor | None = None
    part_index: int = 0
    chunk_offset: int = 0
    span_start: int = 0
    span_end: int = 0
    kind: str = "scanline"
    tile_x: int | None = None
    tile_y: int | None = None
    level_x: int | None = None
    level_y: int | None = None
    packed_sample_table_size: int | None = None
    unpacked_size: int | None = None


@dataclass(frozen=True)
class _ExrContainer:
    data: bytes = field(repr=False)
    magic: int
    version_field: int
    version: int
    version_flags: int
    multipart: bool
    tiled: bool
    deep: bool
    parts: tuple[_ExrPart, ...]
    compression: str
    line_order: int
    data_window: tuple[int, int, int, int]
    display_window: tuple[int, int, int, int]
    lines_per_chunk: int
    expected_chunk_count: int
    offset_table: tuple[int, ...]
    chunks: tuple[_ExrChunk, ...]


@dataclass(frozen=True)
class _ExrReadChunks:
    host_staging: np.ndarray = field(repr=False)
    host_decoded: np.ndarray = field(repr=False)
    stage_offsets: np.ndarray
    stage_sizes: np.ndarray
    decoded_offsets: np.ndarray
    decoded_sizes: np.ndarray
    compressed: np.ndarray
    expected_adler: np.ndarray


class _ExrGpuError(RuntimeError):
    """An already-classified EXR integrity or GPU I/O failure."""


def _parser_error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _gpu_error(*, why: str, what: str, how: str) -> _ExrGpuError:
    return _ExrGpuError(_actionable_error(why=why, what=what, how=how))


def _read_cstring(data: bytes, offset: int, *, limit: int, field_name: str) -> tuple[str, int]:
    if offset >= limit:
        raise _parser_error(
            why=f"the EXR {field_name} is truncated before its null terminator",
            what=f"offset={offset}, limit={limit}",
            how="provide a complete null-terminated EXR header field",
        )
    end = data.find(b"\x00", offset, limit)
    if end < 0:
        raise _parser_error(
            why=f"the EXR {field_name} has no null terminator within the file bounds",
            what=f"offset={offset}, limit={limit}",
            how="provide a complete null-terminated EXR header field",
        )
    try:
        value = data[offset:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise _parser_error(
            why=f"the EXR {field_name} is not valid UTF-8",
            what=f"offset={offset}, bytes={data[offset:end]!r}",
            how="encode EXR header and channel names as valid UTF-8",
        ) from error
    return value, end + 1


def _parse_attributes(data: bytes, offset: int) -> tuple[dict[str, _ExrAttribute], int]:
    attributes: dict[str, _ExrAttribute] = {}
    while True:
        name, offset = _read_cstring(data, offset, limit=len(data), field_name="attribute name")
        if not name:
            return attributes, offset
        if name in attributes:
            raise _parser_error(
                why="the EXR header repeats an attribute name",
                what=f"attribute={name!r}",
                how="store each EXR header attribute exactly once per part",
            )
        attribute_type, offset = _read_cstring(data, offset, limit=len(data), field_name="attribute type")
        if offset + 4 > len(data):
            raise _parser_error(
                why="the EXR attribute size field is truncated",
                what=f"attribute={name!r}, offset={offset}, file_size={len(data)}",
                how="provide the complete four-byte attribute size and payload",
            )
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        payload_start = offset
        payload_end = payload_start + size
        if payload_end > len(data):
            raise _parser_error(
                why="the EXR attribute payload extends beyond the file bounds",
                what=f"attribute={name!r}, payload={payload_start}:{payload_end}, file_size={len(data)}",
                how="provide the complete attribute payload declared by its size",
            )
        attributes[name] = _ExrAttribute(
            name=name,
            attribute_type=attribute_type,
            payload=data[payload_start:payload_end],
            payload_start=payload_start,
            payload_end=payload_end,
        )
        offset = payload_end


def _parse_channel_list(attribute: _ExrAttribute) -> tuple[_ExrChannel, ...]:
    payload = attribute.payload
    channels: list[_ExrChannel] = []
    names: set[str] = set()
    offset = 0
    while offset < len(payload) and payload[offset] != 0:
        end = payload.find(b"\x00", offset)
        if end < 0:
            raise _parser_error(
                why="the EXR channel list ends before its channel name is terminated",
                what=f"offset={offset}, payload_size={len(payload)}",
                how="terminate every EXR channel name and the channel list with null bytes",
            )
        try:
            name = payload[offset:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise _parser_error(
                why="the EXR channel name is not valid UTF-8",
                what=f"bytes={payload[offset:end]!r}",
                how="encode EXR channel names as valid UTF-8",
            ) from error
        offset = end + 1
        if offset + 16 > len(payload):
            raise _parser_error(
                why="the EXR channel list ends before its channel entry is complete",
                what=f"channel={name!r}, entry_bytes={len(payload) - offset}",
                how="pass an EXR with complete 16-byte metadata for every channel entry",
            )
        pixel_type, perceptually_linear, x_sampling, y_sampling = struct.unpack_from("<iB3xii", payload, offset)
        dtype_info = _EXR_DTYPE_INFO.get(pixel_type)
        if dtype_info is None:
            raise _parser_error(
                why="the EXR channel uses an unsupported pixel type",
                what=f"channel={name!r}, pixel_type={pixel_type}",
                how="encode the channel as UINT, HALF, or FLOAT",
            )
        if name in names:
            raise _parser_error(
                why="the EXR channel list repeats a channel name",
                what=f"channel={name!r}",
                how="store each channel label exactly once in a part",
            )
        if x_sampling <= 0 or y_sampling <= 0:
            raise _parser_error(
                why="the EXR channel sampling factors must be positive",
                what=f"channel={name!r}, xSampling={x_sampling}, ySampling={y_sampling}",
                how="encode positive integer xSampling and ySampling values",
            )
        names.add(name)
        channels.append(
            _ExrChannel(
                name=name,
                pixel_type=pixel_type,
                dtype=dtype_info[0],
                bytes_per_sample=dtype_info[1],
                perceptually_linear=bool(perceptually_linear),
                x_sampling=x_sampling,
                y_sampling=y_sampling,
            )
        )
        offset += 16
    if offset >= len(payload) or payload[offset] != 0:
        raise _parser_error(
            why="the EXR channel list lacks its final null terminator",
            what=f"payload_size={len(payload)}, offset={offset}",
            how="terminate the EXR channel list with an additional null byte",
        )
    if offset + 1 != len(payload):
        raise _parser_error(
            why="the EXR channel list contains bytes after its final terminator",
            what=f"trailing_bytes={len(payload) - offset - 1}",
            how="end the channel-list payload immediately after its final null byte",
        )
    if not channels:
        raise _parser_error(
            why="the EXR channel list contains no channels",
            what="channels=()",
            how="provide at least one UINT, HALF, or FLOAT channel",
        )
    return tuple(channels)


def _fixed_payload(
    attributes: Mapping[str, _ExrAttribute],
    name: str,
    *,
    attribute_type: str,
    size: int,
) -> bytes:
    attribute = attributes.get(name)
    if attribute is None:
        raise _parser_error(
            why="the EXR part header lacks a required container attribute",
            what=f"attribute={name!r}, attributes={tuple(attributes)!r}",
            how=f"provide the required {name} attribute",
        )
    if attribute.attribute_type != attribute_type or len(attribute.payload) != size:
        raise _parser_error(
            why="the EXR container attribute has an invalid type or payload size",
            what=(
                f"attribute={name!r}, type={attribute.attribute_type!r}, size={len(attribute.payload)}, "
                f"expected_type={attribute_type!r}, expected_size={size}"
            ),
            how="encode the attribute with its standard OpenEXR type and fixed payload size",
        )
    return attribute.payload


def _parse_box(payload: bytes, *, name: str) -> tuple[int, int, int, int]:
    values = cast(tuple[int, int, int, int], struct.unpack("<iiii", payload))
    x_min, y_min, x_max, y_max = values
    if x_max < x_min or y_max < y_min:
        raise _parser_error(
            why=f"the EXR {name} has inverted bounds",
            what=f"{name}={values!r}",
            how="use inclusive minimum coordinates no greater than the maximum coordinates",
        )
    return values


def _decode_string(attribute: _ExrAttribute | None, *, default: str, part_index: int) -> str:
    if attribute is None:
        return default
    if attribute.attribute_type != "string":
        raise _parser_error(
            why="the EXR part string attribute does not use the string type",
            what=f"part={part_index}, attribute={attribute.name!r}, type={attribute.attribute_type!r}",
            how="encode part name and type attributes with the standard OpenEXR string type",
        )
    try:
        return attribute.payload.rstrip(b"\x00").decode("utf-8")
    except UnicodeDecodeError as error:
        raise _parser_error(
            why="the EXR string attribute is not valid UTF-8",
            what=f"part={part_index}, attribute={attribute.name!r}, payload={attribute.payload!r}",
            how="encode EXR string attributes as valid UTF-8",
        ) from error


def _sampling_axis(minimum: int, maximum: int, sampling: int) -> tuple[int, int]:
    start = minimum + (-minimum % sampling)
    if start > maximum:
        return start, 0
    return start, (maximum - start) // sampling + 1


def _sampling_geometry(
    data_window: tuple[int, int, int, int],
    *,
    x_sampling: int,
    y_sampling: int,
) -> _ExrSamplingGeometry:
    x_min, y_min, x_max, y_max = data_window
    x_start, width = _sampling_axis(x_min, x_max, x_sampling)
    y_start, height = _sampling_axis(y_min, y_max, y_sampling)
    return _ExrSamplingGeometry(
        data_window=data_window,
        x_sampling=x_sampling,
        y_sampling=y_sampling,
        x_start=x_start,
        y_start=y_start,
        width=width,
        height=height,
    )


def _parse_chunk_count(attributes: Mapping[str, _ExrAttribute], *, part_index: int) -> int | None:
    attribute = attributes.get("chunkCount")
    if attribute is None:
        return None
    if attribute.attribute_type != "int" or len(attribute.payload) != 4:
        raise _parser_error(
            why="the EXR part chunkCount attribute has an invalid type or payload size",
            what=(
                f"part={part_index}, attribute='chunkCount', type={attribute.attribute_type!r}, "
                f"size={len(attribute.payload)}, expected_type='int', expected_size=4"
            ),
            how="encode chunkCount as one four-byte OpenEXR int attribute",
        )
    payload = attribute.payload
    count = int(struct.unpack("<i", payload)[0])
    if count < 0:
        raise _parser_error(
            why="the EXR part declares a negative chunk count",
            what=f"part={part_index}, chunkCount={count}",
            how="encode the number of chunks in the part as a non-negative integer",
        )
    return count


def _parse_tile_description(
    attributes: Mapping[str, _ExrAttribute],
    *,
    part_index: int,
    required: bool,
) -> _ExrTileDescription | None:
    attribute = attributes.get("tiles")
    if attribute is None:
        if required:
            raise _parser_error(
                why="the EXR tiled part header lacks its required tile description",
                what=f"part={part_index}, attribute='tiles'",
                how="provide tiledimage and deeptile parts with a tiles attribute",
            )
        return None
    if attribute.attribute_type != "tiledesc" or len(attribute.payload) != 9:
        raise _parser_error(
            why="the EXR part tiles attribute has an invalid type or payload size",
            what=(
                f"part={part_index}, attribute='tiles', type={attribute.attribute_type!r}, "
                f"size={len(attribute.payload)}, expected_type='tiledesc', expected_size=9"
            ),
            how="encode tiles as one nine-byte OpenEXR tiledesc attribute",
        )
    payload = attribute.payload
    x_size, y_size, mode = struct.unpack("<IIB", payload)
    level_mode = mode & 0x0F
    rounding_mode = mode >> 4
    if x_size == 0 or y_size == 0:
        raise _parser_error(
            why="the EXR tile description uses a zero tile dimension",
            what=f"part={part_index}, tile_size=({x_size}, {y_size})",
            how="encode positive tile width and height values",
        )
    if level_mode not in (0, 1, 2) or rounding_mode not in (0, 1):
        raise _parser_error(
            why="the EXR tile description uses an unknown level or rounding mode",
            what=f"part={part_index}, mode=0x{mode:02x}, level_mode={level_mode}, rounding_mode={rounding_mode}",
            how="use ONE_LEVEL, MIPMAP, or RIPMAP with ROUND_DOWN or ROUND_UP",
        )
    return _ExrTileDescription(
        x_size=x_size,
        y_size=y_size,
        level_mode=level_mode,
        rounding_mode=rounding_mode,
    )


def _validate_deep_part_attributes(attributes: Mapping[str, _ExrAttribute], *, part_index: int) -> None:
    payloads: dict[str, bytes] = {}
    for name in ("version", "maxSamplesPerPixel"):
        attribute = attributes.get(name)
        if attribute is None:
            raise _parser_error(
                why="the EXR deep part header lacks a required deep attribute",
                what=f"part={part_index}, attribute={name!r}",
                how="provide deep parts with integer version and maxSamplesPerPixel attributes",
            )
        if attribute.attribute_type != "int" or len(attribute.payload) != 4:
            raise _parser_error(
                why="the EXR deep part attribute has an invalid type or payload size",
                what=(
                    f"part={part_index}, attribute={name!r}, type={attribute.attribute_type!r}, "
                    f"size={len(attribute.payload)}, expected_type='int', expected_size=4"
                ),
                how="encode deep version and maxSamplesPerPixel as four-byte OpenEXR int attributes",
            )
        payloads[name] = attribute.payload
    version = struct.unpack("<i", payloads["version"])[0]
    if version != 1:
        raise _parser_error(
            why="the EXR deep part uses an unsupported deep data version",
            what=f"part={part_index}, version={version}",
            how="encode deep scanline and deep tile parts with version=1",
        )
    max_samples = struct.unpack("<i", payloads["maxSamplesPerPixel"])[0]
    if max_samples < -1:
        raise _parser_error(
            why="the EXR deep part declares an invalid maximum sample count",
            what=f"part={part_index}, maxSamplesPerPixel={max_samples}",
            how="encode -1 for unknown or a non-negative maximum sample count",
        )


def _level_size(size: int, level: int, rounding_mode: int) -> int:
    divisor = 1 << level
    if rounding_mode:
        return max(1, (size + divisor - 1) // divisor)
    return max(1, size // divisor)


def _level_count(size: int, rounding_mode: int) -> int:
    count = 1
    while _level_size(size, count - 1, rounding_mode) > 1:
        count += 1
    return count


def _tile_levels(
    data_window: tuple[int, int, int, int],
    description: _ExrTileDescription,
) -> tuple[_ExrTileLevel, ...]:
    x_min, y_min, x_max, y_max = data_window
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    x_levels = _level_count(width, description.rounding_mode)
    y_levels = _level_count(height, description.rounding_mode)
    identities: tuple[tuple[int, int], ...]
    if description.level_mode == 0:
        identities = ((0, 0),)
    elif description.level_mode == 1:
        identities = tuple((level, level) for level in range(max(x_levels, y_levels)))
    else:
        identities = tuple((level_x, level_y) for level_y in range(y_levels) for level_x in range(x_levels))
    levels: list[_ExrTileLevel] = []
    table_start = 0
    for level_x, level_y in identities:
        level_width = _level_size(width, level_x, description.rounding_mode)
        level_height = _level_size(height, level_y, description.rounding_mode)
        tile_columns = (level_width + description.x_size - 1) // description.x_size
        tile_rows = (level_height + description.y_size - 1) // description.y_size
        table_count = _checked_product(tile_columns, tile_rows, context=f"tile level {(level_x, level_y)}")
        levels.append(
            _ExrTileLevel(
                level_x=level_x,
                level_y=level_y,
                width=level_width,
                height=level_height,
                tile_columns=tile_columns,
                tile_rows=tile_rows,
                table_start=table_start,
                table_count=table_count,
            )
        )
        table_start += table_count
    return tuple(levels)


def _parse_part(
    attributes: Mapping[str, _ExrAttribute],
    *,
    tiled_flag: bool,
    non_image_flag: bool = False,
    multipart: bool = False,
    part_index: int = 0,
) -> _ExrPart:
    channel_attribute = attributes.get("channels")
    data_window_attribute = attributes.get("dataWindow")
    if channel_attribute is None or data_window_attribute is None:
        raise _parser_error(
            why="the EXR part header lacks channels or dataWindow",
            what=f"attributes={tuple(attributes)!r}",
            how="provide every EXR part with channels and dataWindow attributes",
        )
    if channel_attribute.attribute_type != "chlist":
        raise _parser_error(
            why="the EXR channels attribute does not use the chlist type",
            what=f"type={channel_attribute.attribute_type!r}",
            how="encode channels with the standard OpenEXR chlist attribute type",
        )
    channels = _parse_channel_list(channel_attribute)
    data_window = _parse_box(
        _fixed_payload(attributes, "dataWindow", attribute_type="box2i", size=16), name="dataWindow"
    )
    display_window = _parse_box(
        _fixed_payload(attributes, "displayWindow", attribute_type="box2i", size=16), name="displayWindow"
    )
    compression_code = _fixed_payload(attributes, "compression", attribute_type="compression", size=1)[0]
    compression = _EXR_COMPRESSION_NAMES.get(compression_code)
    if compression is None:
        raise _parser_error(
            why="the EXR compression attribute uses an unknown code",
            what=f"compression={compression_code}",
            how="encode the image with a compression code supported by OpenEXR",
        )
    line_order = _fixed_payload(attributes, "lineOrder", attribute_type="lineOrder", size=1)[0]
    if line_order not in (0, 1, 2):
        raise _parser_error(
            why="the EXR lineOrder attribute uses an unknown code",
            what=f"lineOrder={line_order}",
            how="use increasing, decreasing, or random line order",
        )
    requires_layout_identity = multipart or non_image_flag
    if requires_layout_identity:
        for required_attribute in ("name", "type", "chunkCount"):
            if required_attribute not in attributes:
                raise _parser_error(
                    why="the EXR part header lacks a required layout attribute",
                    what=f"part={part_index}, attribute={required_attribute!r}",
                    how="provide every multipart or deep header with name, type, and chunkCount",
                )

    default_type = "tiledimage" if tiled_flag else "scanlineimage"
    image_type_attribute = attributes.get("type")
    image_type = _decode_string(image_type_attribute, default=default_type, part_index=part_index)
    if image_type not in _EXR_PART_TYPES:
        raise _parser_error(
            why="the EXR part uses an unknown type token",
            what=f"part={part_index}, type={image_type!r}",
            how="use scanlineimage, tiledimage, deepscanline, or deeptile",
        )
    deep = image_type in _EXR_DEEP_PART_TYPES
    if not multipart and deep != non_image_flag:
        raise _parser_error(
            why="the EXR non-image flag disagrees with the single-part type",
            what=f"part={part_index}, type={image_type!r}, non_image_flag={non_image_flag}",
            how="set the non-image flag exactly when the single part is deepscanline or deeptile",
        )
    if multipart and deep and not non_image_flag:
        raise _parser_error(
            why="the EXR non-image flag disagrees with a multipart deep type",
            what=f"part={part_index}, type={image_type!r}, non_image_flag={non_image_flag}",
            how="set the non-image flag when any multipart header is deepscanline or deeptile",
        )
    if not multipart and not deep and (image_type == "tiledimage") != tiled_flag:
        raise _parser_error(
            why="the EXR single-tile flag disagrees with the single-part type",
            what=f"part={part_index}, type={image_type!r}, tiled_flag={tiled_flag}",
            how="set the single-tile flag exactly when the regular single part is tiledimage",
        )
    if deep:
        _validate_deep_part_attributes(attributes, part_index=part_index)
    channels = tuple(
        replace(
            channel,
            sampling=_sampling_geometry(
                data_window,
                x_sampling=channel.x_sampling,
                y_sampling=channel.y_sampling,
            ),
        )
        for channel in channels
    )
    if image_type not in _EXR_TILED_PART_TYPES and "tiles" in attributes:
        raise _parser_error(
            why="the EXR tiles attribute appears on a non-tiled part type",
            what=f"part={part_index}, type={image_type!r}, attribute='tiles'",
            how="attach the tiles attribute only to tiledimage and deeptile parts",
        )
    tile_description = _parse_tile_description(
        attributes,
        part_index=part_index,
        required=image_type in _EXR_TILED_PART_TYPES,
    )
    levels = _tile_levels(data_window, tile_description) if tile_description is not None else ()
    x_min, y_min, x_max, y_max = data_window
    height = y_max - y_min + 1
    if levels:
        derived_chunk_count = sum(level.table_count for level in levels)
    else:
        lines_per_chunk = _EXR_LINES_PER_CHUNK[compression]
        derived_chunk_count = (height + lines_per_chunk - 1) // lines_per_chunk
    declared_chunk_count = _parse_chunk_count(attributes, part_index=part_index)
    if declared_chunk_count is not None and declared_chunk_count != derived_chunk_count:
        raise _parser_error(
            why="the EXR part chunk count disagrees with its data window and layout",
            what=(
                f"part={part_index}, declared={declared_chunk_count}, derived={derived_chunk_count}, "
                f"type={image_type!r}, dataWindow={data_window!r}"
            ),
            how="encode one offset-table entry for every declared scanline block or tile",
        )
    return _ExrPart(
        name=_decode_string(attributes.get("name"), default="", part_index=part_index),
        image_type=image_type,
        attributes=attributes,
        channels=channels,
        compression=compression,
        line_order=line_order,
        data_window=data_window,
        display_window=display_window,
        index=part_index,
        deep=deep,
        tile_description=tile_description,
        levels=levels,
        expected_chunk_count=declared_chunk_count if declared_chunk_count is not None else derived_chunk_count,
    )


def _checked_product(*values: int, context: str) -> int:
    result = 1
    for value in values:
        if value < 0 or (value and result > _EXR_MAX_INTEGER // value):
            raise _parser_error(
                why="the EXR container size calculation overflows signed 64-bit bounds",
                what=f"context={context}, factors={values!r}",
                how="use image dimensions and channel counts representable within 64-bit byte offsets",
            )
        result *= value
    return result


def _parse_codec_chunk(
    data: bytes,
    part: _ExrPart,
    chunk: _ExrChunk,
    *,
    width: int,
    lines_per_chunk: int,
    expected_size: int | None = None,
    raw_stored: bool | None = None,
    part_index: int | None = None,
) -> _ExrChunk:
    """Apply decoder geometry and its codec descriptor in one immutable chunk replacement."""
    compression = part.compression
    raw_size = chunk.expected_size if expected_size is None else expected_size
    stored_raw = chunk.raw_stored if raw_stored is None else raw_stored
    dwa: _DwaChunkDescriptor | None = None
    rle: _RleChunkDescriptor | None = None
    pxr24: _Pxr24ChunkDescriptor | None = None
    b44: _B44ChunkDescriptor | None = None
    piz: _PizChunkDescriptor | None = None
    if compression in ("dwaa", "dwab"):
        from pixtreme._io.formats.exr.codec_dwa import _parse_dwa_chunk_descriptor

        dwa = _parse_dwa_chunk_descriptor(
            data,
            part.channels,
            width=width,
            lines_per_chunk=lines_per_chunk,
            chunk_y=chunk.y,
            row_count=chunk.row_count,
            payload_start=chunk.payload_start,
            payload_end=chunk.payload_end,
            expected_size=raw_size,
            raw_stored=stored_raw,
        )
    elif compression == "rle":
        from pixtreme._io.formats.exr.codec_rle import _parse_rle_chunk_descriptor

        rle = _parse_rle_chunk_descriptor(
            data,
            part.channels,
            width=width,
            lines_per_chunk=lines_per_chunk,
            chunk_y=chunk.y,
            row_start=chunk.row_start,
            row_count=chunk.row_count,
            payload_start=chunk.payload_start,
            payload_end=chunk.payload_end,
            expected_raw_size=raw_size,
            raw_stored=stored_raw,
        )
    elif compression == "pxr24":
        from pixtreme._io.formats.exr.codec_pxr24 import _parse_pxr24_chunk_descriptor

        pxr24 = _parse_pxr24_chunk_descriptor(
            data,
            part.channels,
            width=width,
            lines_per_chunk=lines_per_chunk,
            chunk_y=chunk.y,
            row_start=chunk.row_start,
            row_count=chunk.row_count,
            payload_start=chunk.payload_start,
            payload_end=chunk.payload_end,
            expected_raw_size=raw_size,
            raw_stored=stored_raw,
        )
    elif compression in ("b44", "b44a"):
        from pixtreme._io.formats.exr.codec_b44 import _parse_b44_chunk_descriptor

        b44 = _parse_b44_chunk_descriptor(
            compression,
            data,
            part.channels,
            width=width,
            lines_per_chunk=lines_per_chunk,
            chunk_y=chunk.y,
            row_start=chunk.row_start,
            row_count=chunk.row_count,
            payload_start=chunk.payload_start,
            payload_end=chunk.payload_end,
            expected_raw_size=raw_size,
            raw_stored=stored_raw,
        )
    elif compression == "piz":
        from pixtreme._io.formats.exr.codec_piz import _piz_chunk_descriptor

        piz = _piz_chunk_descriptor(
            data,
            part,
            width=width,
            lines_per_chunk=lines_per_chunk,
            chunk_y=chunk.y,
            row_start=chunk.row_start,
            row_count=chunk.row_count,
            payload_start=chunk.payload_start,
            payload_end=chunk.payload_end,
            expected_packed_size=raw_size,
            raw_stored=stored_raw,
        )
    elif expected_size is None and raw_stored is None and part_index is None:
        return chunk
    return replace(
        chunk,
        expected_size=raw_size,
        raw_stored=stored_raw,
        part_index=chunk.part_index if part_index is None else part_index,
        dwa=dwa,
        rle=rle,
        pxr24=pxr24,
        b44=b44,
        piz=piz,
    )


def _codec_gpu_eligible(
    part: _ExrPart,
    chunks: Sequence[_ExrChunk],
    *,
    multipart: bool,
    tiled: bool,
    deep: bool,
) -> bool:
    """Use each compression codec's single eligibility rule for both decoder paths."""
    if multipart or tiled or deep or part.image_type != "scanlineimage":
        return False
    compression = part.compression
    if compression == "none":
        from pixtreme._io.formats.exr.codec_none import _none_gpu_eligible

        return _none_gpu_eligible(part.channels)
    if compression in ("zip", "zips"):
        from pixtreme._io.formats.exr.codec_zip import _zip_gpu_eligible

        return _zip_gpu_eligible(part.channels)
    if compression == "rle":
        from pixtreme._io.formats.exr.codec_rle import _rle_gpu_eligible

        return _rle_gpu_eligible(part.channels, chunks)
    if compression == "pxr24":
        from pixtreme._io.formats.exr.codec_pxr24 import _pxr24_gpu_eligible

        return _pxr24_gpu_eligible(part.channels, chunks)
    if compression in ("b44", "b44a"):
        from pixtreme._io.formats.exr.codec_b44 import _b44_gpu_eligible

        return _b44_gpu_eligible(part.channels, chunks)
    if compression == "piz":
        from pixtreme._io.formats.exr.codec_piz import _piz_gpu_eligible

        return _piz_gpu_eligible(part.channels, chunks)
    if compression in ("dwaa", "dwab"):
        from pixtreme._io.formats.exr.codec_dwa import _dwa_gpu_eligible

        return _dwa_gpu_eligible(part.channels, chunks)
    return False


def _container_gpu_eligible(container: _ExrContainer) -> bool:
    return _codec_gpu_eligible(
        container.parts[0],
        container.chunks,
        multipart=container.multipart,
        tiled=container.tiled,
        deep=container.deep,
    )


def _parse_candidate_chunks(
    data: bytes,
    offset: int,
    part: _ExrPart,
    *,
    lines_per_chunk: int,
) -> tuple[tuple[int, ...], tuple[_ExrChunk, ...]]:
    x_min, y_min, x_max, y_max = part.data_window
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    expected_chunk_count = (height + lines_per_chunk - 1) // lines_per_chunk
    table_size = _checked_product(expected_chunk_count, 8, context="offset table")
    table_end = offset + table_size
    if table_end > len(data):
        raise _parser_error(
            why="the EXR offset table is truncated",
            what=f"table={offset}:{table_end}, file_size={len(data)}, expected_chunks={expected_chunk_count}",
            how="provide one complete eight-byte offset for every scanline chunk",
        )
    offset_table = tuple(struct.unpack_from("<Q", data, offset + index * 8)[0] for index in range(expected_chunk_count))
    if len(set(offset_table)) != len(offset_table):
        raise _parser_error(
            why="the EXR offset table contains duplicate chunk offsets",
            what=f"offsets={offset_table!r}",
            how="point each offset-table entry at one distinct scanline chunk",
        )
    row_bytes = 0
    for channel in part.channels:
        channel_row_bytes = _checked_product(width, channel.bytes_per_sample, context=f"channel {channel.name!r} row")
        if row_bytes > _EXR_MAX_INTEGER - channel_row_bytes:
            raise _parser_error(
                why="the EXR scanline byte count overflows signed 64-bit bounds",
                what=f"row_bytes={row_bytes}, channel={channel.name!r}, channel_bytes={channel_row_bytes}",
                how="use dimensions and a channel layout representable within 64-bit byte offsets",
            )
        row_bytes += channel_row_bytes
    chunks: list[_ExrChunk] = []
    seen_rows: set[int] = set()
    spans: list[tuple[int, int]] = []
    for chunk_offset in offset_table:
        if chunk_offset < table_end or chunk_offset + 8 > len(data):
            raise _parser_error(
                why="the EXR chunk offset does not point to a complete chunk header after the offset table",
                what=f"offset={chunk_offset}, table_end={table_end}, file_size={len(data)}",
                how="point every offset-table entry at an in-file scanline chunk header",
            )
        y, packed_size = struct.unpack_from("<ii", data, chunk_offset)
        if packed_size < 0:
            raise _parser_error(
                why="the EXR scanline chunk declares a negative payload size",
                what=f"offset={chunk_offset}, y={y}, packed_size={packed_size}",
                how="encode a non-negative packed payload size",
            )
        payload_start = chunk_offset + 8
        payload_end = payload_start + packed_size
        if payload_end > len(data):
            raise _parser_error(
                why="the EXR scanline chunk payload extends beyond the file bounds",
                what=f"offset={chunk_offset}, payload={payload_start}:{payload_end}, file_size={len(data)}",
                how="provide the complete packed payload declared by the chunk header",
            )
        row_start = y - y_min
        if row_start < 0 or row_start >= height or row_start % lines_per_chunk:
            raise _parser_error(
                why="the EXR scanline chunk y coordinate is outside or misaligned to the data window",
                what=f"y={y}, dataWindow_y={y_min}:{y_max}, lines_per_chunk={lines_per_chunk}",
                how="align each chunk y coordinate to the data-window minimum and compression block size",
            )
        row_count = min(lines_per_chunk, height - row_start)
        expected_size = _checked_product(row_bytes, row_count, context=f"chunk y={y} uncompressed bytes")
        if part.compression == "none" and packed_size != expected_size:
            raise _parser_error(
                why="the uncompressed EXR chunk size differs from its channel and row layout",
                what=f"y={y}, packed_size={packed_size}, expected_size={expected_size}",
                how="store exactly the expected scanline bytes for NONE compression",
            )
        if part.compression != "none" and packed_size > expected_size:
            raise _parser_error(
                why=f"the {part.compression.upper()} compressed EXR chunk is larger than its expected uncompressed bytes",
                what=f"y={y}, packed_size={packed_size}, expected_size={expected_size}",
                how="store the raw bytes when compression is not smaller than the source chunk",
            )
        if row_start in seen_rows:
            raise _parser_error(
                why="the EXR chunks repeat an output row range",
                what=f"y={y}, row_start={row_start}",
                how="provide exactly one scanline chunk for every expected output row range",
            )
        seen_rows.add(row_start)
        spans.append((chunk_offset, payload_end))
        raw_stored = (
            part.compression == "none"
            or packed_size == expected_size
            or (part.compression == "piz" and packed_size == 0)
        )
        chunk = _ExrChunk(
            y=y,
            row_start=row_start,
            row_count=row_count,
            packed_size=packed_size,
            payload_start=payload_start,
            payload_end=payload_end,
            expected_size=expected_size,
            raw_stored=raw_stored,
            part_index=part.index,
            chunk_offset=chunk_offset,
            span_start=chunk_offset,
            span_end=payload_end,
        )
        chunks.append(_parse_codec_chunk(data, part, chunk, width=width, lines_per_chunk=lines_per_chunk))
    expected_rows = set(range(0, height, lines_per_chunk))
    if seen_rows != expected_rows:
        raise _parser_error(
            why="the EXR chunks leave output row ranges missing or duplicated",
            what=f"observed={tuple(sorted(seen_rows))!r}, expected={tuple(sorted(expected_rows))!r}",
            how="provide exactly one aligned chunk for every row block in the data window",
        )
    ordered_spans = sorted(spans)
    for previous, current in zip(ordered_spans, ordered_spans[1:], strict=False):
        if previous[1] > current[0]:
            raise _parser_error(
                why="the EXR scanline chunk spans intersect",
                what=f"previous={previous!r}, current={current!r}",
                how="store each scanline chunk in a distinct non-overlapping file span",
            )
    return offset_table, tuple(sorted(chunks, key=lambda chunk: chunk.row_start))


def _part_chunk_identities(part: _ExrPart) -> tuple[tuple[int, ...], ...]:
    if part.levels:
        return tuple(
            (tile_x, tile_y, level.level_x, level.level_y)
            for level in part.levels
            for tile_y in range(level.tile_rows)
            for tile_x in range(level.tile_columns)
        )
    _, y_min, _, y_max = part.data_window
    lines_per_chunk = _EXR_LINES_PER_CHUNK[part.compression]
    return tuple((y,) for y in range(y_min, y_max + 1, lines_per_chunk))


def _chunk_identity_context(part: _ExrPart, identity: tuple[int, ...]) -> str:
    if len(identity) == 4:
        tile_x, tile_y, level_x, level_y = identity
        return f"part={part.index}, level={(level_x, level_y)}, tile={(tile_x, tile_y)}"
    return f"part={part.index}, chunk_y={identity[0]}"


def _require_chunk_bytes(data: bytes, cursor: int, size: int, *, context: str) -> None:
    if cursor < 0 or cursor + size > len(data):
        raise _parser_error(
            why="the EXR chunk header is truncated or outside the file bounds",
            what=f"{context}, header={cursor}:{cursor + size}, file_size={len(data)}",
            how="point the owning offset-table entry at a complete in-file chunk",
        )


def _scanline_expected_size(part: _ExrPart, *, y: int, row_count: int) -> int:
    total = 0
    for file_y in range(y, y + row_count):
        for channel in part.channels:
            sampling = channel.sampling
            if sampling is not None and file_y % channel.y_sampling == 0:
                channel_bytes = _checked_product(
                    sampling.width,
                    channel.bytes_per_sample,
                    context=f"part {part.index} channel {channel.name!r} sampled row",
                )
                if total > _EXR_MAX_INTEGER - channel_bytes:
                    raise _parser_error(
                        why="the EXR sampled scanline byte count overflows signed 64-bit bounds",
                        what=f"part={part.index}, channel={channel.name!r}, total={total}, add={channel_bytes}",
                        how="use sampled channel geometry representable within signed 64-bit byte offsets",
                    )
                total += channel_bytes
    return total


def _tile_expected_size(part: _ExrPart, identity: tuple[int, ...]) -> int | None:
    description = part.tile_description
    if description is None or any(channel.x_sampling != 1 or channel.y_sampling != 1 for channel in part.channels):
        return None
    tile_x, tile_y, level_x, level_y = identity
    level = next(item for item in part.levels if (item.level_x, item.level_y) == (level_x, level_y))
    stored_width = min(description.x_size, level.width - tile_x * description.x_size)
    stored_height = min(description.y_size, level.height - tile_y * description.y_size)
    bytes_per_pixel = sum(channel.bytes_per_sample for channel in part.channels)
    return _checked_product(
        stored_width,
        stored_height,
        bytes_per_pixel,
        context=f"part {part.index} tile {(tile_x, tile_y)} level {(level_x, level_y)}",
    )


def _parse_structural_chunk(
    data: bytes,
    chunk_offset: int,
    part: _ExrPart,
    identity: tuple[int, ...],
    *,
    multipart: bool,
    table_end: int,
) -> _ExrChunk:
    context = _chunk_identity_context(part, identity)
    if chunk_offset < table_end:
        raise _parser_error(
            why="the EXR chunk offset points into the header or offset tables",
            what=f"{context}, offset={chunk_offset}, table_end={table_end}",
            how="point every offset-table entry at its owning pixel chunk after all tables",
        )
    cursor = chunk_offset
    if multipart:
        _require_chunk_bytes(data, cursor, 4, context=context)
        observed_part = struct.unpack_from("<i", data, cursor)[0]
        cursor += 4
        if observed_part != part.index:
            raise _parser_error(
                why="the EXR multipart chunk belongs to a different part than its offset table",
                what=f"{context}, observed_part={observed_part}, chunk_offset={chunk_offset}",
                how="prefix each multipart chunk with the index of its owning part header and offset table",
            )

    deep = part.deep
    tiled = len(identity) == 4
    packed_sample_table_size: int | None = None
    unpacked_size: int | None = None
    if tiled:
        if deep:
            _require_chunk_bytes(data, cursor, 40, context=context)
            tile_x, tile_y, level_x, level_y, packed_sample_table_size, packed_size, unpacked_size = struct.unpack_from(
                "<iiiiQQQ", data, cursor
            )
            cursor += 40
            total_packed_size = packed_sample_table_size + packed_size
            if total_packed_size > _EXR_MAX_INTEGER:
                raise _parser_error(
                    why="the EXR deep tile payload size overflows signed 64-bit bounds",
                    what=(f"{context}, packed_sample_table={packed_sample_table_size}, packed_samples={packed_size}"),
                    how="encode deep tile payload sizes representable within signed 64-bit file offsets",
                )
            packed_size = total_packed_size
        else:
            _require_chunk_bytes(data, cursor, 20, context=context)
            tile_x, tile_y, level_x, level_y, packed_size = struct.unpack_from("<iiiii", data, cursor)
            cursor += 20
            if packed_size < 0:
                raise _parser_error(
                    why="the EXR tile declares a negative payload size",
                    what=f"{context}, packed_size={packed_size}",
                    how="encode a non-negative tile payload size",
                )
        observed_identity = (tile_x, tile_y, level_x, level_y)
        if observed_identity != identity:
            raise _parser_error(
                why="the EXR tile chunk identity disagrees with its offset-table position",
                what=f"{context}, observed_level={(level_x, level_y)}, observed_tile={(tile_x, tile_y)}",
                how="store exactly one in-range tile for every level-grid offset-table entry",
            )
        row_start = tile_y * (part.tile_description.y_size if part.tile_description is not None else 0)
        level = next(item for item in part.levels if (item.level_x, item.level_y) == (level_x, level_y))
        row_count = min(
            part.tile_description.y_size if part.tile_description is not None else 0,
            level.height - row_start,
        )
        expected_size = _tile_expected_size(part, identity)
        y = tile_y
        kind = "deep-tile" if deep else "tile"
    else:
        if deep:
            _require_chunk_bytes(data, cursor, 28, context=context)
            y, packed_sample_table_size, packed_size, unpacked_size = struct.unpack_from("<iQQQ", data, cursor)
            cursor += 28
            total_packed_size = packed_sample_table_size + packed_size
            if total_packed_size > _EXR_MAX_INTEGER:
                raise _parser_error(
                    why="the EXR deep scanline payload size overflows signed 64-bit bounds",
                    what=(f"{context}, packed_sample_table={packed_sample_table_size}, packed_samples={packed_size}"),
                    how="encode deep scanline payload sizes representable within signed 64-bit file offsets",
                )
            packed_size = total_packed_size
        else:
            _require_chunk_bytes(data, cursor, 8, context=context)
            y, packed_size = struct.unpack_from("<ii", data, cursor)
            cursor += 8
            if packed_size < 0:
                raise _parser_error(
                    why="the EXR scanline chunk declares a negative payload size",
                    what=f"{context}, packed_size={packed_size}",
                    how="encode a non-negative scanline payload size",
                )
        context = f"part={part.index}, chunk_y={y}"
        _, y_min, _, y_max = part.data_window
        lines_per_chunk = _EXR_LINES_PER_CHUNK[part.compression]
        row_start = y - y_min
        if row_start < 0 or row_start >= y_max - y_min + 1 or row_start % lines_per_chunk:
            raise _parser_error(
                why="the EXR scanline chunk y coordinate is outside or misaligned to its part data window",
                what=(f"{context}, dataWindow_y={(y_min, y_max)}, lines_per_chunk={lines_per_chunk}"),
                how="align each chunk y coordinate to the part data-window minimum and compression block size",
            )
        row_count = min(lines_per_chunk, y_max - y + 1)
        expected_size = unpacked_size if deep else _scanline_expected_size(part, y=y, row_count=row_count)
        kind = "deep-scanline" if deep else "scanline"

    payload_start = cursor
    payload_end = payload_start + packed_size
    if payload_end > len(data):
        raise _parser_error(
            why="the EXR chunk payload is truncated beyond the file bounds",
            what=f"{context}, payload={payload_start}:{payload_end}, file_size={len(data)}",
            how="provide the complete payload declared by the owning chunk header",
        )
    if not deep and part.compression == "none" and expected_size is not None and packed_size != expected_size:
        raise _parser_error(
            why="the uncompressed EXR chunk size differs from its channel sampling geometry",
            what=f"{context}, packed_size={packed_size}, expected_size={expected_size}",
            how="store exactly the samples selected by the part data window and channel sampling lattice",
        )
    return _ExrChunk(
        y=y,
        row_start=row_start,
        row_count=row_count,
        packed_size=packed_size,
        payload_start=payload_start,
        payload_end=payload_end,
        expected_size=expected_size if expected_size is not None else packed_size,
        raw_stored=not deep and (part.compression == "none" or packed_size == expected_size),
        dwa=None,
        rle=None,
        pxr24=None,
        b44=None,
        piz=None,
        part_index=part.index,
        chunk_offset=chunk_offset,
        span_start=chunk_offset,
        span_end=payload_end,
        kind=kind,
        tile_x=identity[0] if tiled else None,
        tile_y=identity[1] if tiled else None,
        level_x=identity[2] if tiled else None,
        level_y=identity[3] if tiled else None,
        packed_sample_table_size=packed_sample_table_size,
        unpacked_size=unpacked_size,
    )


def _parse_part_chunk_ownership(
    data: bytes,
    offset: int,
    parts: tuple[_ExrPart, ...],
    *,
    multipart: bool,
) -> tuple[_ExrPart, ...]:
    total_chunks = sum(part.expected_chunk_count for part in parts)
    table_size = _checked_product(total_chunks, 8, context="all part offset tables")
    table_end = offset + table_size
    if table_end > len(data):
        raise _parser_error(
            why="the EXR part offset tables are truncated",
            what=f"tables={offset}:{table_end}, file_size={len(data)}, expected_chunks={total_chunks}",
            how="provide one complete eight-byte offset for every chunk in every part",
        )

    table_cursor = offset
    parsed_parts: list[_ExrPart] = []
    seen_offsets: dict[int, tuple[int, tuple[int, ...]]] = {}
    all_chunks: list[_ExrChunk] = []
    for part in parts:
        identities = _part_chunk_identities(part)
        if len(identities) != part.expected_chunk_count:
            raise _parser_error(
                why="the EXR part layout does not derive the declared number of chunk identities",
                what=(
                    f"part={part.index}, identities={len(identities)}, "
                    f"expected_chunks={part.expected_chunk_count}, type={part.image_type!r}"
                ),
                how="make chunkCount agree with the part data window, compression, and tile levels",
            )
        part_offsets = tuple(
            struct.unpack_from("<Q", data, table_cursor + entry * 8)[0] for entry in range(part.expected_chunk_count)
        )
        table_cursor += part.expected_chunk_count * 8
        part_chunks: list[_ExrChunk] = []
        for identity, chunk_offset in zip(identities, part_offsets, strict=True):
            previous_owner = seen_offsets.get(chunk_offset)
            if previous_owner is not None:
                raise _parser_error(
                    why="the EXR part offset tables contain a duplicate chunk offset",
                    what=(
                        f"part={part.index}, identity={identity!r}, offset={chunk_offset}, "
                        f"previous_owner={previous_owner!r}"
                    ),
                    how="give every part chunk one distinct offset-table entry and file span",
                )
            seen_offsets[chunk_offset] = (part.index, identity)
            chunk = _parse_structural_chunk(
                data,
                chunk_offset,
                part,
                identity,
                multipart=multipart,
                table_end=table_end,
            )
            part_chunks.append(chunk)
            all_chunks.append(chunk)
        if not part.levels:
            observed_rows = tuple(chunk.row_start for chunk in part_chunks)
            if len(set(observed_rows)) != len(observed_rows):
                raise _parser_error(
                    why="the EXR chunks repeat an output row range within one part",
                    what=f"part={part.index}, observed_row_starts={observed_rows!r}",
                    how="provide exactly one aligned scanline chunk for every output row block in the part",
                )
            expected_rows = {identity[0] - part.data_window[1] for identity in identities}
            if set(observed_rows) != expected_rows:
                raise _parser_error(
                    why="the EXR chunks leave output row ranges missing from one part",
                    what=(
                        f"part={part.index}, observed={tuple(sorted(observed_rows))!r}, "
                        f"expected={tuple(sorted(expected_rows))!r}"
                    ),
                    how="provide exactly one aligned scanline chunk for every output row block in the part",
                )
            part_chunks.sort(key=lambda chunk: chunk.row_start)
        levels = tuple(
            replace(
                level,
                offsets=part_offsets[level.table_start : level.table_start + level.table_count],
                chunks=tuple(part_chunks[level.table_start : level.table_start + level.table_count]),
            )
            for level in part.levels
        )
        parsed_parts.append(replace(part, offset_table=part_offsets, chunks=tuple(part_chunks), levels=levels))

    ordered_spans = sorted(all_chunks, key=lambda chunk: chunk.span_start)
    for previous, current in zip(ordered_spans, ordered_spans[1:], strict=False):
        if previous.span_end > current.span_start:
            raise _parser_error(
                why="the EXR chunks owned by the part offset tables have intersecting file spans",
                what=(
                    f"previous_part={previous.part_index}, previous_span={(previous.span_start, previous.span_end)}, "
                    f"current_part={current.part_index}, current_span={(current.span_start, current.span_end)}"
                ),
                how="store every part chunk header and payload in one distinct non-overlapping file span",
            )
    return tuple(parsed_parts)


def _build_exr_decoder_view(
    container: _ExrContainer,
    part_index: int,
    *,
    tile_chunk: _ExrChunk | None = None,
    tile_size: tuple[int, int] | None = None,
    preserve_sampling: bool = False,
) -> _ExrContainer:
    """Build one dense scanline-shaped view over an owned part or level-zero tile."""
    source_part = container.parts[part_index]
    if source_part.deep:
        raise _parser_error(
            why="a deep EXR part cannot be materialized by the flat decoder view",
            what=f"part={source_part.name!r}, part_index={part_index}",
            how="select channels from a flat scanlineimage or tiledimage part",
        )
    if preserve_sampling and tile_chunk is not None:
        raise _parser_error(
            why="a sampled EXR decoder view cannot reinterpret a tiled chunk",
            what=f"part={source_part.name!r}, part_index={part_index}",
            how="use the sampled decoder view only for scanline parts",
        )
    if tile_chunk is None:
        data_window = source_part.data_window
        source_chunks = source_part.chunks
        lines_per_chunk = _EXR_LINES_PER_CHUNK[source_part.compression]
    else:
        if tile_size is None or tile_chunk.level_x != 0 or tile_chunk.level_y != 0:
            raise _parser_error(
                why="the tiled EXR decoder view requires one level-zero tile size",
                what=(f"part={part_index}, level={(tile_chunk.level_x, tile_chunk.level_y)}, tile_size={tile_size!r}"),
                how="materialize only a complete level (0, 0) tile with its clipped edge dimensions",
            )
        tile_width, tile_height = tile_size
        data_window = (0, 0, tile_width - 1, tile_height - 1)
        source_chunks = (
            replace(
                tile_chunk,
                y=0,
                row_start=0,
                row_count=tile_height,
                part_index=0,
                kind="scanline",
                tile_x=None,
                tile_y=None,
                level_x=None,
                level_y=None,
            ),
        )
        lines_per_chunk = tile_height

    x_min, y_min, x_max, y_max = data_window
    width = x_max - x_min + 1
    channels = (
        source_part.channels
        if preserve_sampling
        else tuple(
            replace(
                channel,
                x_sampling=1,
                y_sampling=1,
                sampling=_sampling_geometry(data_window, x_sampling=1, y_sampling=1),
            )
            for channel in source_part.channels
        )
    )
    decoder_part = replace(
        source_part,
        index=0,
        image_type="scanlineimage",
        channels=channels,
        data_window=data_window,
        display_window=data_window,
        deep=False,
        tile_description=None,
        levels=(),
    )
    row_bytes = _checked_product(
        width,
        sum(channel.bytes_per_sample for channel in channels),
        context=f"part {part_index} decoder-view row bytes",
    )
    chunks: list[_ExrChunk] = []
    for source_chunk in source_chunks:
        expected_size = (
            source_chunk.expected_size
            if preserve_sampling
            else _checked_product(
                row_bytes,
                source_chunk.row_count,
                context=f"part {part_index} decoder-view chunk y={source_chunk.y}",
            )
        )
        raw_stored = (
            source_chunk.raw_stored
            if preserve_sampling
            else (
                source_part.compression == "none"
                or source_chunk.packed_size == expected_size
                or (source_part.compression == "piz" and source_chunk.packed_size == 0)
            )
        )
        chunks.append(
            _parse_codec_chunk(
                container.data,
                decoder_part,
                source_chunk,
                width=width,
                lines_per_chunk=lines_per_chunk,
                expected_size=expected_size,
                raw_stored=raw_stored,
                part_index=0,
            )
        )

    decoded_chunks = tuple(chunks)
    decoder_part = replace(
        decoder_part,
        expected_chunk_count=len(decoded_chunks),
        offset_table=tuple(chunk.chunk_offset for chunk in decoded_chunks),
        chunks=decoded_chunks,
    )
    return replace(
        container,
        multipart=False,
        tiled=False,
        deep=False,
        parts=(decoder_part,),
        compression=source_part.compression,
        line_order=source_part.line_order,
        data_window=data_window,
        display_window=data_window,
        lines_per_chunk=lines_per_chunk,
        expected_chunk_count=len(decoded_chunks),
        offset_table=decoder_part.offset_table,
        chunks=decoded_chunks,
    )


def _parse_exr_container(path: Path) -> _ExrContainer:
    data = path.read_bytes()
    if len(data) < 8:
        raise _parser_error(
            why="the EXR file ends before its magic and version fields",
            what=f"requested=8 bytes, received={len(data)} bytes",
            how="provide a complete EXR file header",
        )
    magic, version_field = struct.unpack_from("<II", data)
    if magic != _EXR_MAGIC:
        raise _parser_error(
            why="the image does not have the required EXR magic value",
            what=f"magic={magic}",
            how="pass a valid OpenEXR file beginning with magic value 20000630",
        )
    version = version_field & 0xFF
    if version != _EXR_VERSION:
        raise _parser_error(
            why="the EXR file uses an unsupported container version",
            what=f"version={version}",
            how="encode the image using OpenEXR file format version 2",
        )
    version_flags = version_field & ~0xFF
    unknown_flags = version_flags & ~_EXR_SUPPORTED_VERSION_FLAGS
    if unknown_flags:
        raise _parser_error(
            why="the EXR version field contains an unknown flag",
            what=f"unknown_flags=0x{unknown_flags:08x}, version_field=0x{version_field:08x}",
            how="encode a version 2 EXR using only tiled, long-name, non-image, and multipart flags",
        )
    multipart = bool(version_flags & _EXR_MULTIPART_FLAG)
    tiled = bool(version_flags & _EXR_TILED_FLAG)
    non_image = bool(version_flags & _EXR_NON_IMAGE_FLAG)
    if tiled and (multipart or non_image):
        raise _parser_error(
            why="the EXR version field combines incompatible layout flags",
            what=(
                f"tiled_flag={tiled}, non_image_flag={non_image}, multipart_flag={multipart}, "
                f"version_field=0x{version_field:08x}"
            ),
            how="use the single-tile flag only for one regular tiledimage part",
        )
    parts: list[_ExrPart] = []
    offset = 8
    while True:
        attributes, offset = _parse_attributes(data, offset)
        if not attributes:
            break
        parts.append(
            _parse_part(
                attributes,
                tiled_flag=tiled,
                non_image_flag=non_image,
                multipart=multipart,
                part_index=len(parts),
            )
        )
        if not multipart:
            break
    if not parts:
        raise _parser_error(
            why="the EXR header contains no readable image parts",
            what="parts=0, dimensions=None",
            how="pass an EXR containing at least one part with channels and a dataWindow",
        )
    if multipart and len(parts) < 2:
        raise _parser_error(
            why="the EXR multipart flag describes fewer than two part headers",
            what=f"parts={len(parts)}, multipart_flag={multipart}",
            how="clear the multipart flag for one part or provide at least two complete part headers",
        )
    deep_parts = tuple((part.index, part.image_type) for part in parts if part.deep)
    if non_image and not deep_parts:
        raise _parser_error(
            why="the EXR non-image flag is set without any deep part",
            what=f"parts={tuple((part.index, part.image_type) for part in parts)!r}, non_image_flag={non_image}",
            how="clear the non-image flag or provide a deepscanline or deeptile part",
        )
    parts = list(
        _parse_part_chunk_ownership(
            data,
            offset,
            tuple(parts),
            multipart=multipart,
        )
    )
    first = parts[0]
    x_min, y_min, x_max, y_max = first.data_window
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    _checked_product(width, height, context="dataWindow pixel count")
    deep = non_image or any(part.deep for part in parts)
    candidate = _codec_gpu_eligible(first, (), multipart=multipart, tiled=tiled, deep=deep)
    lines_per_chunk = _EXR_LINES_PER_CHUNK[first.compression]
    expected_chunk_count = (height + lines_per_chunk - 1) // lines_per_chunk
    offset_table: tuple[int, ...] = ()
    chunks: tuple[_ExrChunk, ...] = ()
    if candidate:
        offset_table, chunks = _parse_candidate_chunks(data, offset, first, lines_per_chunk=lines_per_chunk)
        parts[0] = replace(first, offset_table=offset_table, chunks=chunks)
        first = parts[0]
    return _ExrContainer(
        data=data,
        magic=magic,
        version_field=version_field,
        version=version,
        version_flags=version_flags,
        multipart=multipart,
        tiled=tiled,
        deep=deep,
        parts=tuple(parts),
        compression=first.compression,
        line_order=first.line_order,
        data_window=first.data_window,
        display_window=first.display_window,
        lines_per_chunk=lines_per_chunk,
        expected_chunk_count=expected_chunk_count,
        offset_table=offset_table,
        chunks=chunks,
    )
