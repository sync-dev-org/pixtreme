"""OpenEXR container parsing, descriptors, and validation."""

from __future__ import annotations

import re
import struct
import zlib
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import cast, overload

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error

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
_EXR_GPU_COMPRESSIONS = frozenset(("none", "zip", "zips"))
_EXR_DWA_COMPRESSIONS = frozenset(("dwaa", "dwab"))
_EXR_PHASE3_COMPRESSIONS = frozenset(("rle", "pxr24", "b44", "b44a"))
_EXR_PIZ_COMPRESSION = "piz"
_EXR_PXR24_PLANE_COUNTS = {0: 4, 1: 2, 2: 3}
_PIZ_BITMAP_BYTE_COUNT = 8192
_PIZ_HUFFMAN_LEADER_SIZE = 5 * 4
_DWA_LEADER_SIZE = 11 * 8
_DWA_HUFFMAN_HEADER_SIZE = 5 * 4
_DWA_MAX_HUFFMAN_SYMBOL = 1 << 16
_DWA_MAX_HUFFMAN_CODE_LENGTH = 58
_DWA_HUFFMAN_LOOKAHEAD_BITS = 10
_DWA_HUFFMAN_LOOKAHEAD_SIZE = 1 << _DWA_HUFFMAN_LOOKAHEAD_BITS
_DWA_HUFFMAN_SEGMENT_TOKEN_COUNT = 256
_DWA_STATIC_HUFFMAN = 0
_DWA_DEFLATE = 1
_DWA_SCHEME_NAMES = {0: "unknown", 1: "lossy_dct", 2: "rle"}
_DWA_JPEG_LUMINANCE = np.asarray(
    (
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ),
    dtype=np.float32,
)
_DWA_JPEG_CHROMA = np.asarray(
    (
        17,
        18,
        24,
        47,
        99,
        99,
        99,
        99,
        18,
        21,
        26,
        66,
        99,
        99,
        99,
        99,
        24,
        26,
        56,
        99,
        99,
        99,
        99,
        99,
        47,
        66,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
        99,
    ),
    dtype=np.float32,
)


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
class _DwaByteSpan:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _DwaChannelRule:
    suffix: str
    scheme: str
    pixel_type: int
    csc_index: int | None
    case_insensitive: bool


@dataclass(frozen=True)
class _DwaChannelDescriptor:
    name: str
    suffix: str
    layer_prefix: str
    scheme: str
    csc_index: int | None
    csc_group: int | None


@dataclass(frozen=True)
class _DwaCscGroup:
    channel_names: tuple[str, str, str]


@dataclass(frozen=True)
class _DwaChannelLayout:
    channels: tuple[_DwaChannelDescriptor, ...]
    csc_groups: tuple[_DwaCscGroup, ...]


@dataclass(frozen=True)
class _DwaHuffmanCode:
    symbol: int
    length: int
    code: int


@dataclass(frozen=True)
class _DwaHuffmanTable:
    minimum_symbol: int
    maximum_symbol: int
    table_byte_count: int
    data_bit_count: int
    code_lengths: tuple[int, ...]
    codes: tuple[_DwaHuffmanCode, ...]
    table_span: _DwaByteSpan
    data_span: _DwaByteSpan


@dataclass(frozen=True)
class _DwaLeader:
    version: int
    unknown_uncompressed_size: int
    unknown_compressed_size: int
    ac_compressed_size: int
    dc_compressed_size: int
    rle_compressed_size: int
    rle_uncompressed_size: int
    rle_raw_size: int
    ac_element_count: int
    dc_element_count: int
    ac_compression: int


@dataclass(frozen=True)
class _DwaChunkGeometry:
    lines_per_chunk: int
    row_count: int
    block_columns: int
    block_rows: int
    padded_width: int
    padded_height: int
    mirror_right: int
    mirror_bottom: int


@dataclass(frozen=True)
class _DwaChunkDescriptor:
    geometry: _DwaChunkGeometry
    leader: _DwaLeader | None
    channel_rules: tuple[_DwaChannelRule, ...]
    channel_layout: _DwaChannelLayout | None
    unknown_span: _DwaByteSpan
    ac_span: _DwaByteSpan
    dc_span: _DwaByteSpan
    rle_span: _DwaByteSpan
    huffman: _DwaHuffmanTable | None


@dataclass(frozen=True)
class _DwaWriteStreams:
    unknown: cp.ndarray = field(repr=False)
    unknown_sizes: tuple[int, ...]
    ac_symbols: cp.ndarray = field(repr=False)
    ac_chunk_ids: cp.ndarray = field(repr=False)
    dc: cp.ndarray = field(repr=False)
    dc_sizes: tuple[int, ...]
    rle: cp.ndarray = field(repr=False)
    rle_sizes: tuple[int, ...]


@dataclass(frozen=True)
class _DwaDeflateStreams:
    payload: cp.ndarray = field(repr=False)
    unknown_offsets: tuple[int, ...]
    unknown_sizes: tuple[int, ...]
    dc_offsets: tuple[int, ...]
    dc_sizes: tuple[int, ...]
    rle_offsets: tuple[int, ...]
    rle_sizes: tuple[int, ...]


@dataclass(frozen=True)
class _Phase3ByteSpan:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _Phase3ChannelRow:
    channel_index: int
    channel_name: str
    pixel_type: int
    bytes_per_sample: int
    perceptually_linear: bool
    chunk_row: int
    file_y: int
    output_row: int
    raw_span: _Phase3ByteSpan
    materialized_span: _Phase3ByteSpan


@dataclass(frozen=True)
class _Phase3Plane:
    channel_index: int
    channel_name: str
    chunk_row: int
    output_row: int
    plane_index: int
    materialized_span: _Phase3ByteSpan


@dataclass(frozen=True)
class _Phase3Packet:
    literal: bool
    input_span: _Phase3ByteSpan
    output_span: _Phase3ByteSpan


@dataclass(frozen=True)
class _Phase3RlePackets:
    payload: memoryview = field(repr=False, compare=False)
    payload_start: int
    packet_count: int

    def __len__(self) -> int:
        return self.packet_count

    def __iter__(self) -> Iterator[_Phase3Packet]:
        input_offset = 0
        output_offset = 0
        while input_offset < len(self.payload):
            packet_start = input_offset
            header_byte = int(self.payload[input_offset])
            header = header_byte if header_byte < 128 else header_byte - 256
            input_offset += 1
            literal = header < 0
            output_size = -header if literal else header + 1
            input_offset += output_size if literal else 1
            output_end = output_offset + output_size
            yield _Phase3Packet(
                literal=literal,
                input_span=_Phase3ByteSpan(
                    self.payload_start + packet_start,
                    self.payload_start + input_offset,
                ),
                output_span=_Phase3ByteSpan(output_offset, output_end),
            )
            output_offset = output_end

    def __getitem__(self, index: int) -> _Phase3Packet:
        resolved = index if index >= 0 else self.packet_count + index
        if not 0 <= resolved < self.packet_count:
            raise IndexError(index)
        for packet_index, packet in enumerate(self):
            if packet_index == resolved:
                return packet
        raise IndexError(index)


@dataclass(frozen=True)
class _Phase3ChannelSection:
    channel_index: int
    channel_name: str
    pixel_type: int
    bytes_per_sample: int
    perceptually_linear: bool
    payload_span: _Phase3ByteSpan
    expected_materialized_size: int
    block_start: int
    block_count: int


@dataclass(frozen=True)
class _Phase3Block:
    channel_index: int
    channel_name: str
    block_row: int
    block_column: int
    payload_span: _Phase3ByteSpan
    stored_size: int
    output_row_start: int
    output_row_count: int


@dataclass(frozen=True)
class _Phase3B44Blocks:
    payload: bytes = field(repr=False, compare=False)
    codec: str
    block_sections: tuple[_Phase3ChannelSection, ...]
    block_starts: tuple[int, ...]
    block_columns: int
    row_start: int
    row_count: int

    def __len__(self) -> int:
        return sum(section.block_count for section in self.block_sections)

    def __iter__(self) -> Iterator[_Phase3Block]:
        for section in self.block_sections:
            payload_start = section.payload_span.start
            for local_block in range(section.block_count):
                stored_size = 14 if self.codec == "b44" or self.payload[payload_start + 2] < 0x34 else 3
                yield self._block(section, local_block, payload_start, stored_size)
                payload_start += stored_size

    @overload
    def __getitem__(self, index: int) -> _Phase3Block: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[_Phase3Block, ...]: ...

    def __getitem__(self, index: int | slice) -> _Phase3Block | tuple[_Phase3Block, ...]:
        if isinstance(index, slice):
            return tuple(self[block_index] for block_index in range(*index.indices(len(self))))
        resolved = index if index >= 0 else len(self) + index
        if not 0 <= resolved < len(self):
            raise IndexError(index)
        section_index = bisect_right(self.block_starts, resolved) - 1
        section = self.block_sections[section_index]
        within_section = resolved - section.block_start
        if self.codec == "b44":
            payload_start = section.payload_span.start + within_section * 14
            stored_size = 14
        else:
            payload_start = section.payload_span.start
            for _ in range(within_section):
                payload_start += 3 if self.payload[payload_start + 2] >= 0x34 else 14
            stored_size = 3 if self.payload[payload_start + 2] >= 0x34 else 14
        return self._block(section, within_section, payload_start, stored_size)

    def _block(
        self,
        section: _Phase3ChannelSection,
        within_section: int,
        payload_start: int,
        stored_size: int,
    ) -> _Phase3Block:
        block_row, block_column = divmod(within_section, self.block_columns)
        return _Phase3Block(
            channel_index=section.channel_index,
            channel_name=section.channel_name,
            block_row=block_row,
            block_column=block_column,
            payload_span=_Phase3ByteSpan(payload_start, payload_start + stored_size),
            stored_size=stored_size,
            output_row_start=self.row_start + block_row * 4,
            output_row_count=min(4, self.row_count - block_row * 4),
        )


@dataclass(frozen=True)
class _Phase3ChunkDescriptor:
    codec: str
    lines_per_chunk: int
    chunk_y: int
    row_start: int
    row_count: int
    payload_span: _Phase3ByteSpan
    stored_size: int
    expected_raw_size: int
    expected_materialized_size: int
    raw_stored: bool
    channel_rows: tuple[_Phase3ChannelRow, ...]
    planes: tuple[_Phase3Plane, ...]
    packets: tuple[_Phase3Packet, ...] | _Phase3RlePackets
    channel_sections: tuple[_Phase3ChannelSection, ...]
    blocks: tuple[_Phase3Block, ...] | _Phase3B44Blocks


@dataclass(frozen=True)
class _PizByteSpan:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _PizChannelPlane:
    channel_index: int
    channel_name: str
    pixel_type: int
    bytes_per_sample: int
    sample_count: int
    word_slice_count: int
    word_offset: int
    word_count: int


@dataclass(frozen=True)
class _PizHuffmanLeader:
    minimum_symbol: int
    maximum_symbol: int
    table_byte_count: int
    data_bit_count: int
    reserved: int
    span: _PizByteSpan


@dataclass(frozen=True)
class _PizHuffmanTable:
    minimum_symbol: int
    maximum_symbol: int
    declared_table_byte_count: int
    data_bit_count: int
    reserved: int
    code_lengths: tuple[int, ...]
    codes: tuple[_DwaHuffmanCode, ...]
    table_span: _PizByteSpan
    data_span: _PizByteSpan


@dataclass(frozen=True)
class _PizChunkDescriptor:
    lines_per_chunk: int
    chunk_y: int
    row_start: int
    row_count: int
    output_row_span: tuple[int, int]
    payload_span: _PizByteSpan
    stored_size: int
    expected_packed_size: int
    expected_output_word_count: int
    raw_stored: bool
    channel_planes: tuple[_PizChannelPlane, ...]
    bitmap_range: tuple[int, int] | None
    bitmap_span: _PizByteSpan
    huffman_byte_count: int
    huffman_count_span: _PizByteSpan
    huffman_span: _PizByteSpan
    huffman_leader: _PizHuffmanLeader | None
    trailing_span: _PizByteSpan


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
    dwa: _DwaChunkDescriptor | None
    phase3: _Phase3ChunkDescriptor | None
    piz: _PizChunkDescriptor | None
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
    gpu_eligible: bool
    dwa_eligible: bool
    phase3_eligible: bool
    piz_eligible: bool


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


@dataclass(frozen=True)
class _ExrRleReadChunks:
    host_staging: np.ndarray = field(repr=False)
    stage_offsets: np.ndarray
    stage_sizes: np.ndarray
    decoded_offsets: np.ndarray
    decoded_sizes: np.ndarray
    compressed: np.ndarray
    packet_offsets: np.ndarray
    packet_counts: np.ndarray


@dataclass(frozen=True)
class _ExrPxr24ReadChunks:
    host_staging: np.ndarray = field(repr=False)
    host_materialized: np.ndarray = field(repr=False)
    stage_offsets: np.ndarray
    stage_sizes: np.ndarray
    materialized_offsets: np.ndarray
    materialized_sizes: np.ndarray
    raw_offsets: np.ndarray
    raw_sizes: np.ndarray
    compressed: np.ndarray


@dataclass(frozen=True)
class _ExrB44ReadChunks:
    host_staging: np.ndarray = field(repr=False)
    stage_offsets: np.ndarray
    stage_sizes: np.ndarray
    raw_offsets: np.ndarray
    raw_sizes: np.ndarray
    compressed: np.ndarray
    raw_section_descriptors: np.ndarray
    block_section_descriptors: np.ndarray
    block_output_descriptors: np.ndarray
    block_perceptually_linear: np.ndarray
    b44a: bool


class _ExrGpuError(RuntimeError):
    """An already-classified EXR integrity or GPU I/O failure."""


class _ExrPhase3Error(RuntimeError):
    """An already-classified Phase 3 descriptor integrity failure."""


class _ExrPizError(RuntimeError):
    """An already-classified Phase 4 PIZ descriptor integrity failure."""


def _parser_error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _gpu_error(*, why: str, what: str, how: str) -> _ExrGpuError:
    return _ExrGpuError(_actionable_error(why=why, what=what, how=how))


def _phase3_error(*, why: str, what: str, how: str) -> _ExrPhase3Error:
    return _ExrPhase3Error(_actionable_error(why=why, what=what, how=how))


def _piz_error(*, why: str, what: str, how: str) -> _ExrPizError:
    return _ExrPizError(_actionable_error(why=why, what=what, how=how))


def _dwa_suffix(name: str) -> tuple[str, str]:
    separator = name.rfind(".")
    if separator < 0:
        return "", name
    return name[:separator], name[separator + 1 :]


def _dwa_rule_matches(rule: _DwaChannelRule, channel: _ExrChannel) -> bool:
    _, suffix = _dwa_suffix(channel.name)
    if rule.pixel_type != channel.pixel_type:
        return False
    if rule.case_insensitive:
        return rule.suffix.casefold() == suffix.casefold()
    return rule.suffix == suffix


def _classify_dwa_channels(
    channels: Sequence[_ExrChannel],
    rules: Sequence[_DwaChannelRule],
) -> _DwaChannelLayout:
    provisional: list[tuple[_ExrChannel, str, str, str, int | None]] = []
    for channel in channels:
        prefix, suffix = _dwa_suffix(channel.name)
        matching = tuple(index for index, rule in enumerate(rules) if _dwa_rule_matches(rule, channel))
        if len(matching) > 1:
            raise _parser_error(
                why="the DWA channel rules assign one file channel more than once",
                what=f"channel={channel.name!r}, matching_rules={matching!r}",
                how="store at most one matching suffix and pixel-type rule for each channel",
            )
        if matching:
            rule = rules[matching[0]]
            scheme = rule.scheme
            csc_index = rule.csc_index
        else:
            scheme = "unknown"
            csc_index = None
        provisional.append((channel, prefix, suffix, scheme, csc_index))

    candidates: dict[tuple[str, int, int], dict[int, str]] = {}
    for channel, prefix, _, scheme, csc_index in provisional:
        if scheme != "lossy_dct" or csc_index is None:
            continue
        key = (prefix, channel.x_sampling, channel.y_sampling)
        slots = candidates.setdefault(key, {})
        if csc_index in slots:
            raise _parser_error(
                why="the DWA channel rules assign duplicate RGB components within one layer and sampling group",
                what=f"layer={prefix!r}, csc_index={csc_index}, channels={(slots[csc_index], channel.name)!r}",
                how="provide at most one R, G, and B component per layer and sampling pattern",
            )
        slots[csc_index] = channel.name

    csc_groups: list[_DwaCscGroup] = []
    group_by_channel: dict[str, int] = {}
    for slots in candidates.values():
        if set(slots) != {0, 1, 2}:
            continue
        group_index = len(csc_groups)
        names = (slots[0], slots[1], slots[2])
        csc_groups.append(_DwaCscGroup(channel_names=names))
        group_by_channel.update((name, group_index) for name in names)

    descriptors = tuple(
        _DwaChannelDescriptor(
            name=channel.name,
            suffix=suffix,
            layer_prefix=prefix,
            scheme=scheme,
            csc_index=csc_index,
            csc_group=group_by_channel.get(channel.name),
        )
        for channel, prefix, suffix, scheme, csc_index in provisional
    )
    return _DwaChannelLayout(channels=descriptors, csc_groups=tuple(csc_groups))


def _classify_default_dwa_channels(channels: Sequence[_ExrChannel]) -> _DwaChannelLayout:
    rules = tuple(
        _DwaChannelRule(
            suffix=suffix,
            scheme="lossy_dct",
            pixel_type=pixel_type,
            csc_index={"R": 0, "G": 1, "B": 2}.get(suffix),
            case_insensitive=False,
        )
        for suffix in ("R", "G", "B", "Y", "BY", "RY")
        for pixel_type in (1, 2)
    ) + tuple(
        _DwaChannelRule(
            suffix="A",
            scheme="rle",
            pixel_type=pixel_type,
            csc_index=None,
            case_insensitive=False,
        )
        for pixel_type in (0, 1, 2)
    )
    return _classify_dwa_channels(channels, rules)


def _read_dwa_bits(data: bytes, bit_offset: int, width: int) -> tuple[int, int]:
    if bit_offset + width > len(data) * 8:
        raise _parser_error(
            why="the DWA Huffman code-length table is truncated within a packed field",
            what=f"bit_offset={bit_offset}, requested_bits={width}, table_bytes={len(data)}",
            how="provide the complete six-bit length or eight-bit long-run payload",
        )
    end = bit_offset + width
    byte_start = bit_offset // 8
    byte_end = (end + 7) // 8
    packed = int.from_bytes(data[byte_start:byte_end], "big")
    value = (packed >> (byte_end * 8 - end)) & ((1 << width) - 1)
    return value, end


def _canonical_dwa_codes(minimum_symbol: int, lengths: Sequence[int]) -> tuple[_DwaHuffmanCode, ...]:
    length_array = np.asarray(lengths, dtype=np.uint8)
    observed_offsets = np.flatnonzero(length_array)
    if not observed_offsets.size:
        raise _parser_error(
            why="the DWA Huffman code-length assignment contains no decodable symbol",
            what=f"minimum_symbol={minimum_symbol}, symbol_count={len(lengths)}",
            how="assign a code length between 1 and 58 to at least one symbol",
        )
    observed_lengths = length_array[observed_offsets].astype(np.int64, copy=False)
    counts = np.bincount(observed_lengths, minlength=_DWA_MAX_HUFFMAN_CODE_LENGTH + 1)
    maximum_length = int(observed_lengths.max())
    capacity = 1 << maximum_length
    occupancy = sum(int(counts[length]) << (maximum_length - length) for length in range(1, maximum_length + 1))
    if occupancy > capacity:
        raise _parser_error(
            why="the DWA Huffman code-length assignment is oversubscribed",
            what=f"occupancy={occupancy}, capacity={capacity}, maximum_length={maximum_length}",
            how="reduce the number of short codes so the prefix-code capacity is not exceeded",
        )

    next_code = np.zeros(_DWA_MAX_HUFFMAN_CODE_LENGTH + 1, dtype=np.uint64)
    code = 0
    for length in range(_DWA_MAX_HUFFMAN_CODE_LENGTH, 0, -1):
        next_code[length] = code
        code = (code + int(counts[length])) >> 1

    codes: list[_DwaHuffmanCode] = []
    intervals: list[tuple[int, int, int]] = []
    for assigned_length_value in np.flatnonzero(counts[1:]) + 1:
        assigned_length = int(assigned_length_value)
        offsets = observed_offsets[observed_lengths == assigned_length]
        assigned_codes = next_code[assigned_length] + np.arange(offsets.size, dtype=np.uint64)
        if int(assigned_codes[-1]) >= 1 << assigned_length:
            raise _parser_error(
                why="the DWA canonical Huffman assignment exceeds its declared code length",
                what=(
                    f"symbol={minimum_symbol + int(offsets[-1])}, code={int(assigned_codes[-1])}, "
                    f"length={assigned_length}"
                ),
                how="provide a valid prefix-free code-length assignment",
            )
        for offset_value, assigned_value in zip(offsets, assigned_codes, strict=True):
            offset = int(offset_value)
            assigned = int(assigned_value)
            symbol = minimum_symbol + offset
            interval_start = assigned << (_DWA_MAX_HUFFMAN_CODE_LENGTH - assigned_length)
            interval_end = (assigned + 1) << (_DWA_MAX_HUFFMAN_CODE_LENGTH - assigned_length)
            intervals.append((interval_start, interval_end, symbol))
            codes.append(_DwaHuffmanCode(symbol=symbol, length=assigned_length, code=assigned))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if previous[1] > current[0]:
            raise _parser_error(
                why="the DWA canonical Huffman code-length assignment contains overlapping prefixes",
                what=f"symbols={(previous[2], current[2])!r}, intervals={(previous[:2], current[:2])!r}",
                how="provide a prefix-free canonical code-length assignment",
            )
    return tuple(codes)


def _parse_dwa_huffman_table(
    payload: bytes,
    *,
    base_offset: int = 0,
    expand: bool = True,
) -> _DwaHuffmanTable:
    if len(payload) < _DWA_HUFFMAN_HEADER_SIZE:
        raise _parser_error(
            why="the DWA Huffman stream is truncated before its 20-byte header",
            what=f"received={len(payload)}, required={_DWA_HUFFMAN_HEADER_SIZE}",
            how="provide the complete five-word canonical Huffman header",
        )
    minimum_symbol, maximum_symbol, table_byte_count, data_bit_count, reserved = struct.unpack_from("<IIIII", payload)
    if minimum_symbol > maximum_symbol or maximum_symbol > _DWA_MAX_HUFFMAN_SYMBOL:
        raise _parser_error(
            why="the DWA Huffman symbol range is invalid",
            what=f"minimum={minimum_symbol}, maximum={maximum_symbol}, allowed_max={_DWA_MAX_HUFFMAN_SYMBOL}",
            how="encode an ordered range within the 16-bit alphabet and its repeat pseudo-symbol",
        )
    if reserved != 0:
        raise _parser_error(
            why="the DWA Huffman reserved header field is nonzero",
            what=f"reserved={reserved}",
            how="write zero in the reserved Huffman header word",
        )
    data_byte_count = (data_bit_count + 7) // 8
    expected_size = _DWA_HUFFMAN_HEADER_SIZE + table_byte_count + data_byte_count
    if len(payload) < expected_size:
        raise _parser_error(
            why="the DWA Huffman table or encoded data is truncated relative to its declared byte and bit counts",
            what=f"received={len(payload)}, declared={expected_size}, table_bytes={table_byte_count}, data_bits={data_bit_count}",
            how="provide every declared code-length and encoded-data byte",
        )
    if len(payload) != expected_size:
        raise _parser_error(
            why="the DWA Huffman stream contains bytes outside its declared table and encoded data",
            what=f"received={len(payload)}, declared={expected_size}",
            how="make the table byte count and data bit count cover the complete AC substream",
        )

    table_start = _DWA_HUFFMAN_HEADER_SIZE
    table_end = table_start + table_byte_count
    data_padding_bits = data_byte_count * 8 - data_bit_count
    if data_padding_bits and payload[expected_size - 1] & ((1 << data_padding_bits) - 1):
        raise _parser_error(
            why="the DWA Huffman encoded data has nonzero padding bits beyond its declared data bit count",
            what=f"data_bits={data_bit_count}, padding_bits={data_padding_bits}, final_byte=0x{payload[expected_size - 1]:02x}",
            how="zero the unused low bits of the final encoded-data byte",
        )
    if not expand:
        return _DwaHuffmanTable(
            minimum_symbol=minimum_symbol,
            maximum_symbol=maximum_symbol,
            table_byte_count=table_byte_count,
            data_bit_count=data_bit_count,
            code_lengths=(),
            codes=(),
            table_span=_DwaByteSpan(base_offset + table_start, base_offset + table_end),
            data_span=_DwaByteSpan(base_offset + table_end, base_offset + expected_size),
        )

    packed_table = payload[table_start:table_end]
    symbol_count = maximum_symbol - minimum_symbol + 1
    lengths: list[int] = []
    bit_offset = 0
    while len(lengths) < symbol_count:
        token, bit_offset = _read_dwa_bits(packed_table, bit_offset, 6)
        if token <= _DWA_MAX_HUFFMAN_CODE_LENGTH:
            lengths.append(token)
            continue
        if token == 63:
            extra, bit_offset = _read_dwa_bits(packed_table, bit_offset, 8)
            run = extra + 6
        else:
            run = token - 59 + 2
        if len(lengths) + run > symbol_count:
            raise _parser_error(
                why="the DWA Huffman table-run extends beyond the declared symbol range",
                what=f"decoded={len(lengths)}, run={run}, symbols={symbol_count}, token={token}",
                how="shorten the zero-length run to end within the min/max symbol range",
            )
        lengths.extend((0,) * run)

    consumed_bytes = (bit_offset + 7) // 8
    if consumed_bytes != table_byte_count:
        raise _parser_error(
            why="the DWA Huffman table byte count does not match the decoded code-length fields",
            what=f"declared={table_byte_count}, consumed={consumed_bytes}, consumed_bits={bit_offset}",
            how="set the table byte count to the exact packed code-length size",
        )
    padding_bits = consumed_bytes * 8 - bit_offset
    if padding_bits and packed_table[-1] & ((1 << padding_bits) - 1):
        raise _parser_error(
            why="the DWA Huffman code-length table has nonzero padding bits",
            what=f"padding_bits={padding_bits}, final_byte=0x{packed_table[-1]:02x}",
            how="zero the unused low bits of the final table byte",
        )

    codes = _canonical_dwa_codes(minimum_symbol, lengths)
    return _DwaHuffmanTable(
        minimum_symbol=minimum_symbol,
        maximum_symbol=maximum_symbol,
        table_byte_count=table_byte_count,
        data_bit_count=data_bit_count,
        code_lengths=tuple(lengths),
        codes=codes,
        table_span=_DwaByteSpan(base_offset + table_start, base_offset + table_end),
        data_span=_DwaByteSpan(base_offset + table_end, base_offset + expected_size),
    )


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


def _parse_dwa_channel_rules(
    payload: bytes,
    offset: int,
    *,
    chunk_y: int,
    payload_offset: int,
) -> tuple[tuple[_DwaChannelRule, ...], int]:
    if offset + 2 > len(payload):
        raise _parser_error(
            why="the DWA v2 channel-rule section is truncated before its byte count",
            what=f"chunk_y={chunk_y}, offset={payload_offset + offset}, remaining={len(payload) - offset}",
            how="provide the little-endian two-byte channel-rule section size",
        )
    section_size = struct.unpack_from("<H", payload, offset)[0]
    if section_size < 2:
        raise _parser_error(
            why="the DWA v2 channel-rule section size excludes its own two-byte count",
            what=f"chunk_y={chunk_y}, section_size={section_size}",
            how="declare a channel-rule size of at least two bytes including the size field",
        )
    section_end = offset + section_size
    if section_end > len(payload):
        raise _parser_error(
            why="the DWA v2 channel-rule section is truncated relative to its declared size",
            what=f"chunk_y={chunk_y}, section={payload_offset + offset}:{payload_offset + section_end}, payload_end={payload_offset + len(payload)}",
            how="provide every declared channel-rule record byte",
        )

    cursor = offset + 2
    rules: list[_DwaChannelRule] = []
    while cursor < section_end:
        terminator = payload.find(b"\x00", cursor, section_end)
        if terminator < 0:
            raise _parser_error(
                why="the DWA channel-rule suffix is truncated before its null terminator",
                what=f"chunk_y={chunk_y}, rule_offset={payload_offset + cursor}, section_end={payload_offset + section_end}",
                how="terminate every channel suffix within the declared rule section",
            )
        suffix_bytes = payload[cursor:terminator]
        if not suffix_bytes or len(suffix_bytes) > 128:
            raise _parser_error(
                why="the DWA channel-rule suffix length is outside the supported 1-to-128-byte range",
                what=f"chunk_y={chunk_y}, suffix_length={len(suffix_bytes)}",
                how="store a nonempty channel suffix no longer than 128 bytes",
            )
        try:
            suffix = suffix_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _parser_error(
                why="the DWA channel-rule suffix is not valid UTF-8",
                what=f"chunk_y={chunk_y}, suffix_bytes={suffix_bytes!r}",
                how="encode channel-rule suffixes using valid UTF-8",
            ) from error
        cursor = terminator + 1
        if cursor + 2 > section_end:
            raise _parser_error(
                why="the DWA channel-rule record is truncated before its classification and pixel-type bytes",
                what=f"chunk_y={chunk_y}, suffix={suffix!r}, remaining={section_end - cursor}",
                how="append both packed classification and pixel-type bytes to every suffix",
            )
        packed, pixel_type = payload[cursor], payload[cursor + 1]
        cursor += 2
        csc_value = packed >> 4
        scheme_value = (packed >> 2) & 3
        if packed & 0x02 or csc_value > 3 or scheme_value not in _DWA_SCHEME_NAMES or pixel_type not in _EXR_DTYPE_INFO:
            raise _parser_error(
                why="the DWA channel-rule record contains an invalid or reserved classification value",
                what=(
                    f"chunk_y={chunk_y}, suffix={suffix!r}, packed=0x{packed:02x}, "
                    f"csc={csc_value}, scheme={scheme_value}, pixel_type={pixel_type}"
                ),
                how="use CSC values 0-to-3, schemes UNKNOWN/LOSSY_DCT/RLE, a zero reserved bit, and UINT/HALF/FLOAT",
            )
        rules.append(
            _DwaChannelRule(
                suffix=suffix,
                scheme=_DWA_SCHEME_NAMES[scheme_value],
                pixel_type=pixel_type,
                csc_index=csc_value - 1 if csc_value else None,
                case_insensitive=bool(packed & 1),
            )
        )
    return tuple(rules), section_end


def _dwa_geometry(*, width: int, row_count: int, lines_per_chunk: int) -> _DwaChunkGeometry:
    block_columns = (width + 7) // 8
    block_rows = (row_count + 7) // 8
    padded_width = block_columns * 8
    padded_height = block_rows * 8
    return _DwaChunkGeometry(
        lines_per_chunk=lines_per_chunk,
        row_count=row_count,
        block_columns=block_columns,
        block_rows=block_rows,
        padded_width=padded_width,
        padded_height=padded_height,
        mirror_right=padded_width - width,
        mirror_bottom=padded_height - row_count,
    )


def _zero_dwa_spans(offset: int) -> tuple[_DwaByteSpan, _DwaByteSpan, _DwaByteSpan, _DwaByteSpan]:
    span = _DwaByteSpan(offset, offset)
    return span, span, span, span


def _validate_dwa_declared_layout(
    leader: _DwaLeader,
    layout: _DwaChannelLayout,
    channels: Sequence[_ExrChannel],
    geometry: _DwaChunkGeometry,
    *,
    chunk_y: int,
) -> None:
    descriptors = {descriptor.name: descriptor for descriptor in layout.channels}
    width = geometry.padded_width - geometry.mirror_right
    required_sizes = {"unknown": 0, "rle": 0}
    lossy_block_count = 0
    for channel in channels:
        descriptor = descriptors[channel.name]
        channel_width, channel_rows = _chunk_channel_geometry(
            channel,
            width=width,
            chunk_y=chunk_y,
            row_count=geometry.row_count,
        )
        if descriptor.scheme == "lossy_dct":
            lossy_block_count += ((channel_width + 7) // 8) * ((channel_rows + 7) // 8)
            continue
        if descriptor.scheme in required_sizes:
            channel_size = _checked_product(
                channel_width,
                channel_rows,
                channel.bytes_per_sample,
                context=f"DWA {descriptor.scheme} channel {channel.name!r}",
            )
            required_sizes[descriptor.scheme] += channel_size

    lossless_declarations = (
        (
            "UNKNOWN",
            required_sizes["unknown"],
            leader.unknown_compressed_size,
            leader.unknown_uncompressed_size,
            0,
        ),
        (
            "RLE",
            required_sizes["rle"],
            leader.rle_compressed_size,
            leader.rle_raw_size,
            leader.rle_uncompressed_size,
        ),
    )
    for name, required, compressed_size, raw_size, intermediate_size in lossless_declarations:
        if required:
            if compressed_size == 0 or raw_size < required:
                raise _parser_error(
                    why=f"the DWA {name} declarations cannot hold the channels assigned to that route",
                    what=(
                        f"chunk_y={chunk_y}, required_raw={required}, declared_raw={raw_size}, "
                        f"compressed_size={compressed_size}"
                    ),
                    how=f"derive the DWA {name} sizes from the classified channels and chunk row geometry",
                )
        elif compressed_size or raw_size or intermediate_size:
            raise _parser_error(
                why=f"the DWA {name} declarations are nonzero even though no channel uses that route",
                what=(
                    f"chunk_y={chunk_y}, compressed_size={compressed_size}, raw_size={raw_size}, "
                    f"intermediate_size={intermediate_size}"
                ),
                how=f"zero every DWA {name} declaration when the channel rules assign no {name} channel",
            )

    block_count = lossy_block_count
    if block_count:
        maximum_ac_count = _checked_product(block_count, 63, context=f"DWA AC count for chunk y={chunk_y}")
        if not block_count <= leader.ac_element_count <= maximum_ac_count:
            raise _parser_error(
                why="the DWA AC element count is incompatible with the lossy channel block ownership",
                what=(
                    f"chunk_y={chunk_y}, declared={leader.ac_element_count}, "
                    f"allowed={block_count}:{maximum_ac_count}, lossy_blocks={lossy_block_count}"
                ),
                how="declare between one EOB and 63 AC symbols for every lossy 8x8 channel block",
            )
        if leader.dc_element_count != block_count:
            raise _parser_error(
                why="the DWA DC element count is incompatible with the lossy channel block ownership",
                what=(
                    f"chunk_y={chunk_y}, declared={leader.dc_element_count}, expected={block_count}, "
                    f"lossy_blocks={lossy_block_count}"
                ),
                how="declare exactly one DC element for every lossy 8x8 channel block",
            )
    elif any(
        (
            leader.ac_compressed_size,
            leader.dc_compressed_size,
            leader.ac_element_count,
            leader.dc_element_count,
        )
    ):
        raise _parser_error(
            why="the DWA AC or DC declarations are nonzero even though no channel uses lossy DCT",
            what=(
                f"chunk_y={chunk_y}, ac_size={leader.ac_compressed_size}, dc_size={leader.dc_compressed_size}, "
                f"ac_count={leader.ac_element_count}, dc_count={leader.dc_element_count}"
            ),
            how="zero AC and DC sizes and counts when the channel rules assign no lossy DCT channel",
        )


def _parse_dwa_chunk_payload(
    payload: bytes,
    *,
    payload_offset: int,
    chunk_y: int,
    expected_size: int,
    geometry: _DwaChunkGeometry,
    channels: Sequence[_ExrChannel],
) -> _DwaChunkDescriptor:
    if len(payload) < _DWA_LEADER_SIZE:
        raise _parser_error(
            why="the compressed DWA chunk is truncated before its 88-byte leader",
            what=f"chunk_y={chunk_y}, received={len(payload)}, required={_DWA_LEADER_SIZE}",
            how="provide all eleven little-endian uint64 leader fields",
        )
    values = struct.unpack_from("<11Q", payload)
    leader = _DwaLeader(*values)
    if any(value > _EXR_MAX_INTEGER for value in values[1:10]):
        raise _parser_error(
            why="the DWA leader size or element count exceeds signed 64-bit control-plane bounds",
            what=f"chunk_y={chunk_y}, leader={values!r}",
            how="store sizes and counts representable by the decoder descriptor",
        )
    if leader.version > 2:
        raise _parser_error(
            why="the DWA chunk leader declares an unknown format version",
            what=f"chunk_y={chunk_y}, version={leader.version}",
            how="encode a supported DWA chunk version from 0 through 2",
        )
    if leader.ac_compression not in (_DWA_STATIC_HUFFMAN, _DWA_DEFLATE):
        raise _parser_error(
            why="the DWA chunk leader declares an unknown AC compression scheme",
            what=f"chunk_y={chunk_y}, ac_compression={leader.ac_compression}",
            how="use STATIC_HUFFMAN (0) or DEFLATE (1)",
        )

    offset = _DWA_LEADER_SIZE
    channel_rules: tuple[_DwaChannelRule, ...] = ()
    channel_layout: _DwaChannelLayout | None = None
    if leader.version >= 2:
        channel_rules, offset = _parse_dwa_channel_rules(
            payload,
            offset,
            chunk_y=chunk_y,
            payload_offset=payload_offset,
        )
        channel_layout = _classify_dwa_channels(channels, channel_rules)

    size_pairs = (
        ("UNKNOWN", leader.unknown_compressed_size, leader.unknown_uncompressed_size),
        ("AC", leader.ac_compressed_size, leader.ac_element_count),
        ("DC", leader.dc_compressed_size, leader.dc_element_count),
    )
    for name, compressed_size, declared_output in size_pairs:
        if (compressed_size == 0) != (declared_output == 0):
            raise _parser_error(
                why=f"the DWA {name} compressed size and declared output count disagree about an empty stream",
                what=(f"chunk_y={chunk_y}, compressed_size={compressed_size}, declared_output={declared_output}"),
                how=f"set both DWA {name} declarations to zero or provide a nonempty payload and output count",
            )
    rle_values = (leader.rle_compressed_size, leader.rle_uncompressed_size, leader.rle_raw_size)
    if any(rle_values) and not all(rle_values):
        raise _parser_error(
            why="the DWA RLE compressed, encoded, and raw size declarations disagree about an empty stream",
            what=f"chunk_y={chunk_y}, sizes={rle_values!r}",
            how="set all three RLE sizes to zero or provide all three nonzero sizes",
        )
    if channel_layout is not None:
        _validate_dwa_declared_layout(leader, channel_layout, channels, geometry, chunk_y=chunk_y)
    if leader.unknown_uncompressed_size + leader.rle_raw_size > expected_size:
        raise _parser_error(
            why="the DWA lossless channel declarations exceed the outer chunk's raw byte count",
            what=(
                f"chunk_y={chunk_y}, unknown={leader.unknown_uncompressed_size}, "
                f"rle_raw={leader.rle_raw_size}, outer_raw={expected_size}"
            ),
            how="derive UNKNOWN and RLE raw sizes from the channels and chunk geometry",
        )

    compressed_sizes = (
        leader.unknown_compressed_size,
        leader.ac_compressed_size,
        leader.dc_compressed_size,
        leader.rle_compressed_size,
    )
    declared_end = offset + sum(compressed_sizes)
    if declared_end != len(payload):
        raise _parser_error(
            why="the DWA substream spans do not cover the compressed chunk payload exactly",
            what=(
                f"chunk_y={chunk_y}, leader_and_rules={offset}, substream_sizes={compressed_sizes!r}, "
                f"declared_end={declared_end}, payload_size={len(payload)}"
            ),
            how="make the UNKNOWN, AC, DC, and RLE compressed sizes match their concatenated payload bytes",
        )

    span_start = payload_offset + offset
    spans: list[_DwaByteSpan] = []
    for size in compressed_sizes:
        span_end = span_start + size
        spans.append(_DwaByteSpan(span_start, span_end))
        span_start = span_end
    unknown_span, ac_span, dc_span, rle_span = spans
    huffman = None
    if leader.ac_compressed_size and leader.ac_compression == _DWA_STATIC_HUFFMAN:
        relative_start = ac_span.start - payload_offset
        relative_end = ac_span.end - payload_offset
        huffman = _parse_dwa_huffman_table(
            payload[relative_start:relative_end],
            base_offset=ac_span.start,
            expand=False,
        )
        if huffman.data_bit_count == 0:
            raise _parser_error(
                why="the nonempty DWA AC stream declares zero Huffman data bits",
                what=f"chunk_y={chunk_y}, ac_elements={leader.ac_element_count}",
                how="provide the encoded Huffman bits needed to represent every declared AC element",
            )
    return _DwaChunkDescriptor(
        geometry=geometry,
        leader=leader,
        channel_rules=channel_rules,
        channel_layout=channel_layout,
        unknown_span=unknown_span,
        ac_span=ac_span,
        dc_span=dc_span,
        rle_span=rle_span,
        huffman=huffman,
    )


def _phase3_expected_materialized_size(
    codec: str,
    channels: Sequence[_ExrChannel],
    *,
    width: int,
    chunk_y: int,
    row_count: int,
) -> int:
    total = 0
    for channel in channels:
        channel_width, channel_rows = _chunk_channel_geometry(
            channel,
            width=width,
            chunk_y=chunk_y,
            row_count=row_count,
        )
        stride = _EXR_PXR24_PLANE_COUNTS[channel.pixel_type] if codec == "pxr24" else channel.bytes_per_sample
        channel_size = _checked_product(
            channel_width,
            channel_rows,
            stride,
            context=f"{codec.upper()} channel {channel.name!r} materialized bytes",
        )
        if total > _EXR_MAX_INTEGER - channel_size:
            raise _phase3_error(
                why=f"the {codec.upper()} materialized chunk byte count overflows signed 64-bit bounds",
                what=f"channel={channel.name!r}, total={total}, add={channel_size}",
                how="use channel sampling geometry representable within signed 64-bit byte offsets",
            )
        total += channel_size
    return total


def _chunk_channel_geometry(
    channel: _ExrChannel,
    *,
    width: int,
    chunk_y: int,
    row_count: int,
) -> tuple[int, int]:
    sampling = channel.sampling
    channel_width = width if sampling is None else sampling.width
    channel_rows = sum(file_y % channel.y_sampling == 0 for file_y in range(chunk_y, chunk_y + row_count))
    return channel_width, channel_rows


def _phase3_channel_rows(
    codec: str,
    channels: Sequence[_ExrChannel],
    *,
    width: int,
    chunk_y: int,
    row_start: int,
    row_count: int,
) -> tuple[_Phase3ChannelRow, ...]:
    section_offsets: list[int] = []
    section_cursor = 0
    for channel in channels:
        channel_width, channel_rows = _chunk_channel_geometry(
            channel,
            width=width,
            chunk_y=chunk_y,
            row_count=row_count,
        )
        section_offsets.append(section_cursor)
        section_cursor += channel_width * channel.bytes_per_sample * channel_rows

    rows: list[_Phase3ChannelRow] = []
    raw_cursor = 0
    pxr24_cursor = 0
    for chunk_row in range(row_count):
        file_y = chunk_y + chunk_row
        for channel_index, channel in enumerate(channels):
            if file_y % channel.y_sampling:
                continue
            channel_width, _ = _chunk_channel_geometry(
                channel,
                width=width,
                chunk_y=chunk_y,
                row_count=row_count,
            )
            raw_size = channel_width * channel.bytes_per_sample
            raw_span = _Phase3ByteSpan(raw_cursor, raw_cursor + raw_size)
            if codec == "pxr24":
                materialized_size = channel_width * _EXR_PXR24_PLANE_COUNTS[channel.pixel_type]
                materialized_span = _Phase3ByteSpan(pxr24_cursor, pxr24_cursor + materialized_size)
                pxr24_cursor += materialized_size
            elif codec in ("b44", "b44a"):
                channel_row = sum(candidate_y % channel.y_sampling == 0 for candidate_y in range(chunk_y, file_y))
                materialized_start = section_offsets[channel_index] + channel_row * raw_size
                materialized_span = _Phase3ByteSpan(materialized_start, materialized_start + raw_size)
            else:
                materialized_span = raw_span
            rows.append(
                _Phase3ChannelRow(
                    channel_index=channel_index,
                    channel_name=channel.name,
                    pixel_type=channel.pixel_type,
                    bytes_per_sample=channel.bytes_per_sample,
                    perceptually_linear=channel.perceptually_linear,
                    chunk_row=chunk_row,
                    file_y=file_y,
                    output_row=row_start + chunk_row,
                    raw_span=raw_span,
                    materialized_span=materialized_span,
                )
            )
            raw_cursor += raw_size
    return tuple(rows)


def _parse_phase3_rle_packets(
    payload: bytes | memoryview,
    *,
    payload_start: int,
    chunk_y: int,
    expected_size: int,
) -> _Phase3RlePackets:
    payload_view = memoryview(payload).cast("B")
    input_offset = 0
    output_offset = 0
    packet_count = 0
    while input_offset < len(payload_view):
        packet_start = input_offset
        header_byte = int(payload_view[input_offset])
        header = header_byte if header_byte < 128 else header_byte - 256
        input_offset += 1
        literal = header < 0
        output_size = -header if literal else header + 1
        packet_data_size = output_size if literal else 1
        packet_end = input_offset + packet_data_size
        if packet_end > len(payload_view):
            raise _phase3_error(
                why="the RLE packet payload is truncated after its signed header",
                what=(
                    f"chunk_y={chunk_y}, packet_offset={payload_start + packet_start}, "
                    f"declared_bytes={packet_data_size}, remaining={len(payload_view) - input_offset}"
                ),
                how="provide every literal byte or the repeated run byte declared by the RLE packet head",
            )
        output_end = output_offset + output_size
        if output_end > expected_size:
            raise _phase3_error(
                why="the RLE packet stream expands beyond the descriptor materialized size",
                what=f"chunk_y={chunk_y}, output_end={output_end}, expected={expected_size}",
                how="make the packet output consume the expected transformed chunk bytes exactly",
            )
        packet_count += 1
        input_offset = packet_end
        output_offset = output_end
    if output_offset != expected_size:
        raise _phase3_error(
            why="the RLE packet stream does not produce the descriptor materialized size",
            what=f"chunk_y={chunk_y}, produced={output_offset}, expected={expected_size}",
            how="make the RLE packets consume the input and produce the transformed chunk exactly once",
        )
    return _Phase3RlePackets(
        payload=payload_view,
        payload_start=payload_start,
        packet_count=packet_count,
    )


def _parse_phase3_pxr24_planes(
    payload: bytes,
    channels: Sequence[_ExrChannel],
    *,
    width: int,
    payload_start: int,
    chunk_y: int,
    row_start: int,
    row_count: int,
    expected_size: int,
) -> tuple[_Phase3Plane, ...]:
    if len(payload) < 6:
        raise _phase3_error(
            why="the PXR24 zlib wrapper is truncated before its Deflate payload and Adler-32 trailer",
            what=f"chunk_y={chunk_y}, stored_size={len(payload)}",
            how="provide one complete RFC 1950 stream for the PXR24 chunk",
        )
    cmf, flg = payload[0], payload[1]
    if (cmf & 0x0F) != 8 or (cmf >> 4) > 7 or ((cmf << 8) | flg) % 31:
        raise _phase3_error(
            why="the PXR24 zlib header has an invalid method, window, or FCHECK",
            what=f"chunk_y={chunk_y}, CMF=0x{cmf:02x}, FLG=0x{flg:02x}",
            how="encode one valid RFC 1950 Deflate stream with a correct header check",
        )
    if flg & 0x20:
        raise _phase3_error(
            why="the PXR24 zlib stream requests a preset dictionary",
            what=f"chunk_y={chunk_y}, FLG=0x{flg:02x}",
            how="encode PXR24 without an RFC 1950 preset dictionary",
        )
    decompressor = zlib.decompressobj()
    try:
        materialized = decompressor.decompress(payload, expected_size + 1)
    except zlib.error as error:
        raise _phase3_error(
            why="the PXR24 chunk contains an invalid zlib stream",
            what=f"chunk_y={chunk_y}, payload={payload_start}:{payload_start + len(payload)}, error={error}",
            how="encode one complete zlib stream covering the row-channel plane bytes",
        ) from error
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise _phase3_error(
            why="the PXR24 zlib stream does not end exactly at the chunk payload boundary",
            what=(
                f"chunk_y={chunk_y}, eof={decompressor.eof}, trailing={len(decompressor.unused_data)}, "
                f"unconsumed={len(decompressor.unconsumed_tail)}"
            ),
            how="store exactly one complete RFC 1950 stream and no trailing bytes in the chunk",
        )
    if len(materialized) != expected_size:
        raise _phase3_error(
            why="the PXR24 plane stream does not match the descriptor materialized size",
            what=f"chunk_y={chunk_y}, inflated={len(materialized)}, expected={expected_size}",
            how="emit every row-channel byte plane exactly once in the zlib stream",
        )

    planes: list[_Phase3Plane] = []
    materialized_offset = 0
    for chunk_row in range(row_count):
        file_y = chunk_y + chunk_row
        for channel_index, channel in enumerate(channels):
            if file_y % channel.y_sampling:
                continue
            channel_width, _ = _chunk_channel_geometry(
                channel,
                width=width,
                chunk_y=chunk_y,
                row_count=row_count,
            )
            for plane_index in range(_EXR_PXR24_PLANE_COUNTS[channel.pixel_type]):
                plane_end = materialized_offset + channel_width
                planes.append(
                    _Phase3Plane(
                        channel_index=channel_index,
                        channel_name=channel.name,
                        chunk_row=chunk_row,
                        output_row=row_start + chunk_row,
                        plane_index=plane_index,
                        materialized_span=_Phase3ByteSpan(materialized_offset, plane_end),
                    )
                )
                materialized_offset = plane_end
    return tuple(planes)


@lru_cache(maxsize=32)
def _b44a_section_pattern(block_count: int) -> re.Pattern[bytes]:
    count = str(block_count).encode("ascii")
    return re.compile(rb"(?s:(?:..[\x34-\xff]|..[\x00-\x33].{11})){" + count + rb"}")


def _validated_b44a_section_end(
    data: bytes,
    *,
    cursor: int,
    payload_end: int,
    block_count: int,
    block_columns: int,
    chunk_y: int,
    channel_name: str,
) -> int:
    match = _b44a_section_pattern(block_count).match(data, cursor, payload_end)
    if match is not None:
        return match.end()
    block_cursor = cursor
    for local_block in range(block_count):
        block_row, block_column = divmod(local_block, block_columns)
        if block_cursor + 3 > payload_end:
            raise _phase3_error(
                why="the B44A HALF block is truncated before its three-byte head",
                what=(
                    f"chunk_y={chunk_y}, channel={channel_name!r}, block=({block_row},{block_column}), "
                    f"remaining={payload_end - block_cursor}"
                ),
                how="provide the base and marker bytes for every 4-by-4 HALF block",
            )
        stored_size = 3 if data[block_cursor + 2] >= 0x34 else 14
        block_end = block_cursor + stored_size
        if block_end > payload_end:
            form = "flat" if stored_size == 3 else "dense"
            raise _phase3_error(
                why=f"the B44A {form} HALF block is truncated",
                what=(
                    f"chunk_y={chunk_y}, channel={channel_name!r}, block=({block_row},{block_column}), "
                    f"required={stored_size}, remaining={payload_end - block_cursor}"
                ),
                how=f"provide all {stored_size} bytes for the declared B44A block form",
            )
        block_cursor = block_end
    raise AssertionError("B44A block grammar regex rejected a completely scanned section")


def _parse_phase3_b44_sections(
    codec: str,
    data: bytes,
    channels: Sequence[_ExrChannel],
    *,
    width: int,
    payload_start: int,
    payload_end: int,
    chunk_y: int,
    row_start: int,
    row_count: int,
) -> tuple[tuple[_Phase3ChannelSection, ...], _Phase3B44Blocks]:
    cursor = payload_start
    block_columns = (width + 3) // 4
    total_block_count = 0
    sections: list[_Phase3ChannelSection] = []
    materialized_block_count = 0
    for channel_index, channel in enumerate(channels):
        channel_width, channel_rows = _chunk_channel_geometry(
            channel,
            width=width,
            chunk_y=chunk_y,
            row_count=row_count,
        )
        block_columns = (channel_width + 3) // 4
        block_rows = (channel_rows + 3) // 4
        blocks_per_half_channel = block_rows * block_columns
        if channel.pixel_type == 1:
            total_block_count += blocks_per_half_channel
        section_start = cursor
        block_start = materialized_block_count
        block_count = 0
        materialized_size = channel_width * channel_rows * channel.bytes_per_sample
        if channel.pixel_type != 1:
            cursor += materialized_size
            if cursor > payload_end:
                raise _phase3_error(
                    why=f"the {codec.upper()} raw channel section is truncated",
                    what=(
                        f"chunk_y={chunk_y}, channel={channel.name!r}, section={section_start}:{cursor}, "
                        f"payload_end={payload_end}"
                    ),
                    how="provide every FLOAT or UINT plane byte declared by the channel geometry",
                )
        elif codec == "b44":
            required_size = blocks_per_half_channel * 14
            remaining_size = payload_end - cursor
            if remaining_size < required_size:
                local_block = max(0, remaining_size // 14)
                block_row, block_column = divmod(local_block, block_columns)
                block_cursor = cursor + local_block * 14
                if payload_end - block_cursor < 3:
                    raise _phase3_error(
                        why="the B44 HALF block is truncated before its three-byte head",
                        what=(
                            f"chunk_y={chunk_y}, channel={channel.name!r}, block=({block_row},{block_column}), "
                            f"remaining={payload_end - block_cursor}"
                        ),
                        how="provide the base and marker bytes for every 4-by-4 HALF block",
                    )
                raise _phase3_error(
                    why="the B44 dense HALF block is truncated",
                    what=(
                        f"chunk_y={chunk_y}, channel={channel.name!r}, block=({block_row},{block_column}), "
                        f"required=14, remaining={payload_end - block_cursor}"
                    ),
                    how="provide all 14 bytes for the declared B44 block form",
                )
            block_bytes = np.frombuffer(data, dtype=np.uint8, count=required_size, offset=cursor).reshape(-1, 14)
            invalid_markers = np.flatnonzero(block_bytes[:, 2] >= np.uint8(0x34))
            if invalid_markers.size:
                local_block = int(invalid_markers[0])
                block_row, block_column = divmod(local_block, block_columns)
                marker = int(block_bytes[local_block, 2])
                raise _phase3_error(
                    why="the B44 dense block head contains an invalid shift or flat marker",
                    what=(
                        f"chunk_y={chunk_y}, channel={channel.name!r}, "
                        f"block=({block_row},{block_column}), byte2=0x{marker:02x}"
                    ),
                    how="encode B44 as a 14-byte dense block with byte[2] below 0x34",
                )
            cursor += required_size
            block_count = blocks_per_half_channel
            materialized_block_count += block_count
        else:
            cursor = _validated_b44a_section_end(
                data,
                cursor=cursor,
                payload_end=payload_end,
                block_count=blocks_per_half_channel,
                block_columns=block_columns,
                chunk_y=chunk_y,
                channel_name=channel.name,
            )
            block_count = blocks_per_half_channel
            materialized_block_count += block_count
        sections.append(
            _Phase3ChannelSection(
                channel_index=channel_index,
                channel_name=channel.name,
                pixel_type=channel.pixel_type,
                bytes_per_sample=channel.bytes_per_sample,
                perceptually_linear=channel.perceptually_linear,
                payload_span=_Phase3ByteSpan(section_start, cursor),
                expected_materialized_size=materialized_size,
                block_start=block_start,
                block_count=block_count,
            )
        )
    if cursor != payload_end:
        raise _phase3_error(
            why=f"the {codec.upper()} channel sections do not consume the chunk payload exactly",
            what=f"chunk_y={chunk_y}, consumed_end={cursor}, payload_end={payload_end}",
            how="make the file-channel-order sections cover the compressed payload once with no trailing bytes",
        )
    if materialized_block_count != total_block_count:
        raise AssertionError("B44 block metadata count diverged from the validated channel geometry")
    section_tuple = tuple(sections)
    block_sections = tuple(section for section in section_tuple if section.block_count)
    return section_tuple, _Phase3B44Blocks(
        payload=data,
        codec=codec,
        block_sections=block_sections,
        block_starts=tuple(section.block_start for section in block_sections),
        block_columns=block_columns,
        row_start=row_start,
        row_count=row_count,
    )


def _piz_channel_planes(
    channels: Sequence[_ExrChannel],
    *,
    width: int,
    chunk_y: int,
    row_count: int,
) -> tuple[tuple[_PizChannelPlane, ...], int]:
    word_offset = 0
    planes: list[_PizChannelPlane] = []
    for channel_index, channel in enumerate(channels):
        channel_width, channel_rows = _chunk_channel_geometry(
            channel,
            width=width,
            chunk_y=chunk_y,
            row_count=row_count,
        )
        sample_count = _checked_product(
            channel_width,
            channel_rows,
            context=f"PIZ channel {channel.name!r} sample count",
        )
        word_slice_count = channel.bytes_per_sample // 2
        word_count = _checked_product(
            sample_count,
            word_slice_count,
            context=f"PIZ channel {channel.name!r} word count",
        )
        planes.append(
            _PizChannelPlane(
                channel_index=channel_index,
                channel_name=channel.name,
                pixel_type=channel.pixel_type,
                bytes_per_sample=channel.bytes_per_sample,
                sample_count=sample_count,
                word_slice_count=word_slice_count,
                word_offset=word_offset,
                word_count=word_count,
            )
        )
        if word_offset > _EXR_MAX_INTEGER - word_count:
            raise _piz_error(
                why="the PIZ channel-plane word offsets overflow signed 64-bit bounds",
                what=f"channel={channel.name!r}, word_offset={word_offset}, word_count={word_count}",
                how="use image dimensions and channel counts representable within 64-bit word offsets",
            )
        word_offset += word_count
    return tuple(planes), word_offset


def _piz_chunk_descriptor(
    data: bytes,
    part: _ExrPart,
    *,
    width: int,
    lines_per_chunk: int,
    chunk_y: int,
    row_start: int,
    row_count: int,
    payload_start: int,
    payload_end: int,
    expected_packed_size: int,
    raw_stored: bool,
) -> _PizChunkDescriptor:
    channel_planes, expected_output_word_count = _piz_channel_planes(
        part.channels,
        width=width,
        chunk_y=chunk_y,
        row_count=row_count,
    )
    if expected_output_word_count * 2 != expected_packed_size:
        raise _piz_error(
            why="the PIZ channel-plane word count differs from the packed chunk byte count",
            what=(
                f"chunk_y={chunk_y}, words={expected_output_word_count}, expected_packed_size={expected_packed_size}"
            ),
            how="derive one 16-bit word for HALF and low/high words for FLOAT or UINT samples",
        )
    empty_span = _PizByteSpan(payload_start, payload_start)
    if raw_stored:
        return _PizChunkDescriptor(
            lines_per_chunk=lines_per_chunk,
            chunk_y=chunk_y,
            row_start=row_start,
            row_count=row_count,
            output_row_span=(row_start, row_start + row_count),
            payload_span=_PizByteSpan(payload_start, payload_end),
            stored_size=payload_end - payload_start,
            expected_packed_size=expected_packed_size,
            expected_output_word_count=expected_output_word_count,
            raw_stored=raw_stored,
            channel_planes=channel_planes,
            bitmap_range=None,
            bitmap_span=empty_span,
            huffman_byte_count=0,
            huffman_count_span=empty_span,
            huffman_span=empty_span,
            huffman_leader=None,
            trailing_span=empty_span,
        )

    if payload_end - payload_start < 4:
        raise _piz_error(
            why="the compressed PIZ payload is truncated before its bitmap leader",
            what=f"chunk_y={chunk_y}, stored_size={payload_end - payload_start}, required=4",
            how="provide the little-endian minimum and maximum bitmap byte indices",
        )
    bitmap_minimum, bitmap_maximum = struct.unpack_from("<HH", data, payload_start)
    if bitmap_minimum >= _PIZ_BITMAP_BYTE_COUNT or bitmap_maximum >= _PIZ_BITMAP_BYTE_COUNT:
        raise _piz_error(
            why="the PIZ bitmap leader indexes outside the 8192-byte bitmap",
            what=f"chunk_y={chunk_y}, minimum={bitmap_minimum}, maximum={bitmap_maximum}",
            how="encode bitmap byte indices in the inclusive range 0 through 8191",
        )
    bitmap_size = bitmap_maximum - bitmap_minimum + 1 if bitmap_minimum <= bitmap_maximum else 0
    bitmap_start = payload_start + 4
    bitmap_end = bitmap_start + bitmap_size
    huffman_count_end = bitmap_end + 4
    if huffman_count_end > payload_end:
        raise _piz_error(
            why="the PIZ bitmap slice or Huffman byte count extends beyond the chunk payload",
            what=(
                f"chunk_y={chunk_y}, bitmap={bitmap_start}:{bitmap_end}, "
                f"huffman_count_end={huffman_count_end}, payload_end={payload_end}"
            ),
            how="store the inclusive bitmap slice followed by one complete little-endian u32 Huffman byte count",
        )
    huffman_byte_count = struct.unpack_from("<I", data, bitmap_end)[0]
    huffman_start = huffman_count_end
    huffman_end = huffman_start + huffman_byte_count
    if huffman_end > payload_end:
        raise _piz_error(
            why="the PIZ Huffman section exceeds the remaining chunk payload",
            what=(
                f"chunk_y={chunk_y}, huffman={huffman_start}:{huffman_end}, "
                f"payload_end={payload_end}, hufByteCount={huffman_byte_count}"
            ),
            how="bound hufByteCount to the bytes available after the bitmap slice",
        )
    if huffman_byte_count < _PIZ_HUFFMAN_LEADER_SIZE:
        raise _piz_error(
            why="the PIZ Huffman section is truncated before its fixed-width leader",
            what=f"chunk_y={chunk_y}, hufByteCount={huffman_byte_count}, required={_PIZ_HUFFMAN_LEADER_SIZE}",
            how="provide im, iM, tableByteCount, dataBitCount, and reserved as five little-endian u32 values",
        )
    minimum_symbol, maximum_symbol, table_byte_count, data_bit_count, reserved = struct.unpack_from(
        "<IIIII", data, huffman_start
    )
    if minimum_symbol > maximum_symbol or maximum_symbol > (1 << 16):
        raise _piz_error(
            why="the PIZ Huffman symbol bounds are reversed or exceed the pseudo-symbol domain",
            what=f"chunk_y={chunk_y}, im={minimum_symbol}, iM={maximum_symbol}",
            how="use an inclusive symbol range ending no later than pseudo symbol 65536",
        )
    declared_data_bytes = (data_bit_count + 7) // 8
    if declared_data_bytes > huffman_byte_count - _PIZ_HUFFMAN_LEADER_SIZE:
        raise _piz_error(
            why="the PIZ Huffman data bit count cannot fit within the declared section",
            what=(
                f"chunk_y={chunk_y}, dataBitCount={data_bit_count}, data_bytes={declared_data_bytes}, "
                f"hufByteCount={huffman_byte_count}"
            ),
            how="bound the declared data bits to the Huffman section after its fixed-width leader",
        )
    leader_span = _PizByteSpan(huffman_start, huffman_start + _PIZ_HUFFMAN_LEADER_SIZE)
    return _PizChunkDescriptor(
        lines_per_chunk=lines_per_chunk,
        chunk_y=chunk_y,
        row_start=row_start,
        row_count=row_count,
        output_row_span=(row_start, row_start + row_count),
        payload_span=_PizByteSpan(payload_start, payload_end),
        stored_size=payload_end - payload_start,
        expected_packed_size=expected_packed_size,
        expected_output_word_count=expected_output_word_count,
        raw_stored=raw_stored,
        channel_planes=channel_planes,
        bitmap_range=(bitmap_minimum, bitmap_maximum),
        bitmap_span=_PizByteSpan(bitmap_start, bitmap_end),
        huffman_byte_count=huffman_byte_count,
        huffman_count_span=_PizByteSpan(bitmap_end, huffman_count_end),
        huffman_span=_PizByteSpan(huffman_start, huffman_end),
        huffman_leader=_PizHuffmanLeader(
            minimum_symbol=minimum_symbol,
            maximum_symbol=maximum_symbol,
            table_byte_count=table_byte_count,
            data_bit_count=data_bit_count,
            reserved=reserved,
            span=leader_span,
        ),
        trailing_span=_PizByteSpan(huffman_end, payload_end),
    )


def _piz_reverse_lut(
    bitmap_minimum: int,
    bitmap_maximum: int,
    bitmap_slice: bytes,
) -> tuple[np.ndarray, int]:
    if not 0 <= bitmap_minimum < _PIZ_BITMAP_BYTE_COUNT or not 0 <= bitmap_maximum < _PIZ_BITMAP_BYTE_COUNT:
        raise _piz_error(
            why="the PIZ reverse-LUT bitmap range lies outside the 8192-byte domain",
            what=f"minimum={bitmap_minimum}, maximum={bitmap_maximum}",
            how="use bitmap byte indices in the inclusive range 0 through 8191",
        )
    expected_size = bitmap_maximum - bitmap_minimum + 1 if bitmap_minimum <= bitmap_maximum else 0
    if len(bitmap_slice) != expected_size:
        raise _piz_error(
            why="the PIZ reverse-LUT bitmap slice length differs from its inclusive range",
            what=f"minimum={bitmap_minimum}, maximum={bitmap_maximum}, received={len(bitmap_slice)}, expected={expected_size}",
            how="provide exactly max-min+1 bitmap bytes, or no bytes for an empty range",
        )
    bitmap = np.zeros(_PIZ_BITMAP_BYTE_COUNT, dtype=np.uint8)
    if expected_size:
        bitmap[bitmap_minimum : bitmap_maximum + 1] = np.frombuffer(bitmap_slice, dtype=np.uint8)
    marked = np.flatnonzero(np.unpackbits(bitmap, bitorder="little"))
    marked = marked[marked != 0]
    reverse = np.zeros(1 << 16, dtype=np.uint16)
    reverse[0] = 0
    if marked.size:
        reverse[1 : marked.size + 1] = marked.astype(np.uint16, copy=False)
    return reverse, int(marked.size)


def _piz_uses_w14(max_value: int) -> bool:
    if not 0 <= max_value <= 0xFFFF:
        raise _piz_error(
            why="the PIZ compact maximum lies outside the 16-bit LUT domain",
            what=f"max_value={max_value}",
            how="derive maxValue from the implicit-zero reverse LUT rank",
        )
    return max_value < (1 << 14)


def _read_piz_bits(data: bytes, bit_offset: int, width: int, *, field: str) -> tuple[int, int]:
    if width < 0 or bit_offset < 0 or bit_offset + width > len(data) * 8:
        raise _piz_error(
            why=f"the PIZ Huffman {field} is truncated within a packed field",
            what=f"bit_offset={bit_offset}, requested_bits={width}, available_bits={len(data) * 8}",
            how="provide the complete six-bit length token, long-run count, or encoded symbol",
        )
    value = 0
    for absolute in range(bit_offset, bit_offset + width):
        value = (value << 1) | ((data[absolute // 8] >> (7 - absolute % 8)) & 1)
    return value, bit_offset + width


def _canonical_piz_codes(minimum_symbol: int, lengths: Sequence[int]) -> tuple[_DwaHuffmanCode, ...]:
    length_array = np.asarray(lengths, dtype=np.uint8)
    observed_offsets = np.flatnonzero(length_array)
    if not observed_offsets.size:
        raise _piz_error(
            why="the PIZ Huffman code-length assignment contains no decodable symbol",
            what=f"minimum_symbol={minimum_symbol}, symbol_count={len(lengths)}",
            how="assign a code length between 1 and 58 to an actual symbol and the repeat pseudo-symbol",
        )
    observed_lengths = length_array[observed_offsets].astype(np.int64, copy=False)
    if np.any(observed_lengths > _DWA_MAX_HUFFMAN_CODE_LENGTH):
        raise _piz_error(
            why="the PIZ Huffman code-length assignment exceeds 58 bits",
            what=f"maximum_length={int(observed_lengths.max())}",
            how="encode only canonical lengths in the inclusive range 1 through 58",
        )
    counts = np.bincount(observed_lengths, minlength=_DWA_MAX_HUFFMAN_CODE_LENGTH + 1)
    maximum_length = int(observed_lengths.max())
    capacity = 1 << maximum_length
    occupancy = sum(int(counts[length]) << (maximum_length - length) for length in range(1, maximum_length + 1))
    if occupancy > capacity:
        raise _piz_error(
            why="the PIZ Huffman code-length assignment is oversubscribed",
            what=f"occupancy={occupancy}, capacity={capacity}, maximum_length={maximum_length}",
            how="reduce short code counts until the canonical assignment is prefix-free",
        )
    next_codes = np.zeros(_DWA_MAX_HUFFMAN_CODE_LENGTH + 1, dtype=np.uint64)
    code = 0
    for length in range(_DWA_MAX_HUFFMAN_CODE_LENGTH, 0, -1):
        next_codes[length] = code
        code = (code + int(counts[length])) >> 1
    codes: list[_DwaHuffmanCode] = []
    intervals: list[tuple[int, int, int]] = []
    for offset_value in observed_offsets:
        offset = int(offset_value)
        length = int(length_array[offset])
        assigned = int(next_codes[length])
        next_codes[length] += 1
        symbol = minimum_symbol + offset
        if assigned >= 1 << length:
            raise _piz_error(
                why="the PIZ canonical Huffman code exceeds its declared length",
                what=f"symbol={symbol}, code={assigned}, length={length}",
                how="provide an in-range prefix-free canonical length assignment",
            )
        intervals.append(
            (
                assigned << (_DWA_MAX_HUFFMAN_CODE_LENGTH - length),
                (assigned + 1) << (_DWA_MAX_HUFFMAN_CODE_LENGTH - length),
                symbol,
            )
        )
        codes.append(_DwaHuffmanCode(symbol=symbol, length=length, code=assigned))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if previous[1] > current[0]:
            raise _piz_error(
                why="the PIZ canonical Huffman assignment contains duplicate or prefix-conflicting codes",
                what=f"symbols={(previous[2], current[2])!r}",
                how="provide non-overlapping canonical code lengths across im through iM",
            )
    return tuple(codes)


def _parse_piz_huffman_table(payload: bytes) -> _PizHuffmanTable:
    if len(payload) < _PIZ_HUFFMAN_LEADER_SIZE:
        raise _piz_error(
            why="the PIZ Huffman stream is truncated before its 20-byte leader",
            what=f"received={len(payload)}, required={_PIZ_HUFFMAN_LEADER_SIZE}",
            how="provide im, iM, tableByteCount, dataBitCount, and reserved",
        )
    minimum_symbol, maximum_symbol, declared_table_bytes, data_bit_count, reserved = struct.unpack_from(
        "<IIIII", payload
    )
    if minimum_symbol > maximum_symbol or maximum_symbol > _DWA_MAX_HUFFMAN_SYMBOL:
        raise _piz_error(
            why="the PIZ Huffman symbol range is reversed or outside the pseudo-symbol domain",
            what=f"im={minimum_symbol}, iM={maximum_symbol}",
            how="encode an ordered inclusive range ending no later than 65536",
        )
    symbol_count = maximum_symbol - minimum_symbol + 1
    packed = payload[_PIZ_HUFFMAN_LEADER_SIZE:]
    lengths: list[int] = []
    bit_offset = 0
    while len(lengths) < symbol_count:
        token, bit_offset = _read_piz_bits(packed, bit_offset, 6, field="code-length table")
        if token <= _DWA_MAX_HUFFMAN_CODE_LENGTH:
            lengths.append(token)
            continue
        if token == 63:
            extra, bit_offset = _read_piz_bits(packed, bit_offset, 8, field="code-length long zero run")
            run = extra + 6
        else:
            run = token - 57
        if len(lengths) + run > symbol_count:
            raise _piz_error(
                why="the PIZ Huffman code-length table run overshoots iM+1",
                what=f"decoded={len(lengths)}, run={run}, symbols={symbol_count}, token={token}",
                how="end every zero-length run at or before the declared maximum symbol",
            )
        lengths.extend((0,) * run)
    consumed_table_bytes = (bit_offset + 7) // 8
    table_start = _PIZ_HUFFMAN_LEADER_SIZE
    table_end = table_start + consumed_table_bytes
    data_byte_count = (data_bit_count + 7) // 8
    data_end = table_end + data_byte_count
    if data_end > len(payload):
        raise _piz_error(
            why="the PIZ Huffman encoded data is truncated after the derived code-length table",
            what=(
                f"table_end={table_end}, dataBitCount={data_bit_count}, data_end={data_end}, "
                f"hufByteCount={len(payload)}"
            ),
            how="provide every declared data bit within the bounded Huffman section",
        )
    codes = _canonical_piz_codes(minimum_symbol, lengths)
    if not any(code.symbol != maximum_symbol for code in codes) or not any(
        code.symbol == maximum_symbol for code in codes
    ):
        raise _piz_error(
            why="the PIZ Huffman table does not code both an actual symbol and its repeat pseudo-symbol",
            what=f"im={minimum_symbol}, iM={maximum_symbol}, coded={tuple(code.symbol for code in codes)!r}",
            how="assign a nonzero length to at least one actual symbol and iM",
        )
    return _PizHuffmanTable(
        minimum_symbol=minimum_symbol,
        maximum_symbol=maximum_symbol,
        declared_table_byte_count=declared_table_bytes,
        data_bit_count=data_bit_count,
        reserved=reserved,
        code_lengths=tuple(lengths),
        codes=codes,
        table_span=_PizByteSpan(table_start, table_end),
        data_span=_PizByteSpan(table_end, data_end),
    )


def _decode_piz_huffman_host(
    payload: bytes,
    table: _PizHuffmanTable,
    *,
    expected_count: int,
) -> np.ndarray:
    if expected_count < 0 or table.data_span.start < 0 or table.data_span.end > len(payload):
        raise _piz_error(
            why="the PIZ Huffman output count or encoded-data span is outside its bounded domain",
            what=f"expected={expected_count}, span={table.data_span.start}:{table.data_span.end}, payload={len(payload)}",
            how="decode a non-negative word count from the matching bounded Huffman section",
        )
    codes_by_length: dict[int, dict[int, int]] = {}
    for item in table.codes:
        codes_by_length.setdefault(item.length, {})[item.code] = item.symbol
    encoded = payload[table.data_span.start : table.data_span.end]
    output = np.empty(expected_count, dtype=np.uint16)
    produced = 0
    bit_offset = 0
    while produced < expected_count:
        code = 0
        symbol: int | None = None
        for length in range(1, _DWA_MAX_HUFFMAN_CODE_LENGTH + 1):
            if bit_offset >= table.data_bit_count:
                raise _piz_error(
                    why="the PIZ Huffman data ends before the expected word count",
                    what=f"decoded={produced}, expected={expected_count}, bit_offset={bit_offset}",
                    how="provide a complete canonical code path for every output word",
                )
            bit, bit_offset = _read_piz_bits(encoded, bit_offset, 1, field="encoded symbol")
            code = (code << 1) | bit
            symbol = codes_by_length.get(length, {}).get(code)
            if symbol is not None:
                break
        if symbol is None:
            raise _piz_error(
                why="the PIZ Huffman data contains an invalid canonical prefix",
                what=f"decoded={produced}, bit_offset={bit_offset}",
                how="encode every word with one declared prefix-free canonical code",
            )
        if symbol == table.maximum_symbol:
            if produced == 0:
                raise _piz_error(
                    why="the PIZ Huffman repeat pseudo-symbol has no previous literal",
                    what=f"decoded={produced}, bit_offset={bit_offset}",
                    how="place every repeat pseudo-symbol after a literal word",
                )
            if bit_offset + 8 > table.data_bit_count:
                raise _piz_error(
                    why="the PIZ Huffman repeat count is truncated",
                    what=f"decoded={produced}, bit_offset={bit_offset}, data_bits={table.data_bit_count}",
                    how="append the complete unsigned eight-bit previous-symbol count",
                )
            repeat_count, bit_offset = _read_piz_bits(encoded, bit_offset, 8, field="repeat count")
            if produced + repeat_count > expected_count:
                raise _piz_error(
                    why="the PIZ Huffman repeat would overflow the expected word count",
                    what=f"decoded={produced}, repeat={repeat_count}, expected={expected_count}",
                    how="bound every previous-symbol run to the remaining output capacity",
                )
            output[produced : produced + repeat_count] = output[produced - 1]
            produced += repeat_count
            continue
        if symbol > 0xFFFF:
            raise _piz_error(
                why="the PIZ Huffman literal lies outside the 16-bit word domain",
                what=f"symbol={symbol}, decoded={produced}",
                how="reserve only iM as the non-literal repeat pseudo-symbol",
            )
        output[produced] = symbol
        produced += 1
    return output


def _inverse_piz_pair(a_word: int, b_word: int, *, w14: bool) -> tuple[int, int]:
    if w14:
        low = a_word - 0x10000 if a_word & 0x8000 else a_word
        high = b_word - 0x10000 if b_word & 0x8000 else b_word
        first = low + (high & 1) + (high >> 1)
        second = first - high
        return first & 0xFFFF, second & 0xFFFF
    second = (a_word - (b_word >> 1)) & 0xFFFF
    first = (b_word + second - 32768) & 0xFFFF
    return first, second


def _piz_inverse_wavelet_host(
    words: np.ndarray,
    *,
    nx: int,
    ny: int,
    word_stride: int,
    word_slice: int,
    max_value: int,
) -> None:
    values = np.asarray(words, dtype=np.uint16)
    if nx < 1 or ny < 1 or word_stride < 1 or not 0 <= word_slice < word_stride:
        raise _piz_error(
            why="the PIZ inverse-wavelet field geometry is invalid",
            what=f"nx={nx}, ny={ny}, word_stride={word_stride}, word_slice={word_slice}",
            how="decode a positive sampled plane and an in-range independent word slice",
        )
    required = word_slice + (ny - 1) * word_stride * nx + (nx - 1) * word_stride + 1
    if required > values.size:
        raise _piz_error(
            why="the PIZ inverse-wavelet field extends beyond its owning word plane",
            what=f"required={required}, words={values.size}, nx={nx}, ny={ny}",
            how="match each field geometry and stride to its channel-plane descriptor",
        )
    if min(nx, ny) < 2:
        return
    w14 = _piz_uses_w14(max_value)
    x_stride = word_stride
    y_stride = word_stride * nx
    p2 = 1 << (min(nx, ny).bit_length() - 1)
    p = p2 // 2
    while p >= 1:
        step = p * 2
        for y in range(0, ny - step + 1, step):
            for x in range(0, nx - step + 1, step):
                i00 = word_slice + y * y_stride + x * x_stride
                i01 = i00 + p * x_stride
                i10 = i00 + p * y_stride
                i11 = i10 + p * x_stride
                low0, low1 = _inverse_piz_pair(int(values[i00]), int(values[i10]), w14=w14)
                high0, high1 = _inverse_piz_pair(int(values[i01]), int(values[i11]), w14=w14)
                values[i00], values[i01] = _inverse_piz_pair(low0, high0, w14=w14)
                values[i10], values[i11] = _inverse_piz_pair(low1, high1, w14=w14)
        if nx & p:
            x = (nx // step) * step
            for y in range(0, ny - step + 1, step):
                first = word_slice + y * y_stride + x * x_stride
                second = first + p * y_stride
                values[first], values[second] = _inverse_piz_pair(int(values[first]), int(values[second]), w14=w14)
        if ny & p:
            y = (ny // step) * step
            for x in range(0, nx - step + 1, step):
                first = word_slice + y * y_stride + x * x_stride
                second = first + p * x_stride
                values[first], values[second] = _inverse_piz_pair(int(values[first]), int(values[second]), w14=w14)
        p //= 2


def _phase3_chunk_descriptor(
    data: bytes,
    part: _ExrPart,
    *,
    width: int,
    lines_per_chunk: int,
    chunk_y: int,
    row_start: int,
    row_count: int,
    payload_start: int,
    payload_end: int,
    expected_raw_size: int,
    raw_stored: bool,
) -> _Phase3ChunkDescriptor:
    codec = part.compression
    expected_materialized_size = _phase3_expected_materialized_size(
        codec,
        part.channels,
        width=width,
        chunk_y=chunk_y,
        row_count=row_count,
    )
    channel_rows = _phase3_channel_rows(
        codec,
        part.channels,
        width=width,
        chunk_y=chunk_y,
        row_start=row_start,
        row_count=row_count,
    )
    planes: tuple[_Phase3Plane, ...] = ()
    packets: tuple[_Phase3Packet, ...] | _Phase3RlePackets = ()
    channel_sections: tuple[_Phase3ChannelSection, ...] = ()
    blocks: tuple[_Phase3Block, ...] | _Phase3B44Blocks = ()
    if not raw_stored:
        if codec == "rle":
            packets = _parse_phase3_rle_packets(
                memoryview(data)[payload_start:payload_end],
                payload_start=payload_start,
                chunk_y=chunk_y,
                expected_size=expected_materialized_size,
            )
        elif codec == "pxr24":
            payload = data[payload_start:payload_end]
            planes = _parse_phase3_pxr24_planes(
                payload,
                part.channels,
                width=width,
                payload_start=payload_start,
                chunk_y=chunk_y,
                row_start=row_start,
                row_count=row_count,
                expected_size=expected_materialized_size,
            )
        else:
            channel_sections, blocks = _parse_phase3_b44_sections(
                codec,
                data,
                part.channels,
                width=width,
                payload_start=payload_start,
                payload_end=payload_end,
                chunk_y=chunk_y,
                row_start=row_start,
                row_count=row_count,
            )
    return _Phase3ChunkDescriptor(
        codec=codec,
        lines_per_chunk=lines_per_chunk,
        chunk_y=chunk_y,
        row_start=row_start,
        row_count=row_count,
        payload_span=_Phase3ByteSpan(payload_start, payload_end),
        stored_size=payload_end - payload_start,
        expected_raw_size=expected_raw_size,
        expected_materialized_size=expected_materialized_size,
        raw_stored=raw_stored,
        channel_rows=channel_rows,
        planes=planes,
        packets=packets,
        channel_sections=channel_sections,
        blocks=blocks,
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
                why="the compressed EXR chunk is larger than its expected uncompressed bytes",
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
            or (part.compression == _EXR_PIZ_COMPRESSION and packed_size == 0)
        )
        dwa: _DwaChunkDescriptor | None = None
        phase3: _Phase3ChunkDescriptor | None = None
        piz: _PizChunkDescriptor | None = None
        if part.compression in _EXR_DWA_COMPRESSIONS:
            geometry = _dwa_geometry(width=width, row_count=row_count, lines_per_chunk=lines_per_chunk)
            if raw_stored:
                unknown_span, ac_span, dc_span, rle_span = _zero_dwa_spans(payload_start)
                dwa = _DwaChunkDescriptor(
                    geometry=geometry,
                    leader=None,
                    channel_rules=(),
                    channel_layout=_classify_default_dwa_channels(part.channels),
                    unknown_span=unknown_span,
                    ac_span=ac_span,
                    dc_span=dc_span,
                    rle_span=rle_span,
                    huffman=None,
                )
            else:
                dwa = _parse_dwa_chunk_payload(
                    data[payload_start:payload_end],
                    payload_offset=payload_start,
                    chunk_y=y,
                    expected_size=expected_size,
                    geometry=geometry,
                    channels=part.channels,
                )
        elif part.compression in _EXR_PHASE3_COMPRESSIONS:
            phase3 = _phase3_chunk_descriptor(
                data,
                part,
                width=width,
                lines_per_chunk=lines_per_chunk,
                chunk_y=y,
                row_start=row_start,
                row_count=row_count,
                payload_start=payload_start,
                payload_end=payload_end,
                expected_raw_size=expected_size,
                raw_stored=raw_stored,
            )
        elif part.compression == _EXR_PIZ_COMPRESSION:
            piz = _piz_chunk_descriptor(
                data,
                part,
                width=width,
                lines_per_chunk=lines_per_chunk,
                chunk_y=y,
                row_start=row_start,
                row_count=row_count,
                payload_start=payload_start,
                payload_end=payload_end,
                expected_packed_size=expected_size,
                raw_stored=raw_stored,
            )
        chunks.append(
            _ExrChunk(
                y=y,
                row_start=row_start,
                row_count=row_count,
                packed_size=packed_size,
                payload_start=payload_start,
                payload_end=payload_end,
                expected_size=expected_size,
                raw_stored=raw_stored,
                dwa=dwa,
                phase3=phase3,
                piz=piz,
                part_index=part.index,
                chunk_offset=chunk_offset,
                span_start=chunk_offset,
                span_end=payload_end,
            )
        )
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
        phase3=None,
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
                or (source_part.compression == _EXR_PIZ_COMPRESSION and source_chunk.packed_size == 0)
            )
        )
        dwa: _DwaChunkDescriptor | None = None
        phase3: _Phase3ChunkDescriptor | None = None
        piz: _PizChunkDescriptor | None = None
        if source_part.compression in _EXR_DWA_COMPRESSIONS:
            geometry = _dwa_geometry(
                width=width,
                row_count=source_chunk.row_count,
                lines_per_chunk=lines_per_chunk,
            )
            if raw_stored:
                unknown_span, ac_span, dc_span, rle_span = _zero_dwa_spans(source_chunk.payload_start)
                dwa = _DwaChunkDescriptor(
                    geometry=geometry,
                    leader=None,
                    channel_rules=(),
                    channel_layout=_classify_default_dwa_channels(channels),
                    unknown_span=unknown_span,
                    ac_span=ac_span,
                    dc_span=dc_span,
                    rle_span=rle_span,
                    huffman=None,
                )
            else:
                dwa = _parse_dwa_chunk_payload(
                    container.data[source_chunk.payload_start : source_chunk.payload_end],
                    payload_offset=source_chunk.payload_start,
                    chunk_y=source_chunk.y,
                    expected_size=expected_size,
                    geometry=geometry,
                    channels=channels,
                )
        elif source_part.compression in _EXR_PHASE3_COMPRESSIONS:
            phase3 = _phase3_chunk_descriptor(
                container.data,
                decoder_part,
                width=width,
                lines_per_chunk=lines_per_chunk,
                chunk_y=source_chunk.y,
                row_start=source_chunk.row_start,
                row_count=source_chunk.row_count,
                payload_start=source_chunk.payload_start,
                payload_end=source_chunk.payload_end,
                expected_raw_size=expected_size,
                raw_stored=raw_stored,
            )
        elif source_part.compression == _EXR_PIZ_COMPRESSION:
            piz = _piz_chunk_descriptor(
                container.data,
                decoder_part,
                width=width,
                lines_per_chunk=lines_per_chunk,
                chunk_y=source_chunk.y,
                row_start=source_chunk.row_start,
                row_count=source_chunk.row_count,
                payload_start=source_chunk.payload_start,
                payload_end=source_chunk.payload_end,
                expected_packed_size=expected_size,
                raw_stored=raw_stored,
            )
        chunks.append(
            replace(
                source_chunk,
                expected_size=expected_size,
                raw_stored=raw_stored,
                dwa=dwa,
                phase3=phase3,
                piz=piz,
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
    gpu_eligible = source_part.compression in _EXR_GPU_COMPRESSIONS and all(
        channel.pixel_type in (0, 1, 2) for channel in channels
    )
    dwa_candidate = source_part.compression in _EXR_DWA_COMPRESSIONS
    phase3_candidate = source_part.compression in _EXR_PHASE3_COMPRESSIONS
    piz_candidate = source_part.compression == _EXR_PIZ_COMPRESSION
    dwa_eligible = dwa_candidate and all(
        chunk.raw_stored
        or (
            chunk.dwa is not None
            and chunk.dwa.leader is not None
            and chunk.dwa.leader.version == 2
            and chunk.dwa.channel_layout is not None
            and (chunk.dwa.leader.ac_compressed_size == 0 or chunk.dwa.leader.ac_compression == _DWA_STATIC_HUFFMAN)
        )
        for chunk in decoded_chunks
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
        gpu_eligible=gpu_eligible,
        dwa_eligible=dwa_eligible,
        phase3_eligible=phase3_candidate and all(chunk.phase3 is not None for chunk in decoded_chunks),
        piz_eligible=piz_candidate and all(chunk.piz is not None for chunk in decoded_chunks),
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
    gpu_eligible = (
        not multipart
        and len(parts) == 1
        and not tiled
        and not deep
        and first.image_type == "scanlineimage"
        and first.compression in _EXR_GPU_COMPRESSIONS
        and all(channel.pixel_type in (0, 1, 2) for channel in first.channels)
        and all(channel.x_sampling == 1 and channel.y_sampling == 1 for channel in first.channels)
    )
    dwa_candidate = (
        not multipart
        and len(parts) == 1
        and not tiled
        and not deep
        and first.image_type == "scanlineimage"
        and first.compression in _EXR_DWA_COMPRESSIONS
        and all(channel.x_sampling == 1 and channel.y_sampling == 1 for channel in first.channels)
    )
    phase3_candidate = (
        not multipart
        and len(parts) == 1
        and not tiled
        and not deep
        and first.image_type == "scanlineimage"
        and first.compression in _EXR_PHASE3_COMPRESSIONS
        and all(channel.pixel_type in _EXR_DTYPE_INFO for channel in first.channels)
        and all(channel.x_sampling == 1 and channel.y_sampling == 1 for channel in first.channels)
    )
    piz_candidate = (
        not multipart
        and len(parts) == 1
        and not tiled
        and not deep
        and first.image_type == "scanlineimage"
        and first.compression == _EXR_PIZ_COMPRESSION
        and all(channel.pixel_type in _EXR_DTYPE_INFO for channel in first.channels)
        and all(channel.x_sampling == 1 and channel.y_sampling == 1 for channel in first.channels)
    )
    lines_per_chunk = _EXR_LINES_PER_CHUNK[first.compression]
    expected_chunk_count = (height + lines_per_chunk - 1) // lines_per_chunk
    offset_table: tuple[int, ...] = ()
    chunks: tuple[_ExrChunk, ...] = ()
    if gpu_eligible or dwa_candidate or phase3_candidate or piz_candidate:
        offset_table, chunks = _parse_candidate_chunks(data, offset, first, lines_per_chunk=lines_per_chunk)
        parts[0] = replace(first, offset_table=offset_table, chunks=chunks)
        first = parts[0]
    dwa_eligible = dwa_candidate and all(
        chunk.raw_stored
        or (
            chunk.dwa is not None
            and chunk.dwa.leader is not None
            and chunk.dwa.leader.version == 2
            and chunk.dwa.channel_layout is not None
            and (chunk.dwa.leader.ac_compressed_size == 0 or chunk.dwa.leader.ac_compression == _DWA_STATIC_HUFFMAN)
        )
        for chunk in chunks
    )
    phase3_eligible = phase3_candidate and all(chunk.phase3 is not None for chunk in chunks)
    piz_eligible = piz_candidate and all(chunk.piz is not None for chunk in chunks)
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
        gpu_eligible=gpu_eligible,
        dwa_eligible=dwa_eligible,
        phase3_eligible=phase3_eligible,
        piz_eligible=piz_eligible,
    )
