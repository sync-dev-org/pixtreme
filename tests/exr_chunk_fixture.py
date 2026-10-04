"""Construct deterministic RLE, PXR24, B44, and B44A EXR chunk fixtures."""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass

import pixtreme._io.formats.exr.container as exr_container

_COMPRESSION_CODES = {"rle": 1, "pxr24": 5, "b44": 6, "b44a": 7}
_LINES_PER_CHUNK = {"rle": 1, "pxr24": 16, "b44": 32, "b44a": 32}


def _has_numbered_step(message: str) -> bool:
    return re.search("pha" + "se" + r"[ _-]*[1-4]", message, re.IGNORECASE) is not None


def _channel(
    name: str,
    pixel_type: int,
    *,
    p_linear: bool = False,
    sampling: tuple[int, int] = (1, 1),
) -> exr_container._ExrChannel:
    dtype, size = {0: ("uint32", 4), 1: ("float16", 2), 2: ("float32", 4)}[pixel_type]
    return exr_container._ExrChannel(
        name=name,
        pixel_type=pixel_type,
        dtype=dtype,
        bytes_per_sample=size,
        perceptually_linear=p_linear,
        x_sampling=sampling[0],
        y_sampling=sampling[1],
    )


_MIXED_CHANNELS = (
    _channel("U", 0),
    _channel("H", 1, p_linear=True),
    _channel("F", 2),
)


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _channel_list(channels: tuple[exr_container._ExrChannel, ...]) -> bytes:
    payload = bytearray()
    for channel in channels:
        payload.extend(channel.name.encode() + b"\x00")
        payload.extend(
            struct.pack(
                "<iB3xii",
                channel.pixel_type,
                int(channel.perceptually_linear),
                channel.x_sampling,
                channel.y_sampling,
            )
        )
    payload.append(0)
    return bytes(payload)


def _raw_size(channels: tuple[exr_container._ExrChannel, ...], *, width: int, row_count: int) -> int:
    return width * row_count * sum(channel.bytes_per_sample for channel in channels)


def _materialized_size(
    codec: str,
    channels: tuple[exr_container._ExrChannel, ...],
    *,
    width: int,
    row_count: int,
) -> int:
    if codec != "pxr24":
        return _raw_size(channels, width=width, row_count=row_count)
    plane_counts = {0: 4, 1: 2, 2: 3}
    return width * row_count * sum(plane_counts[channel.pixel_type] for channel in channels)


def _b44_payload(
    codec: str,
    channels: tuple[exr_container._ExrChannel, ...],
    *,
    width: int,
    row_count: int,
) -> bytes:
    sections = bytearray()
    block_count = ((width + 3) // 4) * ((row_count + 3) // 4)
    for channel in channels:
        if channel.pixel_type != 1:
            sections.extend(bytes(width * row_count * channel.bytes_per_sample))
            continue
        for block_index in range(block_count):
            if codec == "b44a" and block_index % 2 == 0:
                sections.extend(b"\x00\x00\xfc")
            else:
                sections.extend(bytes(14))
    return bytes(sections)


def _compressed_payload(
    codec: str,
    channels: tuple[exr_container._ExrChannel, ...],
    *,
    width: int,
    row_count: int,
) -> bytes:
    if codec == "rle":
        expected = _materialized_size(codec, channels, width=width, row_count=row_count)
        assert 1 <= expected <= 128
        return bytes((expected - 1, 0))
    if codec == "pxr24":
        materialized = bytes(_materialized_size(codec, channels, width=width, row_count=row_count))
        return zlib.compress(materialized)
    return _b44_payload(codec, channels, width=width, row_count=row_count)


@dataclass(frozen=True)
class _ExrChunkFixture:
    payload: bytes
    offset_table: tuple[int, ...]
    payload_offsets: tuple[int, ...]


def _build_exr_chunk_fixture(
    codec: str,
    *,
    channels: tuple[exr_container._ExrChannel, ...] = _MIXED_CHANNELS,
    width: int = 8,
    include_raw_chunk: bool = True,
    compressed_payload: bytes | None = None,
    version_flags: int = 0,
) -> _ExrChunkFixture:
    lines_per_chunk = _LINES_PER_CHUNK[codec]
    row_counts = (lines_per_chunk, 1) if include_raw_chunk else (lines_per_chunk,)
    height = sum(row_counts)
    data_window = (-3, 7, -3 + width - 1, 7 + height - 1)
    attributes = (
        _attribute("channels", "chlist", _channel_list(channels)),
        _attribute("compression", "compression", bytes((_COMPRESSION_CODES[codec],))),
        _attribute("dataWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("displayWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("lineOrder", "lineOrder", bytes((2,))),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    header_terminator = b"\x00\x00" if version_flags & 0x1000 else b"\x00"
    header = struct.pack("<II", 20000630, 2 | version_flags) + b"".join(attributes) + header_terminator
    first_payload = compressed_payload
    if first_payload is None:
        first_payload = _compressed_payload(codec, channels, width=width, row_count=lines_per_chunk)
    logical_payloads = [first_payload]
    if include_raw_chunk:
        logical_payloads.append(bytes(_raw_size(channels, width=width, row_count=1)))
    logical_chunks = tuple(
        (
            7 + sum(row_counts[:index]),
            struct.pack("<ii", 7 + sum(row_counts[:index]), len(payload)) + payload,
        )
        for index, payload in enumerate(logical_payloads)
    )

    cursor = len(header) + 8 * len(logical_chunks)
    offsets_by_y: dict[int, int] = {}
    payload_offsets_by_y: dict[int, int] = {}
    physical = bytearray()
    for y, chunk in reversed(logical_chunks):
        offsets_by_y[y] = cursor
        payload_offsets_by_y[y] = cursor + 8
        physical.extend(chunk)
        cursor += len(chunk)
    table_y = tuple(y for y, _ in reversed(logical_chunks))
    offset_table = tuple(offsets_by_y[y] for y in table_y)
    table = b"".join(struct.pack("<Q", offset) for offset in offset_table)
    return _ExrChunkFixture(
        payload=header + table + bytes(physical),
        offset_table=offset_table,
        payload_offsets=tuple(payload_offsets_by_y[y] for y, _ in logical_chunks),
    )
