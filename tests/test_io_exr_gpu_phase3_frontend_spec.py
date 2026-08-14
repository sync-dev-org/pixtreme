"""Specification tests for the Phase 3 OpenEXR codec descriptor front end."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.packing as exr_packing
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

_COMPRESSION_CODES = {"rle": 1, "pxr24": 5, "b44": 6, "b44a": 7}
_LINES_PER_CHUNK = {"rle": 1, "pxr24": 16, "b44": 32, "b44a": 32}


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
class _Phase3Fixture:
    payload: bytes
    offset_table: tuple[int, ...]
    payload_offsets: tuple[int, ...]


def _build_phase3_exr(
    codec: str,
    *,
    channels: tuple[exr_container._ExrChannel, ...] = _MIXED_CHANNELS,
    width: int = 8,
    include_raw_chunk: bool = True,
    compressed_payload: bytes | None = None,
    version_flags: int = 0,
) -> _Phase3Fixture:
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
    return _Phase3Fixture(
        payload=header + table + bytes(physical),
        offset_table=offset_table,
        payload_offsets=tuple(payload_offsets_by_y[y] for y, _ in logical_chunks),
    )


@pytest.mark.parametrize("codec", ("rle", "pxr24", "b44", "b44a"))
def test_phase3_descriptor_covers_mixed_types_raw_storage_partial_rows_and_file_order(
    tmp_path: Path,
    codec: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 2, 6, 7, and 10: the common descriptor fixes every front-end boundary."""
    fixture = _build_phase3_exr(codec)
    path = tmp_path / f"{codec}-descriptor.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    lines_per_chunk = _LINES_PER_CHUNK[codec]
    assert container.compression == codec
    assert container.phase3_eligible is True
    assert container.gpu_eligible is False
    assert container.dwa_eligible is False
    assert container.lines_per_chunk == lines_per_chunk
    assert container.offset_table == fixture.offset_table
    assert tuple((chunk.row_start, chunk.row_count, chunk.raw_stored) for chunk in container.chunks) == (
        (0, lines_per_chunk, False),
        (lines_per_chunk, 1, True),
    )

    compressed, raw = container.chunks
    assert compressed.phase3 is not None
    assert compressed.phase3.codec == codec
    assert compressed.phase3.payload_span == exr_container._Phase3ByteSpan(
        fixture.payload_offsets[0], fixture.payload_offsets[0] + compressed.packed_size
    )
    expected_raw = _raw_size(_MIXED_CHANNELS, width=8, row_count=lines_per_chunk)
    expected_materialized = _materialized_size(codec, _MIXED_CHANNELS, width=8, row_count=lines_per_chunk)
    assert compressed.phase3.expected_raw_size == expected_raw
    assert compressed.phase3.expected_materialized_size == expected_materialized
    assert len(compressed.phase3.channel_rows) == lines_per_chunk * len(_MIXED_CHANNELS)
    assert compressed.phase3.channel_rows[0].channel_name == "U"
    assert compressed.phase3.channel_rows[0].output_row == 0
    assert compressed.phase3.channel_rows[-1].channel_name == "F"
    assert compressed.phase3.channel_rows[-1].output_row == lines_per_chunk - 1

    assert raw.phase3 is not None
    assert raw.phase3.raw_stored is True
    assert raw.phase3.expected_raw_size == _raw_size(_MIXED_CHANNELS, width=8, row_count=1)
    assert raw.phase3.packets == ()
    assert raw.phase3.planes == ()
    assert raw.phase3.channel_sections == ()
    assert raw.phase3.blocks == ()
    assert tuple(row.output_row for row in raw.phase3.channel_rows) == (lines_per_chunk,) * len(_MIXED_CHANNELS)

    if codec == "rle":
        assert len(compressed.phase3.packets) == 1
        assert compressed.phase3.packets[0].output_span == exr_container._Phase3ByteSpan(0, expected_raw)
    elif codec == "pxr24":
        assert expected_materialized < expected_raw
        assert len(compressed.phase3.planes) == lines_per_chunk * (4 + 2 + 3)
        assert {plane.channel_name for plane in compressed.phase3.planes} == {"U", "H", "F"}
    else:
        assert tuple(section.channel_name for section in compressed.phase3.channel_sections) == ("U", "H", "F")
        half_section = compressed.phase3.channel_sections[1]
        assert half_section.perceptually_linear is True
        assert half_section.block_count == 2 * ((lines_per_chunk + 3) // 4)
        assert len(compressed.phase3.blocks) == half_section.block_count
        if codec == "b44":
            assert {block.stored_size for block in compressed.phase3.blocks} == {14}
        else:
            assert {block.stored_size for block in compressed.phase3.blocks} == {3, 14}


def test_phase3_subsampled_structure_is_classified_for_openexr_fallback(tmp_path: Path) -> None:
    """v1-exr-gpu-phase3 acceptance 3: a valid subsampled input remains a fallback case."""
    channels = (_channel("H", 1, sampling=(2, 1)),)
    fixture = _build_phase3_exr("rle", channels=channels)
    path = tmp_path / "fallback-subsampled.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert container.phase3_eligible is False
    assert container.chunks == ()
    assert px.io.read_header(path).format == "EXR"


def test_phase3_pxr24_uses_the_final_custom_cpu_public_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-runtime-independence acceptance 36 and 48: PXR24 public read uses the custom CPU lane."""
    fixture = _build_phase3_exr("pxr24")
    path = tmp_path / "pxr24-openexr-route.exr"
    path.write_bytes(fixture.payload)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    original = io._read_exr_custom_cpu
    calls = 0

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(io, "_read_exr_custom_cpu", spy)
    actual = io._read_exr_pixels(path, container, header, [(0, "H", "H")], unchanged=False)

    assert calls == 1
    assert actual.shape == (17, 8, 1)


@pytest.mark.parametrize("codec", ("rle", "pxr24", "b44", "b44a"))
def test_phase3_compressed_payload_larger_than_raw_is_actionable_corruption(tmp_path: Path, codec: str) -> None:
    """v1-exr-gpu-phase3 acceptance 7: only equality selects raw and an oversized compressed chunk is corrupt."""
    expected_raw = _raw_size(_MIXED_CHANNELS, width=8, row_count=_LINES_PER_CHUNK[codec])
    fixture = _build_phase3_exr(codec, include_raw_chunk=False, compressed_payload=bytes(expected_raw + 1))
    path = tmp_path / f"{codec}-oversized.exr"
    path.write_bytes(fixture.payload)

    with pytest.raises(RuntimeError, match=r"why=.*larger.*what=.*how="):
        px.io.read_header(path)


@pytest.mark.parametrize("codec", ("rle", "pxr24", "b44", "b44a"))
def test_phase3_payload_selection_uses_raw_for_equal_or_larger_codec_output(codec: str) -> None:
    """v1-exr-gpu-phase3 acceptance 7: writer payload selection never stores an equal-size compressed stream."""
    assert codec in _COMPRESSION_CODES
    raw_chunks = (b"raw!", b"same", b"last")
    encoded_chunks = (b"zip", b"code", b"larger")
    raw_sizes = tuple(map(len, raw_chunks))
    encoded_sizes = tuple(map(len, encoded_chunks))
    raw_offsets = tuple(int(value) for value in np.cumsum((0, *raw_sizes[:-1]), dtype=np.int64))
    encoded_offsets = tuple(int(value) for value in np.cumsum((0, *encoded_sizes[:-1]), dtype=np.int64))

    selected, selected_sizes = exr_packing._select_exr_payloads(
        cp.asarray(np.frombuffer(b"".join(raw_chunks), dtype=np.uint8)),
        raw_offsets,
        raw_sizes,
        cp.asarray(np.frombuffer(b"".join(encoded_chunks), dtype=np.uint8)),
        encoded_offsets,
        encoded_sizes,
    )

    assert selected_sizes == (3, 4, 4)
    assert selected.get().tobytes() == b"zip" + raw_chunks[1] + raw_chunks[2]


@pytest.mark.parametrize("codec", ("rle", "pxr24", "b44", "b44a"))
def test_phase3_codec_grammar_errors_are_actionable_before_materialize(tmp_path: Path, codec: str) -> None:
    """v1-exr-gpu-phase3 acceptance 6: malformed packet, plane, and block heads fail in the descriptor front end."""
    channels = (_channel("H", 1),)
    lines_per_chunk = _LINES_PER_CHUNK[codec]
    if codec == "rle":
        malformed = b"\x80"
    elif codec == "pxr24":
        malformed = zlib.compress(b"\x00")
    elif codec == "b44":
        valid = bytearray(_b44_payload(codec, channels, width=8, row_count=lines_per_chunk))
        valid[2] = 0x34
        malformed = bytes(valid)
    else:
        block_count = 2 * ((lines_per_chunk + 3) // 4)
        malformed = b"\x00\x00\xfc" * (block_count - 1) + b"\x00\x00"
    fixture = _build_phase3_exr(
        codec,
        channels=channels,
        include_raw_chunk=False,
        compressed_payload=malformed,
    )
    path = tmp_path / f"{codec}-malformed.exr"
    path.write_bytes(fixture.payload)

    with pytest.raises(RuntimeError, match=rf"why=.*{codec.upper()}.*what=.*how="):
        px.io.read_header(path)
