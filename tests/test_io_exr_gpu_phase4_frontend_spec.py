"""Specification tests for the Phase 4 PIZ descriptor and routing front end."""

from __future__ import annotations

import inspect
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.packing as exr_packing
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

ROOT = Path(__file__).parents[1]


def _channel(
    name: str,
    pixel_type: int,
    *,
    sampling: tuple[int, int] = (1, 1),
) -> exr_container._ExrChannel:
    dtype, size = {0: ("uint32", 4), 1: ("float16", 2), 2: ("float32", 4)}[pixel_type]
    return exr_container._ExrChannel(
        name=name,
        pixel_type=pixel_type,
        dtype=dtype,
        bytes_per_sample=size,
        perceptually_linear=False,
        x_sampling=sampling[0],
        y_sampling=sampling[1],
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


def _header(
    channels: tuple[exr_container._ExrChannel, ...],
    *,
    width: int,
    height: int,
    origin: tuple[int, int],
    line_order: int,
    compression_code: int = 4,
    version_flags: int = 0,
) -> bytes:
    x_min, y_min = origin
    data_window = (x_min, y_min, x_min + width - 1, y_min + height - 1)
    attributes = (
        _attribute("channels", "chlist", _channel_list(channels)),
        _attribute("compression", "compression", bytes((compression_code,))),
        _attribute("dataWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("displayWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("lineOrder", "lineOrder", bytes((line_order,))),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    terminator = b"\x00\x00" if version_flags & 0x1000 else b"\x00"
    return struct.pack("<II", 20000630, 2 | version_flags) + b"".join(attributes) + terminator


def _huffman_stream(
    *,
    minimum_symbol: int = 0,
    maximum_symbol: int = 1,
    table_byte_count: int = 0,
    data_bit_count: int = 0,
    reserved: int = 0,
) -> bytes:
    return struct.pack("<IIIII", minimum_symbol, maximum_symbol, table_byte_count, data_bit_count, reserved)


def _piz_payload(
    *,
    bitmap_minimum: int = 8191,
    bitmap_maximum: int = 0,
    bitmap_slice: bytes | None = None,
    huffman_stream: bytes | None = None,
    huffman_byte_count: int | None = None,
    trailing: bytes = b"",
) -> bytes:
    bitmap_size = bitmap_maximum - bitmap_minimum + 1 if bitmap_minimum <= bitmap_maximum else 0
    bitmap = bytes([1]) * bitmap_size if bitmap_slice is None else bitmap_slice
    assert len(bitmap) == bitmap_size
    stream = _huffman_stream() if huffman_stream is None else huffman_stream
    declared = len(stream) if huffman_byte_count is None else huffman_byte_count
    return struct.pack("<HH", bitmap_minimum, bitmap_maximum) + bitmap + struct.pack("<I", declared) + stream + trailing


def _raw_size(
    channels: tuple[exr_container._ExrChannel, ...],
    *,
    width: int,
    row_count: int,
) -> int:
    return width * row_count * sum(channel.bytes_per_sample for channel in channels)


@dataclass(frozen=True)
class _PizFixture:
    payload: bytes
    offset_table_start: int
    offset_table: tuple[int, ...]
    payload_offsets: tuple[int, ...]


def _build_piz_exr(
    *,
    channels: tuple[exr_container._ExrChannel, ...] = (_channel("H", 1),),
    width: int = 16,
    height: int = 1,
    origin: tuple[int, int] = (-3, 7),
    line_order: int = 0,
    payloads: tuple[bytes, ...] | None = None,
    table_order: tuple[int, ...] | None = None,
    physical_order: tuple[int, ...] | None = None,
    compression_code: int = 4,
    version_flags: int = 0,
) -> _PizFixture:
    lines_per_chunk = 32
    row_starts = tuple(range(0, height, lines_per_chunk))
    if payloads is None:
        payloads = tuple(_piz_payload() for _ in row_starts)
    assert len(payloads) == len(row_starts)
    order = tuple(range(len(row_starts))) if physical_order is None else physical_order
    table_indices = order if table_order is None else table_order
    header = _header(
        channels,
        width=width,
        height=height,
        origin=origin,
        line_order=line_order,
        compression_code=compression_code,
        version_flags=version_flags,
    )
    cursor = len(header) + len(row_starts) * 8
    offsets: dict[int, int] = {}
    payload_offsets: dict[int, int] = {}
    chunks = bytearray()
    for index in order:
        offsets[index] = cursor
        payload_offsets[index] = cursor + 8
        y = origin[1] + row_starts[index]
        chunk = struct.pack("<ii", y, len(payloads[index])) + payloads[index]
        chunks.extend(chunk)
        cursor += len(chunk)
    offset_table = tuple(offsets[index] for index in table_indices)
    table = b"".join(struct.pack("<Q", offset) for offset in offset_table)
    return _PizFixture(
        payload=header + table + bytes(chunks),
        offset_table_start=len(header),
        offset_table=offset_table,
        payload_offsets=tuple(payload_offsets[index] for index in range(len(row_starts))),
    )


def _write_valid_openexr_fallback(path: Path, structure: str) -> tuple[int, str, np.ndarray]:
    from openexr_dev_oracle import OpenEXR

    pixels = np.arange(12, dtype=np.float16).reshape(3, 4)
    if structure == "tiled":
        tiles = OpenEXR.TileDescription()
        tiles.xSize = 2
        tiles.ySize = 2
        OpenEXR.File(
            {"type": OpenEXR.tiledimage, "tiles": tiles, "compression": OpenEXR.PIZ_COMPRESSION},
            {"H": pixels},
        ).write(str(path))
        return 0, "H", pixels
    if structure == "multipart":
        OpenEXR.File(
            [
                OpenEXR.Part({"compression": OpenEXR.PIZ_COMPRESSION}, {"H": pixels}, "beauty"),
                OpenEXR.Part({"compression": OpenEXR.PIZ_COMPRESSION}, {"J": pixels + 1}, "utility"),
            ]
        ).write(str(path))
        return 1, "J", pixels + 1
    if structure == "deep":
        deep_pixels = np.empty((2, 3), dtype=object)
        for row in range(2):
            for column in range(3):
                deep_pixels[row, column] = np.array([row * 3 + column + 0.25], dtype=np.float32)
        OpenEXR.File(
            {
                "name": "deep",
                "type": OpenEXR.deepscanline,
                "compression": OpenEXR.ZIPS_COMPRESSION,
                "maxSamplesPerPixel": 1,
            },
            {"Z": deep_pixels},
        ).write(str(path))
        return 0, "Z", deep_pixels
    raise AssertionError(f"unknown fallback structure: {structure}")


def _assert_openexr_pixels_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if expected.dtype != np.dtype(object):
        np.testing.assert_array_equal(actual, expected)
        return
    for actual_samples, expected_samples in zip(actual.flat, expected.flat, strict=True):
        np.testing.assert_array_equal(actual_samples, expected_samples)


@pytest.mark.parametrize(
    ("height", "row_counts"),
    ((7, (7,)), (32, (32,)), (33, (32, 1)), (63, (32, 31))),
)
def test_piz_descriptor_covers_full_partial_and_subchunk_images(
    tmp_path: Path,
    height: int,
    row_counts: tuple[int, ...],
) -> None:
    """v1-exr-gpu-phase4 acceptance 2, 6, and 11: every 32-line geometry owns one exact output row range."""
    fixture = _build_piz_exr(height=height, physical_order=tuple(reversed(range(len(row_counts)))))
    path = tmp_path / f"piz-{height}.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert container.piz_eligible is True
    assert container.offset_table == fixture.offset_table
    assert tuple((chunk.row_start, chunk.row_count) for chunk in container.chunks) == tuple(
        (sum(row_counts[:index]), row_count) for index, row_count in enumerate(row_counts)
    )
    assert tuple(chunk.piz.output_row_span for chunk in container.chunks if chunk.piz is not None) == tuple(
        (sum(row_counts[:index]), sum(row_counts[: index + 1])) for index in range(len(row_counts))
    )


def test_piz_descriptor_classifies_compressed_equal_zero_and_oversized_storage(tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 6, 7, and 35: size alone fixes raw, compressed, empty, and corrupt paths."""
    channels = (_channel("H", 1),)
    expected_size = _raw_size(channels, width=16, row_count=1)
    full_chunk_size = _raw_size(channels, width=16, row_count=32)
    assert len(_piz_payload()) < expected_size
    payloads = (_piz_payload(), bytes(full_chunk_size), b"")
    fixture = _build_piz_exr(channels=channels, width=16, height=65, payloads=payloads)
    path = tmp_path / "piz-storage.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert tuple((chunk.packed_size, chunk.raw_stored) for chunk in container.chunks) == (
        (len(payloads[0]), False),
        (full_chunk_size, True),
        (0, True),
    )
    compressed, equal_raw, empty_raw = (chunk.piz for chunk in container.chunks)
    assert compressed is not None and compressed.huffman_span.size == len(_huffman_stream())
    assert equal_raw is not None and equal_raw.huffman_span.size == 0
    assert empty_raw is not None and empty_raw.huffman_span.size == 0

    oversized = _build_piz_exr(channels=channels, width=16, payloads=(bytes(expected_size + 1),))
    oversized_path = tmp_path / "piz-oversized.exr"
    oversized_path.write_bytes(oversized.payload)
    with pytest.raises(RuntimeError, match=r"why=.*larger.*what=.*how="):
        px.io.read_header(oversized_path)


def test_piz_payload_selection_uses_raw_for_equal_or_larger_codec_output() -> None:
    """v1-exr-gpu-phase4 acceptance 7: a PIZ writer never stores an equal-size compressed payload."""
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


@pytest.mark.parametrize(
    ("bitmap_minimum", "bitmap_maximum", "bitmap_size"),
    ((3, 5, 3), (4, 4, 1), (8191, 0, 0), (7, 2, 0)),
)
def test_piz_bitmap_and_huffman_spans_are_permissive_but_bounded(
    tmp_path: Path,
    bitmap_minimum: int,
    bitmap_maximum: int,
    bitmap_size: int,
) -> None:
    """v1-exr-gpu-phase4 acceptance 6, 24, 35, and 36: ranges are inclusive, empty ranges and trailing bytes are safe."""
    payload = _piz_payload(
        bitmap_minimum=bitmap_minimum,
        bitmap_maximum=bitmap_maximum,
        trailing=b"permissive-trailing",
    )
    fixture = _build_piz_exr(width=32, payloads=(payload,))
    path = tmp_path / f"bitmap-{bitmap_minimum}-{bitmap_maximum}.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)
    descriptor = container.chunks[0].piz

    assert descriptor is not None
    assert descriptor.bitmap_range == (bitmap_minimum, bitmap_maximum)
    assert descriptor.bitmap_span.size == bitmap_size
    assert descriptor.huffman_span.size == len(_huffman_stream())
    assert descriptor.trailing_span.size == len(b"permissive-trailing")


def test_piz_descriptor_records_absolute_payload_and_huffman_boundaries(tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 6, 24, and 36: every descriptor span and leader field maps to its wire byte."""
    bitmap = b"\x11\x22\x44"
    leader = _huffman_stream(
        minimum_symbol=17,
        maximum_symbol=251,
        table_byte_count=3,
        data_bit_count=13,
        reserved=0x01020304,
    )
    huffman_data = b"\xa5\x5a\xc3"
    trailing = b"\xde\xad\xbe\xef"
    payload = _piz_payload(
        bitmap_minimum=7,
        bitmap_maximum=9,
        bitmap_slice=bitmap,
        huffman_stream=leader + huffman_data,
        trailing=trailing,
    )
    fixture = _build_piz_exr(width=32, payloads=(payload,))
    path = tmp_path / "absolute-piz-spans.exr"
    path.write_bytes(fixture.payload)

    descriptor = exr_container._parse_exr_container(path).chunks[0].piz

    assert descriptor is not None
    payload_start = fixture.payload_offsets[0]
    payload_end = payload_start + len(payload)
    bitmap_start = payload_start + 4
    bitmap_end = bitmap_start + len(bitmap)
    count_start = bitmap_end
    count_end = count_start + 4
    leader_start = count_end
    leader_end = leader_start + len(leader)
    data_end = leader_end + len(huffman_data)
    assert descriptor.payload_span == exr_container._PizByteSpan(payload_start, payload_end)
    assert descriptor.bitmap_span == exr_container._PizByteSpan(bitmap_start, bitmap_end)
    assert descriptor.huffman_count_span == exr_container._PizByteSpan(count_start, count_end)
    assert descriptor.huffman_span == exr_container._PizByteSpan(leader_start, data_end)
    assert descriptor.trailing_span == exr_container._PizByteSpan(data_end, payload_end)
    assert descriptor.huffman_leader is not None
    assert descriptor.huffman_leader.span == exr_container._PizByteSpan(leader_start, leader_end)
    assert (
        descriptor.huffman_leader.minimum_symbol,
        descriptor.huffman_leader.maximum_symbol,
        descriptor.huffman_leader.table_byte_count,
        descriptor.huffman_leader.data_bit_count,
        descriptor.huffman_leader.reserved,
    ) == (17, 251, 3, 13, 0x01020304)
    assert fixture.payload[bitmap_start:bitmap_end] == bitmap
    assert fixture.payload[count_start:count_end] == struct.pack("<I", len(leader) + len(huffman_data))
    assert fixture.payload[leader_start:leader_end] == leader
    assert fixture.payload[leader_end:data_end] == huffman_data
    assert fixture.payload[data_end:payload_end] == trailing


@pytest.mark.parametrize(
    ("payload", "word"),
    (
        (_piz_payload(bitmap_minimum=8192, bitmap_maximum=8192), "bitmap"),
        (_piz_payload(huffman_stream=b"short"), "Huffman"),
        (_piz_payload(huffman_byte_count=len(_huffman_stream()) + 1), "Huffman"),
    ),
)
def test_piz_descriptor_rejects_unsafe_bitmap_and_huffman_bounds(
    tmp_path: Path,
    payload: bytes,
    word: str,
) -> None:
    """v1-exr-gpu-phase4 acceptance 6 and 35: descriptor-time access bounds fail before any codec lane."""
    fixture = _build_piz_exr(width=32, payloads=(payload,))
    path = tmp_path / f"unsafe-{word}.exr"
    path.write_bytes(fixture.payload)

    with pytest.raises(RuntimeError, match=rf"why=.*{word}.*what=.*how="):
        px.io.read_header(path)


def test_piz_channel_planes_preserve_file_order_dotted_names_and_word_slices(tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 6 and 8: HALF and low/high 32-bit word planes have independent offsets."""
    channels = (_channel("A.half", 1), _channel("beauty.F", 2), _channel("mask.U", 0))
    fixture = _build_piz_exr(channels=channels, width=3, height=2)
    path = tmp_path / "piz-planes.exr"
    path.write_bytes(fixture.payload)

    descriptor = exr_container._parse_exr_container(path).chunks[0].piz

    assert descriptor is not None
    assert descriptor.expected_packed_size == 60
    assert descriptor.expected_output_word_count == 30
    assert tuple(
        (plane.channel_name, plane.pixel_type, plane.word_slice_count, plane.word_offset, plane.word_count)
        for plane in descriptor.channel_planes
    ) == (
        ("A.half", 1, 1, 0, 6),
        ("beauty.F", 2, 2, 6, 12),
        ("mask.U", 0, 2, 18, 12),
    )


@pytest.mark.parametrize(
    ("line_order", "table_order", "physical_order"),
    (
        (0, (0, 1, 2), (0, 1, 2)),
        (1, (2, 1, 0), (2, 1, 0)),
        (2, (1, 2, 0), (2, 0, 1)),
    ),
)
def test_piz_offset_table_order_is_independent_of_output_rows(
    tmp_path: Path,
    line_order: int,
    table_order: tuple[int, ...],
    physical_order: tuple[int, ...],
) -> None:
    """v1-exr-gpu-phase4 acceptance 2, 6, and 11: all line orders resolve output ownership from chunk y."""
    fixture = _build_piz_exr(
        width=16,
        height=65,
        line_order=line_order,
        table_order=table_order,
        physical_order=physical_order,
    )
    path = tmp_path / f"line-order-{line_order}.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert container.line_order == line_order
    assert container.offset_table == fixture.offset_table
    assert tuple((chunk.row_start, chunk.row_count) for chunk in container.chunks) == ((0, 32), (32, 32), (64, 1))


def test_piz_offset_spans_and_row_ranges_must_not_intersect_or_repeat(tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 6 and 35: chunk byte spans and output-row ownership are one-to-one."""
    channels = (_channel("H", 1),)
    header = _header(channels, width=16, height=33, origin=(0, 0), line_order=2)
    base = _piz_payload()
    first_offset = len(header) + 16
    embedded_offset = first_offset + 8 + len(base)
    embedded = struct.pack("<ii", 32, len(base)) + base
    first_payload = base + embedded
    crossing = (
        header
        + struct.pack("<QQ", first_offset, embedded_offset)
        + struct.pack("<ii", 0, len(first_payload))
        + first_payload
    )
    crossing_path = tmp_path / "crossing.exr"
    crossing_path.write_bytes(crossing)

    with pytest.raises(RuntimeError, match=r"why=.*intersect.*what=.*how="):
        px.io.read_header(crossing_path)

    repeated = _build_piz_exr(width=16, height=33)
    repeated_bytes = bytearray(repeated.payload)
    second_chunk = repeated.offset_table[1]
    struct.pack_into("<i", repeated_bytes, second_chunk, 7)
    repeated_path = tmp_path / "repeated-row.exr"
    repeated_path.write_bytes(repeated_bytes)
    with pytest.raises(RuntimeError, match=r"why=.*repeat.*row.*what=.*how="):
        px.io.read_header(repeated_path)


def test_piz_offset_table_rejects_an_exact_duplicate_chunk_span(tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 6 and 35: identical offsets cannot make one chunk own two row blocks."""
    fixture = _build_piz_exr(width=16, height=33)
    duplicate = bytearray(fixture.payload)
    first_offset = fixture.offset_table[0]
    second_table_entry = fixture.offset_table_start + 8
    struct.pack_into("<Q", duplicate, second_table_entry, first_offset)
    path = tmp_path / "duplicate-offset-and-span.exr"
    path.write_bytes(duplicate)

    assert struct.unpack_from("<QQ", duplicate, fixture.offset_table_start) == (first_offset, first_offset)
    with pytest.raises(RuntimeError, match=r"why=.*duplicate.*offset.*what=.*how="):
        px.io.read_header(path)


@pytest.mark.parametrize("structure", ("tiled", "multipart"))
def test_tiled_and_multipart_containers_use_internal_read_paths(
    tmp_path: Path,
    structure: str,
) -> None:
    """v1-exr-runtime-independence acceptance 16, 19, 22, and 23: structured flat parts decode internally."""
    path = tmp_path / f"valid-{structure}-internal.exr"
    part_index, channel, expected = _write_valid_openexr_fallback(path, structure)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    locations = [(part_index, channel, channel)]

    assert (container.tiled, container.multipart, container.deep) == {
        "tiled": (True, False, False),
        "multipart": (False, True, False),
    }[structure]
    assert container.piz_eligible is False
    assert container.chunks == ()
    actual = io._read_exr_pixels(path, container, header, locations, unchanged=True)
    _assert_openexr_pixels_equal(actual.get()[..., 0], expected)


def test_explicit_deep_channel_reports_an_actionable_unsupported_error(
    tmp_path: Path,
) -> None:
    """v1-exr-runtime-independence acceptance 26: explicit deep selection fails with actionable context."""
    path = tmp_path / "valid-deep-unsupported.exr"
    part_index, channel, _expected = _write_valid_openexr_fallback(path, "deep")
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)

    assert (container.tiled, container.multipart, container.deep) == (False, False, True)
    with pytest.raises(ValueError, match=rf"why=.*deep.*what=.*{channel}.*how="):
        io._read_exr_pixels(path, container, header, [(part_index, channel, channel)], unchanged=True)


@pytest.mark.parametrize("sampling", ((2, 1), (1, 2)))
def test_synthetic_sampled_piz_corruption_never_falls_back(
    tmp_path: Path,
    sampling: tuple[int, int],
) -> None:
    """v1-exr-runtime-independence acceptance 4, 27, and 43: sampled PIZ corruption fails internally."""
    channels = (_channel("H", 1, sampling=sampling),)
    fixture = _build_piz_exr(channels=channels)
    path = tmp_path / f"sampled-piz-corrupt-{sampling}.exr"
    path.write_bytes(fixture.payload)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)

    assert container.piz_eligible is False
    assert container.chunks == ()
    with pytest.raises(RuntimeError, match=r"why=.*Huffman.*what=.*how="):
        io._read_exr_pixels(path, container, header, [(0, "H", "H")], unchanged=True)


def test_unknown_compression_header_fails_without_an_external_fallback(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 38 and 43: unknown compression fails without an external fallback."""
    fixture = _build_piz_exr(compression_code=10)
    path = tmp_path / "unknown-compression-fallback.exr"
    path.write_bytes(fixture.payload)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    with pytest.raises(RuntimeError, match=r"why=.*unknown.*what=.*compression.*how="):
        io._read_exr_pixels(path, container, header, [(0, "H", "H")], unchanged=True)


def test_piz_uint_and_mixed_public_output_dtype_resolution(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 10-12: UINT and mixed selections resolve before routing."""
    channels = (_channel("F", 2), _channel("H", 1), _channel("U", 0))
    fixture = _build_piz_exr(channels=channels, width=4, height=1)
    path = tmp_path / "piz-mixed-types.exr"
    path.write_bytes(fixture.payload)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)

    assert container.piz_eligible is True
    assert io._exr_output_dtype(header, [(0, "U", "U")], unchanged=False) == "float32"
    assert io._exr_output_dtype(header, [(0, "U", "U")], unchanged=True) == "uint32"
    assert io._exr_output_dtype(header, [(0, "F", "F"), (0, "U", "U")], unchanged=False) == "float32"
    with pytest.raises(ValueError, match=r"why=.*mixed.*what=.*how=.*unchanged=False"):
        io._exr_output_dtype(header, [(0, "F", "F"), (0, "U", "U")], unchanged=True)


def test_piz_public_and_forced_internal_routes_exclude_external_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-runtime-independence acceptance 36, 38, and 48: PIZ routes use only self-owned lanes."""
    fixture = _build_piz_exr(width=16)
    path = tmp_path / "piz-routes.exr"
    path.write_bytes(fixture.payload)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    locations = [(0, "H", "H")]
    calls: list[str] = []

    def read_gpu(*args: object, **kwargs: object) -> object:
        calls.append("read-gpu")
        return cp.zeros((1, 16, 1), dtype=cp.float16)

    def read_custom_cpu(*args: object, **kwargs: object) -> object:
        calls.append("read-custom_cpu")
        return cp.zeros((1, 16, 1), dtype=cp.float16)

    monkeypatch.setattr(io, "_read_exr_gpu", read_gpu)
    monkeypatch.setattr(io, "_read_exr_custom_cpu", read_custom_cpu)

    assert io._EXR_ROUTING[("piz", "read")] == "gpu"
    assert io._EXR_ROUTING[("piz", "write")] == "gpu"
    io._read_exr_pixels(path, container, header, locations, unchanged=True)
    for backend in ("gpu", "custom_cpu"):
        exr_harness._read_exr_pixels_with_backend(path, container, header, locations, unchanged=True, backend=backend)

    frame = SimpleNamespace(data=object(), channels=("R",), colorspace="sRGB")
    monkeypatch.setattr(io, "_write_exr_gpu", lambda *_args, **_kwargs: calls.append("write-gpu"))
    io._write_exr(path, frame, compression="piz", dwa_level=None)
    io._write_exr_with_backend(path, frame, compression="piz", dwa_level=None, backend="gpu")

    assert calls == ["read-gpu", "read-gpu", "read-custom_cpu", "write-gpu", "write-gpu"]


def test_piz_frontend_adds_no_public_selector_and_header_probe_remains_codec_lazy(tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 1 and 39: the public surface and pure header boundary stay unchanged."""
    fixture = _build_piz_exr(width=16)
    path = tmp_path / "piz-header.exr"
    path.write_bytes(fixture.payload)
    for function in (px.io.read_image, px.io.write_image, px.io.read_header):
        assert "backend" not in inspect.signature(function).parameters

    script = """
import importlib.abc
import sys
class BlockCodecs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "OpenEXR" or fullname.startswith("nvidia.nvcomp"):
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, BlockCodecs())
import pixtreme as px
header = px.io.read_header(sys.argv[1])
assert (header.format, header.width, header.height) == ("EXR", 16, 1)
assert "OpenEXR" not in sys.modules
assert "nvidia.nvcomp" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
