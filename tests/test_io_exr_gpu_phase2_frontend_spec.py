"""Specification tests for the Phase 2 OpenEXR DWA control-plane front end."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import pytest

import pixtreme as px
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io


def _pack_bits(fields: list[tuple[int, int]]) -> bytes:
    value = 0
    bit_count = 0
    for field, width in fields:
        assert 0 <= field < 1 << width
        value = (value << width) | field
        bit_count += width
    padding = (-bit_count) % 8
    value <<= padding
    return value.to_bytes((bit_count + padding) // 8, "big")


def _huffman_stream(
    *,
    minimum: int = 0,
    maximum: int = 1,
    table_fields: list[tuple[int, int]] | None = None,
    data_bit_count: int = 1,
    reserved: int = 0,
) -> bytes:
    table = _pack_bits(table_fields or [(1, 6), (1, 6)])
    data = bytes((data_bit_count + 7) // 8)
    return struct.pack("<IIIII", minimum, maximum, len(table), data_bit_count, reserved) + table + data


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


def test_default_dwa_channel_rules_are_case_sensitive_and_group_only_matching_rgb_triplets() -> None:
    """v1-exr-gpu-phase2 acceptance 7: suffix, type, prefix, and sampling determine DWA channel ownership."""
    channels = (
        _channel("beauty.R", 2),
        _channel("beauty.G", 1),
        _channel("beauty.B", 2),
        _channel("matte.R", 1),
        _channel("matte.G", 1, sampling=(2, 1)),
        _channel("matte.B", 1),
        _channel("beauty.A", 0),
        _channel("Y", 1),
        _channel("BY", 2),
        _channel("RY", 1),
        _channel("lower.r", 2),
        _channel("uint.R", 0),
        _channel("depth", 2),
    )

    layout = exr_container._classify_default_dwa_channels(channels)

    by_name = {channel.name: channel for channel in layout.channels}
    assert {name: by_name[name].scheme for name in ("beauty.R", "beauty.G", "beauty.B", "Y", "BY", "RY")} == {
        "beauty.R": "lossy_dct",
        "beauty.G": "lossy_dct",
        "beauty.B": "lossy_dct",
        "Y": "lossy_dct",
        "BY": "lossy_dct",
        "RY": "lossy_dct",
    }
    assert by_name["beauty.A"].scheme == "rle"
    assert {name: by_name[name].scheme for name in ("lower.r", "uint.R", "depth")} == {
        "lower.r": "unknown",
        "uint.R": "unknown",
        "depth": "unknown",
    }
    assert tuple(group.channel_names for group in layout.csc_groups) == (("beauty.R", "beauty.G", "beauty.B"),)
    assert by_name["beauty.R"].csc_group == 0
    assert all(by_name[name].csc_group is None for name in ("matte.R", "matte.G", "matte.B"))


def test_huffman_parser_expands_every_table_run_and_accepts_a_58_bit_code() -> None:
    """v1-exr-gpu-phase2 acceptance 8: the canonical table accepts 59-63 runs and the 58-bit boundary."""
    fields = [
        (59, 6),
        (60, 6),
        (61, 6),
        (62, 6),
        (63, 6),
        (0, 8),
        (58, 6),
    ]
    stream = _huffman_stream(
        minimum=0,
        maximum=20,
        table_fields=fields,
        data_bit_count=58,
    )

    table = exr_container._parse_dwa_huffman_table(stream)

    assert table.minimum_symbol == 0
    assert table.maximum_symbol == 20
    assert table.code_lengths == (0,) * 20 + (58,)
    assert table.codes == (exr_container._DwaHuffmanCode(symbol=20, length=58, code=0),)
    assert table.table_span.size == len(_pack_bits(fields))
    assert table.data_span.size == 8


@pytest.mark.parametrize(
    ("stream", "message"),
    (
        (_huffman_stream(maximum=2, table_fields=[(1, 6), (1, 6), (1, 6)]), "oversubscribed"),
        (_huffman_stream(table_fields=[(1, 6), (2, 6)]), "assignment"),
        (_huffman_stream(reserved=1), "reserved"),
        (_huffman_stream()[:-2], "truncated"),
        (_huffman_stream()[:-1] + b"\x01", "data.*padding"),
    ),
)
def test_huffman_parser_rejects_invalid_or_truncated_tables(stream: bytes, message: str) -> None:
    """v1-exr-gpu-phase2 acceptance 8: malformed Huffman metadata fails before symbol decode."""
    with pytest.raises(ValueError, match=rf"why=.*{message}.*what=.*how="):
        exr_container._parse_dwa_huffman_table(stream)


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


def _rule(suffix: str, *, scheme: int, pixel_type: int, csc_index: int = -1) -> bytes:
    packed = ((csc_index + 1) << 4) | (scheme << 2)
    return suffix.encode() + b"\x00" + bytes((packed, pixel_type))


def _dwa_payload(
    *,
    version: int = 2,
    ac_scheme: int = 0,
    block_rows: int = 4,
    additional_rules: tuple[bytes, ...] = (),
) -> bytes:
    huffman = _huffman_stream()
    rules = b"".join(
        (
            _rule("R", scheme=1, pixel_type=2, csc_index=0),
            _rule("G", scheme=1, pixel_type=2, csc_index=1),
            _rule("B", scheme=1, pixel_type=2, csc_index=2),
        )
        + additional_rules
    )
    rule_section = struct.pack("<H", len(rules) + 2) + rules if version >= 2 else b""
    block_count = 3 * 2 * block_rows
    dc = b"\x00"
    leader = struct.pack(
        "<11Q",
        version,
        0,
        0,
        len(huffman),
        len(dc),
        0,
        0,
        0,
        block_count,
        block_count,
        ac_scheme,
    )
    return leader + rule_section + huffman + dc


@dataclass(frozen=True)
class _DwaFixture:
    payload: bytes
    chunk_offsets: tuple[int, ...]
    payload_offsets: tuple[int, ...]


def _build_dwa_exr(
    *,
    version: int = 2,
    compression: str = "dwaa",
    reverse_physical_chunks: bool = True,
    additional_rules: tuple[bytes, ...] = (),
) -> _DwaFixture:
    channels = (_channel("B", 2), _channel("G", 2), _channel("R", 2))
    width = 16
    lines_per_chunk = {"dwaa": 32, "dwab": 256}[compression]
    compression_code = {"dwaa": 8, "dwab": 9}[compression]
    height = lines_per_chunk + 1
    data_window = (-3, 7, -3 + width - 1, 7 + height - 1)
    attributes = (
        _attribute("channels", "chlist", _channel_list(channels)),
        _attribute("compression", "compression", bytes((compression_code,))),
        _attribute("dataWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("displayWindow", "box2i", struct.pack("<iiii", *data_window)),
        _attribute("lineOrder", "lineOrder", bytes((2,))),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    header = struct.pack("<II", 20000630, 2) + b"".join(attributes) + b"\x00"
    compressed = _dwa_payload(
        version=version,
        block_rows=lines_per_chunk // 8,
        additional_rules=additional_rules,
    )
    raw = bytes(width * 3 * 4)
    logical_chunks = (
        (7, struct.pack("<ii", 7, len(compressed)) + compressed),
        (7 + lines_per_chunk, struct.pack("<ii", 7 + lines_per_chunk, len(raw)) + raw),
    )
    physical_chunks = tuple(reversed(logical_chunks)) if reverse_physical_chunks else logical_chunks
    cursor = len(header) + 16
    offsets_by_y: dict[int, int] = {}
    payload_offsets_by_y: dict[int, int] = {}
    physical = bytearray()
    for y, chunk in physical_chunks:
        offsets_by_y[y] = cursor
        payload_offsets_by_y[y] = cursor + 8
        physical.extend(chunk)
        cursor += len(chunk)
    offset_table = b"".join(struct.pack("<Q", offsets_by_y[y]) for y, _ in logical_chunks)
    return _DwaFixture(
        payload=header + offset_table + bytes(physical),
        chunk_offsets=tuple(offsets_by_y[y] for y, _ in logical_chunks),
        payload_offsets=tuple(payload_offsets_by_y[y] for y, _ in logical_chunks),
    )


def test_dwa_v2_container_descriptor_covers_streams_partial_geometry_and_raw_storage(tmp_path: Path) -> None:
    """v1-exr-gpu-phase2 acceptance 6 and 10: one descriptor covers DWA streams, partial rows, and raw chunks."""
    fixture = _build_dwa_exr()
    path = tmp_path / "dwaa-v2.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert container.compression == "dwaa"
    assert container.lines_per_chunk == 32
    assert container.dwa_eligible is True
    assert container.gpu_eligible is False
    assert container.offset_table == fixture.chunk_offsets
    assert tuple((chunk.y, chunk.row_start, chunk.row_count, chunk.raw_stored) for chunk in container.chunks) == (
        (7, 0, 32, False),
        (39, 32, 1, True),
    )
    compressed, raw = container.chunks
    assert compressed.dwa is not None
    assert compressed.dwa.leader is not None
    assert compressed.dwa.leader.version == 2
    assert compressed.dwa.leader.ac_element_count == 24
    assert compressed.dwa.leader.dc_element_count == 24
    assert compressed.dwa.ac_span.size == compressed.dwa.leader.ac_compressed_size
    assert compressed.dwa.huffman is not None
    assert compressed.dwa.huffman.code_lengths == ()
    assert compressed.dwa.huffman.codes == ()
    assert tuple(rule.suffix for rule in compressed.dwa.channel_rules) == ("R", "G", "B")
    assert tuple(group.channel_names for group in compressed.dwa.channel_layout.csc_groups) == (("R", "G", "B"),)
    assert compressed.dwa.geometry == exr_container._DwaChunkGeometry(
        lines_per_chunk=32,
        row_count=32,
        block_columns=2,
        block_rows=4,
        padded_width=16,
        padded_height=32,
        mirror_right=0,
        mirror_bottom=0,
    )
    assert raw.dwa is not None
    assert raw.dwa.leader is None
    assert raw.dwa.geometry.row_count == 1
    assert raw.dwa.geometry.mirror_bottom == 7


def test_dwa_v2_container_accepts_a_well_formed_rule_unrelated_to_file_channels(tmp_path: Path) -> None:
    """v1-exr-gpu-phase2 acceptance 6 and 7: unused valid rules do not make a DWA v2 file corrupt."""
    fixture = _build_dwa_exr(additional_rules=(_rule("Z", scheme=0, pixel_type=2),))
    path = tmp_path / "dwaa-v2-unused-rule.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert container.dwa_eligible is True
    compressed = container.chunks[0]
    assert compressed.dwa is not None
    assert tuple(rule.suffix for rule in compressed.dwa.channel_rules) == ("R", "G", "B", "Z")
    assert tuple(channel.name for channel in compressed.dwa.channel_layout.channels) == ("B", "G", "R")
    assert tuple(group.channel_names for group in compressed.dwa.channel_layout.csc_groups) == (("R", "G", "B"),)


def test_valid_dwa_v1_chunk_is_described_but_remains_gpu_ineligible(tmp_path: Path) -> None:
    """v1-exr-gpu-phase2 acceptance 10: a valid legacy DWA chunk is a fallback descriptor, not corruption."""
    fixture = _build_dwa_exr(version=1)
    path = tmp_path / "dwaa-v1.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert container.dwa_eligible is False
    assert container.chunks[0].dwa is not None
    assert container.chunks[0].dwa.leader is not None
    assert container.chunks[0].dwa.leader.version == 1


def test_dwab_descriptor_uses_256_lines_and_represents_the_final_partial_chunk(tmp_path: Path) -> None:
    """v1-exr-gpu-phase2 acceptance 10: DWAB parameterizes the shared descriptor with 256-line geometry."""
    fixture = _build_dwa_exr(compression="dwab")
    path = tmp_path / "dwab-v2.exr"
    path.write_bytes(fixture.payload)

    container = exr_container._parse_exr_container(path)

    assert container.compression == "dwab"
    assert container.lines_per_chunk == 256
    assert container.dwa_eligible is True
    assert tuple((chunk.row_start, chunk.row_count) for chunk in container.chunks) == ((0, 256), (256, 1))
    assert container.chunks[0].dwa is not None
    assert container.chunks[0].dwa.geometry.block_rows == 32
    assert container.chunks[1].dwa is not None
    assert container.chunks[1].dwa.geometry.mirror_bottom == 7


@pytest.mark.parametrize(
    "corruption",
    ("truncated-leader", "unknown-ac-scheme", "span-mismatch", "truncated-rules", "dc-count-mismatch"),
)
def test_dwa_container_corruption_is_actionable_before_pixel_decode(tmp_path: Path, corruption: str) -> None:
    """v1-exr-gpu-phase2 acceptance 6 and 10: corrupt DWA control data fails before any codec backend."""
    fixture = _build_dwa_exr()
    payload = bytearray(fixture.payload)
    payload_start = fixture.payload_offsets[0]
    if corruption == "truncated-leader":
        struct.pack_into("<i", payload, fixture.chunk_offsets[0] + 4, 87)
    elif corruption == "unknown-ac-scheme":
        struct.pack_into("<Q", payload, payload_start + 80, 9)
    elif corruption == "span-mismatch":
        ac_size = struct.unpack_from("<Q", payload, payload_start + 24)[0]
        struct.pack_into("<Q", payload, payload_start + 24, ac_size + 1)
    elif corruption == "truncated-rules":
        struct.pack_into("<H", payload, payload_start + 88, 0x7FFF)
    else:
        struct.pack_into("<Q", payload, payload_start + 72, 25)
    path = tmp_path / f"{corruption}.exr"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError, match=r"why=.*DWA.*what=.*how="):
        px.io.read_header(path)


def test_phase2_auto_selection_has_source_fixed_entries_for_every_dwa_direction() -> None:
    """v1-exr-gpu-phase2 acceptance 4-5: every DWA direction has one fixed internal backend."""
    dwa_selection = {key: backend for key, backend in io._EXR_ROUTING.items() if key[0] in {"dwaa", "dwab"}}

    assert dwa_selection == {
        ("dwaa", "read"): "gpu",
        ("dwaa", "write"): "gpu",
        ("dwab", "read"): "gpu",
        ("dwab", "write"): "gpu",
    }
