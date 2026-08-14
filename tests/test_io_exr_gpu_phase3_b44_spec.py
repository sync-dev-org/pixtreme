"""Specification tests for the Phase 3 OpenEXR B44 and B44A lanes."""

from __future__ import annotations

import importlib
import struct
from dataclasses import replace
from pathlib import Path

import cupy as cp
import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.codec_b44 as exr_b44
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io

_B44_EDGES = (
    (0, 4),
    (4, 8),
    (8, 12),
    (0, 1),
    (4, 5),
    (8, 9),
    (12, 13),
    (1, 2),
    (5, 6),
    (9, 10),
    (13, 14),
    (2, 3),
    (6, 7),
    (10, 11),
    (14, 15),
)


def _half_bits(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float16).view(np.uint16)


def _ordered_half(bits: np.ndarray) -> np.ndarray:
    """Independent ordered-HALF oracle from the pinned OpenEXR wire rules."""
    source = np.asarray(bits, dtype=np.uint16)
    special = np.bitwise_and(source, np.uint16(0x7C00)) == np.uint16(0x7C00)
    negative = np.bitwise_and(source, np.uint16(0x8000)) != 0
    return np.where(
        special,
        np.uint16(0x8000),
        np.where(negative, np.bitwise_not(source), np.bitwise_or(source, np.uint16(0x8000))),
    ).astype(np.uint16)


def _half_from_ordered(ordered: np.ndarray) -> np.ndarray:
    source = np.asarray(ordered, dtype=np.uint16)
    return np.where(
        np.bitwise_and(source, np.uint16(0x8000)) != 0,
        np.bitwise_and(source, np.uint16(0x7FFF)),
        np.bitwise_not(source),
    ).astype(np.uint16)


def _shift_and_round(values: np.ndarray, shift: int) -> np.ndarray:
    """Independent integer ties-to-even oracle matching OpenEXR shiftAndRound."""
    source = np.asarray(values, dtype=np.int64) << 1
    addend = (1 << shift) - 1
    next_shift = shift + 1
    tie_parity = np.bitwise_and(source >> next_shift, 1)
    return (source + addend + tie_parity) >> next_shift


def _plinear_luts() -> tuple[np.ndarray, np.ndarray]:
    """Independent all-pattern oracle matching OpenEXR's expf/logf table initializer."""
    bits = np.arange(65536, dtype=np.uint16)
    values = bits.view(np.float16).astype(np.float32)
    special = np.bitwise_and(bits, np.uint16(0x7C00)) == np.uint16(0x7C00)

    encode = np.zeros(65536, dtype=np.uint16)
    saturated = (bits >= np.uint16(0x558C)) & (bits < np.uint16(0x8000))
    encode[saturated] = np.uint16(0x7BFF)
    regular_encode = ~(special | saturated)
    with np.errstate(over="ignore", invalid="ignore"):
        encoded_values = np.exp(values[regular_encode] / np.float32(8.0)).astype(np.float32)
        encode[regular_encode] = encoded_values.astype(np.float16).view(np.uint16)

    decode = np.zeros(65536, dtype=np.uint16)
    regular_decode = ~(special | (bits > np.uint16(0x8000)))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        decoded_values = (np.float32(8.0) * np.log(values[regular_decode])).astype(np.float32)
        decode[regular_decode] = decoded_values.astype(np.float16).view(np.uint16)
    return encode, decode


def _pack_block(
    source_bits: np.ndarray,
    *,
    b44a: bool,
    perceptually_linear: bool = False,
) -> tuple[bytes, int, np.ndarray]:
    """Independent B44/B44A encoder oracle returning bytes, shift, and residuals."""
    bits = np.asarray(source_bits, dtype=np.uint16).reshape(16).copy()
    if perceptually_linear:
        bits = _plinear_luts()[0][bits]
    ordered = _ordered_half(bits)
    maximum = int(ordered.max())
    for shift in range(13):
        differences = _shift_and_round(maximum - ordered.astype(np.int64), shift)
        residuals = np.asarray(
            [differences[source] - differences[target] + 32 for source, target in _B44_EDGES],
            dtype=np.int64,
        )
        if np.all((0 <= residuals) & (residuals <= 63)):
            break
    else:  # pragma: no cover - the 16-bit domain is guaranteed to fit by shift 12
        raise AssertionError("no valid B44 shift")
    if b44a and np.all(residuals == 32):
        base = int(ordered[0])
        return bytes((base >> 8, base & 0xFF, 0xFC)), shift, residuals
    base = int(ordered[0]) if perceptually_linear else maximum - (int(differences[0]) << shift)
    packed = base
    packed = (packed << 6) | shift
    for residual in residuals:
        packed = (packed << 6) | int(residual)
    return packed.to_bytes(14, "big"), shift, residuals


def _decode_block(payload: bytes, *, perceptually_linear: bool = False) -> np.ndarray:
    """Independent B44/B44A decoder oracle for dense and the full flat marker grammar."""
    if len(payload) == 3 or payload[2] >= 0x34:
        ordered = np.full(16, (payload[0] << 8) | payload[1], dtype=np.uint16)
    else:
        packed = int.from_bytes(payload, "big")
        ordered = np.empty(16, dtype=np.uint16)
        ordered[0] = np.uint16((packed >> 96) & 0xFFFF)
        shift = (packed >> 90) & 0x3F
        residuals = tuple((packed >> (84 - 6 * index)) & 0x3F for index in range(15))
        for residual, (source, target) in zip(residuals, _B44_EDGES, strict=True):
            ordered[target] = np.uint16((int(ordered[source]) + ((residual - 32) << shift)) & 0xFFFF)
    bits = _half_from_ordered(ordered)
    if perceptually_linear:
        bits = _plinear_luts()[1][bits]
    return bits


def _fixture_blocks() -> np.ndarray:
    constant = _half_bits([1.25] * 16)
    gradient = _half_bits(np.linspace(-8.0, 8.0, 16, dtype=np.float32))
    signed = _half_bits(
        [-0.0, 0.0, -1.0, 1.0, -2.5, 2.5, -16.0, 16.0, -0.125, 0.125, -64.0, 64.0, -3.0, 3.0, -7.0, 7.0]
    )
    non_finite = np.asarray(
        [0x7C00, 0xFC00, 0x7E00, 0xFE01, 0x0000, 0x8000, 0x3C00, 0xBC00] * 2,
        dtype=np.uint16,
    )
    boundary_d = np.asarray([32, 32, 32, 32, 1, 1, 1, 1, 33, 33, 33, 33, 33, 33, 33, 33], dtype=np.uint16)
    residual_boundary = _half_from_ordered(np.uint16(0x9000) - boundary_d)
    halfway_d = np.asarray([0, 1, 3, 65, 0, 1, 3, 65, 0, 1, 3, 65, 0, 1, 3, 65], dtype=np.uint16)
    halfway = _half_from_ordered(np.uint16(0xA000) - halfway_d)
    return np.stack((constant, gradient, signed, non_finite, residual_boundary, halfway))


def _attribute(name: str, attribute_type: str, payload: bytes) -> bytes:
    return name.encode() + b"\x00" + attribute_type.encode() + b"\x00" + struct.pack("<I", len(payload)) + payload


def _single_channel_header(*, width: int, height: int, compression_code: int) -> bytes:
    channel_payload = b"Y\x00" + struct.pack("<iB3xii", 1, 0, 1, 1) + b"\x00"
    data_window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    attributes = (
        _attribute("channels", "chlist", channel_payload),
        _attribute("compression", "compression", bytes((compression_code,))),
        _attribute("dataWindow", "box2i", data_window),
        _attribute("displayWindow", "box2i", data_window),
        _attribute("lineOrder", "lineOrder", b"\x00"),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )
    return struct.pack("<II", 20000630, 2) + b"".join(attributes) + b"\x00"


def _single_chunk_file(header: bytes, payload: bytes) -> bytes:
    chunk_offset = len(header) + 8
    return header + struct.pack("<Q", chunk_offset) + struct.pack("<ii", 0, len(payload)) + payload


@pytest.mark.parametrize("b44a", (False, True), ids=("b44", "b44a"))
def test_b44_gpu_block_primitive_matches_dense_flat_shift_rounding_and_nonfinite_oracle(b44a: bool) -> None:
    """v1-exr-gpu-phase3 acceptance 24, 25, 26, and 27: GPU blocks match the independent bit oracle."""
    cp = importlib.import_module("cupy")
    blocks = _fixture_blocks()
    expected = tuple(_pack_block(block, b44a=b44a) for block in blocks)

    encoded, offsets, sizes = exr_b44._encode_b44_blocks_gpu(cp.asarray(blocks), b44a=b44a)
    host = encoded.get().tobytes()
    host_offsets = offsets.get().tolist()
    host_sizes = sizes.get().tolist()

    assert host_sizes == [len(item[0]) for item in expected]
    assert [host[offset : offset + size] for offset, size in zip(host_offsets, host_sizes, strict=True)] == [
        item[0] for item in expected
    ]
    dense_payloads = [item[0] for item in expected if len(item[0]) == 14]
    assert dense_payloads
    assert all(payload[2] < 0x34 for payload in dense_payloads)
    assert any(item[1] == 0 for item in expected)
    assert any(item[1] > 0 for item in expected)
    assert any((item[2].min(), item[2].max()) == (0, 63) for item in expected)
    if b44a:
        assert expected[0][0][2] == 0xFC
        assert len(expected[0][0]) == 3
    else:
        assert len(expected[0][0]) == 14

    decoded = exr_b44._decode_b44_blocks_gpu(encoded, offsets, sizes).get()
    independent = np.stack([_decode_block(item[0]) for item in expected])
    np.testing.assert_array_equal(decoded, independent)


@pytest.mark.parametrize("marker", (0x34, 0xFC, 0xFF))
def test_b44a_gpu_decoder_accepts_complete_flat_marker_range(marker: int) -> None:
    """v1-exr-gpu-phase3 acceptance 27: B44A accepts every byte[2] >= 0x34 as a three-byte flat form."""
    cp = importlib.import_module("cupy")
    payload = bytes((0xBC, 0x00, marker))
    decoded = exr_b44._decode_b44_blocks_gpu(
        cp.asarray(np.frombuffer(payload, dtype=np.uint8)),
        cp.asarray(np.asarray([0], dtype=np.int64)),
        cp.asarray(np.asarray([3], dtype=np.int32)),
    ).get()[0]
    np.testing.assert_array_equal(decoded, _decode_block(payload))


def test_b44_plinear_gpu_luts_match_all_half_patterns() -> None:
    """v1-exr-gpu-phase3 acceptance 28: fixed pLinear encode/decode LUTs match all 65,536 HALF patterns."""
    cp = importlib.import_module("cupy")
    bits = np.arange(65536, dtype=np.uint16)
    expected_encode, expected_decode = _plinear_luts()

    actual_encode = exr_harness._b44_plinear_encode_gpu(cp.asarray(bits)).get()
    actual_decode = exr_harness._b44_plinear_decode_gpu(cp.asarray(bits)).get()

    np.testing.assert_array_equal(actual_encode, expected_encode)
    np.testing.assert_array_equal(actual_decode, expected_decode)
    for pattern in (0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0xBC00, 0x558B, 0x558C, 0x7BFF):
        assert actual_encode[pattern] == expected_encode[pattern]
        assert actual_decode[pattern] == expected_decode[pattern]


def test_b44_plinear_block_uses_transformed_first_sample_as_base() -> None:
    """v1-exr-gpu-phase3 acceptance 25 and 28: pLinear blocks use t[0] without exactmax correction."""
    cp = importlib.import_module("cupy")
    block = _half_bits([0.25, 0.5, 1.0, 2.0, 0.375, 0.75, 1.5, 3.0, 0.625, 1.25, 2.5, 5.0, 1.0, 2.0, 4.0, 8.0])
    expected, _, _ = _pack_block(block, b44a=False, perceptually_linear=True)

    encoded, offsets, sizes = exr_b44._encode_b44_blocks_gpu(
        cp.asarray(block.reshape(1, 16)),
        b44a=False,
        perceptually_linear=True,
    )

    assert offsets.get().tolist() == [0]
    assert sizes.get().tolist() == [14]
    assert encoded.get().tobytes() == expected
    transformed_first = _ordered_half(_plinear_luts()[0][block[:1]])[0]
    assert expected[:2] == int(transformed_first).to_bytes(2, "big")
    decoded = exr_b44._decode_b44_blocks_gpu(
        encoded,
        offsets,
        sizes,
        perceptually_linear=True,
    ).get()[0]
    np.testing.assert_array_equal(decoded, _decode_block(expected, perceptually_linear=True))


def test_b44_uint_plane_primitive_is_file_channel_order_and_bit_exact() -> None:
    """v1-exr-gpu-phase3 acceptance 23 and 30: UINT wire sections are raw file-channel-order planes."""
    cp = importlib.import_module("cupy")
    values = np.asarray(
        [
            [[0x01020304, 0xA0B0C0D0], [0x11121314, 0x21222324], [0x31323334, 0x41424344]],
            [[0x51525354, 0x61626364], [0x71727374, 0x81828384], [0x91929394, 0xA1A2A3A4]],
        ],
        dtype=np.uint32,
    )
    raw = values.transpose(0, 2, 1).copy().reshape(-1).view(np.uint8)

    planar = exr_b44._b44_reorder_chunk_planes_gpu(
        cp.asarray(raw),
        (0,),
        (2,),
        width=3,
        channel_count=2,
        bytes_per_sample=4,
    ).get()

    expected = values.transpose(2, 0, 1).copy().reshape(-1).view(np.uint8)
    np.testing.assert_array_equal(planar, expected)


@pytest.mark.parametrize("codec", ("b44", "b44a"))
def test_openexr_b44_uint_sections_are_bit_exact_in_gpu_and_host_materializers(
    tmp_path: Path,
    codec: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 23, 30, and 31: internal materializers preserve UINT32 wire bits."""
    from openexr_dev_oracle import OpenEXR

    cp = importlib.import_module("cupy")
    height, width = 7, 9
    half = np.full((height, width), np.float16(0.5), dtype=np.float16)
    sample_indices = np.arange(height * width, dtype=np.uint32).reshape(height, width)
    unsigned = sample_indices * np.uint32(0x9E3779B1) + np.uint32(0x01020304)
    compression = OpenEXR.B44_COMPRESSION if codec == "b44" else OpenEXR.B44A_COMPRESSION
    path = tmp_path / f"uint-section-{codec}.exr"
    OpenEXR.File({"compression": compression}, {"H": half, "U": unsigned}).write(str(path))
    container = exr_container._parse_exr_container(path)
    prepared = exr_b44._prepare_exr_b44_read_chunks(container)
    assert len(container.chunks) == 1
    assert container.chunks[0].raw_stored is False
    assert any(section.pixel_type == 0 for section in container.chunks[0].phase3.channel_sections)

    gpu_materialized = exr_b44._materialize_b44_gpu(prepared, cp.asarray(prepared.host_staging)).get()
    host_materialized = exr_b44._materialize_b44_host(prepared)
    channels = container.parts[0].channels
    uint_index = next(index for index, channel in enumerate(channels) if channel.name == "U")
    channel_offset = sum(width * channel.bytes_per_sample for channel in channels[:uint_index])
    row_bytes = sum(width * channel.bytes_per_sample for channel in channels)

    def uint_bits(materialized: np.ndarray) -> np.ndarray:
        rows = materialized.reshape(height, row_bytes)
        channel_bytes = np.ascontiguousarray(rows[:, channel_offset : channel_offset + width * 4])
        return channel_bytes.view("<u4").reshape(height, width)

    np.testing.assert_array_equal(uint_bits(gpu_materialized), unsigned)
    np.testing.assert_array_equal(uint_bits(host_materialized), unsigned)


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
@pytest.mark.parametrize("marker", (0x34, 0xFC, 0xFF))
def test_forced_b44a_read_accepts_noncanonical_flat_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    marker: int,
) -> None:
    """v1-exr-gpu-phase3 acceptance 5, 27, and 29: both forced lanes materialize valid flat marker forms."""
    payload = bytes((0xBC, 0x00, marker))
    path = tmp_path / f"flat-{marker:02x}-{backend}.exr"
    path.write_bytes(_single_chunk_file(_single_channel_header(width=4, height=4, compression_code=7), payload))
    monkeypatch.setitem(io._EXR_ROUTING, ("b44a", "read"), backend)

    frame = px.io.read_image(path, channels=["Y"], unchanged=True, colorspace="ACEScg", gamma="linear")

    expected = _decode_block(payload).reshape(4, 4)
    np.testing.assert_array_equal(frame.data.get()[..., 0].view(np.uint16), expected)


@pytest.mark.parametrize(
    ("compression_code", "payload", "reason"),
    (
        (6, b"\x80\x00\x00" + b"\x00" * 10, "dense HALF block is truncated"),
        (7, b"\x80\x00\x00" + b"\x00" * 10, "dense HALF block is truncated"),
        (7, b"\x80\x00\xfc\x00", "channel sections do not consume"),
    ),
    ids=("b44-dense-truncated", "b44a-dense-truncated", "b44a-section-excess"),
)
def test_b44_frontend_rejects_truncated_blocks_and_section_excess(
    tmp_path: Path,
    compression_code: int,
    payload: bytes,
    reason: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 29 and 32: dense/flat sections must consume validated spans exactly."""
    path = tmp_path / f"invalid-b44-{compression_code}-{len(payload)}.exr"
    path.write_bytes(
        _single_chunk_file(_single_channel_header(width=4, height=4, compression_code=compression_code), payload)
    )

    with pytest.raises(RuntimeError) as exc_info:
        px.io.read_header(path)

    message = str(exc_info.value)
    assert "why=" in message
    assert reason in message
    assert "what=" in message
    assert "how=" in message


def test_b44_materializer_rejects_mutated_block_ownership(tmp_path: Path) -> None:
    """v1-exr-gpu-phase3 acceptance 29 and 32: materialize revalidates descriptor block ownership."""
    from openexr_dev_oracle import OpenEXR

    values = np.linspace(-2.0, 3.0, 64, dtype=np.float32).reshape(8, 8).astype(np.float16)
    path = tmp_path / "b44-mutated-ownership.exr"
    OpenEXR.File({"compression": OpenEXR.B44_COMPRESSION}, {"Y": values}).write(str(path))
    container = exr_container._parse_exr_container(path)
    chunk = container.chunks[0]
    descriptor = chunk.phase3
    assert descriptor is not None and descriptor.blocks
    mutated_block = replace(descriptor.blocks[0], block_column=descriptor.blocks[0].block_column + 1)
    mutated_descriptor = replace(descriptor, blocks=(mutated_block, *descriptor.blocks[1:]))
    mutated_container = replace(container, chunks=(replace(chunk, phase3=mutated_descriptor),))

    with pytest.raises(RuntimeError, match=r"why=.*ownership.*what=.*how="):
        exr_b44._prepare_exr_b44_read_chunks(mutated_container)


def test_b44_materializer_rejects_mutated_lazy_block_ownership(tmp_path: Path) -> None:
    """v1-exr-gpu-phase3 acceptance 29 and 32: lazy descriptor ownership failure raises the actionable RuntimeError."""
    from openexr_dev_oracle import OpenEXR

    values = np.linspace(-2.0, 3.0, 64, dtype=np.float32).reshape(8, 8).astype(np.float16)
    path = tmp_path / "b44-mutated-lazy-ownership.exr"
    OpenEXR.File({"compression": OpenEXR.B44_COMPRESSION}, {"Y": values}).write(str(path))
    container = exr_container._parse_exr_container(path)
    chunk = container.chunks[0]
    descriptor = chunk.phase3
    assert descriptor is not None and isinstance(descriptor.blocks, exr_container._Phase3B44Blocks)
    mutated_blocks = replace(descriptor.blocks, codec="b44a")
    mutated_descriptor = replace(descriptor, blocks=mutated_blocks)
    mutated_container = replace(container, chunks=(replace(chunk, phase3=mutated_descriptor),))

    with pytest.raises(RuntimeError, match=r"why=.*ownership.*what=.*how="):
        exr_b44._prepare_exr_b44_read_chunks(mutated_container)


@pytest.mark.parametrize("codec", ("b44", "b44a"))
@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_openexr_b44_read_materializes_mixed_sections_edge_repeat_and_plinear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
    backend: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 23, 28, 29, 30, and 31: mixed OpenEXR chunks decode in both lanes."""
    from openexr_dev_oracle import OpenEXR

    height, width = 7, 5
    half = np.linspace(-4.0, 12.0, height * width, dtype=np.float32).reshape(height, width).astype(np.float16)
    half[:4, :4] = np.float16(0.5)
    single = np.linspace(-8.0, 24.0, height * width, dtype=np.float32).reshape(height, width)
    unsigned = np.arange(height * width, dtype=np.uint32).reshape(height, width) * np.uint32(0x01010101)
    compression = OpenEXR.B44_COMPRESSION if codec == "b44" else OpenEXR.B44A_COMPRESSION
    path = tmp_path / f"openexr-{codec}-{backend}.exr"
    OpenEXR.File(
        {"compression": compression},
        {
            "H": OpenEXR.Channel("H", half, 1, 1, True),
            "F": OpenEXR.Channel("F", single),
            "U": OpenEXR.Channel("U", unsigned),
        },
    ).write(str(path))
    reference_file = OpenEXR.File(str(path), separate_channels=True)
    reference = np.stack(
        [np.asarray(reference_file.channels()[name].pixels).astype(np.float32) for name in ("H", "F")], axis=2
    )
    container = exr_container._parse_exr_container(path)
    assert container.phase3_eligible is True
    assert any(not chunk.raw_stored for chunk in container.chunks)
    assert any(section.pixel_type == 0 for chunk in container.chunks for section in chunk.phase3.channel_sections)
    assert any(section.perceptually_linear for chunk in container.chunks for section in chunk.phase3.channel_sections)
    monkeypatch.setitem(io._EXR_ROUTING, (codec, "read"), backend)

    frame = px.io.read_image(path, channels=["H", "F"], unchanged=False, colorspace="ACEScg", gamma="linear")

    assert frame.channels == ("H", "F")
    assert frame.data.flags.c_contiguous
    np.testing.assert_array_equal(frame.data.get(), reference)


def _b44_channel_oracle(container: exr_container._ExrContainer, channel_name: str) -> np.ndarray:
    width = container.data_window[2] - container.data_window[0] + 1
    height = container.data_window[3] - container.data_window[1] + 1
    channels = container.parts[0].channels
    channel = next(item for item in channels if item.name == channel_name)
    channel_offset = sum(width * item.bytes_per_sample for item in channels[: channels.index(channel)])
    row_bytes = sum(width * item.bytes_per_sample for item in channels)
    output = np.empty((height, width), dtype=np.uint16)
    for chunk in container.chunks:
        if chunk.raw_stored:
            raw = np.frombuffer(container.data[chunk.payload_start : chunk.payload_end], dtype=np.uint8).reshape(
                chunk.row_count, row_bytes
            )
            channel_bytes = np.ascontiguousarray(raw[:, channel_offset : channel_offset + width * 2])
            output[chunk.row_start : chunk.row_start + chunk.row_count] = channel_bytes.view("<u2").reshape(
                chunk.row_count, width
            )
            continue
        descriptor = chunk.phase3
        assert descriptor is not None
        for block in descriptor.blocks:
            if block.channel_name != channel_name:
                continue
            payload = container.data[block.payload_span.start : block.payload_span.end]
            decoded = _decode_block(payload).reshape(4, 4)
            row = block.block_row * 4
            column = block.block_column * 4
            valid_rows = min(4, chunk.row_count - row)
            valid_columns = min(4, width - column)
            output[
                chunk.row_start + row : chunk.row_start + row + valid_rows,
                column : column + valid_columns,
            ] = decoded[:valid_rows, :valid_columns]
    return output


@pytest.mark.parametrize("codec", ("b44", "b44a"))
def test_gpu_b44_write_cross_decodes_in_openexr_with_dense_flat_and_partial_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 9, 23, 25, 26, 27, 29, and 31: GPU payloads cross-decode in OpenEXR."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height, width = 37, 9
    gradient = np.linspace(-16.0, 32.0, height * width, dtype=np.float32).reshape(height, width)
    values = np.stack((gradient, gradient * np.float32(0.5), -gradient), axis=2).astype(np.float16)
    values[:16, :8, 0] = np.float16(0.75)
    values[:8, :, 1] = np.float16(-0.5)
    frame = px.io.from_array(
        cp.asarray(values),
        colorspace="ACEScg",
        gamma="linear",
        channels=("R", "G", "B"),
    )
    path = tmp_path / f"gpu-{codec}.exr"
    calls = 0
    original = io._write_exr_gpu

    def gpu_spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, (codec, "write"), "gpu")
    monkeypatch.setattr(io, "_write_exr_gpu", gpu_spy)

    px.io.write_image(path, frame, compression=codec)

    container = exr_container._parse_exr_container(path)
    assert calls == 1
    assert container.phase3_eligible is True
    assert all(not chunk.raw_stored for chunk in container.chunks)
    block_sizes = tuple(block.stored_size for chunk in container.chunks for block in chunk.phase3.blocks)
    assert 14 in block_sizes
    if codec == "b44a":
        assert 3 in block_sizes
    reference_file = OpenEXR.File(str(path), separate_channels=True)
    assert reference_file.header()["compression"] == (
        OpenEXR.B44_COMPRESSION if codec == "b44" else OpenEXR.B44A_COMPRESSION
    )
    for channel_name in ("R", "G", "B"):
        decoded = np.asarray(reference_file.channels()[channel_name].pixels).view(np.uint16)
        np.testing.assert_array_equal(decoded, _b44_channel_oracle(container, channel_name))


@pytest.mark.parametrize("codec", ("b44", "b44a"))
def test_gpu_b44_float_write_uses_raw_fallback_after_plane_materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 7, 9, 23, and 31: FLOAT plane payloads fall back to exact raw chunks."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    height, width = 33, 7
    values = np.linspace(-32.0, 64.0, height * width, dtype=np.float32).reshape(height, width)
    frame = px.io.from_array(
        cp.asarray(values[..., np.newaxis]),
        colorspace="ACEScg",
        gamma="linear",
        channels=("Y",),
    )
    path = tmp_path / f"gpu-{codec}-float.exr"
    monkeypatch.setitem(io._EXR_ROUTING, (codec, "write"), "gpu")

    px.io.write_image(path, frame, compression=codec, dtype="float32")

    container = exr_container._parse_exr_container(path)
    assert tuple(chunk.raw_stored for chunk in container.chunks) == (True, True)
    reference = OpenEXR.File(str(path), separate_channels=True)
    decoded = np.asarray(reference.channels()["Y"].pixels)
    np.testing.assert_array_equal(decoded.view(np.uint32), values.view(np.uint32))


@pytest.mark.parametrize("codec", ("b44", "b44a"))
def test_forced_custom_cpu_b44_float_read_materializes_all_raw_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 23 and 29: forced host reads handle all-raw chunks."""
    import cupy as cp

    height, width = 33, 7
    values = np.linspace(-32.0, 64.0, height * width, dtype=np.float32).reshape(height, width)
    frame = px.io.from_array(
        cp.asarray(values[..., np.newaxis]),
        colorspace="ACEScg",
        gamma="linear",
        channels=("Y",),
    )
    path = tmp_path / f"forced-custom-cpu-{codec}-float.exr"
    monkeypatch.setitem(io._EXR_ROUTING, (codec, "write"), "gpu")
    px.io.write_image(path, frame, compression=codec, dtype="float32")

    container = exr_container._parse_exr_container(path)
    prepared = exr_b44._prepare_exr_b44_read_chunks(container)
    assert tuple(chunk.raw_stored for chunk in container.chunks) == (True, True)
    assert prepared.block_perceptually_linear.size == 0
    monkeypatch.setitem(io._EXR_ROUTING, (codec, "read"), "custom_cpu")

    decoded = px.io.read_image(path, channels=["Y"], unchanged=True, colorspace="ACEScg", gamma="linear")

    np.testing.assert_array_equal(decoded.data.get()[..., 0].view(np.uint32), values.view(np.uint32))


@pytest.mark.parametrize("codec", ("b44", "b44a"))
def test_b44_lanes_keep_block_parallel_and_image_transfer_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
    request: pytest.FixtureRequest,
) -> None:
    """v1-exr-gpu-phase3 acceptance 8, 9, and 29: B44 block tasks launch in parallel within transfer limits."""
    from openexr_dev_oracle import OpenEXR

    height, width = 65, 65
    values = np.linspace(-2.0, 3.0, height * width, dtype=np.float32).reshape(height, width).astype(np.float16)
    values[:16, :16] = np.float16(0.5)
    compression = OpenEXR.B44_COMPRESSION if codec == "b44" else OpenEXR.B44A_COMPRESSION
    path = tmp_path / f"{codec}-transfer.exr"
    OpenEXR.File({"compression": compression}, {"Y": values}).write(str(path))
    container = exr_container._parse_exr_container(path)
    selected = container.parts[0].channels
    stored_block_sizes = tuple(
        block.stored_size
        for chunk in container.chunks
        if chunk.phase3 is not None and not chunk.raw_stored
        for block in chunk.phase3.blocks
    )
    assert 14 in stored_block_sizes
    if codec == "b44a":
        assert 3 in stored_block_sizes
    else:
        assert set(stored_block_sizes) == {14}
    prepared = exr_b44._prepare_exr_b44_read_chunks(container)
    section_count = int(prepared.block_section_descriptors.shape[0])
    block_count = int(prepared.block_perceptually_linear.size)
    launch_geometries: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class _RecordingRawKernel(cp.RawKernel):
        def __call__(self, grid: object, block: object, *arguments: object, **keywords: object) -> None:
            launch_geometries.append((tuple(int(value) for value in grid), tuple(int(value) for value in block)))
            super().__call__(grid, block, *arguments, **keywords)

    def _clear_module_kernel_caches() -> None:
        for attribute in vars(exr_b44).values():
            cache_clear = getattr(attribute, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()

    # Neutral launch seam: rebuild this module's cached kernels under a recording
    # RawKernel subclass instead of addressing production helpers by name.
    _clear_module_kernel_caches()
    request.addfinalizer(_clear_module_kernel_caches)
    monkeypatch.setattr(cp, "RawKernel", _RecordingRawKernel)
    warmup = exr_b44._read_exr_b44_gpu(container, selected, output_dtype="float32")
    assert warmup.shape == (height, width, 1)
    launch_geometries.clear()
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    gpu_actual = exr_b44._read_exr_b44_gpu(container, selected, output_dtype="float32")

    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=20, max_total_nbytes=300_000, max_shape_elements=65_536
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=5, max_total_nbytes=64, max_shape_elements=4
    )

    gpu_staging = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == prepared.host_staging.nbytes
        and transfer.shape == prepared.host_staging.shape
        and transfer.dtype == prepared.host_staging.dtype.name
    ]
    assert len(gpu_staging) == 1
    assert gpu_actual.shape == (height, width, 1)
    assert gpu_actual.dtype == np.dtype(np.float32)
    assert block_count > exr_container._EXR_THREADS_PER_BLOCK
    threads_per_block = exr_container._EXR_THREADS_PER_BLOCK
    scan_geometry = (((section_count + threads_per_block - 1) // threads_per_block,), (threads_per_block,))
    decode_geometry = (((block_count + threads_per_block - 1) // threads_per_block,), (threads_per_block,))
    scatter_geometry = (((block_count * 16 + threads_per_block - 1) // threads_per_block,), (threads_per_block,))
    assert decode_geometry[0][0] > 1
    assert launch_geometries.count(scan_geometry) >= 1
    assert launch_geometries.count(decode_geometry) == 1
    assert launch_geometries.count(scatter_geometry) >= 1
    assert not [
        transfer for transfer in transfers if transfer.direction == "d2h" and transfer.shape == gpu_actual.shape
    ]

    transfers.clear()
    cpu_actual = exr_b44._read_exr_b44_custom_cpu(container, selected, output_dtype="float32")
    exr_harness._assert_cupy_transfer_budget(
        transfers,
        direction="h2d",
        max_count=1,
        max_total_nbytes=cpu_actual.nbytes,
        max_shape_elements=height * width,
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=0, max_total_nbytes=0, max_shape_elements=0
    )
    image_transfers = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == cpu_actual.nbytes
        and transfer.shape == cpu_actual.shape
        and transfer.dtype == cpu_actual.dtype.name
    ]
    assert len(image_transfers) == 1
    cp.testing.assert_array_equal(cpu_actual, gpu_actual)


@pytest.mark.parametrize("codec", ("b44", "b44a"))
def test_b44_parser_materializes_block_descriptors_only_when_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codec: str,
) -> None:
    """v1-exr-gpu-phase3 acceptance 29 and 36: public parse keeps validated B44 block metadata lazy."""
    from openexr_dev_oracle import OpenEXR

    height, width = 37, 9
    values = np.linspace(-8.0, 16.0, height * width, dtype=np.float32).reshape(height, width).astype(np.float16)
    values[:16, :8] = np.float16(0.5)
    compression = OpenEXR.B44_COMPRESSION if codec == "b44" else OpenEXR.B44A_COMPRESSION
    path = tmp_path / f"lazy-{codec}.exr"
    OpenEXR.File({"compression": compression}, {"Y": values}).write(str(path))
    constructed = 0
    original_block = exr_container._Phase3Block

    def tracked_block(*args: object, **kwargs: object) -> exr_container._Phase3Block:
        nonlocal constructed
        constructed += 1
        return original_block(*args, **kwargs)

    monkeypatch.setattr(exr_container, "_Phase3Block", tracked_block)

    container = exr_container._parse_exr_container(path)
    descriptors = tuple(chunk.phase3 for chunk in container.chunks if chunk.phase3 is not None and not chunk.raw_stored)
    expected_block_count = sum(
        section.block_count for descriptor in descriptors for section in descriptor.channel_sections
    )

    assert constructed == 0
    assert sum(len(descriptor.blocks) for descriptor in descriptors) == expected_block_count
    first = descriptors[0].blocks[0]
    last = descriptors[-1].blocks[-1]
    assert constructed == 2
    assert first.block_row == 0
    assert first.block_column == 0
    assert last.payload_span.end == descriptors[-1].channel_sections[-1].payload_span.end
    for descriptor in descriptors:
        blocks = tuple(descriptor.blocks)
        for section in descriptor.channel_sections:
            if not section.block_count:
                continue
            owned = blocks[section.block_start : section.block_start + section.block_count]
            assert tuple((block.block_row, block.block_column) for block in owned) == tuple(
                divmod(local_block, (width + 3) // 4) for local_block in range(section.block_count)
            )
            assert owned[0].payload_span.start == section.payload_span.start
            assert owned[-1].payload_span.end == section.payload_span.end
            assert sum(block.stored_size for block in owned) == section.payload_span.size
            assert all(left.payload_span.end == right.payload_span.start for left, right in zip(owned, owned[1:]))
    assert constructed == expected_block_count + 2
