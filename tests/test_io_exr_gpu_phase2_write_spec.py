"""Specification tests for the Phase 2 OpenEXR DWA hybrid write lane."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cupy as cp
import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.codec_dwa as exr_dwa
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.packing as exr_packing
import pixtreme._io.formats.exr.selection as io

_JPEG_LUMINANCE = np.asarray(
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
_JPEG_CHROMA = np.asarray(
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
    )
    + (99,) * 32,
    dtype=np.float32,
)
_NATURAL_TO_ZIGZAG = np.asarray(
    (
        0,
        1,
        5,
        6,
        14,
        15,
        27,
        28,
        2,
        4,
        7,
        13,
        16,
        26,
        29,
        42,
        3,
        8,
        12,
        17,
        25,
        30,
        41,
        43,
        9,
        11,
        18,
        24,
        31,
        40,
        44,
        53,
        10,
        19,
        23,
        32,
        39,
        45,
        52,
        54,
        20,
        22,
        33,
        38,
        46,
        51,
        55,
        60,
        21,
        34,
        37,
        47,
        50,
        56,
        59,
        61,
        35,
        36,
        48,
        49,
        57,
        58,
        62,
        63,
    ),
    dtype=np.intp,
)
_HALF_BITS = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
_HALF_VALUES = _HALF_BITS.view(np.float16).astype(np.float32)
_FINITE_HALF = np.isfinite(_HALF_VALUES)


def _half_population(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint16)
    byte_view = values.view(np.uint8).reshape(-1, 2)
    return np.unpackbits(byte_view, axis=1).sum(axis=1)


def _minimum_population_half(value: float, tolerance: float) -> np.uint16:
    source = float(np.float16(value))
    with np.errstate(invalid="ignore"):
        distance = np.abs(_HALF_VALUES - np.float32(source))
    allowed = _FINITE_HALF & (distance < np.float32(tolerance))
    allowed_bits = _HALF_BITS[allowed]
    allowed_distance = distance[allowed]
    assert allowed_bits.size
    population = _half_population(allowed_bits)
    minimum_population = population.min()
    finalists = allowed_bits[population == minimum_population]
    finalist_distance = allowed_distance[population == minimum_population]
    minimum_distance = finalist_distance.min()
    finalists = finalists[finalist_distance == minimum_distance]
    # OpenEXR's tie order chooses the smaller magnitude candidate.
    magnitudes = np.bitwise_and(finalists, np.uint16(0x7FFF))
    return np.uint16(finalists[np.argmin(magnitudes)])


def _read_oracle_bits(data: bytes, bit_offset: int, width: int) -> tuple[int, int]:
    assert bit_offset + width <= len(data) * 8
    value = 0
    for index in range(width):
        absolute = bit_offset + index
        value = (value << 1) | ((data[absolute // 8] >> (7 - absolute % 8)) & 1)
    return value, bit_offset + width


def _decode_candidate_huffman(payload: bytes, *, expected_count: int) -> np.ndarray:
    minimum_symbol, repeat_symbol, table_size, data_bit_count, reserved = struct.unpack_from("<IIIII", payload)
    assert reserved == 0
    packed_table = payload[20 : 20 + table_size]
    lengths: list[int] = []
    table_bit_offset = 0
    while len(lengths) < repeat_symbol - minimum_symbol + 1:
        token, table_bit_offset = _read_oracle_bits(packed_table, table_bit_offset, 6)
        if token <= 58:
            lengths.append(token)
        else:
            if token == 63:
                extra, table_bit_offset = _read_oracle_bits(packed_table, table_bit_offset, 8)
                run = extra + 6
            else:
                run = token - 57
            lengths.extend((0,) * run)
    assert len(lengths) == repeat_symbol - minimum_symbol + 1
    assert (table_bit_offset + 7) // 8 == table_size

    maximum_length = max(lengths)
    counts = [0] * (maximum_length + 1)
    for length in lengths:
        if length:
            counts[length] += 1
    next_code = [0] * (maximum_length + 1)
    code = 0
    for length in range(maximum_length, 0, -1):
        next_code[length] = code
        code = (code + counts[length]) >> 1
    symbols_by_code: dict[tuple[int, int], int] = {}
    for offset, length in enumerate(lengths):
        if length:
            symbols_by_code[(length, next_code[length])] = minimum_symbol + offset
            next_code[length] += 1

    encoded = payload[20 + table_size :]
    output: list[int] = []
    data_bit_offset = 0
    while data_bit_offset < data_bit_count:
        code = 0
        symbol = None
        for length in range(1, maximum_length + 1):
            bit, data_bit_offset = _read_oracle_bits(encoded, data_bit_offset, 1)
            code = (code << 1) | bit
            symbol = symbols_by_code.get((length, code))
            if symbol is not None:
                break
        assert symbol is not None
        if symbol == repeat_symbol:
            assert output
            repeat_count, data_bit_offset = _read_oracle_bits(encoded, data_bit_offset, 8)
            assert repeat_count > 0
            output.extend((output[-1],) * repeat_count)
        else:
            output.append(symbol)
    assert len(output) == expected_count
    return np.asarray(output, dtype=np.uint16)


def _restore_candidate_dc(payload: bytes, *, expected_count: int) -> np.ndarray:
    transformed = np.frombuffer(zlib.decompress(payload), dtype=np.uint8)
    assert transformed.size == expected_count * 2
    cumulative = np.cumsum(transformed, dtype=np.int64)
    grouped = np.bitwise_and(cumulative - np.arange(transformed.size, dtype=np.int64) * 128, 255).astype(np.uint8)
    restored = np.empty_like(grouped)
    half = (grouped.size + 1) // 2
    restored[::2] = grouped[:half]
    restored[1::2] = grouped[half:]
    return restored.view("<u2")


def _candidate_file_coefficients(path: Path) -> np.ndarray:
    container = exr_container._parse_exr_container(path)
    chunks = tuple(chunk for chunk in container.chunks if not chunk.raw_stored)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.dwa is not None
    leader = struct.unpack_from("<11Q", container.data, chunk.payload_start)
    assert leader[0] == 2 and leader[10] == 0
    rule_size = struct.unpack_from("<H", container.data, chunk.payload_start + 88)[0]
    stream_start = chunk.payload_start + 88 + rule_size
    unknown_end = stream_start + leader[2]
    ac_end = unknown_end + leader[3]
    dc_end = ac_end + leader[4]
    assert dc_end + leader[5] == chunk.payload_end
    ac_symbols = _decode_candidate_huffman(container.data[unknown_end:ac_end], expected_count=leader[8])
    dc_values = _restore_candidate_dc(container.data[ac_end:dc_end], expected_count=leader[9])

    geometry = chunk.dwa.geometry
    block_count = geometry.block_columns * geometry.block_rows
    assert block_count and leader[9] % block_count == 0
    component_count = leader[9] // block_count
    zigzag = np.zeros((block_count, component_count, 64), dtype=np.uint16)
    for component in range(component_count):
        start = component * block_count
        zigzag[:, component, 0] = dc_values[start : start + block_count]
    symbol_offset = 0
    for block in range(block_count):
        for component in range(component_count):
            position = 1
            while position < 64:
                symbol = int(ac_symbols[symbol_offset])
                symbol_offset += 1
                if symbol & 0xFF00 == 0xFF00:
                    zero_count = symbol & 0xFF
                    position = 64 if zero_count == 0 else position + zero_count
                    assert position <= 64
                else:
                    zigzag[block, component, position] = symbol
                    position += 1
    assert symbol_offset == ac_symbols.size
    return zigzag[:, :, _NATURAL_TO_ZIGZAG]


def _oracle_forward_coefficients(values: np.ndarray) -> np.ndarray:
    maximum = np.float32(65504.0)
    half_values = np.clip(np.asarray(values, dtype=np.float32), -maximum, maximum).astype(np.float16).astype(np.float32)
    magnitude = np.abs(half_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        nonlinear = np.where(
            magnitude <= np.float32(1.0),
            np.power(magnitude, np.float32(1.0 / 2.2)),
            np.log(magnitude) / np.float32(2.2) + np.float32(1.0),
        )
    nonlinear = np.copysign(nonlinear, half_values).astype(np.float16).astype(np.float32)
    red, green, blue = (nonlinear[..., index] for index in range(3))
    components = np.stack(
        (
            np.float32(0.2126) * red + np.float32(0.7152) * green + np.float32(0.0722) * blue,
            np.float32(-0.1146) * red - np.float32(0.3854) * green + np.float32(0.5) * blue,
            np.float32(0.5) * red - np.float32(0.4542) * green - np.float32(0.0458) * blue,
        )
    )
    sample = np.arange(8, dtype=np.float32)[:, None]
    frequency = np.arange(8, dtype=np.float32)[None, :]
    basis = np.float32(0.5) * np.cos(
        np.float32(np.pi) * frequency * (np.float32(2.0) * sample + np.float32(1.0)) / np.float32(16.0)
    )
    basis[:, 0] = np.float32(1.0 / np.sqrt(8.0))
    block_rows = values.shape[0] // 8
    block_columns = values.shape[1] // 8
    coefficient_planes = []
    for component in components:
        blocks = component.reshape(block_rows, 8, block_columns, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        coefficient_planes.append(np.matmul(np.matmul(basis.T, blocks), basis))
    return np.stack(coefficient_planes, axis=1).reshape(-1, 3, 64)


def _oracle_quantization_tables(dwa_level: float) -> tuple[np.ndarray, np.ndarray]:
    base_error = np.float32(dwa_level / 100000.0)
    return base_error * _JPEG_LUMINANCE / np.float32(10.0), base_error * _JPEG_CHROMA / np.float32(17.0)


def test_dwa_quantization_tables_and_gpu_selection_match_an_independent_half_oracle() -> None:
    """v1-exr-gpu-phase2 acceptance 18 and 20: coefficient errors and HALF selection follow the independent rule."""
    import cupy as cp

    luminance, chroma = exr_dwa._dwa_quantization_tables(45.0)
    jpeg_luminance = np.asarray(
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
    jpeg_chroma = np.asarray(
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
    base_error = np.float32(45.0 / 100000.0)
    np.testing.assert_array_equal(luminance, base_error * jpeg_luminance / np.float32(10.0))
    np.testing.assert_array_equal(chroma, base_error * jpeg_chroma / np.float32(17.0))

    coefficients = np.zeros((1, 64), dtype=np.float32)
    positions = (0, 1, 7, 19, 42, 63)
    coefficients[0, list(positions)] = np.asarray((0.33325, -0.8125, 3.141, -17.75, 0.0019, 128.5))
    actual = exr_dwa._quantize_dwa_half_gpu(cp.asarray(coefficients), cp.asarray(luminance)).get()[0]
    expected = coefficients.astype(np.float16).view(np.uint16)[0]
    for position in positions:
        expected[position] = _minimum_population_half(coefficients[0, position], float(luminance[position]))
    np.testing.assert_array_equal(actual, expected)


def test_dwa_quantizer_breaks_an_equal_population_and_distance_tie_by_magnitude() -> None:
    """v1-exr-gpu-phase2 acceptance 20: an exact distance tie selects the smaller-magnitude HALF candidate."""
    import cupy as cp

    lower = np.uint16(0x0002)
    source = np.uint16(0x0003)
    upper = np.uint16(0x0004)
    source_value = source.reshape(1).view(np.float16).astype(np.float32)[0]
    distance = source_value - lower.reshape(1).view(np.float16).astype(np.float32)[0]
    tolerance = np.nextafter(distance, np.float32(np.inf))
    coefficients = np.zeros((1, 64), dtype=np.float32)
    coefficients[0, 0] = source_value
    tolerances = np.full(64, tolerance, dtype=np.float32)

    actual = exr_dwa._quantize_dwa_half_gpu(cp.asarray(coefficients), cp.asarray(tolerances)).get()[0, 0]

    assert _half_population(np.asarray((lower, upper))).tolist() == [1, 1]
    assert source_value - lower.reshape(1).view(np.float16).astype(np.float32)[0] == (
        upper.reshape(1).view(np.float16).astype(np.float32)[0] - source_value
    )
    assert actual == lower
    assert actual == _minimum_population_half(float(source_value), float(tolerance))


def test_gpu_huffman_encoder_builds_a_histogram_codebook_and_round_trips_runs() -> None:
    """v1-exr-gpu-phase2 acceptance 18 and 24: write-side canonical Huffman and repeat runs decode exactly."""
    import cupy as cp

    symbols = np.asarray(
        (0x0000,) * 40 + (0xFF00,) * 12 + (0x3C00,) * 5 + (0xBC00,) * 2 + (0xFF03,),
        dtype=np.uint16,
    )
    stream = exr_harness._encode_dwa_huffman_gpu(cp.asarray(symbols)).get().tobytes()
    table = exr_container._parse_dwa_huffman_table(stream)
    decoded = exr_dwa._decode_dwa_huffman_host(stream, table, expected_count=symbols.size)
    lengths = dict(zip(range(table.minimum_symbol, table.maximum_symbol + 1), table.code_lengths, strict=True))

    np.testing.assert_array_equal(decoded, symbols)
    assert lengths[0x0000] <= lengths[0x3C00] <= lengths[0xBC00]
    assert table.maximum_symbol == int(symbols.max()) + 1
    assert table.data_bit_count < sum(lengths[int(symbol)] for symbol in symbols)


def test_gpu_huffman_batch_encoder_round_trips_independent_chunk_codebooks() -> None:
    """v1-exr-gpu-phase2 acceptance 17 and 23: AC codebooks and bitstreams are batched but chunk-independent."""
    import cupy as cp

    chunks = (
        np.asarray((0x0000,) * 40 + (0xFF00,) * 12 + (0x3C00,), dtype=np.uint16),
        np.asarray((0xBC00,) * 9 + (0xFF03,) * 4 + (0x0001, 0x0002), dtype=np.uint16),
        np.asarray((0x3555, 0x3556, 0x3557) * 7, dtype=np.uint16),
    )
    sizes = tuple(int(chunk.size) for chunk in chunks)
    offsets = tuple(np.cumsum((0, *sizes[:-1]), dtype=np.int64).tolist())
    symbols = cp.asarray(np.concatenate(chunks))
    chunk_ids = cp.repeat(cp.arange(len(chunks), dtype=cp.int32), cp.asarray(sizes, dtype=cp.int32))

    encoded, encoded_offsets, encoded_sizes, symbol_counts = exr_dwa._encode_dwa_huffman_chunks_gpu(
        symbols,
        chunk_ids,
        len(chunks),
    )
    encoded_host = encoded.get().tobytes()

    assert offsets == (0, sizes[0], sizes[0] + sizes[1])
    assert symbol_counts == sizes
    for expected, offset, size in zip(chunks, encoded_offsets, encoded_sizes, strict=True):
        stream = encoded_host[offset : offset + size]
        table = exr_container._parse_dwa_huffman_table(stream)
        actual = exr_dwa._decode_dwa_huffman_host(stream, table, expected_count=expected.size)
        np.testing.assert_array_equal(actual, expected)


def test_gpu_byte_rle_batch_encoder_round_trips_independent_chunks() -> None:
    """v1-exr-gpu-phase2 acceptance 17 and 23: RLE streams are encoded in parallel without crossing chunks."""
    import cupy as cp

    chunks = (
        bytes(range(127)) + b"repeat" * 23,
        b"\x00\xff" * 141,
        bytes(range(31)),
    )
    sizes = tuple(len(chunk) for chunk in chunks)
    offsets = tuple(np.cumsum((0, *sizes[:-1]), dtype=np.int64).tolist())
    raw = cp.asarray(np.frombuffer(b"".join(chunks), dtype=np.uint8))

    encoded, encoded_offsets, encoded_sizes = exr_dwa._encode_dwa_byte_rle_chunks_gpu(raw, offsets, sizes)
    encoded_host = encoded.get().tobytes()

    for chunk_index, (expected, offset, size) in enumerate(zip(chunks, encoded_offsets, encoded_sizes, strict=True)):
        actual = exr_dwa._decode_dwa_byte_rle_host(
            encoded_host[offset : offset + size],
            expected_size=len(expected),
            chunk_y=chunk_index,
        )
        assert actual == expected


def test_candidate_file_transmits_the_independent_host_oracle_coefficients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase2 acceptance 18 and 20: candidate AC/DC streams carry independently selected coefficients."""
    import cupy as cp

    y, x = np.mgrid[:32, :8]
    values = np.stack(
        (
            np.float32(0.125) + x / np.float32(19.0) + y / np.float32(73.0),
            np.float32(0.75) - x / np.float32(23.0) + y / np.float32(101.0),
            np.float32(-0.2) + x / np.float32(29.0) - y / np.float32(89.0),
        ),
        axis=2,
    ).astype(np.float32)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels="RGB")
    path = tmp_path / "candidate-coefficients-dwaa.exr"
    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "write"), "gpu")

    px.io.write_image(path, frame, compression="dwaa", dwa_level=45.0)

    actual = _candidate_file_coefficients(path)
    source = _oracle_forward_coefficients(values)
    luminance, chroma = _oracle_quantization_tables(45.0)
    expected = source.astype(np.float16).view(np.uint16)
    for block in range(source.shape[0]):
        for component, tolerances in enumerate((luminance, chroma, chroma)):
            for position, tolerance in enumerate(tolerances):
                expected[block, component, position] = _minimum_population_half(
                    float(source[block, component, position]), float(tolerance)
                )

    assert actual.shape == (4, 3, 64)
    np.testing.assert_array_equal(actual, expected)


def _write_frame(dtype: type[np.generic], *, height: int, width: int) -> tuple[px.core.Frame, np.ndarray]:
    import cupy as cp

    y, x = np.mgrid[:height, :width]
    linear = np.stack(
        (
            np.float32(-0.25) + x / np.float32(width + 3) + y / np.float32(height + 7),
            np.float32(0.125) + x / np.float32(width * 2 + 1),
            np.float32(1.5) - y / np.float32(height + 1),
            np.where((x + y) % 7, np.float32(1.0), np.float32(0.25)),
            np.float32(-3.0) + x * np.float32(0.5) + y * np.float32(0.125),
        ),
        axis=2,
    )
    if np.issubdtype(dtype, np.integer):
        maximum = np.float32(np.iinfo(dtype).max)
        stored = np.rint(np.clip(linear, 0.0, 1.0) * maximum).astype(dtype)
        expected = stored.astype(np.float32) * np.float32(1.0 / maximum)
    else:
        stored = linear.astype(dtype)
        expected = stored
    labels = ("beauty.R", "beauty.G", "beauty.B", "beauty.A", "depth.Z")
    frame = px.io.from_array(cp.asarray(stored), colorspace="ACES2065-1", gamma="linear", channels=labels)
    return frame, expected


def _eager_dwa_lossy_components(
    data: cp.ndarray,
    source_rows: cp.ndarray,
    input_indices: tuple[int, ...],
    padded_width: int,
) -> tuple[cp.ndarray, ...]:
    """Test-side rendition of the pre-fusion write transfer and color path."""
    width = int(data.shape[1])
    batched_data = data[source_rows]
    x_indices = exr_dwa._dwa_mirror_indices(width, padded_width)
    components: list[cp.ndarray] = []
    for input_index in input_indices:
        values = batched_data[..., input_index]
        if data.dtype.name == "uint8":
            values = values.astype(cp.float32) * np.float32(1.0 / 255.0)
        elif data.dtype.name == "uint16":
            values = values.astype(cp.float32) * np.float32(1.0 / 65535.0)
        elif data.dtype.name == "float32":
            maximum = np.float32(65504.0)
            values = cp.where(cp.isfinite(values), cp.clip(values, -maximum, maximum), values)
        half_values = cp.ascontiguousarray(values, dtype=cp.float16)
        linear = half_values.astype(cp.float32)
        magnitude = cp.abs(linear)
        nonlinear = cp.where(
            magnitude <= np.float32(1.0),
            cp.power(magnitude, np.float32(1.0 / 2.2)),
            cp.log(magnitude) / np.float32(2.2) + np.float32(1.0),
        )
        nonlinear = cp.copysign(nonlinear, linear)
        nonlinear = cp.where(cp.isfinite(linear), nonlinear, np.float32(0.0))
        components.append(nonlinear.astype(cp.float16).astype(cp.float32)[:, :, x_indices])
    if len(components) == 3:
        red, green, blue = components
        components = [
            np.float32(0.2126) * red + np.float32(0.7152) * green + np.float32(0.0722) * blue,
            np.float32(-0.1146) * red - np.float32(0.3854) * green + np.float32(0.5) * blue,
            np.float32(0.5) * red - np.float32(0.4542) * green - np.float32(0.0458) * blue,
        ]
    return tuple(components)


def _dwa_transfer_trial_frame(
    dtype: type[np.generic], *, height: int, width: int, channels: tuple[str, ...]
) -> px.core.Frame:
    y, x = np.mgrid[:height, :width]
    planes = (
        np.float32(-0.375) + x / np.float32(width + 3) + y / np.float32(height + 7),
        np.float32(0.125) + x / np.float32(width * 2 + 1) - y / np.float32(height + 11),
        np.float32(1.75) - x / np.float32(width + 5) + y / np.float32(height * 3 + 1),
        np.where((x + y) % 5, np.float32(1.0), np.float32(0.25)),
    )
    values = np.stack(planes[: len(channels)], axis=2)
    if np.issubdtype(dtype, np.integer):
        maximum = np.float32(np.iinfo(dtype).max)
        values = np.rint(np.clip(values, 0.0, 1.0) * maximum).astype(dtype)
    else:
        values = values.astype(dtype)
    return px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels=channels)


@pytest.mark.parametrize(
    ("compression", "dwa_level", "dtype", "channels"),
    (
        ("dwaa", 45.0, np.uint8, ("R", "G", "B")),
        ("dwab", 45.0, np.float32, ("R", "G", "B", "A")),
        ("dwaa", 10.0, np.uint16, ("Y",)),
        ("dwab", 100.0, np.float16, ("R", "G", "B")),
        ("dwaa", 23.5, np.float32, ("R", "G", "B", "A")),
        ("dwab", 23.5, np.uint8, ("Y",)),
    ),
)
def test_dwa_transfer_fusion_preserves_encode_bitstream_characterization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    dwa_level: float,
    dtype: type[np.generic],
    channels: tuple[str, ...],
) -> None:
    """characterization: issue #1 RawKernel trial acceptance 1 and 5 freezes the eager DWA bytes until the
    transfer/color contract changes; Phase 2 wire and cross-decode tests independently establish correctness.
    """
    fused_helper = exr_dwa._prepare_dwa_lossy_components_gpu
    height = 33 if compression == "dwaa" else 257
    frame = _dwa_transfer_trial_frame(dtype, height=height, width=9, channels=channels)
    candidate_path = tmp_path / "candidate.exr"
    eager_path = tmp_path / "eager.exr"

    px.io.write_image(candidate_path, frame, compression=compression, dwa_level=dwa_level)
    monkeypatch.setattr(exr_dwa, "_prepare_dwa_lossy_components_gpu", _eager_dwa_lossy_components)
    px.io.write_image(eager_path, frame, compression=compression, dwa_level=dwa_level)

    assert fused_helper is not _eager_dwa_lossy_components
    assert candidate_path.read_bytes() == eager_path.read_bytes()


def test_hybrid_dwaa_writer_clamps_finite_float_samples_to_the_half_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase2 acceptance 18 and 21: finite FLOAT overflow clamps before lossy HALF conversion."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    values = np.empty((32, 64, 1), dtype=np.float32)
    values[:, :32, 0] = np.float32(70_000.0)
    values[:, 32:, 0] = np.float32(-70_000.0)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels=("R",))
    candidate_path = tmp_path / "hybrid-half-range-dwaa.exr"
    reference_path = tmp_path / "reference-half-range-dwaa.exr"
    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "write"), "gpu")

    px.io.write_image(candidate_path, frame, compression="dwaa", dwa_level=45.0, dtype="float32")
    OpenEXR.File(
        {"compression": OpenEXR.DWAA_COMPRESSION, "dwaCompressionLevel": 45.0},
        {"R": values[..., 0]},
    ).write(str(reference_path))

    container = exr_container._parse_exr_container(candidate_path)
    actual = np.asarray(OpenEXR.File(str(candidate_path), separate_channels=True).channels()["R"].pixels)
    expected = np.asarray(OpenEXR.File(str(reference_path), separate_channels=True).channels()["R"].pixels)
    assert any(not chunk.raw_stored for chunk in container.chunks)
    assert np.isfinite(actual).all()
    assert np.all(actual[:, :32] > 0.0)
    assert np.all(actual[:, 32:] < 0.0)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float16, np.float32))
def test_hybrid_dwa_writer_cross_backend_round_trips_every_storage_and_channel_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: str,
    dtype: type[np.generic],
) -> None:
    """v1-exr-gpu-phase2 acceptance 17-23: OpenEXR reads hybrid DWA lossy, RLE, and UNKNOWN channels."""
    from openexr_dev_oracle import OpenEXR

    height = 33 if compression == "dwaa" else 257
    frame, expected = _write_frame(dtype, height=height, width=17)
    path = tmp_path / f"hybrid-{compression}-{np.dtype(dtype).name}.exr"
    calls = 0
    original = io._write_exr_dwa_gpu

    def spy(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setitem(io._EXR_ROUTING, (compression, "write"), "gpu")
    monkeypatch.setattr(io, "_write_exr_dwa_gpu", spy)

    output_dtype = "float16" if np.dtype(dtype) == np.dtype(np.float16) else "float32"
    px.io.write_image(path, frame, compression=compression, dwa_level=23.5, dtype=output_dtype)

    container = exr_container._parse_exr_container(path)
    reference = OpenEXR.File(str(path), separate_channels=True)
    channels = reference.channels()
    assert calls == 1
    assert container.dwa_eligible is True
    assert any(not chunk.raw_stored for chunk in container.chunks)
    assert float(reference.header()["dwaCompressionLevel"]) == 23.5
    assert reference.header()["acesImageContainerFlag"] == 1
    assert (
        reference.header()["compression"]
        == {
            "dwaa": OpenEXR.DWAA_COMPRESSION,
            "dwab": OpenEXR.DWAB_COMPRESSION,
        }[compression]
    )
    np.testing.assert_array_equal(np.asarray(channels["beauty.A"].pixels), expected[..., 3])
    np.testing.assert_array_equal(np.asarray(channels["depth.Z"].pixels), expected[..., 4])
    for label in ("beauty.R", "beauty.G", "beauty.B"):
        decoded = np.asarray(channels[label].pixels)
        assert decoded.shape == (height, 17)
        assert np.isfinite(decoded).all()


def test_hybrid_dwaa_writer_emits_partial_compressed_and_raw_chunks_with_real_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase2 acceptance 19: chunk geometry, raw fallback, leaders, and offsets describe real payloads."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    generator = np.random.default_rng(20260808)
    values = np.zeros((65, 96, 1), dtype=np.float32)
    values[32:64, :, 0] = generator.integers(0, 1 << 32, size=(32, 96), dtype=np.uint32).view(np.float32)
    values[64, :, 0] = np.float32(0.5)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels=("depth.Z",))
    path = tmp_path / "hybrid-mixed-dwaa.exr"
    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "write"), "gpu")

    px.io.write_image(path, frame, compression="dwaa", dtype="float32")

    container = exr_container._parse_exr_container(path)
    assert tuple(chunk.row_count for chunk in container.chunks) == (32, 32, 1)
    assert any(chunk.raw_stored for chunk in container.chunks)
    assert any(not chunk.raw_stored for chunk in container.chunks)
    assert container.offset_table == tuple(chunk.payload_start - 8 for chunk in container.chunks)
    for chunk in container.chunks:
        assert chunk.payload_end - chunk.payload_start == chunk.packed_size
        if not chunk.raw_stored:
            assert chunk.dwa is not None and chunk.dwa.leader is not None
            leader = chunk.dwa.leader
            assert leader.version == 2
            assert leader.unknown_uncompressed_size == chunk.expected_size
            payload = container.data[chunk.dwa.unknown_span.start : chunk.dwa.unknown_span.end]
            assert len(zlib.decompress(payload)) == leader.unknown_uncompressed_size
    decoded = np.asarray(OpenEXR.File(str(path), separate_channels=True).channels()["depth.Z"].pixels)
    np.testing.assert_array_equal(decoded, values[..., 0])


@pytest.mark.parametrize("compression", ("dwaa", "dwab"))
def test_hybrid_dwa_writer_emits_independent_y_chroma_alpha_and_unknown_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: str
) -> None:
    """v1-exr-gpu-phase2 acceptance 18, 21, and 22: Y/BY/RY, A, and UNKNOWN rules interoperate."""
    import cupy as cp
    from openexr_dev_oracle import OpenEXR

    labels = ("luma.Y", "luma.BY", "luma.RY", "matte.A", "other.Q")
    y, x = np.mgrid[:19, :13]
    values = np.stack(tuple((x + y + index) / np.float32(37 + index) for index in range(len(labels))), axis=2).astype(
        np.float16
    )
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels=labels)
    path = tmp_path / f"hybrid-classes-{compression}.exr"
    monkeypatch.setitem(io._EXR_ROUTING, (compression, "write"), "gpu")

    px.io.write_image(path, frame, compression=compression, dwa_level=45.0)

    decoded = OpenEXR.File(str(path), separate_channels=True).channels()
    assert set(decoded) == set(labels)
    np.testing.assert_array_equal(np.asarray(decoded["matte.A"].pixels), values[..., 3])
    np.testing.assert_array_equal(np.asarray(decoded["other.Q"].pixels), values[..., 4])
    for label in labels[:3]:
        assert np.isfinite(np.asarray(decoded[label].pixels)).all()


def test_hybrid_dwa_write_helper_graph_limits_every_device_to_host_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase2 acceptance 17 and 23: DWA transfers control data and one final payload, not pixels."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((9, 16, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    path = tmp_path / "dwa-transfer.exr"
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    px.io.write_image(path, frame, compression="dwaa")

    container = exr_container._parse_exr_container(path)
    payload_bytes = sum(chunk.packed_size for chunk in container.chunks)
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=64, max_total_nbytes=600_000, max_shape_elements=65_537
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=11, max_total_nbytes=524_521, max_shape_elements=65_536
    )
    payload_transfers = [
        transfer
        for transfer in transfers
        if transfer.direction == "d2h"
        and transfer.dtype == "uint8"
        and transfer.shape == (payload_bytes,)
        and transfer.nbytes == payload_bytes
    ]
    assert len(payload_transfers) == 1
    assert not [
        transfer
        for transfer in transfers
        if transfer.direction == "d2h"
        and transfer.shape == frame.data.shape
        and transfer.dtype == frame.data.dtype.name
    ]


def test_unexpected_hybrid_dwa_write_failure_is_actionable_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1-exr-gpu-phase2 acceptance 24-25: an eligible write defect keeps its cause and never enters fallback."""
    import cupy as cp

    frame = px.io.from_array(
        cp.zeros((9, 16, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    path = tmp_path / "hybrid-defect.exr"

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic DWA data-plane failure")

    monkeypatch.setitem(io._EXR_ROUTING, ("dwaa", "write"), "gpu")
    monkeypatch.setattr("pixtreme._io.formats.exr.codec_dwa._encode_dwa_huffman_chunks_gpu", fail)

    with pytest.raises(RuntimeError, match=r"why=.*DWA.*what=.*how=") as error:
        px.io.write_image(path, frame, compression="dwaa")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "synthetic DWA data-plane failure" in str(error.value.__cause__)
    assert not path.exists()


def test_hybrid_dwa_header_uses_the_resolved_default_level() -> None:
    """v1-exr-gpu-phase2 acceptance 19: the hybrid header stores resolved default dwaCompressionLevel as float32."""
    encoded_channels = exr_packing._encode_exr_output_channels(("R",))
    header = exr_packing._exr_write_header(
        width=1,
        height=1,
        encoded_channels=encoded_channels,
        pixel_type=1,
        compression="dwaa",
        chromaticities=(0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767),
        aces_image_container=True,
        dwa_level=45.0,
    )

    marker = b"dwaCompressionLevel\x00float\x00"
    start = header.index(marker) + len(marker)
    size = struct.unpack_from("<I", header, start)[0]
    assert size == 4
    assert struct.unpack_from("<f", header, start + 4)[0] == 45.0
