"""Specification tests for the Phase 4 PIZ read data planes."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TypeAlias

import cupy as cp
import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme._io.formats.exr.codec_piz as exr_piz
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header

BitField: TypeAlias = tuple[int, int]


def _pack_bits(fields: list[BitField]) -> tuple[bytes, int]:
    """Independent MSB-first field packer used only by the PIZ wire oracle."""
    bits: list[int] = []
    for value, width in fields:
        assert 0 <= value < 1 << width
        bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))
    output = bytearray((len(bits) + 7) // 8)
    for offset, bit in enumerate(bits):
        output[offset // 8] |= bit << (7 - offset % 8)
    return bytes(output), len(bits)


def _pack_lengths(lengths: tuple[int, ...]) -> bytes:
    fields: list[BitField] = []
    cursor = 0
    while cursor < len(lengths):
        if lengths[cursor]:
            fields.append((lengths[cursor], 6))
            cursor += 1
            continue
        end = cursor + 1
        while end < len(lengths) and lengths[end] == 0:
            end += 1
        remaining = end - cursor
        while remaining:
            run = min(remaining, 261)
            if run == 1:
                fields.append((0, 6))
            elif run <= 5:
                fields.append((run + 57, 6))
            else:
                fields.extend(((63, 6), (run - 6, 8)))
            remaining -= run
        cursor = end
    return _pack_bits(fields)[0]


def _canonical_codes(lengths: tuple[int, ...], minimum_symbol: int = 0) -> dict[int, tuple[int, int]]:
    counts = [0] * 59
    for length in lengths:
        counts[length] += bool(length)
    bases = [0] * 59
    code = 0
    for length in range(58, 0, -1):
        bases[length] = code
        code = (code + counts[length]) >> 1
    result: dict[int, tuple[int, int]] = {}
    for offset, length in enumerate(lengths):
        if length:
            result[minimum_symbol + offset] = (bases[length], length)
            bases[length] += 1
    return result


def _huffman_stream(
    lengths: tuple[int, ...],
    data_fields: list[BitField],
    *,
    minimum_symbol: int = 0,
    declared_table_bytes: int | None = None,
    declared_data_bits: int | None = None,
    reserved: int = 0,
    table_padding_or: int = 0,
    data_padding_or: int = 0,
    trailing: bytes = b"",
) -> bytes:
    table = bytearray(_pack_lengths(lengths))
    if table:
        table[-1] |= table_padding_or
    data, data_bits = _pack_bits(data_fields)
    encoded = bytearray(data)
    if encoded:
        encoded[-1] |= data_padding_or
    return (
        struct.pack(
            "<IIIII",
            minimum_symbol,
            minimum_symbol + len(lengths) - 1,
            len(table) if declared_table_bytes is None else declared_table_bytes,
            data_bits if declared_data_bits is None else declared_data_bits,
            reserved,
        )
        + table
        + encoded
        + trailing
    )


def _literal_fields(lengths: tuple[int, ...], symbols: tuple[int, ...], *, minimum_symbol: int = 0) -> list[BitField]:
    codes = _canonical_codes(lengths, minimum_symbol)
    return [codes[symbol] for symbol in symbols]


def _decode_huffman_lane(
    lane: str,
    stream: bytes,
    table: exr_container._PizHuffmanTable,
    *,
    expected_count: int,
) -> np.ndarray:
    if lane == "host":
        return exr_container._decode_piz_huffman_host(stream, table, expected_count=expected_count)
    return exr_piz._decode_piz_huffman_gpu(
        cp.asarray(np.frombuffer(stream, dtype=np.uint8)),
        data_offsets=(table.data_span.start,),
        tables=(table,),
        output_counts=(expected_count,),
        record_labels=(0x50495A,),
    ).get()


def _signed_word(value: int) -> int:
    return value - 65536 if value & 0x8000 else value


def _forward_pair14(a_word: int, b_word: int) -> tuple[int, int]:
    a = _signed_word(a_word)
    b = _signed_word(b_word)
    return ((a + b) >> 1) & 0xFFFF, (a - b) & 0xFFFF


def _forward_pair16(a: int, b: int) -> tuple[int, int]:
    offset_a = (a + 32768) & 0xFFFF
    midpoint = (offset_a + b) >> 1
    difference = offset_a - b
    if difference < 0:
        midpoint = (midpoint + 32768) & 0xFFFF
    return midpoint, difference & 0xFFFF


def _forward_wavelet_oracle(
    words: np.ndarray,
    *,
    nx: int,
    ny: int,
    word_stride: int,
    word_slice: int,
    w14: bool,
) -> np.ndarray:
    """Independent scalar rendition of the pinned OpenEXR forward hierarchy."""
    output = np.asarray(words, dtype=np.uint16).copy()
    if min(nx, ny) < 2:
        return output
    pair = _forward_pair14 if w14 else _forward_pair16
    x_stride = word_stride
    y_stride = word_stride * nx
    base = word_slice
    p = 1
    p2 = 2
    while p2 <= min(nx, ny):
        for y in range(0, ny - p2 + 1, p2):
            for x in range(0, nx - p2 + 1, p2):
                i00 = base + y * y_stride + x * x_stride
                i01 = i00 + p * x_stride
                i10 = i00 + p * y_stride
                i11 = i10 + p * x_stride
                low0, high0 = pair(int(output[i00]), int(output[i01]))
                low1, high1 = pair(int(output[i10]), int(output[i11]))
                output[i00], output[i10] = pair(low0, low1)
                output[i01], output[i11] = pair(high0, high1)
        if nx & p:
            x = (nx // p2) * p2
            for y in range(0, ny - p2 + 1, p2):
                first = base + y * y_stride + x * x_stride
                second = first + p * y_stride
                output[first], output[second] = pair(int(output[first]), int(output[second]))
        if ny & p:
            y = (ny // p2) * p2
            for x in range(0, nx - p2 + 1, p2):
                first = base + y * y_stride + x * x_stride
                second = first + p * x_stride
                output[first], output[second] = pair(int(output[first]), int(output[second]))
        p = p2
        p2 *= 2
    return output


def _bitmap_slice(values: set[int], *, include_zero_bit: bool = False) -> tuple[int, int, bytes]:
    bitmap = bytearray(8192)
    for value in values:
        bitmap[value >> 3] |= 1 << (value & 7)
    if include_zero_bit:
        bitmap[0] |= 1
    else:
        bitmap[0] &= 0xFE
    nonzero = [index for index, value in enumerate(bitmap) if value]
    if not nonzero:
        return 8191, 0, b""
    minimum, maximum = nonzero[0], nonzero[-1]
    return minimum, maximum, bytes(bitmap[minimum : maximum + 1])


@pytest.mark.parametrize(
    ("values", "include_zero_bit"),
    (
        (set(), False),
        ({1}, False),
        ({7, 8, 15, 16}, False),
        ({1, 9, 65535}, False),
        ({0, 3, 11}, True),
    ),
)
def test_piz_reverse_lut_matches_absolute_unsigned_rank(
    values: set[int],
    include_zero_bit: bool,
) -> None:
    """v1-exr-gpu-phase4 acceptance 9 and 10: bitmap rank is LSB-first with one implicit zero."""
    minimum, maximum, packed = _bitmap_slice(values, include_zero_bit=include_zero_bit)

    reverse, max_value = exr_container._piz_reverse_lut(minimum, maximum, packed)

    expected = np.asarray((0, *sorted(value for value in values if value)), dtype=np.uint16)
    np.testing.assert_array_equal(reverse[: expected.size], expected)
    assert max_value == expected.size - 1
    assert np.count_nonzero(reverse[expected.size :]) == 0


@pytest.mark.parametrize("compact_max", (16383, 16384))
def test_piz_compact_max_fixes_wavelet_mode_at_the_flip_boundary(compact_max: int) -> None:
    """v1-exr-gpu-phase4 acceptance 10 and 13: compact alphabet size, not original values, selects w14/w16."""
    minimum, maximum, packed = _bitmap_slice(set(range(1, compact_max + 1)))

    _reverse, max_value = exr_container._piz_reverse_lut(minimum, maximum, packed)

    assert max_value == compact_max
    assert exr_container._piz_uses_w14(max_value) is (compact_max < 16384)


@pytest.mark.parametrize("shape", ((2, 2), (4, 3), (3, 4), (5, 5), (7, 31), (8, 1), (1, 8)))
@pytest.mark.parametrize("w14", (True, False))
def test_piz_inverse_wavelet_host_and_gpu_match_independent_all_level_oracle(
    shape: tuple[int, int],
    w14: bool,
) -> None:
    """v1-exr-gpu-phase4 acceptance 14-17: inverse levels, odd edges, corner, wrap, and no-op are exact."""
    nx, ny = shape
    grid = np.arange(nx * ny, dtype=np.uint32).reshape(ny, nx)
    first = ((grid * 811 + 13) % (15000 if w14 else 65536)).astype(np.uint16)
    second = ((65535 - grid * 1237) % (14000 if w14 else 65536)).astype(np.uint16)
    original = np.stack((first, second), axis=2).reshape(-1)
    encoded = original.copy()
    for word_slice in range(2):
        encoded = _forward_wavelet_oracle(
            encoded,
            nx=nx,
            ny=ny,
            word_stride=2,
            word_slice=word_slice,
            w14=w14,
        )

    host = encoded.copy()
    device = cp.asarray(encoded)
    for word_slice in range(2):
        exr_container._piz_inverse_wavelet_host(
            host,
            nx=nx,
            ny=ny,
            word_stride=2,
            word_slice=word_slice,
            max_value=12000 if w14 else 20000,
        )
        exr_harness._piz_inverse_wavelet_gpu(
            device,
            nx=nx,
            ny=ny,
            word_stride=2,
            word_slice=word_slice,
            max_value=12000 if w14 else 20000,
        )

    np.testing.assert_array_equal(host, original)
    np.testing.assert_array_equal(device.get(), original)


@pytest.mark.parametrize("zero_gap", (1, 2, 5, 6, 261, 262))
def test_piz_huffman_length_table_expands_every_zero_run_boundary(zero_gap: int) -> None:
    """v1-exr-gpu-phase4 acceptance 21: all short, long, and split zero-run tokens end at iM+1."""
    lengths = (1,) + (0,) * zero_gap + (1,)
    stream = _huffman_stream(lengths, _literal_fields(lengths, (0,)))

    table = exr_container._parse_piz_huffman_table(stream)

    assert table.code_lengths == lengths
    assert table.table_span.start == 20
    assert table.data_span.start == 20 + len(_pack_lengths(lengths))


@pytest.mark.parametrize("lane", ("host", "gpu"))
def test_piz_huffman_decodes_a_58_bit_code_in_both_lanes(lane: str) -> None:
    """v1-exr-gpu-phase4 acceptance 21 and 23: the full code-length domain decodes in both lanes."""
    lengths = tuple(range(1, 58)) + (58, 58)
    expected_symbol = 57
    stream = _huffman_stream(lengths, _literal_fields(lengths, (expected_symbol,)))
    table = exr_container._parse_piz_huffman_table(stream)

    actual = _decode_huffman_lane(lane, stream, table, expected_count=1)

    np.testing.assert_array_equal(actual, np.asarray((expected_symbol,), dtype=np.uint16))


@pytest.mark.parametrize("lane", ("host", "gpu"))
def test_piz_huffman_decodes_largest_actual_symbol_and_pseudo_repeat_boundary(lane: str) -> None:
    """v1-exr-gpu-phase4 acceptance 21, 23, and 34: im=65535/iM=65536 is literal then repeat."""
    minimum_symbol = 0xFFFF
    lengths = (1, 1)
    codes = _canonical_codes(lengths, minimum_symbol)
    stream = _huffman_stream(
        lengths,
        [codes[0xFFFF], codes[0x10000], (1, 8)],
        minimum_symbol=minimum_symbol,
    )
    table = exr_container._parse_piz_huffman_table(stream)

    actual = _decode_huffman_lane(lane, stream, table, expected_count=2)

    np.testing.assert_array_equal(actual, np.full(2, 0xFFFF, dtype=np.uint16))


def test_piz_huffman_previous_symbol_repeat_accepts_the_maximum_group() -> None:
    """v1-exr-gpu-phase4 acceptance 23: pseudo plus u8 255 expands one literal to 256 occurrences."""
    lengths = (1, 1)
    codes = _canonical_codes(lengths)
    stream = _huffman_stream(lengths, [codes[0], codes[1], (255, 8)])
    table = exr_container._parse_piz_huffman_table(stream)

    host = exr_container._decode_piz_huffman_host(stream, table, expected_count=256)
    device = exr_piz._decode_piz_huffman_gpu(
        cp.asarray(np.frombuffer(stream, dtype=np.uint8)),
        data_offsets=(table.data_span.start,),
        tables=(table,),
        output_counts=(256,),
        record_labels=(11,),
    )

    np.testing.assert_array_equal(host, np.zeros(256, dtype=np.uint16))
    np.testing.assert_array_equal(device.get(), host)


@pytest.mark.parametrize("lane", ("host", "gpu"))
@pytest.mark.parametrize("noncanonical", ("redundant_fields", "trailing_bytes", "extra_declared_data"))
def test_piz_huffman_permissive_read_ignores_safe_noncanonical_fields(lane: str, noncanonical: str) -> None:
    """v1-exr-gpu-phase4 acceptance 24 and 25: both lanes accept safe redundant fields and trailing data."""
    lengths = (1, 1)
    codes = _canonical_codes(lengths)
    fields = [codes[0]]
    options: dict[str, int | bytes] = {}
    if noncanonical == "redundant_fields":
        options = {
            "declared_table_bytes": 0x7FFFFFFF,
            "reserved": 0xA5A5A5A5,
            "table_padding_or": 0x03,
            "data_padding_or": 0x3F,
        }
    elif noncanonical == "trailing_bytes":
        options = {"trailing": b"trailing"}
    else:
        fields.append(codes[0])
    stream = _huffman_stream(lengths, fields, **options)

    table = exr_container._parse_piz_huffman_table(stream)
    actual = _decode_huffman_lane(lane, stream, table, expected_count=1)

    if noncanonical == "redundant_fields":
        assert table.declared_table_byte_count == 0x7FFFFFFF
        assert table.reserved == 0xA5A5A5A5
    np.testing.assert_array_equal(actual, np.asarray((0,), dtype=np.uint16))


@pytest.mark.parametrize(
    ("lengths", "data_fields", "expected_count"),
    (
        pytest.param((1, 1), [(1, 1), (1, 1)], 1, id="pseudo-before-literal"),
        pytest.param((1, 1), [(0, 1), (1, 1)], 2, id="truncated-count"),
        pytest.param((1, 1), [(0, 1), (1, 1), (255, 8)], 2, id="repeat-overflow"),
        pytest.param((1, 1), [(0, 1)], 2, id="word-count-underflow"),
        pytest.param((2, 2), [(1, 1)], 1, id="invalid-prefix"),
    ),
)
@pytest.mark.parametrize("lane", ("host", "gpu"))
def test_piz_huffman_rejects_unsafe_repeat_and_word_count_boundaries(
    lane: str,
    lengths: tuple[int, ...],
    data_fields: list[BitField],
    expected_count: int,
) -> None:
    """v1-exr-gpu-phase4 acceptance 21, 23, 25, and 35: both lanes reject unsafe prefixes and repeats."""
    stream = _huffman_stream(lengths, data_fields)
    table = exr_container._parse_piz_huffman_table(stream)

    with pytest.raises(RuntimeError, match=r"why=.*what=.*how="):
        _decode_huffman_lane(lane, stream, table, expected_count=expected_count)


def test_piz_huffman_rejects_oversubscribed_canonical_lengths() -> None:
    """v1-exr-gpu-phase4 acceptance 21 and 35: canonical assignments must remain prefix-free."""
    stream = _huffman_stream((1, 1, 1), [])

    with pytest.raises(RuntimeError, match=r"why=.*oversubscribed.*what=.*how="):
        exr_container._parse_piz_huffman_table(stream)


def test_piz_huffman_rejects_equal_actual_and_pseudo_symbol_bounds() -> None:
    """v1-exr-gpu-phase4 acceptance 21 and 35: im=iM cannot define both an actual and its pseudo symbol."""
    lengths = (1,)
    stream = _huffman_stream(
        lengths,
        _literal_fields(lengths, (0xFFFF,), minimum_symbol=0xFFFF),
        minimum_symbol=0xFFFF,
    )

    with pytest.raises(RuntimeError, match=r"why=.*actual symbol.*pseudo-symbol.*what=.*how="):
        exr_container._parse_piz_huffman_table(stream)


@pytest.mark.parametrize("zero_gap", (2, 6, 261))
def test_piz_huffman_rejects_length_run_overshoot_and_truncation(zero_gap: int) -> None:
    """v1-exr-gpu-phase4 acceptance 21 and 35: a table cannot cross iM+1 or end inside a token."""
    token = 59 if zero_gap == 2 else 63
    fields: list[BitField] = [(token, 6)]
    if token == 63:
        fields.append((zero_gap - 6, 8))
    packed, _ = _pack_bits(fields)
    stream = struct.pack("<IIIII", 0, 0, len(packed), 0, 0) + packed

    with pytest.raises(RuntimeError, match=r"why=.*table.*what=.*how="):
        exr_container._parse_piz_huffman_table(stream)

    truncated = struct.pack("<IIIII", 0, 1, 1, 0, 0) + b"\xfc"
    with pytest.raises(RuntimeError, match=r"why=.*truncated.*what=.*how="):
        exr_container._parse_piz_huffman_table(truncated)


def _reference_piz_file(path: Path, *, line_order: object) -> dict[str, np.ndarray]:
    from openexr_dev_oracle import OpenEXR

    height, width = 63, 96
    y, x = np.mgrid[:height, :width]
    half = ((x % 13) * np.float32(0.125) - (y % 7) * np.float32(0.25)).astype(np.float16)
    float_bits = np.resize(
        np.asarray(
            (0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x7F800000, 0xFF800000, 0x7FC12345, 0xBF800000),
            dtype=np.uint32,
        ),
        (height, width),
    )
    single = float_bits.view(np.float32)
    unsigned = (x.astype(np.uint32) * np.uint32(0x9E3779B1) + y.astype(np.uint32)).astype(np.uint32)
    zero = np.zeros((height, width), dtype=np.float16)
    origin = (-5, 11)
    maximum = (origin[0] + width - 1, origin[1] + height - 1)
    channels = {"beauty.R": half, "depth.Z": single, "mask.U": unsigned, "zero.H": zero}
    write_line_order = OpenEXR.INCREASING_Y if line_order == OpenEXR.RANDOM_Y else line_order
    header = {
        "compression": OpenEXR.PIZ_COMPRESSION,
        "dataWindow": (np.asarray(origin, dtype=np.int32), np.asarray(maximum, dtype=np.int32)),
        "displayWindow": (np.asarray(origin, dtype=np.int32), np.asarray(maximum, dtype=np.int32)),
        "lineOrder": write_line_order,
    }
    OpenEXR.File(header, dict(channels)).write(str(path))
    if line_order == OpenEXR.RANDOM_Y:
        payload = bytearray(path.read_bytes())
        marker = b"lineOrder\x00lineOrder\x00"
        value_offset = payload.index(marker) + len(marker) + 4
        assert payload[value_offset] == int(OpenEXR.INCREASING_Y)
        payload[value_offset] = int(OpenEXR.RANDOM_Y)
        path.write_bytes(payload)
    reference = OpenEXR.File(str(path), separate_channels=True)
    decoded = {name: np.asarray(reference.channels()[name].pixels) for name in channels}
    for name, expected in channels.items():
        assert decoded[name].dtype == expected.dtype
        np.testing.assert_array_equal(
            decoded[name].view(f"u{expected.dtype.itemsize}"), expected.view(f"u{expected.dtype.itemsize}")
        )
    return decoded


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
@pytest.mark.parametrize("line_order_name", ("INCREASING_Y", "DECREASING_Y", "RANDOM_Y"))
def test_openexr_piz_writer_cross_decodes_in_both_internal_read_lanes(
    tmp_path: Path,
    backend: str,
    line_order_name: str,
) -> None:
    """v1-exr-gpu-phase4 acceptance 27, 30-32, and 34: reference PIZ samples survive every read lane bit-exactly."""
    from openexr_dev_oracle import OpenEXR

    path = tmp_path / f"reference-piz-{backend}-{line_order_name}.exr"
    reference = _reference_piz_file(path, line_order=getattr(OpenEXR, line_order_name))
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    assert container.piz_eligible is True
    assert container.data_window[:2] == (-5, 11)
    assert tuple(chunk.row_count for chunk in container.chunks) == (32, 31)
    assert all(not chunk.raw_stored for chunk in container.chunks)

    for channel, dtype in (("beauty.R", "float16"), ("depth.Z", "float32"), ("zero.H", "float16")):
        actual = exr_harness._read_exr_pixels_with_backend(
            path,
            container,
            header,
            [(0, channel, channel)],
            unchanged=True,
            backend=backend,
        ).get()[..., 0]
        expected = reference[channel]
        assert actual.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(
            actual.view(f"u{actual.dtype.itemsize}"), expected.view(f"u{expected.dtype.itemsize}")
        )

    promoted = exr_harness._read_exr_pixels_with_backend(
        path,
        container,
        header,
        [(0, "beauty.R", "R"), (0, "depth.Z", "Z")],
        unchanged=False,
        backend=backend,
    ).get()
    np.testing.assert_array_equal(promoted[..., 0], reference["beauty.R"].astype(np.float32))
    np.testing.assert_array_equal(promoted[..., 1].view(np.uint32), reference["depth.Z"].view(np.uint32))

    internal_read = io._read_exr_gpu if backend == "gpu" else io._read_exr_custom_cpu
    uint_values = internal_read(container, ["mask.U"], output_dtype="float32").get()[..., 0]
    np.testing.assert_array_equal(uint_values, reference["mask.U"].astype(np.float32))


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_piz_all_zero_bitmap_and_partial_one_row_chunk_decode(backend: str, tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 9, 11, 16, 23, and 32: empty bitmap and ny=1 decode as normal chunks."""
    from openexr_dev_oracle import OpenEXR

    path = tmp_path / f"all-zero-{backend}.exr"
    expected = np.zeros((33, 64), dtype=np.float16)
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, {"H": expected}).write(str(path))
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    assert tuple(chunk.row_count for chunk in container.chunks) == (32, 1)
    assert all(chunk.piz is not None and chunk.piz.bitmap_range == (8191, 0) for chunk in container.chunks)

    actual = exr_harness._read_exr_pixels_with_backend(
        path,
        container,
        header,
        [(0, "H", "H")],
        unchanged=True,
        backend=backend,
    ).get()[..., 0]

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_piz_read_combines_compressed_and_raw_chunks_without_changing_word_bits(backend: str, tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 11, 27, 30, 32, and 34: compressed and raw chunks share one final Frame."""
    from openexr_dev_oracle import OpenEXR

    height, width = 33, 128
    rng = np.random.default_rng(0x50495A)
    half = np.zeros((height, width), dtype=np.float16)
    single = np.zeros((height, width), dtype=np.float32)
    unsigned = np.zeros((height, width), dtype=np.uint32)
    half[-1] = rng.integers(0, 1 << 16, width, dtype=np.uint16).view(np.float16)
    single[-1] = rng.integers(0, 1 << 32, width, dtype=np.uint32).view(np.float32)
    unsigned[-1] = rng.integers(0, 1 << 32, width, dtype=np.uint32)
    path = tmp_path / f"mixed-storage-{backend}.exr"
    OpenEXR.File(
        {"compression": OpenEXR.PIZ_COMPRESSION},
        {"H": half, "F": single, "U": unsigned},
    ).write(str(path))
    reference = OpenEXR.File(str(path), separate_channels=True)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    assert tuple(chunk.raw_stored for chunk in container.chunks) == (False, True)

    for channel, expected in (
        ("H", np.asarray(reference.channels()["H"].pixels)),
        ("F", np.asarray(reference.channels()["F"].pixels)),
    ):
        actual = exr_harness._read_exr_pixels_with_backend(
            path,
            container,
            header,
            [(0, channel, channel)],
            unchanged=True,
            backend=backend,
        ).get()[..., 0]
        np.testing.assert_array_equal(
            actual.view(f"u{actual.dtype.itemsize}"), expected.view(f"u{expected.dtype.itemsize}")
        )


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_piz_zero_byte_outer_raw_chunk_materializes_zero_words(backend: str, tmp_path: Path) -> None:
    """v1-exr-gpu-phase4 acceptance 7, 11, 27, and 30: a zero-byte raw chunk bypasses every PIZ transform."""
    from openexr_dev_oracle import OpenEXR

    path = tmp_path / f"zero-byte-{backend}.exr"
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, {"H": np.zeros((1, 16), dtype=np.float16)}).write(str(path))
    original = exr_container._parse_exr_container(path)
    chunk = original.chunks[0]
    payload = bytearray(path.read_bytes()[: chunk.payload_start])
    struct.pack_into("<i", payload, chunk.payload_start - 4, 0)
    path.write_bytes(payload)
    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    assert container.chunks[0].raw_stored is True
    assert container.chunks[0].packed_size == 0

    actual = exr_harness._read_exr_pixels_with_backend(
        path,
        container,
        header,
        [(0, "H", "H")],
        unchanged=True,
        backend=backend,
    ).get()[..., 0]

    np.testing.assert_array_equal(actual, np.zeros((1, 16), dtype=np.float16))


def test_piz_gpu_huffman_groups_candidate_graphs_below_the_explicit_batch_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-runtime-independence acceptance 32-33: every speculative bit graph is explicitly bounded."""
    lengths = (1, 1)
    streams = tuple(_huffman_stream(lengths, _literal_fields(lengths, (0,))) for _ in range(5))
    tables = tuple(exr_container._parse_piz_huffman_table(stream) for stream in streams)
    stream_offsets = tuple(int(value) for value in np.cumsum((0, *(len(stream) for stream in streams[:-1]))))
    data_offsets = tuple(offset + table.data_span.start for offset, table in zip(stream_offsets, tables, strict=True))
    payload = cp.asarray(np.frombuffer(b"".join(streams), dtype=np.uint8))
    huffman_globals = exr_piz._decode_piz_huffman_gpu.__globals__
    segmented_decoder = huffman_globals["_decode_dwa_huffman_gpu"]
    batch_bit_counts: list[int] = []

    def segmented_spy(*args: object, **kwargs: object) -> cp.ndarray:
        batch_tables = kwargs["tables"]
        assert isinstance(batch_tables, tuple)
        batch_bit_counts.append(sum(table.data_bit_count for table in batch_tables))
        return segmented_decoder(*args, **kwargs)

    monkeypatch.setitem(huffman_globals, "_PIZ_HUFFMAN_SEGMENT_BATCH_BITS", 2)
    monkeypatch.setitem(huffman_globals, "_decode_dwa_huffman_gpu", segmented_spy)

    actual = exr_piz._decode_piz_huffman_gpu(
        payload,
        data_offsets=data_offsets,
        tables=tables,
        output_counts=(1,) * len(tables),
        record_labels=tuple(range(len(tables))),
        fallback_streams=streams,
        fallback_offsets=stream_offsets,
    ).get()

    assert batch_bit_counts == [2, 2, 1]
    assert all(bit_count <= 2 for bit_count in batch_bit_counts)
    np.testing.assert_array_equal(actual, np.zeros(len(tables), dtype=np.uint16))


def test_piz_gpu_huffman_splits_one_record_at_repeat_safe_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-runtime-independence acceptance 32-33: codeword, repeat history, and output ownership cross batches."""
    lengths = (1, 1)
    codes = _canonical_codes(lengths)
    stream = _huffman_stream(lengths, [codes[0], codes[1], (8, 8), codes[1], (8, 8)])
    table = exr_container._parse_piz_huffman_table(stream)
    expected = np.zeros(17, dtype=np.uint16)
    huffman_globals = exr_piz._decode_piz_huffman_gpu.__globals__

    def reject_unbounded_graph(*_args: object, **_kwargs: object) -> cp.ndarray:
        raise AssertionError("an over-limit record reached the whole-graph decoder")

    monkeypatch.setitem(huffman_globals, "_PIZ_HUFFMAN_SEGMENT_BATCH_BITS", 10)
    monkeypatch.setitem(huffman_globals, "_decode_dwa_huffman_gpu", reject_unbounded_graph)

    actual = exr_piz._decode_piz_huffman_gpu(
        cp.asarray(np.frombuffer(stream, dtype=np.uint8)),
        data_offsets=(table.data_span.start,),
        tables=(table,),
        output_counts=(expected.size,),
        record_labels=(0x50495A,),
        fallback_streams=(stream,),
        fallback_offsets=(0,),
    ).get()

    np.testing.assert_array_equal(actual, expected)


def test_piz_gpu_image_read_crosses_multiple_bounded_batches_with_mixed_storage_and_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """v1-exr-runtime-independence acceptance 32-33: a real mixed PIZ image crosses bounded batches."""
    from openexr_dev_oracle import OpenEXR

    height, width = 65, 128
    rng = np.random.default_rng(0xB051DED)
    half = np.zeros((height, width), dtype=np.float16)
    single = np.zeros((height, width), dtype=np.float32)
    unsigned = np.zeros((height, width), dtype=np.uint32)
    half[32:64] = rng.integers(0, 1 << 16, (32, width), dtype=np.uint16).view(np.float16)
    single[32:64] = rng.integers(0, 1 << 32, (32, width), dtype=np.uint32).view(np.float32)
    unsigned[32:64] = rng.integers(0, 1 << 32, (32, width), dtype=np.uint32)
    half[-1] = np.resize(np.asarray((0x0000, 0x3C00, 0xBC00, 0x7E55), dtype=np.uint16), width).view(np.float16)
    single[-1] = np.resize(np.asarray((0x00000000, 0x3F800000, 0xBF800000, 0x7FC12345), dtype=np.uint32), width).view(
        np.float32
    )
    unsigned[-1] = np.arange(width, dtype=np.uint32) * np.uint32(0x9E3779B1)
    path = tmp_path / "bounded-real-image.exr"
    OpenEXR.File(
        {"compression": OpenEXR.PIZ_COMPRESSION},
        {"H": half, "F": single, "U": unsigned},
    ).write(str(path))
    container = exr_container._parse_exr_container(path)

    assert tuple(chunk.row_count for chunk in container.chunks) == (32, 32, 1)
    assert {chunk.raw_stored for chunk in container.chunks} == {False, True}
    compressed = tuple(chunk for chunk in container.chunks if not chunk.raw_stored)
    assert compressed
    assert all(chunk.piz is not None for chunk in compressed)

    huffman_globals = exr_piz._read_exr_piz_gpu.__globals__
    original_batch_kernel = huffman_globals["_piz_huffman_decode_batches_kernel"]()
    allocated_batch_counts: list[int] = []

    def batch_kernel_spy(grid: object, block: object, arguments: tuple[object, ...]) -> None:
        allocated_batch_counts.append(int(arguments[-1]))
        original_batch_kernel(grid, block, arguments)

    monkeypatch.setitem(huffman_globals, "_PIZ_HUFFMAN_SEGMENT_BATCH_BITS", 64)
    monkeypatch.setitem(huffman_globals, "_piz_huffman_decode_batches_kernel", lambda: batch_kernel_spy)
    channels_by_name = {channel.name: channel for channel in container.parts[0].channels}

    actual = exr_piz._read_exr_piz_gpu(
        container,
        tuple(channels_by_name[name] for name in ("H", "F", "U")),
        output_dtype="float32",
    ).get()

    assert allocated_batch_counts
    assert max(allocated_batch_counts) > len(compressed)
    np.testing.assert_array_equal(actual[..., 0], half.astype(np.float32))
    np.testing.assert_array_equal(np.ascontiguousarray(actual[..., 1]).view(np.uint32), single.view(np.uint32))
    np.testing.assert_array_equal(actual[..., 2], unsigned.astype(np.float32))


def test_piz_gpu_lane_keeps_sample_planes_and_symbols_on_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase4 acceptance 27 and 29: PIZ stages file bytes once and never returns host pixels."""
    from openexr_dev_oracle import OpenEXR

    values = np.zeros((33, 64), dtype=np.float16)
    values[-1] = np.linspace(-1.0, 2.0, 64, dtype=np.float16)
    path = tmp_path / "piz-transfer.exr"
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, {"Y": values}).write(str(path))
    container = exr_container._parse_exr_container(path)
    selected = container.parts[0].channels
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    gpu_actual = exr_piz._read_exr_piz_gpu(container, selected, output_dtype="float32")

    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=26, max_total_nbytes=798, max_shape_elements=506
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=4, max_total_nbytes=44, max_shape_elements=8
    )

    file_transfers = [
        transfer
        for transfer in transfers
        if transfer.direction == "h2d"
        and transfer.nbytes == len(container.data)
        and transfer.shape == (len(container.data),)
        and transfer.dtype == "uint8"
    ]
    assert len(file_transfers) == 1
    assert gpu_actual.shape == (33, 64, 1)
    assert gpu_actual.dtype == np.dtype(np.float32)
    assert not [
        transfer for transfer in transfers if transfer.direction == "d2h" and transfer.shape == gpu_actual.shape
    ]

    transfers.clear()
    cpu_actual = exr_piz._read_exr_piz_custom_cpu(container, selected, output_dtype="float32")
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=1, max_total_nbytes=8_448, max_shape_elements=2_112
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


@pytest.mark.parametrize("backend", ("custom_cpu", "gpu"))
def test_piz_corrupt_length_table_fails_in_forced_route_without_openexr_retry(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase4 acceptance 21, 25, 35, and 36: forced eligible routes never retry via OpenEXR."""
    from openexr_dev_oracle import OpenEXR

    path = tmp_path / f"corrupt-piz-table-{backend}.exr"
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, {"H": np.zeros((32, 64), dtype=np.float16)}).write(str(path))
    container = exr_container._parse_exr_container(path)
    descriptor = container.chunks[0].piz
    assert descriptor is not None and descriptor.huffman_leader is not None
    damaged = bytearray(path.read_bytes())
    table_start = descriptor.huffman_leader.span.end
    damaged[table_start] = 0xFC
    damaged[table_start + 1] = 0xFF
    path.write_bytes(damaged)
    damaged_container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(damaged_container)
    with pytest.raises(RuntimeError, match=r"why=.*table.*what=.*how="):
        exr_harness._read_exr_pixels_with_backend(
            path,
            damaged_container,
            header,
            [(0, "H", "H")],
            unchanged=True,
            backend=backend,
        )
