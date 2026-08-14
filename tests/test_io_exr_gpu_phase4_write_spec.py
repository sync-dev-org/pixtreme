"""Specification tests for the Phase 4 PIZ write data plane."""

from __future__ import annotations

import heapq
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import cupy as cp
import exr_test_harness as exr_harness
import numpy as np
import pytest

import pixtreme as px
import pixtreme._io.formats.exr.codec_piz as exr_piz
import pixtreme._io.formats.exr.container as exr_container
import pixtreme._io.formats.exr.packing as exr_packing
import pixtreme._io.formats.exr.selection as io
import pixtreme._io.header as io_header


def _pack_bits(fields: Sequence[tuple[int, int]]) -> tuple[bytes, int]:
    bits: list[int] = []
    for value, width in fields:
        assert 0 <= value < 1 << width
        bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))
    output = bytearray((len(bits) + 7) // 8)
    for offset, bit in enumerate(bits):
        output[offset // 8] |= bit << (7 - offset % 8)
    return bytes(output), len(bits)


def _pack_lengths_oracle(lengths: Sequence[int]) -> bytes:
    fields: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(lengths):
        if lengths[cursor]:
            fields.append((int(lengths[cursor]), 6))
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


def _canonical_codes_oracle(lengths: Sequence[int], minimum_symbol: int = 0) -> dict[int, tuple[int, int]]:
    counts = [0] * 59
    for length in lengths:
        if length:
            counts[int(length)] += 1
    bases = [0] * 59
    code = 0
    for length in range(58, 0, -1):
        bases[length] = code
        code = (code + counts[length]) >> 1
    result: dict[int, tuple[int, int]] = {}
    for offset, length_value in enumerate(lengths):
        length = int(length_value)
        if length:
            result[minimum_symbol + offset] = (bases[length], length)
            bases[length] += 1
    return result


def _weighted_huffman_lengths_oracle(frequencies: Mapping[int, int], pseudo_symbol: int) -> dict[int, int]:
    """Build frequency-weighted lengths with a symbol-minimum tie key, independently of production."""
    heap: list[tuple[int, int, int | tuple[object, object]]] = [
        (int(frequency), int(symbol), int(symbol)) for symbol, frequency in sorted(frequencies.items())
    ]
    heap.append((1, pseudo_symbol, pseudo_symbol))
    heapq.heapify(heap)
    while len(heap) > 1:
        first_frequency, first_key, first_node = heapq.heappop(heap)
        second_frequency, second_key, second_node = heapq.heappop(heap)
        heapq.heappush(
            heap,
            (
                first_frequency + second_frequency,
                min(first_key, second_key),
                (first_node, second_node),
            ),
        )
    lengths: dict[int, int] = {}
    stack: list[tuple[int | tuple[object, object], int]] = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
            continue
        left, right = node
        stack.append((right, depth + 1))
        stack.append((left, depth + 1))
    return lengths


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
    output = np.asarray(words, dtype=np.uint16).copy()
    if min(nx, ny) < 2:
        return output
    pair = _forward_pair14 if w14 else _forward_pair16
    x_stride = word_stride
    y_stride = word_stride * nx
    p = 1
    p2 = 2
    while p2 <= min(nx, ny):
        for y in range(0, ny - p2 + 1, p2):
            for x in range(0, nx - p2 + 1, p2):
                i00 = word_slice + y * y_stride + x * x_stride
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
                first = word_slice + y * y_stride + x * x_stride
                second = first + p * y_stride
                output[first], output[second] = pair(int(output[first]), int(output[second]))
        if ny & p:
            y = (ny // p2) * p2
            for x in range(0, nx - p2 + 1, p2):
                first = word_slice + y * y_stride + x * x_stride
                second = first + p * x_stride
                output[first], output[second] = pair(int(output[first]), int(output[second]))
        p = p2
        p2 *= 2
    return output


def _chunk_payloads(path: Path) -> tuple[bytes, ...]:
    container = exr_container._parse_exr_container(path)
    return tuple(container.data[chunk.payload_start : chunk.payload_end] for chunk in container.chunks)


def _c_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.index(0, offset)
    return data[offset:end].decode("ascii"), end + 1


def _piz_file_chunks_oracle(path: Path) -> tuple[tuple[str, ...], tuple[tuple[int, int, bytes], ...]]:
    """Parse one-part scanline EXR ownership without using the production container parser."""
    data = path.read_bytes()
    assert struct.unpack_from("<I", data)[0] == 20000630
    assert struct.unpack_from("<I", data, 4)[0] & 0xFF == 2
    attributes: dict[str, tuple[str, bytes]] = {}
    offset = 8
    while data[offset] != 0:
        name, offset = _c_string(data, offset)
        type_name, offset = _c_string(data, offset)
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        attributes[name] = (type_name, data[offset : offset + size])
        offset += size
    offset += 1
    assert attributes["compression"] == ("compression", bytes((4,)))
    assert attributes["dataWindow"][0] == "box2i"
    x_min, y_min, x_max, y_max = struct.unpack("<iiii", attributes["dataWindow"][1])
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    assert width > 0 and height > 0

    channel_value = attributes["channels"][1]
    channels: list[str] = []
    channel_offset = 0
    while channel_value[channel_offset] != 0:
        name, channel_offset = _c_string(channel_value, channel_offset)
        pixel_type = struct.unpack_from("<i", channel_value, channel_offset)[0]
        assert pixel_type == 2
        channel_offset += 16
        channels.append(name)
    assert channel_offset + 1 == len(channel_value)

    chunk_count = (height + 31) // 32
    chunk_offsets = struct.unpack_from(f"<{chunk_count}Q", data, offset)
    table_end = offset + chunk_count * 8
    chunks: list[tuple[int, int, bytes]] = []
    physical_spans: list[tuple[int, int]] = []
    for chunk_offset in chunk_offsets:
        y, packed_size = struct.unpack_from("<ii", data, chunk_offset)
        payload_start = chunk_offset + 8
        payload_end = payload_start + packed_size
        assert table_end <= chunk_offset < payload_start <= payload_end <= len(data)
        row_start = y - y_min
        row_count = min(32, height - row_start)
        assert row_start % 32 == 0 and row_count > 0
        chunks.append((row_start, row_count, data[payload_start:payload_end]))
        physical_spans.append((chunk_offset, payload_end))
    sorted_spans = sorted(physical_spans)
    assert sorted_spans[0][0] == table_end
    for previous, current in zip(sorted_spans, sorted_spans[1:], strict=False):
        assert previous[1] == current[0]
    assert sorted_spans[-1][1] == len(data)
    chunks.sort(key=lambda item: item[0])
    assert tuple(row_start for row_start, _row_count, _payload in chunks) == tuple(range(0, height, 32))
    return tuple(channels), tuple(chunks)


def _read_bits(data: bytes, bit_offset: int, width: int) -> tuple[int, int]:
    assert 0 <= bit_offset <= bit_offset + width <= len(data) * 8
    value = 0
    for absolute in range(bit_offset, bit_offset + width):
        value = (value << 1) | ((data[absolute // 8] >> (7 - absolute % 8)) & 1)
    return value, bit_offset + width


def _decode_huffman_stream_oracle(
    stream: bytes,
    *,
    expected_count: int,
) -> tuple[np.ndarray, tuple[int, ...], bytes]:
    """Parse and decode a bare PIZ Huffman stream without production parser or decoder logic."""
    minimum_symbol, maximum_symbol, table_byte_count, data_bit_count, reserved = struct.unpack_from("<IIIII", stream)
    assert reserved == 0
    assert minimum_symbol < maximum_symbol <= 65536
    table_start = 20
    table_end = table_start + table_byte_count
    packed_table = stream[table_start:table_end]
    lengths: list[int] = []
    bit_offset = 0
    expected_length_count = maximum_symbol - minimum_symbol + 1
    while len(lengths) < expected_length_count:
        token, bit_offset = _read_bits(packed_table, bit_offset, 6)
        if token <= 58:
            lengths.append(token)
        elif token == 63:
            extra, bit_offset = _read_bits(packed_table, bit_offset, 8)
            lengths.extend((0,) * (extra + 6))
        else:
            lengths.extend((0,) * (token - 57))
    assert len(lengths) == expected_length_count
    if bit_offset < len(packed_table) * 8:
        padding, bit_offset = _read_bits(packed_table, bit_offset, len(packed_table) * 8 - bit_offset)
        assert padding == 0

    prefixes = {
        (length, code): symbol for symbol, (code, length) in _canonical_codes_oracle(lengths, minimum_symbol).items()
    }
    encoded = stream[table_end:]
    assert len(encoded) == (data_bit_count + 7) // 8
    decoded: list[int] = []
    data_offset = 0
    while len(decoded) < expected_count:
        code = 0
        symbol: int | None = None
        for length in range(1, 59):
            bit, data_offset = _read_bits(encoded, data_offset, 1)
            code = (code << 1) | bit
            symbol = prefixes.get((length, code))
            if symbol is not None:
                break
        assert symbol is not None
        if symbol == maximum_symbol:
            assert decoded
            repeat_count, data_offset = _read_bits(encoded, data_offset, 8)
            assert repeat_count > 0
            decoded.extend((decoded[-1],) * repeat_count)
        else:
            decoded.append(symbol)
        assert len(decoded) <= expected_count
    assert data_offset == data_bit_count
    if data_offset < len(encoded) * 8:
        padding, data_offset = _read_bits(encoded, data_offset, len(encoded) * 8 - data_offset)
        assert padding == 0
    return np.asarray(decoded, dtype=np.uint16), tuple(lengths), packed_table


def _compressed_payload_sections(payload: bytes) -> Mapping[str, bytes]:
    minimum, maximum = struct.unpack_from("<HH", payload)
    bitmap_size = maximum - minimum + 1 if minimum <= maximum else 0
    bitmap_end = 4 + bitmap_size
    huffman_size = struct.unpack_from("<I", payload, bitmap_end)[0]
    huffman_start = bitmap_end + 4
    huffman_end = huffman_start + huffman_size
    assert huffman_end == len(payload)
    minimum_symbol, maximum_symbol, table_bytes, data_bits, reserved = struct.unpack_from(
        "<IIIII", payload, huffman_start
    )
    packed_table = payload[huffman_start + 20 :]
    produced = 0
    bit_offset = 0
    while produced < maximum_symbol - minimum_symbol + 1:
        token, bit_offset = _read_bits(packed_table, bit_offset, 6)
        if token <= 58:
            produced += 1
        elif token == 63:
            extra, bit_offset = _read_bits(packed_table, bit_offset, 8)
            produced += extra + 6
        else:
            produced += token - 57
    derived_table_bytes = (bit_offset + 7) // 8
    assert derived_table_bytes == table_bytes
    data_start = huffman_start + 20 + derived_table_bytes
    data_end = data_start + (data_bits + 7) // 8
    assert data_end == huffman_end
    return {
        "bitmap-leader": payload[:4],
        "bitmap": payload[4:bitmap_end],
        "huffman-count": payload[bitmap_end:huffman_start],
        "huffman-leader": payload[huffman_start : huffman_start + 20],
        "length-table": payload[huffman_start + 20 : data_start],
        "data": payload[data_start:data_end],
        "reserved": struct.pack("<I", reserved),
    }


def _inverse_pair_oracle(a_word: int, b_word: int, *, w14: bool) -> tuple[int, int]:
    if w14:
        low = _signed_word(a_word)
        high = _signed_word(b_word)
        first = low + (high & 1) + (high >> 1)
        return first & 0xFFFF, (first - high) & 0xFFFF
    second = (a_word - (b_word >> 1)) & 0xFFFF
    return (b_word + second - 32768) & 0xFFFF, second


def _inverse_wavelet_oracle(
    words: np.ndarray,
    *,
    nx: int,
    ny: int,
    word_stride: int,
    word_slice: int,
    w14: bool,
) -> None:
    if min(nx, ny) < 2:
        return
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
                low0, low1 = _inverse_pair_oracle(int(words[i00]), int(words[i10]), w14=w14)
                high0, high1 = _inverse_pair_oracle(int(words[i01]), int(words[i11]), w14=w14)
                words[i00], words[i01] = _inverse_pair_oracle(low0, high0, w14=w14)
                words[i10], words[i11] = _inverse_pair_oracle(low1, high1, w14=w14)
        if nx & p:
            x = (nx // step) * step
            for y in range(0, ny - step + 1, step):
                first = word_slice + y * y_stride + x * x_stride
                second = first + p * y_stride
                words[first], words[second] = _inverse_pair_oracle(int(words[first]), int(words[second]), w14=w14)
        if ny & p:
            y = (ny // step) * step
            for x in range(0, nx - step + 1, step):
                first = word_slice + y * y_stride + x * x_stride
                second = first + p * x_stride
                words[first], words[second] = _inverse_pair_oracle(int(words[first]), int(words[second]), w14=w14)
        p //= 2


def _decode_compressed_payload_oracle(
    payload: bytes,
    *,
    expected_word_count: int,
    fields: Sequence[tuple[int, int, int, int, int]],
) -> np.ndarray:
    """Independently parse bitmap/LUT, Huffman/padding, wavelet fields, and their chunk ownership."""
    minimum, maximum = struct.unpack_from("<HH", payload)
    bitmap_size = maximum - minimum + 1 if minimum <= maximum else 0
    bitmap_end = 4 + bitmap_size
    assert bitmap_end + 4 <= len(payload)
    reverse_lut = [0]
    for byte_offset, packed in enumerate(payload[4:bitmap_end], start=minimum):
        for bit in range(8):
            value = byte_offset * 8 + bit
            if value and packed & (1 << bit):
                reverse_lut.append(value)

    huffman_size = struct.unpack_from("<I", payload, bitmap_end)[0]
    huffman_start = bitmap_end + 4
    huffman_end = huffman_start + huffman_size
    assert huffman_end == len(payload)
    minimum_symbol, maximum_symbol, table_bytes, data_bits, reserved = struct.unpack_from(
        "<IIIII", payload, huffman_start
    )
    assert reserved == 0
    assert minimum_symbol < maximum_symbol <= 65536
    table_start = huffman_start + 20
    table_end = table_start + table_bytes
    assert table_end <= huffman_end
    packed_table = payload[table_start:table_end]
    lengths: list[int] = []
    table_bit = 0
    target_length_count = maximum_symbol - minimum_symbol + 1
    while len(lengths) < target_length_count:
        token, table_bit = _read_bits(packed_table, table_bit, 6)
        if token <= 58:
            lengths.append(token)
        elif token == 63:
            extra, table_bit = _read_bits(packed_table, table_bit, 8)
            lengths.extend((0,) * (extra + 6))
        else:
            lengths.extend((0,) * (token - 57))
    assert len(lengths) == target_length_count
    if table_bit < len(packed_table) * 8:
        padding, table_bit = _read_bits(packed_table, table_bit, len(packed_table) * 8 - table_bit)
        assert padding == 0
    assert table_bit == len(packed_table) * 8

    codes = _canonical_codes_oracle(lengths, minimum_symbol)
    prefixes = {(length, code): symbol for symbol, (code, length) in codes.items()}
    encoded = payload[table_end:huffman_end]
    assert len(encoded) == (data_bits + 7) // 8
    words: list[int] = []
    data_bit = 0
    while len(words) < expected_word_count:
        code = 0
        symbol: int | None = None
        for length in range(1, 59):
            bit, data_bit = _read_bits(encoded, data_bit, 1)
            code = (code << 1) | bit
            symbol = prefixes.get((length, code))
            if symbol is not None:
                break
        assert symbol is not None
        if symbol == maximum_symbol:
            assert words
            repeat_count, data_bit = _read_bits(encoded, data_bit, 8)
            assert repeat_count > 0
            words.extend((words[-1],) * repeat_count)
        else:
            assert 0 <= symbol <= 0xFFFF
            words.append(symbol)
        assert len(words) <= expected_word_count
    assert data_bit == data_bits
    if data_bit < len(encoded) * 8:
        padding, data_bit = _read_bits(encoded, data_bit, len(encoded) * 8 - data_bit)
        assert padding == 0

    transformed = np.asarray(words, dtype=np.uint16)
    w14 = len(reverse_lut) - 1 < 16384
    for offset, nx, ny, word_stride, word_slice in fields:
        _inverse_wavelet_oracle(
            transformed[offset:],
            nx=nx,
            ny=ny,
            word_stride=word_stride,
            word_slice=word_slice,
            w14=w14,
        )
    reverse = np.asarray(reverse_lut, dtype=np.uint16)
    assert np.all(transformed < reverse.size)
    return reverse[transformed]


def _encode_single_channel_words(
    words: np.ndarray,
    *,
    width: int,
    pixel_type: int,
) -> tuple[bytes, ...]:
    values = np.ascontiguousarray(words)
    height = int(values.size // width)
    bytes_per_sample = 2 if pixel_type == 1 else 4
    row_counts = tuple(min(32, height - row_start) for row_start in range(0, height, 32))
    raw_sizes = tuple(row_count * width * bytes_per_sample for row_count in row_counts)
    raw_offsets = tuple(int(value) for value in np.cumsum((0, *raw_sizes[:-1]), dtype=np.int64))
    raw = cp.asarray(values.view(np.uint8).reshape(-1))
    encoded, encoded_sizes = exr_piz._encode_piz_chunks_gpu(
        raw,
        raw_offsets,
        raw_sizes,
        row_counts=row_counts,
        width=width,
        channel_count=1,
        pixel_type=pixel_type,
    )
    payload = encoded.get().tobytes()
    offsets = tuple(int(value) for value in np.cumsum((0, *encoded_sizes[:-1]), dtype=np.int64))
    return tuple(payload[offset : offset + size] for offset, size in zip(offsets, encoded_sizes, strict=True))


def _expected_data_bits(symbols: np.ndarray, table: exr_container._PizHuffmanTable) -> int:
    lengths = {
        symbol: int(length)
        for symbol, length in zip(
            range(table.minimum_symbol, table.maximum_symbol + 1), table.code_lengths, strict=True
        )
    }
    pseudo_length = lengths[table.maximum_symbol]
    values = np.asarray(symbols, dtype=np.uint16).reshape(-1)
    total = 0
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[end] == values[cursor]:
            end += 1
        remaining = end - cursor
        while remaining:
            group = min(remaining, 256)
            additional = group - 1
            literal_length = lengths[int(values[cursor])]
            if literal_length + pseudo_length + 8 < literal_length * additional:
                total += literal_length + pseudo_length + 8
            else:
                total += literal_length * group
            remaining -= group
        cursor = end
    return total


def test_piz_huffman_tree_is_frequency_weighted_and_deterministic() -> None:
    """v1-exr-runtime-independence acceptance 34-35: frequency changes lengths under a stable symbol tie key."""
    distributions = ({0: 99, 1: 7, 2: 3, 3: 1}, {0: 1, 1: 3, 2: 7, 3: 99})
    actual_maps: list[dict[int, int]] = []
    for frequencies in distributions:
        symbols = np.concatenate(
            tuple(np.full(frequency, symbol, dtype=np.uint16) for symbol, frequency in frequencies.items())
        )
        stream = exr_harness._encode_piz_huffman_gpu(cp.asarray(symbols)).get().tobytes()
        decoded, lengths, _packed_table = _decode_huffman_stream_oracle(stream, expected_count=symbols.size)
        np.testing.assert_array_equal(decoded, symbols)
        actual = {symbol: lengths[symbol] for symbol in (*frequencies, 4)}
        assert actual == _weighted_huffman_lengths_oracle(frequencies, pseudo_symbol=4)
        actual_maps.append(actual)

    assert actual_maps[0][0] < actual_maps[0][3]
    assert actual_maps[1][3] < actual_maps[1][0]


def test_piz_gpu_huffman_stream_uses_frequency_weighted_lengths_repeatably() -> None:
    """v1-exr-runtime-independence acceptance 34-35: parallel tree construction is weighted and byte-stable."""
    distributions = ({0: 99, 1: 7, 2: 3, 3: 1}, {0: 1, 1: 3, 2: 7, 3: 99})
    length_maps: list[dict[int, int]] = []
    for frequencies in distributions:
        symbols = np.concatenate(
            tuple(np.full(frequency, symbol, dtype=np.uint16) for symbol, frequency in frequencies.items())
        )
        streams = tuple(exr_harness._encode_piz_huffman_gpu(cp.asarray(symbols)).get().tobytes() for _ in range(3))
        assert streams[0] == streams[1] == streams[2]
        table = exr_container._parse_piz_huffman_table(streams[0])
        actual = {
            symbol: int(table.code_lengths[symbol - table.minimum_symbol])
            for symbol in (*frequencies, table.maximum_symbol)
        }
        assert actual == _weighted_huffman_lengths_oracle(frequencies, pseudo_symbol=4)
        length_maps.append(actual)

    assert length_maps[0][0] < length_maps[0][3]
    assert length_maps[1][3] < length_maps[1][0]
    assert length_maps[0] != length_maps[1]


def test_piz_canonical_codes_use_reverse_length_bases_and_symbol_increment() -> None:
    """v1-exr-gpu-phase4 acceptance 19: canonical codes use length 58-to-1 bases and symbol order."""
    symbols = np.asarray((0,) * 13 + (1,) * 8 + (2,) * 5 + (3,) * 3 + (4,), dtype=np.uint16)

    stream = exr_harness._encode_piz_huffman_gpu(cp.asarray(symbols)).get().tobytes()
    decoded, lengths, _packed_table = _decode_huffman_stream_oracle(stream, expected_count=symbols.size)

    np.testing.assert_array_equal(decoded, symbols)
    expected_map = _weighted_huffman_lengths_oracle({0: 13, 1: 8, 2: 5, 3: 3, 4: 1}, pseudo_symbol=5)
    assert lengths == tuple(expected_map[symbol] for symbol in range(6))
    assert len(set(lengths)) > 1


@pytest.mark.parametrize("zero_gap", (1, 2, 5, 6, 261, 262, 522))
def test_piz_length_table_serializes_every_zero_run_boundary(zero_gap: int) -> None:
    """v1-exr-gpu-phase4 acceptance 20: zero runs 1, 2-5, 6-261, and longer split canonically."""
    maximum_actual = zero_gap + 1
    symbols = np.asarray((0, maximum_actual), dtype=np.uint16)
    stream = exr_harness._encode_piz_huffman_gpu(cp.asarray(symbols)).get().tobytes()

    decoded, lengths, packed_table = _decode_huffman_stream_oracle(stream, expected_count=symbols.size)
    expected_map = _weighted_huffman_lengths_oracle({0: 1, maximum_actual: 1}, maximum_actual + 1)
    expected_lengths = tuple(expected_map.get(symbol, 0) for symbol in range(maximum_actual + 2))

    np.testing.assert_array_equal(decoded, symbols)
    assert lengths == expected_lengths
    assert packed_table == _pack_lengths_oracle(expected_lengths)


@pytest.mark.parametrize(
    "run_count",
    (5, 12, 13),
)
def test_piz_repeat_choice_uses_the_strict_reference_cost(run_count: int) -> None:
    """v1-exr-gpu-phase4 acceptance 22: repeat form wins only under strict cost inequality."""
    symbols = np.asarray((0,) * run_count + (1,), dtype=np.uint16)

    stream = exr_harness._encode_piz_huffman_gpu(cp.asarray(symbols)).get().tobytes()
    decoded, _lengths, _packed_table = _decode_huffman_stream_oracle(stream, expected_count=symbols.size)
    table = exr_container._parse_piz_huffman_table(stream)

    np.testing.assert_array_equal(decoded, symbols)
    assert table.data_bit_count == _expected_data_bits(symbols, table)


def test_piz_huffman_gpu_encoder_splits_long_runs_and_writes_a_canonical_stream() -> None:
    """v1-exr-gpu-phase4 acceptance 20, 22, and 26: runs, fields, exhaustion, and padding are canonical."""
    symbols = np.asarray((7,) * 520 + (2, 3) * 19 + (1,), dtype=np.uint16)

    stream = exr_harness._encode_piz_huffman_gpu(cp.asarray(symbols)).get().tobytes()
    table = exr_container._parse_piz_huffman_table(stream)
    decoded = exr_container._decode_piz_huffman_host(stream, table, expected_count=symbols.size)

    np.testing.assert_array_equal(decoded, symbols)
    assert table.maximum_symbol == int(symbols.max()) + 1
    assert table.declared_table_byte_count == table.table_span.size
    assert table.reserved == 0
    assert table.data_bit_count == _expected_data_bits(symbols, table)
    assert table.data_span.end == len(stream)
    if table.data_bit_count % 8:
        assert stream[-1] & ((1 << (8 - table.data_bit_count % 8)) - 1) == 0
    assigned_lengths = tuple(length for length in table.code_lengths if length)
    frequencies = {
        int(symbol): int(count) for symbol, count in zip(*np.unique(symbols, return_counts=True), strict=True)
    }
    actual_lengths = {
        symbol: int(table.code_lengths[symbol - table.minimum_symbol])
        for symbol in (*frequencies, table.maximum_symbol)
    }
    assert actual_lengths == _weighted_huffman_lengths_oracle(frequencies, table.maximum_symbol)
    assert sum(1 << (max(assigned_lengths) - length) for length in assigned_lengths) == 1 << max(assigned_lengths)
    expected_table = _pack_lengths_oracle(table.code_lengths)
    assert table.table_span.size == len(expected_table)
    assert stream[table.table_span.start : table.table_span.end] == expected_table


@pytest.mark.parametrize("run_count", (5, 12, 13))
def test_piz_huffman_gpu_encoder_obeys_false_equal_and_true_repeat_costs(run_count: int) -> None:
    """v1-exr-gpu-phase4 acceptance 22: packed data distinguishes false, equal, and true run costs."""
    symbols = np.asarray((0,) * run_count + (1,), dtype=np.uint16)

    stream = exr_harness._encode_piz_huffman_gpu(cp.asarray(symbols)).get().tobytes()
    table = exr_container._parse_piz_huffman_table(stream)

    assert table.data_bit_count == _expected_data_bits(symbols, table)
    np.testing.assert_array_equal(
        exr_container._decode_piz_huffman_host(stream, table, expected_count=symbols.size), symbols
    )


@pytest.mark.parametrize(
    ("nx", "ny", "word_stride", "word_slice", "max_value"),
    (
        (1, 7, 1, 0, 8191),
        (7, 1, 2, 1, 16384),
        (4, 6, 1, 0, 8191),
        (5, 6, 2, 0, 8191),
        (6, 5, 2, 1, 16384),
        (7, 9, 1, 0, 16384),
    ),
)
def test_piz_forward_wavelet_gpu_matches_the_independent_all_level_oracle(
    nx: int,
    ny: int,
    word_stride: int,
    word_slice: int,
    max_value: int,
) -> None:
    """v1-exr-gpu-phase4 acceptance 13-17: GPU forward w14/w16 obeys every level and odd boundary."""
    rng = np.random.default_rng(0x50495A + nx * 101 + ny)
    words = rng.integers(0, max_value + 1, nx * ny * word_stride, dtype=np.uint16)
    expected = _forward_wavelet_oracle(
        words,
        nx=nx,
        ny=ny,
        word_stride=word_stride,
        word_slice=word_slice,
        w14=max_value < 16384,
    )
    device = cp.asarray(words)

    exr_harness._piz_forward_wavelet_gpu(
        device,
        nx=nx,
        ny=ny,
        word_stride=word_stride,
        word_slice=word_slice,
        max_value=max_value,
    )

    np.testing.assert_array_equal(device.get(), expected)


@pytest.mark.parametrize("fixture", ("zero", "sparse-w14", "dense-w16", "partial", "noise"))
def test_piz_gpu_encoder_emits_deterministic_self_parseable_chunk_payloads(fixture: str) -> None:
    """v1-exr-runtime-independence acceptance 34-35: HALF chunks are deterministic and independently parseable."""
    if fixture == "zero":
        height, width = 32, 64
        bits = np.zeros((height, width), dtype=np.uint16)
    elif fixture == "sparse-w14":
        height, width = 32, 257
        bits = np.resize(np.asarray((0, 1, 7, 8, 1024, 0x3C00, 0xBC00), dtype=np.uint16), (height, width))
    elif fixture == "dense-w16":
        height, width = 32, 1024
        bits = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    elif fixture == "partial":
        height, width = 65, 129
        y, x = np.mgrid[:height, :width]
        bits = ((x * 37 + y * 101) % 4096).astype(np.uint16)
    else:
        height, width = 33, 257
        bits = np.random.default_rng(0x50495A).integers(0, 1 << 16, (height, width), dtype=np.uint16)
    values = np.ascontiguousarray(bits.view(np.float16))
    first = _encode_single_channel_words(values, width=width, pixel_type=1)
    second = _encode_single_channel_words(values, width=width, pixel_type=1)
    raw_sizes = tuple(min(32, height - row_start) * width * 2 for row_start in range(0, height, 32))

    assert first == second
    for payload, raw_size in zip(first, raw_sizes, strict=True):
        if len(payload) >= raw_size:
            assert len(payload) == raw_size
            continue
        sections = _compressed_payload_sections(payload)
        assert sections["reserved"] == bytes(4)
        huffman_stream = b"".join(sections[name] for name in ("huffman-leader", "length-table", "data"))
        table = exr_container._parse_piz_huffman_table(huffman_stream)
        assert table.table_span.size == len(sections["length-table"])
        if table.data_bit_count % 8:
            assert sections["data"][-1] & ((1 << (8 - table.data_bit_count % 8)) - 1) == 0


def test_piz_equal_size_payload_selects_raw_bytes() -> None:
    """v1-exr-gpu-phase4 acceptance 7: equal-size compressed candidates never masquerade as raw chunks."""
    raw = cp.asarray(np.frombuffer(b"raw!", dtype=np.uint8))
    compressed = cp.asarray(np.frombuffer(b"piz!", dtype=np.uint8))

    selected, sizes = exr_packing._select_exr_payloads(raw, (0,), (4,), compressed, (0,), (4,))

    assert selected.get().tobytes() == b"raw!"
    assert sizes == (4,)


def test_piz_internal_uint_low_high_primitive_is_deterministic_and_self_parseable(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 31 and 34-35: UINT low/high fields have deterministic wire data."""
    from openexr_dev_oracle import OpenEXR

    height, width = 33, 1024
    values = np.resize(
        np.asarray(
            (
                0x00000000,
                0xFFFFFFFF,
                0x0000FFFF,
                0xFFFF0000,
                0x80000000,
                0x00000001,
                0x7FFFFFFF,
                0x12345678,
            ),
            dtype=np.uint32,
        ),
        (height, width),
    )
    first = _encode_single_channel_words(values.astype("<u4", copy=False), width=width, pixel_type=0)
    second = _encode_single_channel_words(values.astype("<u4", copy=False), width=width, pixel_type=0)

    assert first == second
    assert any(
        len(payload) < min(32, height - row_start) * width * 4
        for payload, row_start in zip(first, range(0, height, 32), strict=True)
    )
    for payload, row_start in zip(first, range(0, height, 32), strict=True):
        raw_size = min(32, height - row_start) * width * 4
        if len(payload) < raw_size:
            _compressed_payload_sections(payload)
        else:
            assert len(payload) == raw_size

    path = tmp_path / "native-uint.exr"
    OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, {"U": values}).write(str(path))
    template = exr_container._parse_exr_container(path)
    offset_table_start = min(template.offset_table) - len(template.offset_table) * 8
    cursor = offset_table_start + len(template.offset_table) * 8
    offsets: list[int] = []
    chunks = bytearray()
    for chunk, payload in zip(template.chunks, first, strict=True):
        offsets.append(cursor)
        encoded_chunk = struct.pack("<ii", chunk.y, len(payload)) + payload
        chunks.extend(encoded_chunk)
        cursor += len(encoded_chunk)
    offset_table = b"".join(struct.pack("<Q", offset) for offset in offsets)
    path.write_bytes(template.data[:offset_table_start] + offset_table + bytes(chunks))
    container = exr_container._parse_exr_container(path)
    assert container.piz_eligible is True
    for chunk_index, chunk in enumerate(container.chunks):
        row_start = chunk_index * 32
        expected_raw = np.ascontiguousarray(values[row_start : row_start + chunk.row_count]).view(np.uint8).reshape(-1)
        np.testing.assert_array_equal(exr_piz._piz_materialize_chunk_host(container, chunk), expected_raw)
    dev_decoded = np.asarray(OpenEXR.File(str(path), separate_channels=True).channels()["U"].pixels)
    np.testing.assert_array_equal(dev_decoded, values)


def _public_values(dtype: np.dtype[object], *, height: int = 33, width: int = 67) -> np.ndarray:
    shape = (height, width, 3)
    if dtype == np.dtype(np.float16):
        bits = np.resize(
            np.asarray((0x0000, 0x8000, 0x0001, 0x03FF, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E55), dtype=np.uint16),
            shape,
        )
        return bits.view(np.float16)
    if dtype == np.dtype(np.float32):
        bits = np.resize(
            np.asarray(
                (
                    0x00000000,
                    0x80000000,
                    0x00000001,
                    0x007FFFFF,
                    0x3F800000,
                    0xBF800000,
                    0x7F800000,
                    0xFF800000,
                    0x7FC12345,
                ),
                dtype=np.uint32,
            ),
            shape,
        )
        return bits.view(np.float32)
    maximum = np.iinfo(dtype).max
    return (np.arange(np.prod(shape), dtype=np.uint64).reshape(shape) * np.uint64(7919) % (maximum + 1)).astype(dtype)


@pytest.mark.parametrize(
    "dtype",
    (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32)),
)
def test_forced_piz_gpu_write_cross_decodes_in_all_lanes_with_public_adapter_bits_and_metadata(
    tmp_path: Path,
    dtype: np.dtype[object],
) -> None:
    """v1-exr-runtime-independence acceptance 31 and 34: public HALF/FLOAT/UINT lanes cross-decode exactly."""
    from openexr_dev_oracle import OpenEXR

    labels = ("beauty.R", "beauty.G", "matte.A")
    values = _public_values(dtype)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels=labels)
    path = tmp_path / f"gpu-{dtype.name}.exr"

    io._write_exr_with_backend(path, frame, compression="piz", dwa_level=None, backend="gpu")

    container = exr_container._parse_exr_container(path)
    header = io_header._exr_header(container)
    assert container.compression == "piz"
    assert container.piz_eligible is True
    assert (header.color.colorspace, header.color.gamma) == ("ACEScg", None)
    reference_channels = OpenEXR.File(str(path), separate_channels=True).channels()
    if dtype.kind == "f" or dtype == np.dtype(np.uint32):
        expected = values
    else:
        expected = values.astype(np.float32) * np.float32(1.0 / np.iinfo(dtype).max)
    output_dtype = (
        "float16" if dtype == np.dtype(np.float16) else "uint32" if dtype == np.dtype(np.uint32) else "float32"
    )
    locations = [(0, label, label) for label in labels]
    lane_outputs = tuple(
        exr_harness._read_exr_pixels_with_backend(
            path,
            container,
            header,
            locations,
            unchanged=True,
            backend=backend,
        ).get()
        for backend in ("gpu", "custom_cpu")
    )
    for channel_index, label in enumerate(labels):
        reference = np.asarray(reference_channels[label].pixels)
        np.testing.assert_array_equal(
            reference.view(f"u{reference.dtype.itemsize}"),
            expected[..., channel_index].view(f"u{expected.dtype.itemsize}"),
        )
    for actual in lane_outputs:
        assert actual.dtype == np.dtype(output_dtype)
        np.testing.assert_array_equal(
            actual.view(f"u{actual.dtype.itemsize}"), expected.view(f"u{expected.dtype.itemsize}")
        )


def test_public_gpu_writer_is_deterministic_and_passes_all_four_piz_oracles(tmp_path: Path) -> None:
    """v1-exr-runtime-independence acceptance 34-35: deterministic FLOAT wire passes four independent oracles."""
    from openexr_dev_oracle import OpenEXR

    labels = ("beauty.R", "beauty.G", "matte.A")
    values = _public_values(np.dtype(np.float32), height=65, width=193)
    frame = px.io.from_array(cp.asarray(values), colorspace="ACEScg", gamma="linear", channels=labels)
    first_path = tmp_path / "first.exr"
    second_path = tmp_path / "second.exr"

    io._write_exr_with_backend(first_path, frame, compression="piz", dwa_level=None, backend="gpu")
    io._write_exr_with_backend(second_path, frame, compression="piz", dwa_level=None, backend="gpu")

    assert first_path.read_bytes() == second_path.read_bytes()
    container = exr_container._parse_exr_container(first_path)
    header = io_header._exr_header(container)
    locations = [(0, label, label) for label in labels]
    self_decoded = exr_harness._read_exr_pixels_with_backend(
        first_path, container, header, locations, unchanged=True, backend="gpu"
    ).get()
    np.testing.assert_array_equal(self_decoded.view(np.uint32), values.view(np.uint32))
    dev_channels = OpenEXR.File(str(first_path), separate_channels=True).channels()
    for channel_index, label in enumerate(labels):
        actual = np.asarray(dev_channels[label].pixels)
        np.testing.assert_array_equal(actual.view(np.uint32), values[..., channel_index].view(np.uint32))
    file_labels, wire_chunks = _piz_file_chunks_oracle(first_path)
    source_by_label = {label: values[..., channel_index] for channel_index, label in enumerate(labels)}
    assert set(file_labels) == set(labels)
    for row_start, row_count, payload in wire_chunks:
        chunk_values = np.stack(
            tuple(source_by_label[label][row_start : row_start + row_count] for label in file_labels),
            axis=2,
        )
        raw_words = (
            np.ascontiguousarray(chunk_values.transpose(0, 2, 1))
            .view(np.uint16)
            .reshape(row_count, len(file_labels), values.shape[1], 2)
        )
        if len(payload) == raw_words.nbytes:
            assert payload == raw_words.view(np.uint8).reshape(-1).tobytes()
            continue
        assert len(payload) < raw_words.nbytes
        plane_word_count = row_count * values.shape[1] * 2
        fields = tuple(
            (channel_index * plane_word_count, values.shape[1], row_count, 2, word_slice)
            for channel_index in range(len(file_labels))
            for word_slice in range(2)
        )
        independently_restored = _decode_compressed_payload_oracle(
            payload,
            expected_word_count=raw_words.size,
            fields=fields,
        )
        expected_staged = np.ascontiguousarray(raw_words.transpose(1, 0, 2, 3)).reshape(-1)
        np.testing.assert_array_equal(independently_restored, expected_staged)


def test_piz_write_graph_keeps_planes_wavelets_and_symbols_on_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase4 acceptance 28-29 and v1-exr-runtime-independence acceptance 34-35:
    PIZ returns one final payload and no host intermediate arrays.
    """
    values = cp.zeros((33, 64, 3), dtype=cp.float32)
    values[-1] = cp.linspace(-1.0, 2.0, 64 * 3, dtype=cp.float32).reshape(64, 3)
    frame = px.io.from_array(values, colorspace="ACEScg", gamma="linear", channels="RGB")
    path = tmp_path / "piz-write-transfer.exr"
    transfers = exr_harness._record_cupy_transfers(monkeypatch)

    px.io.write_image(path, frame, compression="piz")

    container = exr_container._parse_exr_container(path)
    payload_bytes = sum(chunk.packed_size for chunk in container.chunks)
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="h2d", max_count=30, max_total_nbytes=1_000, max_shape_elements=30
    )
    exr_harness._assert_cupy_transfer_budget(
        transfers, direction="d2h", max_count=17, max_total_nbytes=666, max_shape_elements=444
    )
    payload_transfers = [
        transfer
        for transfer in transfers
        if transfer.direction == "d2h"
        and transfer.has_output_buffer
        and transfer.nbytes == payload_bytes
        and transfer.shape == (payload_bytes,)
        and transfer.dtype == "uint8"
    ]
    assert len(payload_transfers) == 1
    assert not [
        transfer
        for transfer in transfers
        if transfer.direction == "d2h"
        and transfer.shape == frame.data.shape
        and transfer.dtype == frame.data.dtype.name
    ]


def test_unexpected_piz_gpu_write_failure_is_actionable_and_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1-exr-gpu-phase4 acceptance 37: a forced GPU defect preserves its cause and never retries OpenEXR."""
    frame = px.io.from_array(
        cp.zeros((9, 16, 3), dtype=cp.float32), colorspace="ACEScg", gamma="linear", channels="RGB"
    )
    path = tmp_path / "piz-defect.exr"

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic PIZ data-plane failure")

    monkeypatch.setattr(io, "_encode_piz_chunks_gpu", fail)
    with pytest.raises(RuntimeError, match=r"why=.*PIZ.*what=.*how=") as error:
        io._write_exr_with_backend(path, frame, compression="piz", dwa_level=None, backend="gpu")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "synthetic PIZ data-plane failure" in str(error.value.__cause__)
    assert not path.exists()
