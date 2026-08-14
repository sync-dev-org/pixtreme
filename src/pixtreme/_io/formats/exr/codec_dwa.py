"""DWAA and DWAB OpenEXR read/write lane and CUDA kernels."""

from __future__ import annotations

import heapq
import struct
import zlib
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._io.formats.exr.codec_zip import (
    _decode_deflate_chunks,
)
from pixtreme._io.formats.exr.container import (
    _DWA_HUFFMAN_LOOKAHEAD_SIZE,
    _DWA_HUFFMAN_SEGMENT_TOKEN_COUNT,
    _DWA_JPEG_CHROMA,
    _DWA_JPEG_LUMINANCE,
    _DWA_MAX_HUFFMAN_CODE_LENGTH,
    _DWA_MAX_HUFFMAN_SYMBOL,
    _DWA_STATIC_HUFFMAN,
    _EXR_DTYPE_INFO,
    _EXR_LINES_PER_CHUNK,
    _EXR_MAX_GRID_Y,
    _EXR_THREADS_PER_BLOCK,
    _classify_default_dwa_channels,
    _classify_dwa_channels,
    _dwa_suffix,
    _DwaByteSpan,
    _DwaChannelLayout,
    _DwaChunkDescriptor,
    _DwaDeflateStreams,
    _DwaHuffmanTable,
    _DwaLeader,
    _DwaWriteStreams,
    _ExrChannel,
    _ExrChunk,
    _ExrContainer,
    _ExrGpuError,
    _gpu_error,
    _parse_dwa_huffman_table,
)
from pixtreme._io.formats.exr.packing import (
    _adler_kernel,
    _device_i64,
    _encode_deflate_chunks,
    _encode_exr_output_channels,
    _exr_write_header,
    _maximum_block_count,
    _numpy_offsets,
    _pack_exr_gpu,
    _prefix_offsets,
    _restore_even_odd_host,
    _restore_exr_gpu_chunks,
    _restore_predictor_host,
    _select_exr_payloads,
    _transform_and_checksum_chunks,
    _wrap_deflate_chunks,
)

_DWA_NATURAL_TO_ZIGZAG = np.asarray(
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


def _dwa_inverse_basis_host() -> np.ndarray:
    sample = np.arange(8, dtype=np.float32)[:, None]
    frequency = np.arange(8, dtype=np.float32)[None, :]
    basis = np.float32(0.5) * np.cos(
        np.float32(np.pi) * frequency * (np.float32(2.0) * sample + np.float32(1.0)) / np.float32(16.0)
    )
    basis[:, 0] = np.float32(1.0 / np.sqrt(8.0))
    return basis.astype(np.float32)


_DWA_INVERSE_BASIS_HOST = _dwa_inverse_basis_host()


def _dwa_quantization_tables(dwa_level: float) -> tuple[np.ndarray, np.ndarray]:
    base_error = np.float32(np.float32(dwa_level) / np.float32(100000.0))
    luminance = base_error * _DWA_JPEG_LUMINANCE / np.float32(10.0)
    chroma = base_error * _DWA_JPEG_CHROMA / np.float32(17.0)
    return luminance.astype(np.float32), chroma.astype(np.float32)


def _pack_dwa_bit_fields(fields: Sequence[tuple[int, int]]) -> tuple[bytes, int]:
    output = bytearray()
    byte = 0
    occupied = 0
    bit_count = 0
    for value, width in fields:
        if width < 0 or value < 0 or value >= 1 << width:
            raise _gpu_error(
                why="the DWA control plane received an invalid packed bit field",
                what=f"value={value}, width={width}",
                how="pack only non-negative values representable by their declared bit width",
            )
        for shift in range(width - 1, -1, -1):
            byte = (byte << 1) | ((value >> shift) & 1)
            occupied += 1
            bit_count += 1
            if occupied == 8:
                output.append(byte)
                byte = 0
                occupied = 0
    if occupied:
        output.append(byte << (8 - occupied))
    return bytes(output), bit_count


def _build_dwa_huffman_lengths(frequencies: Mapping[int, int], pseudo_symbol: int) -> dict[int, int]:
    leaves = tuple(sorted((*frequencies, pseudo_symbol)))
    heap: list[tuple[int, int, int, int | tuple[object, object]]] = []
    serial = 0
    for symbol in leaves:
        frequency = 1 if symbol == pseudo_symbol else frequencies[symbol]
        heap.append((frequency, symbol, serial, symbol))
        serial += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        merged: tuple[object, object] = (first[3], second[3])
        heapq.heappush(heap, (first[0] + second[0], min(first[1], second[1]), serial, merged))
        serial += 1

    lengths: dict[int, int] = {}
    stack: list[tuple[int | tuple[object, object], int]] = [(heap[0][3], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = max(1, depth)
            continue
        left, right = node
        stack.append((cast(int | tuple[object, object], right), depth + 1))
        stack.append((cast(int | tuple[object, object], left), depth + 1))
    if max(lengths.values()) > _DWA_MAX_HUFFMAN_CODE_LENGTH:
        balanced_length = max(1, (len(lengths) - 1).bit_length())
        lengths = {symbol: balanced_length for symbol in lengths}
    return lengths


def _pack_dwa_huffman_lengths(lengths: Sequence[int] | np.ndarray) -> bytes:
    length_array = np.asarray(lengths, dtype=np.uint8)
    fields: list[tuple[int, int]] = []
    cursor = 0
    for offset in np.flatnonzero(length_array):
        zero_count = int(offset) - cursor
        while zero_count:
            run = min(zero_count, 261)
            if run == 1:
                fields.append((0, 6))
            elif run <= 5:
                fields.append((run + 57, 6))
            else:
                fields.extend(((63, 6), (run - 6, 8)))
            zero_count -= run
        fields.append((int(length_array[offset]), 6))
        cursor = int(offset) + 1
    zero_count = int(length_array.size) - cursor
    while zero_count:
        run = min(zero_count, 261)
        if run == 1:
            fields.append((0, 6))
        elif run <= 5:
            fields.append((run + 57, 6))
        else:
            fields.extend(((63, 6), (run - 6, 8)))
        zero_count -= run
    return _pack_dwa_bit_fields(fields)[0]


def _canonical_dwa_code_array(lengths: np.ndarray) -> np.ndarray:
    observed_offsets = np.flatnonzero(lengths)
    observed_lengths = lengths[observed_offsets].astype(np.int64, copy=False)
    counts = np.bincount(observed_lengths, minlength=_DWA_MAX_HUFFMAN_CODE_LENGTH + 1)
    maximum_length = int(observed_lengths.max())
    capacity = 1 << maximum_length
    occupancy = sum(int(counts[length]) << (maximum_length - length) for length in range(1, maximum_length + 1))
    if occupancy > capacity:
        raise _gpu_error(
            why="the DWA write-side Huffman assignment is oversubscribed",
            what=f"occupancy={occupancy}, capacity={capacity}, maximum_length={maximum_length}",
            how="build a prefix-free histogram code-length assignment before packing AC symbols",
        )
    next_codes = np.zeros(_DWA_MAX_HUFFMAN_CODE_LENGTH + 1, dtype=np.uint64)
    code = 0
    for length in range(_DWA_MAX_HUFFMAN_CODE_LENGTH, 0, -1):
        next_codes[length] = code
        code = (code + int(counts[length])) >> 1
    codes = np.zeros(lengths.size, dtype=np.uint64)
    for assigned_length_value in np.flatnonzero(counts[1:]) + 1:
        assigned_length = int(assigned_length_value)
        offsets = observed_offsets[observed_lengths == assigned_length]
        codes[offsets] = next_codes[assigned_length] + np.arange(offsets.size, dtype=np.uint64)
    return codes


def _encode_dwa_huffman_chunks_gpu(
    symbols: cp.ndarray,
    chunk_ids: cp.ndarray,
    chunk_count: int,
) -> tuple[cp.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    values = cp.ascontiguousarray(symbols, dtype=cp.uint16).reshape(-1)
    value_chunks = cp.ascontiguousarray(chunk_ids, dtype=cp.int32).reshape(-1)
    if int(value_chunks.size) != int(values.size):
        raise _gpu_error(
            why="the batched DWA AC symbols and chunk ownership have different lengths",
            what=f"symbols={values.size}, chunk_ids={value_chunks.size}",
            how="assign exactly one chunk index to every generated AC symbol",
        )
    if chunk_count < 1:
        raise _gpu_error(
            why="the batched DWA AC stream declares no output chunks",
            what=f"chunk_count={chunk_count}, symbol_count={values.size}",
            how="declare at least one output chunk and use zero-based ownership indices",
        )
    if not int(values.size):
        zeros = (0,) * chunk_count
        return cp.empty(0, dtype=cp.uint8), zeros, zeros, zeros

    histogram_indices = value_chunks.astype(cp.int64) * np.int64(1 << 16) + values.astype(cp.int64)
    histograms = cp.bincount(histogram_indices, minlength=chunk_count * (1 << 16)).reshape(chunk_count, 1 << 16)
    histogram_host = histograms.get().astype(np.int64, copy=False)
    symbol_counts = tuple(int(value) for value in histogram_host.sum(axis=1))
    code_values_host = np.zeros((chunk_count, _DWA_MAX_HUFFMAN_SYMBOL + 1), dtype=np.uint64)
    code_lengths_host = np.zeros((chunk_count, _DWA_MAX_HUFFMAN_SYMBOL + 1), dtype=np.uint8)
    minimum_symbols = np.zeros(chunk_count, dtype=np.uint32)
    repeat_symbols = np.zeros(chunk_count, dtype=np.uint32)
    packed_tables: list[bytes] = []
    for chunk_index, histogram in enumerate(histogram_host):
        observed = np.flatnonzero(histogram)
        if not observed.size:
            packed_tables.append(b"")
            continue
        frequencies = {int(symbol): int(histogram[symbol]) for symbol in observed}
        minimum_symbol = int(observed[0])
        maximum_symbol = int(observed[-1])
        pseudo_symbol = maximum_symbol + 1
        if pseudo_symbol > _DWA_MAX_HUFFMAN_SYMBOL:
            raise _gpu_error(
                why="the DWA AC histogram leaves no in-range repeat pseudo-symbol",
                what=f"chunk={chunk_index}, maximum_symbol={maximum_symbol}",
                how="emit finite HALF coefficients and valid DWA zero-run symbols below 0xffff",
            )
        length_by_symbol = _build_dwa_huffman_lengths(frequencies, pseudo_symbol)
        code_lengths = np.zeros(pseudo_symbol - minimum_symbol + 1, dtype=np.uint8)
        assigned_symbols = np.fromiter(length_by_symbol, dtype=np.int64, count=len(length_by_symbol))
        assigned_lengths = np.fromiter(length_by_symbol.values(), dtype=np.uint8, count=len(length_by_symbol))
        code_lengths[assigned_symbols - minimum_symbol] = assigned_lengths
        code_values_host[chunk_index, minimum_symbol : pseudo_symbol + 1] = _canonical_dwa_code_array(code_lengths)
        code_lengths_host[chunk_index, minimum_symbol : pseudo_symbol + 1] = code_lengths
        minimum_symbols[chunk_index] = minimum_symbol
        repeat_symbols[chunk_index] = pseudo_symbol
        packed_tables.append(_pack_dwa_huffman_lengths(code_lengths))

    device_codes = cp.asarray(code_values_host)
    device_lengths = cp.asarray(code_lengths_host)
    device_repeat_symbols = cp.asarray(repeat_symbols)
    boundaries = cp.empty(int(values.size), dtype=cp.bool_)
    boundaries[0] = True
    if int(values.size) > 1:
        boundaries[1:] = (value_chunks[1:] != value_chunks[:-1]) | (values[1:] != values[:-1])
    run_starts = cp.flatnonzero(boundaries).astype(cp.int64)
    run_ends = cp.concatenate((run_starts[1:], cp.asarray((int(values.size),), dtype=cp.int64)))
    run_lengths = run_ends - run_starts
    split_counts = (run_lengths + np.int64(255)) // np.int64(256)
    run_ids = cp.repeat(cp.arange(int(run_starts.size), dtype=cp.int64), split_counts)
    split_bases = cp.cumsum(split_counts, dtype=cp.int64) - split_counts
    split_indices = cp.arange(int(run_ids.size), dtype=cp.int64) - cp.repeat(split_bases, split_counts)
    segment_starts = run_starts[run_ids] + split_indices * np.int64(256)
    segment_lengths = cp.minimum(run_lengths[run_ids] - split_indices * np.int64(256), np.int64(256)).astype(cp.uint16)
    segment_chunks = value_chunks[segment_starts]
    segment_symbols = values[segment_starts].astype(cp.int64)
    literal_codes = device_codes[segment_chunks, segment_symbols]
    literal_lengths = device_lengths[segment_chunks, segment_symbols]
    repeat_codes = device_codes[segment_chunks, device_repeat_symbols[segment_chunks]]
    repeat_lengths = device_lengths[segment_chunks, device_repeat_symbols[segment_chunks]]
    additional = segment_lengths.astype(cp.uint64) - np.uint64(1)
    literal_bits = literal_lengths.astype(cp.uint64)
    use_repeat = (additional > 0) & (
        literal_bits + repeat_lengths.astype(cp.uint64) + np.uint64(8) < literal_bits * additional
    )
    segment_bit_lengths = cp.where(
        use_repeat,
        literal_bits + repeat_lengths.astype(cp.uint64) + np.uint64(8),
        literal_bits * segment_lengths.astype(cp.uint64),
    )
    bit_counts = cp.zeros(chunk_count, dtype=cp.uint64)
    cp.add.at(bit_counts, segment_chunks, segment_bit_lengths)
    bit_count_host = bit_counts.get().astype(np.uint64, copy=False)
    data_sizes = tuple((int(bit_count) + 7) // 8 for bit_count in bit_count_host)
    allocated_sizes = tuple((size + 7) & ~7 for size in data_sizes)
    data_offsets = _prefix_offsets(allocated_sizes)
    encoded_data = cp.zeros(sum(allocated_sizes), dtype=cp.uint8)
    global_bit_offsets = cp.cumsum(segment_bit_lengths, dtype=cp.uint64) - segment_bit_lengths
    chunk_bit_bases = cp.zeros(chunk_count, dtype=cp.uint64)
    if chunk_count > 1:
        chunk_bit_bases[1:] = cp.cumsum(bit_counts[:-1], dtype=cp.uint64)
    local_bit_offsets = global_bit_offsets - chunk_bit_bases[segment_chunks]
    segment_count = int(segment_starts.size)
    _dwa_huffman_pack_kernel()(
        ((segment_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            literal_codes,
            literal_lengths,
            repeat_codes,
            repeat_lengths,
            segment_lengths,
            use_repeat.astype(cp.uint8),
            segment_chunks,
            local_bit_offsets,
            _device_i64(data_offsets),
            encoded_data.view(cp.uint64),
            np.int64(segment_count),
        ),
    )

    prefixes = tuple(
        struct.pack(
            "<IIIII",
            int(minimum_symbols[index]),
            int(repeat_symbols[index]),
            len(packed_tables[index]),
            int(bit_count_host[index]),
            0,
        )
        + packed_tables[index]
        for index in range(chunk_count)
    )
    prefix_sizes = tuple(len(prefix) for prefix in prefixes)
    prefix_offsets = _prefix_offsets(prefix_sizes)
    prefix_blob = cp.asarray(np.frombuffer(b"".join(prefixes), dtype=np.uint8))
    output_sizes = tuple(
        prefix_size + data_size for prefix_size, data_size in zip(prefix_sizes, data_sizes, strict=True)
    )
    output_offsets = _prefix_offsets(output_sizes)
    output = cp.empty(sum(output_sizes), dtype=cp.uint8)
    _dwa_assemble_huffman_kernel()(
        (_maximum_block_count(output_sizes), chunk_count),
        (_EXR_THREADS_PER_BLOCK,),
        (
            prefix_blob,
            _device_i64(prefix_offsets),
            _device_i64(prefix_sizes),
            encoded_data,
            _device_i64(data_offsets),
            _device_i64(data_sizes),
            output,
            _device_i64(output_offsets),
            np.int32(chunk_count),
        ),
    )
    return output, output_offsets, output_sizes, symbol_counts


def _decode_dwa_huffman_host(
    payload: bytes,
    table: _DwaHuffmanTable,
    *,
    expected_count: int,
) -> np.ndarray:
    """Decode one canonical DWA Huffman stream, including its repeat pseudo-symbol."""
    if table.data_span.start < 0 or table.data_span.end > len(payload):
        raise _gpu_error(
            why="the DWA Huffman encoded-data span lies outside the supplied payload",
            what=f"span={table.data_span.start}:{table.data_span.end}, payload_size={len(payload)}",
            how="decode with the complete container payload and its matching Huffman descriptor",
        )
    encoded = payload[table.data_span.start : table.data_span.end]
    codes_by_length: dict[int, dict[int, int]] = {}
    for item in table.codes:
        codes_by_length.setdefault(item.length, {})[item.code] = item.symbol

    output = np.empty(expected_count, dtype=np.uint16)
    output_count = 0
    bit_offset = 0
    while bit_offset < table.data_bit_count:
        code = 0
        symbol: int | None = None
        for length in range(1, _DWA_MAX_HUFFMAN_CODE_LENGTH + 1):
            if bit_offset >= table.data_bit_count:
                raise _gpu_error(
                    why="the DWA Huffman bitstream ends within a symbol",
                    what=f"decoded={output_count}, expected={expected_count}, bit_offset={bit_offset}",
                    how="provide every bit of the final canonical Huffman code",
                )
            code = (code << 1) | ((encoded[bit_offset // 8] >> (7 - bit_offset % 8)) & 1)
            bit_offset += 1
            symbol = codes_by_length.get(length, {}).get(code)
            if symbol is not None:
                break
        if symbol is None:
            raise _gpu_error(
                why="the DWA Huffman bitstream contains no code from its canonical table",
                what=f"decoded={output_count}, bit_offset={bit_offset}",
                how="encode each AC symbol with a declared canonical code",
            )
        if symbol == table.maximum_symbol:
            if output_count == 0 or bit_offset + 8 > table.data_bit_count:
                raise _gpu_error(
                    why="the DWA Huffman repeat symbol has no previous symbol or complete eight-bit count",
                    what=f"decoded={output_count}, bit_offset={bit_offset}, data_bits={table.data_bit_count}",
                    how="place a repeat only after a literal and append its complete count byte",
                )
            repeat_count = 0
            for _ in range(8):
                repeat_count = (repeat_count << 1) | ((encoded[bit_offset // 8] >> (7 - bit_offset % 8)) & 1)
                bit_offset += 1
            if repeat_count == 0 or output_count + repeat_count > expected_count:
                raise _gpu_error(
                    why="the DWA Huffman repeat count is zero or exceeds the declared AC element count",
                    what=f"decoded={output_count}, repeat={repeat_count}, expected={expected_count}",
                    how="encode between one and 255 additional copies within the declared output count",
                )
            output[output_count : output_count + repeat_count] = output[output_count - 1]
            output_count += repeat_count
        else:
            if output_count >= expected_count:
                raise _gpu_error(
                    why="the DWA Huffman bitstream expands beyond the declared AC element count",
                    what=f"decoded={output_count}, expected={expected_count}, symbol={symbol}",
                    how="make the encoded symbols and generic repeats match the declared count",
                )
            output[output_count] = symbol
            output_count += 1

    if output_count != expected_count:
        raise _gpu_error(
            why="the DWA Huffman bitstream expands to a different AC element count",
            what=f"decoded={output_count}, expected={expected_count}, consumed_bits={bit_offset}",
            how="make the encoded symbol stream match the leader AC count exactly",
        )
    return output


def _decode_dwa_huffman_gpu(
    payload: cp.ndarray,
    *,
    data_offsets: Sequence[int],
    tables: Sequence[_DwaHuffmanTable],
    output_counts: Sequence[int],
    record_labels: Sequence[int] | None = None,
) -> cp.ndarray:
    """Decode independent DWA streams through speculative bit nodes and exact parallel segments."""
    record_count = len(tables)
    if len(data_offsets) != record_count or len(output_counts) != record_count:
        raise _gpu_error(
            why="the batched GPU DWA Huffman descriptors have inconsistent record counts",
            what=(f"tables={record_count}, data_offsets={len(data_offsets)}, output_counts={len(output_counts)}"),
            how="provide one encoded-data offset and declared output count for every canonical table",
        )
    labels = tuple(range(record_count)) if record_labels is None else tuple(record_labels)
    if len(labels) != record_count:
        raise _gpu_error(
            why="the batched GPU DWA Huffman labels do not match the decode records",
            what=f"labels={len(labels)}, records={record_count}",
            how="provide one diagnostic label for every canonical table",
        )
    count_array = np.asarray(tuple(output_counts), dtype=np.int64)
    if np.any(count_array < 0):
        raise _gpu_error(
            why="the batched GPU DWA Huffman output count is negative",
            what=f"output_counts={tuple(int(value) for value in count_array)!r}",
            how="declare a non-negative AC element count for every stream",
        )
    output_offsets = np.zeros(record_count, dtype=np.int64)
    if record_count > 1:
        output_offsets[1:] = np.cumsum(count_array[:-1], dtype=np.int64)
    output = cp.empty(max(int(count_array.sum()), 1), dtype=cp.uint16)
    if not record_count:
        return output[:0]

    data_bit_counts = np.fromiter((table.data_bit_count for table in tables), dtype=np.int64, count=record_count)
    repeat_symbols = np.fromiter((table.maximum_symbol for table in tables), dtype=np.uint32, count=record_count)
    minimum_symbols = np.fromiter((table.minimum_symbol for table in tables), dtype=np.uint32, count=record_count)
    symbol_counts = repeat_symbols.astype(np.int64) - minimum_symbols.astype(np.int64) + 1
    symbol_bases = np.zeros(record_count, dtype=np.int64)
    if record_count > 1:
        symbol_bases[1:] = np.cumsum(symbol_counts[:-1], dtype=np.int64)
    symbol_capacity = int(symbol_counts.sum())
    table_byte_counts = np.fromiter((table.table_byte_count for table in tables), dtype=np.int32, count=record_count)
    table_offsets = np.asarray(tuple(data_offsets), dtype=np.int64) - table_byte_counts

    device_symbol_bases = cp.asarray(symbol_bases)
    device_code_counts = cp.zeros(record_count, dtype=cp.int32)
    device_length_counts = cp.zeros((record_count, _DWA_MAX_HUFFMAN_CODE_LENGTH + 1), dtype=cp.int32)
    device_length_offsets = cp.empty((record_count, _DWA_MAX_HUFFMAN_CODE_LENGTH + 1), dtype=cp.int64)
    sparse_symbols = cp.empty(max(symbol_capacity, 1), dtype=cp.uint32)
    sparse_lengths = cp.empty(max(symbol_capacity, 1), dtype=cp.uint8)
    device_codes = cp.empty(max(symbol_capacity, 1), dtype=cp.uint64)
    device_symbols = cp.empty(max(symbol_capacity, 1), dtype=cp.uint32)
    ordered_lengths = cp.empty(max(symbol_capacity, 1), dtype=cp.uint8)
    table_status = cp.zeros(record_count, dtype=cp.int32)
    table_grid = (record_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _dwa_huffman_parse_tables_kernel()(
        (table_grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            payload,
            cp.asarray(table_offsets),
            cp.asarray(table_byte_counts),
            cp.asarray(minimum_symbols),
            cp.asarray(repeat_symbols),
            device_symbol_bases,
            sparse_symbols,
            sparse_lengths,
            device_code_counts,
            device_length_counts,
            table_status,
            np.int32(record_count),
        ),
    )
    _dwa_huffman_build_codes_kernel()(
        (table_grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            device_symbol_bases,
            sparse_symbols,
            sparse_lengths,
            device_code_counts,
            device_length_counts,
            device_length_offsets,
            device_codes,
            device_symbols,
            ordered_lengths,
            table_status,
            np.int32(record_count),
        ),
    )
    table_status_host = table_status.get()
    failed_tables = np.flatnonzero(table_status_host)
    if failed_tables.size:
        table_failures = tuple((labels[int(index)], int(table_status_host[int(index)])) for index in failed_tables)
        raise _gpu_error(
            why="the GPU DWA Huffman table parser rejected a packed length table or canonical assignment",
            what=f"record_status={table_failures!r}",
            how="provide a complete, padding-clean, prefix-free DWA canonical length table",
        )
    del sparse_symbols, sparse_lengths, table_status
    device_lookahead_symbols = cp.zeros(record_count * _DWA_HUFFMAN_LOOKAHEAD_SIZE, dtype=cp.uint32)
    device_lookahead_lengths = cp.zeros(record_count * _DWA_HUFFMAN_LOOKAHEAD_SIZE, dtype=cp.uint8)
    maximum_symbol_count = int(symbol_counts.max(initial=0))
    if maximum_symbol_count:
        lookahead_grid_x = (maximum_symbol_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _dwa_huffman_build_lookahead_kernel()(
            (lookahead_grid_x, record_count),
            (_EXR_THREADS_PER_BLOCK,),
            (
                device_symbol_bases,
                device_code_counts,
                device_codes,
                device_symbols,
                ordered_lengths,
                device_lookahead_symbols,
                device_lookahead_lengths,
                np.int32(record_count),
            ),
        )
    del ordered_lengths

    segment_capacities = np.maximum(
        1,
        (count_array + _DWA_HUFFMAN_SEGMENT_TOKEN_COUNT - 1) // _DWA_HUFFMAN_SEGMENT_TOKEN_COUNT,
    ).astype(np.int32)
    segment_bases = np.zeros(record_count, dtype=np.int64)
    if record_count > 1:
        segment_bases[1:] = np.cumsum(segment_capacities[:-1], dtype=np.int64)
    allocated_segment_count = int(segment_capacities.sum())
    segment_records = np.repeat(np.arange(record_count, dtype=np.int32), segment_capacities)
    node_counts = data_bit_counts + 1
    node_bases_i64 = np.zeros(record_count, dtype=np.int64)
    if record_count > 1:
        node_bases_i64[1:] = np.cumsum(node_counts[:-1], dtype=np.int64)
    node_count = int(node_counts.sum())
    if node_count >= 1 << 32:
        raise _gpu_error(
            why="the batched GPU DWA Huffman candidate graph exceeds its 32-bit node address space",
            what=f"nodes={node_count}, records={record_count}",
            how="decode a smaller image or split the DWA chunk batch before candidate construction",
        )
    node_bases = node_bases_i64.astype(np.uint32)

    device_data_offsets = cp.asarray(np.asarray(tuple(data_offsets), dtype=np.int64))
    device_data_bits = cp.asarray(data_bit_counts)
    device_repeat_symbols = cp.asarray(repeat_symbols)
    device_output_offsets = cp.asarray(output_offsets)
    device_output_counts = cp.asarray(count_array)
    device_segment_bases = cp.asarray(segment_bases)
    segment_bits = cp.empty(allocated_segment_count, dtype=cp.int64)
    segment_outputs = cp.empty(allocated_segment_count, dtype=cp.int64)
    token_nodes = cp.empty(node_count, dtype=cp.uint32)
    token_outputs = cp.empty(node_count, dtype=cp.uint32)
    literal_flags = cp.empty(node_count, dtype=cp.uint8)
    node_grid = (node_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _dwa_huffman_candidates_kernel()(
        (node_grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            payload,
            device_data_offsets,
            device_data_bits,
            device_length_offsets,
            device_length_counts,
            device_codes,
            device_symbols,
            device_lookahead_symbols,
            device_lookahead_lengths,
            device_repeat_symbols,
            cp.asarray(node_bases),
            token_nodes,
            token_outputs,
            literal_flags,
            np.int32(record_count),
            np.uint32(node_count),
        ),
    )
    jump_nodes = cp.empty_like(token_nodes)
    jump_outputs = cp.empty_like(token_outputs)
    source_nodes = token_nodes
    source_outputs = token_outputs
    destination_nodes = jump_nodes
    destination_outputs = jump_outputs
    for _ in range(_DWA_HUFFMAN_SEGMENT_TOKEN_COUNT.bit_length() - 1):
        _dwa_huffman_jump_kernel()(
            (node_grid,),
            (_EXR_THREADS_PER_BLOCK,),
            (
                source_nodes,
                source_outputs,
                destination_nodes,
                destination_outputs,
                np.uint32(node_count),
            ),
        )
        source_nodes, destination_nodes = destination_nodes, source_nodes
        source_outputs, destination_outputs = destination_outputs, source_outputs

    actual_segment_counts = cp.empty(record_count, dtype=cp.int32)
    boundary_status = cp.zeros(record_count, dtype=cp.int32)
    _dwa_huffman_token_boundaries_kernel()(
        (table_grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            token_nodes,
            token_outputs,
            literal_flags,
            source_nodes,
            source_outputs,
            cp.asarray(node_bases),
            device_data_bits,
            device_output_offsets,
            device_output_counts,
            device_segment_bases,
            cp.asarray(segment_capacities),
            segment_bits,
            segment_outputs,
            actual_segment_counts,
            boundary_status,
            np.int32(record_count),
        ),
    )
    boundary_status_host = boundary_status.get()
    failed_boundaries = np.flatnonzero(boundary_status_host)
    if failed_boundaries.size:
        boundary_failures = tuple(
            (labels[int(index)], int(boundary_status_host[int(index)])) for index in failed_boundaries
        )
        raise _gpu_error(
            why="the GPU DWA Huffman token graph could not produce exact parallel segment boundaries",
            what=f"record_status={boundary_failures!r}",
            how="make the canonical token path from bit zero reach the declared AC output boundary",
        )
    del jump_nodes, jump_outputs, token_nodes, token_outputs, literal_flags, boundary_status

    segment_status = cp.zeros(allocated_segment_count, dtype=cp.int32)
    _dwa_huffman_segment_kernel()(
        ((allocated_segment_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            payload,
            device_data_offsets,
            device_data_bits,
            device_length_offsets,
            device_length_counts,
            device_codes,
            device_symbols,
            device_lookahead_symbols,
            device_lookahead_lengths,
            device_repeat_symbols,
            output,
            device_output_offsets,
            device_output_counts,
            cp.asarray(segment_records),
            device_segment_bases,
            segment_bits,
            segment_outputs,
            actual_segment_counts,
            segment_status,
            np.int64(allocated_segment_count),
        ),
    )
    segment_status_host = segment_status.get()
    failed_segments = np.flatnonzero(segment_status_host)
    if failed_segments.size:
        segment_failures = tuple(
            (
                labels[int(segment_records[int(index)])],
                int(index - segment_bases[int(segment_records[int(index)])]),
                int(segment_status_host[int(index)]),
            )
            for index in failed_segments
        )
        raise _gpu_error(
            why="the segmented GPU DWA Huffman decoder rejected a record boundary or output span",
            what=f"record_segment_status={segment_failures!r}",
            how="make every canonical token, repeat run, bit span, and declared AC output boundary agree",
        )
    return output[: int(count_array.sum())]


def _dwa_coefficient_block_spans(
    ac_symbols: cp.ndarray,
    ac_offsets: np.ndarray,
    ac_counts: np.ndarray,
    *,
    expected_block_count: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Locate each DWA coefficient block with parallel prefix/reset operations."""
    block_ac_starts = cp.empty(expected_block_count, dtype=cp.int64)
    dwa_runs = (ac_symbols & np.uint16(0xFF00)) == np.uint16(0xFF00)
    zero_counts = (ac_symbols & np.uint16(0x00FF)).astype(cp.uint32)
    end_of_block = dwa_runs & (zero_counts == 0)
    coefficient_steps = cp.where(dwa_runs, zero_counts, cp.uint32(1))
    cumulative_steps = cp.cumsum(coefficient_steps, dtype=cp.uint64)
    nonempty_chunk_starts = ac_offsets[ac_counts > 0]
    eob_reset_indices = cp.flatnonzero(end_of_block).astype(cp.int64)
    chunk_reset_indices = cp.asarray(nonempty_chunk_starts[1:])
    reset_indices = cp.concatenate((eob_reset_indices, chunk_reset_indices))
    reset_values = cp.concatenate((cumulative_steps[eob_reset_indices], cumulative_steps[chunk_reset_indices - 1]))
    if int(reset_indices.size):
        reset_order = cp.argsort(reset_indices)
        reset_indices = reset_indices[reset_order]
        reset_values = reset_values[reset_order]
        reset_slots = (
            cp.searchsorted(
                reset_indices,
                cp.arange(int(ac_symbols.size), dtype=cp.int64),
                side="right",
            )
            - 1
        )
        prior_resets = cp.where(reset_slots >= 0, reset_values[cp.maximum(reset_slots, 0)], cp.uint64(0))
    else:
        prior_resets = cp.zeros_like(cumulative_steps)
    steps_within_eob_run = cumulative_steps - prior_resets
    steps_before_symbol = steps_within_eob_run - coefficient_steps
    invalid_block_crossings = (~end_of_block) & (
        (steps_before_symbol % np.uint64(63)) + coefficient_steps > np.uint64(63)
    )
    block_ends = end_of_block | ((coefficient_steps != 0) & (steps_within_eob_run % np.uint64(63) == 0))
    block_ac_ends = cp.flatnonzero(block_ends).astype(cp.int64) + np.int64(1)
    decoded_block_count = int(block_ac_ends.size)
    if decoded_block_count:
        final_ac_end, invalid_crossing_count = map(
            int,
            cp.stack((block_ac_ends[-1], cp.count_nonzero(invalid_block_crossings))).get(),
        )
    else:
        final_ac_end = 0
        invalid_crossing_count = int(cp.count_nonzero(invalid_block_crossings).get())
    if decoded_block_count != expected_block_count or final_ac_end != int(ac_symbols.size) or invalid_crossing_count:
        raise _gpu_error(
            why="the GPU DWA AC run expansion did not end at a complete coefficient-block boundary",
            what=(
                f"decoded_blocks={decoded_block_count}, expected_blocks={expected_block_count}, "
                f"consumed_symbols={final_ac_end}, expected_symbols={ac_symbols.size}, "
                f"invalid_block_crossings={invalid_crossing_count}"
            ),
            how="provide valid literals, zero runs, and EOB symbols for every declared lossy block",
        )
    block_ac_starts[0] = 0
    if expected_block_count > 1:
        block_ac_starts[1:] = block_ac_ends[:-1]
    return block_ac_starts, block_ac_ends


def _decompress_dwa_zlib_host(payload: bytes, *, expected_size: int, stream_name: str, chunk_y: int) -> bytes:
    decompressor = zlib.decompressobj()
    try:
        decoded = decompressor.decompress(payload, expected_size + 1)
        decoded += decompressor.flush()
    except zlib.error as error:
        raise _gpu_error(
            why=f"the DWA {stream_name} zlib wrapper or Deflate payload is invalid",
            what=f"chunk_y={chunk_y}, compressed_size={len(payload)}, error={error}",
            how=f"encode one complete RFC 1950 stream for the DWA {stream_name} substream",
        ) from error
    if (
        len(decoded) != expected_size
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or not decompressor.eof
    ):
        raise _gpu_error(
            why=f"the DWA {stream_name} zlib output or stream end differs from its declaration",
            what=(
                f"chunk_y={chunk_y}, decoded={len(decoded)}, expected={expected_size}, "
                f"unconsumed={len(decompressor.unconsumed_tail)}, trailing={len(decompressor.unused_data)}"
            ),
            how=f"make the DWA {stream_name} payload and declared output size describe one exact zlib stream",
        )
    return decoded


def _decode_dwa_byte_rle_host(payload: bytes, *, expected_size: int, chunk_y: int) -> bytes:
    output = bytearray(expected_size)
    source_offset = 0
    output_offset = 0
    while source_offset < len(payload):
        control = int.from_bytes(payload[source_offset : source_offset + 1], "little", signed=True)
        source_offset += 1
        if control < 0:
            count = -control
            end = source_offset + count
            if end > len(payload) or output_offset + count > expected_size:
                raise _gpu_error(
                    why="the DWA byte RLE literal packet is truncated or exceeds its raw output",
                    what=(
                        f"chunk_y={chunk_y}, source_offset={source_offset - 1}, count={count}, "
                        f"decoded={output_offset}, expected={expected_size}"
                    ),
                    how="keep every literal packet within the encoded and declared raw RLE sizes",
                )
            output[output_offset : output_offset + count] = payload[source_offset:end]
            source_offset = end
            output_offset += count
        else:
            count = control + 1
            if source_offset >= len(payload) or output_offset + count > expected_size:
                raise _gpu_error(
                    why="the DWA byte RLE repeat packet is truncated or exceeds its raw output",
                    what=(
                        f"chunk_y={chunk_y}, source_offset={source_offset - 1}, count={count}, "
                        f"decoded={output_offset}, expected={expected_size}"
                    ),
                    how="append one repeat value and keep its run within the declared raw size",
                )
            output[output_offset : output_offset + count] = bytes((payload[source_offset],)) * count
            source_offset += 1
            output_offset += count
    if output_offset != expected_size:
        raise _gpu_error(
            why="the DWA byte RLE packets expand to a different raw size",
            what=f"chunk_y={chunk_y}, decoded={output_offset}, expected={expected_size}",
            how="make the byte RLE packet counts cover the declared raw stream exactly",
        )
    return bytes(output)


def _dwa_sample_array(payload: bytes, channel: _ExrChannel, *, sample_count: int) -> np.ndarray:
    dtype = {0: "<u4", 1: "<f2", 2: "<f4"}[channel.pixel_type]
    return cast(np.ndarray, np.frombuffer(payload, dtype=dtype, count=sample_count).reshape(-1))


def _inverse_dwa_transfer_host(values: np.ndarray) -> np.ndarray:
    nonlinear = np.asarray(values, dtype=np.float16).astype(np.float32)
    magnitude = np.abs(nonlinear)
    with np.errstate(over="ignore", invalid="ignore"):
        linear = np.where(
            magnitude <= np.float32(1.0),
            np.power(magnitude, np.float32(2.2)),
            np.exp(np.float32(2.2) * (magnitude - np.float32(1.0))),
        )
    linear = np.copysign(linear, nonlinear)
    linear = np.where(np.isfinite(nonlinear), linear, np.float32(0.0))
    return linear.astype(np.float16)


def _inverse_dwa_dct_host(coefficients: np.ndarray) -> np.ndarray:
    natural = np.asarray(coefficients, dtype=np.uint16)[_DWA_NATURAL_TO_ZIGZAG].view(np.float16).astype(np.float32)
    block = natural.reshape(8, 8)
    return cast(np.ndarray, (_DWA_INVERSE_BASIS_HOST @ block @ _DWA_INVERSE_BASIS_HOST.T).astype(np.float32))


def _decode_dwa_ac_block_host(symbols: np.ndarray, offset: int, dc: np.uint16) -> tuple[np.ndarray, int]:
    coefficients = np.zeros(64, dtype=np.uint16)
    coefficients[0] = dc
    position = 1
    while position < 64:
        if offset >= symbols.size:
            raise _gpu_error(
                why="the DWA AC symbol stream ends within an 8x8 coefficient block",
                what=f"symbol_offset={offset}, coefficient={position}, available={symbols.size}",
                how="provide a literal, zero run, or EOB that completes every lossy block",
            )
        symbol = int(symbols[offset])
        offset += 1
        if symbol & 0xFF00 == 0xFF00:
            zero_count = symbol & 0xFF
            if zero_count == 0:
                position = 64
            elif position + zero_count > 64:
                raise _gpu_error(
                    why="a DWA AC zero run exceeds its 8x8 coefficient block",
                    what=f"coefficient={position}, zero_run={zero_count}, symbol=0x{symbol:04x}",
                    how="limit each zero run to the remaining AC coefficient positions",
                )
            else:
                position += zero_count
        else:
            coefficients[position] = symbol
            position += 1
    return coefficients, offset


def _dwa_lossy_units(
    channels: Sequence[_ExrChannel],
    layout: _DwaChannelLayout,
) -> tuple[tuple[_ExrChannel, ...], ...]:
    channels_by_name = {channel.name: channel for channel in channels}
    grouped_names = {name for group in layout.csc_groups for name in group.channel_names}
    units: list[tuple[_ExrChannel, ...]] = [
        tuple(channels_by_name[name] for name in group.channel_names) for group in layout.csc_groups
    ]
    descriptors = {descriptor.name: descriptor for descriptor in layout.channels}
    units.extend(
        (channel,)
        for channel in channels
        if descriptors[channel.name].scheme == "lossy_dct" and channel.name not in grouped_names
    )
    return tuple(units)


def _decode_dwa_lossy_host(
    channels: Sequence[_ExrChannel],
    descriptor: _DwaChunkDescriptor,
    ac_symbols: np.ndarray,
    dc_values: np.ndarray,
    *,
    chunk_y: int,
) -> dict[str, np.ndarray]:
    if descriptor.channel_layout is None:
        raise _gpu_error(
            why="the DWA lossy decoder received no channel ownership layout",
            what=f"chunk_y={chunk_y}",
            how="decode only a version-2 chunk with parsed channel rules",
        )
    geometry = descriptor.geometry
    block_count = geometry.block_columns * geometry.block_rows
    output: dict[str, np.ndarray] = {}
    ac_offset = 0
    dc_offset = 0
    for unit in _dwa_lossy_units(channels, descriptor.channel_layout):
        component_blocks = np.empty((len(unit), block_count, 8, 8), dtype=np.float32)
        for block_index in range(block_count):
            for component in range(len(unit)):
                dc_index = dc_offset + component * block_count + block_index
                if dc_index >= dc_values.size:
                    raise _gpu_error(
                        why="the DWA DC stream ends before its channel block ownership",
                        what=f"chunk_y={chunk_y}, dc_index={dc_index}, available={dc_values.size}",
                        how="provide one DC coefficient for every lossy channel block",
                    )
                coefficients, ac_offset = _decode_dwa_ac_block_host(
                    ac_symbols, ac_offset, np.uint16(dc_values[dc_index])
                )
                component_blocks[component, block_index] = _inverse_dwa_dct_host(coefficients)
        dc_offset += len(unit) * block_count
        if len(unit) == 3:
            y_plane = component_blocks[0].copy()
            cb_plane = component_blocks[1].copy()
            cr_plane = component_blocks[2].copy()
            component_blocks[0] = y_plane + np.float32(1.5747) * cr_plane
            component_blocks[1] = y_plane - np.float32(0.1873) * cb_plane - np.float32(0.4682) * cr_plane
            component_blocks[2] = y_plane + np.float32(1.8556) * cb_plane

        for component, channel in enumerate(unit):
            nonlinear = component_blocks[component].astype(np.float16)
            transfer = len(unit) == 3 or not channel.perceptually_linear
            reconstructed = _inverse_dwa_transfer_host(nonlinear) if transfer else nonlinear
            plane = np.empty((geometry.padded_height, geometry.padded_width), dtype=np.float16)
            for block_index in range(block_count):
                block_y, block_x = divmod(block_index, geometry.block_columns)
                plane[block_y * 8 : block_y * 8 + 8, block_x * 8 : block_x * 8 + 8] = reconstructed[block_index]
            width = geometry.padded_width - geometry.mirror_right
            output[channel.name] = plane[: geometry.row_count, :width]

    if ac_offset != ac_symbols.size or dc_offset != dc_values.size:
        raise _gpu_error(
            why="the DWA AC or DC streams do not end at the declared block ownership boundary",
            what=(
                f"chunk_y={chunk_y}, ac_consumed={ac_offset}, ac_declared={ac_symbols.size}, "
                f"dc_consumed={dc_offset}, dc_declared={dc_values.size}"
            ),
            how="make the coefficient counts match the classified lossy channels and 8x8 geometry",
        )
    return output


def _decode_dwa_compressed_chunk_host(
    container: _ExrContainer,
    chunk: _ExrChunk,
) -> dict[str, np.ndarray]:
    descriptor = chunk.dwa
    if descriptor is None or descriptor.leader is None or descriptor.channel_layout is None:
        raise _gpu_error(
            why="the custom DWA decoder received an incomplete compressed-chunk descriptor",
            what=f"chunk_y={chunk.y}, descriptor={descriptor!r}",
            how="decode only an eligible version-2 DWA chunk",
        )
    leader = descriptor.leader
    channels = container.parts[0].channels
    layout_by_name = {item.name: item for item in descriptor.channel_layout.channels}
    width = descriptor.geometry.padded_width - descriptor.geometry.mirror_right
    sample_count = width * descriptor.geometry.row_count
    decoded: dict[str, np.ndarray] = {}

    unknown = b""
    if leader.unknown_compressed_size:
        unknown = _decompress_dwa_zlib_host(
            container.data[descriptor.unknown_span.start : descriptor.unknown_span.end],
            expected_size=leader.unknown_uncompressed_size,
            stream_name="UNKNOWN",
            chunk_y=chunk.y,
        )
    unknown_offset = 0
    for channel in channels:
        if layout_by_name[channel.name].scheme != "unknown":
            continue
        byte_count = sample_count * channel.bytes_per_sample
        end = unknown_offset + byte_count
        if end > len(unknown):
            raise _gpu_error(
                why="the DWA UNKNOWN stream ends before its channel planes",
                what=f"chunk_y={chunk.y}, channel={channel.name!r}, end={end}, available={len(unknown)}",
                how="store every UNKNOWN channel consecutively in file-channel order",
            )
        decoded[channel.name] = _dwa_sample_array(
            unknown[unknown_offset:end], channel, sample_count=sample_count
        ).reshape(descriptor.geometry.row_count, width)
        unknown_offset = end
    if unknown_offset != len(unknown):
        raise _gpu_error(
            why="the DWA UNKNOWN stream has bytes outside its classified channels",
            what=f"chunk_y={chunk.y}, consumed={unknown_offset}, decoded={len(unknown)}",
            how="make the UNKNOWN size equal the classified channel planes",
        )

    rle_raw = b""
    if leader.rle_compressed_size:
        rle_encoded = _decompress_dwa_zlib_host(
            container.data[descriptor.rle_span.start : descriptor.rle_span.end],
            expected_size=leader.rle_uncompressed_size,
            stream_name="RLE",
            chunk_y=chunk.y,
        )
        rle_raw = _decode_dwa_byte_rle_host(rle_encoded, expected_size=leader.rle_raw_size, chunk_y=chunk.y)
    rle_offset = 0
    for channel in channels:
        if layout_by_name[channel.name].scheme != "rle":
            continue
        byte_count = sample_count * channel.bytes_per_sample
        end = rle_offset + byte_count
        if end > len(rle_raw):
            raise _gpu_error(
                why="the DWA RLE stream ends before its channel byte planes",
                what=f"chunk_y={chunk.y}, channel={channel.name!r}, end={end}, available={len(rle_raw)}",
                how="store one complete byte plane per channel sample byte",
            )
        planes = np.frombuffer(rle_raw[rle_offset:end], dtype=np.uint8).reshape(channel.bytes_per_sample, sample_count)
        interleaved = np.ascontiguousarray(planes.T).reshape(-1).tobytes()
        decoded[channel.name] = _dwa_sample_array(interleaved, channel, sample_count=sample_count).reshape(
            descriptor.geometry.row_count, width
        )
        rle_offset = end
    if rle_offset != len(rle_raw):
        raise _gpu_error(
            why="the DWA RLE raw stream has bytes outside its classified channel planes",
            what=f"chunk_y={chunk.y}, consumed={rle_offset}, decoded={len(rle_raw)}",
            how="make the RLE raw size equal the classified byte planes",
        )

    ac_symbols = np.empty(0, dtype=np.uint16)
    if leader.ac_element_count:
        if descriptor.huffman is None:
            raise _gpu_error(
                why="the eligible DWA AC stream has no parsed canonical Huffman table",
                what=f"chunk_y={chunk.y}, ac_elements={leader.ac_element_count}",
                how="use STATIC_HUFFMAN with one complete table and encoded stream",
            )
        table = descriptor.huffman
        if not table.codes:
            table = _parse_dwa_huffman_table(
                container.data[descriptor.ac_span.start : descriptor.ac_span.end],
                base_offset=descriptor.ac_span.start,
            )
        ac_symbols = _decode_dwa_huffman_host(
            container.data,
            table,
            expected_count=leader.ac_element_count,
        )
    dc_values = np.empty(0, dtype=np.uint16)
    if leader.dc_element_count:
        transformed_dc = _decompress_dwa_zlib_host(
            container.data[descriptor.dc_span.start : descriptor.dc_span.end],
            expected_size=leader.dc_element_count * 2,
            stream_name="DC",
            chunk_y=chunk.y,
        )
        restored_dc = _restore_even_odd_host(_restore_predictor_host(np.frombuffer(transformed_dc, dtype=np.uint8)))
        dc_values = restored_dc.view("<u2")
    decoded.update(_decode_dwa_lossy_host(channels, descriptor, ac_symbols, dc_values, chunk_y=chunk.y))
    return decoded


def _raw_exr_chunk_channel(
    container: _ExrContainer,
    chunk: _ExrChunk,
    channel: _ExrChannel,
) -> np.ndarray:
    width = container.data_window[2] - container.data_window[0] + 1
    row_bytes = sum(width * item.bytes_per_sample for item in container.parts[0].channels)
    channel_offset = 0
    for item in container.parts[0].channels:
        if item.name == channel.name:
            break
        channel_offset += width * item.bytes_per_sample
    payload = container.data[chunk.payload_start : chunk.payload_end]
    output = np.empty((chunk.row_count, width), dtype={1: np.float16, 2: np.float32, 0: np.uint32}[channel.pixel_type])
    for row in range(chunk.row_count):
        start = row * row_bytes + channel_offset
        end = start + width * channel.bytes_per_sample
        output[row] = _dwa_sample_array(payload[start:end], channel, sample_count=width)
    return output


def _read_exr_dwa_custom_cpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    width = container.data_window[2] - container.data_window[0] + 1
    height = container.data_window[3] - container.data_window[1] + 1
    if output_dtype == "uint32" and any(channel.pixel_type != 0 for channel in selected):
        raise _gpu_error(
            why="the native DWA UINT read lane received a non-UINT selected channel",
            what=f"channels={tuple((channel.name, channel.pixel_type) for channel in selected)!r}",
            how="route only homogeneous EXR UINT selections to the uint32 output lane",
        )
    host_dtype = {"float16": np.float16, "float32": np.float32, "uint32": np.uint32}[output_dtype]
    host_output = np.empty((height, width, len(selected)), dtype=host_dtype)
    for chunk in container.chunks:
        decoded = None if chunk.raw_stored else _decode_dwa_compressed_chunk_host(container, chunk)
        rows = slice(chunk.row_start, chunk.row_start + chunk.row_count)
        for output_channel, channel in enumerate(selected):
            plane = _raw_exr_chunk_channel(container, chunk, channel) if decoded is None else decoded[channel.name]
            host_output[rows, :, output_channel] = plane.astype(host_dtype, copy=False)
    return cast(cp.ndarray, cp.asarray(np.ascontiguousarray(host_output)))


_DWA_WRITE_GPU_SOURCE = r"""
#include <cuda_fp16.h>

static __device__ int pixtreme_dwa_floor_population(
    const unsigned int value,
    const int population,
    unsigned int* result
) {
    int found = 0;
    unsigned int best = 0;
    if (__popc(value) == population) {
        best = value;
        found = 1;
    }
    for (int pivot = 0; pivot < 15; ++pivot) {
        const unsigned int bit = 1U << pivot;
        if (!(value & bit)) continue;
        const unsigned int low_mask = (1U << (pivot + 1)) - 1U;
        const unsigned int prefix = value & ~low_mask;
        const int remaining = population - __popc(prefix);
        if (remaining < 0 || remaining > pivot) continue;
        const unsigned int suffix = remaining
            ? ((1U << remaining) - 1U) << (pivot - remaining)
            : 0U;
        const unsigned int candidate = prefix | suffix;
        if (candidate <= value && (!found || candidate > best)) {
            best = candidate;
            found = 1;
        }
    }
    *result = best;
    return found;
}

static __device__ int pixtreme_dwa_ceil_population(
    const unsigned int value,
    const int population,
    unsigned int* result
) {
    int found = 0;
    unsigned int best = 0;
    if (__popc(value) == population) {
        best = value;
        found = 1;
    }
    for (int pivot = 0; pivot < 15; ++pivot) {
        const unsigned int bit = 1U << pivot;
        if (value & bit) continue;
        const unsigned int low_mask = (1U << (pivot + 1)) - 1U;
        const unsigned int prefix = (value & ~low_mask) | bit;
        const int remaining = population - __popc(prefix);
        if (remaining < 0 || remaining > pivot) continue;
        const unsigned int suffix = remaining ? (1U << remaining) - 1U : 0U;
        const unsigned int candidate = prefix | suffix;
        if (candidate >= value && candidate <= 0x7bffU && (!found || candidate < best)) {
            best = candidate;
            found = 1;
        }
    }
    *result = best;
    return found;
}

extern "C" __global__ void pixtreme_dwa_quantize_half(
    const float* coefficients,
    const float* tolerances,
    unsigned short* output,
    const long long count
) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned short source_bits = __half_as_ushort(__float2half_rn(coefficients[index]));
    const unsigned int exponent = source_bits & 0x7c00U;
    if (exponent == 0x7c00U) {
        output[index] = source_bits;
        return;
    }
    const unsigned int sign = source_bits & 0x8000U;
    const unsigned int magnitude_bits = source_bits & 0x7fffU;
    const float source = fabsf(__half2float(__ushort_as_half(source_bits)));
    const float tolerance = tolerances[index & 63LL];
    if (source < tolerance) {
        output[index] = 0;
        return;
    }
    for (int population = 0; population <= 15; ++population) {
        unsigned int lower = 0;
        unsigned int upper = 0;
        const int has_lower = pixtreme_dwa_floor_population(magnitude_bits, population, &lower);
        const int has_upper = pixtreme_dwa_ceil_population(magnitude_bits, population, &upper);
        int found = 0;
        unsigned int best = magnitude_bits;
        float best_distance = tolerance;
        if (has_lower) {
            const float candidate = __half2float(__ushort_as_half((unsigned short)lower));
            const float distance = source - candidate;
            if (distance >= 0.0f && distance < best_distance) {
                best = lower;
                best_distance = distance;
                found = 1;
            }
        }
        if (has_upper) {
            const float candidate = __half2float(__ushort_as_half((unsigned short)upper));
            const float distance = candidate - source;
            if (distance >= 0.0f && distance < best_distance) {
                best = upper;
                best_distance = distance;
                found = 1;
            }
        }
        if (found) {
            output[index] = (unsigned short)(sign | best);
            return;
        }
    }
    output[index] = source_bits;
}

extern "C" __global__ void pixtreme_dwa_rle_ac(
    const unsigned short* zigzag,
    unsigned short* fixed_output,
    unsigned char* output_counts,
    const long long block_count
) {
    const long long block = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= block_count) return;
    const unsigned short* source = zigzag + block * 64;
    unsigned short* destination = fixed_output + block * 63;
    int coefficient = 1;
    int output_count = 0;
    while (coefficient < 64) {
        if (source[coefficient] != 0) {
            destination[output_count++] = source[coefficient++];
            continue;
        }
        int run = 1;
        while (coefficient + run < 64 && source[coefficient + run] == 0) ++run;
        if (run == 1) destination[output_count++] = 0;
        else if (coefficient + run == 64) destination[output_count++] = 0xff00U;
        else destination[output_count++] = (unsigned short)(0xff00U | (unsigned int)run);
        coefficient += run;
    }
    output_counts[block] = (unsigned char)output_count;
}

extern "C" __global__ void pixtreme_dwa_byte_rle_encode(
    const unsigned char* source,
    const long long* source_offsets,
    const long long* source_sizes,
    unsigned char* output,
    const long long* output_offsets,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count || index >= source_sizes[chunk]) return;
    const long long packet = index / 127;
    const int within_packet = (int)(index % 127);
    const long long packet_start = packet * 127;
    const int packet_size = (int)min(127LL, source_sizes[chunk] - packet_start);
    const long long destination = output_offsets[chunk] + packet * 128;
    if (within_packet == 0) output[destination] = (unsigned char)(signed char)(-packet_size);
    output[destination + 1 + within_packet] = source[source_offsets[chunk] + index];
}

static __device__ void pixtreme_dwa_write_bits_atomic(
    unsigned long long* output,
    const unsigned long long output_byte_offset,
    const unsigned long long bit_offset,
    const unsigned long long value,
    const int width
) {
    const unsigned long long first_word = bit_offset >> 6;
    unsigned long long first_mask = 0;
    unsigned long long second_mask = 0;
    for (int item = 0; item < width; ++item) {
        const unsigned long long absolute = bit_offset + (unsigned long long)item;
        const unsigned long long word = absolute >> 6;
        const unsigned int byte_lane = (unsigned int)((absolute >> 3) & 7ULL);
        const unsigned int native_bit = byte_lane * 8U + (7U - (unsigned int)(absolute & 7ULL));
        const unsigned long long bit = (value >> (width - 1 - item)) & 1ULL;
        if (word == first_word) first_mask |= bit << native_bit;
        else second_mask |= bit << native_bit;
    }
    unsigned long long* destination = output + (output_byte_offset >> 3) + first_word;
    if (first_mask) atomicOr(destination, first_mask);
    if (second_mask) atomicOr(destination + 1, second_mask);
}

extern "C" __global__ void pixtreme_dwa_huffman_pack(
    const unsigned long long* literal_codes,
    const unsigned char* literal_lengths,
    const unsigned long long* repeat_codes,
    const unsigned char* repeat_lengths,
    const unsigned short* run_lengths,
    const unsigned char* use_repeat,
    const int* chunk_ids,
    const unsigned long long* bit_offsets,
    const long long* output_offsets,
    unsigned long long* output,
    const long long segment_count
) {
    const long long segment = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= segment_count) return;
    const unsigned long long output_offset = (unsigned long long)output_offsets[chunk_ids[segment]];
    unsigned long long bit_offset = bit_offsets[segment];
    if (use_repeat[segment]) {
        pixtreme_dwa_write_bits_atomic(
            output, output_offset, bit_offset, literal_codes[segment], literal_lengths[segment]
        );
        bit_offset += literal_lengths[segment];
        pixtreme_dwa_write_bits_atomic(
            output, output_offset, bit_offset, repeat_codes[segment], repeat_lengths[segment]
        );
        bit_offset += repeat_lengths[segment];
        pixtreme_dwa_write_bits_atomic(
            output, output_offset, bit_offset, (unsigned long long)(run_lengths[segment] - 1), 8
        );
    } else {
        for (int item = 0; item < (int)run_lengths[segment]; ++item) {
            pixtreme_dwa_write_bits_atomic(
                output, output_offset, bit_offset, literal_codes[segment], literal_lengths[segment]
            );
            bit_offset += literal_lengths[segment];
        }
    }
}

extern "C" __global__ void pixtreme_dwa_assemble_huffman(
    const unsigned char* prefixes,
    const long long* prefix_offsets,
    const long long* prefix_sizes,
    const unsigned char* data,
    const long long* data_offsets,
    const long long* data_sizes,
    unsigned char* output,
    const long long* output_offsets,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count || index >= prefix_sizes[chunk] + data_sizes[chunk]) return;
    if (index < prefix_sizes[chunk]) {
        output[output_offsets[chunk] + index] = prefixes[prefix_offsets[chunk] + index];
    } else {
        const long long data_index = index - prefix_sizes[chunk];
        output[output_offsets[chunk] + index] = data[data_offsets[chunk] + data_index];
    }
}

extern "C" __global__ void pixtreme_dwa_assemble_chunks(
    const unsigned char* control,
    const long long* control_offsets,
    const long long* control_sizes,
    const unsigned char* unknown,
    const long long* unknown_offsets,
    const long long* unknown_sizes,
    const unsigned char* ac,
    const long long* ac_offsets,
    const long long* ac_sizes,
    const unsigned char* dc,
    const long long* dc_offsets,
    const long long* dc_sizes,
    const unsigned char* rle,
    const long long* rle_offsets,
    const long long* rle_sizes,
    unsigned char* output,
    const long long* output_offsets,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    if (chunk >= chunk_count) return;
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long output_size = control_sizes[chunk] + unknown_sizes[chunk] + ac_sizes[chunk] +
        dc_sizes[chunk] + rle_sizes[chunk];
    if (index >= output_size) return;
    const long long destination = output_offsets[chunk] + index;
    if (index < control_sizes[chunk]) {
        output[destination] = control[control_offsets[chunk] + index];
        return;
    }
    index -= control_sizes[chunk];
    if (index < unknown_sizes[chunk]) {
        output[destination] = unknown[unknown_offsets[chunk] + index];
        return;
    }
    index -= unknown_sizes[chunk];
    if (index < ac_sizes[chunk]) {
        output[destination] = ac[ac_offsets[chunk] + index];
        return;
    }
    index -= ac_sizes[chunk];
    if (index < dc_sizes[chunk]) {
        output[destination] = dc[dc_offsets[chunk] + index];
        return;
    }
    index -= dc_sizes[chunk];
    output[destination] = rle[rle_offsets[chunk] + index];
}
"""

_DWA_GPU_SOURCE = r"""
__device__ __forceinline__ unsigned int pixtreme_dwa_read_bits(
    const unsigned char* payload,
    const long long byte_offset,
    const long long bit_offset,
    const int width
) {
    unsigned int value = 0U;
    for (int index = 0; index < width; ++index) {
        const long long absolute = bit_offset + index;
        value = (value << 1) |
            ((payload[byte_offset + absolute / 8] >> (7 - absolute % 8)) & 1U);
    }
    return value;
}

extern "C" __global__ void pixtreme_dwa_huffman_parse_tables(
    const unsigned char* payload,
    const long long* table_offsets,
    const int* table_byte_counts,
    const unsigned int* minimum_symbols,
    const unsigned int* maximum_symbols,
    const long long* symbol_bases,
    unsigned int* sparse_symbols,
    unsigned char* sparse_lengths,
    int* code_counts,
    int* length_counts,
    int* status,
    const int record_count
) {
    const int record = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (record >= record_count) return;
    const long long table_offset = table_offsets[record];
    const long long table_bits = (long long)table_byte_counts[record] * 8LL;
    const unsigned int symbol_count = maximum_symbols[record] - minimum_symbols[record] + 1U;
    const long long symbol_base = symbol_bases[record];
    long long bit = 0;
    unsigned int produced = 0U;
    int code_count = 0;
    while (produced < symbol_count) {
        if (bit + 6LL > table_bits) { status[record] = 1; return; }
        const unsigned int token = pixtreme_dwa_read_bits(payload, table_offset, bit, 6);
        bit += 6LL;
        if (token <= 58U) {
            if (token) {
                sparse_symbols[symbol_base + code_count] = minimum_symbols[record] + produced;
                sparse_lengths[symbol_base + code_count] = (unsigned char)token;
                length_counts[(long long)record * 59LL + token] += 1;
                ++code_count;
            }
            ++produced;
            continue;
        }
        unsigned int run = token - 59U + 2U;
        if (token == 63U) {
            if (bit + 8LL > table_bits) { status[record] = 1; return; }
            run = pixtreme_dwa_read_bits(payload, table_offset, bit, 8) + 6U;
            bit += 8LL;
        }
        if (run > symbol_count - produced) { status[record] = 2; return; }
        produced += run;
    }
    const long long consumed_bytes = (bit + 7LL) / 8LL;
    if (consumed_bytes != table_byte_counts[record]) { status[record] = 3; return; }
    const int padding_bits = (int)(consumed_bytes * 8LL - bit);
    if (padding_bits && (payload[table_offset + consumed_bytes - 1LL] & ((1U << padding_bits) - 1U))) {
        status[record] = 4;
        return;
    }
    if (!code_count) { status[record] = 5; return; }
    code_counts[record] = code_count;
}

extern "C" __global__ void pixtreme_dwa_huffman_build_codes(
    const long long* symbol_bases,
    const unsigned int* sparse_symbols,
    const unsigned char* sparse_lengths,
    const int* code_counts,
    int* length_counts,
    long long* length_offsets,
    unsigned long long* codes,
    unsigned int* symbols,
    unsigned char* ordered_lengths,
    int* status,
    const int record_count
) {
    const int record = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (record >= record_count || status[record]) return;
    const long long base = symbol_bases[record];
    const int code_count = code_counts[record];
    int maximum_length = 0;
    for (int length = 1; length <= 58; ++length) {
        if (length_counts[(long long)record * 59LL + length]) maximum_length = length;
    }
    if (!maximum_length) { status[record] = 5; return; }
    const unsigned long long capacity = 1ULL << maximum_length;
    unsigned long long occupancy = 0ULL;
    for (int length = 1; length <= maximum_length; ++length) {
        occupancy += (unsigned long long)length_counts[(long long)record * 59LL + length]
            << (maximum_length - length);
    }
    if (occupancy > capacity) { status[record] = 6; return; }

    unsigned long long next_code[59];
    unsigned long long running = 0ULL;
    for (int length = 58; length >= 1; --length) {
        next_code[length] = running;
        running = (running + (unsigned long long)length_counts[(long long)record * 59LL + length]) >> 1;
    }
    int cursor = 0;
    length_offsets[(long long)record * 59LL] = base;
    for (int length = 1; length <= 58; ++length) {
        length_offsets[(long long)record * 59LL + length] = base + cursor;
        unsigned long long assigned = next_code[length];
        for (int index = 0; index < code_count; ++index) {
            if ((int)sparse_lengths[base + index] != length) continue;
            if (assigned >= (1ULL << length)) { status[record] = 7; return; }
            codes[base + cursor] = assigned++;
            symbols[base + cursor] = sparse_symbols[base + index];
            ordered_lengths[base + cursor] = (unsigned char)length;
            ++cursor;
        }
    }
    if (cursor != code_count) status[record] = 8;
}

extern "C" __global__ void pixtreme_dwa_huffman_build_lookahead(
    const long long* symbol_bases,
    const int* code_counts,
    const unsigned long long* codes,
    const unsigned int* symbols,
    const unsigned char* ordered_lengths,
    unsigned int* lookahead_symbols,
    unsigned char* lookahead_lengths,
    const int record_count
) {
    const int record = (int)blockIdx.y;
    if (record >= record_count) return;
    const int local = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (local >= code_counts[record]) return;
    const long long index = symbol_bases[record] + local;
    const int length = (int)ordered_lengths[index];
    if (length > 10) return;
    const int suffix_count = 1 << (10 - length);
    const int start = (int)(codes[index] << (10 - length));
    const long long lookahead_base = (long long)record * 1024LL;
    for (int suffix = 0; suffix < suffix_count; ++suffix) {
        lookahead_symbols[lookahead_base + start + suffix] = symbols[index];
        lookahead_lengths[lookahead_base + start + suffix] = (unsigned char)length;
    }
}

__device__ __forceinline__ int pixtreme_dwa_decode_symbol(
    const unsigned char* payload,
    const long long data_offset,
    const long long bit_limit,
    long long* bit,
    const int record,
    const long long* length_offsets,
    const int* length_counts,
    const unsigned long long* codes,
    const unsigned int* symbols,
    const unsigned int* lookahead_symbols,
    const unsigned char* lookahead_lengths,
    unsigned int* symbol
) {
    unsigned long long code = 0ULL;
    int first_fallback_length = 1;
    if (bit_limit - *bit >= 10) {
        const unsigned int prefix = pixtreme_dwa_read_bits(payload, data_offset, *bit, 10);
        const long long lookahead_index = (long long)record * 1024 + prefix;
        const int length = (int)lookahead_lengths[lookahead_index];
        if (length) {
            *symbol = lookahead_symbols[lookahead_index];
            *bit += length;
            return 0;
        }
        code = prefix;
        *bit += 10;
        first_fallback_length = 11;
    }
    for (int length = first_fallback_length; length <= 58; ++length) {
        if (*bit >= bit_limit) return 1;
        const unsigned int value =
            (payload[data_offset + *bit / 8] >> (7 - *bit % 8)) & 1U;
        ++*bit;
        code = (code << 1) | value;
        const long long table_index = (long long)record * 59 + length;
        long long low = length_offsets[table_index];
        long long high = low + length_counts[table_index];
        while (low < high) {
            const long long middle = low + (high - low) / 2;
            const unsigned long long candidate = codes[middle];
            if (candidate < code) low = middle + 1;
            else high = middle;
        }
        if (low < length_offsets[table_index] + length_counts[table_index] && codes[low] == code) {
            *symbol = symbols[low];
            return 0;
        }
    }
    return 2;
}

extern "C" __global__ void pixtreme_dwa_huffman_candidates(
    const unsigned char* payload,
    const long long* data_offsets,
    const long long* data_bits,
    const long long* length_offsets,
    const int* length_counts,
    const unsigned long long* codes,
    const unsigned int* symbols,
    const unsigned int* lookahead_symbols,
    const unsigned char* lookahead_lengths,
    const unsigned int* repeat_symbols,
    const unsigned int* node_bases,
    unsigned int* next_nodes,
    unsigned int* output_steps,
    unsigned char* literal_flags,
    const int record_count,
    const unsigned int node_count
) {
    const unsigned int node = (unsigned int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (node >= node_count) return;
    int record = 0;
    int upper = record_count;
    while (record + 1 < upper) {
        const int middle = record + (upper - record) / 2;
        if (node_bases[middle] <= node) record = middle;
        else upper = middle;
    }
    const unsigned int node_base = node_bases[record];
    const long long local_bit = (long long)(node - node_base);
    const long long bit_count = data_bits[record];
    const unsigned int terminal = node_base + (unsigned int)bit_count;
    if (local_bit == bit_count) {
        next_nodes[node] = node;
        output_steps[node] = 0U;
        literal_flags[node] = 0U;
        return;
    }
    long long bit = local_bit;
    unsigned int symbol = 0U;
    const int decode_status = pixtreme_dwa_decode_symbol(
        payload,
        data_offsets[record],
        bit_count,
        &bit,
        record,
        length_offsets,
        length_counts,
        codes,
        symbols,
        lookahead_symbols,
        lookahead_lengths,
        &symbol
    );
    if (decode_status) {
        next_nodes[node] = terminal;
        output_steps[node] = 0U;
        literal_flags[node] = 0U;
        return;
    }
    if (symbol == repeat_symbols[record]) {
        if (bit + 8 > bit_count) {
            next_nodes[node] = terminal;
            output_steps[node] = 0U;
            literal_flags[node] = 0U;
            return;
        }
        const unsigned int repeat = pixtreme_dwa_read_bits(payload, data_offsets[record], bit, 8);
        bit += 8;
        if (!repeat) {
            next_nodes[node] = terminal;
            output_steps[node] = 0U;
            literal_flags[node] = 0U;
            return;
        }
        output_steps[node] = repeat;
        literal_flags[node] = 0U;
    } else {
        if (symbol > 65535U) {
            next_nodes[node] = terminal;
            output_steps[node] = 0U;
            literal_flags[node] = 0U;
            return;
        }
        output_steps[node] = 1U;
        literal_flags[node] = 1U;
    }
    next_nodes[node] = node_base + (unsigned int)bit;
}

extern "C" __global__ void pixtreme_dwa_huffman_jump(
    const unsigned int* source_nodes,
    const unsigned int* source_outputs,
    unsigned int* destination_nodes,
    unsigned int* destination_outputs,
    const unsigned int node_count
) {
    const unsigned int node = (unsigned int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (node >= node_count) return;
    const unsigned int middle = source_nodes[node];
    destination_nodes[node] = source_nodes[middle];
    destination_outputs[node] = source_outputs[node] + source_outputs[middle];
}

extern "C" __global__ void pixtreme_dwa_huffman_token_boundaries(
    const unsigned int* token_nodes,
    const unsigned int* token_outputs,
    const unsigned char* literal_flags,
    const unsigned int* jump_nodes,
    const unsigned int* jump_outputs,
    const unsigned int* node_bases,
    const long long* data_bits,
    const long long* output_offsets,
    const long long* output_counts,
    const long long* segment_bases,
    const int* segment_capacities,
    long long* segment_bits,
    long long* segment_outputs,
    int* segment_counts,
    int* status,
    const int record_count
) {
    const int record = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (record >= record_count) return;
    const unsigned int node_base = node_bases[record];
    const unsigned int terminal = node_base + (unsigned int)data_bits[record];
    const long long segment_base = segment_bases[record];
    const int segment_capacity = segment_capacities[record];
    unsigned int node = node_base;
    unsigned long long produced = 0ULL;
    int segment_count = 1;
    segment_bits[segment_base] = 0;
    segment_outputs[segment_base] = output_offsets[record];
    while (node < terminal) {
        const unsigned int destination = jump_nodes[node];
        const unsigned int step = jump_outputs[node];
        if (destination >= terminal) break;
        if (destination == node || !step) { status[record] = 1; return; }
        node = destination;
        produced += step;
        while (node < terminal && !literal_flags[node]) {
            const unsigned int token_destination = token_nodes[node];
            const unsigned int token_step = token_outputs[node];
            if (token_destination <= node || token_destination > terminal || !token_step) {
                status[record] = 2;
                return;
            }
            node = token_destination;
            produced += token_step;
        }
        if (produced >= (unsigned long long)output_counts[record]) {
            status[record] = 3;
            return;
        }
        if (segment_count >= segment_capacity) { status[record] = 4; return; }
        segment_bits[segment_base + segment_count] = (long long)(node - node_base);
        segment_outputs[segment_base + segment_count] = output_offsets[record] + (long long)produced;
        ++segment_count;
    }
    segment_counts[record] = segment_count;
}

extern "C" __global__ void pixtreme_dwa_huffman_decode_segments(
    const unsigned char* payload,
    const long long* data_offsets,
    const long long* data_bits,
    const long long* length_offsets,
    const int* length_counts,
    const unsigned long long* codes,
    const unsigned int* symbols,
    const unsigned int* lookahead_symbols,
    const unsigned char* lookahead_lengths,
    const unsigned int* repeat_symbols,
    unsigned short* output,
    const long long* output_offsets,
    const long long* output_counts,
    const int* segment_records,
    const long long* segment_bases,
    const long long* segment_bits,
    const long long* segment_outputs,
    const int* segment_counts,
    int* status,
    const long long allocated_segment_count
) {
    const long long global_segment = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (global_segment >= allocated_segment_count) return;
    const int record = segment_records[global_segment];
    const int local_segment = (int)(global_segment - segment_bases[record]);
    const int record_segment_count = segment_counts[record];
    if (local_segment >= record_segment_count) return;
    long long bit = segment_bits[global_segment];
    const long long bit_end = local_segment + 1 < record_segment_count
        ? segment_bits[global_segment + 1]
        : data_bits[record];
    long long out = segment_outputs[global_segment];
    const long long output_end = local_segment + 1 < record_segment_count
        ? segment_outputs[global_segment + 1]
        : output_offsets[record] + output_counts[record];
    unsigned short previous = 0;
    int has_previous = 0;
    while (bit < bit_end) {
        unsigned int symbol = 0;
        const int decode_status = pixtreme_dwa_decode_symbol(
            payload,
            data_offsets[record],
            bit_end,
            &bit,
            record,
            length_offsets,
            length_counts,
            codes,
            symbols,
            lookahead_symbols,
            lookahead_lengths,
            &symbol
        );
        if (decode_status) { status[global_segment] = decode_status; return; }
        if (symbol == repeat_symbols[record]) {
            if (!has_previous || bit + 8 > bit_end) { status[global_segment] = 3; return; }
            const unsigned int repeat = pixtreme_dwa_read_bits(payload, data_offsets[record], bit, 8);
            bit += 8;
            if (repeat == 0 || out + repeat > output_end) { status[global_segment] = 4; return; }
            for (unsigned int index = 0; index < repeat; ++index) output[out++] = previous;
        } else {
            if (out >= output_end || symbol > 65535U) { status[global_segment] = 5; return; }
            previous = (unsigned short)symbol;
            has_previous = 1;
            output[out++] = previous;
        }
    }
    if (out != output_end || bit != bit_end) status[global_segment] = 6;
}

extern "C" __global__ void pixtreme_dwa_expand_coefficient_blocks(
    const unsigned short* ac_symbols,
    const unsigned short* dc_values,
    const long long* dc_indices,
    unsigned short* coefficients,
    const long long* block_ac_starts,
    const long long* block_ac_ends,
    const long long block_count
) {
    const long long block = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= block_count) return;
    unsigned short* dst = coefficients + block * 64;
    for (int index = 0; index < 64; ++index) dst[index] = 0;
    dst[0] = dc_values[dc_indices[block]];
    int position = 1;
    for (long long ac = block_ac_starts[block]; ac < block_ac_ends[block]; ++ac) {
        const unsigned short symbol = ac_symbols[ac];
        if ((symbol & 0xff00U) == 0xff00U) {
            const int zero_count = symbol & 0xffU;
            position = zero_count == 0 ? 64 : position + zero_count;
        } else {
            dst[position++] = symbol;
        }
    }
}

extern "C" __global__ void pixtreme_dwa_byte_rle(
    const unsigned char* encoded,
    const long long* encoded_offsets,
    const long long* encoded_sizes,
    unsigned char* raw,
    const long long* raw_offsets,
    const long long* raw_sizes,
    int* status,
    const int stream_count
) {
    const int stream = (int)blockIdx.x;
    if (stream >= stream_count || threadIdx.x != 0) return;
    long long source = encoded_offsets[stream];
    const long long source_end = source + encoded_sizes[stream];
    long long destination = raw_offsets[stream];
    const long long destination_end = destination + raw_sizes[stream];
    while (source < source_end) {
        const signed char control = (signed char)encoded[source++];
        if (control < 0) {
            const int count = -(int)control;
            if (source + count > source_end || destination + count > destination_end) {
                status[stream] = 1;
                return;
            }
            for (int index = 0; index < count; ++index) raw[destination++] = encoded[source++];
        } else {
            const int count = (int)control + 1;
            if (source >= source_end || destination + count > destination_end) {
                status[stream] = 2;
                return;
            }
            const unsigned char value = encoded[source++];
            for (int index = 0; index < count; ++index) raw[destination++] = value;
        }
    }
    if (destination != destination_end) status[stream] = 3;
}

extern "C" __global__ void pixtreme_dwa_restore_dc(
    const unsigned char* grouped,
    const long long* grouped_offsets,
    const long long* element_offsets,
    const long long* element_counts,
    unsigned short* output,
    const int stream_count
) {
    const long long element = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int stream = (int)blockIdx.y; stream < stream_count; stream += (int)gridDim.y) {
      if (element < element_counts[stream]) {
        const long long byte_count = element_counts[stream] * 2;
        const long long half = (byte_count + 1) / 2;
        const long long low_original = element * 2;
        const long long high_original = low_original + 1;
        const long long low_grouped = (low_original & 1LL) ? half + low_original / 2 : low_original / 2;
        const long long high_grouped = (high_original & 1LL) ? half + high_original / 2 : high_original / 2;
        output[element_offsets[stream] + element] = (unsigned short)(
            grouped[grouped_offsets[stream] + low_grouped]
            | ((unsigned int)grouped[grouped_offsets[stream] + high_grouped] << 8)
        );
      }
    }
}
"""

_DWA_SCATTER_TEMPLATE = r"""
#include <cuda_fp16.h>

typedef __OUTPUT_TYPE__ pixtreme_dwa_output_t;

extern "C" __global__ void pixtreme_dwa_scatter(
    const unsigned char* staging,
    const long long* chunk_stage_offsets,
    const unsigned char* raw_chunks,
    const int* chunk_rows,
    const unsigned char* decoded,
    const long long* unknown_offsets,
    const unsigned char* rle_raw,
    const long long* rle_offsets,
    const unsigned short* lossy_blocks,
    const long long* block_lookup,
    const int* selected_file_channels,
    const int* channel_schemes,
    const int* selected_types,
    const long long* selected_row_offsets,
    pixtreme_dwa_output_t* output,
    const long long element_count,
    const int width,
    const int height,
    const int output_channels,
    const int file_channels,
    const int lines_per_chunk,
    const int block_columns,
    const int maximum_block_rows,
    const long long row_bytes
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) return;
    const int output_channel = (int)(element % output_channels);
    const long long pixel = element / output_channels;
    const int x = (int)(pixel % width);
    const int y = (int)(pixel / width);
    const int chunk = y / lines_per_chunk;
    const int local_y = y - chunk * lines_per_chunk;
    const int file_channel = selected_file_channels[output_channel];
    const int pixel_type = selected_types[output_channel];
    const int byte_count = pixel_type == 1 ? 2 : 4;
    unsigned int bits = 0;
    int decoded_half = pixel_type == 1;
    if (raw_chunks[chunk]) {
        const long long source = chunk_stage_offsets[chunk] + (long long)local_y * row_bytes +
            selected_row_offsets[output_channel] + (long long)x * byte_count;
        for (int byte = 0; byte < byte_count; ++byte) bits |= (unsigned int)staging[source + byte] << (8 * byte);
    } else if (channel_schemes[(long long)chunk * file_channels + file_channel] == 1) {
        const int block_y = local_y / 8;
        const int block_x = x / 8;
        const long long lookup = (((long long)chunk * file_channels + file_channel) * maximum_block_rows + block_y) *
            block_columns + block_x;
        const long long block = block_lookup[lookup];
        if (block < 0) return;
        bits = lossy_blocks[block * 64 + (local_y & 7) * 8 + (x & 7)];
        decoded_half = 1;
    } else if (channel_schemes[(long long)chunk * file_channels + file_channel] == 2) {
        const long long base = rle_offsets[(long long)chunk * file_channels + file_channel];
        const long long samples = (long long)chunk_rows[chunk] * width;
        const long long sample = (long long)local_y * width + x;
        for (int byte = 0; byte < byte_count; ++byte) bits |= (unsigned int)rle_raw[base + byte * samples + sample] << (8 * byte);
    } else {
        const long long base = unknown_offsets[(long long)chunk * file_channels + file_channel];
        const long long sample = (long long)local_y * width + x;
        for (int byte = 0; byte < byte_count; ++byte) bits |= (unsigned int)decoded[base + sample * byte_count + byte] << (8 * byte);
    }
    __WRITE_OUTPUT__
}
"""


@lru_cache(maxsize=1)
def _dwa_huffman_parse_tables_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_huffman_parse_tables")


@lru_cache(maxsize=1)
def _dwa_huffman_build_codes_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_huffman_build_codes")


@lru_cache(maxsize=1)
def _dwa_huffman_build_lookahead_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_huffman_build_lookahead")


@lru_cache(maxsize=1)
def _dwa_huffman_candidates_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_huffman_candidates")


@lru_cache(maxsize=1)
def _dwa_huffman_jump_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_huffman_jump")


@lru_cache(maxsize=1)
def _dwa_huffman_token_boundaries_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_huffman_token_boundaries")


@lru_cache(maxsize=1)
def _dwa_huffman_segment_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_huffman_decode_segments")


@lru_cache(maxsize=1)
def _dwa_expand_coefficient_blocks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_expand_coefficient_blocks")


@lru_cache(maxsize=1)
def _dwa_byte_rle_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_byte_rle")


@lru_cache(maxsize=1)
def _dwa_restore_dc_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_GPU_SOURCE, "pixtreme_dwa_restore_dc")


@lru_cache(maxsize=1)
def _dwa_quantize_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_WRITE_GPU_SOURCE, "pixtreme_dwa_quantize_half")


@lru_cache(maxsize=1)
def _dwa_rle_ac_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_WRITE_GPU_SOURCE, "pixtreme_dwa_rle_ac")


@lru_cache(maxsize=1)
def _dwa_byte_rle_encode_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_WRITE_GPU_SOURCE, "pixtreme_dwa_byte_rle_encode")


@lru_cache(maxsize=1)
def _dwa_huffman_pack_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_WRITE_GPU_SOURCE, "pixtreme_dwa_huffman_pack")


@lru_cache(maxsize=1)
def _dwa_assemble_huffman_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_WRITE_GPU_SOURCE, "pixtreme_dwa_assemble_huffman")


@lru_cache(maxsize=1)
def _dwa_assemble_chunks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DWA_WRITE_GPU_SOURCE, "pixtreme_dwa_assemble_chunks")


@lru_cache(maxsize=3)
def _dwa_scatter_kernel(output_dtype: str) -> cp.RawKernel:
    if output_dtype == "float16":
        source = _DWA_SCATTER_TEMPLATE.replace("__OUTPUT_TYPE__", "unsigned short").replace(
            "__WRITE_OUTPUT__", "output[element] = (unsigned short)bits;"
        )
    elif output_dtype == "uint32":
        source = _DWA_SCATTER_TEMPLATE.replace("__OUTPUT_TYPE__", "unsigned int").replace(
            "__WRITE_OUTPUT__", "output[element] = bits;"
        )
    else:
        source = _DWA_SCATTER_TEMPLATE.replace("__OUTPUT_TYPE__", "float").replace(
            "__WRITE_OUTPUT__",
            (
                "output[element] = pixel_type == 0 ? (float)bits : "
                "(decoded_half ? __half2float(__ushort_as_half((unsigned short)bits)) : __uint_as_float(bits));"
            ),
        )
    return cp.RawKernel(source, "pixtreme_dwa_scatter")


def _quantize_dwa_half_gpu(coefficients: cp.ndarray, tolerances: cp.ndarray) -> cp.ndarray:
    values = cp.ascontiguousarray(coefficients, dtype=cp.float32)
    errors = cp.ascontiguousarray(tolerances, dtype=cp.float32).reshape(-1)
    if int(errors.size) != 64 or int(values.size) % 64:
        raise _gpu_error(
            why="the DWA quantizer received a coefficient or tolerance shape outside 8x8 blocks",
            what=f"coefficients={values.shape!r}, tolerances={errors.shape!r}",
            how="provide complete 64-coefficient blocks and one 64-entry positional tolerance table",
        )
    output = cp.empty(values.shape, dtype=cp.uint16)
    count = int(values.size)
    block_count = (count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _dwa_quantize_kernel()(
        (block_count,),
        (_EXR_THREADS_PER_BLOCK,),
        (values, errors, output, np.int64(count)),
    )
    return cast(cp.ndarray, output)


def _encode_dwa_byte_rle_chunks_gpu(
    raw: cp.ndarray,
    raw_offsets: Sequence[int],
    raw_sizes: Sequence[int],
) -> tuple[cp.ndarray, tuple[int, ...], tuple[int, ...]]:
    source = cp.ascontiguousarray(raw, dtype=cp.uint8).reshape(-1)
    if len(raw_offsets) != len(raw_sizes):
        raise _gpu_error(
            why="the batched DWA byte RLE offsets and sizes have different lengths",
            what=f"offsets={len(raw_offsets)}, sizes={len(raw_sizes)}",
            how="provide one compact source range for every DWA chunk",
        )
    output_sizes = tuple(size + (size + 126) // 127 if size else 0 for size in raw_sizes)
    output_offsets = _prefix_offsets(output_sizes)
    output = cp.empty(sum(output_sizes), dtype=cp.uint8)
    if not output_sizes or not any(output_sizes):
        return output, output_offsets, output_sizes
    _dwa_byte_rle_encode_kernel()(
        (_maximum_block_count(raw_sizes), len(raw_sizes)),
        (_EXR_THREADS_PER_BLOCK,),
        (
            source,
            _device_i64(raw_offsets),
            _device_i64(raw_sizes),
            output,
            _device_i64(output_offsets),
            np.int32(len(raw_sizes)),
        ),
    )
    return output, output_offsets, output_sizes


def _checksum_dwa_chunks(data: cp.ndarray, offsets: Sequence[int], sizes: Sequence[int]) -> cp.ndarray:
    adler = cp.empty(len(sizes), dtype=cp.uint32)
    if not sizes:
        return adler
    _adler_kernel()(
        (len(sizes),),
        (_EXR_THREADS_PER_BLOCK,),
        (data, _device_i64(offsets), _device_i64(sizes), adler, np.int32(len(sizes))),
    )
    return adler


def _encode_dwa_zlib_chunks_gpu(streams: _DwaWriteStreams) -> _DwaDeflateStreams:
    chunk_count = len(streams.unknown_sizes)
    unknown_offsets = _prefix_offsets(streams.unknown_sizes)
    dc_offsets = _prefix_offsets(streams.dc_sizes)
    rle_offsets = _prefix_offsets(streams.rle_sizes)
    rle_encoded, rle_encoded_offsets, rle_encoded_sizes = _encode_dwa_byte_rle_chunks_gpu(
        streams.rle,
        rle_offsets,
        streams.rle_sizes,
    )
    unknown_adler = _checksum_dwa_chunks(streams.unknown, unknown_offsets, streams.unknown_sizes)
    if any(streams.dc_sizes):
        dc_transformed, dc_adler = _transform_and_checksum_chunks(streams.dc, dc_offsets, streams.dc_sizes)
    else:
        dc_transformed = cp.empty(0, dtype=cp.uint8)
        dc_adler = cp.empty(chunk_count, dtype=cp.uint32)
    rle_adler = _checksum_dwa_chunks(rle_encoded, rle_encoded_offsets, rle_encoded_sizes)
    input_blobs = (streams.unknown, dc_transformed, rle_encoded)
    input_sizes_by_stream = (streams.unknown_sizes, streams.dc_sizes, rle_encoded_sizes)
    input_offsets_by_stream = (unknown_offsets, dc_offsets, rle_encoded_offsets)
    blob_bases = _prefix_offsets(tuple(int(blob.size) for blob in input_blobs))
    transformed = _dwa_concatenate(input_blobs, dtype=cp.uint8)
    all_adler = cp.concatenate((unknown_adler, dc_adler, rle_adler))
    descriptors: list[tuple[int, int]] = []
    input_ranges: list[tuple[int, int]] = []
    adler_indices: list[int] = []
    for stream_index, (offsets, sizes) in enumerate(zip(input_offsets_by_stream, input_sizes_by_stream, strict=True)):
        for chunk_index, (offset, size) in enumerate(zip(offsets, sizes, strict=True)):
            if not size:
                continue
            descriptors.append((stream_index, chunk_index))
            input_ranges.append((blob_bases[stream_index] + offset, size))
            adler_indices.append(stream_index * chunk_count + chunk_index)
    empty_offsets = (0,) * chunk_count
    empty_sizes = (0,) * chunk_count
    if not descriptors:
        return _DwaDeflateStreams(
            payload=cp.empty(0, dtype=cp.uint8),
            unknown_offsets=empty_offsets,
            unknown_sizes=empty_sizes,
            dc_offsets=empty_offsets,
            dc_sizes=empty_sizes,
            rle_offsets=empty_offsets,
            rle_sizes=empty_sizes,
        )
    compressed, compressed_offsets, compressed_sizes = _encode_deflate_chunks(transformed, tuple(input_ranges))
    wrapped, wrapped_offsets, wrapped_sizes = _wrap_deflate_chunks(
        compressed,
        compressed_offsets,
        compressed_sizes,
        all_adler[cp.asarray(adler_indices, dtype=cp.int64)],
    )
    output_offsets = [[0] * chunk_count for _ in range(3)]
    output_sizes = [[0] * chunk_count for _ in range(3)]
    for encoded_index, (stream_index, chunk_index) in enumerate(descriptors):
        output_offsets[stream_index][chunk_index] = wrapped_offsets[encoded_index]
        output_sizes[stream_index][chunk_index] = wrapped_sizes[encoded_index]
    return _DwaDeflateStreams(
        payload=wrapped,
        unknown_offsets=tuple(output_offsets[0]),
        unknown_sizes=tuple(output_sizes[0]),
        dc_offsets=tuple(output_offsets[1]),
        dc_sizes=tuple(output_sizes[1]),
        rle_offsets=tuple(output_offsets[2]),
        rle_sizes=tuple(output_sizes[2]),
    )


def _dwa_mirror_indices(size: int, padded_size: int) -> cp.ndarray:
    indices = np.arange(padded_size, dtype=np.int32)
    outside = indices >= size
    indices[outside] = size - (indices[outside] - (size - 1))
    indices[indices < 0] = size - 1
    return cp.asarray(indices)


def _forward_dwa_transfer_gpu(values: cp.ndarray) -> cp.ndarray:
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
    return cast(cp.ndarray, nonlinear.astype(cp.float16).astype(cp.float32))


def _dwa_channel_rules_bytes(channels: Sequence[_ExrChannel]) -> bytes:
    records = bytearray(b"\x00\x00")
    suffixes = tuple(_dwa_suffix(channel.name)[1] for channel in channels)
    pixel_type = channels[0].pixel_type
    if pixel_type == 0:
        struct.pack_into("<H", records, 0, len(records))
        return bytes(records)
    for suffix in ("R", "G", "B", "Y", "BY", "RY", "A"):
        if suffix not in suffixes:
            continue
        if suffix == "A":
            scheme = 2
            csc = 0
        else:
            scheme = 1
            csc = {"R": 1, "G": 2, "B": 3}.get(suffix, 0)
        records.extend(suffix.encode("ascii") + b"\x00")
        records.extend(bytes(((csc << 4) | (scheme << 2), pixel_type)))
    struct.pack_into("<H", records, 0, len(records))
    return bytes(records)


def _dwa_concatenate(parts: Sequence[cp.ndarray], *, dtype: object) -> cp.ndarray:
    nonempty = tuple(part for part in parts if int(part.size))
    if not nonempty:
        return cp.empty(0, dtype=dtype)
    return cast(cp.ndarray, cp.concatenate(nonempty))


def _prepare_dwa_write_streams(
    data: cp.ndarray,
    raw: cp.ndarray,
    channels: Sequence[_ExrChannel],
    input_indices: Mapping[str, int],
    layout: _DwaChannelLayout,
    *,
    row_counts: Sequence[int],
    lines_per_chunk: int,
    dwa_level: float,
) -> _DwaWriteStreams:
    height = int(data.shape[0])
    width = int(data.shape[1])
    chunk_count = len(row_counts)
    row_starts = _prefix_offsets(row_counts)
    row_count_array = np.asarray(row_counts, dtype=np.int64)
    local_rows = np.arange(lines_per_chunk, dtype=np.int64)[None, :]
    mirrored_rows = np.where(
        local_rows < row_count_array[:, None],
        local_rows,
        row_count_array[:, None] - (local_rows - (row_count_array[:, None] - 1)),
    )
    mirrored_rows = np.where(mirrored_rows < 0, row_count_array[:, None] - 1, mirrored_rows)
    source_rows = np.asarray(row_starts, dtype=np.int64)[:, None] + mirrored_rows
    device_source_rows = cp.asarray(source_rows)
    valid_rows = cp.asarray(local_rows < row_count_array[:, None])
    bytes_per_sample = channels[0].bytes_per_sample
    raw_planes = raw.reshape(height, len(channels), width, bytes_per_sample)[device_source_rows]
    descriptor_by_name = {descriptor.name: descriptor for descriptor in layout.channels}
    unknown_indices = tuple(
        index for index, channel in enumerate(channels) if descriptor_by_name[channel.name].scheme == "unknown"
    )
    rle_indices = tuple(
        index for index, channel in enumerate(channels) if descriptor_by_name[channel.name].scheme == "rle"
    )
    unknown_sizes = tuple(row_count * len(unknown_indices) * width * bytes_per_sample for row_count in row_counts)
    rle_sizes = tuple(row_count * len(rle_indices) * width * bytes_per_sample for row_count in row_counts)
    if unknown_indices:
        unknown_staged = cp.take(raw_planes, cp.asarray(unknown_indices, dtype=cp.int32), axis=2).transpose(
            0, 2, 1, 3, 4
        )
        unknown_mask = cp.broadcast_to(valid_rows[:, None, :, None, None], unknown_staged.shape)
        unknown_raw = cp.ascontiguousarray(unknown_staged[unknown_mask])
    else:
        unknown_raw = cp.empty(0, dtype=cp.uint8)
    if rle_indices:
        rle_staged = cp.take(raw_planes, cp.asarray(rle_indices, dtype=cp.int32), axis=2).transpose(0, 2, 4, 1, 3)
        rle_mask = cp.broadcast_to(valid_rows[:, None, None, :, None], rle_staged.shape)
        rle_raw = cp.ascontiguousarray(rle_staged[rle_mask])
    else:
        rle_raw = cp.empty(0, dtype=cp.uint8)

    luminance_tolerance, chroma_tolerance = _dwa_quantization_tables(dwa_level)
    device_luminance_tolerance = cp.asarray(luminance_tolerance)
    device_chroma_tolerance = cp.asarray(chroma_tolerance)
    basis = cp.asarray(_DWA_INVERSE_BASIS_HOST)
    block_columns = (width + 7) // 8
    block_rows_per_chunk = lines_per_chunk // 8
    blocks_per_chunk = block_rows_per_chunk * block_columns
    padded_width = block_columns * 8
    x_indices = _dwa_mirror_indices(width, padded_width)
    zigzag_destination = cp.asarray(_DWA_NATURAL_TO_ZIGZAG)
    block_counts = tuple(((row_count + 7) // 8) * block_columns for row_count in row_counts)
    valid_blocks = cp.arange(blocks_per_chunk, dtype=cp.int64)[None, :] < cp.asarray(block_counts)[:, None]
    batched_data = data[device_source_rows]
    ac_fixed_parts: list[cp.ndarray] = []
    ac_mask_parts: list[cp.ndarray] = []
    dc_fixed_parts: list[cp.ndarray] = []
    dc_mask_parts: list[cp.ndarray] = []
    lossy_units = _dwa_lossy_units(channels, layout)
    for unit in lossy_units:
        components: list[cp.ndarray] = []
        for channel in unit:
            values = batched_data[..., input_indices[channel.name]]
            if data.dtype.name == "uint8":
                values = values.astype(cp.float32) * np.float32(1.0 / 255.0)
            elif data.dtype.name == "uint16":
                values = values.astype(cp.float32) * np.float32(1.0 / 65535.0)
            elif data.dtype.name == "float32":
                maximum = np.float32(65504.0)
                values = cp.where(cp.isfinite(values), cp.clip(values, -maximum, maximum), values)
            half_plane = values.astype(cp.float16)
            nonlinear = _forward_dwa_transfer_gpu(half_plane)
            components.append(nonlinear[:, :, x_indices])
        if len(components) == 3:
            red, green, blue = components
            components = [
                np.float32(0.2126) * red + np.float32(0.7152) * green + np.float32(0.0722) * blue,
                np.float32(-0.1146) * red - np.float32(0.3854) * green + np.float32(0.5) * blue,
                np.float32(0.5) * red - np.float32(0.4542) * green - np.float32(0.0458) * blue,
            ]

        unit_zigzag: list[cp.ndarray] = []
        for component_index, component in enumerate(components):
            blocks = (
                component.reshape(chunk_count, block_rows_per_chunk, 8, block_columns, 8)
                .transpose(0, 1, 3, 2, 4)
                .reshape(chunk_count, blocks_per_chunk, 8, 8)
            )
            coefficients = cp.matmul(cp.matmul(basis.T, blocks), basis)
            tolerances = device_luminance_tolerance if component_index == 0 else device_chroma_tolerance
            natural = _quantize_dwa_half_gpu(coefficients, tolerances).reshape(chunk_count, blocks_per_chunk, 64)
            zigzag = cp.empty_like(natural)
            zigzag[..., zigzag_destination] = natural
            unit_zigzag.append(zigzag)
        dc_fixed = cp.stack(tuple(zigzag[..., 0] for zigzag in unit_zigzag), axis=1)
        dc_mask = cp.broadcast_to(valid_blocks[:, None, :], dc_fixed.shape)
        dc_fixed_parts.append(dc_fixed.reshape(chunk_count, -1))
        dc_mask_parts.append(dc_mask.reshape(chunk_count, -1))
        interleaved = cp.stack(unit_zigzag, axis=2)
        block_components = cp.ascontiguousarray(interleaved).reshape(-1, 64)
        fixed = cp.empty((int(block_components.shape[0]), 63), dtype=cp.uint16)
        counts = cp.empty(int(block_components.shape[0]), dtype=cp.uint8)
        launch_count = (int(block_components.shape[0]) + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _dwa_rle_ac_kernel()(
            (launch_count,),
            (_EXR_THREADS_PER_BLOCK,),
            (block_components, fixed, counts, np.int64(block_components.shape[0])),
        )
        fixed = fixed.reshape(chunk_count, blocks_per_chunk, len(unit), 63)
        counts = counts.reshape(chunk_count, blocks_per_chunk, len(unit))
        positions = cp.arange(63, dtype=cp.uint8)[None, None, None, :]
        ac_mask = valid_blocks[:, :, None, None] & (positions < counts[..., None])
        ac_fixed_parts.append(fixed.reshape(chunk_count, -1))
        ac_mask_parts.append(ac_mask.reshape(chunk_count, -1))

    if ac_fixed_parts:
        ac_fixed = cp.concatenate(ac_fixed_parts, axis=1)
        ac_mask = cp.concatenate(ac_mask_parts, axis=1)
        ac_symbols = cp.ascontiguousarray(ac_fixed[ac_mask])
        ac_chunk_grid = cp.broadcast_to(cp.arange(chunk_count, dtype=cp.int32)[:, None], ac_mask.shape)
        ac_chunk_ids = cp.ascontiguousarray(ac_chunk_grid[ac_mask])
        dc_fixed = cp.concatenate(dc_fixed_parts, axis=1)
        dc_mask = cp.concatenate(dc_mask_parts, axis=1)
        dc_values = cp.ascontiguousarray(dc_fixed[dc_mask])
    else:
        ac_symbols = cp.empty(0, dtype=cp.uint16)
        ac_chunk_ids = cp.empty(0, dtype=cp.int32)
        dc_values = cp.empty(0, dtype=cp.uint16)
    lossy_component_count = sum(len(unit) for unit in lossy_units)
    dc_sizes = tuple(block_count * lossy_component_count * 2 for block_count in block_counts)
    return _DwaWriteStreams(
        unknown=unknown_raw,
        unknown_sizes=unknown_sizes,
        ac_symbols=ac_symbols,
        ac_chunk_ids=ac_chunk_ids,
        dc=dc_values.view(cp.uint8),
        dc_sizes=dc_sizes,
        rle=rle_raw,
        rle_sizes=rle_sizes,
    )


def _encode_dwa_chunks_gpu(
    data: cp.ndarray,
    raw: cp.ndarray,
    channels: Sequence[_ExrChannel],
    input_indices: Mapping[str, int],
    layout: _DwaChannelLayout,
    channel_rules: bytes,
    *,
    row_counts: Sequence[int],
    raw_offsets: Sequence[int],
    raw_sizes: Sequence[int],
    lines_per_chunk: int,
    dwa_level: float,
) -> tuple[cp.ndarray, tuple[int, ...]]:
    streams = _prepare_dwa_write_streams(
        data,
        raw,
        channels,
        input_indices,
        layout,
        row_counts=row_counts,
        lines_per_chunk=lines_per_chunk,
        dwa_level=dwa_level,
    )
    ac_payload, ac_offsets, ac_sizes, ac_counts = _encode_dwa_huffman_chunks_gpu(
        streams.ac_symbols,
        streams.ac_chunk_ids,
        len(row_counts),
    )
    deflate = _encode_dwa_zlib_chunks_gpu(streams)
    rle_encoded_sizes = tuple(size + (size + 126) // 127 if size else 0 for size in streams.rle_sizes)
    control_parts = tuple(
        struct.pack(
            "<11Q",
            2,
            streams.unknown_sizes[index],
            deflate.unknown_sizes[index],
            ac_sizes[index],
            deflate.dc_sizes[index],
            deflate.rle_sizes[index],
            rle_encoded_sizes[index],
            streams.rle_sizes[index],
            ac_counts[index],
            streams.dc_sizes[index] // 2,
            _DWA_STATIC_HUFFMAN,
        )
        + channel_rules
        for index in range(len(row_counts))
    )
    control_sizes = tuple(len(control) for control in control_parts)
    control_offsets = _prefix_offsets(control_sizes)
    control = cp.asarray(np.frombuffer(b"".join(control_parts), dtype=np.uint8))
    compressed_sizes = tuple(
        control_sizes[index]
        + deflate.unknown_sizes[index]
        + ac_sizes[index]
        + deflate.dc_sizes[index]
        + deflate.rle_sizes[index]
        for index in range(len(row_counts))
    )
    compressed_offsets = _prefix_offsets(compressed_sizes)
    compressed = cp.empty(sum(compressed_sizes), dtype=cp.uint8)
    _dwa_assemble_chunks_kernel()(
        (_maximum_block_count(compressed_sizes), len(row_counts)),
        (_EXR_THREADS_PER_BLOCK,),
        (
            control,
            _device_i64(control_offsets),
            _device_i64(control_sizes),
            deflate.payload,
            _device_i64(deflate.unknown_offsets),
            _device_i64(deflate.unknown_sizes),
            ac_payload,
            _device_i64(ac_offsets),
            _device_i64(ac_sizes),
            deflate.payload,
            _device_i64(deflate.dc_offsets),
            _device_i64(deflate.dc_sizes),
            deflate.payload,
            _device_i64(deflate.rle_offsets),
            _device_i64(deflate.rle_sizes),
            compressed,
            _device_i64(compressed_offsets),
            np.int32(len(row_counts)),
        ),
    )
    return _select_exr_payloads(
        raw,
        raw_offsets,
        raw_sizes,
        compressed,
        compressed_offsets,
        compressed_sizes,
    )


def _write_exr_dwa_gpu(
    path: Path,
    data: cp.ndarray,
    channels: Sequence[str],
    *,
    compression: str,
    dwa_level: float,
    chromaticities: Sequence[float],
    aces_image_container: bool,
) -> None:
    encoded_file_channels = _encode_exr_output_channels(tuple(sorted(channels)))
    try:
        raw, file_channels, pixel_type = _pack_exr_gpu(data, channels)
        validated_file_channels = tuple(channel for channel, _ in encoded_file_channels)
        if validated_file_channels != file_channels:
            raise _gpu_error(
                why="the validated DWA channel order diverged from the GPU packing order",
                what=f"validated={validated_file_channels!r}, packed={file_channels!r}",
                how="report this internal hybrid DWA writer defect",
            )
        height, width, channel_count = (int(value) for value in data.shape)
        dtype, bytes_per_sample = _EXR_DTYPE_INFO[pixel_type]
        file_channel_descriptors = tuple(
            _ExrChannel(
                name=name,
                pixel_type=pixel_type,
                dtype=dtype,
                bytes_per_sample=bytes_per_sample,
                perceptually_linear=False,
                x_sampling=1,
                y_sampling=1,
            )
            for name in file_channels
        )
        layout = (
            _classify_dwa_channels(file_channel_descriptors, ())
            if pixel_type == 0
            else _classify_default_dwa_channels(file_channel_descriptors)
        )
        channel_rules = _dwa_channel_rules_bytes(file_channel_descriptors)
        input_indices = {name: index for index, name in enumerate(channels)}
        lines_per_chunk = _EXR_LINES_PER_CHUNK[compression]
        row_starts = tuple(range(0, height, lines_per_chunk))
        row_bytes = width * channel_count * bytes_per_sample
        row_counts = tuple(min(lines_per_chunk, height - row_start) for row_start in row_starts)
        raw_sizes = tuple(row_count * row_bytes for row_count in row_counts)
        raw_offsets = _prefix_offsets(raw_sizes)
        payload_blob, payload_sizes = _encode_dwa_chunks_gpu(
            data,
            raw,
            file_channel_descriptors,
            input_indices,
            layout,
            channel_rules,
            row_counts=row_counts,
            raw_offsets=raw_offsets,
            raw_sizes=raw_sizes,
            lines_per_chunk=lines_per_chunk,
            dwa_level=dwa_level,
        )
        payload_host = payload_blob.get().tobytes()
        header = _exr_write_header(
            width=width,
            height=height,
            encoded_channels=encoded_file_channels,
            pixel_type=pixel_type,
            compression=compression,
            chromaticities=chromaticities,
            aces_image_container=aces_image_container,
            dwa_level=dwa_level,
        )
    except _ExrGpuError:
        raise
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the hybrid GPU DWA data plane could not encode eligible chunks: {error}",
                what=(
                    f"compression={compression!r}, dwa_level={dwa_level!r}, "
                    f"shape={data.shape!r}, channels={tuple(channels)!r}"
                ),
                how="verify NVIDIA GPU availability, CUDA compatibility, nvCOMP, and the DWA Frame layout",
            )
        ) from error

    first_chunk_offset = len(header) + len(payload_sizes) * 8
    chunk_offsets: list[int] = []
    cursor = first_chunk_offset
    for size in payload_sizes:
        chunk_offsets.append(cursor)
        cursor += 8 + size
    try:
        with path.open("wb") as stream:
            stream.write(header)
            stream.write(b"".join(struct.pack("<Q", offset) for offset in chunk_offsets))
            payload_offset = 0
            for row_start, size in zip(row_starts, payload_sizes, strict=True):
                stream.write(struct.pack("<ii", row_start, size))
                stream.write(payload_host[payload_offset : payload_offset + size])
                payload_offset += size
    except OSError as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the hybrid GPU DWA file could not be written: {error}",
                what=str(path),
                how="provide a writable output path whose parent directory already exists",
            )
        ) from error


def _dwa_zlib_gpu_descriptor(
    container: _ExrContainer,
    chunk: _ExrChunk,
    span: _DwaByteSpan,
    *,
    stream_name: str,
    stage_offset: int,
) -> tuple[int, int, int]:
    payload = container.data[span.start : span.end]
    if len(payload) < 6:
        raise _gpu_error(
            why=f"the DWA {stream_name} zlib wrapper is truncated before its Deflate payload and Adler-32 trailer",
            what=f"chunk_y={chunk.y}, packed_size={len(payload)}",
            how="provide a complete RFC 1950 zlib stream for every declared DWA substream",
        )
    cmf, flg = payload[0], payload[1]
    if (cmf & 0x0F) != 8 or (cmf >> 4) > 7 or ((cmf << 8) | flg) % 31:
        raise _gpu_error(
            why=f"the DWA {stream_name} zlib header has an invalid compression method, window, or FCHECK",
            what=f"chunk_y={chunk.y}, CMF=0x{cmf:02x}, FLG=0x{flg:02x}",
            how="encode a valid RFC 1950 Deflate stream with a correct header check",
        )
    if flg & 0x20:
        raise _gpu_error(
            why=f"the DWA {stream_name} zlib stream requests a preset dictionary",
            what=f"chunk_y={chunk.y}, FLG=0x{flg:02x}",
            how="encode DWA zlib substreams without an RFC 1950 preset dictionary",
        )
    deflate_size = len(payload) - 6
    if deflate_size <= 0:
        raise _gpu_error(
            why=f"the DWA {stream_name} zlib wrapper contains no raw Deflate payload",
            what=f"chunk_y={chunk.y}, packed_size={len(payload)}",
            how="provide a complete RFC 1951 payload between the zlib header and trailer",
        )
    input_offset = stage_offset + span.start - chunk.payload_start + 2
    return input_offset, deflate_size, struct.unpack(">I", payload[-4:])[0]


def _read_exr_dwa_gpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    if output_dtype == "uint32" and any(channel.pixel_type != 0 for channel in selected):
        raise _gpu_error(
            why="the native DWA UINT read lane received a non-UINT selected channel",
            what=f"channels={tuple((channel.name, channel.pixel_type) for channel in selected)!r}",
            how="route only homogeneous EXR UINT selections to the uint32 output lane",
        )
    chunks = container.chunks
    channels = container.parts[0].channels
    chunk_count = len(chunks)
    file_channel_count = len(channels)
    width = container.data_window[2] - container.data_window[0] + 1
    height = container.data_window[3] - container.data_window[1] + 1

    stage_sizes = np.fromiter((chunk.packed_size for chunk in chunks), dtype=np.int64, count=chunk_count)
    stage_offsets = _numpy_offsets(stage_sizes)
    host_staging = np.frombuffer(
        b"".join(container.data[chunk.payload_start : chunk.payload_end] for chunk in chunks), dtype=np.uint8
    )
    device_staging = cp.asarray(host_staging)

    stream_records: list[tuple[int, str, _DwaByteSpan, int]] = []
    for stream_name in ("UNKNOWN", "DC", "RLE"):
        for chunk_index, chunk in enumerate(chunks):
            descriptor = chunk.dwa
            if chunk.raw_stored or descriptor is None or descriptor.leader is None:
                continue
            leader = descriptor.leader
            if stream_name == "UNKNOWN" and leader.unknown_compressed_size:
                stream_records.append(
                    (chunk_index, stream_name, descriptor.unknown_span, leader.unknown_uncompressed_size)
                )
            elif stream_name == "DC" and leader.dc_compressed_size:
                stream_records.append((chunk_index, stream_name, descriptor.dc_span, leader.dc_element_count * 2))
            elif stream_name == "RLE" and leader.rle_compressed_size:
                stream_records.append((chunk_index, stream_name, descriptor.rle_span, leader.rle_uncompressed_size))

    stream_sizes = np.asarray([record[3] for record in stream_records], dtype=np.int64)
    stream_offsets = _numpy_offsets(stream_sizes)
    stream_adlers = np.empty(len(stream_records), dtype=np.uint32)
    input_ranges: list[tuple[int, int]] = []
    output_ranges: list[tuple[int, int]] = []
    unknown_decoded_offsets = np.full(chunk_count, -1, dtype=np.int64)
    dc_decoded_offsets = np.full(chunk_count, -1, dtype=np.int64)
    rle_encoded_offsets = np.full(chunk_count, -1, dtype=np.int64)
    for stream_index, (chunk_index, stream_name, span, expected_size) in enumerate(stream_records):
        chunk = chunks[chunk_index]
        input_offset, input_size, adler = _dwa_zlib_gpu_descriptor(
            container,
            chunk,
            span,
            stream_name=stream_name,
            stage_offset=int(stage_offsets[chunk_index]),
        )
        decoded_offset = int(stream_offsets[stream_index])
        input_ranges.append((input_offset, input_size))
        output_ranges.append((decoded_offset, expected_size))
        stream_adlers[stream_index] = adler
        if stream_name == "UNKNOWN":
            unknown_decoded_offsets[chunk_index] = decoded_offset
        elif stream_name == "DC":
            dc_decoded_offsets[chunk_index] = decoded_offset
        else:
            rle_encoded_offsets[chunk_index] = decoded_offset

    decoded_byte_count = int(stream_sizes.sum())
    decoded = cp.empty(max(decoded_byte_count, 1), dtype=cp.uint8)
    if stream_records:
        _decode_deflate_chunks(
            device_staging,
            input_ranges,
            decoded,
            output_ranges,
            verify_output_sizes=False,
        )
        validation = decoded.copy()
        failed_streams = _restore_exr_gpu_chunks(
            validation,
            stream_offsets,
            stream_sizes,
            np.ones(len(stream_records), dtype=np.uint8),
            stream_adlers,
        )
        if failed_streams.size:
            stream_failures = tuple(
                (chunks[stream_records[int(index)][0]].y, stream_records[int(index)][1]) for index in failed_streams
            )
            raise _gpu_error(
                why="the GPU Adler-32 result does not match a DWA zlib trailer",
                what=f"chunk_streams={stream_failures!r}",
                how="verify that each zlib substream and its big-endian Adler-32 trailer are complete and unmodified",
            )
        dc_flags = np.asarray([stream_name == "DC" for _, stream_name, _, _ in stream_records], dtype=np.uint8)
        failed_dc = _restore_exr_gpu_chunks(decoded, stream_offsets, stream_sizes, dc_flags, stream_adlers)
        if failed_dc.size:
            failed_y = tuple(chunks[stream_records[int(index)][0]].y for index in failed_dc)
            raise _gpu_error(
                why="the GPU DC ZIP restoration failed its DWA stream integrity check",
                what=f"chunk_y={failed_y!r}",
                how="verify the DWA DC zlib payload, predictor bytes, and declared coefficient count",
            )

    dc_element_counts = np.fromiter(
        (
            0
            if chunk.raw_stored or chunk.dwa is None or chunk.dwa.leader is None
            else chunk.dwa.leader.dc_element_count
            for chunk in chunks
        ),
        dtype=np.int64,
        count=chunk_count,
    )
    dc_element_offsets = _numpy_offsets(dc_element_counts)
    dc_values = cp.empty(max(int(dc_element_counts.sum()), 1), dtype=cp.uint16)
    dc_chunk_indices = np.flatnonzero(dc_element_counts)
    if dc_chunk_indices.size:
        maximum_dc_count = int(dc_element_counts[dc_chunk_indices].max())
        dc_grid_x = (maximum_dc_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _dwa_restore_dc_kernel()(
            (dc_grid_x, min(int(dc_chunk_indices.size), _EXR_MAX_GRID_Y)),
            (_EXR_THREADS_PER_BLOCK,),
            (
                decoded,
                cp.asarray(dc_decoded_offsets[dc_chunk_indices]),
                cp.asarray(dc_element_offsets[dc_chunk_indices]),
                cp.asarray(dc_element_counts[dc_chunk_indices]),
                dc_values,
                np.int32(dc_chunk_indices.size),
            ),
        )

    ac_counts = np.fromiter(
        (
            0
            if chunk.raw_stored or chunk.dwa is None or chunk.dwa.leader is None
            else chunk.dwa.leader.ac_element_count
            for chunk in chunks
        ),
        dtype=np.int64,
        count=chunk_count,
    )
    ac_offsets = _numpy_offsets(ac_counts)
    ac_symbols = cp.empty(max(int(ac_counts.sum()), 1), dtype=cp.uint16)
    huffman_records: list[tuple[int, _DwaHuffmanTable]] = []
    for chunk_index, chunk in enumerate(chunks):
        if not chunk.raw_stored and chunk.dwa is not None and chunk.dwa.huffman is not None:
            huffman_records.append((chunk_index, chunk.dwa.huffman))
    if huffman_records:
        huffman_data_offsets = tuple(
            int(stage_offsets[chunk_index] + table.data_span.start - chunks[chunk_index].payload_start)
            for chunk_index, table in huffman_records
        )
        huffman_output_counts = tuple(int(ac_counts[chunk_index]) for chunk_index, _ in huffman_records)
        ac_symbols = _decode_dwa_huffman_gpu(
            device_staging,
            data_offsets=huffman_data_offsets,
            tables=tuple(table for _, table in huffman_records),
            output_counts=huffman_output_counts,
            record_labels=tuple(chunks[chunk_index].y for chunk_index, _ in huffman_records),
        )

    rle_chunk_indices = np.flatnonzero(rle_encoded_offsets >= 0)
    rle_raw_sizes = np.asarray(
        [
            cast(_DwaLeader, cast(_DwaChunkDescriptor, chunks[int(index)].dwa).leader).rle_raw_size
            for index in rle_chunk_indices
        ],
        dtype=np.int64,
    )
    rle_raw_offsets = _numpy_offsets(rle_raw_sizes)
    rle_raw = cp.empty(max(int(rle_raw_sizes.sum()), 1), dtype=cp.uint8)
    rle_raw_base_by_chunk = np.full(chunk_count, -1, dtype=np.int64)
    if rle_chunk_indices.size:
        rle_raw_base_by_chunk[rle_chunk_indices] = rle_raw_offsets
        rle_status = cp.zeros(rle_chunk_indices.size, dtype=cp.int32)
        _dwa_byte_rle_kernel()(
            (int(rle_chunk_indices.size),),
            (1,),
            (
                decoded,
                cp.asarray(rle_encoded_offsets[rle_chunk_indices]),
                cp.asarray(
                    stream_sizes[
                        [
                            next(
                                stream_index
                                for stream_index, record in enumerate(stream_records)
                                if record[0] == int(chunk_index) and record[1] == "RLE"
                            )
                            for chunk_index in rle_chunk_indices
                        ]
                    ]
                ),
                rle_raw,
                cp.asarray(rle_raw_offsets),
                cp.asarray(rle_raw_sizes),
                rle_status,
                np.int32(rle_chunk_indices.size),
            ),
        )
        failed_rle = np.flatnonzero(rle_status.get())
        if failed_rle.size:
            rle_failures = tuple(
                (chunks[int(rle_chunk_indices[int(index)])].y, int(rle_status.get()[int(index)]))
                for index in failed_rle
            )
            raise _gpu_error(
                why="the GPU DWA byte RLE decoder rejected a packet boundary or declared raw size",
                what=f"chunk_status={rle_failures!r}",
                how="make the RLE packets consume the encoded stream and produce the declared raw byte count exactly",
            )

    channel_indices = {channel.name: index for index, channel in enumerate(channels)}
    scheme_codes = {"unknown": 0, "lossy_dct": 1, "rle": 2}
    channel_schemes = np.zeros((chunk_count, file_channel_count), dtype=np.int32)
    maximum_block_rows = max(cast(_DwaChunkDescriptor, chunk.dwa).geometry.block_rows for chunk in chunks)
    block_columns = cast(_DwaChunkDescriptor, chunks[0].dwa).geometry.block_columns
    block_lookup = np.full((chunk_count, file_channel_count, maximum_block_rows, block_columns), -1, dtype=np.int64)
    block_offsets = np.empty(chunk_count, dtype=np.int64)
    block_counts = np.zeros(chunk_count, dtype=np.int64)
    dc_index_parts: list[np.ndarray] = []
    transfer_flag_parts: list[np.ndarray] = []
    csc_triplet_parts: list[np.ndarray] = []
    total_block_count = 0
    for chunk_index, chunk in enumerate(chunks):
        descriptor = cast(_DwaChunkDescriptor, chunk.dwa)
        layout = cast(_DwaChannelLayout, descriptor.channel_layout)
        layout_by_name = {item.name: item for item in layout.channels}
        for file_channel, channel in enumerate(channels):
            channel_schemes[chunk_index, file_channel] = scheme_codes[layout_by_name[channel.name].scheme]
        block_offsets[chunk_index] = total_block_count
        if chunk.raw_stored:
            continue
        leader = cast(_DwaLeader, descriptor.leader)
        geometry = descriptor.geometry
        local_block_count = geometry.block_rows * geometry.block_columns
        dc_stream_offset = int(dc_decoded_offsets[chunk_index])
        if leader.dc_element_count and dc_stream_offset < 0:
            raise _gpu_error(
                why="the DWA DC stream has no GPU output range",
                what=f"chunk_y={chunk.y}, decoded_offset={dc_stream_offset}, dc_count={leader.dc_element_count}",
                how="provide one little-endian HALF DC coefficient for every lossy block",
            )
        dc_base = int(dc_element_offsets[chunk_index])
        dc_cursor = 0
        for unit in _dwa_lossy_units(channels, layout):
            unit_size = len(unit)
            local_blocks = np.arange(local_block_count, dtype=np.int64)
            component_indices = np.arange(unit_size, dtype=np.int64)
            global_blocks = total_block_count + np.arange(local_block_count * unit_size, dtype=np.int64).reshape(
                local_block_count, unit_size
            )
            dc_index_parts.append(
                (dc_base + dc_cursor + local_blocks[:, None] + component_indices[None, :] * local_block_count).reshape(
                    -1
                )
            )
            unit_transfer_flags = np.asarray(
                [unit_size == 3 or not channel.perceptually_linear for channel in unit], dtype=np.bool_
            )
            transfer_flag_parts.append(np.broadcast_to(unit_transfer_flags, global_blocks.shape).reshape(-1))
            for component, channel in enumerate(unit):
                block_lookup[
                    chunk_index,
                    channel_indices[channel.name],
                    : geometry.block_rows,
                    : geometry.block_columns,
                ] = global_blocks[:, component].reshape(geometry.block_rows, geometry.block_columns)
            if unit_size == 3:
                csc_triplet_parts.append(global_blocks)
            unit_block_count = local_block_count * unit_size
            total_block_count += unit_block_count
            dc_cursor += unit_block_count
        block_counts[chunk_index] = total_block_count - int(block_offsets[chunk_index])
        if dc_cursor != leader.dc_element_count:
            raise _gpu_error(
                why="the DWA DC stream does not match its classified channel and block ownership",
                what=f"chunk_y={chunk.y}, consumed={dc_cursor}, declared={leader.dc_element_count}",
                how="provide exactly one DC coefficient for each lossy channel block",
            )

    dc_indices = np.concatenate(dc_index_parts) if dc_index_parts else np.empty(0, dtype=np.int64)
    transfer_flags = np.concatenate(transfer_flag_parts) if transfer_flag_parts else np.empty(0, dtype=np.bool_)
    csc_triplets = np.concatenate(csc_triplet_parts, axis=0) if csc_triplet_parts else np.empty((0, 3), dtype=np.int64)
    total_blocks = int(dc_indices.size)
    if total_blocks:
        coefficient_bits = cp.empty(total_blocks * 64, dtype=cp.uint16)
        block_ac_starts, block_ac_ends = _dwa_coefficient_block_spans(
            ac_symbols,
            ac_offsets,
            ac_counts,
            expected_block_count=total_blocks,
        )
        _dwa_expand_coefficient_blocks_kernel()(
            ((total_blocks + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK,),
            (_EXR_THREADS_PER_BLOCK,),
            (
                ac_symbols,
                dc_values,
                cp.asarray(dc_indices),
                coefficient_bits,
                block_ac_starts,
                block_ac_ends,
                np.int64(total_blocks),
            ),
        )
        zigzag = cp.asarray(_DWA_NATURAL_TO_ZIGZAG)
        coefficients = (
            coefficient_bits.reshape(total_blocks, 64)[:, zigzag]
            .view(cp.float16)
            .astype(cp.float32)
            .reshape(total_blocks, 8, 8)
        )
        basis = cp.asarray(_DWA_INVERSE_BASIS_HOST)
        spatial = cp.matmul(cp.matmul(basis, coefficients), basis.T)
        if csc_triplets.size:
            triplets = cp.asarray(csc_triplets)
            y_plane = spatial[triplets[:, 0]].copy()
            cb_plane = spatial[triplets[:, 1]].copy()
            cr_plane = spatial[triplets[:, 2]].copy()
            spatial[triplets[:, 0]] = y_plane + cp.float32(1.5747) * cr_plane
            spatial[triplets[:, 1]] = y_plane - cp.float32(0.1873) * cb_plane - cp.float32(0.4682) * cr_plane
            spatial[triplets[:, 2]] = y_plane + cp.float32(1.8556) * cb_plane
        nonlinear = spatial.astype(cp.float16)
        nonlinear_float = nonlinear.astype(cp.float32)
        magnitude = cp.abs(nonlinear_float)
        linear = cp.where(
            magnitude <= cp.float32(1.0),
            cp.power(magnitude, cp.float32(2.2)),
            cp.exp(cp.float32(2.2) * (magnitude - cp.float32(1.0))),
        )
        linear = cp.copysign(linear, nonlinear_float)
        linear = cp.where(cp.isfinite(nonlinear_float), linear, cp.float32(0.0))
        reconstructed = cp.where(
            cp.asarray(transfer_flags)[:, None, None],
            linear.astype(cp.float16),
            nonlinear,
        )
        lossy_blocks = cp.ascontiguousarray(reconstructed).view(cp.uint16).reshape(-1)
    else:
        lossy_blocks = cp.empty(1, dtype=cp.uint16)

    unknown_offsets = np.full((chunk_count, file_channel_count), -1, dtype=np.int64)
    rle_offsets = np.full((chunk_count, file_channel_count), -1, dtype=np.int64)
    for chunk_index, chunk in enumerate(chunks):
        if chunk.raw_stored:
            continue
        descriptor = cast(_DwaChunkDescriptor, chunk.dwa)
        leader = cast(_DwaLeader, descriptor.leader)
        samples = chunk.row_count * width
        unknown_cursor = int(unknown_decoded_offsets[chunk_index])
        rle_cursor = int(rle_raw_base_by_chunk[chunk_index])
        unknown_start = unknown_cursor
        rle_start = rle_cursor
        for file_channel, channel in enumerate(channels):
            scheme = channel_schemes[chunk_index, file_channel]
            if scheme == 0:
                unknown_offsets[chunk_index, file_channel] = unknown_cursor
                unknown_cursor += samples * channel.bytes_per_sample
            elif scheme == 2:
                rle_offsets[chunk_index, file_channel] = rle_cursor
                rle_cursor += samples * channel.bytes_per_sample
        if unknown_cursor - unknown_start != leader.unknown_uncompressed_size:
            raise _gpu_error(
                why="the DWA UNKNOWN stream size does not match its classified channel planes",
                what=(
                    f"chunk_y={chunk.y}, classified={unknown_cursor - unknown_start}, "
                    f"declared={leader.unknown_uncompressed_size}"
                ),
                how="store UNKNOWN channels consecutively in file-channel order",
            )
        if rle_cursor - rle_start != leader.rle_raw_size:
            raise _gpu_error(
                why="the DWA RLE raw size does not match its classified channel byte planes",
                what=f"chunk_y={chunk.y}, classified={rle_cursor - rle_start}, declared={leader.rle_raw_size}",
                how="store one complete byte plane per RLE channel sample byte",
            )

    row_offsets_by_name: dict[str, int] = {}
    row_bytes = 0
    for channel in channels:
        row_offsets_by_name[channel.name] = row_bytes
        row_bytes += width * channel.bytes_per_sample
    selected_file_channels = np.asarray([channel_indices[channel.name] for channel in selected], dtype=np.int32)
    selected_types = np.asarray([channel.pixel_type for channel in selected], dtype=np.int32)
    selected_row_offsets = np.asarray([row_offsets_by_name[channel.name] for channel in selected], dtype=np.int64)
    chunk_rows = np.asarray([chunk.row_count for chunk in chunks], dtype=np.int32)
    raw_chunks = np.asarray([chunk.raw_stored for chunk in chunks], dtype=np.uint8)
    cupy_dtype = {"float16": cp.float16, "float32": cp.float32, "uint32": cp.uint32}[output_dtype]
    output = cp.empty((height, width, len(selected)), dtype=cupy_dtype)
    element_count = int(output.size)
    grid = (element_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _dwa_scatter_kernel(output_dtype)(
        (grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            device_staging,
            cp.asarray(stage_offsets),
            cp.asarray(raw_chunks),
            cp.asarray(chunk_rows),
            decoded,
            cp.asarray(unknown_offsets.reshape(-1)),
            rle_raw,
            cp.asarray(rle_offsets.reshape(-1)),
            lossy_blocks,
            cp.asarray(block_lookup.reshape(-1)),
            cp.asarray(selected_file_channels),
            cp.asarray(channel_schemes.reshape(-1)),
            cp.asarray(selected_types),
            cp.asarray(selected_row_offsets),
            output,
            np.int64(element_count),
            np.int32(width),
            np.int32(height),
            np.int32(len(selected)),
            np.int32(file_channel_count),
            np.int32(container.lines_per_chunk),
            np.int32(block_columns),
            np.int32(maximum_block_rows),
            np.int64(row_bytes),
        ),
    )
    return cast(cp.ndarray, output)
