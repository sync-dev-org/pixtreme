"""PIZ OpenEXR read/write lane and CUDA kernels."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._io.formats.exr.codec_dwa import (
    _decode_dwa_huffman_gpu,
    _dwa_huffman_pack_kernel,
)
from pixtreme._io.formats.exr.container import (
    _DWA_MAX_HUFFMAN_SYMBOL,
    _EXR_THREADS_PER_BLOCK,
    _PIZ_BITMAP_BYTE_COUNT,
    _decode_piz_huffman_host,
    _DwaByteSpan,
    _DwaHuffmanTable,
    _ExrChannel,
    _ExrChunk,
    _ExrContainer,
    _ExrGpuError,
    _gpu_error,
    _parse_piz_huffman_table,
    _piz_error,
    _piz_inverse_wavelet_host,
    _piz_reverse_lut,
    _piz_uses_w14,
    _PizByteSpan,
    _PizChunkDescriptor,
    _PizHuffmanTable,
)
from pixtreme._io.formats.exr.packing import (
    _device_i64,
    _maximum_block_count,
    _numpy_offsets,
    _prefix_offsets,
    _select_exr_host_pixels,
    _select_exr_payloads,
    _unpack_exr_chunks,
)

_PIZ_HUFFMAN_SEGMENT_BATCH_BITS = 1 << 25


def _encode_piz_huffman_chunks_gpu(
    symbols: cp.ndarray,
    chunk_ids: cp.ndarray,
    chunk_count: int,
) -> tuple[cp.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    values = cp.ascontiguousarray(symbols, dtype=cp.uint16).reshape(-1)
    value_chunks = cp.ascontiguousarray(chunk_ids, dtype=cp.int32).reshape(-1)
    if int(value_chunks.size) != int(values.size):
        raise _gpu_error(
            why="the batched PIZ symbols and chunk ownership have different lengths",
            what=f"symbols={values.size}, chunk_ids={value_chunks.size}",
            how="assign exactly one chunk index to every transformed PIZ word",
        )
    if chunk_count < 1 or not int(values.size):
        raise _gpu_error(
            why="the batched PIZ Huffman encoder received no chunk words",
            what=f"chunk_count={chunk_count}, symbol_count={values.size}",
            how="provide at least one nonempty scanline chunk for PIZ compression",
        )

    histogram_indices = value_chunks.astype(cp.int64) * np.int64(1 << 16) + values.astype(cp.int64)
    histograms = cp.bincount(histogram_indices, minlength=chunk_count * (1 << 16)).reshape(chunk_count, 1 << 16)
    observed = histograms > 0
    observed_counts = cp.sum(observed, axis=1, dtype=cp.uint32)
    symbol_indices = cp.arange(1 << 16, dtype=cp.uint32)
    minimum_symbols = cp.min(cp.where(observed, symbol_indices[None, :], np.uint32(0xFFFF)), axis=1)
    maximum_symbols = cp.max(cp.where(observed, symbol_indices[None, :], np.uint32(0)), axis=1)
    repeat_symbols = maximum_symbols + np.uint32(1)
    symbol_counts = cp.sum(histograms, axis=1, dtype=cp.uint64)
    descriptor_host = (
        cp.stack((minimum_symbols, repeat_symbols, observed_counts, symbol_counts), axis=1)
        .get()
        .astype(np.int64, copy=False)
    )
    empty_chunks = np.flatnonzero(descriptor_host[:, 2] == 0)
    if empty_chunks.size:
        raise _gpu_error(
            why="a PIZ Huffman chunk has no observed words",
            what=f"chunks={tuple(int(value) for value in empty_chunks)!r}, chunk_count={chunk_count}",
            how="keep zero-byte outer chunks out of the PIZ transform and encode nonempty chunks only",
        )

    code_lengths = cp.zeros((chunk_count, _DWA_MAX_HUFFMAN_SYMBOL + 1), dtype=cp.uint8)
    code_values = cp.zeros((chunk_count, _DWA_MAX_HUFFMAN_SYMBOL + 1), dtype=cp.uint64)
    maximum_leaf_count = int(descriptor_host[:, 2].max()) + 1
    node_stride = maximum_leaf_count * 2 - 1
    node_frequencies = cp.empty((chunk_count, node_stride), dtype=cp.uint64)
    node_keys = cp.empty((chunk_count, node_stride), dtype=cp.uint32)
    node_parents = cp.empty((chunk_count, node_stride), dtype=cp.int32)
    heap_nodes = cp.empty((chunk_count, maximum_leaf_count), dtype=cp.int32)
    tree_status = cp.zeros(chunk_count, dtype=cp.int32)
    _piz_huffman_lengths_kernel()(
        (chunk_count,),
        (1,),
        (
            histograms.reshape(-1),
            repeat_symbols,
            observed_counts,
            node_frequencies.reshape(-1),
            node_keys.reshape(-1),
            node_parents.reshape(-1),
            heap_nodes.reshape(-1),
            code_lengths.reshape(-1),
            tree_status,
            np.int32(node_stride),
            np.int32(maximum_leaf_count),
            np.int32(chunk_count),
        ),
    )
    tree_status_host = tree_status.get()
    failed_trees = np.flatnonzero(tree_status_host)
    if failed_trees.size:
        failures = tuple((int(index), int(tree_status_host[index])) for index in failed_trees)
        raise _gpu_error(
            why="the parallel PIZ Huffman tree exceeds the 58-bit wire limit or has invalid ownership",
            what=f"chunk_status={failures!r}",
            how="build one positive frequency-weighted tree per nonempty chunk and store over-limit chunks raw",
        )
    _piz_huffman_codes_kernel()(
        (chunk_count,),
        (1,),
        (
            code_lengths.reshape(-1),
            repeat_symbols,
            code_values.reshape(-1),
            np.int32(chunk_count),
        ),
    )
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
    literal_codes = code_values[segment_chunks, segment_symbols]
    literal_lengths = code_lengths[segment_chunks, segment_symbols]
    repeat_codes = code_values[segment_chunks, repeat_symbols[segment_chunks]]
    repeat_lengths = code_lengths[segment_chunks, repeat_symbols[segment_chunks]]
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

    table_bit_counts = cp.empty(chunk_count, dtype=cp.uint64)
    _piz_huffman_table_bits_kernel()(
        (chunk_count,),
        (1,),
        (
            code_lengths.reshape(-1),
            minimum_symbols,
            repeat_symbols,
            table_bit_counts,
            np.int32(chunk_count),
        ),
    )
    table_bit_count_host = table_bit_counts.get().astype(np.uint64, copy=False)
    table_sizes = tuple((int(bit_count) + 7) // 8 for bit_count in table_bit_count_host)
    output_sizes = tuple(
        20 + table_size + data_size for table_size, data_size in zip(table_sizes, data_sizes, strict=True)
    )
    output_offsets = _prefix_offsets(output_sizes)
    output = cp.zeros(sum(output_sizes), dtype=cp.uint8)
    _piz_assemble_huffman_kernel()(
        (_maximum_block_count(output_sizes), chunk_count),
        (_EXR_THREADS_PER_BLOCK,),
        (
            code_lengths.reshape(-1),
            minimum_symbols,
            repeat_symbols,
            _device_i64(table_sizes),
            bit_counts,
            encoded_data,
            _device_i64(data_offsets),
            _device_i64(data_sizes),
            output,
            _device_i64(output_offsets),
            np.int32(chunk_count),
        ),
    )
    return output, output_offsets, output_sizes, tuple(int(value) for value in descriptor_host[:, 3])


_PIZ_GPU_SOURCE = r"""
__device__ __forceinline__ bool pixtreme_piz_heap_less(
    const int left,
    const int right,
    const unsigned long long* frequencies,
    const unsigned int* keys
) {
    return frequencies[left] < frequencies[right] ||
        (frequencies[left] == frequencies[right] && keys[left] < keys[right]);
}

__device__ __forceinline__ void pixtreme_piz_heap_sift_down(
    int* heap,
    int size,
    int root,
    const unsigned long long* frequencies,
    const unsigned int* keys
) {
    while (true) {
        const int left = root * 2 + 1;
        if (left >= size) return;
        const int right = left + 1;
        int smallest = left;
        if (right < size && pixtreme_piz_heap_less(heap[right], heap[left], frequencies, keys)) {
            smallest = right;
        }
        if (!pixtreme_piz_heap_less(heap[smallest], heap[root], frequencies, keys)) return;
        const int temporary = heap[root];
        heap[root] = heap[smallest];
        heap[smallest] = temporary;
        root = smallest;
    }
}

__device__ __forceinline__ void pixtreme_piz_heap_sift_up(
    int* heap,
    int position,
    const unsigned long long* frequencies,
    const unsigned int* keys
) {
    while (position > 0) {
        const int parent = (position - 1) / 2;
        if (!pixtreme_piz_heap_less(heap[position], heap[parent], frequencies, keys)) return;
        const int temporary = heap[parent];
        heap[parent] = heap[position];
        heap[position] = temporary;
        position = parent;
    }
}

extern "C" __global__ void pixtreme_piz_huffman_lengths(
    const unsigned long long* histograms,
    const unsigned int* repeat_symbols,
    const unsigned int* observed_counts,
    unsigned long long* node_frequencies,
    unsigned int* node_keys,
    int* node_parents,
    int* heap_nodes,
    unsigned char* code_lengths,
    int* status,
    const int node_stride,
    const int heap_stride,
    const int chunk_count
) {
    const int chunk = blockIdx.x;
    if (chunk >= chunk_count || threadIdx.x != 0) return;
    const long long histogram_base = (long long)chunk * 65536LL;
    const long long code_base = (long long)chunk * 65537LL;
    unsigned long long* frequencies = node_frequencies + (long long)chunk * node_stride;
    unsigned int* keys = node_keys + (long long)chunk * node_stride;
    int* parents = node_parents + (long long)chunk * node_stride;
    int* heap = heap_nodes + (long long)chunk * heap_stride;
    int leaf_count = 0;
    for (unsigned int symbol = 0; symbol < 65536U; ++symbol) {
        const unsigned long long frequency = histograms[histogram_base + symbol];
        if (frequency == 0ULL) continue;
        frequencies[leaf_count] = frequency;
        keys[leaf_count] = symbol;
        parents[leaf_count] = -1;
        heap[leaf_count] = leaf_count;
        ++leaf_count;
    }
    const unsigned int pseudo = repeat_symbols[chunk];
    if (leaf_count != (int)observed_counts[chunk] || pseudo > 65536U || leaf_count >= heap_stride) {
        status[chunk] = 1;
        return;
    }
    frequencies[leaf_count] = 1ULL;
    keys[leaf_count] = pseudo;
    parents[leaf_count] = -1;
    heap[leaf_count] = leaf_count;
    ++leaf_count;
    int heap_size = leaf_count;
    for (int root = heap_size / 2 - 1; root >= 0; --root) {
        pixtreme_piz_heap_sift_down(heap, heap_size, root, frequencies, keys);
    }
    int next_node = leaf_count;
    while (heap_size > 1) {
        const int first = heap[0];
        heap[0] = heap[--heap_size];
        pixtreme_piz_heap_sift_down(heap, heap_size, 0, frequencies, keys);
        const int second = heap[0];
        heap[0] = heap[--heap_size];
        pixtreme_piz_heap_sift_down(heap, heap_size, 0, frequencies, keys);
        if (next_node >= node_stride) {
            status[chunk] = 2;
            return;
        }
        frequencies[next_node] = frequencies[first] + frequencies[second];
        keys[next_node] = keys[first] < keys[second] ? keys[first] : keys[second];
        parents[first] = next_node;
        parents[second] = next_node;
        parents[next_node] = -1;
        heap[heap_size] = next_node;
        pixtreme_piz_heap_sift_up(heap, heap_size, frequencies, keys);
        ++heap_size;
        ++next_node;
    }
    for (int leaf = 0; leaf < leaf_count; ++leaf) {
        int depth = 0;
        int node = leaf;
        while (parents[node] >= 0) {
            node = parents[node];
            if (++depth > 58) {
                status[chunk] = 3;
                return;
            }
        }
        code_lengths[code_base + keys[leaf]] = (unsigned char)(depth > 0 ? depth : 1);
    }
}

extern "C" __global__ void pixtreme_piz_huffman_codes(
    const unsigned char* code_lengths,
    const unsigned int* repeat_symbols,
    unsigned long long* code_values,
    const int chunk_count
) {
    const int chunk = blockIdx.x;
    if (chunk >= chunk_count || threadIdx.x != 0) return;
    const long long base = (long long)chunk * 65537LL;
    unsigned int counts[59] = {0U};
    unsigned long long bases[59] = {0ULL};
    const unsigned int maximum = repeat_symbols[chunk];
    for (unsigned int symbol = 0; symbol <= maximum; ++symbol) {
        const unsigned int length = code_lengths[base + symbol];
        if (length > 0U) ++counts[length];
    }
    unsigned long long code = 0ULL;
    for (int length = 58; length >= 1; --length) {
        bases[length] = code;
        code = (code + counts[length]) >> 1;
    }
    for (unsigned int symbol = 0; symbol <= maximum; ++symbol) {
        const unsigned int length = code_lengths[base + symbol];
        if (length > 0U) code_values[base + symbol] = bases[length]++;
    }
}

extern "C" __global__ void pixtreme_piz_huffman_table_bits(
    const unsigned char* code_lengths,
    const unsigned int* minimum_symbols,
    const unsigned int* repeat_symbols,
    unsigned long long* table_bit_counts,
    const int chunk_count
) {
    const int chunk = blockIdx.x;
    if (chunk >= chunk_count || threadIdx.x != 0) return;
    const long long base = (long long)chunk * 65537LL;
    unsigned int symbol = minimum_symbols[chunk];
    const unsigned int maximum = repeat_symbols[chunk];
    unsigned long long bits = 0ULL;
    while (symbol <= maximum) {
        if (code_lengths[base + symbol] != 0U) {
            bits += 6ULL;
            ++symbol;
            continue;
        }
        unsigned int zero_count = 0U;
        while (symbol <= maximum && code_lengths[base + symbol] == 0U) {
            ++zero_count;
            ++symbol;
        }
        while (zero_count > 0U) {
            const unsigned int run = zero_count > 261U ? 261U : zero_count;
            bits += run >= 6U ? 14ULL : 6ULL;
            zero_count -= run;
        }
    }
    table_bit_counts[chunk] = bits;
}

__device__ __forceinline__ void pixtreme_piz_write_table_bits(
    unsigned char* output,
    const long long output_start,
    long long* bit_offset,
    const unsigned int value,
    const int width
) {
    for (int shift = width - 1; shift >= 0; --shift) {
        const unsigned int bit = (value >> shift) & 1U;
        const long long absolute = *bit_offset;
        output[output_start + absolute / 8LL] |= (unsigned char)(bit << (7 - absolute % 8LL));
        ++(*bit_offset);
    }
}

__device__ __forceinline__ unsigned int pixtreme_piz_read_bit(
    const unsigned char* payload,
    const long long data_offset,
    const long long bit_offset
) {
    return (payload[data_offset + bit_offset / 8LL] >> (7 - bit_offset % 8LL)) & 1U;
}

extern "C" __global__ void pixtreme_piz_assemble_huffman(
    const unsigned char* code_lengths,
    const unsigned int* minimum_symbols,
    const unsigned int* repeat_symbols,
    const long long* table_sizes,
    const unsigned long long* data_bit_counts,
    const unsigned char* encoded_data,
    const long long* data_offsets,
    const long long* data_sizes,
    unsigned char* output,
    const long long* output_offsets,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long local = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count) return;
    const long long table_size = table_sizes[chunk];
    const long long data_size = data_sizes[chunk];
    const long long output_size = 20LL + table_size + data_size;
    if (local >= output_size) return;
    const long long destination = output_offsets[chunk] + local;
    if (local < 20LL) {
        const int field = (int)(local / 4LL);
        const int byte = (int)(local % 4LL);
        unsigned long long value = 0ULL;
        if (field == 0) value = minimum_symbols[chunk];
        else if (field == 1) value = repeat_symbols[chunk];
        else if (field == 2) value = (unsigned long long)table_size;
        else if (field == 3) value = data_bit_counts[chunk];
        output[destination] = (unsigned char)((value >> (8 * byte)) & 255ULL);
        return;
    }
    if (local < 20LL + table_size) {
        if (local != 20LL) return;
        const long long base = (long long)chunk * 65537LL;
        const long long table_start = output_offsets[chunk] + 20LL;
        long long bit_offset = 0LL;
        unsigned int symbol = minimum_symbols[chunk];
        const unsigned int maximum = repeat_symbols[chunk];
        while (symbol <= maximum) {
            const unsigned int length = code_lengths[base + symbol];
            if (length != 0U) {
                pixtreme_piz_write_table_bits(output, table_start, &bit_offset, length, 6);
                ++symbol;
                continue;
            }
            unsigned int zero_count = 0U;
            while (symbol <= maximum && code_lengths[base + symbol] == 0U) {
                ++zero_count;
                ++symbol;
            }
            while (zero_count > 0U) {
                const unsigned int run = zero_count > 261U ? 261U : zero_count;
                if (run == 1U) {
                    pixtreme_piz_write_table_bits(output, table_start, &bit_offset, 0U, 6);
                } else if (run <= 5U) {
                    pixtreme_piz_write_table_bits(output, table_start, &bit_offset, run + 57U, 6);
                } else {
                    pixtreme_piz_write_table_bits(output, table_start, &bit_offset, 63U, 6);
                    pixtreme_piz_write_table_bits(output, table_start, &bit_offset, run - 6U, 8);
                }
                zero_count -= run;
            }
        }
        return;
    }
    output[destination] = encoded_data[data_offsets[chunk] + local - 20LL - table_size];
}

extern "C" __global__ void pixtreme_piz_assemble_chunks(
    const unsigned char* bitmaps,
    const unsigned int* bitmap_minimums,
    const unsigned int* bitmap_maximums,
    const unsigned char* huffman,
    const long long* huffman_offsets,
    const long long* huffman_sizes,
    unsigned char* output,
    const long long* output_offsets,
    const long long* output_sizes,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long local = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count || local >= output_sizes[chunk]) return;
    const unsigned int minimum = bitmap_minimums[chunk];
    const unsigned int maximum = bitmap_maximums[chunk];
    const long long bitmap_size = minimum <= maximum ? (long long)maximum - minimum + 1LL : 0LL;
    const long long destination = output_offsets[chunk] + local;
    if (local < 4LL) {
        const unsigned int value = local < 2LL ? minimum : maximum;
        output[destination] = (unsigned char)((value >> (8 * (local % 2LL))) & 255U);
    } else if (local < 4LL + bitmap_size) {
        output[destination] = bitmaps[(long long)chunk * 8192LL + minimum + local - 4LL];
    } else if (local < 8LL + bitmap_size) {
        const long long byte = local - 4LL - bitmap_size;
        output[destination] = (unsigned char)(((unsigned long long)huffman_sizes[chunk] >> (8 * byte)) & 255ULL);
    } else {
        output[destination] = huffman[huffman_offsets[chunk] + local - 8LL - bitmap_size];
    }
}

extern "C" __global__ void pixtreme_piz_huffman_find_batches(
    const unsigned char* payload,
    const long long* data_offsets,
    const long long* data_bit_counts,
    const int* roots,
    const int* left_children,
    const int* right_children,
    const int* node_symbols,
    const unsigned int* pseudo_symbols,
    const long long* output_counts,
    const long long batch_bit_limit,
    const long long* batch_bases,
    const int* batch_capacities,
    long long* batch_bit_starts,
    long long* batch_bit_ends,
    long long* batch_output_starts,
    long long* batch_output_ends,
    unsigned short* batch_previous_symbols,
    unsigned char* batch_has_previous,
    int* batch_counts,
    int* status,
    const int record_count
) {
    const int record = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (record >= record_count) return;
    const long long data_bits = data_bit_counts[record];
    const long long expected = output_counts[record];
    const long long batch_base = batch_bases[record];
    const int batch_capacity = batch_capacities[record];
    long long batch_start_bit = 0LL;
    long long batch_start_output = 0LL;
    unsigned short batch_previous = 0;
    unsigned char batch_previous_valid = 0;
    unsigned short previous = 0;
    unsigned char has_previous = 0;
    long long bit = 0LL;
    long long produced = 0LL;
    int batch_count = 0;
    while (produced < expected) {
        const long long token_start = bit;
        int node = roots[record];
        while (node_symbols[node] < 0) {
            if (bit >= data_bits) { status[record] = 1; return; }
            const unsigned int next = pixtreme_piz_read_bit(payload, data_offsets[record], bit++);
            node = next ? right_children[node] : left_children[node];
            if (node < 0) { status[record] = 2; return; }
        }
        const unsigned int symbol = (unsigned int)node_symbols[node];
        unsigned int repeat = 0U;
        if (symbol == pseudo_symbols[record]) {
            if (!has_previous || bit + 8LL > data_bits) { status[record] = 3; return; }
            for (int index = 0; index < 8; ++index) {
                repeat = (repeat << 1) | pixtreme_piz_read_bit(payload, data_offsets[record], bit++);
            }
            if (repeat == 0U || produced + repeat > expected) { status[record] = 4; return; }
        } else if (symbol > 65535U || produced >= expected) {
            status[record] = 5;
            return;
        }
        if (bit - batch_start_bit > batch_bit_limit) {
            if (token_start == batch_start_bit || batch_count >= batch_capacity - 1) {
                status[record] = 6;
                return;
            }
            const long long batch = batch_base + batch_count++;
            batch_bit_starts[batch] = batch_start_bit;
            batch_bit_ends[batch] = token_start;
            batch_output_starts[batch] = batch_start_output;
            batch_output_ends[batch] = produced;
            batch_previous_symbols[batch] = batch_previous;
            batch_has_previous[batch] = batch_previous_valid;
            batch_start_bit = token_start;
            batch_start_output = produced;
            batch_previous = previous;
            batch_previous_valid = has_previous;
        }
        if (symbol == pseudo_symbols[record]) {
            produced += repeat;
        } else {
            previous = (unsigned short)symbol;
            has_previous = 1;
            ++produced;
        }
    }
    if (batch_count >= batch_capacity || bit - batch_start_bit > batch_bit_limit) {
        status[record] = 7;
        return;
    }
    const long long batch = batch_base + batch_count++;
    batch_bit_starts[batch] = batch_start_bit;
    batch_bit_ends[batch] = bit;
    batch_output_starts[batch] = batch_start_output;
    batch_output_ends[batch] = produced;
    batch_previous_symbols[batch] = batch_previous;
    batch_has_previous[batch] = batch_previous_valid;
    batch_counts[record] = batch_count;
}

extern "C" __global__ void pixtreme_piz_huffman_decode_batches(
    const unsigned char* payload,
    const long long* data_offsets,
    const int* roots,
    const int* left_children,
    const int* right_children,
    const int* node_symbols,
    const unsigned int* pseudo_symbols,
    const long long* output_offsets,
    unsigned short* output,
    const int* batch_records,
    const long long* batch_bases,
    const long long* batch_bit_starts,
    const long long* batch_bit_ends,
    const long long* batch_output_starts,
    const long long* batch_output_ends,
    const unsigned short* batch_previous_symbols,
    const unsigned char* batch_has_previous,
    const int* batch_counts,
    int* status,
    const long long allocated_batch_count
) {
    const long long batch = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= allocated_batch_count) return;
    const int record = batch_records[batch];
    const int local_batch = (int)(batch - batch_bases[record]);
    if (local_batch >= batch_counts[record]) return;
    const long long bit_end = batch_bit_ends[batch];
    const long long output_end = batch_output_ends[batch];
    long long bit = batch_bit_starts[batch];
    long long produced = batch_output_starts[batch];
    unsigned short previous = batch_previous_symbols[batch];
    unsigned char has_previous = batch_has_previous[batch];
    while (produced < output_end) {
        int node = roots[record];
        while (node_symbols[node] < 0) {
            if (bit >= bit_end) { status[batch] = 1; return; }
            const unsigned int next = pixtreme_piz_read_bit(payload, data_offsets[record], bit++);
            node = next ? right_children[node] : left_children[node];
            if (node < 0) { status[batch] = 2; return; }
        }
        const unsigned int symbol = (unsigned int)node_symbols[node];
        if (symbol == pseudo_symbols[record]) {
            if (!has_previous || bit + 8LL > bit_end) { status[batch] = 3; return; }
            unsigned int repeat = 0U;
            for (int index = 0; index < 8; ++index) {
                repeat = (repeat << 1) | pixtreme_piz_read_bit(payload, data_offsets[record], bit++);
            }
            if (repeat == 0U || produced + repeat > output_end) { status[batch] = 4; return; }
            for (unsigned int index = 0; index < repeat; ++index) {
                output[output_offsets[record] + produced++] = previous;
            }
        } else {
            if (symbol > 65535U || produced >= output_end) { status[batch] = 5; return; }
            previous = (unsigned short)symbol;
            has_previous = 1;
            output[output_offsets[record] + produced++] = previous;
        }
    }
    if (bit != bit_end || produced != output_end) status[batch] = 6;
}

__device__ __forceinline__ void pixtreme_piz_inverse_pair(
    const unsigned short low_word,
    const unsigned short high_word,
    const int w14,
    unsigned short* first_word,
    unsigned short* second_word
) {
    if (w14) {
        const int low = (int)((short)low_word);
        const int high = (int)((short)high_word);
        const int first = low + (high & 1) + (high >> 1);
        const int second = first - high;
        *first_word = (unsigned short)first;
        *second_word = (unsigned short)second;
    } else {
        const unsigned int second = ((unsigned int)low_word - ((unsigned int)high_word >> 1)) & 65535U;
        const unsigned int first = ((unsigned int)high_word + second - 32768U) & 65535U;
        *first_word = (unsigned short)first;
        *second_word = (unsigned short)second;
    }
}

__device__ __forceinline__ void pixtreme_piz_forward_pair(
    const unsigned short first_word,
    const unsigned short second_word,
    const int w14,
    unsigned short* low_word,
    unsigned short* high_word
) {
    if (w14) {
        const int first = (int)((short)first_word);
        const int second = (int)((short)second_word);
        *low_word = (unsigned short)((first + second) >> 1);
        *high_word = (unsigned short)(first - second);
    } else {
        const unsigned int offset_first = ((unsigned int)first_word + 32768U) & 65535U;
        unsigned int midpoint = (offset_first + (unsigned int)second_word) >> 1;
        const int difference = (int)offset_first - (int)((unsigned int)second_word);
        if (difference < 0) midpoint = (midpoint + 32768U) & 65535U;
        *low_word = (unsigned short)midpoint;
        *high_word = (unsigned short)difference;
    }
}

extern "C" __global__ void pixtreme_piz_forward_wavelet_level(
    unsigned short* words,
    const long long base,
    const int nx,
    const int ny,
    const int word_stride,
    const int p,
    const int w14,
    const long long full_count,
    const long long vertical_count,
    const long long horizontal_count
) {
    const long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int step = p * 2;
    const int columns = nx / step;
    const int rows = ny / step;
    const long long y_stride = (long long)word_stride * nx;
    if (task < full_count) {
        const int x = (int)(task % columns) * step;
        const int y = (int)(task / columns) * step;
        const long long i00 = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long i01 = i00 + (long long)p * word_stride;
        const long long i10 = i00 + (long long)p * y_stride;
        const long long i11 = i10 + (long long)p * word_stride;
        unsigned short low0, high0, low1, high1;
        pixtreme_piz_forward_pair(words[i00], words[i01], w14, &low0, &high0);
        pixtreme_piz_forward_pair(words[i10], words[i11], w14, &low1, &high1);
        pixtreme_piz_forward_pair(low0, low1, w14, &words[i00], &words[i10]);
        pixtreme_piz_forward_pair(high0, high1, w14, &words[i01], &words[i11]);
        return;
    }
    long long local = task - full_count;
    if (local < vertical_count) {
        const int x = columns * step;
        const int y = (int)local * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * y_stride;
        pixtreme_piz_forward_pair(words[first], words[second], w14, &words[first], &words[second]);
        return;
    }
    local -= vertical_count;
    if (local < horizontal_count) {
        const int x = (int)local * step;
        const int y = rows * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * word_stride;
        pixtreme_piz_forward_pair(words[first], words[second], w14, &words[first], &words[second]);
    }
}

extern "C" __global__ void pixtreme_piz_inverse_wavelet_level(
    unsigned short* words,
    const long long base,
    const int nx,
    const int ny,
    const int word_stride,
    const int p,
    const int w14,
    const long long full_count,
    const long long vertical_count,
    const long long horizontal_count
) {
    const long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int step = p * 2;
    const int columns = nx / step;
    const int rows = ny / step;
    const long long y_stride = (long long)word_stride * nx;
    if (task < full_count) {
        const int x = (int)(task % columns) * step;
        const int y = (int)(task / columns) * step;
        const long long i00 = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long i01 = i00 + (long long)p * word_stride;
        const long long i10 = i00 + (long long)p * y_stride;
        const long long i11 = i10 + (long long)p * word_stride;
        unsigned short low0, low1, high0, high1;
        pixtreme_piz_inverse_pair(words[i00], words[i10], w14, &low0, &low1);
        pixtreme_piz_inverse_pair(words[i01], words[i11], w14, &high0, &high1);
        pixtreme_piz_inverse_pair(low0, high0, w14, &words[i00], &words[i01]);
        pixtreme_piz_inverse_pair(low1, high1, w14, &words[i10], &words[i11]);
        return;
    }
    long long local = task - full_count;
    if (local < vertical_count) {
        const int x = columns * step;
        const int y = (int)local * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * y_stride;
        pixtreme_piz_inverse_pair(words[first], words[second], w14, &words[first], &words[second]);
        return;
    }
    local -= vertical_count;
    if (local < horizontal_count) {
        const int x = (int)local * step;
        const int y = rows * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * word_stride;
        pixtreme_piz_inverse_pair(words[first], words[second], w14, &words[first], &words[second]);
    }
}

extern "C" __global__ void pixtreme_piz_forward_wavelet_fields(
    unsigned short* words,
    const long long* field_descriptors,
    const int* pass_values,
    const int pass_index,
    const int field_count
) {
    const int field = (int)blockIdx.y;
    if (field >= field_count) return;
    const long long descriptor = (long long)field * 5LL;
    const int p = pass_values[(long long)pass_index * field_count + field];
    if (p <= 0) return;
    const long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = (int)field_descriptors[descriptor + 1LL];
    const int ny = (int)field_descriptors[descriptor + 2LL];
    const int word_stride = (int)field_descriptors[descriptor + 3LL];
    const int step = p * 2;
    const int columns = nx / step;
    const int rows = ny / step;
    const long long full_count = (long long)columns * rows;
    const long long vertical_count = (nx & p) ? rows : 0;
    const long long horizontal_count = (ny & p) ? columns : 0;
    const long long y_stride = (long long)word_stride * nx;
    const long long base = field_descriptors[descriptor];
    const int w14 = (int)field_descriptors[descriptor + 4LL];
    if (task < full_count) {
        const int x = (int)(task % columns) * step;
        const int y = (int)(task / columns) * step;
        const long long i00 = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long i01 = i00 + (long long)p * word_stride;
        const long long i10 = i00 + (long long)p * y_stride;
        const long long i11 = i10 + (long long)p * word_stride;
        unsigned short low0, high0, low1, high1;
        pixtreme_piz_forward_pair(words[i00], words[i01], w14, &low0, &high0);
        pixtreme_piz_forward_pair(words[i10], words[i11], w14, &low1, &high1);
        pixtreme_piz_forward_pair(low0, low1, w14, &words[i00], &words[i10]);
        pixtreme_piz_forward_pair(high0, high1, w14, &words[i01], &words[i11]);
        return;
    }
    long long local = task - full_count;
    if (local < vertical_count) {
        const int x = columns * step;
        const int y = (int)local * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * y_stride;
        pixtreme_piz_forward_pair(words[first], words[second], w14, &words[first], &words[second]);
        return;
    }
    local -= vertical_count;
    if (local < horizontal_count) {
        const int x = (int)local * step;
        const int y = rows * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * word_stride;
        pixtreme_piz_forward_pair(words[first], words[second], w14, &words[first], &words[second]);
    }
}

extern "C" __global__ void pixtreme_piz_inverse_wavelet_fields(
    unsigned short* words,
    const long long* field_descriptors,
    const int* pass_values,
    const int pass_index,
    const int field_count
) {
    const int field = (int)blockIdx.y;
    if (field >= field_count) return;
    const long long descriptor = (long long)field * 5LL;
    const int p = pass_values[(long long)pass_index * field_count + field];
    if (p <= 0) return;
    const long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = (int)field_descriptors[descriptor + 1LL];
    const int ny = (int)field_descriptors[descriptor + 2LL];
    const int word_stride = (int)field_descriptors[descriptor + 3LL];
    const int step = p * 2;
    const int columns = nx / step;
    const int rows = ny / step;
    const long long full_count = (long long)columns * rows;
    const long long vertical_count = (nx & p) ? rows : 0;
    const long long horizontal_count = (ny & p) ? columns : 0;
    const long long y_stride = (long long)word_stride * nx;
    const long long base = field_descriptors[descriptor];
    const int w14 = (int)field_descriptors[descriptor + 4LL];
    if (task < full_count) {
        const int x = (int)(task % columns) * step;
        const int y = (int)(task / columns) * step;
        const long long i00 = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long i01 = i00 + (long long)p * word_stride;
        const long long i10 = i00 + (long long)p * y_stride;
        const long long i11 = i10 + (long long)p * word_stride;
        unsigned short low0, low1, high0, high1;
        pixtreme_piz_inverse_pair(words[i00], words[i10], w14, &low0, &low1);
        pixtreme_piz_inverse_pair(words[i01], words[i11], w14, &high0, &high1);
        pixtreme_piz_inverse_pair(low0, high0, w14, &words[i00], &words[i01]);
        pixtreme_piz_inverse_pair(low1, high1, w14, &words[i10], &words[i11]);
        return;
    }
    long long local = task - full_count;
    if (local < vertical_count) {
        const int x = columns * step;
        const int y = (int)local * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * y_stride;
        pixtreme_piz_inverse_pair(words[first], words[second], w14, &words[first], &words[second]);
        return;
    }
    local -= vertical_count;
    if (local < horizontal_count) {
        const int x = (int)local * step;
        const int y = rows * step;
        const long long first = base + (long long)y * y_stride + (long long)x * word_stride;
        const long long second = first + (long long)p * word_stride;
        pixtreme_piz_inverse_pair(words[first], words[second], w14, &words[first], &words[second]);
    }
}

extern "C" __global__ void pixtreme_piz_restore_scatter_planes(
    const unsigned short* words,
    const long long* field_descriptors,
    const unsigned short* reverse_luts,
    const long long* record_descriptors,
    unsigned char* decoded,
    const long long row_bytes,
    const int width,
    int* status,
    const int plane_count
) {
    const int plane = (int)blockIdx.y;
    if (plane >= plane_count) return;
    const long long descriptor = (long long)plane * 6LL;
    const long long sample = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= field_descriptors[descriptor + 5LL]) return;
    const int record = (int)field_descriptors[descriptor];
    const long long row = sample / width;
    const long long x = sample - row * width;
    const int word_stride = (int)field_descriptors[descriptor + 2LL];
    const long long source = field_descriptors[descriptor + 1LL] + sample * word_stride;
    const long long destination =
        field_descriptors[descriptor + 3LL] + row * row_bytes +
        field_descriptors[descriptor + 4LL] + x * word_stride * 2LL;
    const long long record_descriptor = (long long)record * 2LL;
    const long long lut_offset = record_descriptors[record_descriptor];
    const unsigned int max_value = (unsigned int)record_descriptors[record_descriptor + 1LL];
    for (int word_slice = 0; word_slice < word_stride; ++word_slice) {
        const unsigned int compact = words[source + word_slice];
        if (compact > max_value) {
            atomicExch(&status[plane], 1);
            return;
        }
        const unsigned short value = reverse_luts[lut_offset + compact];
        decoded[destination + word_slice * 2] = (unsigned char)(value & 255U);
        decoded[destination + word_slice * 2 + 1] = (unsigned char)(value >> 8);
    }
}
"""


@lru_cache(maxsize=1)
def _piz_huffman_lengths_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_huffman_lengths")


@lru_cache(maxsize=1)
def _piz_huffman_codes_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_huffman_codes")


@lru_cache(maxsize=1)
def _piz_huffman_table_bits_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_huffman_table_bits")


@lru_cache(maxsize=1)
def _piz_assemble_huffman_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_assemble_huffman")


@lru_cache(maxsize=1)
def _piz_assemble_chunks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_assemble_chunks")


@lru_cache(maxsize=1)
def _piz_huffman_find_batches_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_huffman_find_batches")


@lru_cache(maxsize=1)
def _piz_huffman_decode_batches_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_huffman_decode_batches")


@lru_cache(maxsize=1)
def _piz_forward_wavelet_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_forward_wavelet_level")


@lru_cache(maxsize=1)
def _piz_inverse_wavelet_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_inverse_wavelet_level")


@lru_cache(maxsize=1)
def _piz_forward_wavelet_fields_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_forward_wavelet_fields")


@lru_cache(maxsize=1)
def _piz_inverse_wavelet_fields_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_inverse_wavelet_fields")


@lru_cache(maxsize=1)
def _piz_restore_scatter_planes_kernel() -> cp.RawKernel:
    return cp.RawKernel(_PIZ_GPU_SOURCE, "pixtreme_piz_restore_scatter_planes")


def _piz_decode_trees(
    tables: Sequence[_PizHuffmanTable],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    roots: list[int] = []
    left_children: list[int] = []
    right_children: list[int] = []
    symbols: list[int] = []

    def new_node() -> int:
        left_children.append(-1)
        right_children.append(-1)
        symbols.append(-1)
        return len(symbols) - 1

    for table in tables:
        root = new_node()
        roots.append(root)
        for code in table.codes:
            node = root
            for shift in range(code.length - 1, -1, -1):
                if symbols[node] >= 0:
                    raise _piz_error(
                        why="the PIZ Huffman decode tree contains a literal prefix of another code",
                        what=f"symbol={code.symbol}, length={code.length}",
                        how="provide a prefix-free canonical code-length assignment",
                    )
                branch = (code.code >> shift) & 1
                children = right_children if branch else left_children
                child = children[node]
                if child < 0:
                    child = new_node()
                    children[node] = child
                node = child
            if symbols[node] >= 0 or left_children[node] >= 0 or right_children[node] >= 0:
                raise _piz_error(
                    why="the PIZ Huffman decode tree contains a duplicate or conflicting code",
                    what=f"symbol={code.symbol}, length={code.length}, code={code.code}",
                    how="provide one distinct prefix-free canonical code per coded symbol",
                )
            symbols[node] = code.symbol
    return (
        np.asarray(roots, dtype=np.int32),
        np.asarray(left_children, dtype=np.int32),
        np.asarray(right_children, dtype=np.int32),
        np.asarray(symbols, dtype=np.int32),
    )


def _decode_piz_huffman_bounded_gpu(
    payload: cp.ndarray,
    *,
    data_offsets: Sequence[int],
    tables: Sequence[_PizHuffmanTable],
    output_counts: Sequence[int],
    record_labels: Sequence[int],
    batch_bit_limit: int,
) -> cp.ndarray:
    record_count = len(tables)
    if batch_bit_limit < 1:
        raise _piz_error(
            why="the bounded GPU PIZ Huffman batch limit is not positive",
            what=f"batch_bit_limit={batch_bit_limit}",
            how="configure a positive bit limit large enough for one complete Huffman token",
        )
    maximum_token_bits: list[int] = []
    for record, table in enumerate(tables):
        if not table.codes:
            raise _piz_error(
                why="the bounded GPU PIZ Huffman decoder lacks a materialized canonical table",
                what=f"record={record}, label={record_labels[record]!r}, code_count=0",
                how="parse the bounded host Huffman stream before splitting an oversized record",
            )
        pseudo_lengths = tuple(code.length for code in table.codes if code.symbol == table.maximum_symbol)
        if len(pseudo_lengths) != 1:
            raise _piz_error(
                why="the bounded GPU PIZ Huffman table lacks one repeat pseudo-symbol",
                what=f"record={record}, label={record_labels[record]!r}, pseudo_lengths={pseudo_lengths!r}",
                how="provide one canonical largest+1 code for previous-symbol repeats",
            )
        maximum_token_bits.append(max(max(code.length for code in table.codes), pseudo_lengths[0] + 8))
    oversized_tokens = tuple(
        (record_labels[index], token_bits)
        for index, token_bits in enumerate(maximum_token_bits)
        if token_bits > batch_bit_limit
    )
    if oversized_tokens:
        raise _piz_error(
            why="one PIZ Huffman token exceeds the configured bounded-batch bit limit",
            what=f"limit={batch_bit_limit}, label_token_bits={oversized_tokens!r}",
            how="use a batch limit of at least the largest complete literal or repeat token",
        )

    count_array = np.asarray(tuple(output_counts), dtype=np.int64)
    data_bit_counts = np.fromiter((table.data_bit_count for table in tables), dtype=np.int64, count=record_count)
    output_offsets = np.zeros(record_count, dtype=np.int64)
    if record_count > 1:
        output_offsets[1:] = np.cumsum(count_array[:-1], dtype=np.int64)
    batch_capacities = np.fromiter(
        (
            max(
                1,
                (int(data_bits) + max(1, batch_bit_limit - token_bits + 1) - 1)
                // max(1, batch_bit_limit - token_bits + 1)
                + 1,
            )
            for data_bits, token_bits in zip(data_bit_counts, maximum_token_bits, strict=True)
        ),
        dtype=np.int32,
        count=record_count,
    )
    batch_bases = np.zeros(record_count, dtype=np.int64)
    if record_count > 1:
        batch_bases[1:] = np.cumsum(batch_capacities[:-1], dtype=np.int64)
    allocated_batch_count = int(batch_capacities.sum())
    batch_records = np.repeat(np.arange(record_count, dtype=np.int32), batch_capacities)
    roots, left_children, right_children, node_symbols = _piz_decode_trees(tables)
    device_data_offsets = cp.asarray(np.asarray(tuple(data_offsets), dtype=np.int64))
    device_data_bit_counts = cp.asarray(data_bit_counts)
    device_roots = cp.asarray(roots)
    device_left_children = cp.asarray(left_children)
    device_right_children = cp.asarray(right_children)
    device_node_symbols = cp.asarray(node_symbols)
    device_pseudo_symbols = cp.asarray(
        np.fromiter((table.maximum_symbol for table in tables), dtype=np.uint32, count=record_count)
    )
    device_output_counts = cp.asarray(count_array)
    device_batch_bases = cp.asarray(batch_bases)
    device_batch_capacities = cp.asarray(batch_capacities)
    batch_bit_starts = cp.empty(allocated_batch_count, dtype=cp.int64)
    batch_bit_ends = cp.empty(allocated_batch_count, dtype=cp.int64)
    batch_output_starts = cp.empty(allocated_batch_count, dtype=cp.int64)
    batch_output_ends = cp.empty(allocated_batch_count, dtype=cp.int64)
    batch_previous_symbols = cp.empty(allocated_batch_count, dtype=cp.uint16)
    batch_has_previous = cp.empty(allocated_batch_count, dtype=cp.uint8)
    batch_counts = cp.empty(record_count, dtype=cp.int32)
    boundary_status = cp.zeros(record_count, dtype=cp.int32)
    record_grid = (record_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _piz_huffman_find_batches_kernel()(
        (record_grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            payload,
            device_data_offsets,
            device_data_bit_counts,
            device_roots,
            device_left_children,
            device_right_children,
            device_node_symbols,
            device_pseudo_symbols,
            device_output_counts,
            np.int64(batch_bit_limit),
            device_batch_bases,
            device_batch_capacities,
            batch_bit_starts,
            batch_bit_ends,
            batch_output_starts,
            batch_output_ends,
            batch_previous_symbols,
            batch_has_previous,
            batch_counts,
            boundary_status,
            np.int32(record_count),
        ),
    )
    boundary_control = cp.stack((batch_counts, boundary_status), axis=1).get()
    failed_boundaries = np.flatnonzero(boundary_control[:, 1])
    if failed_boundaries.size:
        failures = tuple(
            (record_labels[int(index)], int(boundary_control[int(index), 1])) for index in failed_boundaries
        )
        raise _piz_error(
            why="the bounded GPU PIZ Huffman scanner rejected a codeword, repeat, or batch boundary",
            what=f"record_status={failures!r}, batch_bit_limit={batch_bit_limit}",
            how="keep each complete token within the limit and match declared output ownership",
        )

    output = cp.empty(max(int(count_array.sum()), 1), dtype=cp.uint16)
    batch_status = cp.zeros(allocated_batch_count, dtype=cp.int32)
    _piz_huffman_decode_batches_kernel()(
        ((allocated_batch_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            payload,
            device_data_offsets,
            device_roots,
            device_left_children,
            device_right_children,
            device_node_symbols,
            device_pseudo_symbols,
            cp.asarray(output_offsets),
            output,
            cp.asarray(batch_records),
            device_batch_bases,
            batch_bit_starts,
            batch_bit_ends,
            batch_output_starts,
            batch_output_ends,
            batch_previous_symbols,
            batch_has_previous,
            batch_counts,
            batch_status,
            np.int64(allocated_batch_count),
        ),
    )
    batch_status_host = batch_status.get()
    failed_batches = np.flatnonzero(batch_status_host)
    if failed_batches.size:
        batch_failures = tuple(
            (
                record_labels[int(batch_records[int(index)])],
                int(index - batch_bases[int(batch_records[int(index)])]),
                int(batch_status_host[int(index)]),
            )
            for index in failed_batches
        )
        raise _piz_error(
            why="the bounded GPU PIZ Huffman decoder rejected a saved batch state or output span",
            what=f"record_batch_status={batch_failures!r}",
            how="preserve codeword boundaries, previous-symbol history, and exact output ownership",
        )
    return output[: int(count_array.sum())]


def _decode_piz_huffman_gpu(
    payload: cp.ndarray,
    *,
    data_offsets: Sequence[int],
    tables: Sequence[_PizHuffmanTable],
    output_counts: Sequence[int],
    record_labels: Sequence[int] | None = None,
    fallback_streams: Sequence[bytes] | None = None,
    fallback_offsets: Sequence[int] | None = None,
) -> cp.ndarray:
    record_count = len(tables)
    if len(data_offsets) != record_count or len(output_counts) != record_count:
        raise _piz_error(
            why="the GPU PIZ Huffman descriptor arrays have inconsistent record counts",
            what=f"tables={record_count}, data_offsets={len(data_offsets)}, output_counts={len(output_counts)}",
            how="provide one data span and expected word count per compressed chunk",
        )
    labels = tuple(range(record_count)) if record_labels is None else tuple(record_labels)
    if len(labels) != record_count:
        raise _piz_error(
            why="the GPU PIZ Huffman diagnostic labels do not match its decode records",
            what=f"records={record_count}, labels={len(labels)}",
            how="provide one chunk label for every compressed Huffman stream",
        )
    if (fallback_streams is None) != (fallback_offsets is None) or (
        fallback_streams is not None
        and fallback_offsets is not None
        and (len(fallback_streams) != record_count or len(fallback_offsets) != record_count)
    ):
        raise _piz_error(
            why="the GPU PIZ permissive fallback descriptors have inconsistent record counts",
            what=(
                f"records={record_count}, streams={None if fallback_streams is None else len(fallback_streams)}, "
                f"offsets={None if fallback_offsets is None else len(fallback_offsets)}"
            ),
            how="provide both one bounded Huffman stream and one device base offset per compressed chunk",
        )
    count_array = np.asarray(tuple(output_counts), dtype=np.int64)
    data_offset_array = np.asarray(tuple(data_offsets), dtype=np.int64)
    data_bit_counts = np.fromiter((table.data_bit_count for table in tables), dtype=np.int64, count=record_count)
    if np.any(count_array < 0) or np.any(data_offset_array < 0):
        raise _piz_error(
            why="the GPU PIZ Huffman output count or data offset is negative",
            what=f"output_counts={tuple(count_array)!r}, data_offsets={tuple(data_offset_array)!r}",
            how="use non-negative bounded descriptor values",
        )
    data_ends = data_offset_array + (data_bit_counts + 7) // 8
    if np.any(data_ends > int(payload.size)):
        raise _piz_error(
            why="the GPU PIZ Huffman data span exceeds its device payload",
            what=f"data_ends={tuple(data_ends)!r}, payload_size={int(payload.size)}",
            how="upload the complete bounded Huffman payload before decode",
        )
    if not record_count:
        return cp.empty(0, dtype=cp.uint16)
    canonical_tables = tuple(
        _DwaHuffmanTable(
            minimum_symbol=table.minimum_symbol,
            maximum_symbol=table.maximum_symbol,
            table_byte_count=table.table_span.size,
            data_bit_count=table.data_bit_count,
            code_lengths=table.code_lengths,
            codes=table.codes,
            table_span=_DwaByteSpan(table.table_span.start, table.table_span.end),
            data_span=_DwaByteSpan(table.data_span.start, table.data_span.end),
        )
        for table in tables
    )

    def bounded_decode(batch_start: int, batch_end: int) -> cp.ndarray:
        bounded_tables = tables[batch_start:batch_end]
        bounded_offsets = data_offsets[batch_start:batch_end]
        if any(not table.codes for table in bounded_tables):
            if fallback_streams is None or fallback_offsets is None:
                raise _piz_error(
                    why="an oversized or permissive PIZ Huffman batch lacks its bounded host streams",
                    what=f"batch={batch_start}:{batch_end}, labels={labels[batch_start:batch_end]!r}",
                    how="provide each validated Huffman stream and its device base offset for bounded decode",
                )
            bounded_tables = tuple(
                _parse_piz_huffman_table(stream) for stream in fallback_streams[batch_start:batch_end]
            )
            bounded_offsets = tuple(
                base_offset + table.data_span.start
                for base_offset, table in zip(fallback_offsets[batch_start:batch_end], bounded_tables, strict=True)
            )
        return _decode_piz_huffman_bounded_gpu(
            payload,
            data_offsets=bounded_offsets,
            tables=bounded_tables,
            output_counts=output_counts[batch_start:batch_end],
            record_labels=labels[batch_start:batch_end],
            batch_bit_limit=_PIZ_HUFFMAN_SEGMENT_BATCH_BITS,
        )

    outputs: list[cp.ndarray] = []
    record_index = 0
    while record_index < record_count:
        if int(data_bit_counts[record_index]) > _PIZ_HUFFMAN_SEGMENT_BATCH_BITS:
            outputs.append(bounded_decode(record_index, record_index + 1))
            record_index += 1
            continue
        batch_start = record_index
        batch_bit_count = 0
        while record_index < record_count:
            record_bits = int(data_bit_counts[record_index])
            if record_bits > _PIZ_HUFFMAN_SEGMENT_BATCH_BITS or (
                record_index > batch_start and batch_bit_count + record_bits > _PIZ_HUFFMAN_SEGMENT_BATCH_BITS
            ):
                break
            batch_bit_count += record_bits
            record_index += 1
        try:
            outputs.append(
                _decode_dwa_huffman_gpu(
                    payload,
                    data_offsets=data_offsets[batch_start:record_index],
                    tables=canonical_tables[batch_start:record_index],
                    output_counts=output_counts[batch_start:record_index],
                    record_labels=labels[batch_start:record_index],
                )
            )
        except _ExrGpuError:
            outputs.append(bounded_decode(batch_start, record_index))
    return outputs[0] if len(outputs) == 1 else cp.concatenate(outputs)


def _piz_forward_wavelet_fields_gpu(
    words: cp.ndarray,
    fields: Sequence[tuple[int, int, int, int, int]],
) -> None:
    values = cp.asarray(words)
    if values.dtype != cp.uint16 or values.ndim != 1 or not values.flags.c_contiguous:
        raise _gpu_error(
            why="the batched GPU PIZ forward-wavelet input is not a contiguous uint16 word vector",
            what=f"dtype={values.dtype}, shape={values.shape!r}, contiguous={values.flags.c_contiguous}",
            how="stage all channel-major write fields in one contiguous vector before batched wavelet",
        )
    if not fields:
        return
    field_descriptors = np.empty((len(fields), 5), dtype=np.int64)
    field_passes: list[tuple[int, ...]] = []
    for field_index, (base, nx, ny, word_stride, max_value) in enumerate(fields):
        if base < 0 or nx < 1 or ny < 1 or word_stride < 1:
            raise _gpu_error(
                why="a batched GPU PIZ forward-wavelet field has invalid geometry",
                what=f"field={field_index}, base={base}, nx={nx}, ny={ny}, word_stride={word_stride}",
                how="batch positive descriptor-owned channel fields with non-negative word offsets",
            )
        required = base + (ny - 1) * word_stride * nx + (nx - 1) * word_stride + 1
        if required > int(values.size):
            raise _gpu_error(
                why="a batched GPU PIZ forward-wavelet field exceeds the staged word vector",
                what=f"field={field_index}, required={required}, words={int(values.size)}",
                how="keep each channel word slice within its compressed chunk ownership",
            )
        field_descriptors[field_index] = (base, nx, ny, word_stride, int(_piz_uses_w14(max_value)))
        passes: list[int] = []
        p = 1
        while p * 2 <= min(nx, ny):
            passes.append(p)
            p *= 2
        field_passes.append(tuple(passes))
    pass_count = max((len(passes) for passes in field_passes), default=0)
    if not pass_count:
        return
    pass_values = np.zeros((pass_count, len(fields)), dtype=np.int32)
    maximum_tasks = np.zeros(pass_count, dtype=np.int64)
    for field_index, ((_, nx, ny, _, _), field_pass_values) in enumerate(zip(fields, field_passes, strict=True)):
        for pass_index, p in enumerate(field_pass_values):
            pass_values[pass_index, field_index] = p
            step = p * 2
            columns = nx // step
            rows = ny // step
            task_count = columns * rows + (rows if nx & p else 0) + (columns if ny & p else 0)
            maximum_tasks[pass_index] = max(maximum_tasks[pass_index], task_count)
    device_fields = cp.asarray(field_descriptors)
    device_pass_values = cp.asarray(pass_values)
    for pass_index, maximum_task_count in enumerate(maximum_tasks):
        if not maximum_task_count:
            continue
        grid_x = (int(maximum_task_count) + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _piz_forward_wavelet_fields_kernel()(
            (grid_x, len(fields)),
            (_EXR_THREADS_PER_BLOCK,),
            (
                values,
                device_fields,
                device_pass_values,
                np.int32(pass_index),
                np.int32(len(fields)),
            ),
        )


def _piz_inverse_wavelet_fields_gpu(
    words: cp.ndarray,
    fields: Sequence[tuple[int, int, int, int, int]],
) -> None:
    if words.dtype != cp.uint16 or words.ndim != 1 or not words.flags.c_contiguous:
        raise _piz_error(
            why="the batched GPU PIZ inverse-wavelet input is not a contiguous uint16 word vector",
            what=f"dtype={words.dtype}, shape={words.shape}, contiguous={words.flags.c_contiguous}",
            how="materialize all compressed chunk words in one contiguous vector before batched inverse wavelet",
        )
    if not fields:
        return
    field_descriptors = np.empty((len(fields), 5), dtype=np.int64)
    field_passes: list[tuple[int, ...]] = []
    for field_index, (base, nx, ny, word_stride, max_value) in enumerate(fields):
        if base < 0 or nx < 1 or ny < 1 or word_stride < 1:
            raise _piz_error(
                why="a batched GPU PIZ inverse-wavelet field has invalid geometry",
                what=f"field={field_index}, base={base}, nx={nx}, ny={ny}, word_stride={word_stride}",
                how="batch positive descriptor-owned channel fields with non-negative word offsets",
            )
        required = base + (ny - 1) * word_stride * nx + (nx - 1) * word_stride + 1
        if required > int(words.size):
            raise _piz_error(
                why="a batched GPU PIZ inverse-wavelet field exceeds the materialized word vector",
                what=f"field={field_index}, required={required}, words={int(words.size)}",
                how="keep each channel word slice within its compressed chunk ownership",
            )
        field_descriptors[field_index] = (base, nx, ny, word_stride, int(_piz_uses_w14(max_value)))
        if min(nx, ny) < 2:
            field_passes.append(())
            continue
        p = (1 << (min(nx, ny).bit_length() - 1)) // 2
        passes: list[int] = []
        while p >= 1:
            passes.append(p)
            p //= 2
        field_passes.append(tuple(passes))
    pass_count = max((len(passes) for passes in field_passes), default=0)
    if not pass_count:
        return
    pass_values = np.zeros((pass_count, len(fields)), dtype=np.int32)
    maximum_tasks = np.zeros(pass_count, dtype=np.int64)
    for field_index, ((_, nx, ny, _, _), field_pass_values) in enumerate(zip(fields, field_passes, strict=True)):
        for pass_index, p in enumerate(field_pass_values):
            pass_values[pass_index, field_index] = p
            step = p * 2
            columns = nx // step
            rows = ny // step
            task_count = columns * rows + (rows if nx & p else 0) + (columns if ny & p else 0)
            maximum_tasks[pass_index] = max(maximum_tasks[pass_index], task_count)
    device_fields = cp.asarray(field_descriptors)
    device_pass_values = cp.asarray(pass_values)
    for pass_index, maximum_task_count in enumerate(maximum_tasks):
        if not maximum_task_count:
            continue
        grid_x = (int(maximum_task_count) + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _piz_inverse_wavelet_fields_kernel()(
            (grid_x, len(fields)),
            (_EXR_THREADS_PER_BLOCK,),
            (
                words,
                device_fields,
                device_pass_values,
                np.int32(pass_index),
                np.int32(len(fields)),
            ),
        )


def _piz_restore_and_scatter_planes_gpu(
    words: cp.ndarray,
    reverse_luts: Sequence[np.ndarray],
    max_values: Sequence[int],
    fields: Sequence[tuple[int, int, int, int, int, int]],
    *,
    decoded: cp.ndarray,
    row_bytes: int,
    width: int,
) -> None:
    if len(reverse_luts) != len(max_values):
        raise _piz_error(
            why="the batched GPU PIZ reverse-LUT records have inconsistent counts",
            what=f"luts={len(reverse_luts)}, max_values={len(max_values)}",
            how="provide one compact reverse LUT and maximum value per compressed chunk",
        )
    if not fields:
        return
    compact_luts: list[np.ndarray] = []
    lut_sizes: list[int] = []
    for record, (reverse_lut, max_value) in enumerate(zip(reverse_luts, max_values, strict=True)):
        lut = np.asarray(reverse_lut, dtype=np.uint16)
        if lut.ndim != 1 or not 0 <= max_value < int(lut.size):
            raise _piz_error(
                why="a batched GPU PIZ reverse LUT does not cover its compact alphabet",
                what=f"record={record}, shape={lut.shape!r}, max_value={max_value}",
                how="build each 65536-entry reverse LUT from the matching chunk bitmap",
            )
        compact_luts.append(lut[: max_value + 1])
        lut_sizes.append(max_value + 1)
    lut_offsets = _prefix_offsets(lut_sizes)
    packed_luts = cp.asarray(np.concatenate(compact_luts))
    field_descriptors = np.asarray(tuple(fields), dtype=np.int64)
    if field_descriptors.shape != (len(fields), 6):
        raise _piz_error(
            why="the batched GPU PIZ scatter descriptors do not have six fields",
            what=f"shape={field_descriptors.shape!r}, planes={len(fields)}",
            how="provide record, word offset/stride, destination offsets, and sample count per plane",
        )
    plane_records = field_descriptors[:, 0]
    sample_counts = field_descriptors[:, 5]
    if np.any(plane_records < 0) or np.any(plane_records >= len(reverse_luts)) or np.any(sample_counts < 0):
        raise _piz_error(
            why="a batched GPU PIZ scatter plane has invalid LUT ownership or sample count",
            what=f"plane_records={tuple(plane_records)!r}, sample_counts={tuple(sample_counts)!r}",
            how="map every channel plane to one compressed chunk and a non-negative sample span",
        )
    maximum_sample_count = int(sample_counts.max(initial=0))
    if not maximum_sample_count:
        return
    record_descriptors = np.empty((len(reverse_luts), 2), dtype=np.int64)
    record_descriptors[:, 0] = lut_offsets
    record_descriptors[:, 1] = max_values
    device_fields = cp.asarray(field_descriptors)
    device_records = cp.asarray(record_descriptors)
    status = cp.zeros(len(fields), dtype=cp.int32)
    grid_x = (maximum_sample_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _piz_restore_scatter_planes_kernel()(
        (grid_x, len(fields)),
        (_EXR_THREADS_PER_BLOCK,),
        (
            words,
            device_fields,
            packed_luts,
            device_records,
            decoded,
            np.int64(row_bytes),
            np.int32(width),
            status,
            np.int32(len(fields)),
        ),
    )
    status_host = status.get()
    failed = np.flatnonzero(status_host)
    if failed.size:
        raise _piz_error(
            why="the batched GPU PIZ inverse wavelet produced a LUT index outside the compact alphabet",
            what=f"failed_planes={tuple(int(index) for index in failed)!r}",
            how="verify the Huffman words, wavelet fields, and chunk-local reverse LUT ownership",
        )


def _encode_piz_chunks_gpu(
    raw: cp.ndarray,
    raw_offsets: Sequence[int],
    raw_sizes: Sequence[int],
    *,
    row_counts: Sequence[int],
    width: int,
    channel_count: int,
    pixel_type: int,
) -> tuple[cp.ndarray, tuple[int, ...]]:
    source = cp.ascontiguousarray(raw, dtype=cp.uint8).reshape(-1)
    offsets = tuple(int(value) for value in raw_offsets)
    sizes = tuple(int(value) for value in raw_sizes)
    rows = tuple(int(value) for value in row_counts)
    chunk_count = len(rows)
    word_stride = 1 if pixel_type == 1 else 2
    if pixel_type not in (0, 1, 2) or width < 1 or channel_count < 1 or not rows:
        raise _gpu_error(
            why="the PIZ write descriptor has an invalid pixel type or image geometry",
            what=(f"pixel_type={pixel_type}, width={width}, channel_count={channel_count}, row_counts={rows!r}"),
            how="encode positive scanline chunks containing UINT, HALF, or FLOAT channels",
        )
    expected_sizes = tuple(row_count * width * channel_count * word_stride * 2 for row_count in rows)
    if offsets != _prefix_offsets(sizes) or sizes != expected_sizes or int(source.size) != sum(sizes):
        raise _gpu_error(
            why="the PIZ write chunk descriptors do not consume the packed scanline bytes exactly",
            what=(f"offsets={offsets!r}, sizes={sizes!r}, expected_sizes={expected_sizes!r}, raw_bytes={source.size}"),
            how="derive contiguous 32-line chunk ranges from the packed file-channel rows",
        )

    total_rows = sum(rows)
    scanline_words = source.view(cp.uint16).reshape(total_rows, channel_count, width, word_stride)
    staged_parts: list[cp.ndarray] = []
    row_start = 0
    for row_count in rows:
        staged_parts.append(
            cp.ascontiguousarray(scanline_words[row_start : row_start + row_count].transpose(1, 0, 2, 3)).reshape(-1)
        )
        row_start += row_count
    staged_words = cp.concatenate(staged_parts)
    chunk_word_sizes = tuple(size // 2 for size in sizes)
    chunk_word_offsets = _prefix_offsets(chunk_word_sizes)
    chunk_ids = cp.repeat(cp.arange(chunk_count, dtype=cp.int32), cp.asarray(chunk_word_sizes, dtype=cp.int64))

    histogram_indices = chunk_ids.astype(cp.int64) * np.int64(1 << 16) + staged_words.astype(cp.int64)
    histograms = cp.bincount(histogram_indices, minlength=chunk_count * (1 << 16)).reshape(chunk_count, 1 << 16)
    used = histograms > 0
    used[:, 0] = False
    bitmap_weights = cp.asarray((1, 2, 4, 8, 16, 32, 64, 128), dtype=cp.uint16)
    bitmap_bytes = cp.sum(used.reshape(chunk_count, _PIZ_BITMAP_BYTE_COUNT, 8) * bitmap_weights, axis=2).astype(
        cp.uint8
    )
    bitmap_nonzero = bitmap_bytes != 0
    bitmap_indices = cp.arange(_PIZ_BITMAP_BYTE_COUNT, dtype=cp.int32)
    bitmap_minimum = cp.min(cp.where(bitmap_nonzero, bitmap_indices[None, :], _PIZ_BITMAP_BYTE_COUNT - 1), axis=1)
    bitmap_maximum = cp.max(cp.where(bitmap_nonzero, bitmap_indices[None, :], 0), axis=1)
    included = used.copy()
    included[:, 0] = True
    ranks = cp.cumsum(included, axis=1, dtype=cp.uint32) - np.uint32(1)
    forward_lut = cp.where(included, ranks, 0).astype(cp.uint16)
    max_values = cp.sum(used, axis=1, dtype=cp.uint32)
    descriptor_matrix = cp.stack((bitmap_minimum, bitmap_maximum, max_values), axis=1)
    descriptor_host = descriptor_matrix.get().astype(np.int64, copy=False)

    staged_words = cp.ascontiguousarray(forward_lut[chunk_ids, staged_words], dtype=cp.uint16)
    wavelet_fields: list[tuple[int, int, int, int, int]] = []
    for chunk_index, row_count in enumerate(rows):
        chunk_offset = chunk_word_offsets[chunk_index]
        plane_word_count = row_count * width * word_stride
        max_value = int(descriptor_host[chunk_index, 2])
        for channel_index in range(channel_count):
            plane_offset = chunk_offset + channel_index * plane_word_count
            for word_slice in range(word_stride):
                wavelet_fields.append(
                    (
                        plane_offset + word_slice,
                        width,
                        row_count,
                        word_stride,
                        max_value,
                    )
                )
    _piz_forward_wavelet_fields_gpu(staged_words, wavelet_fields)

    huffman, huffman_offsets, huffman_sizes, symbol_counts = _encode_piz_huffman_chunks_gpu(
        staged_words,
        chunk_ids,
        chunk_count,
    )
    if symbol_counts != chunk_word_sizes:
        raise _gpu_error(
            why="the PIZ Huffman histograms do not cover each transformed chunk exactly",
            what=f"histogram_counts={symbol_counts!r}, expected_words={chunk_word_sizes!r}",
            how="keep transformed word ownership chunk-local through frequency counting",
        )

    bitmap_sizes = np.maximum(descriptor_host[:, 1] - descriptor_host[:, 0] + 1, 0)
    compressed_size_tuple = tuple(
        8 + int(bitmap_size) + huffman_size
        for bitmap_size, huffman_size in zip(bitmap_sizes, huffman_sizes, strict=True)
    )
    compressed_offsets = _prefix_offsets(compressed_size_tuple)
    compressed = cp.empty(sum(compressed_size_tuple), dtype=cp.uint8)
    _piz_assemble_chunks_kernel()(
        (_maximum_block_count(compressed_size_tuple), chunk_count),
        (_EXR_THREADS_PER_BLOCK,),
        (
            bitmap_bytes.reshape(-1),
            bitmap_minimum.astype(cp.uint32),
            bitmap_maximum.astype(cp.uint32),
            huffman,
            _device_i64(huffman_offsets),
            _device_i64(huffman_sizes),
            compressed,
            _device_i64(compressed_offsets),
            _device_i64(compressed_size_tuple),
            np.int32(chunk_count),
        ),
    )
    return _select_exr_payloads(
        source,
        offsets,
        sizes,
        compressed,
        compressed_offsets,
        compressed_size_tuple,
    )


def _piz_chunk_leader_control(
    container: _ExrContainer,
    descriptor: _PizChunkDescriptor,
    *,
    parse_table: bool,
) -> tuple[int, int, bytes, bytes, _PizHuffmanTable]:
    if descriptor.raw_stored or descriptor.bitmap_range is None or descriptor.huffman_leader is None:
        raise _piz_error(
            why="the PIZ compressed decode control received a raw-stored or incomplete descriptor",
            what=f"chunk_y={descriptor.chunk_y}, raw_stored={descriptor.raw_stored}",
            how="bypass raw chunks and decode only descriptors with bitmap and Huffman spans",
        )
    bitmap_minimum, bitmap_maximum = descriptor.bitmap_range
    bitmap_slice = container.data[descriptor.bitmap_span.start : descriptor.bitmap_span.end]
    huffman_stream = container.data[descriptor.huffman_span.start : descriptor.huffman_span.end]
    leader = descriptor.huffman_leader
    table_start = leader.span.size
    table_end = table_start + leader.table_byte_count
    data_end = table_end + (leader.data_bit_count + 7) // 8
    if not parse_table and leader.reserved == 0 and data_end == len(huffman_stream):
        table = _PizHuffmanTable(
            minimum_symbol=leader.minimum_symbol,
            maximum_symbol=leader.maximum_symbol,
            declared_table_byte_count=leader.table_byte_count,
            data_bit_count=leader.data_bit_count,
            reserved=leader.reserved,
            code_lengths=(),
            codes=(),
            table_span=_PizByteSpan(table_start, table_end),
            data_span=_PizByteSpan(table_end, data_end),
        )
    else:
        table = _parse_piz_huffman_table(huffman_stream)
    if (
        table.minimum_symbol != leader.minimum_symbol
        or table.maximum_symbol != leader.maximum_symbol
        or table.declared_table_byte_count != leader.table_byte_count
        or table.data_bit_count != leader.data_bit_count
        or table.reserved != leader.reserved
    ):
        raise _piz_error(
            why="the PIZ Huffman transform table differs from its validated descriptor leader",
            what=f"chunk_y={descriptor.chunk_y}, leader={leader!r}, table={table!r}",
            how="decode the exact bounded Huffman span recorded by the container parser",
        )
    return bitmap_minimum, bitmap_maximum, bitmap_slice, huffman_stream, table


def _piz_chunk_decode_control(
    container: _ExrContainer,
    descriptor: _PizChunkDescriptor,
    *,
    parse_table: bool = True,
    compact_lut: bool = False,
) -> tuple[np.ndarray, int, bytes, _PizHuffmanTable]:
    bitmap_minimum, bitmap_maximum, bitmap_slice, huffman_stream, table = _piz_chunk_leader_control(
        container, descriptor, parse_table=parse_table
    )
    if compact_lut:
        expected_bitmap_size = bitmap_maximum - bitmap_minimum + 1 if bitmap_minimum <= bitmap_maximum else 0
        if len(bitmap_slice) != expected_bitmap_size:
            raise _piz_error(
                why="the compact GPU PIZ reverse-LUT bitmap slice differs from its inclusive range",
                what=(
                    f"minimum={bitmap_minimum}, maximum={bitmap_maximum}, "
                    f"received={len(bitmap_slice)}, expected={expected_bitmap_size}"
                ),
                how="provide exactly max-min+1 bitmap bytes, or no bytes for an empty range",
            )
        relative_marked = np.flatnonzero(np.unpackbits(np.frombuffer(bitmap_slice, dtype=np.uint8), bitorder="little"))
        marked = relative_marked + bitmap_minimum * 8
        marked = marked[marked != 0]
        reverse_lut = np.empty(marked.size + 1, dtype=np.uint16)
        reverse_lut[0] = 0
        if marked.size:
            reverse_lut[1:] = marked.astype(np.uint16, copy=False)
        max_value = int(marked.size)
    else:
        reverse_lut, max_value = _piz_reverse_lut(bitmap_minimum, bitmap_maximum, bitmap_slice)
    return reverse_lut, max_value, huffman_stream, table


def _piz_chunk_decode_controls_gpu(
    container: _ExrContainer,
    descriptors: Sequence[_PizChunkDescriptor],
    *,
    parse_table: bool,
    compact_lut: bool,
) -> tuple[tuple[np.ndarray, int, bytes, _PizHuffmanTable], ...]:
    if parse_table or not compact_lut:
        return tuple(
            _piz_chunk_decode_control(
                container,
                descriptor,
                parse_table=parse_table,
                compact_lut=compact_lut,
            )
            for descriptor in descriptors
        )
    leader_controls = tuple(
        _piz_chunk_leader_control(container, descriptor, parse_table=False) for descriptor in descriptors
    )
    bitmap_sizes = tuple(len(control[2]) for control in leader_controls)
    bitmap_offsets = _prefix_offsets(bitmap_sizes)
    bitmap_blob = b"".join(control[2] for control in leader_controls)
    marked_bits = np.flatnonzero(np.unpackbits(np.frombuffer(bitmap_blob, dtype=np.uint8), bitorder="little"))
    bit_boundaries = np.asarray((*bitmap_offsets, len(bitmap_blob)), dtype=np.int64) * np.int64(8)
    marked_boundaries = np.searchsorted(marked_bits, bit_boundaries)
    controls: list[tuple[np.ndarray, int, bytes, _PizHuffmanTable]] = []
    for record, (bitmap_minimum, bitmap_maximum, bitmap_slice, huffman_stream, table) in enumerate(leader_controls):
        expected_bitmap_size = bitmap_maximum - bitmap_minimum + 1 if bitmap_minimum <= bitmap_maximum else 0
        if len(bitmap_slice) != expected_bitmap_size:
            raise _piz_error(
                why="the batched compact GPU PIZ reverse-LUT bitmap differs from its inclusive range",
                what=(
                    f"record={record}, minimum={bitmap_minimum}, maximum={bitmap_maximum}, "
                    f"received={len(bitmap_slice)}, expected={expected_bitmap_size}"
                ),
                how="provide exactly max-min+1 bitmap bytes, or no bytes for an empty range",
            )
        start = int(marked_boundaries[record])
        end = int(marked_boundaries[record + 1])
        marked = marked_bits[start:end] - bit_boundaries[record] + bitmap_minimum * 8
        marked = marked[marked != 0]
        reverse_lut = np.empty(marked.size + 1, dtype=np.uint16)
        reverse_lut[0] = 0
        if marked.size:
            reverse_lut[1:] = marked.astype(np.uint16, copy=False)
        controls.append((reverse_lut, int(marked.size), huffman_stream, table))
    return tuple(controls)


def _piz_materialize_chunk_host(container: _ExrContainer, chunk: _ExrChunk) -> np.ndarray:
    descriptor = chunk.piz
    if descriptor is None:
        raise _piz_error(
            why="the PIZ host materializer received a chunk without a Phase 4 descriptor",
            what=f"chunk_y={chunk.y}",
            how="materialize only chunks from a PIZ-eligible parsed container",
        )
    if descriptor.raw_stored:
        payload = container.data[descriptor.payload_span.start : descriptor.payload_span.end]
        if not payload:
            return np.zeros(descriptor.expected_packed_size, dtype=np.uint8)
        if len(payload) != descriptor.expected_packed_size:
            raise _piz_error(
                why="the PIZ raw-stored payload differs from its expected packed byte count",
                what=f"chunk_y={chunk.y}, received={len(payload)}, expected={descriptor.expected_packed_size}",
                how="store either zero bytes or exactly one complete raw packed chunk",
            )
        return np.frombuffer(payload, dtype=np.uint8).copy()

    reverse_lut, max_value, huffman_stream, table = _piz_chunk_decode_control(container, descriptor)
    words = _decode_piz_huffman_host(
        huffman_stream,
        table,
        expected_count=descriptor.expected_output_word_count,
    )
    width = container.data_window[2] - container.data_window[0] + 1
    for plane in descriptor.channel_planes:
        plane_end = plane.word_offset + plane.word_count
        if plane_end > words.size:
            raise _piz_error(
                why="the PIZ inverse-wavelet channel plane exceeds the materialized word vector",
                what=f"chunk_y={chunk.y}, channel={plane.channel_name!r}, plane_end={plane_end}, words={words.size}",
                how="match channel-plane ownership to the descriptor output word count",
            )
        plane_words = words[plane.word_offset : plane_end]
        for word_slice in range(plane.word_slice_count):
            _piz_inverse_wavelet_host(
                plane_words,
                nx=width,
                ny=descriptor.row_count,
                word_stride=plane.word_slice_count,
                word_slice=word_slice,
                max_value=max_value,
            )
    invalid = np.flatnonzero(words > max_value)
    if invalid.size:
        first = int(invalid[0])
        raise _piz_error(
            why="the PIZ inverse wavelet produced a LUT index outside the compact alphabet",
            what=f"chunk_y={chunk.y}, word={first}, value={int(words[first])}, maxValue={max_value}",
            how="verify the Huffman words, wavelet mode, field strides, and level barriers",
        )
    words = reverse_lut[words]

    row_bytes = sum(width * channel.bytes_per_sample for channel in container.parts[0].channels)
    if row_bytes * descriptor.row_count != descriptor.expected_packed_size:
        raise _piz_error(
            why="the PIZ output row layout differs from the descriptor packed byte count",
            what=(
                f"chunk_y={chunk.y}, row_bytes={row_bytes}, rows={descriptor.row_count}, "
                f"expected={descriptor.expected_packed_size}"
            ),
            how="derive output rows from the same file-channel ordering as the descriptor",
        )
    output = np.empty(descriptor.expected_packed_size, dtype=np.uint8)
    output_rows = output.reshape(descriptor.row_count, row_bytes)
    channel_offset = 0
    for plane in descriptor.channel_planes:
        plane_words = words[plane.word_offset : plane.word_offset + plane.word_count]
        plane_bytes = (
            plane_words.astype("<u2", copy=False)
            .view(np.uint8)
            .reshape(
                descriptor.row_count,
                width * plane.bytes_per_sample,
            )
        )
        channel_end = channel_offset + width * plane.bytes_per_sample
        output_rows[:, channel_offset:channel_end] = plane_bytes
        channel_offset = channel_end
    if channel_offset != row_bytes:
        raise _piz_error(
            why="the PIZ channel-plane scatter does not own one complete output row",
            what=f"chunk_y={chunk.y}, consumed={channel_offset}, row_bytes={row_bytes}",
            how="scatter every file channel once in alphabetical file order",
        )
    return output


def _read_exr_piz_custom_cpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    sizes = np.fromiter(
        (chunk.expected_size for chunk in container.chunks), dtype=np.int64, count=len(container.chunks)
    )
    offsets = _numpy_offsets(sizes)
    restored = np.empty(int(sizes.sum()), dtype=np.uint8)
    for index, chunk in enumerate(container.chunks):
        materialized = _piz_materialize_chunk_host(container, chunk)
        start = int(offsets[index])
        end = start + int(sizes[index])
        if materialized.size != int(sizes[index]):
            raise _piz_error(
                why="the PIZ host materializer returned a different packed chunk size",
                what=f"chunk_y={chunk.y}, received={materialized.size}, expected={int(sizes[index])}",
                how="materialize exactly one descriptor-owned raw byte span per output chunk",
            )
        restored[start:end] = materialized
    host_selected = _select_exr_host_pixels(container, selected, restored, output_dtype=output_dtype)
    return cp.asarray(host_selected)


def _read_exr_piz_gpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    width = container.data_window[2] - container.data_window[0] + 1
    row_bytes = sum(width * channel.bytes_per_sample for channel in container.parts[0].channels)
    decoded_sizes = np.fromiter(
        (chunk.expected_size for chunk in container.chunks), dtype=np.int64, count=len(container.chunks)
    )
    decoded_offsets = _numpy_offsets(decoded_sizes)
    decoded = cp.zeros(int(decoded_sizes.sum()), dtype=cp.uint8)
    device_file = cp.asarray(np.frombuffer(container.data, dtype=np.uint8))

    compressed_chunks: list[tuple[int, _PizChunkDescriptor]] = []
    data_offsets: list[int] = []
    tables: list[_PizHuffmanTable] = []
    fallback_streams: list[bytes] = []
    fallback_offsets: list[int] = []
    output_counts: list[int] = []
    labels: list[int] = []
    for chunk_index, chunk in enumerate(container.chunks):
        descriptor = chunk.piz
        if descriptor is None:
            raise _piz_error(
                why="the PIZ GPU materializer received a chunk without a Phase 4 descriptor",
                what=f"chunk_y={chunk.y}",
                how="materialize only chunks from a PIZ-eligible parsed container",
            )
        destination = int(decoded_offsets[chunk_index])
        if descriptor.raw_stored:
            if descriptor.stored_size == 0:
                continue
            if descriptor.stored_size != descriptor.expected_packed_size:
                raise _piz_error(
                    why="the PIZ GPU raw-stored payload differs from its expected packed byte count",
                    what=(
                        f"chunk_y={chunk.y}, stored={descriptor.stored_size}, "
                        f"expected={descriptor.expected_packed_size}"
                    ),
                    how="store either zero bytes or exactly one complete raw packed chunk",
                )
            decoded[destination : destination + descriptor.expected_packed_size] = device_file[
                descriptor.payload_span.start : descriptor.payload_span.end
            ]
            continue
        compressed_chunks.append((chunk_index, descriptor))

    compressed_records: list[tuple[int, _PizChunkDescriptor, np.ndarray, int, _PizHuffmanTable]] = []
    controls = _piz_chunk_decode_controls_gpu(
        container,
        tuple(descriptor for _, descriptor in compressed_chunks),
        parse_table=False,
        compact_lut=True,
    )
    for (chunk_index, descriptor), (reverse_lut, max_value, huffman_stream, table) in zip(
        compressed_chunks, controls, strict=True
    ):
        compressed_records.append((chunk_index, descriptor, reverse_lut, max_value, table))
        data_offsets.append(descriptor.huffman_span.start + table.data_span.start)
        tables.append(table)
        fallback_streams.append(huffman_stream)
        fallback_offsets.append(descriptor.huffman_span.start)
        output_counts.append(descriptor.expected_output_word_count)
        labels.append(descriptor.chunk_y)

    words = _decode_piz_huffman_gpu(
        device_file,
        data_offsets=data_offsets,
        tables=tables,
        output_counts=output_counts,
        record_labels=labels,
        fallback_streams=fallback_streams,
        fallback_offsets=fallback_offsets,
    )
    word_offsets = _prefix_offsets(output_counts)
    channel_offsets: list[int] = []
    channel_cursor = 0
    for channel in container.parts[0].channels:
        channel_offsets.append(channel_cursor)
        channel_cursor += width * channel.bytes_per_sample
    if channel_cursor != row_bytes:
        raise _piz_error(
            why="the PIZ GPU channel offsets do not cover one complete packed row",
            what=f"consumed={channel_cursor}, row_bytes={row_bytes}",
            how="derive each channel offset once in alphabetical file order",
        )

    wavelet_fields: list[tuple[int, int, int, int, int]] = []
    scatter_fields: list[tuple[int, int, int, int, int, int]] = []
    reverse_luts: list[np.ndarray] = []
    max_values: list[int] = []
    for record_index, (chunk_index, descriptor, reverse_lut, max_value, _table) in enumerate(compressed_records):
        word_start = word_offsets[record_index]
        reverse_luts.append(reverse_lut)
        max_values.append(max_value)
        destination = int(decoded_offsets[chunk_index])
        for plane in descriptor.channel_planes:
            plane_end = plane.word_offset + plane.word_count
            if plane_end > descriptor.expected_output_word_count:
                raise _piz_error(
                    why="the GPU PIZ inverse-wavelet channel plane exceeds its materialized word vector",
                    what=(
                        f"chunk_y={descriptor.chunk_y}, channel={plane.channel_name!r}, "
                        f"plane_end={plane_end}, words={descriptor.expected_output_word_count}"
                    ),
                    how="match channel-plane ownership to the descriptor output word count",
                )
            for word_slice in range(plane.word_slice_count):
                wavelet_fields.append(
                    (
                        word_start + plane.word_offset + word_slice,
                        width,
                        descriptor.row_count,
                        plane.word_slice_count,
                        max_value,
                    )
                )
            scatter_fields.append(
                (
                    record_index,
                    word_start + plane.word_offset,
                    plane.word_slice_count,
                    destination,
                    channel_offsets[plane.channel_index],
                    plane.sample_count,
                )
            )
    _piz_inverse_wavelet_fields_gpu(words, wavelet_fields)
    _piz_restore_and_scatter_planes_gpu(
        words,
        reverse_luts,
        max_values,
        scatter_fields,
        decoded=decoded,
        row_bytes=row_bytes,
        width=width,
    )
    return _unpack_exr_chunks(
        container,
        selected,
        decoded,
        decoded_offsets,
        decoded_sizes,
        np.zeros(len(container.chunks), dtype=np.uint8),
        output_dtype=output_dtype,
    )
