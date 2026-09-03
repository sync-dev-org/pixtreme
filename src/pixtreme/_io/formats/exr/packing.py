"""Shared OpenEXR packing, transforms, checksums, and GPU staging."""

from __future__ import annotations

import os
import struct
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, cast

import cupy as cp
import numpy as np

from pixtreme._io.formats.exr.container import (
    _EXR_COMPRESSION_CODES,
    _EXR_HOST_RESTORE_BATCH_BYTES,
    _EXR_LONG_NAMES_FLAG,
    _EXR_MAGIC,
    _EXR_MAX_GRID_X,
    _EXR_MAX_GRID_Y,
    _EXR_RESTORE_TILE_BYTES,
    _EXR_THREADS_PER_BLOCK,
    _EXR_VERSION,
    _ExrChannel,
    _ExrContainer,
    _ExrReadChunks,
    _gpu_error,
    _parser_error,
)


def _read_worker_count(task_count: int) -> int:
    return max(1, min(task_count, os.cpu_count() or 1, 16))


def _numpy_offsets(sizes: np.ndarray) -> np.ndarray:
    offsets = np.empty_like(sizes)
    if sizes.size:
        offsets[0] = 0
        np.cumsum(sizes[:-1], out=offsets[1:])
    return offsets


def _restore_predictor_host(transformed: np.ndarray) -> np.ndarray:
    values = np.asarray(transformed, dtype=np.uint8)
    cumulative = np.cumsum(values, dtype=np.int64)
    cumulative -= np.arange(values.size, dtype=np.int64) * 128
    return np.bitwise_and(cumulative, 255).astype(np.uint8)


def _restore_even_odd_host(grouped: np.ndarray) -> np.ndarray:
    restored = np.empty_like(grouped)
    half = (grouped.size + 1) // 2
    restored[::2] = grouped[:half]
    restored[1::2] = grouped[half:]
    return restored


def _restore_host_batch(transformed: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(transformed, axis=1, dtype=np.int64)
    cumulative -= np.arange(transformed.shape[1], dtype=np.int64)[None, :] * 128
    grouped = np.bitwise_and(cumulative, 255).astype(np.uint8)
    restored = np.empty_like(grouped)
    half = (grouped.shape[1] + 1) // 2
    restored[:, 0::2] = grouped[:, :half]
    restored[:, 1::2] = grouped[:, half:]
    return restored


def _restore_exr_host_chunks(prepared: _ExrReadChunks) -> np.ndarray:
    restored = prepared.host_decoded.copy()
    compressed_indices = np.flatnonzero(prepared.compressed)
    if not compressed_indices.size:
        return restored
    batches: list[tuple[int, np.ndarray]] = []
    run_start = 0
    while run_start < compressed_indices.size:
        first_index = int(compressed_indices[run_start])
        chunk_size = int(prepared.decoded_sizes[first_index])
        run_end = run_start + 1
        while (
            run_end < compressed_indices.size
            and compressed_indices[run_end] == compressed_indices[run_end - 1] + 1
            and prepared.decoded_sizes[compressed_indices[run_end]] == chunk_size
        ):
            run_end += 1
        maximum_chunks = max(1, _EXR_HOST_RESTORE_BATCH_BYTES // chunk_size)
        for batch_start in range(run_start, run_end, maximum_chunks):
            batch_end = min(batch_start + maximum_chunks, run_end)
            batch_first_index = int(compressed_indices[batch_start])
            batch_count = batch_end - batch_start
            offset = int(prepared.decoded_offsets[batch_first_index])
            size = batch_count * chunk_size
            transformed = prepared.host_decoded[offset : offset + size].reshape(batch_count, chunk_size)
            batches.append((offset, transformed))
        run_start = run_end
    with ThreadPoolExecutor(max_workers=_read_worker_count(len(batches))) as executor:
        restored_batches = executor.map(_restore_host_batch, (transformed for _, transformed in batches))
        for (offset, _), batch in zip(batches, restored_batches, strict=True):
            restored[offset : offset + batch.size] = batch.reshape(-1)
    return restored


@lru_cache(maxsize=8)
def _nvcomp_deflate_codec(device_id: int, stream_ptr: int) -> Any:
    from nvidia import nvcomp

    with cp.cuda.Device(device_id):
        return nvcomp.Codec(
            algorithm="Deflate",
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            cuda_stream=stream_ptr,
        )


_EXR_RESTORE_SOURCE = r"""
extern "C" __global__ void pixtreme_exr_restore_reduce(
    const unsigned char* data,
    const long long* tile_offsets,
    const int* tile_sizes,
    unsigned int* tile_sum_256,
    unsigned int* tile_sum_adler,
    unsigned int* tile_weighted_adler,
    const int tile_count
) {
    const int tid = (int)threadIdx.x;
    __shared__ unsigned int sum_256[256];
    __shared__ unsigned long long sum_a[256];
    __shared__ unsigned long long sum_b[256];
    for (int tile = (int)blockIdx.x; tile < tile_count; tile += (int)gridDim.x) {
        const long long base = tile_offsets[tile];
        const int size = tile_sizes[tile];
        unsigned int local_256 = 0;
        unsigned long long local_a = 0;
        unsigned long long local_b = 0;
        for (int index = tid; index < size; index += (int)blockDim.x) {
            const unsigned long long value = data[base + index];
            local_256 += (unsigned int)value;
            local_a += value;
            local_b += (unsigned long long)(size - index) * value;
        }
        sum_256[tid] = local_256;
        sum_a[tid] = local_a;
        sum_b[tid] = local_b;
        __syncthreads();
        for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sum_256[tid] += sum_256[tid + stride];
                sum_a[tid] += sum_a[tid + stride];
                sum_b[tid] += sum_b[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            tile_sum_256[tile] = sum_256[0] & 255U;
            tile_sum_adler[tile] = (unsigned int)(sum_a[0] % 65521ULL);
            tile_weighted_adler[tile] = (unsigned int)(sum_b[0] % 65521ULL);
        }
        __syncthreads();
    }
}

extern "C" __global__ void pixtreme_exr_restore_offsets(
    const long long* chunk_sizes,
    const unsigned char* compressed,
    const unsigned int* expected_adler,
    const long long* chunk_first_tile,
    const int* chunk_tile_counts,
    const int* tile_sizes,
    const unsigned int* tile_sum_256,
    const unsigned int* tile_sum_adler,
    const unsigned int* tile_weighted_adler,
    unsigned int* tile_prefix_256,
    int* status,
    const int validate_adler,
    const int chunk_count
) {
    for (int chunk = (int)blockIdx.x; chunk < chunk_count; chunk += (int)gridDim.x) {
        if (threadIdx.x == 0 && compressed[chunk]) {
            const long long first = chunk_first_tile[chunk];
            const int count = chunk_tile_counts[chunk];
            const unsigned long long size = (unsigned long long)chunk_sizes[chunk];
            unsigned int prefix_256 = 0;
            unsigned long long sum_a = 0;
            unsigned long long sum_b = 0;
            unsigned long long consumed = 0;
            for (int local_tile = 0; local_tile < count; ++local_tile) {
                const long long tile = first + local_tile;
                const unsigned long long tile_size = (unsigned long long)tile_sizes[tile];
                tile_prefix_256[tile] = prefix_256;
                prefix_256 = (prefix_256 + tile_sum_256[tile]) & 255U;
                sum_a = (sum_a + tile_sum_adler[tile]) % 65521ULL;
                const unsigned long long remaining = size - consumed - tile_size;
                sum_b = (
                    sum_b
                    + tile_weighted_adler[tile]
                    + (remaining % 65521ULL) * tile_sum_adler[tile]
                ) % 65521ULL;
                consumed += tile_size;
            }
            const unsigned int s1 = (unsigned int)((1ULL + sum_a) % 65521ULL);
            const unsigned int s2 = (unsigned int)((size + sum_b) % 65521ULL);
            const unsigned int observed = (s2 << 16) | s1;
            status[chunk] = !validate_adler || observed == expected_adler[chunk] ? 0 : 1;
        }
    }
}

extern "C" __global__ void pixtreme_exr_restore_finalize(
    unsigned char* data,
    const long long* tile_offsets,
    const int* tile_sizes,
    const long long* tile_chunk_byte_offsets,
    const unsigned int* tile_prefix_256,
    const int tile_count
) {
    const int tid = (int)threadIdx.x;
    __shared__ unsigned int scan[256];
    for (int tile = (int)blockIdx.x; tile < tile_count; tile += (int)gridDim.x) {
        const long long base = tile_offsets[tile];
        const int size = tile_sizes[tile];
        const int items_per_thread = (size + (int)blockDim.x - 1) / (int)blockDim.x;
        const int begin = tid * items_per_thread;
        const int end = begin + items_per_thread < size ? begin + items_per_thread : size;
        unsigned int local_total = 0;
        for (int index = begin; index < end; ++index) {
            local_total += (unsigned int)data[base + index];
        }
        scan[tid] = local_total;
        __syncthreads();
        for (int step = 1; step < (int)blockDim.x; step <<= 1) {
            const unsigned int addend = tid >= step ? scan[tid - step] : 0U;
            __syncthreads();
            if (tid >= step) {
                scan[tid] += addend;
            }
            __syncthreads();
        }
        unsigned int running = scan[tid] - local_total;
        const unsigned int tile_prefix = tile_prefix_256[tile];
        const long long chunk_byte_offset = tile_chunk_byte_offsets[tile];
        for (int index = begin; index < end; ++index) {
            running += (unsigned int)data[base + index];
            const unsigned long long predictor_index = (unsigned long long)(chunk_byte_offset + index);
            data[base + index] = (unsigned char)(
                (tile_prefix + running - (unsigned int)(128ULL * predictor_index)) & 255U
            );
        }
        __syncthreads();
    }
}
"""

_EXR_GATHER_SOURCE = r"""
extern "C" __global__ void pixtreme_exr_gather_raw(
    const unsigned char* staging,
    const long long* source_offsets,
    unsigned char* decoded,
    const long long* destination_offsets,
    const long long* sizes,
    const int range_count
) {
    for (int range = (int)blockIdx.x; range < range_count; range += (int)gridDim.x) {
        for (long long index = (long long)threadIdx.x; index < sizes[range]; index += blockDim.x) {
            decoded[destination_offsets[range] + index] = staging[source_offsets[range] + index];
        }
    }
}
"""

_EXR_UNPACK_TEMPLATE = r"""
#include <cuda_fp16.h>

typedef __OUTPUT_TYPE__ pixtreme_exr_output_t;

extern "C" __global__ void pixtreme_exr_unpack(
    const unsigned char* data,
    const long long* chunk_offsets,
    const long long* chunk_sizes,
    const unsigned char* even_odd_grouped,
    const long long* channel_offsets,
    const int* channel_types,
    pixtreme_exr_output_t* output,
    const long long element_count,
    const int width,
    const int height,
    const int output_channels,
    const int lines_per_chunk,
    const long long row_bytes
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {
        return;
    }
    const int output_channel = (int)(element % output_channels);
    const long long pixel = element / output_channels;
    const int x = (int)(pixel % width);
    const int y = (int)(pixel / width);
    const int chunk = y / lines_per_chunk;
    const int row_in_chunk = y - chunk * lines_per_chunk;
    const int pixel_type = channel_types[output_channel];
    const int byte_count = pixel_type == 1 ? 2 : 4;
    const long long raw_offset =
        (long long)row_in_chunk * row_bytes + channel_offsets[output_channel] + (long long)x * byte_count;
    const long long chunk_size = chunk_sizes[chunk];
    const long long half = (chunk_size + 1) / 2;
    unsigned int bits = 0;
    for (int byte_index = 0; byte_index < byte_count; ++byte_index) {
        const long long original = raw_offset + byte_index;
        const long long stored = even_odd_grouped[chunk]
            ? ((original & 1LL) ? half + original / 2 : original / 2)
            : original;
        bits |= (unsigned int)data[chunk_offsets[chunk] + stored] << (8 * byte_index);
    }
    __WRITE_OUTPUT__
}
"""

_EXR_PACK_TEMPLATE = r"""
typedef __INPUT_TYPE__ pixtreme_exr_input_t;

extern "C" __global__ void pixtreme_exr_pack(
    const pixtreme_exr_input_t* input,
    const int* channel_indices,
    unsigned char* output,
    const long long sample_count,
    const int width,
    const int channels,
    const int row_prefix_bytes
) {
    const long long sample = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (sample >= sample_count) {
        return;
    }
    const int x = (int)(sample % width);
    const long long row_channel = sample / width;
    const int output_channel = (int)(row_channel % channels);
    const long long y = row_channel / channels;
    const int input_channel = channel_indices[output_channel];
    const long long input_index = (y * width + x) * channels + input_channel;
    const long long row_bytes = (long long)width * channels * __BYTE_COUNT__;
    const long long output_index = y * (row_bytes + row_prefix_bytes) + row_prefix_bytes
        + ((long long)output_channel * width + x) * __BYTE_COUNT__;
    if (row_prefix_bytes == 8 && output_channel == 0 && x == 0) {
        const unsigned int chunk_y = (unsigned int)y;
        const unsigned int packed_size = (unsigned int)row_bytes;
        for (int byte_index = 0; byte_index < 4; ++byte_index) {
            output[y * (row_bytes + row_prefix_bytes) + byte_index] =
                (unsigned char)(chunk_y >> (8 * byte_index));
            output[y * (row_bytes + row_prefix_bytes) + 4 + byte_index] =
                (unsigned char)(packed_size >> (8 * byte_index));
        }
    }
    __BITS_TYPE__ bits = __READ_BITS__;
    for (int byte_index = 0; byte_index < __BYTE_COUNT__; ++byte_index) {
        output[output_index + byte_index] = (unsigned char)(bits >> (8 * byte_index));
    }
}
"""

_EXR_TRANSFORM_SOURCE = r"""
extern "C" __global__ void pixtreme_exr_transform(
    const unsigned char* raw,
    unsigned char* transformed,
    const long long* chunk_offsets,
    const long long* chunk_sizes,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long grouped_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count || grouped_index >= chunk_sizes[chunk]) {
        return;
    }
    const long long size = chunk_sizes[chunk];
    const long long half = (size + 1) / 2;
    const long long original_index = grouped_index < half
        ? grouped_index * 2
        : (grouped_index - half) * 2 + 1;
    const unsigned int current = raw[chunk_offsets[chunk] + original_index];
    unsigned int predicted = current;
    if (grouped_index > 0) {
        const long long previous_grouped = grouped_index - 1;
        const long long previous_original = previous_grouped < half
            ? previous_grouped * 2
            : (previous_grouped - half) * 2 + 1;
        const unsigned int previous = raw[chunk_offsets[chunk] + previous_original];
        predicted = (current - previous + 128U) & 255U;
    }
    transformed[chunk_offsets[chunk] + grouped_index] = (unsigned char)predicted;
}
"""

_EXR_ADLER_SOURCE = r"""
extern "C" __global__ void pixtreme_exr_adler(
    const unsigned char* data,
    const long long* chunk_offsets,
    const long long* chunk_sizes,
    unsigned int* adler,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (chunk >= chunk_count) {
        return;
    }
    const long long base = chunk_offsets[chunk];
    const long long size = chunk_sizes[chunk];
    const unsigned long long mod = 65521ULL;
    unsigned long long local_a = 0;
    unsigned long long local_b = 0;
    for (long long index = tid; index < size; index += blockDim.x) {
        const unsigned long long value = data[base + index];
        local_a = (local_a + value) % mod;
        local_b = (local_b + ((unsigned long long)(size - index) % mod) * value) % mod;
    }
    __shared__ unsigned long long sum_a[256];
    __shared__ unsigned long long sum_b[256];
    sum_a[tid] = local_a;
    sum_b[tid] = local_b;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_a[tid] = (sum_a[tid] + sum_a[tid + stride]) % mod;
            sum_b[tid] = (sum_b[tid] + sum_b[tid + stride]) % mod;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const unsigned int s1 = (unsigned int)((1ULL + sum_a[0]) % mod);
        const unsigned int s2 = (unsigned int)(((unsigned long long)size + sum_b[0]) % mod);
        adler[chunk] = (s2 << 16) | s1;
    }
}
"""

_EXR_WRAP_SOURCE = r"""
extern "C" __global__ void pixtreme_exr_wrap(
    const unsigned char* compressed,
    const long long* compressed_offsets,
    const long long* compressed_sizes,
    const unsigned int* adler,
    unsigned char* wrapped,
    const long long* wrapped_offsets,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count || index >= compressed_sizes[chunk] + 6) {
        return;
    }
    const long long compressed_size = compressed_sizes[chunk];
    unsigned char value;
    if (index == 0) {
        value = 0x78;
    } else if (index == 1) {
        value = 0x9c;
    } else if (index < compressed_size + 2) {
        value = compressed[compressed_offsets[chunk] + index - 2];
    } else {
        const int trailer_byte = (int)(index - compressed_size - 2);
        value = (unsigned char)(adler[chunk] >> (8 * (3 - trailer_byte)));
    }
    wrapped[wrapped_offsets[chunk] + index] = value;
}
"""

_EXR_SELECT_SOURCE = r"""
extern "C" __global__ void pixtreme_exr_select_payload(
    const unsigned char* raw,
    const long long* raw_offsets,
    const long long* raw_sizes,
    const unsigned char* wrapped,
    const long long* wrapped_offsets,
    const long long* wrapped_sizes,
    const unsigned char* use_wrapped,
    unsigned char* output,
    const long long* output_offsets,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count) {
        return;
    }
    const long long size = use_wrapped[chunk] ? wrapped_sizes[chunk] : raw_sizes[chunk];
    if (index >= size) {
        return;
    }
    output[output_offsets[chunk] + index] = use_wrapped[chunk]
        ? wrapped[wrapped_offsets[chunk] + index]
        : raw[raw_offsets[chunk] + index];
}
"""


@lru_cache(maxsize=1)
def _restore_reduce_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_RESTORE_SOURCE, "pixtreme_exr_restore_reduce")


@lru_cache(maxsize=1)
def _restore_offsets_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_RESTORE_SOURCE, "pixtreme_exr_restore_offsets")


@lru_cache(maxsize=1)
def _restore_finalize_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_RESTORE_SOURCE, "pixtreme_exr_restore_finalize")


@lru_cache(maxsize=1)
def _gather_raw_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_GATHER_SOURCE, "pixtreme_exr_gather_raw")


@lru_cache(maxsize=2)
def _unpack_kernel(output_dtype: str) -> cp.RawKernel:
    if output_dtype == "float16":
        source = _EXR_UNPACK_TEMPLATE.replace("__OUTPUT_TYPE__", "unsigned short").replace(
            "__WRITE_OUTPUT__", "output[element] = (unsigned short)bits;"
        )
    elif output_dtype == "uint32":
        source = _EXR_UNPACK_TEMPLATE.replace("__OUTPUT_TYPE__", "unsigned int").replace(
            "__WRITE_OUTPUT__", "output[element] = bits;"
        )
    else:
        source = _EXR_UNPACK_TEMPLATE.replace("__OUTPUT_TYPE__", "float").replace(
            "__WRITE_OUTPUT__",
            (
                "output[element] = pixel_type == 0 ? (float)bits : "
                "(pixel_type == 1 ? __half2float(__ushort_as_half((unsigned short)bits)) : __uint_as_float(bits));"
            ),
        )
    return cp.RawKernel(source, "pixtreme_exr_unpack")


@lru_cache(maxsize=4)
def _pack_kernel(input_dtype: str) -> cp.RawKernel:
    replacements = {
        "float16": ("unsigned short", "unsigned short", "input[input_index]", "2"),
        "float32": ("float", "unsigned int", "__float_as_uint(input[input_index])", "4"),
        "uint8": (
            "unsigned char",
            "unsigned int",
            "__float_as_uint((float)input[input_index] * (1.0f / 255.0f))",
            "4",
        ),
        "uint16": (
            "unsigned short",
            "unsigned int",
            "__float_as_uint((float)input[input_index] * (1.0f / 65535.0f))",
            "4",
        ),
        "uint32": ("unsigned int", "unsigned int", "input[input_index]", "4"),
    }
    input_type, bits_type, read_bits, byte_count = replacements[input_dtype]
    source = (
        _EXR_PACK_TEMPLATE.replace("__INPUT_TYPE__", input_type)
        .replace("__BITS_TYPE__", bits_type)
        .replace("__READ_BITS__", read_bits)
        .replace("__BYTE_COUNT__", byte_count)
    )
    return cp.RawKernel(source, "pixtreme_exr_pack")


@lru_cache(maxsize=1)
def _transform_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_TRANSFORM_SOURCE, "pixtreme_exr_transform")


@lru_cache(maxsize=1)
def _adler_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_ADLER_SOURCE, "pixtreme_exr_adler")


@lru_cache(maxsize=1)
def _wrap_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_WRAP_SOURCE, "pixtreme_exr_wrap")


@lru_cache(maxsize=1)
def _select_payload_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_SELECT_SOURCE, "pixtreme_exr_select_payload")


def _prefix_offsets(sizes: Sequence[int]) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for size in sizes:
        offsets.append(cursor)
        cursor += size
    return tuple(offsets)


def _device_i64(values: Sequence[int]) -> cp.ndarray:
    return cp.asarray(np.asarray(values, dtype=np.int64))


def _maximum_block_count(sizes: Sequence[int]) -> int:
    return max((size + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK for size in sizes)


def _chunk_launch_ranges(chunk_count: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(start + _EXR_MAX_GRID_Y, chunk_count)) for start in range(0, chunk_count, _EXR_MAX_GRID_Y))


def _restore_tile_descriptors(
    chunk_offsets: np.ndarray,
    chunk_sizes: np.ndarray,
    compressed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    compressed_indices = np.flatnonzero(compressed)
    chunk_first_tile = np.full(chunk_sizes.size, -1, dtype=np.int64)
    chunk_tile_counts = np.zeros(chunk_sizes.size, dtype=np.int32)
    if not compressed_indices.size:
        empty_i64 = np.empty(0, dtype=np.int64)
        return empty_i64, np.empty(0, dtype=np.int32), empty_i64, chunk_first_tile, chunk_tile_counts
    counts = ((chunk_sizes[compressed_indices] + _EXR_RESTORE_TILE_BYTES - 1) // _EXR_RESTORE_TILE_BYTES).astype(
        np.int32
    )
    first_tiles = _numpy_offsets(counts.astype(np.int64))
    chunk_first_tile[compressed_indices] = first_tiles
    chunk_tile_counts[compressed_indices] = counts
    tile_chunk_indices = np.repeat(compressed_indices, counts)
    tile_first = np.repeat(first_tiles, counts)
    tile_local_indices = np.arange(tile_chunk_indices.size, dtype=np.int64) - tile_first
    tile_chunk_byte_offsets = tile_local_indices * _EXR_RESTORE_TILE_BYTES
    tile_offsets = chunk_offsets[tile_chunk_indices] + tile_chunk_byte_offsets
    tile_sizes = np.minimum(
        _EXR_RESTORE_TILE_BYTES,
        chunk_sizes[tile_chunk_indices] - tile_chunk_byte_offsets,
    ).astype(np.int32)
    return tile_offsets, tile_sizes, tile_chunk_byte_offsets, chunk_first_tile, chunk_tile_counts


def _restore_exr_gpu_chunks(
    decoded: cp.ndarray,
    chunk_offsets: np.ndarray,
    chunk_sizes: np.ndarray,
    compressed: np.ndarray,
    expected_adler: np.ndarray | None,
) -> np.ndarray:
    chunk_offsets = np.asarray(chunk_offsets, dtype=np.int64)
    chunk_sizes = np.asarray(chunk_sizes, dtype=np.int64)
    compressed = np.asarray(compressed, dtype=np.uint8)
    validate_adler = expected_adler is not None
    expected_adler = (
        np.zeros(chunk_sizes.size, dtype=np.uint32)
        if expected_adler is None
        else np.asarray(expected_adler, dtype=np.uint32)
    )
    tile_offsets, tile_sizes, tile_chunk_byte_offsets, chunk_first_tile, chunk_tile_counts = _restore_tile_descriptors(
        chunk_offsets, chunk_sizes, compressed
    )
    tile_count = int(tile_offsets.size)
    if not tile_count:
        return np.empty(0, dtype=np.int64)

    device_tile_offsets = cp.asarray(tile_offsets)
    device_tile_sizes = cp.asarray(tile_sizes)
    device_tile_chunk_byte_offsets = cp.asarray(tile_chunk_byte_offsets)
    device_chunk_sizes = cp.asarray(chunk_sizes)
    device_compressed = cp.asarray(compressed)
    device_expected_adler = cp.asarray(expected_adler)
    device_chunk_first_tile = cp.asarray(chunk_first_tile)
    device_chunk_tile_counts = cp.asarray(chunk_tile_counts)
    tile_sum_256 = cp.empty(tile_count, dtype=cp.uint32)
    tile_sum_adler = cp.empty(tile_count, dtype=cp.uint32)
    tile_weighted_adler = cp.empty(tile_count, dtype=cp.uint32)
    tile_prefix_256 = cp.empty(tile_count, dtype=cp.uint32)
    status = cp.zeros(chunk_sizes.size, dtype=cp.int32)
    tile_grid = min(tile_count, _EXR_MAX_GRID_X)
    _restore_reduce_kernel()(
        (tile_grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            decoded,
            device_tile_offsets,
            device_tile_sizes,
            tile_sum_256,
            tile_sum_adler,
            tile_weighted_adler,
            np.int32(tile_count),
        ),
    )
    _restore_offsets_kernel()(
        (min(int(chunk_sizes.size), _EXR_MAX_GRID_X),),
        (1,),
        (
            device_chunk_sizes,
            device_compressed,
            device_expected_adler,
            device_chunk_first_tile,
            device_chunk_tile_counts,
            device_tile_sizes,
            tile_sum_256,
            tile_sum_adler,
            tile_weighted_adler,
            tile_prefix_256,
            status,
            np.int32(validate_adler),
            np.int32(chunk_sizes.size),
        ),
    )
    _restore_finalize_kernel()(
        (tile_grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            decoded,
            device_tile_offsets,
            device_tile_sizes,
            device_tile_chunk_byte_offsets,
            tile_prefix_256,
            np.int32(tile_count),
        ),
    )
    if not validate_adler:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(status.get())


def _gather_raw_chunks(
    device_staging: cp.ndarray,
    decoded: cp.ndarray,
    stage_offsets: np.ndarray,
    decoded_offsets: np.ndarray,
    decoded_sizes: np.ndarray,
    compressed: np.ndarray,
) -> None:
    raw_indices = np.flatnonzero(np.logical_not(compressed))
    if not raw_indices.size:
        return
    source_offsets = cp.asarray(stage_offsets[raw_indices])
    destination_offsets = cp.asarray(decoded_offsets[raw_indices])
    sizes = cp.asarray(decoded_sizes[raw_indices])
    range_count = int(raw_indices.size)
    _gather_raw_kernel()(
        (min(range_count, _EXR_MAX_GRID_X),),
        (_EXR_THREADS_PER_BLOCK,),
        (device_staging, source_offsets, decoded, destination_offsets, sizes, np.int32(range_count)),
    )


def _unpack_exr_chunks(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    decoded: cp.ndarray,
    decoded_offsets: np.ndarray,
    decoded_sizes: np.ndarray,
    even_odd_grouped: np.ndarray,
    *,
    output_dtype: str,
) -> cp.ndarray:
    width = container.data_window[2] - container.data_window[0] + 1
    height = container.data_window[3] - container.data_window[1] + 1
    channel_offsets_by_name: dict[str, int] = {}
    row_bytes = 0
    for channel in container.parts[0].channels:
        channel_offsets_by_name[channel.name] = row_bytes
        row_bytes += width * channel.bytes_per_sample
    if output_dtype == "uint32":
        if any(channel.pixel_type != 0 for channel in selected):
            raise _gpu_error(
                why="the native UINT unpack lane received a non-UINT selected channel",
                what=f"channels={tuple((channel.name, channel.pixel_type) for channel in selected)!r}",
                how="route only homogeneous EXR UINT selections to the uint32 output lane",
            )
        for chunk, size in zip(container.chunks, decoded_sizes, strict=True):
            expected_size = chunk.row_count * row_bytes
            if int(size) != expected_size:
                raise _gpu_error(
                    why="a native UINT chunk differs from its channel-derived row size",
                    what=f"chunk_y={chunk.y}, decoded={int(size)}, expected={expected_size}",
                    how="materialize every file-order channel sample byte exactly once before UINT unpack",
                )
    device_chunk_offsets = cp.asarray(decoded_offsets)
    device_chunk_sizes = cp.asarray(decoded_sizes)
    device_even_odd_grouped = cp.asarray(even_odd_grouped)
    device_channel_offsets = cp.asarray(
        np.asarray([channel_offsets_by_name[channel.name] for channel in selected], dtype=np.int64)
    )
    device_channel_types = cp.asarray(np.asarray([channel.pixel_type for channel in selected], dtype=np.int32))
    cupy_dtype = {"float16": cp.float16, "float32": cp.float32, "uint32": cp.uint32}[output_dtype]
    output = cp.empty((height, width, len(selected)), dtype=cupy_dtype)
    element_count = int(output.size)
    block_count = (element_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _unpack_kernel(output_dtype)(
        (block_count,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            decoded,
            device_chunk_offsets,
            device_chunk_sizes,
            device_even_odd_grouped,
            device_channel_offsets,
            device_channel_types,
            output,
            np.int64(element_count),
            np.int32(width),
            np.int32(height),
            np.int32(len(selected)),
            np.int32(container.lines_per_chunk),
            np.int64(row_bytes),
        ),
    )
    return cast(cp.ndarray, output)


def _unpack_exr_output(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    decoded: cp.ndarray,
    decoded_offsets: np.ndarray,
    decoded_sizes: np.ndarray,
    even_odd_grouped: np.ndarray,
    *,
    output_dtype: str,
) -> cp.ndarray:
    return _unpack_exr_chunks(
        container,
        selected,
        decoded,
        decoded_offsets,
        decoded_sizes,
        even_odd_grouped,
        output_dtype=output_dtype,
    )


def _select_exr_host_pixels(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    restored: np.ndarray,
    *,
    output_dtype: str,
) -> np.ndarray:
    width = container.data_window[2] - container.data_window[0] + 1
    height = container.data_window[3] - container.data_window[1] + 1
    channel_offsets_by_name: dict[str, int] = {}
    row_bytes = 0
    for channel in container.parts[0].channels:
        channel_offsets_by_name[channel.name] = row_bytes
        row_bytes += width * channel.bytes_per_sample
    rows = restored.reshape(height, row_bytes)
    host_selected = np.empty((height, width, len(selected)), dtype=output_dtype)
    for output_channel, channel in enumerate(selected):
        channel_start = channel_offsets_by_name[channel.name]
        channel_end = channel_start + width * channel.bytes_per_sample
        channel_bytes = np.ascontiguousarray(rows[:, channel_start:channel_end])
        source_dtype = np.dtype({0: "<u4", 1: "<f2", 2: "<f4"}[channel.pixel_type])
        values = channel_bytes.view(source_dtype).reshape(height, width)
        host_selected[..., output_channel] = values.astype(output_dtype, copy=False)
    return host_selected


def _pack_exr_gpu(
    data: cp.ndarray,
    channels: Sequence[str],
    *,
    row_prefix_bytes: int = 0,
) -> tuple[cp.ndarray, tuple[str, ...], int]:
    input_dtype = data.dtype.name
    if input_dtype not in ("uint8", "uint16", "uint32", "float16", "float32"):
        raise _gpu_error(
            why="the GPU EXR writer received an unsupported Frame storage dtype",
            what=f"dtype={input_dtype!r}",
            how="provide uint8, uint16, uint32, float16, or float32 Frame storage",
        )
    ordered = tuple(sorted(enumerate(channels), key=lambda item: item[1]))
    ordered_channels = tuple(label for _, label in ordered)
    channel_indices = cp.asarray(np.asarray([index for index, _ in ordered], dtype=np.int32))
    pixel_type = 0 if input_dtype == "uint32" else 1 if input_dtype == "float16" else 2
    bytes_per_sample = 2 if pixel_type == 1 else 4
    if row_prefix_bytes not in (0, 8):
        raise _gpu_error(
            why="the GPU EXR packer received an unsupported row prefix size",
            what=f"row_prefix_bytes={row_prefix_bytes}",
            how="use zero for codec payloads or eight bytes for NONE scanline records",
        )
    sample_count = int(data.size)
    output = cp.empty(sample_count * bytes_per_sample + int(data.shape[0]) * row_prefix_bytes, dtype=cp.uint8)
    block_count = (sample_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _pack_kernel(input_dtype)(
        (block_count,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            data,
            channel_indices,
            output,
            np.int64(sample_count),
            np.int32(data.shape[1]),
            np.int32(data.shape[2]),
            np.int32(row_prefix_bytes),
        ),
    )
    return output, ordered_channels, pixel_type


def _transform_and_checksum_chunks(
    raw: cp.ndarray,
    chunk_offsets: Sequence[int],
    chunk_sizes: Sequence[int],
) -> tuple[cp.ndarray, cp.ndarray]:
    device_offsets = _device_i64(chunk_offsets)
    device_sizes = _device_i64(chunk_sizes)
    transformed = cp.empty_like(raw)
    adler = cp.empty(len(chunk_sizes), dtype=cp.uint32)
    block_count = _maximum_block_count(chunk_sizes)
    for start, end in _chunk_launch_ranges(len(chunk_sizes)):
        batch_size = end - start
        batch_offsets = device_offsets[start:end]
        batch_sizes = device_sizes[start:end]
        _transform_kernel()(
            (block_count, batch_size),
            (_EXR_THREADS_PER_BLOCK,),
            (raw, transformed, batch_offsets, batch_sizes, np.int32(batch_size)),
        )
        _adler_kernel()(
            (batch_size,),
            (_EXR_THREADS_PER_BLOCK,),
            (transformed, batch_offsets, batch_sizes, adler[start:end], np.int32(batch_size)),
        )
    return transformed, adler


def _checksum_exr_chunks(
    payload: cp.ndarray,
    chunk_offsets: Sequence[int],
    chunk_sizes: Sequence[int],
) -> cp.ndarray:
    if len(chunk_offsets) != len(chunk_sizes) or not chunk_sizes:
        raise _gpu_error(
            why="the EXR checksum batch received mismatched or empty chunk descriptors",
            what=f"offsets={len(chunk_offsets)}, sizes={len(chunk_sizes)}",
            how="provide one positive-size payload range for every checksum output",
        )
    device_offsets = _device_i64(chunk_offsets)
    device_sizes = _device_i64(chunk_sizes)
    adler = cp.empty(len(chunk_sizes), dtype=cp.uint32)
    for start, end in _chunk_launch_ranges(len(chunk_sizes)):
        batch_size = end - start
        _adler_kernel()(
            (batch_size,),
            (_EXR_THREADS_PER_BLOCK,),
            (payload, device_offsets[start:end], device_sizes[start:end], adler[start:end], np.int32(batch_size)),
        )
    return cast(cp.ndarray, adler)


def _transform_exr_chunks(
    raw: cp.ndarray,
    chunk_offsets: Sequence[int],
    chunk_sizes: Sequence[int],
) -> cp.ndarray:
    device_offsets = _device_i64(chunk_offsets)
    device_sizes = _device_i64(chunk_sizes)
    transformed = cp.empty_like(raw)
    block_count = _maximum_block_count(chunk_sizes)
    for start, end in _chunk_launch_ranges(len(chunk_sizes)):
        batch_size = end - start
        _transform_kernel()(
            (block_count, batch_size),
            (_EXR_THREADS_PER_BLOCK,),
            (raw, transformed, device_offsets[start:end], device_sizes[start:end], np.int32(batch_size)),
        )
    return transformed


def _encode_deflate_chunks(
    transformed: cp.ndarray,
    input_ranges: Sequence[tuple[int, int]],
) -> tuple[cp.ndarray, tuple[int, ...], tuple[int, ...]]:
    from nvidia import nvcomp

    stream = cp.cuda.get_current_stream()
    inputs = nvcomp.as_arrays(
        [transformed[offset : offset + size] for offset, size in input_ranges], cuda_stream=int(stream.ptr)
    )
    codec = _nvcomp_deflate_codec(cp.cuda.Device().id, int(stream.ptr))
    encoded = codec.encode(inputs)
    stream.synchronize()
    encoded_arrays = [cp.asarray(value).view(cp.uint8) for value in encoded]
    sizes = tuple(int(value.size) for value in encoded_arrays)
    offsets = _prefix_offsets(sizes)
    return cp.concatenate(encoded_arrays), offsets, sizes


def _wrap_deflate_chunks(
    compressed: cp.ndarray,
    compressed_offsets: Sequence[int],
    compressed_sizes: Sequence[int],
    adler: cp.ndarray,
) -> tuple[cp.ndarray, tuple[int, ...], tuple[int, ...]]:
    wrapped_sizes = tuple(size + 6 for size in compressed_sizes)
    wrapped_offsets = _prefix_offsets(wrapped_sizes)
    wrapped = cp.empty(sum(wrapped_sizes), dtype=cp.uint8)
    device_compressed_offsets = _device_i64(compressed_offsets)
    device_compressed_sizes = _device_i64(compressed_sizes)
    device_wrapped_offsets = _device_i64(wrapped_offsets)
    block_count = _maximum_block_count(wrapped_sizes)
    for start, end in _chunk_launch_ranges(len(wrapped_sizes)):
        batch_size = end - start
        _wrap_kernel()(
            (block_count, batch_size),
            (_EXR_THREADS_PER_BLOCK,),
            (
                compressed,
                device_compressed_offsets[start:end],
                device_compressed_sizes[start:end],
                adler[start:end],
                wrapped,
                device_wrapped_offsets[start:end],
                np.int32(batch_size),
            ),
        )
    return wrapped, wrapped_offsets, wrapped_sizes


def _select_exr_payloads(
    raw: cp.ndarray,
    raw_offsets: Sequence[int],
    raw_sizes: Sequence[int],
    wrapped: cp.ndarray,
    wrapped_offsets: Sequence[int],
    wrapped_sizes: Sequence[int],
) -> tuple[cp.ndarray, tuple[int, ...]]:
    use_wrapped = tuple(
        wrapped_size < raw_size for wrapped_size, raw_size in zip(wrapped_sizes, raw_sizes, strict=True)
    )
    output_sizes = tuple(
        wrapped_size if selected else raw_size
        for selected, wrapped_size, raw_size in zip(use_wrapped, wrapped_sizes, raw_sizes, strict=True)
    )
    output_offsets = _prefix_offsets(output_sizes)
    output = cp.empty(sum(output_sizes), dtype=cp.uint8)
    device_raw_offsets = _device_i64(raw_offsets)
    device_raw_sizes = _device_i64(raw_sizes)
    device_wrapped_offsets = _device_i64(wrapped_offsets)
    device_wrapped_sizes = _device_i64(wrapped_sizes)
    device_use_wrapped = cp.asarray(np.asarray(use_wrapped, dtype=np.uint8))
    device_output_offsets = _device_i64(output_offsets)
    block_count = _maximum_block_count(output_sizes)
    for start, end in _chunk_launch_ranges(len(output_sizes)):
        batch_size = end - start
        _select_payload_kernel()(
            (block_count, batch_size),
            (_EXR_THREADS_PER_BLOCK,),
            (
                raw,
                device_raw_offsets[start:end],
                device_raw_sizes[start:end],
                wrapped,
                device_wrapped_offsets[start:end],
                device_wrapped_sizes[start:end],
                device_use_wrapped[start:end],
                output,
                device_output_offsets[start:end],
                np.int32(batch_size),
            ),
        )
    return output, output_sizes


def _attribute_bytes(name: str, attribute_type: str, payload: bytes) -> bytes:
    return (
        name.encode("ascii")
        + b"\x00"
        + attribute_type.encode("ascii")
        + b"\x00"
        + struct.pack("<I", len(payload))
        + payload
    )


def _encode_exr_output_channels(channels: Sequence[str]) -> tuple[tuple[str, bytes], ...]:
    encoded_channels: list[tuple[str, bytes]] = []
    for channel in channels:
        try:
            encoded = channel.encode("utf-8")
        except UnicodeEncodeError as error:
            raise _parser_error(
                why="an EXR output channel label is not valid UTF-8",
                what=f"channel={channel!r}",
                how="use channel labels that can be represented as UTF-8",
            ) from error
        if not encoded or b"\x00" in encoded or len(encoded) > 255:
            raise _parser_error(
                why="an EXR output channel label is empty, contains a null byte, or exceeds 255 UTF-8 bytes",
                what=f"channel={channel!r}, encoded_size={len(encoded)}",
                how="use a non-empty channel label of at most 255 UTF-8 bytes without null characters",
            )
        encoded_channels.append((channel, encoded))
    return tuple(encoded_channels)


def _exr_write_header(
    *,
    width: int,
    height: int,
    encoded_channels: Sequence[tuple[str, bytes]],
    pixel_type: int,
    compression: str,
    chromaticities: Sequence[float],
    aces_image_container: bool,
    dwa_level: float | None = None,
) -> bytes:
    channel_payload = bytearray()
    for _, encoded in encoded_channels:
        channel_payload.extend(encoded + b"\x00")
        channel_payload.extend(struct.pack("<iB3xii", pixel_type, 0, 1, 1))
    channel_payload.append(0)
    data_window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    attributes = [
        _attribute_bytes("channels", "chlist", bytes(channel_payload)),
        _attribute_bytes("compression", "compression", bytes((_EXR_COMPRESSION_CODES[compression],))),
        _attribute_bytes("dataWindow", "box2i", data_window),
        _attribute_bytes("displayWindow", "box2i", data_window),
        _attribute_bytes("lineOrder", "lineOrder", b"\x00"),
        _attribute_bytes("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute_bytes("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute_bytes("screenWindowWidth", "float", struct.pack("<f", 1.0)),
        _attribute_bytes("chromaticities", "chromaticities", struct.pack("<8f", *chromaticities)),
    ]
    if dwa_level is not None:
        attributes.append(_attribute_bytes("dwaCompressionLevel", "float", struct.pack("<f", dwa_level)))
    if aces_image_container:
        attributes.append(_attribute_bytes("acesImageContainerFlag", "int", struct.pack("<i", 1)))
    version_field = _EXR_VERSION
    if any(len(encoded) > 31 for _, encoded in encoded_channels):
        version_field |= _EXR_LONG_NAMES_FLAG
    return struct.pack("<II", _EXR_MAGIC, version_field) + b"".join(attributes) + b"\x00"
