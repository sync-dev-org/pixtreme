"""RLE OpenEXR read/write lane and CUDA kernels."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import cupy as cp
import numpy as np

from pixtreme._io.formats.exr.container import (
    _EXR_MAX_GRID_X,
    _EXR_MAX_GRID_Y,
    _EXR_THREADS_PER_BLOCK,
    _ExrChannel,
    _ExrContainer,
    _ExrRleReadChunks,
    _gpu_error,
)
from pixtreme._io.formats.exr.packing import (
    _device_i64,
    _gather_raw_chunks,
    _numpy_offsets,
    _prefix_offsets,
    _restore_exr_gpu_chunks,
    _unpack_exr_output,
)


def _prepare_exr_rle_read_chunks(container: _ExrContainer) -> _ExrRleReadChunks:
    chunks = container.chunks
    stage_sizes = np.fromiter((chunk.packed_size for chunk in chunks), dtype=np.int64, count=len(chunks))
    stage_offsets = np.fromiter((chunk.payload_start for chunk in chunks), dtype=np.int64, count=len(chunks))
    decoded_sizes = np.fromiter((chunk.expected_size for chunk in chunks), dtype=np.int64, count=len(chunks))
    decoded_offsets = _numpy_offsets(decoded_sizes)
    compressed = np.fromiter((not chunk.raw_stored for chunk in chunks), dtype=np.uint8, count=len(chunks))
    host_staging = np.frombuffer(container.data, dtype=np.uint8)
    packet_counts = np.zeros(len(chunks), dtype=np.int64)
    for chunk_index, chunk in enumerate(chunks):
        descriptor = chunk.phase3
        if descriptor is None or descriptor.codec != "rle":
            raise _gpu_error(
                why="the RLE read batch received a chunk without its validated RLE descriptor",
                what=f"chunk_y={chunk.y}, descriptor={descriptor!r}",
                how="parse every eligible RLE chunk before materializing its packet stream",
            )
        packet_counts[chunk_index] = len(descriptor.packets)
    return _ExrRleReadChunks(
        host_staging=host_staging,
        stage_offsets=stage_offsets,
        stage_sizes=stage_sizes,
        decoded_offsets=decoded_offsets,
        decoded_sizes=decoded_sizes,
        compressed=compressed,
        packet_offsets=_numpy_offsets(packet_counts),
        packet_counts=packet_counts,
    )


def _copy_rle_raw_ranges_host(prepared: _ExrRleReadChunks, decoded: np.ndarray) -> None:
    raw_indices = np.flatnonzero(np.logical_not(prepared.compressed))
    for chunk_index in raw_indices:
        source = int(prepared.stage_offsets[chunk_index])
        destination = int(prepared.decoded_offsets[chunk_index])
        size = int(prepared.decoded_sizes[chunk_index])
        decoded[destination : destination + size] = prepared.host_staging[source : source + size]


def _materialize_rle_host(prepared: _ExrRleReadChunks) -> np.ndarray:
    decoded_size = int(prepared.decoded_sizes.sum())
    decoded_memory = cp.cuda.alloc_pinned_memory(decoded_size)
    decoded = np.frombuffer(decoded_memory, dtype=np.uint8, count=decoded_size)
    _copy_rle_raw_ranges_host(prepared, decoded)
    for chunk_index in np.flatnonzero(prepared.compressed):
        input_offset = int(prepared.stage_offsets[chunk_index])
        input_end = input_offset + int(prepared.stage_sizes[chunk_index])
        output_offset = int(prepared.decoded_offsets[chunk_index])
        while input_offset < input_end:
            header_byte = int(prepared.host_staging[input_offset])
            header = header_byte if header_byte < 128 else header_byte - 256
            input_offset += 1
            literal = header < 0
            size = -header if literal else header + 1
            output_end = output_offset + size
            if literal:
                decoded[output_offset:output_end] = prepared.host_staging[input_offset : input_offset + size]
                input_offset += size
            else:
                decoded[output_offset:output_end] = prepared.host_staging[input_offset]
                input_offset += 1
            output_offset = output_end
    return decoded


_EXR_RLE_SOURCE = r"""
extern "C" __global__ void pixtreme_exr_rle_scan_packets(
    const unsigned char* staging,
    const long long* stage_offsets,
    const long long* stage_sizes,
    const unsigned char* compressed,
    const long long* packet_offsets,
    long long* packet_descriptors,
    const int chunk_count
) {
    for (int chunk = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         chunk < chunk_count;
         chunk += (int)(gridDim.x * blockDim.x)) {
        if (!compressed[chunk]) {
            continue;
        }
        long long input_offset = stage_offsets[chunk];
        const long long input_end = input_offset + stage_sizes[chunk];
        long long output_offset = 0;
        long long packet = packet_offsets[chunk];
        while (input_offset < input_end) {
            const long long header_offset = input_offset;
            const signed char header = (signed char)staging[input_offset++];
            const bool literal = header < 0;
            const long long size = literal ? -(long long)header : (long long)header + 1LL;
            packet_descriptors[packet * 3LL] = header_offset;
            packet_descriptors[packet * 3LL + 1LL] = (long long)chunk;
            packet_descriptors[packet * 3LL + 2LL] = output_offset;
            input_offset += literal ? size : 1LL;
            output_offset += size;
            ++packet;
        }
    }
}

extern "C" __global__ void pixtreme_exr_rle_decode_packets(
    const unsigned char* staging,
    const long long* packet_descriptors,
    const long long* chunk_output_offsets,
    unsigned char* output,
    const long long packet_count
) {
    const long long warps_per_block = (long long)blockDim.x / 32LL;
    const long long warp = (long long)blockIdx.x * warps_per_block + (long long)threadIdx.x / 32LL;
    const long long lane = (long long)threadIdx.x & 31LL;
    for (long long packet = warp; packet < packet_count; packet += (long long)gridDim.x * warps_per_block) {
        const long long header_offset = packet_descriptors[packet * 3LL];
        const long long chunk = packet_descriptors[packet * 3LL + 1LL];
        const long long packet_output_offset = packet_descriptors[packet * 3LL + 2LL];
        const signed char header = (signed char)staging[header_offset];
        const bool literal = header < 0;
        const long long size = literal ? -(long long)header : (long long)header + 1LL;
        const long long source = header_offset + 1LL;
        const long long destination = chunk_output_offsets[chunk] + packet_output_offset;
        for (long long index = lane; index < size; index += 32LL) {
            output[destination + index] = staging[source + (literal ? index : 0LL)];
        }
    }
}

extern "C" __global__ void pixtreme_exr_rle_encode_chunks(
    const unsigned char* transformed,
    const long long* chunk_offsets,
    const long long* chunk_sizes,
    const long long* capacity_offsets,
    unsigned char* output,
    long long* output_sizes,
    const int chunk_count
) {
    __shared__ long long shared_input;
    __shared__ long long shared_output;
    __shared__ long long shared_copy_input;
    __shared__ long long shared_copy_output;
    __shared__ long long shared_copy_size;
    __shared__ int shared_done;
    const int lane = (int)threadIdx.x;
    for (int chunk = (int)blockIdx.x; chunk < chunk_count; chunk += (int)gridDim.x) {
        const long long chunk_offset = chunk_offsets[chunk];
        const long long chunk_size = chunk_sizes[chunk];
        const long long output_start = capacity_offsets[chunk];
        if (lane == 0) {
            shared_input = 0;
            shared_output = output_start;
            shared_done = 0;
        }
        __syncwarp();
        while (true) {
            if (lane == 0) {
                shared_copy_size = 0;
                if (shared_input >= chunk_size) {
                    shared_done = 1;
                } else {
                    long long input = shared_input;
                    long long output_offset = shared_output;
                    long long run_length = 1;
                    const unsigned char value = transformed[chunk_offset + input];
                    while (
                        input + run_length < chunk_size
                        && transformed[chunk_offset + input + run_length] == value
                    ) {
                        ++run_length;
                    }
                    if (run_length >= 3) {
                        long long remaining = run_length;
                        while (remaining >= 3) {
                            const long long packet_size = remaining > 128 ? 128 : remaining;
                            output[output_offset++] = (unsigned char)(packet_size - 1LL);
                            output[output_offset++] = value;
                            input += packet_size;
                            remaining -= packet_size;
                        }
                    } else {
                        const long long literal_start = input;
                        long long literal_size = 0;
                        while (input < chunk_size && literal_size < 127) {
                            run_length = 1;
                            const unsigned char literal_value = transformed[chunk_offset + input];
                            while (
                                input + run_length < chunk_size
                                && transformed[chunk_offset + input + run_length] == literal_value
                            ) {
                                ++run_length;
                            }
                            if (run_length >= 3) {
                                break;
                            }
                            const long long available = 127LL - literal_size;
                            const long long consumed = run_length < available ? run_length : available;
                            input += consumed;
                            literal_size += consumed;
                        }
                        output[output_offset++] = (unsigned char)(256LL - literal_size);
                        shared_copy_input = chunk_offset + literal_start;
                        shared_copy_output = output_offset;
                        shared_copy_size = literal_size;
                        output_offset += literal_size;
                    }
                    shared_input = input;
                    shared_output = output_offset;
                }
            }
            __syncwarp();
            if (shared_done) {
                break;
            }
            for (long long index = lane; index < shared_copy_size; index += 32LL) {
                output[shared_copy_output + index] = transformed[shared_copy_input + index];
            }
            __syncwarp();
        }
        if (lane == 0) {
            output_sizes[chunk] = shared_output - output_start;
        }
        __syncwarp();
    }
}

extern "C" __global__ void pixtreme_exr_rle_compact_chunks(
    const unsigned char* source,
    const long long* source_offsets,
    const long long* chunk_sizes,
    unsigned char* destination,
    const long long* destination_offsets,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count || index >= chunk_sizes[chunk]) {
        return;
    }
    destination[destination_offsets[chunk] + index] = source[source_offsets[chunk] + index];
}
"""


@lru_cache(maxsize=1)
def _rle_scan_packets_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_RLE_SOURCE, "pixtreme_exr_rle_scan_packets")


@lru_cache(maxsize=1)
def _rle_decode_packets_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_RLE_SOURCE, "pixtreme_exr_rle_decode_packets")


@lru_cache(maxsize=1)
def _rle_encode_chunks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_RLE_SOURCE, "pixtreme_exr_rle_encode_chunks")


@lru_cache(maxsize=1)
def _rle_compact_chunks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_RLE_SOURCE, "pixtreme_exr_rle_compact_chunks")


def _materialize_rle_gpu(prepared: _ExrRleReadChunks, device_staging: cp.ndarray) -> cp.ndarray:
    decoded = cp.empty(int(prepared.decoded_sizes.sum()), dtype=cp.uint8)
    _gather_raw_chunks(
        device_staging,
        decoded,
        prepared.stage_offsets,
        prepared.decoded_offsets,
        prepared.decoded_sizes,
        prepared.compressed,
    )
    packet_count = int(prepared.packet_counts.sum())
    if not packet_count:
        return decoded
    device_stage_offsets = cp.asarray(prepared.stage_offsets)
    device_stage_sizes = cp.asarray(prepared.stage_sizes)
    device_compressed = cp.asarray(prepared.compressed)
    device_packet_offsets = cp.asarray(prepared.packet_offsets)
    packet_descriptors = cp.empty(packet_count * 3, dtype=cp.int64)
    chunk_count = int(prepared.stage_sizes.size)
    scan_blocks = min(
        (chunk_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK,
        _EXR_MAX_GRID_X,
    )
    _rle_scan_packets_kernel()(
        (scan_blocks,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            device_staging,
            device_stage_offsets,
            device_stage_sizes,
            device_compressed,
            device_packet_offsets,
            packet_descriptors,
            np.int32(chunk_count),
        ),
    )
    warps_per_block = _EXR_THREADS_PER_BLOCK // 32
    decode_blocks = min((packet_count + warps_per_block - 1) // warps_per_block, _EXR_MAX_GRID_X)
    _rle_decode_packets_kernel()(
        (decode_blocks,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            device_staging,
            packet_descriptors,
            cp.asarray(prepared.decoded_offsets),
            decoded,
            np.int64(packet_count),
        ),
    )
    return decoded


def _encode_rle_packets_gpu(
    transformed: cp.ndarray,
    chunk_offsets: Sequence[int],
    chunk_sizes: Sequence[int],
) -> tuple[cp.ndarray, tuple[int, ...], tuple[int, ...]]:
    offsets = tuple(int(value) for value in chunk_offsets)
    sizes = tuple(int(value) for value in chunk_sizes)
    if len(offsets) != len(sizes) or not sizes:
        raise _gpu_error(
            why="the RLE packet encoder received mismatched or empty chunk descriptors",
            what=f"offsets={len(offsets)}, sizes={len(sizes)}",
            how="provide one nonempty transformed byte range for every RLE scanline chunk",
        )
    if offsets != _prefix_offsets(sizes) or any(size <= 0 for size in sizes):
        raise _gpu_error(
            why="the RLE packet encoder received non-contiguous or empty transformed chunks",
            what=f"offsets={offsets!r}, sizes={sizes!r}",
            how="concatenate positive-size transformed chunks in output-row order before packet scanning",
        )
    values = cp.ascontiguousarray(transformed, dtype=cp.uint8).reshape(-1)
    total_size = sum(sizes)
    if int(values.size) != total_size:
        raise _gpu_error(
            why="the RLE packet encoder byte count differs from its chunk descriptors",
            what=f"transformed={int(values.size)}, described={total_size}",
            how="make the chunk ranges consume the transformed image bytes exactly once",
        )

    capacity_sizes = tuple(size + (size + 126) // 127 for size in sizes)
    capacity_offsets = _prefix_offsets(capacity_sizes)
    capacity_output = cp.empty(sum(capacity_sizes), dtype=cp.uint8)
    device_payload_sizes = cp.empty(len(sizes), dtype=cp.int64)
    chunk_count = len(sizes)
    encode_blocks = min(chunk_count, _EXR_MAX_GRID_X)
    _rle_encode_chunks_kernel()(
        (encode_blocks,),
        (32,),
        (
            values,
            _device_i64(offsets),
            _device_i64(sizes),
            _device_i64(capacity_offsets),
            capacity_output,
            device_payload_sizes,
            np.int32(chunk_count),
        ),
    )
    payload_sizes = tuple(int(value) for value in cp.asnumpy(device_payload_sizes))
    payload_offsets = _prefix_offsets(payload_sizes)
    output = cp.empty(sum(payload_sizes), dtype=cp.uint8)
    maximum_blocks = max((size + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK for size in payload_sizes)
    device_capacity_offsets = _device_i64(capacity_offsets)
    device_payload_offsets = _device_i64(payload_offsets)
    device_payload_sizes = _device_i64(payload_sizes)
    for start in range(0, chunk_count, _EXR_MAX_GRID_Y):
        end = min(start + _EXR_MAX_GRID_Y, chunk_count)
        batch_size = end - start
        _rle_compact_chunks_kernel()(
            (maximum_blocks, batch_size),
            (_EXR_THREADS_PER_BLOCK,),
            (
                capacity_output,
                device_capacity_offsets[start:end],
                device_payload_sizes[start:end],
                output,
                device_payload_offsets[start:end],
                np.int32(batch_size),
            ),
        )
    return output, payload_offsets, payload_sizes


def _read_exr_rle_gpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    prepared = _prepare_exr_rle_read_chunks(container)
    device_staging = cp.asarray(prepared.host_staging)
    decoded = _materialize_rle_gpu(prepared, device_staging)
    _restore_exr_gpu_chunks(
        decoded,
        prepared.decoded_offsets,
        prepared.decoded_sizes,
        prepared.compressed,
        None,
    )
    return _unpack_exr_output(
        container,
        selected,
        decoded,
        prepared.decoded_offsets,
        prepared.decoded_sizes,
        even_odd_grouped=prepared.compressed,
        output_dtype=output_dtype,
    )


def _read_exr_rle_custom_cpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    prepared = _prepare_exr_rle_read_chunks(container)
    materialized = _materialize_rle_host(prepared)
    decoded = cp.asarray(materialized)
    cp.cuda.get_current_stream().synchronize()
    _restore_exr_gpu_chunks(
        decoded,
        prepared.decoded_offsets,
        prepared.decoded_sizes,
        prepared.compressed,
        None,
    )
    return _unpack_exr_output(
        container,
        selected,
        decoded,
        prepared.decoded_offsets,
        prepared.decoded_sizes,
        even_odd_grouped=prepared.compressed,
        output_dtype=output_dtype,
    )
