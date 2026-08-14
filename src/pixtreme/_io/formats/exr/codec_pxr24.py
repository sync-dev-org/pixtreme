"""PXR24 OpenEXR read/write lane and CUDA kernels."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import cast

import cupy as cp
import numpy as np

from pixtreme._io.formats.exr.codec_zip import (
    _decode_deflate_chunks,
)
from pixtreme._io.formats.exr.container import (
    _EXR_PXR24_PLANE_COUNTS,
    _EXR_THREADS_PER_BLOCK,
    _ExrChannel,
    _ExrContainer,
    _ExrPxr24ReadChunks,
    _gpu_error,
    _Phase3ChunkDescriptor,
)
from pixtreme._io.formats.exr.packing import (
    _gather_raw_chunks,
    _numpy_offsets,
    _read_worker_count,
    _unpack_exr_output,
)


def _prepare_exr_pxr24_read_chunks(
    container: _ExrContainer,
    *,
    materialize_host: bool,
) -> _ExrPxr24ReadChunks:
    chunks = container.chunks
    raw_sizes = np.fromiter((chunk.expected_size for chunk in chunks), dtype=np.int64, count=len(chunks))
    raw_offsets = _numpy_offsets(raw_sizes)
    materialized_sizes = np.fromiter(
        (cast(_Phase3ChunkDescriptor, chunk.phase3).expected_materialized_size for chunk in chunks),
        dtype=np.int64,
        count=len(chunks),
    )
    materialized_offsets = _numpy_offsets(materialized_sizes)
    compressed = np.fromiter((not chunk.raw_stored for chunk in chunks), dtype=np.uint8, count=len(chunks))
    payloads: list[bytes] = []
    for chunk_index, chunk in enumerate(chunks):
        descriptor = chunk.phase3
        if descriptor is None or descriptor.codec != "pxr24":
            raise _gpu_error(
                why="the PXR24 read batch received a chunk without its validated PXR24 descriptor",
                what=f"chunk_y={chunk.y}, descriptor={descriptor!r}",
                how="parse every eligible PXR24 chunk before materializing its row-channel planes",
            )
        payload = container.data[chunk.payload_start : chunk.payload_end]
        if compressed[chunk_index]:
            stage_payload = payload[2:-4]
        else:
            stage_payload = payload
        payloads.append(stage_payload)
    if materialize_host:
        materialized_byte_count = int(materialized_sizes.sum())
        stage_sizes = np.where(compressed, np.int64(0), raw_sizes)
        stage_offsets = _numpy_offsets(stage_sizes) + materialized_byte_count
        staging_byte_count = materialized_byte_count + int(stage_sizes.sum())
        staging_memory = cp.cuda.alloc_pinned_memory(staging_byte_count)
        host_staging = np.frombuffer(staging_memory, dtype=np.uint8, count=staging_byte_count)
        host_materialized = host_staging[:materialized_byte_count]
        compressed_indices = np.flatnonzero(compressed)
        compressed_payloads = tuple(
            container.data[chunks[int(index)].payload_start : chunks[int(index)].payload_end]
            for index in compressed_indices
        )
        with ThreadPoolExecutor(max_workers=min(_read_worker_count(len(compressed_payloads)), 8)) as executor:
            materialized_payloads = tuple(executor.map(zlib.decompress, compressed_payloads))
        for compressed_index, materialized in zip(compressed_indices, materialized_payloads, strict=True):
            chunk_index = int(compressed_index)
            materialized_offset = int(materialized_offsets[chunk_index])
            materialized_size = int(materialized_sizes[chunk_index])
            if len(materialized) != materialized_size:
                raise _gpu_error(
                    why="the prepared PXR24 plane stream differs from its descriptor materialized size",
                    what=(
                        f"chunk_y={chunks[chunk_index].y}, prepared={len(materialized)}, expected={materialized_size}"
                    ),
                    how="emit every row-channel byte plane exactly once in the zlib stream",
                )
            host_materialized[materialized_offset : materialized_offset + materialized_size] = np.frombuffer(
                materialized,
                dtype=np.uint8,
            )
        for raw_index in np.flatnonzero(np.logical_not(compressed)):
            chunk_index = int(raw_index)
            stage_offset = int(stage_offsets[chunk_index])
            stage_size = int(stage_sizes[chunk_index])
            host_staging[stage_offset : stage_offset + stage_size] = np.frombuffer(
                payloads[chunk_index],
                dtype=np.uint8,
            )
    else:
        stage_sizes = np.fromiter((len(payload) for payload in payloads), dtype=np.int64, count=len(payloads))
        stage_offsets = _numpy_offsets(stage_sizes)
        host_staging = np.frombuffer(b"".join(payloads), dtype=np.uint8)
        host_materialized = np.empty(0, dtype=np.uint8)
    return _ExrPxr24ReadChunks(
        host_staging=host_staging,
        host_materialized=host_materialized,
        stage_offsets=stage_offsets,
        stage_sizes=stage_sizes,
        materialized_offsets=materialized_offsets,
        materialized_sizes=materialized_sizes,
        raw_offsets=raw_offsets,
        raw_sizes=raw_sizes,
        compressed=compressed,
    )


def _encode_pxr24_rows_gpu(bits: cp.ndarray, pixel_type: int) -> cp.ndarray:
    plane_count = _EXR_PXR24_PLANE_COUNTS.get(pixel_type)
    values = cp.ascontiguousarray(bits, dtype=cp.uint32)
    if plane_count is None or values.ndim != 2:
        raise _gpu_error(
            why="the PXR24 row encoder received an unsupported pixel type or row shape",
            what=f"pixel_type={pixel_type}, shape={values.shape!r}",
            how="provide a two-dimensional UINT, HALF, or FLOAT bit-pattern array",
        )
    row_count, width = (int(value) for value in values.shape)
    planes = cp.empty((row_count, plane_count, width), dtype=cp.uint8)
    sample_count = int(values.size)
    block_count = (sample_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _pxr24_encode_rows_kernel()(
        (block_count,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            values,
            planes,
            np.int64(sample_count),
            np.int32(width),
            np.int32(pixel_type),
            np.int32(plane_count),
        ),
    )
    return cast(cp.ndarray, planes)


def _decode_pxr24_rows_host(planes: np.ndarray, pixel_type: int) -> np.ndarray:
    plane_count = _EXR_PXR24_PLANE_COUNTS.get(pixel_type)
    source = np.ascontiguousarray(planes, dtype=np.uint8)
    if plane_count is None or source.ndim != 3 or source.shape[1] != plane_count:
        raise _gpu_error(
            why="the host PXR24 row decoder received an unsupported pixel type or plane shape",
            what=f"pixel_type={pixel_type}, shape={source.shape!r}",
            how="provide row-major UINT=4, HALF=2, or FLOAT24=3 byte planes",
        )
    difference = np.zeros((source.shape[0], source.shape[2]), dtype=np.uint64)
    for plane in range(plane_count):
        difference |= source[:, plane].astype(np.uint64) << np.uint64(8 * (plane_count - plane - 1))
    mask = np.uint64((1 << (plane_count * 8)) - 1)
    values = (np.cumsum(difference, axis=1, dtype=np.uint64) & mask).astype(np.uint32)
    return values << np.uint32(8) if pixel_type == 2 else values


_EXR_PXR24_SOURCE = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ unsigned int pixtreme_exr_float24(const unsigned int bits) {
    const unsigned int sign = bits & 0x80000000U;
    const unsigned int exponent = bits & 0x7f800000U;
    const unsigned int mantissa = bits & 0x007fffffU;
    const unsigned int magnitude = exponent | mantissa;
    const unsigned int truncated = magnitude >> 8;
    const unsigned int rounded = (magnitude + (mantissa & 0x80U)) >> 8;
    unsigned int encoded = rounded;
    if (exponent != 0x7f800000U && (rounded & 0x7fffffU) == 0x7f8000U) {
        encoded = truncated;
    } else if (exponent == 0x7f800000U && mantissa != 0U) {
        encoded = truncated;
        if ((encoded & 0x7fffU) == 0U) encoded |= 1U;
    }
    return (sign >> 8) | encoded;
}

extern "C" __global__ void pixtreme_exr_pxr24_encode_rows(
    const unsigned int* bits,
    unsigned char* planes,
    const long long sample_count,
    const int width,
    const int pixel_type,
    const int plane_count
) {
    const long long sample = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (sample >= sample_count) return;
    const int x = (int)(sample % width);
    const long long row = sample / width;
    const unsigned int mask = plane_count == 4 ? 0xffffffffU : ((1U << (8 * plane_count)) - 1U);
    unsigned int current = bits[sample] & mask;
    unsigned int previous = x == 0 ? 0U : bits[sample - 1] & mask;
    if (pixel_type == 2) {
        current = pixtreme_exr_float24(bits[sample]);
        previous = x == 0 ? 0U : pixtreme_exr_float24(bits[sample - 1]);
    }
    const unsigned int difference = (current - previous) & mask;
    for (int plane = 0; plane < plane_count; ++plane) {
        const int shift = 8 * (plane_count - plane - 1);
        planes[(row * plane_count + plane) * width + x] = (unsigned char)(difference >> shift);
    }
}

extern "C" __global__ void pixtreme_exr_pxr24_scatter_f16(
    const unsigned char* materialized,
    const long long* output_rows,
    const long long* plane_offsets,
    unsigned short* output,
    const int row_count,
    const int width,
    const int output_channels,
    const int output_channel,
    const int pixel_type,
    const int plane_count
) {
    const int record = (int)blockDim.x * blockIdx.x + threadIdx.x;
    if (record >= row_count) return;
    const unsigned int mask = plane_count == 4 ? 0xffffffffU : ((1U << (8 * plane_count)) - 1U);
    unsigned int current = 0U;
    const long long destination = output_rows[record] * (long long)width * output_channels + output_channel;
    for (int x = 0; x < width; ++x) {
        unsigned int difference = 0U;
        for (int plane = 0; plane < plane_count; ++plane) {
            const long long offset = plane_offsets[(long long)record * plane_count + plane];
            difference |= (unsigned int)materialized[offset + x] << (8 * (plane_count - plane - 1));
        }
        current = (current + difference) & mask;
        const unsigned short bits = pixel_type == 1
            ? (unsigned short)current
            : __half_as_ushort(__float2half_rn(__uint_as_float(current << 8)));
        output[destination + (long long)x * output_channels] = bits;
    }
}

extern "C" __global__ void pixtreme_exr_pxr24_scatter_f32(
    const unsigned char* materialized,
    const long long* output_rows,
    const long long* plane_offsets,
    float* output,
    const int row_count,
    const int width,
    const int output_channels,
    const int output_channel,
    const int pixel_type,
    const int plane_count
) {
    const int record = (int)blockDim.x * blockIdx.x + threadIdx.x;
    if (record >= row_count) return;
    const unsigned int mask = plane_count == 4 ? 0xffffffffU : ((1U << (8 * plane_count)) - 1U);
    unsigned int current = 0U;
    const long long destination = output_rows[record] * (long long)width * output_channels + output_channel;
    for (int x = 0; x < width; ++x) {
        unsigned int difference = 0U;
        for (int plane = 0; plane < plane_count; ++plane) {
            const long long offset = plane_offsets[(long long)record * plane_count + plane];
            difference |= (unsigned int)materialized[offset + x] << (8 * (plane_count - plane - 1));
        }
        current = (current + difference) & mask;
        output[destination + (long long)x * output_channels] = pixel_type == 0
            ? (float)current
            : (pixel_type == 1 ? __half2float(__ushort_as_half((unsigned short)current)) : __uint_as_float(current << 8));
    }
}

extern "C" __global__ void pixtreme_exr_pxr24_scatter_u32(
    const unsigned char* materialized,
    const long long* output_rows,
    const long long* plane_offsets,
    unsigned int* output,
    const int row_count,
    const int width,
    const int output_channels,
    const int output_channel,
    const int pixel_type,
    const int plane_count
) {
    const int record = (int)blockDim.x * blockIdx.x + threadIdx.x;
    if (record >= row_count || pixel_type != 0 || plane_count != 4) return;
    unsigned int current = 0U;
    const long long destination = output_rows[record] * (long long)width * output_channels + output_channel;
    for (int x = 0; x < width; ++x) {
        unsigned int difference = 0U;
        for (int plane = 0; plane < plane_count; ++plane) {
            const long long offset = plane_offsets[(long long)record * plane_count + plane];
            difference |= (unsigned int)materialized[offset + x] << (8 * (plane_count - plane - 1));
        }
        current += difference;
        output[destination + (long long)x * output_channels] = current;
    }
}
"""


@lru_cache(maxsize=1)
def _pxr24_encode_rows_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_PXR24_SOURCE, "pixtreme_exr_pxr24_encode_rows")


@lru_cache(maxsize=3)
def _pxr24_scatter_kernel(output_dtype: str) -> cp.RawKernel:
    suffix = {"float16": "f16", "float32": "f32", "uint32": "u32"}[output_dtype]
    return cp.RawKernel(_EXR_PXR24_SOURCE, f"pixtreme_exr_pxr24_scatter_{suffix}")


def _pxr24_channel_row_records(
    container: _ExrContainer,
    prepared: _ExrPxr24ReadChunks,
    channel: _ExrChannel,
) -> tuple[np.ndarray, np.ndarray]:
    width = container.data_window[2] - container.data_window[0] + 1
    file_channels = container.parts[0].channels
    channel_index = next(index for index, file_channel in enumerate(file_channels) if file_channel.name == channel.name)
    plane_count = _EXR_PXR24_PLANE_COUNTS[channel.pixel_type]
    channel_count = len(file_channels)
    row_plane_count = sum(_EXR_PXR24_PLANE_COUNTS[file_channel.pixel_type] for file_channel in file_channels)
    channel_plane_start = sum(
        _EXR_PXR24_PLANE_COUNTS[file_channel.pixel_type] for file_channel in file_channels[:channel_index]
    )
    compressed_indices = np.flatnonzero(prepared.compressed)
    record_count = sum(container.chunks[int(index)].row_count for index in compressed_indices)
    output_rows = np.empty(record_count, dtype=np.int64)
    plane_offsets = np.empty((record_count, plane_count), dtype=np.int64)
    record_index = 0
    for chunk_index_value in compressed_indices:
        chunk_index = int(chunk_index_value)
        chunk = container.chunks[chunk_index]
        descriptor = cast(_Phase3ChunkDescriptor, chunk.phase3)
        materialized_base = int(prepared.materialized_offsets[chunk_index])
        if (
            len(descriptor.channel_rows) != chunk.row_count * channel_count
            or len(descriptor.planes) != chunk.row_count * row_plane_count
        ):
            channel_row_count = sum(row.channel_index == channel_index for row in descriptor.channel_rows)
            channel_plane_count = sum(plane.channel_index == channel_index for plane in descriptor.planes)
            raise _gpu_error(
                why="the PXR24 descriptor has incomplete row-channel plane ownership",
                what=(
                    f"chunk_y={chunk.y}, channel={channel.name!r}, rows={channel_row_count}, "
                    f"planes={channel_plane_count}, expected_rows={chunk.row_count}, "
                    f"expected_planes={chunk.row_count * plane_count}"
                ),
                how="assign every validated PXR24 plane to one row and file-order channel",
            )
        for chunk_row in range(chunk.row_count):
            row = descriptor.channel_rows[chunk_row * channel_count + channel_index]
            plane_start = chunk_row * row_plane_count + channel_plane_start
            planes = descriptor.planes[plane_start : plane_start + plane_count]
            observed_plane_indices = tuple(plane.plane_index for plane in planes)
            if (
                row.channel_name != channel.name
                or row.channel_index != channel_index
                or row.chunk_row != chunk_row
                or row.pixel_type != channel.pixel_type
                or row.materialized_span.size != width * plane_count
                or observed_plane_indices != tuple(range(plane_count))
                or any(
                    plane.channel_index != channel_index
                    or plane.channel_name != channel.name
                    or plane.chunk_row != chunk_row
                    or plane.output_row != row.output_row
                    or plane.materialized_span.size != width
                    for plane in planes
                )
                or planes[0].materialized_span.start != row.materialized_span.start
                or planes[-1].materialized_span.end != row.materialized_span.end
                or any(
                    previous.materialized_span.end != current.materialized_span.start
                    for previous, current in zip(planes, planes[1:], strict=False)
                )
            ):
                raise _gpu_error(
                    why="the PXR24 row-channel descriptor does not own its complete byte-plane span",
                    what=(
                        f"chunk_y={chunk.y}, channel={channel.name!r}, pixel_type={row.pixel_type}, "
                        f"span={row.materialized_span.size}, plane_indices={observed_plane_indices!r}, "
                        f"expected_span={width * plane_count}"
                    ),
                    how="assign every row-channel plane byte to exactly one validated descriptor",
                )
            output_rows[record_index] = row.output_row
            for plane_index, plane in enumerate(planes):
                plane_offsets[record_index, plane_index] = materialized_base + plane.materialized_span.start
            record_index += 1
    return output_rows, plane_offsets


def _scatter_pxr24_gpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    prepared: _ExrPxr24ReadChunks,
    materialized: cp.ndarray,
    output: cp.ndarray,
    *,
    output_dtype: str,
) -> None:
    width = container.data_window[2] - container.data_window[0] + 1
    for output_channel, channel in enumerate(selected):
        output_rows, plane_offsets = _pxr24_channel_row_records(container, prepared, channel)
        if not output_rows.size:
            continue
        if output_dtype == "uint32" and channel.pixel_type != 0:
            raise _gpu_error(
                why="the native PXR24 UINT scatter received a non-UINT selected channel",
                what=f"channel={channel.name!r}, pixel_type={channel.pixel_type}",
                how="route only homogeneous EXR UINT selections to the uint32 output lane",
            )
        if output_dtype != "uint32" and channel.pixel_type not in (1, 2):
            raise _gpu_error(
                why="the floating PXR24 scatter received a selected UINT channel",
                what=f"channel={channel.name!r}, pixel_type={channel.pixel_type}",
                how="select uint32 output when materializing a PXR24 UINT channel",
            )
        row_count = int(output_rows.size)
        block_count = (row_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _pxr24_scatter_kernel(output_dtype)(
            (block_count,),
            (_EXR_THREADS_PER_BLOCK,),
            (
                materialized,
                cp.asarray(output_rows),
                cp.asarray(plane_offsets).reshape(-1),
                output,
                np.int32(row_count),
                np.int32(width),
                np.int32(len(selected)),
                np.int32(output_channel),
                np.int32(channel.pixel_type),
                np.int32(plane_offsets.shape[1]),
            ),
        )


def _read_exr_pxr24_gpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    prepared = _prepare_exr_pxr24_read_chunks(container, materialize_host=False)
    device_staging = cp.asarray(prepared.host_staging)
    materialized = cp.empty(int(prepared.materialized_sizes.sum()), dtype=cp.uint8)
    compressed_indices = np.flatnonzero(prepared.compressed)
    if compressed_indices.size:
        deflate_inputs = tuple(
            zip(
                prepared.stage_offsets[compressed_indices].tolist(),
                prepared.stage_sizes[compressed_indices].tolist(),
                strict=True,
            )
        )
        deflate_outputs = tuple(
            zip(
                prepared.materialized_offsets[compressed_indices].tolist(),
                prepared.materialized_sizes[compressed_indices].tolist(),
                strict=True,
            )
        )
        _decode_deflate_chunks(
            device_staging,
            deflate_inputs,
            materialized,
            deflate_outputs,
            verify_output_sizes=False,
        )
    raw_decoded = cp.empty(int(prepared.raw_sizes.sum()), dtype=cp.uint8)
    _gather_raw_chunks(
        device_staging,
        raw_decoded,
        prepared.stage_offsets,
        prepared.raw_offsets,
        prepared.raw_sizes,
        prepared.compressed,
    )
    output = _unpack_exr_output(
        container,
        selected,
        raw_decoded,
        prepared.raw_offsets,
        prepared.raw_sizes,
        even_odd_grouped=np.zeros_like(prepared.compressed),
        output_dtype=output_dtype,
    )
    _scatter_pxr24_gpu(container, selected, prepared, materialized, output, output_dtype=output_dtype)
    return output


def _read_exr_pxr24_custom_cpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    prepared = _prepare_exr_pxr24_read_chunks(container, materialize_host=True)
    device_batch = cp.asarray(prepared.host_staging)
    cp.cuda.get_current_stream().synchronize()
    materialized_byte_count = int(prepared.materialized_sizes.sum())
    materialized = device_batch[:materialized_byte_count]
    raw_decoded = cp.empty(int(prepared.raw_sizes.sum()), dtype=cp.uint8)
    _gather_raw_chunks(
        device_batch,
        raw_decoded,
        prepared.stage_offsets,
        prepared.raw_offsets,
        prepared.raw_sizes,
        prepared.compressed,
    )
    output = _unpack_exr_output(
        container,
        selected,
        raw_decoded,
        prepared.raw_offsets,
        prepared.raw_sizes,
        even_odd_grouped=np.zeros_like(prepared.compressed),
        output_dtype=output_dtype,
    )
    _scatter_pxr24_gpu(container, selected, prepared, materialized, output, output_dtype=output_dtype)
    return output
