"""B44 and B44A OpenEXR read/write lane and CUDA kernels."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import cast

import cupy as cp
import numpy as np

from pixtreme._io.formats.exr.container import (
    _EXR_DTYPE_INFO,
    _EXR_MAX_GRID_X,
    _EXR_THREADS_PER_BLOCK,
    _ExrB44ReadChunks,
    _ExrChannel,
    _ExrContainer,
    _gpu_error,
    _Phase3B44Blocks,
    _Phase3Block,
)
from pixtreme._io.formats.exr.packing import (
    _chunk_launch_ranges,
    _device_i64,
    _gather_raw_chunks,
    _maximum_block_count,
    _numpy_offsets,
    _prefix_offsets,
    _select_exr_host_pixels,
    _unpack_exr_output,
)


def _b44_block_boundary_matches(
    block: _Phase3Block,
    *,
    channel_index: int,
    channel_name: str,
    block_row: int,
    block_column: int,
    chunk_row_start: int,
    chunk_row_count: int,
) -> bool:
    return (
        block.channel_index == channel_index
        and block.channel_name == channel_name
        and block.block_row == block_row
        and block.block_column == block_column
        and block.output_row_start == chunk_row_start + block_row * 4
        and block.output_row_count == min(4, chunk_row_count - block_row * 4)
        and block.stored_size in (3, 14)
    )


def _prepare_exr_b44_read_chunks(container: _ExrContainer) -> _ExrB44ReadChunks:
    chunks = container.chunks
    channels = container.parts[0].channels
    width = container.data_window[2] - container.data_window[0] + 1
    stage_sizes = np.fromiter((chunk.packed_size for chunk in chunks), dtype=np.int64, count=len(chunks))
    stage_offsets = _numpy_offsets(stage_sizes)
    raw_sizes = np.fromiter((chunk.expected_size for chunk in chunks), dtype=np.int64, count=len(chunks))
    raw_offsets = _numpy_offsets(raw_sizes)
    compressed = np.fromiter((not chunk.raw_stored for chunk in chunks), dtype=np.uint8, count=len(chunks))
    host_staging = np.frombuffer(
        b"".join(container.data[chunk.payload_start : chunk.payload_end] for chunk in chunks),
        dtype=np.uint8,
    )
    channel_row_sizes = np.fromiter(
        (width * channel.bytes_per_sample for channel in channels), dtype=np.int64, count=len(channels)
    )
    channel_row_offsets = _numpy_offsets(channel_row_sizes)
    row_bytes = int(channel_row_sizes.sum())
    block_columns = (width + 3) // 4
    raw_section_records: list[tuple[int, int, int, int, int, int, int]] = []
    block_section_records: list[tuple[int, int, int, int]] = []
    block_output_records: list[tuple[int, int, int, int, int, int, int]] = []
    block_flag_parts: list[np.ndarray] = []
    materialized_block_count = 0
    for chunk_index in np.flatnonzero(compressed):
        index = int(chunk_index)
        chunk = chunks[index]
        descriptor = chunk.phase3
        if descriptor is None or descriptor.codec not in ("b44", "b44a"):
            raise _gpu_error(
                why="the B44 read batch received a chunk without its validated B44/B44A descriptor",
                what=f"chunk_y={chunk.y}, descriptor={descriptor!r}",
                how="parse every eligible B44/B44A chunk before materializing its channel sections",
            )
        if len(descriptor.channel_sections) != len(channels):
            raise _gpu_error(
                why="the B44 descriptor does not assign one section to every file-order channel",
                what=(f"chunk_y={chunk.y}, sections={len(descriptor.channel_sections)}, channels={len(channels)}"),
                how="assign the compressed payload to every file channel exactly once",
            )
        stage_base = int(stage_offsets[index])
        raw_base = int(raw_offsets[index])
        observed_blocks = 0
        for channel_index, (channel, section) in enumerate(zip(channels, descriptor.channel_sections, strict=True)):
            expected_materialized_size = width * chunk.row_count * channel.bytes_per_sample
            if (
                section.channel_index != channel_index
                or section.channel_name != channel.name
                or section.pixel_type != channel.pixel_type
                or section.bytes_per_sample != channel.bytes_per_sample
                or section.perceptually_linear != channel.perceptually_linear
                or section.expected_materialized_size != expected_materialized_size
            ):
                raise _gpu_error(
                    why="the B44 channel section ownership differs from the parsed file channel",
                    what=(
                        f"chunk_y={chunk.y}, section_channel={section.channel_name!r}, "
                        f"file_channel={channel.name!r}, expected_size={expected_materialized_size}, "
                        f"observed_size={section.expected_materialized_size}"
                    ),
                    how="keep each B44 section bound to one file-order channel and its complete geometry",
                )
            source_offset = stage_base + section.payload_span.start - chunk.payload_start
            channel_offset = int(channel_row_offsets[channel_index])
            if channel.pixel_type != 1:
                if (
                    section.block_start != observed_blocks
                    or section.block_count != 0
                    or section.payload_span.size != expected_materialized_size
                ):
                    raise _gpu_error(
                        why="the B44 raw channel section has blocks or an incomplete plane span",
                        what=(
                            f"chunk_y={chunk.y}, channel={channel.name!r}, blocks={section.block_count}, "
                            f"stored={section.payload_span.size}, expected={expected_materialized_size}"
                        ),
                        how="store every UINT or FLOAT plane byte exactly once without HALF block descriptors",
                    )
                raw_section_records.append(
                    (
                        source_offset,
                        raw_base,
                        channel_offset,
                        width,
                        chunk.row_count,
                        channel.bytes_per_sample,
                        row_bytes,
                    )
                )
                continue
            expected_block_count = ((chunk.row_count + 3) // 4) * block_columns
            if section.block_start != observed_blocks or section.block_count != expected_block_count:
                raise _gpu_error(
                    why="the B44 HALF section block ownership differs from its four-by-four geometry",
                    what=(
                        f"chunk_y={chunk.y}, channel={channel.name!r}, block_start={section.block_start}, "
                        f"observed={section.block_count}, expected={expected_block_count}"
                    ),
                    how="assign every block-row and block-column to its HALF channel exactly once",
                )
            block_end = section.block_start + section.block_count
            if block_end > len(descriptor.blocks):
                raise _gpu_error(
                    why="the B44 HALF section references blocks outside the validated descriptor",
                    what=(
                        f"chunk_y={chunk.y}, channel={channel.name!r}, block_end={block_end}, "
                        f"expected={expected_block_count}"
                    ),
                    how="keep the section block range within the descriptor block table",
                )
            block_rows = (chunk.row_count + 3) // 4
            if isinstance(descriptor.blocks, _Phase3B44Blocks):
                half_sections = tuple(item for item in descriptor.channel_sections if item.block_count)
                boundaries_valid = (
                    descriptor.blocks.payload is container.data
                    and descriptor.blocks.codec == descriptor.codec
                    and descriptor.blocks.block_sections == half_sections
                    and descriptor.blocks.block_columns == block_columns
                    and descriptor.blocks.row_start == chunk.row_start
                    and descriptor.blocks.row_count == chunk.row_count
                )
                boundary_evidence = (
                    f"lazy codec={descriptor.blocks.codec!r} vs {descriptor.codec!r}, "
                    f"rows=({descriptor.blocks.row_start},{descriptor.blocks.row_count}) "
                    f"vs ({chunk.row_start},{chunk.row_count}), "
                    f"block_columns={descriptor.blocks.block_columns} vs {block_columns}"
                )
            else:
                first_block = descriptor.blocks[section.block_start]
                last_block = descriptor.blocks[block_end - 1]
                boundaries_valid = (
                    _b44_block_boundary_matches(
                        first_block,
                        channel_index=channel_index,
                        channel_name=channel.name,
                        block_row=0,
                        block_column=0,
                        chunk_row_start=chunk.row_start,
                        chunk_row_count=chunk.row_count,
                    )
                    and first_block.payload_span.start == section.payload_span.start
                    and _b44_block_boundary_matches(
                        last_block,
                        channel_index=channel_index,
                        channel_name=channel.name,
                        block_row=block_rows - 1,
                        block_column=block_columns - 1,
                        chunk_row_start=chunk.row_start,
                        chunk_row_count=chunk.row_count,
                    )
                    and last_block.payload_span.end == section.payload_span.end
                )
                boundary_evidence = (
                    f"first=({first_block.block_row},{first_block.block_column}), "
                    f"last=({last_block.block_row},{last_block.block_column})"
                )
            if not boundaries_valid:
                raise _gpu_error(
                    why="the B44 block descriptor has incomplete channel, geometry, or payload ownership",
                    what=f"chunk_y={chunk.y}, channel={channel.name!r}, {boundary_evidence}",
                    how="bind every validated block section boundary to its four-by-four output geometry",
                )
            block_section_records.append(
                (source_offset, section.payload_span.size, materialized_block_count, expected_block_count)
            )
            block_output_records.append(
                (
                    raw_base,
                    row_bytes,
                    channel_offset,
                    width,
                    chunk.row_count,
                    materialized_block_count,
                    expected_block_count,
                )
            )
            block_flag_parts.append(np.full(expected_block_count, channel.perceptually_linear, dtype=np.uint8))
            materialized_block_count += expected_block_count
            observed_blocks += expected_block_count
        if observed_blocks != len(descriptor.blocks):
            raise _gpu_error(
                why="the B44 descriptor block table is not consumed exactly by its HALF sections",
                what=f"chunk_y={chunk.y}, consumed={observed_blocks}, blocks={len(descriptor.blocks)}",
                how="remove unowned or duplicate B44 block descriptors",
            )
    block_perceptually_linear = np.concatenate(block_flag_parts) if block_flag_parts else np.empty(0, dtype=np.uint8)
    return _ExrB44ReadChunks(
        host_staging=host_staging,
        stage_offsets=stage_offsets,
        stage_sizes=stage_sizes,
        raw_offsets=raw_offsets,
        raw_sizes=raw_sizes,
        compressed=compressed,
        raw_section_descriptors=np.asarray(raw_section_records, dtype=np.int64).reshape(-1, 7),
        block_section_descriptors=np.asarray(block_section_records, dtype=np.int64).reshape(-1, 4),
        block_output_descriptors=np.asarray(block_output_records, dtype=np.int64).reshape(-1, 7),
        block_perceptually_linear=block_perceptually_linear,
        b44a=container.compression == "b44a",
    )


def _copy_b44_raw_ranges_host(prepared: _ExrB44ReadChunks, decoded: np.ndarray) -> None:
    raw_indices = np.flatnonzero(np.logical_not(prepared.compressed))
    if not raw_indices.size:
        return
    sizes = prepared.raw_sizes[raw_indices]
    range_ids = np.repeat(np.arange(raw_indices.size, dtype=np.int64), sizes)
    range_offsets = _numpy_offsets(sizes)
    within = np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(range_offsets, sizes)
    source = prepared.stage_offsets[raw_indices][range_ids] + within
    destination = prepared.raw_offsets[raw_indices][range_ids] + within
    decoded[destination] = prepared.host_staging[source]


def _b44_block_ranges_host(prepared: _ExrB44ReadChunks) -> tuple[np.ndarray, np.ndarray]:
    block_count = int(prepared.block_perceptually_linear.size)
    block_offsets = np.empty(block_count, dtype=np.int64)
    block_sizes = np.empty(block_count, dtype=np.int32)
    for descriptor in prepared.block_section_descriptors:
        source, section_size, first_block, section_block_count = map(int, descriptor)
        if not prepared.b44a:
            block_end = first_block + section_block_count
            block_offsets[first_block:block_end] = source + np.arange(section_block_count, dtype=np.int64) * 14
            block_sizes[first_block:block_end] = 14
            if section_block_count * 14 != section_size:
                raise _gpu_error(
                    why="the host B44 block-head geometry does not consume its dense channel section exactly",
                    what=(
                        f"section={source}:{source + section_size}, blocks={section_block_count}, "
                        f"expected_size={section_block_count * 14}"
                    ),
                    how="make every B44 HALF channel contain one fourteen-byte payload per block",
                )
            continue
        cursor = source
        section_end = source + section_size
        for local_block in range(section_block_count):
            if cursor + 3 > section_end:
                raise _gpu_error(
                    why="the host B44 block-head scan reached a truncated block header",
                    what=f"section={source}:{section_end}, block={local_block}, cursor={cursor}",
                    how="provide the complete three-byte head for every declared B44 block",
                )
            marker = int(prepared.host_staging[cursor + 2])
            if not prepared.b44a and marker >= 0x34:
                raise _gpu_error(
                    why="the host B44 block-head scan found an invalid dense marker",
                    what=f"section={source}:{section_end}, block={local_block}, marker=0x{marker:02x}",
                    how="encode every B44 block as a fourteen-byte dense form with byte[2] below 0x34",
                )
            stored_size = 3 if prepared.b44a and marker >= 0x34 else 14
            if cursor + stored_size > section_end:
                raise _gpu_error(
                    why="the host B44 block-head scan reached a truncated block body",
                    what=(
                        f"section={source}:{section_end}, block={local_block}, cursor={cursor}, "
                        f"stored_size={stored_size}"
                    ),
                    how="provide every byte selected by the B44/B44A block marker",
                )
            block_index = first_block + local_block
            block_offsets[block_index] = cursor
            block_sizes[block_index] = stored_size
            cursor += stored_size
        if cursor != section_end:
            raise _gpu_error(
                why="the host B44 block-head scan did not consume its channel section exactly",
                what=f"section={source}:{section_end}, consumed_end={cursor}",
                how="make the declared block sequence cover the HALF channel section without trailing bytes",
            )
    return block_offsets, block_sizes


def _decode_b44_blocks_host(
    payload: np.ndarray,
    offsets: np.ndarray,
    sizes: np.ndarray,
    perceptually_linear: np.ndarray,
) -> np.ndarray:
    source = np.asarray(payload, dtype=np.uint8).reshape(-1)
    block_offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    block_sizes = np.asarray(sizes, dtype=np.int32).reshape(-1)
    flags = np.asarray(perceptually_linear, dtype=np.uint8).reshape(-1)
    block_count = int(block_offsets.size)
    if block_count == 0 or block_sizes.size != block_count or flags.size != block_count:
        raise _gpu_error(
            why="the host B44 batch decoder received empty or mismatched block metadata",
            what=f"offsets={block_count}, sizes={block_sizes.size}, flags={flags.size}",
            how="provide one validated range and pLinear flag per B44/B44A block",
        )
    invalid_sizes = np.flatnonzero((block_sizes != 3) & (block_sizes != 14))
    if invalid_sizes.size:
        raise _gpu_error(
            why="the host B44 batch decoder received a block outside the three- or fourteen-byte grammar",
            what=f"block={int(invalid_sizes[0])}, stored_size={int(block_sizes[invalid_sizes[0]])}",
            how="provide one validated B44A flat or B44/B44A dense block",
        )
    byte_indices = block_offsets[:, np.newaxis] + np.minimum(
        np.arange(14, dtype=np.int64)[np.newaxis, :],
        block_sizes[:, np.newaxis] - 1,
    )
    dense = source[byte_indices].astype(np.uint16)
    ordered = np.empty((block_count, 16), dtype=np.uint16)
    ordered[:, 0] = (dense[:, 0] << 8) | dense[:, 1]
    flat = (block_sizes == 3) | (dense[:, 2] >= np.uint16(0x34))
    shift = dense[:, 2] >> 2
    residuals = np.column_stack(
        (
            ((dense[:, 2] & 0x03) << 4) | (dense[:, 3] >> 4),
            ((dense[:, 3] & 0x0F) << 2) | (dense[:, 4] >> 6),
            dense[:, 4] & 0x3F,
            dense[:, 5] >> 2,
            ((dense[:, 5] & 0x03) << 4) | (dense[:, 6] >> 4),
            ((dense[:, 6] & 0x0F) << 2) | (dense[:, 7] >> 6),
            dense[:, 7] & 0x3F,
            dense[:, 8] >> 2,
            ((dense[:, 8] & 0x03) << 4) | (dense[:, 9] >> 4),
            ((dense[:, 9] & 0x0F) << 2) | (dense[:, 10] >> 6),
            dense[:, 10] & 0x3F,
            dense[:, 11] >> 2,
            ((dense[:, 11] & 0x03) << 4) | (dense[:, 12] >> 4),
            ((dense[:, 12] & 0x0F) << 2) | (dense[:, 13] >> 6),
            dense[:, 13] & 0x3F,
        )
    ).astype(np.int32, copy=False)
    residuals[flat] = 32
    shift[flat] = 0
    for residual_index, (source_index, target_index) in enumerate(
        (
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
    ):
        delta = np.left_shift(residuals[:, residual_index] - 32, shift)
        ordered[:, target_index] = np.bitwise_and(
            ordered[:, source_index].astype(np.int32) + delta,
            0xFFFF,
        ).astype(np.uint16)
    bits = np.where(
        np.bitwise_and(ordered, np.uint16(0x8000)) != 0,
        np.bitwise_and(ordered, np.uint16(0x7FFF)),
        np.bitwise_not(ordered),
    ).astype(np.uint16)
    flagged = flags.astype(bool)
    if np.any(flagged):
        bits[flagged] = _b44_plinear_luts_host()[1][bits[flagged]]
    return bits


def _materialize_b44_host(prepared: _ExrB44ReadChunks) -> np.ndarray:
    decoded = np.empty(int(prepared.raw_sizes.sum()), dtype=np.uint8)
    _copy_b44_raw_ranges_host(prepared, decoded)
    for descriptor in prepared.raw_section_descriptors:
        source, destination, channel_offset, width, row_count, bytes_per_sample, row_bytes = map(int, descriptor)
        channel_row_bytes = width * bytes_per_sample
        plane = prepared.host_staging[source : source + row_count * channel_row_bytes].reshape(
            row_count, channel_row_bytes
        )
        output_rows = decoded[destination : destination + row_count * row_bytes].reshape(row_count, row_bytes)
        output_rows[:, channel_offset : channel_offset + channel_row_bytes] = plane
    block_count = int(prepared.block_perceptually_linear.size)
    if block_count:
        block_offsets, block_sizes = _b44_block_ranges_host(prepared)
        blocks = _decode_b44_blocks_host(
            prepared.host_staging,
            block_offsets,
            block_sizes,
            prepared.block_perceptually_linear,
        )
        for section in prepared.block_output_descriptors:
            destination, row_bytes, channel_offset, width, row_count, first_block, section_block_count = map(
                int, section
            )
            block_columns = (width + 3) // 4
            block_rows = (row_count + 3) // 4
            plane_bits = (
                blocks[first_block : first_block + section_block_count]
                .reshape(block_rows, block_columns, 4, 4)
                .transpose(0, 2, 1, 3)
                .reshape(block_rows * 4, block_columns * 4)[:row_count, :width]
            )
            plane_bytes = plane_bits.astype("<u2", copy=False).view(np.uint8).reshape(row_count, width * 2)
            output_rows = decoded[destination : destination + row_count * row_bytes].reshape(row_count, row_bytes)
            output_rows[:, channel_offset : channel_offset + width * 2] = plane_bytes
    return decoded


@lru_cache(maxsize=1)
def _b44_plinear_luts_host() -> tuple[np.ndarray, np.ndarray]:
    bits = np.arange(65536, dtype=np.uint16)
    values = bits.view(np.float16).astype(np.float32)
    special = np.bitwise_and(bits, np.uint16(0x7C00)) == np.uint16(0x7C00)

    encode = np.zeros(65536, dtype=np.uint16)
    saturated = (bits >= np.uint16(0x558C)) & (bits < np.uint16(0x8000))
    encode[saturated] = np.uint16(0x7BFF)
    regular_encode = ~(special | saturated)
    with np.errstate(over="ignore", invalid="ignore"):
        encoded = np.exp(values[regular_encode] / np.float32(8.0)).astype(np.float32)
        encode[regular_encode] = encoded.astype(np.float16).view(np.uint16)

    decode = np.zeros(65536, dtype=np.uint16)
    regular_decode = ~(special | (bits > np.uint16(0x8000)))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        decoded = (np.float32(8.0) * np.log(values[regular_decode])).astype(np.float32)
        decode[regular_decode] = decoded.astype(np.float16).view(np.uint16)
    encode.flags.writeable = False
    decode.flags.writeable = False
    return encode, decode


@lru_cache(maxsize=1)
def _b44_plinear_luts_gpu() -> tuple[cp.ndarray, cp.ndarray]:
    encode, decode = _b44_plinear_luts_host()
    return cp.asarray(encode), cp.asarray(decode)


def _decode_b44_block_host(
    payload: np.ndarray,
    *,
    perceptually_linear: bool,
) -> np.ndarray:
    source = np.asarray(payload, dtype=np.uint8).reshape(-1)
    if source.size not in (3, 14):
        raise _gpu_error(
            why="the host B44 decoder received a block outside the three- or fourteen-byte grammar",
            what=f"stored_size={source.size}",
            how="provide one validated B44A flat or B44/B44A dense block",
        )
    ordered = np.empty(16, dtype=np.uint16)
    if source.size == 3 or source[2] >= 0x34:
        ordered.fill((int(source[0]) << 8) | int(source[1]))
    else:
        packed = int.from_bytes(source.tobytes(), "big")
        ordered[0] = np.uint16((packed >> 96) & 0xFFFF)
        shift = (packed >> 90) & 0x3F
        for residual_index, (source_index, target_index) in enumerate(
            (
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
        ):
            residual = (packed >> (84 - 6 * residual_index)) & 0x3F
            ordered[target_index] = np.uint16((int(ordered[source_index]) + ((residual - 32) << shift)) & 0xFFFF)
    bits = np.where(
        np.bitwise_and(ordered, np.uint16(0x8000)) != 0,
        np.bitwise_and(ordered, np.uint16(0x7FFF)),
        np.bitwise_not(ordered),
    ).astype(np.uint16)
    if perceptually_linear:
        bits = _b44_plinear_luts_host()[1][bits]
    return bits


def _b44_block_flags(perceptually_linear: bool | cp.ndarray, block_count: int) -> cp.ndarray:
    if isinstance(perceptually_linear, bool):
        return cp.full(block_count, perceptually_linear, dtype=cp.uint8)
    flags = cp.ascontiguousarray(perceptually_linear, dtype=cp.uint8).reshape(-1)
    if int(flags.size) != block_count:
        raise _gpu_error(
            why="the B44 block primitive received a pLinear flag count that differs from its block count",
            what=f"flags={int(flags.size)}, blocks={block_count}",
            how="provide one pLinear flag for every independent four-by-four HALF block",
        )
    return flags


def _encode_b44_blocks_gpu(
    blocks: cp.ndarray,
    *,
    b44a: bool,
    perceptually_linear: bool | cp.ndarray = False,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    values = cp.ascontiguousarray(blocks, dtype=cp.uint16)
    if values.ndim != 2 or values.shape[1] != 16 or values.shape[0] == 0:
        raise _gpu_error(
            why="the B44 block encoder received an empty or non-four-by-four block matrix",
            what=f"shape={values.shape!r}",
            how="provide a two-dimensional uint16 array with sixteen HALF patterns per row",
        )
    block_count = int(values.shape[0])
    flags = _b44_block_flags(perceptually_linear, block_count)
    dense = cp.empty(block_count * 14, dtype=cp.uint8)
    sizes = cp.empty(block_count, dtype=cp.int32)
    encode_lut, _ = _b44_plinear_luts_gpu()
    grid = (block_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _b44_encode_blocks_kernel()(
        (grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (values, flags, encode_lut, dense, sizes, np.int64(block_count), np.int32(b44a)),
    )
    offsets = cp.empty(block_count, dtype=cp.int64)
    offsets[0] = 0
    if block_count > 1:
        offsets[1:] = cp.cumsum(sizes[:-1], dtype=cp.int64)
    output_size = int((offsets[-1] + sizes[-1]).item())
    output = cp.empty(output_size, dtype=cp.uint8)
    _b44_assemble_blocks_kernel()(
        (grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (dense, sizes, offsets, output, np.int64(block_count)),
    )
    return cast(cp.ndarray, output), cast(cp.ndarray, offsets), cast(cp.ndarray, sizes)


def _decode_b44_blocks_gpu(
    payload: cp.ndarray,
    offsets: cp.ndarray,
    sizes: cp.ndarray,
    *,
    perceptually_linear: bool | cp.ndarray = False,
) -> cp.ndarray:
    source = cp.ascontiguousarray(payload, dtype=cp.uint8).reshape(-1)
    block_offsets = cp.ascontiguousarray(offsets, dtype=cp.int64).reshape(-1)
    block_sizes = cp.ascontiguousarray(sizes, dtype=cp.int32).reshape(-1)
    block_count = int(block_offsets.size)
    if block_count == 0 or int(block_sizes.size) != block_count:
        raise _gpu_error(
            why="the B44 block decoder received empty or mismatched offset and size arrays",
            what=f"offsets={block_count}, sizes={int(block_sizes.size)}",
            how="provide one validated three- or fourteen-byte range for every B44 block",
        )
    flags = _b44_block_flags(perceptually_linear, block_count)
    output = cp.empty((block_count, 16), dtype=cp.uint16)
    _, decode_lut = _b44_plinear_luts_gpu()
    grid = (block_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _b44_decode_blocks_kernel()(
        (grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (source, block_offsets, block_sizes, flags, decode_lut, output, np.int64(block_count)),
    )
    return cast(cp.ndarray, output)


def _b44_reorder_chunk_planes_gpu(
    raw: cp.ndarray,
    chunk_offsets: Sequence[int],
    row_counts: Sequence[int],
    *,
    width: int,
    channel_count: int,
    bytes_per_sample: int,
) -> cp.ndarray:
    offsets = tuple(int(value) for value in chunk_offsets)
    rows = tuple(int(value) for value in row_counts)
    if len(offsets) != len(rows) or not rows or min(rows) <= 0 or min(width, channel_count, bytes_per_sample) <= 0:
        raise _gpu_error(
            why="the B44 plane reorder received empty or mismatched chunk geometry",
            what=(
                f"offsets={len(offsets)}, rows={rows!r}, width={width}, channels={channel_count}, "
                f"bytes_per_sample={bytes_per_sample}"
            ),
            how="provide one positive row count and raw offset for every B44 chunk",
        )
    sizes = tuple(row_count * width * channel_count * bytes_per_sample for row_count in rows)
    source = cp.ascontiguousarray(raw, dtype=cp.uint8).reshape(-1)
    if any(offset < 0 or offset + size > int(source.size) for offset, size in zip(offsets, sizes, strict=True)):
        raise _gpu_error(
            why="the B44 plane reorder chunk range exceeds the packed raw input",
            what=f"raw_size={int(source.size)}, ranges={tuple(zip(offsets, sizes, strict=True))!r}",
            how="provide complete scanline-interleaved packed chunk bytes",
        )
    output_offsets = _prefix_offsets(sizes)
    output = cp.empty(sum(sizes), dtype=cp.uint8)
    block_count = _maximum_block_count(sizes)
    device_offsets = _device_i64(offsets)
    device_sizes = _device_i64(sizes)
    device_output_offsets = _device_i64(output_offsets)
    device_rows = cp.asarray(np.asarray(rows, dtype=np.int32))
    for start, end in _chunk_launch_ranges(len(rows)):
        batch_size = end - start
        _b44_reorder_planes_kernel()(
            (block_count, batch_size),
            (_EXR_THREADS_PER_BLOCK,),
            (
                source,
                device_offsets[start:end],
                device_sizes[start:end],
                device_output_offsets[start:end],
                output,
                device_rows[start:end],
                np.int32(width),
                np.int32(channel_count),
                np.int32(bytes_per_sample),
                np.int32(batch_size),
            ),
        )
    return cast(cp.ndarray, output)


def _encode_b44_chunks_gpu(
    raw: cp.ndarray,
    raw_offsets: Sequence[int],
    row_counts: Sequence[int],
    *,
    width: int,
    channel_count: int,
    pixel_type: int,
    codec: str,
    perceptually_linear: Sequence[bool] | None = None,
) -> tuple[cp.ndarray, tuple[int, ...], tuple[int, ...]]:
    offsets = tuple(int(value) for value in raw_offsets)
    rows = tuple(int(value) for value in row_counts)
    if codec not in ("b44", "b44a") or pixel_type not in _EXR_DTYPE_INFO:
        raise _gpu_error(
            why="the B44 chunk encoder received an unsupported codec or pixel type",
            what=f"codec={codec!r}, pixel_type={pixel_type}",
            how="provide B44/B44A chunks containing UINT, HALF, or FLOAT samples",
        )
    bytes_per_sample = _EXR_DTYPE_INFO[pixel_type][1]
    raw_sizes = tuple(row_count * width * channel_count * bytes_per_sample for row_count in rows)
    if pixel_type != 1:
        planes = _b44_reorder_chunk_planes_gpu(
            raw,
            offsets,
            rows,
            width=width,
            channel_count=channel_count,
            bytes_per_sample=bytes_per_sample,
        )
        return planes, _prefix_offsets(raw_sizes), raw_sizes

    channel_flags = (
        tuple(False for _ in range(channel_count)) if perceptually_linear is None else tuple(perceptually_linear)
    )
    if len(channel_flags) != channel_count:
        raise _gpu_error(
            why="the B44 chunk encoder received a channel pLinear flag count mismatch",
            what=f"flags={len(channel_flags)}, channels={channel_count}",
            how="provide one pLinear declaration per file-order HALF channel",
        )
    row_counts_array = np.asarray(rows, dtype=np.int32)
    block_columns = (width + 3) // 4
    chunk_block_counts = ((row_counts_array.astype(np.int64) + 3) // 4) * block_columns * channel_count
    chunk_block_starts = _numpy_offsets(chunk_block_counts)
    block_count = int(chunk_block_counts.sum())
    blocks = cp.empty((block_count, 16), dtype=cp.uint16)
    block_flags = cp.empty(block_count, dtype=cp.uint8)
    device_chunk_offsets = cp.asarray(np.asarray(offsets, dtype=np.int64))
    device_row_counts = cp.asarray(row_counts_array)
    device_chunk_block_starts = cp.asarray(chunk_block_starts)
    device_channel_flags = cp.asarray(np.asarray(channel_flags, dtype=np.uint8))
    sample_count = block_count * 16
    grid = (sample_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
    _b44_gather_blocks_kernel()(
        (grid,),
        (_EXR_THREADS_PER_BLOCK,),
        (
            raw,
            device_chunk_offsets,
            device_row_counts,
            device_chunk_block_starts,
            device_channel_flags,
            blocks,
            block_flags,
            np.int64(block_count),
            np.int32(len(rows)),
            np.int32(width),
            np.int32(channel_count),
        ),
    )
    encoded, _, block_sizes = _encode_b44_blocks_gpu(
        blocks,
        b44a=codec == "b44a",
        perceptually_linear=block_flags,
    )
    device_chunk_sizes = cp.add.reduceat(block_sizes.astype(cp.int64), device_chunk_block_starts)
    output_sizes = tuple(int(value) for value in device_chunk_sizes.get())
    return encoded, _prefix_offsets(output_sizes), output_sizes


_EXR_B44_SOURCE = r"""
__device__ __forceinline__ int pixtreme_b44_shift_round(int value, int shift) {
    value <<= 1;
    const int addend = (1 << shift) - 1;
    const int next_shift = shift + 1;
    const int parity = (value >> next_shift) & 1;
    return (value + addend + parity) >> next_shift;
}

__device__ __forceinline__ unsigned short pixtreme_b44_ordered(unsigned short bits) {
    if ((bits & 0x7c00U) == 0x7c00U) return 0x8000U;
    if (bits & 0x8000U) return (unsigned short)(~bits);
    return (unsigned short)(bits | 0x8000U);
}

__device__ __forceinline__ unsigned short pixtreme_b44_half(unsigned short ordered) {
    if (ordered & 0x8000U) return (unsigned short)(ordered & 0x7fffU);
    return (unsigned short)(~ordered);
}

extern "C" __global__ void pixtreme_exr_b44_encode_blocks(
    const unsigned short* blocks,
    const unsigned char* perceptually_linear,
    const unsigned short* encode_lut,
    unsigned char* dense,
    int* sizes,
    const long long block_count,
    const int flat_fields
) {
    const long long block = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (block >= block_count) return;
    const int edge_from[15] = {0, 4, 8, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14};
    const int edge_to[15] = {4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    unsigned short ordered[16];
    int difference[16];
    int residual[15];
    unsigned short maximum = 0U;
    for (int index = 0; index < 16; ++index) {
        unsigned short bits = blocks[block * 16LL + index];
        if (perceptually_linear[block]) bits = encode_lut[bits];
        ordered[index] = pixtreme_b44_ordered(bits);
        if (ordered[index] > maximum) maximum = ordered[index];
    }
    int shift = -1;
    int minimum;
    int maximum_residual;
    do {
        ++shift;
        for (int index = 0; index < 16; ++index) {
            difference[index] = pixtreme_b44_shift_round((int)maximum - (int)ordered[index], shift);
        }
        for (int index = 0; index < 15; ++index) {
            residual[index] = difference[edge_from[index]] - difference[edge_to[index]] + 32;
        }
        minimum = residual[0];
        maximum_residual = residual[0];
        for (int index = 1; index < 15; ++index) {
            if (residual[index] < minimum) minimum = residual[index];
            if (residual[index] > maximum_residual) maximum_residual = residual[index];
        }
    } while (minimum < 0 || maximum_residual > 63);

    const long long output = block * 14LL;
    if (flat_fields && minimum == 32 && maximum_residual == 32) {
        dense[output] = (unsigned char)(ordered[0] >> 8);
        dense[output + 1] = (unsigned char)ordered[0];
        dense[output + 2] = 0xfcU;
        for (int index = 3; index < 14; ++index) dense[output + index] = 0U;
        sizes[block] = 3;
        return;
    }
    const unsigned short base = perceptually_linear[block]
        ? ordered[0]
        : (unsigned short)((int)maximum - (difference[0] << shift));
    dense[output] = (unsigned char)(base >> 8);
    dense[output + 1] = (unsigned char)base;
    dense[output + 2] = (unsigned char)((shift << 2) | (residual[0] >> 4));
    dense[output + 3] = (unsigned char)((residual[0] << 4) | (residual[1] >> 2));
    dense[output + 4] = (unsigned char)((residual[1] << 6) | residual[2]);
    dense[output + 5] = (unsigned char)((residual[3] << 2) | (residual[4] >> 4));
    dense[output + 6] = (unsigned char)((residual[4] << 4) | (residual[5] >> 2));
    dense[output + 7] = (unsigned char)((residual[5] << 6) | residual[6]);
    dense[output + 8] = (unsigned char)((residual[7] << 2) | (residual[8] >> 4));
    dense[output + 9] = (unsigned char)((residual[8] << 4) | (residual[9] >> 2));
    dense[output + 10] = (unsigned char)((residual[9] << 6) | residual[10]);
    dense[output + 11] = (unsigned char)((residual[11] << 2) | (residual[12] >> 4));
    dense[output + 12] = (unsigned char)((residual[12] << 4) | (residual[13] >> 2));
    dense[output + 13] = (unsigned char)((residual[13] << 6) | residual[14]);
    sizes[block] = 14;
}

extern "C" __global__ void pixtreme_exr_b44_assemble_blocks(
    const unsigned char* dense,
    const int* sizes,
    const long long* offsets,
    unsigned char* output,
    const long long block_count
) {
    const long long block = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (block >= block_count) return;
    for (int index = 0; index < sizes[block]; ++index) {
        output[offsets[block] + index] = dense[block * 14LL + index];
    }
}

extern "C" __global__ void pixtreme_exr_b44_scan_block_heads(
    const unsigned char* payload,
    const long long* sections,
    int* block_sizes,
    int* section_status,
    const int section_count,
    const int b44a
) {
    const int section = (int)blockDim.x * blockIdx.x + threadIdx.x;
    if (section >= section_count) return;
    const long long* descriptor = sections + (long long)section * 4LL;
    long long cursor = descriptor[0];
    const long long section_end = cursor + descriptor[1];
    const long long first_block = descriptor[2];
    const long long block_count = descriptor[3];
    for (long long local_block = 0; local_block < block_count; ++local_block) {
        if (cursor + 3LL > section_end) {
            section_status[section] = 1;
            return;
        }
        const unsigned char marker = payload[cursor + 2LL];
        if (!b44a && marker >= 0x34U) {
            section_status[section] = 2;
            return;
        }
        const int stored_size = b44a && marker >= 0x34U ? 3 : 14;
        if (cursor + stored_size > section_end) {
            section_status[section] = 3;
            return;
        }
        block_sizes[first_block + local_block] = stored_size;
        cursor += stored_size;
    }
    if (cursor != section_end) section_status[section] = 4;
}

__device__ __forceinline__ int pixtreme_b44_read6(
    const unsigned char* payload,
    long long offset,
    int residual_index
) {
    const int first_bit = 22 + residual_index * 6;
    int value = 0;
    for (int bit = 0; bit < 6; ++bit) {
        const int absolute = first_bit + bit;
        value = (value << 1) | ((payload[offset + absolute / 8] >> (7 - absolute % 8)) & 1U);
    }
    return value;
}

extern "C" __global__ void pixtreme_exr_b44_decode_blocks(
    const unsigned char* payload,
    const long long* offsets,
    const int* sizes,
    const unsigned char* perceptually_linear,
    const unsigned short* decode_lut,
    unsigned short* blocks,
    const long long block_count
) {
    const long long block = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (block >= block_count) return;
    const int edge_from[15] = {0, 4, 8, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14};
    const int edge_to[15] = {4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    const long long input = offsets[block];
    unsigned short ordered[16];
    ordered[0] = (unsigned short)(((unsigned short)payload[input] << 8) | payload[input + 1]);
    if (sizes[block] == 3 || payload[input + 2] >= 0x34U) {
        for (int index = 1; index < 16; ++index) ordered[index] = ordered[0];
    } else {
        const int shift = payload[input + 2] >> 2;
        for (int index = 0; index < 15; ++index) {
            const int residual = pixtreme_b44_read6(payload, input, index);
            ordered[edge_to[index]] = (unsigned short)(
                (int)ordered[edge_from[index]] + ((residual - 32) << shift)
            );
        }
    }
    for (int index = 0; index < 16; ++index) {
        unsigned short bits = pixtreme_b44_half(ordered[index]);
        if (perceptually_linear[block]) bits = decode_lut[bits];
        blocks[block * 16LL + index] = bits;
    }
}

extern "C" __global__ void pixtreme_exr_b44_reorder_planes(
    const unsigned char* raw,
    const long long* chunk_offsets,
    const long long* chunk_sizes,
    const long long* output_offsets,
    unsigned char* output,
    const int* row_counts,
    const int width,
    const int channel_count,
    const int bytes_per_sample,
    const int chunk_count
) {
    const int chunk = (int)blockIdx.y;
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count || index >= chunk_sizes[chunk]) return;
    const long long channel_span = (long long)row_counts[chunk] * width * bytes_per_sample;
    const int channel = (int)(index / channel_span);
    const long long within_channel = index - (long long)channel * channel_span;
    const int row = (int)(within_channel / ((long long)width * bytes_per_sample));
    const long long within_row = within_channel - (long long)row * width * bytes_per_sample;
    const long long source = (long long)row * width * channel_count * bytes_per_sample
        + (long long)channel * width * bytes_per_sample + within_row;
    output[output_offsets[chunk] + index] = raw[chunk_offsets[chunk] + source];
}

extern "C" __global__ void pixtreme_exr_b44_gather_blocks(
    const unsigned char* raw,
    const long long* chunk_offsets,
    const int* row_counts,
    const long long* chunk_block_starts,
    const unsigned char* channel_flags,
    unsigned short* blocks,
    unsigned char* block_flags,
    const long long block_count,
    const int chunk_count,
    const int width,
    const int channel_count
) {
    const long long sample = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (sample >= block_count * 16LL) return;
    const long long block = sample / 16LL;
    const int local = (int)(sample - block * 16LL);
    int lower = 0;
    int upper = chunk_count;
    while (lower + 1 < upper) {
        const int middle = lower + (upper - lower) / 2;
        if (chunk_block_starts[middle] <= block) lower = middle;
        else upper = middle;
    }
    const int chunk = lower;
    const int row_count = row_counts[chunk];
    const int block_columns = (width + 3) / 4;
    const int block_rows = (row_count + 3) / 4;
    const long long blocks_per_channel = (long long)block_rows * block_columns;
    const long long within_chunk = block - chunk_block_starts[chunk];
    const int channel = (int)(within_chunk / blocks_per_channel);
    const long long within_channel = within_chunk - (long long)channel * blocks_per_channel;
    const int block_row = (int)(within_channel / block_columns);
    const int block_column = (int)(within_channel - (long long)block_row * block_columns);
    int row = block_row * 4 + local / 4;
    int column = block_column * 4 + local % 4;
    if (row >= row_count) row = row_count - 1;
    if (column >= width) column = width - 1;
    const long long byte_offset = chunk_offsets[chunk] +
        ((long long)row * channel_count * width + (long long)channel * width + column) * 2LL;
    blocks[sample] = (unsigned short)(raw[byte_offset] | ((unsigned short)raw[byte_offset + 1] << 8));
    if (local == 0) block_flags[block] = channel_flags[channel];
}

extern "C" __global__ void pixtreme_exr_b44_copy_sections(
    const unsigned char* staging,
    unsigned char* decoded,
    const long long* descriptors,
    const int section_count
) {
    for (int section = (int)blockIdx.x; section < section_count; section += (int)gridDim.x) {
        const long long* descriptor = descriptors + (long long)section * 7LL;
        const long long source = descriptor[0];
        const long long destination = descriptor[1];
        const long long channel_offset = descriptor[2];
        const long long width = descriptor[3];
        const long long row_count = descriptor[4];
        const long long bytes_per_sample = descriptor[5];
        const long long row_bytes = descriptor[6];
        const long long size = row_count * width * bytes_per_sample;
        for (long long index = threadIdx.x; index < size; index += blockDim.x) {
            const long long channel_row_bytes = width * bytes_per_sample;
            const long long row = index / channel_row_bytes;
            const long long within = index - row * channel_row_bytes;
            decoded[destination + row * row_bytes + channel_offset + within] = staging[source + index];
        }
    }
}

extern "C" __global__ void pixtreme_exr_b44_scatter_blocks(
    const unsigned short* blocks,
    unsigned char* decoded,
    const long long* sections,
    const int section_count,
    const long long block_count
) {
    const long long sample = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (sample >= block_count * 16LL) return;
    const long long block = sample / 16LL;
    const int local = (int)(sample - block * 16LL);
    int lower = 0;
    int upper = section_count;
    while (lower + 1 < upper) {
        const int middle = lower + (upper - lower) / 2;
        if (sections[(long long)middle * 7LL + 5LL] <= block) lower = middle;
        else upper = middle;
    }
    const long long* descriptor = sections + (long long)lower * 7LL;
    const long long destination = descriptor[0];
    const long long row_bytes = descriptor[1];
    const long long channel_offset = descriptor[2];
    const int width = (int)descriptor[3];
    const int row_count = (int)descriptor[4];
    const long long first_block = descriptor[5];
    const long long section_block_count = descriptor[6];
    const long long within_section = block - first_block;
    if (within_section < 0 || within_section >= section_block_count) return;
    const int block_columns = (width + 3) / 4;
    const int block_row = (int)(within_section / block_columns);
    const int block_column = (int)(within_section - (long long)block_row * block_columns);
    const int row = block_row * 4 + local / 4;
    const int column = block_column * 4 + local % 4;
    if (row >= row_count || column >= width) return;
    const unsigned short bits = blocks[sample];
    const long long byte_offset = destination + (long long)row * row_bytes + channel_offset + (long long)column * 2LL;
    decoded[byte_offset] = (unsigned char)bits;
    decoded[byte_offset + 1] = (unsigned char)(bits >> 8);
}
"""


@lru_cache(maxsize=1)
def _b44_encode_blocks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_encode_blocks")


@lru_cache(maxsize=1)
def _b44_assemble_blocks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_assemble_blocks")


@lru_cache(maxsize=1)
def _b44_scan_block_heads_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_scan_block_heads")


@lru_cache(maxsize=1)
def _b44_decode_blocks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_decode_blocks")


@lru_cache(maxsize=1)
def _b44_reorder_planes_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_reorder_planes")


@lru_cache(maxsize=1)
def _b44_gather_blocks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_gather_blocks")


@lru_cache(maxsize=1)
def _b44_copy_sections_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_copy_sections")


@lru_cache(maxsize=1)
def _b44_scatter_blocks_kernel() -> cp.RawKernel:
    return cp.RawKernel(_EXR_B44_SOURCE, "pixtreme_exr_b44_scatter_blocks")


def _materialize_b44_gpu(prepared: _ExrB44ReadChunks, device_staging: cp.ndarray) -> cp.ndarray:
    decoded = cp.empty(int(prepared.raw_sizes.sum()), dtype=cp.uint8)
    _gather_raw_chunks(
        device_staging,
        decoded,
        prepared.stage_offsets,
        prepared.raw_offsets,
        prepared.raw_sizes,
        prepared.compressed,
    )
    section_count = int(prepared.raw_section_descriptors.shape[0])
    if section_count:
        _b44_copy_sections_kernel()(
            (min(section_count, _EXR_MAX_GRID_X),),
            (_EXR_THREADS_PER_BLOCK,),
            (device_staging, decoded, cp.asarray(prepared.raw_section_descriptors), np.int32(section_count)),
        )
    block_count = int(prepared.block_perceptually_linear.size)
    if block_count:
        device_sections = cp.asarray(prepared.block_section_descriptors)
        section_count = int(device_sections.shape[0])
        block_sizes = cp.zeros(block_count, dtype=cp.int32)
        section_status = cp.zeros(section_count, dtype=cp.int32)
        section_grid = (section_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _b44_scan_block_heads_kernel()(
            (section_grid,),
            (_EXR_THREADS_PER_BLOCK,),
            (
                device_staging,
                device_sections,
                block_sizes,
                section_status,
                np.int32(section_count),
                np.int32(prepared.b44a),
            ),
        )
        host_section_status = section_status.get()
        failed_head_sections = np.flatnonzero(host_section_status)
        if failed_head_sections.size:
            failures = tuple((int(section), int(host_section_status[section])) for section in failed_head_sections)
            raise _gpu_error(
                why="the GPU B44 block-head scan rejected a channel section",
                what=f"section_status={failures!r}",
                how="provide complete three- or fourteen-byte blocks with markers valid for the selected codec",
            )
        block_output_ends = cp.cumsum(block_sizes, dtype=cp.int64)
        block_output_starts = block_output_ends - block_sizes
        section_first_blocks = device_sections[:, 2]
        section_prefixes = block_output_starts[section_first_blocks]
        block_section_indices = cp.repeat(cp.arange(section_count, dtype=cp.int64), device_sections[:, 3])
        block_offsets = (
            block_output_starts - section_prefixes[block_section_indices] + device_sections[block_section_indices, 0]
        )
        section_last_blocks = section_first_blocks + device_sections[:, 3] - 1
        section_consumed_sizes = block_output_ends[section_last_blocks] - section_prefixes
        failed_consumption_sections = cp.flatnonzero(section_consumed_sizes != device_sections[:, 1]).get()
        if failed_consumption_sections.size:
            consumption_failures = tuple(int(section) for section in failed_consumption_sections)
            raise _gpu_error(
                why="the GPU B44 block prefix scan did not consume each channel section exactly",
                what=f"section_indices={consumption_failures!r}",
                how="make every HALF section contain exactly its declared regular block geometry",
            )
        blocks = _decode_b44_blocks_gpu(
            device_staging,
            block_offsets,
            block_sizes,
            perceptually_linear=cp.asarray(prepared.block_perceptually_linear),
        )
        sample_count = block_count * 16
        grid = (sample_count + _EXR_THREADS_PER_BLOCK - 1) // _EXR_THREADS_PER_BLOCK
        _b44_scatter_blocks_kernel()(
            (grid,),
            (_EXR_THREADS_PER_BLOCK,),
            (
                blocks,
                decoded,
                cp.asarray(prepared.block_output_descriptors),
                np.int32(prepared.block_output_descriptors.shape[0]),
                np.int64(block_count),
            ),
        )
    return cast(cp.ndarray, decoded)


def _read_exr_b44_gpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    prepared = _prepare_exr_b44_read_chunks(container)
    device_staging = cp.asarray(prepared.host_staging)
    decoded = _materialize_b44_gpu(prepared, device_staging)
    return _unpack_exr_output(
        container,
        selected,
        decoded,
        prepared.raw_offsets,
        prepared.raw_sizes,
        even_odd_grouped=np.zeros_like(prepared.compressed),
        output_dtype=output_dtype,
    )


def _read_exr_b44_custom_cpu(
    container: _ExrContainer,
    selected: Sequence[_ExrChannel],
    *,
    output_dtype: str,
) -> cp.ndarray:
    prepared = _prepare_exr_b44_read_chunks(container)
    materialized = _materialize_b44_host(prepared)
    if output_dtype == "uint32":
        return _unpack_exr_output(
            container,
            selected,
            cp.asarray(materialized),
            prepared.raw_offsets,
            prepared.raw_sizes,
            even_odd_grouped=np.zeros_like(prepared.compressed),
            output_dtype=output_dtype,
        )
    host_selected = _select_exr_host_pixels(container, selected, materialized, output_dtype=output_dtype)
    return cp.asarray(host_selected)
