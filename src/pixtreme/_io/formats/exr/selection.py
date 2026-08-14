"""Source-fixed EXR codec routing and high-level I/O orchestration."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from pathlib import Path

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._io.common import (
    _EXR_COMPRESSION_TOKENS,
    _EXR_DWA_COMPRESSION_TOKENS,
    _colorspace_chromaticities,
)
from pixtreme._io.formats.exr.codec_b44 import (
    _decode_b44_block_host,
    _encode_b44_chunks_gpu,
    _read_exr_b44_custom_cpu,
    _read_exr_b44_gpu,
)
from pixtreme._io.formats.exr.codec_dwa import (
    _decode_dwa_ac_block_host,
    _decode_dwa_byte_rle_host,
    _decode_dwa_huffman_host,
    _decompress_dwa_zlib_host,
    _dwa_lossy_units,
    _dwa_sample_array,
    _inverse_dwa_dct_host,
    _inverse_dwa_transfer_host,
    _read_exr_dwa_custom_cpu,
    _read_exr_dwa_gpu,
    _write_exr_dwa_gpu,
)
from pixtreme._io.formats.exr.codec_none import (
    _read_exr_none,
)
from pixtreme._io.formats.exr.codec_piz import (
    _encode_piz_chunks_gpu,
    _piz_chunk_decode_control,
    _read_exr_piz_custom_cpu,
    _read_exr_piz_gpu,
)
from pixtreme._io.formats.exr.codec_pxr24 import (
    _decode_pxr24_rows_host,
    _encode_pxr24_rows_gpu,
    _prepare_exr_pxr24_read_chunks,
    _read_exr_pxr24_custom_cpu,
    _read_exr_pxr24_gpu,
)
from pixtreme._io.formats.exr.codec_rle import (
    _encode_rle_packets_gpu,
    _materialize_rle_host,
    _prepare_exr_rle_read_chunks,
    _read_exr_rle_custom_cpu,
    _read_exr_rle_gpu,
)
from pixtreme._io.formats.exr.codec_zip import (
    _decode_deflate_chunks,
    _prepare_exr_read_chunks,
    _read_exr_zip_custom_cpu,
)
from pixtreme._io.formats.exr.container import (
    _EXR_LINES_PER_CHUNK,
    _EXR_PIZ_COMPRESSION,
    _EXR_PXR24_PLANE_COUNTS,
    _build_exr_decoder_view,
    _classify_default_dwa_channels,
    _decode_piz_huffman_host,
    _DwaChannelLayout,
    _ExrChannel,
    _ExrChunk,
    _ExrContainer,
    _ExrGpuError,
    _ExrPart,
    _ExrPizError,
    _gpu_error,
    _parse_dwa_huffman_table,
    _piz_inverse_wavelet_host,
)
from pixtreme._io.formats.exr.packing import (
    _checksum_exr_chunks,
    _encode_deflate_chunks,
    _encode_exr_output_channels,
    _exr_write_header,
    _gather_raw_chunks,
    _pack_exr_gpu,
    _prefix_offsets,
    _restore_even_odd_host,
    _restore_exr_gpu_chunks,
    _restore_exr_host_chunks,
    _restore_predictor_host,
    _select_exr_payloads,
    _transform_and_checksum_chunks,
    _transform_exr_chunks,
    _unpack_exr_chunks,
    _wrap_deflate_chunks,
)
from pixtreme._io.formats.nvimgcodec import (
    _validate_raster_encode_options,
)
from pixtreme._io.models import (
    ImageHeader,
)

# Public EXR routing is a source contract. Runtime capabilities, environment, and measured performance never alter it.
_EXR_ROUTING = {
    ("none", "read"): "native",
    ("none", "write"): "gpu",
    ("piz", "read"): "gpu",
    ("piz", "write"): "gpu",
    ("rle", "read"): "gpu",
    ("rle", "write"): "gpu",
    ("zip", "read"): "custom_cpu",
    ("zip", "write"): "gpu",
    ("zips", "read"): "custom_cpu",
    ("zips", "write"): "gpu",
    ("pxr24", "read"): "custom_cpu",
    ("pxr24", "write"): "gpu",
    ("b44", "read"): "gpu",
    ("b44", "write"): "gpu",
    ("b44a", "read"): "gpu",
    ("b44a", "write"): "gpu",
    ("dwaa", "read"): "gpu",
    ("dwaa", "write"): "gpu",
    ("dwab", "read"): "gpu",
    ("dwab", "write"): "gpu",
}


def _exr_output_dtype(
    header: ImageHeader,
    locations: list[tuple[int, str, str]],
    *,
    unchanged: bool,
) -> str:
    selected_dtypes = [header.parts[part_index].channels[channel] for part_index, channel, _ in locations]
    if unchanged and len(set(selected_dtypes)) != 1:
        raise ValueError(
            _actionable_error(
                why="mixed EXR channel types cannot produce one unchanged Frame dtype",
                what=repr(tuple(selected_dtypes)),
                how="use unchanged=False to promote selected channels to float32",
            )
        )
    return selected_dtypes[0] if unchanged else "float32"


def _read_exr_gpu_pixels(
    container: _ExrContainer,
    header: ImageHeader,
    locations: list[tuple[int, str, str]],
    *,
    unchanged: bool,
) -> cp.ndarray:
    dtype = _exr_output_dtype(header, locations, unchanged=unchanged)
    if any(part_index != 0 for part_index, _, _ in locations):
        raise RuntimeError(
            _actionable_error(
                why="the internal single-part GPU EXR route received a channel from another part",
                what=f"locations={locations!r}",
                how="build a part-local decoder view before entering the GPU codec lane",
            )
        )
    return _read_exr_gpu(container, [channel for _, channel, _ in locations], output_dtype=dtype)


def _read_exr_custom_cpu_pixels(
    container: _ExrContainer,
    header: ImageHeader,
    locations: list[tuple[int, str, str]],
    *,
    unchanged: bool,
) -> cp.ndarray:
    dtype = _exr_output_dtype(header, locations, unchanged=unchanged)
    if any(part_index != 0 for part_index, _, _ in locations):
        raise RuntimeError(
            _actionable_error(
                why="the internal single-part custom CPU EXR route received a channel from another part",
                what=f"locations={locations!r}",
                how="build a part-local decoder view before entering the custom CPU codec lane",
            )
        )
    return _read_exr_custom_cpu(container, [channel for _, channel, _ in locations], output_dtype=dtype)


def _read_exr_native(
    container: _ExrContainer,
    selected_channels: Sequence[str],
    *,
    output_dtype: str,
) -> cp.ndarray:
    channels_by_name = {channel.name: channel for channel in container.parts[0].channels}
    selected = [channels_by_name[name] for name in selected_channels]
    return _read_exr_none(container, selected, output_dtype=output_dtype)


def _decode_exr_view(
    container: _ExrContainer,
    selected_channels: Sequence[str],
    *,
    output_dtype: str,
) -> cp.ndarray:
    backend = _EXR_ROUTING.get((container.compression, "read"))
    if backend == "native":
        return _read_exr_native(container, selected_channels, output_dtype=output_dtype)
    if backend == "gpu":
        return _read_exr_gpu(container, selected_channels, output_dtype=output_dtype)
    if backend == "custom_cpu":
        return _read_exr_custom_cpu(container, selected_channels, output_dtype=output_dtype)
    raise RuntimeError(
        _actionable_error(
            why="the source-fixed EXR read table contains an unknown internal lane",
            what=f"compression={container.compression!r}, backend={backend!r}",
            how="map every supported compression to native, gpu, or custom_cpu",
        )
    )


def _read_exr_tiled_part(
    container: _ExrContainer,
    part_index: int,
    selected_channels: Sequence[str],
    *,
    output_dtype: str,
) -> cp.ndarray:
    part = container.parts[part_index]
    description = part.tile_description
    if description is None:
        raise RuntimeError(
            _actionable_error(
                why="the tiled EXR part has no tile description at materialization time",
                what=f"part={part.name!r}, part_index={part_index}",
                how="provide the validated tileDescription required by a tiledimage part",
            )
        )
    level = next(item for item in part.levels if (item.level_x, item.level_y) == (0, 0))
    cupy_dtype = {"float16": cp.float16, "float32": cp.float32, "uint32": cp.uint32}[output_dtype]
    output = cp.empty((level.height, level.width, len(selected_channels)), dtype=cupy_dtype)
    for chunk in level.chunks:
        if chunk.tile_x is None or chunk.tile_y is None:
            raise RuntimeError(
                _actionable_error(
                    why="a level-zero EXR tile lacks its output tile coordinates",
                    what=f"part={part.name!r}, chunk_offset={chunk.chunk_offset}",
                    how="preserve tile x/y identity from the owning offset-table entry",
                )
            )
        x_start = chunk.tile_x * description.x_size
        y_start = chunk.tile_y * description.y_size
        tile_width = min(description.x_size, level.width - x_start)
        tile_height = min(description.y_size, level.height - y_start)
        view = _build_exr_decoder_view(
            container,
            part_index,
            tile_chunk=chunk,
            tile_size=(tile_width, tile_height),
        )
        tile = _decode_exr_view(view, selected_channels, output_dtype=output_dtype)
        output[y_start : y_start + tile_height, x_start : x_start + tile_width] = tile
    return output


def _sampled_channel_chunk_shape(channel: _ExrChannel, chunk: _ExrChunk) -> tuple[int, int]:
    sampling = channel.sampling
    if sampling is None:
        raise RuntimeError(
            _actionable_error(
                why="a sampled EXR channel has no parsed sampling geometry",
                what=f"channel={channel.name!r}, sampling={(channel.x_sampling, channel.y_sampling)!r}",
                how="derive the channel lattice from its part dataWindow before materializing pixels",
            )
        )
    rows = sum(file_y % channel.y_sampling == 0 for file_y in range(chunk.y, chunk.y + chunk.row_count))
    return sampling.width, rows


def _sampled_stored_payload(
    container: _ExrContainer,
    chunk: _ExrChunk,
    *,
    compression: str | None = None,
) -> bytes:
    payload = container.data[chunk.payload_start : chunk.payload_end]
    if not payload and (container.compression if compression is None else compression) == _EXR_PIZ_COMPRESSION:
        return bytes(chunk.expected_size)
    return payload


def _sampled_zip_payloads(view: _ExrContainer) -> tuple[bytes, ...]:
    prepared = _prepare_exr_read_chunks(view, include_staging=False)
    restored = _restore_exr_host_chunks(prepared)
    return tuple(
        restored[int(offset) : int(offset + size)].tobytes()
        for offset, size in zip(prepared.decoded_offsets, prepared.decoded_sizes, strict=True)
    )


def _sampled_rle_payloads(view: _ExrContainer) -> tuple[bytes, ...]:
    prepared = _prepare_exr_rle_read_chunks(view)
    transformed = _materialize_rle_host(prepared)
    payloads: list[bytes] = []
    for chunk_index, (offset, size) in enumerate(zip(prepared.decoded_offsets, prepared.decoded_sizes, strict=True)):
        start = int(offset)
        end = start + int(size)
        payload = transformed[start:end]
        if prepared.compressed[chunk_index]:
            payload = _restore_even_odd_host(_restore_predictor_host(payload))
        payloads.append(payload.tobytes())
    return tuple(payloads)


def _sampled_pxr24_payloads(view: _ExrContainer) -> tuple[bytes, ...]:
    prepared = _prepare_exr_pxr24_read_chunks(view, materialize_host=True)
    payloads: list[bytes] = []
    for chunk_index, chunk in enumerate(view.chunks):
        if not prepared.compressed[chunk_index]:
            payloads.append(_sampled_stored_payload(view, chunk))
            continue
        descriptor = chunk.phase3
        if descriptor is None or descriptor.codec != "pxr24":
            raise RuntimeError(
                _actionable_error(
                    why="a sampled PXR24 chunk has no validated plane descriptor",
                    what=f"chunk_y={chunk.y}, descriptor={descriptor!r}",
                    how="parse the PXR24 zlib stream and its row-channel plane ownership before decode",
                )
            )
        raw = np.empty(chunk.expected_size, dtype=np.uint8)
        materialized_base = int(prepared.materialized_offsets[chunk_index])
        channels = view.parts[0].channels
        for row in descriptor.channel_rows:
            channel = channels[row.channel_index]
            width = row.raw_span.size // channel.bytes_per_sample
            plane_count = _EXR_PXR24_PLANE_COUNTS[channel.pixel_type]
            start = materialized_base + row.materialized_span.start
            planes = prepared.host_materialized[start : start + row.materialized_span.size].reshape(plane_count, width)
            bits = _decode_pxr24_rows_host(planes[None, ...], channel.pixel_type)[0]
            if channel.pixel_type == 1:
                row_bytes = bits.astype("<u2", copy=False).tobytes()
            else:
                row_bytes = bits.astype("<u4", copy=False).tobytes()
            raw[row.raw_span.start : row.raw_span.end] = np.frombuffer(row_bytes, dtype=np.uint8)
        payloads.append(raw.tobytes())
    return tuple(payloads)


def _sampled_b44_payloads(view: _ExrContainer) -> tuple[bytes, ...]:
    payloads: list[bytes] = []
    channels = view.parts[0].channels
    for chunk in view.chunks:
        if chunk.raw_stored:
            payloads.append(_sampled_stored_payload(view, chunk))
            continue
        descriptor = chunk.phase3
        if descriptor is None or descriptor.codec not in ("b44", "b44a"):
            raise RuntimeError(
                _actionable_error(
                    why="a sampled B44 chunk has no validated channel-section descriptor",
                    what=f"chunk_y={chunk.y}, descriptor={descriptor!r}",
                    how="parse every B44 channel section and HALF block before decode",
                )
            )
        materialized = np.empty(
            sum(section.expected_materialized_size for section in descriptor.channel_sections), dtype=np.uint8
        )
        materialized_cursor = 0
        for channel, section in zip(channels, descriptor.channel_sections, strict=True):
            width, row_count = _sampled_channel_chunk_shape(channel, chunk)
            section_end = materialized_cursor + section.expected_materialized_size
            if channel.pixel_type != 1:
                materialized[materialized_cursor:section_end] = np.frombuffer(
                    view.data[section.payload_span.start : section.payload_span.end],
                    dtype=np.uint8,
                )
            else:
                block_columns = (width + 3) // 4
                block_rows = (row_count + 3) // 4
                plane = np.empty((block_rows * 4, block_columns * 4), dtype=np.uint16)
                payload_cursor = section.payload_span.start
                for block_index in range(block_rows * block_columns):
                    stored_size = 14
                    if descriptor.codec == "b44a" and view.data[payload_cursor + 2] >= 0x34:
                        stored_size = 3
                    block_end = payload_cursor + stored_size
                    bits = _decode_b44_block_host(
                        np.frombuffer(view.data[payload_cursor:block_end], dtype=np.uint8),
                        perceptually_linear=channel.perceptually_linear,
                    ).reshape(4, 4)
                    block_y, block_x = divmod(block_index, block_columns)
                    plane[block_y * 4 : block_y * 4 + 4, block_x * 4 : block_x * 4 + 4] = bits
                    payload_cursor = block_end
                if payload_cursor != section.payload_span.end:
                    raise RuntimeError(
                        _actionable_error(
                            why="the sampled B44 block decoder did not consume its channel section exactly",
                            what=(
                                f"chunk_y={chunk.y}, channel={channel.name!r}, "
                                f"consumed={payload_cursor}, expected_end={section.payload_span.end}"
                            ),
                            how="make the channel section contain exactly its sampled 4-by-4 block grid",
                        )
                    )
                plane_bytes = plane[:row_count, :width].astype("<u2", copy=False).view(np.uint8).reshape(-1)
                materialized[materialized_cursor:section_end] = plane_bytes
            materialized_cursor = section_end
        raw = np.empty(chunk.expected_size, dtype=np.uint8)
        for row in descriptor.channel_rows:
            raw[row.raw_span.start : row.raw_span.end] = materialized[
                row.materialized_span.start : row.materialized_span.end
            ]
        payloads.append(raw.tobytes())
    return tuple(payloads)


def _sampled_piz_payloads(view: _ExrContainer) -> tuple[bytes, ...]:
    payloads: list[bytes] = []
    channels = view.parts[0].channels
    for chunk in view.chunks:
        if chunk.raw_stored:
            payloads.append(_sampled_stored_payload(view, chunk))
            continue
        descriptor = chunk.piz
        if descriptor is None:
            raise RuntimeError(
                _actionable_error(
                    why="a sampled PIZ chunk has no validated bitmap and Huffman descriptor",
                    what=f"chunk_y={chunk.y}",
                    how="parse the PIZ bitmap, Huffman stream, and sampled channel planes before decode",
                )
            )
        reverse_lut, maximum, huffman_stream, table = _piz_chunk_decode_control(view, descriptor)
        words = _decode_piz_huffman_host(
            huffman_stream,
            table,
            expected_count=descriptor.expected_output_word_count,
        )
        channel_rows: dict[str, np.ndarray] = {}
        for plane in descriptor.channel_planes:
            channel = channels[plane.channel_index]
            width, row_count = _sampled_channel_chunk_shape(channel, chunk)
            plane_words = words[plane.word_offset : plane.word_offset + plane.word_count]
            for word_slice in range(plane.word_slice_count):
                _piz_inverse_wavelet_host(
                    plane_words,
                    nx=width,
                    ny=row_count,
                    word_stride=plane.word_slice_count,
                    word_slice=word_slice,
                    max_value=maximum,
                )
            invalid = np.flatnonzero(plane_words > maximum)
            if invalid.size:
                first = int(invalid[0])
                raise RuntimeError(
                    _actionable_error(
                        why="the sampled PIZ inverse wavelet produced a LUT index outside the compact alphabet",
                        what=(
                            f"chunk_y={chunk.y}, channel={channel.name!r}, word={first}, "
                            f"value={int(plane_words[first])}, maxValue={maximum}"
                        ),
                        how="verify the Huffman words, wavelet geometry, and sampled channel ownership",
                    )
                )
            channel_rows[channel.name] = (
                reverse_lut[plane_words]
                .astype("<u2", copy=False)
                .view(np.uint8)
                .reshape(row_count, width * channel.bytes_per_sample)
            )
        raw = bytearray()
        channel_row_indices = {channel.name: 0 for channel in channels}
        for file_y in range(chunk.y, chunk.y + chunk.row_count):
            for channel in channels:
                if file_y % channel.y_sampling:
                    continue
                row_index = channel_row_indices[channel.name]
                raw.extend(channel_rows[channel.name][row_index].tobytes())
                channel_row_indices[channel.name] = row_index + 1
        if len(raw) != chunk.expected_size:
            raise RuntimeError(
                _actionable_error(
                    why="the sampled PIZ materializer produced the wrong raw chunk size",
                    what=f"chunk_y={chunk.y}, produced={len(raw)}, expected={chunk.expected_size}",
                    how="scatter every inverse-wavelet channel sample onto its file-coordinate lattice",
                )
            )
        payloads.append(bytes(raw))
    return tuple(payloads)


def _sampled_dwa_lossy_planes(
    channels: Sequence[_ExrChannel],
    chunk: _ExrChunk,
    layout: _DwaChannelLayout,
    ac_symbols: np.ndarray,
    dc_values: np.ndarray,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    ac_offset = 0
    dc_offset = 0
    for unit in _dwa_lossy_units(channels, layout):
        width, row_count = _sampled_channel_chunk_shape(unit[0], chunk)
        if any(_sampled_channel_chunk_shape(channel, chunk) != (width, row_count) for channel in unit[1:]):
            raise RuntimeError(
                _actionable_error(
                    why="a DWA color-transform unit spans different sampled channel shapes",
                    what=f"chunk_y={chunk.y}, channels={tuple(channel.name for channel in unit)!r}",
                    how="group DWA color components only when their x/y sampling factors match",
                )
            )
        block_columns = (width + 7) // 8
        block_rows = (row_count + 7) // 8
        block_count = block_columns * block_rows
        component_blocks = np.empty((len(unit), block_count, 8, 8), dtype=np.float32)
        for block_index in range(block_count):
            for component in range(len(unit)):
                dc_index = dc_offset + component * block_count + block_index
                if dc_index >= dc_values.size:
                    raise RuntimeError(
                        _actionable_error(
                            why="the sampled DWA DC stream ends before its channel block ownership",
                            what=f"chunk_y={chunk.y}, dc_index={dc_index}, available={dc_values.size}",
                            how="provide one DC coefficient for every sampled lossy channel block",
                        )
                    )
                coefficients, ac_offset = _decode_dwa_ac_block_host(
                    ac_symbols,
                    ac_offset,
                    np.uint16(dc_values[dc_index]),
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
            plane = np.empty((block_rows * 8, block_columns * 8), dtype=np.float16)
            for block_index in range(block_count):
                block_y, block_x = divmod(block_index, block_columns)
                plane[block_y * 8 : block_y * 8 + 8, block_x * 8 : block_x * 8 + 8] = reconstructed[block_index]
            output[channel.name] = plane[:row_count, :width]
    if ac_offset != ac_symbols.size or dc_offset != dc_values.size:
        raise RuntimeError(
            _actionable_error(
                why="the sampled DWA coefficient streams do not end at their channel block boundaries",
                what=(
                    f"chunk_y={chunk.y}, ac_consumed={ac_offset}, ac_declared={ac_symbols.size}, "
                    f"dc_consumed={dc_offset}, dc_declared={dc_values.size}"
                ),
                how="match the coefficient counts to every sampled lossy channel block",
            )
        )
    return output


def _sampled_dwa_payloads(view: _ExrContainer) -> tuple[bytes, ...]:
    payloads: list[bytes] = []
    channels = view.parts[0].channels
    source_dtypes = {0: "<u4", 1: "<f2", 2: "<f4"}
    for chunk in view.chunks:
        if chunk.raw_stored:
            payloads.append(_sampled_stored_payload(view, chunk))
            continue
        descriptor = chunk.dwa
        if descriptor is None or descriptor.leader is None:
            raise RuntimeError(
                _actionable_error(
                    why="a sampled DWA chunk has no validated leader and stream descriptor",
                    what=f"chunk_y={chunk.y}, descriptor={descriptor!r}",
                    how="parse the DWA leader, channel rules, and compressed substreams before decode",
                )
            )
        leader = descriptor.leader
        layout = descriptor.channel_layout or _classify_default_dwa_channels(channels)
        layout_by_name = {item.name: item for item in layout.channels}
        decoded: dict[str, np.ndarray] = {}

        unknown = b""
        if leader.unknown_compressed_size:
            unknown = _decompress_dwa_zlib_host(
                view.data[descriptor.unknown_span.start : descriptor.unknown_span.end],
                expected_size=leader.unknown_uncompressed_size,
                stream_name="UNKNOWN",
                chunk_y=chunk.y,
            )
        unknown_offset = 0
        for channel in channels:
            if layout_by_name[channel.name].scheme != "unknown":
                continue
            width, row_count = _sampled_channel_chunk_shape(channel, chunk)
            sample_count = width * row_count
            end = unknown_offset + sample_count * channel.bytes_per_sample
            decoded[channel.name] = _dwa_sample_array(
                unknown[unknown_offset:end],
                channel,
                sample_count=sample_count,
            ).reshape(row_count, width)
            unknown_offset = end
        if unknown_offset != len(unknown):
            raise RuntimeError(
                _actionable_error(
                    why="the sampled DWA UNKNOWN stream has bytes outside its channels",
                    what=f"chunk_y={chunk.y}, consumed={unknown_offset}, decoded={len(unknown)}",
                    how="make the UNKNOWN size equal the sampled channel planes",
                )
            )

        rle_raw = b""
        if leader.rle_compressed_size:
            rle_encoded = _decompress_dwa_zlib_host(
                view.data[descriptor.rle_span.start : descriptor.rle_span.end],
                expected_size=leader.rle_uncompressed_size,
                stream_name="RLE",
                chunk_y=chunk.y,
            )
            rle_raw = _decode_dwa_byte_rle_host(
                rle_encoded,
                expected_size=leader.rle_raw_size,
                chunk_y=chunk.y,
            )
        rle_offset = 0
        for channel in channels:
            if layout_by_name[channel.name].scheme != "rle":
                continue
            width, row_count = _sampled_channel_chunk_shape(channel, chunk)
            sample_count = width * row_count
            end = rle_offset + sample_count * channel.bytes_per_sample
            planes = np.frombuffer(rle_raw[rle_offset:end], dtype=np.uint8).reshape(
                channel.bytes_per_sample,
                sample_count,
            )
            interleaved = np.ascontiguousarray(planes.T).reshape(-1).tobytes()
            decoded[channel.name] = _dwa_sample_array(
                interleaved,
                channel,
                sample_count=sample_count,
            ).reshape(row_count, width)
            rle_offset = end
        if rle_offset != len(rle_raw):
            raise RuntimeError(
                _actionable_error(
                    why="the sampled DWA RLE stream has bytes outside its channels",
                    what=f"chunk_y={chunk.y}, consumed={rle_offset}, decoded={len(rle_raw)}",
                    how="make the RLE raw size equal the sampled channel byte planes",
                )
            )

        ac_symbols = np.empty(0, dtype=np.uint16)
        if leader.ac_element_count:
            table = descriptor.huffman
            if table is None:
                raise RuntimeError(
                    _actionable_error(
                        why="a sampled DWA AC stream has no parsed Huffman table",
                        what=f"chunk_y={chunk.y}, ac_elements={leader.ac_element_count}",
                        how="provide a complete canonical Huffman table for the AC stream",
                    )
                )
            if not table.codes:
                table = _parse_dwa_huffman_table(
                    view.data[descriptor.ac_span.start : descriptor.ac_span.end],
                    base_offset=descriptor.ac_span.start,
                )
            ac_symbols = _decode_dwa_huffman_host(view.data, table, expected_count=leader.ac_element_count)
        dc_values = np.empty(0, dtype=np.uint16)
        if leader.dc_element_count:
            transformed_dc = _decompress_dwa_zlib_host(
                view.data[descriptor.dc_span.start : descriptor.dc_span.end],
                expected_size=leader.dc_element_count * 2,
                stream_name="DC",
                chunk_y=chunk.y,
            )
            restored_dc = _restore_even_odd_host(_restore_predictor_host(np.frombuffer(transformed_dc, dtype=np.uint8)))
            dc_values = restored_dc.view("<u2")
        decoded.update(
            _sampled_dwa_lossy_planes(
                channels,
                chunk,
                layout,
                ac_symbols,
                dc_values,
            )
        )

        raw = bytearray()
        channel_row_indices = {channel.name: 0 for channel in channels}
        for file_y in range(chunk.y, chunk.y + chunk.row_count):
            for channel in channels:
                if file_y % channel.y_sampling:
                    continue
                row_index = channel_row_indices[channel.name]
                raw.extend(
                    np.asarray(decoded[channel.name][row_index], dtype=source_dtypes[channel.pixel_type]).tobytes()
                )
                channel_row_indices[channel.name] = row_index + 1
        if len(raw) != chunk.expected_size:
            raise RuntimeError(
                _actionable_error(
                    why="the sampled DWA materializer produced the wrong raw chunk size",
                    what=f"chunk_y={chunk.y}, produced={len(raw)}, expected={chunk.expected_size}",
                    how="scatter every decoded DWA channel sample onto its file-coordinate lattice",
                )
            )
        payloads.append(bytes(raw))
    return tuple(payloads)


def _sampled_raw_payloads(container: _ExrContainer, part: _ExrPart) -> tuple[bytes, ...]:
    if all(chunk.raw_stored for chunk in part.chunks):
        return tuple(_sampled_stored_payload(container, chunk, compression=part.compression) for chunk in part.chunks)
    view = _build_exr_decoder_view(container, part.index, preserve_sampling=True)
    if part.compression in ("zip", "zips"):
        return _sampled_zip_payloads(view)
    if part.compression == "rle":
        return _sampled_rle_payloads(view)
    if part.compression == "pxr24":
        return _sampled_pxr24_payloads(view)
    if part.compression in ("b44", "b44a"):
        return _sampled_b44_payloads(view)
    if part.compression == "piz":
        return _sampled_piz_payloads(view)
    if part.compression in ("dwaa", "dwab"):
        return _sampled_dwa_payloads(view)
    raise RuntimeError(
        _actionable_error(
            why="the sampled EXR part names an unknown compression codec",
            what=f"part={part.name!r}, compression={part.compression!r}",
            how="use one of the ten supported EXR compression modes",
        )
    )


def _read_exr_sampled_part(
    container: _ExrContainer,
    part_index: int,
    selected_channels: Sequence[str],
    *,
    output_dtype: str,
) -> dict[str, cp.ndarray]:
    part = container.parts[part_index]
    selected = set(selected_channels)
    host_dtype = {"float16": np.float16, "float32": np.float32, "uint32": np.uint32}[output_dtype]
    host_outputs = {
        channel.name: np.empty(channel.sampling.shape, dtype=host_dtype)
        for channel in part.channels
        if channel.name in selected and channel.sampling is not None
    }
    payloads = _sampled_raw_payloads(container, part)
    source_dtype_names = {0: "<u4", 1: "<f2", 2: "<f4"}
    for chunk, payload in zip(part.chunks, payloads, strict=True):
        cursor = 0
        for file_y in range(chunk.y, chunk.y + chunk.row_count):
            for channel in part.channels:
                sampling = channel.sampling
                if sampling is None or file_y % channel.y_sampling:
                    continue
                byte_count = sampling.width * channel.bytes_per_sample
                end = cursor + byte_count
                if end > len(payload):
                    raise RuntimeError(
                        _actionable_error(
                            why="a sampled EXR chunk ends inside a channel row",
                            what=(
                                f"part={part.name!r}, chunk_y={chunk.y}, channel={channel.name!r}, "
                                f"row_y={file_y}, required_end={end}, payload_size={len(payload)}"
                            ),
                            how="store every sample on the channel's file-coordinate sampling lattice",
                        )
                    )
                if channel.name in host_outputs:
                    output_row = (file_y - sampling.y_start) // channel.y_sampling
                    values = np.frombuffer(payload[cursor:end], dtype=source_dtype_names[channel.pixel_type])
                    host_outputs[channel.name][output_row] = values.astype(host_dtype, copy=False)
                cursor = end
        if cursor != len(payload):
            raise RuntimeError(
                _actionable_error(
                    why="a sampled EXR chunk has bytes outside its channel lattice rows",
                    what=f"part={part.name!r}, chunk_y={chunk.y}, consumed={cursor}, payload_size={len(payload)}",
                    how="derive the raw chunk size from each channel's x/y sampling geometry",
                )
            )
    return {name: cp.asarray(np.ascontiguousarray(values)) for name, values in host_outputs.items()}


def _read_exr_container_pixels(
    container: _ExrContainer,
    header: ImageHeader,
    locations: list[tuple[int, str, str]],
    *,
    unchanged: bool,
) -> cp.ndarray:
    output_dtype = _exr_output_dtype(header, locations, unchanged=unchanged)
    for part_index, channel_name, _ in locations:
        part = container.parts[part_index]
        if part.deep:
            raise ValueError(
                _actionable_error(
                    why="the selected EXR channel belongs to an unsupported deep part",
                    what=f"part={part.name!r}, part_index={part_index}, channel={channel_name!r}",
                    how="select channels from a flat scanlineimage or tiledimage part",
                )
            )

    requested_by_part: dict[int, list[str]] = {}
    for part_index, channel_name, _ in locations:
        channels = requested_by_part.setdefault(part_index, [])
        if channel_name not in channels:
            channels.append(channel_name)

    planes: dict[tuple[int, str], cp.ndarray] = {}
    for part_index, selected_channels in requested_by_part.items():
        part = container.parts[part_index]
        if any(channel.x_sampling != 1 or channel.y_sampling != 1 for channel in part.channels):
            sampled = _read_exr_sampled_part(
                container,
                part_index,
                selected_channels,
                output_dtype=output_dtype,
            )
            for channel_name in selected_channels:
                planes[(part_index, channel_name)] = sampled[channel_name]
            continue
        if part.levels:
            decoded = _read_exr_tiled_part(
                container,
                part_index,
                selected_channels,
                output_dtype=output_dtype,
            )
        else:
            view = _build_exr_decoder_view(container, part_index)
            decoded = _decode_exr_view(view, selected_channels, output_dtype=output_dtype)
        for channel_index, channel_name in enumerate(selected_channels):
            planes[(part_index, channel_name)] = decoded[..., channel_index]

    selected_planes = [planes[(part_index, channel_name)] for part_index, channel_name, _ in locations]
    shape_records = tuple(
        (location[2], tuple(int(value) for value in plane.shape))
        for plane, location in zip(selected_planes, locations, strict=True)
    )
    if len({shape for _, shape in shape_records}) != 1:
        raise ValueError(
            _actionable_error(
                why="selected EXR channel dimensions do not match",
                what=repr(shape_records),
                how="select channels with identical sampled or dataWindow dimensions",
            )
        )
    return cp.stack(selected_planes, axis=2)


def _read_exr_pixels(
    _path: Path,
    container: _ExrContainer,
    header: ImageHeader,
    locations: list[tuple[int, str, str]],
    *,
    unchanged: bool,
) -> cp.ndarray:
    return _read_exr_container_pixels(container, header, locations, unchanged=unchanged)


def _validate_exr_write_options(
    *,
    quality: int | None,
    compression: str | None,
    compression_level: int | None,
    lossless: bool | None,
    dwa_level: float | None,
) -> tuple[str, float | None]:
    _validate_raster_encode_options(
        "EXR",
        quality=quality,
        compression=None,
        compression_level=compression_level,
        lossless=lossless,
    )
    if compression is not None and (type(compression) is not str or compression not in _EXR_COMPRESSION_TOKENS):
        raise ValueError(
            _actionable_error(
                why="EXR compression is not a supported token",
                what=repr(compression),
                how=f"use one of {_EXR_COMPRESSION_TOKENS!r}",
            )
        )
    compression_token = "zip" if compression is None else compression
    if compression_token not in _EXR_DWA_COMPRESSION_TOKENS:
        if dwa_level is not None:
            raise ValueError(
                _actionable_error(
                    why="dwa_level is supported only for DWAA and DWAB output",
                    what=f"dwa_level={dwa_level!r}, compression={compression_token!r}",
                    how="omit dwa_level or use compression='dwaa' or compression='dwab'",
                )
            )
        return compression_token, None
    candidate_dwa_level = 45.0 if dwa_level is None else dwa_level
    resolved_dwa_level: float | None = None
    if type(candidate_dwa_level) is float:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            resolved_dwa_level = float(np.float32(candidate_dwa_level))
    if resolved_dwa_level is None or not math.isfinite(resolved_dwa_level) or resolved_dwa_level <= 0.0:
        raise ValueError(
            _actionable_error(
                why="dwa_level must remain positive and finite as an OpenEXR float",
                what=repr(dwa_level),
                how="pass a positive finite float such as dwa_level=45.0",
            )
        )
    return compression_token, resolved_dwa_level


def _validate_exr_write_channels(frame: Frame) -> None:
    if len(set(frame.channels)) != len(frame.channels):
        duplicates = tuple(label for label in dict.fromkeys(frame.channels) if frame.channels.count(label) > 1)
        raise ValueError(
            _actionable_error(
                why="EXR output channel labels must be unique",
                what=f"channels={frame.channels!r}, duplicates={duplicates!r}",
                how="provide one unique label for each EXR output channel",
            )
        )


def _write_exr_with_backend(
    path: Path,
    frame: Frame,
    *,
    compression: str,
    dwa_level: float | None,
    backend: str,
) -> None:
    _validate_exr_write_channels(frame)
    if backend == "gpu":
        if compression in _EXR_DWA_COMPRESSION_TOKENS:
            if dwa_level is None:
                raise RuntimeError(
                    _actionable_error(
                        why="the hybrid GPU DWA writer received no resolved compression level",
                        what=f"compression={compression!r}",
                        how="validate DWA output options before selecting the internal GPU backend",
                    )
                )
            _write_exr_dwa_gpu(
                path,
                frame.data,
                frame.channels,
                compression=compression,
                dwa_level=dwa_level,
                chromaticities=_colorspace_chromaticities(frame.colorspace),
                aces_image_container=frame.colorspace == "ACES2065-1",
            )
            return
        if compression in ("none", "rle", "zip", "zips", "piz", "pxr24", "b44", "b44a"):
            _write_exr_gpu(
                path,
                frame.data,
                frame.channels,
                compression=compression,
                chromaticities=_colorspace_chromaticities(frame.colorspace),
                aces_image_container=frame.colorspace == "ACES2065-1",
            )
            return
        raise RuntimeError(
            _actionable_error(
                why="the internal GPU EXR writer received an unsupported compression",
                what=f"compression={compression!r}",
                how="force only a GPU-capable scanline compression from an internal benchmark harness",
            )
        )
    raise RuntimeError(
        _actionable_error(
            why="the internal EXR write route received an unknown backend",
            what=f"backend={backend!r}, compression={compression!r}",
            how="force the gpu lane only from an internal correctness or benchmark harness",
        )
    )


def _write_exr(path: Path, frame: Frame, *, compression: str, dwa_level: float | None) -> None:
    backend = _EXR_ROUTING[(compression, "write")]
    _write_exr_with_backend(
        path,
        frame,
        compression=compression,
        dwa_level=dwa_level,
        backend=backend,
    )


def _read_exr_gpu(
    container: _ExrContainer,
    selected_channels: Sequence[str],
    *,
    output_dtype: str,
) -> cp.ndarray:
    try:
        if not (
            container.gpu_eligible or container.dwa_eligible or container.phase3_eligible or container.piz_eligible
        ):
            raise _gpu_error(
                why="the internal GPU EXR reader received a container outside the eligible scanline codecs",
                what=(
                    f"multipart={container.multipart}, tiled={container.tiled}, deep={container.deep}, "
                    f"compression={container.compression!r}"
                ),
                how="build a supported flat scanline decoder view before entering the GPU codec lane",
            )
        part = container.parts[0]
        channels_by_name = {channel.name: channel for channel in part.channels}
        selected = [channels_by_name[name] for name in selected_channels]
        if container.piz_eligible:
            return _read_exr_piz_gpu(container, selected, output_dtype=output_dtype)
        if container.dwa_eligible:
            return _read_exr_dwa_gpu(container, selected, output_dtype=output_dtype)
        if container.phase3_eligible:
            if container.compression == "rle":
                return _read_exr_rle_gpu(container, selected, output_dtype=output_dtype)
            if container.compression == "pxr24":
                return _read_exr_pxr24_gpu(container, selected, output_dtype=output_dtype)
            return _read_exr_b44_gpu(container, selected, output_dtype=output_dtype)
        prepared = _prepare_exr_read_chunks(container)
        host_staging = prepared.host_staging
        device_staging = cp.asarray(host_staging)
        if np.any(prepared.compressed):
            decoded = cp.empty(int(prepared.decoded_sizes.sum()), dtype=cp.uint8)
            compressed_indices = np.flatnonzero(prepared.compressed)
            deflate_inputs = list(
                zip(
                    prepared.stage_offsets[compressed_indices].tolist(),
                    prepared.stage_sizes[compressed_indices].tolist(),
                    strict=True,
                )
            )
            deflate_outputs = list(
                zip(
                    prepared.decoded_offsets[compressed_indices].tolist(),
                    prepared.decoded_sizes[compressed_indices].tolist(),
                    strict=True,
                )
            )
            _decode_deflate_chunks(device_staging, deflate_inputs, decoded, deflate_outputs)
            _gather_raw_chunks(
                device_staging,
                decoded,
                prepared.stage_offsets,
                prepared.decoded_offsets,
                prepared.decoded_sizes,
                prepared.compressed,
            )
        else:
            decoded = device_staging
        failed_chunks = _restore_exr_gpu_chunks(
            decoded,
            prepared.decoded_offsets,
            prepared.decoded_sizes,
            prepared.compressed,
            prepared.expected_adler,
        )
        if failed_chunks.size:
            failed_y = tuple(container.chunks[int(index)].y for index in failed_chunks)
            raise _gpu_error(
                why="the GPU Adler-32 result does not match the EXR zlib trailer",
                what=f"chunk_y={failed_y!r}",
                how="verify that the zlib payload and big-endian Adler-32 trailer are complete and unmodified",
            )
        return _unpack_exr_chunks(
            container,
            selected,
            decoded,
            prepared.decoded_offsets,
            prepared.decoded_sizes,
            prepared.compressed,
            output_dtype=output_dtype,
        )
    except (_ExrGpuError, _ExrPizError):
        raise
    except Exception as error:
        if container.piz_eligible:
            raise RuntimeError(
                _actionable_error(
                    why=f"CUDA could not decode the eligible PIZ chunks: {error}",
                    what=f"chunks={len(container.chunks)}, dataWindow={container.data_window!r}",
                    how="verify CUDA availability and the PIZ bitmap, Huffman, wavelet, LUT, and ownership bounds",
                )
            ) from error
        raise RuntimeError(
            _actionable_error(
                why=f"nvCOMP/CUDA could not decode the eligible EXR chunks: {error}",
                what=(
                    f"compression={container.compression!r}, chunks={len(container.chunks)}, "
                    f"dataWindow={container.data_window!r}"
                ),
                how="verify NVIDIA GPU availability, CUDA compatibility, nvCOMP, and the EXR payload integrity",
            )
        ) from error


def _read_exr_custom_cpu(
    container: _ExrContainer,
    selected_channels: Sequence[str],
    *,
    output_dtype: str,
) -> cp.ndarray:
    try:
        if not (
            container.gpu_eligible or container.dwa_eligible or container.phase3_eligible or container.piz_eligible
        ):
            raise _gpu_error(
                why="the internal custom CPU EXR reader received a container outside the eligible scanline codecs",
                what=(
                    f"multipart={container.multipart}, tiled={container.tiled}, deep={container.deep}, "
                    f"compression={container.compression!r}"
                ),
                how="build a supported flat scanline decoder view before entering the custom CPU codec lane",
            )
        part = container.parts[0]
        channels_by_name = {channel.name: channel for channel in part.channels}
        selected = [channels_by_name[name] for name in selected_channels]
        if container.compression in ("zip", "zips"):
            return _read_exr_zip_custom_cpu(container, selected, output_dtype=output_dtype)
        if container.piz_eligible:
            return _read_exr_piz_custom_cpu(container, selected, output_dtype=output_dtype)
        if container.dwa_eligible:
            return _read_exr_dwa_custom_cpu(container, selected, output_dtype=output_dtype)
        if container.phase3_eligible:
            if container.compression == "rle":
                return _read_exr_rle_custom_cpu(container, selected, output_dtype=output_dtype)
            if container.compression == "pxr24":
                return _read_exr_pxr24_custom_cpu(container, selected, output_dtype=output_dtype)
            return _read_exr_b44_custom_cpu(container, selected, output_dtype=output_dtype)
        prepared = _prepare_exr_read_chunks(container)
        restored = _restore_exr_host_chunks(prepared)
        decoded = cp.asarray(restored)
        raw_flags = np.zeros_like(prepared.compressed)
        return _unpack_exr_chunks(
            container,
            selected,
            decoded,
            prepared.decoded_offsets,
            prepared.decoded_sizes,
            raw_flags,
            output_dtype=output_dtype,
        )
    except (_ExrGpuError, _ExrPizError):
        raise
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the custom CPU lane could not decode the eligible EXR chunks: {error}",
                what=(
                    f"compression={container.compression!r}, chunks={len(container.chunks)}, "
                    f"dataWindow={container.data_window!r}"
                ),
                how="verify the EXR payload integrity, host memory availability, and the GPU channel unpack boundary",
            )
        ) from error


def _write_exr_gpu(
    path: Path,
    data: cp.ndarray,
    channels: Sequence[str],
    *,
    compression: str,
    chromaticities: Sequence[float],
    aces_image_container: bool,
) -> None:
    encoded_file_channels = _encode_exr_output_channels(tuple(sorted(channels)))
    try:
        raw, file_channels, pixel_type = _pack_exr_gpu(
            data,
            channels,
            row_prefix_bytes=8 if compression == "none" else 0,
        )
        validated_file_channels = tuple(channel for channel, _ in encoded_file_channels)
        if validated_file_channels != file_channels:
            raise _gpu_error(
                why="the validated EXR channel order diverged from the GPU packing order",
                what=f"validated={validated_file_channels!r}, packed={file_channels!r}",
                how="report this internal GPU EXR writer defect",
            )
        height, width, channel_count = (int(value) for value in data.shape)
        bytes_per_sample = 2 if pixel_type == 1 else 4
        row_bytes = width * channel_count * bytes_per_sample
        lines_per_chunk = _EXR_LINES_PER_CHUNK[compression]
        row_starts = tuple(range(0, height, lines_per_chunk))
        row_counts = tuple(min(lines_per_chunk, height - row_start) for row_start in row_starts)
        raw_sizes = tuple(row_count * row_bytes for row_count in row_counts)
        raw_offsets = _prefix_offsets(raw_sizes)
        if compression == "none":
            payload_blob = raw
            payload_sizes = raw_sizes
        elif compression == _EXR_PIZ_COMPRESSION:
            payload_blob, payload_sizes = _encode_piz_chunks_gpu(
                raw,
                raw_offsets,
                raw_sizes,
                row_counts=row_counts,
                width=width,
                channel_count=channel_count,
                pixel_type=pixel_type,
            )
        elif compression == "rle":
            transformed = _transform_exr_chunks(raw, raw_offsets, raw_sizes)
            encoded, encoded_offsets, encoded_sizes = _encode_rle_packets_gpu(
                transformed,
                raw_offsets,
                raw_sizes,
            )
            payload_blob, payload_sizes = _select_exr_payloads(
                raw,
                raw_offsets,
                raw_sizes,
                encoded,
                encoded_offsets,
                encoded_sizes,
            )
        elif compression == "pxr24":
            plane_count = _EXR_PXR24_PLANE_COUNTS[pixel_type]
            if pixel_type == 1:
                row_channel_bits = raw.view(cp.uint16).reshape(height * channel_count, width).astype(cp.uint32)
            else:
                row_channel_bits = raw.view(cp.uint32).reshape(height * channel_count, width)
            transformed = _encode_pxr24_rows_gpu(row_channel_bits, pixel_type).reshape(-1)
            materialized_sizes = tuple(
                min(lines_per_chunk, height - row_start) * width * channel_count * plane_count
                for row_start in row_starts
            )
            materialized_offsets = _prefix_offsets(materialized_sizes)
            adler = _checksum_exr_chunks(transformed, materialized_offsets, materialized_sizes)
            compressed, compressed_offsets, compressed_sizes = _encode_deflate_chunks(
                transformed,
                tuple(zip(materialized_offsets, materialized_sizes, strict=True)),
            )
            wrapped, wrapped_offsets, wrapped_sizes = _wrap_deflate_chunks(
                compressed,
                compressed_offsets,
                compressed_sizes,
                adler,
            )
            payload_blob, payload_sizes = _select_exr_payloads(
                raw,
                raw_offsets,
                raw_sizes,
                wrapped,
                wrapped_offsets,
                wrapped_sizes,
            )
        elif compression in ("b44", "b44a"):
            encoded, encoded_offsets, encoded_sizes = _encode_b44_chunks_gpu(
                raw,
                raw_offsets,
                row_counts,
                width=width,
                channel_count=channel_count,
                pixel_type=pixel_type,
                codec=compression,
            )
            payload_blob, payload_sizes = _select_exr_payloads(
                raw,
                raw_offsets,
                raw_sizes,
                encoded,
                encoded_offsets,
                encoded_sizes,
            )
        else:
            transformed, adler = _transform_and_checksum_chunks(raw, raw_offsets, raw_sizes)
            compressed, compressed_offsets, compressed_sizes = _encode_deflate_chunks(
                transformed, tuple(zip(raw_offsets, raw_sizes, strict=True))
            )
            wrapped, wrapped_offsets, wrapped_sizes = _wrap_deflate_chunks(
                compressed, compressed_offsets, compressed_sizes, adler
            )
            payload_blob, payload_sizes = _select_exr_payloads(
                raw, raw_offsets, raw_sizes, wrapped, wrapped_offsets, wrapped_sizes
            )
        payload_memory = cp.cuda.alloc_pinned_memory(int(payload_blob.nbytes))
        payload_host = np.frombuffer(payload_memory, dtype=np.uint8, count=int(payload_blob.size))
        payload_blob.get(out=payload_host)
        payload_view = memoryview(payload_host)
        header = _exr_write_header(
            width=width,
            height=height,
            encoded_channels=encoded_file_channels,
            pixel_type=pixel_type,
            compression=compression,
            chromaticities=chromaticities,
            aces_image_container=aces_image_container,
        )
    except _ExrGpuError:
        raise
    except Exception as error:
        encoder = "PIZ/CUDA" if compression == _EXR_PIZ_COMPRESSION else "nvCOMP/CUDA"
        raise RuntimeError(
            _actionable_error(
                why=f"{encoder} could not encode the eligible EXR chunks: {error}",
                what=f"compression={compression!r}, shape={data.shape!r}, channels={tuple(channels)!r}",
                how=(
                    "verify NVIDIA GPU availability, CUDA compatibility, and the output Frame layout"
                    if compression == _EXR_PIZ_COMPRESSION
                    else "verify NVIDIA GPU availability, CUDA compatibility, nvCOMP, and the output Frame layout"
                ),
            )
        ) from error

    first_chunk_offset = len(header) + len(payload_sizes) * 8
    chunk_offsets: list[int] = []
    cursor = first_chunk_offset
    for size in payload_sizes:
        chunk_offsets.append(cursor)
        cursor += 8 + size
    offset_table = b"".join(struct.pack("<Q", offset) for offset in chunk_offsets)
    try:
        with path.open("wb") as stream:
            if compression == "zips":
                file_parts: list[bytes | memoryview] = [header, offset_table]
                payload_offset = 0
                for row_start, size in zip(row_starts, payload_sizes, strict=True):
                    file_parts.append(struct.pack("<ii", row_start, size))
                    file_parts.append(payload_view[payload_offset : payload_offset + size])
                    payload_offset += size
                stream.write(b"".join(file_parts))
            else:
                stream.write(header)
                stream.write(offset_table)
            if compression == "none":
                stream.write(payload_view)
            elif compression != "zips":
                payload_offset = 0
                for row_start, size in zip(row_starts, payload_sizes, strict=True):
                    stream.write(struct.pack("<ii", row_start, size))
                    stream.write(payload_view[payload_offset : payload_offset + size])
                    payload_offset += size
    except OSError as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the GPU-generated EXR file could not be written: {error}",
                what=str(path),
                how="provide a writable output path whose parent directory already exists",
            )
        ) from error
