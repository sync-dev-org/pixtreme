"""Per-channel native-dtype OpenEXR write orchestration."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._io.common import _colorspace_chromaticities
from pixtreme._io.formats.exr.codec_b44 import _encode_b44_chunks_gpu
from pixtreme._io.formats.exr.codec_dwa import (
    _dwa_mixed_channel_rules_bytes,
    _encode_dwa_channel_chunks_gpu,
)
from pixtreme._io.formats.exr.codec_piz import _encode_piz_chunks_gpu
from pixtreme._io.formats.exr.codec_pxr24 import _encode_pxr24_rows_gpu
from pixtreme._io.formats.exr.codec_rle import _encode_rle_packets_gpu
from pixtreme._io.formats.exr.container import (
    _EXR_DTYPE_INFO,
    _EXR_LINES_PER_CHUNK,
    _classify_default_dwa_channels,
    _ExrChannel,
    _ExrGpuError,
    _ExrPizError,
    _gpu_error,
)
from pixtreme._io.formats.exr.packing import (
    _checksum_exr_chunks,
    _encode_deflate_chunks,
    _encode_exr_output_channels,
    _exr_write_header,
    _prefix_offsets,
    _select_exr_payloads,
    _transform_and_checksum_chunks,
    _transform_exr_chunks,
    _wrap_deflate_chunks,
)

_NATIVE_PIXEL_TYPES = {"uint32": 0, "float16": 1, "float32": 2}


@dataclass(frozen=True)
class _ExrWriteChannel:
    descriptor: _ExrChannel
    plane: cp.ndarray


class _ExrFileWriteError(RuntimeError):
    """An already-classified mixed-channel EXR file I/O failure."""


def _validate_exr_channel_frames(frames: object) -> tuple[Frame, ...]:
    if isinstance(frames, (str, bytes, bytearray)) or not isinstance(frames, Sequence):
        raise ValueError(
            _actionable_error(
                why="write_exr_channels requires a nonempty Sequence of Frame objects",
                what=f"frames_type={type(frames).__module__}.{type(frames).__qualname__}",
                how="pass one or more pixtreme Frames in a list or tuple",
            )
        )
    resolved = tuple(frames)
    if not resolved:
        raise ValueError(
            _actionable_error(
                why="write_exr_channels cannot write an empty Frame sequence",
                what="frames=()",
                how="pass at least one pixtreme Frame",
            )
        )
    for index, frame in enumerate(resolved):
        if not isinstance(frame, Frame):
            raise ValueError(
                _actionable_error(
                    why="write_exr_channels accepts only Frame elements",
                    what=f"frame_index={index}, type={type(frame).__module__}.{type(frame).__qualname__}",
                    how="replace every sequence element with a pixtreme Frame",
                )
            )
    return resolved


def _validated_write_channels(frames: object) -> tuple[tuple[Frame, ...], tuple[_ExrWriteChannel, ...], int, int, int]:
    resolved = _validate_exr_channel_frames(frames)
    first = resolved[0]
    height, width = first.height, first.width
    device_id = int(first.data.device.id)
    colorspace = first.colorspace
    owners: dict[str, int] = {}
    channels: list[_ExrWriteChannel] = []
    for frame_index, frame in enumerate(resolved):
        if (frame.height, frame.width) != (height, width):
            raise ValueError(
                _actionable_error(
                    why="all write_exr_channels Frames must have the same width and height",
                    what=(
                        f"frame_index={frame_index}, shape={frame.shape!r}, expected_height_width={(height, width)!r}"
                    ),
                    how="resize inputs explicitly so every Frame has identical width and height",
                )
            )
        frame_device = int(frame.data.device.id)
        if frame_device != device_id:
            raise ValueError(
                _actionable_error(
                    why="all write_exr_channels Frames must reside on the same CUDA device",
                    what=f"frame_index={frame_index}, device={frame_device}, expected_device={device_id}",
                    how="move inputs explicitly onto one CUDA device before writing",
                )
            )
        if frame.colorspace != colorspace:
            raise ValueError(
                _actionable_error(
                    why="all write_exr_channels Frames must declare the same colorspace",
                    what=(
                        f"frame_index={frame_index}, colorspace={frame.colorspace!r}, "
                        f"expected_colorspace={colorspace!r}"
                    ),
                    how="convert or relabel every input to one common colorspace before writing",
                )
            )
        dtype_name = frame.data.dtype.name
        pixel_type = _NATIVE_PIXEL_TYPES.get(dtype_name)
        if pixel_type is None:
            raise ValueError(
                _actionable_error(
                    why="write_exr_channels maps only native float16, float32, and uint32 Frame storage",
                    what=f"frame_index={frame_index}, dtype={dtype_name!r}",
                    how=(
                        "use px.values.cast_dtype(frame, dtype='uint32') for literal ID or code values; "
                        "use px.values.recode_dtype(frame, dtype='float16' | 'float32') for normalized images"
                    ),
                )
            )
        dtype, bytes_per_sample = _EXR_DTYPE_INFO[pixel_type]
        for channel_index, label in enumerate(frame.channels):
            if not isinstance(label, str):
                raise ValueError(
                    _actionable_error(
                        why="EXR output channel labels must be strings",
                        what=f"frame_index={frame_index}, channel_index={channel_index}, label={label!r}",
                        how="use a nonempty UTF-8 string label for every Frame channel",
                    )
                )
            prior_owner = owners.get(label)
            if prior_owner is not None:
                raise ValueError(
                    _actionable_error(
                        why="EXR output channel labels must be unique across all Frames",
                        what=f"label={label!r}, frame_index={prior_owner}, conflicting_frame_index={frame_index}",
                        how="assign one globally unique label to every output channel",
                    )
                )
            owners[label] = frame_index
            descriptor = _ExrChannel(
                name=label,
                pixel_type=pixel_type,
                dtype=dtype,
                bytes_per_sample=bytes_per_sample,
                perceptually_linear=False,
                x_sampling=1,
                y_sampling=1,
            )
            channels.append(_ExrWriteChannel(descriptor=descriptor, plane=frame.data[..., channel_index]))
    ordered = tuple(sorted(channels, key=lambda channel: channel.descriptor.name))
    _encode_exr_output_channels(tuple(channel.descriptor.name for channel in ordered))
    return resolved, ordered, height, width, device_id


def _pack_channel_rows(
    channels: Sequence[_ExrWriteChannel],
    *,
    height: int,
    width: int,
) -> tuple[cp.ndarray, int]:
    row_bytes = width * sum(channel.descriptor.bytes_per_sample for channel in channels)
    rows = cp.empty((height, row_bytes), dtype=cp.uint8)
    channel_offset = 0
    for channel in channels:
        channel_row_bytes = width * channel.descriptor.bytes_per_sample
        source = cp.ascontiguousarray(channel.plane).view(cp.uint8).reshape(height, channel_row_bytes)
        rows[:, channel_offset : channel_offset + channel_row_bytes] = source
        channel_offset += channel_row_bytes
    return rows.reshape(-1), row_bytes


def _encode_pxr24_channel_chunks(
    raw: cp.ndarray,
    channels: Sequence[_ExrWriteChannel],
    *,
    raw_offsets: Sequence[int],
    raw_sizes: Sequence[int],
    row_starts: Sequence[int],
    row_counts: Sequence[int],
    width: int,
) -> tuple[cp.ndarray, tuple[int, ...]]:
    transformed_channels: list[cp.ndarray] = []
    for channel in channels:
        plane = cp.ascontiguousarray(channel.plane)
        bits = plane.view(cp.uint16).astype(cp.uint32) if channel.descriptor.pixel_type == 1 else plane.view(cp.uint32)
        transformed_channels.append(_encode_pxr24_rows_gpu(bits.reshape(plane.shape), channel.descriptor.pixel_type))
    chunk_parts: list[cp.ndarray] = []
    materialized_sizes: list[int] = []
    for row_start, row_count in zip(row_starts, row_counts, strict=True):
        row_end = row_start + row_count
        row_channels = tuple(
            transformed[row_start:row_end].reshape(row_count, -1) for transformed in transformed_channels
        )
        chunk = cp.concatenate(row_channels, axis=1).reshape(-1)
        chunk_parts.append(chunk)
        materialized_sizes.append(int(chunk.size))
    transformed = cp.concatenate(chunk_parts)
    materialized_size_tuple = tuple(materialized_sizes)
    materialized_offsets = _prefix_offsets(materialized_size_tuple)
    adler = _checksum_exr_chunks(transformed, materialized_offsets, materialized_size_tuple)
    compressed, compressed_offsets, compressed_sizes = _encode_deflate_chunks(
        transformed,
        tuple(zip(materialized_offsets, materialized_size_tuple, strict=True)),
    )
    wrapped, wrapped_offsets, wrapped_sizes = _wrap_deflate_chunks(
        compressed,
        compressed_offsets,
        compressed_sizes,
        adler,
    )
    return _select_exr_payloads(
        raw,
        raw_offsets,
        raw_sizes,
        wrapped,
        wrapped_offsets,
        wrapped_sizes,
    )


def _encode_b44_channel_chunks(
    raw: cp.ndarray,
    channels: Sequence[_ExrWriteChannel],
    *,
    raw_offsets: Sequence[int],
    raw_sizes: Sequence[int],
    row_counts: Sequence[int],
    width: int,
    codec: str,
) -> tuple[cp.ndarray, tuple[int, ...]]:
    channel_payloads: list[tuple[cp.ndarray, tuple[int, ...], tuple[int, ...]]] = []
    for channel in channels:
        source = cp.ascontiguousarray(channel.plane).view(cp.uint8).reshape(-1)
        channel_raw_sizes = tuple(row_count * width * channel.descriptor.bytes_per_sample for row_count in row_counts)
        payload, offsets, sizes = _encode_b44_chunks_gpu(
            source,
            _prefix_offsets(channel_raw_sizes),
            row_counts,
            width=width,
            channel_count=1,
            pixel_type=channel.descriptor.pixel_type,
            codec=codec,
        )
        channel_payloads.append((payload, offsets, sizes))
    encoded_parts: list[cp.ndarray] = []
    encoded_sizes: list[int] = []
    for chunk_index in range(len(row_counts)):
        chunk_parts = tuple(
            payload[offsets[chunk_index] : offsets[chunk_index] + sizes[chunk_index]]
            for payload, offsets, sizes in channel_payloads
        )
        encoded_parts.extend(chunk_parts)
        encoded_sizes.append(sum(sizes[chunk_index] for _, _, sizes in channel_payloads))
    encoded = cp.concatenate(encoded_parts)
    encoded_size_tuple = tuple(encoded_sizes)
    return _select_exr_payloads(
        raw,
        raw_offsets,
        raw_sizes,
        encoded,
        _prefix_offsets(encoded_size_tuple),
        encoded_size_tuple,
    )


def _encode_channel_chunks(
    raw: cp.ndarray,
    channels: Sequence[_ExrWriteChannel],
    *,
    compression: str,
    dwa_level: float | None,
    raw_offsets: Sequence[int],
    raw_sizes: Sequence[int],
    row_starts: Sequence[int],
    row_counts: Sequence[int],
    width: int,
) -> tuple[cp.ndarray, tuple[int, ...]]:
    descriptors = tuple(channel.descriptor for channel in channels)
    if compression == "none":
        return raw, tuple(raw_sizes)
    if compression == "rle":
        transformed = _transform_exr_chunks(raw, raw_offsets, raw_sizes)
        encoded, encoded_offsets, encoded_sizes = _encode_rle_packets_gpu(transformed, raw_offsets, raw_sizes)
        return _select_exr_payloads(
            raw,
            raw_offsets,
            raw_sizes,
            encoded,
            encoded_offsets,
            encoded_sizes,
        )
    if compression == "piz":
        return _encode_piz_chunks_gpu(
            raw,
            raw_offsets,
            raw_sizes,
            row_counts=row_counts,
            width=width,
            channel_count=len(channels),
            pixel_type=descriptors[0].pixel_type,
            channel_pixel_types=tuple(channel.pixel_type for channel in descriptors),
        )
    if compression == "pxr24":
        return _encode_pxr24_channel_chunks(
            raw,
            channels,
            raw_offsets=raw_offsets,
            raw_sizes=raw_sizes,
            row_starts=row_starts,
            row_counts=row_counts,
            width=width,
        )
    if compression in ("b44", "b44a"):
        return _encode_b44_channel_chunks(
            raw,
            channels,
            raw_offsets=raw_offsets,
            raw_sizes=raw_sizes,
            row_counts=row_counts,
            width=width,
            codec=compression,
        )
    if compression in ("dwaa", "dwab"):
        if dwa_level is None:
            raise _gpu_error(
                why="the mixed DWA writer received no resolved compression level",
                what=f"compression={compression!r}",
                how="validate DWA options before encoding mixed EXR channels",
            )
        layout = _classify_default_dwa_channels(descriptors)
        channel_rules = _dwa_mixed_channel_rules_bytes(descriptors)
        return _encode_dwa_channel_chunks_gpu(
            {channel.descriptor.name: channel.plane for channel in channels},
            raw,
            descriptors,
            layout,
            channel_rules,
            row_counts=row_counts,
            raw_offsets=raw_offsets,
            raw_sizes=raw_sizes,
            lines_per_chunk=_EXR_LINES_PER_CHUNK[compression],
            dwa_level=dwa_level,
        )
    transformed, adler = _transform_and_checksum_chunks(raw, raw_offsets, raw_sizes)
    compressed, compressed_offsets, compressed_sizes = _encode_deflate_chunks(
        transformed,
        tuple(zip(raw_offsets, raw_sizes, strict=True)),
    )
    wrapped, wrapped_offsets, wrapped_sizes = _wrap_deflate_chunks(
        compressed,
        compressed_offsets,
        compressed_sizes,
        adler,
    )
    return _select_exr_payloads(
        raw,
        raw_offsets,
        raw_sizes,
        wrapped,
        wrapped_offsets,
        wrapped_sizes,
    )


def _write_exr_channel_payloads(
    path: Path,
    header: bytes,
    payload: cp.ndarray,
    payload_sizes: Sequence[int],
    row_starts: Sequence[int],
) -> None:
    payload_memory = cp.cuda.alloc_pinned_memory(int(payload.nbytes))
    payload_host = np.frombuffer(payload_memory, dtype=np.uint8, count=int(payload.size))
    payload.get(out=payload_host)
    payload_view = memoryview(payload_host)
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
                stream.write(payload_view[payload_offset : payload_offset + size])
                payload_offset += size
    except OSError as error:
        raise _ExrFileWriteError(
            _actionable_error(
                why=f"the mixed-channel EXR file could not be written: {error}",
                what=str(path),
                how="provide a writable output path whose parent directory already exists",
            )
        ) from error


def _write_exr_channels(path: Path, frames: object, *, compression: str, dwa_level: float | None) -> None:
    resolved, channels, height, width, device_id = _validated_write_channels(frames)
    descriptors = tuple(channel.descriptor for channel in channels)
    encoded_channels = _encode_exr_output_channels(tuple(channel.name for channel in descriptors))
    try:
        with cp.cuda.Device(device_id):
            raw, row_bytes = _pack_channel_rows(channels, height=height, width=width)
            lines_per_chunk = _EXR_LINES_PER_CHUNK[compression]
            row_starts = tuple(range(0, height, lines_per_chunk))
            row_counts = tuple(min(lines_per_chunk, height - row_start) for row_start in row_starts)
            raw_sizes = tuple(row_count * row_bytes for row_count in row_counts)
            raw_offsets = _prefix_offsets(raw_sizes)
            payload, payload_sizes = _encode_channel_chunks(
                raw,
                channels,
                compression=compression,
                dwa_level=dwa_level,
                raw_offsets=raw_offsets,
                raw_sizes=raw_sizes,
                row_starts=row_starts,
                row_counts=row_counts,
                width=width,
            )
            header = _exr_write_header(
                width=width,
                height=height,
                encoded_channels=encoded_channels,
                pixel_type=descriptors[0].pixel_type,
                channel_pixel_types=tuple(channel.pixel_type for channel in descriptors),
                compression=compression,
                chromaticities=_colorspace_chromaticities(resolved[0].colorspace),
                aces_image_container=resolved[0].colorspace == "ACES2065-1",
                dwa_level=dwa_level,
            )
            _write_exr_channel_payloads(path, header, payload, payload_sizes, row_starts)
    except (_ExrGpuError, _ExrPizError, _ExrFileWriteError):
        raise
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why=f"CUDA or the EXR codec could not encode mixed native channel types: {error}",
                what=(
                    f"compression={compression!r}, frames={len(resolved)}, "
                    f"channels={tuple(channel.name for channel in descriptors)!r}"
                ),
                how="verify CUDA availability and provide valid native EXR channel storage",
            )
        ) from error
