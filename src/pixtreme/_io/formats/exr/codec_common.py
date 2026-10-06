"""Shared lower-level EXR codec data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pixtreme._core.errors import _actionable_error

if TYPE_CHECKING:
    from pixtreme._io.formats.exr.container import _ExrChannel


@dataclass(frozen=True)
class _CanonicalHuffmanCode:
    symbol: int
    length: int
    code: int


@dataclass(frozen=True)
class _ExrByteSpan:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _ExrChannelRow:
    channel_index: int
    channel_name: str
    pixel_type: int
    bytes_per_sample: int
    perceptually_linear: bool
    chunk_row: int
    file_y: int
    output_row: int
    raw_span: _ExrByteSpan
    materialized_span: _ExrByteSpan


@dataclass(frozen=True)
class _ExrChunkDescriptor:
    codec: str
    lines_per_chunk: int
    chunk_y: int
    row_start: int
    row_count: int
    payload_span: _ExrByteSpan
    stored_size: int
    expected_raw_size: int
    expected_materialized_size: int
    raw_stored: bool
    channel_rows: tuple[_ExrChannelRow, ...]


class _ExrCodecError(RuntimeError):
    """An already-classified compression-specific EXR integrity failure."""


def _codec_error(*, why: str, what: str, how: str) -> _ExrCodecError:
    return _ExrCodecError(_actionable_error(why=why, what=what, how=how))


def _chunk_channel_geometry(
    channel: _ExrChannel,
    *,
    width: int,
    chunk_y: int,
    row_count: int,
) -> tuple[int, int]:
    sampling = channel.sampling
    channel_width = width if sampling is None else sampling.width
    channel_rows = sum(file_y % channel.y_sampling == 0 for file_y in range(chunk_y, chunk_y + row_count))
    return channel_width, channel_rows


def _channel_rows(
    channels: Sequence[_ExrChannel],
    *,
    width: int,
    chunk_y: int,
    row_start: int,
    row_count: int,
    materialized_bytes_per_sample: Sequence[int] | None = None,
    materialized_channel_offsets: Sequence[int] | None = None,
) -> tuple[_ExrChannelRow, ...]:
    """Describe raw scanlines and materialized scanlines or channel planes in one pass."""
    rows: list[_ExrChannelRow] = []
    raw_cursor = 0
    materialized_cursor = 0
    channel_cursors = list(materialized_channel_offsets) if materialized_channel_offsets is not None else None
    for chunk_row in range(row_count):
        file_y = chunk_y + chunk_row
        for channel_index, channel in enumerate(channels):
            if file_y % channel.y_sampling:
                continue
            channel_width, _ = _chunk_channel_geometry(channel, width=width, chunk_y=chunk_y, row_count=row_count)
            raw_size = channel_width * channel.bytes_per_sample
            raw_span = _ExrByteSpan(raw_cursor, raw_cursor + raw_size)
            materialized_span = raw_span
            if materialized_bytes_per_sample is not None or channel_cursors is not None:
                materialized_size = (
                    raw_size
                    if materialized_bytes_per_sample is None
                    else channel_width * materialized_bytes_per_sample[channel_index]
                )
                materialized_start = materialized_cursor if channel_cursors is None else channel_cursors[channel_index]
                materialized_span = _ExrByteSpan(materialized_start, materialized_start + materialized_size)
                if channel_cursors is None:
                    materialized_cursor = materialized_span.end
                else:
                    channel_cursors[channel_index] = materialized_span.end
            rows.append(
                _ExrChannelRow(
                    channel_index=channel_index,
                    channel_name=channel.name,
                    pixel_type=channel.pixel_type,
                    bytes_per_sample=channel.bytes_per_sample,
                    perceptually_linear=channel.perceptually_linear,
                    chunk_row=chunk_row,
                    file_y=file_y,
                    output_row=row_start + chunk_row,
                    raw_span=raw_span,
                    materialized_span=materialized_span,
                )
            )
            raw_cursor += raw_size
    return tuple(rows)
