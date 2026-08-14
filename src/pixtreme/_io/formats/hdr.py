"""Radiance HDR parsing, decoding, CUDA conversion, and encoding."""

from __future__ import annotations

import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, NoReturn

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    ChannelInput,
    Frame,
)
from pixtreme._io.common import (
    _binary_stream,
    _resolve_channel_locations,
    _resolve_metadata,
)
from pixtreme._io.dtype import (
    _prepare_write_frame,
)
from pixtreme._io.models import (
    ImageHeader,
    _ImageColorInfo,
    _ImagePart,
)

_HDR_THREADS_PER_BLOCK = 256

_HDR_READ_KERNEL_SOURCE = r"""
extern "C" __global__ void pixtreme_hdr_read(
    const unsigned char* __restrict__ source,
    float* __restrict__ output,
    const long long element_count,
    const int output_channels,
    const int* __restrict__ source_indices
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {
        return;
    }
    const int output_channel = (int)(element % output_channels);
    const long long pixel = element / output_channels;
    const long long source_base = pixel * 4;
    const unsigned char exponent = source[source_base + 3];
    if (exponent == 0) {
        output[element] = 0.0f;
        return;
    }
    const unsigned char mantissa = source[source_base + source_indices[output_channel]];
    output[element] = ldexpf((float)mantissa + 0.5f, (int)exponent - 136);
}
"""

_HDR_WRITE_KERNEL_SOURCE = r"""
extern "C" __global__ void pixtreme_hdr_write(
    const float* __restrict__ source,
    unsigned char* __restrict__ output,
    const long long pixel_count,
    const int source_channels,
    const int source_r,
    const int source_g,
    const int source_b
) {
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const long long source_base = pixel * source_channels;
    const long long output_base = pixel * 4;
    const float r = source[source_base + source_r];
    const float g = source[source_base + source_g];
    const float b = source[source_base + source_b];
    float maximum = r > g ? r : g;
    if (b > maximum) {
        maximum = b;
    }
    if (maximum <= 1e-32f) {
        output[output_base] = 0;
        output[output_base + 1] = 0;
        output[output_base + 2] = 0;
        output[output_base + 3] = 0;
        return;
    }
    int exponent;
    const float scale = frexpf(maximum, &exponent) * 256.0f / maximum;
    output[output_base] = r > 0.0f ? (unsigned char)(r * scale) : 0;
    output[output_base + 1] = g > 0.0f ? (unsigned char)(g * scale) : 0;
    output[output_base + 2] = b > 0.0f ? (unsigned char)(b * scale) : 0;
    output[output_base + 3] = (unsigned char)(exponent + 128);
}
"""


class _UnsupportedHdrError(ValueError):
    """Mark a valid Radiance header configuration that is outside HDR scope."""


@dataclass(frozen=True)
class _HdrLayout:
    width: int
    height: int
    data_offset: int
    raw_variables: dict[str, object]


def _unsupported_hdr(*, why: str, what: str, how: str) -> NoReturn:
    raise _UnsupportedHdrError(_actionable_error(why=why, what=what, how=how))


def _read_hdr_line(stream: BinaryIO, *, field: str) -> str:
    raw = stream.readline()
    if not raw:
        raise ValueError(
            _actionable_error(
                why=f"the Radiance HDR payload ended before the {field} line",
                what=f"stream_offset={stream.tell()}",
                how="provide a complete ASCII header, blank separator, resolution line, and pixel payload",
            )
        )
    if not raw.endswith(b"\n"):
        raise ValueError(
            _actionable_error(
                why=f"the Radiance HDR {field} line is not newline-terminated",
                what=f"line_bytes={len(raw)}",
                how="terminate every Radiance header and resolution line with a newline",
            )
        )
    return raw.rstrip(b"\r\n").decode("ascii")


def _parse_hdr_layout(source: Path | bytes) -> _HdrLayout:
    with _binary_stream(source) as stream:
        signature = _read_hdr_line(stream, field="signature")
        if signature != "#?RADIANCE":
            raise ValueError(
                _actionable_error(
                    why="the Radiance HDR signature is missing or unsupported",
                    what=repr(signature),
                    how="begin the file with '#?RADIANCE'",
                )
            )

        format_values: list[str] = []
        raw_variables: defaultdict[str, list[str]] = defaultdict(list)
        while True:
            line = _read_hdr_line(stream, field="information header")
            if not line:
                break
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name == "FORMAT":
                format_values.append(value)
            elif name in ("EXPOSURE", "PRIMARIES", "COLORCORR"):
                raw_variables[name].append(value)

        if len(format_values) != 1:
            raise ValueError(
                _actionable_error(
                    why="the Radiance HDR header must contain exactly one FORMAT assignment",
                    what=f"FORMAT values={tuple(format_values)!r}",
                    how="write one FORMAT=32-bit_rle_rgbe line before the blank header terminator",
                )
            )
        format_value = format_values[0]
        if format_value == "32-bit_rle_xyze":
            _unsupported_hdr(
                why="Radiance XYZE storage is outside the supported HDR RGBE format",
                what=f"FORMAT={format_value}",
                how="convert the image to FORMAT=32-bit_rle_rgbe before reading",
            )
        if format_value != "32-bit_rle_rgbe":
            raise ValueError(
                _actionable_error(
                    why="the Radiance HDR FORMAT assignment is not supported",
                    what=f"FORMAT={format_value}",
                    how="use FORMAT=32-bit_rle_rgbe",
                )
            )

        resolution = _read_hdr_line(stream, field="resolution")
        match = re.fullmatch(r"-Y[ \t]+([1-9][0-9]*)[ \t]+\+X[ \t]+([1-9][0-9]*)", resolution)
        if match is None:
            _unsupported_hdr(
                why="the Radiance HDR orientation is outside standard top-down '-Y H +X W' order",
                what=repr(resolution),
                how="store scanlines in standard -Y H +X W orientation",
            )
        height, width = (int(value) for value in match.groups())
        normalized_variables: dict[str, object] = {name: tuple(values) for name, values in raw_variables.items()}
        return _HdrLayout(
            width=width,
            height=height,
            data_offset=stream.tell(),
            raw_variables=normalized_variables,
        )


def _hdr_header(layout: _HdrLayout) -> ImageHeader:
    return ImageHeader(
        format="HDR",
        width=layout.width,
        height=layout.height,
        parts=(_ImagePart(name="", channels={"R": "float32", "G": "float32", "B": "float32"}),),
        color=_ImageColorInfo(
            raw=layout.raw_variables,
            colorspace="Rec.709",
            gamma="linear",
            mappable=True,
        ),
    )


def _parse_hdr(source: Path | bytes) -> ImageHeader:
    return _hdr_header(_parse_hdr_layout(source))


def _decode_hdr_old_scanline(data: bytes, offset: int, width: int, *, first_pixel: bytes = b"") -> tuple[bytes, int]:
    output = bytearray(first_pixel)
    previous = first_pixel or None
    repeat_shift = 0
    while len(output) // 4 < width:
        end = offset + 4
        if end > len(data):
            raise ValueError(
                _actionable_error(
                    why="the flat or old-style Radiance HDR scanline is truncated",
                    what=f"decoded_pixels={len(output) // 4}, expected_pixels={width}",
                    how="provide one complete RGBE value or repeat marker for every declared pixel",
                )
            )
        pixel = data[offset:end]
        offset = end
        if pixel[:3] == b"\x01\x01\x01":
            if previous is None:
                raise ValueError(
                    _actionable_error(
                        why="an old-style Radiance HDR repeat marker has no previous pixel",
                        what=f"decoded_pixels={len(output) // 4}",
                        how="begin each scanline with a literal RGBE pixel",
                    )
                )
            if repeat_shift >= 63:
                raise ValueError(
                    _actionable_error(
                        why="an old-style Radiance HDR repeat count exceeds supported image dimensions",
                        what=f"repeat_shift={repeat_shift}, width={width}",
                        how="limit consecutive repeat markers to the declared scanline width",
                    )
                )
            repeat_count = pixel[3] << repeat_shift
            decoded_pixels = len(output) // 4
            if decoded_pixels + repeat_count > width:
                raise ValueError(
                    _actionable_error(
                        why="an old-style Radiance HDR repeat marker exceeds the declared scanline width",
                        what=f"decoded_pixels={decoded_pixels}, repeat_count={repeat_count}, width={width}",
                        how="limit repeat counts to the remaining pixels in the scanline",
                    )
                )
            output.extend(previous * repeat_count)
            repeat_shift += 8
        else:
            output.extend(pixel)
            previous = pixel
            repeat_shift = 0
    return bytes(output), offset


def _decode_hdr_new_scanline(data: bytes, offset: int, width: int) -> tuple[bytes, int]:
    components: list[bytearray] = []
    for component in range(4):
        values = bytearray()
        while len(values) < width:
            if offset >= len(data):
                raise ValueError(
                    _actionable_error(
                        why="the new-style Radiance HDR scanline is truncated before a component packet",
                        what=f"component={component}, decoded_values={len(values)}, width={width}",
                        how="provide complete adaptive RLE packets for all four RGBE components",
                    )
                )
            code = data[offset]
            offset += 1
            if code == 0:
                raise ValueError(
                    _actionable_error(
                        why="a new-style Radiance HDR component packet has a zero count",
                        what=f"component={component}, code={code}, decoded_values={len(values)}",
                        how="use literal counts 1 through 128 or run counts 1 through 127",
                    )
                )
            if code > 128:
                count = code - 128
                if offset >= len(data):
                    raise ValueError(
                        _actionable_error(
                            why="a new-style Radiance HDR run packet is truncated",
                            what=f"component={component}, count={count}",
                            how="provide one component value after every run packet code",
                        )
                    )
                value = data[offset]
                offset += 1
                packet = bytes((value,)) * count
            else:
                count = code
                end = offset + count
                if end > len(data):
                    raise ValueError(
                        _actionable_error(
                            why="a new-style Radiance HDR literal packet is truncated",
                            what=f"component={component}, count={count}, available={len(data) - offset}",
                            how="provide every component value declared by the literal packet",
                        )
                    )
                packet = data[offset:end]
                offset = end
            if len(values) + count > width:
                raise ValueError(
                    _actionable_error(
                        why="a new-style Radiance HDR component packet exceeds the declared scanline width",
                        what=f"component={component}, decoded_values={len(values)}, count={count}, width={width}",
                        how="limit component packets to exactly the declared scanline width",
                    )
                )
            values.extend(packet)
        components.append(values)
    scanline = np.stack(tuple(np.frombuffer(component, dtype=np.uint8) for component in components), axis=1)
    return scanline.tobytes(), offset


def _hdr_pixel_bytes(data: bytes, layout: _HdrLayout) -> bytes:
    offset = layout.data_offset
    output = bytearray()
    for row in range(layout.height):
        if 8 <= layout.width <= 32767:
            end = offset + 4
            if end > len(data):
                raise ValueError(
                    _actionable_error(
                        why="the Radiance HDR scanline is truncated before its first RGBE value or RLE marker",
                        what=f"row={row}, available={len(data) - offset}",
                        how="provide a complete scanline for every declared row",
                    )
                )
            prefix = data[offset:end]
            offset = end
            if prefix[0] == 2 and prefix[1] == 2 and not (prefix[2] & 0x80):
                declared_width = (prefix[2] << 8) | prefix[3]
                if declared_width != layout.width:
                    raise ValueError(
                        _actionable_error(
                            why="the new-style Radiance HDR scanline length differs from the resolution string",
                            what=f"row={row}, scanline_length={declared_width}, width={layout.width}",
                            how="write the same width in every scanline marker and the resolution string",
                        )
                    )
                scanline, offset = _decode_hdr_new_scanline(data, offset, layout.width)
            else:
                scanline, offset = _decode_hdr_old_scanline(data, offset, layout.width, first_pixel=prefix)
        else:
            scanline, offset = _decode_hdr_old_scanline(data, offset, layout.width)
        output.extend(scanline)
    return bytes(output)


@lru_cache(maxsize=1)
def _hdr_read_kernel() -> cp.RawKernel:
    return cp.RawKernel(_HDR_READ_KERNEL_SOURCE, "pixtreme_hdr_read")


@lru_cache(maxsize=1)
def _hdr_write_kernel() -> cp.RawKernel:
    return cp.RawKernel(_HDR_WRITE_KERNEL_SOURCE, "pixtreme_hdr_write")


def _read_hdr_frame(
    path: Path,
    *,
    channels: ChannelInput | None,
    colorspace: str | None,
    gamma: str | None,
) -> Frame:
    try:
        data = path.read_bytes()
        layout = _parse_hdr_layout(data)
        pixel_bytes = _hdr_pixel_bytes(data, layout)
    except _UnsupportedHdrError:
        raise
    except (OSError, ValueError, struct.error) as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the Radiance HDR pixels could not be decoded: {error}",
                what=str(path),
                how="verify that the file has a complete RGBE header, resolution, and supported scanlines",
            )
        ) from error

    header = _hdr_header(layout)
    resolved_colorspace, resolved_gamma = _resolve_metadata(header, colorspace=colorspace, gamma=gamma)
    locations = _resolve_channel_locations(header, channels)
    source_indices = {"R": 0, "G": 1, "B": 2}
    selected_indices = [source_indices[channel] for _, channel, _ in locations]
    device_pixels = cp.asarray(np.frombuffer(pixel_bytes, dtype=np.uint8))
    device_channel_indices = cp.asarray(selected_indices, dtype=cp.int32)
    output = cp.empty((layout.height, layout.width, len(locations)), dtype=cp.float32)
    element_count = int(output.size)
    block_count = (element_count + _HDR_THREADS_PER_BLOCK - 1) // _HDR_THREADS_PER_BLOCK
    _hdr_read_kernel()(
        (block_count,),
        (_HDR_THREADS_PER_BLOCK,),
        (
            device_pixels,
            output,
            np.int64(element_count),
            np.int32(len(locations)),
            device_channel_indices,
        ),
    )
    return Frame(
        data=output,
        colorspace=resolved_colorspace,
        gamma=resolved_gamma,
        channels=tuple(label for _, _, label in locations),
    )


def _validate_hdr_write_layout(frame: Frame) -> tuple[int, int, int]:
    channels = frame.channels
    if len(channels) != 3 or frozenset(channels) != frozenset(("R", "G", "B")):
        duplicates = tuple(label for label in dict.fromkeys(channels) if channels.count(label) > 1)
        raise ValueError(
            _actionable_error(
                why="Radiance HDR output requires one unique RGB channel set",
                what=f"channels={channels!r}, duplicates={duplicates!r}",
                how="provide exactly one R, G, and B channel without alpha or duplicates",
            )
        )
    if not 8 <= frame.width <= 32767:
        raise ValueError(
            _actionable_error(
                why="Radiance HDR new-style RLE output requires width from 8 through 32767",
                what=f"width={frame.width}",
                how="resize the image to width 8..32767 or use another output format",
            )
        )
    return (frame.channels.index("R"), frame.channels.index("G"), frame.channels.index("B"))


def _hdr_write_data(frame: Frame) -> cp.ndarray:
    source_indices = _validate_hdr_write_layout(frame)
    source_frame = _prepare_write_frame("HDR", frame)
    output = cp.empty((source_frame.height, source_frame.width, 4), dtype=cp.uint8)
    pixel_count = source_frame.height * source_frame.width
    block_count = (pixel_count + _HDR_THREADS_PER_BLOCK - 1) // _HDR_THREADS_PER_BLOCK
    _hdr_write_kernel()(
        (block_count,),
        (_HDR_THREADS_PER_BLOCK,),
        (
            source_frame.data,
            output,
            np.int64(pixel_count),
            np.int32(len(source_frame.channels)),
            np.int32(source_indices[0]),
            np.int32(source_indices[1]),
            np.int32(source_indices[2]),
        ),
    )
    return output


def _append_hdr_literal_packets(output: bytearray, values: np.ndarray, start: int, end: int) -> None:
    while start < end:
        packet_end = min(start + 128, end)
        output.append(packet_end - start)
        output.extend(values[start:packet_end].tobytes())
        start = packet_end


def _append_hdr_run_packets(output: bytearray, value: np.uint8, count: int) -> None:
    while count > 0:
        packet_count = min(127, count)
        output.extend((128 + packet_count, int(value)))
        count -= packet_count


def _encode_hdr_component(values: np.ndarray) -> bytes:
    output = bytearray()
    equal_edges = values[1:] == values[:-1]
    padded = np.zeros(equal_edges.size + 2, dtype=np.bool_)
    padded[1:-1] = equal_edges
    run_starts = np.flatnonzero(np.logical_and(padded[1:], np.logical_not(padded[:-1])))
    run_ends = np.flatnonzero(np.logical_and(np.logical_not(padded[1:]), padded[:-1])) + 1
    cursor = 0
    for run_start_value, run_end_value in zip(run_starts, run_ends, strict=True):
        run_start = int(run_start_value)
        run_end = int(run_end_value)
        if run_end - run_start < 4:
            continue
        _append_hdr_literal_packets(output, values, cursor, run_start)
        _append_hdr_run_packets(output, values[run_start], run_end - run_start)
        cursor = run_end
    _append_hdr_literal_packets(output, values, cursor, int(values.size))
    return bytes(output)


def _encode_hdr_file(frame: Frame) -> bytes:
    host_rgbe = cp.asnumpy(_hdr_write_data(frame))
    output = bytearray(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
    output.extend(f"-Y {frame.height} +X {frame.width}\n".encode("ascii"))
    for row in host_rgbe:
        output.extend((2, 2, frame.width >> 8, frame.width & 0xFF))
        for component in range(4):
            output.extend(_encode_hdr_component(row[:, component]))
    return bytes(output)
