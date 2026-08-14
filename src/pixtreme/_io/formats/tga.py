"""TGA parsing, decoding, CUDA conversion, and encoding."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NoReturn

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    ChannelInput,
    Frame,
)
from pixtreme._io.common import (
    _binary_stream,
    _empty_color,
    _read_exact,
    _resolve_channel_locations,
    _resolve_metadata,
)
from pixtreme._io.models import (
    ImageHeader,
    _ImagePart,
)
from pixtreme._values.cast import recode_dtype

_TGA_HEADER = struct.Struct("<BBBHHBHHHHBB")
_TGA_FOOTER = struct.pack("<II", 0, 0) + b"TRUEVISION-XFILE.\x00"
_TGA_THREADS_PER_BLOCK = 256

_TGA_READ_KERNEL_TEMPLATE = r"""
typedef __OUTPUT_TYPE__ pixtreme_tga_output_t;

extern "C" __global__ void pixtreme_tga_read(
    const unsigned char* __restrict__ source,
    pixtreme_tga_output_t* __restrict__ output,
    const long long element_count,
    const int width,
    const int height,
    const int source_channels,
    const int output_channels,
    const int top_origin,
    const int* __restrict__ source_indices
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {
        return;
    }
    const int output_channel = (int)(element % output_channels);
    const long long output_pixel = element / output_channels;
    const int output_y = (int)(output_pixel / width);
    const int output_x = (int)(output_pixel % width);
    const int source_y = top_origin ? output_y : height - 1 - output_y;
    const long long source_base = ((long long)source_y * width + output_x) * source_channels;
    const int source_channel = source_indices[output_channel];
    const unsigned char value = source[source_base + source_channel];
    output[element] = __OUTPUT_EXPRESSION__;
}
"""

_TGA_WRITE_KERNEL_SOURCE = r"""
extern "C" __global__ void pixtreme_tga_write(
    const unsigned char* __restrict__ source,
    unsigned char* __restrict__ output,
    const long long element_count,
    const int source_channels,
    const int source_b,
    const int source_g,
    const int source_r,
    const int source_a
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {
        return;
    }
    const int output_channel = (int)(element % source_channels);
    const long long pixel = element / source_channels;
    int source_channel = source_b;
    if (output_channel == 1) {
        source_channel = source_g;
    } else if (output_channel == 2) {
        source_channel = source_r;
    } else if (output_channel == 3) {
        source_channel = source_a;
    }
    output[element] = source[pixel * source_channels + source_channel];
}
"""


class _UnsupportedTgaError(ValueError):
    """Mark a valid TGA header whose storage configuration is out of scope."""


@dataclass(frozen=True)
class _TgaLayout:
    image_type: int
    width: int
    height: int
    channels: int
    data_offset: int
    top_origin: bool


def _unsupported_tga(*, why: str, what: str, how: str) -> NoReturn:
    raise _UnsupportedTgaError(_actionable_error(why=why, what=what, how=how))


def _parse_tga_layout(source: Path | bytes) -> _TgaLayout:
    with _binary_stream(source) as stream:
        (
            image_id_length,
            color_map_type,
            image_type,
            color_map_origin,
            color_map_length,
            color_map_depth,
            _x_origin,
            _y_origin,
            width,
            height,
            pixel_depth,
            descriptor,
        ) = _TGA_HEADER.unpack(_read_exact(stream, _TGA_HEADER.size))
        _read_exact(stream, image_id_length)

    if image_type not in (2, 10):
        _unsupported_tga(
            why="the TGA image type is outside uncompressed and RLE true-color",
            what=f"image_type={image_type}",
            how="use image type 2 or 10 true-color TGA data",
        )
    if pixel_depth not in (24, 32):
        _unsupported_tga(
            why="the TGA pixel depth is outside 24-bit BGR and 32-bit BGRA",
            what=f"pixel_depth={pixel_depth}",
            how="use 24-bit or 32-bit true-color TGA data",
        )
    if descriptor & 0xC0:
        _unsupported_tga(
            why="the TGA image descriptor uses reserved interleaving bits",
            what=f"descriptor=0x{descriptor:02x}, reserved_bits={(descriptor >> 6) & 0x03}",
            how="clear image descriptor bits 6 and 7",
        )
    if descriptor & 0x10:
        _unsupported_tga(
            why="the TGA uses a right-to-left origin that is outside the supported origin set",
            what=f"descriptor=0x{descriptor:02x}, right-to-left origin bit=1",
            how="store pixels with a bottom-left or top-left origin",
        )
    attribute_bits = descriptor & 0x0F
    expected_attribute_bits = 8 if pixel_depth == 32 else 0
    if attribute_bits != expected_attribute_bits:
        _unsupported_tga(
            why="the TGA attribute-bit count does not match its supported true-color depth",
            what=(
                f"pixel_depth={pixel_depth}, attribute_bits={attribute_bits}, "
                f"expected_attribute_bits={expected_attribute_bits}"
            ),
            how=("use attribute bits 0 for 24-bit BGR or attribute bits 8 for 32-bit BGRA"),
        )
    if color_map_type != 0:
        _unsupported_tga(
            why="color-mapped TGA storage is outside the supported true-color set",
            what=f"color_map_type={color_map_type}",
            how="use a true-color TGA with color map type 0",
        )
    if (color_map_origin, color_map_length, color_map_depth) != (0, 0, 0):
        _unsupported_tga(
            why="a TGA without a color map must have an empty color-map specification",
            what=(
                f"color_map_origin={color_map_origin}, color_map_length={color_map_length}, "
                f"color_map_depth={color_map_depth}"
            ),
            how="set color-map origin, length, and entry depth to zero",
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            _actionable_error(
                why="the TGA dimensions must both be positive",
                what=f"width={width}, height={height}",
                how="write positive 16-bit width and height fields",
            )
        )
    return _TgaLayout(
        image_type=image_type,
        width=width,
        height=height,
        channels=pixel_depth // 8,
        data_offset=_TGA_HEADER.size + image_id_length,
        top_origin=bool(descriptor & 0x20),
    )


def _tga_header(layout: _TgaLayout) -> ImageHeader:
    labels = ("R", "G", "B", "A")[: layout.channels]
    return ImageHeader(
        format="TGA",
        width=layout.width,
        height=layout.height,
        parts=(_ImagePart(name="", channels=dict.fromkeys(labels, "uint8")),),
        color=_empty_color(),
    )


def _parse_tga(source: Path | bytes) -> ImageHeader:
    return _tga_header(_parse_tga_layout(source))


def _decode_tga_rle(data: bytes, layout: _TgaLayout) -> bytes:
    pixel_size = layout.channels
    pixel_count = layout.width * layout.height
    output_size = pixel_count * pixel_size
    output = bytearray(output_size)
    source_offset = layout.data_offset
    output_offset = 0
    while output_offset < output_size:
        if source_offset >= len(data):
            raise ValueError(
                _actionable_error(
                    why="the TGA RLE payload ended before the next packet header",
                    what=f"decoded_bytes={output_offset}, expected_bytes={output_size}",
                    how="provide complete RLE packet data for every declared pixel",
                )
            )
        packet_header = data[source_offset]
        source_offset += 1
        packet_pixels = (packet_header & 0x7F) + 1
        packet_bytes = packet_pixels * pixel_size
        if output_offset + packet_bytes > output_size:
            raise ValueError(
                _actionable_error(
                    why="a TGA RLE packet exceeds the declared image dimensions",
                    what=(
                        f"decoded_bytes={output_offset}, packet_pixels={packet_pixels}, expected_bytes={output_size}"
                    ),
                    how="limit RLE packets to the declared pixel count",
                )
            )
        if packet_header & 0x80:
            pixel_end = source_offset + pixel_size
            if pixel_end > len(data):
                raise ValueError(
                    _actionable_error(
                        why="the TGA RLE run packet has a truncated pixel value",
                        what=f"packet_offset={source_offset - 1}, pixel_size={pixel_size}",
                        how="provide one complete BGR or BGRA value after the packet header",
                    )
                )
            pixel = data[source_offset:pixel_end]
            source_offset = pixel_end
            output[output_offset : output_offset + packet_bytes] = pixel * packet_pixels
        else:
            packet_end = source_offset + packet_bytes
            if packet_end > len(data):
                raise ValueError(
                    _actionable_error(
                        why="the TGA RLE raw packet has truncated pixel values",
                        what=(
                            f"packet_offset={source_offset - 1}, requested_bytes={packet_bytes}, "
                            f"available_bytes={len(data) - source_offset}"
                        ),
                        how="provide every BGR or BGRA value declared by the raw packet",
                    )
                )
            output[output_offset : output_offset + packet_bytes] = data[source_offset:packet_end]
            source_offset = packet_end
        output_offset += packet_bytes
    return bytes(output)


def _tga_pixel_bytes(data: bytes, layout: _TgaLayout) -> bytes:
    if layout.image_type == 10:
        return _decode_tga_rle(data, layout)
    payload_size = layout.width * layout.height * layout.channels
    payload_end = layout.data_offset + payload_size
    if payload_end > len(data):
        raise ValueError(
            _actionable_error(
                why="the uncompressed TGA pixel payload is truncated",
                what=(f"requested_bytes={payload_size}, available_bytes={len(data) - layout.data_offset}"),
                how="provide one complete BGR or BGRA value for every declared pixel",
            )
        )
    return data[layout.data_offset : payload_end]


@lru_cache(maxsize=2)
def _tga_read_kernel(unchanged: bool) -> cp.RawKernel:
    output_type = "unsigned char" if unchanged else "float"
    output_expression = "value" if unchanged else "(float)value / 255.0f"
    source = _TGA_READ_KERNEL_TEMPLATE.replace("__OUTPUT_TYPE__", output_type).replace(
        "__OUTPUT_EXPRESSION__", output_expression
    )
    return cp.RawKernel(source, "pixtreme_tga_read")


@lru_cache(maxsize=1)
def _tga_write_kernel() -> cp.RawKernel:
    return cp.RawKernel(_TGA_WRITE_KERNEL_SOURCE, "pixtreme_tga_write")


def _read_tga_frame(
    path: Path,
    *,
    channels: ChannelInput | None,
    unchanged: bool,
    colorspace: str | None,
    gamma: str | None,
) -> Frame:
    try:
        data = path.read_bytes()
        layout = _parse_tga_layout(data)
        pixel_bytes = _tga_pixel_bytes(data, layout)
    except _UnsupportedTgaError:
        raise
    except (OSError, ValueError, struct.error) as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the TGA pixels could not be decoded: {error}",
                what=str(path),
                how="verify that the file contains complete supported true-color TGA data",
            )
        ) from error

    header = _tga_header(layout)
    resolved_colorspace, resolved_gamma = _resolve_metadata(header, colorspace=colorspace, gamma=gamma)
    locations = _resolve_channel_locations(header, channels)
    file_channel_indices = {"R": 2, "G": 1, "B": 0, "A": 3}
    selected_indices = [file_channel_indices[channel] for _, channel, _ in locations]

    device_pixels = cp.asarray(np.frombuffer(pixel_bytes, dtype=np.uint8))
    device_channel_indices = cp.asarray(selected_indices, dtype=cp.int32)
    output_dtype = cp.uint8 if unchanged else cp.float32
    output = cp.empty((layout.height, layout.width, len(locations)), dtype=output_dtype)
    element_count = int(output.size)
    block_count = (element_count + _TGA_THREADS_PER_BLOCK - 1) // _TGA_THREADS_PER_BLOCK
    _tga_read_kernel(unchanged)(
        (block_count,),
        (_TGA_THREADS_PER_BLOCK,),
        (
            device_pixels,
            output,
            np.int64(element_count),
            np.int32(layout.width),
            np.int32(layout.height),
            np.int32(layout.channels),
            np.int32(len(locations)),
            np.int32(layout.top_origin),
            device_channel_indices,
        ),
    )
    return Frame(
        data=output,
        colorspace=resolved_colorspace,
        gamma=resolved_gamma,
        channels=tuple(label for _, _, label in locations),
    )


def _validate_tga_write_layout(frame: Frame) -> tuple[str, ...]:
    channels = frame.channels
    channel_set = frozenset(channels)
    if len(channels) == 3 and channel_set == frozenset(("R", "G", "B")):
        return ("R", "G", "B")
    if len(channels) == 4 and channel_set == frozenset(("R", "G", "B", "A")):
        return ("R", "G", "B", "A")
    raise ValueError(
        _actionable_error(
            why="TGA output requires one unique RGB or RGBA channel set",
            what=f"channels={channels!r}",
            how="provide RGB or RGBA Frame channels without duplicates",
        )
    )


def _tga_write_data(frame: Frame) -> cp.ndarray:
    canonical = _validate_tga_write_layout(frame)
    source_frame = frame
    if frame.dtype != np.dtype(np.uint8):
        source_frame = recode_dtype(frame, dtype="uint8")
    source_indices = {label: source_frame.channels.index(label) for label in canonical}
    output = cp.empty(source_frame.shape, dtype=cp.uint8)
    element_count = int(output.size)
    block_count = (element_count + _TGA_THREADS_PER_BLOCK - 1) // _TGA_THREADS_PER_BLOCK
    _tga_write_kernel()(
        (block_count,),
        (_TGA_THREADS_PER_BLOCK,),
        (
            source_frame.data,
            output,
            np.int64(element_count),
            np.int32(len(canonical)),
            np.int32(source_indices["B"]),
            np.int32(source_indices["G"]),
            np.int32(source_indices["R"]),
            np.int32(source_indices.get("A", 0)),
        ),
    )
    return output


def _append_tga_raw_packets(output: bytearray, row: np.ndarray, start: int, end: int) -> None:
    while start < end:
        packet_end = min(start + 128, end)
        output.append(packet_end - start - 1)
        output.extend(row[start:packet_end].tobytes())
        start = packet_end


def _append_tga_run_packets(output: bytearray, pixel: np.ndarray, count: int) -> None:
    while count > 0:
        packet_count = min(128, count)
        output.append(0x80 | (packet_count - 1))
        output.extend(pixel.tobytes())
        count -= packet_count


def _encode_tga_rle(host: np.ndarray) -> bytes:
    output = bytearray()
    width = int(host.shape[1])
    channels = int(host.shape[2])
    for row in host:
        pixels = row.reshape(width, channels)
        equal_edges = np.all(pixels[1:] == pixels[:-1], axis=1)
        padded = np.zeros(equal_edges.size + 2, dtype=np.bool_)
        padded[1:-1] = equal_edges
        run_starts = np.flatnonzero(np.logical_and(padded[1:], np.logical_not(padded[:-1])))
        run_ends = np.flatnonzero(np.logical_and(np.logical_not(padded[1:]), padded[:-1]))
        cursor = 0
        for run_start_value, run_end_value in zip(run_starts, run_ends, strict=True):
            run_start = int(run_start_value)
            run_end = int(run_end_value) + 1
            _append_tga_raw_packets(output, pixels, cursor, run_start)
            _append_tga_run_packets(output, pixels[run_start], run_end - run_start)
            cursor = run_end
        _append_tga_raw_packets(output, pixels, cursor, width)
    return bytes(output)


def _encode_tga_file(frame: Frame) -> bytes:
    if frame.width > 65535 or frame.height > 65535:
        raise ValueError(
            _actionable_error(
                why="TGA dimensions must fit unsigned 16-bit header fields",
                what=f"width={frame.width}, height={frame.height}",
                how="write an image no larger than 65535 pixels on either axis",
            )
        )
    device_bgra = _tga_write_data(frame)
    host_bgra = cp.asnumpy(device_bgra)
    channels = int(host_bgra.shape[2])
    descriptor = 0x20 | (8 if channels == 4 else 0)
    header = _TGA_HEADER.pack(
        0,
        0,
        10,
        0,
        0,
        0,
        0,
        0,
        frame.width,
        frame.height,
        channels * 8,
        descriptor,
    )
    return header + _encode_tga_rle(host_bgra) + _TGA_FOOTER
