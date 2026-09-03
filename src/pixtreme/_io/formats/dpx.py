"""DPX parsing, decoding, CUDA conversion, and encoding."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NoReturn, cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    ChannelInput,
    Frame,
)
from pixtreme._io.common import (
    _binary_stream,
    _read_exact,
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

_DPX_THREADS_PER_BLOCK = 256
_DPX_GENERIC_HEADER_SIZE = 1664
_DPX_INDUSTRY_HEADER_SIZE = 384
_DPX_DATA_OFFSET = 2048

_DPX_READ_KERNEL_TEMPLATE = r"""
typedef __OUTPUT_TYPE__ pixtreme_dpx_output_t;

extern "C" __global__ void pixtreme_dpx_read(
    const unsigned char* __restrict__ source,
    pixtreme_dpx_output_t* __restrict__ output,
    const long long element_count,
    const int width,
    const int source_channels,
    const int output_channels,
    const int row_stride,
    const int bit_depth,
    const int little_endian,
    const int* __restrict__ source_indices
) {
    const long long element = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= element_count) {
        return;
    }
    const int output_channel = (int)(element % output_channels);
    const long long pixel = element / output_channels;
    const int y = (int)(pixel / width);
    const int x = (int)(pixel % width);
    const int sample = x * source_channels + source_indices[output_channel];
    const long long row_offset = (long long)y * row_stride;
    unsigned int value;
    if (bit_depth == 8) {
        value = source[row_offset + sample];
    } else if (bit_depth == 10) {
        const long long byte_offset = row_offset + (sample / 3) * 4;
        unsigned int word;
        if (little_endian) {
            word = (unsigned int)source[byte_offset]
                | ((unsigned int)source[byte_offset + 1] << 8)
                | ((unsigned int)source[byte_offset + 2] << 16)
                | ((unsigned int)source[byte_offset + 3] << 24);
        } else {
            word = ((unsigned int)source[byte_offset] << 24)
                | ((unsigned int)source[byte_offset + 1] << 16)
                | ((unsigned int)source[byte_offset + 2] << 8)
                | (unsigned int)source[byte_offset + 3];
        }
        value = (word >> (22 - (sample % 3) * 10)) & 1023u;
    } else {
        const long long byte_offset = row_offset + sample * 2;
        const unsigned int word = little_endian
            ? ((unsigned int)source[byte_offset] | ((unsigned int)source[byte_offset + 1] << 8))
            : (((unsigned int)source[byte_offset] << 8) | (unsigned int)source[byte_offset + 1]);
        value = bit_depth == 12 ? word >> 4 : word;
    }
    output[element] = __OUTPUT_EXPRESSION__;
}
"""

_DPX_WRITE_KERNEL_SOURCE = r"""
__device__ __forceinline__ unsigned int pixtreme_dpx_quantize(
    const float* source,
    const long long sample,
    const int channels,
    const int* source_indices,
    const unsigned int maximum
) {
    const long long pixel = sample / channels;
    const int channel = (int)(sample % channels);
    const float value = fminf(fmaxf(source[pixel * channels + source_indices[channel]], 0.0f), 1.0f);
    return (unsigned int)floorf(value * (float)maximum + 0.5f);
}

extern "C" __global__ void pixtreme_dpx_write(
    const float* __restrict__ source,
    unsigned char* __restrict__ output,
    const long long unit_count,
    const int samples_per_row,
    const int units_per_row,
    const int channels,
    const int row_bytes,
    const int bit_depth,
    const int* __restrict__ source_indices
) {
    const long long unit = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (unit >= unit_count) {
        return;
    }
    const long long row = unit / units_per_row;
    const int row_unit = (int)(unit % units_per_row);
    const long long row_sample = row * samples_per_row;
    const long long output_offset = row * row_bytes;
    const unsigned int maximum = bit_depth == 8
        ? 255u
        : (bit_depth == 10 ? 1023u : (bit_depth == 12 ? 4095u : 65535u));
    if (bit_depth == 8) {
        output[output_offset + row_unit] = (unsigned char)pixtreme_dpx_quantize(
            source, row_sample + row_unit, channels, source_indices, maximum
        );
        return;
    }
    if (bit_depth == 10) {
        const int first = row_unit * 3;
        unsigned int values[3] = {0u, 0u, 0u};
        for (int lane = 0; lane < 3; ++lane) {
            if (first + lane < samples_per_row) {
                values[lane] = pixtreme_dpx_quantize(
                    source, row_sample + first + lane, channels, source_indices, maximum
                );
            }
        }
        const unsigned int word = (values[0] << 22) | (values[1] << 12) | (values[2] << 2);
        const long long byte_offset = output_offset + row_unit * 4;
        output[byte_offset] = (unsigned char)(word >> 24);
        output[byte_offset + 1] = (unsigned char)(word >> 16);
        output[byte_offset + 2] = (unsigned char)(word >> 8);
        output[byte_offset + 3] = (unsigned char)word;
        return;
    }
    unsigned int word = pixtreme_dpx_quantize(
        source, row_sample + row_unit, channels, source_indices, maximum
    );
    if (bit_depth == 12) {
        word <<= 4;
    }
    const long long byte_offset = output_offset + row_unit * 2;
    output[byte_offset] = (unsigned char)(word >> 8);
    output[byte_offset + 1] = (unsigned char)word;
}
"""


class _UnsupportedDpxError(ValueError):
    """Mark a valid DPX header configuration that is outside DPX scope."""


@dataclass(frozen=True)
class _DpxLayout:
    width: int
    height: int
    channels: int
    bit_depth: int
    little_endian: bool
    data_offset: int
    row_bytes: int
    row_stride: int
    transfer: int
    descriptor: int
    packing: int
    encoding: int


def _unsupported_dpx(*, why: str, what: str, how: str) -> NoReturn:
    raise _UnsupportedDpxError(_actionable_error(why=why, what=what, how=how))


def _parse_dpx_layout(source: Path | bytes) -> _DpxLayout:
    with _binary_stream(source) as stream:
        header = _read_exact(stream, 820)
        stream.seek(0, os.SEEK_END)
        actual_file_size = stream.tell()
    magic = header[:4]
    if magic == b"SDPX":
        endian = ">"
        little_endian = False
    elif magic == b"XPDS":
        endian = "<"
        little_endian = True
    else:
        raise ValueError(
            _actionable_error(
                why="the DPX magic does not declare big-endian SDPX or little-endian XPDS storage",
                what=f"magic={magic!r}",
                how="begin the file with b'SDPX' or b'XPDS'",
            )
        )

    file_offset = struct.unpack_from(f"{endian}I", header, 4)[0]
    total_file_size = struct.unpack_from(f"{endian}I", header, 16)[0]
    generic_header_length, industry_header_length, user_header_length = struct.unpack_from(f"{endian}III", header, 24)
    encryption_key = struct.unpack_from(f"{endian}I", header, 660)[0]
    orientation, elements = struct.unpack_from(f"{endian}HH", header, 768)
    width, height, signed = struct.unpack_from(f"{endian}III", header, 772)
    descriptor, transfer, bit_depth = header[800], header[801], header[803]
    packing, encoding = struct.unpack_from(f"{endian}HH", header, 804)
    data_offset, eol_padding = struct.unpack_from(f"{endian}II", header, 808)
    if encryption_key != 0xFFFFFFFF:
        _unsupported_dpx(
            why="encrypted DPX image data is outside the supported file boundary",
            what=f"encryption_key=0x{encryption_key:08x}",
            how="decrypt the image data and set the encryption key field to 0xFFFFFFFF",
        )
    if total_file_size != actual_file_size:
        raise ValueError(
            _actionable_error(
                why="the DPX file is truncated or its declared total file size is inconsistent",
                what=f"declared_total_file_size={total_file_size}, actual_file_size={actual_file_size}",
                how="make the total file size field equal the complete file length in bytes",
            )
        )
    if generic_header_length != _DPX_GENERIC_HEADER_SIZE:
        raise ValueError(
            _actionable_error(
                why="the DPX generic header length does not match the ST 268-1 generic section",
                what=f"generic_header_length={generic_header_length}",
                how=f"declare a {_DPX_GENERIC_HEADER_SIZE}-byte generic header",
            )
        )
    declared_header_size = generic_header_length + industry_header_length + user_header_length
    if file_offset < declared_header_size:
        raise ValueError(
            _actionable_error(
                why="the DPX core file offset overlaps the declared header sections",
                what=f"file_offset={file_offset}, declared_header_size={declared_header_size}",
                how="place image data at or after the generic, industry, and user header sections",
            )
        )
    if data_offset < declared_header_size:
        raise ValueError(
            _actionable_error(
                why="the DPX image element offset overlaps the declared header sections",
                what=f"element_offset={data_offset}, declared_header_size={declared_header_size}",
                how="place the image element after the generic, industry, and user header sections",
            )
        )
    if data_offset != file_offset:
        raise ValueError(
            _actionable_error(
                why="the single-element DPX core and image element offsets disagree",
                what=f"file_offset={file_offset}, element_offset={data_offset}",
                how="use the same byte offset for the file image data and sole image element",
            )
        )
    if file_offset > total_file_size:
        raise ValueError(
            _actionable_error(
                why="the DPX image data offset lies outside the declared file size",
                what=f"file_offset={file_offset}, total_file_size={total_file_size}",
                how="place image data within the complete file boundary",
            )
        )
    if orientation != 0:
        _unsupported_dpx(
            why="DPX orientation other than top-to-bottom left-to-right is outside the supported set",
            what=f"orientation={orientation}",
            how="store orientation=0 or reorient the image before encoding",
        )
    if elements != 1:
        _unsupported_dpx(
            why="DPX reading supports exactly one image element",
            what=f"image_elements={elements}",
            how="flatten the source to one RGB or RGBA image element",
        )
    if signed != 0:
        _unsupported_dpx(
            why="signed DPX sample storage is outside the supported unsigned integer set",
            what=f"data_sign={signed}",
            how="store unsigned integer RGB or RGBA samples with data sign 0",
        )
    if descriptor not in (50, 51):
        _unsupported_dpx(
            why="the DPX image element descriptor is not supported",
            what=f"descriptor={descriptor}",
            how="use descriptor 50 for RGB or 51 for RGBA",
        )
    if bit_depth not in (8, 10, 12, 16):
        _unsupported_dpx(
            why="the DPX bit depth is outside the supported integer set",
            what=f"bit_depth={bit_depth}",
            how="use 8, 10, 12, or 16-bit unsigned samples",
        )
    expected_packing = 1 if bit_depth in (10, 12) else 0
    if packing != expected_packing:
        _unsupported_dpx(
            why="the DPX packing method does not match the supported depth-specific layout",
            what=f"bit_depth={bit_depth}, packing={packing}",
            how=("use Method A filled packing=1 for 10/12-bit data or packing=0 for 8/16-bit data"),
        )
    if encoding != 0:
        _unsupported_dpx(
            why="RLE or another encoded DPX payload is outside the supported uncompressed set",
            what=f"encoding={encoding}",
            how="write an uncompressed image element with encoding=0",
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            _actionable_error(
                why="the DPX dimensions must both be positive",
                what=f"width={width}, height={height}",
                how="write positive pixels-per-line and lines-per-element fields",
            )
        )
    channels = 3 if descriptor == 50 else 4
    samples_per_row = width * channels
    if bit_depth == 8:
        row_bytes = samples_per_row
    elif bit_depth == 10:
        row_bytes = ((samples_per_row + 2) // 3) * 4
    else:
        row_bytes = samples_per_row * 2
    return _DpxLayout(
        width=width,
        height=height,
        channels=channels,
        bit_depth=bit_depth,
        little_endian=little_endian,
        data_offset=data_offset,
        row_bytes=row_bytes,
        row_stride=row_bytes + eol_padding,
        transfer=transfer,
        descriptor=descriptor,
        packing=packing,
        encoding=encoding,
    )


def _dpx_gamma(bit_depth: int, transfer: int) -> tuple[str, bool]:
    if transfer in (1, 3, 13):
        return "Cineon", True
    if transfer == 2:
        return "linear", True
    if transfer in (4, 5, 6, 7, 8, 9, 10):
        return "Rec.709", True
    fallback = "Cineon" if bit_depth == 10 else ("Rec.709" if bit_depth == 8 else "linear")
    return fallback, False


def _dpx_header(layout: _DpxLayout) -> ImageHeader:
    labels = ("R", "G", "B", "A")[: layout.channels]
    dtype = "uint8" if layout.bit_depth == 8 else "uint16"
    gamma, mappable = _dpx_gamma(layout.bit_depth, layout.transfer)
    return ImageHeader(
        format="DPX",
        width=layout.width,
        height=layout.height,
        parts=(_ImagePart(name="", channels=dict.fromkeys(labels, dtype)),),
        color=_ImageColorInfo(
            raw={
                "bit_depth": layout.bit_depth,
                "byte_order": "little" if layout.little_endian else "big",
                "descriptor": layout.descriptor,
                "transfer": layout.transfer,
                "packing": layout.packing,
                "encoding": layout.encoding,
                "data_offset": layout.data_offset,
                "row_stride": layout.row_stride,
            },
            colorspace="Rec.709",
            gamma=gamma,
            mappable=mappable,
        ),
    )


def _parse_dpx(source: Path | bytes) -> ImageHeader:
    return _dpx_header(_parse_dpx_layout(source))


def _dpx_pixel_bytes(data: bytes, layout: _DpxLayout) -> bytes:
    payload_size = layout.height * layout.row_stride
    payload_end = layout.data_offset + payload_size
    if payload_end > len(data):
        raise ValueError(
            _actionable_error(
                why="the uncompressed DPX pixel payload is truncated",
                what=f"requested_bytes={payload_size}, available_bytes={len(data) - layout.data_offset}",
                how="provide every packed scanline and declared end-of-line padding byte",
            )
        )
    return data[layout.data_offset : payload_end]


@lru_cache(maxsize=3)
def _dpx_read_kernel(output_dtype: str) -> cp.RawKernel:
    output_type = {"uint8": "unsigned char", "uint16": "unsigned short", "float32": "float"}[output_dtype]
    expression = "value"
    if output_dtype == "float32":
        expression = "(float)value / (float)((1u << bit_depth) - 1u)"
    source = _DPX_READ_KERNEL_TEMPLATE.replace("__OUTPUT_TYPE__", output_type).replace(
        "__OUTPUT_EXPRESSION__", expression
    )
    return cp.RawKernel(source, "pixtreme_dpx_read")


@lru_cache(maxsize=1)
def _dpx_write_kernel() -> cp.RawKernel:
    return cp.RawKernel(_DPX_WRITE_KERNEL_SOURCE, "pixtreme_dpx_write")


def _read_dpx_frame(
    path: Path,
    *,
    channels: ChannelInput | None,
    unchanged: bool,
    colorspace: str | None,
    gamma: str | None,
) -> Frame:
    try:
        data = path.read_bytes()
        layout = _parse_dpx_layout(data)
        pixel_bytes = _dpx_pixel_bytes(data, layout)
    except _UnsupportedDpxError:
        raise
    except (OSError, ValueError, struct.error) as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the DPX pixels could not be decoded: {error}",
                what=str(path),
                how="verify that the file has a complete supported header and uncompressed packed payload",
            )
        ) from error

    header = _dpx_header(layout)
    resolved_colorspace, resolved_gamma = _resolve_metadata(header, colorspace=colorspace, gamma=gamma)
    locations = _resolve_channel_locations(header, channels)
    source_indices = {label: index for index, label in enumerate(("R", "G", "B", "A")[: layout.channels])}
    selected_indices = [source_indices[channel] for _, channel, _ in locations]
    device_pixels = cp.asarray(np.frombuffer(pixel_bytes, dtype=np.uint8))
    device_channel_indices = cp.asarray(selected_indices, dtype=cp.int32)
    if unchanged:
        output_dtype = "uint8" if layout.bit_depth == 8 else "uint16"
        cupy_dtype = cp.uint8 if layout.bit_depth == 8 else cp.uint16
    else:
        output_dtype = "float32"
        cupy_dtype = cp.float32
    output = cp.empty((layout.height, layout.width, len(locations)), dtype=cupy_dtype)
    element_count = int(output.size)
    block_count = (element_count + _DPX_THREADS_PER_BLOCK - 1) // _DPX_THREADS_PER_BLOCK
    _dpx_read_kernel(output_dtype)(
        (block_count,),
        (_DPX_THREADS_PER_BLOCK,),
        (
            device_pixels,
            output,
            np.int64(element_count),
            np.int32(layout.width),
            np.int32(layout.channels),
            np.int32(len(locations)),
            np.int32(layout.row_stride),
            np.int32(layout.bit_depth),
            np.int32(layout.little_endian),
            device_channel_indices,
        ),
    )
    return Frame(
        data=output,
        colorspace=resolved_colorspace,
        gamma=resolved_gamma,
        channels=tuple(label for _, _, label in locations),
    )


def _validate_dpx_write_layout(frame: Frame) -> tuple[int, ...]:
    channels = frame.channels
    canonical: tuple[str, ...]
    if len(channels) == 3 and frozenset(channels) == frozenset(("R", "G", "B")):
        canonical = ("R", "G", "B")
    elif len(channels) == 4 and frozenset(channels) == frozenset(("R", "G", "B", "A")):
        canonical = ("R", "G", "B", "A")
    else:
        duplicates = tuple(label for label in dict.fromkeys(channels) if channels.count(label) > 1)
        raise ValueError(
            _actionable_error(
                why="DPX output requires one unique RGB or RGBA channel set",
                what=f"channels={channels!r}, duplicates={duplicates!r}",
                how="provide exactly one R, G, and B channel with optional A and no duplicates",
            )
        )
    if frame.width > 0xFFFFFFFF or frame.height > 0xFFFFFFFF:
        raise ValueError(
            _actionable_error(
                why="DPX dimensions must fit unsigned 32-bit header fields",
                what=f"width={frame.width}, height={frame.height}",
                how="resize the image so each dimension is at most 4294967295 pixels",
            )
        )
    return tuple(frame.channels.index(label) for label in canonical)


def _validate_dpx_bit_depth(bit_depth: int) -> int:
    if type(bit_depth) is not int or bit_depth not in (8, 10, 12, 16):
        raise ValueError(
            _actionable_error(
                why="DPX bit_depth is outside the supported integer set",
                what=f"bit_depth={bit_depth!r}",
                how="pass bit_depth=8, 10, 12, or 16",
            )
        )
    return bit_depth


def _dpx_transfer_from_gamma(gamma: str) -> int:
    if gamma in ("Cineon", "REDlogFilm"):
        return 1
    if gamma == "linear":
        return 2
    if gamma in (
        "S-Log",
        "S-Log2",
        "S-Log3",
        "ARRI-LogC3",
        "ARRI-LogC4",
        "Blackmagic-Film-Gen-5",
        "DaVinci-Intermediate",
        "RED-Log3G10",
    ):
        return 3
    return 6


def _dpx_write_data(frame: Frame, bit_depth: int) -> tuple[cp.ndarray, int]:
    source_indices = _validate_dpx_write_layout(frame)
    source_frame = _prepare_write_frame("DPX", frame)
    channels = len(source_indices)
    samples_per_row = source_frame.width * channels
    if bit_depth == 8:
        row_bytes = samples_per_row
        units_per_row = samples_per_row
    elif bit_depth == 10:
        units_per_row = (samples_per_row + 2) // 3
        row_bytes = units_per_row * 4
    else:
        units_per_row = samples_per_row
        row_bytes = samples_per_row * 2
    output = cp.empty((source_frame.height, row_bytes), dtype=cp.uint8)
    device_source_indices = cp.asarray(source_indices, dtype=cp.int32)
    unit_count = source_frame.height * units_per_row
    block_count = (unit_count + _DPX_THREADS_PER_BLOCK - 1) // _DPX_THREADS_PER_BLOCK
    _dpx_write_kernel()(
        (block_count,),
        (_DPX_THREADS_PER_BLOCK,),
        (
            source_frame.data,
            output,
            np.int64(unit_count),
            np.int32(samples_per_row),
            np.int32(units_per_row),
            np.int32(channels),
            np.int32(row_bytes),
            np.int32(bit_depth),
            device_source_indices,
        ),
    )
    return output, row_bytes


def _encode_dpx_file(frame: Frame, *, bit_depth: int) -> bytes:
    device_payload, row_bytes = _dpx_write_data(frame, bit_depth)
    host_payload = cast(bytes, cp.asnumpy(device_payload).tobytes())
    file_size = _DPX_DATA_OFFSET + len(host_payload)
    if file_size > 0xFFFFFFFF:
        raise ValueError(
            _actionable_error(
                why="the DPX output size does not fit its unsigned 32-bit file-size field",
                what=f"file_size={file_size}, row_bytes={row_bytes}",
                how="reduce the image dimensions or choose another output format",
            )
        )
    channels = len(frame.channels)
    packing = 1 if bit_depth in (10, 12) else 0
    header = bytearray(b"\xff" * _DPX_DATA_OFFSET)
    header[36:660] = bytes(624)
    for element in range(8):
        description_start = 820 + element * 72
        header[description_start : description_start + 32] = bytes(32)
    header[1432:1620] = bytes(188)
    header[1664:1712] = bytes(48)
    header[1732:1864] = bytes(132)
    header[1931] = 0
    header[:4] = b"SDPX"
    header[8:16] = b"V2.0\0\0\0\0"
    struct.pack_into(">I", header, 4, _DPX_DATA_OFFSET)
    struct.pack_into(">I", header, 16, file_size)
    struct.pack_into(">I", header, 20, 1)
    struct.pack_into(">I", header, 24, _DPX_GENERIC_HEADER_SIZE)
    struct.pack_into(">I", header, 28, _DPX_INDUSTRY_HEADER_SIZE)
    struct.pack_into(">I", header, 32, 0)
    struct.pack_into(">I", header, 660, 0xFFFFFFFF)
    struct.pack_into(">H", header, 768, 0)
    struct.pack_into(">H", header, 770, 1)
    struct.pack_into(">I", header, 772, frame.width)
    struct.pack_into(">I", header, 776, frame.height)
    struct.pack_into(">I", header, 780, 0)
    struct.pack_into(">I", header, 784, 0)
    struct.pack_into(">f", header, 788, 0.0)
    struct.pack_into(">I", header, 792, (1 << bit_depth) - 1)
    struct.pack_into(">f", header, 796, 1.0)
    header[800] = 50 if channels == 3 else 51
    header[801] = _dpx_transfer_from_gamma(frame.gamma)
    header[802] = 6
    header[803] = bit_depth
    struct.pack_into(">H", header, 804, packing)
    struct.pack_into(">H", header, 806, 0)
    struct.pack_into(">I", header, 808, _DPX_DATA_OFFSET)
    struct.pack_into(">I", header, 812, 0)
    struct.pack_into(">I", header, 816, 0)
    struct.pack_into(">I", header, 1408, 0)
    struct.pack_into(">I", header, 1412, 0)
    struct.pack_into(">I", header, 1424, frame.width)
    struct.pack_into(">I", header, 1428, frame.height)
    return bytes(header) + host_payload
