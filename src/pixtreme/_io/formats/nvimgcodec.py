"""Shared nvImageCodec decode and encode helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import ChannelInput, Frame
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_kernel import _cuda_load_expression, _cuda_storage_type, _cuda_store_expression
from pixtreme._io.common import (
    _DTYPE_MAXIMA,
    _ENCODE_CODEC_EXTENSIONS,
    _TIFF_COMPRESSION_TOKENS,
    _empty_color,
    _resolve_channel_locations,
    _resolve_metadata,
)
from pixtreme._io.dtype import _WRITE_DEFAULT_DTYPES, _WRITE_NATIVE_DTYPES
from pixtreme._io.models import ImageHeader, _ImagePart
from pixtreme._values.cast import _recode_dtype_expression, _recode_dtype_parameter

_RASTER_REPACK_THREADS_PER_BLOCK = 256


def _new_raster_header(
    format_name: str,
    *,
    width: int,
    height: int,
    component_count: int,
    dtype: str,
) -> ImageHeader:
    labels = {1: ("Y",), 3: ("R", "G", "B"), 4: ("R", "G", "B", "A")}.get(component_count)
    if labels is None or width <= 0 or height <= 0:
        raise ValueError(
            _actionable_error(
                why=f"the {format_name} header has unsupported components or non-positive dimensions",
                what=f"components={component_count}, width={width}, height={height}",
                how="use one Y, three RGB, or four RGBA components with positive dimensions",
            )
        )
    return ImageHeader(
        format=format_name,
        width=width,
        height=height,
        parts=(_ImagePart(name="", channels=dict.fromkeys(labels, dtype)),),
        color=_empty_color(),
    )


def _read_raster_pixels(source: Path | bytes, header: ImageHeader) -> cp.ndarray:
    source_value: str | bytes = str(source) if isinstance(source, Path) else source
    source_description = str(source) if isinstance(source, Path) else f"{len(source)} encoded bytes"
    try:
        from nvidia import nvimgcodec

        code_stream = nvimgcodec.CodeStream(source_value)
        params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.UNCHANGED, allow_any_depth=True)
        decoded = nvimgcodec.Decoder().decode(code_stream, params=params)
        if decoded is None:
            raise RuntimeError(
                _actionable_error(
                    why="nvImageCodec returned no decoded image",
                    what=source_description,
                    how="verify that the encoded content is complete and uses a supported raster format",
                )
            )
        device_image = decoded.cuda()
        if device_image is None:
            raise RuntimeError(
                _actionable_error(
                    why="the decoded image could not be transferred to CUDA memory",
                    what=source_description,
                    how="verify NVIDIA GPU availability, CUDA compatibility, and the encoded image",
                )
            )
        array = cp.from_dlpack(device_image)
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why=f"nvImageCodec could not decode the image: {error}",
                what=source_description,
                how="verify that the encoded content is complete and not corrupt",
            )
        ) from error
    expected_channels = len(header.parts[0].channels)
    if array.ndim != 3 or array.shape[2] != expected_channels:
        raise RuntimeError(
            _actionable_error(
                why="decoded channel layout differs from the parsed file header",
                what=f"decoded shape {array.shape!r}, expected {expected_channels} channels",
                how="use a PNG, JPEG, or TIFF with RGB/RGBA/gray channels",
            )
        )
    return cast(cp.ndarray, array)


@lru_cache(maxsize=128)
def _raster_repack_kernel(
    source_dtype: str,
    destination_dtype: str,
    source_channels: int,
    output_channels: int,
    conversion: str,
    index_mode: str,
) -> cp.RawKernel:
    if conversion == "copy" and source_dtype != destination_dtype:
        raise ValueError("raster copy conversion requires matching dtypes")
    assignments: list[str] = []
    for output_channel in range(output_channels):
        source_index = (
            f"pixel * {source_channels} + channel_index_{output_channel}"
            if index_mode == "hwc"
            else f"source_base + channel_index_{output_channel} * channel_stride"
        )
        if conversion == "copy":
            stored = f"source[{source_index}]"
        elif conversion == "normalize":
            loaded = _cuda_load_expression(source_dtype, f"source[{source_index}]")
            stored = _cuda_store_expression(destination_dtype, f"({loaded} / parameter)")
        elif conversion == "cast":
            loaded = _cuda_load_expression(source_dtype, f"source[{source_index}]")
            stored = _cuda_store_expression(destination_dtype, loaded)
        elif conversion == "recode":
            stored = _recode_dtype_expression(source_dtype, destination_dtype, f"source[{source_index}]")
        else:
            raise ValueError(f"unsupported raster repack conversion: {conversion}")
        assignments.append(f"    destination[pixel * {output_channels} + {output_channel}] = {stored};")

    strided_prelude = ""
    if index_mode == "strided":
        strided_prelude = """
    const long long row = pixel / width;
    const long long column = pixel - row * width;
    const long long source_base = row * row_stride + column * column_stride;
"""
    channel_parameters = ",\n    ".join(
        f"const int channel_index_{output_channel}" for output_channel in range(output_channels)
    )
    source = f"""
#include <cuda_fp16.h>

extern "C" __global__ void pixtreme_raster_repack(
    const {_cuda_storage_type(source_dtype)}* __restrict__ source,
    {_cuda_storage_type(destination_dtype)}* __restrict__ destination,
    const long long pixel_count,
    const long long width,
    const long long row_stride,
    const long long column_stride,
    const long long channel_stride,
    const float parameter,
    {channel_parameters}
) {{
    const long long pixel = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (pixel >= pixel_count) {{
        return;
    }}
{strided_prelude}
{chr(10).join(assignments)}
}}
"""
    return cp.RawKernel(source, "pixtreme_raster_repack")


def _repack_raster_data(
    source: cp.ndarray,
    channel_indices: tuple[int, ...],
    *,
    destination_dtype: str,
    conversion: str,
    parameter: np.float32 = np.float32(0.0),
) -> cp.ndarray:
    output = cp.empty((*source.shape[:2], len(channel_indices)), dtype=destination_dtype)
    pixel_count = int(source.shape[0] * source.shape[1])
    if pixel_count == 0:
        return output
    source_dtype = np.dtype(source.dtype).name
    source_channels = int(source.shape[2])
    if any(index < 0 or index >= source_channels for index in channel_indices):
        raise ValueError("raster repack channel index is outside the source layout")
    itemsize = int(source.dtype.itemsize)
    row_stride, column_stride, channel_stride = (np.int64(stride // itemsize) for stride in source.strides)
    index_mode = "hwc" if source.flags.c_contiguous else "strided"
    block_count = (pixel_count + _RASTER_REPACK_THREADS_PER_BLOCK - 1) // _RASTER_REPACK_THREADS_PER_BLOCK
    _raster_repack_kernel(
        source_dtype,
        destination_dtype,
        source_channels,
        len(channel_indices),
        conversion,
        index_mode,
    )(
        (block_count,),
        (_RASTER_REPACK_THREADS_PER_BLOCK,),
        (
            source,
            output,
            np.int64(pixel_count),
            np.int64(source.shape[1]),
            row_stride,
            column_stride,
            channel_stride,
            parameter,
            *(np.int32(index) for index in channel_indices),
        ),
    )
    return output


def _decode_raster_data(decoded: cp.ndarray, indices: tuple[int, ...], *, unchanged: bool) -> cp.ndarray:
    dtype_name = np.dtype(decoded.dtype).name
    if unchanged or dtype_name == "float32":
        return _repack_raster_data(
            decoded,
            indices,
            destination_dtype=dtype_name,
            conversion="copy",
        )
    if dtype_name in _DTYPE_MAXIMA:
        return _repack_raster_data(
            decoded,
            indices,
            destination_dtype="float32",
            conversion="normalize",
            parameter=_DTYPE_MAXIMA[dtype_name],
        )
    if dtype_name == "float16":
        return _repack_raster_data(
            decoded,
            indices,
            destination_dtype="float32",
            conversion="cast",
        )
    raise ValueError(
        _actionable_error(
            why="decoded raster dtype is outside the supported Frame dtype set",
            what=dtype_name,
            how="use an 8/16-bit integer or float32 source image",
        )
    )


def _decode_raster_frame(
    source: Path | bytes,
    header: ImageHeader,
    *,
    channels: ChannelInput | None,
    unchanged: bool,
    colorspace: str | None,
    gamma: str | None,
) -> Frame:
    resolved_colorspace, resolved_gamma = _resolve_metadata(header, colorspace=colorspace, gamma=gamma)
    locations = _resolve_channel_locations(header, channels)
    decoded = _read_raster_pixels(source, header)
    indices = tuple(tuple(header.parts[0].channels).index(channel) for _, channel, _ in locations)
    output = _decode_raster_data(decoded, indices, unchanged=unchanged)
    output_labels = tuple(label for _, _, label in locations)
    return Frame(
        data=output,
        colorspace=resolved_colorspace,
        gamma=resolved_gamma,
        channels=output_labels,
    )


def _validate_new_format_layout(
    format_name: str,
    channels: tuple[str, ...],
    *,
    output_extension: str | None,
) -> None:
    allowed = {
        "JPEG2000": (("Y",), ("R", "G", "B"), ("R", "G", "B", "A")),
        "WEBP": (("R", "G", "B"),),
        "BMP": (("Y",), ("R", "G", "B")),
        "PNM": (("Y",), ("R", "G", "B")),
    }.get(format_name)
    if allowed is None:
        return
    channel_set = frozenset(channels)
    accepted = any(len(channels) == len(layout) and channel_set == frozenset(layout) for layout in allowed)
    if not accepted:
        raise ValueError(
            _actionable_error(
                why=f"{format_name} output does not support the Frame channel layout",
                what=f"channels={channels!r}",
                how=f"use one of {allowed!r} without duplicate channel labels",
            )
        )
    if output_extension == ".pgm" and channel_set != frozenset(("Y",)):
        raise ValueError(
            _actionable_error(
                why="PGM output requires a one-channel Y Frame",
                what=f"channels={channels!r}",
                how="use a Y Frame or choose .ppm/.pnm for RGB output",
            )
        )
    if output_extension == ".ppm" and channel_set != frozenset(("R", "G", "B")):
        raise ValueError(
            _actionable_error(
                why="PPM output requires an RGB Frame",
                what=f"channels={channels!r}",
                how="use an RGB Frame or choose .pgm/.pnm for grayscale output",
            )
        )


def _raster_write_data(frame: Frame, *, format_name: str) -> cp.ndarray:
    canonical = {
        frozenset(("R", "G", "B")): ("R", "G", "B"),
        frozenset(("R", "G", "B", "A")): ("R", "G", "B", "A"),
        frozenset(("Y",)): ("Y",),
        frozenset(("Y", "A")): ("Y", "A"),
    }.get(frozenset(frame.channels))
    if canonical is None or len(canonical) != len(frame.channels):
        raise ValueError(
            _actionable_error(
                why="raster output requires RGB, RGBA, Y, or YA channel labels",
                what=repr(frame.channels),
                how="select and order standard output channels before writing",
            )
        )
    indices = tuple(frame.channels.index(label) for label in canonical)
    source_dtype = frame.dtype.name
    destination_dtype = (
        source_dtype if source_dtype in _WRITE_NATIVE_DTYPES[format_name] else _WRITE_DEFAULT_DTYPES[format_name]
    )
    return _repack_raster_data(
        frame.data,
        indices,
        destination_dtype=destination_dtype,
        conversion="copy" if source_dtype == destination_dtype else "recode",
        parameter=_recode_dtype_parameter(source_dtype, destination_dtype),
    )


def _validate_raster_encode_options(
    format_name: str,
    *,
    quality: int | None,
    compression: str | None,
    compression_level: int | None,
    lossless: bool | None,
) -> tuple[str, float] | None:
    if quality is not None and format_name not in ("JPEG", "WEBP"):
        raise ValueError(
            _actionable_error(
                why="quality is supported only for JPEG and WebP output",
                what=f"quality={quality!r}, format={format_name}",
                how="omit quality or select JPEG or WebP output",
            )
        )
    if quality is not None and (type(quality) is not int or not 1 <= quality <= 100):
        raise ValueError(
            _actionable_error(
                why="JPEG and WebP quality must be an integer from 1 through 100",
                what=repr(quality),
                how="pass quality=1 through quality=100",
            )
        )
    if compression is not None and format_name != "TIFF":
        raise ValueError(
            _actionable_error(
                why="compression is supported only for TIFF output",
                what=f"compression={compression!r}, format={format_name}",
                how="omit compression or select TIFF output",
            )
        )
    if compression is not None:
        compression = _normalized_closed_token(
            compression,
            axis="compression",
            accepted=_TIFF_COMPRESSION_TOKENS,
            why="TIFF compression is not a supported token",
            how=f"use one of the canonical tokens {_TIFF_COMPRESSION_TOKENS!r}",
        )
    if compression_level is not None and format_name != "PNG":
        raise ValueError(
            _actionable_error(
                why="compression_level is supported only for PNG output",
                what=f"compression_level={compression_level!r}, format={format_name}",
                how="omit compression_level or select PNG output",
            )
        )
    if compression_level is not None and (type(compression_level) is not int or not 0 <= compression_level <= 9):
        raise ValueError(
            _actionable_error(
                why="PNG compression_level must be an integer from 0 through 9",
                what=repr(compression_level),
                how="pass compression_level=0 through compression_level=9",
            )
        )
    if lossless is not None and type(lossless) is not bool:
        raise ValueError(
            _actionable_error(
                why="lossless must be an exact bool or None",
                what=repr(lossless),
                how="pass lossless=True, lossless=False, or omit it",
            )
        )
    if lossless is not None and format_name not in ("JPEG2000", "WEBP"):
        raise ValueError(
            _actionable_error(
                why="lossless is supported only for JPEG 2000 and WebP output",
                what=f"lossless={lossless!r}, format={format_name}",
                how="omit lossless or select JPEG 2000 or WebP output",
            )
        )
    if format_name == "WEBP" and quality is not None and lossless is True:
        raise ValueError(
            _actionable_error(
                why="WebP quality conflicts with lossless=True",
                what=f"quality={quality}, lossless={lossless}",
                how="omit quality for lossless WebP or use lossless=False for lossy quality control",
            )
        )
    if quality is not None:
        return "quality", float(quality)
    if compression is not None:
        return "lossless", float(_TIFF_COMPRESSION_TOKENS.index(compression))
    if compression_level is not None:
        return "lossless", float(compression_level)
    if lossless is True:
        return "lossless", 0.0
    return None


def _encode_raster(
    frame: Frame,
    format_name: str,
    *,
    quality: int | None,
    compression: str | None,
    compression_level: int | None,
    lossless: bool | None,
    output_extension: str | None = None,
    jpeg2000_bitstream: str = "jp2",
) -> bytes:
    _validate_new_format_layout(format_name, frame.channels, output_extension=output_extension)
    parameter = _validate_raster_encode_options(
        format_name,
        quality=quality,
        compression=compression,
        compression_level=compression_level,
        lossless=lossless,
    )
    try:
        from nvidia import nvimgcodec

        parameter_options: dict[str, object] = {}
        if parameter is not None:
            parameter_kind, parameter_value = parameter
            quality_type = nvimgcodec.QUALITY if parameter_kind == "quality" else nvimgcodec.LOSSLESS
            parameter_options.update(quality_type=quality_type, quality_value=parameter_value)
        if format_name == "JPEG2000":
            bitstream_type = nvimgcodec.J2K if jpeg2000_bitstream == "j2k" else nvimgcodec.JP2
            parameter_options["jpeg2k_encode_params"] = nvimgcodec.Jpeg2kEncodeParams(bitstream_type=bitstream_type)
        params = nvimgcodec.EncodeParams(**parameter_options) if parameter_options else None
        encoded = nvimgcodec.Encoder().encode(
            _raster_write_data(frame, format_name=format_name),
            _ENCODE_CODEC_EXTENSIONS[format_name],
            params=params,
        )
        if encoded is None:
            raise RuntimeError(
                _actionable_error(
                    why="nvImageCodec returned no encoded code stream",
                    what=f"format={format_name}, dtype={frame.dtype}, channels={frame.channels!r}",
                    how="verify the output format, dtype, and standard channel labels",
                )
            )
        return bytes(encoded)
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(
            _actionable_error(
                why=f"nvImageCodec could not encode the image: {error}",
                what=f"format={format_name}, dtype={frame.dtype}, channels={frame.channels!r}",
                how="verify the output format, dtype, and standard channel labels",
            )
        ) from error
