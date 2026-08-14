"""Public image read, decode, encode, and write orchestration."""

from __future__ import annotations

import os
import struct

import cupy as cp

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    ChannelInput,
    Frame,
)
from pixtreme._io.common import (
    _ENCODE_FORMAT_NAMES,
    _ENCODE_FORMAT_TOKENS,
    _path_and_format,
    _resolve_channel_locations,
    _resolve_metadata,
)
from pixtreme._io.dtype import _prepare_exr_write_frame
from pixtreme._io.formats.dpx import (
    _encode_dpx_file,
    _read_dpx_frame,
    _validate_dpx_bit_depth,
)
from pixtreme._io.formats.exr.selection import (
    _read_exr_pixels,
    _validate_exr_write_options,
    _write_exr,
)
from pixtreme._io.formats.hdr import (
    _encode_hdr_file,
    _read_hdr_frame,
)
from pixtreme._io.formats.nvimgcodec import (
    _decode_raster_frame,
    _encode_raster,
    _validate_raster_encode_options,
)
from pixtreme._io.formats.tga import (
    _encode_tga_file,
    _read_tga_frame,
)
from pixtreme._io.header import (
    _exr_header,
    _parse_exr,
    _read_raster_header,
    _sniff_raster_format,
    read_header,
)


def read_image(
    path: str | os.PathLike[str],
    *,
    channels: ChannelInput | None = None,
    unchanged: bool = False,
    colorspace: str | None = None,
    gamma: str | None = None,
) -> Frame:
    """Decode a supported image file into an HWC GPU Frame.

    ``path`` selects PNG, JPEG, TIFF, EXR, JPEG 2000, WebP, BMP, PNM, TGA, HDR, or DPX
    case-insensitively by extension.
    ``channels`` accepts a compact channel string or a sequence of labels and
    preserves the requested order. With no selection, raster files retain their
    RGB, RGBA, one-channel ``Y``, or two-channel ``YA`` layout; EXR selects
    unique R, G, B, and an optional A across its parts. Ambiguous EXR labels
    require ``part.channel`` qualification.

    The default converts ordinary integer storage, including DPX code values, to
    normalized float32 and promotes EXR HALF to float32. EXR UINT is instead
    converted literally to float32 without normalization. ``unchanged=True``
    preserves uint8, uint16, uint32, float16, or float32 storage where
    representable, including exact EXR UINT sample bits. HDR is natively decoded to float32,
    so ``unchanged=True`` returns the same values and dtype as its default read.
    Explicit ``colorspace`` and ``gamma``
    tokens override mapped file metadata, which overrides the fixed sRGB/srgb
    raster, ACES2065-1/linear EXR, or Rec.709/linear HDR defaults; these are metadata claims and do not
    transform pixel values.

    Codec-backed raster pixels decode through nvImageCodec into CUDA memory. TGA
    structure and RLE decode on the CPU before a GPU kernel produces the Frame.
    HDR likewise parses and expands RGBE scanlines on the CPU, then transfers
    only flat uint8 RGBE bytes for one-pass GPU ldexp decoding. EXPOSURE,
    PRIMARIES, and COLORCORR header variables are retained for header inspection
    but are not applied to pixels or metadata.
    DPX transfers only raw packed bytes and performs endian resolution, filled
    unpacking, normalization, and channel selection in one GPU pass.
    EXR scanline and tiled parts use the source-fixed native, custom-CPU, or GPU
    codec lane for their compression; multipart and sampled selections are
    materialized from part-local self-owned decoder views. Encoded bytes use
    :func:`decode_image`; EXR, TGA, HDR, and DPX are file-only formats. Missing files raise
    :class:`FileNotFoundError`; unsupported extensions, invalid tokens, channel
    selection, or incompatible unchanged EXR channels raise :class:`ValueError`;
    header or codec failures raise :class:`RuntimeError`.
    """
    file_path, format_name = _path_and_format(path, require_exists=True)
    if format_name == "TGA":
        return _read_tga_frame(
            file_path,
            channels=channels,
            unchanged=unchanged,
            colorspace=colorspace,
            gamma=gamma,
        )
    if format_name == "HDR":
        return _read_hdr_frame(
            file_path,
            channels=channels,
            colorspace=colorspace,
            gamma=gamma,
        )
    if format_name == "DPX":
        return _read_dpx_frame(
            file_path,
            channels=channels,
            unchanged=unchanged,
            colorspace=colorspace,
            gamma=gamma,
        )
    if format_name != "EXR":
        header = read_header(file_path)
        return _decode_raster_frame(
            file_path,
            header,
            channels=channels,
            unchanged=unchanged,
            colorspace=colorspace,
            gamma=gamma,
        )
    try:
        container = _parse_exr(file_path)
        header = _exr_header(container)
    except (OSError, ValueError, struct.error) as error:
        raise RuntimeError(
            _actionable_error(
                why=f"the EXR header could not be parsed: {error}",
                what=str(file_path),
                how="verify that the file is a valid, supported image",
            )
        ) from error

    resolved_colorspace, resolved_gamma = _resolve_metadata(header, colorspace=colorspace, gamma=gamma)
    try:
        locations = _resolve_channel_locations(header, channels)
    except ValueError as error:
        if channels is None and header.format == "EXR":
            raise ValueError(
                _actionable_error(
                    why="EXR default reading requires unique R, G, and B channels",
                    what=str(file_path),
                    how="inspect and select explicit channels with px.io.read_header(path)",
                )
            ) from error
        raise

    output = _read_exr_pixels(file_path, container, header, locations, unchanged=unchanged)
    output_labels = tuple(label for _, _, label in locations)
    return Frame(
        data=cp.ascontiguousarray(output),
        colorspace=resolved_colorspace,
        gamma=resolved_gamma,
        channels=output_labels,
    )


def decode_image(
    data: bytes,
    *,
    channels: ChannelInput | None = None,
    unchanged: bool = False,
    colorspace: str | None = None,
    gamma: str | None = None,
) -> Frame:
    """Decode supported raster bytes into an HWC GPU Frame.

    ``data`` accepts an encoded ``bytes`` payload in JPEG, PNG, TIFF, JPEG 2000,
    WebP, BMP, or PNM format and identifies it from its signature. EXR, TGA, HDR,
    and DPX are file-only formats. ``channels`` accepts a compact string or label
    sequence and preserves the requested order. ``unchanged=True`` preserves a
    supported native integer depth; the default normalizes ordinary integer
    storage to float32. Explicit ``colorspace`` and ``gamma`` metadata claims
    override embedded metadata and fixed raster defaults without transforming
    pixel values.

    Returns a new C-contiguous HWC GPU Frame with decoded pixels, selected channel
    labels, and resolved metadata. Unsupported or unidentifiable formats, malformed
    headers, invalid tokens, and invalid channel selections raise
    :class:`ValueError`; codec failures raise :class:`RuntimeError`.
    """
    format_name = _sniff_raster_format(data)
    header = _read_raster_header(data, format_name)
    return _decode_raster_frame(
        data,
        header,
        channels=channels,
        unchanged=unchanged,
        colorspace=colorspace,
        gamma=gamma,
    )


def _require_frame(frame: Frame, *, operation: str) -> None:
    if not isinstance(frame, Frame):
        raise ValueError(
            _actionable_error(
                why=f"{operation} requires a Frame",
                what=type(frame).__name__,
                how="pass a pixtreme Frame as the image input",
            )
        )


def encode_image(
    frame: Frame,
    *,
    format: str,
    quality: int | None = None,
    compression: str | None = None,
    compression_level: int | None = None,
    lossless: bool | None = None,
) -> bytes:
    """Encode a Frame as a supported raster byte stream.

    ``format`` is a case-sensitive image-format token. ``quality`` applies to
    JPEG and lossy WebP, ``compression`` is TIFF-only with ``"none"``/``"lzw"``,
    ``compression_level`` is PNG-only from 0 through 9, and ``lossless`` applies
    to JPEG 2000 and WebP. JPEG 2000 bytes use a JP2 container. Every format
    accepts all five Frame storage dtypes; native uint dtypes are preserved and
    other inputs are meaning-preservingly converted to uint8 on the GPU.
    """
    if type(format) is not str or format not in _ENCODE_FORMAT_TOKENS:
        raise ValueError(
            _actionable_error(
                why="format is not a supported encoded image token",
                what=repr(format),
                how=f"use one of {_ENCODE_FORMAT_TOKENS!r}",
            )
        )
    _require_frame(frame, operation="encode_image")
    format_name = _ENCODE_FORMAT_NAMES[format]
    return _encode_raster(
        frame,
        format_name,
        quality=quality,
        compression=compression,
        compression_level=compression_level,
        lossless=lossless,
    )


def write_image(
    path: str | os.PathLike[str],
    frame: Frame,
    *,
    quality: int | None = None,
    compression: str | None = None,
    compression_level: int | None = None,
    lossless: bool | None = None,
    dwa_level: float | None = None,
    bit_depth: int | None = None,
    dtype: str | None = None,
) -> None:
    """Write a Frame using the format selected from ``path``'s extension.

    Every format accepts uint8, uint16, uint32, float16, and float32 Frame storage.
    Native uint raster dtypes are preserved; other raster inputs are
    meaning-preservingly converted to uint8 on the GPU. EXR defaults to HALF for
    every Frame dtype except uint32, which defaults to native UINT. EXR-only
    ``dtype`` may explicitly select ``"float16"``, ``"float32"``, or ``"uint32"``;
    conversion follows :func:`pixtreme.values.recode_dtype`. TGA writes uint8 RLE
    under the same conversion meaning.
    HDR uses float32 as its native/default container and writes RGBE new-style RLE.
    DPX uses float32 as its native/default container and writes uncompressed big-endian data.
    No colorspace or gamma conversion is performed. Codec-backed raster output
    validates each format's channel layout and encodes through nvImageCodec from
    GPU data. EXR accepts unique channel
    labels, uses the source-fixed CPU/GPU backend table, writes chromaticities
    derived from ``frame.colorspace``, and marks ACES2065-1 containers.

    Metadata persistence is format-bound: raster output writes no public color
    vocabulary metadata, so ``frame.colorspace`` and ``frame.gamma`` are not
    preserved there. EXR persists only the chromaticities and ACES container
    flag derived from ``frame.colorspace``; ``frame.gamma`` is not stored there.
    HDR writes no EXPOSURE, PRIMARIES, COLORCORR, or public color-vocabulary metadata.
    DPX records a transfer characteristic derived from ``frame.gamma``.

    ``quality`` applies to JPEG and lossy WebP and accepts integers 1 through
    100. ``compression`` selects TIFF ``"none"``/``"lzw"`` or EXR ``"none"``,
    ``"rle"``, ``"zip"``, ``"zips"``, ``"piz"``, ``"pxr24"``, ``"b44"``,
    ``"b44a"``, ``"dwaa"``, or ``"dwab"`` according to the output format.
    EXR defaults to ``"zip"``. ``dwa_level`` applies only to DWAA/DWAB and
    defaults to 45.0 when omitted. ``compression_level`` is PNG-only and
    accepts integers 0 through 9. ``lossless`` applies to JPEG 2000 and WebP.
    ``bit_depth`` applies only to DPX, accepts 8, 10, 12, or 16, and defaults to 10.
    ``dtype`` applies only to EXR and accepts ``"float16"``, ``"float32"``,
    ``"uint32"``, or ``None``.
    Successful writes return ``None``.

    Unsupported extensions, channel layouts, or encode options raise
    :class:`ValueError`. Codec failures and unwritable output paths
    raise :class:`RuntimeError`; parent directories are not created. Use
    :func:`encode_image` when encoded raster bytes rather than a file are needed.
    """
    file_path, format_name = _path_and_format(path, require_exists=False)
    _require_frame(frame, operation="write_image")
    if format_name != "EXR" and dtype is not None:
        raise ValueError(
            _actionable_error(
                why="dtype is supported only for EXR output",
                what=f"dtype={dtype!r}, format={format_name}",
                how="omit dtype for this format or select an EXR (.exr) output path",
            )
        )
    if format_name != "DPX" and bit_depth is not None:
        raise ValueError(
            _actionable_error(
                why="bit_depth is supported only for DPX output",
                what=f"bit_depth={bit_depth!r}, format={format_name}",
                how="omit bit_depth or select a .dpx output path",
            )
        )
    if format_name == "EXR":
        write_frame = _prepare_exr_write_frame(frame, dtype=dtype)
        compression_token, resolved_dwa_level = _validate_exr_write_options(
            quality=quality,
            compression=compression,
            compression_level=compression_level,
            lossless=lossless,
            dwa_level=dwa_level,
        )
        _write_exr(file_path, write_frame, compression=compression_token, dwa_level=resolved_dwa_level)
    elif format_name == "DPX":
        if dwa_level is not None:
            raise ValueError(
                _actionable_error(
                    why="dwa_level is supported only for EXR DWAA and DWAB output",
                    what=f"dwa_level={dwa_level!r}, format={format_name}",
                    how="omit dwa_level or select EXR output with compression='dwaa' or compression='dwab'",
                )
            )
        _validate_raster_encode_options(
            format_name,
            quality=quality,
            compression=compression,
            compression_level=compression_level,
            lossless=lossless,
        )
        resolved_bit_depth = _validate_dpx_bit_depth(10 if bit_depth is None else bit_depth)
        payload = _encode_dpx_file(frame, bit_depth=resolved_bit_depth)
        try:
            file_path.write_bytes(payload)
        except OSError as error:
            raise RuntimeError(
                _actionable_error(
                    why=f"the encoded image could not be written: {error}",
                    what=str(file_path),
                    how="verify that the output directory exists and is writable",
                )
            ) from error
    elif format_name == "TGA":
        if dwa_level is not None:
            raise ValueError(
                _actionable_error(
                    why="dwa_level is supported only for EXR DWAA and DWAB output",
                    what=f"dwa_level={dwa_level!r}, format={format_name}",
                    how="omit dwa_level or select EXR output with compression='dwaa' or compression='dwab'",
                )
            )
        _validate_raster_encode_options(
            format_name,
            quality=quality,
            compression=compression,
            compression_level=compression_level,
            lossless=lossless,
        )
        payload = _encode_tga_file(frame)
        try:
            file_path.write_bytes(payload)
        except OSError as error:
            raise RuntimeError(
                _actionable_error(
                    why=f"the encoded image could not be written: {error}",
                    what=str(file_path),
                    how="verify that the output directory exists and is writable",
                )
            ) from error
    elif format_name == "HDR":
        if dwa_level is not None:
            raise ValueError(
                _actionable_error(
                    why="dwa_level is supported only for EXR DWAA and DWAB output",
                    what=f"dwa_level={dwa_level!r}, format={format_name}",
                    how="omit dwa_level or select EXR output with compression='dwaa' or compression='dwab'",
                )
            )
        _validate_raster_encode_options(
            format_name,
            quality=quality,
            compression=compression,
            compression_level=compression_level,
            lossless=lossless,
        )
        payload = _encode_hdr_file(frame)
        try:
            file_path.write_bytes(payload)
        except OSError as error:
            raise RuntimeError(
                _actionable_error(
                    why=f"the encoded image could not be written: {error}",
                    what=str(file_path),
                    how="verify that the output directory exists and is writable",
                )
            ) from error
    else:
        if dwa_level is not None:
            raise ValueError(
                _actionable_error(
                    why="dwa_level is supported only for EXR DWAA and DWAB output",
                    what=f"dwa_level={dwa_level!r}, format={format_name}",
                    how="omit dwa_level or select EXR output with compression='dwaa' or compression='dwab'",
                )
            )
        output_extension = file_path.suffix.lower()
        payload = _encode_raster(
            frame,
            format_name,
            quality=quality,
            compression=compression,
            compression_level=compression_level,
            lossless=lossless,
            output_extension=output_extension,
            jpeg2000_bitstream="j2k" if output_extension in (".j2k", ".j2c") else "jp2",
        )
        try:
            file_path.write_bytes(payload)
        except OSError as error:
            raise RuntimeError(
                _actionable_error(
                    why=f"the encoded image could not be written: {error}",
                    what=str(file_path),
                    how="verify that the output directory exists and is writable",
                )
            ) from error
