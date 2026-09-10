"""JPEG header parsing."""

from __future__ import annotations

import struct
from pathlib import Path

from pixtreme._core.errors import _actionable_error
from pixtreme._io.common import _binary_stream, _read_exact
from pixtreme._io.icc import (
    _MAX_JPEG_CARRIER_SIZE,
    _MAX_JPEG_SEGMENT_DATA,
    _icc_carrier_color,
    _IccCarrier,
)
from pixtreme._io.models import ImageHeader, _ImagePart
from pixtreme._io.orientation import _oriented_dimensions, _parse_exif_orientation

_JPEG_SOF_MARKERS = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))


def _parse_jpeg(source: Path | bytes) -> ImageHeader:
    with _binary_stream(source) as stream:
        signature = _read_exact(stream, 2)
        if signature != b"\xff\xd8":
            raise ValueError(
                _actionable_error(
                    why="the image does not have a valid JPEG start-of-image signature",
                    what=f"signature={signature!r}",
                    how="pass a JPEG file beginning with the b'\\xff\\xd8' marker",
                )
            )
        orientation = 1
        saw_exif = False
        image_values: tuple[int, int, int, int] | None = None
        icc_present = False
        icc_invalid = False
        icc_expected_count: int | None = None
        icc_total = 0
        icc_pieces: dict[int, bytes] = {}
        while True:
            byte = _read_exact(stream, 1)[0]
            while byte != 0xFF:
                byte = _read_exact(stream, 1)[0]
            marker = _read_exact(stream, 1)[0]
            while marker == 0xFF:
                marker = _read_exact(stream, 1)[0]
            if marker == 0xD9:
                break
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            length = struct.unpack(">H", _read_exact(stream, 2))[0]
            if length < 2:
                raise ValueError(
                    _actionable_error(
                        why="the JPEG marker length is smaller than its two-byte length field",
                        what=f"marker=0x{marker:02x}, length={length}",
                        how="pass a JPEG whose variable-length markers declare lengths of at least 2 bytes",
                    )
                )
            payload_size = length - 2
            if marker == 0xE2:
                prefix_size = min(payload_size, 14)
                prefix = _read_exact(stream, prefix_size)
                remaining = payload_size - prefix_size
                if prefix.startswith(b"ICC_PROFILE\x00"):
                    icc_present = True
                    if len(prefix) < 14:
                        icc_invalid = True
                        icc_pieces.clear()
                        icc_total = 0
                        stream.seek(remaining, 1)
                    else:
                        sequence, count = prefix[12], prefix[13]
                        data_size = remaining
                        invalid_segment = (
                            icc_invalid
                            or count == 0
                            or sequence == 0
                            or sequence > count
                            or data_size > _MAX_JPEG_SEGMENT_DATA
                            or (icc_expected_count is not None and count != icc_expected_count)
                            or sequence in icc_pieces
                            or icc_total + data_size > _MAX_JPEG_CARRIER_SIZE
                        )
                        if invalid_segment:
                            icc_invalid = True
                            icc_pieces.clear()
                            icc_total = 0
                            stream.seek(remaining, 1)
                        else:
                            if icc_expected_count is None:
                                icc_expected_count = count
                            icc_pieces[sequence] = _read_exact(stream, remaining)
                            icc_total += data_size
                else:
                    stream.seek(remaining, 1)
                payload = b""
            elif marker in _JPEG_SOF_MARKERS or marker in (0xE1, 0xDA):
                payload = _read_exact(stream, payload_size)
            else:
                payload = b""
                stream.seek(payload_size, 1)
            if marker in _JPEG_SOF_MARKERS:
                precision, height, width, component_count = struct.unpack(">BHHB", payload[:6])
                image_values = (precision, height, width, component_count)
            elif marker == 0xE1:
                if not saw_exif and payload.startswith(b"Exif\x00\x00"):
                    orientation = _parse_exif_orientation(payload, description="JPEG APP1 Exif")
                    saw_exif = True
            if marker == 0xDA:
                break

    if not icc_present:
        carrier = _IccCarrier(present=False, profile=None)
    elif (
        icc_invalid
        or icc_expected_count is None
        or len(icc_pieces) != icc_expected_count
        or set(icc_pieces) != set(range(1, icc_expected_count + 1))
    ):
        carrier = _IccCarrier(present=True, profile=None)
    else:
        carrier = _IccCarrier(
            present=True,
            profile=b"".join(icc_pieces[index] for index in range(1, icc_expected_count + 1)),
        )

    if image_values is None:
        raise ValueError(
            _actionable_error(
                why="the JPEG container has no supported start-of-frame marker",
                what="no baseline, extended, progressive, or lossless frame header was found",
                how="pass a complete JPEG image with a supported start-of-frame marker",
            )
        )
    precision, height, width, component_count = image_values
    labels: tuple[str, ...]
    if component_count == 1:
        labels = ("Y",)
    elif component_count == 3:
        labels = ("R", "G", "B")
    else:
        labels = tuple(f"channel-{index}" for index in range(component_count))
    dtype = "uint8" if precision <= 8 else "uint16"
    width, height = _oriented_dimensions(width, height, orientation)
    color = _icc_carrier_color(
        carrier,
        compatible=labels in (("R", "G", "B"), ("R", "G", "B", "A")),
    )
    return ImageHeader(
        format="JPEG",
        width=width,
        height=height,
        parts=(_ImagePart(name="", channels=dict.fromkeys(labels, dtype)),),
        color=color,
        orientation=orientation,
    )
