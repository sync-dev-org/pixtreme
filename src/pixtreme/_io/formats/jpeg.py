"""JPEG header parsing."""

from __future__ import annotations

import os
import struct
from pathlib import Path

from pixtreme._core.errors import _actionable_error
from pixtreme._io.common import _binary_stream, _empty_color, _read_exact
from pixtreme._io.models import ImageHeader, _ImagePart

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
        while True:
            byte = _read_exact(stream, 1)[0]
            while byte != 0xFF:
                byte = _read_exact(stream, 1)[0]
            marker = _read_exact(stream, 1)[0]
            while marker == 0xFF:
                marker = _read_exact(stream, 1)[0]
            if marker in _JPEG_SOF_MARKERS:
                length = struct.unpack(">H", _read_exact(stream, 2))[0]
                payload = _read_exact(stream, length - 2)
                precision, height, width, component_count = struct.unpack(">BHHB", payload[:6])
                labels: tuple[str, ...]
                if component_count == 1:
                    labels = ("Y",)
                elif component_count == 3:
                    labels = ("R", "G", "B")
                else:
                    labels = tuple(f"channel-{index}" for index in range(component_count))
                dtype = "uint8" if precision <= 8 else "uint16"
                return ImageHeader(
                    format="JPEG",
                    width=width,
                    height=height,
                    parts=(_ImagePart(name="", channels=dict.fromkeys(labels, dtype)),),
                    color=_empty_color(),
                )
            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
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
            stream.seek(length - 2, os.SEEK_CUR)
