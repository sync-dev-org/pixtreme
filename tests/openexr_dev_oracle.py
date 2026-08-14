"""Explicit dev-only OpenEXR oracle import for EXR fixture and cross-backend tests."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol

import cupy as cp
import numpy as np

from pixtreme._io.common import _colorspace_chromaticities


class _OpenEXRProxy:
    def __getattr__(self, name: str) -> object:
        return getattr(import_module("OpenEXR"), name)


OpenEXR = _OpenEXRProxy()


class _FrameLike(Protocol):
    data: cp.ndarray
    channels: tuple[str, ...]
    colorspace: str


def write_frame(path: Path, frame: _FrameLike, *, compression: str, dwa_level: float | None) -> None:
    """Write a Frame through the independent OpenEXR dev oracle."""
    compression_values = {
        "none": OpenEXR.NO_COMPRESSION,
        "rle": OpenEXR.RLE_COMPRESSION,
        "zip": OpenEXR.ZIP_COMPRESSION,
        "zips": OpenEXR.ZIPS_COMPRESSION,
        "piz": OpenEXR.PIZ_COMPRESSION,
        "pxr24": OpenEXR.PXR24_COMPRESSION,
        "b44": OpenEXR.B44_COMPRESSION,
        "b44a": OpenEXR.B44A_COMPRESSION,
        "dwaa": OpenEXR.DWAA_COMPRESSION,
        "dwab": OpenEXR.DWAB_COMPRESSION,
    }
    host = cp.asnumpy(frame.data)
    channels = {label: np.ascontiguousarray(host[..., index]) for index, label in enumerate(frame.channels)}
    header: dict[str, object] = {
        "type": OpenEXR.scanlineimage,
        "compression": compression_values[compression],
        "chromaticities": _colorspace_chromaticities(frame.colorspace),
    }
    if dwa_level is not None:
        header["dwaCompressionLevel"] = dwa_level
    if frame.colorspace == "ACES2065-1":
        header["acesImageContainerFlag"] = 1
    OpenEXR.File(header, channels).write(str(path))


def read_frame(path: Path) -> dict[str, np.ndarray]:
    """Read and materialize every channel through the independent OpenEXR dev oracle."""
    channels = OpenEXR.File(str(path), separate_channels=True).channels()
    return {name: np.ascontiguousarray(channel.pixels) for name, channel in channels.items()}
