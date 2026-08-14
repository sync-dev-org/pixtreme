"""Image write dtype contracts and conversion."""

from __future__ import annotations

import numpy as np

from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import (
    Frame,
)
from pixtreme._core.vocabulary import _DTYPE_TOKENS
from pixtreme._values.cast import recode_dtype


def _is_exr_write_dtype(token: str) -> bool:
    dtype = np.dtype(token)
    return (dtype.kind == "f" and dtype.itemsize in (2, 4)) or (dtype.kind == "u" and dtype.itemsize == 4)


_EXR_WRITE_DTYPES = tuple(
    sorted(
        (token for token in _DTYPE_TOKENS if _is_exr_write_dtype(token)),
        key=lambda token: (np.dtype(token).kind != "f", np.dtype(token).itemsize),
    )
)

_WRITE_NATIVE_DTYPES = {
    "PNG": frozenset(("uint8", "uint16")),
    "JPEG": frozenset(("uint8",)),
    "TIFF": frozenset(("uint8", "uint16")),
    "EXR": frozenset(_EXR_WRITE_DTYPES),
    "JPEG2000": frozenset(("uint8", "uint16")),
    "WEBP": frozenset(("uint8",)),
    "BMP": frozenset(("uint8",)),
    "PNM": frozenset(("uint8", "uint16")),
    "HDR": frozenset(("float32",)),
    "DPX": frozenset(("float32",)),
}
_WRITE_DEFAULT_DTYPES = {
    "PNG": "uint8",
    "JPEG": "uint8",
    "TIFF": "uint8",
    "EXR": "float16",
    "JPEG2000": "uint8",
    "WEBP": "uint8",
    "BMP": "uint8",
    "PNM": "uint8",
    "HDR": "float32",
    "DPX": "float32",
}


def _prepare_write_frame(format_name: str, frame: Frame) -> Frame:
    if frame.dtype.name in _WRITE_NATIVE_DTYPES[format_name]:
        return frame
    return recode_dtype(frame, dtype=_WRITE_DEFAULT_DTYPES[format_name])


def _prepare_exr_write_frame(frame: Frame, *, dtype: str | None) -> Frame:
    if dtype is not None and (type(dtype) is not str or dtype not in _EXR_WRITE_DTYPES):
        raise ValueError(
            _actionable_error(
                why="EXR dtype is a closed, case-sensitive output storage token",
                what=f"received dtype={dtype!r}",
                how=f"pass one of {_EXR_WRITE_DTYPES!r}, or omit dtype to use the Frame-dependent default",
            )
        )
    resolved = ("uint32" if frame.dtype.name == "uint32" else "float16") if dtype is None else dtype
    if frame.dtype.name == resolved:
        return frame
    return recode_dtype(frame, dtype=resolved)
