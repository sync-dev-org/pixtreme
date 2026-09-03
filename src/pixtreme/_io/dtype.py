"""Image write dtype contracts and conversion."""

from __future__ import annotations

from typing import cast

import numpy as np

from pixtreme._core.frame import (
    Frame,
)
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.vocabulary import _DTYPE_TOKENS, Dtype
from pixtreme._values.cast import recode_dtype


def _is_exr_write_dtype(token: str) -> bool:
    dtype = np.dtype(token)
    return (dtype.kind == "f" and dtype.itemsize in (2, 4)) or (dtype.kind == "u" and dtype.itemsize == 4)


_EXR_WRITE_DTYPES = cast(
    tuple[Dtype, ...],
    tuple(
        sorted(
            (token for token in _DTYPE_TOKENS if _is_exr_write_dtype(token)),
            key=lambda token: (np.dtype(token).kind != "f", np.dtype(token).itemsize),
        )
    ),
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
_WRITE_DEFAULT_DTYPES: dict[str, Dtype] = {
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


def _prepare_exr_write_frame(frame: Frame, *, dtype: Dtype | None) -> Frame:
    checked_dtype = (
        None
        if dtype is None
        else _normalized_closed_token(
            dtype,
            axis="dtype",
            accepted=_EXR_WRITE_DTYPES,
            how=f"pass one of the canonical tokens {_EXR_WRITE_DTYPES!r}, or omit dtype to use the default",
        )
    )
    resolved: Dtype = (
        ("uint32" if frame.dtype.name == "uint32" else "float16") if checked_dtype is None else checked_dtype
    )
    if frame.dtype.name == resolved:
        return frame
    return recode_dtype(frame, dtype=resolved)
