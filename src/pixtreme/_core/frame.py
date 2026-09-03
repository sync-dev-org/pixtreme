"""Frame construction and boundary protocols."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TypeVar, cast

import cupy as cp
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from pixtreme._core.errors import _actionable_error as _actionable_error
from pixtreme._core.validation import _normalized_closed_token
from pixtreme._core.value_domain import _float32_conversion_guidance
from pixtreme._core.vocabulary import (
    _COLORSPACE_TOKENS as _COLORSPACE_TOKENS,
)
from pixtreme._core.vocabulary import (
    _DTYPE_TOKENS as _DTYPE_TOKEN_NAMES,
)
from pixtreme._core.vocabulary import (
    _GAMMA_TOKENS as _GAMMA_TOKENS,
)
from pixtreme._core.vocabulary import _LAYOUT_TOKENS as _LAYOUT_TOKENS
from pixtreme._core.vocabulary import (
    _MATRIX_TOKENS as _MATRIX_TOKENS,
)
from pixtreme._core.vocabulary import (
    Colorspace,
    Gamma,
    Matrix,
)

_CHANNEL_LABELS = ("R", "G", "B", "H", "S", "V", "A", "Y", "Cb", "Cr", "Z")

_GREEDY_CHANNEL_LABELS = tuple(sorted(_CHANNEL_LABELS, key=len, reverse=True))
_DTYPE_TOKENS = {token: np.dtype(token) for token in _DTYPE_TOKEN_NAMES}
_ACCEPTED_DTYPES = frozenset(_DTYPE_TOKENS.values())

ChannelInput = str | Sequence[str]


@lru_cache(maxsize=128)
def _compact_channels(value: str) -> tuple[str, ...]:
    if not value:
        raise ValueError(
            _actionable_error(
                why="a channels compact string cannot describe any channels when it is empty",
                what=f"received channels={value!r}",
                how="pass a non-empty compact string such as channels='RGB' or a sequence such as channels=('R', 'G', 'B')",
            )
        )
    result: list[str] = []
    offset = 0
    while offset < len(value):
        label = next((candidate for candidate in _GREEDY_CHANNEL_LABELS if value.startswith(candidate, offset)), None)
        if label is None:
            raise ValueError(
                _actionable_error(
                    why="channels compact strings accept only known case-sensitive labels",
                    what=f"received channels={value!r}, with an unknown label at offset {offset}",
                    how=f"use labels from {_CHANNEL_LABELS!r}, or pass application-defined labels as a sequence",
                )
            )
        result.append(label)
        offset += len(label)
    return tuple(result)


def channels(value: ChannelInput) -> tuple[str, ...]:
    """Normalize channel labels to their canonical tuple representation.

    Parameters
    ----------
    value:
        A compact string of known, case-sensitive labels parsed by greedy
        longest match, or a Sequence of non-empty strings. Sequence input also
        permits application-defined labels unknown to the compact vocabulary.

    Returns
    -------
    tuple[str, ...]
        A non-empty tuple preserving the normalized label order.

    Raises
    ------
    ValueError
        If a compact string is empty or contains an unknown label, or if a
        sequence is empty or contains a non-string or empty label.
    """
    if isinstance(value, str):
        return _compact_channels(value)
    if not isinstance(value, Sequence):
        raise ValueError(
            _actionable_error(
                why="channels must be declared as a compact string or a sequence of string labels",
                what=f"received channels={value!r} ({type(value).__module__}.{type(value).__qualname__})",
                how="pass channels='RGB' or channels=('R', 'G', 'B')",
            )
        )
    sequence_result = tuple(value)
    if not sequence_result or any(not isinstance(label, str) or not label for label in sequence_result):
        raise ValueError(
            _actionable_error(
                why="channels sequences must contain one or more non-empty string labels",
                what=f"received channels={value!r}",
                how="remove empty/non-string entries and pass at least one label, such as channels=('R', 'G', 'B')",
            )
        )
    return sequence_result


_normalize_channels = channels


_Token = TypeVar("_Token", bound=str)


def _validate_token(value: object, *, axis: str, accepted: tuple[_Token, ...]) -> _Token:
    return _normalized_closed_token(value, axis=axis, accepted=accepted)


class Frame(BaseModel):
    """A mutable metadata-bearing HWC image stored on an NVIDIA GPU.

    ``data`` is always a C-contiguous CuPy array. Keep this Frame alive for as
    long as a consumer uses a raw pointer obtained from its data. Frame is a
    DLPack producer: consumer arguments and stream information are passed
    through unchanged to the underlying CuPy array.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    data: cp.ndarray
    colorspace: Colorspace
    gamma: Gamma
    channels: tuple[str, ...]
    matrix: Matrix | None = None

    @field_validator("data")
    @classmethod
    def _validate_data(cls, value: cp.ndarray, info: ValidationInfo) -> cp.ndarray:
        if not isinstance(value, cp.ndarray):
            raise ValueError(
                _actionable_error(
                    why="Frame data must use device-resident cupy.ndarray storage",
                    what=f"received Frame data as {type(value).__module__}.{type(value).__qualname__}",
                    how="pass a cupy.ndarray, or construct through px.io.from_array(cp.asarray(data), ...)",
                )
            )
        if value.ndim != 3:
            raise ValueError(
                _actionable_error(
                    why="Frame data uses HWC rank 3 storage",
                    what=f"received Frame data shape={value.shape!r}",
                    how="reshape or transpose data to (height, width, channels) before constructing Frame",
                )
            )
        if any(size < 1 for size in value.shape):
            raise ValueError(
                _actionable_error(
                    why="Frame data height, width, and channel dimensions must all be nonempty",
                    what=f"received Frame data shape={value.shape!r}",
                    how="pass HWC data whose height, width, and channel count are each at least 1",
                )
            )
        if np.dtype(value.dtype) not in _ACCEPTED_DTYPES:
            raise ValueError(
                _actionable_error(
                    why="Frame data dtype must be one of float32, float16, uint8, uint16, or uint32",
                    what=f"received Frame data dtype={value.dtype!s}",
                    how="convert data with data.astype(cp.float32) or another supported dtype before constructing Frame",
                )
            )
        contiguous = cp.ascontiguousarray(value)
        assigned_channels = info.data.get("channels")
        if isinstance(assigned_channels, tuple) and len(assigned_channels) != contiguous.shape[2]:
            raise ValueError(
                _actionable_error(
                    why="Frame data and channels metadata must describe the same channel count",
                    what=(
                        f"received Frame data shape={contiguous.shape!r} with {contiguous.shape[2]} channels and "
                        f"channels={assigned_channels!r} with {len(assigned_channels)} labels"
                    ),
                    how=(
                        f"assign data with {len(assigned_channels)} channels, or construct a new Frame with "
                        f"{contiguous.shape[2]} channel labels"
                    ),
                )
            )
        return contiguous

    @field_validator("colorspace", mode="before")
    @classmethod
    def _validate_colorspace(cls, value: object) -> Colorspace:
        return _validate_token(value, axis="colorspace", accepted=_COLORSPACE_TOKENS)

    @field_validator("gamma", mode="before")
    @classmethod
    def _validate_gamma(cls, value: object) -> Gamma:
        return _validate_token(value, axis="gamma", accepted=_GAMMA_TOKENS)

    @field_validator("matrix", mode="before")
    @classmethod
    def _validate_matrix(cls, value: object) -> Matrix | None:
        return None if value is None else _validate_token(value, axis="matrix", accepted=_MATRIX_TOKENS)

    @field_validator("channels", mode="before")
    @classmethod
    def _normalize_channels(cls, value: ChannelInput) -> tuple[str, ...]:
        return _normalize_channels(value)

    @field_validator("channels")
    @classmethod
    def _validate_channel_count(cls, value: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        data = info.data.get("data")
        if isinstance(data, cp.ndarray) and len(value) != data.shape[2]:
            raise ValueError(
                _actionable_error(
                    why="channels metadata and Frame data must describe the same channel count",
                    what=f"received channels={value!r} with {len(value)} labels for Frame data shape={data.shape!r}",
                    how=f"pass exactly {data.shape[2]} channel labels to match Frame data.shape[2]",
                )
            )
        return value

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        """HWC data shape."""
        return cast(tuple[int, int, int], self.data.shape)

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Data storage dtype."""
        return cast(np.dtype[np.generic], self.data.dtype)

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object:
        """Delegate DLPack export to data, preserving consumer arguments."""
        return cast(
            object,
            self.data.__dlpack__(stream=stream, max_version=max_version, dl_device=dl_device, copy=copy),
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        """Delegate the DLPack device query to data."""
        return cast(tuple[int, int], self.data.__dlpack_device__())

    def __repr__(self) -> str:
        """Represent known channel labels compactly and preserve custom labels explicitly."""
        channel_repr: str | tuple[str, ...]
        if all(label in _CHANNEL_LABELS for label in self.channels):
            channel_repr = "".join(self.channels)
        else:
            channel_repr = self.channels
        return (
            f"Frame(data={self.data!r}, colorspace={self.colorspace!r}, "
            f"gamma={self.gamma!r}, channels={channel_repr!r}, matrix={self.matrix!r})"
        )


def _new_frame(frame: Frame, output: cp.ndarray) -> Frame:
    return Frame(
        data=output,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=frame.matrix,
    )


def _validate_frame(frame: object, *, operation: str) -> Frame:
    if not isinstance(frame, Frame):
        raise ValueError(
            _actionable_error(
                why=f"{operation} operates on metadata-bearing Frame values only",
                what=f"received {type(frame).__module__}.{type(frame).__qualname__}",
                how=f"construct a Frame with px.io.from_array before calling px.{operation}",
            )
        )
    return frame


def _validate_float32_frame(frame: object, *, operation: str) -> Frame:
    checked_frame = _validate_frame(frame, operation=operation)
    dtype = np.dtype(checked_frame.dtype)
    if dtype != np.dtype(np.float32):
        raise ValueError(
            _actionable_error(
                why=f"{operation} requires float32 Frame data",
                what=f"received Frame data dtype {dtype.name}",
                how=_float32_conversion_guidance(dtype),
            )
        )
    return checked_frame
