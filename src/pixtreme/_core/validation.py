"""Shared semantic primitives for validating public-boundary values."""

from __future__ import annotations

import math
from numbers import Real
from typing import TypeVar, cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error

_Token = TypeVar("_Token")


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be a finite real number",
                what=f"received {name}={value!r}",
                how=f"pass a finite int or float for {name}",
            )
        )
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(
            _actionable_error(
                why=f"{name} must be finite",
                what=f"received {name}={value!r}",
                how=f"pass a finite int or float for {name}",
            )
        )
    return resolved


def _positive_real(value: object, *, name: str) -> float:
    resolved = _finite_real(value, name=name)
    if resolved <= 0.0:
        raise ValueError(
            _actionable_error(
                why=f"{name} must be greater than 0",
                what=f"received {name}={value!r}",
                how=f"pass a finite positive real number for {name}",
            )
        )
    return resolved


def _bounded_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    why: str,
    how: str,
) -> float:
    resolved = _finite_real(value, name=name)
    if (minimum is not None and resolved < minimum) or (maximum is not None and resolved > maximum):
        raise ValueError(
            _actionable_error(
                why=why,
                what=f"received {name}={value!r}",
                how=how,
            )
        )
    return resolved


def _strict_bool(value: object, *, name: str, why: str, how: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            _actionable_error(
                why=why,
                what=f"received {name}={value!r}",
                how=how,
            )
        )
    return value


def _host_array(value: object, *, why: str, how: str) -> np.ndarray:
    if isinstance(value, cp.ndarray):
        return cast(np.ndarray, cp.asnumpy(value))
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            _actionable_error(
                why=why,
                what=f"received value={value!r}",
                how=how,
            )
        ) from error


def _finite_pair(value: object, *, name: str) -> tuple[float, float]:
    why = f"{name} must be one (x, y) coordinate pair"
    how = f"pass {name} as a two-element finite real sequence"
    try:
        array = _host_array(value, why=why, how=how)
    except ValueError:
        array = np.asarray((), dtype=np.float32)
    if array.shape != (2,):
        raise ValueError(
            _actionable_error(
                why=why,
                what=f"received {name}={value!r}",
                how=how,
            )
        )
    return (
        _finite_real(array[0].item() if isinstance(array[0], np.generic) else array[0], name=f"{name}[0]"),
        _finite_real(array[1].item() if isinstance(array[1], np.generic) else array[1], name=f"{name}[1]"),
    )


def _positive_scalar_or_pair(
    value: object,
    *,
    name: str,
    why: str,
    how: str,
) -> tuple[float, float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        resolved = _positive_real(value, name=name)
        return resolved, resolved
    try:
        array = _host_array(value, why=why, how=how)
    except ValueError:
        array = np.asarray((), dtype=np.float32)
    if array.shape != (2,):
        raise ValueError(
            _actionable_error(
                why=why,
                what=f"received {name}={value!r}",
                how=how,
            )
        )
    return (
        _positive_real(array[0].item() if isinstance(array[0], np.generic) else array[0], name=f"{name}[0]"),
        _positive_real(array[1].item() if isinstance(array[1], np.generic) else array[1], name=f"{name}[1]"),
    )


def _closed_token_error(
    value: object,
    *,
    axis: str,
    accepted: tuple[str, ...],
    why: str | None,
    how: str | None,
) -> ValueError:
    return ValueError(
        _actionable_error(
            why=f"{axis} is a closed, case-sensitive token axis" if why is None else why,
            what=f"received {axis}={value!r}",
            how=f"pass one of {accepted!r}" if how is None else how,
        )
    )


def _closed_token(
    value: _Token,
    *,
    axis: str,
    accepted: tuple[str, ...],
    why: str | None = None,
    how: str | None = None,
) -> _Token:
    if value not in accepted:
        raise _closed_token_error(value, axis=axis, accepted=accepted, why=why, how=how)
    return value


def _normalized_closed_token(
    value: object,
    *,
    axis: str,
    accepted: tuple[str, ...],
    why: str | None = None,
    how: str | None = None,
) -> str:
    return str(_closed_token(value, axis=axis, accepted=accepted, why=why, how=how))


def _closed_str_token(
    value: object,
    *,
    axis: str,
    accepted: tuple[str, ...],
    why: str | None = None,
    how: str | None = None,
) -> str:
    if not isinstance(value, str) or value not in accepted:
        raise _closed_token_error(value, axis=axis, accepted=accepted, why=why, how=how)
    return value
