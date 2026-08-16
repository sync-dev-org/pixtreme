"""GPU-resident one- and three-dimensional LUT values."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import cast

import cupy as cp
import numpy as np

from pixtreme._core.errors import _actionable_error

_DEFAULT_DOMAIN_MIN = (0.0, 0.0, 0.0)
_DEFAULT_DOMAIN_MAX = (1.0, 1.0, 1.0)


def _domain(value: object, *, name: str, owner: str = "Lut") -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        values: tuple[object, ...] = ()
    else:
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            values = ()
    if len(values) != 3 or any(isinstance(item, bool) or not isinstance(item, Real) for item in values):
        raise ValueError(
            _actionable_error(
                why=f"{owner} {name} must contain exactly three real RGB values",
                what=f"received {name}={value!r}",
                how=f"pass {name}=(red, green, blue)",
            )
        )
    real_values = cast(tuple[Real, Real, Real], values)
    try:
        result = (float(real_values[0]), float(real_values[1]), float(real_values[2]))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            _actionable_error(
                why=f"{owner} {name} values must be convertible to finite Python floats",
                what=f"failed to normalize {name} ({type(error).__name__})",
                how=f"pass {name} as three finite values representable by Python float",
            )
        ) from None
    if not all(np.isfinite(result)):
        raise ValueError(
            _actionable_error(
                why=f"{owner} {name} values must be finite",
                what=f"received {name}={result!r}",
                how="replace NaN or infinite domain endpoints with finite values",
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class Lut:
    """A validated 3D RGB lookup grid stored on an NVIDIA GPU.

    ``data`` has shape ``(N, N, N, 3)`` indexed by red, green, and blue,
    with float32 values and ``N >= 2``. The array is retained by reference,
    without a construction-time copy; callers are responsible for avoiding
    mutation while the LUT is in use. ``domain_min`` and ``domain_max`` map
    input RGB values onto the grid independently by channel.
    """

    data: cp.ndarray
    domain_min: tuple[float, float, float] = _DEFAULT_DOMAIN_MIN
    domain_max: tuple[float, float, float] = _DEFAULT_DOMAIN_MAX

    def __post_init__(self) -> None:
        if not isinstance(self.data, cp.ndarray):
            raise ValueError(
                _actionable_error(
                    why="Lut data must reside on the GPU as a cupy.ndarray",
                    what=f"received {type(self.data).__module__}.{type(self.data).__qualname__}",
                    how="transfer a float32 (N, N, N, 3) grid with cupy.asarray",
                )
            )
        if self.data.ndim != 4 or self.data.shape[-1] != 3:
            raise ValueError(
                _actionable_error(
                    why="Lut data must have shape (N, N, N, 3)",
                    what=f"received shape {self.data.shape!r}",
                    how="provide a rank-four RGB lookup grid",
                )
            )
        size = int(self.data.shape[0])
        if size < 2 or self.data.shape[:3] != (size, size, size):
            raise ValueError(
                _actionable_error(
                    why="Lut data must be a cubic grid with N at least 2",
                    what=f"received shape {self.data.shape!r}",
                    how="provide data shaped (N, N, N, 3) with N >= 2",
                )
            )
        if np.dtype(self.data.dtype) != np.dtype(np.float32):
            raise ValueError(
                _actionable_error(
                    why="Lut data uses the float32 working representation",
                    what=f"received dtype {self.data.dtype!s}",
                    how="convert the lookup grid with data.astype(cupy.float32)",
                )
            )
        domain_min = _domain(self.domain_min, name="domain_min")
        domain_max = _domain(self.domain_max, name="domain_max")
        if not all(lower < upper for lower, upper in zip(domain_min, domain_max)):
            raise ValueError(
                _actionable_error(
                    why="Lut domain_min must be strictly less than domain_max in every RGB channel",
                    what=f"received domain_min={domain_min!r}, domain_max={domain_max!r}",
                    how="choose increasing finite endpoints for red, green, and blue",
                )
            )
        object.__setattr__(self, "domain_min", domain_min)
        object.__setattr__(self, "domain_max", domain_max)


@dataclass(frozen=True, slots=True)
class Lut1D:
    """A validated per-channel RGB lookup curve stored on an NVIDIA GPU.

    ``data`` has shape ``(N, 3)`` with float32 values and ``N >= 2``. Each
    column is the independent red, green, or blue output curve. The array is
    retained by reference without a construction-time copy; callers are
    responsible for avoiding mutation while the LUT is in use. Per-channel
    ``domain_min`` and ``domain_max`` values map inputs onto sample indices.
    Table outputs may be non-monotonic or outside 0..1.
    """

    data: cp.ndarray
    domain_min: tuple[float, float, float] = _DEFAULT_DOMAIN_MIN
    domain_max: tuple[float, float, float] = _DEFAULT_DOMAIN_MAX

    def __post_init__(self) -> None:
        if not isinstance(self.data, cp.ndarray):
            raise ValueError(
                _actionable_error(
                    why="Lut1D data must reside on the GPU as a cupy.ndarray",
                    what=f"received {type(self.data).__module__}.{type(self.data).__qualname__}",
                    how="transfer a float32 (N, 3) curve table with cupy.asarray",
                )
            )
        if self.data.ndim != 2 or self.data.shape[1] != 3:
            raise ValueError(
                _actionable_error(
                    why="Lut1D data must have shape (N, 3)",
                    what=f"received shape {self.data.shape!r}",
                    how="provide a rank-two table with independent red, green, and blue columns",
                )
            )
        if self.data.shape[0] < 2:
            raise ValueError(
                _actionable_error(
                    why="Lut1D data requires at least two samples",
                    what=f"received shape {self.data.shape!r}",
                    how="provide data shaped (N, 3) with N >= 2",
                )
            )
        if np.dtype(self.data.dtype) != np.dtype(np.float32):
            raise ValueError(
                _actionable_error(
                    why="Lut1D data uses the float32 working representation",
                    what=f"received dtype {self.data.dtype!s}",
                    how="convert the curve table with data.astype(cupy.float32)",
                )
            )
        domain_min = _domain(self.domain_min, name="domain_min", owner="Lut1D")
        domain_max = _domain(self.domain_max, name="domain_max", owner="Lut1D")
        if not all(lower < upper for lower, upper in zip(domain_min, domain_max)):
            raise ValueError(
                _actionable_error(
                    why="Lut1D domain_min must be strictly less than domain_max in every RGB channel",
                    what=f"received domain_min={domain_min!r}, domain_max={domain_max!r}",
                    how="choose increasing finite endpoints for red, green, and blue",
                )
            )
        object.__setattr__(self, "domain_min", domain_min)
        object.__setattr__(self, "domain_max", domain_max)
