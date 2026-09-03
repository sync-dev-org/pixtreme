"""GPU box convolution filter."""

from __future__ import annotations

from pixtreme._core.border import _resolve_border
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame, _validate_float32_frame
from pixtreme._core.vocabulary import Border
from pixtreme._filter.box import _convolve_box
from pixtreme._filter.common import _validate_odd_size


def _resolve_box_size(size: object) -> tuple[int, int]:
    if type(size) is int:
        resolved = _validate_odd_size(size, operation="filter.convolve_box")
        return resolved, resolved
    if type(size) is not tuple or len(size) != 2:
        raise ValueError(
            _actionable_error(
                why="convolve_box size must be one int or a two-element height-width tuple",
                what=f"received size={size!r}",
                how="pass one positive odd int, or pass (height, width) with two positive odd ints",
            )
        )
    height = _validate_odd_size(size[0], operation="filter.convolve_box height")
    width = _validate_odd_size(size[1], operation="filter.convolve_box width")
    return height, width


def convolve_box(
    frame: Frame,
    *,
    size: int | tuple[int, int],
    normalize: bool,
    border: Border = "mirror",
    border_value: float | None = None,
) -> Frame:
    """Apply a rectangular moving sum or moving mean independently per channel.

    ``size`` is one positive odd integer or ``(height, width)``. ``normalize``
    must be passed explicitly: true returns the window mean and false returns the
    window sum. Border defaults to ``mirror`` (edge-excluding reflection);
    ``replicate`` clamps to the edge and ``wrap`` uses periodic indices.
    ``constant`` uses ``border_value`` for every virtual pixel outside the image;
    ``border_value`` is required with ``constant`` and forbidden for every other
    border. The fp32 calculation does not clamp negative values or values above
    1 and always returns new storage, including size 1.
    """
    checked_frame = _validate_float32_frame(frame, operation="filter.convolve_box")
    height, width = _resolve_box_size(size)
    if type(normalize) is not bool:
        raise ValueError(
            _actionable_error(
                why="convolve_box normalize is an explicit boolean choice",
                what=f"received normalize={normalize!r}",
                how="pass normalize=True for a mean or normalize=False for a sum",
            )
        )
    checked_border, checked_border_value = _resolve_border(border, border_value)
    return _convolve_box(
        checked_frame,
        height=height,
        width=width,
        normalize=normalize,
        border=checked_border,
        border_value=checked_border_value,
    )
