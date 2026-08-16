"""Reference-white resolution and physical device white-point simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pixtreme._color.semantics import _validate_rgb_transfer_frame
from pixtreme._color.transform import _transform_data
from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.validation import _closed_str_token, _finite_pair
from pixtreme._core.vocabulary import _REFERENCE_WHITE_TOKENS, ReferenceWhite

_Float64Matrix = NDArray[np.float64]

_REFERENCE_WHITE_COORDINATES: Mapping[ReferenceWhite, tuple[float, float]] = {
    "d65": (0.3127, 0.3290),
    "d93": (0.2831, 0.2971),
    "d50": (0.3457, 0.3585),
    "aces": (0.32168, 0.33767),
}


def _error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _validate_xy(value: object, *, name: str) -> tuple[float, float]:
    x, y = _finite_pair(value, name=name)
    if x <= 0.0 or y <= 0.0 or x + y >= 1.0:
        raise _error(
            why=f"{name} must be a CIE 1931 xy chromaticity inside the positive XYZ domain",
            what=f"received {name}=({x!r}, {y!r})",
            how=f"pass {name}=(x, y) with x > 0, y > 0, and x + y < 1",
        )
    return x, y


def _resolve_reference_white(value: object, *, name: str) -> tuple[float, float]:
    if isinstance(value, str):
        token = _closed_str_token(
            value,
            axis=name,
            accepted=_REFERENCE_WHITE_TOKENS,
            why=f"{name} must be a documented reference-white token or CIE 1931 xy pair",
            how=f"pass {name} as one of {_REFERENCE_WHITE_TOKENS!r} or as a valid (x, y) pair",
        )
        return _REFERENCE_WHITE_COORDINATES[cast(ReferenceWhite, token)]
    return _validate_xy(value, name=name)


def _xy_to_xyz(white: tuple[float, float]) -> _Float64Matrix:
    x, y = (np.float64(component) for component in white)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.asarray((x / y, np.float64(1.0), (np.float64(1.0) - x - y) / y), dtype=np.float64)


def _device_matrix(colorspace: str, white: tuple[float, float]) -> _Float64Matrix:
    primaries = _COLORSPACE_DEFINITIONS[colorspace][0]
    unscaled = np.asarray(
        (
            tuple(x / y for x, y in primaries),
            (1.0, 1.0, 1.0),
            tuple((1.0 - x - y) / y for x, y in primaries),
        ),
        dtype=np.float64,
    )
    white_xyz = _xy_to_xyz(white)
    try:
        scale = np.linalg.solve(unscaled, white_xyz)
        matrix = np.asarray(unscaled @ np.diag(scale), dtype=np.float64)
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise _error(
            why="the reference white and Frame primaries must construct a nonsingular RGB device matrix",
            what=f"received white={white!r}, colorspace={colorspace!r}",
            how="choose a valid xy whose normalized RGB device matrix is finite and nonsingular",
        ) from error
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(white_xyz).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(matrix).all()
        or not np.isfinite(inverse).all()
    ):
        raise _error(
            why="the reference white and Frame primaries must construct a finite nonsingular RGB device matrix",
            what=f"received white={white!r}, colorspace={colorspace!r}",
            how="choose a valid xy whose normalized RGB device matrix is finite and nonsingular",
        )
    return matrix


def _white_point_simulation_matrix(
    colorspace: str,
    input_white: tuple[float, float],
    output_white: tuple[float, float],
) -> _Float64Matrix:
    input_device = _device_matrix(colorspace, input_white)
    output_device = _device_matrix(colorspace, output_white)
    try:
        matrix = np.asarray(np.linalg.inv(output_device) @ input_device, dtype=np.float64)
    except np.linalg.LinAlgError as error:
        raise _error(
            why="input and output device matrices must compose a nonsingular RGB transform",
            what=(f"received input_white={input_white!r}, output_white={output_white!r}, colorspace={colorspace!r}"),
            how="choose input and output whites that each construct a finite nonsingular device matrix",
        ) from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise _error(
            why="input and output device matrices must compose a finite 3x3 RGB transform",
            what=(f"received input_white={input_white!r}, output_white={output_white!r}, colorspace={colorspace!r}"),
            how="choose input and output whites that each construct a finite nonsingular device matrix",
        )
    return matrix


def white_point_simulation(
    frame: Frame,
    *,
    input_white: ReferenceWhite | Sequence[float] | None = None,
    output_white: ReferenceWhite | Sequence[float],
) -> Frame:
    """Physically re-encode absolute colorimetry between display whites.

    The input and output devices use the Frame colorspace primaries normalized
    to their respective white points. The Frame transfer is decoded, the
    absolute device matrix is applied, and the original transfer is encoded in
    one GPU pass. This operation does not perform chromatic adaptation.
    """
    validated_frame = _validate_rgb_transfer_frame(frame, operation="white_point_simulation")
    resolved_input = (
        _COLORSPACE_DEFINITIONS[validated_frame.colorspace][1]
        if input_white is None
        else _resolve_reference_white(input_white, name="input_white")
    )
    resolved_output = _resolve_reference_white(output_white, name="output_white")
    if resolved_input == resolved_output:
        # Equal whites still must construct a finite nonsingular device matrix.
        _device_matrix(validated_frame.colorspace, resolved_input)
        output_data = validated_frame.data.copy()
    else:
        matrix = _white_point_simulation_matrix(
            validated_frame.colorspace,
            resolved_input,
            resolved_output,
        )
        try:
            output_data = _transform_data(
                validated_frame.data,
                validated_frame.channels,
                input_gamma=validated_frame.gamma,
                output_gamma=validated_frame.gamma,
                matrix=np.asarray(matrix, dtype=np.float32),
            )
        except Exception as error:
            raise _error(
                why="white_point_simulation could not execute its fused GPU pixel pass",
                what=f"backend raised {type(error).__module__}.{type(error).__qualname__}: {error}",
                how="verify the CUDA runtime and retry with a valid float32 Frame on an available NVIDIA GPU",
            ) from error
    return Frame(
        data=output_data,
        colorspace=validated_frame.colorspace,
        gamma=validated_frame.gamma,
        channels=validated_frame.channels,
        matrix=None,
    )
