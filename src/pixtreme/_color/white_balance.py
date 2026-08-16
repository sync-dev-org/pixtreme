"""Explicit white-point and Temperature / Tint chromatic adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pixtreme._color.semantics import _validate_rgb_transfer_frame
from pixtreme._color.transform import _RGB_TO_XYZ, _transform_data
from pixtreme._color.white_point import _resolve_reference_white, _validate_xy
from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS
from pixtreme._core.errors import _actionable_error
from pixtreme._core.frame import Frame
from pixtreme._core.validation import _closed_str_token, _finite_real
from pixtreme._core.vocabulary import (
    _CHROMATIC_ADAPTATION_TOKENS,
    ChromaticAdaptation,
    ReferenceWhite,
)

_Float64Matrix = NDArray[np.float64]

_CAT_BASES: Mapping[ChromaticAdaptation, _Float64Matrix] = {
    "bradford": np.asarray(
        (
            (0.8951000, 0.2664000, -0.1614000),
            (-0.7502000, 1.7135000, 0.0367000),
            (0.0389000, -0.0685000, 1.0296000),
        ),
        dtype=np.float64,
    ),
    "cat02": np.asarray(
        (
            (0.7328, 0.4296, -0.1624),
            (-0.7036, 1.6975, 0.0061),
            (0.0030, 0.0136, 0.9834),
        ),
        dtype=np.float64,
    ),
    "cat16": np.asarray(
        (
            (0.401288, 0.650173, -0.051461),
            (-0.250268, 1.204414, 0.045854),
            (-0.002079, 0.048952, 0.953127),
        ),
        dtype=np.float64,
    ),
    "von-kries": np.asarray(
        (
            (0.4002400, 0.7076000, -0.0808100),
            (-0.2263000, 1.1653200, 0.0457000),
            (0.0000000, 0.0000000, 0.9182200),
        ),
        dtype=np.float64,
    ),
}

# Adobe DNG SDK 1.7.1 dng_temperature.cpp kTempTable. Records are
# (reciprocal megakelvin, CIE 1960 u, CIE 1960 v, iso-temperature slope).
_DNG_TEMPERATURE_TABLE = (
    (0.0, 0.18006, 0.26352, -0.24341),
    (10.0, 0.18066, 0.26589, -0.25479),
    (20.0, 0.18133, 0.26846, -0.26876),
    (30.0, 0.18208, 0.27119, -0.28539),
    (40.0, 0.18293, 0.27407, -0.30470),
    (50.0, 0.18388, 0.27709, -0.32675),
    (60.0, 0.18494, 0.28021, -0.35156),
    (70.0, 0.18611, 0.28342, -0.37915),
    (80.0, 0.18740, 0.28668, -0.40955),
    (90.0, 0.18880, 0.28997, -0.44278),
    (100.0, 0.19032, 0.29326, -0.47888),
    (125.0, 0.19462, 0.30141, -0.58204),
    (150.0, 0.19962, 0.30921, -0.70471),
    (175.0, 0.20525, 0.31647, -0.84901),
    (200.0, 0.21142, 0.32312, -1.0182),
    (225.0, 0.21807, 0.32909, -1.2168),
    (250.0, 0.22511, 0.33439, -1.4512),
    (275.0, 0.23247, 0.33904, -1.7298),
    (300.0, 0.24010, 0.34308, -2.0637),
    (325.0, 0.24702, 0.34655, -2.4681),
    (350.0, 0.25591, 0.34951, -2.9641),
    (375.0, 0.26400, 0.35200, -3.5814),
    (400.0, 0.27218, 0.35407, -4.3633),
    (425.0, 0.28039, 0.35577, -5.3762),
    (450.0, 0.28863, 0.35714, -6.7262),
    (475.0, 0.29685, 0.35823, -8.5955),
    (500.0, 0.30505, 0.35907, -11.324),
    (525.0, 0.31320, 0.35968, -15.628),
    (550.0, 0.32129, 0.36011, -23.325),
    (575.0, 0.32931, 0.36038, -40.770),
    (600.0, 0.33724, 0.36051, -116.45),
)

_MINIMUM_TEMPERATURE = 1_000_000.0 / 600.0


def _error(*, why: str, what: str, how: str) -> ValueError:
    return ValueError(_actionable_error(why=why, what=what, how=how))


def _validate_cat(value: object) -> ChromaticAdaptation:
    return cast(
        ChromaticAdaptation,
        _closed_str_token(
            value,
            axis="cat",
            accepted=_CHROMATIC_ADAPTATION_TOKENS,
            why="cat selects one documented chromatic adaptation transform",
            how=f"pass cat as one of {_CHROMATIC_ADAPTATION_TOKENS!r}",
        ),
    )


def _xy_to_xyz(xy: tuple[float, float]) -> _Float64Matrix:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _chromatic_adaptation_matrix(
    input_white: tuple[float, float],
    output_white: tuple[float, float],
    cat: str,
) -> _Float64Matrix:
    """Compose one full-adaptation XYZ matrix in host float64."""
    basis = _CAT_BASES[cast(ChromaticAdaptation, cat)]
    input_xyz = _xy_to_xyz(input_white)
    output_xyz = _xy_to_xyz(output_white)
    input_response = basis @ input_xyz
    output_response = basis @ output_xyz
    zero_scale = max(
        1.0,
        float(
            np.linalg.norm(basis, ord=np.inf)
            * max(np.linalg.norm(input_xyz, ord=np.inf), np.linalg.norm(output_xyz, ord=np.inf))
        ),
    )
    zero_threshold = np.finfo(np.float64).eps * zero_scale * 8.0
    if (
        not np.isfinite(input_response).all()
        or not np.isfinite(output_response).all()
        or np.any(np.abs(input_response) <= zero_threshold)
        or np.any(np.abs(output_response) <= zero_threshold)
    ):
        raise _error(
            why="white points must produce finite nonzero cone responses for the selected CAT",
            what=(
                f"received input_white={input_white!r}, output_white={output_white!r}, cat={cat!r}; "
                f"input_response={tuple(input_response)!r}, output_response={tuple(output_response)!r}"
            ),
            how="choose valid input and output xy values whose selected CAT cone responses are finite and nonzero",
        )
    matrix = np.linalg.inv(basis) @ np.diag(output_response / input_response) @ basis
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise _error(
            why="white points and CAT must compose a finite 3x3 adaptation matrix",
            what=f"received input_white={input_white!r}, output_white={output_white!r}, cat={cat!r}",
            how="choose a valid input/output xy pair and one documented CAT token",
        )
    return np.asarray(matrix, dtype=np.float64)


def _temperature_to_xy(temperature: object, tint: object = 0.0) -> tuple[float, float]:
    """Map Kelvin and signed raw Duv to CIE 1931 xy in host float64."""
    resolved_temperature = _finite_real(temperature, name="temperature")
    if resolved_temperature < _MINIMUM_TEMPERATURE:
        raise _error(
            why="temperature must stay within the DNG table's finite 0-to-600-mired domain",
            what=f"received temperature={temperature!r}",
            how=f"pass temperature >= {_MINIMUM_TEMPERATURE!r} Kelvin with no finite upper bound",
        )
    resolved_tint = _finite_real(tint, name="tint")
    reciprocal_temperature = np.float64(1_000_000.0) / np.float64(resolved_temperature)
    upper_index = next(
        (
            index
            for index, record in enumerate(_DNG_TEMPERATURE_TABLE)
            if np.float64(record[0]) >= reciprocal_temperature
        ),
        len(_DNG_TEMPERATURE_TABLE) - 1,
    )
    lower_index = max(0, upper_index - 1)
    lower = _DNG_TEMPERATURE_TABLE[lower_index]
    upper = _DNG_TEMPERATURE_TABLE[upper_index]
    fraction = (
        np.float64(0.0)
        if lower_index == upper_index
        else (reciprocal_temperature - np.float64(lower[0])) / (np.float64(upper[0]) - np.float64(lower[0]))
    )
    u = np.float64(lower[1]) + fraction * (np.float64(upper[1]) - np.float64(lower[1]))
    v = np.float64(lower[2]) + fraction * (np.float64(upper[2]) - np.float64(lower[2]))
    slope = np.float64(lower[3]) + fraction * (np.float64(upper[3]) - np.float64(lower[3]))
    length = np.sqrt(np.float64(1.0) + slope * slope)
    u_tinted = u - np.float64(resolved_tint) / length
    v_tinted = v - slope * np.float64(resolved_tint) / length
    denominator = np.float64(2.0) * u_tinted - np.float64(8.0) * v_tinted + np.float64(4.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xy = (
            float(np.float64(3.0) * u_tinted / denominator),
            float(np.float64(2.0) * v_tinted / denominator),
        )
    return _validate_xy(xy, name="temperature/tint-derived input_white")


def _apply_adaptation(
    frame: Frame,
    *,
    input_white: tuple[float, float],
    output_white: tuple[float, float],
    cat: ChromaticAdaptation,
    operation: str,
) -> Frame:
    adaptation = _chromatic_adaptation_matrix(input_white, output_white, cat)
    if input_white == output_white:
        output_data = frame.data.copy()
    else:
        rgb_to_xyz = _RGB_TO_XYZ[frame.colorspace]
        rgb_matrix = np.linalg.inv(rgb_to_xyz) @ adaptation @ rgb_to_xyz
        if not np.isfinite(rgb_matrix).all():
            raise _error(
                why="white points, CAT, and Frame colorspace must compose a finite RGB matrix",
                what=(
                    f"received input_white={input_white!r}, output_white={output_white!r}, "
                    f"cat={cat!r}, colorspace={frame.colorspace!r}"
                ),
                how="choose a valid input/output xy pair and documented CAT for the Frame colorspace",
            )
        try:
            output_data = _transform_data(
                frame.data,
                frame.channels,
                input_gamma=frame.gamma,
                output_gamma=frame.gamma,
                matrix=np.asarray(rgb_matrix, dtype=np.float32),
            )
        except Exception as error:
            raise _error(
                why=f"{operation} could not execute its fused GPU pixel pass",
                what=f"backend raised {type(error).__module__}.{type(error).__qualname__}: {error}",
                how="verify the CUDA runtime and retry with a valid float32 Frame on an available NVIDIA GPU",
            ) from error
    return Frame(
        data=output_data,
        colorspace=frame.colorspace,
        gamma=frame.gamma,
        channels=frame.channels,
        matrix=None,
    )


def chromatic_adaptation(
    frame: Frame,
    *,
    input_white: ReferenceWhite | Sequence[float],
    output_white: ReferenceWhite | Sequence[float],
    cat: ChromaticAdaptation = "cat02",
) -> Frame:
    """Adapt a float32 RGB Frame between explicit CIE 1931 xy white points.

    The Frame transfer is decoded, the selected CAT is composed through the
    Frame colorspace, and the original transfer is encoded in one GPU pass.
    RGB channels are label-driven; all other channels are copied bit-for-bit.
    """
    validated_frame = _validate_rgb_transfer_frame(frame, operation="chromatic_adaptation")
    validated_input = _resolve_reference_white(input_white, name="input_white")
    validated_output = _resolve_reference_white(output_white, name="output_white")
    validated_cat = _validate_cat(cat)
    return _apply_adaptation(
        validated_frame,
        input_white=validated_input,
        output_white=validated_output,
        cat=validated_cat,
        operation="chromatic_adaptation",
    )


def white_balance(
    frame: Frame,
    *,
    temperature: float,
    tint: float = 0.0,
    cat: ChromaticAdaptation = "cat02",
) -> Frame:
    """Correct a source illuminant described by Kelvin and signed raw Duv.

    The target is the Frame colorspace's nominal white. Positive Tint describes
    a source on the green side of the DNG black-body locus and therefore moves
    the corrected output toward magenta.
    """
    validated_frame = _validate_rgb_transfer_frame(frame, operation="white_balance")
    input_white = _temperature_to_xy(temperature, tint)
    output_white = _COLORSPACE_DEFINITIONS[validated_frame.colorspace][1]
    validated_cat = _validate_cat(cat)
    return _apply_adaptation(
        validated_frame,
        input_white=input_white,
        output_white=output_white,
        cat=validated_cat,
        operation="white_balance",
    )
