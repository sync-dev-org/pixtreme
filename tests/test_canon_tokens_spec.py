"""Specification tests for Canon Cinema Gamut and Canon Log tokens."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import cupy as cp
import numpy as np
import pytest
from repository_contracts import latest_changelog_section, require_repo_file

import pixtreme as px

ROOT = Path(__file__).resolve().parents[1]

_COLORSPACES = (
    "sRGB",
    "Rec.709",
    "Rec.2020",
    "P3-DCI",
    "P3-D60",
    "P3-D65",
    "SMPTE-C",
    "ACES2065-1",
    "ACEScg",
    "S-Gamut",
    "S-Gamut3",
    "S-Gamut3.Cine",
    "ARRI-Wide-Gamut-3",
    "ARRI-Wide-Gamut-4",
    "Blackmagic-Wide-Gamut-Gen-5",
    "DaVinci-Wide-Gamut",
    "REDWideGamutRGB",
    "DRAGONcolor",
    "DRAGONcolor2",
    "REDcolor2",
    "REDcolor3",
    "REDcolor4",
    "Canon-Cinema-Gamut",
    "V-Gamut",
    "D-Gamut",
    "F-Gamut-C",
    "Apple-Wide-Gamut",
)
_GAMMAS = (
    "linear",
    "sRGB",
    "Rec.709",
    "BT.1886",
    "PQ",
    "HLG",
    "ACEScc",
    "ACEScct",
    "S-Log",
    "S-Log2",
    "S-Log3",
    "ARRI-LogC3",
    "ARRI-LogC4",
    "Blackmagic-Film-Gen-5",
    "DaVinci-Intermediate",
    "RED-Log3G10",
    "REDlogFilm",
    "Canon-Log",
    "Canon-Log-2",
    "Canon-Log-3",
    "V-Log",
    "D-Log",
    "F-Log",
    "F-Log2",
    "N-Log",
    "L-Log",
    "Apple-Log",
    "Samsung-Log",
    "Cineon",
    "Gamma-2.2",
    "Gamma-2.4",
    "Gamma-2.5",
    "Gamma-2.6",
)
_ALIASES = (
    px.core.ChromaticAdaptation,
    px.core.ReferenceWhite,
    px.core.Colorspace,
    px.core.Gamma,
    px.core.Matrix,
    px.core.Dtype,
    px.core.Layout,
    px.core.Tonemap,
    px.core.Range,
    px.core.Interpolation,
    px.core.Border,
    px.core.ChromaSiting,
    px.core.StackDirection,
    px.core.SobelDirection,
    px.core.TemplateMatchingMethod,
    px.core.Blend,
    px.core.Alpha,
    px.core.Antialiasing,
    px.core.TextLanguage,
    px.core.TextAnchor,
    px.core.TextAlign,
    px.core.TextFont,
    px.core.GeneratorKind,
    px.core.ColorBarsStandard,
    px.core.ColorBarsOutput,
    px.core.MorphologyShape,
    px.core.ImageFormat,
    px.core.TiffCompression,
    px.core.ExrCompression,
    px.core.VectorBlurShutter,
)

_D65 = (0.3127, 0.3290)
_ACES_WHITE = (0.32168, 0.33767)
_REC709 = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65)
_ACES2065 = (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), _ACES_WHITE)
_CANON_GAMUT = (((0.7400, 0.2700), (0.1700, 1.1400), (0.0800, -0.1000)), _D65)
_CANON_RGB_TO_XYZ = np.asarray(
    (
        (0.716049646551520, 0.129683477875740, 0.104722802624412),
        (0.261261357525555, 0.869642145754960, -0.130903503280515),
        (-0.009676346575021, -0.236481636126349, 1.335215733461248),
    ),
    dtype=np.float64,
)
_CANON_CAT02_TO_ACES = np.asarray(
    (
        (0.763064454775734, 0.149021161137060, 0.0879143840872056),
        (0.00365745670512393, 1.106960380376220, -0.110617837081339),
        (-0.00940779404571890, -0.218383304989987, 1.227791099035710),
    ),
    dtype=np.float64,
)
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)
_CAT02 = np.asarray(
    ((0.7328, 0.4296, -0.1624), (-0.7036, 1.6975, 0.0061), (0.0030, 0.0136, 0.9834)),
    dtype=np.float64,
)

_CURVES = {
    "Canon-Log": {
        "a": np.float64("0.45310179"),
        "b": np.float64("10.1596"),
        "c": np.float64("0.12512248"),
    },
    "Canon-Log-2": {
        "a": np.float64("0.24136077"),
        "b": np.float64("87.099375"),
        "c": np.float64("0.092864125"),
    },
    "Canon-Log-3": {
        "a": np.float64("0.36726845"),
        "b": np.float64("14.98325"),
        "m": np.float64("1.9754798"),
        "c": np.float64("0.12512219"),
        "c_pos": np.float64("0.12240537"),
        "c_neg": np.float64("0.12783901"),
        "cut": np.float64("0.014"),
    },
}


def _piecewise(
    values: np.ndarray,
    masks_and_functions: tuple[tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]], ...],
) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for mask, function in masks_and_functions:
        result[mask] = function(values[mask])
    return result


def _canon_encode(gamma: str, reflectance: np.ndarray) -> np.ndarray:
    constants = _CURVES[gamma]
    x = np.asarray(reflectance, dtype=np.float64) / np.float64("0.9")
    a = constants["a"]
    b = constants["b"]
    c = constants["c"]
    if gamma != "Canon-Log-3":
        positive = x >= 0.0
        return _piecewise(
            x,
            (
                (~positive, lambda part: -a * np.log10(1.0 - b * part) + c),
                (positive, lambda part: a * np.log10(1.0 + b * part) + c),
            ),
        )
    cut = constants["cut"]
    lower = x < -cut
    upper = x > cut
    linear = ~(lower | upper)
    return _piecewise(
        x,
        (
            (lower, lambda part: -a * np.log10(1.0 - b * part) + constants["c_neg"]),
            (linear, lambda part: constants["m"] * part + c),
            (upper, lambda part: a * np.log10(1.0 + b * part) + constants["c_pos"]),
        ),
    )


def _canon_decode(gamma: str, encoded: np.ndarray) -> np.ndarray:
    values = np.asarray(encoded, dtype=np.float64)
    constants = _CURVES[gamma]
    a = constants["a"]
    b = constants["b"]
    c = constants["c"]
    if gamma != "Canon-Log-3":
        positive = values >= c
        x = _piecewise(
            values,
            (
                (~positive, lambda part: -(np.float64(10.0) ** ((c - part) / a) - 1.0) / b),
                (positive, lambda part: (np.float64(10.0) ** ((part - c) / a) - 1.0) / b),
            ),
        )
        return np.float64("0.9") * x
    lower_cut = c - constants["m"] * constants["cut"]
    upper_cut = c + constants["m"] * constants["cut"]
    lower = values < lower_cut
    upper = values > upper_cut
    linear = ~(lower | upper)
    x = _piecewise(
        values,
        (
            (
                lower,
                lambda part: -(np.float64(10.0) ** ((constants["c_neg"] - part) / a) - 1.0) / b,
            ),
            (linear, lambda part: (part - c) / constants["m"]),
            (
                upper,
                lambda part: (np.float64(10.0) ** ((part - constants["c_pos"]) / a) - 1.0) / b,
            ),
        ),
    )
    return np.float64("0.9") * x


def _frame(
    values: np.ndarray | tuple[float, ...],
    *,
    colorspace: str = "ACEScg",
    gamma: str = "linear",
    auxiliary: bool = False,
) -> px.core.Frame:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = np.repeat(array[:, None], 3, axis=1)
    channels: tuple[str, ...] = ("R", "G", "B")
    if auxiliary:
        auxiliary_values = (np.arange(array.shape[0], dtype=np.float32) + np.float32(16.0))[:, None]
        array = np.concatenate((auxiliary_values, array[:, (2, 0, 1)]), axis=1)
        channels = ("Z", "B", "R", "G")
    return px.io.from_array(
        cp.asarray(array[None, :, :]),
        colorspace=colorspace,
        gamma=gamma,
        channels=channels,
        matrix="native",
    )


def _rgb_values(frame: px.core.Frame) -> np.ndarray:
    array = px.io.to_array(frame).get()[0]
    return array[:, [frame.channels.index(label) for label in ("R", "G", "B")]]


def _literal_strings(annotation: object) -> tuple[str, ...]:
    if get_origin(annotation) is Literal:
        return tuple(value for value in get_args(annotation) if isinstance(value, str))
    return tuple(value for argument in get_args(annotation) for value in _literal_strings(argument))


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.asarray((x / y, 1.0, (1.0 - x - y) / y), dtype=np.float64)


def _rgb_to_xyz(
    primaries: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    white: tuple[float, float],
) -> np.ndarray:
    unscaled = np.asarray(
        (tuple(x / y for x, y in primaries), (1.0, 1.0, 1.0), tuple((1.0 - x - y) / y for x, y in primaries)),
        dtype=np.float64,
    )
    return unscaled @ np.diag(np.linalg.solve(unscaled, _xy_to_xyz(white)))


def _adaptation(source: tuple[float, float], target: tuple[float, float], cone_response: np.ndarray) -> np.ndarray:
    source_cones = cone_response @ _xy_to_xyz(source)
    target_cones = cone_response @ _xy_to_xyz(target)
    return np.linalg.inv(cone_response) @ np.diag(target_cones / source_cones) @ cone_response


def _conversion(
    source: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
    target: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
    *,
    cone_response: np.ndarray = _BRADFORD,
) -> np.ndarray:
    source_primaries, source_white = source
    target_primaries, target_white = target
    return (
        np.linalg.inv(_rgb_to_xyz(target_primaries, target_white))
        @ _adaptation(source_white, target_white, cone_response)
        @ _rgb_to_xyz(source_primaries, source_white)
    )


def test_canon_tokens_extend_canonical_vocabulary_and_public_static_surfaces() -> None:
    """v1-canon-tokens acceptance 76-77; v1-panasonic-tokens acceptance 99-100;
    v1-standard-tokens acceptance 117; v1-vendor-a-tokens acceptance 140-141;
    v1-vendor-b-tokens acceptance 166-167:
    expose only the exact current canonical vocabulary.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS
    frame = _frame((0.18,), colorspace="Canon-Cinema-Gamut", gamma="Canon-Log-3")
    assert (frame.colorspace, frame.gamma) == ("Canon-Cinema-Gamut", "Canon-Log-3")
    assert "colorspace='Canon-Cinema-Gamut'" in repr(frame)
    assert "gamma='Canon-Log-3'" in repr(frame)
    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    assert len(_PERMANENT_TOKEN_ALIASES) == 4
    assert not any("Canon" in item for alias in _PERMANENT_TOKEN_ALIASES for item in alias)


def test_canon_token_keys_and_invalid_inputs_follow_the_shared_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1-canon-tokens acceptance 78 and 92; v1-vendor-a-tokens acceptance 142 and 160:
    normalize keys and reject raw invalid values before GPU work.
    """
    from pixtreme._core.validation import _normalized_closed_token

    translation = str.maketrans("", "", " .-_")
    expected = {
        "Canon-Cinema-Gamut": "canoncinemagamut",
        "Canon-Log": "canonlog",
        "Canon-Log-2": "canonlog2",
        "Canon-Log-3": "canonlog3",
    }
    assert {token: token.translate(translation).casefold() for token in expected} == expected
    assert len({token.translate(translation).casefold() for token in _COLORSPACES}) == len(_COLORSPACES)
    assert len({token.translate(translation).casefold() for token in _GAMMAS}) == len(_GAMMAS)
    variants = {
        "Canon-Cinema-Gamut": ("canon cinema gamut", "CANON.CINEMA_GAMUT", "CanonCinemaGamut"),
        "Canon-Log": ("canon log", "CANON.LOG", "CanonLog"),
        "Canon-Log-2": ("canon log 2", "CANON.LOG_2", "CanonLog2"),
        "Canon-Log-3": ("canon log 3", "CANON.LOG_3", "CanonLog3"),
    }
    for canonical, spellings in variants.items():
        accepted = _COLORSPACES if canonical in _COLORSPACES else _GAMMAS
        axis = "colorspace" if canonical in _COLORSPACES else "gamma"
        for spelling in spellings:
            assert _normalized_closed_token(spelling, axis=axis, accepted=accepted) == canonical
    for rejected in ("CLog2", "CLog3", "Cinema Gamut", "Canon Raw"):
        accepted = _COLORSPACES if rejected == "Cinema Gamut" else _GAMMAS
        axis = "colorspace" if accepted is _COLORSPACES else "gamma"
        with pytest.raises(ValueError):
            _normalized_closed_token(rejected, axis=axis, accepted=accepted)
    with pytest.raises(ValueError):
        _normalized_closed_token("Canon-Log", axis="colorspace", accepted=_COLORSPACES)

    import pixtreme._color.semantics as semantics

    monkeypatch.setattr(semantics, "_color_semantics_kernel", lambda: pytest.fail("GPU work must not start"))
    source = _frame((0.18,))
    for rejected in ("Canon Raw", 17):
        with pytest.raises(ValueError) as captured:
            px.color.linear_to_gamma(source, gamma=rejected)
        message = str(captured.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert f"received gamma={rejected!r}" in message
        assert repr(_GAMMAS) in message


@pytest.mark.parametrize(
    ("gamma", "reflectance", "float64_codes", "integer_codes"),
    (
        (
            "Canon-Log",
            (-0.25, 0.0, 0.18, 0.9, 1.5, 7.2),
            (128.00029704, 351.28761123, 613.60966955, 1015.77567097),
            (128, 351, 614, 1016),
        ),
        (
            "Canon-Log-2",
            (-0.25, 0.0, 0.18, 0.9, 1.5, 57.6),
            (94.99999988, 407.41454995, 575.23726233, 1019.99999151),
            (95, 407, 575, 1020),
        ),
    ),
)
def test_canon_log_and_log2_encode_match_independent_piecewise_oracles(
    gamma: str,
    reflectance: tuple[float, ...],
    float64_codes: tuple[float, ...],
    integer_codes: tuple[int, ...],
) -> None:
    """v1-canon-tokens acceptance 79 and 81: encode signed scene values and published anchors."""
    values = np.asarray(reflectance, dtype=np.float32)
    source = _frame(values, auxiliary=True)
    before = source.data.copy()
    encoded = px.color.linear_to_gamma(source, gamma=gamma)
    actual = _rgb_values(encoded)[:, 0]
    expected = _canon_encode(gamma, values.astype(np.float64))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)
    anchor_indices = np.asarray((1, 2, 3, 5))
    actual_codes = actual[anchor_indices].astype(np.float64) * np.float64(1023.0)
    np.testing.assert_allclose(actual_codes, float64_codes, rtol=0.0, atol=2.1e-4)
    np.testing.assert_array_equal(np.rint(actual_codes).astype(np.int64), integer_codes)
    positive = _canon_encode(gamma, np.asarray((0.25,), dtype=np.float64))[0]
    negative = _canon_encode(gamma, np.asarray((-0.25,), dtype=np.float64))[0]
    assert negative != -positive
    assert actual[0] < np.float32(_CURVES[gamma]["c"])
    assert actual[-1] > 0.0
    assert cp.array_equal(encoded.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (encoded.colorspace, encoded.gamma, encoded.channels, encoded.matrix) == (
        "ACEScg",
        gamma,
        source.channels,
        None,
    )
    assert encoded.data.dtype == cp.float32


@pytest.mark.parametrize("gamma", ("Canon-Log", "Canon-Log-2"))
def test_canon_log_and_log2_decode_grids_ulp_and_round_trips_match_independent_inverses(gamma: str) -> None:
    """v1-canon-tokens acceptance 80 and 82: decode full grids and preserve signed overshoot round trips."""
    c = np.float32(_CURVES[gamma]["c"])
    grid = np.linspace(-0.5, 1.5, 200_001, dtype=np.float32)
    special = np.asarray(
        (
            np.nextafter(c, np.float32(-np.inf)),
            c,
            np.nextafter(c, np.float32(np.inf)),
            0.0,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )
    encoded_values = np.unique(np.concatenate((grid, special)))
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(encoded_values, gamma=gamma), gamma=gamma))[:, 0]
    expected = _canon_decode(gamma, encoded_values.astype(np.float64))
    np.testing.assert_allclose(decoded, expected, rtol=2e-6, atol=2e-7)
    anchor_index = int(np.flatnonzero(encoded_values == np.float32(1.0))[0])
    actual_bits = int(decoded[anchor_index : anchor_index + 1].view(np.uint32)[0])
    oracle_bits = int(np.asarray((expected[anchor_index],), dtype=np.float32).view(np.uint32)[0])
    assert abs(actual_bits - oracle_bits) <= 16

    linear_values = np.asarray((-2.0, -0.25, 0.0, 0.18, 0.9, 1.5, 64.0), dtype=np.float32)
    restored = px.color.gamma_to_linear(
        px.color.linear_to_gamma(_frame(linear_values), gamma=gamma),
        gamma=gamma,
    )
    np.testing.assert_allclose(_rgb_values(restored)[:, 0], linear_values, rtol=2e-6, atol=2e-7)
    round_trip_encoded = np.asarray((-0.75, c, 0.5, 1.0, 1.5, 1.6), dtype=np.float32)
    reencoded = px.color.linear_to_gamma(
        px.color.gamma_to_linear(_frame(round_trip_encoded, gamma=gamma), gamma=gamma),
        gamma=gamma,
    )
    np.testing.assert_allclose(_rgb_values(reencoded)[:, 0], round_trip_encoded, rtol=2e-6, atol=2e-7)


def test_canon_log3_encode_includes_both_cuts_in_the_linear_branch_and_matches_anchors() -> None:
    """v1-canon-tokens acceptance 83: encode the three branches, exact cut policy, and published anchors."""
    cut = np.float32(_CURVES["Canon-Log-3"]["cut"])
    x_values = np.asarray(
        (
            -0.25,
            np.nextafter(-cut, np.float32(-np.inf)),
            -cut,
            np.nextafter(-cut, np.float32(np.inf)),
            0.0,
            np.nextafter(cut, np.float32(-np.inf)),
            cut,
            np.nextafter(cut, np.float32(np.inf)),
            0.2,
            1.0,
            16.0,
        ),
        dtype=np.float32,
    )
    reflectance = x_values * np.float32(0.9)
    encoded = px.color.linear_to_gamma(_frame(reflectance, auxiliary=True), gamma="Canon-Log-3")
    actual = _rgb_values(encoded)[:, 0]
    expected = _canon_encode("Canon-Log-3", reflectance.astype(np.float64))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)
    anchor_indices = np.asarray((4, 8, 9, 10))
    float64_codes = (128.00000037, 351.28732589, 577.45647524, 1020.00016109)
    actual_codes = actual[anchor_indices].astype(np.float64) * np.float64(1023.0)
    np.testing.assert_allclose(actual_codes, float64_codes, rtol=0.0, atol=2.1e-4)
    np.testing.assert_array_equal(np.rint(actual_codes).astype(np.int64), (128, 351, 577, 1020))
    constants = _CURVES["Canon-Log-3"]
    linear_cuts = constants["c"] + constants["m"] * np.asarray((-constants["cut"], constants["cut"]))
    log_cuts = np.asarray(
        (
            -constants["a"] * np.log10(1.0 + constants["b"] * constants["cut"]) + constants["c_neg"],
            constants["a"] * np.log10(1.0 + constants["b"] * constants["cut"]) + constants["c_pos"],
        )
    )
    np.testing.assert_array_equal(linear_cuts.astype(np.float32), log_cuts.astype(np.float32))
    assert np.all(np.nextafter(linear_cuts.astype(np.float32), np.float32(np.inf)) > log_cuts)
    assert (encoded.gamma, encoded.matrix, encoded.channels) == ("Canon-Log-3", None, ("Z", "B", "R", "G"))


def test_canon_log3_decode_thresholds_monotonicity_ulp_and_round_trips_match_oracle() -> None:
    """v1-canon-tokens acceptance 84-85: decode derived cuts uniquely and preserve monotone signed overshoot."""
    constants = _CURVES["Canon-Log-3"]
    lower = np.float32(constants["c"] - constants["m"] * constants["cut"])
    upper = np.float32(constants["c"] + constants["m"] * constants["cut"])
    grid = np.linspace(-0.5, 1.5, 200_001, dtype=np.float32)
    special = np.asarray(
        (
            np.nextafter(lower, np.float32(-np.inf)),
            lower,
            np.nextafter(lower, np.float32(np.inf)),
            np.nextafter(upper, np.float32(-np.inf)),
            upper,
            np.nextafter(upper, np.float32(np.inf)),
            0.0,
            1.0,
            1.5,
        ),
        dtype=np.float32,
    )
    encoded_values = np.unique(np.concatenate((grid, special)))
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(encoded_values, gamma="Canon-Log-3"), gamma="Canon-Log-3"))[
        :, 0
    ]
    expected = _canon_decode("Canon-Log-3", encoded_values.astype(np.float64))
    np.testing.assert_allclose(decoded, expected, rtol=2e-6, atol=2e-7)
    threshold_values = np.asarray((lower, upper), dtype=np.float32)
    threshold_decoded = _rgb_values(
        px.color.gamma_to_linear(_frame(threshold_values, gamma="Canon-Log-3"), gamma="Canon-Log-3")
    )[:, 0]
    float32_constants = {name: np.float32(value) for name, value in constants.items()}
    linear_inverse = np.float32(0.9) * ((threshold_values - float32_constants["c"]) / float32_constants["m"])
    log_inverse = np.asarray(
        (
            -np.float32(0.9)
            * (np.float32(10.0) ** ((float32_constants["c_neg"] - lower) / float32_constants["a"]) - np.float32(1.0))
            / float32_constants["b"],
            np.float32(0.9)
            * (np.float32(10.0) ** ((upper - float32_constants["c_pos"]) / float32_constants["a"]) - np.float32(1.0))
            / float32_constants["b"],
        ),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(threshold_decoded.view(np.uint32), linear_inverse.view(np.uint32))
    assert np.all(threshold_decoded.view(np.uint32) != log_inverse.view(np.uint32))
    anchor_index = int(np.flatnonzero(encoded_values == np.float32(1.0))[0])
    actual_bits = int(decoded[anchor_index : anchor_index + 1].view(np.uint32)[0])
    oracle_bits = int(np.asarray((expected[anchor_index],), dtype=np.float32).view(np.uint32)[0])
    assert abs(actual_bits - oracle_bits) <= 16

    fine_x = np.linspace(-0.05, 0.05, 131_073, dtype=np.float32)
    fine_r = fine_x * np.float32(0.9)
    fine_encoded = _rgb_values(px.color.linear_to_gamma(_frame(fine_r), gamma="Canon-Log-3"))[:, 0]
    assert np.all(np.diff(fine_encoded) >= 0.0)
    linear_values = np.asarray((-2.0, -0.25, -0.0126, 0.0, 0.0126, 0.18, 0.9, 14.4, 20.0), dtype=np.float32)
    restored = px.color.gamma_to_linear(
        px.color.linear_to_gamma(_frame(linear_values), gamma="Canon-Log-3"),
        gamma="Canon-Log-3",
    )
    np.testing.assert_allclose(_rgb_values(restored)[:, 0], linear_values, rtol=2e-6, atol=2e-7)
    encoded_round_trip = np.asarray((-0.75, lower, upper, 0.5, 1.0, 1.5, 1.6), dtype=np.float32)
    reencoded = px.color.linear_to_gamma(
        px.color.gamma_to_linear(_frame(encoded_round_trip, gamma="Canon-Log-3"), gamma="Canon-Log-3"),
        gamma="Canon-Log-3",
    )
    np.testing.assert_allclose(_rgb_values(reencoded)[:, 0], encoded_round_trip, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("gamma", ("Canon-Log", "Canon-Log-2", "Canon-Log-3"))
def test_canon_transfer_standalone_and_fused_paths_are_bit_identical(gamma: str) -> None:
    """v1-canon-tokens acceptance 86: keep standalone and fused transfer paths bit-identical."""
    linear_values = np.asarray((-0.25, -0.0126, 0.0, 0.18, 1.0, 1.5), dtype=np.float32)
    linear = _frame(linear_values, auxiliary=True)
    standalone_encoded = px.color.linear_to_gamma(linear, gamma=gamma)
    fused_encoded = px.color.rgb_to_rgb(linear, output_gamma=gamma)
    np.testing.assert_array_equal(standalone_encoded.data.get(), fused_encoded.data.get())
    encoded_values = _canon_encode(gamma, linear_values.astype(np.float64)).astype(np.float32)
    encoded = _frame(encoded_values, gamma=gamma, auxiliary=True)
    standalone_decoded = px.color.gamma_to_linear(encoded, gamma=gamma)
    fused_decoded = px.color.rgb_to_rgb(encoded, output_gamma="linear")
    np.testing.assert_array_equal(standalone_decoded.data.get(), fused_decoded.data.get())
    assert cp.array_equal(standalone_encoded.data[..., 0], linear.data[..., 0])
    assert cp.array_equal(standalone_decoded.data[..., 0], encoded.data[..., 0])


def test_canon_cinema_gamut_matrix_conversions_and_native_row_match_independent_oracles() -> None:
    """v1-canon-tokens acceptance 87: derive the normalized gamut, conversions, and native luma row."""
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    assert _COLORSPACE_DEFINITIONS["Canon-Cinema-Gamut"] == _CANON_GAMUT
    derived = _rgb_to_xyz(*_CANON_GAMUT)
    np.testing.assert_allclose(derived, _CANON_RGB_TO_XYZ, rtol=0.0, atol=1e-12)
    values = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)),
        dtype=np.float64,
    )
    source = _frame(values, colorspace="Canon-Cinema-Gamut")
    converted = px.color.rgb_to_rgb(source, output_colorspace="Rec.709")
    expected = values @ _conversion(_CANON_GAMUT, _REC709).T
    np.testing.assert_allclose(_rgb_values(converted), expected, rtol=0.0, atol=6e-6)
    grayscale = px.color.rgb_to_grayscale(source, colorspace="Canon-Cinema-Gamut", gamma="linear", matrix="native")
    np.testing.assert_allclose(
        px.io.to_array(grayscale).get()[0, :, 0],
        values @ _CANON_RGB_TO_XYZ[1],
        rtol=0.0,
        atol=6e-6,
    )
    assert np.any(_rgb_values(converted) < 0.0) and np.any(_rgb_values(converted) > 1.0)


def test_canon_cinema_gamut_uses_d65_identity_bradford_and_cat02_only_as_auxiliary_oracle() -> None:
    """v1-canon-tokens acceptance 88: preserve D65 identity, Bradford production, and independent CAT02 evidence."""
    from pixtreme._color.transform import _compose_matrix

    canon_to_rec709 = _conversion(_CANON_GAMUT, _REC709)
    np.testing.assert_allclose(_compose_matrix("Canon-Cinema-Gamut", "Rec.709"), canon_to_rec709, rtol=0.0, atol=6e-6)
    canon_to_aces_bradford = _conversion(_CANON_GAMUT, _ACES2065)
    np.testing.assert_allclose(
        _compose_matrix("Canon-Cinema-Gamut", "ACES2065-1"),
        canon_to_aces_bradford,
        rtol=0.0,
        atol=6e-6,
    )
    canon_to_aces_cat02 = _conversion(_CANON_GAMUT, _ACES2065, cone_response=_CAT02)
    np.testing.assert_allclose(canon_to_aces_cat02, _CANON_CAT02_TO_ACES, rtol=0.0, atol=5e-13)
    assert np.max(np.abs(canon_to_aces_bradford - _CANON_CAT02_TO_ACES)) > 5e-3


@pytest.mark.parametrize("gamma", ("Canon-Log", "Canon-Log-2", "Canon-Log-3"))
@pytest.mark.parametrize("target", ("Rec.709", "ACES2065-1"))
def test_canon_frames_convert_end_to_end_with_independent_transfer_and_gamut_oracle(gamma: str, target: str) -> None:
    """v1-canon-tokens acceptance 89: fuse Canon decode and gamut conversion with auxiliary-bit preservation."""
    linear_rgb = np.asarray(((-0.25, 0.18, 1.5), (0.18, 1.25, -0.05)), dtype=np.float64)
    encoded_rgb = _canon_encode(gamma, linear_rgb).astype(np.float32)
    source = _frame(encoded_rgb, colorspace="Canon-Cinema-Gamut", gamma=gamma, auxiliary=True)
    before = source.data.copy()
    converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
    target_definition = _REC709 if target == "Rec.709" else _ACES2065
    expected = (
        _canon_decode(gamma, _rgb_values(source).astype(np.float64)) @ _conversion(_CANON_GAMUT, target_definition).T
    )
    np.testing.assert_allclose(_rgb_values(converted), expected, rtol=2e-6, atol=8e-6)
    assert cp.array_equal(converted.data[..., 0], source.data[..., 0])
    assert cp.array_equal(source.data, before)
    assert (converted.colorspace, converted.gamma, converted.channels, converted.matrix) == (
        target,
        "linear",
        source.channels,
        None,
    )
    assert np.any(_rgb_values(converted) < 0.0) and np.any(_rgb_values(converted) > 1.0)


def test_existing_token_bits_remain_at_the_pre_canon_baseline() -> None:
    """v1-canon-tokens acceptance 90: preserve representative existing transfer and gamut output bits."""
    # Provenance: captured from the complete pre-Canon commit
    # 7b2881e3c415da58bcb2d37d5a816da345a493a6.  Reproduce from a detached worktree at that SHA with:
    #
    #   UV_PROJECT_ENVIRONMENT=<main-repo>/.venv PYTHONPATH=<baseline>/src uv run --no-sync python - <<'PY'
    #   import cupy as cp
    #   import numpy as np
    #   import pixtreme as px
    #   linear = np.asarray((-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5), np.float32)
    #   rgb = np.repeat(linear[:, None], 3, axis=1)[None]
    #   base = px.io.from_array(cp.asarray(rgb), colorspace="ACEScg", gamma="linear", channels="RGB")
    #   for gamma in ("S-Log3", "ARRI-LogC4", "Blackmagic-Film-Gen-5", "DaVinci-Intermediate",
    #                 "RED-Log3G10", "Cineon", "Gamma-2.4"):
    #       out = px.io.to_array(px.color.linear_to_gamma(base, gamma=gamma)).get()[0, :, 0]
    #       print(gamma, tuple(int(value) for value in out.view(np.uint32)))
    #   pixels = np.asarray(((-0.25, 0.18, 1.5), (0.02, -0.1, 1.0)), np.float32)[None]
    #   for colorspace in ("Rec.2020", "ACEScg", "S-Gamut3.Cine", "ARRI-Wide-Gamut-4",
    #                      "Blackmagic-Wide-Gamut-Gen-5", "DaVinci-Wide-Gamut", "REDWideGamutRGB"):
    #       frame = px.io.from_array(cp.asarray(pixels), colorspace=colorspace, gamma="linear", channels="RGB")
    #       out = px.io.to_array(px.color.rgb_to_rgb(
    #           frame, output_colorspace="Rec.709", output_gamma="linear"
    #       )).get().view(np.uint32).reshape(-1)
    #       print(colorspace, tuple(int(value) for value in out))
    #   PY
    gamma_fixtures = {
        "S-Log3": (3217556478, 3168455511, 1035874188, 1053963405, 1058575679, 1059324708),
        "ARRI-LogC4": (3221400801, 2995575096, 1035874188, 1049528807, 1054532562, 1055775089),
        "Blackmagic-Film-Gen-5": (3221044577, 3177835909, 1035820719, 1053057585, 1057476139, 1058064820),
        "DaVinci-Intermediate": (3223788473, 3191938634, 0, 1051463133, 1057196762, 1057911650),
        "RED-Log3G10": (3228130337, 3187323085, 1035698008, 1051372189, 1056744777, 1057508475),
        "Cineon": (3204351021, 3193857572, 1035874182, 1055532491, 1059810011, 1060668676),
        "Gamma-2.4": (3205475542, 3191882770, 0, 1056610176, 1065353216, 1066897169),
    }
    linear = (-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5)
    for gamma, expected_bits in gamma_fixtures.items():
        encoded = px.color.linear_to_gamma(_frame(linear), gamma=gamma)
        np.testing.assert_array_equal(_rgb_values(encoded)[:, 0].view(np.uint32), expected_bits)
        assert encoded.gamma == gamma

    gamut_fixtures = {
        "Rec.2020": (3206632196, 1046732887, 1070927681, 1016900116, 3187554328, 1066430520),
        "ACEScg": (3207184178, 1046702756, 1071327960, 1012225928, 3187820956, 1066740601),
        "S-Gamut3.Cine": (3206699341, 3178418026, 1072369260, 3112583168, 3200582444, 1067526363),
        "ARRI-Wide-Gamut-4": (3209182827, 1035601944, 1071419179, 997395808, 3196821229, 1066861949),
        "Blackmagic-Wide-Gamut-Gen-5": (3205367535, 3187568618, 1073070301, 1025198451, 3200919674, 1068080043),
        "DaVinci-Wide-Gamut": (3209093926, 3190676685, 1074284506, 1009788360, 3203518916, 1069265258),
        "REDWideGamutRGB": (3209143998, 3190614948, 1075355334, 1027943654, 3203640764, 1071129191),
    }
    pixels = np.asarray(((-0.25, 0.18, 1.5), (0.02, -0.1, 1.0)), dtype=np.float32)
    for colorspace, expected_bits in gamut_fixtures.items():
        converted = px.color.rgb_to_rgb(
            _frame(pixels, colorspace=colorspace), output_colorspace="Rec.709", output_gamma="linear"
        )
        np.testing.assert_array_equal(
            px.io.to_array(converted).get().view(np.uint32).reshape(-1),
            np.asarray(expected_bits, dtype=np.uint32),
        )


def test_canon_dpx_transfer_codes_are_logarithmic_and_existing_mappings_remain_unchanged(tmp_path: Path) -> None:
    """v1-canon-tokens acceptance 91: classify Canon logs as DPX logarithmic without disturbing prior mappings."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    expected = {
        "Canon-Log": 3,
        "Canon-Log-2": 3,
        "Canon-Log-3": 3,
        "S-Log3": 3,
        "ARRI-LogC4": 3,
        "Blackmagic-Film-Gen-5": 3,
        "DaVinci-Intermediate": 3,
        "RED-Log3G10": 3,
        "Cineon": 1,
        "REDlogFilm": 1,
        "linear": 2,
        "Gamma-2.4": 6,
    }
    assert {gamma: _dpx_transfer_from_gamma(gamma) for gamma in expected} == expected
    pixels = cp.asarray([[[0.18, 0.18, 0.18]]], dtype=cp.float32)
    for gamma, transfer in expected.items():
        frame = px.io.from_array(pixels.copy(), colorspace="Rec.709", gamma=gamma, channels="RGB")
        path = tmp_path / f"{gamma}.dpx"
        px.io.write_image(path, frame)
        assert path.read_bytes()[801] == transfer
        if gamma.startswith("Canon-Log"):
            assert px.io.read_image(path).gamma == "Cineon"


def test_canon_token_reference_requirements_changelog_and_public_docstrings_are_synchronized() -> None:
    """v1-canon-tokens acceptance 93; v1-panasonic-tokens acceptance 112; v1-vendor-a-tokens acceptance 161;
    v1-vendor-b-tokens acceptance 188:
    synchronize canonical counts, numeric contracts, boundaries, and public prose.
    """
    token_reference = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest_section = latest_changelog_section(changelog)
    for token in ("Canon-Cinema-Gamut", "Canon-Log", "Canon-Log-2", "Canon-Log-3"):
        assert f"`{token}`" in token_reference
        assert token in latest_section
    for fragment in (
        "x = r / 0.9",
        "0.45310179",
        "0.24136077",
        "0.36726845",
        "0.014",
        "0.7400",
        "1.1400",
        "-0.1000",
        "D65",
        "Bradford",
        "CAT02",
        "Canon Raw",
        "native",
    ):
        assert fragment in token_reference
    assert "27 Colorspace" in requirements
    assert "33 Gamma" in requirements
    assert "188 canonical tokens" in requirements
    assert "Canon Raw" in latest_section and "bit" in latest_section
    for operation in (px.color.rgb_to_rgb, px.color.gamma_to_linear, px.color.linear_to_gamma):
        docstring = inspect.getdoc(operation)
        assert docstring is not None
        assert "Canon-Cinema-Gamut" in docstring or operation is not px.color.rgb_to_rgb
        assert "Canon-Log" in docstring
        assert "Canon-Log-2" in docstring
        assert "Canon-Log-3" in docstring
