"""Specification tests for standards-derived colorspace and transfer tokens."""

from __future__ import annotations

import inspect
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
_P3_PRIMARIES = ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))
_DEFINITIONS = {
    "P3-DCI": (_P3_PRIMARIES, (0.3140, 0.3510)),
    "P3-D60": (_P3_PRIMARIES, _ACES_WHITE),
    "P3-D65": (_P3_PRIMARIES, _D65),
    "SMPTE-C": (((0.630, 0.340), (0.310, 0.595), (0.155, 0.070)), _D65),
}
_RGB_TO_XYZ = {
    "P3-DCI": np.asarray(
        (
            (0.445169815564552, 0.277134409206778, 0.172282669815565),
            (0.209491677912731, 0.721595254161044, 0.068913067926226),
            (0.0, 0.047060560053981, 0.907355394361973),
        ),
        dtype=np.float64,
    ),
    "P3-D60": np.asarray(
        (
            (0.504949534191744, 0.264681488895262, 0.183015051482840),
            (0.237623310207880, 0.689170669198984, 0.073206020593136),
            (0.0, 0.044945913208629, 0.963879271142956),
        ),
        dtype=np.float64,
    ),
    "P3-D65": np.asarray(
        (
            (0.486570948648216, 0.265667693169093, 0.198217285234362),
            (0.228974564069749, 0.691738521836506, 0.079286914093745),
            (0.0, 0.045113381858903, 1.043944368900976),
        ),
        dtype=np.float64,
    ),
    "SMPTE-C": np.asarray(
        (
            (0.393520903659390, 0.365258076717604, 0.191676946674678),
            (0.212376360705067, 0.701059856925723, 0.086563782369210),
            (0.018739090650447, 0.111933926736040, 0.958384733373392),
        ),
        dtype=np.float64,
    ),
}
_REC709 = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), _D65)
_ACES2065 = (((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), _ACES_WHITE)
_BRADFORD = np.asarray(
    ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    dtype=np.float64,
)


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


def _adaptation(source: tuple[float, float], target: tuple[float, float]) -> np.ndarray:
    if source == target:
        return np.eye(3, dtype=np.float64)
    source_cones = _BRADFORD @ _xy_to_xyz(source)
    target_cones = _BRADFORD @ _xy_to_xyz(target)
    return np.linalg.inv(_BRADFORD) @ np.diag(target_cones / source_cones) @ _BRADFORD


def _conversion(
    source: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
    target: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], tuple[float, float]],
) -> np.ndarray:
    source_primaries, source_white = source
    target_primaries, target_white = target
    return (
        np.linalg.inv(_rgb_to_xyz(target_primaries, target_white))
        @ _adaptation(source_white, target_white)
        @ _rgb_to_xyz(source_primaries, source_white)
    )


def _acescc_encode(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    result = np.empty_like(x)
    nonpositive = x <= 0.0
    lower = (x > 0.0) & (x < 2.0**-15)
    upper = ~(nonpositive | lower)
    result[nonpositive] = (-16.0 + 9.72) / 17.52
    result[lower] = (np.log2(2.0**-16 + x[lower] / 2.0) + 9.72) / 17.52
    result[upper] = (np.log2(x[upper]) + 9.72) / 17.52
    return result


def _acescc_decode(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    lower = y <= (9.72 - 15.0) / 17.52
    result = np.empty_like(y)
    result[lower] = 2.0 * (2.0 ** (17.52 * y[lower] - 9.72) - 2.0**-16)
    result[~lower] = 2.0 ** (17.52 * y[~lower] - 9.72)
    return result


def _acescct_encode(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    linear = x <= 0.0078125
    result = np.empty_like(x)
    result[linear] = 10.5402377416545 * x[linear] + 0.0729055341958355
    result[~linear] = (np.log2(x[~linear]) + 9.72) / 17.52
    return result


def _acescct_decode(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    linear = y <= 0.155251141552511
    result = np.empty_like(y)
    result[linear] = (y[linear] - 0.0729055341958355) / 10.5402377416545
    result[~linear] = 2.0 ** (17.52 * y[~linear] - 9.72)
    return result


def _decode(values: np.ndarray, gamma: str) -> np.ndarray:
    if gamma == "linear":
        return np.asarray(values, dtype=np.float64)
    if gamma == "Gamma-2.5":
        values = np.asarray(values, dtype=np.float64)
        return np.copysign(np.abs(values) ** 2.5, values)
    if gamma == "ACEScc":
        return _acescc_decode(values)
    if gamma == "ACEScct":
        return _acescct_decode(values)
    raise AssertionError(gamma)


def _adjacent_float32(center: np.float32, radius: int = 3000) -> np.ndarray:
    below = np.empty(radius, dtype=np.float32)
    value = center
    for index in range(radius - 1, -1, -1):
        value = np.nextafter(value, np.float32(-np.inf))
        below[index] = value
    above = np.empty(radius, dtype=np.float32)
    value = center
    for index in range(radius):
        value = np.nextafter(value, np.float32(np.inf))
        above[index] = value
    return np.concatenate((below, np.asarray((center,), dtype=np.float32), above))


def test_standard_tokens_extend_only_the_canonical_vocabulary_and_public_surfaces() -> None:
    """v1-standard-tokens acceptance 117-119 and 134; v1-vendor-a-tokens acceptance 140-142;
    v1-vendor-b-tokens acceptance 166-168:
    expose and normalize only current canonical tokens.
    """
    assert get_args(px.core.Colorspace) == _COLORSPACES
    assert get_args(px.core.Gamma) == _GAMMAS
    assert len(_ALIASES) == 30
    assert sum(len(get_args(alias)) for alias in _ALIASES) == 188
    assert _literal_strings(get_type_hints(px.color.linear_to_gamma)["gamma"]) == _GAMMAS
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["input_colorspace"]) == _COLORSPACES
    assert _literal_strings(get_type_hints(px.color.rgb_to_rgb)["output_gamma"]) == _GAMMAS

    from pixtreme._core.validation import _normalized_closed_token
    from pixtreme._core.vocabulary import _PERMANENT_TOKEN_ALIASES

    assert len(_PERMANENT_TOKEN_ALIASES) == 4
    assert not any(
        token in {"P3-DCI", "P3-D60", "P3-D65", "SMPTE-C", "ACEScc", "ACEScct", "Gamma-2.5"}
        for alias in _PERMANENT_TOKEN_ALIASES
        for token in alias
    )
    resolved = {
        "P3 DCI": ("colorspace", _COLORSPACES, "P3-DCI"),
        "p3_d65": ("colorspace", _COLORSPACES, "P3-D65"),
        "SMPTEC": ("colorspace", _COLORSPACES, "SMPTE-C"),
        "ACES cc": ("gamma", _GAMMAS, "ACEScc"),
        "Gamma 2.5": ("gamma", _GAMMAS, "Gamma-2.5"),
        "gamma25": ("gamma", _GAMMAS, "Gamma-2.5"),
    }
    for spelling, (axis, accepted, expected) in resolved.items():
        assert _normalized_closed_token(spelling, axis=axis, accepted=accepted) == expected
    for spelling, axis, accepted in (
        ("DCI-P3", "colorspace", _COLORSPACES),
        ("Display P3", "colorspace", _COLORSPACES),
        ("DisplayP3", "colorspace", _COLORSPACES),
        ("DCI P3 D65", "colorspace", _COLORSPACES),
        ("P3-D65 (Scene)", "colorspace", _COLORSPACES),
        ("Rec.709-A", "gamma", _GAMMAS),
        ("DCI", "gamma", _GAMMAS),
        ("SMPTE 170M", "colorspace", _COLORSPACES),
        ("SMPTE-170M", "colorspace", _COLORSPACES),
        ("NTSC", "colorspace", _COLORSPACES),
    ):
        with pytest.raises(ValueError) as captured:
            _normalized_closed_token(spelling, axis=axis, accepted=accepted)
        message = str(captured.value)
        assert message.index("why=") < message.index("what=") < message.index("how=")
        assert repr(spelling) in message and repr(accepted) in message
    with pytest.raises(ValueError, match=r"what=received gamma=17"):
        px.color.linear_to_gamma(_frame((0.18,)), gamma=17)  # type: ignore[arg-type]

    frame = _frame((0.18,), colorspace="p3_d65", gamma="ACES cc")
    assert (frame.colorspace, frame.gamma) == ("P3-D65", "ACEScc")
    assert "colorspace='P3-D65'" in repr(frame) and "gamma='ACEScc'" in repr(frame)


def test_gamma_25_matches_the_signed_power_oracle_and_rejects_neighbor_exponents() -> None:
    """v1-standard-tokens acceptance 120: evaluate the exact sign-preserving 2.5 power pair."""
    values = np.asarray((-4.0, -1.0, -0.18, 0.0, 0.18, 1.0, 4.0), dtype=np.float32)
    encoded = _rgb_values(px.color.linear_to_gamma(_frame(values), gamma="Gamma-2.5"))[:, 0]
    expected = np.copysign(np.abs(values.astype(np.float64)) ** 0.4, values)
    np.testing.assert_allclose(encoded, expected, rtol=0.0, atol=2e-6)
    assert encoded[4] == pytest.approx(0.5036269964912325, abs=2e-6)
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(encoded, gamma="Gamma-2.5"), gamma="Gamma-2.5"))[:, 0]
    np.testing.assert_allclose(decoded, values, rtol=2e-6, atol=2e-7)
    for wrong_exponent in (1.0 / 2.4, 1.0 / 2.6):
        mutant = np.copysign(np.abs(values.astype(np.float64)) ** wrong_exponent, values)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(mutant, expected, rtol=0.0, atol=2e-6)

    encoded_values = np.linspace(-2.0, 2.0, 40001, dtype=np.float64).astype(np.float32)
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(encoded_values, gamma="Gamma-2.5"), gamma="Gamma-2.5"))[:, 0]
    expected_decoded = np.copysign(np.abs(encoded_values.astype(np.float64)) ** 2.5, encoded_values)
    np.testing.assert_allclose(decoded, expected_decoded, rtol=0.0, atol=2e-6)
    for wrong_exponent in (2.4, 2.6):
        mutant = np.copysign(np.abs(encoded_values.astype(np.float64)) ** wrong_exponent, encoded_values)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(mutant, expected_decoded, rtol=0.0, atol=2e-6)
    reencoded = _rgb_values(px.color.linear_to_gamma(_frame(decoded), gamma="Gamma-2.5"))[:, 0]
    np.testing.assert_allclose(reencoded, encoded_values, rtol=2e-6, atol=2e-7)


def test_acescc_encode_matches_independent_branches_anchors_and_monotonicity() -> None:
    """v1-standard-tokens acceptance 121: preserve ACEScc encode branches, cuts, anchors, and collapse."""
    anchors = np.asarray((0.0, 2.0**-15, 0.0078125, 0.18, 1.0, 65504.0), dtype=np.float32)
    broad = np.concatenate(
        (
            np.linspace(-0.5, 64.0, 10001, dtype=np.float64).astype(np.float32),
            np.geomspace(2.0**-24, 65505.0, 10001, dtype=np.float64).astype(np.float32),
            anchors,
        )
    )
    actual = _rgb_values(px.color.linear_to_gamma(_frame(broad), gamma="ACEScc"))[:, 0]
    np.testing.assert_allclose(actual, _acescc_encode(broad), rtol=2e-7, atol=2e-7)
    assert np.unique(actual[broad <= 0.0]).size == 1
    assert np.isfinite(actual[broad == 0.0]).all()
    for center in (np.float32(0.0), np.float32(2.0**-15)):
        near = _adjacent_float32(center)
        encoded = _rgb_values(px.color.linear_to_gamma(_frame(near), gamma="ACEScc"))[:, 0]
        assert np.all(np.diff(encoded) >= 0.0)


def test_acescc_decode_is_unclipped_and_round_trips_only_its_injective_domain() -> None:
    """v1-standard-tokens acceptance 122-123: decode ACEScc analytically without upper clip."""
    values = np.linspace(-0.5, 1.5, 200001, dtype=np.float64).astype(np.float32)
    actual = _rgb_values(px.color.gamma_to_linear(_frame(values, gamma="ACEScc"), gamma="ACEScc"))[:, 0]
    expected = _acescc_decode(values)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert actual[np.searchsorted(values, np.float32(1.5))] > 65504.0
    for encoded, linear in ((0.0, 0.0011857371917920374), (1.0, 222.8609442038076)):
        result = _rgb_values(px.color.gamma_to_linear(_frame((encoded,), gamma="ACEScc"), gamma="ACEScc"))[0, 0]
        assert result == pytest.approx(linear, rel=2e-6, abs=2e-7)
    decoded_one = _rgb_values(px.color.gamma_to_linear(_frame((1.0,), gamma="ACEScc"), gamma="ACEScc"))[0, 0]
    np.testing.assert_array_max_ulp(decoded_one, np.float32(222.8609442038076), maxulp=16)
    near = _adjacent_float32(np.float32((9.72 - 15.0) / 17.52))
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(near, gamma="ACEScc"), gamma="ACEScc"))[:, 0]
    assert np.all(np.diff(decoded) >= 0.0)

    positive = np.geomspace(2.0**-24, 65505.0, 10001, dtype=np.float64).astype(np.float32)
    restored = px.color.gamma_to_linear(px.color.linear_to_gamma(_frame(positive), gamma="ACEScc"), gamma="ACEScc")
    np.testing.assert_allclose(_rgb_values(restored)[:, 0], positive, rtol=2e-6, atol=2e-7)
    encoded_domain = values[values >= np.float32((-16.0 + 9.72) / 17.52)]
    reencoded = px.color.linear_to_gamma(
        px.color.gamma_to_linear(_frame(encoded_domain, gamma="ACEScc"), gamma="ACEScc"), gamma="ACEScc"
    )
    np.testing.assert_allclose(_rgb_values(reencoded)[:, 0], encoded_domain, rtol=2e-6, atol=2e-7)


def test_acescct_matches_public_branches_cut_residual_and_round_trip() -> None:
    """v1-standard-tokens acceptance 124-125: evaluate ACEScct with its published decimal cuts."""
    anchors = np.asarray((0.0, 2.0**-15, 0.0078125, 0.18, 1.0, 65504.0), dtype=np.float32)
    linear_values = np.concatenate(
        (
            np.linspace(-0.5, 64.0, 40001, dtype=np.float64).astype(np.float32),
            np.geomspace(2.0**-24, 65505.0, 10001, dtype=np.float64).astype(np.float32),
            anchors,
        )
    )
    encoded = _rgb_values(px.color.linear_to_gamma(_frame(linear_values), gamma="ACEScct"))[:, 0]
    np.testing.assert_allclose(encoded, _acescct_encode(linear_values), rtol=2e-7, atol=2e-7)
    near_encode = _adjacent_float32(np.float32(0.0078125))
    near_encoded = _rgb_values(px.color.linear_to_gamma(_frame(near_encode), gamma="ACEScct"))[:, 0]
    assert np.all(np.diff(near_encoded) >= 0.0)

    encoded_values = np.linspace(-0.5, 1.5, 200001, dtype=np.float64).astype(np.float32)
    decoded = _rgb_values(px.color.gamma_to_linear(_frame(encoded_values, gamma="ACEScct"), gamma="ACEScct"))[:, 0]
    np.testing.assert_allclose(decoded, _acescct_decode(encoded_values), rtol=2e-6, atol=2e-7)
    assert decoded[-1] > 65504.0
    decoded_one = _rgb_values(px.color.gamma_to_linear(_frame((1.0,), gamma="ACEScct"), gamma="ACEScct"))[0, 0]
    np.testing.assert_array_max_ulp(decoded_one, np.float32(222.8609442038076), maxulp=16)

    encoded_cut = np.float32(0.155251141552511)
    cut_neighbors = np.asarray(
        (
            np.nextafter(encoded_cut, np.float32(-np.inf)),
            encoded_cut,
            np.nextafter(encoded_cut, np.float32(np.inf)),
        ),
        dtype=np.float32,
    )
    decoded_cut_neighbors = _rgb_values(
        px.color.gamma_to_linear(_frame(cut_neighbors, gamma="ACEScct"), gamma="ACEScct")
    )[:, 0]
    linear_expected = (cut_neighbors[:2] - np.float32(0.0729055341958355)) / np.float32(10.5402377416545)
    fused_exponent = np.float32(
        np.float64(np.float32(17.52)) * np.float64(cut_neighbors[2]) + np.float64(np.float32(-9.72))
    )
    logarithmic_expected = np.exp2(np.asarray((fused_exponent,), dtype=np.float32))[0]
    # The cut and its lower neighbour are owned by the linear inverse, whose float32 evaluation is exact, so
    # their bits are fixed. The upper neighbour is owned by the logarithmic inverse; CUDA exp2f carries up to
    # 2 ULP, so that point is bound within 2 ULP of the log branch rather than to a single GPU's bits.
    np.testing.assert_array_equal(decoded_cut_neighbors[:2].view(np.uint32), linear_expected.view(np.uint32))
    np.testing.assert_array_max_ulp(decoded_cut_neighbors[2], logarithmic_expected, maxulp=2)

    near_decode = _adjacent_float32(encoded_cut)
    near_decoded = _rgb_values(px.color.gamma_to_linear(_frame(near_decode, gamma="ACEScct"), gamma="ACEScct"))[:, 0]
    inversions = np.flatnonzero(np.diff(near_decoded) < 0.0)
    assert len(inversions) <= 1
    if len(inversions) == 1:
        index = int(inversions[0])
        np.testing.assert_array_max_ulp(near_decoded[index], near_decoded[index + 1], maxulp=1)

    restored = px.color.gamma_to_linear(
        px.color.linear_to_gamma(_frame(linear_values), gamma="ACEScct"), gamma="ACEScct"
    )
    np.testing.assert_allclose(_rgb_values(restored)[:, 0], linear_values, rtol=2e-6, atol=2e-7)

    encoded_domain = np.concatenate((encoded_values, cut_neighbors, np.asarray((0.0, 1.0, 1.5), dtype=np.float32)))
    decoded_domain = _rgb_values(px.color.gamma_to_linear(_frame(encoded_domain, gamma="ACEScct"), gamma="ACEScct"))[
        :, 0
    ]
    representable = np.isfinite(encoded_domain) & np.isfinite(decoded_domain)
    reencoded = _rgb_values(px.color.linear_to_gamma(_frame(decoded_domain[representable]), gamma="ACEScct"))[:, 0]
    np.testing.assert_allclose(reencoded, encoded_domain[representable], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("gamma", ("Gamma-2.5", "ACEScc", "ACEScct"))
def test_new_transfers_match_standalone_and_fused_paths(gamma: str) -> None:
    """v1-standard-tokens acceptance 126: keep standalone and fused transfer evaluation identical."""
    values = np.asarray(((-0.25, 0.18, 1.5), (0.01, -0.1, 2.0)), dtype=np.float32)
    source = _frame(values, colorspace="P3-D65", auxiliary=True)
    before = source.data.copy()
    standalone = px.color.linear_to_gamma(source, gamma=gamma)
    fused = px.color.rgb_to_rgb(source, output_gamma=gamma)
    assert cp.array_equal(standalone.data, fused.data)
    assert cp.array_equal(source.data, before)
    assert cp.array_equal(standalone.data[..., 0], source.data[..., 0])
    assert (standalone.colorspace, standalone.gamma, standalone.channels, standalone.matrix) == (
        "P3-D65",
        gamma,
        source.channels,
        None,
    )


def test_standard_gamut_definitions_matrices_native_rows_and_adaptation_are_independent() -> None:
    """v1-standard-tokens acceptance 127-130: derive P3 and SMPTE-C matrices and Bradford conversion from xy."""
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS
    from pixtreme._core.vocabulary import ReferenceWhite

    assert "DCI" not in get_args(ReferenceWhite)
    samples = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.25, 0.18, 1.5)), np.float32)
    for token, definition in _DEFINITIONS.items():
        assert _COLORSPACE_DEFINITIONS[token] == definition
        derived = _rgb_to_xyz(*definition)
        np.testing.assert_allclose(derived, _RGB_TO_XYZ[token], rtol=0.0, atol=1e-12)
        source = _frame(samples, colorspace=token)
        grayscale = px.color.rgb_to_grayscale(source, matrix="native")
        expected_luma = samples.astype(np.float64) @ _RGB_TO_XYZ[token][1]
        np.testing.assert_allclose(px.io.to_array(grayscale).get()[0, :, 0], expected_luma, rtol=0.0, atol=6e-6)
        for target, target_definition in (("Rec.709", _REC709), ("ACES2065-1", _ACES2065)):
            converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
            expected = samples.astype(np.float64) @ _conversion(definition, target_definition).T
            np.testing.assert_allclose(_rgb_values(converted), expected, rtol=0.0, atol=6e-6)

    eg_p3 = {
        "P3-DCI": np.asarray(((0.44517, 0.27713, 0.17228), (0.20949, 0.72160, 0.06891), (0.0, 0.04706, 0.90736))),
        "P3-D65": np.asarray(((0.48657, 0.26567, 0.19822), (0.22897, 0.69174, 0.07929), (0.0, 0.04511, 1.04394))),
    }
    for token, matrix in eg_p3.items():
        np.testing.assert_allclose(_rgb_to_xyz(*_DEFINITIONS[token]), matrix, rtol=0.0, atol=6e-6)
    eg_d60 = _rgb_to_xyz(_P3_PRIMARIES, (0.3217, 0.3378))
    eg_d60_table = np.asarray(((0.50474, 0.26474, 0.18286), (0.23752, 0.68933, 0.07314), (0.0, 0.04496, 0.96304)))
    np.testing.assert_allclose(eg_d60, eg_d60_table, rtol=0.0, atol=6e-6)
    assert np.max(np.abs(eg_d60 - _RGB_TO_XYZ["P3-D60"])) > 8e-4
    np.testing.assert_allclose(
        _xy_to_xyz(_ACES_WHITE),
        np.asarray((0.95264607456985, 1.0, 1.00882518435159)),
        rtol=0.0,
        atol=5e-13,
    )


def test_representative_frames_compose_independent_transfer_and_gamut_oracles() -> None:
    """v1-standard-tokens acceptance 131: compose every new metadata token without coupling axes."""
    encoded = np.asarray(((-0.25, 0.18, 1.5), (0.01, -0.1, 2.0)), dtype=np.float32)
    cases = (
        ("P3-DCI", "Gamma-2.5"),
        ("P3-D60", "linear"),
        ("P3-D65", "ACEScc"),
        ("SMPTE-C", "ACEScct"),
        ("ACEScg", "ACEScc"),
        ("ACEScg", "ACEScct"),
    )
    assert {colorspace for colorspace, _ in cases} >= {"P3-DCI", "P3-D60", "P3-D65", "SMPTE-C"}
    assert {gamma for _, gamma in cases} >= {"Gamma-2.5", "ACEScc", "ACEScct"}
    target_definitions = {"Rec.709": _REC709, "ACES2065-1": _ACES2065}
    from pixtreme._core.colorspace import _COLORSPACE_DEFINITIONS

    for colorspace, gamma in cases:
        source = _frame(encoded, colorspace=colorspace, gamma=gamma, auxiliary=True)
        before = source.data.copy()
        linear = _decode(_rgb_values(source), gamma)
        for target, target_definition in target_definitions.items():
            converted = px.color.rgb_to_rgb(source, output_colorspace=target, output_gamma="linear")
            expected = linear @ _conversion(_COLORSPACE_DEFINITIONS[colorspace], target_definition).T
            np.testing.assert_allclose(_rgb_values(converted), expected, rtol=2e-6, atol=8e-6)
            assert cp.array_equal(converted.data[..., 0], source.data[..., 0])
            assert (converted.colorspace, converted.gamma, converted.channels, converted.matrix) == (
                target,
                "linear",
                source.channels,
                None,
            )
        assert cp.array_equal(source.data, before)


def test_existing_token_bits_remain_at_the_pre_standard_baseline() -> None:
    """v1-standard-tokens acceptance 132: preserve every existing transfer and gamut fixture bit."""
    # Provenance: captured from complete pre-standard-token commit
    # 675c5b235b9b4155b42a58bbde8a07ea4d97feb4. Reproduce at that exact SHA with a float32 grid
    # (-0.25, -0.018056996166706085, 0, 0.18000000715255737, 1, 1.5), encode each Gamma,
    # decode its encoded output, and view channel R as uint32. For each Colorspace, convert the two RGB pixels
    # ((-0.25, 0.18, 1.5), (0.02, -0.1, 1)) to Rec.709/linear and view the flattened output as uint32.
    gamma_fixtures = {
        "linear": (
            (3196059648, 3163810884, 0, 1043878380, 1065353216, 1069547520),
            (3196059648, 3163810884, 0, 1043878380, 1065353216, 1069547520),
        ),
        "sRGB": (
            (3226384466, 3194938688, 0, 1055667935, 1065353215, 1066982086),
            (3196059648, 3163810884, 0, 1043878379, 1065353216, 1069547519),
        ),
        "Rec.709": (
            (3213885440, 3181799884, 0, 1053911414, 1065353216, 1067198555),
            (3196059648, 3163810884, 0, 1043878377, 1065353216, 1069547517),
        ),
        "BT.1886": (
            (3205475542, 3191882770, 0, 1056610176, 1065353216, 1066897169),
            (3196059647, 3163810883, 0, 1043878379, 1065353216, 1069547519),
        ),
        "PQ": (
            (3210348844, 3205597225, 893662952, 1062265261, 1065353216, 1065706239),
            (3196059617, 3163810756, 0, 1043878383, 1065353216, 1069548157),
        ),
        "HLG": (
            (3208450448, 3194901793, 0, 1059856297, 1065353216, 1065973579),
            (3196059645, 3163810884, 0, 1043878379, 1065353217, 1069547517),
        ),
        "S-Log": (
            (3213684627, 991558804, 1035255064, 1053104870, 1059289119, 1060354155),
            (3196059647, 3163810884, 828794470, 1043878380, 1065353216, 1069547519),
        ),
        "S-Log2": (
            (3208701273, 1021371850, 1035255064, 1051580213, 1058392199, 1059445478),
            (3196059648, 3163810884, 832750646, 1043878379, 1065353215, 1069547518),
        ),
        "S-Log3": (
            (3217556478, 3168455511, 1035874188, 1053963405, 1058575679, 1059324708),
            (3196059648, 3163810884, 789333745, 1043878379, 1065353213, 1069547519),
        ),
        "ARRI-LogC3": (
            (3214926503, 3146174940, 1035866836, 1053307405, 1058149604, 1058874277),
            (3196059648, 3163810884, 0, 1043878381, 1065353215, 1069547521),
        ),
        "ARRI-LogC4": (
            (3221400801, 2995575096, 1035874188, 1049528807, 1054532562, 1055775089),
            (3196059648, 3163810884, 0, 1043878379, 1065353214, 1069547520),
        ),
        "Blackmagic-Film-Gen-5": (
            (3221044577, 3177835909, 1035820719, 1053057585, 1057476139, 1058064820),
            (3196059648, 3163810884, 0, 1043878381, 1065353217, 1069547524),
        ),
        "DaVinci-Intermediate": (
            (3223788473, 3191938634, 0, 1051463133, 1057196762, 1057911650),
            (3196059648, 3163810884, 0, 1043878378, 1065353219, 1069547523),
        ),
        "RED-Log3G10": (
            (3228130337, 3187323085, 1035698008, 1051372189, 1056744777, 1057508475),
            (3196059648, 3163810884, 0, 1043878381, 1065353218, 1069547523),
        ),
        "REDlogFilm": (
            (3204351021, 3193857572, 1035874182, 1055532491, 1059810011, 1060668676),
            (3196059643, 3163810880, 2973898886, 1043878376, 1065353217, 1069547520),
        ),
        "Canon-Log": (
            (3188591788, 1035304199, 1040195611, 1051709637, 1058957330, 1060205148),
            (3196059651, 3163810882, 0, 1043878381, 1065353217, 1069547521),
        ),
        "Canon-Log-2": (
            (3195747939, 3159770779, 1035874188, 1053550602, 1058193192, 1058900243),
            (3196059649, 3163810886, 0, 1043878380, 1065353220, 1069547526),
        ),
        "Canon-Log-3": (
            (3188272132, 1034941615, 1040195592, 1051709627, 1058311446, 1059345448),
            (3196059649, 3163810884, 0, 1043878378, 1065353215, 1069547524),
        ),
        "V-Log": (
            (3215143756, 1019453554, 1040187336, 1054391367, 1058627527, 1059335953),
            (3196059647, 3163810884, 0, 1043878382, 1065353217, 1069547521),
        ),
        "Cineon": (
            (3204351021, 3193857572, 1035874182, 1055532491, 1059810011, 1060668676),
            (3196059643, 3163810880, 2973898886, 1043878376, 1065353217, 1069547520),
        ),
        "Gamma-2.2": (
            (3204993860, 3190105392, 0, 1055577349, 1065353216, 1067050896),
            (3196059647, 3163810885, 0, 1043878380, 1065353216, 1069547519),
        ),
        "Gamma-2.4": (
            (3205475542, 3191882770, 0, 1056610176, 1065353216, 1066897169),
            (3196059647, 3163810883, 0, 1043878379, 1065353216, 1069547519),
        ),
        "Gamma-2.6": (
            (3205903348, 3193612852, 0, 1057251335, 1065353216, 1066768924),
            (3196059649, 3163810885, 0, 1043878381, 1065353216, 1069547519),
        ),
    }
    linear_values = (-0.25, -0.018056996166706085, 0.0, 0.18000000715255737, 1.0, 1.5)
    for gamma, (encode_bits, decode_bits) in gamma_fixtures.items():
        encoded = px.color.linear_to_gamma(_frame(linear_values), gamma=gamma)
        np.testing.assert_array_equal(_rgb_values(encoded)[:, 0].view(np.uint32), encode_bits)
        decoded = px.color.gamma_to_linear(encoded, gamma=gamma)
        np.testing.assert_array_equal(_rgb_values(decoded)[:, 0].view(np.uint32), decode_bits)

    gamut_fixtures = {
        "sRGB": (3196059648, 1043878380, 1069547520, 1017370378, 3184315597, 1065353216),
        "Rec.709": (3196059648, 1043878380, 1069547520, 1017370378, 3184315597, 1065353216),
        "Rec.2020": (3206632196, 1046732887, 1070927681, 1016900116, 3187554328, 1066430520),
        "ACES2065-1": (3216325669, 1043331488, 1071467162, 3194295248, 3195324180, 1066891212),
        "ACEScg": (3207184178, 1046702756, 1071327960, 1012225928, 3187820956, 1066740601),
        "S-Gamut": (3208442050, 1020641384, 1071575628, 1023941360, 3198167830, 1066938173),
        "S-Gamut3": (3208442050, 1020641384, 1071575628, 1023941360, 3198167830, 1066938173),
        "S-Gamut3.Cine": (3206699341, 3178418026, 1072369260, 3112583168, 3200582444, 1067526363),
        "ARRI-Wide-Gamut-3": (3206485810, 3188555999, 1072370346, 1002402032, 3201056342, 1067620899),
        "ARRI-Wide-Gamut-4": (3209182827, 1035601944, 1071419179, 997395808, 3196821229, 1066861949),
        "Blackmagic-Wide-Gamut-Gen-5": (3205367535, 3187568618, 1073070301, 1025198451, 3200919674, 1068080043),
        "DaVinci-Wide-Gamut": (3209093926, 3190676685, 1074284506, 1009788360, 3203518916, 1069265258),
        "REDWideGamutRGB": (3209143998, 3190614948, 1075355334, 1027943654, 3203640764, 1071129191),
        "DRAGONcolor": (3205933323, 1007608160, 1072416519, 3186131858, 3197073364, 1067429716),
        "DRAGONcolor2": (3199014795, 1044384842, 1072072492, 1017369241, 3187784779, 1067180719),
        "REDcolor2": (3204995483, 1032083194, 1072885404, 3183172050, 3196333414, 1067614689),
        "REDcolor3": (3203501702, 1032960070, 1071542172, 3178541746, 3193460298, 1066843002),
        "REDcolor4": (3198600811, 1043676433, 1071394414, 1016703677, 3187207329, 1066732646),
        "Canon-Cinema-Gamut": (3209689344, 3186602936, 1074144546, 3151829184, 3202611611, 1069424635),
        "V-Gamut": (3208528264, 1033315073, 1071627965, 3148457696, 3196726112, 1066985781),
    }
    pixels = np.asarray(((-0.25, 0.18, 1.5), (0.02, -0.1, 1.0)), dtype=np.float32)
    for colorspace, expected_bits in gamut_fixtures.items():
        converted = px.color.rgb_to_rgb(
            _frame(pixels, colorspace=colorspace), output_colorspace="Rec.709", output_gamma="linear"
        )
        np.testing.assert_array_equal(
            px.io.to_array(converted).get().view(np.uint32).reshape(-1), np.asarray(expected_bits, dtype=np.uint32)
        )


def test_standard_transfer_dpx_codes_preserve_existing_mapping(tmp_path: Path) -> None:
    """v1-standard-tokens acceptance 133: classify ACES curves as log and Gamma-2.5 as video."""
    from pixtreme._io.formats.dpx import _dpx_transfer_from_gamma

    expected = {
        "ACEScc": 3,
        "ACEScct": 3,
        "Gamma-2.5": 6,
        "V-Log": 3,
        "Cineon": 1,
        "REDlogFilm": 1,
        "linear": 2,
        "Gamma-2.4": 6,
    }
    assert {gamma: _dpx_transfer_from_gamma(gamma) for gamma in expected} == expected
    for gamma, transfer in expected.items():
        frame = _frame((0.18,), colorspace="Rec.709", gamma=gamma)
        path = tmp_path / f"{gamma}.dpx"
        px.io.write_image(path, frame)
        assert path.read_bytes()[801] == transfer
        if gamma in {"ACEScc", "ACEScct"}:
            assert px.io.read_image(path).gamma == "Cineon"
        elif gamma == "Gamma-2.5":
            assert px.io.read_image(path).gamma == "Rec.709"


def test_standard_token_reference_requirements_changelog_and_docstrings_are_synchronized() -> None:
    """v1-standard-tokens acceptance 135; v1-vendor-a-tokens acceptance 161; v1-vendor-b-tokens acceptance 188:
    synchronize every public vocabulary and numeric contract surface.
    """
    token_reference = (ROOT / "docs_site" / "tokens.md").read_text(encoding="utf-8")
    requirements = require_repo_file("docs/requirements.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest = latest_changelog_section(changelog)
    for token in ("P3-DCI", "P3-D60", "P3-D65", "SMPTE-C", "ACEScc", "ACEScct", "Gamma-2.5"):
        assert f"`{token}`" in token_reference
        assert token in latest
    for fragment in (
        "0.3140",
        "0.3510",
        "0.32168",
        "0.33767",
        "0.630",
        "0.595",
        "Bradford",
        "0.5036269964912325",
        "-0.35844748858447484",
        "0.155251141552511",
        "222.8609442038076",
        "65504",
        "1 ULP",
        "Display P3",
        "white_point_simulation",
        "chromatic_adaptation",
        "AP1",
        "native",
    ):
        assert fragment in token_reference
    assert "27 Colorspace" in requirements
    assert "33 Gamma" in requirements
    assert "188 canonical tokens" in requirements
    for fragment in ("P3-D60", "Resolve", "Gamma-2.5", "no-upper-clip", "AP1", "bit-identical"):
        assert fragment in latest
    for operation in (
        px.color.rgb_to_rgb,
        px.color.rgb_to_ycbcr,
        px.color.ycbcr_to_rgb,
        px.color.rgb_to_grayscale,
        px.color.gamma_to_linear,
        px.color.linear_to_gamma,
    ):
        docstring = inspect.getdoc(operation)
        assert docstring is not None
        assert "ACEScc" in docstring and "ACEScct" in docstring and "Gamma-2.5" in docstring
        assert "P3-D65" in docstring or operation in {px.color.gamma_to_linear, px.color.linear_to_gamma}
